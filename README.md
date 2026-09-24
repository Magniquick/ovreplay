# ovreplay

Record one OpenVINO GPU inference, then replay it over plain OpenCL with no
OpenVINO in the process. The replay does no graph compilation and no oneDNN
kernel planning. It creates a context, loads one device binary per program,
uploads the weights once, and binds every kernel argument ahead of time. A run
is the input copies, the recorded kernel launches and a `clFinish`.

The project has two halves:

- `ovreplay-record` (Python) runs a model once under an `LD_PRELOAD` OpenCL
  interposer and writes a recording directory.
- the `ovreplay` crate (Rust) loads that directory and runs it.

## Why

A GPU model in OpenVINO is slow to start even when the model cache hits. On
import the GPU plugin rebuilds its graph and oneDNN plans each distinct
convolution shape again through its `jit:ir` path, about 8 ms per shape, and
the weights are copied into place again. For one mid-sized fused face model
(detector, landmarks, recognizer and an anti-spoof head in one graph) on an
Intel Arc iGPU, OpenVINO's cached import took 0.35 to 1.2 s before the first
inference could run.

The replay of the same graph is ready about 70 to 90 ms after the process
starts, including the upload of 142 MB of weights, and its outputs are
bit-identical to OpenVINO's.

## How it works

The interposer (`python/ovreplay/shim.c`) wraps the OpenCL entry points and
the Intel USM extension functions, including the ones oneDNN resolves through
`dlsym`. It tracks every allocation, program, kernel and kernel argument.

The recorder compiles the model, fills its inputs with random data and runs
three warm-up inferences so that every kernel is built and every allocation
settled. It then snapshots the contents of every live allocation, refills the
inputs with different random data, and records one inference. During that
inference the shim logs each kernel launch with its arguments, and each host
write with its source pointer, so each write can be matched to a named input
tensor. Output tensors are located by address in the host USM allocations.

After the inference every snapshotted allocation is read back and compared.
An allocation whose contents changed is scratch; the replay allocates it
empty. One whose contents did not change holds constants and goes into
`weights.bin`. The recorded input differs from the warm-up input, so buffers
that depend on the input change and are classified as scratch. A buffer that
is written but comes out byte-identical is treated as a constant, which is
still correct because the replay starts it with those same bytes.

The recorder also saves the recorded inference's input and output bytes to
`golden.bin`. Before saving, it runs the same input through OpenVINO a second
time and refuses the recording if the outputs differ, since a
nondeterministic graph could never pass a bit-exact check.

A recording directory holds:

| file | contents |
| --- | --- |
| `replay.txt` | device identity, inputs, outputs, allocation list, programs, kernels, launches |
| `weights.bin` | the constant allocations, page aligned |
| `prog_<n>.bin` | the device binaries OpenVINO built |
| `golden.bin` | the recorded inference's inputs and outputs |

### Device and driver checks

Device binaries are specific to the GPU they were built for. `Replay::load`
refuses a recording made on a device with a different name or PCI device id
and returns `Error::DeviceMismatch`.

A driver update usually leaves the binaries valid, so a changed
`CL_DRIVER_VERSION` does not refuse the recording outright. Instead `load`
replays the golden input once and requires outputs bit-identical to the ones
OpenVINO produced at record time. If any byte differs it returns
`Error::GoldenMismatch`. This catches a driver that changed the kernel ABI and
would otherwise produce wrong results without an error, and it avoids
re-recording after every driver update. `Replay::verify` runs the same check
on demand.

On either error, fall back to OpenVINO and record again.

## Usage

### Recording

Needs Python 3.11 or newer, OpenVINO, NumPy, a C compiler and the OpenCL
headers. The shim is compiled on first use into `~/.cache/ovreplay`.

```sh
uvx --from /path/to/ovreplay ovreplay-record model.xml model.replay \
    --config INFERENCE_PRECISION_HINT=dynamic --config PERFORMANCE_HINT=LATENCY
```

```
ovreplay-record MODEL OUTDIR [--device GPU] [--config KEY=VALUE ...]
                [--plugins-xml PATH] [--check]
```

`--config` passes compile properties to `compile_model`, so the recording
captures the kernels OpenVINO picks for those settings. `--plugins-xml` points
OpenVINO at a specific GPU plugin build. `--check` runs `ovreplay-run` on the
new recording and fails if the replay does not match; it finds the binary
through `$OVREPLAY_RUN`, then `PATH`, then this source tree's `target/`.

### Replaying

```rust
let mut replay = ovreplay::Replay::load("model.replay")?;
replay.set_input("frame", &frame_bytes)?;
replay.run()?;
let scores: &[f32] = replay.output_f32("score")?;
```

| item | purpose |
| --- | --- |
| `Replay::load(dir)` | load onto the first GPU, with the device and driver checks above |
| `Replay::inputs()` | `(name, bytes)` for each input |
| `Replay::outputs()` | `(name, bytes, element type)` for each output |
| `Replay::set_input(name, &[u8])` | copy one input; the length must match exactly |
| `Replay::run()` | run the launches and wait for them |
| `Replay::output(name)` | output bytes, valid until the next run |
| `Replay::output_f32(name)` | an `f32` output as `&[f32]` |
| `Replay::verify()` | replay the golden input and compare; overwrites inputs and outputs |

`Replay` is `Send`. It takes `&mut self` for anything that touches the queue.

### Example runner

```sh
cargo build --release --example ovreplay-run
target/release/examples/ovreplay-run model.replay [--input NAME=FILE ...] [--bench N]
```

With no `--input` it replays the golden input, reports whether the outputs
match, and exits with status 1 if they do not. Each `--input` file holds the
raw bytes of one input. It prints the load time, the first and steady-state
run times over `N` runs (default 10), and the first values of each output.

Measured on the face model above (Intel Arc iGPU in an Arrow Lake H, Linux
compute-runtime 26.35):

```
load       77.17 ms (ready 80.00 ms after exec)
verify     pass, outputs bit-identical to the recording (63.85 ms)
steady     median 18.87 ms, min 17.93 ms over 200 runs
```

## Limitations

- Static shapes only. Reshape a dynamic model to static shapes before recording.
- The trace must use Intel USM pointers only, which is what OpenVINO's GPU
  plugin produces on Intel GPUs. `cl_mem` kernel arguments, OpenCL images,
  buffer maps, and device-side copies or fills inside the recorded inference
  are refused at record time.
- Outputs must live in host USM, as OpenVINO allocates them on this path.
- A recording is tied to the device it was made on, and on a driver change it
  is only trusted after the golden check passes.
- The replay uses the first GPU OpenCL reports.
- Tested only on an Intel Arc iGPU (Arrow Lake H) with OpenVINO 2026.4 and
  compute-runtime 26.35.

## Prior art

GPUReplay ([paper](https://par.nsf.gov/servlets/purl/10325027)) records GPU
work below the driver on mobile GPUs and replays it without the GPU stack.
ovreplay works one level higher. It records OpenCL calls, so the driver stays
in place and only OpenVINO and oneDNN are removed.

Intel's [opencl-intercept-layer](https://github.com/intel/opencl-intercept-layer)
can capture a single kernel launch with its inputs and replay it, which is
meant for debugging one kernel. ovreplay records every launch of a whole
inference and binds them once for repeated runs.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
