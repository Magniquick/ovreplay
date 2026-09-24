//! Replay a recorded `OpenVINO` GPU inference over plain `OpenCL`.
//!
//! `ovreplay-record` (the Python half of this project) runs one inference of
//! a model under an `OpenCL` interposer and saves the device binaries, the
//! constant buffers, the launch list, and the inputs and outputs of that
//! inference. [`Replay`] loads it with nothing but the `OpenCL` runtime: no
//! graph compilation, no `oneDNN` kernel planning, no weight repacking.
//! Loading costs a context, one binary load per program and one upload of
//! the weights.
//!
//! A recording holds the exact kernels chosen on the GPU it was made on.
//! [`Replay::load`] refuses another device with [`Error::DeviceMismatch`].
//! When only the driver version differs it replays the recorded input once
//! and requires bit-identical outputs, returning [`Error::GoldenMismatch`]
//! otherwise. On either error, fall back to `OpenVINO` and re-record.
//!
//! ```no_run
//! let mut replay = ovreplay::Replay::load("model.replay")?;
//! replay.set_input("frame", &vec![0u8; 640 * 360])?;
//! replay.run()?;
//! let scores = replay.output_f32("score")?;
//! # Ok::<(), ovreplay::Error>(())
//! ```

mod cl;
mod plan;

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use opencl_sys::{CL_DEVICE_NAME, CL_DRIVER_VERSION, cl_device_info};

pub use cl::ClError;
use plan::{Arg, Plan};

/// `CL_DEVICE_ID_INTEL` from `cl_intel_device_attribute_query`.
const CL_DEVICE_ID_INTEL: cl_device_info = 0x4251;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("{path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("recorded on {recorded}, running on {found}; re-record")]
    DeviceMismatch { recorded: String, found: String },
    #[error("output {output} differs from the recording in {differing} of {bytes} bytes; re-record")]
    GoldenMismatch { output: String, differing: usize, bytes: usize },
    #[error("no input named {0}")]
    NoInput(String),
    #[error("no output named {0}")]
    NoOutput(String),
    #[error("input {name} takes {expected} bytes, got {got}")]
    InputSize { name: String, expected: usize, got: usize },
    #[error("output {name} is {element_type}, not {wanted}")]
    OutputType { name: String, element_type: String, wanted: &'static str },
    #[error("OpenCL: {0}")]
    Cl(#[from] ClError),
}

struct Launch {
    kernel: cl::Kernel,
    dim: u32,
    global: [usize; 3],
    local: [usize; 3],
    offset: [usize; 3],
}

struct Io {
    name: String,
    ptr: *mut c_void,
    bytes: usize,
    element_type: String,
    golden: usize,
}

/// One allocation of the recording, as bound on this device.
#[derive(Clone, Copy)]
struct Slot {
    ptr: *mut c_void,
    size: usize,
    host: bool,
}

/// A loaded recording. Every kernel argument is bound once at load, so a run
/// is the input copies, the launches, and a finish.
pub struct Replay {
    // Fields drop in order: kernels first, then the USM behind every bound
    // pointer, then the context those were made on.
    launches: Vec<Launch>,
    inputs: Vec<Io>,
    outputs: Vec<Io>,
    golden: PathBuf,
    _weights: cl::Alloc,
    _scratch: Vec<cl::Alloc>,
    _host: Vec<cl::Alloc>,
    ctx: cl::Context,
}

// SAFETY: every handle and pointer is owned by the Replay and used through
// `&mut self`/`&self` from one thread at a time, which OpenCL allows.
unsafe impl Send for Replay {}

fn describe(name: &str, id: Option<u32>, driver: &str) -> String {
    id.map_or_else(|| format!("{name} / {driver}"), |id| format!("{name} [{id:#06x}] / {driver}"))
}

impl Replay {
    /// Load the recording in `dir` onto the first GPU.
    ///
    /// The device name, and the PCI device id where both sides report one,
    /// must match the recording. If the driver version differs, the recorded
    /// input is replayed once as in [`Replay::verify`].
    ///
    /// # Errors
    ///
    /// [`Error::DeviceMismatch`] on another GPU, [`Error::GoldenMismatch`]
    /// when a different driver gives different results; otherwise I/O,
    /// format or `OpenCL` errors.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let plan = Plan::read(dir)?;
        let ctx = cl::Context::first_gpu()?;
        let name = ctx.device_string(CL_DEVICE_NAME)?;
        let driver = ctx.device_string(CL_DRIVER_VERSION)?;
        let id = ctx.device_uint(CL_DEVICE_ID_INTEL);
        let id_differs = matches!((plan.device_id, id), (Some(a), Some(b)) if a != b);
        if name != plan.device_name || id_differs {
            return Err(Error::DeviceMismatch {
                recorded: describe(&plan.device_name, plan.device_id, &plan.driver_version),
                found: describe(&name, id, &driver),
            });
        }
        let mut replay = Self::build(ctx, &plan, dir)?;
        if driver != plan.driver_version {
            replay.verify()?;
        }
        Ok(replay)
    }

    fn build(ctx: cl::Context, plan: &Plan, dir: &Path) -> Result<Self, Error> {
        let parse_err = |message: String| Error::Parse { path: dir.join("replay.txt"), message };

        // Constants: one device allocation, filled once from weights.bin.
        let path = dir.join("weights.bin");
        let bytes = std::fs::read(&path).map_err(|source| Error::Io { path, source })?;
        let weights = ctx.device_alloc(bytes.len())?;
        ctx.upload(&weights, 0, &bytes)?;
        let weights_len = bytes.len();
        drop(bytes);

        let mut slots: Vec<Option<Slot>> = vec![None; plan.max_alloc_id() + 1];
        let mut bind = |id: usize, slot: Slot| match slots.get_mut(id) {
            Some(s @ None) => {
                *s = Some(slot);
                Ok(())
            }
            _ => Err(parse_err(format!("allocation {id} listed twice"))),
        };
        for w in &plan.weights {
            if w.off.checked_add(w.size).is_none_or(|end| end > weights_len) {
                return Err(parse_err(format!("allocation {} lies outside weights.bin", w.id)));
            }
            bind(w.id, Slot { ptr: weights.ptr().wrapping_byte_add(w.off), size: w.size, host: false })?;
        }
        let mut scratch = Vec::with_capacity(plan.scratch.len());
        for s in &plan.scratch {
            let a = ctx.device_alloc(s.size)?;
            bind(s.id, Slot { ptr: a.ptr(), size: s.size, host: false })?;
            scratch.push(a);
        }
        let mut host = Vec::with_capacity(plan.host.len());
        for h in &plan.host {
            let a = ctx.host_alloc(h.size)?;
            bind(h.id, Slot { ptr: a.ptr(), size: h.size, host: true })?;
            host.push(a);
        }
        // A pointer `off` bytes into allocation `id`, `len` bytes of which
        // must lie inside it.
        let at = |id: usize, off: usize, len: usize| -> Result<(Slot, *mut c_void), Error> {
            let slot = slots
                .get(id)
                .copied()
                .flatten()
                .ok_or_else(|| parse_err(format!("allocation {id} missing")))?;
            if off.checked_add(len).is_none_or(|end| end > slot.size) {
                return Err(parse_err(format!("{len} bytes at {off} overrun allocation {id} of {}", slot.size)));
            }
            Ok((slot, slot.ptr.wrapping_byte_add(off)))
        };

        let programs = plan.build_programs(&ctx, dir)?;
        let mut launches = Vec::with_capacity(plan.launches.len());
        for l in &plan.launches {
            let program = programs.get(&l.program).ok_or_else(|| parse_err(format!("no program {}", l.program)))?;
            let kernel = cl::Context::kernel(program, &l.name)?;
            for (i, arg) in l.args.iter().enumerate() {
                let index = u32::try_from(i).map_err(|_| parse_err("too many arguments".into()))?;
                match arg {
                    Arg::Ptr { alloc, off } => ctx.set_arg_ptr(&kernel, index, at(*alloc, *off, 0)?.1)?,
                    Arg::Value(bytes) => cl::Context::set_arg_value(&kernel, index, bytes)?,
                    Arg::Local(size) => cl::Context::set_arg_local(&kernel, index, *size)?,
                    Arg::Null => cl::Context::set_arg_null(&kernel, index)?,
                }
            }
            launches.push(Launch { kernel, dim: l.dim, global: l.global, local: l.local, offset: l.offset });
        }

        let resolve = |p: &plan::Port, output: bool| -> Result<Io, Error> {
            let (slot, ptr) = at(p.alloc, p.off, p.bytes)?;
            if output && !slot.host {
                return Err(parse_err(format!("output {} is not in host memory", p.name)));
            }
            if p.element_type == "f32" && (ptr.align_offset(4) != 0 || !p.bytes.is_multiple_of(4)) {
                return Err(parse_err(format!("f32 port {} is misaligned", p.name)));
            }
            Ok(Io { name: p.name.clone(), ptr, bytes: p.bytes, element_type: p.element_type.clone(), golden: p.golden })
        };
        let inputs = plan.inputs.iter().map(|p| resolve(p, false)).collect::<Result<_, _>>()?;
        let outputs = plan.outputs.iter().map(|p| resolve(p, true)).collect::<Result<_, _>>()?;
        Ok(Self {
            launches,
            inputs,
            outputs,
            golden: dir.join("golden.bin"),
            _weights: weights,
            _scratch: scratch,
            _host: host,
            ctx,
        })
    }

    /// Names and byte sizes of the inputs.
    pub fn inputs(&self) -> impl Iterator<Item = (&str, usize)> {
        self.inputs.iter().map(|i| (i.name.as_str(), i.bytes))
    }

    /// Names, byte sizes and element types (`OpenVINO` names: `f32`, `f16`,
    /// `u8`, ...) of the outputs.
    pub fn outputs(&self) -> impl Iterator<Item = (&str, usize, &str)> {
        self.outputs.iter().map(|o| (o.name.as_str(), o.bytes, o.element_type.as_str()))
    }

    fn write(&self, input: &Io, data: &[u8]) -> Result<(), Error> {
        if data.len() != input.bytes {
            return Err(Error::InputSize { name: input.name.clone(), expected: input.bytes, got: data.len() });
        }
        // SAFETY: the input region holds `bytes` bytes (checked at load);
        // `data` is that long.
        unsafe { self.ctx.copy(input.ptr, data.as_ptr().cast(), data.len())? };
        Ok(())
    }

    /// Copy `data` into input `name`, which must be its exact byte size.
    ///
    /// # Errors
    ///
    /// Unknown name, wrong size, or an `OpenCL` copy failure.
    pub fn set_input(&mut self, name: &str, data: &[u8]) -> Result<(), Error> {
        let input = self.inputs.iter().find(|i| i.name == name).ok_or_else(|| Error::NoInput(name.to_owned()))?;
        self.write(input, data)
    }

    /// Run the recorded launches and wait for them.
    ///
    /// # Errors
    ///
    /// An `OpenCL` enqueue or finish failure.
    pub fn run(&mut self) -> Result<(), Error> {
        for l in &self.launches {
            self.ctx.launch(&l.kernel, l.dim, &l.global, &l.local, &l.offset)?;
        }
        Ok(self.ctx.finish()?)
    }

    /// Replay the recorded inference's inputs and require outputs
    /// bit-identical to what `OpenVINO` produced for them. Overwrites every
    /// input and output.
    ///
    /// # Errors
    ///
    /// [`Error::GoldenMismatch`] on the first output that differs; I/O errors
    /// reading `golden.bin`, or `OpenCL` errors.
    pub fn verify(&mut self) -> Result<(), Error> {
        let path = &self.golden.clone();
        let golden = std::fs::read(path).map_err(|source| Error::Io { path: path.clone(), source })?;
        let recorded = |io: &Io| {
            io.golden.checked_add(io.bytes).and_then(|end| golden.get(io.golden..end)).ok_or_else(|| Error::Parse {
                path: path.clone(),
                message: format!("too short for {}", io.name),
            })
        };
        for input in &self.inputs {
            self.write(input, recorded(input)?)?;
        }
        self.run()?;
        for o in &self.outputs {
            let want = recorded(o)?;
            let got = Self::bytes(o);
            if got != want {
                let differing = got.iter().zip(want).filter(|(a, b)| a != b).count();
                return Err(Error::GoldenMismatch { output: o.name.clone(), differing, bytes: o.bytes });
            }
        }
        Ok(())
    }

    fn bytes(o: &Io) -> &[u8] {
        // SAFETY: host USM of at least `bytes` bytes (checked at load), written
        // by a finished run, and not written again while `&self` is borrowed.
        unsafe { std::slice::from_raw_parts(o.ptr.cast::<u8>(), o.bytes) }
    }

    /// Output `name` as raw bytes, valid until the next [`Replay::run`].
    ///
    /// # Errors
    ///
    /// Unknown name.
    pub fn output(&self, name: &str) -> Result<&[u8], Error> {
        let o = self.outputs.iter().find(|o| o.name == name).ok_or_else(|| Error::NoOutput(name.to_owned()))?;
        Ok(Self::bytes(o))
    }

    /// Output `name` as f32 values.
    ///
    /// # Errors
    ///
    /// Unknown name, or an output that is not f32.
    pub fn output_f32(&self, name: &str) -> Result<&[f32], Error> {
        let o = self.outputs.iter().find(|o| o.name == name).ok_or_else(|| Error::NoOutput(name.to_owned()))?;
        if o.element_type != "f32" {
            return Err(Error::OutputType { name: name.to_owned(), element_type: o.element_type.clone(), wanted: "f32" });
        }
        // SAFETY: as `bytes`; load checked f32 ports are 4-byte aligned and a
        // multiple of 4 bytes long, and every bit pattern is a valid f32.
        Ok(unsafe { std::slice::from_raw_parts(o.ptr.cast::<f32>(), o.bytes / 4) })
    }
}
