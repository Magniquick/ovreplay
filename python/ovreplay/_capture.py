"""Runs one inference under the recorder shim. Invoked by `ovreplay-record`
in a child process with LD_PRELOAD set; not meant to be run directly.

    python -m ovreplay._capture MODEL OUTDIR DEVICE CONFIG_JSON [PLUGINS_XML]
"""

import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import openvino as ov


def fill(arr, rng):
    """Random contents valid for the dtype. Different on every call, so the
    recorded inference touches scratch memory differently from the warm-up."""
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr[...] = rng.integers(max(info.min, 0), min(info.max, 255) + 1, arr.shape, dtype=arr.dtype)
    elif arr.dtype == np.bool_:
        arr[...] = rng.integers(0, 2, arr.shape).astype(bool)
    else:
        arr[...] = rng.standard_normal(arr.shape).astype(arr.dtype)


def main():
    model, outdir, device, config = sys.argv[1], Path(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
    plugins = sys.argv[5] if len(sys.argv) > 5 else None
    outdir.mkdir(parents=True, exist_ok=True)
    shim = ctypes.CDLL(None)
    shim.shim_locate.restype = ctypes.c_int
    shim.shim_locate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]

    core = ov.Core(plugins) if plugins else ov.Core()
    if "INFERENCE_PRECISION_HINT" in config and config["INFERENCE_PRECISION_HINT"] == "dynamic":
        config["INFERENCE_PRECISION_HINT"] = ov.Type.dynamic
    compiled = core.compile_model(model, device, config)
    req = compiled.create_infer_request()

    inputs = {}
    for port in compiled.inputs:
        shape = port.get_partial_shape()
        if shape.is_dynamic:
            raise SystemExit(f"input {port.get_any_name()} has a dynamic shape; reshape the model to static first")
        arr = np.zeros(shape.to_shape(), dtype=port.get_element_type().to_dtype())
        req.set_tensor(port.get_any_name(), ov.Tensor(arr, shared_memory=True))
        inputs[port.get_any_name()] = arr

    rng = np.random.default_rng(0)
    for arr in inputs.values():
        fill(arr, rng)
    for _ in range(3):  # compile kernels, settle allocations
        req.infer()

    shim.shim_start(str(outdir).encode())
    for arr in inputs.values():
        fill(arr, rng)
    req.infer()
    shim.shim_stop()

    # The recorded inference's inputs and outputs, for replay verification.
    golden = [("INPUT", name, arr.tobytes()) for name, arr in inputs.items()]
    outputs = [port.get_any_name() for port in compiled.outputs]
    golden += [("OUTPUT", name, np.asarray(req.get_tensor(name).data).tobytes()) for name in outputs]

    # A replay is checked for bit-identical outputs, so OpenVINO itself must
    # give the same bytes for the same input.
    req.infer()
    for kind, name, data in golden:
        if kind == "OUTPUT" and np.asarray(req.get_tensor(name).data).tobytes() != data:
            raise SystemExit(f"output {name} differs between two inferences of the same input; "
                             "a nondeterministic graph cannot be verified")

    off = ctypes.c_size_t(0)
    with open(outdir / "io.txt", "w") as f, open(outdir / "golden.bin", "wb") as g:
        for name, arr in inputs.items():
            f.write(f"INPUT {name} {arr.ctypes.data:#x} {arr.nbytes}\n")
        for name in outputs:
            t = req.get_tensor(name)
            alloc = shim.shim_locate(ctypes.c_void_p(t.data.ctypes.data), ctypes.byref(off))
            if alloc < 0:
                raise SystemExit(f"output {name} is not in device-visible host memory; unsupported")
            f.write(f"OUTPUT {name} {alloc} {off.value} {t.byte_size} {t.element_type.get_type_name()}\n")
        pos = 0
        for kind, name, data in golden:
            f.write(f"GOLDEN {kind} {name} {pos} {len(data)}\n")
            g.write(data)
            pos += len(data)


if __name__ == "__main__":
    main()
