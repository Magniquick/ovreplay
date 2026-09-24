"""ovreplay-record: record one OpenVINO GPU inference for replay.

    ovreplay-record MODEL.xml OUTDIR [--device GPU] [--config KEY=VALUE ...]
                    [--plugins-xml PATH] [--check]

Compiles the recorder shim, runs one inference of MODEL under it, and packs
the result into OUTDIR for the ovreplay crate. The recording is tied to the
GPU it was made on; on a driver change the crate replays the recorded
inference and requires identical outputs before trusting it.

--check runs ovreplay-run on OUTDIR afterwards to confirm the replay matches
OpenVINO. It looks for the binary in $OVREPLAY_RUN, then PATH, then this
source tree's target/ (cargo build --release --example ovreplay-run).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .pack import pack

SHIM_SRC = Path(__file__).with_name("shim.c")


def build_shim() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ovreplay"
    shim = cache / "libovreplay_shim.so"
    if not shim.exists() or shim.stat().st_mtime < SHIM_SRC.stat().st_mtime:
        cache.mkdir(parents=True, exist_ok=True)
        cc = os.environ.get("CC", "cc")
        subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", str(shim), str(SHIM_SRC), "-ldl", "-lpthread"],
                       check=True)
    return shim


def find_runner() -> str | None:
    if env := os.environ.get("OVREPLAY_RUN"):
        return env
    if found := shutil.which("ovreplay-run"):
        return found
    root = Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        candidate = root / "target" / profile / "examples" / "ovreplay-run"
        if candidate.is_file():
            return str(candidate)
    return None


def main():
    ap = argparse.ArgumentParser(prog="ovreplay-record", description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--config", action="append", default=[], metavar="KEY=VALUE",
                    help="compile property, e.g. INFERENCE_PRECISION_HINT=dynamic")
    ap.add_argument("--plugins-xml", help="OpenVINO plugins.xml, to record through a specific plugin build")
    ap.add_argument("--check", action="store_true", help="replay the recording with ovreplay-run and compare")
    a = ap.parse_args()
    config = dict(kv.split("=", 1) for kv in a.config)

    shim = build_shim()
    with tempfile.TemporaryDirectory(prefix="ovreplay-") as raw:
        cmd = [sys.executable, "-m", "ovreplay._capture", a.model, raw, a.device, json.dumps(config)]
        if a.plugins_xml:
            cmd.append(a.plugins_xml)
        env = {**os.environ, "LD_PRELOAD": str(shim)}
        if subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL).returncode != 0:
            raise SystemExit("ovreplay-record: capture failed")
        if a.outdir.exists():
            shutil.rmtree(a.outdir)
        info = pack(Path(raw), a.outdir)
    print(f"{a.outdir}: {info['launches']} launches, {info['weights']} constant buffers "
          f"({info['weight_bytes'] / 1e6:.0f} MB), recorded on {info['device']} / {info['driver']}", flush=True)
    if a.check:
        runner = find_runner()
        if runner is None:
            raise SystemExit("--check: ovreplay-run not found; set OVREPLAY_RUN or put it on PATH")
        if subprocess.run([runner, str(a.outdir), "--bench", "0"]).returncode != 0:
            raise SystemExit("--check: replay does not match the recording")


if __name__ == "__main__":
    main()
