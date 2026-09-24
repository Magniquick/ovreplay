"""Turn a raw recording into the files the ovreplay crate loads.

Reads rec.txt, io.txt and the snapshots from the capture directory and writes
into OUTDIR:
  replay.txt   device identity, inputs, outputs, allocation manifest, the
               program/kernel table and the launch list
  weights.bin  every constant allocation, page-aligned and concatenated
  prog_*.bin   the recorded device binaries
  golden.bin   the recorded inference's inputs and outputs
"""

import shutil
from pathlib import Path

ALIGN = 4096
FORMAT = "ovreplay 1"


def pack(raw: Path, out: Path) -> dict:
    size, kind, dirty, used, snap, unsupported, device, writes = {}, {}, set(), set(), {}, [], {}, []
    lines = (raw / "rec.txt").read_text().splitlines()
    for line in lines:
        p = line.split()
        if not p:
            continue
        tag = p[0]
        if tag == "UNSUPPORTED":
            unsupported.append(line)
        elif tag == "ALLOC":
            size[int(p[1])], kind[int(p[1])] = int(p[3]), int(p[2])
        elif tag == "DIRTY":
            dirty.add(int(p[1]))
        elif tag == "SNAP":
            snap[int(p[1])] = int(p[2])
        elif tag in ("DEVICE_NAME", "DRIVER_VERSION", "DEVICE_ID"):
            device[tag] = line[len(tag) + 1:]
        elif tag == "U":
            used.add(int(p[1]))
        elif tag == "M":
            unsupported.append(f"cl_mem buffer argument (alloc {p[1]})")
        elif tag in ("COPY", "FILL"):
            unsupported.append(f"device-side {tag.lower()} during the inference: {line}")
        elif tag == "WRITE":
            writes.append((int(p[1]), int(p[2]), int(p[3]), int(p[5], 16), int(p[4])))
    if unsupported:
        raise SystemExit("recording uses unsupported OpenCL features:\n  " + "\n  ".join(unsupported[:10]))

    # Each host write must come from one of the model's input tensors.
    io = [l.split() for l in (raw / "io.txt").read_text().splitlines()]
    inputs = [(n, int(ptr, 16), int(nb)) for tag, n, ptr, nb in (x for x in io if x[0] == "INPUT")]
    input_lines, written = [], {}
    for name, ptr, nbytes in inputs:
        hit = [w for w in writes if ptr <= w[3] < ptr + nbytes]
        if len(hit) != 1 or hit[0][2] != nbytes:
            raise SystemExit(f"input {name}: expected one full write of {nbytes} bytes, found {len(hit)}")
        dst, off, _, _, blob = hit[0]
        input_lines.append(f"INPUT {name} {dst} {off} {nbytes}")
        written[name] = (raw / f"blob_{blob}.bin").read_bytes()
    if len(writes) != len(inputs):
        raise SystemExit(f"{len(writes)} host writes but {len(inputs)} inputs; unsupported")
    output_lines = [" ".join(x) for x in io if x[0] == "OUTPUT"]

    # golden.bin must hold what the recorded inference actually consumed and
    # one entry of the right size per port.
    golden = (raw / "golden.bin").read_bytes()
    golden_lines = [" ".join(x) for x in io if x[0] == "GOLDEN"]
    ports = {("INPUT", n): int(nb) for _, n, _, nb in (x for x in io if x[0] == "INPUT")}
    ports |= {("OUTPUT", x[1]): int(x[4]) for x in io if x[0] == "OUTPUT"}
    seen = {}
    for _, port, name, goff, gbytes in (x for x in io if x[0] == "GOLDEN"):
        goff, gbytes = int(goff), int(gbytes)
        if ports.get((port, name)) != gbytes or goff + gbytes > len(golden):
            raise SystemExit(f"golden {port.lower()} {name}: bad entry")
        seen[(port, name)] = golden[goff:goff + gbytes]
    if seen.keys() != ports.keys():
        raise SystemExit("golden.bin does not cover every input and output")
    for name, data in written.items():
        if seen[("INPUT", name)] != data:
            raise SystemExit(f"input {name}: bytes sent to the device differ from the golden input")

    # Kinds: 1 device, 2 host, 3 shared. Constants go to weights.bin; memory
    # changed during the inference (or never snapshotted) is scratch; host
    # memory holds outputs.
    out.mkdir(parents=True, exist_ok=True)
    manifest, offset, n_weights = [], 0, 0
    with open(out / "weights.bin", "wb") as wf:
        for a in sorted(used | {int(l.split()[2]) for l in input_lines}):
            k = kind.get(a, 1)
            if k == 2:
                manifest.append(f"H {a} {size[a]}")
            elif a in dirty or a not in snap:
                manifest.append(f"S {a} {size[a]} {k}")
            else:
                data = (raw / f"blob_{snap[a]}.bin").read_bytes()
                if len(data) != size[a]:
                    raise SystemExit(f"alloc {a}: snapshot {len(data)} bytes, allocation {size[a]}")
                pad = (-offset) % ALIGN
                wf.write(b"\0" * pad)
                offset += pad
                manifest.append(f"W {a} {offset} {size[a]}")
                wf.write(data)
                offset += len(data)
                n_weights += 1

    body = [f"FORMAT {FORMAT}"] + [f"{k} {v}" for k, v in device.items()] + input_lines + output_lines + golden_lines + manifest
    arg_tags = (" U", " M", " L", " N", " V", " X")
    for line in lines:
        if line[:2] in arg_tags or line.split()[:1] and line.split()[0] in ("PROG", "KERN", "NDR"):
            body.append(line)
    (out / "replay.txt").write_text("\n".join(body) + "\n")
    for prog in raw.glob("prog_*.bin"):
        shutil.copy2(prog, out / prog.name)
    shutil.copy2(raw / "golden.bin", out / "golden.bin")
    return {
        "device": device.get("DEVICE_NAME"),
        "driver": device.get("DRIVER_VERSION"),
        "weights": n_weights,
        "weight_bytes": offset,
        "launches": sum(1 for l in body if l.startswith("NDR")),
    }
