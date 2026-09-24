//! Parsing `replay.txt`, the recording `ovreplay-record` writes.
//!
//! One record per line, space-separated:
//!
//! ```text
//! FORMAT ovreplay 1
//! DEVICE_NAME <text> / DRIVER_VERSION <text> / DEVICE_ID <n>
//! INPUT <name> <alloc> <byte offset> <bytes>
//! OUTPUT <name> <alloc> <byte offset> <bytes> <element type>
//! GOLDEN INPUT|OUTPUT <name> <offset in golden.bin> <bytes>
//! W <alloc> <offset in weights.bin> <size>      a constant buffer
//! S <alloc> <size> <kind>                       scratch
//! H <alloc> <size>                              host memory (outputs)
//! PROG <program> <binary size>                  prog_<program>.bin
//! KERN <kernel> <program> <name>
//! NDR <kernel> <dim> <global x3> <local x3> <offset x3> <nargs>
//!   followed by nargs lines:
//!   " U <alloc> <offset>"   a USM pointer
//!   " V <n> <hex bytes>"    a by-value argument
//!   " L <n>"                n bytes of local memory
//!   " N"                    a null pointer
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::{Error, cl};

pub const FORMAT: &str = "ovreplay 1";

pub struct Region {
    pub id: usize,
    pub off: usize,
    pub size: usize,
}

pub struct Port {
    pub name: String,
    pub alloc: usize,
    pub off: usize,
    pub bytes: usize,
    pub element_type: String,
    /// Byte offset of this port's recorded contents in `golden.bin`.
    pub golden: usize,
}

pub enum Arg {
    Ptr { alloc: usize, off: usize },
    Value(Vec<u8>),
    Local(usize),
    Null,
}

pub struct LaunchPlan {
    pub program: usize,
    pub name: String,
    pub dim: u32,
    pub global: [usize; 3],
    pub local: [usize; 3],
    pub offset: [usize; 3],
    pub args: Vec<Arg>,
}

pub struct Plan {
    pub device_name: String,
    pub driver_version: String,
    pub device_id: Option<u32>,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    pub weights: Vec<Region>,
    pub scratch: Vec<Region>,
    pub host: Vec<Region>,
    pub programs: Vec<usize>,
    pub launches: Vec<LaunchPlan>,
}

fn bad(line: &str) -> String {
    format!("bad line: {line}")
}

fn num<T: std::str::FromStr>(field: Option<&str>, line: &str) -> Result<T, String> {
    field.and_then(|f| f.parse().ok()).ok_or_else(|| bad(line))
}

fn triple(it: &mut std::str::SplitWhitespace<'_>, line: &str) -> Result<[usize; 3], String> {
    Ok([num(it.next(), line)?, num(it.next(), line)?, num(it.next(), line)?])
}

fn port(it: &mut std::str::SplitWhitespace<'_>, line: &str, typed: bool) -> Result<Port, String> {
    Ok(Port {
        name: it.next().ok_or_else(|| bad(line))?.to_owned(),
        alloc: num(it.next(), line)?,
        off: num(it.next(), line)?,
        bytes: num(it.next(), line)?,
        element_type: if typed { it.next().unwrap_or("u8").to_owned() } else { "u8".to_owned() },
        golden: usize::MAX,
    })
}

fn arg(line: &str) -> Result<Arg, String> {
    let mut a = line.split_whitespace();
    Ok(match a.next() {
        Some("U") => Arg::Ptr { alloc: num(a.next(), line)?, off: num(a.next(), line)? },
        Some("L") => Arg::Local(num(a.next(), line)?),
        Some("N") => Arg::Null,
        Some("V") => {
            let n: usize = num(a.next(), line)?;
            let bytes = a
                .map(|h| u8::from_str_radix(h, 16).map_err(|_| format!("bad arg: {line}")))
                .collect::<Result<Vec<u8>, String>>()?;
            if bytes.len() != n {
                return Err(format!("value arg has {} of {n} bytes: {line}", bytes.len()));
            }
            Arg::Value(bytes)
        }
        _ => return Err(format!("unsupported arg: {line}")),
    })
}

impl Plan {
    pub fn read(dir: &Path) -> Result<Self, Error> {
        let path = dir.join("replay.txt");
        let text = std::fs::read_to_string(&path).map_err(|source| Error::Io { path: path.clone(), source })?;
        Self::parse(&text).map_err(|message| Error::Parse { path, message })
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut plan = Self {
            device_name: String::new(),
            driver_version: String::new(),
            device_id: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: Vec::new(),
            scratch: Vec::new(),
            host: Vec::new(),
            programs: Vec::new(),
            launches: Vec::new(),
        };
        let mut format = None;
        let mut kernels: HashMap<usize, (usize, String)> = HashMap::new();
        let mut golden: HashMap<(bool, String), (usize, usize)> = HashMap::new();
        let mut lines = text.lines();
        while let Some(line) = lines.next() {
            let mut it = line.split_whitespace();
            let Some(tag) = it.next() else { continue };
            let rest = || line.get(tag.len()..).unwrap_or("").trim().to_owned();
            match tag {
                "FORMAT" => format = Some(rest()),
                "DEVICE_NAME" => plan.device_name = rest(),
                "DRIVER_VERSION" => plan.driver_version = rest(),
                "DEVICE_ID" => plan.device_id = Some(num(it.next(), line)?),
                "INPUT" => plan.inputs.push(port(&mut it, line, false)?),
                "OUTPUT" => plan.outputs.push(port(&mut it, line, true)?),
                "GOLDEN" => {
                    let is_input = match it.next() {
                        Some("INPUT") => true,
                        Some("OUTPUT") => false,
                        _ => return Err(bad(line)),
                    };
                    let name = it.next().ok_or_else(|| bad(line))?.to_owned();
                    golden.insert((is_input, name), (num(it.next(), line)?, num(it.next(), line)?));
                }
                "W" => plan.weights.push(Region {
                    id: num(it.next(), line)?,
                    off: num(it.next(), line)?,
                    size: num(it.next(), line)?,
                }),
                "S" => plan.scratch.push(Region { id: num(it.next(), line)?, off: 0, size: num(it.next(), line)? }),
                "H" => plan.host.push(Region { id: num(it.next(), line)?, off: 0, size: num(it.next(), line)? }),
                "PROG" => plan.programs.push(num(it.next(), line)?),
                "KERN" => {
                    let kid = num(it.next(), line)?;
                    let prog = num(it.next(), line)?;
                    let name = it.next().ok_or_else(|| bad(line))?.to_owned();
                    kernels.insert(kid, (prog, name));
                }
                "NDR" => {
                    let kid: usize = num(it.next(), line)?;
                    let dim = num(it.next(), line)?;
                    // The replayer hands clEnqueueNDRangeKernel three-element
                    // arrays, so a larger dim would read past them.
                    if !(1..=3).contains(&dim) {
                        return Err(format!("launch with {dim} dimensions: {line}"));
                    }
                    let global = triple(&mut it, line)?;
                    let local = triple(&mut it, line)?;
                    let offset = triple(&mut it, line)?;
                    let nargs: usize = num(it.next(), line)?;
                    let (program, name) =
                        kernels.get(&kid).cloned().ok_or_else(|| format!("launch of unknown kernel {kid}"))?;
                    let args = (0..nargs)
                        .map(|_| lines.next().ok_or_else(|| "launch truncated".to_owned()).and_then(arg))
                        .collect::<Result<_, _>>()?;
                    plan.launches.push(LaunchPlan { program, name, dim, global, local, offset, args });
                }
                other => return Err(format!("unknown record {other}")),
            }
        }
        if format.as_deref() != Some(FORMAT) {
            return Err(format!("not an {FORMAT} recording (FORMAT {})", format.unwrap_or_default()));
        }
        if plan.device_name.is_empty() || plan.inputs.is_empty() || plan.outputs.is_empty() || plan.launches.is_empty()
        {
            return Err("recording is missing its device, inputs, outputs or launches".into());
        }
        for (is_input, ports) in [(true, &mut plan.inputs), (false, &mut plan.outputs)] {
            for p in ports {
                let (off, bytes) = golden
                    .get(&(is_input, p.name.clone()))
                    .copied()
                    .ok_or_else(|| format!("no golden data for {}", p.name))?;
                if bytes != p.bytes {
                    return Err(format!("golden data for {} is {bytes} bytes, port is {}", p.name, p.bytes));
                }
                p.golden = off;
            }
        }
        Ok(plan)
    }

    pub fn max_alloc_id(&self) -> usize {
        self.weights.iter().chain(&self.scratch).chain(&self.host).map(|r| r.id).max().unwrap_or(0)
    }

    /// Where each scratch allocation lives inside one shared arena, and the
    /// arena's size. Scratch allocations whose lifetimes do not overlap share
    /// memory: an allocation is live from the first launch that uses it to
    /// the last, and all of the run if it holds an input or an output. The
    /// placement is greedy, largest first, each at the lowest aligned offset
    /// that does not collide with an already placed allocation live at the
    /// same time. Offsets are in the order of [`Plan::scratch`].
    #[must_use]
    pub fn scratch_layout(&self, align: usize) -> (Vec<usize>, usize) {
        let align = align.max(1);
        let index: HashMap<usize, usize> = self.scratch.iter().enumerate().map(|(i, r)| (r.id, i)).collect();
        // Launch indices; inputs are written before launch 0 and outputs
        // read after the last one.
        let end = self.launches.len();
        let mut live: Vec<Option<(usize, usize)>> = vec![None; self.scratch.len()];
        let mut touch = |alloc: usize, at: (usize, usize)| {
            if let Some(slot) = index.get(&alloc).and_then(|&i| live.get_mut(i)) {
                *slot = Some(slot.map_or(at, |(a, b)| (a.min(at.0), b.max(at.1))));
            }
        };
        for (i, launch) in self.launches.iter().enumerate() {
            for arg in &launch.args {
                if let Arg::Ptr { alloc, .. } = arg {
                    touch(*alloc, (i, i));
                }
            }
        }
        for port in &self.inputs {
            touch(port.alloc, (0, 0));
        }
        for port in &self.outputs {
            touch(port.alloc, (end, end));
        }

        let round = |n: usize| n.div_ceil(align).saturating_mul(align);
        let mut order: Vec<usize> = (0..self.scratch.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.scratch.get(i).map_or(0, |r| r.size)));
        let mut offsets = vec![0; self.scratch.len()];
        // (offset, size, lifetime) of every allocation placed so far.
        let mut placed: Vec<(usize, usize, (usize, usize))> = Vec::new();
        let mut total = 0usize;
        for i in order {
            let (Some(region), Some(Some(when))) = (self.scratch.get(i), live.get(i).copied()) else {
                continue; // never used: offset 0 is as good as any
            };
            let size = round(region.size);
            let mut clashes: Vec<(usize, usize)> = placed
                .iter()
                .filter(|p| p.2.0 <= when.1 && when.0 <= p.2.1)
                .map(|p| (p.0, p.0.saturating_add(p.1)))
                .collect();
            clashes.sort_unstable();
            let mut at = 0usize;
            for (start, stop) in clashes {
                if at.saturating_add(size) <= start {
                    break;
                }
                at = at.max(stop);
            }
            if let Some(o) = offsets.get_mut(i) {
                *o = at;
            }
            placed.push((at, size, when));
            total = total.max(at.saturating_add(size));
        }
        (offsets, total)
    }

    pub fn build_programs(&self, ctx: &cl::Context, dir: &Path) -> Result<HashMap<usize, cl::Program>, Error> {
        let mut out = HashMap::with_capacity(self.programs.len());
        for &p in &self.programs {
            let path = dir.join(format!("prog_{p}.bin"));
            let binary = std::fs::read(&path).map_err(|source| Error::Io { path, source })?;
            out.insert(p, ctx.program_from_binary(&binary)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "FORMAT ovreplay 1
DEVICE_NAME Intel(R) Arc(TM) Graphics
DRIVER_VERSION 26.35.39758
DEVICE_ID 32081
INPUT frame 2 0 128
OUTPUT score 5 0 4 f32
GOLDEN INPUT frame 0 128
GOLDEN OUTPUT score 128 4
W 1 0 64
S 2 128 1
H 5 4
PROG 3 100
KERN 4 3 gen_conv
NDR 4 3 16 1 1 16 1 1 0 0 0 4
 U 1 0
 V 4 00 00 80 3f
 L 256
 N
";

    #[test]
    fn parses_a_recording() {
        let p = Plan::parse(SAMPLE).unwrap();
        assert_eq!(p.device_name, "Intel(R) Arc(TM) Graphics");
        assert_eq!(p.driver_version, "26.35.39758");
        assert_eq!(p.device_id, Some(32081));
        assert_eq!(p.inputs[0].name, "frame");
        assert_eq!(p.inputs[0].golden, 0);
        assert_eq!(p.outputs[0].element_type, "f32");
        assert_eq!(p.outputs[0].golden, 128);
        assert_eq!(p.launches[0].name, "gen_conv");
        let args = &p.launches[0].args;
        assert!(matches!(args[0], Arg::Ptr { alloc: 1, off: 0 }));
        assert!(matches!(args[1], Arg::Value(ref b) if b == &[0, 0, 0x80, 0x3f]));
        assert!(matches!(args[2], Arg::Local(256)));
        assert!(matches!(args[3], Arg::Null));
        assert_eq!(p.max_alloc_id(), 5);
    }

    #[test]
    fn refuses_other_formats_unknown_records_and_short_values() {
        assert!(Plan::parse(&SAMPLE.replace("ovreplay 1", "ovreplay 2")).is_err());
        assert!(Plan::parse(&SAMPLE.replace("PROG 3 100", "MAP 3")).is_err());
        assert!(Plan::parse(&SAMPLE.replace(" V 4 00 00 80 3f", " V 4 00 00")).is_err());
        assert!(Plan::parse(&SAMPLE.replace(" N\n", " M 3\n")).is_err());
        assert!(Plan::parse(&SAMPLE.replace(" N\n", "")).is_err());
        assert!(Plan::parse(&SAMPLE.replace("NDR 4 3 ", "NDR 4 4 ")).is_err());
        assert!(Plan::parse(&SAMPLE.replace("NDR 4 3 ", "NDR 4 0 ")).is_err());
    }

    fn launch(allocs: &[usize]) -> LaunchPlan {
        LaunchPlan {
            program: 0,
            name: "k".into(),
            dim: 1,
            global: [1, 1, 1],
            local: [0, 0, 0],
            offset: [0, 0, 0],
            args: allocs.iter().map(|&alloc| Arg::Ptr { alloc, off: 0 }).collect(),
        }
    }

    fn chain(sizes: &[usize], launches: Vec<LaunchPlan>) -> Plan {
        let mut p = Plan::parse(SAMPLE).unwrap();
        p.scratch = sizes.iter().enumerate().map(|(i, &size)| Region { id: 100 + i, off: 0, size }).collect();
        p.inputs.clear();
        p.outputs.clear();
        p.launches = launches;
        p
    }

    #[test]
    fn scratch_that_is_never_live_together_shares_memory() {
        // a -> b -> c: a and c are never live at the same time as each other
        // once b has consumed a, so c can reuse a's memory.
        let p = chain(&[1000, 500, 1000], vec![launch(&[100, 101]), launch(&[101, 102])]);
        let (off, total) = p.scratch_layout(256);
        assert_eq!(off[0], off[2], "{off:?}");
        assert_ne!(off[1], off[0]);
        assert_eq!(total, 1024 + 512);
    }

    #[test]
    fn scratch_live_together_never_overlaps() {
        let p = chain(&[300, 300, 300], vec![launch(&[100, 101, 102])]);
        let (off, total) = p.scratch_layout(256);
        let mut spans: Vec<(usize, usize)> = off.iter().map(|&o| (o, o + 512)).collect();
        spans.sort_unstable();
        assert!(spans.windows(2).all(|w| w[0].1 <= w[1].0), "{spans:?}");
        assert_eq!(total, 3 * 512);
    }

    #[test]
    fn inputs_and_outputs_live_for_the_whole_run() {
        let mut p = chain(&[256, 256], vec![launch(&[100]), launch(&[101])]);
        let (off, _) = p.scratch_layout(256);
        assert_eq!(off[0], off[1], "disjoint lifetimes share");
        let port = |alloc| Port { name: "x".into(), alloc, off: 0, bytes: 4, element_type: "f32".into(), golden: 0 };
        p.outputs.push(port(100));
        let (off, _) = p.scratch_layout(256);
        assert_ne!(off[0], off[1], "an output stays live to the end");
    }

    #[test]
    fn requires_golden_data_of_the_right_size() {
        assert!(Plan::parse(&SAMPLE.replace("GOLDEN OUTPUT score 128 4\n", "")).is_err());
        assert!(Plan::parse(&SAMPLE.replace("GOLDEN INPUT frame 0 128", "GOLDEN INPUT frame 0 64")).is_err());
        assert!(Plan::parse(&SAMPLE.replace("GOLDEN INPUT", "GOLDEN SIDEWAYS")).is_err());
    }
}
