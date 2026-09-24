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
    }

    #[test]
    fn requires_golden_data_of_the_right_size() {
        assert!(Plan::parse(&SAMPLE.replace("GOLDEN OUTPUT score 128 4\n", "")).is_err());
        assert!(Plan::parse(&SAMPLE.replace("GOLDEN INPUT frame 0 128", "GOLDEN INPUT frame 0 64")).is_err());
        assert!(Plan::parse(&SAMPLE.replace("GOLDEN INPUT", "GOLDEN SIDEWAYS")).is_err());
    }
}
