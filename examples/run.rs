//! Load a recording, run it, and print its outputs and timings.
//!
//! ```text
//! ovreplay-run DIR [--input NAME=FILE ...] [--bench N] [--profile]
//! ```
//!
//! Each `--input` file holds the raw bytes of one input. With no `--input`,
//! the recorded inference's own inputs are replayed and the outputs checked
//! against the recorded ones; the exit status is 1 if they differ. `--bench`
//! sets the number of timed runs after the first (default 10). `--profile`
//! then times every launch on the GPU over five runs and prints the median
//! time per kernel family and the costliest launches.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use ovreplay::{Error, Replay};

struct Args {
    dir: String,
    inputs: Vec<(String, String)>,
    bench: usize,
    profile: bool,
}

fn usage() -> String {
    "usage: ovreplay-run DIR [--input NAME=FILE ...] [--bench N] [--profile]".to_owned()
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let (mut dir, mut inputs, mut bench, mut profile) = (None, Vec::new(), 10, false);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => {
                let v = it.next().ok_or_else(usage)?;
                let (name, file) = v.split_once('=').ok_or_else(|| format!("--input wants NAME=FILE, got {v}"))?;
                inputs.push((name.to_owned(), file.to_owned()));
            }
            "--bench" => bench = it.next().and_then(|n| n.parse().ok()).ok_or_else(usage)?,
            "--profile" => profile = true,
            "-h" | "--help" => return Err(usage()),
            _ if dir.is_none() && !a.starts_with('-') => dir = Some(a),
            _ => return Err(usage()),
        }
    }
    Ok(Args { dir: dir.ok_or_else(usage)?, inputs, bench, profile })
}

/// Time since this process was exec'd, from /proc (10 ms resolution).
fn since_exec() -> Option<Duration> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Field 22, counting from the pid; the command name may contain spaces.
    let start_ticks: f64 = stat.rsplit_once(')')?.1.split_whitespace().nth(19)?.parse().ok()?;
    let uptime: f64 = std::fs::read_to_string("/proc/uptime").ok()?.split_whitespace().next()?.parse().ok()?;
    Duration::try_from_secs_f64(uptime - start_ticks / 100.0).ok()
}

fn ms(d: Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1e3)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 if frac == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// The first few values of an output, decoded by element type.
fn preview(bytes: &[u8], element_type: &str) -> String {
    const N: usize = 6;
    fn first<const K: usize>(bytes: &[u8], show: impl Fn([u8; K]) -> String) -> String {
        bytes.as_chunks::<K>().0.iter().take(N).map(|c| show(*c)).collect::<Vec<_>>().join(" ")
    }
    match element_type {
        "f32" => first(bytes, |c| format!("{:.5}", f32::from_le_bytes(c))),
        "f16" => first(bytes, |c| format!("{:.4}", f16_to_f32(u16::from_le_bytes(c)))),
        "i32" => first(bytes, |c| i32::from_le_bytes(c).to_string()),
        "i64" => first(bytes, |c| i64::from_le_bytes(c).to_string()),
        "u8" | "boolean" => first(bytes, |[b]: [u8; 1]| b.to_string()),
        _ => first(bytes, |[b]: [u8; 1]| format!("{b:02x}")),
    }
}

fn run(args: &Args) -> Result<bool, Error> {
    let t = Instant::now();
    let mut replay = Replay::load(&args.dir)?;
    let load = t.elapsed();
    let ready = since_exec();
    println!(
        "load       {}{}",
        ms(load),
        ready.map_or_else(String::new, |d| format!(" (ready {} after exec)", ms(d)))
    );

    let mut passed = true;
    let t = Instant::now();
    if args.inputs.is_empty() {
        match replay.verify() {
            Ok(()) => println!("verify     pass, outputs bit-identical to the recording ({})", ms(t.elapsed())),
            Err(e @ Error::GoldenMismatch { .. }) => {
                println!("verify     FAIL: {e}");
                passed = false;
            }
            Err(e) => return Err(e),
        }
    } else {
        let given: Vec<&str> = args.inputs.iter().map(|(n, _)| n.as_str()).collect();
        if let Some((missing, _)) = replay.inputs().find(|(n, _)| !given.contains(n)) {
            return Err(Error::NoInput(format!("{missing} (pass --input {missing}=FILE)")));
        }
        for (name, file) in &args.inputs {
            let data = std::fs::read(file).map_err(|source| Error::Io { path: file.into(), source })?;
            replay.set_input(name, &data)?;
        }
        replay.run()?;
        println!("first run  {}", ms(t.elapsed()));
    }

    let mut times = Vec::with_capacity(args.bench);
    for _ in 0..args.bench {
        let t = Instant::now();
        replay.run()?;
        times.push(t.elapsed());
    }
    times.sort_unstable();
    if let (Some(min), Some(median)) = (times.first(), times.get(times.len() / 2)) {
        println!("steady     median {}, min {} over {} runs", ms(*median), ms(*min), times.len());
    }

    if args.profile {
        print_profile(&mut replay)?;
    }

    let outputs: Vec<(String, usize, String)> =
        replay.outputs().map(|(n, b, t)| (n.to_owned(), b, t.to_owned())).collect();
    for (name, bytes, element_type) in outputs {
        println!("output     {name}  {element_type}  {bytes} B  [{}]", preview(replay.output(&name)?, &element_type));
    }
    Ok(passed)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("ovreplay-run: {e}");
            ExitCode::FAILURE
        }
    }
}

/// A kernel name without the hash and index suffixes `OpenVINO` appends:
/// `reorder_data_7009384538133017897_0_0` is `reorder_data`.
fn family(name: &str) -> &str {
    let mut end = name.len();
    for part in name.rsplit('_') {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            break;
        }
        end = end.saturating_sub(part.len() + 1);
    }
    name.get(..end).unwrap_or(name)
}

fn print_profile(replay: &mut Replay) -> Result<(), Error> {
    let runs: Vec<Vec<(String, Duration)>> = (0..5).map(|_| replay.profile()).collect::<Result<_, _>>()?;
    let Some(first) = runs.first() else { return Ok(()) };
    // Median over the runs, per launch.
    let launches: Vec<(&str, Duration)> = (0..first.len())
        .map(|i| {
            let mut t: Vec<Duration> = runs.iter().filter_map(|r| r.get(i).map(|x| x.1)).collect();
            t.sort_unstable();
            (first.get(i).map_or("", |x| x.0.as_str()), t.get(t.len() / 2).copied().unwrap_or_default())
        })
        .collect();
    let total: Duration = launches.iter().map(|l| l.1).sum();
    let mut families: Vec<(&str, usize, Duration)> = Vec::new();
    for &(name, t) in &launches {
        let f = family(name);
        match families.iter_mut().find(|x| x.0 == f) {
            Some(x) => {
                x.1 += 1;
                x.2 += t;
            }
            None => families.push((f, 1, t)),
        }
    }
    families.sort_by_key(|x| std::cmp::Reverse(x.2));
    println!("profile    {} launches, {} of GPU time (median of 5 runs)", launches.len(), ms(total));
    let share = |t: Duration| 100.0 * t.as_secs_f64() / total.as_secs_f64().max(f64::MIN_POSITIVE);
    for (f, n, t) in &families {
        println!("  {:>9}  {:>5.1}%  {n:>4}x  {f}", ms(*t), share(*t));
    }
    let mut top: Vec<(usize, &str, Duration)> = launches.iter().enumerate().map(|(i, l)| (i, l.0, l.1)).collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.2));
    println!("costliest launches:");
    for (i, name, t) in top.iter().take(12) {
        println!("  {:>9}  #{i:<4} {name}", ms(*t));
    }
    Ok(())
}
