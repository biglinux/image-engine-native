//! Run BiRefNet-lite on raw RGB images and time it.
//!
//!     birefnet-bench WEIGHTS SIDE [--threads N] [--repeat N] [--tag TAG] IMAGE.rgb...
//!
//! Each IMAGE.rgb holds SIDE x SIDE pixels of 8-bit RGB, row by row
//! (`tools/birefnet/evaluate.py prepare` writes them). With `--tag`, the logits go to
//! IMAGE.TAG.logits as little-endian f32 rows. Prints the median time of
//! `--repeat` runs per image and the process's peak resident memory.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use birefnet_native::Model;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("birefnet-bench: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage =
        "usage: birefnet-bench WEIGHTS SIDE [--threads N] [--repeat N] [--tag TAG] IMAGE.rgb...";
    let weights = PathBuf::from(args.next().ok_or(usage)?);
    let side: usize = args.next().ok_or(usage)?.parse()?;
    let (mut threads, mut repeat, mut tag, mut images) = (0, 1, None, Vec::new());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => threads = args.next().ok_or(usage)?.parse()?,
            "--repeat" => repeat = args.next().ok_or(usage)?.parse::<usize>()?.max(1),
            "--tag" => tag = Some(args.next().ok_or(usage)?),
            _ => images.push(PathBuf::from(arg)),
        }
    }
    let start = Instant::now();
    let mut model = Model::load(&weights, threads)?;
    let device = match model.gpu() {
        Some(gpu) => format!("GPU {gpu}"),
        None => format!("CPU, SIMD tier {}", birefnet_native::simd_tier()),
    };
    println!(
        "loaded in {:.3}s on {device}",
        start.elapsed().as_secs_f64()
    );
    for image in &images {
        let rgb = std::fs::read(image).map_err(|e| format!("{}: {e}", image.display()))?;
        let mut times = Vec::new();
        let mut logits = Vec::new();
        for _ in 0..repeat {
            let start = Instant::now();
            logits = model.run(&rgb, side)?;
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        println!("{} {:.3}s", stem(image), times[times.len() / 2]);
        if let Some(tag) = &tag {
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(image.with_extension(format!("{tag}.logits")), bytes)?;
        }
    }
    if let Ok(status) = std::fs::read_to_string("/proc/self/status")
        && let Some(line) = status.lines().find(|l| l.starts_with("VmHWM"))
    {
        println!("peak {}", line.trim_start_matches("VmHWM:").trim());
    }
    Ok(())
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned())
}
