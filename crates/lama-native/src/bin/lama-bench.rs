//! Run LaMa on raw RGB images and time it.
//!
//!     lama-bench WEIGHTS WIDTH HEIGHT [--threads N] [--repeat N] [--tag TAG] IMAGE.rgb...
//!
//! Each IMAGE.rgb holds WIDTH x HEIGHT pixels of 8-bit RGB, row by row, and
//! IMAGE.mask next to it one byte per pixel, not zero where the image is to be
//! filled (`tools/lama/evaluate.py prepare` writes both). With `--tag`, the
//! network's prediction for every pixel goes to IMAGE.TAG.out as little-endian
//! f32 RGB rows, neither clamped nor rounded. Prints the median time of
//! `--repeat` runs per image and the process's peak resident memory. The
//! model runs on a GPU when there is one; `IMAGE_ENGINE_NATIVE_GPU=0` keeps it on the
//! CPU (see `Model::load`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use lama_native::Model;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lama-bench: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: lama-bench WEIGHTS WIDTH HEIGHT [--threads N] [--repeat N] [--tag TAG] IMAGE.rgb...";
    let weights = PathBuf::from(args.next().ok_or(usage)?);
    let width: usize = args.next().ok_or(usage)?.parse()?;
    let height: usize = args.next().ok_or(usage)?.parse()?;
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
        None => format!("CPU, SIMD tier {}", lama_native::simd_tier()),
    };
    println!(
        "loaded in {:.3}s on {device}",
        start.elapsed().as_secs_f64()
    );
    for image in &images {
        let rgb = std::fs::read(image).map_err(|e| format!("{}: {e}", image.display()))?;
        let input: Vec<f32> = rgb.iter().map(|&v| f32::from(v) / 255.0).collect();
        let mask_path = image.with_extension("mask");
        let mask =
            std::fs::read(&mask_path).map_err(|e| format!("{}: {e}", mask_path.display()))?;
        let mask: Vec<f32> = mask.iter().map(|&m| f32::from(u8::from(m != 0))).collect();
        let mut times = Vec::new();
        let mut out = Vec::new();
        for _ in 0..repeat {
            let start = Instant::now();
            out = model.run_float(&input, &mask, width, height)?;
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        println!("{} {:.3}s", stem(image), times[times.len() / 2]);
        if let Some(tag) = &tag {
            let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(image.with_extension(format!("{tag}.out")), bytes)?;
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
