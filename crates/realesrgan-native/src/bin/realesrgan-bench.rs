//! Run Real-ESRGAN on raw RGB images and time it.
//!
//!     realesrgan-bench WEIGHTS WIDTH HEIGHT [--threads N] [--repeat N] [--tag TAG]
//!                  [--tile N --margin N] IMAGE.rgb...
//!
//! Each IMAGE.rgb holds WIDTH x HEIGHT pixels of 8-bit RGB, row by row
//! (`tools/realesrgan/evaluate.py prepare` writes them). With `--tag`, the
//! network's output, four times larger on each side, goes to IMAGE.TAG.out as
//! little-endian f32 RGB rows, neither clamped nor rounded. With `--tile`, the image runs in squares of that side with
//! `--margin` pixels of context (see `Model::run_float_tiled`). Prints the
//! median time of `--repeat` runs per image and the process's peak resident
//! memory. The model runs on a GPU when there is one; `IMAGE_ENGINE_NATIVE_GPU=0`
//! keeps it on the CPU (see `Model::load`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use realesrgan_native::Model;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("realesrgan-bench: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: realesrgan-bench WEIGHTS WIDTH HEIGHT [--threads N] [--repeat N] [--tag TAG] [--tile N --margin N] IMAGE.rgb...";
    let weights = PathBuf::from(args.next().ok_or(usage)?);
    let width: usize = args.next().ok_or(usage)?.parse()?;
    let height: usize = args.next().ok_or(usage)?.parse()?;
    let (mut threads, mut repeat, mut tag, mut images) = (0, 1, None, Vec::new());
    let (mut tile, mut margin) = (None, 0);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => threads = args.next().ok_or(usage)?.parse()?,
            "--repeat" => repeat = args.next().ok_or(usage)?.parse::<usize>()?.max(1),
            "--tag" => tag = Some(args.next().ok_or(usage)?),
            "--tile" => tile = Some(args.next().ok_or(usage)?.parse()?),
            "--margin" => margin = args.next().ok_or(usage)?.parse()?,
            _ => images.push(PathBuf::from(arg)),
        }
    }
    let start = Instant::now();
    let mut model = Model::load(&weights, threads)?;
    let device = match model.gpu() {
        Some(gpu) => format!("GPU {gpu}"),
        None => format!("CPU, SIMD tier {}", realesrgan_native::simd_tier()),
    };
    println!(
        "loaded in {:.3}s on {device}",
        start.elapsed().as_secs_f64()
    );
    for image in &images {
        let rgb = std::fs::read(image).map_err(|e| format!("{}: {e}", image.display()))?;
        let input: Vec<f32> = rgb.iter().map(|&v| f32::from(v) / 255.0).collect();
        let mut times = Vec::new();
        let mut out = Vec::new();
        for _ in 0..repeat {
            let start = Instant::now();
            out = match tile {
                Some(tile) => model.run_float_tiled(&input, width, height, tile, margin, |_| {
                    std::ops::ControlFlow::Continue(())
                })?,
                None => model.run_float(&input, width, height)?,
            };
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
