//! The model on a GPU against the CPU engine, on a directory of `NAME.rgb`
//! photos and their `NAME.mask`, of one size (see "Checking results" in
//! README.md):
//!
//!     LAMA_NATIVE_WEIGHTS=$PWD/weights/big-lama-f16.weights \
//!     LAMA_NATIVE_REFERENCE=corpus LAMA_NATIVE_SIZE=512x512 \
//!     cargo test --release --test gpu -- --ignored

use std::path::PathBuf;

use lama_native::Model;

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

/// PSNR in 8 bits of `got` against the photo where the mask is set.
fn psnr_in_holes(got: &[f32], photo: &[u8], mask: &[u8]) -> f64 {
    let mut squares = 0.0;
    let mut n = 0;
    for (p, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        for c in 0..3 {
            let v = (got[p * 3 + c].clamp(0.0, 1.0) * 255.0).round();
            squares += (f64::from(v) - f64::from(photo[p * 3 + c])).powi(2);
            n += 1;
        }
    }
    10.0 * (255.0f64.powi(2) * f64::from(n) / squares).log10()
}

#[test]
#[ignore = "needs a Vulkan GPU, weights and photos"]
fn gpu_matches_cpu_engine() {
    let weights = PathBuf::from(var("LAMA_NATIVE_WEIGHTS"));
    let dir = PathBuf::from(var("LAMA_NATIVE_REFERENCE"));
    let size = var("LAMA_NATIVE_SIZE");
    let (width, height) = size.split_once('x').expect("SIZE is WIDTHxHEIGHT");
    let (width, height): (usize, usize) = (width.parse().unwrap(), height.parse().unwrap());
    let mut gpu = Model::load(&weights, 0).unwrap();
    assert!(
        gpu.gpu().is_some(),
        "no usable GPU (IMAGE_ENGINE_NATIVE_GPU=1 says why)"
    );
    let mut cpu = Model::load_cpu(&weights, 0).unwrap();
    let (mut sum_gpu, mut sum_cpu, mut checked) = (0.0, 0.0, 0);
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "rgb") {
            continue;
        }
        let photo = std::fs::read(&path).unwrap();
        let mask = std::fs::read(path.with_extension("mask")).unwrap();
        let image: Vec<f32> = photo.iter().map(|&v| f32::from(v) / 255.0).collect();
        let holes: Vec<f32> = mask.iter().map(|&m| f32::from(u8::from(m != 0))).collect();
        let got = gpu.run_float(&image, &holes, width, height).unwrap();
        let want = cpu.run_float(&image, &holes, width, height).unwrap();
        assert_eq!(got.len(), want.len(), "{}: output size", path.display());
        // LaMa invents what fills the holes, and small differences grow into
        // other plausible fills: compare each with the photo itself. On the
        // 20 test photos half floats move a photo's PSNR by 0.15 dB at most
        // at 512 x 512, and the mean by 0.002 dB; at 1024 x 1024 by 0.42 and
        // 0.03 dB on the Vega 7.
        let (g, c) = (
            psnr_in_holes(&got, &photo, &mask),
            psnr_in_holes(&want, &photo, &mask),
        );
        assert!(
            (g - c).abs() < 0.5,
            "{}: {g:.2} dB against {c:.2}",
            path.display()
        );
        (sum_gpu, sum_cpu, checked) = (sum_gpu + g, sum_cpu + c, checked + 1);
    }
    assert!(checked > 0, "no NAME.rgb in {}", dir.display());
    let (g, c) = (sum_gpu / f64::from(checked), sum_cpu / f64::from(checked));
    assert!((g - c).abs() < 0.05, "mean {g:.3} dB against {c:.3}");
}
