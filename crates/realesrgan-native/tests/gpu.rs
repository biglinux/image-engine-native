//! The model on a GPU against the CPU engine, on a directory of `NAME.rgb`
//! photos of one size (see "Checking results" in README.md):
//!
//!     REALESRGAN_NATIVE_WEIGHTS=weights/realesr-general-x4v3-dn50.weights \
//!     REALESRGAN_NATIVE_REFERENCE=corpus REALESRGAN_NATIVE_SIZE=250x167 \
//!     cargo test --release --test gpu -- --ignored

use std::ops::ControlFlow;
use std::path::PathBuf;

use realesrgan_native::{MARGIN, Model};

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn to_8bit(v: f32) -> f64 {
    (v.clamp(0.0, 1.0) * 255.0).round().into()
}

#[test]
#[ignore = "needs a Vulkan GPU, weights and photos"]
fn gpu_matches_cpu_engine() {
    let weights = PathBuf::from(var("REALESRGAN_NATIVE_WEIGHTS"));
    let dir = PathBuf::from(var("REALESRGAN_NATIVE_REFERENCE"));
    let size = var("REALESRGAN_NATIVE_SIZE");
    let (width, height) = size.split_once('x').expect("SIZE is WIDTHxHEIGHT");
    let (width, height): (usize, usize) = (width.parse().unwrap(), height.parse().unwrap());
    let mut gpu = Model::load(&weights, 0).unwrap();
    assert!(
        gpu.gpu().is_some(),
        "no usable GPU (IMAGE_ENGINE_NATIVE_GPU=1 says why)"
    );
    let mut cpu = Model::load_cpu(&weights, 0).unwrap();
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "rgb") {
            continue;
        }
        let image: Vec<f32> = std::fs::read(&path)
            .unwrap()
            .iter()
            .map(|&v| f32::from(v) / 255.0)
            .collect();
        let got = gpu.run_float(&image, width, height).unwrap();
        let want = cpu.run_float(&image, width, height).unwrap();
        // Half floats move a share of 8-bit values by a step or a few: 52 to
        // 57 dB on the 20 test photos, with the same PSNR against the
        // originals to within 0.02 dB.
        let squares: f64 = got
            .iter()
            .zip(&want)
            .map(|(&g, &w)| (to_8bit(g) - to_8bit(w)).powi(2))
            .sum();
        let psnr = 10.0 * (255.0f64.powi(2) * want.len() as f64 / squares).log10();
        assert!(psnr > 48.0, "{}: {psnr:.1} dB", path.display());
        // Every output pixel is computed the same way wherever its tile is.
        let tiled = gpu
            .run_float_tiled(&image, width, height, 64, MARGIN, |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        assert!(
            tiled
                .iter()
                .zip(&got)
                .all(|(t, g)| t.to_bits() == g.to_bits()),
            "{}: tiles differ from the whole image",
            path.display()
        );
        checked += 1;
    }
    assert!(checked > 0, "no NAME.rgb in {}", dir.display());
}
