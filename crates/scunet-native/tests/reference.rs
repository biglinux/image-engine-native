//! The model against PyTorch's own outputs, and its progress reports.
//!
//! The reference test needs float32 weights and a directory of `NAME.rgb`
//! inputs with `NAME.torch.out` references, all of one size (see "Checking
//! results" in README.md); the progress test needs only weights:
//!
//!     SCUNET_NATIVE_WEIGHTS=weights/scunet-color-real-psnr.weights \
//!     SCUNET_NATIVE_REFERENCE=corpus SCUNET_NATIVE_SIZE=768x512 \
//!     cargo test --release -- --ignored

use std::ops::ControlFlow;
use std::path::PathBuf;

use scunet_native::{Error, Model};

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn to_8bit(v: f32) -> i32 {
    (v.clamp(0.0, 1.0) * 255.0).round() as i32
}

#[test]
#[ignore = "needs weights and reference outputs"]
fn matches_pytorch_reference() {
    let weights = PathBuf::from(var("SCUNET_NATIVE_WEIGHTS"));
    let dir = PathBuf::from(var("SCUNET_NATIVE_REFERENCE"));
    let size = var("SCUNET_NATIVE_SIZE");
    let (width, height) = size.split_once('x').expect("SIZE is WIDTHxHEIGHT");
    let (width, height): (usize, usize) = (width.parse().unwrap(), height.parse().unwrap());
    // The GPU computes in half floats; this bound is the CPU engine's.
    let mut model = Model::load_cpu(&weights, 0).unwrap();
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "rgb") {
            continue;
        }
        let Ok(bytes) = std::fs::read(path.with_extension("torch.out")) else {
            continue;
        };
        let want: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        let image: Vec<f32> = std::fs::read(&path)
            .unwrap()
            .iter()
            .map(|&v| f32::from(v) / 255.0)
            .collect();
        let got = model.run_float(&image, width, height).unwrap();
        assert_eq!(got.len(), want.len(), "{}: output size", path.display());
        // A float32 run of the same network: within 1e-5 (rounding differences
        // reach 2e-6), and no 8-bit value more than one step away.
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        let worst_8bit = got
            .iter()
            .zip(&want)
            .map(|(&g, &w)| (to_8bit(g) - to_8bit(w)).abs())
            .max()
            .unwrap();
        assert!(
            worst < 1e-5 && worst_8bit <= 1,
            "{}: difference {worst}, 8-bit {worst_8bit}",
            path.display()
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no NAME.rgb with NAME.torch.out in {}",
        dir.display()
    );
}

#[test]
#[ignore = "needs weights"]
fn progress_reaches_one_and_cancel_stops() {
    let mut model = Model::load(&PathBuf::from(var("SCUNET_NATIVE_WEIGHTS")), 0).unwrap();
    let image = vec![0.5f32; 128 * 96 * 3];
    // Four 64 x 48 tiles of 28 blocks each.
    let mut seen = Vec::new();
    model
        .run_float_tiled(&image, 128, 96, 64, 16, |f| {
            seen.push(f);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(seen.len(), 4 * 28);
    assert!(seen.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(seen.last(), Some(&1.0));

    let mut calls = 0;
    let result = model.run_float_tiled(&image, 128, 96, 64, 16, |f| {
        calls += 1;
        if f >= 0.5 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(calls, 2 * 28);
}
