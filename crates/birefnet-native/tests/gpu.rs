//! The model on a GPU against the CPU engine, on a directory of `NAME.rgb`
//! photos of one size (see "Checking results" in README.md):
//!
//!     BIREFNET_NATIVE_WEIGHTS=$PWD/weights/birefnet-lite-f16.weights \
//!     BIREFNET_NATIVE_REFERENCE=corpus cargo test --release --test gpu -- --ignored

use std::path::PathBuf;

use birefnet_native::Model;

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn alpha(logit: f32) -> i32 {
    (255.0 / (1.0 + (-f64::from(logit)).exp())).round() as i32
}

#[test]
#[ignore = "needs a Vulkan GPU, weights and photos"]
fn gpu_matches_cpu_engine() {
    let weights = PathBuf::from(var("BIREFNET_NATIVE_WEIGHTS"));
    let dir = PathBuf::from(var("BIREFNET_NATIVE_REFERENCE"));
    let mut gpu = Model::load(&weights, 0).unwrap();
    assert!(
        gpu.gpu().is_some(),
        "no usable GPU (IMAGE_ENGINE_NATIVE_GPU=1 says why)"
    );
    let mut cpu = Model::load_cpu(&weights, 0).unwrap();
    let (mut ious, mut moves) = (Vec::new(), Vec::new());
    let (mut border, mut border_off) = (0, 0);
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "rgb") {
            continue;
        }
        let rgb = std::fs::read(&path).unwrap();
        let side = ((rgb.len() / 3) as f64).sqrt() as usize;
        let got: Vec<i32> = gpu
            .run(&rgb, side)
            .unwrap()
            .into_iter()
            .map(alpha)
            .collect();
        let want: Vec<i32> = cpu
            .run(&rgb, side)
            .unwrap()
            .into_iter()
            .map(alpha)
            .collect();
        let (mut both, mut either, mut off, mut moved) = (0, 0, 0, 0);
        for (&g, &w) in got.iter().zip(&want) {
            both += usize::from(g >= 128 && w >= 128);
            either += usize::from(g >= 128 || w >= 128);
            off += usize::from((g - w).abs() > 25);
            moved += (g - w).unsigned_abs() as usize;
        }
        let iou = if either == 0 {
            1.0
        } else {
            both as f64 / either as f64
        };
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let (y, x) = (i / side, i % side);
            if y == 0 || x == 0 || y == side - 1 || x == side - 1 {
                border += 1;
                border_off += usize::from((g - w).abs() > 25);
            }
        }
        let off = off as f64 / got.len() as f64;
        // Half floats move the masks' edges: IoU 0.997 to 1 on most of 46
        // test photos. On one or two per GPU, a different one on each, the
        // rounding tips an object the model is unsure of: IoU 0.92 to 0.98,
        // up to 1.7 % of the pixels more than 25 steps away.
        assert!(
            iou > 0.9 && off < 0.025,
            "{}: IoU {iou:.4}, {:.2} % of pixels off",
            path.display(),
            off * 100.0
        );
        ious.push(iou);
        moves.push(moved as f64 / got.len() as f64);
    }
    assert!(!ious.is_empty(), "no NAME.rgb in {}", dir.display());
    // The image's border runs in a pass of its own, too few pixels for the
    // checks above: 1 in 106 000 border pixels of 26 photos is more than 25
    // steps off, 225 without that pass.
    assert!(
        border_off * 2000 < border,
        "{border_off} of {border} border pixels off"
    );
    let mean = ious.iter().sum::<f64>() / ious.len() as f64;
    assert!(mean > 0.995, "mean IoU {mean:.4}");
    // The typical photo's opacities move by 0.004 steps on average, on both
    // test sets; a 5 % error in the deformable modulators, or the shifted
    // windows' masks left out, moves them by 0.05 or more.
    moves.sort_by(f64::total_cmp);
    let median = moves[moves.len() / 2];
    assert!(median < 0.02, "median mean opacity change {median:.4}");
}
