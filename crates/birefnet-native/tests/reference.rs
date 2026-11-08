//! The model against PyTorch's own outputs.
//!
//! Needs float32 weights and a directory of `NAME.rgb` inputs with `NAME.torch.logits`
//! references (see "Checking results" in README.md):
//!
//!     BIREFNET_NATIVE_WEIGHTS=weights/birefnet-lite.weights \
//!     BIREFNET_NATIVE_REFERENCE=corpus cargo test --release -- --ignored

use std::path::PathBuf;

use birefnet_native::Model;

fn alpha(logit: f32) -> i32 {
    (255.0 / (1.0 + (-f64::from(logit)).exp())).round() as i32
}

#[test]
#[ignore = "needs weights and reference outputs"]
fn matches_pytorch_reference() {
    let weights =
        PathBuf::from(std::env::var("BIREFNET_NATIVE_WEIGHTS").expect("BIREFNET_NATIVE_WEIGHTS"));
    let dir = PathBuf::from(
        std::env::var("BIREFNET_NATIVE_REFERENCE").expect("BIREFNET_NATIVE_REFERENCE"),
    );
    let mut model = Model::load_cpu(&weights, 0).unwrap();
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "rgb") {
            continue;
        }
        let reference = path.with_extension("torch.logits");
        let Ok(bytes) = std::fs::read(&reference) else {
            continue;
        };
        let want: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        let rgb = std::fs::read(&path).unwrap();
        let side = (want.len() as f64).sqrt() as usize;
        assert_eq!(
            side * side,
            want.len(),
            "{}: reference size",
            reference.display()
        );
        let got = model.run(&rgb, side).unwrap();
        assert_eq!(got.len(), want.len(), "{}: output size", path.display());
        assert!(
            got.iter().chain(&want).all(|v| v.is_finite()),
            "{}: logits not finite",
            path.display()
        );
        // A float32 run of the same network: logits within 0.01, and no 8-bit
        // opacity more than one step away.
        let worst_logit = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        let worst_alpha = got
            .iter()
            .zip(&want)
            .map(|(&g, &w)| (alpha(g) - alpha(w)).abs())
            .max()
            .unwrap();
        assert!(
            worst_logit < 0.01 && worst_alpha <= 1,
            "{}: logit {worst_logit}, alpha {worst_alpha}",
            path.display()
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no NAME.rgb with NAME.torch.logits in {}",
        dir.display()
    );
}
