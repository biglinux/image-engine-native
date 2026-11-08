//! Compile the GPU shaders to SPIR-V with `glslc`, from shaderc.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let shaders: [(&str, &str, &[&str]); 12] = [
        ("add", "add", &[]),
        ("attention", "attention", &[]),
        ("conv3", "conv3", &[]),
        ("conv_shared", "conv_shared", &[]),
        ("conv_shared_32", "conv_shared", &["-DCOUT=32"]),
        ("gemm", "gemm", &[]),
        ("gemm_2", "gemm", &["-DWAVES=2"]),
        ("gemm_4", "gemm", &["-DWAVES=4"]),
        ("gemm_shared", "gemm_shared", &[]),
        ("gemm_shared_32", "gemm_shared", &["-DCOLS=32"]),
        ("mlp_32", "mlp", &["-DK=32"]),
        ("mlp_64", "mlp", &["-DK=64"]),
    ];
    println!("cargo::rerun-if-changed=shaders/gemm_layout.glsl");
    println!("cargo::rerun-if-changed=shaders/gelu.glsl");
    for (name, source, defines) in shaders {
        let source = format!("shaders/{source}.comp");
        println!("cargo::rerun-if-changed={source}");
        let status = Command::new("glslc")
            .args(["--target-env=vulkan1.2", "-O"])
            .args(defines)
            .args([&source, "-o"])
            .arg(out.join(format!("{name}.spv")))
            .status()
            .unwrap_or_else(|e| panic!("glslc, from shaderc, compiles the GPU shaders: {e}"));
        assert!(status.success(), "glslc failed on {source}: {status}");
    }
}
