//! Compile the GPU shaders to SPIR-V with `glslc`, from shaderc.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let shaders: [(&str, &str, &[&str]); 18] = [
        ("add", "add", &[]),
        ("conv3", "conv_lanes", &[]),
        ("conv3_s2", "conv_lanes", &["-DS=2"]),
        (
            "conv7",
            "conv",
            &["-DK=7", "-DS=1", "-DCO=32", "-DWAVES=2", "-DCK=4"],
        ),
        ("conv7_last", "conv7_last", &[]),
        (
            "conv7_px4",
            "conv",
            &["-DK=7", "-DS=1", "-DCO=32", "-DPX=4", "-DCK=4"],
        ),
        (
            "conv7_px2",
            "conv",
            &["-DK=7", "-DS=1", "-DCO=32", "-DPX=2", "-DCK=4"],
        ),
        ("conv_shared", "conv_shared", &[]),
        ("conv_shared_s2", "conv_shared", &["-DS=2"]),
        ("dft", "dft", &[]),
        ("fft_64", "fft", &["-DC=16", "-DNMAX=64"]),
        ("fft", "fft", &[]),
        ("fft_256", "fft", &["-DC=4", "-DNMAX=256"]),
        ("fft_512", "fft", &["-DC=2", "-DNMAX=512"]),
        ("gemm", "gemm", &[]),
        ("gemm_2", "gemm", &["-DWAVES=2"]),
        ("gemm_4", "gemm", &["-DWAVES=4"]),
        ("gemm_shared", "gemm_shared", &[]),
    ];
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
