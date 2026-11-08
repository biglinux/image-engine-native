//! Compile the GPU shaders to SPIR-V with `glslc`, from shaderc.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    // Each SPIR-V module: its name, its source and the macros it is compiled with.
    for (name, source, defines) in [
        ("conv_scalar", "conv_scalar", &[][..]),
        ("conv_shared", "conv_shared", &[]),
        ("conv_shared_px4", "conv_shared", &["-DPX=4"]),
        ("shuffle", "shuffle", &[]),
    ] {
        let source = format!("shaders/{source}.comp");
        println!("cargo::rerun-if-changed={source}");
        let status = Command::new("glslc")
            .args(["--target-env=vulkan1.2", "-O", &source])
            .args(defines)
            .arg("-o")
            .arg(out.join(format!("{name}.spv")))
            .status()
            .unwrap_or_else(|e| panic!("glslc, from shaderc, compiles the GPU shaders: {e}"));
        assert!(status.success(), "glslc failed on {source}: {status}");
    }
}
