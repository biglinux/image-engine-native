# image-engine-native

Run image AI models without PyTorch, ONNX Runtime or Python. Four models,
written out by hand in plain Rust, running on the CPU or on any GPU with a
Vulkan driver.

| Model | What it does | Crate | Weights |
|---|---|---|---:|
| BiRefNet-lite | removes the background | [birefnet-native](crates/birefnet-native) | 90 MB |
| SCUNet | cleans up noise and JPEG artifacts | [scunet-native](crates/scunet-native) | 37 MB |
| Real-ESRGAN general x4v3 | upscales 4x | [realesrgan-native](crates/realesrgan-native) | 4.9 MB |
| Big LaMa | fills in a masked area | [lama-native](crates/lama-native) | 102 MB |

- No dependencies beyond Rust's standard library.
- Uses the GPU when there is one, falls back to the CPU when there isn't.
- Every model is checked against the original model running in PyTorch.

Each crate's README covers its API, bigger images, memory use and how to
check its output.

## How fast is it?

We ran the same models through what we would otherwise have shipped: ONNX
Runtime 1.31, OpenVINO 2026.4 and PyTorch 2.15. Same input size, 6 threads,
median of 5 runs after a warm-up. The tables use each runtime's float (f32)
model; the half-float (f16) ones come further down.

The machines:

| | Laptop | Desktop |
|---|---|---|
| CPU | AMD Ryzen 5 5600H (6 cores, 12 threads) | Intel Core i5-13400 (6 performance + 4 efficient cores) |
| RAM | 40 GB DDR4-3200 | 32 GB DDR5-5200 |
| GPU | AMD Radeon Vega 7 (integrated) | NVIDIA GeForce GTX 1660 Ti, 6 GB |
| Drivers | Mesa 26.2 (RADV) | NVIDIA 615.78, CUDA 13.4, cuDNN 9.27 |
| OS | BigLinux with BigDesktop, Linux 7.2 | BigLinux with BigDesktop, Linux 7.2 |

### Laptop, CPU (Ryzen 5 5600H)

| Time | image-engine-native | ONNX Runtime | OpenVINO | PyTorch |
|---|---:|---:|---:|---:|
| BiRefNet-lite, 1024 x 1024 | **1.36 s** | 8.27 s | 6.45 s | 9.48 s |
| SCUNet, 768 x 512 | **2.07 s** | 8.61 s | 5.79 s | 10.4 s |
| Real-ESRGAN, 384 x 256 | **0.23 s** | 0.70 s | 0.46 s | 1.04 s |
| Big LaMa, 512 x 512 | **0.74 s** | 1.91 s | 1.40 s | 1.45 s |

| Memory | image-engine-native | ONNX Runtime | OpenVINO | PyTorch |
|---|---:|---:|---:|---:|
| BiRefNet-lite | **0.29 GB** | 11.7 GB | 4.85 GB | 2.19 GB |
| SCUNet | **0.39 GB** | 8.55 GB | 1.20 GB | 1.78 GB |
| Real-ESRGAN | **0.09 GB** | 0.18 GB | 0.15 GB | 0.39 GB |
| Big LaMa | **0.24 GB** | 0.86 GB | 1.23 GB | 1.74 GB |

1.9 to 4.7 times faster than the best of the rest, with 1.7 to 7.6 times less
memory.

### Laptop, GPU (Radeon Vega 7, through Vulkan)

| | image-engine-native | vs. our own CPU path | vs. the best of the rest on the CPU |
|---|---:|---:|---:|
| BiRefNet-lite, 1024 x 1024 | **0.60 s** | 2.3x faster | 10.8x faster |
| SCUNet, 768 x 512 | **0.75 s** | 2.8x faster | 7.7x faster |
| Real-ESRGAN, 384 x 256 | **0.11 s** | 2.1x faster | 4.2x faster |
| Big LaMa, 512 x 512 | **0.31 s** | 2.4x faster | 4.5x faster |

The others can't use this GPU at all: OpenVINO's GPU support is Intel-only,
and CUDA and ROCm don't cover the Vega 7.

### Desktop, CPU (Core i5-13400)

Pinned to one thread per performance core.

| Time | image-engine-native | ONNX Runtime | OpenVINO | PyTorch |
|---|---:|---:|---:|---:|
| BiRefNet-lite, 1024 x 1024 | **1.04 s** | 3.96 s | 3.05 s | 6.11 s |
| SCUNet, 768 x 512 | **1.44 s** | 4.63 s | 2.75 s | 5.14 s |
| Real-ESRGAN, 384 x 256 | **0.14 s** | 0.41 s | 0.33 s | 0.46 s |
| Big LaMa, 512 x 512 | **0.53 s** | 1.40 s | 0.86 s | 0.86 s |

| Memory | image-engine-native | ONNX Runtime | OpenVINO | PyTorch |
|---|---:|---:|---:|---:|
| BiRefNet-lite | **0.29 GB** | 11.7 GB | 4.85 GB | 3.24 GB |
| SCUNet | **0.39 GB** | 8.55 GB | 1.19 GB | 2.28 GB |
| Real-ESRGAN | **0.09 GB** | 0.18 GB | 0.15 GB | 0.70 GB |
| Big LaMa | **0.24 GB** | 0.87 GB | 1.23 GB | 2.04 GB |

1.6 to 2.9 times faster than the best of the rest, with 1.7 to 11 times less
memory.

### Desktop, GPU (GeForce GTX 1660 Ti)

ONNX Runtime here is the distro's 1.29 package, which only finds cuDNN with
`LD_PRELOAD=libcudnn.so.9`.

| Time | image-engine-native | ONNX Runtime + CUDA | PyTorch + CUDA |
|---|---:|---:|---:|
| BiRefNet-lite, 1024 x 1024 | **0.17 s** | out of memory | 0.34 s |
| SCUNet, 768 x 512 | **0.22 s** | 0.69 s | 0.69 s |
| Real-ESRGAN, 384 x 256 | **0.051 s** | 0.068 s | 0.053 s |
| Big LaMa, 512 x 512 | **0.081 s** | 0.23 s | 0.15 s |

| GPU memory | image-engine-native | ONNX Runtime + CUDA | PyTorch + CUDA |
|---|---:|---:|---:|
| BiRefNet-lite | **0.38 GB** | out of memory | 2.53 GB |
| SCUNet | **0.26 GB** | 4.36 GB | 1.31 GB |
| Real-ESRGAN | **0.04 GB** | 0.37 GB | 0.16 GB |
| Big LaMa | **0.18 GB** | 1.10 GB | 0.50 GB |

Tied with PyTorch on Real-ESRGAN, 1.8 to 3.1 times faster on the rest, with
2.8 to 6.7 times less GPU memory. ONNX Runtime can't fit BiRefNet into 6 GB at
all.

### Install size and start-up

| Libraries | CPU only | With GPU support |
|---|---:|---:|
| image-engine-native | 2 MB per model | 2 MB per model |
| ONNX Runtime | 29 MB | 2.5 GB (with CUDA and cuDNN) |
| OpenVINO | 86 MB | no GPU support here |
| PyTorch | 0.86 GB | 5.2 GB (with CUDA) |

| Model files | image-engine-native | ONNX Runtime, OpenVINO | PyTorch |
|---|---:|---:|---:|
| BiRefNet-lite | 90 MB | 224 MB | 178 MB |
| SCUNet | 37 MB | 74 MB | 72 MB |
| Real-ESRGAN | 4.9 MB | 4.9 MB | 4.9 MB |
| Big LaMa | 102 MB | 208 MB | 381 MB |

| Load time, laptop CPU | image-engine-native | ONNX Runtime | OpenVINO | PyTorch |
|---|---:|---:|---:|---:|
| BiRefNet-lite | 0.10 s | 2.7 s | 4.3 s | 2.3 s |
| SCUNet | 0.04 s | 1.9 s | 2.4 s | 1.4 s |
| Real-ESRGAN | 0.01 s | 0.04 s | 0.07 s | 0.03 s |
| Big LaMa | 0.11 s | 7.0 s | 4.2 s | 2.8 s |

OpenVINO compiles the network before its first run, hence the wait. On the
GPU, image-engine-native takes up to 0.4 s to load.

### Half floats didn't help the others

f16 model files are half the size, and that's about it. The same runs with
the f16 models (LaMa has no f16 ONNX, and converting it fails):

| f16 time | BiRefNet-lite | SCUNet | Real-ESRGAN | Big LaMa |
|---|---:|---:|---:|---:|
| ONNX Runtime, laptop CPU | 8.36 s | 10.7 s | 1.31 s | - |
| OpenVINO, laptop CPU | 6.83 s | 6.20 s | 0.46 s | 1.41 s |
| ONNX Runtime, desktop CPU | 4.10 s | 5.21 s | 0.69 s | - |
| OpenVINO, desktop CPU | 3.03 s | 2.78 s | 0.33 s | 0.97 s |
| ONNX Runtime + CUDA, GTX 1660 Ti | 1.45 s | 1.61 s | 0.24 s | - |
| PyTorch + CUDA, GTX 1660 Ti | 1.10 s | 1.67 s | 0.24 s | 0.56 s |

None of them beats its own f32 run. These CPUs can't do math in half floats,
OpenVINO quietly turns the weights back into f32 when it loads them, and ONNX
Runtime and PyTorch run slower. Watch out for ONNX Runtime 1.29 in
particular: it runs f16 models on the CPU on a single core (13 s for
BiRefNet, and the official 1.29 build didn't finish in 400 s). 1.31 fixed
that.

### Are the results the same?

Close enough that you won't see it. Against PyTorch's own output, worst case
over three photos per model:

| | SCUNet: worst pixel (out of 255) | Real-ESRGAN: worst pixel | BiRefNet: mask overlap | LaMa: average difference in the hole |
|---|---:|---:|---:|---:|
| image-engine-native, CPU | 1 | 1 | 99.95% | 0.40 |
| image-engine-native, Vega 7 | 2 | 8 | 99.78% | 1.74 |
| ONNX Runtime and OpenVINO, f32 | 1 | 1 | 97.56% | 0 |
| ONNX Runtime and OpenVINO, f16 | 1 | 3 | 97.52% | 0.97 |

The published BiRefNet ONNX file is actually further from PyTorch than we
are. LaMa's fills shift a little with f16 weights no matter who runs them,
without getting worse against the original photos (details in its README).

## Why is it fast?

### On the CPU

- **One hand-tuned matrix kernel** for every convolution and linear layer,
  running at 82% of a Zen 3 core's peak. It comes in AVX2, AVX and SSE2
  flavors, picked at start-up (`IMAGE_ENGINE_NATIVE_TIER=N` forces an older
  one).
- **Half-float weights**, packed once at load in the order the kernel reads
  them and widened to f32 in registers. Half the memory traffic, half the RAM,
  half the disk.
- **No im2col.** Convolutions read their input in place through row pointers,
  padding included.
- **Winograd** for 3x3 convolutions: 2.25 to 4 times fewer multiplications.
- **Fused layers.** Bias, activations and residual sums are applied while the
  output is still in cache, instead of in separate passes.
- **Memory that comes and goes.** Each intermediate map lives only as long as
  it's needed, and buffers get reused. SCUNet and Real-ESRGAN can take big
  images in tiles, so memory doesn't grow with image size.
- **Deterministic threads.** Work is split by rows, never inside a sum, so
  you get the same result with 1 thread or 16.

### On the GPU (Vulkan)

- **Plain Vulkan compute shaders**, loaded at run time. No SDK, no CUDA, no
  ROCm; if there's no Vulkan, the CPU path runs instead.
- **Half floats everywhere**, two values per instruction. Long sums are added
  up in f32 so the error stays small.
- **Different kernels for different GPUs.** On AMD, weights go through the
  scalar cache that every lane shares, leaving registers free for the actual
  work. On NVIDIA and others, weights and inputs are staged in shared memory
  and each thread computes a small block of outputs.
- **Fused layers here too.** Layer norm, GELU, bias and residual sums happen
  inside the matrix kernels, and SCUNet's MLPs run as a single kernel on AMD.
- **Model-specific tricks.** LaMa's Fourier transforms run in shared memory
  and its 3-channel last layer has its own kernel; BiRefNet's products that
  are too small to fill the GPU split their work along k.
- **Wide writes**, 16 bytes at a time, so results leave in whole cache lines.

## Repository layout

- `crates/native-engine`: the CPU core shared by all models (matrix kernel,
  Winograd, Swin window attention, thread pool, weight loading, SIMD
  dispatch).
- `crates/native-vulkan`: the Vulkan side. Opens `libvulkan.so.1` at run time,
  so machines without Vulkan just use the CPU.
- `crates/<model>-native`: one model each, with its shaders, a bench tool and
  tests.
- `tools/<model>`: Python scripts to export the published weights, produce
  reference outputs with the original model and compare results.
  `tools/checkpoint.py` reads PyTorch checkpoints without PyTorch and without
  running anything stored in them.

## Building

You need Rust 1.95 or newer, and `glslc` (from shaderc) for the shaders.

```bash
cargo build --release --locked
```

The weights aren't in the repository: each crate's README shows how to
download the published files and export them into `weights/`.
`pkgbuild/PKGBUILD` builds everything and splits it into one package per
model.

## Testing

```bash
cargo test --release --workspace
```

Each model also has ignored tests that compare it against the original
model's output; its README shows how to run them.

## License

The code is MIT ([LICENSE](LICENSE)). The engine's exponential follows
XNNPACK's and keeps its BSD-3-Clause licence, and its GELU uses Eigen's
coefficients; [NOTICE](NOTICE) has both. The weights keep their original
licenses, listed in each crate's NOTICE.
