# realesrgan-native

Real-ESRGAN image upscaling on the CPU or a Vulkan GPU, in plain Rust: an
image goes in, one four times larger on each side comes out, with no
PyTorch, ONNX Runtime or other inference library underneath.

- **No dependencies.** The library and its bench tool use only `std`; the
  GPU is reached through the system's `libvulkan.so.1`, opened at run time
  when it is there.
- **The original model's results.** On the CPU, checked against PyTorch
  running the same network and weights: within 2.0e-4 on 20 CC0 photos, the
  same 8-bit image but for rounding ties. The GPU computes in half floats
  and scores the same against the original photos, to within 0.02 dB.
- **Fast enough for photos.** About 2.1 s per megapixel of input on a
  six-core laptop's CPU and 1.4 s on its integrated GPU, with 0.1 GB of
  memory for small images and 0.4 GB plus the output for large ones. See
  [Performance](#performance).

The model is `realesr-general-x4v3`, Real-ESRGAN's small network for general
photos: 1.2 million weights, 4.9 MB. Real-ESRGAN publishes it with a twin,
`realesr-general-wdn-x4v3`, that removes less noise; its own command line
blends the two half and half by default, and so do the weights below.

## Build

You need Rust 1.95 or newer and `glslc`, from shaderc, which compiles the
GPU shaders. From the repository's root:

```bash
cargo build --release --locked -p realesrgan-native
```

The weights are not in the repository. They keep Real-ESRGAN's BSD-3-Clause
licence and are converted from its two PyTorch checkpoints into a weights file
under the ignored `weights/` directory. The conversion needs Python with
`numpy`, but not PyTorch: the checkpoints are read with an unpickler that
builds nothing but arrays.

```bash
release=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0
curl -LO $release/realesr-general-x4v3.pth
curl -LO $release/realesr-general-wdn-x4v3.pth
mkdir -p weights
tools/realesrgan/export_weights.py --weak realesr-general-wdn-x4v3.pth --strength 0.5 \
  realesr-general-x4v3.pth weights/realesr-general-x4v3-dn50.weights
```

`--strength` is Real-ESRGAN's `--denoise_strength`: 1 keeps
`realesr-general-x4v3` alone, the strongest denoising, and 0 the weak twin
alone. The checkpoints' SHA-256 are
`8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` and
`1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca`.

## Use

```rust,no_run
use std::path::Path;
use realesrgan_native::{Model, SCALE};

fn main() -> Result<(), realesrgan_native::Error> {
    // Load once; 0 threads means one per CPU the process may use.
    let mut model = Model::load(Path::new("weights/realesr-general-x4v3-dn50.weights"), 0)?;
    let (width, height) = (320, 240);
    let small = vec![0u8; width * height * 3];
    let large = model.run(&small, width, height)?;
    assert_eq!(large.len(), width * SCALE * height * SCALE * 3);
    Ok(())
}
```

`Model::load` puts the model on a GPU that has Vulkan 1.2 with half floats (a
discrete one first), and on the CPU when there is none or it cannot be used;
`Model::gpu` names the GPU in use. `IMAGE_ENGINE_NATIVE_GPU=0` keeps the model
on the CPU, and `IMAGE_ENGINE_NATIVE_GPU=1` turns a missing GPU into an error
that says why. `Model::load_cpu` always uses the CPU. CPU implementations of
Vulkan, such as llvmpipe, are slower than the CPU engine and never chosen.

The input and output are 8-bit RGB, row by row, of any size. `Model::run`
works in tiles of up to 512 x 512 input pixels (768 x 768 on a GPU), each
with 34 pixels of context on every side: the network's 34 3x3 convolutions
see no further, so tiles give the whole image's result, to within rounding
(6e-6) on the CPU and bit for bit on a GPU, while memory stays bounded.
`Model::run_with_progress` reports the share of the work done after each
layer, 34 times per tile, and stops with `Error::Cancelled` when the callback
returns `ControlFlow::Break`. `Model::run_float` runs the whole image at once
on values in 0..=1 and returns the network's output unclamped; on a GPU it
runs in tiles all the same, which give the same result.

`realesrgan-bench WEIGHTS WIDTH HEIGHT [--threads N] [--repeat N] [--tag TAG]
[--tile N --margin N] IMAGE.rgb...` times the model on raw RGB files and, with
`--tag`, writes its output.

## How it works

The network is written out layer by layer (`src/network.rs`): a 3x3
convolution from RGB to 64 channels, 32 from 64 to 64, each followed by a
PReLU, and one to 48 channels that a pixel shuffle turns into the 4x RGB
image, added to the input enlarged by repeating its pixels.

- Every convolution after the first uses Winograd's F(4x4, 3x3) (the engine's
  `winograd.rs`): a 4x4 block of output pixels takes 36 products of the
  transformed input and weights instead of 144. Each of them runs on one matrix
  product kernel (the engine's `gemm.rs`): six rows by sixteen columns of
  vector accumulators and weights packed into 16-column panels.
- Two 64-channel maps at the input's size take turns as a layer's input and
  output; the task that computes a layer's blocks applies its PReLU to them
  while they are in cache.

On a GPU (`src/gpu.rs`, `shaders/`) each layer is one dispatch of a 3x3
convolution in half floats, and its own submission, so no submission runs
long and a run can stop between layers. Activations stay on the GPU, two
packed 64-channel maps that take turns; only the weights, once, and the
output are copied. The convolution kernel depends on the GPU:

- On AMD's, each wave of 64 lanes computes an 8x8 tile for all 64 output
  channels, one pixel per lane, and reads the weights at an index the same in
  every lane, so they come through the scalar cache into scalar registers
  and only the input goes through LDS (`conv_scalar.comp`).
- On others, a workgroup stages a block of 16 input channels' weights in
  shared memory and each invocation computes a row of 8 pixels for 8
  channels, reading each input row once for the three kernel columns
  (`conv_shared.comp`).

Each is 1.6 to 2.3 times faster than the other on its own kind of GPU.
Winograd's convolution did not pay on either: its four times as many
registers per output leave too few workgroups to hide the GPU's latency.

Work is split between threads by blocks of pixels; no sum is ever split, so
the result does not depend on the number of threads. The kernels come in three
tiers, chosen when the process starts: AVX2 with FMA (tier 2: Haswell, Zen and
later), AVX (tier 1: Sandy and Ivy Bridge, Bulldozer), and SSE2 for any other
x86-64 CPU (tier 0), with plain Rust on other architectures.
`IMAGE_ENGINE_NATIVE_TIER=N` caps the tier.

## Performance

As direct convolutions a run costs 2.4 MFLOP per input pixel, 238 GFLOP for a
384 x 256 image; the Winograd convolutions do it with four times fewer
multiplications. On a Ryzen 5 5600H (Zen 3, 6 cores, 12 threads, AVX2), 6
threads, peak resident memory of the whole process, one image:

| Input | Output | Time | Peak memory |
|---|---|---:|---:|
| 250 x 167 | 1000 x 668 | 0.09 s | 46 MB |
| 384 x 256 | 1536 x 1024 | 0.19 s | 96 MB |
| 1024 x 768, whole image | 4096 x 3072 | 1.63 s | 561 MB |
| 1024 x 768, tiles (`Model::run`) | 4096 x 3072 | 2.01 s | 401 MB |

The bench keeps float copies of the image and of the output; `Model::run`
keeps only the 8-bit output and a tile's floats, 0.23 GB for a 1500 x 1000
photo. All 12 threads take 1.50 s for 1024 x 768 whole. On a Core i3-2375M
(Sandy Bridge, 2 cores and 4 threads at 1.5 GHz, AVX without AVX2), 384 x 256
takes 2.5 s with 4 threads, 2.8 s with 2, and 5.6 s with SSE2
(`IMAGE_ENGINE_NATIVE_TIER=0`).

On GPUs, in half floats, with the time of the bench's whole run (the same
images; the 1000 x 668 one runs as two tiles):

| GPU | 384 x 256 | 1000 x 668 |
|---|---:|---:|
| Radeon Vega 7 (the 5600H's, RADV) | 0.10 s | 0.90 s |
| GeForce GTX 1660 Ti (NVIDIA 615) | 0.04 s | 0.40 s |

against 0.19 s and 1.39 s on the 5600H's CPU. In single floats the GPUs
take about twice as long, which is why they compute in half floats.

## Precision

Against PyTorch's float32 output on the 20 photos, reduced to 384 x 256 and to
250 x 167:

| Weights | Largest difference | 8-bit values that differ |
|---|---:|---:|
| float32 | 2.0e-4 | 0.003 % (rounding ties) |
| float16 (`--half`) | 1.2e-2 | 1.6 % |

Half floats save 2.4 MB and no time here, and cost three 8-bit steps at
worst; the float32 weights are the ones to use.

The GPU computes in half floats throughout. Against the CPU engine on the
same 20 photos at 250 x 167, it moves a share of 8-bit values by a step,
and a few by up to 12 (8-bit PSNR 52 to 57.6 dB); against the original
photos its PSNR is within 0.02 dB of the CPU's, above or below, on each of
them.

## Checking results

`tools/realesrgan/evaluate.py prepare` makes a reduced copy of each photo to
upscale and keeps the larger one as the reference;
`tools/realesrgan/torch_reference.py` writes PyTorch's outputs for them;
`tools/realesrgan/evaluate.py compare` prints the PSNR of a Lanczos upscale
and of the outputs against the reference, and the differences between two sets
of outputs. PSNR favours blur: Real-ESRGAN draws plausible texture where
Lanczos stays soft, and scores lower. With a directory of `NAME.rgb` inputs
and their `NAME.torch.out`, the ignored tests check float32 weights against
them and the progress reports:

```bash
cargo test --release
REALESRGAN_NATIVE_WEIGHTS=$PWD/weights/realesr-general-x4v3-dn50.weights \
  REALESRGAN_NATIVE_REFERENCE=DIR REALESRGAN_NATIVE_SIZE=384x256 \
  cargo test --release --test reference -- --ignored
```

The ignored `gpu` test needs a GPU and checks it against the CPU engine on
the same photos: an 8-bit PSNR above 48 dB, and tiles that give the whole
image's result bit for bit.

## Credits and license

Real-ESRGAN is the work of Xintao Wang, Liangbin Xie, Chao Dong and Ying
Shan, published under BSD-3-Clause. [NOTICE](NOTICE) lists the papers and the
code this builds on and carries Real-ESRGAN's licence, which applies to the
weights.

The Rust implementation is by Bruno Gonçalves
([BigLinux](https://www.biglinux.com.br)) and is licensed MIT
([LICENSE](../../LICENSE)).