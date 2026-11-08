# birefnet-native

BiRefNet-lite background removal on the CPU or a Vulkan GPU, in plain Rust:
the image goes in, an opacity per pixel comes out, with no ONNX Runtime,
PyTorch or other inference library underneath.

- **No dependencies.** The library and its bench tool use only `std`; the
  GPU is reached through the system's `libvulkan.so.1`, opened at run time
  when it is there.
- **The original model's results.** Checked against PyTorch running
  BiRefNet's own code and weights: logits within 0.004, and no 8-bit opacity
  more than one step away, on 20 CC0 photos (`tools/birefnet/corpus.tsv`).
- **Faster and lighter than ONNX Runtime.** 1.3 s and 0.26 GiB per
  1024 x 1024 image on a six-core laptop, where ONNX Runtime takes 3.0 s and
  4.1 GiB at best; 0.61 s on the laptop's own GPU and 0.17 s on a GeForce GTX
  1660 Ti. See [Performance](#performance).

## Build

You need Rust 1.95 or newer and `glslc`, from shaderc, which compiles the
GPU shaders. From the repository's root:

```bash
cargo build --release --locked -p birefnet-native
```

The weights are not in the repository. They keep BiRefNet's MIT licence and
are converted from one of its ONNX exports into a weights file under the
ignored `weights/` directory. The export needs Python with `onnx` and `numpy`:

```bash
curl -L -o birefnet-lite.onnx \
  https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/de15b22ba131738a16dff04aab8bdf8dc32e3ac1/onnx/model.onnx
mkdir -p weights
tools/birefnet/export_weights.py birefnet-lite.onnx weights/birefnet-lite.weights
tools/birefnet/export_weights.py --half birefnet-lite.onnx weights/birefnet-lite-f16.weights
```

The repository's `pkgbuild/PKGBUILD` does the same steps; its `birefnet-native`
package holds the half-float weights alone, in `/usr/share/birefnet-native`,
with the bench tool and the licences.

`AssetFurnace/birefnet-lite-onnx` works too: both exports carry the same
tensors and give the same weights file, byte for byte. Run as ONNX models they
do not give the same masks, though: onnx-community's export breaks each
deformable convolution into gathers and scatters that do not compute what
PyTorch does, and departs from PyTorch on 19 of the 20 photos below (mean IoU
0.996, worst 0.976, with curly hair and thin structures losing most), while the
AssetFurnace export, with ONNX's own `DeformConv`, matches PyTorch.

## Use

```rust,no_run
use std::path::Path;
use birefnet_native::Model;

fn main() -> Result<(), birefnet_native::Error> {
    // Load once; 0 threads means one per CPU the process may use.
    let mut model = Model::load(Path::new("weights/birefnet-lite.weights"), 0)?;
    let rgb = vec![0u8; 1024 * 1024 * 3]; // the image resized to 1024 x 1024
    let logits = model.run(&rgb, 1024)?;
    let alpha: Vec<u8> = logits
        .iter()
        .map(|&l| (255.0 / (1.0 + (-l).exp())).round() as u8)
        .collect();
    assert_eq!(alpha.len(), 1024 * 1024);
    Ok(())
}
```

`Model::load` puts the model on a GPU that has Vulkan 1.2 with half floats (a
discrete one first), and on the CPU when there is none or it cannot be used;
`Model::gpu` names the GPU in use. `IMAGE_ENGINE_NATIVE_GPU=0` keeps the model
on the CPU, and `IMAGE_ENGINE_NATIVE_GPU=1` turns a missing GPU into an error
that says why. `Model::load_cpu` always uses the CPU.

The input is 8-bit RGB, row by row. The network was trained at 1024 x 1024;
smaller inputs run, but change the result well beyond what saving time is
worth (see [Precision](#precision-and-resolution)). Resizing the image to the
input side and the mask back belongs to the caller.

`birefnet-bench WEIGHTS SIDE [--threads N] [--repeat N] [--tag TAG]
IMAGE.rgb...` times the model on raw RGB files and, with `--tag`, writes its
logits.

## How it works

The network is written out layer by layer (`src/swin.rs`, `src/decoder.rs`)
instead of interpreted from a graph, which lets every layer take the shape that
suits it:

- Every linear layer and convolution runs on one matrix product kernel (the
  engine's `gemm.rs`), except a few tiny ones with a single output channel or
  a single input pixel: six rows by sixteen columns of vector accumulators,
  weights packed once into 16-column panels, and the input read through row
  pointers, so a convolution never copies its input into columns.
- The decoder's 3x3 convolutions use Winograd's F(4x4, 3x3), or F(2x2, 3x3)
  where their channels are many (the engine's `winograd.rs`): 36 products per
  4x4 block of output pixels instead of 144, or 16 per 2x2 block instead of
  36, each one product of the same kernel.
- A task takes up to four Swin windows: gather, normalise, project, attend,
  project and add back, all in cache. The MLP runs over chunks of tokens in
  place.
- The deformable ASPP runs a chunk of pixels through its four branches and
  their 1x1 fusion in one task, so the 1024 channels of its four branches
  never exist for the whole map; its pooling branch, constant over the image,
  is folded into a bias.
- The export folds every batch norm, merges each deformable convolution's
  offset and modulator convolutions, and moves the last 1x1 convolution before
  the final upsampling (both are linear), so nothing is computed with 96
  channels at full resolution.

On a GPU (`src/gpu.rs`, `shaders/`) the same network runs in half floats, with
layer norms, softmax, GELU, the biases and the deformable offsets in floats.
Nearly every layer is one matrix product kernel: the linear layers, the patch
merges and every convolution, whose taps it reads from the map itself. It
normalises its input rows when a layer norm comes first, with their means and
deviations taken once per product (`shaders/stats.comp`) rather than in every
block of columns, gathers the Swin windows (rolled, padded) and scatters the
attention back into place, so no tokens are copied; it sums each block of 16
inputs in half floats and the blocks in floats. The attention runs one
workgroup per window and head, two lanes per token, each with half the head's
channels. The deformable ASPP samples a chunk of 8192 pixels at a time, and
its pooling branch becomes a bias the 1x1 fusion adds onto. The
full-resolution branch runs as its composed 5x5 convolution, a pixel per
invocation, except on the image's border, where the two 3x3 convolutions run
in a pass of their own. As in realesrgan-native, the matrix kernel reads
weights through AMD's scalar cache, with 32 columns per wave so that three or
four waves fit on each SIMD, or stages them in shared memory on other GPUs;
either way it loads its input as blocks of four half floats along k,
transposed into the shared tile. A product with too few rows and columns to
fill the GPU, such as a 3x3 convolution over 2880 channels on the 32 x 32 map,
splits k between workgroups, whose float partial sums `reduce.comp` adds up.
Each invocation writes its row's columns eight at a time, and the workgroups
run in bands of row blocks (eight on AMD, four elsewhere) across all column
blocks, so the ones in flight share inputs and weights in the L2 cache. Each
Swin block is one submission, and each decoder block a few.

Work is split between threads by rows; no sum is ever split, so the result
does not depend on the number of threads. The kernels come in three tiers,
chosen when the process starts: AVX2 with FMA and F16C (tier 2: Haswell, Zen
and later), AVX (tier 1: Sandy and Ivy Bridge, Bulldozer), and SSE2 for any
other x86-64 CPU (tier 0), with plain Rust on other architectures. Results are
identical from run to run and machine to machine within a tier; tiers 1 and 0
give the same bits, while tier 2 fuses each multiply-add and differs from them
in the last bits. `IMAGE_ENGINE_NATIVE_TIER=N` caps the tier, to run an older
CPU's code on a newer one.

## Performance

One 1024 x 1024 image on a Ryzen 5 5600H (Zen 3, 6 cores, 12 threads, AVX2),
idle, median of five runs after a first one, peak resident memory of the
whole process:

| Engine | 4 threads | 6 threads | 12 threads | Peak memory | Mask vs PyTorch |
|---|---:|---:|---:|---:|---|
| birefnet-native, `--half` weights | 1.63 s | 1.24 s | 1.20 s | 0.26 GiB | IoU 0.99996 |
| birefnet-native, float32 weights | 1.84 s | 1.33 s | 1.28 s | 0.37 GiB | same |
| ONNX Runtime 1.31, AssetFurnace export | 3.26 s | 2.99 s | 3.31 s | 4.1 GiB | same |
| ONNX Runtime 1.31, onnx-community export | 8.00 s | 7.91 s | 8.42 s | 11.7 GiB | IoU 0.996 |
| PyTorch 2, BiRefNet's code | | 8.7 s | | | reference |
| candle-birefnet 0.1, candle 0.9.2 | | | 31.5 s | 9.9 GiB | same |

ONNX Runtime and PyTorch ran from Python (memory includes the interpreter).
candle-birefnet needs candle's conv2d default switched to `FullIm2Col`; with
candle 0.9.2 as published its masks are wrong. Starting a process, loading the
half-float weights and processing one image takes 2.1 s on four threads.

On a Core i3-2375M (Sandy Bridge, 2 cores and 4 threads at 1.5 GHz, AVX
without AVX2), at 0.34 GiB:

| Tier | 2 threads | 4 threads |
|---|---:|---:|
| AVX (tier 1) | 17.9 s | 18.0 s |
| SSE2 (tier 0, `IMAGE_ENGINE_NATIVE_TIER=0`) | 38.1 s | 33.7 s |

The AVX kernel keeps the multiply and add units busy on its own, so a core's
second thread only competes for them; the SSE2 kernel is bound by how many
instructions a core issues, which the second thread helps fill.

A run is 299 GMAC (multiply-adds), 46 % of them in the deformable ASPP and 38 %
in the two backbone passes. The matrix kernel sustains about 88 % of the CPU's
multiply-add peak; the rest of the time goes to attention, sampling,
normalisation and the moments a thread waits for the others. Half-float
weights are faster as well as smaller: the kernel streams half as many weight
bytes, and widening them costs less than that saves.

On GPUs, in half floats, median time of the bench's whole run over the 20
photos (the GPUs' own memory is not in the peaks above):

| GPU | Time per 1024 x 1024 image |
|---|---:|
| Radeon Vega 7 (the 5600H's, RADV) | 0.61 s |
| GeForce GTX 1660 Ti (NVIDIA 615) | 0.17 s |

against 1.22 s on the 5600H's CPU in the same runs. On the ASPP's largest
product the Vega 7 sustains about a third of its half-float multiply-add
peak.

## Precision and resolution

A weights file stores its weight matrices as 32-bit floats, or with `--half`
as 16-bit floats. Half floats halve the file and, at tier 2, the weights in
memory; the lower tiers widen them to 32 bits at load. The arithmetic stays
32-bit.

Measured on the 20 photos of `tools/birefnet/corpus.tsv` against PyTorch's
float32 output, as 8-bit masks:

| Weights or input | Mean IoU | Worst IoU | Pixels off by more than 25/255, mean / worst image |
|---|---:|---:|---:|
| float32 (this crate) | 1.00000 | 1.00000 | 0 / 0 |
| float16 (`--half`) | 0.99996 | 0.99953 | 0 / 0 |
| int8 per channel, weights only | 0.998 | 0.985 | 0.16 % / 2.8 % |
| int8 weights and activations, per row | 0.997 | 0.969 | 0.07 % / 0.66 % |
| float32, 768 x 768 input | 0.867 | 0.000 | 1.4 % / 5.9 % |
| float32, 512 x 512 input | 0.852 | 0.000 | 2.3 % / 16 % |

The GPU computes in half floats. Against PyTorch on the 20 photos its masks
reach IoU 0.997 or better on both GPUs, with at most 0.09 % of the pixels more
than 25 steps away. A half-float run is not always that close: on 26 other
photos, one per GPU, a different one on each, loses part of an object the
model is unsure of: IoU 0.94 against PyTorch on the GTX 1660 Ti and 0.92 on
the Vega 7, with up to 2.3 % of the pixels more than 25 steps away. In the
backbone's third stage a few tokens grow activations in the hundreds through
four blocks of feedback, and rounding upstream decides whether they do.
Summing the products in floats left that photo as it was, at 1.5 to 3 times
the time, so the GPU keeps half-float sums. `IMAGE_ENGINE_NATIVE_GPU=0` gives
the CPU's float results.

On these photos 8-bit quantization stays near the onnx-community export's
error, but plain per-channel scales break some images outright: a near-black
portrait came out with IoU 0.35 (weights only) and 0.20 (with activations),
the background turned to foreground. The backbone's attention projections and
MLP output layers are the sensitive ones. Lower resolutions change which
object is cut out. Neither is offered. `tools/birefnet/quantization_study.py`
reproduces the int8 rows in PyTorch.

## Checking results

`tools/birefnet/corpus.tsv` lists the evaluation photos, all CC0, with where
to download them. `tools/birefnet/evaluate.py` prepares inputs from images,
runs ONNX Runtime on them and compares two sets of logits as masks;
`tools/birefnet/torch_reference.py` writes the PyTorch references. With a
directory of `NAME.rgb` inputs and their `NAME.torch.logits`, the ignored test
checks float32 weights against them; half-float weights stay within the
precision table above, not within one 8-bit step:

```bash
cargo test --release
BIREFNET_NATIVE_WEIGHTS=weights/birefnet-lite.weights BIREFNET_NATIVE_REFERENCE=DIR \
  cargo test --release --test reference -- --ignored
```

The ignored `gpu` test needs a GPU and checks it against the CPU engine on
the same photos: IoU above 0.9 and fewer than 2.5 % of the pixels more than
25 steps away on each, mean IoU above 0.995, and the typical photo's
opacities moved by under 0.02 steps on average (0.004 measured).

## Credits and license

BiRefNet is the work of Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma
Laaksonen, Wanli Ouyang and Nicu Sebe, published under MIT. [NOTICE](NOTICE)
lists the papers and the code this builds on.

The Rust implementation is by Bruno Gonçalves
([BigLinux](https://www.biglinux.com.br)) and is licensed MIT
([LICENSE](../../LICENSE)), as the model is.