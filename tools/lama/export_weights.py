#!/usr/bin/env python3
"""Export Big LaMa's generator into a lama-native weights file.

    export_weights.py [--half] big-lama.zip OUT.weights

big-lama.zip is the archive LaMa's README links, with big-lama/models/
best.ckpt inside. The checkpoint is read without PyTorch by
tools/checkpoint.py, with the classes of its training configuration and
callbacks turned into inert placeholders, so nothing in the file can run. The
generator's tensors are written in the layouts
crates/lama-native/src/network.rs reads:

* every batch norm is folded into the convolution before it;
* a convolution weight becomes a matrix of (tap row, tap column, input
  channel) rows by output channel columns, the order the engine's NHWC
  convolutions read their input in;
* in each fast Fourier convolution, the local-to-local and global-to-local
  convolutions become one convolution of all 512 channels ("local"), the
  local-to-global one is "cross", and the spectral transform's three 1x1
  convolutions are "spectral_in", "fourier" and "spectral_out", the last two
  carrying the global batch norm's scale;
* the Fourier unit's channels, which alternate real and imaginary parts in
  the reference, are reordered to all real parts, then all imaginary ones;
* each stride-2 transposed convolution becomes four products, one per output
  phase (even or odd row, even or odd column), of the input pixels and kernel
  taps that phase reads.

With --half, the weight matrices are stored as IEEE half floats (biases stay
float32): half the size, and on CPUs with AVX2 they stay half in memory too.

File format, little-endian: the bytes b"LAMA\\0\\0\\0\\0", u32 version 1, u32
tensor count, then per tensor a u16 name length, the UTF-8 name, a u8 type (0
for float32, 1 for float16), a u8 rank, u32 dimensions, and the values.
"""

import argparse
import io
import pathlib
import struct
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from checkpoint import load as load_checkpoint

BN_EPS = 1e-5
LOCAL, GLOBAL = 128, 384
SPECTRAL = GLOBAL // 2
BLOCKS = 18
# Output phase taps along one axis, as (input offset, kernel tap): an even
# output 2m reads input m through tap 1, an odd one 2m + 1 reads m through
# tap 2 and m + 1 through tap 0. crates/lama-native/src/network.rs lists the
# same offsets.
PHASE_TAPS = [[(0, 1)], [(0, 2), (1, 0)]]


def load(path):
    with zipfile.ZipFile(path) as outer:
        ckpt = outer.read("big-lama/models/best.ckpt")
    state = load_checkpoint(io.BytesIO(ckpt), inert=True)["state_dict"]
    prefix = "generator.model."
    return {
        k.removeprefix(prefix): np.asarray(v, np.float32)
        for k, v in state.items()
        if k.startswith(prefix) and not k.endswith("num_batches_tracked")
    }


class Source:
    def __init__(self, tensors):
        self.tensors = tensors
        self.used = set()

    def __getitem__(self, name):
        self.used.add(name)
        return self.tensors[name]

    def bn(self, name):
        """Scale and shift of a batch norm in evaluation."""
        scale = self[f"{name}.weight"] / np.sqrt(self[f"{name}.running_var"] + BN_EPS)
        return scale, self[f"{name}.bias"] - self[f"{name}.running_mean"] * scale


def conv(w):
    """[out, in, kh, kw] to [(kh, kw, in), out]."""
    o, i, kh, kw = w.shape
    return w.transpose(2, 3, 1, 0).reshape(kh * kw * i, o)


def export(src):
    out = {}

    def put(name, matrix, bias=None):
        out[f"{name}.weight"] = matrix
        if bias is not None:
            out[f"{name}.bias"] = bias

    scale, shift = src.bn("1.bn_l")
    put("first", conv(src["1.ffc.convl2l.weight"]) * scale, shift)
    for i, layer in enumerate((2, 3)):
        scale, shift = src.bn(f"{layer}.bn_l")
        put(f"down{i}", conv(src[f"{layer}.ffc.convl2l.weight"]) * scale, shift)
    weight = np.concatenate([src["4.ffc.convl2l.weight"], src["4.ffc.convl2g.weight"]])
    scale_l, shift_l = src.bn("4.bn_l")
    scale_g, shift_g = src.bn("4.bn_g")
    scale, shift = (
        np.concatenate([scale_l, scale_g]),
        np.concatenate([shift_l, shift_g]),
    )
    put("down2", conv(weight) * scale, shift)

    planar = np.array([2 * c + j for j in range(2) for c in range(SPECTRAL)])
    for b in range(BLOCKS):
        for f, half in enumerate(("conv1", "conv2")):
            p = f"{5 + b}.{half}"
            name = f"block{b}.ffc{f}"
            scale_l, shift_l = src.bn(f"{p}.bn_l")
            scale_g, shift_g = src.bn(f"{p}.bn_g")
            local = np.concatenate(
                [src[f"{p}.ffc.convl2l.weight"], src[f"{p}.ffc.convg2l.weight"]], axis=1
            )
            put(f"{name}.local", conv(local) * scale_l, shift_l)
            put(
                f"{name}.cross", conv(src[f"{p}.ffc.convl2g.weight"]) * scale_g, shift_g
            )
            g = f"{p}.ffc.convg2g"
            scale, shift = src.bn(f"{g}.conv1.1")
            put(f"{name}.spectral_in", conv(src[f"{g}.conv1.0.weight"]) * scale, shift)
            scale, shift = src.bn(f"{g}.fu.bn")
            fourier = conv(src[f"{g}.fu.conv_layer.weight"]) * scale
            put(f"{name}.fourier", fourier[planar][:, planar], shift[planar])
            put(f"{name}.spectral_out", conv(src[f"{g}.conv2.weight"]) * scale_g)

    for i, (layer, bn) in enumerate(((24, 25), (27, 28), (30, 31))):
        weight = src[f"{layer}.weight"]  # [in, out, ky, kx]
        scale, shift = src.bn(str(bn))
        bias = (src[f"{layer}.bias"] * scale) + shift
        for phase in range(4):
            ys, xs = PHASE_TAPS[phase // 2], PHASE_TAPS[phase % 2]
            taps = [weight[:, :, ky, kx] for _, ky in ys for _, kx in xs]
            put(f"up{i}.phase{phase}", np.concatenate(taps) * scale, bias)
    put("last", conv(src["34.weight"]), src["34.bias"])

    unused = sorted(set(src.tensors) - src.used)
    if unused:
        raise SystemExit(f"generator tensors not exported: {unused[:5]}")
    return out


def write(path, tensors, half):
    with open(path, "wb") as f:
        f.write(b"LAMA\0\0\0\0")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, array in tensors.items():
            array = np.ascontiguousarray(array, dtype=np.float32)
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)))
            f.write(encoded)
            as_half = half and array.ndim == 2
            f.write(struct.pack("<BB", int(as_half), array.ndim))
            f.write(struct.pack(f"<{array.ndim}I", *array.shape))
            f.write(array.astype("<f2" if as_half else "<f4").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--half", action="store_true")
    parser.add_argument("archive")
    parser.add_argument("out")
    args = parser.parse_args()
    tensors = export(Source(load(args.archive)))
    write(args.out, tensors, args.half)
    count = sum(a.size for a in tensors.values())
    print(f"{len(tensors)} tensors, {count} floats")


if __name__ == "__main__":
    main()
