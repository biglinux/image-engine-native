#!/usr/bin/env python3
"""Export BiRefNet-lite's weights from its ONNX file into a birefnet-native weights file.

    export_weights.py [--half] MODEL.onnx OUT.weights

MODEL.onnx is the onnx-community export
(onnx-community/BiRefNet_lite-ONNX, onnx/model.onnx) or the AssetFurnace one
(AssetFurnace/birefnet-lite-onnx); both carry the same 437 tensors. The file
holds them in the layouts crates/birefnet-native/src/swin.rs and decoder.rs
read, folded where the network allows:

* every convolution weight becomes a matrix of (tap row, tap column, input
  channel) rows by output channel columns, the order the engine's NHWC
  convolutions read their input in;
* each batch norm after a deformable convolution is folded into it;
* a deformable convolution's offset and modulator convolutions, which read
  the same input with the same kernel, become one convolution;
* the last 1x1 convolution is split: its 96 decoder channels go before the
  final upsampling (both are linear, so they commute), and its 24 input-patch
  channels are folded into the 3x3 convolution that produces them.

With --half, the weight matrices are stored as IEEE half floats (biases,
normalisations and position biases stay float32): half the size, and on CPUs
with AVX2 they stay half in memory too.

File format, little-endian: the bytes b"BIREFNET", u32 version 1, u32 tensor
count, then per tensor a u16 name length, the UTF-8 name, a u8 type (0 for
float32, 1 for float16), a u8 rank, u32 dimensions, and the values.
"""

import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

BN_EPS = 1e-5
STAGES = [(96, 3, 2), (192, 6, 2), (384, 12, 6), (768, 24, 2)]  # dim, heads, depth


def gemm_conv(w):
    """[out, in, kh, kw] to [(kh, kw, in), out]."""
    o, i, kh, kw = w.shape
    return w.transpose(2, 3, 1, 0).reshape(kh * kw * i, o)


class Source:
    def __init__(self, path):
        model = onnx.load(path)
        consumer = {}
        for node in model.graph.node:
            for name in node.input:
                consumer.setdefault(name, node.op_type)
        self.named = {}
        # Unnamed tensors in file order, apart for convolutions (the decoder)
        # and the rest (the backbone).
        self.anonymous = {"Conv": [], "other": []}
        for t in model.graph.initializer:
            name = t.name.removeprefix("m.")
            array = numpy_helper.to_array(t).astype(np.float64)
            if name.startswith(("bb.", "squeeze_module.", "decoder.")):
                self.named[name] = array
            else:
                kind = "Conv" if consumer.get(t.name) == "Conv" else "other"
                self.anonymous[kind].append((t.name, array))
        self.used = set()
        self.cursor = {"Conv": 0, "other": 0}

    def get(self, name, shape):
        array = self.named[name]
        if array.shape != tuple(shape):
            raise SystemExit(f"{name}: shape {array.shape}, expected {tuple(shape)}")
        self.used.add(name)
        return array

    def next(self, shape, kind="other"):
        """The next unnamed tensor of `kind`, which must have `shape`: the ONNX
        export folded these from the network in module order."""
        queue = self.anonymous[kind]
        if self.cursor[kind] == len(queue):
            raise SystemExit(f"ran out of unnamed tensors looking for {shape}")
        name, array = queue[self.cursor[kind]]
        if array.shape != tuple(shape):
            raise SystemExit(f"{name}: shape {array.shape}, expected {tuple(shape)}")
        self.cursor[kind] += 1
        return array

    def conv(self, shape):
        return self.next(shape, "Conv")


def export(src):
    out = {}

    def put(name, array):
        out[name] = np.ascontiguousarray(array, dtype=np.float32)

    # Backbone. The unnamed tensors run per block: qkv, the relative position
    # bias already gathered into [1, heads, 49, 49], proj, fc1, fc2, and after
    # each stage but the last the patch merging reduction.
    put(
        "patch_embed.weight",
        gemm_conv(src.get("bb.patch_embed.proj.weight", (96, 3, 4, 4))),
    )
    put("patch_embed.bias", src.get("bb.patch_embed.proj.bias", (96,)))
    put("patch_embed.norm.weight", src.get("bb.patch_embed.norm.weight", (96,)))
    put("patch_embed.norm.bias", src.get("bb.patch_embed.norm.bias", (96,)))
    for s, (c, heads, depth) in enumerate(STAGES):
        for b in range(depth):
            p = f"bb.layers.{s}.blocks.{b}."
            q = f"stage{s}.block{b}."
            for norm in ("norm1", "norm2"):
                put(q + norm + ".weight", src.get(p + norm + ".weight", (c,)))
                put(q + norm + ".bias", src.get(p + norm + ".bias", (c,)))
            put(q + "qkv.weight", src.next((c, 3 * c)))
            put(q + "qkv.bias", src.get(p + "attn.qkv.bias", (3 * c,)))
            put(q + "position_bias", src.next((1, heads, 49, 49))[0])
            put(q + "proj.weight", src.next((c, c)))
            put(q + "proj.bias", src.get(p + "attn.proj.bias", (c,)))
            put(q + "fc1.weight", src.next((c, 4 * c)))
            put(q + "fc1.bias", src.get(p + "mlp.fc1.bias", (4 * c,)))
            put(q + "fc2.weight", src.next((4 * c, c)))
            put(q + "fc2.bias", src.get(p + "mlp.fc2.bias", (c,)))
        if s < 3:
            p = f"bb.layers.{s}.downsample."
            put(f"stage{s}.merge.norm.weight", src.get(p + "norm.weight", (4 * c,)))
            put(f"stage{s}.merge.norm.bias", src.get(p + "norm.bias", (4 * c,)))
            put(f"stage{s}.merge.weight", src.next((4 * c, 2 * c)))
        put(f"stage{s}.norm.weight", src.get(f"bb.norm{s}.weight", (c,)))
        put(f"stage{s}.norm.bias", src.get(f"bb.norm{s}.bias", (c,)))

    # Decoder blocks. Their conv_in, ASPP pooling branch, ASPP conv1 and
    # conv_out had their batch norms folded by the ONNX export and lost their
    # names; the gradient attention convolutions of blocks 4..2 follow them.
    blocks = [
        ("squeeze_module.0.", "squeeze.", 2880, 1536, False),
        ("decoder.decoder_block4.", "block4.", 1728, 768, True),
        ("decoder.decoder_block3.", "block3.", 960, 384, True),
        ("decoder.decoder_block2.", "block2.", 480, 192, True),
        ("decoder.decoder_block1.", "block1.", 240, 96, False),
    ]
    for p, q, cin, cout, gdt in blocks:
        put(q + "conv_in.weight", gemm_conv(src.conv((64, cin, 3, 3))))
        put(q + "conv_in.bias", src.conv((64,)))
        put(q + "aspp.pool.weight", gemm_conv(src.conv((256, 64, 1, 1))))
        put(q + "aspp.pool.bias", src.conv((256,)))
        put(q + "aspp.conv1.weight", gemm_conv(src.conv((64, 1280, 1, 1))))
        put(q + "aspp.conv1.bias", src.conv((64,)))
        put(q + "conv_out.weight", gemm_conv(src.conv((cout, 64, 3, 3))))
        put(q + "conv_out.bias", src.conv((cout,)))
        if gdt:
            put(q + "gdt.weight", gemm_conv(src.conv((16, cout, 3, 3))))
            put(q + "gdt.bias", src.conv((16,)))
        branches = [
            ("aspp1", 1),
            ("aspp_deforms.0", 1),
            ("aspp_deforms.1", 3),
            ("aspp_deforms.2", 7),
        ]
        for i, (branch, k) in enumerate(branches):
            d = f"{p}dec_att.{branch}."
            taps = k * k
            offset = src.get(d + "atrous_conv.offset_conv.weight", (2 * taps, 64, k, k))
            modulator = src.get(
                d + "atrous_conv.modulator_conv.weight", (taps, 64, k, k)
            )
            put(
                f"{q}aspp.branch{i}.offset.weight",
                gemm_conv(np.concatenate([offset, modulator])),
            )
            put(
                f"{q}aspp.branch{i}.offset.bias",
                np.concatenate(
                    [
                        src.get(d + "atrous_conv.offset_conv.bias", (2 * taps,)),
                        src.get(d + "atrous_conv.modulator_conv.bias", (taps,)),
                    ]
                ),
            )
            weight = src.get(d + "atrous_conv.regular_conv.weight", (256, 64, k, k))
            gamma = src.get(d + "bn.weight", (256,))
            beta = src.get(d + "bn.bias", (256,))
            mean = src.get(d + "bn.running_mean", (256,))
            var = src.get(d + "bn.running_var", (256,))
            scale = gamma / np.sqrt(var + BN_EPS)
            put(
                f"{q}aspp.branch{i}.weight",
                gemm_conv(weight * scale[:, None, None, None]),
            )
            put(f"{q}aspp.branch{i}.bias", beta - mean * scale)

    for n, c in ((4, 768), (3, 384), (2, 192)):
        put(
            f"block{n}.gdt_attn.weight",
            src.get(f"decoder.gdt_convs_attn_{n}.0.weight", (1, 16, 1, 1)).reshape(
                16, 1
            ),
        )
        put(
            f"block{n}.gdt_attn.bias",
            src.get(f"decoder.gdt_convs_attn_{n}.0.bias", (1,)),
        )
        put(
            f"lateral{n}.weight",
            gemm_conv(src.get(f"decoder.lateral_block{n}.conv.weight", (c, c, 1, 1))),
        )
        put(f"lateral{n}.bias", src.get(f"decoder.lateral_block{n}.conv.bias", (c,)))

    # Input-patch branches: two 3x3 convolutions without an activation.
    for n, cin, cout in ((5, 3072, 192), (4, 768, 192), (3, 192, 96), (2, 48, 48)):
        p = f"decoder.ipt_blk{n}."
        put(
            f"patches{n}.conv1.weight",
            gemm_conv(src.get(p + "conv1.weight", (64, cin, 3, 3))),
        )
        put(f"patches{n}.conv1.bias", src.get(p + "conv1.bias", (64,)))
        put(
            f"patches{n}.conv_out.weight",
            gemm_conv(src.get(p + "conv_out.weight", (cout, 64, 3, 3))),
        )
        put(f"patches{n}.conv_out.bias", src.get(p + "conv_out.bias", (cout,)))

    head = src.get("decoder.conv_out1.0.weight", (1, 120, 1, 1))[0, :, 0, 0]
    put("head.weight", head[:96].reshape(96, 1))
    put("head.bias", src.get("decoder.conv_out1.0.bias", (1,)))
    # The head's 24 patch channels times ipt_blk1.conv_out: one output channel.
    conv_out = src.get("decoder.ipt_blk1.conv_out.weight", (24, 64, 3, 3))
    folded = np.einsum("k,kcij->cij", head[96:], conv_out)[None]
    put(
        "patches1.conv1.weight",
        gemm_conv(src.get("decoder.ipt_blk1.conv1.weight", (64, 3, 3, 3))),
    )
    put("patches1.conv1.bias", src.get("decoder.ipt_blk1.conv1.bias", (64,)))
    put("patches1.conv_out.weight", gemm_conv(folded))
    put(
        "patches1.conv_out.bias",
        np.array([head[96:] @ src.get("decoder.ipt_blk1.conv_out.bias", (24,))]),
    )

    unused = sorted(set(src.named) - src.used)
    left = sum(len(q) - src.cursor[k] for k, q in src.anonymous.items())
    if unused or left:
        raise SystemExit(f"unexported tensors: {unused[:5]} and {left} unnamed")
    return out


def write(path, tensors, half):
    with open(path, "wb") as f:
        f.write(b"BIREFNET")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, array in tensors.items():
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)))
            f.write(encoded)
            as_half = half and array.ndim == 2 and name.endswith(".weight")
            f.write(struct.pack("<BB", int(as_half), array.ndim))
            f.write(struct.pack(f"<{array.ndim}I", *array.shape))
            f.write(array.astype("<f2" if as_half else "<f4").tobytes())


def main():
    args = sys.argv[1:]
    half = "--half" in args
    args = [a for a in args if a != "--half"]
    if len(args) != 2:
        raise SystemExit(__doc__.split("\n\n")[1])
    tensors = export(Source(args[0]))
    write(args[1], tensors, half)
    count = sum(a.size for a in tensors.values())
    print(f"{len(tensors)} tensors, {count} floats")


if __name__ == "__main__":
    main()
