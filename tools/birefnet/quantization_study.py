#!/usr/bin/env python3
"""What lower precision would do to BiRefNet_lite's masks, simulated in PyTorch.

    quantization_study.py RGB_DIR VARIANT [--threads N]

Runs BiRefNet_lite's own code (see torch_reference.py) with every linear and
convolution layer, deformable ones included, rounded as VARIANT says, and
writes NAME.VARIANT.logits next to every NAME.rgb. The deformable layer's
forward restates DeformableConv2d's from BiRefNet's
models/modules/deform_conv.py (Copyright (c) 2024 ZhengPeng, under the MIT
licence in crates/birefnet-native/NOTICE, notice A). Compare with
`evaluate.py compare RGB_DIR torch VARIANT`.

    fp16, bf16  weights rounded to that format
    w8          weights to int8, symmetric, one scale per output channel
    w8a8r       w8, and inputs to uint8 with a scale and zero point per row
                (per token for linear layers, per pixel for convolutions)
    w8a8t       w8, and inputs to uint8 with one scale per tensor
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_reference import MEAN, REVISION, STD
from torchvision.ops import deform_conv2d
from transformers import AutoModelForImageSegmentation

VARIANTS = ("fp16", "bf16", "w8", "w8a8r", "w8a8t")


def round_weight(w, variant):
    if variant == "fp16":
        return w.half().float()
    if variant == "bf16":
        return w.bfloat16().float()
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(1).clamp_min(1e-12) / 127
    return (
        torch.round(flat / scale[:, None]).clamp(-127, 127) * scale[:, None]
    ).reshape(w.shape)


def round_input(x, channel_dim, variant):
    if not variant.startswith("w8a8"):
        return x
    if variant == "w8a8t":
        lo, hi = x.min(), x.max()
    else:
        lo = x.amin(dim=channel_dim, keepdim=True)
        hi = x.amax(dim=channel_dim, keepdim=True)
    lo = torch.minimum(lo, torch.zeros_like(lo))
    hi = torch.maximum(hi, torch.zeros_like(hi))
    scale = ((hi - lo) / 255).clamp_min(1e-12)
    zero = torch.round(-lo / scale)
    return (torch.round(x / scale + zero).clamp(0, 255) - zero) * scale


def patch(model, variant):
    def linear(self, x):
        return F.linear(
            round_input(x, -1, variant), round_weight(self.weight, variant), self.bias
        )

    def conv(self, x):
        return self._conv_forward(
            round_input(x, 1, variant), round_weight(self.weight, variant), self.bias
        )

    def deformable(self, x):
        x = round_input(x, 1, variant)
        offset = F.conv2d(
            x,
            round_weight(self.offset_conv.weight, variant),
            self.offset_conv.bias,
            padding=self.padding,
        )
        modulator = 2.0 * torch.sigmoid(
            F.conv2d(
                x,
                round_weight(self.modulator_conv.weight, variant),
                self.modulator_conv.bias,
                padding=self.padding,
            )
        )
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=round_weight(self.regular_conv.weight, variant),
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )

    nn.Linear.forward = linear
    nn.Conv2d.forward = conv
    sys.modules[type(model).__module__].DeformableConv2d.forward = deformable


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rgb_dir")
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet_lite", trust_remote_code=True, revision=REVISION
    ).eval()
    patch(model, args.variant)
    for path in sorted(pathlib.Path(args.rgb_dir).glob("*.rgb")):
        data = np.fromfile(path, dtype=np.uint8)
        side = round((data.size // 3) ** 0.5)
        x = (
            data.reshape(side, side, 3).astype(np.float32) / np.float32(255.0) - MEAN
        ) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
        with torch.no_grad():
            logits = model(x)[-1]
        logits.numpy().astype("<f4").tofile(path.with_suffix(f".{args.variant}.logits"))
        print(path.stem, flush=True)


if __name__ == "__main__":
    main()
