#!/usr/bin/env python3
"""Reference outputs from Real-ESRGAN's compact network in PyTorch.

    torch_reference.py MODEL.pth [--weak WDN.pth --strength S] WIDTH HEIGHT RGB_DIR

Writes NAME.torch.out (little-endian float32 RGB rows, four times larger on
each side, neither clamped nor rounded) next to every NAME.rgb in RGB_DIR,
each WIDTH x HEIGHT 8-bit RGB (`evaluate.py prepare` writes them). The
network class restates SRVGGNetCompact from Real-ESRGAN's
realesrgan/archs/srvgg_arch.py, Copyright (c) 2021, Xintao Wang, under the
BSD-3-Clause licence in crates/realesrgan-native/NOTICE (notice A); Real-ESRGAN's package needs basicsr;
loading the checkpoint with strict=True checks it layer by layer. --weak and
--strength blend two checkpoints as RealESRGANer.dni does.
"""

import argparse
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def params(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["params_ema" if "params_ema" in checkpoint else "params"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--weak")
    parser.add_argument("--strength", type=float)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("rgb_dir")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    state = params(args.model)
    if args.weak:
        weak = params(args.weak)
        state = {
            k: args.strength * v + (1 - args.strength) * weak[k]
            for k, v in state.items()
        }
    model = SRVGGNetCompact()
    model.load_state_dict(state, strict=True)
    model.eval()
    for path in sorted(pathlib.Path(args.rgb_dir).glob("*.rgb")):
        data = np.fromfile(path, dtype=np.uint8).reshape(args.height, args.width, 3)
        x = torch.from_numpy(data.transpose(2, 0, 1)[None].astype(np.float32) / 255)
        with torch.no_grad():
            out = model(x)[0].numpy().transpose(1, 2, 0)
        out.astype("<f4").tofile(path.with_suffix(".torch.out"))
        print(path.stem, flush=True)


if __name__ == "__main__":
    main()
