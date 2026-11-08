#!/usr/bin/env python3
"""Reference outputs from SCUNet's own PyTorch code and weights.

    torch_reference.py SCUNET_SRC MODEL.pth WIDTH HEIGHT RGB_DIR [--threads N]

SCUNET_SRC is a checkout of https://github.com/cszn/SCUNet (checked against
commit 52e440a80a655b01e0b41e9dd9bfe599bc11625e), whose network code this
runs; it needs torch, einops, timm and thop. Writes NAME.torch.out
(little-endian float32 RGB rows, neither clamped nor rounded) next to every
NAME.rgb in RGB_DIR, each WIDTH x HEIGHT 8-bit RGB (`evaluate.py prepare`
writes them).
"""

import argparse
import pathlib
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("model")
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("rgb_dir")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from models.network_scunet import SCUNet

    torch.set_num_threads(args.threads)
    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
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
