#!/usr/bin/env python3
"""Reference logits from BiRefNet_lite's own PyTorch code and weights.

    torch_reference.py RGB_DIR [--threads N] [--revision REV]

Writes NAME.torch.logits (little-endian float32 rows) next to every NAME.rgb
in RGB_DIR (`evaluate.py prepare` writes those). Needs torch, torchvision,
transformers, timm, einops, kornia and Hugging Face access; this runs the code
published with the model, so pin the revision you have reviewed.
"""

import argparse
import pathlib

import numpy as np
import torch
from transformers import AutoModelForImageSegmentation

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# ZhengPeng7/BiRefNet_lite revision the weights files were checked against.
REVISION = "aa62cd87eafb9cc43056d08ef3615a14628b831d"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rgb_dir")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet_lite", trust_remote_code=True, revision=args.revision
    ).eval()
    for path in sorted(pathlib.Path(args.rgb_dir).glob("*.rgb")):
        data = np.fromfile(path, dtype=np.uint8)
        side = round((data.size // 3) ** 0.5)
        x = (
            data.reshape(side, side, 3).astype(np.float32) / np.float32(255.0) - MEAN
        ) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
        with torch.no_grad():
            logits = model(x)[-1]
        logits.numpy().astype("<f4").tofile(path.with_suffix(".torch.logits"))
        print(path.stem, flush=True)


if __name__ == "__main__":
    main()
