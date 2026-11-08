#!/usr/bin/env python3
"""Reference predictions from LaMa's own generator code in PyTorch.

    torch_reference.py LAMA_SRC big-lama.zip WIDTH HEIGHT RGB_DIR [--threads N]

LAMA_SRC is a checkout of https://github.com/advimman/lama (checked against
commit 786f5936b27fb3dacd2b1ad799e4de968ea697e7). This runs its
saicinpainting/training/modules/ffc.py as is, with the four modules it
imports replaced by what the Big LaMa generator uses of them:
get_activation, restated from base.py (Copyright 2021 Samsung Research,
under the Apache-2.0 licence in crates/lama-native/NOTICE, notice A), and
names it never calls. The weights are
read with tools/checkpoint.py and loaded with strict=True, which
checks every tensor; the generator's settings are those of
configs/training/big-lama.yaml.

For every NAME.rgb in RGB_DIR (WIDTH x HEIGHT 8-bit RGB) with its NAME.mask
(a byte per pixel, not zero to fill), writes the generator's prediction for
every pixel as NAME.torch.out (little-endian float32 RGB rows), after padding
to a multiple of 8 as LaMa's predict.py does.
"""

import argparse
import importlib.util
import io
import pathlib
import sys
import types
import zipfile

import numpy as np
import torch
from torch import nn


def get_activation(kind="tanh"):
    if kind == "tanh":
        return nn.Tanh()
    if kind == "sigmoid":
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f"Unknown activation kind {kind}")


def load_ffc(source):
    stubs = {
        "saicinpainting": {},
        "saicinpainting.training": {},
        "saicinpainting.training.modules": {},
        "saicinpainting.training.modules.base": {
            "get_activation": get_activation,
            "BaseDiscriminator": nn.Module,
        },
        "saicinpainting.training.modules.spatial_transform": {
            "LearnableSpatialTransformWrapper": None
        },
        "saicinpainting.training.modules.squeeze_excitation": {"SELayer": None},
        "saicinpainting.utils": {"get_shape": None},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
    path = pathlib.Path(source) / "saicinpainting/training/modules/ffc.py"
    spec = importlib.util.spec_from_file_location("ffc", path)
    ffc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ffc)
    return ffc


def state_dict(archive):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from checkpoint import load

    with zipfile.ZipFile(archive) as outer:
        ckpt = outer.read("big-lama/models/best.ckpt")
    state = load(io.BytesIO(ckpt), inert=True)["state_dict"]
    return {
        k.removeprefix("generator."): torch.from_numpy(np.array(v))
        for k, v in state.items()
        if k.startswith("generator.")
    }


def pad_to_8(array):
    """np.pad "symmetric" of the bottom rows and right columns, as LaMa's
    pad_img_to_modulo."""
    h, w = array.shape[-2:]
    pad = ((0, 0), (0, -h % 8), (0, -w % 8))
    return np.pad(array, pad, mode="symmetric")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("archive")
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("rgb_dir")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    ffc = load_ffc(args.source)
    fixed = {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}
    generator = ffc.FFCResNetGenerator(
        input_nc=4,
        output_nc=3,
        ngf=64,
        n_downsampling=3,
        n_blocks=18,
        add_out_act="sigmoid",
        init_conv_kwargs=fixed,
        downsample_conv_kwargs=fixed,
        resnet_conv_kwargs={"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
    )
    generator.load_state_dict(state_dict(args.archive), strict=True)
    generator.eval()
    h, w = args.height, args.width
    for path in sorted(pathlib.Path(args.rgb_dir).glob("*.rgb")):
        image = np.fromfile(path, np.uint8).reshape(h, w, 3).transpose(2, 0, 1)
        image = image.astype(np.float32) / 255
        mask = np.fromfile(path.with_suffix(".mask"), np.uint8).reshape(1, h, w)
        mask = (mask > 0).astype(np.float32)
        image, mask = pad_to_8(image), pad_to_8(mask)
        x = torch.from_numpy(np.concatenate([image * (1 - mask), mask])[None])
        with torch.no_grad():
            predicted = generator(x)[0].numpy()[:, :h, :w].transpose(1, 2, 0)
        predicted.astype("<f4").tofile(path.with_suffix(".torch.out"))
        print(path.stem, flush=True)


if __name__ == "__main__":
    main()
