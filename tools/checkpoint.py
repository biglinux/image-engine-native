"""Reading a PyTorch checkpoint without PyTorch.

A checkpoint saved by torch.save is a zip of a pickle and the raw data of its
tensors. load() unpickles it with an unpickler that builds numpy arrays for
the tensors, and dicts, ordered dicts, lists and ints for the containers
around them, and nothing else: any other class the pickle names is refused,
or with inert=True replaced by a placeholder that takes any arguments and
state and holds nothing, for checkpoints that also store their training
configuration. Nothing in the file can run code, and a tensor that would
read past its storage is refused.
"""

import collections
import operator
import pickle
import zipfile

import numpy as np

STORAGES = {"FloatStorage": "<f4", "LongStorage": "<i8"}
CONTAINERS = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("builtins", "dict"): dict,
    ("builtins", "list"): list,
    ("builtins", "int"): int,
}


class Inert:
    """Stands for a class the checkpoint names that is not a tensor or a
    plain container: built from anything, holding nothing, running nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __setitem__(self, key, value):
        pass

    def append(self, value):
        pass

    def extend(self, values):
        pass


def rebuild_tensor(storage, offset, size, stride, *_):
    # as_strided reads wherever the strides lead, so the view must stay
    # inside a one-dimensional storage, its bounds summed in Python integers,
    # which do not wrap as numpy's do.
    if not (isinstance(storage, np.ndarray) and storage.ndim == 1):
        raise pickle.UnpicklingError("tensor storage is not a plain array")
    offset = operator.index(offset)
    size = [operator.index(n) for n in size]
    stride = [operator.index(s) for s in stride]
    if len(size) != len(stride) or min([offset, *size, *stride]) < 0:
        raise pickle.UnpicklingError(f"bad tensor view {offset} {size} {stride}")
    if 0 not in size:
        last = offset + sum((n - 1) * s for n, s in zip(size, stride, strict=True))
        if last >= len(storage):
            raise pickle.UnpicklingError(
                f"tensor view past its storage of {len(storage)}"
            )
    item = storage.itemsize
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=size, strides=[s * item for s in stride]
    )
    return view.copy()


class Unpickler(pickle.Unpickler):
    def __init__(self, archive, prefix, inert):
        self.archive, self.prefix, self.inert = archive, prefix, inert
        super().__init__(archive.open(f"{prefix}/data.pkl"))

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild_tensor
        if (module, name) in CONTAINERS:
            return CONTAINERS[module, name]
        if module == "torch" and name in STORAGES:
            return STORAGES[name]
        if self.inert:
            return Inert
        raise pickle.UnpicklingError(f"refusing {module}.{name}")

    def persistent_load(self, pid):
        kind, dtype, key, _location, count = pid
        if kind != "storage" or dtype not in STORAGES.values():
            raise pickle.UnpicklingError(f"unexpected storage {pid!r}")
        data = self.archive.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(data, dtype=dtype, count=count)


def load(file, inert=False):
    """The object the checkpoint `file` (a path or a binary file) holds,
    with numpy arrays for its tensors."""
    with zipfile.ZipFile(file) as archive:
        prefix = archive.namelist()[0].split("/")[0]
        return Unpickler(archive, prefix, inert).load()
