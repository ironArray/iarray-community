import struct

import caterva as cat
from .iarray import IArray
from .config_params import *


def add_meta(dtype, **kwargs):
    if "meta" not in kwargs:
        kwargs["meta"] = {}
    sdata = b"000" if dtype == np.float64 else b"010"
    kwargs["meta"]["iarray"] = sdata
    return kwargs


def empty(shape, **kwargs):
    with config(**kwargs) as cfg:
        dtype = np.dtype(cfg.dtype)
        arr = IArray(**cfg.kwargs)
        kwargs = add_meta(dtype, **arr._cfg.cat_kwargs)
        cat.ext.empty(arr, shape, dtype.itemsize, **kwargs)
    return arr


def zeros(shape, **kwargs):
    with config(**kwargs) as cfg:
        dtype = np.dtype(cfg.dtype)
        arr = IArray(**cfg.kwargs)
        kwargs = add_meta(dtype, **arr._cfg.cat_kwargs)
        cat.ext.zeros(arr, shape, dtype.itemsize, **kwargs)
    return arr


def full(shape, fill_value, **kwargs):
    with config(**kwargs) as cfg:
        dtype = np.dtype(cfg.dtype)
        arr = IArray(**cfg.kwargs)
        kwargs = add_meta(dtype, **arr._cfg.cat_kwargs)
        if dtype.itemsize == 8:
            fill_bytes = struct.pack("d", fill_value)
        else:
            fill_bytes = struct.pack("f", fill_value)
        cat.ext.full(arr, shape, fill_bytes, **kwargs)
    return arr


def ones(shape, **kwargs):
    return full(shape, 1., **kwargs)

