import cat4py as cat
from .iarray import Array
import numpy as np


def empty(shape, dtype, **kwargs):
    dtype = np.dtype(dtype)
    arr = Array(dtype, **kwargs)
    kwargs = arr.kwargs
    cat.ext.empty(arr, shape, dtype.itemsize, **kwargs)
    return arr

def zeros():
    pass

def from_buffer(buffer, shape, itemsize, dtype, **kwargs):

    arr = Array(dtype, **kwargs)
    kwargs = arr.kwargs
    cat.ext.from_buffer(arr, buffer, shape, itemsize, **kwargs)
    return arr


def copy(array, **kwargs):

    arr = Array(array._dtype, **kwargs)
    kwargs = arr.kwargs
    cat.ext.copy(arr, array, **kwargs)

    return arr


def open(urlpath):

    arr = cat.NDArray()
    cat.ext.from_file(arr, urlpath)
    print("hola")

    if "iarray" in arr.meta:
        arr = Array.cast(arr)
    else:
        raise AttributeError(f"File {urlpath} not contains an ironArray object")

    return arr
