import struct

import caterva as cat
from .iarray import IArray, add_meta
from .config_params import *



def empty(shape, **kwargs):
    """Return an empty array.

    An empty array has no data and needs to be filled via a write iterator.

    Parameters
    ----------
    shape : tuple, list
        The shape of the array to be created.
    kwargs : dict
        A dictionary for setting some or all of the fields in the Config
        dataclass that should override the current configuration.

    Returns
    -------
    IArray
        The new array.
    """
    with config(**kwargs) as cfg:
        dtype = np.dtype(cfg.dtype)
        arr = IArray(**cfg.kwargs)
        kwargs = add_meta(dtype, **arr._cfg.cat_kwargs)
        cat.ext.empty(arr, shape, dtype.itemsize, **kwargs)
    return arr


def zeros(shape, **kwargs):
    """Return a new array of given shape and type, filled with zeros.

    `shape` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    IArray
        The new array.

    See Also
    --------
    empty : Create an empty array.
    ones : Create an array filled with ones.
    """
    with config(**kwargs) as cfg:
        dtype = np.dtype(cfg.dtype)
        arr = IArray(**cfg.kwargs)
        kwargs = add_meta(dtype, **arr._cfg.cat_kwargs)
        cat.ext.zeros(arr, shape, dtype.itemsize, **kwargs)
    return arr


def full(shape, fill_value, **kwargs):
    """Return a new array of given shape and type, filled with `fill_value`.

    `shape` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    IArray
        The new array.

    See Also
    --------
    empty : Create an empty array.
    zeros : Create an array filled with zeros.
    """
    with config(**kwargs) as cfg:
        dtype = np.dtype(cfg.dtype)
        arr = IArray(**cfg.kwargs)
        kwargs = add_meta(arr.dtype, **arr._cfg.cat_kwargs)
        fill_bytes = dtype.type(fill_value).tobytes()
        cat.ext.full(arr, shape, fill_bytes, **kwargs)
    return arr


def ones(shape, **kwargs):
    """Return a new array of given shape and type, filled with ones.

    `shape` and `kwargs` are the same than for :func:`empty`.

    Returns
    -------
    IArray
        The new array.

    See Also
    --------
    empty : Create an empty array.
    zeros : Create an array filled with zeros.
    """
    return full(shape, 1., **kwargs)

