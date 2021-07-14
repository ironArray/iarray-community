import numpy as np
import iarray_community as ia
import caterva as cat
from .constructors import add_meta

def iarray2numpy(iarr) -> np.ndarray:
    """Convert an ironArray array into a NumPy array.

    Parameters
    ----------
    iarr : IArray
        The array to convert.

    Returns
    -------
    np.ndarray
        The new NumPy array.

    See Also
    --------
    numpy2iarray
    """
    return iarr[:]


def numpy2iarray(ndarray, **kwargs) -> ia.IArray:
    """Convert a NumPy array into an ironArray array.

    `kwargs` are the same than for :func:`empty`.

    Parameters
    ----------
    arr : np.ndarray
        The array to convert.

    Returns
    -------
    IArray
        The new ironArray array.

    See Also
    --------
    iarray2numpy
    """
    with ia.config(**kwargs) as cfg:
        kwargs = cfg.kwargs
        if ndarray.dtype == np.float64:
            kwargs["dtype"] = np.dtype(np.float64)
        elif ndarray.dtype == np.float32:
            kwargs["dtype"] = np.dtype(np.float32)
        else:
            raise NotImplementedError("Only float32 and float64 types are supported for now")
        iarr = ia.IArray(**kwargs)
        kwargs = add_meta(kwargs["dtype"], **iarr._cfg.cat_kwargs)
        cat.ext.asarray(iarr, ndarray, **kwargs)
        return iarr


def copy(array, **kwargs):
    with ia.config(**kwargs) as cfg:
        kwargs = cfg.kwargs
        if array.dtype == np.float64:
            kwargs["dtype"] = np.dtype(np.float64)
        elif array.dtype == np.float32:
            kwargs["dtype"] = np.dtype(np.float32)
        else:
            raise NotImplementedError("Only float32 and float64 types are supported for now")
        arr = ia.IArray(**kwargs)
        kwargs = add_meta(kwargs["dtype"], **arr._cfg.cat_kwargs)
        cat.ext.copy(arr, array, **kwargs)

    return arr


def open(urlpath):
    """Open an array from a binary file in ironArray ``.iarray`` format. The array data will lazily
    be read when necessary.

    Parameters
    ----------
    filename : str
        The file name to read.

    Returns
    -------
    IArray
        The new opened array.

    """
    arr = cat.NDArray()
    cat.ext.from_file(arr, urlpath)

    if "iarray" in arr.meta:
        arr = ia.IArray.cast(arr)
    else:
        raise AttributeError(f"File {urlpath} not contains an ironArray object")

    return arr


def remove(urlpath):
    cat.remove(urlpath)
