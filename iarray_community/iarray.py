import caterva as cat
import msgpack
import numpy as np
from .info import InfoReporter
import iarray_community as ia
import os


dtype_to_meta = {
    np.dtype('float64'): 0,
    np.dtype('float32'): 1,
    # np.dtype('float16'): 2,
    # np.dtype('float8'): 3,
    np.dtype('int64'): 10,
    np.dtype('int32'): 11,
    np.dtype('int16'): 12,
    np.dtype('int8'): 13,
    np.dtype('uint64'): 16,
    np.dtype('uint32'): 17,
    np.dtype('uint16'): 18,
    np.dtype('uint8'): 19,
    np.dtype('bool'): 24
}
meta_to_dtype = {v: k for k, v in dtype_to_meta.items()}

supported_dtypes = list(dtype_to_meta.keys())


def add_meta(dtype, **kwargs):
    if "meta" not in kwargs:
        kwargs["meta"] = {}
    s_version = 0
    s_dtype = dtype_to_meta[dtype]
    s_unused = 0
    s_meta = msgpack.packb([s_version, s_dtype, s_unused])
    kwargs["meta"]["iarray"] = s_meta
    return kwargs


class IArray(cat.NDArray):
    def __init__(self, **kwargs):
        self.pre_init(**kwargs)
        self._cfg = ia.Config(**kwargs)
        super(IArray, self).__init__(**self._cfg.cat_kwargs)

    def pre_init(self, **kwargs):
        dtype = np.dtype(kwargs["dtype"])
        if dtype not in dtype_to_meta:
            raise AttributeError("dtype is not supported")
        self._dtype = dtype

        urlpath = kwargs["urlpath"]
        if urlpath and os.path.exists(urlpath):
            raise FileExistsError("Remove file first!")

    @classmethod
    def cast(cls, cont):
        cont.__class__ = cls
        assert isinstance(cont, IArray)
        iarray_meta = cont.meta["iarray"]
        _, s_dtype, _ = msgpack.unpackb(iarray_meta)
        cont._dtype = meta_to_dtype[s_dtype]

        return cont

    @property
    def dtype(self):
        """
        The data type for the container.
        """
        return self._dtype

    @property
    def data(self):
        """
        Get a ndarray with array data.
        """
        return self[:]

    @property
    def info(self):
        """
        Print information about this array.
        """
        return InfoReporter(self)

    @property
    def info_items(self):
        items = []
        items += [("type", self.__class__.__name__)]
        items += [("dtype", self.dtype)]
        items += [("shape", self.shape)]
        items += [("chunks", self.chunks)]
        items += [("blocks", self.blocks)]
        items += [("cratio", f"{self.cratio:.2f}")]
        return items

    def __getitem__(self, key):
        return super(IArray, self).__getitem__(key).view(self.dtype)

    def slice(self, key, **kwargs):
        kwargs = add_meta(self.dtype, **kwargs)
        arr = super(IArray, self).slice(key, **kwargs)
        return self.cast(arr)

    def copy(self, **kwargs):
        return ia.copy(self, **kwargs)

    def resize(self, newshape):
        super(IArray, self).resize(newshape)
        return self

