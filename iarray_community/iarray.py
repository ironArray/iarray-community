import caterva as cat
import numpy as np
from .info import InfoReporter
import iarray_community as ia
import os


class IArray(cat.NDArray):
    def __init__(self, **kwargs):
        self.pre_init(**kwargs)
        self._cfg = ia.Config(**kwargs)
        super(IArray, self).__init__(**self._cfg.cat_kwargs)

    def pre_init(self, **kwargs):
        dtype = kwargs["dtype"]
        if dtype != np.float64 and dtype != np.float32:
            raise AttributeError("dtype can only be np.float32 or np.float64")
        self._dtype = dtype

        urlpath = kwargs["urlpath"]
        if urlpath and os.path.exists(urlpath):
            raise FileExistsError("Remove file first!")

    @classmethod
    def cast(cls, cont):
        cont.__class__ = cls
        assert isinstance(cont, IArray)
        cont._dtype = np.float64 if cont.itemsize == 8 else np.float32
        return cont

    @property
    def dtype(self):
        """The data type for the container."""
        return self._dtype

    @property
    def data(self):
        return self[:]

    @property
    def is_plainbuffer(self):
        return self.chunks is None

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
        items += [("shape", self.shape)]
        items += [("chunks", self.chunks)]
        items += [("blocks", self.blocks)]
        items += [("cratio", f"{self.cratio:.2f}")]
        return items

    def __getitem__(self, key):
        return np.array(self.slice(key)).view(self.dtype)

    def slice(self, key, **kwargs):
        arr = super(IArray, self).slice(key, **kwargs)
        return self.cast(arr)

    def copy(self, **kwargs):
        return ia.copy(self, **kwargs)
