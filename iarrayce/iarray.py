import cat4py as cat
import numpy as np


class Array(cat.NDArray):
    def __init__(self, dtype, **kwargs):
        if type(self) == Array:
            self.pre_init(dtype, **kwargs)
        super(Array, self).__init__(**self.kwargs)

    def pre_init(self, dtype, **kwargs):
        if dtype != np.float64 and dtype != np.float32:
            raise AttributeError("dtype can only be np.float32 or np.float64")

        self._dtype = dtype
        if "metalayers" not in kwargs:
            kwargs["metalayers"] = {}
        sdata = b"000" if dtype == np.float64 else b"010"
        kwargs["metalayers"]["iarray"] = sdata

        self.kwargs = kwargs


    @classmethod
    def cast(cls, cont):
        cont.__class__ = cls
        assert isinstance(cont, Array)
        cont._dtype = np.float64 if cont.itemsize == 8 else np.float32
        return cont

    @property
    def dtype(self):
        """The data type for the container."""
        return self._dtype

    def __getitem__(self, key):
        """ Get a (multidimensional) slice as specified in key.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter is not honored yet in
            slices.

        Returns
        -------
        out: NDTArray
            An array, stored in a non-compressed buffer, with the requested data.
        """
        return np.array(self.slice(key)).view(self.dtype)

    def slice(self, key, **kwargs):
        """ Get a (multidimensional) slice as specified in key. Generalizes :py:meth:`__getitem__`.

       Parameters
       ----------
       key: int, slice or sequence of slices
           The index for the slices to be updated. Note that step parameter is not honored yet in
           slices.

       Other Parameters
       ----------------
       kwargs: dict, optional
           Keyword arguments that are supported by the :py:meth:`cat4py.empty` constructor.

       Returns
       -------
       out: NDTArray
           An array with the requested data.
       """
        arr = super(Array, self).slice(key, **kwargs)
        return self.cast(arr)
