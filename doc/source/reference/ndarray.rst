-------------------
N-Dimensional Array
-------------------
The multidimensional data array class. This consists of a series of parameters and functions that allow not only to construct an array, but also to handle it (for example, copying this to another array, either storing it into the native ironArray format file or converting it into a NumPy array).

.. currentmodule:: iarray_community


Attributes
==========


.. autosummary::
   :toctree: autofiles/iarray
   :nosignatures:

   IArray.blocks
   IArray.chunks
   IArray.cratio
   IArray.data
   IArray.dtype
   IArray.is_plainbuffer
   IArray.ndim
   IArray.shape
   IArray.info


Methods
=======

.. autosummary::
   :toctree: autofiles/iarray
   :nosignatures:

   IArray.copy


Utilities
=========

.. autosummary::
   :toctree: autofiles/iarray
   :nosignatures:

   open
   iarray2numpy
   numpy2iarray
