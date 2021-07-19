-------------------
N-Dimensional Array
-------------------
The multidimensional data array class. This class consists of a set of useful parameters and functions that allow not only to define an array correctly, but also to handle it in a simple way, being able to copy this array and transform it from a file or NumPy format to the ironArray format.

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
