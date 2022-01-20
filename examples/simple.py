import iarray_community as ia
import numpy as np
import os
import caterva as cat
import msgpack


shape = (55, 123, 72)
chunks = (10, 12, 25)
blocks = (2, 3, 7)
chunks2 = (55, 1, 72)
blocks2 = (2, 1, 5)

dtype = np.float64

urlpath = "test_open.iarray"
if os.path.exists(urlpath):
    ia.remove(urlpath)

with ia.config(chunks=chunks, blocks=blocks, urlpath=urlpath):

    a = ia.full(shape, 3, dtype=dtype)
    print(a.info)

    b = ia.open(urlpath)
    print(b.info)


if os.path.exists(urlpath):
    ia.remove(urlpath)