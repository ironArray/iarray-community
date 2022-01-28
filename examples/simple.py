import iarray_community as ia
import numpy as np
import os
import caterva as cat
import msgpack


shape = (10, 10)
chunks = (5, 5)
blocks = (2, 3)

dtype = np.float64

urlpath = "test_open.iarray"
if os.path.exists(urlpath):
    ia.remove(urlpath)

with ia.config(chunks=chunks, blocks=blocks, urlpath=urlpath):

    a = ia.full(shape, 3, dtype=dtype)
    print(a.info)

    b = ia.open(urlpath)
    print(b.info)

    b.resize((20, 10))

    print(b.info)

if os.path.exists(urlpath):
    ia.remove(urlpath)