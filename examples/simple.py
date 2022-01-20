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

    aux = np.linspace(0, 1, int(np.prod(shape)), dtype=dtype).reshape(shape)
    # a = ia.numpy2iarray(aux)
    a = ia.full(shape, 3, dtype=dtype)
    print(a.info)

    b = cat.open(urlpath)
    print(msgpack.unpackb(b.meta["caterva"]))
    print(b.meta["iarray"])

    print(b.itemsize)

if os.path.exists(urlpath):
    ia.remove(urlpath)