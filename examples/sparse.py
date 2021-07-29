# Example on how to read a sparse file inside iarray
# A comparison is done against h5py

import iarray_community as ia
import scipy.io
import numpy as np
import h5py
import hdf5plugin
from time import time
import os

# Downnload from https://sparse.tamu.edu/ML_Graph/worms20_10NN
sparse_path = os.path.expanduser("~/Downloads/worms20_10NN/worms20_10NN.mtx")
w = scipy.io.mmread(sparse_path).toarray()

t0 = time()
wia = ia.empty(w.shape, chunks=(500, 2000), blocks=(250, 500),
               codec=ia.Codecs.ZSTD, clevel=1, filters=[])
wia[:] = w
print("Time to store (iarray): %.3fs" % (time() - t0))
csize_MB = np.prod(w.shape) * w.itemsize / wia.cratio / 2**20
print("Size (iarray): %.3fMB" % (csize_MB))

t0 = time()
f = h5py.File('worms20_10NN.h5', 'w')
f.create_dataset("data", w.shape, chunks=(500, 2000), data=w,
                 **hdf5plugin.Blosc(cname='zstd', clevel=1, shuffle=0))
print("Time to store (hdf5): %.3fs" % (time() - t0))
file_size = os.path.getsize('worms20_10NN.h5')
csize_MB = file_size / 2**20
print("Size (hdf5): %.3fMB" % (csize_MB))
