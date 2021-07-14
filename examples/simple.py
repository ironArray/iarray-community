import iarray_community as ia
import numpy as np

arr = np.arange(1000 * 1000, dtype=np.float64)
shape = (1000, 1000)

with ia.config(chunks=(250, 250), blocks=(100, 100), clevel=9) as cfg:
    a = ia.empty(shape)
    print(a.cratio)

    with ia.config(clevel=2, codec=ia.Codecs.LZ4):
        a2 = ia.ones((1000, 1000), dtype=np.float64, clevel=3)
        print(a2.cratio)

a3 = a2.copy(chunks=(200, 200), blocks=(10, 200))
print(a3.meta.keys())

print(a3.info)

iarr = ia.ones((3, 2), dtype=np.float32)

print("iarr.shape->", iarr.shape)
print("first time:", iarr[1])
print("second time:", iarr[1])

b = ia.open("data/test_normal_float64_loc3_scale5.iarray")

print(b.info)

c = b[2500:10000]

print(type(c))
