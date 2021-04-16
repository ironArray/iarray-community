import iarrayce as ia
import numpy as np

kwargs = {
    "chunkshape": (25, 25),
    "blockshape": (10, 10),
}
arr = np.arange(100 * 100, dtype=np.float64)
a = ia.from_buffer(arr.tobytes(), (100, 100), itemsize=8, dtype=np.float64, **kwargs)
print(a.info)


b = ia.from_file("data/test_normal_float64_loc3_scale5.iarray")

print(b.info)

c = b[2500:10000]

