import pytest
import numpy as np
import iarray_community as ia

shapes_names = "shape, newshape, chunks, blocks"
shapes_values = [
    ((10, 10), (200, 10), (5, 5), (3, 2)),
    ((100, 100, 100), (200, 200, 101), (20, 20, 20), (20, 20, 5)),
    ((55, 123, 72), (200, 123, 72), (10, 12, 25), (2, 3, 7)),
]
dtype_names = "dtype"
dtype_values = [
    np.float32,
    np.float64,
]


@pytest.mark.parametrize(shapes_names, shapes_values)
@pytest.mark.parametrize(dtype_names, dtype_values)
def test_resize(shape, newshape, chunks, blocks, dtype):
    with ia.config(chunks=chunks, blocks=blocks):
        a = ia.ones(shape, dtype=dtype)

    assert a.shape == shape

    a.resize(newshape)

    assert a.shape == newshape

    slides = tuple(slice(s) for s in shape)
    np.testing.assert_allclose(a[slides], 1)
