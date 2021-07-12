import pytest
import numpy as np
from src import iarrayce as ia

shapes_names = "shape, chunks, blocks"
shapes_values = [
    ((55, 123, 72), (10, 12, 25), (2, 3, 7)),
    ((10, 12, 5), None, None),
]
dtype_names = "dtype"
dtype_values = [
    np.float32,
    np.float64,
]


@pytest.mark.parametrize(shapes_names, shapes_values)
@pytest.mark.parametrize(dtype_names, dtype_values)
def test_empty(shape, chunks, blocks, dtype):
    if blocks is None or chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)
    with ia.config(store=store):
        a = ia.empty(shape, dtype=dtype)

    size = np.prod(shape) * np.dtype(dtype).itemsize
    assert dtype == a.dtype
    assert size == a.size
    assert shape == a.shape
    assert chunks == a.chunks
    assert blocks == a.blocks


@pytest.mark.parametrize(shapes_names, shapes_values)
@pytest.mark.parametrize(dtype_names, dtype_values)
def test_zeros(shape, chunks, blocks, dtype):
    if blocks is None or chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)
    with ia.config(store=store):
        a = ia.zeros(shape, dtype=dtype)
    b = ia.iarray2numpy(a)
    np.testing.assert_allclose(b, 0.)


@pytest.mark.parametrize(shapes_names, shapes_values)
@pytest.mark.parametrize(dtype_names, dtype_values)
def test_ones(shape, chunks, blocks, dtype):
    if blocks is None or chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)
    with ia.config(store=store):
        a = ia.ones(shape, dtype=dtype)
    b = ia.iarray2numpy(a)
    np.testing.assert_allclose(b, 1.)



@pytest.mark.parametrize(shapes_names, shapes_values)
@pytest.mark.parametrize(dtype_names, dtype_values)
def test_full(shape, chunks, blocks, dtype):
    fill_value = 3.141516171819
    if blocks is None or chunks is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks)
    with ia.config(store=store):
        a = ia.full(shape, fill_value, dtype=dtype)
    b = ia.iarray2numpy(a)
    np.testing.assert_allclose(b, fill_value)
