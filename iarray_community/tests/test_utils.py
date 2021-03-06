import pytest
import numpy as np
import iarray_community as ia
import os


shapes_names = "shape, chunks, blocks"
shapes_values = [
    ((55, 123, 72), (10, 12, 25), (2, 3, 7)),
]
dtype_names = "dtype"
dtype_values = [
    np.float32,
    np.float64,
]


@pytest.mark.parametrize(shapes_names, shapes_values)
@pytest.mark.parametrize(dtype_names, dtype_values)
def test_numpy(shape, chunks, blocks, dtype):
    with ia.config(chunks=chunks, blocks=blocks):
        b = np.linspace(0, 1, int(np.prod(shape)), dtype=dtype).reshape(shape)
        c = ia.numpy2iarray(b)
        assert c.chunks == chunks
        assert c.blocks == blocks
        d = ia.iarray2numpy(c)

        np.testing.assert_allclose(b, d)


shapes_names = "shape, chunks, blocks, chunks2, blocks2"
shapes_values = [
    ((55, 123, 72), (10, 12, 25), (2, 3, 7), (55, 1, 72), (2, 1, 5)),
    ((55, 123), (10, 12), (2, 3), (10, 12), (2, 3)),
]


@pytest.mark.parametrize(shapes_names, shapes_values)
@pytest.mark.parametrize(dtype_names, dtype_values)
def test_copy(shape, chunks, blocks, chunks2, blocks2, dtype):
    with ia.config(chunks=chunks, blocks=blocks):
        b = np.linspace(0, 1, int(np.prod(shape)), dtype=dtype).reshape(shape)
        c = ia.numpy2iarray(b)
        assert c.chunks == chunks
        assert c.blocks == blocks
        d = ia.copy(c, chunks=chunks2, blocks=blocks2, clevel=0)
        e = d[:]

        np.testing.assert_allclose(b, e)


shapes_values = [
    ((55, 123, 72), (10, 12, 25), (2, 3, 7), (55, 1, 72), (2, 1, 5)),
    ((55, 123), (10, 12), (2, 3), (10, 12), (2, 3)),
]

const_names = "const"
const_values = [
    "empty",
    "zeros",
    "ones",
    "copy",
    "numpy2iarray"
]

@pytest.mark.parametrize(shapes_names, shapes_values)
@pytest.mark.parametrize(dtype_names, dtype_values)
@pytest.mark.parametrize(const_names, const_values)
def test_open(shape, chunks, blocks, chunks2, blocks2, dtype, const):
    urlpath = "test_open.iarray"
    if os.path.exists(urlpath):
        ia.remove(urlpath)

    with ia.config(chunks=chunks, blocks=blocks, urlpath=urlpath):
        if const == "empty":
            a = ia.empty(shape)
        elif const == "zeros":
            a = ia.zeros(shape)
        elif const == "ones":  # Also full
            a = ia.ones(shape)
        elif const == "numpy2iarray":
            aux = np.linspace(0, 1, int(np.prod(shape)), dtype=dtype).reshape(shape)
            a = ia.numpy2iarray(aux)
        else:
            aux = ia.ones(shape, urlpath=None)
            a = ia.copy(aux, chunks=chunks2, blocks=blocks2)

        b = ia.open(urlpath)

        np.testing.assert_allclose(a[:], b[:])

    if os.path.exists(urlpath):
        ia.remove(urlpath)
