import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from matops import CupyMatOps, DataType, MatDevice, MatLib, to_cupy_type, to_type_name


def _gpu_available() -> bool:
    try:
        return bool(cp.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="CuPy CUDA device unavailable")


@pytest.fixture
def ops():
    return CupyMatOps()


def test_cupy_basic_reduction_parity(ops):
    x = ops.mat([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
    result = ops.to_numpy(ops.mean(x, dim=0))
    np.testing.assert_allclose(result, np.array([2.0, 3.0], dtype=np.float32))


def test_cupy_reshape_and_flatten_parity(ops):
    x = ops.mat([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
    np.testing.assert_array_equal(ops.to_numpy(ops.flatten(x)), np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(ops.to_numpy(ops.reshape(x, (4,))), np.array([1, 2, 3, 4]))


def test_cupy_backend_device_and_dtype_detection(ops):
    x = ops.zeros((2,), dtype=cp.float32)
    assert MatLib.which(x) is MatLib.CUPY
    assert MatDevice.which(x) is MatDevice.CUDA
    assert DataType.which(x) is DataType.FLOAT32


def test_cupy_type_helpers():
    assert to_type_name(cp.uint8) == "uint8"
    assert to_cupy_type(np.float32) is cp.float32


def test_cupy_explicit_device(ops):
    x = ops.ones((2,), dtype=cp.float32, device="cuda:0")
    assert x.device.id == 0
