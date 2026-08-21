import ctypes

import numpy as np
import pytest
import torch

from matops import (
    DataType,
    MatDevice,
    MatLib,
    NumpyMatOps,
    TorchMatOps,
    to_ctypes_type,
    to_np_type,
    to_torch_type,
    to_type_name,
)


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "numpy":
        return NumpyMatOps(), np.float32
    return TorchMatOps(), torch.float32


def test_basic_reduction_parity(backend):
    ops, dtype = backend
    x = ops.mat([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    result = ops.to_numpy(ops.mean(x, dim=0))
    np.testing.assert_allclose(result, np.array([2.0, 3.0]))


def test_reshape_and_flatten_parity(backend):
    ops, dtype = backend
    x = ops.mat([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    flat = ops.to_numpy(ops.flatten(x))
    reshaped = ops.to_numpy(ops.reshape(x, (4,)))
    np.testing.assert_array_equal(flat, reshaped)


def test_box_conversion_round_trip():
    boxes = np.array([[10.0, 20.0, 40.0, 60.0]])
    x, y, w, h = NumpyMatOps.from_xyxy_to_xywh(boxes)
    xywh = np.stack((x, y, w, h), axis=1)
    x1, y1, x2, y2 = NumpyMatOps.from_xywh_to_xyxy(xywh)
    result = np.stack((x1, y1, x2, y2), axis=1)
    np.testing.assert_allclose(result, boxes)


def test_backend_detection():
    assert MatLib.which(np.zeros(1)) is MatLib.NUMPY
    assert MatLib.which(torch.zeros(1)) is MatLib.TORCH
    assert MatDevice.which(np.zeros(1)) is MatDevice.CPU
    assert MatDevice.which(torch.zeros(1)) is MatDevice.CPU


def test_dtype_detection():
    assert DataType.which(np.zeros(1, dtype=np.float32)) is DataType.FLOAT32
    assert DataType.which(torch.zeros(1, dtype=torch.int64)) is DataType.INT64
    assert DataType.from_dtype(torch.bfloat16) is DataType.BFLOAT16


def test_type_map_helpers():
    assert to_type_name(np.uint8) == "uint8"
    assert to_type_name(torch.float32) == "float32"
    assert to_type_name(ctypes.c_int32) == "int32"
    assert to_np_type("float32") is np.float32
    assert to_torch_type("float32") is torch.float32
    assert to_ctypes_type("float32") is ctypes.c_float
