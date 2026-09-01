from __future__ import annotations

import ctypes
import enum
from contextlib import nullcontext
from typing import Any, ClassVar, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

try:
    import cupy as cp
except ImportError:  # CuPy is an optional, CUDA-specific dependency.
    cp = None

if cp is not None:
    ArrayLike = Union[np.ndarray, torch.Tensor, cp.ndarray]
else:
    ArrayLike = Union[np.ndarray, torch.Tensor]


def _require_cupy():
    if cp is None:
        raise ImportError(
            "CuPy support is optional. Install a CUDA-matched CuPy package, "
            "for example `pip install matops[cupy-cuda12]`."
        )
    return cp


def _cupy_device_context(device: Optional[Union[str, int, Any]]):
    cupy = _require_cupy()
    if device is None or device == "cuda":
        return nullcontext()
    if isinstance(device, str):
        if not device.startswith("cuda:"):
            raise ValueError(f"Unsupported CuPy device: {device!r}")
        device = int(device.split(":", 1)[1])
    return cupy.cuda.Device(device)


class MatOps(BaseModel):
    """Small backend-neutral matrix/tensor operation interface.

    Subclasses implement the same common operations for NumPy arrays, PyTorch
    tensors, and CuPy arrays. This is intentionally lightweight; it is useful when
    result payloads may be produced on either backend but downstream code wants
    a consistent operation surface.
    """

    int32: ClassVar[Any] = None
    uint8: ClassVar[Any] = None
    float32: ClassVar[Any] = None
    float16: ClassVar[Any] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def mat(self, data: Any, dtype: Any, device: Optional[Union[str, Any]] = None) -> ArrayLike:
        raise NotImplementedError

    def ndim(self, x: ArrayLike): return x.ndim

    def shape(self, x: ArrayLike): return x.shape

    def sin(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError

    def cos(self, x: ArrayLike) -> ArrayLike: raise NotImplementedError

    def eye(self, size: int, dtype: Any, device: Optional[Union[str, Any]] = None) -> ArrayLike:
        raise NotImplementedError

    def ones(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> ArrayLike:
        raise NotImplementedError

    def zeros(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> ArrayLike:
        raise NotImplementedError

    def hstack(self, arrays: Sequence[ArrayLike]) -> ArrayLike:
        raise NotImplementedError

    def norm(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def dot(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def cross(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def to_numpy(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def mean(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def median(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def std(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def max(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def min(self, x: ArrayLike, dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def abs(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def stack(self, xs: Sequence[ArrayLike], dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def cat(self, xs: Sequence[ArrayLike], dim: int = 0) -> ArrayLike:
        raise NotImplementedError

    def reshape(self, x: ArrayLike, shape: Sequence[int]) -> ArrayLike:
        raise NotImplementedError

    def copy_mat(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def logical_and(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def logical_or(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def clip(self, x: ArrayLike, min_val: Any, max_val: Any) -> ArrayLike:
        raise NotImplementedError

    def astype_int32(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def astype_uint8(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def astype_float32(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def astype_float16(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def nonzero(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError
    
    def flatten(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @staticmethod
    def from_xyxy_to_xywh(data):
        x1,y1,x2,y2 = data.T
        w = x2 - x1
        h = y2 - y1
        return x1,y1,w,h
    
    @staticmethod
    def from_xywh_to_xyxy(data):
        x1,y1,w,h = data.T
        x2 = x1 + w
        y2 = y1 + h
        return x1,y1,x2,y2
    
    @staticmethod
    def from_cxcywh_to_xyxy(data):
        cx,cy,w,h = data.T
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return x1,y1,x2,y2
    
    @staticmethod
    def from_xyxy_to_cxcywh(data):
        x1,y1,x2,y2 = data.T
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        return cx,cy,w,h
    
    @staticmethod
    def from_xywh_to_cxcywh(data):
        x1,y1,w,h = data.T
        cx = x1 + w/2
        cy = y1 + h/2
        return cx,cy,w,h
    
    @staticmethod
    def from_cxcywh_to_xywh(data):
        cx,cy,w,h = data.T
        x1 = cx - w/2
        y1 = cy - h/2
        return x1,y1,w,h

class NumpyMatOps(MatOps):
    """NumPy implementation of :class:`MatOps`."""

    int32: ClassVar[Any] = np.int32
    uint8: ClassVar[Any] = np.uint8
    float32: ClassVar[Any] = np.float32
    float16: ClassVar[Any] = np.float16

    def mat(self, data: Any, dtype: Any, device: Optional[Union[str, Any]] = None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def sin(self, x): return np.sin(x)

    def cos(self, x): return np.cos(x)
    
    def eye(self, size: int, dtype: Any, device: Optional[Union[str, Any]] = None) -> np.ndarray:
        return np.eye(size, dtype=dtype)

    def ones(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def hstack(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        return np.hstack(arrays)

    def norm(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x)

    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)

    def cross(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.cross(a, b)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)

    def mean(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.mean(x, axis=dim)

    def median(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.median(x, axis=dim)

    def std(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.std(x, axis=dim)

    def max(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.max(x, axis=dim)

    def min(self, x: np.ndarray, dim: int = 0) -> np.ndarray:
        return np.min(x, axis=dim)

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def stack(self, xs: Sequence[np.ndarray], dim: int = 0) -> np.ndarray:
        return np.stack(xs, axis=dim)

    def cat(self, xs: Sequence[np.ndarray], dim: int = 0) -> np.ndarray:
        return np.concatenate(xs, axis=dim)

    def reshape(self, x: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        return np.reshape(x, shape)

    def copy_mat(self, x: np.ndarray) -> np.ndarray:
        return x.copy()

    def logical_and(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.logical_and(a, b)

    def logical_or(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.logical_or(a, b)

    def clip(self, x: np.ndarray, min_val: Any, max_val: Any) -> np.ndarray:
        return np.clip(x, min_val, max_val)

    def astype_int32(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.int32)

    def astype_uint8(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.uint8)

    def astype_float32(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)

    def astype_float16(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float16)

    def nonzero(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        return np.nonzero(x)
    
    def flatten(self, x: np.ndarray) -> np.ndarray:
        return x.flatten()

class TorchMatOps(MatOps):
    """PyTorch implementation of :class:`MatOps`."""

    int32: ClassVar[Any] = None if torch is None else torch.int32
    uint8: ClassVar[Any] = None if torch is None else torch.uint8
    float32: ClassVar[Any] = None if torch is None else torch.float32
    float16: ClassVar[Any] = None if torch is None else torch.float16

    def mat(self, data: torch.Tensor, dtype: Any, device: Optional[Union[str, Any]] = None) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, device=device)

    def sin(self, x): return torch.sin(x)

    def cos(self, x): return torch.cos(x)

    def eye(self, size: int, dtype: Any, device: Optional[Union[str, Any]] = None) -> torch.Tensor:
        return torch.eye(size, dtype=dtype, device=device)

    def ones(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> torch.Tensor:
        return torch.ones(tuple(shape), dtype=dtype, device=device)

    def zeros(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, Any]] = None) -> torch.Tensor:
        return torch.zeros(tuple(shape), dtype=dtype, device=device)

    def hstack(self, arrays: Sequence[Any]) -> torch.Tensor:
        return torch.cat(tuple(arrays), dim=1)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x)

    def dot(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.dot(a, b)

    def cross(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.cross(a, b)

    def matmul(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.matmul(a, b)

    def to_numpy(self, x: torch.Tensor) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    def mean(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.mean(x, dim=dim)

    def median(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.median(x, dim=dim).values

    def std(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.std(x, dim=dim, unbiased=False)

    def max(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.max(x, dim=dim).values

    def min(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.min(x, dim=dim).values

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)

    def stack(self, xs: Sequence[Any], dim: int = 0) -> torch.Tensor:
        return torch.stack(tuple(xs), dim=dim)

    def cat(self, xs: Sequence[Any], dim: int = 0) -> torch.Tensor:
        return torch.cat(tuple(xs), dim=dim)

    def reshape(self, x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        return x.reshape(tuple(shape))

    def copy_mat(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    def logical_and(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.logical_and(a, b)

    def logical_or(self, a: torch.Tensor, b: Any) -> torch.Tensor:
        return torch.logical_or(a, b)

    def clip(self, x: torch.Tensor, min_val: Any, max_val: Any) -> torch.Tensor:
        return torch.clamp(x, min=min_val, max=max_val)

    def astype_int32(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.int32)

    def astype_uint8(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.uint8)

    def astype_float32(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float32)

    def astype_float16(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float16)

    def nonzero(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(x)

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten()


class CupyMatOps(MatOps):
    """CuPy implementation of :class:`MatOps`.

    CuPy is optional. Instantiating this class is always safe, but calling an
    operation requires a CuPy installation compatible with the local CUDA
    runtime.
    """

    int32: ClassVar[Any] = None if cp is None else cp.int32
    uint8: ClassVar[Any] = None if cp is None else cp.uint8
    float32: ClassVar[Any] = None if cp is None else cp.float32
    float16: ClassVar[Any] = None if cp is None else cp.float16

    def mat(self, data: Any, dtype: Any, device: Optional[Union[str, int, Any]] = None) -> Any:
        cupy = _require_cupy()
        with _cupy_device_context(device):
            return cupy.array(data, dtype=dtype)

    def sin(self, x): return _require_cupy().sin(x)

    def cos(self, x): return _require_cupy().cos(x)
    
    def eye(self, size: int, dtype: Any, device: Optional[Union[str, int, Any]] = None) -> Any:
        cupy = _require_cupy()
        with _cupy_device_context(device):
            return cupy.eye(size, dtype=dtype)

    def ones(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, int, Any]] = None) -> Any:
        cupy = _require_cupy()
        with _cupy_device_context(device):
            return cupy.ones(tuple(shape), dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype: Any, device: Optional[Union[str, int, Any]] = None) -> Any:
        cupy = _require_cupy()
        with _cupy_device_context(device):
            return cupy.zeros(tuple(shape), dtype=dtype)

    def hstack(self, arrays: Sequence[Any]) -> Any:
        return _require_cupy().hstack(tuple(arrays))

    def norm(self, x: Any) -> Any:
        return _require_cupy().linalg.norm(x)

    def dot(self, a: Any, b: Any) -> Any:
        return _require_cupy().dot(a, b)

    def cross(self, a: Any, b: Any) -> Any:
        return _require_cupy().cross(a, b)

    def matmul(self, a: Any, b: Any) -> Any:
        return _require_cupy().matmul(a, b)

    def to_numpy(self, x: Any) -> np.ndarray:
        return _require_cupy().asnumpy(x)

    def mean(self, x: Any, dim: int = 0) -> Any:
        return _require_cupy().mean(x, axis=dim)

    def median(self, x: Any, dim: int = 0) -> Any:
        return _require_cupy().median(x, axis=dim)

    def std(self, x: Any, dim: int = 0) -> Any:
        return _require_cupy().std(x, axis=dim, ddof=0)

    def max(self, x: Any, dim: int = 0) -> Any:
        return _require_cupy().max(x, axis=dim)

    def min(self, x: Any, dim: int = 0) -> Any:
        return _require_cupy().min(x, axis=dim)

    def abs(self, x: Any) -> Any:
        return _require_cupy().abs(x)

    def stack(self, xs: Sequence[Any], dim: int = 0) -> Any:
        return _require_cupy().stack(tuple(xs), axis=dim)

    def cat(self, xs: Sequence[Any], dim: int = 0) -> Any:
        return _require_cupy().concatenate(tuple(xs), axis=dim)

    def reshape(self, x: Any, shape: Sequence[int]) -> Any:
        return _require_cupy().reshape(x, tuple(shape))

    def copy_mat(self, x: Any) -> Any:
        _require_cupy()
        return x.copy()

    def logical_and(self, a: Any, b: Any) -> Any:
        return _require_cupy().logical_and(a, b)

    def logical_or(self, a: Any, b: Any) -> Any:
        return _require_cupy().logical_or(a, b)

    def clip(self, x: Any, min_val: Any, max_val: Any) -> Any:
        return _require_cupy().clip(x, min_val, max_val)

    def astype_int32(self, x: Any) -> Any:
        return x.astype(_require_cupy().int32)

    def astype_uint8(self, x: Any) -> Any:
        return x.astype(_require_cupy().uint8)

    def astype_float32(self, x: Any) -> Any:
        return x.astype(_require_cupy().float32)

    def astype_float16(self, x: Any) -> Any:
        return x.astype(_require_cupy().float16)

    def nonzero(self, x: Any) -> Any:
        return _require_cupy().nonzero(x)

    def flatten(self, x: Any) -> Any:
        _require_cupy()
        return x.flatten()


class MatLib(str, enum.Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    CUPY = "cupy"

    @staticmethod
    def which(data):
        if isinstance(data, np.ndarray):
            return MatLib.NUMPY
        if isinstance(data, torch.Tensor):
            return MatLib.TORCH
        if cp is not None and isinstance(data, cp.ndarray):
            return MatLib.CUPY
        raise ValueError(f"Unsupported data type: {type(data)}")

class MatDevice(str, enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"
    CUDA0 = "cuda:0"
    CUDA1 = "cuda:1"
    # MPS = "mps"
    
    @staticmethod
    def which(data):
        if isinstance(data, np.ndarray):
            return MatDevice.CPU
        if isinstance(data, torch.Tensor):
            return MatDevice.CPU if data.device.type == "cpu" else MatDevice.CUDA
        if cp is not None and isinstance(data, cp.ndarray):
            return MatDevice.CUDA
        raise ValueError(f"Unsupported data type: {type(data)}")

class DataType(str, enum.Enum):
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    UINT8 = "uint8"
    INT32 = "int32"
    INT64 = "int64"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_dtype(cls, dtype: Any, *, strict: bool = False) -> "DataType | None":
        """Return the matching DataType for a NumPy/Torch/CuPy/Python dtype.

        Returns None for unsupported dtypes unless strict=True.
        """
        if isinstance(dtype, cls):
            return dtype

        # NumPy dtype, NumPy scalar type, or dtype string such as "uint8".
        try:
            np_dtype = np.dtype(dtype)
        except TypeError:
            np_dtype = None

        if np_dtype is not None:
            result = {
                np.dtype(np.float64): cls.FLOAT64,
                np.dtype(np.float32): cls.FLOAT32,
                np.dtype(np.float16): cls.FLOAT16,
                np.dtype(np.uint8): cls.UINT8,
                np.dtype(np.int32): cls.INT32,
                np.dtype(np.int64): cls.INT64,
            }.get(np_dtype)
            if result is not None:
                return result

        # Torch dtype objects, for example torch.float32.
        result = {
            torch.float64: cls.FLOAT64,
            torch.float32: cls.FLOAT32,
            torch.float16: cls.FLOAT16,
            torch.bfloat16: cls.BFLOAT16,
            torch.uint8: cls.UINT8,
            torch.int32: cls.INT32,
            torch.int64: cls.INT64,
        }.get(dtype)
        if result is not None:
            return result

        if strict:
            raise TypeError(f"Unsupported dtype: {dtype!r}")
        return None

    @classmethod
    def which(cls, data: Any, *, strict: bool = True) -> "DataType | None":
        """Infer DataType from an array, tensor, dtype object, enum, or dtype string.

        Existing calls like DataType.which(data) still work. Unsupported inputs return
        None by default; pass strict=True to raise a TypeError instead.
        """
        if isinstance(data, np.ndarray):
            return cls.from_dtype(data.dtype, strict=strict)

        if isinstance(data, torch.Tensor):
            return cls.from_dtype(data.dtype, strict=strict)

        if cp is not None and isinstance(data, cp.ndarray):
            return cls.from_dtype(data.dtype, strict=strict)

        return cls.from_dtype(data, strict=strict)

TypeMap = {
    # Tuple order preserves the original public layout: NumPy, Torch, ctypes, CuPy.
    "uint8": (np.uint8, torch.uint8, ctypes.c_uint8, None if cp is None else cp.uint8),
    "int8": (np.int8, torch.int8, ctypes.c_int8, None if cp is None else cp.int8),
    "uint16": (
        np.uint16,
        getattr(torch, "uint16", None),
        ctypes.c_uint16,
        None if cp is None else cp.uint16,
    ),
    "int16": (np.int16, torch.int16, ctypes.c_int16, None if cp is None else cp.int16),
    "uint32": (
        np.uint32,
        getattr(torch, "uint32", None),
        ctypes.c_uint32,
        None if cp is None else cp.uint32,
    ),
    "int32": (np.int32, torch.int32, ctypes.c_int32, None if cp is None else cp.int32),
    "uint64": (
        np.uint64,
        getattr(torch, "uint64", None),
        ctypes.c_uint64,
        None if cp is None else cp.uint64,
    ),
    "int64": (np.int64, torch.int64, ctypes.c_int64, None if cp is None else cp.int64),
    "float16": (np.float16, torch.float16, None, None if cp is None else cp.float16),
    "float32": (np.float32, torch.float32, ctypes.c_float, None if cp is None else cp.float32),
    "float64": (np.float64, torch.float64, ctypes.c_double, None if cp is None else cp.float64),
    "bool": (np.bool_, torch.bool, ctypes.c_bool, None if cp is None else cp.bool_),
}

def to_type_name(dtype: Any) -> str:
    """
    Convert a NumPy / PyTorch / CuPy / ctypes dtype into the canonical string key
    used by TypeMap.

    Examples:
        np.uint8          -> "uint8"
        np.dtype("uint8") -> "uint8"
        torch.uint8       -> "uint8"
        cupy.uint8        -> "uint8"
        ctypes.c_uint8    -> "uint8"
        "uint8"           -> "uint8"
    """
    if dtype is None:
        raise TypeError("dtype cannot be None")

    if isinstance(dtype, str):
        if dtype in TypeMap:
            return dtype
        raise KeyError(f"Unknown dtype name: {dtype!r}")

    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        np_dtype = None

    for name, info in TypeMap.items():
        if np_dtype is not None and np_dtype == info[0]:
            return name

        if info[1] is not None and dtype is info[1]:
            return name

        if info[2] is not None and dtype is info[2]:
            return name

        if info[3] is not None and dtype is info[3]:
            return name

    raise TypeError(f"Unsupported dtype: {dtype!r}")


def to_np_type(dtype: Any) -> np.dtype:
    return TypeMap[to_type_name(dtype)][0]


def to_torch_type(dtype: Any) -> torch.dtype:
    result = TypeMap[to_type_name(dtype)][1]
    if result is None:
        raise TypeError(f"No PyTorch dtype mapping for {dtype!r}")
    return result


def to_cupy_type(dtype: Any) -> Any:
    _require_cupy()
    result = TypeMap[to_type_name(dtype)][3]
    if result is None:
        raise TypeError(f"No CuPy dtype mapping for {dtype!r}")
    return result


def to_ctypes_type(dtype: Any):
    result = TypeMap[to_type_name(dtype)][2]
    if result is None:
        raise TypeError(f"No ctypes dtype mapping for {dtype!r}")
    return result
