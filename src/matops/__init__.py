"""Backend-neutral matrix/tensor operations for NumPy, PyTorch, and CuPy."""

from .mat import (
    ArrayLike,
    CupyMatOps,
    DataType,
    MatDevice,
    MatLib,
    MatOps,
    NumpyMatOps,
    TorchMatOps,
    TypeMap,
    to_ctypes_type,
    to_cupy_type,
    to_np_type,
    to_torch_type,
    to_type_name,
)

__all__ = [
    "ArrayLike",
    "CupyMatOps",
    "DataType",
    "MatDevice",
    "MatLib",
    "MatOps",
    "NumpyMatOps",
    "TorchMatOps",
    "TypeMap",
    "to_ctypes_type",
    "to_cupy_type",
    "to_np_type",
    "to_torch_type",
    "to_type_name",
]

__version__ = "0.2.0"
