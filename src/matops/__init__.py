"""Backend-neutral matrix/tensor operations for NumPy and PyTorch."""

from .mat import (
    ArrayLike,
    DataType,
    MatDevice,
    MatLib,
    MatOps,
    NumpyMatOps,
    TorchMatOps,
    TypeMap,
    to_ctypes_type,
    to_np_type,
    to_torch_type,
    to_type_name,
)

__all__ = [
    "ArrayLike",
    "DataType",
    "MatDevice",
    "MatLib",
    "MatOps",
    "NumpyMatOps",
    "TorchMatOps",
    "TypeMap",
    "to_ctypes_type",
    "to_np_type",
    "to_torch_type",
    "to_type_name",
]

__version__ = "0.1.0"
