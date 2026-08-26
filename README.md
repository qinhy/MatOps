# MatOps

A tiny backend-neutral operation layer for code that needs to work with NumPy,
PyTorch, or CuPy without duplicating every basic matrix/tensor operation.

`MatOps` defines a small common surface. `NumpyMatOps`, `TorchMatOps`, and
`CupyMatOps` implement it for their respective backends. The package also
includes helpers for backend/device detection, dtype conversion, and common
bounding-box coordinate conversions.

## Why this exists

Sometimes a result may be produced as a `numpy.ndarray`, `torch.Tensor`, or
`cupy.ndarray`, while downstream logic only needs a small set of common
operations. MatOps keeps that downstream code simple and explicit without
trying to replace NumPy, PyTorch, or CuPy.

## Installation

For development from a clone:

```bash
python -m pip install -e ".[dev]"
```

NumPy and PyTorch support are installed by default. CuPy is optional because
its wheel must match the CUDA major version on the machine.

For CUDA 12:

```bash
python -m pip install -e ".[dev,cupy-cuda12]"
```

For CUDA 13:

```bash
python -m pip install -e ".[dev,cupy-cuda13]"
```

Install only one CuPy package in an environment.

## Quick start

### NumPy

```python
import numpy as np

from matops import MatLib, NumpyMatOps

x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

ops = NumpyMatOps()
print(MatLib.which(x))       # MatLib.NUMPY
print(ops.mean(x, dim=0))    # [2. 3.]
print(ops.to_numpy(x))
```

### PyTorch

```python
import torch

from matops import TorchMatOps

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
ops = TorchMatOps()
print(ops.mean(x, dim=0))
```

### CuPy

```python
import cupy as cp

from matops import CupyMatOps, MatLib

ops = CupyMatOps()
x = ops.mat([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)

print(MatLib.which(x))       # MatLib.CUPY
print(ops.mean(x, dim=0))    # CuPy array on the GPU
print(ops.to_numpy(x))       # Explicit GPU -> CPU copy
```

Array creation can target a specific CUDA device:

```python
x = ops.zeros((1024, 1024), dtype=cp.float32, device="cuda:1")
```

If `device` is omitted, CuPy's current CUDA device is used.

## Supported operation surface

All three backends implement the same focused interface:

- array creation: `mat`, `eye`, `ones`, `zeros`
- linear algebra: `norm`, `dot`, `cross`, `matmul`
- reductions: `mean`, `median`, `std`, `max`, `min`
- shape/composition: `hstack`, `stack`, `cat`, `reshape`, `flatten`
- utility operations: `abs`, `clip`, `copy_mat`, logical operations, casts,
  `nonzero`, and `to_numpy`

The goal is useful parity, not complete NumPy/PyTorch/CuPy API emulation.

## Bounding-box conversions

The shared base class provides helpers for the common formats `xyxy`, `xywh`,
and `cxcywh`:

```python
import numpy as np

from matops import NumpyMatOps

boxes = np.array([[10, 20, 40, 60]])
x, y, w, h = NumpyMatOps.from_xyxy_to_xywh(boxes)
```

Because these helpers use basic array operations, they work with NumPy,
PyTorch, and CuPy arrays.

## Backend, device, and dtype helpers

```python
from matops import DataType, MatDevice, MatLib

MatLib.which(x)      # NUMPY, TORCH, or CUPY
MatDevice.which(x)   # CPU or CUDA
DataType.which(x)    # normalized DataType enum
```

`TypeMap` and the conversion helpers translate dtype names across supported
libraries:

```python
from matops import to_cupy_type, to_np_type, to_torch_type

to_np_type("float32")
to_torch_type("float32")
to_cupy_type("float32")
```

## Optional CuPy behavior

Importing `matops` does not require CuPy. `CupyMatOps` is always importable so
applications can expose optional GPU support cleanly. Calling a CuPy operation
without CuPy installed raises an `ImportError` with installation guidance.

## Development

Run the tests:

```bash
pytest
```

CuPy tests are automatically skipped when CuPy or a usable CUDA device is not
available. On a CUDA-equipped development machine, install the matching CuPy
extra and run the same `pytest` command to exercise backend parity.

Run linting:

```bash
ruff check .
```

## Project status

This repository intentionally stays small. The goal is a stable, predictable
adapter for a focused set of operations—not a general tensor framework.

MIT
