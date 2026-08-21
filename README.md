# MatOps

A tiny backend-neutral operation layer for code that needs to work with either
NumPy arrays or PyTorch tensors without duplicating every basic matrix/tensor
operation.

`MatOps` defines a small common surface. `NumpyMatOps` and `TorchMatOps`
implement it for their respective backends. The package also includes helpers
for backend/device detection, dtype conversion, and common bounding-box
coordinate conversions.

## Why this exists

Sometimes a result may be produced as either a `numpy.ndarray` or a
`torch.Tensor`, while the downstream logic only needs a small set of common
operations. MatOps keeps that downstream code simple and explicit without
trying to replace NumPy or PyTorch.

## Installation

For development from a clone:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np

from matops import MatLib, NumpyMatOps

x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

ops = NumpyMatOps()
print(MatLib.which(x))       # MatLib.NUMPY
print(ops.mean(x, dim=0))    # [2. 3.]
print(ops.to_numpy(x))
```

The PyTorch backend exposes the same operation names:

```python
import torch

from matops import TorchMatOps

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
ops = TorchMatOps()
print(ops.mean(x, dim=0))
```

## Bounding-box conversions

The shared base class provides helpers for the common formats `xyxy`, `xywh`,
and `cxcywh`:

```python
import numpy as np

from matops import NumpyMatOps

boxes = np.array([[10, 20, 40, 60]])
x, y, w, h = NumpyMatOps.from_xyxy_to_xywh(boxes)
```

## Public API

- `MatOps` — common interface
- `NumpyMatOps` — NumPy implementation
- `TorchMatOps` — PyTorch implementation
- `MatLib` — NumPy/PyTorch backend detection
- `MatDevice` — CPU/CUDA detection
- `DataType` — normalized dtype enum
- `TypeMap` and `to_*_type` helpers — NumPy/PyTorch/ctypes dtype conversion

## Development

Run the tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

## Project status

This repository intentionally stays small. The goal is a stable, predictable
adapter for a focused set of operations—not a general tensor framework.

Before publishing publicly, choose and add the license you want for the project.
