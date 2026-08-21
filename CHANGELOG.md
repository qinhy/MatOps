# Changelog

## 0.2.0

- Add optional CuPy backend through `CupyMatOps`.
- Add `MatLib.CUPY` and CuPy-aware device/dtype detection.
- Add `to_cupy_type` and CuPy entries to `TypeMap` while preserving the
  original NumPy/Torch/ctypes tuple positions.
- Add optional CUDA 12 and CUDA 13 installation extras.
- Add GPU-only CuPy parity tests that skip cleanly without a CUDA device.

## 0.1.0

- Initial NumPy and PyTorch backend release.
