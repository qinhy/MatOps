from __future__ import annotations

import numpy as np
import torch

from src.matops.mat import MatOps, MatLib, NumpyMatOps, TorchMatOps, CupyMatOps

try:
    import cupy as cp
except ImportError:
    cp = None


OPS:dict[str,MatOps] = {
    MatLib.NUMPY: NumpyMatOps(),
    MatLib.TORCH: TorchMatOps(),
    MatLib.CUPY: CupyMatOps(),
}

VALID_DISTORTION_SIZES = (4, 5, 8, 12, 14)


def _tilt_matrix(tau_x, tau_y, *, ops: MatOps, dtype, device=None):
    """OpenCV tilted-sensor projection matrix."""
    cx, sx = ops.cos(tau_x), ops.sin(tau_x)
    cy, sy = ops.cos(tau_y), ops.sin(tau_y)

    rx = ops.eye(3, dtype=dtype, device=device)
    rx[1, 1], rx[1, 2], rx[2, 1], rx[2, 2] = cx, sx, -sx, cx

    ry = ops.eye(3, dtype=dtype, device=device)
    ry[0, 0], ry[0, 2], ry[2, 0], ry[2, 2] = cy, -sy, sy, cy

    r = ops.matmul(ry, rx)

    pz = ops.eye(3, dtype=dtype, device=device)
    pz[0, 0] = pz[1, 1] = r[2, 2]
    pz[0, 2], pz[1, 2] = -r[0, 2], -r[1, 2]

    return ops.matmul(pz, r)


def project_camera_points_opencv_model(
    points_camera,
    K,
    distortion=None,
    *,
    dtype,
    device=None,
):
    """Project Nx3 camera-frame points using OpenCV's distortion model."""
    p = points_camera
    ops = OPS[MatLib.which(p)]

    if ops.ndim(p) != 2 or ops.shape(p)[1] != 3:
        raise ValueError(f"points_camera must be Nx3, got {ops.shape(p)}")
    if ops.shape(K) != (3, 3):
        raise ValueError(f"K must be 3x3, got {ops.shape(K)}")

    x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]

    if distortion is not None:
        src = ops.reshape(distortion, (-1,))
        n = len(src)

        if n not in VALID_DISTORTION_SIZES:
            raise ValueError(
                f"distortion must contain one of {VALID_DISTORTION_SIZES}"
            )

        # OpenCV layout:
        # k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tau_x,tau_y
        d = ops.zeros((14,), dtype=dtype, device=device)
        d[:n] = src
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty = d

        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2

        radial = (
            1 + k1 * r2 + k2 * r4 + k3 * r6
        ) / (
            1 + k4 * r2 + k5 * r4 + k6 * r6
        )

        xy = x * y
        xd = x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * xy + s3 * r2 + s4 * r4

        tilt = _tilt_matrix(tx, ty, ops=ops, dtype=dtype, device=device)

        zt = tilt[2, 0] * xd + tilt[2, 1] * yd + tilt[2, 2]
        x = (tilt[0, 0] * xd + tilt[0, 1] * yd + tilt[0, 2]) / zt
        y = (tilt[1, 0] * xd + tilt[1, 1] * yd + tilt[1, 2]) / zt

    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 0] * x + K[1, 1] * y + K[1, 2]

    return ops.stack((u, v), dim=1)


# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------

POINTS = [
    [-0.5, -0.2, 2.0],
    [0.0, 0.0, 1.0],
    [0.4, 0.3, 1.5],
    [1.0, -0.4, 3.0],
]

CAMERA_MATRIX = [
    [800.0, 0.0, 640.0],
    [0.0, 810.0, 360.0],
    [0.0, 0.0, 1.0],
]

DISTORTION = [
    -0.12, 0.03, 0.001, -0.0005, 0.002,
    0.0, 0.0, 0.0,
    0.0001, -0.00005, 0.00008, -0.00003,
    0.01, -0.015,
]


def numpy_example():
    dtype = np.float64
    points = np.asarray(POINTS, dtype=dtype)
    K = np.asarray(CAMERA_MATRIX, dtype=dtype)
    distortion = np.asarray(DISTORTION, dtype=dtype)

    pixels = project_camera_points_opencv_model(
        points, K, distortion, dtype=dtype
    )
    print("\n=== NumPy ===\n", pixels)

    try:
        import cv2

        expected, _ = cv2.projectPoints(
            points,
            np.zeros(3, dtype=dtype),
            np.zeros(3, dtype=dtype),
            K,
            distortion,
        )
        expected = expected.reshape(-1, 2)

        print("max abs error:", np.abs(pixels - expected).max())
        print("allclose:", np.allclose(pixels, expected, rtol=1e-10, atol=1e-10))
    except ImportError:
        print("OpenCV not installed; verification skipped.")

    return pixels


def torch_example(device="cpu"):
    device = torch.device(device)
    dtype = torch.float64

    make = lambda x: torch.tensor(x, dtype=dtype, device=device)

    pixels = project_camera_points_opencv_model(
        make(POINTS),
        make(CAMERA_MATRIX),
        make(DISTORTION),
        dtype=dtype,
        device=device,
    )

    print(f"\n=== PyTorch ({device}) ===\n", pixels)
    return pixels


def cupy_example():
    if cp is None:
        print("\n=== CuPy ===\nCuPy not installed; skipping.")
        return None

    dtype = cp.float64
    pixels = project_camera_points_opencv_model(
        cp.asarray(POINTS, dtype=dtype),
        cp.asarray(CAMERA_MATRIX, dtype=dtype),
        cp.asarray(DISTORTION, dtype=dtype),
        dtype=dtype,
        device="cuda",
    )

    print("\n=== CuPy ===\n", pixels)
    return pixels


if __name__ == "__main__":
    numpy_example()
    torch_example()

    if torch.cuda.is_available():
        torch_example("cuda")

    cupy_example()