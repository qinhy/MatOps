from pathlib import Path
from typing import Literal

import cv2

from matops.mat import ArrayLike, DataType, MatOps, NumpyMatOps

ColorOrder = Literal["RGB", "BGR"]


def _scalar(x, ops: MatOps):
    """Convert a backend scalar to a Python scalar."""
    return ops.to_numpy(x).item()


def _tilt_matrix(tau_x, tau_y, *, ops: MatOps, dtype):
    """OpenCV tilted-sensor projection matrix."""
    cx, sx = ops.cos(tau_x), ops.sin(tau_x)
    cy, sy = ops.cos(tau_y), ops.sin(tau_y)

    rx = ops.eye(3, dtype=dtype)
    rx[1, 1], rx[1, 2], rx[2, 1], rx[2, 2] = cx, sx, -sx, cx

    ry = ops.eye(3, dtype=dtype)
    ry[0, 0], ry[0, 2], ry[2, 0], ry[2, 2] = cy, -sy, sy, cy

    r = ops.matmul(ry, rx)

    pz = ops.eye(3, dtype=dtype)
    pz[0, 0] = pz[1, 1] = r[2, 2]
    pz[0, 2], pz[1, 2] = -r[0, 2], -r[1, 2]

    return ops.matmul(pz, r)


def _undistort_normalized(
    xd,
    yd,
    distortion,
    *,
    ops: MatOps,
    iterations: int = 5,
):
    src = ops.flatten(distortion)
    n = ops.numel(src)

    if not n:
        return xd, yd

    if n not in (4, 5, 8, 12, 14):
        raise ValueError(
            "distortion must contain 4, 5, 8, 12, or 14 coefficients"
        )

    d = ops.zeros(
        (14,),
        dtype=ops.dtype(xd),
    )
    d[:n] = src

    (
        k1, k2,
        p1, p2,
        k3, k4, k5, k6,
        s1, s2, s3, s4,
        tx, ty,
    ) = d

    if n == 14:
        tilt = _tilt_matrix(
            tx,
            ty,
            ops=ops,
            dtype=ops.dtype(xd),
        )

        inv_tilt = ops.inv(tilt)

        # Equivalent to torch.ones_like(xd),
        # without requiring another MatOps primitive.
        ones = xd * 0 + 1

        h = ops.stack(
            (xd, yd, ones),
            dim=1,
        )

        h = ops.matmul(
            h,
            inv_tilt.T,
        )

        x0 = h[:, 0] / h[:, 2]
        y0 = h[:, 1] / h[:, 2]

    else:
        x0 = xd
        y0 = yd

    x = ops.copy_mat(x0)
    y = ops.copy_mat(y0)

    for _ in range(iterations):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        icdist = (
            1
            + k4 * r2
            + k5 * r4
            + k6 * r6
        ) / (
            1
            + k1 * r2
            + k2 * r4
            + k3 * r6
        )

        xy = x * y

        dx = (
            2 * p1 * xy
            + p2 * (r2 + 2 * x * x)
            + s1 * r2
            + s2 * r4
        )

        dy = (
            p1 * (r2 + 2 * y * y)
            + 2 * p2 * xy
            + s3 * r2
            + s4 * r4
        )

        x = (x0 - dx) * icdist
        y = (y0 - dy) * icdist

    return x, y


def gray8(image: ArrayLike, *, ops: MatOps) -> ArrayLike:
    """
    Convert an image to 8-bit grayscale.

    Input color order is assumed to be BGR, matching the original
    cv2.COLOR_BGR2GRAY behavior.
    """
    a = image
    src_dtype = DataType.which(ops.dtype(a), strict=False)
    converted_color = False

    # BGR / BGRA / >=3 channels -> grayscale
    if ops.ndim(a) == 3 and ops.shape(a)[2] >= 3:
        b = a[:, :, 0]
        g = a[:, :, 1]
        r = a[:, :, 2]

        a = 0.114 * b + 0.587 * g + 0.299 * r
        converted_color = True

    # HxWx1 -> HxW
    elif ops.ndim(a) == 3 and ops.shape(a)[2] == 1:
        a = a[:, :, 0]

    elif ops.ndim(a) != 2:
        raise ValueError(
            f"Unsupported image shape: {ops.shape(a)}"
        )

    # uint8 is already in the desired value range.
    #
    # For color input, grayscale arithmetic promoted the values to
    # floating point, so round back to uint8.
    if src_dtype == DataType.UINT8:
        if converted_color:
            a = ops.round(a)
            a = ops.clip(a, 0, 255)
            return ops.astype_uint8(a)

        return a

    # OpenCV's uint16 grayscale conversion produces integer grayscale
    # values before the later normalization, so round here as well.
    if src_dtype == DataType.UINT16 and converted_color:
        a = ops.round(a)

    # Float image in [0, 1] -> [0, 255].
    if (
        ops.is_floating_point(a)
        and ops.numel(a)
        and _scalar(ops.nanmax(a), ops) <= 1
    ):
        a = a * 255

    # Preserve your original uint16 behavior:
    # scale the image's actual maximum to 255.
    if src_dtype == DataType.UINT16 and ops.numel(a):
        max_value = _scalar(
            ops.max(a, dim=None),
            ops,
        )

        if max_value > 0:
            a = ops.astype_float32(a)
            a = a * (255 / max_value)

    return ops.astype_uint8(
        ops.clip(a, 0, 255)
    )


def rgb8(
    colors: ArrayLike,
    order: ColorOrder = "RGB",
    *,
    ops: MatOps,
) -> ArrayLike:
    """
    Convert Nx3-like colors to uint8 RGB/BGR.
    """
    a = colors

    if ops.ndim(a) != 2 or ops.shape(a)[1] < 3:
        raise ValueError(
            f"colors must be Nx3, got {ops.shape(a)}"
        )

    a = a[:, :3]

    # Float colors in [0, 1] -> [0, 255].
    if (
        ops.is_floating_point(a)
        and ops.numel(a)
        and _scalar(ops.nanmax(a), ops) <= 1
    ):
        a = a * 255

    a = ops.round(a)
    a = ops.clip(a, 0, 255)
    a = ops.astype_uint8(a)

    if order == "BGR":
        a = ops.flip(a, dim=1)

    return a

    

def read_image(
    path: str | Path,
    *,
    color: bool = True,
    ops: MatOps = NumpyMatOps(),
):
    """
    Read an image from disk.

    color=True:
        Returns BGR uint8 image, matching cv2.imread(..., IMREAD_COLOR).

    color=False:
        Returns a 2D grayscale uint8 image.

    If `ops` is supplied, the resulting NumPy image is transferred to
    that backend.
    """
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}"
        )
    
    if not color:
        image = gray8(
            ops.from_numpy(image),
            ops=ops,
        )

    return image