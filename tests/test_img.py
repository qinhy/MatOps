from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from src.matops.mat import (
    ArrayLike, CupyMatOps, DataType, MatDevice, MatLib, MatOps,
    NumpyMatOps, TorchMatOps,
)

ColorOrder = Literal["RGB", "BGR"]
TranslationUnit = Literal["m", "cm", "mm"]


def _scalar(x, ops: MatOps):
    """Convert a backend scalar to a Python scalar."""
    return ops.to_numpy(x).item()


def _to_ops(value: ArrayLike, src: MatOps, dst: MatOps) -> ArrayLike:
    """Move an array between MatOps backends through NumPy."""
    return dst.from_numpy(src.to_numpy(value))


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
    xd, yd, distortion, *, ops: MatOps, iterations: int = 5
):
    src = ops.flatten(distortion)
    n = ops.numel(src)
    if not n:
        return xd, yd
    if n not in (4, 5, 8, 12, 14):
        raise ValueError("distortion must contain 4, 5, 8, 12, or 14 coefficients")

    d = ops.zeros((14,), dtype=ops.dtype(xd))
    d[:n] = src
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty = d

    if n == 14:
        tilt = _tilt_matrix(tx, ty, ops=ops, dtype=ops.dtype(xd))
        h = ops.stack((xd, yd, xd * 0 + 1), dim=1)
        h = ops.matmul(h, ops.inv(tilt).T)
        x0, y0 = h[:, 0] / h[:, 2], h[:, 1] / h[:, 2]
    else:
        x0, y0 = xd, yd

    x, y = ops.copy_mat(x0), ops.copy_mat(y0)
    for _ in range(iterations):
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        icdist = (1 + k4 * r2 + k5 * r4 + k6 * r6) / (
            1 + k1 * r2 + k2 * r4 + k3 * r6
        )
        xy = x * y
        dx = 2 * p1 * xy + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * xy + s3 * r2 + s4 * r4
        x, y = (x0 - dx) * icdist, (y0 - dy) * icdist
    return x, y


def gray8(image: ArrayLike, *, ops: MatOps) -> ArrayLike:
    """Convert a BGR/BGRA image to 8-bit grayscale."""
    a = image
    src_dtype = DataType.which(ops.dtype(a), strict=False)
    converted_color = False

    if ops.ndim(a) == 3 and ops.shape(a)[2] >= 3:
        b, g, r = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        a = 0.114 * b + 0.587 * g + 0.299 * r
        converted_color = True
    elif ops.ndim(a) == 3 and ops.shape(a)[2] == 1:
        a = a[:, :, 0]
    elif ops.ndim(a) != 2:
        raise ValueError(f"Unsupported image shape: {ops.shape(a)}")

    if src_dtype == DataType.UINT8:
        if converted_color:
            return ops.astype_uint8(ops.clip(ops.round(a), 0, 255))
        return a

    if src_dtype == DataType.UINT16 and converted_color:
        a = ops.round(a)
    if ops.is_floating_point(a) and ops.numel(a) and _scalar(ops.nanmax(a), ops) <= 1:
        a = a * 255

    if src_dtype == DataType.UINT16 and ops.numel(a):
        max_value = _scalar(ops.max(a, dim=None), ops)
        if max_value > 0:
            a = ops.astype_float32(a) * (255 / max_value)

    return ops.astype_uint8(ops.clip(a, 0, 255))


def rgb8(
    colors: ArrayLike, order: ColorOrder = "RGB", *, ops: MatOps
) -> ArrayLike:
    """Convert Nx3-like colors to uint8 RGB/BGR."""
    a = colors
    if ops.ndim(a) != 2 or ops.shape(a)[1] < 3:
        raise ValueError(f"colors must be Nx3, got {ops.shape(a)}")

    a = a[:, :3]
    if ops.is_floating_point(a) and ops.numel(a) and _scalar(ops.nanmax(a), ops) <= 1:
        a = a * 255

    a = ops.astype_uint8(ops.clip(ops.round(a), 0, 255))
    return ops.flip(a, dim=1) if order == "BGR" else a


def read_image(
    path: str | Path, *, color: bool = True, ops: MatOps = NumpyMatOps()
):
    """Read an image from disk; grayscale output uses the selected MatOps backend."""
    image = cv2.imread(
        str(path), cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    )
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image if color else gray8(ops.from_numpy(image), ops=ops)


DEFAULT_CALIBRATION: dict[str, Any] = {'rgb_resolution': [4056, 3040],
 'left_resolution': [1280, 800],
 'right_resolution': [1280, 800],
 'stereo_translation_units_hint': 'Calibration extrinsics units from device; Luxonis stereo baseline override API uses '
                                  'centimeters.',
 'rgb_intrinsics': [[2430.31884765625, 0.0, 2063.196044921875], [0.0, 2429.41748046875, 1490.1956787109375],
                    [0.0, 0.0, 1.0]],
 'left_intrinsics': [[570.8507690429688, 0.0, 653.754150390625], [0.0, 570.580810546875, 390.99169921875],
                     [0.0, 0.0, 1.0]],
 'right_intrinsics': [[567.8758544921875, 0.0, 655.560546875], [0.0, 567.7424926757812, 393.97039794921875],
                      [0.0, 0.0, 1.0]],
 'left_to_right_extrinsics': [[0.9998766183853149, 0.0023975009098649025, -0.015519456937909126, -7.537897109985352],
                              [-0.0024368134327232838, 0.9999938607215881, -0.002514647087082267, 0.09707357734441757],
                              [0.01551333349198103, 0.0025521547067910433, 0.9998764395713806, -0.08006280660629272],
                              [0.0, 0.0, 0.0, 1.0]],
 'left_to_rgb_extrinsics': [[0.999745786190033, -0.010174884460866451, -0.020119857043027878, -3.7557337284088135],
                            [0.010090984404087067, 0.9999399781227112, -0.004267154261469841, -0.004705727566033602],
                            [0.02016206830739975, 0.004063040018081665, 0.9997884631156921, -0.04603101313114166],
                            [0.0, 0.0, 0.0, 1.0]],
 'rgb_distortion': [11.808209419250488, 11.02328872680664, 0.0005683265044353902, -0.0014364976668730378,
                    -1.831695795059204, 11.769153594970703, 14.672675132751465, -1.088363766670227, 0.0, 0.0, 0.0, 0.0,
                    -0.00932407472282648, -0.015433108434081078],
 'left_distortion': [5.454817771911621, 1.694711446762085, 8.319105836562812e-05, -5.4938958783168346e-05,
                     0.029059873893857002, 5.82321310043335, 3.369436502456665, 0.23804199695587158, 0.0, 0.0, 0.0, 0.0,
                     -0.004699581768363714, -0.0014164879685267806],
 'right_distortion': [5.091114521026611, 1.5919005870819092, -7.720104804320727e-06, 2.0027317077619955e-05,
                      0.029687780886888504, 5.4577412605285645, 3.150493621826172, 0.22900323569774628, 0.0, 0.0, 0.0,
                      0.0, -0.0026084419805556536, -0.002354657743126154]}


@dataclass(frozen=True)
class StereoRgbCalibration:
    rgb_resolution: tuple[int, int]
    left_resolution: tuple[int, int]
    right_resolution: tuple[int, int]
    rgb_intrinsics: ArrayLike
    left_intrinsics: ArrayLike
    right_intrinsics: ArrayLike
    rgb_distortion: ArrayLike
    left_distortion: ArrayLike
    right_distortion: ArrayLike
    left_to_right: ArrayLike
    left_to_rgb: ArrayLike
    source_translation_unit: TranslationUnit = "cm"
    ops: MatOps = field(default_factory=NumpyMatOps)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_translation_unit: TranslationUnit = "cm",
        ops: MatOps = NumpyMatOps(),
    ):
        def arr(x, name: str, shape: tuple[int, ...] | None = None) -> ArrayLike:
            value = ops.mat(x, dtype=ops.float64)
            if shape is not None and value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            return ops.copy_mat(value)

        scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[source_translation_unit]
        lr = arr(data["left_to_right_extrinsics"], "left_to_right_extrinsics", (4, 4))
        lrgb = arr(data["left_to_rgb_extrinsics"], "left_to_rgb_extrinsics", (4, 4))
        lr[:3, 3] *= scale
        lrgb[:3, 3] *= scale

        return cls(
            data["rgb_resolution"],
            data["left_resolution"],
            data["right_resolution"],
            arr(data["rgb_intrinsics"], "rgb_intrinsics", (3, 3)),
            arr(data["left_intrinsics"], "left_intrinsics", (3, 3)),
            arr(data["right_intrinsics"], "right_intrinsics", (3, 3)),
            arr(data["rgb_distortion"], "rgb_distortion").reshape(-1, 1),
            arr(data["left_distortion"], "left_distortion").reshape(-1, 1),
            arr(data["right_distortion"], "right_distortion").reshape(-1, 1),
            lr,
            lrgb,
            source_translation_unit,
            ops=ops,
        )

    def get_rectifier(self, alpha: float = 0.0, zero_disparity: bool = True):
        return StereoRectifier(self, alpha, zero_disparity, self.ops)

    @property
    def stereo_baseline_m(self) -> float:
        return abs(float(self.left_to_right[0, 3]))

    @property
    def stereo_baseline_cm(self) -> float:
        return self.stereo_baseline_m * 100

    @property
    def stereo_translation_norm_m(self) -> float:
        return float(self.ops.norm(self.left_to_right[:3, 3]))

    @property
    def left_to_right_rotation(self):
        return self.ops.copy_mat(self.left_to_right[:3, :3])

    @property
    def left_to_right_translation_m(self):
        return self.ops.copy_mat(self.left_to_right[:3, 3:4])

    @property
    def left_to_rgb_rotation(self):
        return self.ops.copy_mat(self.left_to_rgb[:3, :3])

    @property
    def left_to_rgb_translation_m(self):
        return self.ops.copy_mat(self.left_to_rgb[:3, 3:4])

    def as_ops(self, ops: MatOps | None = None):
        ops = ops or NumpyMatOps()
        move = lambda x: _to_ops(x, self.ops, ops)
        return self.__class__(
            self.rgb_resolution,
            self.left_resolution,
            self.right_resolution,
            move(self.rgb_intrinsics),
            move(self.left_intrinsics),
            move(self.right_intrinsics),
            move(self.rgb_distortion),
            move(self.left_distortion),
            move(self.right_distortion),
            move(self.left_to_right),
            move(self.left_to_rgb),
            self.source_translation_unit,
            ops,
        )


@dataclass(frozen=True)
class StereoRectification:
    image_size: tuple[int, int]
    left_map_x: ArrayLike
    left_map_y: ArrayLike
    right_map_x: ArrayLike
    right_map_y: ArrayLike
    R1: ArrayLike
    R2: ArrayLike
    P1: ArrayLike
    P2: ArrayLike
    Q: ArrayLike
    valid_roi_left: tuple[int, int, int, int]
    valid_roi_right: tuple[int, int, int, int]
    ops: MatOps = field(default_factory=NumpyMatOps)

    def disparity_to_points_rectified(
        self,
        disparity: ArrayLike,
        *,
        min_disparity: float = 0.5,
        min_depth_m: float | None = 0.01,
        max_depth_m: float | None = 10.0,
        stride: int = 1,
        mask: ArrayLike | None = None,
    ) -> tuple[ArrayLike, ArrayLike]:

        ops = self.ops

        if ops.ndim(disparity) != 2:
            raise ValueError(
                f"disparity must be HxW, got {ops.shape(disparity)}"
            )

        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        # ---------------------------------------------------------
        # Work in float32.
        #
        # Q may originally be float64, but float32 is normally plenty
        # for disparity reprojection and considerably better on GPU.
        # ---------------------------------------------------------
        d = ops.astype_float32(disparity)
        q = ops.astype_float32(self.Q)

        # ---------------------------------------------------------
        # Select useful pixels BEFORE computing XYZ.
        # ---------------------------------------------------------
        valid = ops.logical_and(
            ops.isfinite(d),
            d > float(min_disparity),
        )

        if mask is not None:
            if tuple(ops.shape(mask)) != tuple(ops.shape(d)):
                raise ValueError(
                    f"mask shape {ops.shape(mask)} "
                    f"does not match disparity shape {ops.shape(d)}"
                )

            valid = ops.logical_and(valid, mask)

        # ---------------------------------------------------------
        # Subsample before reprojection.
        # ---------------------------------------------------------
        if stride > 1:
            sample = ops.zeros(
                ops.shape(valid),
                dtype=ops.uint8,
            )

            sample[::stride, ::stride] = 1

            valid = ops.logical_and(
                valid,
                sample != 0,
            )

        # ---------------------------------------------------------
        # Only retrieve valid pixel locations.
        # ---------------------------------------------------------
        y, x = ops.nonzero(valid)

        if ops.numel(x) == 0:
            return (
                ops.zeros((0, 3), dtype=ops.float64),
                ops.zeros((0, 2), dtype=ops.int32),
            )

        disp = d[y, x]

        # Coordinates participate in floating-point Q projection.
        xf = ops.astype_float32(x)
        yf = ops.astype_float32(y)

        # ---------------------------------------------------------
        # Q @ [x, y, disparity, 1]
        #
        # Avoid constructing Nx4 homogeneous vectors and avoid a
        # general matrix multiply. Q is only 4x4, so evaluate it
        # directly.
        # ---------------------------------------------------------
        Xh = (
            q[0, 0] * xf
            + q[0, 1] * yf
            + q[0, 2] * disp
            + q[0, 3]
        )

        Yh = (
            q[1, 0] * xf
            + q[1, 1] * yf
            + q[1, 2] * disp
            + q[1, 3]
        )

        Zh = (
            q[2, 0] * xf
            + q[2, 1] * yf
            + q[2, 2] * disp
            + q[2, 3]
        )

        Wh = (
            q[3, 0] * xf
            + q[3, 1] * yf
            + q[3, 2] * disp
            + q[3, 3]
        )

        # ---------------------------------------------------------
        # Reject bad homogeneous coordinates before division.
        # ---------------------------------------------------------
        good = ops.logical_and(
            ops.isfinite(Wh),
            ops.abs(Wh) > 1e-12,
        )

        Xh = Xh[good]
        Yh = Yh[good]
        Zh = Zh[good]
        Wh = Wh[good]

        x = x[good]
        y = y[good]

        # ---------------------------------------------------------
        # Homogeneous -> Cartesian
        # ---------------------------------------------------------
        X = Xh / Wh
        Y = Yh / Wh
        Z = Zh / Wh

        # ---------------------------------------------------------
        # 3D filtering
        # ---------------------------------------------------------
        good = (
            ops.isfinite(X)
            & ops.isfinite(Y)
            & ops.isfinite(Z)
            & (Z > 0)
        )

        if min_depth_m is not None:
            good &= Z >= float(min_depth_m)

        if max_depth_m is not None:
            good &= Z <= float(max_depth_m)

        X = X[good]
        Y = Y[good]
        Z = Z[good]

        x = x[good]
        y = y[good]

        # ---------------------------------------------------------
        # Nx3 points and Nx2 image coordinates
        # ---------------------------------------------------------
        points = ops.stack(
            (X, Y, Z),
            dim=1,
        )

        pixels = ops.stack(
            (x, y),
            dim=1,
        )

        # Preserve your previous public API.
        points = ops.astype_float64(points)
        pixels = ops.astype_int32(pixels)

        return points, pixels

    def as_ops(self, ops: MatOps | None = None):
        ops = ops or NumpyMatOps()
        move = lambda x: _to_ops(x, self.ops, ops)
        return self.__class__(
            self.image_size,
            move(self.left_map_x),
            move(self.left_map_y),
            move(self.right_map_x),
            move(self.right_map_y),
            move(self.R1),
            move(self.R2),
            move(self.P1),
            move(self.P2),
            move(self.Q),
            self.valid_roi_left,
            self.valid_roi_right,
            ops,
        )


class BilinearRemap:
    def __init__(
        self,
        map_x: ArrayLike,
        map_y: ArrayLike,
        image_size: tuple[int, int],
        ops: MatOps,
    ):
        self.ops = ops
        self.image_size = tuple(image_size)
        src_w, src_h = self.image_size

        if ops.ndim(map_x) != 2 or ops.ndim(map_y) != 2:
            raise ValueError(
                f"map_x/map_y must be 2D, got {ops.shape(map_x)} and {ops.shape(map_y)}"
            )
        if tuple(ops.shape(map_x)) != tuple(ops.shape(map_y)):
            raise ValueError(f"map shapes differ: {ops.shape(map_x)} != {ops.shape(map_y)}")

        mx, my = ops.astype_float32(map_x), ops.astype_float32(map_y)
        finite = ops.logical_and(ops.isfinite(mx), ops.isfinite(my))
        mx, my = ops.where(finite, mx, 0.0), ops.where(finite, my, 0.0)

        x0f, y0f = ops.floor(mx), ops.floor(my)
        wx, wy = mx - x0f, my - y0f
        x0, y0 = ops.astype_int64(x0f), ops.astype_int64(y0f)
        x1, y1 = x0 + 1, y0 + 1

        valid = (
            finite & (x0 >= 0) & (x0 < src_w) & (y0 >= 0) & (y0 < src_h),
            finite & (x1 >= 0) & (x1 < src_w) & (y0 >= 0) & (y0 < src_h),
            finite & (x0 >= 0) & (x0 < src_w) & (y1 >= 0) & (y1 < src_h),
            finite & (x1 >= 0) & (x1 < src_w) & (y1 >= 0) & (y1 < src_h),
        )

        self.x0, self.x1 = ops.clip(x0, 0, src_w - 1), ops.clip(x1, 0, src_w - 1)
        self.y0, self.y1 = ops.clip(y0, 0, src_h - 1), ops.clip(y1, 0, src_h - 1)

        iwx, iwy = 1.0 - wx, 1.0 - wy
        self.w00 = ops.where(valid[0], iwx * iwy, 0.0)
        self.w01 = ops.where(valid[1], wx * iwy, 0.0)
        self.w10 = ops.where(valid[2], iwx * wy, 0.0)
        self.w11 = ops.where(valid[3], wx * wy, 0.0)

    def remap(self, src: ArrayLike, out: ArrayLike | None = None) -> ArrayLike:
        ops = self.ops
        ndim = ops.ndim(src)
        if ndim not in (2, 3):
            raise ValueError(f"src must be HxW or HxWxC, got {ops.shape(src)}")

        src_w, src_h = self.image_size
        if tuple(ops.shape(src)[:2]) != (src_h, src_w):
            actual = (ops.shape(src)[1], ops.shape(src)[0])
            raise ValueError(f"Input size {actual} != remap size {self.image_size}")

        dtype = ops.dtype(src)
        with ops.no_grad():
            image = ops.astype_float32(src)
            p00, p01 = image[self.y0, self.x0], image[self.y0, self.x1]
            p10, p11 = image[self.y1, self.x0], image[self.y1, self.x1]
            w00, w01, w10, w11 = self.w00, self.w01, self.w10, self.w11

            if ndim == 3:
                w00, w01, w10, w11 = (
                    w00[..., None], w01[..., None], w10[..., None], w11[..., None]
                )

            result = p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11
            if not ops.is_floating_point(src):
                result = ops.round(result)
            result = ops.astype(result, dtype)

            if out is None:
                return result
            if tuple(ops.shape(out)) != tuple(ops.shape(result)):
                raise ValueError(f"out shape {ops.shape(out)} != result shape {ops.shape(result)}")
            if ops.dtype(out) != dtype:
                raise ValueError(f"out dtype {ops.dtype(out)} != src dtype {dtype}")
            return ops.copyto(out, result)


def scale_K(
    K: ArrayLike,
    from_wh: tuple[int, int],
    to_wh: tuple[int, int],
    ops: MatOps,
) -> ArrayLike:
    K = ops.copy_mat(ops.astype_float64(K))
    if from_wh != to_wh:
        sx, sy = to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]
        K[0, [0, 2]] *= sx
        K[1, [1, 2]] *= sy
    return K


class StereoRectifier:
    def __init__(
        self,
        calibration: StereoRgbCalibration,
        alpha=0.0,
        zero_disparity=True,
        ops: MatOps | None = None,
    ):
        self.calibration = calibration
        self.alpha = alpha
        self.zero_disparity = zero_disparity
        self.ops = ops or NumpyMatOps()
        self.rectification = self.image_size = None
        self.left_remapper = self.right_remapper = None

    def make(self, image_size=None):
        size = image_size or self.calibration.left_resolution
        if self.rectification is not None and size == self.image_size:
            return self.rectification

        cal = self.calibration.as_ops(NumpyMatOps())
        self.image_size = size
        if size != cal.left_resolution:
            print("[Warning]: Input size differs from calibration; intrinsics are scaled.")

        K1 = scale_K(cal.left_intrinsics, cal.left_resolution, size, cal.ops)
        K2 = scale_K(cal.right_intrinsics, cal.right_resolution, size, cal.ops)
        flags = cv2.CALIB_ZERO_DISPARITY if self.zero_disparity else 0

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1,
            cal.left_distortion,
            K2,
            cal.right_distortion,
            size,
            cal.left_to_right_rotation,
            cal.left_to_right_translation_m,
            flags=flags,
            alpha=float(self.alpha),
        )
        left_maps = cv2.initUndistortRectifyMap(
            K1, cal.left_distortion, R1, P1, size, cv2.CV_32FC1
        )
        right_maps = cv2.initUndistortRectifyMap(
            K2, cal.right_distortion, R2, P2, size, cv2.CV_32FC1
        )

        self.rectification = StereoRectification(
            size,
            left_maps[0],
            left_maps[1],
            right_maps[0],
            right_maps[1],
            R1,
            R2,
            P1,
            P2,
            Q,
            tuple(map(int, roi1)),
            tuple(map(int, roi2)),
        ).as_ops(self.ops)

        self.left_remapper = self.right_remapper = None
        return self.rectification

    def rectify(self, left: ArrayLike, right: ArrayLike):
        lh, lw = self.ops.shape(left)[:2]
        rh, rw = self.ops.shape(right)[:2]
        if (lh, lw) != (rh, rw):
            raise ValueError(f"Left/right sizes differ: {(lw, lh)} != {(rw, rh)}")

        r = self.make((lw, lh))
        if self.left_remapper is None:
            self.left_remapper = BilinearRemap(
                r.left_map_x, r.left_map_y, r.image_size, self.ops
            )
            self.right_remapper = BilinearRemap(
                r.right_map_x, r.right_map_y, r.image_size, self.ops
            )
        return self.left_remapper.remap(left), self.right_remapper.remap(right), r


def rectified_left_to_original_left(
    points_rectified_m: ArrayLike,
    rectification: StereoRectification,
) -> ArrayLike:
    ops = rectification.ops

    p = ops.astype_float64(points_rectified_m)

    if ops.ndim(p) != 2 or ops.shape(p)[1] != 3:
        raise ValueError(
            f"points must be Nx3, got {ops.shape(p)}"
        )

    # OpenCV:
    #
    #   X_rect = R1 @ X_original
    #
    # therefore:
    #
    #   X_original = R1.T @ X_rect
    #
    # For row-vector Nx3 representation:
    #
    #   P_original = P_rect @ R1
    #
    return ops.matmul(p, rectification.R1)


def transform_points(
    points_m: ArrayLike,
    transform_4x4: ArrayLike,
    ops: MatOps,
) -> ArrayLike:
    p = ops.astype_float64(points_m)
    T = ops.astype_float64(transform_4x4)

    if ops.ndim(p) != 2 or ops.shape(p)[1] != 3:
        raise ValueError(
            f"points must be Nx3, got {ops.shape(p)}"
        )

    if tuple(ops.shape(T)) != (4, 4):
        raise ValueError(
            f"transform must be 4x4, got {ops.shape(T)}"
        )

    R = T[:3, :3]
    t = T[:3, 3]

    # Row-vector representation:
    #
    #   p' = p @ R.T + t
    #
    Rt = ops.permute(R, (1, 0))

    return ops.matmul(p, Rt) + t


def project_camera_points_opencv_model(
    points_camera,
    K,
    distortion=None,
    *,
    dtype,
    ops
):
    """Project Nx3 camera-frame points using OpenCV's distortion model."""    
    VALID_DISTORTION_SIZES = (4, 5, 8, 12, 14)
    p = points_camera

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
        d = ops.zeros((14,), dtype=dtype)
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

        tilt = _tilt_matrix(tx, ty, ops=ops, dtype=dtype)

        zt = tilt[2, 0] * xd + tilt[2, 1] * yd + tilt[2, 2]
        x = (tilt[0, 0] * xd + tilt[0, 1] * yd + tilt[0, 2]) / zt
        y = (tilt[1, 0] * xd + tilt[1, 1] * yd + tilt[1, 2]) / zt

    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 0] * x + K[1, 1] * y + K[1, 2]

    return ops.stack((u, v), dim=1)


def project_points_to_rgb_pixels(
    points_left_m: ArrayLike,
    rgb_image: ArrayLike,
    calibration: StereoRgbCalibration,
    *,
    rgb_image_is_undistorted: bool = False,
) -> tuple[ArrayLike, ArrayLike]:

    ops = calibration.ops

    # HxW or HxWxC
    rgb_h, rgb_w = ops.shape(rgb_image)[:2]

    K = scale_K(
        calibration.rgb_intrinsics,
        calibration.rgb_resolution,
        (rgb_w, rgb_h),
        ops,
    )

    points_rgb = transform_points(
        points_left_m,
        calibration.left_to_rgb,
        ops,
    )

    distortion = (
        None
        if rgb_image_is_undistorted
        else calibration.rgb_distortion
    )

    pixels = project_camera_points_opencv_model(
        points_rgb,
        K,
        distortion,
        ops=ops,
    )

    return pixels, points_rgb


def points_left_to_rgb_depth(
    points_left_m: ArrayLike,
    rgb_image: ArrayLike,
    calibration: StereoRgbCalibration,
    *,
    rgb_image_is_undistorted: bool = False,
    splat_px: int = 0,
) -> tuple[ArrayLike, ArrayLike]:

    ops = calibration.ops

    rgb_h, rgb_w = ops.shape(rgb_image)[:2]

    pixels, points_rgb = project_points_to_rgb_pixels(
        points_left_m,
        rgb_image,
        calibration,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
    )

    # ---------------------------------------------------------
    # Round projected pixels to nearest integer.
    # ---------------------------------------------------------
    u0 = ops.astype_int64(
        ops.round(pixels[:, 0])
    )

    v0 = ops.astype_int64(
        ops.round(pixels[:, 1])
    )

    z = ops.astype_float32(
        points_rgb[:, 2]
    )

    # ---------------------------------------------------------
    # Basic validity independent of splat offset.
    # ---------------------------------------------------------
    pixel_valid = ops.all(
        ops.isfinite(pixels),
        dim=1,
    )

    z_valid = ops.logical_and(
        ops.isfinite(z),
        z > 0,
    )

    base_valid = ops.logical_and(
        pixel_valid,
        z_valid,
    )

    # ---------------------------------------------------------
    # Flat depth buffer.
    #
    # Flattening lets scatter_min work identically on
    # NumPy / Torch / CuPy.
    # ---------------------------------------------------------
    depth = (
        ops.ones(
            (rgb_h * rgb_w,),
            dtype=ops.float32,
        )
        * float("inf")
    )

    r = max(0, int(splat_px))

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):

            u = u0 + dx
            v = v0 + dy

            inside = (
                (u >= 0)
                & (u < rgb_w)
                & (v >= 0)
                & (v < rgb_h)
            )

            ok = ops.logical_and(
                base_valid,
                inside,
            )

            u_ok = u[ok]
            v_ok = v[ok]
            z_ok = z[ok]

            # 2D (v,u) -> flat index
            index = v_ok * rgb_w + u_ok

            ops.scatter_min(
                depth,
                index,
                z_ok,
            )

    depth = ops.reshape(
        depth,
        (rgb_h, rgb_w),
    )

    valid = ops.isfinite(depth)

    depth = ops.where(
        valid,
        depth,
        float("nan"),
    )

    return depth, valid

    
OPS: dict[tuple[str, str], MatOps] = {
    (MatLib.NUMPY, MatDevice.CPU): NumpyMatOps(),
    (MatLib.TORCH, MatDevice.CPU): TorchMatOps(),
    (MatLib.TORCH, MatDevice.CUDA): TorchMatOps(device=MatDevice.CUDA),
    (MatLib.CUPY, MatDevice.CUDA): CupyMatOps(),
}


for (lib, device), op in OPS.items():
    print()
    print("=" * 80)
    print(f"Testing: {lib} / {device}")
    print("=" * 80)

    try:
        # -------------------------------------------------------------
        # Calibration
        # -------------------------------------------------------------
        calib = StereoRgbCalibration.from_dict(
            DEFAULT_CALIBRATION,
            ops=op,
        )

        print("Calibration created")
        print("  left K :", op.shape(calib.left_intrinsics))
        print("  right K:", op.shape(calib.right_intrinsics))
        print("  baseline:", calib.stereo_baseline_m, "m")

        assert op.shape(calib.left_intrinsics) == (3, 3)
        assert op.shape(calib.right_intrinsics) == (3, 3)
        assert op.shape(calib.left_to_right) == (4, 4)

        # -------------------------------------------------------------
        # Conversion back to NumPy
        # -------------------------------------------------------------
        calib_np = calib.as_ops(NumpyMatOps())

        assert isinstance(calib_np.left_intrinsics, np.ndarray)
        assert isinstance(calib_np.left_to_right, np.ndarray)

        print("Calibration -> NumPy conversion OK")

        # -------------------------------------------------------------
        # Stereo rectification
        # -------------------------------------------------------------
        rectifier = StereoRectifier(
            calib,
            ops=op,
        )

        rect = rectifier.make(calib.left_resolution)

        print("Rectification created")
        print("  image_size:", rect.image_size)
        print("  map shape :", op.shape(rect.left_map_x))
        print("  Q shape   :", op.shape(rect.Q))

        w, h = rect.image_size

        assert op.shape(rect.left_map_x) == (h, w)
        assert op.shape(rect.left_map_y) == (h, w)
        assert op.shape(rect.right_map_x) == (h, w)
        assert op.shape(rect.right_map_y) == (h, w)
        assert op.shape(rect.Q) == (4, 4)

        # -------------------------------------------------------------
        # Synthetic grayscale image
        # -------------------------------------------------------------
        img_np = np.arange(
            h * w,
            dtype=np.uint8,
        ).reshape(h, w)

        img = op.from_numpy(img_np)

        left_rect, right_rect, rect2 = rectifier.rectify(
            img,
            img,
        )

        print("Remap completed")
        print("  input :", op.shape(img))
        print("  left  :", op.shape(left_rect))
        print("  right :", op.shape(right_rect))

        assert op.shape(left_rect) == (h, w)
        assert op.shape(right_rect) == (h, w)

        # -------------------------------------------------------------
        # Check output can be converted back to NumPy
        # -------------------------------------------------------------
        left_np = op.to_numpy(left_rect)
        right_np = op.to_numpy(right_rect)

        assert left_np.shape == (h, w)
        assert right_np.shape == (h, w)

        assert np.isfinite(left_np).all()
        assert np.isfinite(right_np).all()

        print("Output conversion OK")
        print("PASS")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")