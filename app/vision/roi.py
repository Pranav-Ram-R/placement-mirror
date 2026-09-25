"""Rotated region of interest (ROI) crop in numpy.

Follows the qai_hub_models MediaPipe app (templates/mediapipe/app.py): the ROI is the
detector box scaled by 1.1 and rotated by the angle of the eye to eye keypoint vector.
Its top left, bottom left and top right corners map to the corners of the landmark
model input. The warp matches cv2.warpAffine with bilinear sampling and a constant
zero border, without OpenCV (CLAUDE.md rule 10). warp_reference is the Day 1
implementation, kept to check the vectorized warp_affine_bilinear against.

Affine matrices are 2x3 and map source image pixels to output pixels, the same
convention as cv2.warpAffine.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def roi_corners(box_xyxy: np.ndarray, kp_start: np.ndarray, kp_end: np.ndarray, scale: float = 1.1,
                offset: float = 0.0) -> np.ndarray:
    """Corners (top left, bottom left, top right, bottom right) of the rotated ROI. Shape (4, 2)."""
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float64)
    kp_start = np.asarray(kp_start, dtype=np.float64)
    kp_end = np.asarray(kp_end, dtype=np.float64)
    theta = np.arctan2(kp_start[1] - kp_end[1], kp_start[0] - kp_end[0])
    xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    if offset:
        vec = kp_end - kp_start
        xc, yc = np.array([xc, yc]) + offset * w * vec / np.linalg.norm(vec)
    return rect_corners(xc, yc, w * scale, h * scale, theta)


def rect_corners(xc: float, yc: float, w: float, h: float, theta: float) -> np.ndarray:
    """Corners (top left, bottom left, top right, bottom right) of a w x h box centered on
    (xc, yc) and rotated by theta radians, as compute_box_corners_with_rotation in
    qai_hub_models. Shape (4, 2)."""
    unit = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64) * [w / 2, h / 2]
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return unit @ rot.T + [xc, yc]


def keypoint_roi_corners(kp_center: np.ndarray, kp_end: np.ndarray, scale: float,
                         rotation_offset: float) -> np.ndarray:
    """Square ROI centered on one keypoint, as the qai_hub_models MediaPipe pose app does.

    Side = 2 * scale * |kp_center - kp_end|. Rotation = angle of kp_center - kp_end minus
    rotation_offset (pi / 2 for the pose detector's hip to head keypoint vector).
    """
    c = np.asarray(kp_center, dtype=np.float64)
    e = np.asarray(kp_end, dtype=np.float64)
    theta = np.arctan2(c[1] - e[1], c[0] - e[0]) - rotation_offset
    side = 2.0 * scale * float(np.hypot(*(c - e)))
    return rect_corners(c[0], c[1], side, side, theta)


def landmarks_roi_corners(points_xy: np.ndarray, start_idx: int, end_idx: int, scale: float) -> np.ndarray:
    """Square ROI around landmarks, for tracking a face from the previous frame.

    Rotation = angle of points[start_idx] - points[end_idx], as for the detector keypoints.
    The landmarks are rotated into that frame, and the ROI is centered on their bounding
    box with side = scale * the box's long side.
    """
    p = np.asarray(points_xy, dtype=np.float64)
    d = p[start_idx] - p[end_idx]
    theta = np.arctan2(d[1], d[0])
    c, s = np.cos(theta), np.sin(theta)
    aligned = p @ np.array([[c, -s], [s, c]])  # rotate by -theta
    lo, hi = aligned.min(axis=0), aligned.max(axis=0)
    center = np.array([[c, -s], [s, c]]) @ ((lo + hi) / 2)
    side = scale * float((hi - lo).max())
    return rect_corners(center[0], center[1], side, side, theta)


def affine_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 affine that maps 3 source points onto 3 destination points."""
    src = np.asarray(src, dtype=np.float64)
    a = np.hstack([src, np.ones((3, 1))])
    return np.linalg.solve(a, np.asarray(dst, dtype=np.float64)).T


def crop_affine(corners: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Affine that maps the ROI top left, bottom left and top right corners to the output corners."""
    dst = np.array([[0, 0], [0, out_h - 1], [out_w - 1, 0]], dtype=np.float64)
    return affine_from_points(np.asarray(corners)[:3], dst)


def letterbox_affine(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and center with zero padding, like qai_hub_models resize_pad.

    Returns the affine, the scale and the (left, top) padding. The affine uses half pixel
    centers, the same sampling as a bilinear resize with align_corners=False.
    """
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = int(np.floor(src_w * scale)), int(np.floor(src_h * scale))
    pad_left, pad_top = (dst_w - new_w) // 2, (dst_h - new_h) // 2
    m = np.array([[scale, 0, pad_left + 0.5 * scale - 0.5],
                  [0, scale, pad_top + 0.5 * scale - 0.5]], dtype=np.float64)
    return m, scale, (pad_left, pad_top)


def invert_affine(m: np.ndarray) -> np.ndarray:
    full = np.vstack([np.asarray(m, dtype=np.float64), [0, 0, 1]])
    return np.linalg.inv(full)[:2]


def apply_affine(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ np.asarray(m)[:, :2].T + np.asarray(m)[:, 2]


def affine_scale(m: np.ndarray) -> float:
    """Uniform scale factor of an affine (square root of the absolute determinant)."""
    return float(np.sqrt(abs(np.linalg.det(np.asarray(m, dtype=np.float64)[:, :2]))))


@lru_cache(maxsize=16)
def _grid(out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Output pixel coordinates u, v as flat float32 arrays, computed once per output size."""
    u, v = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    u, v = u.ravel(), v.ravel()
    u.flags.writeable = False
    v.flags.writeable = False
    return u, v


def pad_border(image: np.ndarray, value: float = 0.0) -> np.ndarray:
    """Copy of an HxWxC image with a one pixel border of value. uint8 stays uint8 when value fits."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[..., None]
    h, w, c = img.shape
    dtype = img.dtype if (img.dtype != np.uint8 or (float(value).is_integer() and 0 <= value <= 255)) else np.float32
    out = np.empty((h + 2, w + 2, c), dtype)
    out[0] = value
    out[-1] = value
    out[1:-1, 0] = value
    out[1:-1, -1] = value
    out[1:-1, 1:-1] = img
    return out


def warp_padded(padded: np.ndarray, m: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """warp_affine_bilinear on an image already passed through pad_border.

    m maps pixels of the unpadded image to output pixels. Lets one padded frame serve
    several crops.
    """
    hp, wp, c = padded.shape
    h, w = hp - 2, wp - 2
    flat = padded.reshape(hp * wp, c)
    inv = invert_affine(m).astype(np.float32)
    u, v = _grid(out_w, out_h)
    x = inv[0, 0] * u + inv[0, 1] * v + inv[0, 2]
    y = inv[1, 0] * u + inv[1, 1] * v + inv[1, 2]
    # A tap at -1 or at w (h) reads the border. Clamping here keeps every tap in the padded
    # image, and a coordinate that is clamped samples the border with full weight.
    np.clip(x, -1, w, out=x)
    np.clip(y, -1, h, out=y)
    x0f = np.floor(x)
    y0f = np.floor(y)
    fx = x - x0f
    fy = y - y0f
    gx = 1 - fx
    gy = 1 - fy
    i0 = (y0f.astype(np.int32) + 1) * wp + (x0f.astype(np.int32) + 1)
    i1 = i0 + wp

    def planar(index: np.ndarray) -> np.ndarray:
        # Taps with zero weight at the far edge can point one past the image. mode="clip"
        # keeps them in bounds and their weight keeps them out of the result.
        return np.ascontiguousarray(np.take(flat, index, axis=0, mode="clip").T)

    out = planar(i0) * (gy * gx)
    tmp = np.empty_like(out)
    out += np.multiply(planar(i0 + 1), gy * fx, out=tmp)
    out += np.multiply(planar(i1), fy * gx, out=tmp)
    out += np.multiply(planar(i1 + 1), fy * fx, out=tmp)
    return out.T.reshape(out_h, out_w, c)


def warp_affine_bilinear(image: np.ndarray, m: np.ndarray, out_w: int, out_h: int,
                         border_value: float = 0.0) -> np.ndarray:
    """Warp an HxWxC image with a source to output affine. Returns float32 (out_h, out_w, C).

    Each output pixel is sampled from the source with bilinear interpolation. Taps that
    fall outside the source use border_value. Same sampling and the same float32 operation
    order as warp_reference, vectorized with no Python loops and no float64:
    - output coordinates come from precomputed flat grids
    - a one pixel border replaces the per tap range masks
    - the source is gathered as is (uint8 stays uint8), so the full frame is never converted
    - the weighted sum runs on channel planar (C, N) arrays, where numpy's inner loop is long

    The result is an (out_h, out_w, C) view of channel planar memory, so
    result.transpose(2, 0, 1) is contiguous NCHW data with no copy.
    """
    return warp_padded(pad_border(image, border_value), m, out_w, out_h)


def warp_reference(image: np.ndarray, m: np.ndarray, out_w: int, out_h: int,
                         border_value: float = 0.0) -> np.ndarray:
    """Day 1 implementation, kept as the reference for warp_affine_bilinear.

    Warps an HxWxC image with a source to output affine and returns float32
    (out_h, out_w, C). Each output pixel is sampled from the source with bilinear
    interpolation. Taps that fall outside the source use border_value.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    h, w = img.shape[:2]
    inv = invert_affine(m).astype(np.float32)
    u, v = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    x = inv[0, 0] * u + inv[0, 1] * v + inv[0, 2]
    y = inv[1, 0] * u + inv[1, 1] * v + inv[1, 2]
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)

    def tap(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        vals = img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]
        vals[~inside] = border_value
        return vals

    out = (tap(y0, x0) * ((1 - fx) * (1 - fy)) + tap(y0, x0 + 1) * (fx * (1 - fy))
           + tap(y0 + 1, x0) * ((1 - fx) * fy) + tap(y0 + 1, x0 + 1) * (fx * fy))
    return out.astype(np.float32)
