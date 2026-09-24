"""Rotated region of interest (ROI) crop in numpy.

Follows the qai_hub_models MediaPipe app (templates/mediapipe/app.py): the ROI is the
detector box scaled by 1.1 and rotated by the angle of the eye to eye keypoint vector.
Its top left, bottom left and top right corners map to the corners of the landmark
model input. The warp matches cv2.warpAffine with bilinear sampling and a constant
zero border, without OpenCV (CLAUDE.md rule 10).

Affine matrices are 2x3 and map source image pixels to output pixels, the same
convention as cv2.warpAffine.
"""

from __future__ import annotations

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
    w, h = w * scale, h * scale
    unit = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64) * [w / 2, h / 2]
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return unit @ rot.T + [xc, yc]


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


def warp_affine_bilinear(image: np.ndarray, m: np.ndarray, out_w: int, out_h: int,
                         border_value: float = 0.0) -> np.ndarray:
    """Warp an HxWxC image with a source to output affine. Returns float32 (out_h, out_w, C).

    Each output pixel is sampled from the source with bilinear interpolation. Taps that
    fall outside the source use border_value.
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
