"""Head pose from 3D face landmarks with Kabsch alignment in numpy.

Six MediaPipe face mesh landmarks (nose tip, chin, outer eye corners, mouth corners)
are aligned to a generic 3D face model. The rotation comes from the SVD of the
cross-covariance between the centered model points and the centered detected points.
Kabsch is invariant to uniform scale, so only the relative scale of x, y and z in the
detected landmarks matters.

Frames:
- Detected landmarks are in image pixels: x right, y down, z from the face mesh model
  where smaller z is closer to the camera, scaled to the same pixel units as x and y.
- Both point sets are aligned in a frame with X right, Y up and Z toward the camera.

Angles in degrees, from R = Rz(roll) @ Ry(yaw) @ Rx(pitch):
- yaw > 0: the nose turns toward the image right
- pitch > 0: the nose turns down
- roll > 0: the face tilts counterclockwise as seen in the image
"""

from __future__ import annotations

import numpy as np

# MediaPipe face mesh indices. "image left" is the side on the left of the image,
# which is the subject's right side for a person facing the camera.
MESH_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "eye_outer_image_left": 33,
    "eye_outer_image_right": 263,
    "mouth_corner_image_left": 61,
    "mouth_corner_image_right": 291,
}

# Generic 3D face model (arbitrary units, X right, Y up, Z toward the camera), same order as MESH_IDS.
CANONICAL = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
])


def to_model_frame(points_xyz: np.ndarray) -> np.ndarray:
    """Image frame (x right, y down, smaller z closer) to X right, Y up, Z toward the camera."""
    p = np.asarray(points_xyz, dtype=np.float64)
    return p * np.array([1.0, -1.0, -1.0])


def kabsch(model: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Rotation R (3x3, det +1) that best maps centered model points onto centered observed points."""
    p = np.asarray(model, dtype=np.float64)
    q = np.asarray(observed, dtype=np.float64)
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def euler_degrees(r: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees for R = Rz(roll) @ Ry(yaw) @ Rx(pitch)."""
    yaw = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    pitch = np.arctan2(r[2, 1], r[2, 2])
    roll = np.arctan2(r[1, 0], r[0, 0])
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def head_pose(landmarks_xyz: np.ndarray) -> dict:
    """Head pose from face mesh landmarks of shape (468, 3) in image pixels.

    Returns yaw, pitch and roll in degrees and the rotation matrix.
    """
    pts = np.asarray(landmarks_xyz, dtype=np.float64)[list(MESH_IDS.values())]
    r = kabsch(CANONICAL, to_model_frame(pts))
    yaw, pitch, roll = euler_degrees(r)
    return {"yaw": yaw, "pitch": pitch, "roll": roll, "rotation": r}
