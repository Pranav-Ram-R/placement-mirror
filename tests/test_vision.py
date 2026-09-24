import numpy as np
import pytest

from app.vision.head_pose import CANONICAL, MESH_IDS, euler_degrees, head_pose, kabsch, to_model_frame
from app.vision.roi import (
    affine_scale,
    apply_affine,
    crop_affine,
    invert_affine,
    letterbox_affine,
    roi_corners,
    warp_affine_bilinear,
)


def rotation(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return rz @ ry @ rx


@pytest.mark.parametrize("angles", [(0, 0, 0), (20, -10, 5), (-35, 15, -8), (10, 25, 0)])
def test_kabsch_recovers_rotation_under_scale_translation_and_noise(angles):
    rng = np.random.default_rng(0)
    r = rotation(*angles)
    observed = 0.37 * CANONICAL @ r.T + [120.0, -40.0, 7.0] + rng.normal(0, 0.5, CANONICAL.shape)
    est = kabsch(CANONICAL, observed)
    assert np.allclose(np.linalg.det(est), 1.0)
    assert np.allclose(euler_degrees(est), angles, atol=0.5)


def test_head_pose_from_image_frame_landmarks():
    yaw, pitch, roll = 18.0, -12.0, 4.0
    model = CANONICAL @ rotation(yaw, pitch, roll).T * 0.5 + [320.0, -240.0, 0.0]
    landmarks = np.zeros((468, 3))
    landmarks[list(MESH_IDS.values())] = to_model_frame(model)  # the frame flip is its own inverse
    pose = head_pose(landmarks)
    assert np.allclose([pose["yaw"], pose["pitch"], pose["roll"]], [yaw, pitch, roll], atol=1e-6)


def test_yaw_sign_nose_toward_image_right():
    model = CANONICAL @ rotation(30, 0, 0).T
    assert model[0, 0] > model[4:6, 0].mean()  # nose tip moves toward +X, the image right


def test_crop_affine_maps_corners():
    corners = roi_corners([100, 80, 200, 200], kp_start=[170, 120], kp_end=[130, 118])
    m = crop_affine(corners, 192, 192)
    assert np.allclose(apply_affine(m, corners[:3]), [[0, 0], [0, 191], [191, 0]], atol=1e-9)
    assert np.allclose(apply_affine(invert_affine(m), [[0, 0], [0, 191], [191, 0]]), corners[:3], atol=1e-9)
    assert np.isclose(affine_scale(invert_affine(m)) * affine_scale(m), 1.0)


def test_roi_corners_upright_box_is_axis_aligned_and_scaled():
    corners = roi_corners([0, 0, 100, 50], kp_start=[70, 20], kp_end=[30, 20], scale=1.1)
    assert np.allclose(corners, [[-5, -2.5], [-5, 52.5], [105, -2.5], [105, 52.5]])


def test_letterbox_affine_matches_resize_pad_layout():
    m, scale, pad = letterbox_affine(640, 480, 256, 256)
    assert scale == pytest.approx(0.4)
    assert pad == (0, 32)
    x, y = apply_affine(m, [[0, 0]])[0]
    assert x == pytest.approx(0.5 * 0.4 - 0.5) and y == pytest.approx(32 + 0.5 * 0.4 - 0.5)


def test_warp_identity_and_translation():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (40, 50, 3)).astype(np.uint8)
    ident = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert np.array_equal(warp_affine_bilinear(img, ident, 50, 40), img.astype(np.float32))
    shift = np.array([[1.0, 0, 3], [0, 1.0, 2]])
    out = warp_affine_bilinear(img, shift, 50, 40)
    assert np.array_equal(out[2:, 3:], img[:-2, :-3].astype(np.float32))
    assert np.all(out[:2] == 0) and np.all(out[:, :3] == 0)


def test_warp_matches_cv2_warp_affine():
    # cv2 is a test-only reference here (qai_hub_models crops with cv2.warpAffine). cv2 rounds
    # sub-pixel weights to 1/32, so compare interior pixels of a smooth image tightly and
    # bound the rest by that rounding.
    cv2 = pytest.importorskip("cv2")
    yy, xx = np.mgrid[0:120, 0:160].astype(np.float32)
    smooth = np.stack([xx * 1.5, yy * 2.0, 128 + 100 * np.sin(xx / 9) * np.cos(yy / 7)], axis=-1)
    corners = roi_corners([30, 20, 130, 110], kp_start=[95, 50], kp_end=[60, 58])
    m = crop_affine(corners, 64, 64)
    ours = warp_affine_bilinear(smooth, m, 64, 64)
    ref = cv2.warpAffine(smooth, m, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    interior = warp_affine_bilinear(np.ones((120, 160, 1), np.float32), m, 64, 64)[..., 0] == 1.0
    diff = np.abs(ours - ref)
    assert interior.sum() > 0.8 * interior.size
    assert diff[interior].max() < 0.5
    assert diff.max() < smooth.max() / 32 + 0.5
    assert diff.mean() < 0.1
