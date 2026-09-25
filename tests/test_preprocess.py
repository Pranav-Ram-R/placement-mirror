import numpy as np
import pytest

from app.vision.preprocess import NCHW, NHWC, image_tensor, input_layout
from app.vision.roi import crop_affine, letterbox_affine, roi_corners, warp_affine_bilinear

# Input shapes of the fetched models (onnx and precompiled_qnn_onnx), read from the graphs
# in models/ on 2026-09-25.
SHAPES = {
    ("face_detector", "onnx"): [1, 3, 256, 256],
    ("face_detector", "precompiled_qnn_onnx"): [1, 256, 256, 3],
    ("face_landmark", "onnx"): [1, 3, 192, 192],
    ("face_landmark", "precompiled_qnn_onnx"): [1, 192, 192, 3],
    ("pose_detector", "onnx"): [1, 3, 128, 128],
    ("pose_detector", "precompiled_qnn_onnx"): [1, 128, 128, 3],
    ("pose_landmark", "onnx"): [1, 3, 256, 256],
    ("pose_landmark", "precompiled_qnn_onnx"): [1, 256, 256, 3],
}


@pytest.mark.parametrize("key", list(SHAPES))
def test_layout_from_model_input_shape(key):
    assert input_layout(SHAPES[key]) == (NCHW if key[1] == "onnx" else NHWC)


def test_layout_rejects_ambiguous_shapes():
    for shape in ([1, 3, 3, 3], [1, 80, 3000], [1, 4, 256, 256]):
        with pytest.raises(ValueError):
            input_layout(shape)


def test_nhwc_and_nchw_tensors_hold_identical_values_for_one_frame():
    frame = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    m_lb, _, _ = letterbox_affine(640, 480, 256, 256)
    corners = roi_corners(np.array([250.0, 150.0, 400.0, 330.0]), np.array([290.0, 210.0]),
                          np.array([360.0, 200.0]), scale=1.1)
    for image, size in ((warp_affine_bilinear(frame, m_lb, 256, 256), 256),
                        (warp_affine_bilinear(frame, crop_affine(corners, 192, 192), 192, 192), 192)):
        nhwc, nchw = image_tensor(image, NHWC), image_tensor(image, NCHW)
        assert nhwc.shape == (1, size, size, 3) and nchw.shape == (1, 3, size, size)
        assert nhwc.dtype == nchw.dtype == np.float32
        assert nhwc.flags.c_contiguous and nchw.flags.c_contiguous
        assert np.array_equal(nhwc.transpose(0, 3, 1, 2), nchw)
        assert np.array_equal(nhwc[0], (image.astype(np.float32) * np.float32(1 / 255)))
