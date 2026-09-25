"""Model input tensors from RGB images, in the layout the loaded session expects.

The AI Hub onnx vision models take NCHW input. The precompiled_qnn_onnx variants were
compiled with --force_channel_last_input and take NHWC. Values are RGB scaled to [0, 1]
in both, as in qai_hub_models mediapipe (image / 255).
"""

from __future__ import annotations

import numpy as np

NCHW, NHWC = "NCHW", "NHWC"
INV_255 = np.float32(1.0 / 255.0)


def input_layout(shape) -> str:
    """Layout of a 4D image input from its declared shape, for example [1, 3, 256, 256]."""
    if len(shape) != 4:
        raise ValueError(f"expected a 4D image input, got shape {shape}")
    if shape[1] == 3 and shape[3] != 3:
        return NCHW
    if shape[3] == 3 and shape[1] != 3:
        return NHWC
    raise ValueError(f"cannot tell NCHW from NHWC for shape {shape}")


def image_tensor(rgb: np.ndarray, layout: str) -> np.ndarray:
    """(H, W, 3) RGB, uint8 or float in 0..255, to a float32 (1, ...) tensor in [0, 1]."""
    x = rgb.astype(np.float32, copy=False) * INV_255
    if layout == NCHW:
        x = x.transpose(2, 0, 1)
    elif layout != NHWC:
        raise ValueError(f"unknown layout {layout}")
    return np.ascontiguousarray(x[None])
