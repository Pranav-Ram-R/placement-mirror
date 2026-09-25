"""BlazeFace and BlazePose detector post-processing in numpy.

Follows qai_hub_models templates/mediapipe (decode, sigmoid score, hard NMS). The anchors
are generated here with the MediaPipe SSD anchor rule (ssd_anchors_calculator with
fixed_anchor_size) so the app does not need the qai_hub_models asset files. tests check
them against anchors_face_back.npy and anchors_pose.npy when those files are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCORE_CLIP = 100.0


def ssd_anchors(input_size: int, strides: list[int], anchors_per_layer: int = 2) -> np.ndarray:
    """(N, 4) anchors (x_center, y_center, w, h) normalized to the input, w = h = 1.

    Consecutive layers with the same stride share one feature map, so their anchors are
    interleaved per grid cell, as in MediaPipe's SsdAnchorsCalculator.
    """
    out = []
    i = 0
    while i < len(strides):
        j = i
        while j < len(strides) and strides[j] == strides[i]:
            j += 1
        size = int(np.ceil(input_size / strides[i]))
        per_cell = anchors_per_layer * (j - i)
        ys, xs = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size, indexing="ij")
        centers = np.stack([xs.ravel(), ys.ravel()], axis=1)
        centers = np.repeat(centers, per_cell, axis=0)
        out.append(np.hstack([centers, np.ones_like(centers)]))
        i = j
    return np.vstack(out).astype(np.float32)


# Anchor options of the two detectors (MediaPipe face_detection_full_range 256x256 "back"
# model and the BlazePose detector 128x128), as in the MediaPipePyTorch README.
FACE_ANCHORS = ssd_anchors(256, [16, 32, 32, 32])
POSE_ANCHORS = ssd_anchors(128, [8, 16, 16, 16])


@dataclass
class Detection:
    score: float
    box: np.ndarray  # (4,) x1, y1, x2, y2 in detector input pixels
    keypoints: np.ndarray  # (K, 2) in detector input pixels


def decode(coords: np.ndarray, scores: np.ndarray, anchors: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw outputs (N, 4 + 2K) and (N, 1) to boxes (N, 4) xyxy, keypoints (N, K, 2) and probabilities (N,)."""
    pairs = coords.reshape(len(anchors), -1, 2).astype(np.float32)
    offset = anchors[:, None, 0:2] * size
    offset = np.repeat(offset, pairs.shape[1], axis=1)
    offset[:, 1] = 0.0  # the (w, h) pair gets no offset
    pairs = pairs * anchors[:, None, 2:4] + offset
    xc, yc, w, h = pairs[:, 0, 0], pairs[:, 0, 1], pairs[:, 1, 0], pairs[:, 1, 1]
    boxes = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)
    clipped = np.clip(scores.reshape(-1).astype(np.float64), -SCORE_CLIP, SCORE_CLIP)  # exp(100) overflows float32
    probs = 1.0 / (1.0 + np.exp(-clipped))
    return boxes, pairs[:, 2:], probs


def nms(boxes: np.ndarray, probs: np.ndarray, min_score: float, iou_threshold: float) -> list[int]:
    """Hard NMS, highest score first. Returns kept indices."""
    order = np.flatnonzero(probs >= min_score)
    order = order[np.argsort(-probs[order], kind="stable")]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def best_detection(outputs: dict[str, np.ndarray], anchors: np.ndarray, size: int, min_score: float,
                   iou_threshold: float) -> Detection | None:
    """Highest scoring detection after NMS from the four detector outputs, or None."""
    coords = np.concatenate([outputs["box_coords_1"][0], outputs["box_coords_2"][0]], axis=0)
    scores = np.concatenate([outputs["box_scores_1"][0], outputs["box_scores_2"][0]], axis=0)
    boxes, kps, probs = decode(coords, scores, anchors, size)
    keep = nms(boxes, probs, min_score, iou_threshold)
    if not keep:
        return None
    i = keep[0]
    return Detection(score=float(probs[i]), box=boxes[i].astype(np.float64), keypoints=kps[i].astype(np.float64))
