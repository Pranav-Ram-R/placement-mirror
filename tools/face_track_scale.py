"""Derive VisionConfig.face_track_roi_scale on the qai_hub_models mediapipe_face sample photo.

The face detector ROI is the detector box scaled by face_box_scale (1.1). With a track,
the next ROI comes from the landmarks: a square of side face_track_roi_scale times the
long side of the landmark bounding box in the face aligned frame. This script runs the
detector path once and prints the scale that makes both ROIs the same size, then runs
five tracked frames with that scale and prints the landmark score and ROI size of each,
to show the track holds.

Runs in the eval venv (needs qai_hub_models for the photo and PIL to resize it to the
640x480 capture size). Models run on CPU through ModelRunner.

Usage: python tools/face_track_scale.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.config import VISION  # noqa: E402
from app.runtime.runner import ModelRunner  # noqa: E402
from app.vision.head_pose import MESH_IDS  # noqa: E402
from app.vision.pipeline import VisionPipeline, Frame  # noqa: E402
from app.vision.roi import letterbox_affine, warp_affine_bilinear  # noqa: E402


def side(corners: np.ndarray) -> float:
    return float(np.hypot(*(corners[2] - corners[0])))


def main() -> int:
    from PIL import Image
    from qai_hub_models.models.mediapipe_face.demo import INPUT_IMAGE_ADDRESS

    img = Image.open(INPUT_IMAGE_ADDRESS.fetch()).convert("RGB").resize((640, 480), Image.BILINEAR)
    rgb = np.asarray(img)
    face_in = warp_affine_bilinear(rgb, letterbox_affine(640, 480, 256, 256)[0], 256, 256).round().astype(np.uint8)
    pose_in = warp_affine_bilinear(rgb, letterbox_affine(640, 480, 128, 128)[0], 128, 128).round().astype(np.uint8)
    runner = ModelRunner()

    pipe = VisionPipeline(runner, lambda e: None, replace(VISION, face_track_roi_scale=1.0))
    ev = pipe.process(Frame(0, 0.0, 0.0, rgb, face_in, pose_in))
    print("detector frame:", ev["face"]["state"], "landmark score", round(ev["face"]["landmark_score"], 3))
    # Detector ROI side for this frame, recomputed the way _face does it.
    from app.vision.detection import FACE_ANCHORS, best_detection
    from app.vision.preprocess import image_tensor
    from app.vision.roi import roi_corners

    out, _ = runner.run("face_detector", {"image": image_tensor(face_in, pipe.layouts["face_detector"])})
    det = best_detection(out, FACE_ANCHORS, 256, VISION.face_detector_min_score, VISION.face_nms_iou)
    _, s, (pl, pt) = letterbox_affine(640, 480, 256, 256)
    box = ((det.box.reshape(2, 2) - [pl, pt]) / s).reshape(4)
    kp = (det.keypoints - [pl, pt]) / s
    det_side = side(roi_corners(box, kp[1], kp[0], scale=VISION.face_box_scale))
    bbox_side = side(pipe.track.roi)  # track ROI with scale 1.0 = landmark bounding box long side
    scale = det_side / bbox_side
    print(f"detector ROI side {det_side:.1f} px, landmark box long side {bbox_side:.1f} px, scale {scale:.3f}")

    pipe = VisionPipeline(runner, lambda e: None, replace(VISION, face_track_roi_scale=round(scale, 2)))
    for i in range(6):
        ev = pipe.process(Frame(i, 0.0, 0.0, rgb, face_in if i == 0 else None, pose_in))
        print(f"frame {i}: {ev['face']['state']:9} score {ev['face']['landmark_score']:.3f} "
              f"yaw {ev['face']['yaw']:+.2f} pitch {ev['face']['pitch']:+.2f} next ROI side {side(pipe.track.roi):.1f} px")
    print(f"eye corners used for the track rotation: {MESH_IDS['eye_outer_image_right']}, {MESH_IDS['eye_outer_image_left']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
