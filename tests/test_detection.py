from pathlib import Path

import numpy as np
import pytest

from app.vision.detection import FACE_ANCHORS, POSE_ANCHORS, best_detection, decode, nms, ssd_anchors

# qai_hub_models clones the MediaPipePyTorch repo here. Present on the dev machine only.
QAIHM_REPOS = Path.home() / ".qaihm"


def reference_anchor_file(name: str) -> Path:
    found = sorted(QAIHM_REPOS.glob(f"**/mediapipe/{name}")) if QAIHM_REPOS.exists() else []
    if not found:
        pytest.skip(f"{name} from the qai_hub_models MediaPipe repo is not on this machine")
    return found[0]


@pytest.mark.parametrize("name,anchors", [("anchors_face_back.npy", FACE_ANCHORS), ("anchors_pose.npy", POSE_ANCHORS)])
def test_generated_anchors_match_the_mediapipe_files(name, anchors):
    ref = np.load(reference_anchor_file(name))
    assert anchors.shape == ref.shape == (896, 4)
    assert np.abs(anchors - ref).max() == 0.0


def test_anchor_counts_match_detector_outputs():
    # face detector outputs 512 + 384 boxes, pose detector 512 + 384
    for anchors in (ssd_anchors(256, [16, 32, 32, 32]), ssd_anchors(128, [8, 16, 16, 16])):
        assert len(anchors) == 896 and np.all(anchors[:, 2:] == 1.0)


def test_decode_and_nms_find_the_planted_box():
    anchors = FACE_ANCHORS
    coords = np.zeros((896, 16), np.float32)
    scores = np.full((896, 1), -20.0, np.float32)
    i = 200
    # box of 40 x 50 px centered 3 px right of the anchor, first keypoint 5 px left of it
    coords[i, :4] = [3.0, 0.0, 40.0, 50.0]
    coords[i, 4:6] = [-5.0, 0.0]
    scores[i] = 5.0
    scores[i + 1] = 4.0  # same place, lower score: removed by NMS
    coords[i + 1] = coords[i]
    boxes, kps, probs = decode(coords, scores, anchors, 256)
    ax, ay = anchors[i, :2] * 256
    assert np.allclose(boxes[i], [ax + 3 - 20, ay - 25, ax + 3 + 20, ay + 25])
    assert np.allclose(kps[i, 0], [ax - 5, ay])
    assert nms(boxes, probs, 0.8, 0.3) == [i]
    outputs = {"box_coords_1": coords[None, :512], "box_coords_2": coords[None, 512:],
               "box_scores_1": scores[None, :512], "box_scores_2": scores[None, 512:]}
    det = best_detection(outputs, anchors, 256, 0.8, 0.3)
    assert det is not None and det.score == pytest.approx(1 / (1 + np.exp(-5.0)))
    scores[:] = -20.0
    assert best_detection(outputs | {"box_scores_1": scores[None, :512]}, anchors, 256, 0.8, 0.3) is None
