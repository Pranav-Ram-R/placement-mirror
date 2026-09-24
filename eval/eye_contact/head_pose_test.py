"""Eye contact feasibility (Task C): head pose from the AI Hub mediapipe_face models.

Pipeline per frame, all on the local x86 CPU:
1. letterbox the RGB frame to 256x256 (app.vision.roi, numpy)
2. face detector, onnx target model, ONNX Runtime CPU execution provider
3. anchor decode, score sigmoid and hard NMS, following qai_hub_models templates/mediapipe
4. rotated ROI and 192x192 crop (app.vision.roi, numpy bilinear affine warp)
5. face landmark detector, onnx target model, ONNX Runtime CPU execution provider
6. landmarks back to image pixels, z scaled by the ROI scale
7. head pose by Kabsch alignment on 6 landmarks (app.vision.head_pose)

The detector runs on every frame. cv2 is used only to read the video file.

Analysis, using the labels written by record_scripted.py:
- calibration: mean yaw and pitch over the first --calib-s seconds of CAMERA frames is zero
- per segment: calibrated yaw and pitch mean and standard deviation
- facing vs away: facing = CAMERA, away = every other label. A threshold on the angular
  distance from zero is chosen on frames before --split-s (calibration frames excluded)
  and tested on frames from --split-s on
- CAMERA vs SCREEN overlap in yaw and pitch

Usage:
  python eval/eye_contact/head_pose_test.py --video <file> --labels <labels.csv>
                                            [--calib-s 3] [--split-s 30] [--out-dir eval/eye_contact/results]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.vision.head_pose import head_pose  # noqa: E402
from app.vision.roi import (  # noqa: E402
    affine_scale,
    apply_affine,
    crop_affine,
    invert_affine,
    letterbox_affine,
    roi_corners,
    warp_affine_bilinear,
)
from benchmarks.schema import Measurement, Source, save_json  # noqa: E402

MODELS_DIR = REPO / "models" / "mediapipe_face" / "mediapipe_face-onnx-float"
DET_SIZE, LM_SIZE = 256, 192
# Values from qai_hub_models mediapipe_face (app.py and model.py defaults).
MIN_BOX_SCORE, NMS_IOU, MIN_LANDMARK_SCORE = 0.8, 0.3, 0.5
SCORE_CLIP, BOX_SCALE, BOX_OFFSET = 100.0, 1.1, 0.0
KP_START, KP_END = 1, 0  # RIGHT_EYE_KEYPOINT_INDEX to LEFT_EYE_KEYPOINT_INDEX
STAGES = ("letterbox", "detector", "decode_roi", "roi_warp", "landmark", "head_pose", "total")


def load_anchors() -> np.ndarray:
    """Detector anchors (896, 4) = (x, y, w, h), from the repo qai_hub_models uses for this model."""
    from qai_hub_models.utils.asset_loaders import always_answer_prompts

    with always_answer_prompts(True):
        from qai_hub_models.models.mediapipe_face.model import MEDIAPIPE_REPO_DIR
    return np.load(Path(MEDIAPIPE_REPO_DIR) / "anchors_face_back.npy").astype(np.float32)


def decode(coords: np.ndarray, scores: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw detector outputs to boxes (N, 4) xyxy and keypoints (N, 6, 2) in 256 px detector space."""
    pairs = coords.reshape(-1, 8, 2)
    offset = anchors[:, None, 0:2] * DET_SIZE
    offset = np.where(np.arange(8)[None, :, None] == 1, 0.0, offset)  # the (w, h) pair gets no offset
    pairs = pairs * anchors[:, None, 2:4] + offset
    xc, yc, w, h = pairs[:, 0, 0], pairs[:, 0, 1], pairs[:, 1, 0], pairs[:, 1, 1]
    boxes = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores.reshape(-1).astype(np.float64), -SCORE_CLIP, SCORE_CLIP)))
    return boxes, pairs[:, 2:], probs


def nms(boxes: np.ndarray, probs: np.ndarray) -> list[int]:
    keep, order = [], [i for i in np.argsort(-probs) if probs[i] >= MIN_BOX_SCORE]
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = np.array(order, dtype=int)
        if not len(rest):
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
        order = [int(j) for j, o in zip(rest, iou) if o <= NMS_IOU]
    return keep


class Pipeline:
    def __init__(self, models_dir: Path):
        import onnxruntime as ort

        self.ort = ort
        opts = dict(providers=["CPUExecutionProvider"])
        self.det = ort.InferenceSession(str(models_dir / "face_detector.onnx"), **opts)
        self.lm = ort.InferenceSession(str(models_dir / "face_landmark_detector.onnx"), **opts)
        self.providers = {"detector": self.det.get_providers(), "landmark": self.lm.get_providers()}
        self.anchors = load_anchors()
        self.det_outputs = [o.name for o in self.det.get_outputs()]

    def __call__(self, rgb: np.ndarray) -> dict:
        t = {}
        start = time.perf_counter()
        h, w = rgb.shape[:2]
        m_lb, scale, (pad_l, pad_t) = letterbox_affine(w, h, DET_SIZE, DET_SIZE)
        det_in = (warp_affine_bilinear(rgb, m_lb, DET_SIZE, DET_SIZE) / 255.0).transpose(2, 0, 1)[None]
        t["letterbox"] = time.perf_counter() - start

        s = time.perf_counter()
        out = dict(zip(self.det_outputs, self.det.run(None, {"image": det_in.astype(np.float32)})))
        t["detector"] = time.perf_counter() - s

        s = time.perf_counter()
        coords = np.concatenate([out["box_coords_1"][0], out["box_coords_2"][0]], axis=0)
        scores = np.concatenate([out["box_scores_1"][0], out["box_scores_2"][0]], axis=0)
        boxes, kps, probs = decode(coords, scores, self.anchors)
        keep = nms(boxes, probs)
        result = {"detected": bool(keep), "box_score": float(probs[keep[0]]) if keep else None}
        if not keep:
            t["decode_roi"] = time.perf_counter() - s
            t["total"] = time.perf_counter() - start
            return {**result, "times": t}
        i = keep[0]
        unpad = lambda p: (p - [pad_l, pad_t]) / scale  # noqa: E731  zoo denormalize_coordinates
        box = unpad(boxes[i].reshape(2, 2)).reshape(4)
        kp = unpad(kps[i])
        corners = roi_corners(box, kp[KP_START], kp[KP_END], scale=BOX_SCALE, offset=BOX_OFFSET)
        m_crop = crop_affine(corners, LM_SIZE, LM_SIZE)
        t["decode_roi"] = time.perf_counter() - s

        s = time.perf_counter()
        crop = warp_affine_bilinear(rgb, m_crop, LM_SIZE, LM_SIZE)
        t["roi_warp"] = time.perf_counter() - s

        s = time.perf_counter()
        lm_score, lm = self.lm.run(None, {"image": (crop / 255.0).transpose(2, 0, 1)[None].astype(np.float32)})
        t["landmark"] = time.perf_counter() - s
        result.update(landmark_score=float(lm_score.reshape(-1)[0]), landmark_count=int(lm.shape[1]))
        if result["landmark_score"] < MIN_LANDMARK_SCORE:
            t["total"] = time.perf_counter() - start
            return {**result, "times": t}

        s = time.perf_counter()
        inv = invert_affine(m_crop)
        pts = lm[0].astype(np.float64)
        img_xy = apply_affine(inv, pts[:, :2] * LM_SIZE)
        img_z = pts[:, 2:3] * LM_SIZE * affine_scale(inv)
        pose = head_pose(np.hstack([img_xy, img_z]))
        t["head_pose"] = time.perf_counter() - s
        t["total"] = time.perf_counter() - start
        return {**result, "yaw": pose["yaw"], "pitch": pose["pitch"], "roll": pose["roll"], "times": t}


def read_labels(path: Path) -> dict[int, tuple[float, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["frame"]): (float(r["t_s"]), r["label"]) for r in csv.DictReader(f)}


def segments_of(frames: list[dict]) -> list[dict]:
    segs = []
    for f in frames:
        if not segs or segs[-1]["label"] != f["label"]:
            segs.append({"label": f["label"], "frames": []})
        segs[-1]["frames"].append(f)
    return segs


def best_threshold(d: np.ndarray, positive: np.ndarray) -> tuple[float | None, float | None]:
    """Threshold t (predict positive when d < t) with the best balanced accuracy."""
    if positive.all() or not positive.any():
        return None, None
    values = np.unique(d)
    candidates = np.concatenate([[values[0] - 1e-6], (values[:-1] + values[1:]) / 2, [values[-1] + 1e-6]])
    best = (None, -1.0)
    for c in candidates:
        pred = d < c
        bacc = 0.5 * (pred[positive].mean() + (~pred[~positive]).mean())
        if bacc > best[1]:
            best = (float(c), float(bacc))
    return best


def evaluate(d: np.ndarray, positive: np.ndarray, threshold: float) -> dict:
    pred = d < threshold
    tp, tn = int((pred & positive).sum()), int((~pred & ~positive).sum())
    fp, fn = int((pred & ~positive).sum()), int((~pred & positive).sum())
    n = len(d)
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return {
        "frames": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / n if n else None,
        "accuracy_formula": "Derived: (tp + tn) / frames",
        "balanced_accuracy": 0.5 * (tpr + tnr) if tpr is not None and tnr is not None else None,
        "balanced_accuracy_formula": "Derived: (tp / (tp + fn) + tn / (tn + fp)) / 2",
    }


def overlap(a: np.ndarray, b: np.ndarray) -> dict:
    ia, ib = np.percentile(a, [5, 95]), np.percentile(b, [5, 95])
    best = 0.5
    for sign in (1, -1):
        t, bacc = best_threshold(sign * np.concatenate([a, b]), np.r_[np.ones(len(a), bool), np.zeros(len(b), bool)])
        if bacc is not None:
            best = max(best, bacc)
    return {"camera_p5_p95": ia.round(2).tolist(), "screen_p5_p95": ib.round(2).tolist(),
            "intervals_overlap": bool(ia[0] <= ib[1] and ib[0] <= ia[1]),
            "best_single_threshold_balanced_accuracy": round(best, 4),
            "formula": "Derived: best balanced accuracy of one threshold on this axis, CAMERA vs SCREEN frames"}


def plot(frames: list[dict], segs: list[dict], calib_end: float, threshold: float | None, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"CAMERA": "#4c9f70", "SCREEN": "#4a7bd0", "AWAY_LEFT": "#d0874a", "DOWN": "#b04ab0"}
    valid = [f for f in frames if "yaw_c" in f]
    t = np.array([f["t"] for f in valid])
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    series = [("yaw_c", "yaw (deg)"), ("pitch_c", "pitch (deg)"), ("dist", "distance from zero (deg)")]
    for ax, (key, name) in zip(axes, series):
        for seg in segs:
            ax.axvspan(seg["frames"][0]["t"], seg["frames"][-1]["t"], color=colors.get(seg["label"], "#999999"),
                       alpha=0.15, lw=0)
        ax.plot(t, [f[key] for f in valid], lw=1, color="#222222")
        ax.set_ylabel(name)
        ax.axvline(calib_end, color="#666666", ls=":", lw=1)
        ax.grid(alpha=0.3)
    if threshold is not None:
        axes[2].axhline(threshold, color="#c03030", ls="--", lw=1, label=f"threshold {threshold:.1f} deg")
        axes[2].legend(loc="upper right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in colors.values()]
    axes[0].legend(handles, list(colors), loc="upper right", ncol=4, fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time (s), dotted line = end of calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--calib-s", type=float, default=3.0)
    ap.add_argument("--split-s", type=float, default=30.0)
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    import cv2  # eval only, used only to read the recorded video file

    video, out_dir = Path(args.video), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = video.stem
    labels = read_labels(Path(args.labels))
    pipe = Pipeline(Path(args.models_dir))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    frames, idx = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx in labels:
            t_s, label = labels[idx]
            r = pipe(np.ascontiguousarray(bgr[..., ::-1]))
            frames.append({"frame": idx, "t": t_s, "label": label, **r})
        idx += 1
    cap.release()
    if not frames:
        raise SystemExit("No labeled frames were read")

    lm_counts = sorted({f["landmark_count"] for f in frames if "landmark_count" in f})
    valid = [f for f in frames if "yaw" in f]
    cam_start = next(f["t"] for f in frames if f["label"] == "CAMERA")
    calib = [f for f in valid if f["label"] == "CAMERA" and f["t"] < cam_start + args.calib_s]
    if not calib:
        raise SystemExit("No CAMERA frames with a head pose in the calibration window")
    zero = {"yaw": float(np.mean([f["yaw"] for f in calib])), "pitch": float(np.mean([f["pitch"] for f in calib]))}
    calib_ids = {f["frame"] for f in calib}
    for f in valid:
        f["yaw_c"], f["pitch_c"] = f["yaw"] - zero["yaw"], f["pitch"] - zero["pitch"]
        f["dist"] = float(np.hypot(f["yaw_c"], f["pitch_c"]))

    segs = segments_of(frames)
    seg_stats = []
    for seg in segs:
        v = [f for f in seg["frames"] if "yaw_c" in f]
        y, p = np.array([f["yaw_c"] for f in v]), np.array([f["pitch_c"] for f in v])
        seg_stats.append({
            "label": seg["label"], "start_s": seg["frames"][0]["t"], "end_s": seg["frames"][-1]["t"],
            "frames": len(seg["frames"]), "frames_with_pose": len(v),
            "yaw_mean": float(y.mean()) if len(v) else None, "yaw_std": float(y.std()) if len(v) else None,
            "pitch_mean": float(p.mean()) if len(v) else None, "pitch_std": float(p.std()) if len(v) else None,
        })

    train = [f for f in valid if f["t"] < args.split_s and f["frame"] not in calib_ids]
    test = [f for f in valid if f["t"] >= args.split_s]
    arr = lambda fs, k: np.array([f[k] for f in fs])  # noqa: E731
    pos = lambda fs: np.array([f["label"] == "CAMERA" for f in fs], dtype=bool)  # noqa: E731
    threshold, train_bacc = best_threshold(arr(train, "dist"), pos(train)) if train else (None, None)
    classifier = {
        "definition": "facing = CAMERA, away = every other label. Predict facing when the calibrated "
                      "angular distance sqrt(yaw^2 + pitch^2) is below the threshold",
        "train": f"frames with t < {args.split_s} s, calibration frames excluded",
        "test": f"frames with t >= {args.split_s} s",
        "train_labels": sorted({f["label"] for f in train}), "test_labels": sorted({f["label"] for f in test}),
        "threshold_deg": threshold, "train_balanced_accuracy": train_bacc,
        "train_eval": evaluate(arr(train, "dist"), pos(train), threshold) if threshold is not None else None,
        "test_eval": evaluate(arr(test, "dist"), pos(test), threshold) if threshold is not None and test else None,
    }
    cam = [f for f in valid if f["label"] == "CAMERA"]
    scr = [f for f in valid if f["label"] == "SCREEN"]
    cam_vs_screen = None
    if cam and scr:
        cam_vs_screen = {"yaw": overlap(arr(cam, "yaw_c"), arr(scr, "yaw_c")),
                         "pitch": overlap(arr(cam, "pitch_c"), arr(scr, "pitch_c"))}
        cam_vs_screen["overlap_in_yaw_and_pitch"] = (cam_vs_screen["yaw"]["intervals_overlap"]
                                                     and cam_vs_screen["pitch"]["intervals_overlap"])

    # Per-frame stage times: raw samples plus Measurement records (p50, p95, min, max).
    import onnxruntime

    samples = {s: [f["times"][s] * 1000 for f in frames if s in f["times"]] for s in STAGES}
    sample_path = out_dir / f"{tag}_frame_times.json"
    sample_path.write_text(json.dumps({"video": video.name, "unit": "ms", "source": Source.LOCAL_X86_CPU.value,
                                       "samples": samples}, indent=1) + "\n", encoding="utf-8")
    rel = sample_path.relative_to(REPO).as_posix() if sample_path.is_relative_to(REPO) else sample_path.as_posix()
    stage_meta = {
        "letterbox": ("app.vision.roi letterbox warp 640x480 to 256x256", "numpy"),
        "detector": ("mediapipe_face_float_face_detector", "onnx"),
        "decode_roi": ("anchor decode, NMS and ROI geometry", "numpy"),
        "roi_warp": ("app.vision.roi.warp_affine_bilinear 192x192 ROI crop", "numpy"),
        "landmark": ("mediapipe_face_float_face_landmark_detector", "onnx"),
        "head_pose": ("app.vision.head_pose landmark mapping and Kabsch", "numpy"),
        "total": ("face pipeline, detector on every frame", "onnx+numpy"),
    }
    measurements = []
    for stage, values in samples.items():
        if not values:
            continue
        model, runtime = stage_meta[stage]
        a = np.asarray(values)
        base = (f"video {video.name}, {len(a)} frames, ONNX Runtime {onnxruntime.__version__} providers "
                f"{pipe.providers['detector']}, CPU {platform.processor()}, per-frame times in {rel}")
        for metric, value, how in (
            ("frame_time_p50", np.percentile(a, 50, method="linear"), "numpy.percentile 50, method linear"),
            ("frame_time_p95", np.percentile(a, 95, method="linear"), "numpy.percentile 95, method linear"),
            ("frame_time_min", a.min(), "min of the per-frame times"),
            ("frame_time_max", a.max(), "max of the per-frame times"),
        ):
            measurements.append(Measurement(
                model=model, metric=f"{stage}_{metric}", value=float(value), unit="ms", source=Source.LOCAL_X86_CPU,
                runtime=runtime, compute_unit="CPU", precision="float32", notes=f"{base} | {how}"))
    save_json(measurements, out_dir / f"{tag}_timings.json")

    plot(frames, segs, cam_start + args.calib_s, threshold, out_dir / f"{tag}_yaw_pitch.png",
         f"{video.name}: calibrated head pose (Source: local x86 CPU)")
    results = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": Source.LOCAL_X86_CPU.value, "video": video.name, "labels": Path(args.labels).name,
        "frames_labeled": len(frames), "frames_with_face": sum(f["detected"] for f in frames),
        "frames_with_pose": len(valid),
        "landmark_count": lm_counts, "iris_points": any(c >= 478 for c in lm_counts),
        "landmark_output": "landmarks [1, 468, 3] = (x, y, z) divided by 192. z confirmed from MediaPipePyTorch "
                           "blazeface_landmark.py (view(-1, 468, 3) / 192, confidence is the separate flag output)",
        "providers": pipe.providers, "calibration": {"window_s": [cam_start, cam_start + args.calib_s],
                                                    "frames": len(calib), "zero_deg": zero},
        "segments": seg_stats, "classifier": classifier, "camera_vs_screen": cam_vs_screen,
        "angle_convention": "yaw > 0 nose toward image right, pitch > 0 nose down (app.vision.head_pose)",
        "timings_file": f"{tag}_timings.json", "frame_times_file": sample_path.name,
    }
    (out_dir / f"{tag}_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Source: local x86 CPU. Video {video.name}, {len(frames)} labeled frames, "
          f"{results['frames_with_face']} with a face, {len(valid)} with a head pose")
    print(f"Landmark count: {lm_counts}. Iris points: {'yes' if results['iris_points'] else 'no'}")
    print(f"Calibration zero over {len(calib)} CAMERA frames: yaw {zero['yaw']:.2f}, pitch {zero['pitch']:.2f} deg")
    print("\n| segment | start s | end s | frames | yaw mean | yaw std | pitch mean | pitch std |")
    print("|---|---|---|---|---|---|---|---|")
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
    for s in seg_stats:
        print(f"| {s['label']} | {s['start_s']:.1f} | {s['end_s']:.1f} | {s['frames_with_pose']}/{s['frames']} | "
              f"{fmt(s['yaw_mean'])} | {fmt(s['yaw_std'])} | {fmt(s['pitch_mean'])} | {fmt(s['pitch_std'])} |")
    te = classifier["test_eval"]
    print(f"\nFacing vs away threshold: {fmt(threshold)} deg, chosen on {classifier['train_labels']}")
    if te:
        print(f"Held-out ({classifier['test_labels']}): accuracy {fmt(te['accuracy'])}, balanced accuracy "
              f"{fmt(te['balanced_accuracy'])} (Derived, formulas in results JSON)")
    else:
        print("Held-out accuracy: not available (no threshold or no test frames)")
    if cam_vs_screen:
        for axis in ("yaw", "pitch"):
            o = cam_vs_screen[axis]
            print(f"CAMERA vs SCREEN {axis}: p5-p95 CAMERA {o['camera_p5_p95']} SCREEN {o['screen_p5_p95']}, "
                  f"overlap {o['intervals_overlap']}, best one-threshold balanced accuracy "
                  f"{o['best_single_threshold_balanced_accuracy']}")
        print(f"CAMERA and SCREEN overlap in both yaw and pitch: {cam_vs_screen['overlap_in_yaw_and_pitch']}")
    roi = [m for m in measurements if m.metric.startswith("roi_warp_frame_time_p50")]
    if roi:
        print(f"ROI warp p50 per frame: {roi[0].value:.3f} ms (Source: local x86 CPU)")
    print(f"\nWrote results, timings, frame times and plot to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
