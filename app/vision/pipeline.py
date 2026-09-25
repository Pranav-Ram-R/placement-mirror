"""Vision pipeline for one session: face landmarks and head pose on every frame, pose on
every other frame.

Frames come from the browser over the session WebSocket (see parse_frame for the binary
format). Each frame is stamped with time.monotonic() on arrival and put in a single slot
mailbox. The arrival time drives calibration windows and fps. Durations use
time.perf_counter(), also stamped on arrival, because time.monotonic() on Windows is
GetTickCount64 with 15.625 ms resolution. A worker thread takes the latest frame and processes it. A frame that arrives
while an older one is still waiting replaces it (a server side drop) and the pipeline
tells the browser it is busy, so the browser drops frames until the worker catches up.

Face: the face detector runs only when there is no track. Its input is the letterboxed
256x256 image the browser makes in a canvas and sends while the server asks for it. With
a track, the ROI comes from the previous frame's landmarks. A landmark score below
config.face_track_min_score ends the track. Head pose is Kabsch on 6 landmarks
(app.vision.head_pose). Facing = |yaw| and |pitch| relative to calibration under the
config limits (provisional until Task C results).

Pose: on every other processed frame, pose detector on the browser's 128x128 letterbox,
ROI crop, pose landmarks. Posture score from shoulder tilt plus head drop relative to
calibration (config, provisional).

Calibration: the first config.calibration_s seconds after the user presses Calibrate.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.config import VISION, VisionConfig
from app.vision.detection import FACE_ANCHORS, POSE_ANCHORS, best_detection
from app.vision.head_pose import MESH_IDS, head_pose
from app.vision.preprocess import image_tensor, input_layout
from app.vision.roi import (
    affine_scale,
    apply_affine,
    crop_affine,
    invert_affine,
    keypoint_roi_corners,
    landmarks_roi_corners,
    letterbox_affine,
    pad_border,
    roi_corners,
    warp_padded,
)

VISION_MODELS = ("face_detector", "face_landmark", "pose_detector", "pose_landmark")

# Binary frame message, little endian:
#   magic b"PMF1", frame_id uint32, capture_ms float64 (browser performance.now()),
#   width uint16, height uint16, face_input_size uint16 (0 = absent),
#   pose_input_size uint16 (0 = absent)
# then width*height*3 bytes RGB, then the face detector input (size*size*3 RGB) if
# present, then the pose detector input if present.
HEADER = struct.Struct("<4sIdHHHH")
MAGIC = b"PMF1"

FACE_KP_START, FACE_KP_END = 1, 0  # detector eye keypoints, image right eye to image left eye
POSE_KP_CENTER, POSE_KP_END = 2, 3  # qai_hub_models mediapipe_pose POSE_KEYPOINT_INDEX_START / END
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 11, 12  # BlazePose landmark indices
FACE_LM_SIZE, POSE_LM_SIZE = 192, 256


@dataclass
class Frame:
    frame_id: int
    capture_ms: float
    arrival: float  # time.monotonic() on arrival
    rgb: np.ndarray
    arrival_perf: float = 0.0  # time.perf_counter() on arrival, for durations
    face_input: np.ndarray | None = None
    pose_input: np.ndarray | None = None


def parse_frame(data: bytes, arrival: float, arrival_perf: float = 0.0) -> Frame:
    if len(data) < HEADER.size:
        raise ValueError(f"frame message of {len(data)} bytes is shorter than the header")
    magic, frame_id, capture_ms, w, h, face_s, pose_s = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"bad frame magic {magic!r}")
    sizes = [w * h * 3, face_s * face_s * 3, pose_s * pose_s * 3]
    if len(data) != HEADER.size + sum(sizes):
        raise ValueError(f"frame message is {len(data)} bytes, header says {HEADER.size + sum(sizes)}")
    buf = np.frombuffer(data, dtype=np.uint8, offset=HEADER.size)
    rgb = buf[:sizes[0]].reshape(h, w, 3)
    face = buf[sizes[0]:sizes[0] + sizes[1]].reshape(face_s, face_s, 3) if face_s else None
    pose = buf[sizes[0] + sizes[1]:].reshape(pose_s, pose_s, 3) if pose_s else None
    return Frame(frame_id, capture_ms, arrival, rgb, arrival_perf, face, pose)


class Mailbox:
    """Single slot. put() replaces an unread item and reports that it did."""

    def __init__(self):
        self._cv = threading.Condition()
        self._item: Any = None
        self._closed = False
        self.replaced = 0

    def put(self, item) -> bool:
        with self._cv:
            replaced = self._item is not None
            self.replaced += replaced
            self._item = item
            self._cv.notify()
            return replaced

    def get(self, timeout: float | None = None):
        with self._cv:
            if self._item is None and not self._closed:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item

    def empty(self) -> bool:
        with self._cv:
            return self._item is None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class StageTimes:
    """Per stage durations in seconds, from time.perf_counter()."""

    def __init__(self):
        self.samples: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def add(self, stage: str, seconds: float) -> None:
        with self._lock:
            self.samples.setdefault(stage, []).append(seconds)

    def summary(self) -> dict[str, dict]:
        with self._lock:
            out = {}
            for stage, values in self.samples.items():
                arr = np.asarray(values) * 1000
                out[stage] = {"count": len(values), "p50_ms": float(np.percentile(arr, 50)),
                              "p95_ms": float(np.percentile(arr, 95))}
            return out

    def reset(self) -> None:
        with self._lock:
            self.samples = {}


def posture_metrics(nose: np.ndarray, left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> dict:
    """Shoulder line angle (degrees) and head height ratio from image pixel points.

    head_ratio = (shoulder midpoint y - nose y) / shoulder width. It falls when the head
    drops or moves forward toward the shoulders.
    """
    d = np.asarray(left_shoulder, float) - np.asarray(right_shoulder, float)
    width = float(np.hypot(d[0], d[1]))
    mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
    return {"tilt_deg": float(np.degrees(np.arctan2(d[1], d[0]))),
            "head_ratio": float((mid_y - nose[1]) / width) if width > 0 else float("nan"),
            "shoulder_width_px": width}


def posture_score(metrics: dict, baseline: dict, cfg: VisionConfig) -> dict:
    tilt = abs((metrics["tilt_deg"] - baseline["tilt_deg"] + 180.0) % 360.0 - 180.0)
    drop = baseline["head_ratio"] - metrics["head_ratio"]
    score = tilt / cfg.posture_max_tilt_deg + max(0.0, drop) / cfg.posture_max_head_drop
    return {"tilt_change_deg": tilt, "head_drop": drop, "score": score,
            "posture": "ok" if score < cfg.posture_max_score else "check"}


class Calibration:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.state = "not calibrated"
        self.window: tuple[float, float] | None = None
        self.face_samples: list[tuple[float, float]] = []
        self.pose_samples: list[tuple[float, float]] = []
        self.face: dict | None = None
        self.pose: dict | None = None
        self.message = ""

    def start(self, now: float) -> None:
        self.window = (now, now + self.cfg.calibration_s)
        self.face_samples, self.pose_samples = [], []
        self.state, self.message = "running", ""

    def in_window(self, t: float) -> bool:
        return self.state == "running" and self.window is not None and self.window[0] <= t <= self.window[1]

    def add_face(self, t: float, yaw: float, pitch: float) -> None:
        if self.in_window(t):
            self.face_samples.append((yaw, pitch))

    def add_pose(self, t: float, metrics: dict) -> None:
        if self.in_window(t) and np.isfinite(metrics["head_ratio"]):
            self.pose_samples.append((metrics["tilt_deg"], metrics["head_ratio"]))

    def update(self, t: float) -> bool:
        """Finish the calibration once a frame arrives after the window. True when it finished."""
        if self.state != "running" or t <= self.window[1]:
            return False
        cfg, notes = self.cfg, []
        if len(self.face_samples) >= cfg.calibration_min_face_frames:
            yaw, pitch = np.median(np.asarray(self.face_samples), axis=0)
            self.face = {"yaw": float(yaw), "pitch": float(pitch), "frames": len(self.face_samples)}
        else:
            notes.append(f"face seen in {len(self.face_samples)} frames, need {cfg.calibration_min_face_frames}")
        if len(self.pose_samples) >= cfg.calibration_min_pose_frames:
            tilt, ratio = np.median(np.asarray(self.pose_samples), axis=0)
            self.pose = {"tilt_deg": float(tilt), "head_ratio": float(ratio), "frames": len(self.pose_samples)}
        else:
            notes.append(f"shoulders seen in {len(self.pose_samples)} frames, need {cfg.calibration_min_pose_frames}")
        self.state = "done" if not notes else ("partial" if (self.face or self.pose) else "failed")
        self.message = ". ".join(notes)
        return True

    def as_dict(self) -> dict:
        return {"state": self.state, "face": self.face is not None, "pose": self.pose is not None,
                "message": self.message}


@dataclass
class FaceTrack:
    roi: np.ndarray | None = None
    lost_reason: str = "no track yet"


@dataclass
class SessionCounters:
    received: int = 0
    processed: int = 0
    bad_messages: int = 0
    face_detector_runs: int = 0
    started: float = field(default_factory=time.monotonic)


class VisionPipeline:
    def __init__(self, runner, emit: Callable[[dict], None], cfg: VisionConfig = VISION):
        self.runner = runner
        self.emit = emit
        self.cfg = cfg
        self.layouts = {}
        self.sizes = {}
        for name in VISION_MODELS:
            shape = next(s for n, s, _ in runner.input_specs(name) if n == "image")
            self.layouts[name] = input_layout(shape)
            self.sizes[name] = int(shape[2] if self.layouts[name] == "NCHW" else shape[1])
        if (self.sizes["face_detector"], self.sizes["pose_detector"]) != (cfg.face_detector_size, cfg.pose_detector_size):
            raise ValueError(f"detector input sizes {self.sizes} do not match the config")
        self.mailbox = Mailbox()
        self.times = StageTimes()
        self.counters = SessionCounters()
        self.calibration = Calibration(cfg)
        self.track = FaceTrack()
        self.last_pose: dict | None = None
        self.want = {"face": True, "pose": True}
        self._busy = False
        self._busy_lock = threading.Lock()
        self._done_times: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ session control

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="vision-worker", daemon=True)
        self._thread.start()
        self.emit({"type": "detector_inputs", **self.want})

    def stop(self) -> None:
        self._running = False
        self.mailbox.close()
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, data: bytes, arrival: float, arrival_perf: float | None = None) -> None:
        try:
            frame = parse_frame(data, arrival, time.perf_counter() if arrival_perf is None else arrival_perf)
        except ValueError as e:
            self.counters.bad_messages += 1
            self.emit({"type": "error", "message": str(e)})
            return
        self.counters.received += 1
        if self.mailbox.put(frame):
            self._set_busy(True)

    def calibrate(self, now: float) -> None:
        self.calibration.start(now)
        self.emit({"type": "calibration", **self.calibration.as_dict()})

    def _set_busy(self, busy: bool) -> None:
        with self._busy_lock:
            if busy == self._busy:
                return
            self._busy = busy
        self.emit({"type": "busy", "busy": busy})

    def _loop(self) -> None:
        while self._running:
            frame = self.mailbox.get(timeout=0.5)
            if frame is None:
                continue
            try:
                event = self.process(frame)
            except Exception as e:  # noqa: BLE001  keep the session alive, report the error
                self.emit({"type": "error", "message": f"frame {frame.frame_id}: {type(e).__name__}: {e}"})
                continue
            self.emit(event)
            if self.mailbox.empty():
                self._set_busy(False)

    # ------------------------------------------------------------------ per frame

    def process(self, frame: Frame) -> dict:
        t_start = time.perf_counter()
        self.times.add("queue_wait", t_start - frame.arrival_perf)
        calibration_finished = self.calibration.update(frame.arrival)
        padded: list[np.ndarray] = []

        def get_padded() -> np.ndarray:
            if not padded:
                padded.append(pad_border(frame.rgb))
            return padded[0]

        face = self._face(frame, get_padded)
        pose_frame = self.counters.processed % self.cfg.pose_every_n_frames == 0
        if pose_frame:
            t0 = time.perf_counter()
            self.last_pose = self._pose(frame, get_padded)
            self.times.add("pose", time.perf_counter() - t0)
        self.counters.processed += 1
        want = {"face": self.track.roi is None, "pose": True}
        if want != self.want:
            self.want = want
            self.emit({"type": "detector_inputs", **want})
        if calibration_finished:
            self.emit({"type": "calibration", **self.calibration.as_dict()})

        done = time.monotonic()
        self._done_times = [t for t in self._done_times if t > done - 1.0] + [done]
        self.times.add("total", time.perf_counter() - t_start)
        return {
            "type": "indicators", "frame_id": frame.frame_id, "capture_ms": frame.capture_ms,
            "server_latency_ms": (time.perf_counter() - frame.arrival_perf) * 1000, "processed_fps": len(self._done_times),
            "face": face, "pose": self._pose_indicator(), "pose_updated": pose_frame,
            "calibration": self.calibration.as_dict(),
        }

    def _face(self, frame: Frame, get_padded) -> dict:
        cfg = self.cfg
        roi = self.track.roi
        detected = False
        if roi is None:
            if frame.face_input is None:
                return {"state": "waiting for detector input", "facing": None}
            t0 = time.perf_counter()
            size = self.sizes["face_detector"]
            out, _ = self.runner.run("face_detector", {"image": image_tensor(frame.face_input, self.layouts["face_detector"])})
            det = best_detection(out, FACE_ANCHORS, size, cfg.face_detector_min_score, cfg.face_nms_iou)
            self.times.add("face_detector", time.perf_counter() - t0)
            self.counters.face_detector_runs += 1
            if det is None:
                self.track.lost_reason = "no face detected"
                return {"state": "no face", "facing": None}
            h, w = frame.rgb.shape[:2]
            _, scale, (pad_l, pad_t) = letterbox_affine(w, h, size, size)
            box = ((det.box.reshape(2, 2) - [pad_l, pad_t]) / scale).reshape(4)
            kp = (det.keypoints - [pad_l, pad_t]) / scale
            roi = roi_corners(box, kp[FACE_KP_START], kp[FACE_KP_END], scale=cfg.face_box_scale)
            detected = True

        t0 = time.perf_counter()
        m = crop_affine(roi, FACE_LM_SIZE, FACE_LM_SIZE)
        crop = warp_padded(get_padded(), m, FACE_LM_SIZE, FACE_LM_SIZE)
        self.times.add("roi_warp", time.perf_counter() - t0)

        t0 = time.perf_counter()
        out, _ = self.runner.run("face_landmark", {"image": image_tensor(crop, self.layouts["face_landmark"])})
        self.times.add("face_landmark", time.perf_counter() - t0)
        score = float(np.asarray(out["scores"]).reshape(-1)[0])
        if score < cfg.face_track_min_score:
            self.track = FaceTrack(None, f"landmark score {score:.2f} below {cfg.face_track_min_score}")
            return {"state": "lost", "facing": None, "landmark_score": score}

        t0 = time.perf_counter()
        inv = invert_affine(m)
        pts = np.asarray(out["landmarks"])[0].astype(np.float64)
        xy = apply_affine(inv, pts[:, :2] * FACE_LM_SIZE)
        z = pts[:, 2:3] * FACE_LM_SIZE * affine_scale(inv)
        pose = head_pose(np.hstack([xy, z]))
        self.times.add("head_pose", time.perf_counter() - t0)
        self.track = FaceTrack(landmarks_roi_corners(xy, MESH_IDS["eye_outer_image_right"],
                                                     MESH_IDS["eye_outer_image_left"], cfg.face_track_roi_scale), "")

        self.calibration.add_face(frame.arrival, pose["yaw"], pose["pitch"])
        result = {"state": "detected" if detected else "tracking", "landmark_score": score,
                  "yaw": pose["yaw"], "pitch": pose["pitch"], "roll": pose["roll"], "facing": None}
        base = self.calibration.face
        if base is not None:
            d_yaw, d_pitch = pose["yaw"] - base["yaw"], pose["pitch"] - base["pitch"]
            result.update(yaw_change=d_yaw, pitch_change=d_pitch,
                          facing=bool(abs(d_yaw) < cfg.facing_max_yaw_deg and abs(d_pitch) < cfg.facing_max_pitch_deg))
        return result

    def _pose(self, frame: Frame, get_padded) -> dict:
        cfg = self.cfg
        if frame.pose_input is None:
            return {"state": "waiting for detector input"}
        t0 = time.perf_counter()
        size = self.sizes["pose_detector"]
        out, _ = self.runner.run("pose_detector", {"image": image_tensor(frame.pose_input, self.layouts["pose_detector"])})
        det = best_detection(out, POSE_ANCHORS, size, cfg.pose_detector_min_score, cfg.pose_nms_iou)
        self.times.add("pose_detector", time.perf_counter() - t0)
        if det is None:
            return {"state": "no person"}
        h, w = frame.rgb.shape[:2]
        _, scale, (pad_l, pad_t) = letterbox_affine(w, h, size, size)
        kp = (det.keypoints - [pad_l, pad_t]) / scale
        roi = keypoint_roi_corners(kp[POSE_KP_CENTER], kp[POSE_KP_END], cfg.pose_roi_scale, np.pi / 2)

        t0 = time.perf_counter()
        m = crop_affine(roi, POSE_LM_SIZE, POSE_LM_SIZE)
        crop = warp_padded(get_padded(), m, POSE_LM_SIZE, POSE_LM_SIZE)
        self.times.add("pose_warp", time.perf_counter() - t0)

        t0 = time.perf_counter()
        out, _ = self.runner.run("pose_landmark", {"image": image_tensor(crop, self.layouts["pose_landmark"])})
        self.times.add("pose_landmark", time.perf_counter() - t0)
        score = float(np.asarray(out["scores"]).reshape(-1)[0])
        if score < cfg.pose_min_score:
            return {"state": "no person", "landmark_score": score}
        lm = np.asarray(out["landmarks"])[0].astype(np.float64)  # (25, 4): x, y, z, visibility logit, all / 256
        xy = apply_affine(invert_affine(m), lm[:, :2] * POSE_LM_SIZE)
        visibility = 1.0 / (1.0 + np.exp(-np.clip(lm[:, 3] * POSE_LM_SIZE, -50, 50)))
        needed = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]
        if (visibility[needed] < cfg.pose_min_visibility).any():
            return {"state": "shoulders not visible", "landmark_score": score,
                    "visibility": [float(v) for v in visibility[needed]]}
        metrics = posture_metrics(xy[NOSE], xy[LEFT_SHOULDER], xy[RIGHT_SHOULDER])
        self.calibration.add_pose(frame.arrival, metrics)
        return {"state": "tracking", "landmark_score": score, **metrics,
                "points": {"nose": xy[NOSE].tolist(), "left_shoulder": xy[LEFT_SHOULDER].tolist(),
                           "right_shoulder": xy[RIGHT_SHOULDER].tolist()}}

    def _pose_indicator(self) -> dict | None:
        p = self.last_pose
        if p is None:
            return None
        out = {k: v for k, v in p.items() if k != "points"}
        out["posture"] = None
        base = self.calibration.pose
        if base is not None and p.get("state") == "tracking":
            out.update(posture_score(p, base, self.cfg))
        return out

    # ------------------------------------------------------------------ report

    def report(self) -> dict:
        c = self.counters
        return {"duration_s": time.monotonic() - c.started, "frames_received": c.received,
                "frames_processed": c.processed, "frames_replaced_in_mailbox": self.mailbox.replaced,
                "bad_messages": c.bad_messages, "face_detector_runs": c.face_detector_runs,
                "stages": self.times.summary()}
