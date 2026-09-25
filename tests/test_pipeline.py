import struct
import threading

import numpy as np
import pytest

from app.config import VISION
from app.vision.head_pose import CANONICAL, MESH_IDS, to_model_frame
from app.vision.pipeline import HEADER, Mailbox, VisionPipeline, parse_frame, posture_metrics

W, H = 640, 480


def encode_frame(frame_id, rgb, face=None, pose=None, capture_ms=0.0):
    """Same layout as app.js sendFrame."""
    fs = face.shape[0] if face is not None else 0
    ps = pose.shape[0] if pose is not None else 0
    parts = [HEADER.pack(b"PMF1", frame_id, capture_ms, rgb.shape[1], rgb.shape[0], fs, ps), rgb.tobytes()]
    parts += [a.tobytes() for a in (face, pose) if a is not None]
    return b"".join(parts)


def rotation(yaw, pitch):
    y, p = np.radians([yaw, pitch])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return ry @ rx


def detector_outputs(n_coords, index, values):
    coords = np.zeros((896, n_coords), np.float32)
    scores = np.full((896, 1), -20.0, np.float32)
    coords[index, :len(values)] = values
    scores[index] = 5.0
    return {"box_coords_1": coords[None, :512], "box_coords_2": coords[None, 512:],
            "box_scores_1": scores[None, :512], "box_scores_2": scores[None, 512:]}


class FakeRunner:
    """Planted detections and synthetic landmarks. Layouts as the onnx (NCHW) models."""

    SHAPES = {"face_detector": [1, 3, 256, 256], "face_landmark": [1, 3, 192, 192],
              "pose_detector": [1, 3, 128, 128], "pose_landmark": [1, 3, 256, 256]}

    def __init__(self, unit="NPU"):
        self.unit = unit
        self.yaw = 0.0
        self.pitch = 0.0
        self.face_score = 1.0
        self.nose_y = 0.30  # pose landmark nose y in the crop, normalized
        self.calls = []

    def status(self):
        return {name: {"compute_unit": self.unit} for name in self.SHAPES}

    def input_specs(self, name):
        return [("image", self.SHAPES[name], "tensor(float)")]

    def run(self, name, inputs):
        assert inputs["image"].shape == tuple(self.SHAPES[name])
        self.calls.append(name)
        if name == "face_detector":
            # 60x60 px box on the anchor at index 200, eye keypoints 12 px left and right of center
            return detector_outputs(16, 200, [0, 0, 60, 60, 12, 0, -12, 0]), 0.0
        if name == "face_landmark":
            pts = np.tile([0.5, 0.5, 0.0], (468, 1))
            ring = np.linspace(0, 2 * np.pi, 468, endpoint=False)
            pts[:, 0] += 0.3 * np.cos(ring)
            pts[:, 1] += 0.35 * np.sin(ring)
            face = to_model_frame(CANONICAL @ rotation(self.yaw, self.pitch).T) * (0.25 / 225) + [0.5, 0.5, 0]
            pts[list(MESH_IDS.values())] = face
            return {"scores": np.array([self.face_score], np.float32),
                    "landmarks": pts[None].astype(np.float32)}, 0.0
        if name == "pose_detector":
            # center keypoint (index 2) on the anchor, end keypoint (index 3) 20 px above it
            return detector_outputs(12, 300, [0, 0, 40, 40, 0, 0, 0, 0, 0, 0, 0, -20]), 0.0
        if name == "pose_landmark":
            lm = np.zeros((25, 4), np.float32)
            lm[:, 3] = 10 / 256  # visibility logit 10, stored / 256 like the model output
            lm[0, :2] = [0.5, self.nose_y]
            lm[11, :2] = [0.65, 0.5]
            lm[12, :2] = [0.35, 0.5]
            return {"scores": np.array([1.0], np.float32), "landmarks": lm[None]}, 0.0
        raise KeyError(name)


def make(runner=None):
    events = []
    pipe = VisionPipeline(runner or FakeRunner(), events.append)
    return pipe, events


RGB = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
FACE_IN = np.zeros((256, 256, 3), np.uint8)
POSE_IN = np.zeros((128, 128, 3), np.uint8)


def frame(i, t, face=True, pose=True):
    return parse_frame(encode_frame(i, RGB, FACE_IN if face else None, POSE_IN if pose else None), t, t)


def test_parse_frame_roundtrip_and_errors():
    data = encode_frame(7, RGB, FACE_IN, None, capture_ms=12.5)
    f = parse_frame(data, 3.0)
    assert (f.frame_id, f.capture_ms, f.arrival) == (7, 12.5, 3.0)
    assert np.array_equal(f.rgb, RGB) and f.face_input.shape == (256, 256, 3) and f.pose_input is None
    with pytest.raises(ValueError):
        parse_frame(data[:-1], 0.0)
    with pytest.raises(ValueError):
        parse_frame(b"XXXX" + data[4:], 0.0)
    with pytest.raises(ValueError):
        parse_frame(b"PM", 0.0)


def test_mailbox_keeps_only_the_latest_frame():
    box = Mailbox()
    assert box.put(1) is False
    assert box.put(2) is True and box.replaced == 1
    assert box.get(0) == 2 and box.get(0) is None
    got = []
    t = threading.Thread(target=lambda: got.append(box.get(2.0)))
    t.start()
    box.put(3)
    t.join()
    assert got == [3]


def test_face_detector_runs_only_without_a_track_and_is_requested_again_after_loss():
    runner = FakeRunner()
    pipe, events = make(runner)
    ev = pipe.process(frame(0, 0.0, face=False))
    assert ev["face"]["state"] == "waiting for detector input" and "face_detector" not in runner.calls
    ev = pipe.process(frame(1, 0.03))
    assert ev["face"]["state"] == "detected" and runner.calls.count("face_detector") == 1
    assert {"type": "detector_inputs", "face": False, "pose": True} in events
    ev = pipe.process(frame(2, 0.06, face=False))
    assert ev["face"]["state"] == "tracking" and runner.calls.count("face_detector") == 1
    runner.face_score = 0.2
    ev = pipe.process(frame(3, 0.09, face=False))
    assert ev["face"]["state"] == "lost" and events[-1] == {"type": "detector_inputs", "face": True, "pose": True}
    runner.face_score = 1.0
    ev = pipe.process(frame(4, 0.12))
    assert ev["face"]["state"] == "detected" and runner.calls.count("face_detector") == 2


def test_npu_mode_runs_pose_every_other_frame_at_30_fps():
    runner = FakeRunner("NPU")
    pipe, events = make(runner)
    pipe.start()
    pipe.stop()
    assert events[0] == {"type": "mode", "mode": "npu", "label": "NPU mode", "face_max_fps": 30.0, "pose_fps": 15.0}
    flags = [pipe.process(frame(i, i / 30))["pose_updated"] for i in range(6)]
    assert flags == [True, False, True, False, True, False]
    assert runner.calls.count("pose_landmark") == 3


def test_cpu_fallback_mode_limits_face_to_15_and_pose_to_5_fps():
    runner = FakeRunner("CPU")
    pipe, events = make(runner)
    assert pipe.mode_event() == {"type": "mode", "mode": "cpu_fallback", "label": "CPU fallback mode, reduced frame rate",
                                 "face_max_fps": 15.0, "pose_fps": 5.0}
    frames = [frame(i, i / 30) for i in range(60)]  # 2 s at 30 fps
    accepted = [f for f in frames if pipe.accept(f)]
    assert len(accepted) == 30 and pipe.counters.rate_limited == 30  # 15 fps
    flags = [pipe.process(f)["pose_updated"] for f in accepted]
    assert sum(flags) == 10  # 5 fps over 2 s


def test_cpu_fallback_mode_skips_pose_while_asr_transcribes():
    runner = FakeRunner("CPU")
    pipe, _ = make(runner)
    pipe.priority.busy.set()
    flags = [pipe.process(frame(i, i * 0.25))["pose_updated"] for i in range(4)]
    assert flags == [False] * 4 and pipe.counters.pose_skipped_for_asr == 4
    pipe.priority.busy.clear()
    assert pipe.process(frame(4, 1.0))["pose_updated"] is True
    npu, _ = make(FakeRunner("NPU"))
    npu.priority.busy.set()
    assert npu.process(frame(0, 0.0))["pose_updated"] is True  # NPU mode runs at full rate


def calibrate(pipe, runner):
    pipe.calibrate(1.0)
    for i in range(20):
        pipe.process(frame(i, 1.0 + i * 0.1))  # 1.0 to 2.9 s, inside the 3 s window
    ev = pipe.process(frame(20, 4.1))  # first frame after the window finishes calibration
    assert pipe.calibration.state == "done", pipe.calibration.message
    return ev


def test_facing_flips_when_the_head_turns_away_after_calibration():
    runner = FakeRunner()
    pipe, events = make(runner)
    ev = pipe.process(frame(0, 0.0))
    assert ev["face"]["facing"] is None  # not calibrated
    runner.yaw, runner.pitch = 4.0, -3.0  # calibration pose, not exactly zero
    ev = calibrate(pipe, runner)
    assert {"type": "calibration", "state": "done", "face": True, "pose": True, "message": ""} in events
    assert ev["face"]["facing"] is True
    runner.yaw = 4.0 + VISION.facing_max_yaw_deg + 10
    assert pipe.process(frame(21, 4.2))["face"]["facing"] is False
    runner.yaw, runner.pitch = 4.0, -3.0 + VISION.facing_max_pitch_deg + 8
    assert pipe.process(frame(22, 4.3))["face"]["facing"] is False
    runner.pitch = -3.0
    ev = pipe.process(frame(23, 4.4))
    assert ev["face"]["facing"] is True and abs(ev["face"]["yaw_change"]) < 1.0


def test_posture_flips_on_slouch_after_calibration():
    runner = FakeRunner()
    pipe, _ = make(runner)
    calibrate(pipe, runner)
    evs = [pipe.process(frame(i, 4.2 + (i - 21) * 0.033)) for i in (21, 22)]  # one of the two is a pose frame
    ev = next(e for e in evs if e["pose_updated"])
    assert ev["pose"]["posture"] == "ok"
    runner.nose_y = 0.42  # nose drops toward the shoulders
    for i in range(23, 25):
        ev = pipe.process(frame(i, 4.2 + (i - 22) * 0.033))
    assert ev["pose"]["posture"] == "check" and ev["pose"]["head_drop"] > VISION.posture_max_head_drop


def test_calibration_without_a_face_reports_why():
    runner = FakeRunner()
    pipe, _ = make(runner)
    pipe.calibrate(0.0)
    for i in range(5):
        pipe.process(frame(i, i * 0.5, face=False))  # no detector input, face never found
    pipe.process(frame(5, 3.5, face=False))
    assert pipe.calibration.state == "partial" and pipe.calibration.face is None
    assert "face seen in 0 frames" in pipe.calibration.message


def test_busy_is_signalled_when_a_frame_is_replaced_and_cleared_when_the_worker_catches_up():
    runner = FakeRunner()
    pipe, events = make(runner)
    gate = threading.Event()
    real_process = pipe.process

    def slow_process(f):
        gate.wait(2.0)
        return real_process(f)

    pipe.process = slow_process
    pipe.start()
    data = encode_frame(0, RGB, FACE_IN, POSE_IN)
    pipe.submit(data, 0.0, 0.0)
    for _ in range(100):  # wait until the worker holds frame 0
        if pipe.mailbox.empty():
            break
        threading.Event().wait(0.01)
    pipe.submit(data, 0.1, 0.1)
    pipe.submit(data, 0.2, 0.2)  # replaces the unread frame
    assert {"type": "busy", "busy": True} in events
    gate.set()
    for _ in range(200):
        if {"type": "busy", "busy": False} in events:
            break
        threading.Event().wait(0.01)
    pipe.stop()
    assert {"type": "busy", "busy": False} in events
    report = pipe.report()
    assert report["frames_received"] == 3 and report["frames_replaced_in_mailbox"] == 1
    assert report["frames_processed"] == 2 and "total" in report["stages"]


def test_bad_message_is_reported_not_raised():
    pipe, events = make()
    pipe.submit(b"nonsense", 0.0)
    assert events[-1]["type"] == "error" and pipe.counters.bad_messages == 1


def test_posture_metrics_geometry():
    m = posture_metrics(np.array([320.0, 100.0]), np.array([400.0, 200.0]), np.array([240.0, 200.0]))
    assert m["tilt_deg"] == pytest.approx(0.0) and m["shoulder_width_px"] == pytest.approx(160.0)
    assert m["head_ratio"] == pytest.approx(100 / 160)
    tilted = posture_metrics(np.array([320.0, 100.0]), np.array([400.0, 220.0]), np.array([240.0, 180.0]))
    assert tilted["tilt_deg"] == pytest.approx(np.degrees(np.arctan2(40, 160)))
