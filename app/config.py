"""App settings.

Values marked PROVISIONAL are placeholders. They are not measured or tuned yet and will
be replaced by the evaluation named next to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisionConfig:
    # Capture. The browser sends frames of this size and letterboxes detector inputs to
    # the detector sizes below.
    frame_width: int = 640
    frame_height: int = 480
    face_detector_size: int = 256
    pose_detector_size: int = 128

    # Calibration: the first seconds after the user presses Calibrate. Baselines are the
    # median of the values seen in that window.
    calibration_s: float = 3.0
    calibration_min_face_frames: int = 5
    calibration_min_pose_frames: int = 3

    # Facing camera: |yaw| and |pitch| relative to calibration below these limits.
    # PROVISIONAL until Task C results (eye contact evaluation on a scripted recording).
    facing_max_yaw_deg: float = 15.0
    facing_max_pitch_deg: float = 12.0

    # Face detector and landmark tracking. Detector values are the qai_hub_models
    # mediapipe_face app defaults. The detector runs only when there is no track. A track
    # ends when the landmark score drops below face_track_min_score (qai_hub_models
    # min_landmark_score).
    face_detector_min_score: float = 0.8
    face_nms_iou: float = 0.3
    face_box_scale: float = 1.1
    face_track_min_score: float = 0.5
    # Side of the next frame's face ROI divided by the long side of the landmark bounding
    # box (in the face aligned frame). Set so the tracked ROI matches the detector ROI
    # (box side x 1.1) on the qai_hub_models mediapipe_face sample photo resized to
    # 640x480: 201.9 px / 156.4 px = 1.29 (tools/face_track_scale.py, 2026-09-25).
    face_track_roi_scale: float = 1.29

    # Frame rates by mode. NPU mode: every vision model runs on the NPU, full rates.
    # CPU fallback mode: at least one vision model runs on CPU, so face runs at up to
    # cpu_face_max_fps and pose at cpu_pose_fps, pose frames are skipped while the ASR
    # worker transcribes, and the vision worker thread runs below the ASR thread's
    # priority. The browser is told the face rate and sends no faster.
    npu_face_max_fps: float = 30.0
    npu_pose_fps: float = 15.0
    cpu_face_max_fps: float = 15.0
    cpu_pose_fps: float = 5.0

    # Pose. The pose detector runs on every pose frame because the AI Hub pose landmark model
    # outputs only the 25 body landmarks, not the auxiliary keypoints MediaPipe uses to
    # track the next ROI. Detector and ROI values are the qai_hub_models mediapipe_pose
    # app defaults.
    pose_detector_min_score: float = 0.75
    pose_nms_iou: float = 0.3
    pose_roi_scale: float = 1.5
    pose_min_score: float = 0.5
    pose_min_visibility: float = 0.5

    # Posture: two independent flags relative to calibration (Checkpoint 3 decision).
    # slouching: head_drop, the fall in (shoulder midpoint y - nose y) / shoulder width,
    # reaches slouch_max_head_drop. leaning: the change in shoulder line angle reaches
    # lean_max_tilt_deg. The UI names the worse one (larger value / threshold).
    # PROVISIONAL: not yet tested on real recordings.
    slouch_max_head_drop: float = 0.15
    lean_max_tilt_deg: float = 8.0


VISION = VisionConfig()


@dataclass(frozen=True)
class AudioConfig:
    # Capture: sounddevice, 16 kHz mono, 512 sample (32 ms) blocks, the block size Silero
    # VAD takes at 16 kHz.
    sample_rate: int = 16000
    block_samples: int = 512

    # Silero VAD. threshold and neg_threshold (threshold - 0.15) are the silero-vad
    # VADIterator defaults: speech starts at a probability >= threshold and continues while
    # it stays >= neg_threshold.
    vad_threshold: float = 0.5
    vad_neg_threshold: float = 0.35
    # ASR segments (Checkpoint 3 decision): a segment ends after segment_end_silence_ms of
    # silence or segment_max_s of audio, with speech_pad_ms of audio kept before and after
    # the speech.
    speech_pad_ms: int = 200
    segment_end_silence_ms: int = 1000
    segment_max_s: float = 20.0
    # A pause event is sent for a silence longer than this, after speech has started.
    # Independent of the ASR segment settings.
    pause_min_s: float = 2.0

    # Whisper. use_prompt feeds the Task B filler prompt before the decoder prefix.
    use_prompt: bool = False

    # Live indicators: WPM = words in the last wpm_window_s seconds / wpm_window_s * 60.
    wpm_window_s: float = 30.0
    indicator_interval_s: float = 1.0


AUDIO = AudioConfig()


@dataclass(frozen=True)
class RuntimeConfig:
    # ONNX Runtime threads per session. Option names from the ORT docs "Thread management":
    # SessionOptions.intra_op_num_threads ("Controls the total number of INTRA threads to use
    # to run the model", the calling thread included), SessionOptions.inter_op_num_threads,
    # and the session config entry "session.intra_op.allow_spinning" = "0" so idle pool
    # threads sleep instead of spinning.
    intra_op_threads: dict = field(default_factory=lambda: {
        "face_detector": 2, "face_landmark": 2, "pose_detector": 2, "pose_landmark": 2,
        "whisper_tiny_encoder": 4, "whisper_tiny_decoder": 4, "silero_vad": 1,
    })
    inter_op_threads: int = 1
    intra_op_allow_spinning: str = "0"


RUNTIME = RuntimeConfig()


@dataclass(frozen=True)
class SessionConfig:
    # An answer stops by itself at the question's suggested_time_s plus this.
    auto_stop_extra_s: float = 60.0
    # A calibration counts when the face was calibrated. Posture needs the pose part too,
    # and without it the session goes on with posture marked unavailable.
    require_face_calibration: bool = True


SESSION = SessionConfig()


GUIDELINE_KIND = "coaching guideline, not a measurement"


@dataclass(frozen=True)
class Guideline:
    """A target range for one report metric. low and high are inclusive, None means no limit."""
    metric: str
    label: str
    unit: str
    low: float | None = None
    high: float | None = None
    kind: str = GUIDELINE_KIND


@dataclass(frozen=True)
class ReportConfig:
    # The report splits the answer into windows of window_s seconds for facing camera,
    # posture and WPM. A last window shorter than min_last_window_s is merged into the
    # window before it, so a short tail does not show as a spike.
    window_s: float = 10.0
    min_last_window_s: float = 5.0

    # Coaching guidelines the feedback compares each metric with. These are coaching
    # targets, not measurements. PROVISIONAL: chosen by the developer, need author review.
    guidelines: tuple = (
        Guideline("facing_camera_pct", "Facing the camera", "% of the answer", low=70.0),
        Guideline("slouching_pct", "Slouching", "% of the answer", high=10.0),
        Guideline("leaning_pct", "Leaning", "% of the answer", high=10.0),
        Guideline("wpm", "Speaking pace", "words per minute", low=120.0, high=160.0),
        Guideline("fillers_per_min", "Filler words", "per minute", high=2.0),
        Guideline("long_pauses_per_min", "Long pauses", "per minute", high=1.0),
    )


REPORT = ReportConfig()
