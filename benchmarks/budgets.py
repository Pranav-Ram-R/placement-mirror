"""Derived budgets for the shipping configuration, from AI Hub hosted Snapdragon X Elite p50s.

- steady state vision per camera frame on the NPU: face landmark p50 every frame plus pose
  landmark p50 on the share of frames that run pose (config npu_pose_fps / npu_face_max_fps).
  The face and pose detectors run only when a track is lost, so they are left out.
- that time as a share of the frame interval, 1 / npu_face_max_fps
- whisper_tiny encoder p50 on CPU and GPU against the NPU, as ratios, all three from the same
  onnx float32 artifact profiled with --compute_unit cpu, gpu and npu

Per-segment ASR cost is left symbolic, encoder p50 + n * decoder p50 for n decoder calls,
since no decoded token count was measured on a device. It needs no Derived record.

Writes benchmarks/raw/budgets_derived.json. Usage: python -m benchmarks.budgets
"""

from __future__ import annotations

from pathlib import Path

from app.config import VISION
from benchmarks.schema import Derived, Measurement, save_json
from benchmarks.table import load_all

OUT = Path(__file__).parent / "raw" / "budgets_derived.json"
JOBS = {
    "face_landmark": "jgzllz465",  # mediapipe_face landmark detector, precompiled_qnn_onnx, NPU
    "pose_landmark": "jpyoo7885",  # mediapipe_pose landmark detector, precompiled_qnn_onnx, NPU
    "encoder_npu": "jp0mxyj2g",  # whisper_tiny encoder, shipping: precompiled_qnn_onnx float32 I/O
    "decoder_npu": "j57e8z9qp",  # whisper_tiny decoder, shipping: precompiled_qnn_onnx float16 I/O
    "encoder_cpu": "j57eeo9rp",  # whisper_tiny encoder, onnx, CPU
    "encoder_gpu": "jp4yye3lp",  # whisper_tiny encoder, onnx, GPU (some ops on CPU)
    "encoder_onnx_npu": "jpxll0x9p",  # whisper_tiny encoder, onnx float32, NPU (same artifact as the two above)
}


def p50(records: list[Measurement], job: str) -> Measurement:
    [m] = [r for r in records if r.job_id == job and r.metric == "inference_time_p50"]
    return m


def main() -> None:
    records = load_all()
    m = {k: p50(records, j) for k, j in JOBS.items()}
    face_fps, pose_fps = VISION.npu_face_max_fps, VISION.npu_pose_fps
    interval_us = 1e6 / face_fps
    per_frame = m["face_landmark"].value + m["pose_landmark"].value * pose_fps / face_fps
    out = [
        Derived(name="vision_npu_steady_state_per_frame", value=per_frame, unit="us",
                formula=f"p50(face landmark, {JOBS['face_landmark']}) + p50(pose landmark, {JOBS['pose_landmark']}) * "
                f"npu_pose_fps / npu_face_max_fps (config {pose_fps:g} / {face_fps:g}). Detectors run only on "
                "track loss and are left out",
                inputs=[m["face_landmark"], m["pose_landmark"]]),
        Derived(name="vision_npu_steady_state_share_of_frame_interval", value=per_frame / interval_us * 100,
                unit="%", formula=f"vision_npu_steady_state_per_frame / (1e6 / npu_face_max_fps) * 100, frame interval "
                f"{interval_us:g} us at config npu_face_max_fps {face_fps:g}",
                inputs=[m["face_landmark"], m["pose_landmark"]]),
        Derived(name="whisper_tiny_encoder_cpu_over_npu", value=m["encoder_cpu"].value / m["encoder_onnx_npu"].value,
                unit="x", formula=f"p50(encoder onnx CPU, {JOBS['encoder_cpu']}) / p50(encoder onnx NPU, {JOBS['encoder_onnx_npu']}), same onnx float32 artifact",
                inputs=[m["encoder_cpu"], m["encoder_onnx_npu"]]),
        Derived(name="whisper_tiny_encoder_gpu_over_npu", value=m["encoder_gpu"].value / m["encoder_onnx_npu"].value,
                unit="x", formula=f"p50(encoder onnx GPU, {JOBS['encoder_gpu']}) / p50(encoder onnx NPU, {JOBS['encoder_onnx_npu']}), same onnx float32 artifact",
                inputs=[m["encoder_gpu"], m["encoder_onnx_npu"]]),
    ]
    save_json(out, OUT)
    for d in out:
        print(d.label)


if __name__ == "__main__":
    main()
