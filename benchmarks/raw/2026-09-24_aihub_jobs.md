# AI Hub jobs 2026-09-24

Device: Snapdragon X Elite CRD (Windows 11). qai-hub 0.55.0, qai-hub-models 0.63.0, torch 2.10.0.
Base URL for every job: https://workbench.aihub.qualcomm.com/jobs/<job_id>/

Numbers from finished profile jobs are stored as Measurement records in the JSON files in this folder. Nothing in this file is a performance number.

## Day 0 profile: mediapipe_face (target runtime onnx, float)

| Component | Compile | Profile | Inference |
|---|---|---|---|
| face_detector | jgnzz7nrg | jp2rrvw4g | jgk22942g |
| face_landmark_detector | jprlln09p | jpyoo7x75 | jglyy1x85 |

Profile pages:
- https://workbench.aihub.qualcomm.com/jobs/jp2rrvw4g/
- https://workbench.aihub.qualcomm.com/jobs/jpyoo7x75/

Measurements: 2026-09-24_mediapipe_face_x_elite.json

## Overnight: mediapipe_pose (target runtime precompiled_qnn_onnx, float)

qai-hub-models 0.63.0 does not offer plain onnx for this model.

| Component | Compile | Profile | Inference |
|---|---|---|---|
| pose_detector | jgnzz71kg | jp2rrvorg | jp0mmvo9g |
| pose_landmark_detector | jprllnx0p | jpyoo7885 | j5688do6g |

Measurements: 2026-09-24_mediapipe_pose_x_elite.json

## Overnight: whisper_tiny (target runtime precompiled_qnn_onnx, float)

Checkpoint openai/whisper-tiny (multilingual). Smallest Whisper in qai-hub-models 0.63.0. Plain onnx is not offered for this model.

| Component | Compile | Profile | Inference |
|---|---|---|---|
| encoder | jp1nnvn2g | j5m00d0wg | jprllnl9p |
| decoder | jp4yy9yvp | jgnzz7zrg | jp2rrvr4g |

Measurements: 2026-09-24_whisper_tiny_x_elite.json

## Overnight: distil_whisper (target runtime onnx, float)

Checkpoint distil-whisper/distil-small.en. The only English-only Whisper variant in qai-hub-models 0.63.0.

| Component | Compile | Profile |
|---|---|---|
| encoder | jp4yy971p | j5m00d79g |
| decoder | jpxlldqlp | jgnzz74qg |

Inference job seen at time of logging: jprllnr7p.

## Void jobs

These were submitted by qai-hub-models 0.48.0, which requested the retired QAIRT 2.42. Do not use them: jgdddzvlg, jp4yy9xlp, jpyoo7445, jp3zzw4z5.
