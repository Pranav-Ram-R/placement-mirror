# Placement Mirror
Real-time mock interview coach that runs fully on-device on Snapdragon X laptops.
It analyzes delivery (eye contact, posture, speaking pace, filler words, pauses)
live while the user answers, then produces a report. The headline is concurrent
multimodal inference on the Hexagon NPU. LLM content feedback is secondary.

## Stack
Python app. ONNX Runtime with the QNN Execution Provider on Windows ARM64.
Browser capture (getUserMedia, canvas resize, raw RGB over WebSocket).
Local web UI on localhost. Models from Qualcomm AI Hub
(MediaPipe face, landmarks and pose, Whisper) plus Silero VAD and a small local LLM.
Development happens on Windows x86. AI Hub hosted devices are used for NPU compile
and profile jobs.

## Hard rules
1. Never invent, estimate or extrapolate a performance number. Every number comes
   from a real run and is stored as benchmarks.schema.Measurement with a Source.
2. Derived numbers use benchmarks.schema.Derived and show their formula.
3. Anything only confirmable on a physical Snapdragon device is labeled
   "not yet validated on device".
4. Every dependency must support Windows ARM64. Do not add the mediapipe pip
   package. Check before adding any new dependency and say what you checked.
5. The app must work fully offline after install. No network calls at runtime.
6. If QNN EP fails to load, fall back to CPU and show a visible notice.
7. Do not change scope or add features beyond the task you were given.
8. User-facing text and docs: no em dashes, no semicolons, no hype words.
9. Never commit model files, recordings, API tokens or session data.
10. No OpenCV in app code. Capture and resize happen in the browser. Image
    math uses numpy.
11. Never trust the requested execution provider. After creating each
    session, verify where the model was placed (strict session with CPU
    fallback disabled, or the EP device API), and report the real compute
    unit to the UI. A model that silently falls back to CPU must show as CPU.

## Layout
app/ runtime code by stage. aihub/ AI Hub scripts. benchmarks/ measurements.
eval/ evaluation data and scripts. questions/ question bank. docs/ architecture,
description doc, deck.
