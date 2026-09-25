"""QNN HTP emulation correctness check for the precompiled face landmark model.

Correctness only. No timings are recorded. Result label:
"local x86 CPU, QNN HTP emulation, correctness only".

Step 1, eval venv (needs qai_hub_models for the sample photo and the detector anchors):
  python eval/qnn_emulation/face_landmark_check.py crop --out <crop.npy>
  Loads the mediapipe_face sample photo (qai_hub_models asset store, face.jpeg), runs the
  onnx face detector on CPU and writes the 192x192 RGB face crop.

Step 2, app venv (onnxruntime-qnn 2.6.0):
  python eval/qnn_emulation/face_landmark_check.py compare --crop <crop.npy> --out <result.json>
  Loads models/face_landmark/precompiled_qnn_onnx on the QNN EP device listed by
  get_ep_devices (on x86 a CPU-type device, the HTP emulator) with
  session.disable_cpu_ep_fallback=1, and models/face_landmark/onnx on
  CPUExecutionProvider. Runs the crop through both and reports the max abs difference of
  the landmarks, also saved as Measurement records next to --out. If the QNN session fails,
  the exact error is recorded and nothing else runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.vision.preprocess import image_tensor, input_layout  # noqa: E402

LABEL = "local x86 CPU, QNN HTP emulation, correctness only"
LM = REPO / "models" / "face_landmark"


def crop(out: Path) -> None:
    from PIL import Image
    from qai_hub_models.models.mediapipe_face.demo import INPUT_IMAGE_ADDRESS

    sys.path.insert(0, str(REPO / "eval" / "eye_contact"))
    import head_pose_test as hp

    src = INPUT_IMAGE_ADDRESS.fetch()
    rgb = np.asarray(Image.open(src).convert("RGB"))
    pipe = hp.Pipeline.__new__(hp.Pipeline)
    import onnxruntime as ort

    pipe.det = ort.InferenceSession(str(REPO / "models/face_detector/onnx/model.onnx"), providers=["CPUExecutionProvider"])
    pipe.anchors = hp.load_anchors()
    pipe.det_outputs = [o.name for o in pipe.det.get_outputs()]
    h, w = rgb.shape[:2]
    m_lb, scale, (pad_l, pad_t) = hp.letterbox_affine(w, h, hp.DET_SIZE, hp.DET_SIZE)
    det_in = image_tensor(hp.warp_affine_bilinear(rgb, m_lb, hp.DET_SIZE, hp.DET_SIZE), "NCHW")
    o = dict(zip(pipe.det_outputs, pipe.det.run(None, {"image": det_in})))
    coords = np.concatenate([o["box_coords_1"][0], o["box_coords_2"][0]], axis=0)
    scores = np.concatenate([o["box_scores_1"][0], o["box_scores_2"][0]], axis=0)
    boxes, kps, probs = hp.decode(coords, scores, pipe.anchors)
    keep = hp.nms(boxes, probs)
    if not keep:
        raise SystemExit("no face detected in the sample photo")
    i = keep[0]
    unpad = lambda p: (p - [pad_l, pad_t]) / scale  # noqa: E731
    box, kp = unpad(boxes[i].reshape(2, 2)).reshape(4), unpad(kps[i])
    corners = hp.roi_corners(box, kp[hp.KP_START], kp[hp.KP_END], scale=hp.BOX_SCALE, offset=hp.BOX_OFFSET)
    face = hp.warp_affine_bilinear(rgb, hp.crop_affine(corners, hp.LM_SIZE, hp.LM_SIZE), hp.LM_SIZE, hp.LM_SIZE)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, face.astype(np.float32))
    out.with_suffix(".json").write_text(json.dumps({
        "photo": "qai_hub_models asset store, mediapipe_face v3 face.jpeg", "photo_url": getattr(INPUT_IMAGE_ADDRESS, "url", None), "photo_size": [w, h],
        "detector": "models/face_detector/onnx on CPUExecutionProvider", "box_score": float(probs[i]),
    }, indent=2), encoding="utf-8")
    print(f"crop {face.shape} from {src} (box score {probs[i]:.3f}) -> {out}")


def compare(crop_path: Path, out: Path) -> int:
    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep

    face = np.load(crop_path)
    meta_path = crop_path.with_suffix(".json")
    result = {
        "label": LABEL, "check": "face_landmark precompiled_qnn_onnx on QNN HTP emulation vs onnx on CPU",
        "machine": platform.machine(), "processor": platform.processor(),
        "onnxruntime": ort.__version__, "onnxruntime_qnn": qnn_ep.__version__,
        "input": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else str(crop_path),
        "timings_recorded": False,
    }
    ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    result["qnn_devices"] = [str(d.device.type) for d in devices]
    precompiled = LM / "precompiled_qnn_onnx" / "model.onnx"
    try:
        so = ort.SessionOptions()
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_provider_for_devices(devices, {"backend_path": qnn_ep.get_qnn_htp_path()})
        qnn = ort.InferenceSession(str(precompiled), sess_options=so)
        result["qnn_session"] = {"created": True, "providers": qnn.get_providers()}
        q_in = qnn.get_inputs()[0]
        q_out = dict(zip([o.name for o in qnn.get_outputs()],
                         qnn.run(None, {q_in.name: image_tensor(face, input_layout(q_in.shape))})))
    except Exception as e:  # noqa: BLE001
        result["qnn_session"] = {"created": "qnn" in locals(), "error": f"{type(e).__name__}: {e}",
                                 "traceback": traceback.format_exc()}
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"QNN step FAILED: {type(e).__name__}: {e}")
        return 1

    cpu = ort.InferenceSession(str(LM / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])
    c_in = cpu.get_inputs()[0]
    c_out = dict(zip([o.name for o in cpu.get_outputs()],
                     cpu.run(None, {c_in.name: image_tensor(face, input_layout(c_in.shape))})))
    lq, lc = q_out["landmarks"].astype(np.float64), c_out["landmarks"].astype(np.float64)
    diff = np.abs(lq - lc)
    result.update({
        "qnn_input_layout": input_layout(q_in.shape), "cpu_input_layout": input_layout(c_in.shape),
        "landmarks_shape": list(lq.shape), "landmark_units": "model output (x, y, z) / 192, crop normalized",
        "landmarks_max_abs_diff": float(diff.max()),
        "landmarks_max_abs_diff_xyz": [float(v) for v in diff.reshape(-1, 3).max(axis=0)],
        "landmarks_max_abs_diff_px_192": float(diff.max() * 192),
        "score_qnn": float(q_out["scores"].reshape(-1)[0]), "score_cpu": float(c_out["scores"].reshape(-1)[0]),
    })
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "input"}, indent=2))

    from benchmarks.schema import Measurement, Source, save_json

    notes = (f"{LABEL}. precompiled_qnn_onnx landmarks on the QNN EP {result['qnn_devices']} device "
             f"(strict, session.disable_cpu_ep_fallback=1) vs models/face_landmark/onnx on CPUExecutionProvider, "
             f"one 192x192 face crop of the qai_hub_models mediapipe_face sample photo. "
             f"onnxruntime_qnn {qnn_ep.__version__}. Result file {out.resolve().relative_to(REPO).as_posix()}")
    common = dict(model="face_landmark", source=Source.LOCAL_X86_CPU, runtime="precompiled_qnn_onnx",
                  compute_unit="QNN HTP emulation on CPU", precision="float32", ort_version=ort.__version__)
    save_json([
        Measurement(metric="landmarks_max_abs_diff_vs_onnx_cpu", value=result["landmarks_max_abs_diff"],
                    unit="normalized crop coordinate (landmark / 192)", notes=notes, **common),
        Measurement(metric="landmarks_max_abs_diff_vs_onnx_cpu_px", value=result["landmarks_max_abs_diff_px_192"],
                    unit="px in the 192x192 crop", notes=notes, **common),
    ], out.with_name(out.stem + "_measurements.json"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("crop")
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("compare")
    b.add_argument("--crop", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "crop":
        crop(args.out)
        return 0
    return compare(args.crop, args.out)


if __name__ == "__main__":
    sys.exit(main())
