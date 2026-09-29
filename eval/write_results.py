"""Write eval/RESULTS.md from the eval result files (Day 3 item 2d).

Every number is read from a results file, nothing is typed in:
- fillers (Task B): eval/fillers/results/filler_results.json (run_filler_test.py)
- eye contact (Task C): eval/eye_contact/results/<name>_results.json (head_pose_test.py)
- pace: eval/pace/results/pace_results.json (pace/run_pace_test.py)

Task B verdict: PASS if the app pipeline's whisper_tiny recall (app_whisper_tiny) is
>= 0.70 in either condition, else FAIL. A missing fillers or pace results file gives a
"not run" section and no verdict.

Usage: python eval/write_results.py --eye-contact eval/eye_contact/results/<name>_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.config import VISION  # noqa: E402

RECALL_TO_PASS = 0.70
NOT_ON_DEVICE = "NPU fp16 transcripts not yet validated on device"
LIMITS = ("Limitations: one speaker recorded in one room with one laptop microphone and camera, on {n}, "
          "so these numbers describe that setup only and are not a general accuracy estimate.")


def clips_text(n: int) -> str:
    return f"{n} clip{'' if n == 1 else 's'}"


def f3(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def f1(v) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def not_run(title: str, path: Path, why: str) -> list[str]:
    return [f"## {title}", "", f"Not run: {why} No results file at {path.relative_to(HERE.parent).as_posix()}.", ""]


def fillers(path: Path) -> tuple[list[str], str]:
    if not path.exists():
        return not_run("Filler detection (Task B)", path, "no hand labeled filler counts were made for the "
                       "recorded answers, so there is nothing to score the transcripts against."), "not run"
    r = json.loads(path.read_text(encoding="utf-8"))
    rows = r["summary"]
    clips = rows[0]["clips"] if rows else 0
    app_id = r["verdict_model"]
    app = [s for s in rows if s["model"] == app_id]
    passed = any(s["recall"] is not None and s["recall"] >= RECALL_TO_PASS for s in app)
    verdict = "PASS" if passed else "FAIL"
    models = r["models"]
    app_models = ", ".join(f"{n} {m['runtime']} {m['precision']} on {m['compute_unit']}"
                           for n, m in models[app_id]["models"].items())
    refs = ", ".join(f"{k} = {m['checkpoint']}" for k, m in models.items() if m["pipeline"] == "reference")
    lines = ["## Filler detection (Task B)", "",
             f"Source: {r['source']}. {app_id} is the app's own speech path (VAD segments, the app's Whisper "
             f"decode loop, {models[app_id]['runtime']}: {app_models}). The other rows are "
             f"{next(m['runtime'] for m in models.values() if m['pipeline'] == 'reference')} float32 reference "
             f"models for comparison ({refs}). {r['not_validated']}. Clips: {clips}. Labeled fillers: "
             f"{rows[0]['total_labeled'] if rows else 0}. Prompt in the disfluent_prompt condition: "
             f"\"{r['prompt']}\"", "",
             "| Pipeline | Model | Condition | Clips | Labeled | Found | Matched | Recall | Precision |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        lines.append(f"| {s['pipeline']} | {s['model']} | {s['condition']} | {s['clips']} | {s['total_labeled']} | "
                     f"{s['total_found']} | {s['total_matched']} | {f3(s['recall'])} | {f3(s['precision'])} |")
    lines += ["", "Recall = matched / labeled, precision = matched / found, matched = min(found, labeled) per clip "
              "and filler type (Derived). The reference rows are for comparison only.", "",
              f"Verdict: {verdict} ({app_id} recall {' and '.join(f3(s['recall']) for s in app)} "
              f"against {RECALL_TO_PASS:.2f} in either condition).", "",
              LIMITS.format(n=clips_text(clips)), ""]
    return lines, verdict


def eye_contact(path: Path) -> list[str]:
    r = json.loads(path.read_text(encoding="utf-8"))
    c = r["classifier"]
    lines = ["## Eye contact (Task C)", "",
             f"Source: {r['source']}. Video: {r['video']}, labels: {r['labels']}. Labeled frames: "
             f"{r['frames_labeled']}, with a head pose: {r['frames_with_pose']}. Calibration: "
             f"{r['calibration']['frames']} CAMERA frames.", "",
             "| Segment | Start s | End s | Frames with pose | Yaw mean | Yaw std | Pitch mean | Pitch std |",
             "|---|---|---|---|---|---|---|---|"]
    for s in r["segments"]:
        lines.append(f"| {s['label']} | {f1(s['start_s'])} | {f1(s['end_s'])} | {s['frames_with_pose']} | "
                     f"{f1(s['yaw_mean'])} | {f1(s['yaw_std'])} | {f1(s['pitch_mean'])} | {f1(s['pitch_std'])} |")
    lines += ["", "Angles in degrees, calibrated to the first CAMERA seconds. " + r["angle_convention"] + ".", "",
              "| Facing vs away | Frames | Threshold deg | Accuracy | Balanced accuracy |", "|---|---|---|---|---|"]
    for name, key in (("Train (chosen here)", "train_eval"), ("Held out", "test_eval")):
        e = c.get(key)
        lines.append(f"| {name}: {c['train'] if key == 'train_eval' else c['test']} | "
                     f"{e['frames'] if e else 'n/a'} | {f1(c['threshold_deg'])} | {f3(e['accuracy']) if e else 'n/a'} | "
                     f"{f3(e['balanced_accuracy']) if e else 'n/a'} |")
    lines += ["", c["definition"] + ".", ""]
    te = c.get("test_eval")
    lines += [f"The app's limit (VisionConfig.facing_max_angle_deg) stays at {VISION.facing_max_angle_deg:g} deg "
              f"and stays PROVISIONAL. The {f1(c['threshold_deg'])} deg threshold above is not used: it was chosen "
              "on the CAMERA and SCREEN segments only, where SCREEN counts as away, and the CAMERA and SCREEN "
              "ranges overlap in both yaw and pitch (next table)" + (f". Its held-out balanced accuracy is {f3(te['balanced_accuracy'])}" if te else "")
              + ". Treating SCREEN as facing would change the test definition and was not done here.", ""]
    cs = r.get("camera_vs_screen")
    if cs:
        lines += ["| Camera vs screen | Camera p5 to p95 | Screen p5 to p95 | Intervals overlap | "
                  "Best one-threshold balanced accuracy |", "|---|---|---|---|---|"]
        for axis in ("yaw", "pitch"):
            o = cs[axis]
            lines.append(f"| {axis} | {o['camera_p5_p95']} | {o['screen_p5_p95']} | {o['intervals_overlap']} | "
                         f"{f3(o['best_single_threshold_balanced_accuracy'])} |")
        lines.append("")
    lines += [LIMITS.format(n="one scripted recording"), ""]
    return lines


def pace(path: Path) -> list[str]:
    if not path.exists():
        return not_run("Speaking pace word count", path, "no manual word counts were made for the recorded "
                       "answers.")
    r = json.loads(path.read_text(encoding="utf-8"))
    lines = ["## Speaking pace word count", "",
             f"Source: {r['source']}. The app's own speech path offline (Silero VAD segments, Whisper tiny on "
             f"{r['compute_units']['whisper_tiny_encoder']}, prompt {'on' if r['use_prompt'] else 'off'}), words "
             f"counted by {r['word_counter']}. {NOT_ON_DEVICE}. Clips: {len(r['results'])}.", "",
             "| Clip | Audio s | App words | Manual words | Error % |", "|---|---|---|---|---|"]
    for x in r["results"]:
        lines.append(f"| {x['clip']} | {f1(x['audio_s'])} | {x['app_words']} | {x['manual_words']} | "
                     f"{f1(x['error_pct'])} |")
    lines += ["", "Error % = (app words - manual words) / manual words * 100 (Derived).", "",
              LIMITS.format(n=clips_text(len(r['results']))), ""]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fillers", type=Path, default=HERE / "fillers" / "results" / "filler_results.json")
    ap.add_argument("--eye-contact", type=Path, required=True)
    ap.add_argument("--pace", type=Path, default=HERE / "pace" / "results" / "pace_results.json")
    ap.add_argument("--out", type=Path, default=HERE / "RESULTS.md")
    args = ap.parse_args()
    filler_lines, verdict = fillers(args.fillers)
    lines = ["# Evaluation results", "",
             "Generated by eval/write_results.py from the result files named in each section. No claims beyond "
             "these numbers.", ""]
    lines += filler_lines + eye_contact(args.eye_contact) + pace(args.pace)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.out}. Task B verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
