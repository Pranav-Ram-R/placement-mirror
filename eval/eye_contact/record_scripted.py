"""Scripted 60 s webcam recording for the eye contact test (Task C).

Shows full-screen text prompts and logs one label per captured frame:
  3 s countdown, then
  "Look at the CAMERA" 15 s, "Look at the CENTER of the screen" 15 s,
  "Look AWAY to the left" 10 s, "Look DOWN at the desk" 10 s, "Look at the CAMERA" 10 s.

Saves <name>.mp4 and <name>_labels.csv (frame, t_s, label) under
eval/eye_contact/recordings/, which is gitignored. Eval tool for the x86 dev machine
only, so it uses cv2 for capture and display. Press Esc to abort.

Usage: python eval/eye_contact/record_scripted.py [--camera 0] [--out-dir eval/eye_contact/recordings]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCHEDULE = [
    ("CAMERA", "Look at the CAMERA", 15.0),
    ("SCREEN", "Look at the CENTER of the screen", 15.0),
    ("AWAY_LEFT", "Look AWAY to the left", 10.0),
    ("DOWN", "Look DOWN at the desk", 10.0),
    ("CAMERA", "Look at the CAMERA", 10.0),
]
COUNTDOWN_S = 3
WIDTH, HEIGHT, FPS = 640, 480, 30
WINDOW = "Placement Mirror eye contact recording"


def label_at(t: float) -> tuple[str, str, float] | None:
    end = 0.0
    for label, text, seconds in SCHEDULE:
        end += seconds
        if t < end:
            return label, text, end - t
    return None


def screen(text: str, sub: str = "") -> np.ndarray:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for line, scale, y in ((text, 2.6, 520), (sub, 1.2, 640)):
        if line:
            (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
            cv2.putText(img, line, ((1920 - w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4,
                        cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "recordings"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "scripted_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path, labels_path = out_dir / f"{name}.mp4", out_dir / f"{name}_labels.csv"

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return 1
    ok, frame = cap.read()
    if not ok:
        print("Camera returned no frame")
        return 1
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    start = time.monotonic()
    while (left := COUNTDOWN_S - (time.monotonic() - start)) > 0:
        cap.read()  # keep the camera buffer fresh
        cv2.imshow(WINDOW, screen(f"Starting in {int(np.ceil(left))}", "Sit as you would in an interview"))
        if cv2.waitKey(1) == 27:
            return 1

    rows, aborted = [], False
    start = time.monotonic()
    while True:
        ok, frame = cap.read()
        t = time.monotonic() - start
        now = label_at(t)
        if not ok or now is None:
            break
        label, text, left = now
        writer.write(frame)
        rows.append((len(rows), round(t, 4), label))
        cv2.imshow(WINDOW, screen(text, f"{left:4.1f} s"))
        if cv2.waitKey(1) == 27:
            aborted = True
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "t_s", "label"])
        wr.writerows(rows)
    duration = rows[-1][1] if rows else 0.0
    print(f"{'Aborted' if aborted else 'Done'}: {len(rows)} frames over {duration:.1f} s")
    print(f"Video: {video_path}\nLabels: {labels_path}")
    return 0 if not aborted else 1


if __name__ == "__main__":
    sys.exit(main())
