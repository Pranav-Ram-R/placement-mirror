"""Record one clip from the default microphone at 16 kHz mono to the next numbered WAV.

Files are named clip_01.wav, clip_02.wav and so on in the output folder. WAV files
under eval/ are gitignored.

Usage: python eval/fillers/record_clip.py [--seconds 45] [--out-dir eval/fillers]
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

RATE = 16000


def next_path(out_dir: Path) -> Path:
    n = 1
    while (out_dir / f"clip_{n:02d}.wav").exists():
        n += 1
    return out_dir / f"clip_{n:02d}.wav"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = next_path(out_dir)
    device = sd.query_devices(kind="input")
    print(f"Input device: {device['name']}")
    print(f"Recording {args.seconds:g} s to {path}")
    for n in (3, 2, 1):
        print(f"\rStarting in {n} ...   ", end="", flush=True)
        time.sleep(1)

    frames = int(round(args.seconds * RATE))
    audio = sd.rec(frames, samplerate=RATE, channels=1, dtype="int16")
    start = time.monotonic()
    while True:
        left = args.seconds - (time.monotonic() - start)
        if left <= 0:
            break
        print(f"\rRECORDING  {left:5.1f} s left   ", end="", flush=True)
        time.sleep(0.1)
    sd.wait()
    print("\rDone.                       ")

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())
    if int(np.abs(audio).max()) == 0:
        print("Warning: the recording is all zeros. Check the microphone.")
    print(f"Saved {path}")
    print(f"Add a row for {path.name} to labels.csv: clip,um_count,uh_count,hmm_count")
    return 0


if __name__ == "__main__":
    sys.exit(main())
