"""ROI warp time before and after vectorization (Task G), on this machine's CPU.

Times app.vision.roi.warp_reference (the Day 1 warp) and warp_affine_bilinear (Task G)
on a 640x480 uint8 RGB frame, the size the browser sends:
- face ROI crop to 192x192 (face landmark input)
- pose ROI crop to 256x256 (pose landmark input)
The two implementations alternate on every iteration so both see the same machine load.
Warp time does not depend on pixel values, so the frame is random.

Usage: python -m benchmarks.bench_roi_warp [--iterations 1000] [--save PATH]
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

from app.vision.roi import crop_affine, keypoint_roi_corners, roi_corners, warp_affine_bilinear, warp_reference
from benchmarks.schema import Measurement, Source, save_json

CASES = {
    # face box 150x180 px with the eyes 70 px apart, as the face detector gives at arm's length
    "roi_warp_face_192": (roi_corners([250, 150, 400, 330], [360, 200], [290, 210], 1.1), 192),
    # pose ROI from hip center (320, 470) and head keypoint (310, 190), scale 1.5
    "roi_warp_pose_256": (keypoint_roi_corners([320, 470], [310, 190], 1.5, np.pi / 2), 256),
}
IMPLS = {"warp_reference (Day 1)": warp_reference, "warp_affine_bilinear (Task G)": warp_affine_bilinear}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--save", type=Path)
    args = ap.parse_args()
    if platform.machine().upper() not in ("AMD64", "X86_64"):
        sys.exit("This benchmark labels results local x86 CPU. Run it on the x86 dev machine.")

    frame = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    records = []
    print(f"numpy {np.__version__}, {platform.processor()}, {args.iterations} iterations after {args.warmup} warmup\n")
    print("| crop | implementation | p50 ms | p95 ms |")
    print("|---|---|---|---|")
    for model, (corners, size) in CASES.items():
        m = crop_affine(corners, size, size)
        times = {name: [] for name in IMPLS}
        for i in range(args.warmup + args.iterations):
            for name, fn in IMPLS.items():
                t0 = time.perf_counter()
                fn(frame, m, size, size)
                dt = time.perf_counter() - t0
                if i >= args.warmup:
                    times[name].append(dt * 1000)
        for name, ts in times.items():
            arr = np.asarray(ts)
            p50, p95 = (float(np.percentile(arr, q, method="linear")) for q in (50, 95))
            print(f"| {model} | {name} | {p50:.3f} | {p95:.3f} |")
            notes = (f"app.vision.roi {name.split(' ')[0]}, 640x480 uint8 RGB frame to {size}x{size} float32, "
                     f"{args.iterations} iterations after {args.warmup} warmup, alternating with the other "
                     f"implementation, numpy.percentile linear, numpy {np.__version__}, "
                     f"CPU {platform.processor()}, benchmarks/bench_roi_warp.py")
            for q, value in ((50, p50), (95, p95)):
                records.append(Measurement(
                    model=model, metric=f"warp_time_p{q}", value=value, unit="ms", source=Source.LOCAL_X86_CPU,
                    runtime=f"numpy {name}", compute_unit="CPU", precision="float32", notes=notes))
    if args.save:
        save_json(records, args.save)
        print(f"\nSaved {len(records)} measurements to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
