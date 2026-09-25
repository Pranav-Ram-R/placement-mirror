"""Smoke test for the packaged app (Task L, CI).

Starts PlacementMirror.exe --no-browser in its own process group, waits for /health, checks
the reported mode, then sends Ctrl+Break (uvicorn's Windows shutdown signal) and checks the
app exits by itself with code 0 and prints its shutdown line.

Usage: python tools/smoke_test.py EXE [--expect-mode "CPU fallback mode"] [--port 8765] [--timeout 600]
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("--expect-mode", default="CPU fallback mode")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for /health (model loading)")
    args = ap.parse_args()

    exe = str(Path(args.exe).resolve())
    proc = subprocess.Popen([exe, "--no-browser", "--port", str(args.port)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    output: list[str] = []
    reader = threading.Thread(target=lambda: output.extend(proc.stdout), daemon=True)
    reader.start()
    failures = []
    health = None
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            with _LOCAL.open(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                health = json.load(r)
            break
        except OSError:
            time.sleep(1)
    if health is None:
        failures.append(f"/health did not answer within {args.timeout:.0f} s (exit code {proc.poll()})")
    else:
        print("/health:", json.dumps(health, indent=1))
        if health.get("status") != "ok":
            failures.append(f"status is {health.get('status')!r}, expected 'ok'")
        if not str(health.get("mode_label", "")).startswith(args.expect_mode):
            failures.append(f"mode_label is {health.get('mode_label')!r}, expected {args.expect_mode!r}")

    if proc.poll() is None:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
            failures.append("the app did not exit within 60 s of Ctrl+Break and was killed")
    else:
        code = proc.returncode
    reader.join(timeout=5)
    text = "".join(output)
    print("---- app output ----\n" + text + "--------------------")
    if code != 0:
        failures.append(f"exit code {code}, expected 0")
    if "Placement Mirror stopped." not in text:
        failures.append("the app did not print its shutdown line")
    if failures:
        print("SMOKE TEST FAILED:\n- " + "\n- ".join(failures))
        return 1
    print(f"SMOKE TEST PASSED: /health ok, {health['mode_label']}, clean exit (code 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
