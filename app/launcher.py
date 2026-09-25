"""Placement Mirror launcher: starts the local server and opens the browser.

The packaged app (PlacementMirror.exe) runs this. It takes the first free port from --port
(default 8000) upwards, starts the server on 127.0.0.1, waits until /health answers, then
opens the default browser on the practice page. Press Ctrl+C or close the window to stop.
Other options go to the server (python -m app.server --help).

Usage: PlacementMirror.exe [--port 8000] [--no-browser] [server options]
       python -m app.launcher [--port 8000] [--no-browser] [server options]
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

PORT_TRIES = 20
# Loopback only, never through a system proxy.
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def free_port(start: int, tries: int = PORT_TRIES) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"No free port from {start} to {start + tries - 1}")


def wait_for_health(port: int, timeout_s: float) -> dict | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with _LOCAL.open(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                return json.load(r)
        except OSError:
            time.sleep(0.5)
    return None


def open_when_ready(port: int, timeout_s: float = 600.0) -> None:
    url = f"http://127.0.0.1:{port}/"
    health = wait_for_health(port, timeout_s)
    if health is None:
        print(f"The server did not answer at {url} within {timeout_s:.0f} s.", flush=True)
        return
    print(f"Placement Mirror is running at {url} ({health['mode_label']}). Opening the browser.", flush=True)
    webbrowser.open(url)


def _stop_signal_after_shutdown(sig, frame) -> None:
    """uvicorn shuts down cleanly on Ctrl+C or Ctrl+Break, then raises the same signal again
    with the handler that was set before it started. With Python's defaults that would end
    the process with a KeyboardInterrupt or exit code 3, so the launcher sets this one."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Start Placement Mirror and open it in the browser.")
    ap.add_argument("--port", type=int, default=8000, help="first port to try (default 8000)")
    ap.add_argument("--no-browser", action="store_true", help="start the server without opening the browser")
    args, server_args = ap.parse_known_args(argv)
    port = free_port(args.port)
    from app import server  # loads ONNX Runtime and the app code

    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(port,), name="open-browser", daemon=True).start()
    print(f"Starting Placement Mirror on http://127.0.0.1:{port}/ . Loading models. Press Ctrl+C to stop.",
          flush=True)
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            signal.signal(sig, _stop_signal_after_shutdown)
    code = server.main(["--port", str(port), *server_args])
    print("Placement Mirror stopped.", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
