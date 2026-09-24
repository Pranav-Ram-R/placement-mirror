"""Windows ARM64 dependency kill test.

Installs each runtime dependency alone in a fresh venv, imports it, and
records exactly what happens. Never exits early. Always writes a markdown
report (default ci/deps_report.md).

Usage: python ci/check_deps.py [--out PATH] [package ...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import venv
from pathlib import Path

# pip name -> import name
PACKAGES = {
    "numpy": "numpy",
    "opencv-python": "cv2",
    "onnxruntime-qnn": "onnxruntime",
    "sounddevice": "sounddevice",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pyinstaller": "PyInstaller",
    "onnx": "onnx",
}
PIP_TIMEOUT_S = 1200
PROBE_TIMEOUT_S = 300


# ---------------------------------------------------------------- tiny ONNX model
# Hand-encoded protobuf so this test does not depend on the onnx package.


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _int(field: int, n: int) -> bytes:
    return _varint(field << 3) + _varint(n)


def _len(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _str(field: int, s: str) -> bytes:
    return _len(field, s.encode())


def _value_info(name: str, dims: list[int]) -> bytes:
    shape = b"".join(_len(1, _int(1, d)) for d in dims)  # TensorShapeProto.dim.dim_value
    tensor = _int(1, 1) + _len(2, shape)  # elem_type FLOAT, shape
    return _str(1, name) + _len(2, _len(1, tensor))  # ValueInfoProto.name, .type.tensor_type


def build_add_model() -> bytes:
    """C = A + B, float32 [1, 4]. IR version 8, opset 13."""
    node = _str(1, "A") + _str(1, "B") + _str(2, "C") + _str(3, "add") + _str(4, "Add")
    graph = (
        _len(1, node)
        + _str(2, "add_graph")
        + _len(11, _value_info("A", [1, 4]))
        + _len(11, _value_info("B", [1, 4]))
        + _len(12, _value_info("C", [1, 4]))
    )
    return _int(1, 8) + _str(2, "check_deps") + _len(7, graph) + _len(8, _int(2, 13))


# ---------------------------------------------------------------- probes run inside each venv

IMPORT_PROBE = r"""
import importlib, importlib.metadata, json, sys
mod = importlib.import_module(sys.argv[1])
print(json.dumps({"module_version": str(getattr(mod, "__version__", None)),
                  "dist_version": importlib.metadata.version(sys.argv[2]),
                  "module_file": getattr(mod, "__file__", None)}))
"""

ORT_QNN_PROBE = r"""
import json, os, sys, traceback
import numpy as np
import onnxruntime as ort

res = {}
try:
    ort.set_default_logger_severity(1)
except Exception as e:
    res["set_default_logger_severity_error"] = repr(e)
res["ort_version"] = ort.__version__
res["available_providers"] = ort.get_available_providers()
capi = os.path.join(os.path.dirname(ort.__file__), "capi")
res["qnn_dlls_in_package"] = sorted(f for f in os.listdir(capi) if f.lower().startswith("qnn") and f.lower().endswith(".dll"))
model = sys.argv[1]
a = np.arange(4, dtype=np.float32).reshape(1, 4)
b = np.ones((1, 4), dtype=np.float32)

def attempt(label, providers, disable_cpu_fallback=False):
    sys.stderr.write(f"\n===== attempt: {label} =====\n")
    sys.stderr.flush()
    r = {"label": label, "providers_requested": repr(providers), "created": False}
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 1
        if disable_cpu_fallback:
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        s = ort.InferenceSession(model, sess_options=so, providers=providers)
        r["created"] = True
        r["session_providers"] = s.get_providers()
        out = s.run(None, {"A": a, "B": b})[0]
        r["output"] = out.tolist()
        r["output_correct"] = bool(np.allclose(out, a + b))
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        r["traceback"] = traceback.format_exc()
    sys.stderr.flush()
    return r

qnn = ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})
res["attempts"] = [
    attempt("QNNExecutionProvider backend_path=QnnHtp.dll", [qnn]),
    attempt("CPUExecutionProvider", ["CPUExecutionProvider"]),
    attempt("QNNExecutionProvider backend_path=QnnHtp.dll with session.disable_cpu_ep_fallback=1", [qnn], True),
]
print(json.dumps(res))
"""

SOUNDDEVICE_PROBE = r"""
import json
res = {}
try:
    import sounddevice as sd
    res["portaudio_loaded"] = True
    res["portaudio_version"] = list(sd.get_portaudio_version())
    res["portaudio_lib"] = str(getattr(sd, "_libname", None))
    try:
        devices = sd.query_devices()
        res["device_count"] = len(devices)
    except Exception as e:
        res["query_devices_error"] = f"{type(e).__name__}: {e}"
except Exception as e:
    res["portaudio_loaded"] = False
    res["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(res))
"""


# ---------------------------------------------------------------- helpers


def run(cmd: list[str], timeout: int) -> dict:
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "seconds": round(time.monotonic() - start, 1)}
    except subprocess.TimeoutExpired as e:
        return {"rc": None, "stdout": str(e.stdout or ""), "stderr": f"TIMEOUT after {timeout} s\n{e.stderr or ''}",
                "seconds": round(time.monotonic() - start, 1)}
    except Exception as e:
        return {"rc": None, "stdout": "", "stderr": f"{type(e).__name__}: {e}", "seconds": round(time.monotonic() - start, 1)}


def last_json(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def first_error_line(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        if l.startswith("ERROR:") or l.startswith("error:"):
            return l
    for l in reversed(lines):
        if re.search(r"error|exception|timeout", l, re.I):
            return l
    return lines[-1] if lines else "no output"


def artifacts(pip_output: str, pkg: str) -> list[str]:
    norm = re.sub(r"[-_.]+", "_", pkg).lower()
    files = re.findall(r"(?:Downloading|Using cached)\s+(\S+?\.(?:whl|tar\.gz|zip))", pip_output)
    return sorted({f.rsplit("/", 1)[-1] for f in files if re.sub(r"[-_.]+", "_", f.rsplit("/", 1)[-1]).lower().startswith(norm + "_")})


def cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# ---------------------------------------------------------------- one package


def check(pkg: str, import_name: str, work: Path, model_path: Path) -> dict:
    r = {"package": pkg, "install": "not run", "import": "not run", "version": "", "notes": "", "details": {}}
    try:
        env_dir = work / f"venv_{re.sub(r'[^A-Za-z0-9]', '_', pkg)}"
        venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
        py = str(venv_python(env_dir))
        r["details"]["pip_version"] = run([py, "-m", "pip", "--version"], 120)["stdout"].strip()

        inst = run([py, "-m", "pip", "install", "--no-input", pkg], PIP_TIMEOUT_S)
        out = inst["stdout"] + "\n" + inst["stderr"]
        r["details"]["install_seconds"] = inst["seconds"]
        r["details"]["artifacts"] = artifacts(out, pkg)
        r["details"]["built_from_source"] = bool(re.search(r"Building wheels? for", out))
        r["details"]["install_output_tail"] = "\n".join(out.strip().splitlines()[-40:])
        if inst["rc"] != 0:
            r["install"] = "FAIL"
            r["notes"] = first_error_line(out)
            return r
        r["install"] = "pass"
        notes = []
        if r["details"]["artifacts"]:
            notes.append("from " + ", ".join(r["details"]["artifacts"]))
        if r["details"]["built_from_source"]:
            notes.append("built from source")

        imp = run([py, "-c", IMPORT_PROBE, import_name, pkg], PROBE_TIMEOUT_S)
        r["details"]["import_stderr_tail"] = "\n".join(imp["stderr"].strip().splitlines()[-40:])
        info = last_json(imp["stdout"])
        if imp["rc"] != 0 or info is None:
            r["import"] = "FAIL"
            notes.append(first_error_line(imp["stderr"] or imp["stdout"]))
            r["notes"] = ". ".join(notes)
            return r
        r["import"] = "pass"
        r["version"] = info["dist_version"]
        r["details"]["import_info"] = info

        if pkg == "onnxruntime-qnn":
            probe = run([py, "-c", ORT_QNN_PROBE, str(model_path)], PROBE_TIMEOUT_S)
            r["details"]["ort_probe_stderr"] = probe["stderr"]
            res = last_json(probe["stdout"])
            r["details"]["ort_probe"] = res if res is not None else {"rc": probe["rc"], "stdout": probe["stdout"]}
            if res is None:
                notes.append("ORT probe crashed: " + first_error_line(probe["stderr"] or probe["stdout"]))
            else:
                notes.append("providers " + ", ".join(res.get("available_providers", [])))
                for a in res.get("attempts", []):
                    if a.get("created"):
                        notes.append(f"{a['label']}: session created, providers {a.get('session_providers')}, "
                                     f"output correct {a.get('output_correct')}{', ' + a['error'] if a.get('error') else ''}")
                    else:
                        notes.append(f"{a['label']}: FAILED, {a.get('error', '').splitlines()[0][:300]}")
        if pkg == "sounddevice":
            probe = run([py, "-c", SOUNDDEVICE_PROBE], PROBE_TIMEOUT_S)
            res = last_json(probe["stdout"])
            r["details"]["sounddevice_probe"] = res if res is not None else {"rc": probe["rc"], "stderr": probe["stderr"]}
            if res and res.get("portaudio_loaded"):
                notes.append(f"PortAudio loaded ({res.get('portaudio_version')}), lib {res.get('portaudio_lib')}, "
                             + (f"devices {res['device_count']}" if "device_count" in res else f"query_devices {res.get('query_devices_error')}"))
            else:
                notes.append("PortAudio did not load: " + str((res or {}).get("error") or first_error_line(probe["stderr"])))
        r["notes"] = ". ".join(notes)
    except Exception:
        r["notes"] = (r["notes"] + ". " if r["notes"] else "") + "check_deps internal error, see details"
        r["details"]["internal_error"] = traceback.format_exc()
    return r


# ---------------------------------------------------------------- report


def write_report(path: Path, results: list[dict], header: dict, fatal: str | None) -> None:
    lines = ["# Windows ARM64 dependency report", ""]
    for k, v in header.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "Each package is installed alone in a fresh venv with `pip install <package>` and no version pin.", ""]
    lines += ["| package | install | import | version | notes |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {cell(r['package'])} | {r['install']} | {r['import']} | {cell(r['version'])} | {cell(r['notes'])} |")
    if fatal:
        lines += ["", "## check_deps internal error", "", "```", fatal, "```"]
    lines += ["", "## Details", ""]
    for r in results:
        lines += [f"### {r['package']}", ""]
        for k, v in r["details"].items():
            if isinstance(v, str) and "\n" in v:
                lines += [f"<details><summary>{k}</summary>", "", "```", v.strip(), "```", "", "</details>", ""]
            else:
                lines += [f"- {k}: `{json.dumps(v) if not isinstance(v, str) else v}`"]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("deps_report.md")))
    ap.add_argument("packages", nargs="*", default=list(PACKAGES))
    args = ap.parse_args()

    header = {
        "date (UTC)": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "platform.machine()": platform.machine(),
        "platform.platform()": platform.platform(),
        "sys.version": sys.version.replace("\n", " "),
        "runner": f"{os.environ.get('ImageOS', 'local')} {os.environ.get('ImageVersion', '')}".strip(),
    }
    print("\n".join(f"{k}: {v}" for k, v in header.items()), flush=True)
    results: list[dict] = []
    fatal = None
    try:
        work = Path(tempfile.mkdtemp(prefix="check_deps_"))
        model_path = work / "add.onnx"
        model_path.write_bytes(build_add_model())
        for pkg in args.packages:
            print(f"--- {pkg}", flush=True)
            r = check(pkg, PACKAGES.get(pkg, pkg), work, model_path)
            print(f"    install {r['install']}, import {r['import']}, version {r['version']}, {r['notes']}", flush=True)
            results.append(r)
    except Exception:
        fatal = traceback.format_exc()
        print(fatal, flush=True)
    finally:
        write_report(Path(args.out), results, header, fatal)
        print(f"report written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
