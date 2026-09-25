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
    "wsproto": "wsproto",
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
res["qnn_dlls_in_onnxruntime_capi"] = sorted(f for f in os.listdir(capi) if f.lower().startswith("qnn") and f.lower().endswith(".dll"))


def file_version(path):
    # Windows version resource: fixed FileVersion plus the ProductVersion string.
    import ctypes
    from ctypes import wintypes
    class FIXED(ctypes.Structure):
        _fields_ = [(n, wintypes.DWORD) for n in ("sig", "struc", "fvMS", "fvLS", "pvMS", "pvLS", "mask", "flags", "os", "type", "sub", "dMS", "dLS")]
    ver = ctypes.windll.version
    size = ver.GetFileVersionInfoSizeW(path, None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    ver.GetFileVersionInfoW(path, 0, size, buf)
    p, n = ctypes.c_void_p(), wintypes.UINT()
    out = {}
    if ver.VerQueryValueW(buf, "\\", ctypes.byref(p), ctypes.byref(n)):
        f = ctypes.cast(p, ctypes.POINTER(FIXED)).contents
        out["file_version"] = f"{f.fvMS >> 16}.{f.fvMS & 0xFFFF}.{f.fvLS >> 16}.{f.fvLS & 0xFFFF}"
    if ver.VerQueryValueW(buf, "\\VarFileInfo\\Translation", ctypes.byref(p), ctypes.byref(n)) and n.value >= 4:
        lang, cp = ctypes.cast(p, ctypes.POINTER(wintypes.WORD * 2)).contents
        for key in ("ProductVersion", "FileDescription"):
            q = f"\\StringFileInfo\\{lang:04x}{cp:04x}\\{key}"
            if ver.VerQueryValueW(buf, q, ctypes.byref(p), ctypes.byref(n)) and n.value:
                out[key] = ctypes.wstring_at(p, n.value - 1)
    return out


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

# Plugin QNN EP (onnxruntime-qnn 2.x). Flow from the onnxruntime/onnxruntime-qnn README,
# "Getting Started with the Plugin QNN EP", Python example.
plugin = {}
try:
    import onnxruntime_qnn as qnn_ep
    from onnxruntime_qnn import build_and_package_info as info
    plugin["onnxruntime_qnn_version"] = qnn_ep.__version__
    plugin["bundled_qnn_version_from_package_info"] = getattr(info, "qnn_version", None)
    plugin["library_path"] = qnn_ep.get_library_path()
    plugin["qnn_htp_path"] = qnn_ep.get_qnn_htp_path()
    pkg_dir = os.path.dirname(qnn_ep.__file__)
    plugin["qnn_package_dlls"] = sorted(os.path.relpath(os.path.join(d, f), pkg_dir).replace(os.sep, "/")
                                        for d, _, fs in os.walk(pkg_dir) for f in fs if f.lower().endswith(".dll"))
    lib_dir = os.path.dirname(plugin["library_path"])
    plugin["dll_version_info"] = {}
    for name in ("onnxruntime_providers_qnn.dll", "QnnHtp.dll", "QnnSystem.dll", "QnnHtpV73Stub.dll"):
        try:
            plugin["dll_version_info"][name] = file_version(os.path.join(lib_dir, name))
        except Exception as e:
            plugin["dll_version_info"][name] = f"{type(e).__name__}: {e}"
except Exception as e:
    plugin["import_error"] = f"{type(e).__name__}: {e}"
    plugin["import_traceback"] = traceback.format_exc()

if "library_path" in plugin:
    name = "QNNExecutionProvider"
    try:
        ort.register_execution_provider_library(name, plugin["library_path"])
        plugin["registration"] = "succeeded"
    except Exception as e:
        plugin["registration"] = f"FAILED: {type(e).__name__}: {e}"
    try:
        devices = ort.get_ep_devices()
        plugin["ep_devices"] = [{
            "ep_name": d.ep_name, "ep_vendor": d.ep_vendor, "ep_metadata": dict(d.ep_metadata),
            "ep_options": dict(d.ep_options), "hw_type": str(d.device.type), "hw_vendor": d.device.vendor,
            "hw_vendor_id": d.device.vendor_id, "hw_device_id": d.device.device_id, "hw_metadata": dict(d.device.metadata),
        } for d in devices]
    except Exception as e:
        devices = []
        plugin["ep_devices_error"] = f"{type(e).__name__}: {e}"
    selected = [d for d in devices if d.ep_name == name]
    plugin["selected_qnn_devices"] = [str(d.device.type) for d in selected]
    sys.stderr.write("\n===== attempt: plugin QNN EP devices with session.disable_cpu_ep_fallback=1 =====\n")
    sys.stderr.flush()
    strict = {"created": False}
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 1
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_provider_for_devices(selected, {"backend_path": plugin["qnn_htp_path"]})
        s = ort.InferenceSession(sys.argv[1], sess_options=so)
        strict["created"] = True
        strict["session_providers"] = s.get_providers()
        import numpy as _np
        _a = _np.arange(4, dtype=_np.float32).reshape(1, 4)
        out = s.run(None, {"A": _a, "B": _np.ones((1, 4), _np.float32)})[0]
        strict["output_correct"] = bool(_np.allclose(out, _a + 1))
        del s
    except Exception as e:
        strict["error"] = f"{type(e).__name__}: {e}"
        strict["traceback"] = traceback.format_exc()
    sys.stderr.flush()
    plugin["strict_session"] = strict
    try:
        ort.unregister_execution_provider_library(name)
    except Exception as e:
        plugin["unregister_error"] = f"{type(e).__name__}: {e}"
res["plugin"] = plugin
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


AIHUB_QNN_VERSION = "2.50.0"  # QAIRT used to compile the Day 0 and Day 1 AI Hub context binaries


def compare_qnn(bundled: str | None) -> str:
    if not bundled:
        return f"bundled QNN version not found, cannot compare with AI Hub {AIHUB_QNN_VERSION}"
    if bundled == AIHUB_QNN_VERSION:
        return f"bundled QNN {bundled} matches AI Hub {AIHUB_QNN_VERSION}"
    same_minor = bundled.split(".")[:2] == AIHUB_QNN_VERSION.split(".")[:2]
    return (f"bundled QNN {bundled} vs AI Hub {AIHUB_QNN_VERSION}: mismatch"
            + (" at patch level, same major.minor" if same_minor else ""))


def plugin_summary(p: dict) -> str:
    if "import_error" in p:
        return f"plugin QNN EP: import onnxruntime_qnn FAILED, {p['import_error']}"
    parts = [f"plugin QNN EP (onnxruntime_qnn {p.get('onnxruntime_qnn_version')}): registration {p.get('registration')}"]
    devs = p.get("ep_devices", [])
    parts.append("get_ep_devices " + ", ".join(f"{d['ep_name']} on {d['hw_type']}" for d in devs) if devs
                 else f"get_ep_devices {p.get('ep_devices_error', 'empty')}")
    strict = p.get("strict_session", {})
    if strict.get("created"):
        parts.append(f"strict session on QNN devices {p.get('selected_qnn_devices')}: created, providers "
                     f"{strict.get('session_providers')}, output correct {strict.get('output_correct')}")
    else:
        parts.append(f"strict session on QNN devices {p.get('selected_qnn_devices')}: FAILED, "
                     f"{(strict.get('error') or '').splitlines()[0][:300] if strict.get('error') else 'no error text'}")
    parts.append(compare_qnn(p.get("bundled_qnn_version_from_package_info")))
    dll = (p.get("dll_version_info") or {}).get("QnnHtp.dll")
    if isinstance(dll, dict):
        parts.append(f"QnnHtp.dll version info {dll}")
    return ". ".join(parts)


def opencv_headless_wheel_check(work: Path) -> dict:
    """Informational only. Downloads, never installs."""
    dest = work / "opencv_headless_download"
    cmd = [sys.executable, "-m", "pip", "download", "opencv-python-headless", "--only-binary=:all:",
           "--platform", "win_arm64", "-d", str(dest)]
    r = run(cmd, PIP_TIMEOUT_S)
    out = r["stdout"] + "\n" + r["stderr"]
    wheels = sorted(p.name for p in dest.glob("opencv_python_headless*.whl")) if dest.exists() else []
    return {"command": " ".join(cmd[2:]), "rc": r["rc"], "wheels": wheels,
            "result": (f"wheel exists: {', '.join(wheels)}" if wheels else f"no wheel: {first_error_line(out)}"),
            "output_tail": "\n".join(out.strip().splitlines()[-15:])}


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
                notes.append(plugin_summary(res.get("plugin", {})))
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


def write_report(path: Path, results: list[dict], header: dict, fatal: str | None, info: dict | None = None) -> None:
    lines = ["# Windows ARM64 dependency report", ""]
    for k, v in header.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "Each package is installed alone in a fresh venv with `pip install <package>` and no version pin.", ""]
    lines += ["| package | install | import | version | notes |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {cell(r['package'])} | {r['install']} | {r['import']} | {cell(r['version'])} | {cell(r['notes'])} |")
    if info:
        lines += ["", "## Informational", ""]
        for key, value in info.items():
            lines.append(f"- {key}: `{value.get('command')}` gives {value.get('result')}")
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
    for key, value in (info or {}).items():
        lines += [f"### {key}", "", "<details><summary>output_tail</summary>", "", "```",
                  value.get("output_tail", "").strip(), "```", "", "</details>", ""]
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
    info: dict = {}
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
        print("--- informational: opencv-python-headless win_arm64 wheel check", flush=True)
        try:
            info["opencv-python-headless win_arm64 wheel"] = opencv_headless_wheel_check(work)
        except Exception:
            info["opencv-python-headless win_arm64 wheel"] = {"command": "pip download", "result": "check crashed",
                                                              "output_tail": traceback.format_exc()}
        print("    " + info["opencv-python-headless win_arm64 wheel"]["result"], flush=True)
    except Exception:
        fatal = traceback.format_exc()
        print(fatal, flush=True)
    finally:
        write_report(Path(args.out), results, header, fatal, info)
        print(f"report written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
