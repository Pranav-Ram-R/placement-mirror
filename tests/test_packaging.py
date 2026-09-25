import hashlib
import http.server
import json
import socket
import threading
import zipfile
from importlib import metadata

import pytest

from app import launcher
from tools import build_app, fetch_models, package_models, third_party_licenses


def fake_models(root, content=b"weights"):
    files = [("models/a/onnx/model.onnx", content), ("models/b/onnx/model.onnx", b"other")]
    for rel, data in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    manifest = {"models": [{"name": rel.split("/")[1], "files": [
        {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]} for rel, data in files]}
    (root / "models" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_models_zip_is_deterministic_and_checked_against_the_manifest(tmp_path):
    fake_models(tmp_path)
    a = package_models.build_zip(tmp_path / "a.zip", tmp_path)
    b = package_models.build_zip(tmp_path / "b.zip", tmp_path)
    assert a == b  # same files, same zip
    with zipfile.ZipFile(tmp_path / "a.zip") as zf:
        assert zf.namelist() == ["models/manifest.json", "models/a/onnx/model.onnx", "models/b/onnx/model.onnx"]
        assert {i.date_time for i in zf.infolist()} == {package_models.FIXED_TIME}
    (tmp_path / "models/a/onnx/model.onnx").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="Missing or changed"):
        package_models.build_zip(tmp_path / "c.zip", tmp_path)


def test_fetch_checks_the_hash_and_extracts_only_model_paths(tmp_path, monkeypatch):
    fake_models(tmp_path)
    digest = package_models.build_zip(tmp_path / "m.zip", tmp_path)
    hash_file = tmp_path / "models-v1.sha256"
    monkeypatch.setattr(fetch_models, "HASH_FILE", hash_file)
    hash_file.write_text("0" * 64 + "  models-v1.zip\n")
    monkeypatch.setattr("sys.argv", ["fetch", "--zip", str(tmp_path / "m.zip"), "--dest", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="expects"):
        fetch_models.main()
    hash_file.write_text(f"{digest}  models-v1.zip\n")
    assert fetch_models.main() == 0
    assert (tmp_path / "out/models/a/onnx/model.onnx").read_bytes() == b"weights"
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(SystemExit, match="Unexpected path"):
        fetch_models.extract(tmp_path / "bad.zip", tmp_path / "out")


def test_launcher_skips_busy_ports_and_opens_the_browser_after_health(monkeypatch):
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    port = busy.getsockname()[1]
    assert launcher.free_port(port) > port

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok", "mode_label": "CPU fallback mode, reduced frame rate"}).encode()
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    opened = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    launcher.open_when_ready(server.server_address[1], timeout_s=10)
    server.shutdown()
    busy.close()
    assert opened == [f"http://127.0.0.1:{server.server_address[1]}/"]


class StatusRunner:
    def __init__(self, unit):
        self.unit = unit

    def status(self):
        names = ("face_detector", "face_landmark", "pose_detector", "pose_landmark", "silero_vad")
        return {n: {"compute_unit": "CPU" if n == "silero_vad" else self.unit, "reason": "test"} for n in names}


def test_health_reports_the_mode(monkeypatch):
    from app import server

    monkeypatch.setitem(server.STATE, "runner", StatusRunner("CPU"))
    cpu = json.loads(server.health().body)
    assert cpu["status"] == "ok" and cpu["mode"] == "cpu_fallback"
    assert cpu["mode_label"] == "CPU fallback mode, reduced frame rate" and cpu["notice"]
    monkeypatch.setitem(server.STATE, "runner", StatusRunner("NPU"))
    npu = json.loads(server.health().body)
    assert npu["mode_label"] == "NPU mode" and npu["notice"] is None  # silero_vad is CPU by design


def test_third_party_licenses_cover_the_models_and_every_package_given():
    text, _ = third_party_licenses.render([metadata.distribution("wsproto"), metadata.distribution("h11")])
    for heading in ("## Models", "### MediaPipe face detector and face landmark models",
                    "### MediaPipe pose detector and pose landmark models", "### Whisper tiny encoder and decoder",
                    "### Silero VAD", "## Native libraries", "### Qualcomm AI Engine Direct (QNN) libraries",
                    "### PortAudio", "## Python packages", "### wsproto 1.3.2", "### h11 0.16.0"):
        assert heading in text
    assert "models/whisper_tiny_decoder/onnx/model.onnx" in text and "Copyright (c) 2020-present Silero Team" in text
    assert "—" not in text.split("## Python packages")[0]  # our own text (CLAUDE.md rule 8)


def test_requirement_closure_includes_every_requirement():
    names = {d.metadata["Name"].lower() for d in
             third_party_licenses.dependency_closure(third_party_licenses.requirement_roots())}
    assert {"numpy", "onnxruntime-qnn", "onnxruntime", "fastapi", "uvicorn", "sounddevice", "wsproto", "h11"} <= names


def test_bundled_modules_are_read_from_the_pyinstaller_toc(tmp_path):
    (tmp_path / build_app.NAME).mkdir()
    toc = [("numpy.core", "C:/x/numpy/core/__init__.py", "PYMODULE"), ("wsproto", "C:/x/wsproto.py", "PYMODULE"),
           ("_socket", "C:/x/_socket.pyd", "EXTENSION"), ("app\\ui\\static\\index.html", "C:/x", "DATA")]
    (tmp_path / build_app.NAME / "PYZ-00.toc").write_text(repr(("PYZ", [toc])), encoding="utf-8")
    assert build_app.bundled_modules(tmp_path) == {"numpy", "wsproto", "_socket"}
