import json
from pathlib import Path

import numpy as np
import pytest

from app.runtime.runner import STEP_CPU, STEP_ONNX_QNN, STEP_PRECOMPILED, ModelRunner, QnnBackend
from ci.check_deps import build_add_model

A = np.arange(4, dtype=np.float32).reshape(1, 4)
B = np.ones((1, 4), dtype=np.float32)
FALLBACK_ERROR = ("[ONNXRuntimeError] : 1 : FAIL : This session contains graph nodes that are assigned to the "
                  "default CPU EP, but fallback to CPU EP has been explicitly disabled by the user.")


def make_manifest(tmp_path: Path, names=("m1",), runtimes=("onnx", "precompiled_qnn_onnx")) -> Path:
    models = []
    for name in names:
        for runtime in runtimes:
            folder = tmp_path / "models" / name / runtime
            folder.mkdir(parents=True)
            (folder / "model.onnx").write_bytes(build_add_model())
            models.append({"name": name, "runtime": runtime, "precision": "float32",
                           "onnx": f"models/{name}/{runtime}/model.onnx"})
    path = tmp_path / "models" / "manifest.json"
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return path


class FakeSession:
    """A real CPU session that reports the providers we tell it to."""

    def __init__(self, path, providers):
        import onnxruntime as ort

        self._s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._providers = list(providers)

    def get_providers(self):
        return self._providers

    def __getattr__(self, item):
        return getattr(self._s, item)


class FakeQnn:
    def __init__(self, available=True, reason=None, error=None, providers=("QNNExecutionProvider", "CPUExecutionProvider")):
        self.available = available
        self.reason = reason
        self.devices = ["npu"] if available else []
        self.listing = []
        self.error = error
        self.providers = providers
        self.calls = []

    def session(self, path, config):
        self.calls.append((Path(path), dict(config)))
        if self.error:
            raise self.error
        return FakeSession(path, self.providers)


def test_without_qnn_every_model_lands_on_cpu_with_reason(tmp_path):
    qnn = FakeQnn(available=False, reason="No QNN devices listed")
    runner = ModelRunner(make_manifest(tmp_path, names=("a", "b", "c")), qnn=qnn, cache_dir=tmp_path / "cache")
    for name in runner.names():
        st = runner.load(name)
        assert st.path == STEP_CPU and st.compute_unit == "CPU"
        assert st.providers == ["CPUExecutionProvider"]
        assert [a.step for a in st.attempts] == [STEP_PRECOMPILED, STEP_ONNX_QNN, STEP_CPU]
        assert all("No QNN devices listed" in a.reason for a in st.attempts[:2])
        assert "No QNN devices listed" in st.reason()
        assert st.load_s is not None and st.total_load_s >= st.load_s
    assert qnn.calls == []  # no QNN devices: straight to CPU


def test_real_backend_on_this_machine_lands_on_cpu_with_reason(tmp_path):
    backend = QnnBackend()
    if backend.available:
        pytest.skip("this machine lists a QNN NPU device")
    runner = ModelRunner(make_manifest(tmp_path), qnn=backend, cache_dir=tmp_path / "cache")
    st = runner.load("m1")
    assert st.compute_unit == "CPU" and st.path == STEP_CPU
    assert backend.reason and backend.reason in st.reason()


def test_qnn_failures_fall_back_to_cpu_with_each_reason(tmp_path):
    qnn = FakeQnn(error=RuntimeError(FALLBACK_ERROR))
    runner = ModelRunner(make_manifest(tmp_path), qnn=qnn, cache_dir=tmp_path / "cache")
    st = runner.load("m1")
    assert st.path == STEP_CPU and st.compute_unit == "CPU"
    failed = [a for a in st.attempts if not a.ok]
    assert [a.step for a in failed] == [STEP_PRECOMPILED, STEP_ONNX_QNN]
    assert all("fallback to CPU EP has been explicitly disabled" in a.reason for a in failed)
    (p1, c1), (p2, c2) = qnn.calls
    assert p1.parent.name == "precompiled_qnn_onnx" and c1 == {}
    assert p2.parent.name == "onnx"
    assert c2 == {"ep.context_enable": "1", "ep.context_file_path": str(tmp_path / "cache" / "m1_ctx.onnx")}


def test_precompiled_on_qnn_reports_npu_and_runs(tmp_path):
    runner = ModelRunner(make_manifest(tmp_path), qnn=FakeQnn(), cache_dir=tmp_path / "cache")
    st = runner.load("m1")
    assert st.path == STEP_PRECOMPILED and st.compute_unit == "NPU" and st.runtime == "precompiled_qnn_onnx"
    assert st.reason() == "first choice"
    out, wall = runner.run("m1", {"A": A, "B": B})
    assert np.allclose(out["C"], A + B) and wall >= 0


def test_session_without_qnn_provider_counts_as_failure(tmp_path):
    runner = ModelRunner(make_manifest(tmp_path), qnn=FakeQnn(providers=("CPUExecutionProvider",)),
                         cache_dir=tmp_path / "cache")
    st = runner.load("m1")
    assert st.compute_unit == "CPU" and st.path == STEP_CPU
    assert "session reports providers ['CPUExecutionProvider']" in st.reason()


def test_missing_precompiled_then_cached_ep_context_is_loaded(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "m1_ctx.onnx").write_bytes(build_add_model())
    qnn = FakeQnn()
    runner = ModelRunner(make_manifest(tmp_path, runtimes=("onnx",)), qnn=qnn, cache_dir=cache)
    st = runner.load("m1")
    assert st.path == STEP_ONNX_QNN and st.compute_unit == "NPU"
    assert "no precompiled_qnn_onnx model for m1" in st.attempts[0].reason
    assert qnn.calls == [(cache / "m1_ctx.onnx", {})]


def test_cpu_only_model_skips_qnn(tmp_path):
    qnn = FakeQnn()
    runner = ModelRunner(make_manifest(tmp_path, names=("silero_vad",), runtimes=("onnx",)), qnn=qnn,
                         cache_dir=tmp_path / "cache")
    st = runner.load("silero_vad")
    assert st.compute_unit == "CPU" and "CPU by design" in st.reason()
    assert qnn.calls == []


def test_status_is_json_serializable(tmp_path):
    runner = ModelRunner(make_manifest(tmp_path), qnn=FakeQnn(available=False, reason="No QNN devices listed"),
                         cache_dir=tmp_path / "cache")
    runner.load("m1")
    status = json.loads(json.dumps(runner.status()))
    assert status["m1"]["compute_unit"] == "CPU" and "No QNN devices listed" in status["m1"]["reason"]
