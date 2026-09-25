"""ModelRunner: the one loader every model goes through.

Load order per model (models/manifest.json lists the files):
1. precompiled_qnn_onnx on a QNN EP NPU device, through the onnxruntime-qnn plugin EP API
   (register_execution_provider_library, get_ep_devices, add_provider_for_devices, as in
   the onnxruntime/onnxruntime-qnn README "Getting Started with the Plugin QNN EP")
2. plain onnx on the QNN EP NPU device with EP context caching. Session config keys
   "ep.context_enable" = "1" and "ep.context_file_path", from the onnxruntime-qnn docs
   "QNN context binary cache feature". A cached context model is loaded when present.
3. onnx on CPUExecutionProvider

Every session gets the thread settings in app.config.RUNTIME: intra_op_num_threads per
model, inter_op_num_threads and session.intra_op.allow_spinning.

Both QNN attempts set session.disable_cpu_ep_fallback=1, so a partial CPU fallback fails
the attempt instead of passing silently. If no QNN NPU device is listed, both QNN attempts
are skipped with that reason. After every session is created, the providers it reports
are checked against the step (CLAUDE.md rule 11), and status() reports the real compute
unit for the UI.

Nothing here has been validated on a Snapdragon device yet. On machines without an NPU,
every model lands on CPU with the reason recorded.

Usage: python -m app.runtime.runner --check [--save PATH]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import RUNTIME, RuntimeConfig
from app.storage import paths

REPO = Path(__file__).resolve().parents[2]  # in the packaged app, its _internal folder
MANIFEST = REPO / "models" / "manifest.json"
# The packaged app can sit in a read-only folder, so it keeps the EP context cache with the
# user's data (%LOCALAPPDATA%\PlacementMirror).
CACHE_DIR = (paths.data_root() if getattr(sys, "frozen", False) else REPO / "models") / "ep_context_cache"
QNN_EP = "QNNExecutionProvider"
STRICT = {"session.disable_cpu_ep_fallback": "1"}
STEP_PRECOMPILED = "precompiled_qnn_onnx on QNN NPU"
STEP_ONNX_QNN = "onnx on QNN NPU with EP context cache"
STEP_CPU = "onnx on CPU"
# Models that run on CPU by design, with the reason.
CPU_ONLY = {"silero_vad": "CPU by design (docs/architecture.md VAD row)"}


def first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}"[:400]


class QnnBackend:
    """Finds the onnxruntime-qnn plugin EP and its NPU devices. Records why when it cannot."""

    def __init__(self, ort=None):
        import onnxruntime

        self.ort = ort or onnxruntime
        self.devices: list = []
        self.listing: list[str] = []
        self.htp_path: str | None = None
        self.reason: str | None = None
        try:
            import onnxruntime_qnn as qnn_ep
        except ImportError as e:
            self.reason = f"onnxruntime_qnn not installed ({e})"
            return
        if not hasattr(self.ort, "register_execution_provider_library"):
            self.reason = f"onnxruntime {self.ort.__version__} has no plugin EP API"
            return
        try:
            self.ort.register_execution_provider_library(QNN_EP, qnn_ep.get_library_path())
        except Exception as e:  # noqa: BLE001
            self.reason = f"QNN EP registration failed: {first_line(e)}"
            return
        self.htp_path = qnn_ep.get_qnn_htp_path()
        all_devices = self.ort.get_ep_devices()
        self.listing = [f"{d.ep_name} on {str(d.device.type).split('.')[-1]}" for d in all_devices]
        qnn = [d for d in all_devices if d.ep_name == QNN_EP]
        self.devices = [d for d in qnn if str(d.device.type).split(".")[-1] == "NPU"]
        if not self.devices:
            self.reason = ("No QNN devices listed" if not qnn else
                           "No QNN NPU device listed (QNN lists " + ", ".join(
                               str(d.device.type).split(".")[-1] for d in qnn) + ")")

    @property
    def available(self) -> bool:
        return bool(self.devices)

    def session(self, path: Path, config: dict[str, str], so=None):
        so = so if so is not None else self.ort.SessionOptions()
        for key, value in {**STRICT, **config}.items():
            so.add_session_config_entry(key, value)
        so.add_provider_for_devices(self.devices, {"backend_path": self.htp_path})
        return self.ort.InferenceSession(str(path), sess_options=so)


@dataclass
class Attempt:
    step: str
    ok: bool
    reason: str | None
    seconds: float


@dataclass
class ModelStatus:
    name: str
    path: str | None = None
    compute_unit: str | None = None
    providers: list[str] = field(default_factory=list)
    runtime: str | None = None
    precision: str | None = None
    model_file: str | None = None
    ep_context_cache: str | None = None
    threads: dict = field(default_factory=dict)
    load_s: float | None = None
    total_load_s: float | None = None
    attempts: list[Attempt] = field(default_factory=list)

    def reason(self) -> str:
        """Why earlier steps were not used, with steps that share a reason grouped."""
        grouped: dict[str, list[str]] = {}
        for a in self.attempts:
            if not a.ok:
                grouped.setdefault(a.reason or "", []).append(a.step)
        if not grouped:
            return "first choice"
        return ". ".join(f"{' and '.join(steps)}: {reason}" for reason, steps in grouped.items())


class ModelRunner:
    def __init__(self, manifest: Path = MANIFEST, qnn: QnnBackend | None = None, cache_dir: Path = CACHE_DIR,
                 runtime: RuntimeConfig = RUNTIME):
        import onnxruntime

        self.ort = onnxruntime
        self.root = Path(manifest).resolve().parent.parent
        entries = json.loads(Path(manifest).read_text(encoding="utf-8"))["models"]
        self.variants: dict[str, dict[str, dict]] = {}
        for e in entries:
            self.variants.setdefault(e["name"], {})[e["runtime"]] = e
        self.qnn = qnn if qnn is not None else QnnBackend(onnxruntime)
        self.cache_dir = Path(cache_dir)
        self.runtime = runtime
        self.sessions: dict[str, Any] = {}
        self.statuses: dict[str, ModelStatus] = {}

    def names(self) -> list[str]:
        return list(self.variants)

    def _file(self, name: str, runtime: str) -> Path:
        entry = self.variants[name].get(runtime)
        if entry is None:
            raise FileNotFoundError(f"no {runtime} model for {name} in the manifest")
        path = self.root / entry["onnx"]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing, run python -m aihub.fetch_models")
        return path

    def options(self, name: str):
        """SessionOptions with the thread settings for this model (app.config.RUNTIME)."""
        so = self.ort.SessionOptions()
        so.intra_op_num_threads = self.runtime.intra_op_threads.get(name, 0)
        so.inter_op_num_threads = self.runtime.inter_op_threads
        so.add_session_config_entry("session.intra_op.allow_spinning", self.runtime.intra_op_allow_spinning)
        return so

    def _threads(self, name: str) -> dict:
        return {"intra_op_num_threads": self.runtime.intra_op_threads.get(name, 0),
                "inter_op_num_threads": self.runtime.inter_op_threads,
                "session.intra_op.allow_spinning": self.runtime.intra_op_allow_spinning}

    def _precompiled(self, name: str, st: ModelStatus):
        path = self._file(name, "precompiled_qnn_onnx")
        st.runtime, st.model_file = "precompiled_qnn_onnx", str(path)
        return self.qnn.session(path, {}, self.options(name))

    def _onnx_qnn(self, name: str, st: ModelStatus):
        path = self._file(name, "onnx")
        cache = self.cache_dir / f"{name}_ctx.onnx"
        st.runtime, st.ep_context_cache = "onnx", str(cache)
        if cache.exists():
            st.model_file = str(cache)
            return self.qnn.session(cache, {}, self.options(name))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        st.model_file = str(path)
        return self.qnn.session(path, {"ep.context_enable": "1", "ep.context_file_path": str(cache)},
                                self.options(name))

    def _cpu(self, name: str, st: ModelStatus):
        path = self._file(name, "onnx")
        st.runtime, st.model_file = "onnx", str(path)
        return self.ort.InferenceSession(str(path), sess_options=self.options(name), providers=["CPUExecutionProvider"])

    def load(self, name: str) -> ModelStatus:
        if name in self.statuses and name in self.sessions:
            return self.statuses[name]
        st = ModelStatus(name=name, threads=self._threads(name))
        start = time.perf_counter()
        steps = [(STEP_PRECOMPILED, self._precompiled, QNN_EP), (STEP_ONNX_QNN, self._onnx_qnn, QNN_EP),
                 (STEP_CPU, self._cpu, "CPUExecutionProvider")]
        for step, make, expected in steps:
            if expected == QNN_EP and name in CPU_ONLY:
                st.attempts.append(Attempt(step, False, f"skipped: {CPU_ONLY[name]}", 0.0))
                continue
            if expected == QNN_EP and not self.qnn.available:
                st.attempts.append(Attempt(step, False, f"skipped: {self.qnn.reason}", 0.0))
                continue
            t0 = time.perf_counter()
            try:
                session = make(name, st)
                providers = list(session.get_providers())
                if expected not in providers or (expected == "CPUExecutionProvider" and providers != [expected]):
                    raise RuntimeError(f"session reports providers {providers}, expected {expected}")
            except Exception as e:  # noqa: BLE001
                st.attempts.append(Attempt(step, False, first_line(e), time.perf_counter() - t0))
                continue
            st.load_s = time.perf_counter() - t0
            st.attempts.append(Attempt(step, True, None, st.load_s))
            st.path, st.providers = step, providers
            st.compute_unit = "NPU" if expected == QNN_EP else "CPU"
            st.precision = self.variants[name].get(st.runtime, {}).get("precision")
            self.sessions[name] = session
            break
        st.total_load_s = time.perf_counter() - start
        self.statuses[name] = st
        if name not in self.sessions:
            raise RuntimeError(f"{name}: every load step failed: {st.reason()}")
        return st

    def status(self) -> dict[str, dict]:
        out = {}
        for name, st in self.statuses.items():
            d = asdict(st)
            d["reason"] = st.reason()
            out[name] = d
        return out

    def input_specs(self, name: str) -> list[tuple[str, list, str]]:
        self.load(name)
        return [(i.name, i.shape, i.type) for i in self.sessions[name].get_inputs()]

    def run(self, name: str, inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
        self.load(name)
        session = self.sessions[name]
        t0 = time.perf_counter()
        outputs = session.run(None, inputs)
        wall = time.perf_counter() - t0
        return {o.name: v for o, v in zip(session.get_outputs(), outputs)}, wall


def _source():
    from benchmarks.schema import Source

    return Source.LOCAL_X86_CPU if platform.machine().upper() in ("AMD64", "X86_64") else Source.PHYSICAL_SNAPDRAGON


def check(save: Path | None) -> int:
    runner = ModelRunner()
    print(f"onnxruntime {runner.ort.__version__}, QNN: "
          f"{'NPU devices ' + str(len(runner.qnn.devices)) if runner.qnn.available else runner.qnn.reason}")
    if runner.qnn.listing:
        print("get_ep_devices: " + ", ".join(runner.qnn.listing))
    rows, failed = [], []
    for name in runner.names():
        try:
            st = runner.load(name)
        except RuntimeError as e:
            failed.append(str(e))
            continue
        rows.append(st)
    print("\n| model | path taken | compute unit | reason earlier steps were not used | load ms |")
    print("|---|---|---|---|---|")
    for st in rows:
        print(f"| {st.name} | {st.path} | {st.compute_unit} | {st.reason()} | {st.load_s * 1000:,.1f} |")
    for f in failed:
        print(f"FAILED: {f}")
    src = _source()
    print(f"\nLoad times: Source {src.value}. {len(rows)} of {len(runner.names())} models loaded.")
    if save:
        from benchmarks.schema import Measurement, save_json

        save_json([Measurement(
            model=st.name, metric="session_load_time", value=st.load_s * 1000, unit="ms", source=src,
            runtime=st.runtime, compute_unit=st.compute_unit, precision=st.precision or "unknown",
            ort_version=runner.ort.__version__,
            notes=(f"ModelRunner --check, path {st.path}, providers {st.providers}, earlier steps: {st.reason()}, "
                   f"file {Path(st.model_file).relative_to(REPO).as_posix()}, CPU {platform.processor()}"),
        ) for st in rows], save)
        print(f"Saved {len(rows)} load time measurements to {save}")
    return 0 if not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="load every model in the manifest and print a table")
    ap.add_argument("--save", type=Path, help="with --check, save load times as Measurement records here")
    args = ap.parse_args()
    if not args.check:
        ap.print_help()
        return 2
    return check(args.save)


if __name__ == "__main__":
    sys.exit(main())
