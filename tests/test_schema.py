import dataclasses
import json

import pytest

from benchmarks.schema import PRECISIONS, Derived, Measurement, Source, load_json, save_json


def make_measurement(**overrides):
    kwargs = dict(
        model="mediapipe_face",
        metric="inference_latency",
        value=1.5,
        unit="ms",
        source=Source.LOCAL_X86_CPU,
        runtime="onnx",
        compute_unit="CPU",
        precision="float32",
    )
    kwargs.update(overrides)
    return Measurement(**kwargs)


def test_source_has_exactly_three_values():
    assert {s.value for s in Source} == {
        "AI Hub hosted Snapdragon X Elite",
        "local x86 CPU",
        "physical Snapdragon device",
    }


def test_measurement_without_source_cannot_be_constructed():
    with pytest.raises(TypeError):
        Measurement(
            model="m",
            metric="latency",
            value=1.0,
            unit="ms",
            runtime="onnx",
            compute_unit="CPU",
            precision="float32",
        )


@pytest.mark.parametrize("bad_source", [None, "local x86 CPU"])
def test_measurement_source_must_be_a_source(bad_source):
    with pytest.raises(TypeError):
        make_measurement(source=bad_source)


@pytest.mark.parametrize("job_id", [None, "", "   "])
def test_aihub_without_job_id_raises(job_id):
    with pytest.raises(ValueError):
        make_measurement(source=Source.AIHUB_X_ELITE, compute_unit="NPU", job_id=job_id)


def test_aihub_with_job_id_is_valid():
    m = make_measurement(source=Source.AIHUB_X_ELITE, compute_unit="NPU", job_id="jabc123")
    assert m.job_id == "jabc123"


def test_local_measurement_does_not_need_job_id():
    assert make_measurement().job_id is None


@pytest.mark.parametrize("formula", ["", "   "])
def test_derived_with_empty_formula_raises(formula):
    with pytest.raises(ValueError):
        Derived(name="fps", value=10.0, unit="fps", formula=formula, inputs=[make_measurement()])


def test_derived_without_inputs_raises():
    with pytest.raises(ValueError):
        Derived(name="fps", value=10.0, unit="fps", formula="1000 / latency_ms", inputs=[])


def test_derived_label_starts_with_derived():
    d = Derived(
        name="fps",
        value=666.67,
        unit="fps",
        formula="1000 / latency_ms",
        inputs=[make_measurement()],
    )
    assert d.label.startswith("Derived:")
    assert "1000 / latency_ms" in d.label


def test_json_round_trip_preserves_all_fields(tmp_path):
    aihub = make_measurement(
        source=Source.AIHUB_X_ELITE,
        compute_unit="NPU",
        precision="w8a8",
        runtime="precompiled_qnn_onnx",
        qnn_version="2.50.0",
        ort_version="1.27.1",
        job_id="jabc123",
        timestamp="2026-09-24T10:00:00+00:00",
        notes="profile job",
    )
    local = make_measurement(value=7, notes="int value")
    derived = Derived(
        name="fps",
        value=666.67,
        unit="fps",
        formula="1000 / latency_ms",
        inputs=[aihub, local],
    )
    records = [aihub, local, derived]

    path = tmp_path / "records.json"
    save_json(records, path)
    loaded = load_json(path)

    assert loaded == records
    for original, restored in zip(records, loaded):
        assert type(restored) is type(original)
        for f in dataclasses.fields(original):
            assert getattr(restored, f.name) == getattr(original, f.name), f.name
    assert loaded[0].source is Source.AIHUB_X_ELITE
    assert isinstance(loaded[1].value, float)
    assert loaded[2].inputs[0].job_id == "jabc123"


def test_allowed_precisions():
    assert PRECISIONS == {"float32", "float16", "w8a8", "w8a16", "w4a16", "int8", "unknown"}
    for precision in sorted(PRECISIONS):
        assert make_measurement(precision=precision).precision == precision


@pytest.mark.parametrize("precision", ["", "float", "fp16", "FLOAT32", None])
def test_invalid_precision_raises(precision):
    with pytest.raises(ValueError):
        make_measurement(precision=precision)


@pytest.mark.parametrize("runtime", ["", "   ", None])
def test_empty_runtime_raises(runtime):
    with pytest.raises(ValueError):
        make_measurement(runtime=runtime)


def test_versions_default_to_none():
    m = make_measurement()
    assert m.qnn_version is None and m.ort_version is None


def test_load_old_format_record(tmp_path):
    old = {
        "kind": "measurement",
        "model": "mediapipe_face_float_face_detector",
        "metric": "estimated_inference_time",
        "value": 683.0,
        "unit": "us",
        "source": "AI Hub hosted Snapdragon X Elite",
        "compute_unit": "NPU",
        "precision": "float",
        "job_id": "jp2rrvw4g",
        "timestamp": "2026-09-24T18:33:05",
        "notes": "https://workbench.aihub.qualcomm.com/jobs/jp2rrvw4g/",
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps([old]), encoding="utf-8")

    [m] = load_json(path)

    assert m.runtime == "unknown"
    assert m.precision == "unknown"
    assert "legacy precision: 'float'" in m.notes
    assert m.notes.startswith(old["notes"])
    assert m.qnn_version is None and m.ort_version is None
    assert (m.value, m.job_id, m.timestamp) == (683.0, "jp2rrvw4g", "2026-09-24T18:33:05")


def test_load_old_format_record_keeps_valid_precision(tmp_path):
    old = {
        "kind": "measurement",
        "model": "m",
        "metric": "latency",
        "value": 1.0,
        "unit": "ms",
        "source": "local x86 CPU",
        "compute_unit": "CPU",
        "precision": "float32",
        "job_id": None,
        "timestamp": "2026-09-24T00:00:00+00:00",
        "notes": "",
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps([old]), encoding="utf-8")

    [m] = load_json(path)

    assert m.runtime == "unknown"
    assert m.precision == "float32"
    assert m.notes == ""
