import dataclasses

import pytest

from benchmarks.schema import Derived, Measurement, Source, load_json, save_json


def make_measurement(**overrides):
    kwargs = dict(
        model="mediapipe_face",
        metric="inference_latency",
        value=1.5,
        unit="ms",
        source=Source.LOCAL_X86_CPU,
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
