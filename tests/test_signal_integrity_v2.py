import json
from pathlib import Path

import numpy as np
import pytest

from neuros.contracts import (
    ClockDomain,
    QualityFlag,
    SignalFrame,
    StreamDescriptor,
    frame_channel_count,
    validate_frame_against_descriptor,
)
from neuros.recording import ARCHIVE_SCHEMA_VERSION, SessionArchiveReader, SessionArchiveWriter


def _frame(**overrides):
    values = {
        "stream_id": "eeg",
        "sequence_id": 3,
        "data": np.arange(8, dtype=np.float32),
        "sample_rate_hz": 250.0,
        "host_receive_time_ns": 100,
        "device_time_ns": 200,
        "synchronized_time_ns": 300,
        "clock_domain": ClockDomain.SYNCHRONIZED,
        "quality": QualityFlag.GOOD,
        "metadata": {"modality": "eeg"},
    }
    values.update(overrides)
    return SignalFrame(**values)


def _descriptor(**overrides):
    values = {
        "stream_id": "eeg",
        "modality": "eeg",
        "sample_rate_hz": 250.0,
        "channel_names": tuple(f"C{index}" for index in range(8)),
        "channel_types": ("eeg",) * 8,
        "units": ("uV",) * 8,
        "clock_domain": ClockDomain.SYNCHRONIZED,
        "metadata": {"reference": "common-average", "nested": {"version": 1}},
    }
    values.update(overrides)
    return StreamDescriptor(**values)


def test_signal_frame_detaches_and_freezes_sample_buffer():
    source = np.arange(8, dtype=np.float32)
    frame = _frame(data=source)

    source[0] = 999.0
    assert frame.data[0] == 0.0
    assert not frame.data.flags.writeable
    with pytest.raises(ValueError):
        frame.data[0] = 1.0


def test_signal_frame_recursively_detaches_metadata():
    metadata = {"nested": {"labels": ["left", "right"]}}
    frame = _frame(metadata=metadata)

    metadata["nested"]["labels"][0] = "changed"
    assert frame.metadata["nested"]["labels"] == ("left", "right")
    with pytest.raises(TypeError):
        frame.metadata["nested"]["new"] = 1


def test_metadata_requires_mapping_at_contract_root():
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        _frame(metadata=[("modality", "eeg")])
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        _descriptor(metadata=[("reference", "average")])


def test_provenance_metadata_rejects_unstable_python_values():
    with pytest.raises(TypeError, match="bytes"):
        _descriptor(metadata={"blob": b"opaque"})
    with pytest.raises(TypeError, match="set"):
        _descriptor(metadata={"unordered": {"C3", "C4"}})
    with pytest.raises(ValueError, match="non-finite"):
        _descriptor(metadata={"value": float("nan")})


def test_numpy_metadata_is_detached_into_archive_stable_values():
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    descriptor = _descriptor(metadata={"coordinates": source})
    before = descriptor.fingerprint()
    source[0, 0] = 99.0
    assert descriptor.metadata["coordinates"] == ((1.0, 2.0), (3.0, 4.0))
    assert descriptor.fingerprint() == before


@pytest.mark.parametrize("bad_rate", [float("nan"), float("inf"), 0.0, -1.0])
def test_sample_rates_must_be_finite_and_positive(bad_rate):
    with pytest.raises(ValueError):
        _descriptor(sample_rate_hz=bad_rate)
    with pytest.raises(ValueError):
        _frame(sample_rate_hz=bad_rate)


@pytest.mark.parametrize("bad_rate", [True, "250"])
def test_sample_rates_do_not_accept_boolean_or_string_coercion(bad_rate):
    with pytest.raises(TypeError):
        _descriptor(sample_rate_hz=bad_rate)
    with pytest.raises(TypeError):
        _frame(sample_rate_hz=bad_rate)


@pytest.mark.parametrize("field", ["sequence_id", "host_receive_time_ns", "device_time_ns"])
@pytest.mark.parametrize("bad_value", [True, 1.5])
def test_frame_integer_identity_fields_reject_lossy_coercion(field, bad_value):
    with pytest.raises(TypeError):
        _frame(**{field: bad_value})


def test_quality_flags_do_not_accept_lossy_or_undefined_states():
    with pytest.raises(TypeError):
        _frame(quality=1.5)
    with pytest.raises(TypeError):
        _frame(quality=True)
    with pytest.raises(ValueError, match="undefined"):
        _frame(quality=1 << 20)


def test_declared_clock_domain_requires_corresponding_timestamp():
    with pytest.raises(ValueError, match="device_time_ns"):
        _frame(clock_domain=ClockDomain.DEVICE, device_time_ns=None)
    with pytest.raises(ValueError, match="synchronized_time_ns"):
        _frame(clock_domain=ClockDomain.SYNCHRONIZED, synchronized_time_ns=None)


def test_timestamp_authority_follows_declared_clock_domain():
    assert _frame(clock_domain=ClockDomain.DEVICE).timestamp_ns == 200
    assert _frame(clock_domain=ClockDomain.HOST_MONOTONIC).timestamp_ns == 100
    assert _frame(clock_domain=ClockDomain.SYNCHRONIZED).timestamp_ns == 300


def test_legacy_conversion_places_timestamp_in_declared_domain():
    synchronized = SignalFrame.from_legacy(
        stream_id="legacy",
        sequence_id=0,
        timestamp_seconds=1.25,
        data=np.ones(2),
        sample_rate_hz=100.0,
        clock_domain=ClockDomain.SYNCHRONIZED,
    )
    assert synchronized.synchronized_time_ns == 1_250_000_000
    assert synchronized.device_time_ns is None
    assert synchronized.timestamp_ns == 1_250_000_000


def test_frame_rejects_nonfinite_and_object_signal_payloads():
    with pytest.raises(ValueError, match="finite"):
        _frame(data=np.array([1.0, np.nan]))
    with pytest.raises(TypeError, match="numeric dtype"):
        _frame(data=np.array(["C3", "C4"], dtype=object))


def test_stream_descriptor_requires_unique_nonempty_channel_identity():
    with pytest.raises(ValueError, match="unique"):
        _descriptor(
            channel_names=("C3", "C3"),
            channel_types=("eeg", "eeg"),
            units=("uV", "uV"),
        )
    with pytest.raises(ValueError, match="non-empty"):
        _descriptor(
            channel_names=("C3", ""),
            channel_types=("eeg", "eeg"),
            units=("uV", "uV"),
        )


def test_descriptor_fingerprint_is_deterministic_and_semantically_sensitive():
    first = _descriptor(metadata={"a": 1, "b": {"x": [1, 2]}})
    reordered = _descriptor(metadata={"b": {"x": (1, 2)}, "a": 1})
    changed = _descriptor(metadata={"a": 1, "b": {"x": [1, 3]}})

    assert first.fingerprint() == reordered.fingerprint()
    assert first.fingerprint() != changed.fingerprint()
    assert len(first.fingerprint()) == 64


def test_multidimensional_channel_geometry_is_never_guessed():
    ambiguous = _frame(data=np.zeros((5, 8)), metadata={"modality": "eeg"})
    with pytest.raises(ValueError, match="axis_order"):
        frame_channel_count(ambiguous)

    explicit = _frame(
        data=np.zeros((5, 8)),
        metadata={"modality": "eeg", "axis_order": ("sample", "channel")},
    )
    assert frame_channel_count(explicit) == 8


def test_descriptor_frame_validator_binds_stream_rate_clock_and_channels():
    descriptor = _descriptor()
    valid = _frame()
    validate_frame_against_descriptor(descriptor, valid)

    with pytest.raises(ValueError, match="stream_id"):
        validate_frame_against_descriptor(descriptor, _frame(stream_id="other"))
    with pytest.raises(ValueError, match="sample_rate"):
        validate_frame_against_descriptor(descriptor, _frame(sample_rate_hz=200.0))
    with pytest.raises(ValueError, match="clock_domain"):
        validate_frame_against_descriptor(
            descriptor,
            _frame(clock_domain=ClockDomain.DEVICE),
        )

    wrong_channels = _frame(data=np.zeros(7, dtype=np.float32))
    with pytest.raises(ValueError, match="channel geometry"):
        validate_frame_against_descriptor(descriptor, wrong_channels)


def test_descriptor_frame_validator_checks_declared_channel_names():
    descriptor = _descriptor()
    frame = _frame(metadata={"channel_names": tuple(reversed(descriptor.channel_names))})
    with pytest.raises(ValueError, match="channel_names"):
        validate_frame_against_descriptor(descriptor, frame)


@pytest.mark.asyncio
async def test_archive_v2_persists_and_verifies_descriptor_identity(tmp_path: Path):
    descriptor = _descriptor(metadata={"coordinates": np.arange(4).reshape(2, 2)})
    frame = _frame()
    root = tmp_path / "session"

    writer = SessionArchiveWriter(root, session_id="identity")
    writer.register_stream(descriptor)
    await writer.write(frame)
    await writer.close()

    reader = SessionArchiveReader(root)
    restored_descriptor = reader.descriptor("eeg")
    restored_frame = list(reader.iter_frames("eeg"))[0]
    assert reader.schema_version == ARCHIVE_SCHEMA_VERSION == 2
    assert reader.descriptor_integrity == "sha256-bound"
    assert restored_descriptor.fingerprint() == descriptor.fingerprint()
    assert not restored_frame.data.flags.writeable
    np.testing.assert_array_equal(restored_frame.data, frame.data)
    assert restored_frame.timestamp_ns == frame.timestamp_ns


def _downgrade_archive_to_legacy_v1(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    for stream in manifest["streams"].values():
        stream.pop("descriptor_fingerprint_sha256", None)
        stream["descriptor"].pop("fingerprint_sha256", None)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    descriptor_path = root / "streams" / "eeg" / "descriptor.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor.pop("fingerprint_sha256", None)
    descriptor_path.write_text(json.dumps(descriptor, sort_keys=True), encoding="utf-8")


def test_archive_v2_descriptor_tampering_is_detected(tmp_path: Path):
    root = tmp_path / "session"
    writer = SessionArchiveWriter(root, session_id="tamper")
    writer.register_stream(_descriptor())

    descriptor_path = root / "streams" / "eeg" / "descriptor.json"
    encoded = json.loads(descriptor_path.read_text(encoding="utf-8"))
    encoded["sample_rate_hz"] = 200.0
    descriptor_path.write_text(json.dumps(encoded), encoding="utf-8")

    reader = SessionArchiveReader(root)
    with pytest.raises(IOError, match="fingerprint mismatch"):
        reader.descriptor("eeg")


@pytest.mark.asyncio
async def test_archive_v2_index_does_not_coerce_malformed_identity_fields(tmp_path: Path):
    root = tmp_path / "strict-index"
    writer = SessionArchiveWriter(root, session_id="strict-index")
    writer.register_stream(_descriptor())
    await writer.write(_frame())
    await writer.close()

    index_path = root / "streams" / "eeg" / "index.ndjson"
    row = json.loads(index_path.read_text(encoding="utf-8").strip())
    row["sequence_id"] = 3.5
    index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    reader = SessionArchiveReader(root)
    with pytest.raises(TypeError, match="sequence_id must be an integer"):
        list(reader.iter_frames("eeg"))


@pytest.mark.asyncio
async def test_archive_v1_remains_readable_without_claiming_v2_integrity(tmp_path: Path):
    root = tmp_path / "legacy"
    writer = SessionArchiveWriter(root, session_id="legacy")
    writer.register_stream(_descriptor())
    await writer.write(_frame())
    await writer.close()
    _downgrade_archive_to_legacy_v1(root)

    reader = SessionArchiveReader(root)
    assert reader.schema_version == 1
    assert reader.descriptor_integrity == "legacy-unbound"
    assert len(list(reader.iter_frames("eeg"))) == 1
    summary = reader.summary()
    assert summary["schema_version"] == 1
    assert summary["descriptor_integrity"] == "legacy-unbound"


@pytest.mark.asyncio
async def test_archive_rejects_frame_descriptor_mismatch_before_writing(tmp_path: Path):
    descriptor = _descriptor()
    root = tmp_path / "session"
    writer = SessionArchiveWriter(root, session_id="mismatch")
    writer.register_stream(descriptor)

    with pytest.raises(ValueError, match="channel geometry"):
        await writer.write(_frame(data=np.zeros(7, dtype=np.float32)))

    assert not list((root / "streams" / "eeg" / "frames").glob("*.npy"))


@pytest.mark.asyncio
async def test_archive_auto_registration_requires_explicit_multidimensional_axis(tmp_path: Path):
    writer = SessionArchiveWriter(tmp_path / "session", session_id="ambiguous")
    ambiguous = _frame(
        stream_id="unregistered",
        data=np.zeros((4, 2)),
        metadata={"modality": "eeg"},
    )
    with pytest.raises(ValueError, match="axis_order"):
        await writer.write(ambiguous)
