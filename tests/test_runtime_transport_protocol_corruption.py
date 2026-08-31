from __future__ import annotations

import copy

import numpy as np
import pytest

from neuros.contracts import DecoderOutput, SignalFrame
from neuros.runtime.transport import NeuralTransportProtocolError, SharedMemoryMailbox


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


def test_transport_manifest_rejects_coerced_and_misaligned_array_geometry():
    box = SharedMemoryMailbox(4096)
    try:
        envelope = box.encode(np.arange(8, dtype=np.float32), lease_id=7)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["offset"] = 1
        with pytest.raises(NeuralTransportProtocolError, match="aligned"):
            box.decode(corrupted, expected_lease_id=7)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["offset"] = "0"
        with pytest.raises(NeuralTransportProtocolError, match="exact integer"):
            box.decode(corrupted, expected_lease_id=7)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["nbytes"] = float(envelope["manifest"]["nbytes"])
        with pytest.raises(NeuralTransportProtocolError, match="exact integer"):
            box.decode(corrupted, expected_lease_id=7)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["shape"] = tuple(envelope["manifest"]["shape"])
        with pytest.raises(NeuralTransportProtocolError, match="exact list"):
            box.decode(corrupted, expected_lease_id=7)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["shape"] = ListSubclass(envelope["manifest"]["shape"])
        with pytest.raises(NeuralTransportProtocolError, match="exact list"):
            box.decode(corrupted, expected_lease_id=7)
    finally:
        box.close_and_unlink()


def test_transport_envelope_rejects_integer_coercion_and_unreferenced_tail_bytes():
    box = SharedMemoryMailbox(4096)
    try:
        envelope = box.encode(np.arange(4, dtype=np.float32), lease_id=3)

        corrupted = copy.deepcopy(envelope)
        corrupted["lease_id"] = "3"
        with pytest.raises(NeuralTransportProtocolError, match="exact integer"):
            box.decode(corrupted, expected_lease_id=3)

        corrupted = copy.deepcopy(envelope)
        corrupted["bytes_used"] = float(envelope["bytes_used"])
        with pytest.raises(NeuralTransportProtocolError, match="exact integer"):
            box.decode(corrupted, expected_lease_id=3)

        corrupted = copy.deepcopy(envelope)
        corrupted["bytes_used"] = True
        with pytest.raises(NeuralTransportProtocolError, match="exact integer"):
            box.decode(corrupted, expected_lease_id=3)

        corrupted = copy.deepcopy(envelope)
        corrupted["bytes_used"] += 64
        with pytest.raises(NeuralTransportProtocolError, match="referenced payload boundary"):
            box.decode(corrupted, expected_lease_id=3)
    finally:
        box.close_and_unlink()


@pytest.mark.parametrize(
    "expected_lease_id",
    (True, np.bool_(True), "3", 3.0, np.int64(3), 0, -1),
)
def test_expected_lease_identity_rejects_coercion(expected_lease_id):
    box = SharedMemoryMailbox(4096)
    try:
        envelope = box.encode(np.arange(4, dtype=np.float32), lease_id=3)
        with pytest.raises((TypeError, ValueError), match="expected_lease_id"):
            box.decode(envelope, expected_lease_id=expected_lease_id)
    finally:
        box.close_and_unlink()


def test_transport_rejects_mapping_subclasses_not_emitted_by_writer():
    box = SharedMemoryMailbox(4096)
    try:
        envelope = box.encode(np.arange(4, dtype=np.float32), lease_id=4)

        with pytest.raises(NeuralTransportProtocolError, match="exact mapping"):
            box.decode(DictSubclass(envelope), expected_lease_id=4)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"] = DictSubclass(corrupted["manifest"])
        with pytest.raises(NeuralTransportProtocolError, match="exact mapping"):
            box.decode(corrupted, expected_lease_id=4)
    finally:
        box.close_and_unlink()


def test_mailbox_ownership_cannot_be_forged_when_attaching():
    owner = SharedMemoryMailbox(4096)
    try:
        with pytest.raises(TypeError, match="unexpected keyword argument 'owner'"):
            SharedMemoryMailbox(
                4096,
                name=owner.name,
                create=False,
                owner=True,
            )
    finally:
        owner.close_and_unlink()


def test_canonical_frame_manifest_rejects_lossy_field_coercion():
    frame = SignalFrame(
        stream_id="eeg",
        sequence_id=8,
        data=np.arange(4, dtype=np.float32),
        sample_rate_hz=250.0,
        host_receive_time_ns=1000,
    )
    box = SharedMemoryMailbox(4096)
    try:
        envelope = box.encode(frame, lease_id=4)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["stream_id"] = 123
        with pytest.raises(NeuralTransportProtocolError, match="stream_id must be a string"):
            box.decode(corrupted, expected_lease_id=4)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["sequence_id"] = 8.9
        with pytest.raises(NeuralTransportProtocolError, match="exact integer"):
            box.decode(corrupted, expected_lease_id=4)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["host_receive_time_ns"] = "1000"
        with pytest.raises(NeuralTransportProtocolError, match="exact integer"):
            box.decode(corrupted, expected_lease_id=4)
    finally:
        box.close_and_unlink()


def test_canonical_constructor_rejection_is_reported_as_protocol_failure():
    output = DecoderOutput(prediction=1, confidence=0.5)
    box = SharedMemoryMailbox(4096)
    try:
        envelope = box.encode(output, lease_id=5)
        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["confidence"]["value"] = 2.0
        with pytest.raises(NeuralTransportProtocolError, match="DecoderOutput"):
            box.decode(corrupted, expected_lease_id=5)
    finally:
        box.close_and_unlink()


def test_sequence_manifest_requires_writer_container_shape():
    box = SharedMemoryMailbox(4096)
    try:
        envelope = box.encode((1, 2, 3), lease_id=6)
        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["items"] = tuple(corrupted["manifest"]["items"])
        with pytest.raises(NeuralTransportProtocolError, match="exact list"):
            box.decode(corrupted, expected_lease_id=6)

        corrupted = copy.deepcopy(envelope)
        corrupted["manifest"]["items"] = ListSubclass(corrupted["manifest"]["items"])
        with pytest.raises(NeuralTransportProtocolError, match="exact list"):
            box.decode(corrupted, expected_lease_id=6)
    finally:
        box.close_and_unlink()
