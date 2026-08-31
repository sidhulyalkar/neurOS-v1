from __future__ import annotations

import copy

import numpy as np
import pytest

from neuros.runtime.transport import NeuralTransportProtocolError, SharedMemoryMailbox


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
        with pytest.raises(NeuralTransportProtocolError, match="shape must be a list"):
            box.decode(corrupted, expected_lease_id=7)
    finally:
        box.close_and_unlink()


def test_transport_envelope_rejects_integer_coercion_for_identity_and_boundary():
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
    finally:
        box.close_and_unlink()
