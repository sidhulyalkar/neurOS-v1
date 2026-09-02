from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from neuros.contracts import DecoderOutput
from neuros.runtime.transport import SharedMemoryMailbox


def _output() -> DecoderOutput:
    return DecoderOutput(
        prediction=np.array([1, 2], dtype=np.int64),
        confidence=0.75,
        uncertainty=0.25,
        probabilities=np.array([0.25, 0.75], dtype=np.float32),
        logits=np.array([-0.5, 0.5], dtype=np.float32),
        embedding=np.arange(6, dtype=np.float32).reshape(2, 3),
        model_id="immutability-test",
        model_version="1",
        inference_time_ns=123,
        metadata={
            "labels": ["left", "right"],
            "nested": {"session": "A", "trial": 3},
            "array": np.array([4, 5], dtype=np.int16),
        },
    )


def test_decoder_output_detaches_all_array_fields_from_caller_storage():
    prediction = np.array([1, 2], dtype=np.int64)
    probabilities = np.array([0.2, 0.8], dtype=np.float32)
    logits = np.array([-1.0, 1.0], dtype=np.float32)
    embedding = np.arange(4, dtype=np.float32).reshape(2, 2)

    output = DecoderOutput(
        prediction=prediction,
        probabilities=probabilities,
        logits=logits,
        embedding=embedding,
    )

    prediction[:] = 99
    probabilities[:] = 99
    logits[:] = 99
    embedding[:] = 99

    assert np.array_equal(output.prediction, np.array([1, 2], dtype=np.int64))
    assert np.array_equal(output.probabilities, np.array([0.2, 0.8], dtype=np.float32))
    assert np.array_equal(output.logits, np.array([-1.0, 1.0], dtype=np.float32))
    assert np.array_equal(output.embedding, np.arange(4, dtype=np.float32).reshape(2, 2))
    assert not output.prediction.flags.writeable
    assert not output.probabilities.flags.writeable
    assert not output.logits.flags.writeable
    assert not output.embedding.flags.writeable


@pytest.mark.parametrize("field_name", ["prediction", "probabilities", "logits", "embedding"])
def test_decoder_output_array_fields_reject_direct_mutation(field_name):
    output = _output()
    array = getattr(output, field_name)
    with pytest.raises(ValueError):
        array.reshape(-1)[0] = 123


def test_decoder_output_recursively_detaches_prediction_and_metadata_containers():
    prediction = {
        "labels": ["left", "right"],
        "scores": np.array([1, 2], dtype=np.int64),
    }
    metadata = {
        "nested": {"labels": ["C3", "C4"]},
        "array": np.array([7, 8], dtype=np.int16),
    }

    output = DecoderOutput(prediction=prediction, metadata=metadata)

    prediction["labels"].append("mutated")
    prediction["scores"][:] = 99
    metadata["nested"]["labels"].append("mutated")
    metadata["array"][:] = 99

    assert isinstance(output.prediction, MappingProxyType)
    assert output.prediction["labels"] == ("left", "right")
    assert np.array_equal(output.prediction["scores"], np.array([1, 2], dtype=np.int64))
    assert not output.prediction["scores"].flags.writeable
    assert isinstance(output.metadata, MappingProxyType)
    assert isinstance(output.metadata["nested"], MappingProxyType)
    assert output.metadata["nested"]["labels"] == ("C3", "C4")
    assert output.metadata["array"] == (7, 8)

    with pytest.raises(TypeError):
        output.prediction["new"] = 1
    with pytest.raises(TypeError):
        output.metadata["new"] = 1


def test_decoder_output_string_prediction_arrays_become_transportable_immutable_sequences():
    labels = np.array(["left", "right"], dtype="U5")
    output = DecoderOutput(prediction=labels)
    labels[:] = "other"
    assert output.prediction == ("left", "right")


def test_decoder_output_replace_recanonicalizes_and_does_not_share_array_storage():
    original = _output()
    replaced = replace(original, metadata={**dict(original.metadata), "phase": "replaced"})

    assert replaced is not original
    assert replaced.prediction is not original.prediction
    assert replaced.probabilities is not original.probabilities
    assert replaced.logits is not original.logits
    assert replaced.embedding is not original.embedding
    assert np.array_equal(replaced.prediction, original.prediction)
    assert np.array_equal(replaced.probabilities, original.probabilities)
    assert np.array_equal(replaced.logits, original.logits)
    assert np.array_equal(replaced.embedding, original.embedding)
    assert not replaced.prediction.flags.writeable
    assert not replaced.probabilities.flags.writeable
    assert replaced.metadata["phase"] == "replaced"


def test_decoder_output_rejects_nondeterministic_or_nonnumeric_mutable_payloads():
    with pytest.raises(TypeError, match="unordered set"):
        DecoderOutput(prediction={"left", "right"})

    with pytest.raises(TypeError, match="mapping keys must be strings"):
        DecoderOutput(prediction={1: "left"})

    with pytest.raises(TypeError, match="probabilities must use a boolean or numeric dtype"):
        DecoderOutput(prediction=1, probabilities=["bad", "data"])


@pytest.mark.parametrize("field_name", ["probabilities", "logits", "embedding"])
def test_decoder_output_numeric_array_fields_accept_array_like_and_canonicalize(field_name):
    output = DecoderOutput(prediction=1, **{field_name: [1, 2, 3]})
    value = getattr(output, field_name)
    assert isinstance(value, np.ndarray)
    assert np.array_equal(value, np.array([1, 2, 3]))
    assert not value.flags.writeable


def test_decoder_output_shared_mailbox_round_trip_preserves_immutability():
    output = _output()
    box = SharedMemoryMailbox(32 * 1024)
    try:
        envelope = box.encode(output, lease_id=41)
        decoded = box.decode(envelope, expected_lease_id=41)
    finally:
        box.close_and_unlink()

    assert isinstance(decoded, DecoderOutput)
    assert np.array_equal(decoded.prediction, output.prediction)
    assert np.array_equal(decoded.probabilities, output.probabilities)
    assert np.array_equal(decoded.logits, output.logits)
    assert np.array_equal(decoded.embedding, output.embedding)
    assert decoded.metadata["labels"] == ("left", "right")
    assert decoded.metadata["nested"]["session"] == "A"
    assert not decoded.prediction.flags.writeable
    assert not decoded.probabilities.flags.writeable
    assert not decoded.logits.flags.writeable
    assert not decoded.embedding.flags.writeable


def test_decoder_output_keeps_existing_confidence_and_uncertainty_validation():
    with pytest.raises(ValueError, match="confidence"):
        DecoderOutput(prediction=1, confidence=1.01)
    with pytest.raises(ValueError, match="uncertainty"):
        DecoderOutput(prediction=1, uncertainty=-0.01)
