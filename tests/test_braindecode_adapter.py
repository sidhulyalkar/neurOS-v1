import importlib.metadata

import numpy as np
import pytest

from neuros.models import BraindecodeDecoder


def test_braindecode_adapter_rejects_unqualified_or_ambiguous_contracts():
    with pytest.raises(ValueError, match="Qualified adapter models"):
        BraindecodeDecoder("SignalJEPA", 2, 128, 2)

    with pytest.raises(ValueError, match="cannot override"):
        BraindecodeDecoder(
            "EEGNet",
            2,
            128,
            2,
            model_options={"n_chans": 99},
        )

    decoder = BraindecodeDecoder("EEGNet", 2, 128, 2, n_epochs=1)
    with pytest.raises(ValueError, match="batch, channels, time"):
        decoder.train(np.ones((2, 128), dtype=np.float32), np.array([0, 1]))
    with pytest.raises(ValueError, match="geometry"):
        decoder.train(np.ones((2, 3, 128), dtype=np.float32), np.array([0, 1]))


def test_braindecode_adapter_trains_upstream_eegnet_without_hidden_preprocessing():
    version = importlib.metadata.version("braindecode")
    assert version.startswith("1.7."), version

    rng = np.random.default_rng(11)
    X = rng.normal(size=(8, 2, 128)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

    decoder = BraindecodeDecoder(
        "EEGNet",
        n_channels=2,
        n_times=128,
        n_classes=2,
        sample_rate_hz=128.0,
        learning_rate=1e-3,
        n_epochs=1,
        batch_size=4,
        device="cpu",
        random_state=11,
    )
    decoder.train(X, y)

    output = decoder.infer(X[:1])
    assert output.model_id == "Braindecode:EEGNet"
    assert output.model_version == version
    assert output.probabilities is not None
    probabilities = np.asarray(output.probabilities)
    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)
    assert output.confidence is not None
    assert output.metadata["backend"] == "braindecode/torch"
    assert output.metadata["upstream_training"] == "EEGClassifier"
    assert output.metadata["hidden_preprocessing"] is False
    assert output.metadata["input_contract"] == "batch,channel,time"

    module = decoder.analysis_model()
    assert module is not None
    manifest = decoder.analysis_manifest()
    assert manifest.architecture == "EEGNet"
    assert manifest.input_axes == ("batch", "channel", "time")
    assert manifest.mechint_ready is False
