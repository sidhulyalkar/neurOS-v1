import warnings
import numpy as np
import pytest
import torch

from neuros.models import (
    AttentionFusionModel,
    CNNModel,
    EEGConformerModel,
    EEGNetModel,
    LSTMModel,
    TransformerModel,
    validate_manifest_paths,
)


def _assert_manifest(model):
    backend = model.analysis_model()
    manifest = model.analysis_manifest()
    assert manifest.mechint_ready
    assert not validate_manifest_paths(backend, manifest)
    assert manifest.fingerprint() == model.analysis_manifest().fingerprint()


def test_eegnet_manifest_and_shapes():
    model = EEGNetModel(n_channels=4, n_classes=3, temporal_kernel=15, separable_kernel=7, n_epochs=1)
    _assert_manifest(model)
    x = torch.randn(2, 4, 64)
    backend = model.analysis_model()
    assert backend(x).shape == (2, 3)
    assert backend.forward_features(x).shape[0] == 2


def test_conformer_manifest_and_mechint_replacement():
    model = EEGConformerModel(
        n_channels=4,
        n_classes=2,
        embedding_dim=16,
        n_heads=4,
        n_layers=2,
        pool_length=8,
        pool_stride=4,
        n_epochs=1,
    )
    _assert_manifest(model)
    adapter = model.mechint_adapter()
    assert "embedding_norm" in adapter.recommended_paths
    backend = model.analysis_model()
    x = torch.randn(3, 4, 64, device=next(backend.parameters()).device)
    cache = adapter.capture_outputs(x, ["embedding_norm"])
    clean = adapter.forward(x)
    zeroed = adapter.forward_with_replacements(x, {"embedding_norm": torch.zeros_like(cache["embedding_norm"])})
    assert clean.shape == zeroed.shape == (3, 2)
    assert not torch.allclose(clean, zeroed)


def test_transformer_is_real_torch_transformer_not_generator():
    model = TransformerModel(n_channels=4, n_classes=2, d_model=16, n_heads=4, n_layers=2, n_epochs=1)
    _assert_manifest(model)
    names = dict(model.analysis_model().named_modules())
    assert "encoder.layers.0.self_attn" in names
    assert "encoder.layers.0.linear1" in names
    x = torch.randn(2, 4, 32)
    assert model.analysis_model()(x).shape == (2, 2)


def test_temporal_cnn_and_lstm_surfaces_exist():
    cnn = CNNModel(n_channels=3, n_classes=2, hidden_channels=8, n_blocks=2, n_epochs=1)
    lstm = LSTMModel(n_channels=3, n_classes=2, lstm_units=8, n_lstm_layers=1, n_epochs=1)
    _assert_manifest(cnn)
    _assert_manifest(lstm)
    assert "blocks.1" in cnn.analysis_manifest().recommended_paths
    assert "embedding_norm" in lstm.analysis_manifest().recommended_paths


def test_attention_fusion_is_sample_dependent_and_normalized():
    model = AttentionFusionModel([3, 5, 2], n_classes=2, fusion_dim=8, n_epochs=1)
    _assert_manifest(model)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 10)).astype("float32")
    y = (x[:, 0] > 0).astype(int)
    model.train(x, y)
    weights = model.get_attention_weights(x[:6])
    assert weights.shape == (6, 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert np.std(weights, axis=0).max() > 0


def test_training_infer_exposes_logits_and_embedding():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 4, 64)).astype("float32")
    y = (X[:, 0, :].mean(axis=1) > 0).astype(int)
    model = CNNModel(
        n_channels=4,
        n_classes=2,
        hidden_channels=8,
        n_blocks=1,
        n_epochs=1,
        batch_size=8,
        random_state=1,
    )
    model.train(X, y)
    assert model.is_trained
    probs = model.predict_proba(X[:2])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    out = model.infer(X[:1])
    assert out.logits is not None
    assert out.embedding is not None
    assert out.metadata["mechint_ready"] is True


def test_dinov3_compatibility_path_fails_closed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from neuros.models.dino_v3_model import DinoV3Model
        model = DinoV3Model()
    with pytest.raises(RuntimeError, match="no longer fabricates"):
        model.train(np.zeros((2, 2)), np.zeros(2))
