"""
Tests for NDT (Neural Data Transformer) foundation models.

These tests verify that NDT2 and NDT3 models work correctly with or without
PyTorch, handle multi-context and cross-subject scenarios, and maintain
compatibility with the neurOS BaseModel interface.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuros.foundation_models import NDT2Model, NDT3Model


class TestNDT2Model:
    """Tests for NDT2 multi-context pretraining model."""

    def test_initialization(self):
        """Test basic NDT2 model initialization."""
        model = NDT2Model(n_neurons=96, sequence_length=1.0, bin_size=0.005)
        assert model.n_neurons == 96
        assert model.n_bins == 200  # 1.0 / 0.005
        assert model.dim == 256
        assert model.depth == 6
        assert not model.is_trained

    def test_initialization_custom_params(self):
        """Test NDT2 initialization with custom parameters."""
        model = NDT2Model(
            n_neurons=128,
            sequence_length=2.0,
            bin_size=0.01,
            dim=512,
            depth=8,
            num_heads=16,
            dropout=0.2,
            max_contexts=200,
        )
        assert model.n_neurons == 128
        assert model.n_bins == 200  # 2.0 / 0.01
        assert model.dim == 512
        assert model.depth == 8
        assert model.num_heads == 16
        assert model.dropout == 0.2
        assert model.max_contexts == 200

    def test_train_basic(self):
        """Test basic training without context IDs."""
        model = NDT2Model(n_neurons=96, sequence_length=1.0, bin_size=0.005)
        X = np.random.randn(50, 96, 200)
        y = np.random.randn(50, 96, 200)

        model.train(X, y)
        assert model.is_trained

    def test_train_with_context_ids(self):
        """Test training with context identifiers."""
        model = NDT2Model(n_neurons=96, max_contexts=5)
        X = np.random.randn(50, 96, 200)
        y = np.random.randn(50, 96, 200)
        context_ids = np.random.randint(0, 5, size=50)

        model.train(X, y, context_ids=context_ids)
        assert model.is_trained

    def test_train_2d_input(self):
        """Test training with 2D input (auto-reshape to 3D)."""
        model = NDT2Model(n_neurons=96, sequence_length=1.0, bin_size=0.005)
        X = np.random.randn(50, 96)  # 2D input
        y = np.random.randn(50, 96)

        model.train(X, y)
        assert model.is_trained

    def test_predict_basic(self):
        """Test basic prediction."""
        model = NDT2Model(n_neurons=96, sequence_length=1.0, bin_size=0.005)
        X_train = np.random.randn(50, 96, 200)
        y_train = np.random.randn(50, 96, 200)
        model.train(X_train, y_train)

        X_test = np.random.randn(20, 96, 200)
        predictions = model.predict(X_test)

        assert predictions.shape == (20, 96)

    def test_predict_with_context(self):
        """Test prediction with context ID."""
        model = NDT2Model(n_neurons=96)
        X_train = np.random.randn(50, 96, 200)
        y_train = np.random.randn(50, 96, 200)
        model.train(X_train, y_train)

        X_test = np.random.randn(20, 96, 200)
        predictions = model.predict(X_test, context_id=2)

        assert predictions.shape == (20, 96)

    def test_predict_untrained_raises(self):
        """Test that prediction before training raises an error."""
        model = NDT2Model(n_neurons=96)
        X = np.random.randn(20, 96, 200)

        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X)

    def test_encode(self):
        """Test encoding neural data to latent space."""
        model = NDT2Model(n_neurons=96, dim=256)
        X_train = np.random.randn(50, 96, 200)
        y_train = np.random.randn(50, 96, 200)
        model.train(X_train, y_train)

        X_test = np.random.randn(20, 96, 200)
        latents = model.encode(X_test)

        assert latents.shape == (20, 256)

    def test_encode_with_context(self):
        """Test encoding with context ID."""
        model = NDT2Model(n_neurons=96, dim=256)
        X_train = np.random.randn(50, 96, 200)
        y_train = np.random.randn(50, 96, 200)
        model.train(X_train, y_train)

        X_test = np.random.randn(20, 96, 200)
        latents = model.encode(X_test, context_id=1)

        assert latents.shape == (20, 256)

    def test_decode(self):
        """Test decoding from latent space."""
        model = NDT2Model(n_neurons=96, dim=256)
        X_train = np.random.randn(50, 96, 200)
        y_train = np.random.randn(50, 96, 200)
        model.train(X_train, y_train)

        latents = np.random.randn(20, 256)
        decoded = model.decode(latents)

        assert decoded.shape == (20, 96)

    def test_save_and_load(self):
        """Test saving and loading model checkpoints."""
        model = NDT2Model(n_neurons=96, dim=256)
        X = np.random.randn(50, 96, 200)
        y = np.random.randn(50, 96, 200)
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "ndt2_model.pt"
            model.save_checkpoint(str(save_path))

            # Load the model
            loaded_model = NDT2Model.from_pretrained(str(save_path), n_neurons=96, dim=256)
            assert loaded_model.is_trained
            assert loaded_model.n_neurons == 96
            assert loaded_model.dim == 256


class TestNDT3Model:
    """Tests for NDT3 intracortical motor decoding model."""

    def test_initialization(self):
        """Test basic NDT3 model initialization."""
        model = NDT3Model(n_neurons=192, output_dim=2)
        assert model.n_neurons == 192
        assert model.output_dim == 2
        assert model.sequence_length == 0.5
        assert model.bin_size == 0.02
        assert model.n_bins == 25  # 0.5 / 0.02
        assert model.depth == 4
        assert not model.is_trained

    def test_initialization_custom_params(self):
        """Test NDT3 initialization with custom parameters."""
        model = NDT3Model(
            n_neurons=256,
            output_dim=3,
            sequence_length=1.0,
            bin_size=0.01,
            dim=512,
            depth=6,
            num_heads=16,
            dropout=0.2,
            use_subject_embedding=False,
            max_subjects=100,
        )
        assert model.n_neurons == 256
        assert model.output_dim == 3
        assert model.n_bins == 100  # 1.0 / 0.01
        assert model.dim == 512
        assert model.depth == 6
        assert not model.use_subject_embedding

    def test_train_basic(self):
        """Test basic training for motor decoding."""
        model = NDT3Model(n_neurons=192, output_dim=2)
        X = np.random.randn(100, 192, 25)
        y = np.random.randn(100, 2)  # Motor outputs (e.g., cursor velocity)

        model.train(X, y)
        assert model.is_trained

    def test_train_with_subject_ids(self):
        """Test training with subject identifiers."""
        model = NDT3Model(n_neurons=192, output_dim=2, max_subjects=10)
        X = np.random.randn(100, 192, 25)
        y = np.random.randn(100, 2)
        subject_ids = np.random.randint(0, 10, size=100)

        model.train(X, y, subject_ids=subject_ids)
        assert model.is_trained

    def test_train_2d_input(self):
        """Test training with 2D input (auto-reshape to 3D)."""
        model = NDT3Model(n_neurons=192, output_dim=2)
        X = np.random.randn(100, 192)  # 2D
        y = np.random.randn(100, 2)

        model.train(X, y)
        assert model.is_trained

    def test_predict_motor_output(self):
        """Test predicting motor outputs."""
        model = NDT3Model(n_neurons=192, output_dim=2)
        X_train = np.random.randn(100, 192, 25)
        y_train = np.random.randn(100, 2)
        model.train(X_train, y_train)

        X_test = np.random.randn(20, 192, 25)
        predictions = model.predict(X_test)

        assert predictions.shape == (20, 2)

    def test_predict_with_subject_id(self):
        """Test prediction with subject ID for cross-subject transfer."""
        model = NDT3Model(n_neurons=192, output_dim=2)
        X_train = np.random.randn(100, 192, 25)
        y_train = np.random.randn(100, 2)
        model.train(X_train, y_train)

        X_test = np.random.randn(20, 192, 25)
        predictions = model.predict(X_test, subject_id=5)

        assert predictions.shape == (20, 2)

    def test_predict_untrained_raises(self):
        """Test that prediction before training raises an error."""
        model = NDT3Model(n_neurons=192, output_dim=2)
        X = np.random.randn(20, 192, 25)

        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X)

    def test_encode(self):
        """Test encoding neural data to latent space."""
        model = NDT3Model(n_neurons=192, output_dim=2, dim=256)
        X_train = np.random.randn(100, 192, 25)
        y_train = np.random.randn(100, 2)
        model.train(X_train, y_train)

        X_test = np.random.randn(20, 192, 25)
        latents = model.encode(X_test)

        assert latents.shape == (20, 256)

    def test_decode_to_motor_output(self):
        """Test decoding latents to motor outputs."""
        model = NDT3Model(n_neurons=192, output_dim=2)
        X_train = np.random.randn(100, 192, 25)
        y_train = np.random.randn(100, 2)
        model.train(X_train, y_train)

        latents = np.random.randn(20, 256)
        motor_outputs = model.decode(latents)

        assert motor_outputs.shape == (20, 2)

    def test_fine_tune(self):
        """Test fine-tuning on a new subject."""
        model = NDT3Model(n_neurons=192, output_dim=2)

        # Pretrain on multiple subjects
        X_pretrain = np.random.randn(200, 192, 25)
        y_pretrain = np.random.randn(200, 2)
        model.train(X_pretrain, y_pretrain)

        # Fine-tune on new subject
        X_finetune = np.random.randn(20, 192, 25)
        y_finetune = np.random.randn(20, 2)
        model.fine_tune(X_finetune, y_finetune, subject_id=99, n_epochs=5)

        # Should still be able to predict
        predictions = model.predict(X_finetune, subject_id=99)
        assert predictions.shape == (20, 2)

    def test_fine_tune_untrained_raises(self):
        """Test that fine-tuning before pretraining raises an error."""
        model = NDT3Model(n_neurons=192, output_dim=2)
        X = np.random.randn(20, 192, 25)
        y = np.random.randn(20, 2)

        with pytest.raises(ValueError, match="must be pretrained"):
            model.fine_tune(X, y)

    def test_save_and_load(self):
        """Test saving and loading NDT3 model."""
        model = NDT3Model(n_neurons=192, output_dim=2, dim=256)
        X = np.random.randn(100, 192, 25)
        y = np.random.randn(100, 2)
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "ndt3_model.pt"
            model.save_checkpoint(str(save_path))

            loaded_model = NDT3Model.from_pretrained(
                str(save_path), n_neurons=192, output_dim=2, dim=256
            )
            assert loaded_model.is_trained
            assert loaded_model.n_neurons == 192
            assert loaded_model.output_dim == 2


class TestNDTIntegration:
    """Integration tests comparing NDT2 and NDT3."""

    def test_ndt2_vs_ndt3_architecture(self):
        """Test architectural differences between NDT2 and NDT3."""
        ndt2 = NDT2Model(n_neurons=96, sequence_length=1.0, bin_size=0.005)
        ndt3 = NDT3Model(n_neurons=192, sequence_length=0.5, bin_size=0.02)

        # NDT2 has longer sequences and finer temporal resolution
        assert ndt2.n_bins == 200
        assert ndt3.n_bins == 25

        # NDT3 has shallower network for real-time performance
        assert ndt2.depth == 6
        assert ndt3.depth == 4

    def test_context_vs_subject_embeddings(self):
        """Test that NDT2 uses contexts and NDT3 uses subjects."""
        ndt2 = NDT2Model(n_neurons=96, max_contexts=100)
        ndt3 = NDT3Model(n_neurons=192, max_subjects=50)

        assert ndt2.max_contexts == 100
        assert ndt3.max_subjects == 50

    def test_different_output_types(self):
        """Test that NDT2 predicts neural activity and NDT3 predicts motor outputs."""
        ndt2 = NDT2Model(n_neurons=96)
        ndt3 = NDT3Model(n_neurons=192, output_dim=2)

        # Train both models
        X2 = np.random.randn(50, 96, 200)
        y2 = np.random.randn(50, 96, 200)  # Next neural activity
        ndt2.train(X2, y2)

        X3 = np.random.randn(50, 192, 25)
        y3 = np.random.randn(50, 2)  # Motor outputs
        ndt3.train(X3, y3)

        # Predictions have different shapes
        pred2 = ndt2.predict(X2[:10])
        pred3 = ndt3.predict(X3[:10])

        assert pred2.shape == (10, 96)  # Neural activity
        assert pred3.shape == (10, 2)  # Motor outputs
