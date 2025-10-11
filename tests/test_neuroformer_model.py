"""
Tests for Neuroformer (Multimodal Generative Pretraining) foundation model.

These tests verify that Neuroformer works correctly with pretraining,
fine-tuning, zero-shot, few-shot, and generation capabilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuros.foundation_models import NeuroformerModel


class TestNeuroformerInitialization:
    """Tests for Neuroformer model initialization."""

    def test_basic_initialization(self):
        """Test basic Neuroformer initialization."""
        model = NeuroformerModel(input_dim=100, output_dim=2)
        assert model.input_dim == 100
        assert model.output_dim == 2
        assert model.n_modalities == 1
        assert model.dim == 512
        assert model.depth == 8
        assert model.task_type == "classification"
        assert not model.is_trained

    def test_initialization_custom_params(self):
        """Test Neuroformer initialization with custom parameters."""
        model = NeuroformerModel(
            input_dim=200,
            output_dim=4,
            n_modalities=3,
            dim=768,
            depth=12,
            num_heads=12,
            dropout=0.2,
            mask_ratio=0.25,
            pretrain_mode=True,
            task_type="regression",
        )
        assert model.input_dim == 200
        assert model.output_dim == 4
        assert model.n_modalities == 3
        assert model.dim == 768
        assert model.depth == 12
        assert model.num_heads == 12
        assert model.dropout == 0.2
        assert model.mask_ratio == 0.25
        assert model.pretrain_mode is True
        assert model.task_type == "regression"

    def test_invalid_task_type_raises(self):
        """Test that invalid task type raises an error."""
        with pytest.raises(ValueError, match="task_type must be"):
            NeuroformerModel(input_dim=100, task_type="invalid")


class TestNeuroformerPretraining:
    """Tests for Neuroformer pretraining."""

    def test_pretrain_basic(self):
        """Test basic pretraining."""
        model = NeuroformerModel(input_dim=100, pretrain_mode=True)
        X = np.random.randn(1000, 50, 100)  # (samples, seq_len, input_dim)

        model.pretrain(X, n_epochs=10)
        assert model.is_trained

    def test_pretrain_2d_input(self):
        """Test pretraining with 2D input (auto-adds sequence dim)."""
        model = NeuroformerModel(input_dim=100, pretrain_mode=True)
        X = np.random.randn(1000, 100)  # 2D input

        model.pretrain(X)
        assert model.is_trained

    def test_pretrain_with_modality_ids(self):
        """Test pretraining with modality identifiers."""
        model = NeuroformerModel(input_dim=100, n_modalities=3)
        X = np.random.randn(1000, 50, 100)
        modality_ids = np.random.randint(0, 3, size=(1000, 3))

        model.pretrain(X, modality_ids=modality_ids, n_epochs=5)
        assert model.is_trained


class TestNeuroformerFineTuning:
    """Tests for Neuroformer fine-tuning."""

    def test_finetune_classification(self):
        """Test fine-tuning for classification task."""
        model = NeuroformerModel(input_dim=100, output_dim=3, task_type="classification")
        X = np.random.randn(500, 50, 100)
        y = np.random.randint(0, 3, size=500)

        model.train(X, y)
        assert model.is_trained

    def test_finetune_regression(self):
        """Test fine-tuning for regression task."""
        model = NeuroformerModel(input_dim=100, output_dim=2, task_type="regression")
        X = np.random.randn(500, 50, 100)
        y = np.random.randn(500, 2)

        model.train(X, y)
        assert model.is_trained

    def test_finetune_2d_input(self):
        """Test fine-tuning with 2D input."""
        model = NeuroformerModel(input_dim=100, output_dim=2)
        X = np.random.randn(500, 100)  # 2D
        y = np.random.randint(0, 2, size=500)

        model.train(X, y)
        assert model.is_trained

    def test_finetune_with_modality_ids(self):
        """Test fine-tuning with modality identifiers."""
        model = NeuroformerModel(input_dim=100, n_modalities=2)
        X = np.random.randn(500, 50, 100)
        y = np.random.randint(0, 2, size=500)
        modality_ids = np.random.randint(0, 2, size=(500, 2))

        model.train(X, y, modality_ids=modality_ids)
        assert model.is_trained


class TestNeuroformerPrediction:
    """Tests for Neuroformer prediction."""

    def test_predict_classification(self):
        """Test prediction for classification."""
        model = NeuroformerModel(input_dim=100, output_dim=3, task_type="classification")
        X_train = np.random.randn(500, 50, 100)
        y_train = np.random.randint(0, 3, size=500)
        model.train(X_train, y_train)

        X_test = np.random.randn(50, 50, 100)
        predictions = model.predict(X_test)

        assert predictions.shape == (50,)
        assert predictions.dtype == np.int64

    def test_predict_regression(self):
        """Test prediction for regression."""
        model = NeuroformerModel(input_dim=100, output_dim=2, task_type="regression")
        X_train = np.random.randn(500, 50, 100)
        y_train = np.random.randn(500, 2)
        model.train(X_train, y_train)

        X_test = np.random.randn(50, 50, 100)
        predictions = model.predict(X_test)

        assert predictions.shape == (50, 2)

    def test_predict_untrained_raises(self):
        """Test that prediction before training raises error."""
        model = NeuroformerModel(input_dim=100, output_dim=2)
        X = np.random.randn(50, 50, 100)

        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X)

    def test_predict_2d_input(self):
        """Test prediction with 2D input."""
        model = NeuroformerModel(input_dim=100, output_dim=2)
        X_train = np.random.randn(500, 100)
        y_train = np.random.randint(0, 2, size=500)
        model.train(X_train, y_train)

        X_test = np.random.randn(50, 100)
        predictions = model.predict(X_test)

        assert predictions.shape[0] == 50


class TestNeuroformerEncoding:
    """Tests for Neuroformer encoding and decoding."""

    def test_encode(self):
        """Test encoding neural data to latent space."""
        model = NeuroformerModel(input_dim=100, dim=512)
        X_train = np.random.randn(500, 50, 100)
        y_train = np.random.randint(0, 2, size=500)
        model.train(X_train, y_train)

        X_test = np.random.randn(50, 50, 100)
        embeddings = model.encode(X_test)

        assert embeddings.shape == (50, 512)

    def test_decode(self):
        """Test decoding from latent space."""
        model = NeuroformerModel(input_dim=100, dim=512)
        X = np.random.randn(500, 50, 100)
        y = np.random.randint(0, 2, size=500)
        model.train(X, y)

        latents = np.random.randn(50, 512)
        decoded = model.decode(latents)

        assert decoded.shape == (50, 100)

    def test_encode_untrained_raises(self):
        """Test that encoding before training raises error."""
        model = NeuroformerModel(input_dim=100)
        X = np.random.randn(50, 50, 100)

        with pytest.raises(ValueError, match="must be trained"):
            model.encode(X)


class TestNeuroformerZeroShot:
    """Tests for zero-shot prediction."""

    def test_zero_shot_predict(self):
        """Test zero-shot prediction with task description."""
        model = NeuroformerModel(input_dim=100, output_dim=2)
        # Pretrain first
        X_pretrain = np.random.randn(1000, 50, 100)
        model.pretrain(X_pretrain, n_epochs=10)

        # Zero-shot on new task
        X_test = np.random.randn(50, 50, 100)
        predictions = model.zero_shot_predict(
            X_test, task_description="Predict left vs right arm movement"
        )

        assert predictions.shape == (50, 2)

    def test_zero_shot_untrained_raises(self):
        """Test that zero-shot without pretraining raises error."""
        model = NeuroformerModel(input_dim=100, output_dim=2)
        X = np.random.randn(50, 50, 100)

        with pytest.raises(ValueError, match="must be pretrained"):
            model.zero_shot_predict(X, task_description="Some task")


class TestNeuroformerFewShot:
    """Tests for few-shot adaptation."""

    def test_few_shot_adapt_classification(self):
        """Test few-shot adaptation for classification."""
        model = NeuroformerModel(input_dim=100, output_dim=3, task_type="classification")
        # Pretrain
        X_pretrain = np.random.randn(1000, 50, 100)
        model.pretrain(X_pretrain, n_epochs=10)

        # Few-shot adaptation
        X_support = np.random.randn(15, 50, 100)  # 5 shots per class
        y_support = np.array([0] * 5 + [1] * 5 + [2] * 5)
        X_query = np.random.randn(30, 50, 100)

        predictions = model.few_shot_adapt(X_support, y_support, X_query, n_shots=5)

        assert predictions.shape == (30,)

    def test_few_shot_adapt_regression(self):
        """Test few-shot adaptation for regression."""
        model = NeuroformerModel(input_dim=100, output_dim=2, task_type="regression")
        X_pretrain = np.random.randn(1000, 50, 100)
        model.pretrain(X_pretrain, n_epochs=10)

        X_support = np.random.randn(10, 50, 100)
        y_support = np.random.randn(10, 2)
        X_query = np.random.randn(30, 50, 100)

        predictions = model.few_shot_adapt(X_support, y_support, X_query, n_shots=10)

        assert predictions.shape == (30, 2)

    def test_few_shot_untrained_raises(self):
        """Test that few-shot without pretraining raises error."""
        model = NeuroformerModel(input_dim=100, output_dim=2)
        X_support = np.random.randn(10, 50, 100)
        y_support = np.random.randint(0, 2, size=10)
        X_query = np.random.randn(30, 50, 100)

        with pytest.raises(ValueError, match="must be pretrained"):
            model.few_shot_adapt(X_support, y_support, X_query)


class TestNeuroformerGeneration:
    """Tests for neural data generation."""

    def test_generate_basic(self):
        """Test basic generation of synthetic neural data."""
        model = NeuroformerModel(input_dim=100, task_type="generation")
        X = np.random.randn(1000, 50, 100)
        model.pretrain(X, n_epochs=10)

        generated = model.generate(n_samples=50, sequence_length=100)

        assert generated.shape == (50, 100, 100)

    def test_generate_with_context(self):
        """Test generation conditioned on context."""
        model = NeuroformerModel(input_dim=100, task_type="generation")
        X = np.random.randn(1000, 50, 100)
        model.pretrain(X, n_epochs=10)

        context = np.random.randn(10, 20, 100)  # Context sequences
        generated = model.generate(context=context, n_samples=10, sequence_length=50)

        assert generated.shape == (10, 50, 100)

    def test_generate_untrained_raises(self):
        """Test that generation before training raises error."""
        model = NeuroformerModel(input_dim=100)

        with pytest.raises(ValueError, match="must be trained"):
            model.generate(n_samples=10)


class TestNeuroformerSaveLoad:
    """Tests for saving and loading Neuroformer models."""

    def test_save_and_load(self):
        """Test saving and loading Neuroformer model."""
        model = NeuroformerModel(input_dim=100, output_dim=2, dim=256, depth=4)
        X = np.random.randn(500, 50, 100)
        y = np.random.randint(0, 2, size=500)
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "neuroformer_model.pt"
            model.save_checkpoint(str(save_path))

            loaded_model = NeuroformerModel.from_pretrained(
                str(save_path), input_dim=100, output_dim=2, dim=256, depth=4
            )
            assert loaded_model.is_trained
            assert loaded_model.input_dim == 100
            assert loaded_model.output_dim == 2
            assert loaded_model.dim == 256


class TestNeuroformerIntegration:
    """Integration tests for complete Neuroformer workflows."""

    def test_pretrain_finetune_workflow(self):
        """Test complete pretrain -> fine-tune workflow."""
        model = NeuroformerModel(input_dim=96, output_dim=3, dim=256, depth=4)

        # Step 1: Pretrain on large unlabeled dataset
        X_pretrain = np.random.randn(2000, 50, 96)
        model.pretrain(X_pretrain, n_epochs=20)
        assert model.is_trained

        # Step 2: Fine-tune on smaller labeled dataset
        X_finetune = np.random.randn(200, 50, 96)
        y_finetune = np.random.randint(0, 3, size=200)
        model.train(X_finetune, y_finetune)

        # Step 3: Predict on test data
        X_test = np.random.randn(50, 50, 96)
        predictions = model.predict(X_test)
        assert predictions.shape == (50,)

    def test_multimodal_workflow(self):
        """Test multimodal workflow with multiple data types."""
        model = NeuroformerModel(
            input_dim=100, output_dim=2, n_modalities=3, dim=256, depth=4
        )

        # Multimodal data: spikes, LFP, behavior
        X = np.random.randn(500, 50, 100)
        modality_ids = np.random.randint(0, 3, size=(500, 3))
        y = np.random.randint(0, 2, size=500)

        # Train with modality information
        model.train(X, y, modality_ids=modality_ids)

        # Predict
        X_test = np.random.randn(50, 50, 100)
        predictions = model.predict(X_test)
        assert predictions.shape == (50,)

    def test_zero_few_shot_workflow(self):
        """Test pretrain -> zero-shot -> few-shot workflow."""
        model = NeuroformerModel(input_dim=96, output_dim=4, dim=256, depth=4)

        # Pretrain
        X_pretrain = np.random.randn(2000, 50, 96)
        model.pretrain(X_pretrain, n_epochs=20)

        # Zero-shot prediction
        X_test = np.random.randn(50, 50, 96)
        zero_shot_preds = model.zero_shot_predict(
            X_test, task_description="Classify brain state"
        )
        assert zero_shot_preds.shape == (50, 4)

        # Few-shot adaptation
        X_support = np.random.randn(20, 50, 96)
        y_support = np.random.randint(0, 4, size=20)
        few_shot_preds = model.few_shot_adapt(X_support, y_support, X_test, n_shots=5)
        assert few_shot_preds.shape == (50,)

    def test_generative_workflow(self):
        """Test generative workflow for data augmentation."""
        model = NeuroformerModel(input_dim=96, dim=256, depth=4, task_type="generation")

        # Train generative model
        X_train = np.random.randn(1000, 50, 96)
        model.pretrain(X_train, n_epochs=50)

        # Generate synthetic data
        synthetic = model.generate(n_samples=100, sequence_length=50)
        assert synthetic.shape == (100, 50, 96)

        # Generate with conditioning
        context = X_train[:10, :20, :]  # Use first 20 timesteps as context
        conditioned = model.generate(context=context, n_samples=10, sequence_length=30)
        assert conditioned.shape == (10, 30, 96)
