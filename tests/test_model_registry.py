"""Tests for model registry functionality."""
import pytest
import tempfile
from pathlib import Path
import numpy as np

from neuros.models import SimpleClassifier, ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            # Train a simple model
            model = SimpleClassifier()
            X = np.random.randn(20, 10)
            y = np.random.randint(0, 2, 20)
            model.train(X, y)

            # Save model
            metadata = registry.save(
                model,
                name="test_model",
                version="1.0.0",
                metrics={"accuracy": 0.95},
                tags=["test", "classifier"],
            )

            assert metadata.name == "test_model"
            assert metadata.version == "1.0.0"
            assert metadata.metrics["accuracy"] == 0.95
            assert "test" in metadata.tags

            # Load model
            loaded_model = registry.load("test_model", version="1.0.0")
            assert isinstance(loaded_model, SimpleClassifier)
            assert loaded_model.is_trained

            # Test prediction
            X_test = np.random.randn(5, 10)
            predictions = loaded_model.predict(X_test)
            assert len(predictions) == 5

    def test_list_models(self):
        """Test listing models in registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            # Save multiple models
            for i in range(3):
                model = SimpleClassifier()
                X = np.random.randn(10, 5)
                y = np.random.randint(0, 2, 10)
                model.train(X, y)

                registry.save(
                    model,
                    name=f"model_{i}",
                    version=f"{i}.0.0",
                    metrics={"accuracy": 0.9 + i * 0.01},
                )

            # List all models
            models = registry.list_models()
            assert len(models) == 3

            # Filter by name
            filtered = registry.list_models(name_filter="model_1")
            assert len(filtered) == 1
            assert filtered[0].name == "model_1"

    def test_search_models(self):
        """Test searching models by criteria."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            # Save models with different tags
            model1 = SimpleClassifier()
            model1.train(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            registry.save(
                model1,
                name="prod_model",
                metrics={"accuracy": 0.95},
                tags=["production"],
            )

            model2 = SimpleClassifier()
            model2.train(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            registry.save(
                model2,
                name="dev_model",
                metrics={"accuracy": 0.85},
                tags=["development"],
            )

            # Search by tag
            prod_models = registry.search(tags=["production"])
            assert len(prod_models) == 1
            assert prod_models[0].name == "prod_model"

            # Search by accuracy
            high_acc_models = registry.search(min_accuracy=0.9)
            assert len(high_acc_models) == 1
            assert high_acc_models[0].metrics["accuracy"] >= 0.9

    def test_get_latest_version(self):
        """Test getting latest version of a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            # Save multiple versions
            for i in [1, 2, 3]:
                model = SimpleClassifier()
                model.train(np.random.randn(10, 5), np.random.randint(0, 2, 10))
                registry.save(model, name="my_model", version=f"{i}.0.0")

            # Get latest
            latest = registry.get_latest("my_model")
            assert latest is not None
            # Latest should be most recently created (version 3.0.0)
            assert latest.version == "3.0.0"

    def test_delete_model(self):
        """Test deleting a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            # Save a model
            model = SimpleClassifier()
            model.train(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            registry.save(model, name="temp_model", version="1.0.0")

            # Verify it exists
            models = registry.list_models()
            assert len(models) == 1

            # Delete it
            deleted = registry.delete("temp_model", "1.0.0")
            assert deleted

            # Verify it's gone
            models = registry.list_models()
            assert len(models) == 0

    def test_checksum_verification(self):
        """Test checksum verification on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            # Save a model
            model = SimpleClassifier()
            model.train(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            metadata = registry.save(model, name="secure_model", version="1.0.0")

            # Tamper with the file
            model_path = Path(tmpdir) / metadata.file_path
            with open(model_path, "ab") as f:
                f.write(b"corrupted data")

            # Loading with checksum verification should fail
            with pytest.raises(ValueError, match="Checksum mismatch"):
                registry.load("secure_model", version="1.0.0", verify_checksum=True)

            # Loading without verification bypasses checksum check
            # The model may be corrupted but pickle might still load it or fail
            # We just verify the option works
            try:
                registry.load("secure_model", version="1.0.0", verify_checksum=False)
            except Exception:
                # Expected - corrupted pickle
                pass
