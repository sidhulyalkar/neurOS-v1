"""
Model definitions for neurOS.

Models encapsulate training and prediction logic.  The base class defines a
common interface so that custom algorithms can be dropped in easily.  A simple
logistic regression classifier is provided as a baseline.
"""

from neuros.models.base_model import BaseModel  # noqa: F401
from neuros.models.simple_classifier import SimpleClassifier  # noqa: F401

# extended models
from neuros.models.eegnet_model import EEGNetModel  # noqa: F401
from neuros.models.cnn_model import CNNModel  # noqa: F401
from neuros.models.random_forest_model import RandomForestModel  # noqa: F401
from neuros.models.svm_model import SVMModel  # noqa: F401
from neuros.models.knn_model import KNNModel  # noqa: F401
from neuros.models.gbdt_model import GBDTModel  # noqa: F401

# additional advanced models
from neuros.models.transformer_model import TransformerModel  # noqa: F401
from neuros.models.dino_v3_model import DinoV3Model  # noqa: F401
from neuros.models.lstm_model import LSTMModel  # noqa: F401

# composite multi‑modal models
from neuros.models.composite_model import CompositeModel  # noqa: F401
from neuros.models.attention_fusion_model import AttentionFusionModel  # noqa: F401

# model management
from neuros.models.model_registry import ModelRegistry, ModelMetadata  # noqa: F401
