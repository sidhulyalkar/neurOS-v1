"""
Model definitions for neurOS.

Models encapsulate training and prediction logic.  The base class defines a
common interface so that custom algorithms can be dropped in easily.  A simple
logistic regression classifier is provided as a baseline.
"""

from .base_model import BaseModel  # noqa: F401
from .simple_classifier import SimpleClassifier  # noqa: F401

# extended models
from .eegnet_model import EEGNetModel  # noqa: F401
from .cnn_model import CNNModel  # noqa: F401
from .random_forest_model import RandomForestModel  # noqa: F401
from .svm_model import SVMModel  # noqa: F401
from .knn_model import KNNModel  # noqa: F401
from .gbdt_model import GBDTModel  # noqa: F401

# additional advanced models
from .transformer_model import TransformerModel  # noqa: F401
from .dino_v3_model import DinoV3Model  # noqa: F401