"""Task-specific decoders and analysis contracts for neurOS."""

from neuros.models.analysis import (
    AnalysisCapability,
    AnalysisSurface,
    InterpretabilityManifest,
    MechanisticallyInspectable,
    validate_manifest_paths,
)
from neuros.models.artifact import (
    ArtifactBackedDecoder,
    ModelArtifactManifest,
    ModelInputContract,
    export_model_artifact,
    load_model_artifact,
    verify_model_artifact,
)
from neuros.models.artifact_store import ModelArtifactStore
from neuros.models.attention_fusion_model import AttentionFusionModel
from neuros.models.base_model import BaseModel
from neuros.models.braindecode_adapter import BraindecodeDecoder
from neuros.models.catalog import DecoderCard, get_decoder_card, list_decoder_cards
from neuros.models.cnn_model import CNNModel
from neuros.models.composite_model import CompositeModel
from neuros.models.dino_v3_model import DinoV3Model
from neuros.models.eeg_conformer_model import EEGConformerModel
from neuros.models.eegnet_model import EEGNetModel
from neuros.models.gbdt_model import GBDTModel
from neuros.models.knn_model import KNNModel
from neuros.models.lstm_model import LSTMModel
from neuros.models.model_registry import ModelMetadata, ModelRegistry
from neuros.models.random_forest_model import RandomForestModel
from neuros.models.simple_classifier import SimpleClassifier
from neuros.models.svm_model import SVMModel
from neuros.models.transformer_model import TemporalTransformerModel, TransformerModel

__all__ = [
    "AnalysisCapability",
    "AnalysisSurface",
    "ArtifactBackedDecoder",
    "AttentionFusionModel",
    "BaseModel",
    "BraindecodeDecoder",
    "CNNModel",
    "CompositeModel",
    "DecoderCard",
    "DinoV3Model",
    "EEGConformerModel",
    "EEGNetModel",
    "GBDTModel",
    "InterpretabilityManifest",
    "KNNModel",
    "LSTMModel",
    "MechanisticallyInspectable",
    "ModelArtifactManifest",
    "ModelArtifactStore",
    "ModelInputContract",
    "ModelMetadata",
    "ModelRegistry",
    "RandomForestModel",
    "SVMModel",
    "SimpleClassifier",
    "TemporalTransformerModel",
    "TransformerModel",
    "export_model_artifact",
    "get_decoder_card",
    "list_decoder_cards",
    "load_model_artifact",
    "validate_manifest_paths",
    "verify_model_artifact",
]
