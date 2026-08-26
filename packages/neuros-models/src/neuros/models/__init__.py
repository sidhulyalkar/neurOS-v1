"""Task-specific decoders and analysis contracts for neurOS."""

from neuros.models.analysis import (
    AnalysisCapability,
    AnalysisSurface,
    InterpretabilityManifest,
    MechanisticallyInspectable,
    validate_manifest_paths,
)
from neuros.models.artifacts import (
    ARTIFACT_KIND,
    ARTIFACT_SCHEMA_VERSION,
    TorchDecoderStateSnapshot,
    learning_state_sha256,
    parameter_state_sha256_from_tensors,
    read_torch_decoder_artifact,
    resolved_torch_decoder_config,
    restore_torch_decoder_state,
    snapshot_torch_decoder_state,
    torch_parameter_state_sha256,
    write_torch_decoder_artifact,
)
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
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "AnalysisCapability",
    "AnalysisSurface",
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
    "ModelMetadata",
    "ModelRegistry",
    "RandomForestModel",
    "SVMModel",
    "SimpleClassifier",
    "TemporalTransformerModel",
    "TorchDecoderStateSnapshot",
    "TransformerModel",
    "get_decoder_card",
    "learning_state_sha256",
    "list_decoder_cards",
    "parameter_state_sha256_from_tensors",
    "read_torch_decoder_artifact",
    "resolved_torch_decoder_config",
    "restore_torch_decoder_state",
    "snapshot_torch_decoder_state",
    "torch_parameter_state_sha256",
    "validate_manifest_paths",
    "write_torch_decoder_artifact",
]
