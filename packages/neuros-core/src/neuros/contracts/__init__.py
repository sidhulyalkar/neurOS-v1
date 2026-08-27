"""Stable public contracts for the neurOS kernel."""

from .artifacts import ModelArtifactManifest
from .models import Decoder, DecoderCapabilities, DecoderOutput, TrainableDecoder
from .operators import Monitor, OutputSubscriber, Sink, Source, Transform, TransformEmission
from .signal import (
    ClockDomain,
    QualityFlag,
    SignalFrame,
    StreamDescriptor,
    frame_channel_count,
    validate_frame_against_descriptor,
)
from .window import NeuralWindow, WindowSpec

__all__ = [
    "ClockDomain",
    "Decoder",
    "DecoderCapabilities",
    "DecoderOutput",
    "ModelArtifactManifest",
    "Monitor",
    "NeuralWindow",
    "OutputSubscriber",
    "QualityFlag",
    "SignalFrame",
    "Sink",
    "Source",
    "StreamDescriptor",
    "TrainableDecoder",
    "Transform",
    "TransformEmission",
    "WindowSpec",
    "frame_channel_count",
    "validate_frame_against_descriptor",
]
