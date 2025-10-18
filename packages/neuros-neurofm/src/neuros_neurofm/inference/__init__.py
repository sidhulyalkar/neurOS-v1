"""
Real-time inference module for NeuroFM-X.

Provides optimized inference pipelines for production deployment.
"""

from neuros_neurofm.inference.realtime_pipeline import (
    RealtimeInferencePipeline,
    DynamicBatcher,
    ModelCache,
    LatencyProfiler,
    InferenceRequest,
    InferenceResult,
)

__all__ = [
    'RealtimeInferencePipeline',
    'DynamicBatcher',
    'ModelCache',
    'LatencyProfiler',
    'InferenceRequest',
    'InferenceResult',
]
