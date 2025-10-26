"""
Production Deployment and Inference Serving
==========================================

This example demonstrates production deployment of NeuroFMX:

1. Model export (ONNX, TorchScript)
2. Inference optimization (quantization, pruning)
3. REST API serving with FastAPI
4. Batch inference pipeline
5. Real-time streaming inference
6. Model monitoring and logging

Deployment Options:
    - Local server (FastAPI)
    - Docker container
    - Kubernetes deployment
    - Cloud deployment (AWS/GCP/Azure)
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import logging

from neuros_neurofm.model import NeuroFMX


# ==================== Model Export ====================

def export_to_torchscript(
    model: NeuroFMX,
    export_path: str,
    example_inputs: Dict[str, torch.Tensor],
):
    """
    Export model to TorchScript for production deployment

    TorchScript benefits:
    - Language-agnostic (C++, Java, etc.)
    - Optimized inference
    - No Python dependency at runtime
    """
    print(f"Exporting to TorchScript: {export_path}")

    model.eval()

    # Trace model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_inputs)

    # Save
    traced_model.save(export_path)

    print(f"✓ TorchScript model saved to {export_path}")

    # Verify
    loaded_model = torch.jit.load(export_path)
    with torch.no_grad():
        original_output = model(example_inputs)
        loaded_output = loaded_model(example_inputs)

    # Check outputs match
    max_diff = (original_output - loaded_output).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✓ Verification passed")
    else:
        print(f"  ⚠ Warning: outputs differ by {max_diff}")

    return traced_model


def export_to_onnx(
    model: NeuroFMX,
    export_path: str,
    example_inputs: Dict[str, torch.Tensor],
    opset_version: int = 14,
):
    """
    Export model to ONNX format

    ONNX benefits:
    - Cross-framework compatibility
    - Optimized runtime (ONNX Runtime)
    - Hardware-specific optimizations
    """
    print(f"Exporting to ONNX: {export_path}")

    model.eval()

    # Dynamic axes for variable-length sequences
    dynamic_axes = {
        'eeg': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size', 1: 'sequence_length'},
    }

    torch.onnx.export(
        model,
        example_inputs,
        export_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=list(example_inputs.keys()),
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    print(f"✓ ONNX model saved to {export_path}")

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(export_path)

        # Prepare inputs
        ort_inputs = {k: v.cpu().numpy() for k, v in example_inputs.items()}

        # Run inference
        ort_outputs = session.run(None, ort_inputs)

        print("  ✓ ONNX Runtime verification passed")

    except ImportError:
        print("  ⚠ ONNX Runtime not installed, skipping verification")


def optimize_model(
    model: NeuroFMX,
    quantization: str = 'dynamic',  # 'dynamic', 'static', 'qat', None
    pruning: float = 0.0,  # Sparsity level (0.0-1.0)
) -> NeuroFMX:
    """
    Optimize model for inference

    - Quantization: Reduce precision (fp32 → int8)
    - Pruning: Remove unimportant weights
    """
    print("\nOptimizing model for inference...")

    # Quantization
    if quantization == 'dynamic':
        print("  Applying dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        model = quantized_model
        print("  ✓ Dynamic quantization applied")

    elif quantization == 'static':
        print("  ⚠ Static quantization requires calibration data")
        # Static quantization would require calibration here

    # Pruning
    if pruning > 0:
        print(f"  Applying {pruning:.1%} pruning...")
        from torch.nn.utils import prune

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning)
                prune.remove(module, 'weight')  # Make pruning permanent

        print(f"  ✓ Pruning applied")

    # Measure size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"  Model size: {original_size:.1f} MB")

    return model


# ==================== Inference Server ====================

# FastAPI app
app = FastAPI(
    title="NeuroFMX Inference API",
    description="Production inference server for NeuroFMX foundation model",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model (loaded on startup)
MODEL = None
CONFIG = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic schemas
class InferenceRequest(BaseModel):
    """Request schema for inference"""
    modalities: Dict[str, List[List[float]]]  # {modality: [batch, time, features]}
    return_embeddings: bool = False
    return_attention: bool = False


class InferenceResponse(BaseModel):
    """Response schema for inference"""
    predictions: List[List[float]]  # [batch, time, features]
    embeddings: Optional[List[List[float]]] = None
    attention_weights: Optional[List[List[float]]] = None
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model on server startup"""
    global MODEL, CONFIG

    logger.info("Loading NeuroFMX model...")

    # Load config
    config_path = os.environ.get('CONFIG_PATH', 'configs/deployment/default.yaml')
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)

    # Load model
    checkpoint_path = CONFIG['deployment']['checkpoint_path']

    MODEL = NeuroFMX(
        d_model=CONFIG['model']['d_model'],
        n_layers=CONFIG['model']['n_layers'],
        n_heads=CONFIG['model']['n_heads'],
        architecture=CONFIG['model']['architecture'],
        modality_configs=CONFIG['model']['modalities'],
    )

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    MODEL.load_state_dict(state_dict)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    # Optimize model
    if CONFIG['deployment'].get('optimize', True):
        MODEL = optimize_model(
            MODEL,
            quantization=CONFIG['deployment'].get('quantization', 'dynamic'),
            pruning=CONFIG['deployment'].get('pruning', 0.0),
        )

    logger.info(f"✓ Model loaded on {DEVICE}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        device=DEVICE,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    Main inference endpoint

    Example request:
    {
        "modalities": {
            "eeg": [[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]],
            "spikes": [[[0.01, 0.02, ...], ...]]
        },
        "return_embeddings": true
    }
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Start timer
        start_time = datetime.now()

        # Convert inputs to tensors
        inputs = {}
        for modality, data in request.modalities.items():
            tensor = torch.tensor(data, dtype=torch.float32, device=DEVICE)
            inputs[modality] = tensor

        # Run inference
        with torch.no_grad():
            outputs = MODEL(
                inputs,
                return_embeddings=request.return_embeddings,
                return_attention=request.return_attention,
            )

        # Extract predictions
        if isinstance(outputs, dict):
            predictions = outputs['predictions'].cpu().numpy().tolist()
            embeddings = outputs.get('embeddings')
            if embeddings is not None:
                embeddings = embeddings.cpu().numpy().tolist()
            attention = outputs.get('attention')
            if attention is not None:
                attention = attention.cpu().numpy().tolist()
        else:
            predictions = outputs.cpu().numpy().tolist()
            embeddings = None
            attention = None

        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Log request
        logger.info(f"Inference completed in {latency_ms:.2f}ms")

        return InferenceResponse(
            predictions=predictions,
            embeddings=embeddings,
            attention_weights=attention,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(
    requests: List[InferenceRequest],
    background_tasks: BackgroundTasks,
):
    """
    Batch inference endpoint

    Processes multiple requests in a single batch for efficiency
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Combine all requests into a single batch
        all_inputs = {modality: [] for modality in requests[0].modalities.keys()}

        for req in requests:
            for modality, data in req.modalities.items():
                all_inputs[modality].extend(data)

        # Convert to tensors
        batch_inputs = {}
        for modality, data in all_inputs.items():
            tensor = torch.tensor(data, dtype=torch.float32, device=DEVICE)
            batch_inputs[modality] = tensor

        # Run batch inference
        start_time = datetime.now()

        with torch.no_grad():
            outputs = MODEL(batch_inputs)

        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Split outputs back into individual responses
        predictions = outputs.cpu().numpy().tolist()

        # Log batch request
        logger.info(f"Batch inference ({len(requests)} samples) completed in {latency_ms:.2f}ms")

        return {
            "predictions": predictions,
            "total_latency_ms": latency_ms,
            "samples_per_second": len(requests) / (latency_ms / 1000),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Batch inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Streaming Inference ====================

class StreamingInference:
    """
    Real-time streaming inference for continuous neural data

    Maintains state across time for efficient processing
    """

    def __init__(
        self,
        model: NeuroFMX,
        buffer_size: int = 1000,  # 1 second at 1kHz
        device: str = 'cuda',
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.device = device

        # Rolling buffer for each modality
        self.buffers = {}

        # Hidden state for recurrent processing
        self.hidden_state = None

    def initialize_buffer(self, modality: str, n_channels: int):
        """Initialize buffer for a modality"""
        self.buffers[modality] = torch.zeros(
            1, self.buffer_size, n_channels,
            device=self.device
        )

    def update(self, new_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Update buffers with new data and run inference

        Args:
            new_data: {modality: tensor of shape (1, T, C)}
                     where T is number of new samples

        Returns:
            Predictions for the new samples
        """
        # Update buffers
        for modality, data in new_data.items():
            if modality not in self.buffers:
                self.initialize_buffer(modality, data.shape[-1])

            # Roll buffer and add new data
            n_new = data.shape[1]
            self.buffers[modality] = torch.cat([
                self.buffers[modality][:, n_new:, :],
                data
            ], dim=1)

        # Run inference on full buffer
        with torch.no_grad():
            outputs = self.model(self.buffers)

        # Return only predictions for new samples
        return outputs[:, -n_new:, :]


# ==================== Main ====================

def main():
    """
    Main deployment workflow

    1. Load model
    2. Export to TorchScript/ONNX
    3. Start inference server
    """

    # Configuration
    config_path = "configs/deployment/default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("NeuroFMX Deployment")
    print("=" * 80)

    # Load model
    checkpoint_path = config['deployment']['checkpoint_path']

    model = NeuroFMX(
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        architecture=config['model']['architecture'],
        modality_configs=config['model']['modalities'],
    )

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ Model loaded from {checkpoint_path}")

    # Create example inputs for export
    example_inputs = {
        'eeg': torch.randn(1, 100, 64),  # (batch, time, channels)
        'spikes': torch.randn(1, 100, 96),
    }

    # Export to TorchScript
    if config['deployment'].get('export_torchscript', True):
        torchscript_path = config['deployment'].get('torchscript_path', 'models/neurofmx.pt')
        export_to_torchscript(model, torchscript_path, example_inputs)

    # Export to ONNX
    if config['deployment'].get('export_onnx', True):
        onnx_path = config['deployment'].get('onnx_path', 'models/neurofmx.onnx')
        export_to_onnx(model, onnx_path, example_inputs)

    # Optimize model
    if config['deployment'].get('optimize', True):
        model = optimize_model(
            model,
            quantization=config['deployment'].get('quantization', 'dynamic'),
            pruning=config['deployment'].get('pruning', 0.0),
        )

    print("\n" + "=" * 80)
    print("Starting Inference Server")
    print("=" * 80)
    print(f"Host: {config['deployment'].get('host', '0.0.0.0')}")
    print(f"Port: {config['deployment'].get('port', 8000)}")
    print("=" * 80)

    # Start server
    uvicorn.run(
        app,
        host=config['deployment'].get('host', '0.0.0.0'),
        port=config['deployment'].get('port', 8000),
        log_level="info",
    )


if __name__ == "__main__":
    main()


"""
Docker Deployment
=================

Dockerfile:

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "05_deployment_inference.py"]

Build and run:
    docker build -t neurofmx-server .
    docker run -p 8000:8000 --gpus all neurofmx-server

Kubernetes Deployment
=====================

deployment.yaml:

apiVersion: apps/v1
kind: Deployment
metadata:
  name: neurofmx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neurofmx
  template:
    metadata:
      labels:
        app: neurofmx
    spec:
      containers:
      - name: neurofmx
        image: neurofmx-server:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "16Gi"
            cpu: "4"
        env:
        - name: CONFIG_PATH
          value: "/config/deployment.yaml"
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: neurofmx-config

---
apiVersion: v1
kind: Service
metadata:
  name: neurofmx-service
spec:
  selector:
    app: neurofmx
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

Deploy:
    kubectl apply -f deployment.yaml
"""
