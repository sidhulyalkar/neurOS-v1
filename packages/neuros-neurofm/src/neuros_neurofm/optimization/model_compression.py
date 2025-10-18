"""
Model compression and optimization utilities for NeuroFM-X.

Provides tools for reducing model size and improving inference speed:
- Quantization (INT8, FP16)
- Pruning (magnitude-based, structured)
- Knowledge distillation
- TorchScript export
"""

from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic, quantize_qat


class ModelQuantizer:
    """Quantize models for faster inference and reduced memory.

    Supports dynamic quantization (INT8) and static quantization (QAT).

    Parameters
    ----------
    model : nn.Module
        Model to quantize.
    quantization_type : str, optional
        Type of quantization ('dynamic', 'qat', 'static').
        Default: 'dynamic'.
    backend : str, optional
        Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM).
        Default: 'fbgemm'.

    Examples
    --------
    >>> quantizer = ModelQuantizer(model, quantization_type='dynamic')
    >>> quantized_model = quantizer.quantize()
    >>> # Model is now INT8, 4x smaller and faster
    """

    def __init__(
        self,
        model: nn.Module,
        quantization_type: str = 'dynamic',
        backend: str = 'fbgemm',
    ):
        self.model = model
        self.quantization_type = quantization_type
        self.backend = backend

    def quantize(self) -> nn.Module:
        """Apply quantization to the model.

        Returns
        -------
        nn.Module
            Quantized model.
        """
        torch.backends.quantized.engine = self.backend

        if self.quantization_type == 'dynamic':
            # Dynamic quantization: quantize weights statically, activations dynamically
            quantized_model = quantize_dynamic(
                self.model,
                {nn.Linear, nn.LSTM, nn.GRU},  # Quantize these layer types
                dtype=torch.qint8,
            )

        elif self.quantization_type == 'static':
            # Static quantization: requires calibration data
            # Prepare model for quantization
            self.model.qconfig = torch.quantization.get_default_qconfig(self.backend)
            quantized_model = torch.quantization.prepare(self.model, inplace=False)
            # Note: User must run calibration data through quantized_model before converting
            print("Static quantization prepared. Run calibration data, then call convert().")
            return quantized_model

        elif self.quantization_type == 'qat':
            # Quantization-aware training: train with fake quantization
            self.model.qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
            quantized_model = torch.quantization.prepare_qat(self.model, inplace=False)
            print("QAT prepared. Continue training, then call convert().")
            return quantized_model

        else:
            raise ValueError(f"Unknown quantization type: {self.quantization_type}")

        return quantized_model

    @staticmethod
    def convert_to_quantized(prepared_model: nn.Module) -> nn.Module:
        """Convert prepared model to quantized model.

        Use after calibration (static) or QAT training.

        Parameters
        ----------
        prepared_model : nn.Module
            Prepared model from quantize() with qconfig.

        Returns
        -------
        nn.Module
            Fully quantized model.
        """
        return torch.quantization.convert(prepared_model, inplace=False)


class ModelPruner:
    """Prune model weights to reduce size and improve speed.

    Supports magnitude-based pruning and structured pruning.

    Parameters
    ----------
    model : nn.Module
        Model to prune.
    pruning_method : str, optional
        Pruning method ('magnitude', 'structured').
        Default: 'magnitude'.
    amount : float, optional
        Fraction of weights to prune (0.0 to 1.0).
        Default: 0.3 (30%).

    Examples
    --------
    >>> pruner = ModelPruner(model, amount=0.5)
    >>> pruned_model = pruner.prune()
    >>> # 50% of weights are now zero
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_method: str = 'magnitude',
        amount: float = 0.3,
    ):
        self.model = model
        self.pruning_method = pruning_method
        self.amount = amount

    def prune(self) -> nn.Module:
        """Apply pruning to the model.

        Returns
        -------
        nn.Module
            Pruned model.
        """
        import torch.nn.utils.prune as prune

        if self.pruning_method == 'magnitude':
            # Magnitude-based unstructured pruning
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=self.amount)
                    # Make pruning permanent
                    prune.remove(module, 'weight')

        elif self.pruning_method == 'structured':
            # Structured pruning: remove entire filters/neurons
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=self.amount,
                        n=2,  # L2 norm
                        dim=0,  # Output dimension
                    )
                    prune.remove(module, 'weight')

        else:
            raise ValueError(f"Unknown pruning method: {self.pruning_method}")

        return self.model

    def compute_sparsity(self) -> Dict[str, float]:
        """Compute sparsity statistics for the model.

        Returns
        -------
        dict
            Sparsity metrics (overall, per-layer).
        """
        total_zeros = 0
        total_params = 0
        layer_sparsity = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = module.weight.data
                zeros = (weight == 0).sum().item()
                params = weight.numel()

                total_zeros += zeros
                total_params += params
                layer_sparsity[name] = zeros / params if params > 0 else 0.0

        overall_sparsity = total_zeros / total_params if total_params > 0 else 0.0

        return {
            'overall_sparsity': overall_sparsity,
            'layer_sparsity': layer_sparsity,
            'total_zeros': total_zeros,
            'total_params': total_params,
        }


class KnowledgeDistiller:
    """Distill knowledge from large teacher to small student model.

    Implements standard knowledge distillation (Hinton et al., 2015).

    Parameters
    ----------
    teacher_model : nn.Module
        Large pre-trained teacher model.
    student_model : nn.Module
        Smaller student model to train.
    temperature : float, optional
        Softmax temperature for distillation.
        Default: 4.0.
    alpha : float, optional
        Weight for distillation loss vs. ground truth loss.
        Default: 0.7 (70% distillation, 30% ground truth).

    Examples
    --------
    >>> distiller = KnowledgeDistiller(teacher, student)
    >>> loss = distiller.distillation_loss(student_logits, teacher_logits, targets)
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

        # Set teacher to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Parameters
        ----------
        student_logits : torch.Tensor
            Student model outputs.
        teacher_logits : torch.Tensor
            Teacher model outputs.
        targets : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Combined distillation loss.
        """
        # Distillation loss: KL divergence between teacher and student
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)

        distill_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean',
        ) * (self.temperature ** 2)

        # Ground truth loss
        gt_loss = F.cross_entropy(student_logits, targets)

        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * gt_loss

        return total_loss

    def distill_regression(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Distillation loss for regression tasks.

        Parameters
        ----------
        student_output : torch.Tensor
            Student model predictions.
        teacher_output : torch.Tensor
            Teacher model predictions.
        targets : torch.Tensor
            Ground truth targets.

        Returns
        -------
        torch.Tensor
            Combined regression distillation loss.
        """
        # Distillation loss: MSE between teacher and student
        distill_loss = F.mse_loss(student_output, teacher_output)

        # Ground truth loss
        gt_loss = F.mse_loss(student_output, targets)

        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * gt_loss

        return total_loss


class TorchScriptExporter:
    """Export models to TorchScript for production deployment.

    TorchScript models can run without Python and are optimized for inference.

    Parameters
    ----------
    model : nn.Module
        Model to export.
    example_inputs : tuple or torch.Tensor
        Example inputs for tracing.
    mode : str, optional
        Export mode ('trace' or 'script').
        Default: 'trace'.

    Examples
    --------
    >>> exporter = TorchScriptExporter(model, example_inputs)
    >>> script_model = exporter.export()
    >>> exporter.save(script_model, 'model.pt')
    """

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        mode: str = 'trace',
    ):
        self.model = model
        self.example_inputs = example_inputs
        self.mode = mode

    def export(self) -> torch.jit.ScriptModule:
        """Export model to TorchScript.

        Returns
        -------
        torch.jit.ScriptModule
            TorchScript model.
        """
        self.model.eval()

        if self.mode == 'trace':
            # Tracing: run model with example inputs
            script_model = torch.jit.trace(self.model, self.example_inputs)

        elif self.mode == 'script':
            # Scripting: compile model directly
            script_model = torch.jit.script(self.model)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Optimize for inference
        script_model = torch.jit.optimize_for_inference(script_model)

        return script_model

    def save(self, script_model: torch.jit.ScriptModule, save_path: str):
        """Save TorchScript model to file.

        Parameters
        ----------
        script_model : torch.jit.ScriptModule
            TorchScript model.
        save_path : str
            Path to save model.
        """
        script_model.save(save_path)
        print(f"TorchScript model saved to {save_path}")

    @staticmethod
    def load(load_path: str) -> torch.jit.ScriptModule:
        """Load TorchScript model from file.

        Parameters
        ----------
        load_path : str
            Path to load model from.

        Returns
        -------
        torch.jit.ScriptModule
            Loaded TorchScript model.
        """
        return torch.jit.load(load_path)


class MixedPrecisionOptimizer:
    """Optimize model for mixed precision (FP16/BF16) training and inference.

    Reduces memory usage and increases speed on modern GPUs.

    Parameters
    ----------
    model : nn.Module
        Model to optimize.
    precision : str, optional
        Precision type ('fp16' or 'bf16').
        Default: 'fp16'.

    Examples
    --------
    >>> optimizer = MixedPrecisionOptimizer(model, precision='fp16')
    >>> model_fp16 = optimizer.convert()
    """

    def __init__(
        self,
        model: nn.Module,
        precision: str = 'fp16',
    ):
        self.model = model
        self.precision = precision

    def convert(self) -> nn.Module:
        """Convert model to mixed precision.

        Returns
        -------
        nn.Module
            Model with mixed precision.
        """
        if self.precision == 'fp16':
            self.model = self.model.half()
        elif self.precision == 'bf16':
            self.model = self.model.to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown precision: {self.precision}")

        return self.model

    def get_autocast_context(self):
        """Get autocast context for mixed precision training.

        Returns
        -------
        torch.cuda.amp.autocast or torch.cpu.amp.autocast
            Autocast context manager.
        """
        if torch.cuda.is_available():
            if self.precision == 'fp16':
                return torch.cuda.amp.autocast(dtype=torch.float16)
            elif self.precision == 'bf16':
                return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            # CPU autocast
            if self.precision == 'bf16':
                return torch.cpu.amp.autocast(dtype=torch.bfloat16)

        # Fallback: no autocast
        from contextlib import nullcontext
        return nullcontext()


def compute_model_size(model: nn.Module) -> Dict[str, float]:
    """Compute model size in bytes and megabytes.

    Parameters
    ----------
    model : nn.Module
        Model to measure.

    Returns
    -------
    dict
        Model size statistics.
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        'total_bytes': total_size,
        'total_mb': total_size / (1024 ** 2),
        'param_bytes': param_size,
        'param_mb': param_size / (1024 ** 2),
        'buffer_bytes': buffer_size,
        'buffer_mb': buffer_size / (1024 ** 2),
    }


def compare_model_sizes(
    original_model: nn.Module,
    compressed_model: nn.Module,
) -> Dict[str, Any]:
    """Compare sizes of original and compressed models.

    Parameters
    ----------
    original_model : nn.Module
        Original model.
    compressed_model : nn.Module
        Compressed model.

    Returns
    -------
    dict
        Comparison statistics.
    """
    original_size = compute_model_size(original_model)
    compressed_size = compute_model_size(compressed_model)

    compression_ratio = original_size['total_mb'] / compressed_size['total_mb']
    size_reduction = original_size['total_mb'] - compressed_size['total_mb']

    return {
        'original_mb': original_size['total_mb'],
        'compressed_mb': compressed_size['total_mb'],
        'compression_ratio': compression_ratio,
        'size_reduction_mb': size_reduction,
        'size_reduction_percent': (size_reduction / original_size['total_mb']) * 100,
    }


def save_compression_config(
    config: Dict[str, Any],
    save_path: str,
):
    """Save compression configuration to JSON.

    Parameters
    ----------
    config : dict
        Compression configuration.
    save_path : str
        Path to save JSON file.
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Compression config saved to {save_path}")


def load_compression_config(load_path: str) -> Dict[str, Any]:
    """Load compression configuration from JSON.

    Parameters
    ----------
    load_path : str
        Path to JSON file.

    Returns
    -------
    dict
        Compression configuration.
    """
    with open(load_path, 'r') as f:
        config = json.load(f)

    return config
