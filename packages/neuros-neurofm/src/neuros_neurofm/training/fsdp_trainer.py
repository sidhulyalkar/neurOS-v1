"""
FSDP (Fully Sharded Data Parallel) Trainer for NeuroFMX
Enables distributed training of large models across multiple GPUs with memory efficiency.
"""

import os
from typing import Any, Dict, Optional
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import pytorch_lightning as pl


class FSDPConfig:
    """Configuration for FSDP training"""

    def __init__(
        self,
        sharding_strategy: str = "FULL_SHARD",
        cpu_offload: bool = False,
        mixed_precision: bool = True,
        activation_checkpointing: bool = True,
        backward_prefetch: str = "BACKWARD_PRE",
        forward_prefetch: bool = True,
        auto_wrap_policy: str = "transformer",
        min_num_params: int = 1_000_000,
    ):
        self.sharding_strategy = self._parse_sharding_strategy(sharding_strategy)
        self.cpu_offload = cpu_offload
        self.mixed_precision = mixed_precision
        self.activation_checkpointing = activation_checkpointing
        self.backward_prefetch = self._parse_backward_prefetch(backward_prefetch)
        self.forward_prefetch = forward_prefetch
        self.auto_wrap_policy = auto_wrap_policy
        self.min_num_params = min_num_params

    @staticmethod
    def _parse_sharding_strategy(strategy: str) -> ShardingStrategy:
        """Parse sharding strategy string to enum"""
        mapping = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,  # ZeRO-3
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2
            "NO_SHARD": ShardingStrategy.NO_SHARD,  # DDP
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,  # Hybrid
        }
        return mapping.get(strategy, ShardingStrategy.FULL_SHARD)

    @staticmethod
    def _parse_backward_prefetch(prefetch: str) -> Optional[BackwardPrefetch]:
        """Parse backward prefetch string to enum"""
        mapping = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        return mapping.get(prefetch, BackwardPrefetch.BACKWARD_PRE)


class FSDPTrainer:
    """
    Wrapper for FSDP training setup

    Example:
        >>> config = FSDPConfig(sharding_strategy="FULL_SHARD")
        >>> trainer = FSDPTrainer(config)
        >>> model = trainer.wrap_model(model)
    """

    def __init__(self, config: FSDPConfig):
        self.config = config
        self._setup_mixed_precision()

    def _setup_mixed_precision(self):
        """Configure mixed precision settings"""
        if self.config.mixed_precision:
            # Use bfloat16 for parameters, fp32 for reductions
            self.mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        else:
            self.mixed_precision_policy = None

    def get_auto_wrap_policy(self, model: nn.Module):
        """
        Get auto-wrap policy for FSDP

        Auto-wrap determines which submodules to wrap with FSDP.
        Transformer layers are typically wrapped individually.
        """
        if self.config.auto_wrap_policy == "transformer":
            # Wrap transformer blocks (common pattern for NeuroFMX)
            def lambda_policy(module):
                # Wrap if module is a transformer layer or has enough params
                if hasattr(module, "self_attn") or hasattr(module, "mamba_block"):
                    return True
                num_params = sum(p.numel() for p in module.parameters())
                return num_params >= self.config.min_num_params

            return lambda_policy

        elif self.config.auto_wrap_policy == "size_based":
            # Wrap based on parameter count only
            return partial(
                size_based_auto_wrap_policy,
                min_num_params=self.config.min_num_params
            )

        else:
            return None

    def wrap_model(self, model: nn.Module, device_id: Optional[int] = None) -> FSDP:
        """
        Wrap model with FSDP

        Args:
            model: PyTorch model to wrap
            device_id: GPU device ID (default: current device)

        Returns:
            FSDP-wrapped model
        """
        # Get device
        if device_id is None:
            device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

        # Get auto-wrap policy
        auto_wrap_policy = self.get_auto_wrap_policy(model)

        # Wrap with FSDP
        fsdp_model = FSDP(
            model,
            sharding_strategy=self.config.sharding_strategy,
            mixed_precision=self.mixed_precision_policy,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=self.config.backward_prefetch,
            forward_prefetch=self.config.forward_prefetch,
            cpu_offload=self.config.cpu_offload,
            device_id=device_id,
            sync_module_states=True,  # Broadcast rank 0 states
            limit_all_gathers=True,  # Reduce bandwidth usage
        )

        # Enable activation checkpointing if requested
        if self.config.activation_checkpointing:
            self._apply_activation_checkpointing(fsdp_model)

        return fsdp_model

    def _apply_activation_checkpointing(self, model: FSDP):
        """
        Apply activation checkpointing to transformer layers

        This trades compute for memory by recomputing activations during backward pass.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )

        def check_fn(submodule):
            # Checkpoint transformer/mamba blocks
            return (
                hasattr(submodule, "self_attn") or
                hasattr(submodule, "mamba_block") or
                submodule.__class__.__name__ in ["TransformerBlock", "MambaBlock"]
            )

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=check_fn,
        )

    def save_checkpoint(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        path: str,
    ):
        """
        Save FSDP checkpoint

        Args:
            model: FSDP-wrapped model
            optimizer: Optimizer
            epoch: Current epoch
            path: Checkpoint save path
        """
        # Save full state dict on rank 0
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
        ):
            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            # Only save on rank 0
            if torch.distributed.get_rank() == 0:
                torch.save(state_dict, path)
                print(f"Checkpoint saved to {path}")

    def load_checkpoint(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        path: str,
    ) -> int:
        """
        Load FSDP checkpoint

        Args:
            model: FSDP-wrapped model
            optimizer: Optimizer
            path: Checkpoint path

        Returns:
            Loaded epoch number
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location="cpu")

        # Load model state
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
        ):
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {path} (epoch {epoch})")

        return epoch


class FSDPLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module with FSDP support

    Example:
        >>> model = MyModel()
        >>> fsdp_config = FSDPConfig()
        >>> lightning_model = FSDPLightningModule(model, fsdp_config)
        >>> trainer = pl.Trainer(strategy="fsdp", ...)
        >>> trainer.fit(lightning_model, dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        fsdp_config: Optional[FSDPConfig] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.fsdp_config = fsdp_config or FSDPConfig()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def configure_model(self):
        """Configure model with FSDP (called by Lightning)"""
        if self.trainer.strategy.strategy_name == "fsdp":
            fsdp_trainer = FSDPTrainer(self.fsdp_config)
            self.model = fsdp_trainer.wrap_model(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss = self.model.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss = self.model.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_steps,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def create_fsdp_trainer_from_config(config_dict: Dict[str, Any]) -> FSDPTrainer:
    """
    Create FSDP trainer from configuration dictionary

    Args:
        config_dict: Configuration dictionary (from YAML)

    Returns:
        Configured FSDPTrainer instance
    """
    strategy_config = config_dict.get("strategy", {})

    fsdp_config = FSDPConfig(
        sharding_strategy=strategy_config.get("sharding_strategy", "FULL_SHARD"),
        cpu_offload=strategy_config.get("cpu_offload", False),
        mixed_precision=strategy_config.get("mixed_precision", {}).get("param_dtype") is not None,
        activation_checkpointing=strategy_config.get("activation_checkpointing", True),
        backward_prefetch=strategy_config.get("backward_prefetch", "BACKWARD_PRE"),
        forward_prefetch=strategy_config.get("forward_prefetch", True),
        auto_wrap_policy=strategy_config.get("auto_wrap_policy", "transformer"),
        min_num_params=int(strategy_config.get("min_num_params", 1e6)),
    )

    return FSDPTrainer(fsdp_config)
