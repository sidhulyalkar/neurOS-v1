"""
ENGRAM-FMx Synthetic Training Script.

Trains ENGRAM-FMx on synthetic tasks (associative recall, delayed copy, neural dynamics)
to validate the architecture before scaling to real neural data.

Usage:
    python -m neuros_neurofm.training.train_engram_synthetic --config configs/engram_fmx/tiny_synthetic.yaml
    python -m neuros_neurofm.training.train_engram_synthetic --task associative_recall --max_steps 1000
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from neuros_neurofm.backbones.engram_fmx import ENGRAMBackbone, ENGRAMFMxConfig
from neuros_neurofm.data.synthetic import (
    AssociativeRecallDataset,
    DelayedCopyDataset,
    NeuralDynamicsDataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Task
    task: str = "associative_recall"  # associative_recall, delayed_copy, neural_dynamics

    # Model
    model_size: str = "tiny"  # tiny, small, medium, large
    hidden_dim: Optional[int] = None  # Override model size
    num_layers: Optional[int] = None
    num_latents: Optional[int] = None
    memory_slots: Optional[int] = None

    # Ablations
    use_local_processing: bool = True
    use_ssm: bool = True
    use_latent_workspace: bool = True
    use_attractor_memory: bool = True
    use_operator_dynamics: bool = True
    use_sparse_anchor_attention: bool = True

    # Data
    seq_length: int = 256
    num_train_samples: int = 10000
    num_val_samples: int = 1000
    batch_size: int = 8

    # Training
    max_steps: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0

    # Hardware
    device: str = "auto"
    precision: str = "fp32"  # fp32, fp16, bf16

    # Logging
    log_interval: int = 50
    eval_interval: int = 200
    save_interval: int = 500
    output_dir: str = "runs/engram_fmx"
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.task}_{self.model_size}_{timestamp}"


def load_config(config_path: str) -> TrainingConfig:
    """Load config from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for config files: pip install pyyaml")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return TrainingConfig(**config_dict)


def create_model(config: TrainingConfig) -> ENGRAMBackbone:
    """Create ENGRAM-FMx model from config."""
    # Get base config
    if config.model_size == "tiny":
        model_config = ENGRAMFMxConfig.tiny()
    elif config.model_size == "small":
        model_config = ENGRAMFMxConfig.small()
    elif config.model_size == "medium":
        model_config = ENGRAMFMxConfig.medium()
    elif config.model_size == "large":
        model_config = ENGRAMFMxConfig.large()
    else:
        model_config = ENGRAMFMxConfig()

    # Override with explicit values
    if config.hidden_dim is not None:
        model_config.hidden_dim = config.hidden_dim
        model_config.input_dim = config.hidden_dim
        model_config.output_dim = config.hidden_dim
    if config.num_layers is not None:
        model_config.num_layers = config.num_layers
    if config.num_latents is not None:
        model_config.num_latents = config.num_latents
    if config.memory_slots is not None:
        model_config.memory_slots = config.memory_slots

    # Ablation flags
    model_config.use_local_processing = config.use_local_processing
    model_config.use_ssm = config.use_ssm
    model_config.use_latent_workspace = config.use_latent_workspace
    model_config.use_attractor_memory = config.use_attractor_memory
    model_config.use_operator_dynamics = config.use_operator_dynamics
    model_config.use_sparse_anchor_attention = config.use_sparse_anchor_attention

    return ENGRAMBackbone(model_config)


def create_datasets(config: TrainingConfig):
    """Create train and validation datasets."""
    hidden_dim = config.hidden_dim or 128  # Default for tiny

    if config.task == "associative_recall":
        train_dataset = AssociativeRecallDataset(
            num_samples=config.num_train_samples,
            seq_length=config.seq_length,
            hidden_dim=hidden_dim,
            seed=42,
        )
        val_dataset = AssociativeRecallDataset(
            num_samples=config.num_val_samples,
            seq_length=config.seq_length,
            hidden_dim=hidden_dim,
            seed=123,
        )
    elif config.task == "delayed_copy":
        train_dataset = DelayedCopyDataset(
            num_samples=config.num_train_samples,
            seq_length=config.seq_length,
            hidden_dim=hidden_dim,
            seed=42,
        )
        val_dataset = DelayedCopyDataset(
            num_samples=config.num_val_samples,
            seq_length=config.seq_length,
            hidden_dim=hidden_dim,
            seed=123,
        )
    elif config.task == "neural_dynamics":
        train_dataset = NeuralDynamicsDataset(
            num_samples=config.num_train_samples,
            seq_length=config.seq_length,
            hidden_dim=hidden_dim,
            seed=42,
        )
        val_dataset = NeuralDynamicsDataset(
            num_samples=config.num_val_samples,
            seq_length=config.seq_length,
            hidden_dim=hidden_dim,
            seed=123,
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")

    return train_dataset, val_dataset


def compute_loss(
    output,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked MSE loss."""
    # output.sequence_output: [B, T, D]
    # targets: [B, T, D]
    # mask: [B, T]

    pred = output.sequence_output
    diff = (pred - targets) ** 2  # [B, T, D]
    diff = diff.mean(dim=-1)  # [B, T]

    # Apply mask
    mask_float = mask.float()
    if mask_float.sum() > 0:
        loss = (diff * mask_float).sum() / mask_float.sum()
    else:
        loss = diff.mean()

    return loss


def evaluate(
    model: ENGRAMBackbone,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets, mask in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            output = model(inputs)
            loss = compute_loss(output, targets, mask)

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return {"val_loss": total_loss / max(num_batches, 1)}


def train(config: TrainingConfig) -> Dict[str, Any]:
    """Main training loop."""
    # Setup
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {config.device}")

    # Create model
    model = create_model(config)
    model = model.to(config.device)

    num_params = model.get_num_params()
    logger.info(f"Model parameters: {num_params:,}")

    # Create datasets
    train_dataset, val_dataset = create_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    model.train()
    step = 0
    epoch = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        epoch += 1

        for inputs, targets, mask in train_loader:
            if step >= config.max_steps:
                break

            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            mask = mask.to(config.device)

            # Forward
            output = model(inputs)
            loss = compute_loss(output, targets, mask)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            step += 1
            pbar.update(1)

            # Logging
            if step % config.log_interval == 0:
                avg_loss = sum(train_losses[-config.log_interval:]) / config.log_interval
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                logger.info(f"Step {step}: loss={avg_loss:.4f}, lr={lr:.2e}")

            # Evaluation
            if step % config.eval_interval == 0:
                val_metrics = evaluate(model, val_loader, config.device)
                val_losses.append(val_metrics["val_loss"])
                logger.info(f"Step {step}: val_loss={val_metrics['val_loss']:.4f}")

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": best_val_loss,
                        },
                        output_dir / "best_model.pt",
                    )

            # Save checkpoint
            if step % config.save_interval == 0:
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    },
                    output_dir / f"checkpoint_{step}.pt",
                )

    pbar.close()

    # Final evaluation
    final_metrics = evaluate(model, val_loader, config.device)

    # Save final results
    results = {
        "final_train_loss": sum(train_losses[-100:]) / min(100, len(train_losses)),
        "final_val_loss": final_metrics["val_loss"],
        "best_val_loss": best_val_loss,
        "num_params": num_params,
        "total_steps": step,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if not isinstance(v, list)}, f, indent=2)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train ENGRAM-FMx on synthetic tasks")

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--task", type=str, default="associative_recall",
                        choices=["associative_recall", "delayed_copy", "neural_dynamics"])
    parser.add_argument("--model_size", type=str, default="tiny",
                        choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="runs/engram_fmx")

    # Ablation flags
    parser.add_argument("--no_memory", action="store_true", help="Disable attractor memory")
    parser.add_argument("--no_operator", action="store_true", help="Disable operator dynamics")
    parser.add_argument("--no_sparse_attention", action="store_true", help="Disable sparse anchor attention")
    parser.add_argument("--ssm_only", action="store_true", help="SSM-only baseline")

    args = parser.parse_args()

    # Load config from file or create from args
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfig(
            task=args.task,
            model_size=args.model_size,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            output_dir=args.output_dir,
        )

    # Apply ablation flags
    if args.no_memory:
        config.use_attractor_memory = False
    if args.no_operator:
        config.use_operator_dynamics = False
    if args.no_sparse_attention:
        config.use_sparse_anchor_attention = False
    if args.ssm_only:
        config.use_latent_workspace = False
        config.use_attractor_memory = False
        config.use_operator_dynamics = False
        config.use_sparse_anchor_attention = False

    # Train
    results = train(config)

    return results


if __name__ == "__main__":
    main()
