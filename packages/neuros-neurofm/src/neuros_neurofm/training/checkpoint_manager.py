"""
Checkpoint Manager for NeuroFMX
Handles saving, loading, and resumption of training state
"""

import os
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from datetime import datetime


@dataclass
class TrainingState:
    """Training state for resumption"""
    global_step: int
    epoch: int
    best_val_loss: float
    data_cursor: Dict[str, Any]  # Shard index, sample offset
    timestamp: str


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup and resumption support

    Features:
    - Save model, optimizer, scheduler, and data cursor
    - Keep top-K checkpoints by validation loss
    - Automatic cleanup of old checkpoints
    - Resume from latest or best checkpoint
    - Periodic saving and emergency saves

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir="checkpoints",
        ...     save_every_n_steps=1000,
        ...     keep_top_k=3
        ... )
        >>> manager.save(model, optimizer, scheduler, global_step=5000,
        ...              metrics={"val_loss": 0.5}, data_cursor={"shard": 10, "offset": 500})
        >>> state = manager.load_latest(model, optimizer, scheduler)
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_every_n_steps: int = 1000,
        keep_top_k: int = 3,
        keep_last_n: int = 2,
        save_best_only: bool = False,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every_n_steps: Save frequency
            keep_top_k: Keep top K checkpoints by validation loss
            keep_last_n: Keep last N checkpoints regardless of performance
            save_best_only: Only save when validation improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_every_n_steps = save_every_n_steps
        self.keep_top_k = keep_top_k
        self.keep_last_n = keep_last_n
        self.save_best_only = save_best_only

        self.best_val_loss = float("inf")
        self.checkpoint_history: List[Dict[str, Any]] = []

        # Load checkpoint history if exists
        self._load_history()

    def _load_history(self):
        """Load checkpoint history from disk"""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                self.checkpoint_history = json.load(f)

            # Update best val loss
            if self.checkpoint_history:
                self.best_val_loss = min(
                    ckpt.get("val_loss", float("inf"))
                    for ckpt in self.checkpoint_history
                )

    def _save_history(self):
        """Save checkpoint history to disk"""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        with open(history_path, "w") as f:
            json.dump(self.checkpoint_history, f, indent=2)

    def should_save(self, global_step: int, val_loss: Optional[float] = None) -> bool:
        """
        Determine if checkpoint should be saved

        Args:
            global_step: Current training step
            val_loss: Current validation loss

        Returns:
            True if should save checkpoint
        """
        # Always save at specified intervals
        if global_step % self.save_every_n_steps == 0:
            if self.save_best_only and val_loss is not None:
                return val_loss < self.best_val_loss
            return True

        # Save if new best
        if val_loss is not None and val_loss < self.best_val_loss:
            return True

        return False

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        global_step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        data_cursor: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save checkpoint

        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            global_step: Current training step
            epoch: Current epoch
            metrics: Training metrics (must include "val_loss")
            data_cursor: Data loading state for resumption
            is_best: Mark as best checkpoint

        Returns:
            Path to saved checkpoint
        """
        metrics = metrics or {}
        val_loss = metrics.get("val_loss", float("inf"))

        # Update best val loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            is_best = True

        # Create checkpoint
        checkpoint = {
            "global_step": global_step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "metrics": metrics,
            "data_cursor": data_cursor or {},
            "timestamp": datetime.now().isoformat(),
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save checkpoint
        checkpoint_name = f"checkpoint_step_{global_step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Add to history
        self.checkpoint_history.append({
            "path": str(checkpoint_path),
            "global_step": global_step,
            "epoch": epoch,
            "val_loss": val_loss,
            "timestamp": checkpoint["timestamp"],
            "is_best": is_best,
        })

        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            shutil.copy(checkpoint_path, best_path)
            print(f"Best checkpoint saved: {best_path}")

        # Save latest checkpoint link
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy(checkpoint_path, latest_path)

        # Save history
        self._save_history()

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        return str(checkpoint_path)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints based on keep_top_k and keep_last_n"""
        if len(self.checkpoint_history) <= self.keep_top_k + self.keep_last_n:
            return

        # Sort by val_loss (ascending)
        sorted_by_loss = sorted(
            self.checkpoint_history,
            key=lambda x: x.get("val_loss", float("inf"))
        )

        # Sort by step (descending for recent)
        sorted_by_step = sorted(
            self.checkpoint_history,
            key=lambda x: x["global_step"],
            reverse=True
        )

        # Keep top K by performance
        keep_top = set(ckpt["path"] for ckpt in sorted_by_loss[:self.keep_top_k])

        # Keep last N by recency
        keep_last = set(ckpt["path"] for ckpt in sorted_by_step[:self.keep_last_n])

        # Keep best checkpoint
        keep_best = set(
            ckpt["path"] for ckpt in self.checkpoint_history if ckpt.get("is_best")
        )

        # Combine all checkpoints to keep
        keep_paths = keep_top | keep_last | keep_best

        # Remove checkpoints not in keep set
        new_history = []
        for ckpt in self.checkpoint_history:
            if ckpt["path"] in keep_paths:
                new_history.append(ckpt)
            else:
                # Delete checkpoint file
                checkpoint_path = Path(ckpt["path"])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print(f"Removed old checkpoint: {checkpoint_path}")

        self.checkpoint_history = new_history
        self._save_history()

    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> TrainingState:
        """
        Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            strict: Strict state dict loading

        Returns:
            TrainingState with resumption info
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load model
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Load optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Create training state
        state = TrainingState(
            global_step=checkpoint.get("global_step", 0),
            epoch=checkpoint.get("epoch", 0),
            best_val_loss=checkpoint.get("best_val_loss", float("inf")),
            data_cursor=checkpoint.get("data_cursor", {}),
            timestamp=checkpoint.get("timestamp", ""),
        )

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Step: {state.global_step}, Epoch: {state.epoch}")
        print(f"  Best val loss: {state.best_val_loss:.4f}")

        return state

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Optional[TrainingState]:
        """
        Load latest checkpoint

        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Scheduler

        Returns:
            TrainingState or None if no checkpoint exists
        """
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if not latest_path.exists():
            print("No checkpoint found, starting from scratch")
            return None

        return self.load(str(latest_path), model, optimizer, scheduler)

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Optional[TrainingState]:
        """
        Load best checkpoint

        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Scheduler

        Returns:
            TrainingState or None if no best checkpoint exists
        """
        best_path = self.checkpoint_dir / "best_checkpoint.pt"
        if not best_path.exists():
            print("No best checkpoint found")
            return None

        return self.load(str(best_path), model, optimizer, scheduler)

    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information for resuming training

        Returns:
            Dictionary with resume information or None
        """
        if not self.checkpoint_history:
            return None

        latest = max(self.checkpoint_history, key=lambda x: x["global_step"])
        return {
            "checkpoint_path": latest["path"],
            "global_step": latest["global_step"],
            "epoch": latest["epoch"],
            "val_loss": latest.get("val_loss"),
        }


class DistributedCheckpointManager(CheckpointManager):
    """
    Checkpoint manager for distributed training

    Only rank 0 saves checkpoints, all ranks can load
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_main_process = self._is_main_process()

    @staticmethod
    def _is_main_process() -> bool:
        """Check if current process is main (rank 0)"""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def save(self, *args, **kwargs) -> str:
        """Save checkpoint (rank 0 only)"""
        if not self.is_main_process:
            return ""

        return super().save(*args, **kwargs)

    def _cleanup_checkpoints(self):
        """Cleanup checkpoints (rank 0 only)"""
        if not self.is_main_process:
            return

        super()._cleanup_checkpoints()
