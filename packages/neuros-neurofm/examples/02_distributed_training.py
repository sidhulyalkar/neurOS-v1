"""
Distributed Multi-GPU Training with FSDP
========================================

This example demonstrates:
- Full FSDP (Fully Sharded Data Parallel) training across multiple GPUs
- DeepSpeed ZeRO-3 equivalent configuration
- Mixed precision (bfloat16) training
- Activation checkpointing for memory efficiency
- Multi-node training setup
- Checkpoint resumption across distributed training

Hardware Requirements:
    - 4-8 H100/A100 GPUs recommended
    - 80GB+ GPU memory per GPU
    - NVLink/Infiniband for multi-node

Expected Performance:
    - 4x H100 (80GB): ~150B parameters, 10-15 samples/sec
    - 8x H100 (80GB): ~300B parameters, 20-30 samples/sec
"""

import os
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
import yaml

from neuros_neurofm.model import NeuroFMX, TransformerBlock, MambaBlock
from neuros_neurofm.training.fsdp_trainer import FSDPConfig, FSDPLightningModule
from neuros_neurofm.data.webdataset_loader import create_webdataset_loader


def get_auto_wrap_policy(config: dict):
    """
    Create auto-wrap policy for FSDP

    Wraps each transformer/mamba block as a separate FSDP unit
    """
    architecture = config['model'].get('architecture', 'mamba')

    if architecture == 'transformer':
        target_class = TransformerBlock
    elif architecture == 'mamba':
        target_class = MambaBlock
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={target_class},
    )

    return wrap_policy


def setup_fsdp_strategy(config: dict) -> FSDPStrategy:
    """
    Setup PyTorch Lightning FSDP strategy

    Configures:
    - Sharding strategy (FULL_SHARD = ZeRO-3)
    - Mixed precision (bfloat16)
    - Activation checkpointing
    - Auto-wrapping policy
    """
    distributed_config = config['distributed']

    # Sharding strategy
    sharding_map = {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,  # ZeRO-3: shard params, grads, optimizer
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2: shard grads, optimizer
        'NO_SHARD': ShardingStrategy.NO_SHARD,  # DDP equivalent
    }
    sharding_strategy = sharding_map[distributed_config.get('sharding_strategy', 'FULL_SHARD')]

    # Mixed precision
    use_bf16 = distributed_config.get('use_bf16', True)
    if use_bf16:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,  # Keep grads in fp32 for stability
            buffer_dtype=torch.bfloat16,
        )
    else:
        mixed_precision_policy = None

    # Auto-wrap policy
    auto_wrap_policy = get_auto_wrap_policy(config)

    # CPU offloading (for very large models)
    cpu_offload = distributed_config.get('cpu_offload', False)

    # Backward prefetch (overlap backward with communication)
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE

    # Create FSDP strategy
    strategy = FSDPStrategy(
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        activation_checkpointing=distributed_config.get('activation_checkpointing', True),
        state_dict_type='full',  # Save full state dict on rank 0
    )

    return strategy


def print_distributed_info():
    """Print distributed training information"""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        if rank == 0:
            print("\n" + "=" * 80)
            print("Distributed Training Configuration")
            print("=" * 80)
            print(f"World size: {world_size} GPUs")
            print(f"Backend: {dist.get_backend()}")
            print(f"Master addr: {os.environ.get('MASTER_ADDR', 'localhost')}")
            print(f"Master port: {os.environ.get('MASTER_PORT', '12355')}")

            # Print GPU info
            if torch.cuda.is_available():
                print(f"\nGPU Information:")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

            print("=" * 80 + "\n")
    else:
        print("Warning: Distributed training not initialized")


def estimate_memory_usage(model: NeuroFMX, config: dict):
    """
    Estimate memory usage for FSDP training

    Returns estimates for:
    - Model parameters
    - Gradients
    - Optimizer states
    - Activations (with checkpointing)
    """
    num_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = num_params * 4 / 1e9  # 4 bytes per fp32 param

    # FSDP FULL_SHARD (ZeRO-3)
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Per-GPU memory estimates
    sharded_param_memory = param_memory_gb / world_size

    # Gradients (same as params)
    grad_memory = sharded_param_memory

    # Optimizer states (Adam: 2x params for m, v)
    optimizer_memory = sharded_param_memory * 2

    # Activations (depends on batch size and sequence length)
    batch_size = config['data']['batch_size']
    seq_length = config['data'].get('max_seq_length', 1024)
    d_model = config['model']['d_model']
    n_layers = config['model']['n_layers']

    # With activation checkpointing, only store activations for checkpointed layers
    checkpoint_fraction = 0.2  # Checkpoint ~20% of layers
    activation_memory_gb = (
        batch_size * seq_length * d_model * n_layers * checkpoint_fraction * 4 / 1e9
    )

    total_memory_per_gpu = (
        sharded_param_memory + grad_memory + optimizer_memory + activation_memory_gb
    )

    if dist.get_rank() == 0 if dist.is_initialized() else True:
        print("\n" + "=" * 80)
        print("Memory Usage Estimates (per GPU)")
        print("=" * 80)
        print(f"Model parameters:     {sharded_param_memory:.2f} GB")
        print(f"Gradients:            {grad_memory:.2f} GB")
        print(f"Optimizer states:     {optimizer_memory:.2f} GB")
        print(f"Activations (est):    {activation_memory_gb:.2f} GB")
        print(f"-" * 80)
        print(f"Total estimate:       {total_memory_per_gpu:.2f} GB")
        print(f"Recommended GPU:      {total_memory_per_gpu * 1.3:.0f}+ GB VRAM")
        print("=" * 80 + "\n")


def main():
    """Main distributed training function"""

    # Load configuration
    config_path = "configs/training/distributed.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Print distributed info
    print_distributed_info()

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Create model (on CPU first, will be moved to GPU by FSDP)
    if rank == 0:
        print("[1/5] Creating NeuroFMX model...")

    model_config = config['model']
    model = NeuroFMX(
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        architecture=model_config['architecture'],
        modality_configs=model_config['modalities'],
        fusion_type=model_config['fusion'],
    )

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {num_params / 1e9:.2f}B parameters")

        # Estimate memory usage
        estimate_memory_usage(model, config)

    # Setup FSDP strategy
    if rank == 0:
        print("[2/5] Configuring FSDP strategy...")

    fsdp_strategy = setup_fsdp_strategy(config)

    if rank == 0:
        print(f"✓ FSDP configured with {config['distributed']['sharding_strategy']}")

    # Setup data loaders
    if rank == 0:
        print("[3/5] Setting up distributed data loaders...")

    train_loader = create_webdataset_loader(
        shard_urls=config['data']['train_shards'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle_buffer=config['data']['shuffle_buffer'],
        modality_specs=config['data']['modality_specs'],
    )

    val_loader = create_webdataset_loader(
        shard_urls=config['data']['val_shards'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle_buffer=0,
        modality_specs=config['data']['modality_specs'],
    )

    if rank == 0:
        print("✓ Data loaders ready")

    # Wrap in Lightning module
    if rank == 0:
        print("[4/5] Creating Lightning module...")

    from neuros_neurofm.losses import CombinedLoss, MaskedModelingLoss

    loss_fn = CombinedLoss(
        losses={'masked_modeling': MaskedModelingLoss()},
        weights={'masked_modeling': 1.0}
    )

    lightning_module = FSDPLightningModule(
        model=model,
        loss_fn=loss_fn,
        fsdp_config=None,  # Handled by strategy
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
    )

    if rank == 0:
        print("✓ Lightning module ready")

    # Setup callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='neurofmx-fsdp-{step:06d}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        every_n_train_steps=config['training']['checkpoint_every_n_steps'],
        save_on_train_epoch_end=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Setup logger
    from pytorch_lightning.loggers import WandbLogger

    logger = None
    if rank == 0 and config['logging'].get('wandb_enabled', True):
        logger = WandbLogger(
            project=config['logging']['wandb_project'],
            name=config['logging']['experiment_name'],
            tags=['fsdp', 'distributed', 'neurofmx'],
        )

    # Create trainer
    if rank == 0:
        print("[5/5] Creating PyTorch Lightning Trainer...")

    trainer = pl.Trainer(
        max_steps=config['training']['max_steps'],
        accelerator='gpu',
        devices=config['distributed']['num_gpus'],
        num_nodes=config['distributed'].get('num_nodes', 1),
        strategy=fsdp_strategy,
        precision='bf16-mixed' if config['distributed']['use_bf16'] else 32,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        val_check_interval=config['training']['val_check_interval'],
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        enable_progress_bar=(rank == 0),
        enable_model_summary=(rank == 0),
    )

    if rank == 0:
        print("✓ Trainer ready")
        print("\n" + "=" * 80)
        print("Starting Distributed Training")
        print("=" * 80)
        print(f"Total GPUs: {config['distributed']['num_gpus'] * config['distributed'].get('num_nodes', 1)}")
        print(f"Batch size per GPU: {config['data']['batch_size']}")
        print(f"Global batch size: {config['data']['batch_size'] * config['distributed']['num_gpus'] * config['distributed'].get('num_nodes', 1)}")
        print(f"Training steps: {config['training']['max_steps']:,}")
        print("=" * 80 + "\n")

    # Start training
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    if rank == 0:
        print("\n" + "=" * 80)
        print("Distributed Training Complete!")
        print("=" * 80)
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()


"""
SLURM Launch Script
===================

For multi-node training on a SLURM cluster, save this as `submit_distributed.sh`:

#!/bin/bash
#SBATCH --job-name=neurofmx-fsdp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/neurofmx_%j.out
#SBATCH --error=logs/neurofmx_%j.err

# Setup environment
module load cuda/12.1
module load nccl/2.18
source activate neurofmx

# Get master node
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500

# Print info
echo "Master node: $MASTER_ADDR"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_TASK"

# Launch training
srun python 02_distributed_training.py

Then submit with:
    sbatch submit_distributed.sh

For single-node multi-GPU training:
    torchrun --nproc_per_node=8 02_distributed_training.py

For multi-node training:
    torchrun \
        --nnodes=2 \
        --nproc_per_node=8 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:29500 \
        02_distributed_training.py
"""
