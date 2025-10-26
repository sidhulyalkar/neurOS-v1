"""
Complete Training Workflow for NeuroFMX
=======================================

This example demonstrates the full training pipeline including:
- Multi-modal data loading with WebDataset
- Curriculum learning (unimodal → pairwise → multimodal)
- Multiple training objectives (masked modeling, forecasting, diffusion, contrastive)
- Distributed training with FSDP
- Checkpoint management and resumption
- MLflow experiment tracking

Requirements:
    - 4+ GPUs recommended (can run on 1 GPU for testing)
    - 100+ GB disk space for checkpoints
    - MLflow server running (optional)
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
import yaml

from neuros_neurofm.model import NeuroFMX
from neuros_neurofm.data.webdataset_loader import create_webdataset_loader
from neuros_neurofm.training.curriculum_scheduler import CurriculumScheduler, CurriculumConfig
from neuros_neurofm.training.fsdp_trainer import FSDPConfig, FSDPLightningModule
from neuros_neurofm.training.checkpoint_manager import CheckpointManager
from neuros_neurofm.losses import (
    MaskedModelingLoss,
    MultiHorizonForecastingLoss,
    DiffusionLoss,
    CrossModalContrastiveLoss,
    CombinedLoss
)
from neuros_neurofm.augmentation import MultiModalAugmentation


def setup_model(config: dict) -> NeuroFMX:
    """Create NeuroFMX model from config"""
    model_config = config['model']

    model = NeuroFMX(
        # Backbone
        d_model=model_config.get('d_model', 768),
        n_layers=model_config.get('n_layers', 12),
        n_heads=model_config.get('n_heads', 12),
        architecture=model_config.get('architecture', 'mamba'),  # 'mamba' or 'transformer'

        # Modality configuration
        modality_configs=model_config['modalities'],

        # Fusion
        fusion_type=model_config.get('fusion', 'perceiver'),  # 'perceiver' or 'attention'
        fusion_latents=model_config.get('fusion_latents', 256),

        # Optional components
        enable_lora=model_config.get('enable_lora', False),
        lora_rank=model_config.get('lora_rank', 8),
    )

    return model


def setup_losses(config: dict) -> CombinedLoss:
    """Setup multi-objective loss function"""
    loss_config = config['losses']

    losses = {}
    weights = {}

    # Masked modeling loss
    if loss_config.get('masked_modeling', {}).get('enabled', True):
        mm_config = loss_config['masked_modeling']
        losses['masked_modeling'] = MaskedModelingLoss(
            mask_ratio=mm_config.get('mask_ratio', 0.15),
            mask_strategy=mm_config.get('strategy', 'random'),  # random, block, adaptive
        )
        weights['masked_modeling'] = mm_config.get('weight', 1.0)

    # Multi-horizon forecasting
    if loss_config.get('forecasting', {}).get('enabled', True):
        fc_config = loss_config['forecasting']
        losses['forecasting'] = MultiHorizonForecastingLoss(
            horizons_ms=fc_config.get('horizons_ms', [100, 250, 500, 1000]),
            distance_weighting=fc_config.get('distance_weighting', 'exponential'),
        )
        weights['forecasting'] = fc_config.get('weight', 0.5)

    # Diffusion denoising
    if loss_config.get('diffusion', {}).get('enabled', False):
        diff_config = loss_config['diffusion']
        losses['diffusion'] = DiffusionLoss(
            num_timesteps=diff_config.get('num_timesteps', 1000),
            noise_schedule=diff_config.get('noise_schedule', 'cosine'),
        )
        weights['diffusion'] = diff_config.get('weight', 0.3)

    # Cross-modal contrastive
    if loss_config.get('contrastive', {}).get('enabled', True):
        cont_config = loss_config['contrastive']
        losses['contrastive'] = CrossModalContrastiveLoss(
            temperature=cont_config.get('temperature', 0.07),
            negative_sampling=cont_config.get('negative_sampling', 'hard'),
        )
        weights['contrastive'] = cont_config.get('weight', 0.3)

    combined_loss = CombinedLoss(
        losses=losses,
        weights=weights,
        adaptive_weights=loss_config.get('adaptive_weights', True)
    )

    return combined_loss


def setup_data_loaders(config: dict, stage: str = 'train'):
    """Setup WebDataset data loaders with curriculum"""
    data_config = config['data']

    # Base WebDataset configuration
    loader_config = {
        'shard_urls': data_config[f'{stage}_shards'],
        'batch_size': data_config.get('batch_size', 32),
        'num_workers': data_config.get('num_workers', 8),
        'shuffle_buffer': data_config.get('shuffle_buffer', 10000) if stage == 'train' else 0,
        'modality_specs': data_config['modality_specs'],
    }

    loader = create_webdataset_loader(**loader_config)

    return loader


def setup_augmentation(config: dict) -> MultiModalAugmentation:
    """Setup data augmentation pipeline"""
    aug_config = config['augmentation']

    augmentation = MultiModalAugmentation(
        modality_dropout_prob=aug_config.get('modality_dropout_prob', 0.1),
        min_modalities=aug_config.get('min_modalities', 1),
        time_mask_param=aug_config.get('time_mask_param', 20),
        channel_mask_param=aug_config.get('channel_mask_param', 5),
        noise_std=aug_config.get('noise_std', 0.05),
        apply_time_warp=aug_config.get('apply_time_warp', False),
        apply_mixup=aug_config.get('apply_mixup', False),
    )

    return augmentation


def setup_curriculum(config: dict) -> CurriculumScheduler:
    """Setup curriculum learning scheduler"""
    curriculum_config = config.get('curriculum', {})

    if not curriculum_config.get('enabled', True):
        return None

    # Define 3-stage curriculum
    stages = [
        # Stage 1: Unimodal learning
        {
            'name': 'unimodal',
            'duration_steps': curriculum_config.get('unimodal_steps', 10000),
            'modality_pairs': None,  # Single modalities only
            'learning_rate_multiplier': 1.0,
            'loss_weights': {'masked_modeling': 1.0}
        },
        # Stage 2: Pairwise cross-modal
        {
            'name': 'pairwise',
            'duration_steps': curriculum_config.get('pairwise_steps', 20000),
            'modality_pairs': curriculum_config.get('modality_pairs', [
                ('eeg', 'spikes'),
                ('eeg', 'video'),
                ('spikes', 'video'),
            ]),
            'learning_rate_multiplier': 0.5,
            'loss_weights': {
                'masked_modeling': 0.7,
                'contrastive': 0.3
            }
        },
        # Stage 3: Full multimodal
        {
            'name': 'multimodal',
            'duration_steps': curriculum_config.get('multimodal_steps', 50000),
            'modality_pairs': 'all',
            'learning_rate_multiplier': 0.3,
            'loss_weights': {
                'masked_modeling': 0.4,
                'forecasting': 0.3,
                'contrastive': 0.3
            }
        }
    ]

    curriculum = CurriculumScheduler(
        stages=stages,
        transition_steps=curriculum_config.get('transition_steps', 1000),
        warmup_steps=curriculum_config.get('warmup_steps', 1000),
    )

    return curriculum


def main():
    """Main training function"""

    # Load configuration
    config_path = "configs/training/default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("NeuroFMX Complete Training Workflow")
    print("=" * 80)

    # 1. Setup model
    print("\n[1/8] Creating NeuroFMX model...")
    model = setup_model(config)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # 2. Setup losses
    print("\n[2/8] Setting up multi-objective losses...")
    loss_fn = setup_losses(config)
    print(f"✓ Combined loss with {len(loss_fn.losses)} objectives")

    # 3. Setup augmentation
    print("\n[3/8] Setting up augmentation pipeline...")
    augmentation = setup_augmentation(config)
    print("✓ Augmentation configured")

    # 4. Setup curriculum
    print("\n[4/8] Setting up curriculum learning...")
    curriculum = setup_curriculum(config)
    if curriculum:
        print(f"✓ Curriculum with {len(curriculum.stages)} stages")
    else:
        print("✓ Curriculum disabled")

    # 5. Setup data loaders
    print("\n[5/8] Setting up data loaders...")
    train_loader = setup_data_loaders(config, stage='train')
    val_loader = setup_data_loaders(config, stage='val')
    print(f"✓ Data loaders ready")

    # 6. Setup distributed training (FSDP)
    print("\n[6/8] Configuring distributed training...")
    fsdp_config = FSDPConfig(
        sharding_strategy=config['distributed'].get('sharding_strategy', 'FULL_SHARD'),
        cpu_offload=config['distributed'].get('cpu_offload', False),
        mixed_precision='bf16' if config['training'].get('use_bf16', True) else 'fp32',
        activation_checkpointing=config['distributed'].get('activation_checkpointing', True),
    )

    lightning_module = FSDPLightningModule(
        model=model,
        loss_fn=loss_fn,
        augmentation=augmentation,
        curriculum=curriculum,
        fsdp_config=fsdp_config,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training'].get('warmup_steps', 1000),
        max_steps=config['training']['max_steps'],
    )
    print("✓ FSDP configured")

    # 7. Setup callbacks and logging
    print("\n[7/8] Setting up callbacks and logging...")

    # Checkpoint manager
    checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints/')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='neurofmx-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        every_n_train_steps=config['training'].get('checkpoint_every_n_steps', 5000),
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # MLflow logger
    mlflow_logger = None
    if config['logging'].get('mlflow_enabled', True):
        mlflow_logger = MLFlowLogger(
            experiment_name=config['logging'].get('experiment_name', 'neurofmx'),
            tracking_uri=config['logging'].get('mlflow_uri', 'http://localhost:5000'),
            tags={
                'model': 'NeuroFMX',
                'curriculum': curriculum is not None,
                'distributed': 'FSDP',
            }
        )

    print("✓ Callbacks and logging ready")

    # 8. Create trainer and start training
    print("\n[8/8] Initializing PyTorch Lightning Trainer...")

    trainer = pl.Trainer(
        max_steps=config['training']['max_steps'],
        accelerator='gpu',
        devices=config['distributed'].get('num_gpus', -1),  # -1 = all available
        strategy='ddp',  # FSDP wrapped in LightningModule
        precision='bf16-mixed' if config['training'].get('use_bf16', True) else 32,
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        val_check_interval=config['training'].get('val_check_interval', 1000),
        callbacks=[checkpoint_callback, lr_monitor],
        logger=mlflow_logger,
        log_every_n_steps=config['logging'].get('log_every_n_steps', 10),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print("✓ Trainer ready")

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"Total steps: {config['training']['max_steps']:,}")
    print(f"Validation every: {config['training'].get('val_check_interval', 1000)} steps")
    print(f"Checkpoint every: {config['training'].get('checkpoint_every_n_steps', 5000)} steps")
    print(f"Using {trainer.num_devices} GPU(s)")
    print("=" * 80 + "\n")

    # Start training
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
