"""
Main training script for NMR-ChemBERTa
"""

import os
import torch
import logging
from tqdm import tqdm
import time
from pathlib import Path

from config import Config, get_default_config
from nmr_chemberta_model import NMRChemBERTa
from nmr_dataset import create_data_loaders
from hardware_utils import HardwareOptimizer
from training_utils import (
    MultiTaskLoss,
    MetricsCalculator,
    EarlyStopping,
    ModelCheckpoint,
    setup_optimizer_and_scheduler,
    prepare_batch_targets,
    TrainingLogger,
    GradientClipping,
    compute_model_size
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, 
                device, scaler, use_amp, gradient_clipper):
    """Train for one epoch"""
    model.train()
    metrics_calculator = MetricsCalculator()
    epoch_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device, non_blocking=True)
            elif isinstance(value, dict):
                batch_device[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        batch_device[key][subkey] = subvalue.to(device, non_blocking=True)
            else:
                batch_device[key] = value
        
        # Prepare targets
        targets, masks = prepare_batch_targets(batch_device, device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(
                input_ids=batch_device['input_ids'],
                attention_mask=batch_device['attention_mask'],
                coords=batch_device['coords'],
                atom_types=batch_device['atom_types'],
                atom_mask=batch_device['atom_mask'],
                nmr_features=batch_device['nmr_features']
            )
            
            # Compute losses
            losses = loss_fn(predictions, targets, masks)
            loss = losses['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = gradient_clipper.clip_gradients(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = gradient_clipper.clip_gradients(model)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        metrics_calculator.update(predictions, targets, masks)
        epoch_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'grad_norm': f'{grad_norm:.2f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    # Compute epoch metrics
    metrics = metrics_calculator.compute_metrics()
    metrics['total_loss'] = epoch_loss / len(train_loader)
    
    return metrics


def validate_epoch(model, val_loader, loss_fn, device, use_amp):
    """Validate for one epoch"""
    model.eval()
    metrics_calculator = MetricsCalculator()
    epoch_loss = 0.0
    
    progress_bar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device, non_blocking=True)
                elif isinstance(value, dict):
                    batch_device[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, torch.Tensor):
                            batch_device[key][subkey] = subvalue.to(device, non_blocking=True)
                else:
                    batch_device[key] = value
            
            # Prepare targets
            targets, masks = prepare_batch_targets(batch_device, device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                predictions = model(
                    input_ids=batch_device['input_ids'],
                    attention_mask=batch_device['attention_mask'],
                    coords=batch_device['coords'],
                    atom_types=batch_device['atom_types'],
                    atom_mask=batch_device['atom_mask'],
                    nmr_features=batch_device['nmr_features']
                )
                
                # Compute losses
                losses = loss_fn(predictions, targets, masks)
                loss = losses['total_loss']
            
            # Update metrics
            metrics_calculator.update(predictions, targets, masks)
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute epoch metrics
    metrics = metrics_calculator.compute_metrics()
    metrics['total_loss'] = epoch_loss / len(val_loader)
    
    return metrics


def main():
    """Main training function"""
    
    # Load configuration
    config = get_default_config()
    
    # You can also load from YAML if you prefer
    # config = Config.from_yaml('config.yaml')
    
    # Setup hardware
    hardware_optimizer = HardwareOptimizer(config)
    device, scaler, use_amp = hardware_optimizer.setup()
    
    logger.info(f"Using device: {device}")
    logger.info(f"Mixed precision training: {use_amp}")
    
    # Create data loaders
    logger.info("Loading data...")
    try:
        train_loader, val_loader, test_loader, dataset = create_data_loaders(config)
        logger.info(f"Data loaded successfully!")
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Create model
    logger.info("Creating model...")
    model = NMRChemBERTa(config)
    model = hardware_optimizer.optimize_model(model)
    
    # Log model size
    model_size = compute_model_size(model)
    logger.info(f"Model created with {model_size['trainable_parameters']:,} trainable parameters")
    logger.info(f"Estimated memory: {model_size['estimated_memory_mb']:.1f} MB")
    
    # Setup loss function
    loss_fn = MultiTaskLoss(config)
    
    # Setup optimizer and scheduler
    num_training_steps = len(train_loader) * config.training.num_epochs
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config, num_training_steps)
    
    # Setup training utilities
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        restore_best_weights=True
    )
    
    checkpoint_manager = ModelCheckpoint(
        checkpoint_dir=config.logging.checkpoint_dir,
        save_every_n_epochs=config.training.save_every_n_epochs
    )
    
    training_logger = TrainingLogger(config, use_wandb=config.logging.use_wandb)
    gradient_clipper = GradientClipping(max_norm=config.training.gradient_clip_norm)
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(1, config.training.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, 
            loss_fn, device, scaler, use_amp, gradient_clipper
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, use_amp
        )
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        training_logger.log_epoch(
            epoch, train_metrics, val_metrics, current_lr, epoch_time
        )
        
        # Save checkpoint
        is_best = val_metrics['total_loss'] < checkpoint_manager.best_score
        checkpoint_manager.save(
            model, optimizer, scheduler, epoch, 
            val_metrics['total_loss'], is_best
        )
        
        # Check early stopping
        if early_stopping(val_metrics['total_loss'], model):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save final model
    final_path = Path(config.logging.checkpoint_dir) / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training completed! Final model saved to {final_path}")
    
    # Save training history
    history_path = Path(config.logging.log_dir) / 'training_history.json'
    training_logger.save_history(str(history_path))
    
    # Test model if test loader exists
    if test_loader:
        logger.info("Running test evaluation...")
        test_metrics = validate_epoch(
            model, test_loader, loss_fn, device, use_amp
        )
        logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()