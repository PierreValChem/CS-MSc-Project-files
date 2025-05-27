"""
Training utilities, loss functions, and metrics for NMR-ChemBERTa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import time
from collections import defaultdict
import wandb
from pathlib import Path

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """Multi-task loss function for NMR-ChemBERTa"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Individual loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.l1_loss = nn.L1Loss(reduction='none')
        
        # Loss weights
        self.nmr_weight = config.training.nmr_loss_weight
        self.position_weight = config.training.position_loss_weight
        self.atom_type_weight = config.training.atom_type_loss_weight
        self.smiles_pos_weight = config.training.smiles_position_loss_weight
    
    def forward(self, predictions: Dict, targets: Dict, masks: Dict) -> Dict:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            masks: Masks for valid data points
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # NMR chemical shift prediction loss
        if 'nmr_shifts' in predictions and 'nmr_shifts' in targets:
            nmr_loss = self._compute_nmr_loss(
                predictions['nmr_shifts'],
                targets['nmr_shifts'],
                masks['nmr_mask']
            )
            losses['nmr_loss'] = nmr_loss
            total_loss += self.nmr_weight * nmr_loss
        
        # 3D position prediction loss
        if 'positions' in predictions and 'positions' in targets:
            pos_loss = self._compute_position_loss(
                predictions['positions'],
                targets['positions'],
                masks['atom_mask']
            )
            losses['position_loss'] = pos_loss
            total_loss += self.position_weight * pos_loss
        
        # Atom type classification loss
        if 'atom_types' in predictions and 'atom_types' in targets:
            atom_loss = self._compute_atom_type_loss(
                predictions['atom_types'],
                targets['atom_types'],
                masks['atom_mask']
            )
            losses['atom_type_loss'] = atom_loss
            total_loss += self.atom_type_weight * atom_loss
        
        # SMILES position prediction loss
        if 'smiles_positions' in predictions and 'smiles_positions' in targets:
            smiles_loss = self._compute_smiles_position_loss(
                predictions['smiles_positions'],
                targets['smiles_positions'],
                masks['atom_mask']
            )
            losses['smiles_position_loss'] = smiles_loss
            total_loss += self.smiles_pos_weight * smiles_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_nmr_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute NMR chemical shift prediction loss"""
        # pred: (batch_size, max_atoms, 2) - H and C shifts
        # target: (batch_size, max_atoms, 2)
        # mask: (batch_size, max_atoms, 2) - separate masks for H and C
        
        loss = self.mse_loss(pred, target)  # (batch_size, max_atoms, 2)
        masked_loss = loss * mask
        
        # Average over valid predictions
        num_valid = mask.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=pred.device)
    
    def _compute_position_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute 3D position prediction loss"""
        # pred: (batch_size, max_atoms, 3)
        # target: (batch_size, max_atoms, 3)
        # mask: (batch_size, max_atoms)
        
        loss = self.mse_loss(pred, target)  # (batch_size, max_atoms, 3)
        
        # Apply mask
        mask_3d = mask.unsqueeze(-1).expand_as(loss)
        masked_loss = loss * mask_3d
        
        num_valid = mask_3d.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=pred.device)
    
    def _compute_atom_type_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute atom type classification loss"""
        # pred: (batch_size, max_atoms, num_atom_types)
        # target: (batch_size, max_atoms)
        # mask: (batch_size, max_atoms)
        
        batch_size, max_atoms, num_classes = pred.shape
        
        # Debug: Check target values
        # print(f"Target min: {target.min()}, max: {target.max()}")
        # print(f"Unique targets: {torch.unique(target)}")
        
        # Replace -1 (padding) with 0, which will be masked out
        valid_target = torch.clamp(target, min=0, max=num_classes-1)
        
        # Reshape for cross entropy
        pred_flat = pred.view(-1, num_classes)
        target_flat = valid_target.view(-1)
        mask_flat = mask.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(pred_flat, target_flat, reduction='none')
        
        # Apply mask to ignore padding
        masked_loss = loss * mask_flat
        
        num_valid = mask_flat.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=pred.device)
    
    def _compute_smiles_position_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute SMILES position prediction loss"""
        # pred: (batch_size, max_atoms, 1)
        # target: (batch_size, max_atoms, 1)
        # mask: (batch_size, max_atoms)
        
        loss = self.l1_loss(pred, target).squeeze(-1)  # (batch_size, max_atoms)
        masked_loss = loss * mask
        
        num_valid = mask.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=pred.device)


class MetricsCalculator:
    """Calculate various metrics for model evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.metrics = defaultdict(list)
    
    def update(self, predictions: Dict, targets: Dict, masks: Dict):
        """Update metrics with new batch"""
        
        # NMR shift metrics
        if 'nmr_shifts' in predictions and 'nmr_shifts' in targets:
            self._update_nmr_metrics(
                predictions['nmr_shifts'],
                targets['nmr_shifts'],
                masks['nmr_mask']
            )
        
        # Position metrics
        if 'positions' in predictions and 'positions' in targets:
            self._update_position_metrics(
                predictions['positions'],
                targets['positions'],
                masks['atom_mask']
            )
        
        # Atom type metrics
        if 'atom_types' in predictions and 'atom_types' in targets:
            self._update_atom_type_metrics(
                predictions['atom_types'],
                targets['atom_types'],
                masks['atom_mask']
            )
    
    def _update_nmr_metrics(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Update NMR prediction metrics"""
        valid_pred = pred[mask.bool()]
        valid_target = target[mask.bool()]
        
        if len(valid_pred) > 0:
            # MAE for both H and C shifts
            mae = torch.abs(valid_pred - valid_target)
            self.metrics['nmr_mae'].extend(mae.cpu().numpy())
            
            # Separate H and C metrics
            h_mask = mask[:, :, 0].bool()
            c_mask = mask[:, :, 1].bool()
            
            if h_mask.any():
                h_mae = torch.abs(pred[:, :, 0][h_mask] - target[:, :, 0][h_mask])
                self.metrics['h_nmr_mae'].extend(h_mae.cpu().numpy())
            
            if c_mask.any():
                c_mae = torch.abs(pred[:, :, 1][c_mask] - target[:, :, 1][c_mask])
                self.metrics['c_nmr_mae'].extend(c_mae.cpu().numpy())
    
    def _update_position_metrics(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Update 3D position prediction metrics"""
        mask_3d = mask.unsqueeze(-1).expand_as(pred)
        valid_pred = pred[mask_3d.bool()].view(-1, 3)
        valid_target = target[mask_3d.bool()].view(-1, 3)
        
        if len(valid_pred) > 0:
            # RMSD
            squared_diff = torch.sum((valid_pred - valid_target) ** 2, dim=1)
            rmsd = torch.sqrt(squared_diff)
            self.metrics['position_rmsd'].extend(rmsd.cpu().numpy())
    
    def _update_atom_type_metrics(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Update atom type classification metrics"""
        pred_classes = torch.argmax(pred, dim=-1)
        
        valid_pred = pred_classes[mask.bool()]
        valid_target = target[mask.bool()]
        
        if len(valid_pred) > 0:
            accuracy = (valid_pred == valid_target).float()
            self.metrics['atom_type_accuracy'].extend(accuracy.cpu().numpy())
    
    def compute_metrics(self) -> Dict:
        """Compute final metrics from accumulated values"""
        computed_metrics = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                computed_metrics[f'{metric_name}_mean'] = np.mean(values)
                computed_metrics[f'{metric_name}_std'] = np.std(values)
        
        return computed_metrics


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop early
        
        Args:
            score: Validation score (lower is better)
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self.restore_checkpoint(model)
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        if self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def restore_checkpoint(self, model: nn.Module):
        """Restore best model weights"""
        if self.best_weights is not None:
            model.load_state_dict({k: v.cuda() if torch.cuda.is_available() else v 
                                 for k, v in self.best_weights.items()})


class ModelCheckpoint:
    """Save model checkpoints during training"""
    
    def __init__(self, checkpoint_dir: str, save_best: bool = True, save_every_n_epochs: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        
        self.best_score = float('inf')
    
    def save(self, model: nn.Module, optimizer, scheduler, epoch: int, 
             score: float = None, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'score': score,
            'timestamp': time.time()
        }
        
        # Save regular checkpoint
        if epoch % self.save_every_n_epochs == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best and score is not None and score < self.best_score:
            self.best_score = score
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path} (score: {score:.6f})")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)


def setup_optimizer_and_scheduler(model: nn.Module, config, num_training_steps: int):
    """Setup optimizer and learning rate scheduler"""
    
    # Separate parameters for different components
    chemberta_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'chemberta' in name:
                chemberta_params.append(param)
            else:
                other_params.append(param)
    
    # Different learning rates for pre-trained and new components
    param_groups = []
    
    if chemberta_params:
        param_groups.append({
            'params': chemberta_params,
            'lr': config.training.learning_rate * 0.1,  # Lower LR for pre-trained
            'name': 'chemberta'
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': config.training.learning_rate,
            'name': 'other'
        })
    
    # Create optimizer
    optimizer = AdamW(
        param_groups,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        eps=1e-8
    )
    
    # Create scheduler
    if config.training.warmup_steps > 0:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[group['lr'] for group in param_groups],
            total_steps=num_training_steps,
            pct_start=config.training.warmup_steps / num_training_steps,
            anneal_strategy='cos'
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_training_steps // 4,
            T_mult=2,
            eta_min=1e-7
        )
    
    return optimizer, scheduler


def prepare_batch_targets(batch: Dict, device: torch.device) -> Tuple[Dict, Dict]:
    """Prepare targets and masks from batch data"""
    
    targets = {}
    masks = {}
    
    # NMR targets and masks
    nmr_features = batch['nmr_features']
    targets['nmr_shifts'] = torch.stack([
        nmr_features['h_shifts'],
        nmr_features['c_shifts']
    ], dim=-1).to(device)  # (batch_size, max_atoms, 2)
    
    masks['nmr_mask'] = torch.stack([
        nmr_features['h_mask'],
        nmr_features['c_mask']
    ], dim=-1).to(device)  # (batch_size, max_atoms, 2)
    
    # Position targets
    targets['positions'] = batch['coords'].to(device)
    
    # Atom type targets
    targets['atom_types'] = batch['atom_types'].to(device)
    
    # SMILES position targets (placeholder - would need proper implementation)
    targets['smiles_positions'] = torch.zeros(
        batch['atom_types'].shape + (1,), 
        device=device, 
        dtype=torch.float32
    )
    
    # Atom mask
    masks['atom_mask'] = batch['atom_mask'].to(device)
    
    return targets, masks


class TrainingLogger:
    """Enhanced logging for training process"""
    
    def __init__(self, config, use_wandb: bool = False):
        self.config = config
        self.use_wandb = use_wandb
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="nmr-chemberta",
                name=config.logging.experiment_name,
                config=config.__dict__
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                  lr: float, epoch_time: float):
        """Log metrics for an epoch"""
        
        # Store in history
        self.history['train_loss'].append(train_metrics.get('total_loss', 0))
        self.history['val_loss'].append(val_metrics.get('total_loss', 0))
        self.history['learning_rate'].append(lr)
        self.history['epoch_times'].append(epoch_time)
        
        # Console logging
        logger.info(f"Epoch {epoch:3d} | "
                   f"Train Loss: {train_metrics.get('total_loss', 0):.6f} | "
                   f"Val Loss: {val_metrics.get('total_loss', 0):.6f} | "
                   f"LR: {lr:.2e} | "
                   f"Time: {epoch_time:.1f}s")
        
        # Detailed metrics logging
        for key, value in train_metrics.items():
            if key != 'total_loss':
                logger.debug(f"Train {key}: {value:.6f}")
        
        for key, value in val_metrics.items():
            if key != 'total_loss':
                logger.debug(f"Val {key}: {value:.6f}")
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {
                'epoch': epoch,
                'learning_rate': lr,
                'epoch_time': epoch_time
            }
            
            # Add training metrics
            for key, value in train_metrics.items():
                log_dict[f'train/{key}'] = value
            
            # Add validation metrics
            for key, value in val_metrics.items():
                log_dict[f'val/{key}'] = value
            
            wandb.log(log_dict)
    
    def log_batch(self, step: int, loss: float, lr: float):
        """Log batch-level metrics"""
        if step % 100 == 0:  # Log every 100 steps
            logger.debug(f"Step {step:5d} | Loss: {loss:.6f} | LR: {lr:.2e}")
            
            if self.use_wandb:
                wandb.log({
                    'step': step,
                    'batch_loss': loss,
                    'learning_rate': lr
                })
    
    def save_history(self, filepath: str):
        """Save training history to file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training history saved to {filepath}")


def compute_model_size(model: nn.Module) -> Dict:
    """Compute model size statistics"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    param_memory_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per float32
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_param_count,
        'frozen_parameters': param_count - trainable_param_count,
        'estimated_memory_mb': param_memory_mb
    }


class GradientClipping:
    """Gradient clipping utility with monitoring"""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
        self.grad_norms = []
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the gradient norm"""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm
        )
        
        self.grad_norms.append(grad_norm.item())
        return grad_norm.item()
    
    def get_recent_grad_norm_stats(self, window: int = 100) -> Dict:
        """Get statistics for recent gradient norms"""
        if not self.grad_norms:
            return {}
        
        recent_norms = self.grad_norms[-window:]
        return {
            'mean_grad_norm': np.mean(recent_norms),
            'max_grad_norm': np.max(recent_norms),
            'std_grad_norm': np.std(recent_norms)
        }


def count_parameters_by_component(model: nn.Module) -> Dict:
    """Count parameters for each component of the model"""
    component_counts = {}
    
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        component_counts[name] = {
            'total': param_count,
            'trainable': trainable_count,
            'frozen': param_count - trainable_count
        }
    
    return component_counts


def validate_model_outputs(predictions: Dict, batch_size: int, max_atoms: int) -> bool:
    """Validate model output shapes and values"""
    try:
        # Check required keys
        required_keys = ['nmr_shifts', 'positions', 'atom_types', 'smiles_positions']
        for key in required_keys:
            if key not in predictions:
                logger.error(f"Missing key in predictions: {key}")
                return False
        
        # Check shapes
        expected_shapes = {
            'nmr_shifts': (batch_size, max_atoms, 2),
            'positions': (batch_size, max_atoms, 3),
            'atom_types': (batch_size, max_atoms, -1),  # -1 means any size for last dim
            'smiles_positions': (batch_size, max_atoms, 1)
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = predictions[key].shape
            
            # Check first dimensions
            if len(actual_shape) != len(expected_shape):
                logger.error(f"Wrong number of dimensions for {key}: {actual_shape}")
                return False
            
            for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
                if expected != -1 and actual != expected:
                    logger.error(f"Wrong shape for {key} dim {i}: {actual} vs {expected}")
                    return False
        
        # Check for NaN or Inf values
        for key, tensor in predictions.items():
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any():
                    logger.error(f"NaN values found in {key}")
                    return False
                if torch.isinf(tensor).any():
                    logger.error(f"Inf values found in {key}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating model outputs: {e}")
        return False


def profile_training_step(model: nn.Module, batch: Dict, device: torch.device) -> Dict:
    """Profile a single training step for performance analysis"""
    import time
    
    profiling_results = {}
    
    # Move batch to device
    start_time = time.time()
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    batch[key][subkey] = subvalue.to(device, non_blocking=True)
    
    profiling_results['data_transfer_time'] = time.time() - start_time
    
    # Forward pass
    start_time = time.time()
    with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
        predictions = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            coords=batch['coords'],
            atom_types=batch['atom_types'],
            atom_mask=batch['atom_mask'],
            nmr_features=batch['nmr_features']
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    profiling_results['forward_time'] = time.time() - start_time
    profiling_results['batch_size'] = batch['input_ids'].shape[0]
    
    return profiling_results