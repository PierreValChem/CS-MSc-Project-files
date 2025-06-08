"""
Debug script to test actual training step
"""
import torch
import numpy as np
from config import get_default_config
from nmr_dataset import create_data_loaders
from nmr_chemberta_model import NMRChemBERTa
from training_utils import MultiTaskLoss, prepare_batch_targets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_training_step():
    """Debug a single training step with real data"""
    config = get_default_config()
    config.model.chemberta_name = 'DeepChem/ChemBERTa-77M-MLM'  # Fix the name
    config.data.max_files_limit = 15000  # Use more files
    config.data.batch_size = 4  # Small batch for debugging
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    logger.info("Loading data...")
    train_loader, _, _, _ = create_data_loaders(config)
    
    # Create model and loss
    logger.info("Creating model...")
    model = NMRChemBERTa(config).to(device)
    loss_fn = MultiTaskLoss(config)
    
    # Get one batch
    batch = next(iter(train_loader))
    
    # Move batch to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in value.items()}
    
    logger.info(f"\nBatch info:")
    logger.info(f"  Batch size: {batch['input_ids'].shape[0]}")
    logger.info(f"  Sequence length: {batch['input_ids'].shape[1]}")
    logger.info(f"  Num atoms: {batch['coords'].shape[1]}")
    
    # Check input data
    logger.info("\nChecking input data...")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            has_nan = torch.isnan(value).any()
            has_inf = torch.isinf(value).any()
            logger.info(f"{key}: shape={value.shape}, NaN={has_nan}, Inf={has_inf}")
            if not has_nan and not has_inf and value.dtype in [torch.float32, torch.float64]:
                logger.info(f"  Range: [{value.min().item():.4f}, {value.max().item():.4f}]")
        elif key == 'nmr_features':
            for sub_key, sub_value in value.items():
                has_nan = torch.isnan(sub_value).any()
                has_inf = torch.isinf(sub_value).any()
                logger.info(f"{key}.{sub_key}: shape={sub_value.shape}, NaN={has_nan}, Inf={has_inf}")
                if not has_nan and not has_inf and sub_value.dtype in [torch.float32, torch.float64]:
                    # Get non-zero values for shifts
                    if 'shift' in sub_key:
                        non_zero = sub_value[sub_value != 0]
                        if len(non_zero) > 0:
                            logger.info(f"  Non-zero range: [{non_zero.min().item():.4f}, {non_zero.max().item():.4f}]")
                    else:
                        logger.info(f"  Range: [{sub_value.min().item():.4f}, {sub_value.max().item():.4f}]")
    
    # Forward pass
    logger.info("\nRunning forward pass...")
    model.train()
    predictions = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        coords=batch['coords'],
        atom_types=batch['atom_types'],
        atom_mask=batch['atom_mask'],
        nmr_features=batch['nmr_features']
    )
    
    # Check predictions
    logger.info("\nChecking predictions...")
    for key, pred in predictions.items():
        if isinstance(pred, torch.Tensor):
            has_nan = torch.isnan(pred).any()
            has_inf = torch.isinf(pred).any()
            logger.info(f"{key}: shape={pred.shape}, NaN={has_nan}, Inf={has_inf}")
            if not has_nan and not has_inf:
                logger.info(f"  Range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    
    # Prepare targets
    logger.info("\nPreparing targets...")
    targets, masks = prepare_batch_targets(batch, device)
    
    # Check targets
    logger.info("\nChecking targets...")
    for key, target in targets.items():
        if isinstance(target, torch.Tensor):
            has_nan = torch.isnan(target).any()
            has_inf = torch.isinf(target).any()
            logger.info(f"{key}: shape={target.shape}, NaN={has_nan}, Inf={has_inf}")
            if not has_nan and not has_inf and target.dtype in [torch.float32, torch.float64]:
                logger.info(f"  Range: [{target.min().item():.4f}, {target.max().item():.4f}]")
    
    # Check masks
    logger.info("\nChecking masks...")
    for key, mask in masks.items():
        if isinstance(mask, torch.Tensor):
            logger.info(f"{key}: shape={mask.shape}, sum={mask.sum().item()}")
    
    # Compute loss
    logger.info("\nComputing loss...")
    loss_dict = loss_fn(predictions, targets, masks)
    
    logger.info("\nLoss components:")
    for key, loss in loss_dict.items():
        if isinstance(loss, torch.Tensor):
            logger.info(f"  {key}: {loss.item():.6f}")
            if torch.isnan(loss):
                logger.error(f"  {key} is NaN!")
    
    # Backward pass
    logger.info("\nRunning backward pass...")
    loss = loss_dict['total_loss']
    loss.backward()
    
    # Check gradients
    logger.info("\nChecking gradients...")
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            has_nan = torch.isnan(grad).any()
            has_inf = torch.isinf(grad).any()
            if has_nan or has_inf:
                logger.error(f"{name}: grad NaN={has_nan}, Inf={has_inf}")
            else:
                grad_norm = grad.norm().item()
                if grad_norm > 100:
                    logger.warning(f"{name}: large gradient norm={grad_norm:.2f}")
                grad_stats[name] = grad_norm
    
    # Show top gradients
    if grad_stats:
        sorted_grads = sorted(grad_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("\nTop 5 gradients:")
        for name, norm in sorted_grads:
            logger.info(f"  {name}: {norm:.4f}")

if __name__ == "__main__":
    try:
        debug_training_step()
    except Exception as e:
        logger.error(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()