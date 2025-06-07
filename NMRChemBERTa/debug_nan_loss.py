"""
Debug script to identify why we're getting NaN losses
"""
import torch
import numpy as np
from config import Config
from nmr_dataset import create_data_loaders
from nmr_chemberta_model import NMRChemBERTa
from training_utils import MultiTaskLoss, prepare_batch_targets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_batch_data(batch, device):
    """Check a batch for NaN or extreme values"""
    issues = []
    
    # Check each tensor in the batch
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value.to(device)
            if torch.isnan(tensor).any():
                issues.append(f"{key} contains NaN")
            if torch.isinf(tensor).any():
                issues.append(f"{key} contains Inf")
            
            # Check for extreme values
            if tensor.dtype in [torch.float32, torch.float64]:
                max_val = tensor.abs().max().item()
                if max_val > 1e6:
                    issues.append(f"{key} has extreme values (max abs: {max_val})")
                    
        elif isinstance(value, dict):
            # Check NMR features
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    tensor = sub_value.to(device)
                    if torch.isnan(tensor).any():
                        issues.append(f"{key}.{sub_key} contains NaN")
                    if torch.isinf(tensor).any():
                        issues.append(f"{key}.{sub_key} contains Inf")
                    
                    if tensor.dtype in [torch.float32, torch.float64]:
                        # Get non-zero values for NMR shifts
                        non_zero = tensor[tensor != 0]
                        if len(non_zero) > 0:
                            max_val = non_zero.abs().max().item()
                            mean_val = non_zero.mean().item()
                            logger.info(f"{key}.{sub_key}: max={max_val:.2f}, mean={mean_val:.2f}")
    
    return issues

def test_model_initialization():
    """Test if model weights are properly initialized"""
    config = Config.from_yaml('config.yaml')
    model = NMRChemBERTa(config)
    
    # Check for NaN in model parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"NaN found in parameter: {name}")
            return False
        
        # Check for extreme values
        if param.abs().max() > 100:
            logger.warning(f"Large values in {name}: max={param.abs().max().item()}")
    
    return True

def test_forward_pass():
    """Test a single forward pass"""
    config = Config.from_yaml('config.yaml')
    config.data.max_files_limit = 10
    
    # Create model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NMRChemBERTa(config).to(device)
    train_loader, _, _, _ = create_data_loaders(config)
    
    # Get one batch
    batch = next(iter(train_loader))
    
    # Check batch data
    issues = check_batch_data(batch, device)
    if issues:
        logger.error(f"Batch data issues: {issues}")
        return
    
    # Move batch to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in value.items()}
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            predictions = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                coords=batch['coords'],
                atom_types=batch['atom_types'],
                atom_mask=batch['atom_mask'],
                nmr_features=batch['nmr_features']
            )
            
            # Check predictions
            for key, pred in predictions.items():
                if isinstance(pred, torch.Tensor):
                    if torch.isnan(pred).any():
                        logger.error(f"NaN in prediction: {key}")
                    else:
                        logger.info(f"{key}: shape={pred.shape}, "
                                   f"mean={pred.mean().item():.4f}, "
                                   f"std={pred.std().item():.4f}")
                        
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

def test_loss_computation():
    """Test loss computation"""
    config = Config.from_yaml('config.yaml')
    config.data.max_files_limit = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NMRChemBERTa(config).to(device)
    loss_fn = MultiTaskLoss(config)
    train_loader, _, _, _ = create_data_loaders(config)
    
    batch = next(iter(train_loader))
    
    # Move to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in value.items()}
    
    # Forward pass
    predictions = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        coords=batch['coords'],
        atom_types=batch['atom_types'],
        atom_mask=batch['atom_mask'],
        nmr_features=batch['nmr_features']
    )
    
    # Prepare targets
    targets, masks = prepare_batch_targets(batch, device)
    
    # Check targets for NaN
    for key, target in targets.items():
        if torch.isnan(target).any():
            logger.error(f"NaN in target: {key}")
            # Print some statistics
            logger.info(f"{key} shape: {target.shape}")
            logger.info(f"{key} non-nan values: {(~torch.isnan(target)).sum()}")
    
    # Compute loss
    try:
        loss_dict = loss_fn(predictions, targets, masks)
        logger.info(f"Loss components: {loss_dict}")
        
        if torch.isnan(loss_dict['total_loss']):
            logger.error("Total loss is NaN!")
            # Check individual losses
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value):
                    logger.error(f"{key} is NaN")
                    
    except Exception as e:
        logger.error(f"Loss computation failed: {e}")
        import traceback
        traceback.print_exc()

def check_nmr_normalization():
    """Check if NMR values are properly normalized"""
    config = Config.from_yaml('config.yaml')
    config.data.max_files_limit = 100
    
    from nmr_dataset import NMRDataset
    dataset = NMRDataset(config, split='train')
    
    all_h_shifts = []
    all_c_shifts = []
    
    for i in range(min(50, len(dataset))):
        sample = dataset[i]
        h_shifts = sample['nmr_features']['h_shifts']
        c_shifts = sample['nmr_features']['c_shifts']
        
        # Get non-zero values
        h_nonzero = h_shifts[h_shifts != 0]
        c_nonzero = c_shifts[c_shifts != 0]
        
        if len(h_nonzero) > 0:
            all_h_shifts.extend(h_nonzero.numpy())
        if len(c_nonzero) > 0:
            all_c_shifts.extend(c_nonzero.numpy())
    
    if all_h_shifts:
        logger.info(f"H NMR shifts - mean: {np.mean(all_h_shifts):.3f}, "
                   f"std: {np.std(all_h_shifts):.3f}, "
                   f"range: [{np.min(all_h_shifts):.3f}, {np.max(all_h_shifts):.3f}]")
    
    if all_c_shifts:
        logger.info(f"C NMR shifts - mean: {np.mean(all_c_shifts):.3f}, "
                   f"std: {np.std(all_c_shifts):.3f}, "
                   f"range: [{np.min(all_c_shifts):.3f}, {np.max(all_c_shifts):.3f}]")
    
    # Check if normalization is working
    if all_h_shifts and (np.mean(all_h_shifts) > 5 or np.std(all_h_shifts) > 10):
        logger.warning("H NMR shifts don't appear to be normalized!")
    if all_c_shifts and (np.mean(all_c_shifts) > 20 or np.std(all_c_shifts) > 50):
        logger.warning("C NMR shifts don't appear to be normalized!")

if __name__ == "__main__":
    logger.info("=== Testing Model Initialization ===")
    test_model_initialization()
    
    logger.info("\n=== Checking NMR Normalization ===")
    check_nmr_normalization()
    
    logger.info("\n=== Testing Forward Pass ===")
    test_forward_pass()
    
    logger.info("\n=== Testing Loss Computation ===")
    test_loss_computation()