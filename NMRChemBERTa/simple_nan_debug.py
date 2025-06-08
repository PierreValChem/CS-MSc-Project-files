"""
Simple script to debug NaN issues in the model
"""
import torch
import numpy as np
from config import get_default_config
from nmr_dataset import NMRDataset
from nmr_chemberta_model import NMRChemBERTa
from training_utils import MultiTaskLoss, prepare_batch_targets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dataset_samples():
    """Check a few dataset samples for issues"""
    config = get_default_config()
    config.data.max_files_limit = 10
    
    dataset = NMRDataset(config, split='train')
    
    logger.info(f"\nChecking {min(5, len(dataset))} samples...")
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        logger.info(f"\n--- Sample {i} ---")
        
        # Check NMR features
        h_shifts = sample['nmr_features']['h_shifts']
        c_shifts = sample['nmr_features']['c_shifts']
        
        # Get non-zero values
        h_nonzero = h_shifts[h_shifts != 0]
        c_nonzero = c_shifts[c_shifts != 0]
        
        logger.info(f"H shifts - count: {len(h_nonzero)}, range: [{h_nonzero.min():.2f}, {h_nonzero.max():.2f}]" if len(h_nonzero) > 0 else "H shifts - no data")
        logger.info(f"C shifts - count: {len(c_nonzero)}, range: [{c_nonzero.min():.2f}, {c_nonzero.max():.2f}]" if len(c_nonzero) > 0 else "C shifts - no data")
        
        # Check for NaN
        if torch.isnan(h_shifts).any():
            logger.error(f"Sample {i}: NaN in H shifts!")
        if torch.isnan(c_shifts).any():
            logger.error(f"Sample {i}: NaN in C shifts!")
        
        # Check coordinates
        coords = sample['coords']
        logger.info(f"Coords range: [{coords.min():.2f}, {coords.max():.2f}]")
        
        if torch.isnan(coords).any():
            logger.error(f"Sample {i}: NaN in coordinates!")

def test_model_initialization():
    """Test model initialization values"""
    config = get_default_config()
    config.model.chemberta_name = 'DeepChem/ChemBERTa-77M-MLM'  # Fix the name
    
    model = NMRChemBERTa(config)
    
    logger.info("\nChecking model parameters...")
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only check trainable parameters
            param_data = param.data
            has_nan = torch.isnan(param_data).any()
            has_inf = torch.isinf(param_data).any()
            
            if has_nan or has_inf:
                logger.error(f"{name}: NaN={has_nan}, Inf={has_inf}")
            
            # Check for very large values
            max_val = param_data.abs().max().item()
            if max_val > 10:
                logger.warning(f"{name}: Large values detected (max={max_val:.2f})")

def test_single_forward():
    """Test a single forward pass with minimal data"""
    config = get_default_config()
    config.model.chemberta_name = 'DeepChem/ChemBERTa-77M-MLM'  # Fix the name
    config.data.max_files_limit = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NMRChemBERTa(config).to(device)
    model.eval()
    
    # Create minimal dummy input
    batch_size = 1
    seq_len = 10
    max_atoms = 10
    
    dummy_batch = {
        'input_ids': torch.randint(0, 100, (batch_size, seq_len)).to(device),
        'attention_mask': torch.ones(batch_size, seq_len).to(device),
        'coords': torch.randn(batch_size, max_atoms, 3).to(device) * 5,  # Scale down
        'atom_types': torch.randint(0, 10, (batch_size, max_atoms)).to(device),
        'atom_mask': torch.ones(batch_size, max_atoms).to(device),
        'nmr_features': {
            'h_shifts': torch.randn(batch_size, max_atoms).to(device) * 0.1,  # Small values
            'c_shifts': torch.randn(batch_size, max_atoms).to(device) * 0.1,  # Small values
            'h_mask': torch.ones(batch_size, max_atoms).to(device),
            'c_mask': torch.ones(batch_size, max_atoms).to(device)
        }
    }
    
    logger.info("\nTesting forward pass with dummy data...")
    
    with torch.no_grad():
        predictions = model(
            input_ids=dummy_batch['input_ids'],
            attention_mask=dummy_batch['attention_mask'],
            coords=dummy_batch['coords'],
            atom_types=dummy_batch['atom_types'],
            atom_mask=dummy_batch['atom_mask'],
            nmr_features=dummy_batch['nmr_features']
        )
    
    # Check predictions
    for key, pred in predictions.items():
        if isinstance(pred, torch.Tensor):
            has_nan = torch.isnan(pred).any()
            has_inf = torch.isinf(pred).any()
            
            logger.info(f"{key}: shape={pred.shape}, NaN={has_nan}, Inf={has_inf}")
            if not has_nan and not has_inf:
                logger.info(f"  Range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

def test_loss_computation():
    """Test loss computation with controlled inputs"""
    config = get_default_config()
    config.model.chemberta_name = 'DeepChem/ChemBERTa-77M-MLM'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = MultiTaskLoss(config)
    
    batch_size = 2
    max_atoms = 10
    
    # Create predictions with known good values
    predictions = {
        'nmr_shifts': torch.randn(batch_size, max_atoms, 2).to(device) * 0.1,
        'positions': torch.randn(batch_size, max_atoms, 3).to(device),
        'atom_types': torch.randn(batch_size, max_atoms, 10).to(device),
        'smiles_positions': torch.randn(batch_size, max_atoms, 1).to(device)
    }
    
    # Create targets
    targets = {
        'nmr_shifts': torch.randn(batch_size, max_atoms, 2).to(device) * 0.1,
        'positions': torch.randn(batch_size, max_atoms, 3).to(device),
        'atom_types': torch.randint(0, 10, (batch_size, max_atoms)).to(device),
        'smiles_positions': torch.randn(batch_size, max_atoms, 1).to(device)
    }
    
    # Create masks
    masks = {
        'nmr_mask': torch.ones(batch_size, max_atoms, 2).to(device),
        'atom_mask': torch.ones(batch_size, max_atoms).to(device)
    }
    
    logger.info("\nTesting loss computation...")
    
    losses = loss_fn(predictions, targets, masks)
    
    for key, loss in losses.items():
        if isinstance(loss, torch.Tensor):
            logger.info(f"{key}: {loss.item():.6f}")
            if torch.isnan(loss):
                logger.error(f"{key} is NaN!")

if __name__ == "__main__":
    logger.info("=== Simple NaN Debugging ===\n")
    
    try:
        logger.info("1. Checking dataset samples...")
        check_dataset_samples()
    except Exception as e:
        logger.error(f"Dataset check failed: {e}")
    
    try:
        logger.info("\n2. Testing model initialization...")
        test_model_initialization()
    except Exception as e:
        logger.error(f"Model init failed: {e}")
    
    try:
        logger.info("\n3. Testing forward pass...")
        test_single_forward()
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
    
    try:
        logger.info("\n4. Testing loss computation...")
        test_loss_computation()
    except Exception as e:
        logger.error(f"Loss computation failed: {e}")
    
    logger.info("\n=== Debugging complete ===")