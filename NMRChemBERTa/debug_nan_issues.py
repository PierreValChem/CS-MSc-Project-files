"""
Debug script to identify sources of NaN values in NMR-ChemBERTa
"""

import torch
import numpy as np
import logging
from config import get_default_config
from nmr_chemberta_model import NMRChemBERTa
from nmr_dataset import NMRDataset
from training_utils import MultiTaskLoss, prepare_batch_targets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dataset_statistics(dataset):
    """Check dataset for potential issues"""
    logger.info("\n=== Dataset Statistics ===")
    
    # Check for NaN in coordinates
    coords_with_nan = 0
    coords_range = {'min': float('inf'), 'max': float('-inf')}
    
    # Check for NaN in NMR shifts
    h_shifts_range = {'min': float('inf'), 'max': float('-inf')}
    c_shifts_range = {'min': float('inf'), 'max': float('-inf')}
    
    for i in range(min(100, len(dataset))):  # Check first 100 samples
        sample = dataset[i]
        
        # Check coordinates
        coords = sample['coords'].numpy()
        if np.isnan(coords).any():
            coords_with_nan += 1
        else:
            coords_range['min'] = min(coords_range['min'], coords.min())
            coords_range['max'] = max(coords_range['max'], coords.max())
        
        # Check NMR shifts
        h_shifts = sample['nmr_features']['h_shifts'].numpy()
        c_shifts = sample['nmr_features']['c_shifts'].numpy()
        
        # Only check non-zero values (zeros are padding)
        h_nonzero = h_shifts[h_shifts != 0]
        c_nonzero = c_shifts[c_shifts != 0]
        
        if len(h_nonzero) > 0:
            h_shifts_range['min'] = min(h_shifts_range['min'], h_nonzero.min())
            h_shifts_range['max'] = max(h_shifts_range['max'], h_nonzero.max())
        
        if len(c_nonzero) > 0:
            c_shifts_range['min'] = min(c_shifts_range['min'], c_nonzero.min())
            c_shifts_range['max'] = max(c_shifts_range['max'], c_nonzero.max())
    
    logger.info(f"Samples with NaN coordinates: {coords_with_nan}")
    logger.info(f"Coordinate range: {coords_range}")
    logger.info(f"H NMR shift range: {h_shifts_range}")
    logger.info(f"C NMR shift range: {c_shifts_range}")
    
    return coords_with_nan == 0


def test_model_components():
    """Test individual model components"""
    logger.info("\n=== Testing Model Components ===")
    
    config = get_default_config()
    model = NMRChemBERTa(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    max_atoms = config.model.max_atoms
    seq_len = 64
    
    # Create inputs with known good values
    dummy_inputs = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'attention_mask': torch.ones(batch_size, seq_len).to(device),
        'coords': torch.randn(batch_size, max_atoms, 3).to(device) * 10,  # Scale coordinates
        'atom_types': torch.randint(0, 10, (batch_size, max_atoms)).to(device),
        'atom_mask': torch.ones(batch_size, max_atoms).to(device),
        'nmr_features': {
            'h_shifts': torch.rand(batch_size, max_atoms).to(device) * 10,  # Typical H NMR range
            'c_shifts': torch.rand(batch_size, max_atoms).to(device) * 200,  # Typical C NMR range
            'h_mask': torch.ones(batch_size, max_atoms).to(device),
            'c_mask': torch.ones(batch_size, max_atoms).to(device)
        }
    }
    
    # Test each component
    with torch.no_grad():
        # Test SMILES encoding
        logger.info("Testing SMILES encoding...")
        smiles_embeddings = model._encode_smiles(
            dummy_inputs['input_ids'], 
            dummy_inputs['attention_mask']
        )
        logger.info(f"SMILES embeddings: shape={smiles_embeddings.shape}, "
                   f"contains_nan={torch.isnan(smiles_embeddings).any()}")
        
        # Test atom encoding
        logger.info("Testing atom encoding...")
        atom_representations = model._encode_atoms(
            dummy_inputs['coords'],
            dummy_inputs['atom_types'],
            dummy_inputs['atom_mask'],
            dummy_inputs['nmr_features']
        )
        logger.info(f"Atom representations: shape={atom_representations.shape}, "
                   f"contains_nan={torch.isnan(atom_representations).any()}")
        
        # Test full forward pass
        logger.info("Testing full forward pass...")
        predictions = model(**dummy_inputs)
        
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"{key}: shape={value.shape}, "
                           f"contains_nan={torch.isnan(value).any()}, "
                           f"min={value.min().item():.4f}, "
                           f"max={value.max().item():.4f}")


def test_loss_computation():
    """Test loss computation with known inputs"""
    logger.info("\n=== Testing Loss Computation ===")
    
    config = get_default_config()
    loss_fn = MultiTaskLoss(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 2
    max_atoms = 10
    
    # Create predictions and targets with known values
    predictions = {
        'nmr_shifts': torch.randn(batch_size, max_atoms, 2).to(device),
        'positions': torch.randn(batch_size, max_atoms, 3).to(device),
        'atom_types': torch.randn(batch_size, max_atoms, 10).to(device),
        'smiles_positions': torch.randn(batch_size, max_atoms, 1).to(device)
    }
    
    targets = {
        'nmr_shifts': torch.randn(batch_size, max_atoms, 2).to(device),
        'positions': torch.randn(batch_size, max_atoms, 3).to(device),
        'atom_types': torch.randint(0, 10, (batch_size, max_atoms)).to(device),
        'smiles_positions': torch.randn(batch_size, max_atoms, 1).to(device)
    }
    
    masks = {
        'nmr_mask': torch.ones(batch_size, max_atoms, 2).to(device),
        'atom_mask': torch.ones(batch_size, max_atoms).to(device)
    }
    
    # Compute losses
    losses = loss_fn(predictions, targets, masks)
    
    logger.info("Loss components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.item():.6f}")


def check_data_pipeline():
    """Check the entire data pipeline"""
    logger.info("\n=== Checking Data Pipeline ===")
    
    config = get_default_config()
    config.data.max_files_limit = 10  # Just check a few files
    
    try:
        dataset = NMRDataset(config, split='train')
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Check a few samples
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            logger.info(f"\nSample {i}:")
            logger.info(f"  SMILES: {sample['smiles']}")
            
            # Check for extreme values
            coords = sample['coords']
            if torch.isnan(coords).any():
                logger.warning(f"  NaN found in coordinates!")
            else:
                logger.info(f"  Coords range: [{coords.min():.2f}, {coords.max():.2f}]")
            
            # Check NMR data
            h_shifts = sample['nmr_features']['h_shifts']
            c_shifts = sample['nmr_features']['c_shifts']
            
            h_nonzero = h_shifts[h_shifts != 0]
            c_nonzero = c_shifts[c_shifts != 0]
            
            if len(h_nonzero) > 0:
                logger.info(f"  H shifts range: [{h_nonzero.min():.2f}, {h_nonzero.max():.2f}]")
            if len(c_nonzero) > 0:
                logger.info(f"  C shifts range: [{c_nonzero.min():.2f}, {c_nonzero.max():.2f}]")
                
    except Exception as e:
        logger.error(f"Error in data pipeline: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all debugging tests"""
    logger.info("Running NMR-ChemBERTa debugging tests...")
    
    # Test 1: Check dataset
    logger.info("\n" + "="*50)
    config = get_default_config()
    config.data.max_files_limit = 100
    
    try:
        dataset = NMRDataset(config, split='train')
        check_dataset_statistics(dataset)
    except Exception as e:
        logger.error(f"Dataset check failed: {e}")
    
    # Test 2: Check model components
    logger.info("\n" + "="*50)
    test_model_components()
    
    # Test 3: Check loss computation
    logger.info("\n" + "="*50)
    test_loss_computation()
    
    # Test 4: Check data pipeline
    logger.info("\n" + "="*50)
    check_data_pipeline()
    
    logger.info("\n" + "="*50)
    logger.info("Debugging complete. Check the output for any issues.")


if __name__ == "__main__":
    main()