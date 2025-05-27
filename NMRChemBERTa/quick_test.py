"""
Quick test script to verify the model can run without full training
"""

import torch
import logging
from config import get_default_config
from nmr_chemberta_model import NMRChemBERTa
from nmr_dataset import NMRDataset
from training_utils import MultiTaskLoss, prepare_batch_targets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_forward_pass():
    """Test that the model can do a forward pass"""
    logger.info("Testing model forward pass...")
    
    # Load config
    config = get_default_config()
    config.data.max_files_limit = 10  # Just load a few files for testing
    
    # Create model
    model = NMRChemBERTa(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Model created on {device}")
    logger.info(f"Model parameters: {model.get_parameter_count()}")
    
    # Create dataset
    try:
        dataset = NMRDataset(config, split='train')
        logger.info(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Make sure your .nmredata files are in the CSV_to_NMRe_output_v3/ directory")
        return False
    
    # Get a single sample
    sample = dataset[0]
    logger.info(f"Sample SMILES: {sample['smiles']}")
    
    # Create a batch of size 1
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, dict):
            batch[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    batch[key][subkey] = subvalue.unsqueeze(0).to(device)
        else:
            batch[key] = value
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            predictions = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                coords=batch['coords'],
                atom_types=batch['atom_types'],
                atom_mask=batch['atom_mask'],
                nmr_features=batch['nmr_features']
            )
        
        logger.info("✓ Forward pass successful!")
        
        # Check predictions
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape {value.shape}")
        
        # Test loss computation
        loss_fn = MultiTaskLoss(config)
        targets, masks = prepare_batch_targets(batch, device)
        losses = loss_fn(predictions, targets, masks)
        
        logger.info(f"✓ Loss computation successful!")
        logger.info(f"  Total loss: {losses['total_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test that data loader works properly"""
    logger.info("\nTesting data loader...")
    
    from nmr_dataset import create_data_loaders
    
    config = get_default_config()
    config.data.max_files_limit = 20  # Limit for testing
    config.data.batch_size = 2
    
    try:
        train_loader, val_loader, test_loader, dataset = create_data_loaders(config)
        
        logger.info(f"✓ Data loaders created successfully!")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        # Test loading a batch
        batch = next(iter(train_loader))
        logger.info(f"✓ Successfully loaded a batch!")
        logger.info(f"  Batch size: {batch['input_ids'].shape[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data loader test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("Running NMR-ChemBERTa tests...\n")
    
    # Test 1: Model forward pass
    if test_model_forward_pass():
        logger.info("✓ Model test passed!\n")
    else:
        logger.error("✗ Model test failed!\n")
        return
    
    # Test 2: Data loader
    if test_data_loader():
        logger.info("✓ Data loader test passed!\n")
    else:
        logger.error("✗ Data loader test failed!\n")
        return
    
    logger.info("All tests passed! You can now run the full training with: python train.py")


if __name__ == "__main__":
    main()