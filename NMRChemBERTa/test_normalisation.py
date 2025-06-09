"""
Quick test to verify normalization fix
"""
import torch
import numpy as np
from config import Config
from nmr_dataset import create_data_loaders
from enhanced_smiles_nmr_model import EnhancedSMILEStoNMRModel

def test_normalization():
    """Test if normalization is working correctly"""
    
    config = Config.from_yaml('config_smiles_nmr.yaml')
    config.data.max_files_limit = 100  # Quick test
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    train_loader, _, _, _ = create_data_loaders(config)
    
    # Get one batch
    batch = next(iter(train_loader))
    
    # Check normalized values
    h_shifts = batch['nmr_features']['h_shifts']
    c_shifts = batch['nmr_features']['c_shifts']
    h_mask = batch['nmr_features']['h_mask']
    c_mask = batch['nmr_features']['c_mask']
    
    # Get non-zero values
    h_values = h_shifts[h_mask.bool()].numpy()
    c_values = c_shifts[c_mask.bool()].numpy()
    
    print(f"\nNormalized H NMR stats:")
    print(f"  Mean: {np.mean(h_values):.3f} (should be ~0)")
    print(f"  Std: {np.std(h_values):.3f} (should be ~1)")
    print(f"  Range: [{np.min(h_values):.3f}, {np.max(h_values):.3f}]")
    
    print(f"\nNormalized C NMR stats:")
    print(f"  Mean: {np.mean(c_values):.3f} (should be ~0)")
    print(f"  Std: {np.std(c_values):.3f} (should be ~1)")
    print(f"  Range: [{np.min(c_values):.3f}, {np.max(c_values):.3f}]")
    
    # Test model forward pass
    print("\nTesting model...")
    model = EnhancedSMILEStoNMRModel(config).to(device)
    
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Check model is on GPU
    print(f"Model on GPU: {next(model.parameters()).is_cuda}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Output shape: {outputs['nmr_shifts'].shape}")
    print(f"Output device: {outputs['nmr_shifts'].device}")
    
    # Denormalize predictions for one sample
    pred_h = outputs['nmr_shifts'][0, :, 0].cpu().numpy()
    pred_c = outputs['nmr_shifts'][0, :, 1].cpu().numpy()
    
    # Denormalize
    pred_h_ppm = pred_h * 2.1 + 3.5
    pred_c_ppm = pred_c * 50.3 + 79.8
    
    print(f"\nSample predictions (denormalized):")
    print(f"  H NMR range: [{np.min(pred_h_ppm):.1f}, {np.max(pred_h_ppm):.1f}] ppm")
    print(f"  C NMR range: [{np.min(pred_c_ppm):.1f}, {np.max(pred_c_ppm):.1f}] ppm")

if __name__ == "__main__":
    test_normalization()