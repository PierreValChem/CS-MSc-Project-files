"""
Check the configuration saved with the model
"""
import torch
from pathlib import Path

def check_model_config():
    """Check what configuration was used for training"""
    
    results_dir = Path('results_smiles_nmr')
    checkpoint_path = results_dir / 'best_model.pt'
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    print("Checkpoint contents:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Check if config was saved
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\nModel configuration:")
        print(f"  hidden_dim: {config.model.hidden_dim}")
        print(f"  nmr_hidden_dim: {config.model.nmr_hidden_dim}")
        print(f"  num_attention_heads: {config.model.num_attention_heads}")
        print(f"  dropout: {config.model.dropout}")
    
    print(f"\nTraining info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    if 'val_metrics' in checkpoint:
        print(f"\nValidation metrics at best epoch:")
        for k, v in checkpoint['val_metrics'].items():
            print(f"  {k}: {v:.4f}")
    
    # Check model state dict shapes
    print(f"\nSome model layer shapes:")
    state_dict = checkpoint['model_state_dict']
    for key in ['projection.0.weight', 'h_nmr_predictor.0.weight', 'c_nmr_predictor.0.weight']:
        if key in state_dict:
            print(f"  {key}: {state_dict[key].shape}")

if __name__ == "__main__":
    check_model_config()