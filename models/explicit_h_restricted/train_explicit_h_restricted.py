"""
Training script for implicit H restricted model
Uses implicit hydrogen SMILES WITH molecule validation
"""

import sys
import os

# Add parent directories to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Change working directory to this model's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import the main training script
from nmr_to_smiles_chemberta import main

# Model-specific configuration with improved validity parameters
MODEL_CONFIG = {
    'model_name': 'implicit_h_restricted',
    'use_explicit_h': True,
    'use_molecule_validation': True,  # With validation
    'data_dir': 'C:\\Users\\pierr\\Desktop\\CS MSc Project files\\peaklist\\fakesmall_expH',
    'rebuild_vocab': False,  # Set to True if you want to rebuild
    
    # Validity enforcement parameters
    'validity_weight': 1.0,  # Increased from 0.5 for stronger enforcement
    'learning_rate': 1e-5,   # Lower learning rate for stability
    'batch_size': 8,         # Smaller batch size for better learning
    'num_epochs': 100,       # More epochs
    'temperature': 0.7,      # Lower temperature for more conservative generation
    
    # Additional parameters for better training
    'gradient_clip': 0.5,    # Stronger gradient clipping
    'dropout': 0.2,          # More dropout for regularization
    'warmup_ratio': 0.2,     # Longer warmup
    'save_metrics_every': 10,
    'calculate_roc_every': 20,
    
    # Generation parameters
    'max_generation_attempts': 10,  # Fewer attempts to speed up
    'beam_size': 3,                 # Smaller beam size
}

if __name__ == "__main__":
    print("="*60)
    print("Training Implicit H Restricted Model with Validity Constraints")
    print("="*60)
    print("Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60)
    main(model_config=MODEL_CONFIG)