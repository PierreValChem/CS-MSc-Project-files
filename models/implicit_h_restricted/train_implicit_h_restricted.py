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

# Model-specific configuration
MODEL_CONFIG = {
    'model_name': 'implicit_h_restricted',
    'use_explicit_h': False,
    'use_molecule_validation': True,  # With validation
    'data_dir': 'C:\\Users\\pierr\\Desktop\\CS MSc Project files\\peaklist\\fakesmall',
    'rebuild_vocab': True,

    'validity_weight': 0.5,  # Increased from 0.2
    'learning_rate': 3e-5,   # Slightly lower for stability
    'batch_size': 16,        # Smaller batch for better validity learning
    'num_epochs': 100,       # More epochs to learn validity
    'temperature': 0.8
}

if __name__ == "__main__":
    print("="*60)
    print("Training Implicit H Restricted Model with Validity Constraints")
    print("="*60)
    main(model_config=MODEL_CONFIG)