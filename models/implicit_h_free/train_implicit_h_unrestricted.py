"""
Training script for implicit H unrestricted model
Uses implicit hydrogen SMILES without molecule validation
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
    'model_name': 'implicit_h_unrestricted',
    'use_explicit_h': False,
    'use_molecule_validation': False,  # No validation
    'data_dir': 'C:\\Users\\pierr\\Desktop\\CS MSc Project files\\peaklist\\complete_compounds_only',
    'rebuild_vocab': False,
}

if __name__ == "__main__":
    print("="*60)
    print("Training Implicit H Unrestricted Model")
    print("="*60)
    main(model_config=MODEL_CONFIG)