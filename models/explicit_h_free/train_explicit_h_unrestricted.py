"""
Training script for explicit H unrestricted model
Uses explicit hydrogen SMILES without molecule validation
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
    'model_name': 'explicit_h_unrestricted',
    'use_explicit_h': True,
    'use_molecule_validation': False,  # No validation
    'data_dir': 'C:\\Users\\pierr\\Desktop\\CS MSc Project files\\peaklist\\complete_compounds_only_explicit_h',
    'rebuild_vocab': True,
}

if __name__ == "__main__":
    print("="*60)
    print("Training Explicit H Unrestricted Model")
    print("="*60)
    main(model_config=MODEL_CONFIG)