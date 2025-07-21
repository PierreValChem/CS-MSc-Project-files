"""
Training script for implicit H restricted model with comprehensive rewards
Uses comprehensive reward/penalty system to ensure proper molecule generation with correct H/C counts
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

# Model-specific configuration with comprehensive reward system
MODEL_CONFIG = {
    'model_name': 'implicit_h_restricted_comprehensive',
    'use_explicit_h': False,
    'use_molecule_validation': True,
    'use_comprehensive_rewards': True,  # Enable comprehensive reward/penalty system
    'data_dir': 'C:\\Users\\pierr\\Desktop\\CS MSc Project files\\peaklist\\fakesmall',
    'rebuild_vocab': False,
    
    # COMPREHENSIVE REWARD SYSTEM WEIGHTS (All integrated during training & testing)
    'base_loss_weight': 0.2,           # 20% standard CrossEntropyLoss
    'reward_weight': 0.5,              # 50% molecule reward/penalty system  
    'token_accuracy_weight': 0.3,      # 30% token accuracy rewards with synergy bonuses
    
    # H/C VALIDATION PARAMETERS (focuses on NMR-visible atoms only)
    'hc_count_tolerance': 0.05,        # Very strict 5% tolerance for H/C counts
    
    # TRAINING PARAMETERS (optimized for comprehensive rewards)
    'learning_rate': 3e-6,             # Very low LR for stability with complex loss
    'batch_size': 8,                   # Small batch for detailed reward tracking
    'num_epochs': 200,                 # More epochs for reward system convergence
    'temperature': 0.5,                # Conservative generation for better chemistry
    'gradient_clip': 0.25,             # Strong clipping for complex gradients
    'dropout': 0.1,                    # Moderate dropout
    'warmup_ratio': 0.4,               # Long warmup for complex loss landscape
    
    # ENHANCED TRACKING (all reward systems monitored)
    'save_metrics_every': 5,           # Frequent saves to track reward evolution
    'calculate_roc_every': 10,
    'save_comprehensive_predictions': True,     # Save detailed predictions with all scores
    'log_detailed_rewards': True,              # Log all reward components
    'track_token_accuracy_synergy': True,      # Track synergy bonuses
    'track_reward_evolution': True,            # Track how rewards change over time
    'save_reward_breakdown': True,             # Save detailed reward analysis
    
    # GENERATION PARAMETERS
    'max_generation_attempts': 5,      # Fewer attempts (rely on reward system)
    'beam_size': 3,                    # Small beam size
    'use_reward_guided_generation': True,  # Use rewards to guide generation
}

if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE REWARD SYSTEM TRAINING")
    
    print("\nConfiguration Summary:")
    print(f"  Model: {MODEL_CONFIG['model_name']}")
    print(f"  Base Loss Weight: {MODEL_CONFIG['base_loss_weight']} (20%)")
    print(f"  Reward Weight: {MODEL_CONFIG['reward_weight']} (50%)")
    print(f"  Token Accuracy Weight: {MODEL_CONFIG['token_accuracy_weight']} (30%)")
    print(f"  H/C Tolerance: {MODEL_CONFIG['hc_count_tolerance']*100}% (very strict)")
    print(f"  Learning Rate: {MODEL_CONFIG['learning_rate']} (conservative)")
    print(f"  Epochs: {MODEL_CONFIG['num_epochs']} (extended for convergence)")
    print("="*80)
    
    main(model_config=MODEL_CONFIG)