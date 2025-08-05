# Updated train_implicit_h_restricted.py for ChemBERTa integration

"""
Training script for implicit H restricted model with ChemBERTa and validity-first rewards
Optimized for full dataset (35k compounds) with pretrained ChemBERTa
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

# Model-specific configuration for ChemBERTa + Full Dataset
MODEL_CONFIG = {
    'model_name': 'chemberta_validity_first_35k',
    'use_explicit_h': False,
    'use_molecule_validation': True,
    'use_comprehensive_rewards': True,
    'data_dir': 'C:\\Users\\pierr\\Desktop\\CS MSc Project files\\peaklist\\complete_compounds_only',
    'rebuild_vocab': True,
    
    # CHEMBERTA CONFIGURATION
    'use_pretrained_chemberta': True,  # Enable pretrained ChemBERTa
    'chemberta_model_name': 'DeepChem/ChemBERTa-77M-MLM',  # 77M parameter model
    'freeze_chemberta_layers': 6,      # Freeze first 6 layers of ChemBERTa
    'use_pretrained_decoder': False,   # Use standard decoder (can set True for full ChemBERTa)
    
    # VALIDITY-FIRST REWARD SYSTEM (same as before)
    'hc_count_tolerance': 0.08,
    
    # TRAINING PARAMETERS (adjusted for pretrained model)
    'learning_rate': 2e-5,             # Lower LR for pretrained model
    'batch_size': 24,                  # Slightly smaller due to larger model
    'num_epochs': 80,                  # Fewer epochs with pretrained
    'temperature': 0.7,
    'gradient_clip': 1.0,
    'dropout': 0.1,
    'warmup_ratio': 0.1,               # More warmup for pretrained
    
    # OPTIMIZER SETTINGS
    'weight_decay': 1e-4,              # More regularization for pretrained
    'adam_epsilon': 1e-8,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    
    # LEARNING RATE SCHEDULE
    'scheduler_type': 'cosine_with_restarts',  # Better for fine-tuning
    'min_lr': 1e-6,
    'T_0': 10,                         # Restart every 10 epochs
    
    # MODEL ARCHITECTURE (ChemBERTa-compatible)
    'hidden_dim': 384,                 # ChemBERTa hidden dimension (FIXED)
    'num_encoder_layers': 3,           # Additional processing layers
    'num_decoder_layers': 6,           # Decoder layers
    'attention_heads': 12,             # ChemBERTa-compatible heads
    
    # GENERATION PARAMETERS
    'generation_temperature': 0.8,
    'generation_top_k': 40,
    'generation_top_p': 0.9,
    'max_generation_attempts': 5,
    'beam_size': 5,
    
    # VALIDATION & CHECKPOINTING
    'validate_every': 1,
    'save_metrics_every': 5,
    'calculate_roc_every': 10,
    'save_best_only': True,
    'save_comprehensive_predictions': True,
    'log_detailed_rewards': True,
    
    # EARLY STOPPING
    'early_stopping_patience': 15,
    'early_stopping_metric': 'validity_rate',
    'early_stopping_mode': 'max',
    'early_stopping_min_delta': 0.001,
    
    # DATA HANDLING
    'use_data_augmentation': False,
    'num_workers': 4,
    'pin_memory': True,
    
    # LOGGING
    'log_every_n_steps': 100,
    'track_gradients': True,           # Monitor pretrained model gradients
    'track_reward_components': True,
    'log_reward_distribution': True,
    
    # PERFORMANCE OPTIMIZATIONS
    'gradient_accumulation_steps': 2,  # Accumulate for effective batch size 48
    'mixed_precision': True,           # Important for large models
    'find_unused_parameters': False,
    
    # CHECKPOINT MANAGEMENT
    'save_total_limit': 5,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'validity_rate',
}

if __name__ == "__main__":
    print("="*80)
    print("CHEMBERTA + VALIDITY-FIRST TRAINING - FULL DATASET (35K)")
    print("="*80)
    print("\n🧪 ChemBERTa Configuration:")
    print(f"  Model: {MODEL_CONFIG['chemberta_model_name']}")
    print(f"  Hidden Dimension: 768 (fixed for ChemBERTa)")
    print(f"  Frozen Layers: First {MODEL_CONFIG['freeze_chemberta_layers']} layers")
    print(f"  Fine-tuning: Upper layers only")
    print("\n📊 Dataset:")
    print(f"  Path: complete_compounds_only (~35,000 molecules)")
    print(f"  Rebuild Vocab: {MODEL_CONFIG['rebuild_vocab']}")
    print("\n🎯 Reward Priority: VALIDITY > Token Accuracy > Structure > H/C matching")
    print("\n⚙️ Training Configuration:")
    print(f"  Learning Rate: {MODEL_CONFIG['learning_rate']} (lower for pretrained)")
    print(f"  Batch Size: {MODEL_CONFIG['batch_size']} × {MODEL_CONFIG['gradient_accumulation_steps']} = {MODEL_CONFIG['batch_size'] * MODEL_CONFIG['gradient_accumulation_steps']} effective")
    print(f"  Epochs: {MODEL_CONFIG['num_epochs']}")
    print(f"  Warmup: {MODEL_CONFIG['warmup_ratio']*100}% of steps")
    print(f"  LR Schedule: {MODEL_CONFIG['scheduler_type']}")
    print(f"  Mixed Precision: {MODEL_CONFIG['mixed_precision']}")
    print("="*80)
    print("\n🚀 Expected Benefits of ChemBERTa:")
    print("  ✓ Faster convergence (pretrained on 77M molecules)")
    print("  ✓ Better understanding of chemical structures")
    print("  ✓ Improved validity rates")
    print("  ✓ Better generalization")
    print("  ✓ More chemically meaningful representations")
    print("="*80)
    print("\n⏱️ Estimated Training Time:")
    print("  • ~3-4 minutes per epoch on GPU")
    print("  • ~4-5 hours total training time")
    print("  • Faster convergence than training from scratch")
    print("="*80)
    
    # Check dependencies
    try:
        from transformers import RobertaModel
        model_test = RobertaModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
        print("\n✅ ChemBERTa model successfully loaded for testing!")
        del model_test  # Free memory
    except Exception as e:
        print(f"\n❌ Error loading ChemBERTa: {e}")
        print("Please ensure transformers library is installed: pip install transformers")
    
    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        print(f"\n✓ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠ Warning: CUDA not available, training will be very slow on CPU")
    
    print("="*80)
    
    main(model_config=MODEL_CONFIG)