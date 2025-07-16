"""
Diagnose why the NMR model has high MAE
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config
from nmr_dataset import create_data_loaders
from enhanced_smiles_nmr_model import EnhancedSMILEStoNMRModel

def diagnose_model_performance():
    """Analyze model predictions to understand high MAE"""
    
    # Load config and model
    config = Config.from_yaml('config_smiles_nmr_improved.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best model
    results_dir = Path('results_smiles_nmr')
    checkpoint = torch.load(results_dir / 'best_model.pt', weights_only=False)
    
    model = EnhancedSMILEStoNMRModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    # Load data
    _, val_loader, _, _ = create_data_loaders(config)
    
    # Analyze predictions
    all_h_pred = []
    all_c_pred = []
    all_h_true = []
    all_c_true = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10:  # Analyze first 10 batches
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_shifts = outputs['nmr_shifts'].cpu().numpy()
            
            # Get true values
            h_true = batch['nmr_features']['h_shifts'].numpy()
            c_true = batch['nmr_features']['c_shifts'].numpy()
            h_mask = batch['nmr_features']['h_mask'].numpy()
            c_mask = batch['nmr_features']['c_mask'].numpy()
            
            # Collect valid predictions
            for b in range(pred_shifts.shape[0]):
                h_valid = h_mask[b].astype(bool)
                c_valid = c_mask[b].astype(bool)
                
                all_h_pred.extend(pred_shifts[b, h_valid, 0])
                all_c_pred.extend(pred_shifts[b, c_valid, 1])
                all_h_true.extend(h_true[b, h_valid])
                all_c_true.extend(c_true[b, c_valid])
    
    # Convert to arrays
    all_h_pred = np.array(all_h_pred)
    all_c_pred = np.array(all_c_pred)
    all_h_true = np.array(all_h_true)
    all_c_true = np.array(all_c_true)
    
    # Analyze predictions
    print("\n" + "="*50)
    print("NORMALIZED SPACE ANALYSIS")
    print("="*50)
    
    print(f"\nH NMR (normalized):")
    print(f"  True range: [{all_h_true.min():.3f}, {all_h_true.max():.3f}]")
    print(f"  Pred range: [{all_h_pred.min():.3f}, {all_h_pred.max():.3f}]")
    print(f"  True mean: {all_h_true.mean():.3f}, std: {all_h_true.std():.3f}")
    print(f"  Pred mean: {all_h_pred.mean():.3f}, std: {all_h_pred.std():.3f}")
    
    print(f"\nC NMR (normalized):")
    print(f"  True range: [{all_c_true.min():.3f}, {all_c_true.max():.3f}]")
    print(f"  Pred range: [{all_c_pred.min():.3f}, {all_c_pred.max():.3f}]")
    print(f"  True mean: {all_c_true.mean():.3f}, std: {all_c_true.std():.3f}")
    print(f"  Pred mean: {all_c_pred.mean():.3f}, std: {all_c_pred.std():.3f}")
    
    # Denormalize
    h_pred_ppm = all_h_pred * 2.07 + 3.51
    c_pred_ppm = all_c_pred * 50.26 + 79.81
    h_true_ppm = all_h_true * 2.07 + 3.51
    c_true_ppm = all_c_true * 50.26 + 79.81
    
    print("\n" + "="*50)
    print("PPM SPACE ANALYSIS")
    print("="*50)
    
    print(f"\nH NMR (ppm):")
    print(f"  True range: [{h_true_ppm.min():.1f}, {h_true_ppm.max():.1f}]")
    print(f"  Pred range: [{h_pred_ppm.min():.1f}, {h_pred_ppm.max():.1f}]")
    print(f"  MAE: {np.abs(h_pred_ppm - h_true_ppm).mean():.3f} ppm")
    
    print(f"\nC NMR (ppm):")
    print(f"  True range: [{c_true_ppm.min():.1f}, {c_true_ppm.max():.1f}]")
    print(f"  Pred range: [{c_pred_ppm.min():.1f}, {c_pred_ppm.max():.1f}]")
    print(f"  MAE: {np.abs(c_pred_ppm - c_true_ppm).mean():.3f} ppm")
    
    # Check if model is outputting constant values
    print("\n" + "="*50)
    print("PREDICTION DIVERSITY CHECK")
    print("="*50)
    
    print(f"H NMR unique predictions: {len(np.unique(np.round(all_h_pred, 3)))}")
    print(f"C NMR unique predictions: {len(np.unique(np.round(all_c_pred, 3)))}")
    
    if all_h_pred.std() < 0.1:
        print("WARNING: H NMR predictions have very low variance!")
    if all_c_pred.std() < 0.1:
        print("WARNING: C NMR predictions have very low variance!")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # H NMR distribution
    ax = axes[0, 0]
    ax.hist(h_true_ppm, bins=50, alpha=0.5, label='True', color='blue')
    ax.hist(h_pred_ppm, bins=50, alpha=0.5, label='Pred', color='red')
    ax.set_xlabel('Chemical Shift (ppm)')
    ax.set_ylabel('Count')
    ax.set_title('¹H NMR Distribution')
    ax.legend()
    
    # C NMR distribution
    ax = axes[0, 1]
    ax.hist(c_true_ppm, bins=50, alpha=0.5, label='True', color='blue')
    ax.hist(c_pred_ppm, bins=50, alpha=0.5, label='Pred', color='red')
    ax.set_xlabel('Chemical Shift (ppm)')
    ax.set_ylabel('Count')
    ax.set_title('¹³C NMR Distribution')
    ax.legend()
    
    # H NMR scatter
    ax = axes[1, 0]
    ax.scatter(h_true_ppm, h_pred_ppm, alpha=0.5, s=10)
    ax.plot([h_true_ppm.min(), h_true_ppm.max()], 
            [h_true_ppm.min(), h_true_ppm.max()], 'r--')
    ax.set_xlabel('True ¹H (ppm)')
    ax.set_ylabel('Predicted ¹H (ppm)')
    ax.set_title('¹H NMR Correlation')
    
    # C NMR scatter
    ax = axes[1, 1]
    ax.scatter(c_true_ppm, c_pred_ppm, alpha=0.5, s=10)
    ax.plot([c_true_ppm.min(), c_true_ppm.max()], 
            [c_true_ppm.min(), c_true_ppm.max()], 'r--')
    ax.set_xlabel('True ¹³C (ppm)')
    ax.set_ylabel('Predicted ¹³C (ppm)')
    ax.set_title('¹³C NMR Correlation')
    
    plt.tight_layout()
    plt.savefig('nmr_diagnosis.png', dpi=150)
    plt.close()
    
    print(f"\nDiagnostic plots saved to nmr_diagnosis.png")
    
    # Check model weights
    print("\n" + "="*50)
    print("MODEL WEIGHT ANALYSIS")
    print("="*50)
    
    for name, param in model.named_parameters():
        if 'nmr_predictor' in name and 'weight' in name:
            print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

if __name__ == "__main__":
    diagnose_model_performance()