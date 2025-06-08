"""
Improved training script for SMILES-to-NMR prediction with early stopping and comprehensive reporting
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import pandas as pd
from scipy import stats

from config import Config
from nmr_dataset import create_data_loaders
from enhanced_smiles_nmr_model import EnhancedSMILEStoNMRModel, AdaptiveLoss
from training_utils import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping with patience"""
    def __init__(self, patience=15, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        score = -val_loss  # Negative because we want to minimize
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
        return self.early_stop


def analyze_predictions(model, dataloader, device, num_samples=100):
    """Analyze model predictions for confidence and error distribution"""
    model.eval()
    
    all_h_errors = []
    all_c_errors = []
    all_h_predictions = []
    all_c_predictions = []
    all_h_targets = []
    all_c_targets = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Prepare NMR targets
            h_targets = batch['nmr_features']['h_shifts'].to(device)
            c_targets = batch['nmr_features']['c_shifts'].to(device)
            h_mask = batch['nmr_features']['h_mask'].to(device)
            c_mask = batch['nmr_features']['c_mask'].to(device)
            
            # Forward pass
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_shifts = predictions['nmr_shifts']
            
            # Extract valid predictions and targets
            h_pred = pred_shifts[:, :, 0][h_mask.bool()]
            c_pred = pred_shifts[:, :, 1][c_mask.bool()]
            h_tgt = h_targets[h_mask.bool()]
            c_tgt = c_targets[c_mask.bool()]
            
            # Calculate errors (in normalized space)
            h_errors = (h_pred - h_tgt).cpu().numpy()
            c_errors = (c_pred - c_tgt).cpu().numpy()
            
            # Store for analysis
            all_h_errors.extend(h_errors)
            all_c_errors.extend(c_errors)
            all_h_predictions.extend(h_pred.cpu().numpy())
            all_c_predictions.extend(c_pred.cpu().numpy())
            all_h_targets.extend(h_tgt.cpu().numpy())
            all_c_targets.extend(c_tgt.cpu().numpy())
            
            sample_count += batch['input_ids'].size(0)
    
    # Convert to arrays
    all_h_errors = np.array(all_h_errors)
    all_c_errors = np.array(all_c_errors)
    all_h_predictions = np.array(all_h_predictions)
    all_c_predictions = np.array(all_c_predictions)
    all_h_targets = np.array(all_h_targets)
    all_c_targets = np.array(all_c_targets)
    
    # Denormalize errors for interpretability
    h_errors_ppm = all_h_errors * 2.5  # Assuming std=2.5 for H NMR
    c_errors_ppm = all_c_errors * 50.0  # Assuming std=50.0 for C NMR
    
    # Calculate statistics
    h_stats = {
        'mae': np.mean(np.abs(h_errors_ppm)),
        'rmse': np.sqrt(np.mean(h_errors_ppm**2)),
        'std': np.std(h_errors_ppm),
        'median_abs_error': np.median(np.abs(h_errors_ppm)),
        'percentile_90': np.percentile(np.abs(h_errors_ppm), 90),
        'percentile_95': np.percentile(np.abs(h_errors_ppm), 95),
        'r2': stats.pearsonr(all_h_predictions, all_h_targets)[0]**2,
        'count': len(all_h_errors)
    }
    
    c_stats = {
        'mae': np.mean(np.abs(c_errors_ppm)),
        'rmse': np.sqrt(np.mean(c_errors_ppm**2)),
        'std': np.std(c_errors_ppm),
        'median_abs_error': np.median(np.abs(c_errors_ppm)),
        'percentile_90': np.percentile(np.abs(c_errors_ppm), 90),
        'percentile_95': np.percentile(np.abs(c_errors_ppm), 95),
        'r2': stats.pearsonr(all_c_predictions, all_c_targets)[0]**2,
        'count': len(all_c_errors)
    }
    
    return {
        'h_nmr': h_stats,
        'c_nmr': c_stats,
        'h_errors': h_errors_ppm,
        'c_errors': c_errors_ppm,
        'h_predictions': all_h_predictions,
        'c_predictions': all_c_predictions,
        'h_targets': all_h_targets,
        'c_targets': all_c_targets
    }


def create_error_plots(analysis_results, save_dir):
    """Create detailed error analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # H NMR error distribution
    ax = axes[0, 0]
    ax.hist(analysis_results['h_errors'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Error (ppm)')
    ax.set_ylabel('Count')
    ax.set_title('¹H NMR Error Distribution')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # C NMR error distribution
    ax = axes[0, 1]
    ax.hist(analysis_results['c_errors'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Error (ppm)')
    ax.set_ylabel('Count')
    ax.set_title('¹³C NMR Error Distribution')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # H NMR correlation plot
    ax = axes[0, 2]
    ax.scatter(analysis_results['h_targets'], analysis_results['h_predictions'], 
               alpha=0.5, s=10, color='blue')
    ax.plot([analysis_results['h_targets'].min(), analysis_results['h_targets'].max()],
            [analysis_results['h_targets'].min(), analysis_results['h_targets'].max()],
            'r--', alpha=0.5)
    ax.set_xlabel('True H shifts (normalized)')
    ax.set_ylabel('Predicted H shifts (normalized)')
    ax.set_title(f'¹H NMR Correlation (R² = {analysis_results["h_nmr"]["r2"]:.3f})')
    
    # C NMR correlation plot
    ax = axes[1, 0]
    ax.scatter(analysis_results['c_targets'], analysis_results['c_predictions'], 
               alpha=0.5, s=10, color='green')
    ax.plot([analysis_results['c_targets'].min(), analysis_results['c_targets'].max()],
            [analysis_results['c_targets'].min(), analysis_results['c_targets'].max()],
            'r--', alpha=0.5)
    ax.set_xlabel('True C shifts (normalized)')
    ax.set_ylabel('Predicted C shifts (normalized)')
    ax.set_title(f'¹³C NMR Correlation (R² = {analysis_results["c_nmr"]["r2"]:.3f})')
    
    # Error percentiles
    ax = axes[1, 1]
    percentiles = [50, 75, 90, 95, 99]
    h_percentiles = [np.percentile(np.abs(analysis_results['h_errors']), p) for p in percentiles]
    c_percentiles = [np.percentile(np.abs(analysis_results['c_errors']), p) for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.35
    ax.bar(x - width/2, h_percentiles, width, label='¹H NMR', color='blue', alpha=0.7)
    ax.bar(x + width/2, c_percentiles, width, label='¹³C NMR', color='green', alpha=0.7)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Absolute Error (ppm)')
    ax.set_title('Error Percentiles')
    ax.set_xticks(x)
    ax.set_xticklabels(percentiles)
    ax.legend()
    
    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    Summary Statistics:
    
    ¹H NMR:
    • MAE: {analysis_results['h_nmr']['mae']:.3f} ppm
    • RMSE: {analysis_results['h_nmr']['rmse']:.3f} ppm
    • Median: {analysis_results['h_nmr']['median_abs_error']:.3f} ppm
    • 90th %ile: {analysis_results['h_nmr']['percentile_90']:.3f} ppm
    • R²: {analysis_results['h_nmr']['r2']:.3f}
    
    ¹³C NMR:
    • MAE: {analysis_results['c_nmr']['mae']:.3f} ppm
    • RMSE: {analysis_results['c_nmr']['rmse']:.3f} ppm
    • Median: {analysis_results['c_nmr']['median_abs_error']:.3f} ppm
    • 90th %ile: {analysis_results['c_nmr']['percentile_90']:.3f} ppm
    • R²: {analysis_results['c_nmr']['r2']:.3f}
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_final_report(config, history, test_analysis, val_analysis, training_time, save_dir):
    """Generate comprehensive training report"""
    report_content = f"""
# SMILES-to-NMR Model Training Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Configuration
- Architecture: ChemBERTa-based SMILES-to-NMR predictor
- Base model: {config.model.chemberta_name}
- Hidden dimension: {config.model.hidden_dim}
- Maximum atoms: {config.model.max_atoms}
- ChemBERTa frozen: {config.model.freeze_chemberta}

## Dataset Information
- Total samples: {config.data.train_split + config.data.val_split + config.data.test_split:.0%} of available data
- Train/Val/Test split: {config.data.train_split:.0%}/{config.data.val_split:.0%}/{config.data.test_split:.0%}
- Batch size: {config.data.batch_size}

## Training Summary
- Total epochs trained: {len(history['train_loss'])}
- Total training time: {training_time/3600:.2f} hours
- Best validation loss: {min(history['val_loss']):.4f}
- Best epoch: {np.argmin(history['val_loss']) + 1}
- Final learning rate: {history['learning_rate'][-1]:.2e}

## Final Performance Metrics

### Test Set Results
**¹H NMR Predictions:**
- Mean Absolute Error: {test_analysis['h_nmr']['mae']:.3f} ppm
- Root Mean Square Error: {test_analysis['h_nmr']['rmse']:.3f} ppm
- Median Absolute Error: {test_analysis['h_nmr']['median_abs_error']:.3f} ppm
- 90th Percentile Error: {test_analysis['h_nmr']['percentile_90']:.3f} ppm
- 95th Percentile Error: {test_analysis['h_nmr']['percentile_95']:.3f} ppm
- R² Score: {test_analysis['h_nmr']['r2']:.3f}
- Number of predictions: {test_analysis['h_nmr']['count']:,}

**¹³C NMR Predictions:**
- Mean Absolute Error: {test_analysis['c_nmr']['mae']:.3f} ppm
- Root Mean Square Error: {test_analysis['c_nmr']['rmse']:.3f} ppm
- Median Absolute Error: {test_analysis['c_nmr']['median_abs_error']:.3f} ppm
- 90th Percentile Error: {test_analysis['c_nmr']['percentile_90']:.3f} ppm
- 95th Percentile Error: {test_analysis['c_nmr']['percentile_95']:.3f} ppm
- R² Score: {test_analysis['c_nmr']['r2']:.3f}
- Number of predictions: {test_analysis['c_nmr']['count']:,}

### Model Confidence Analysis
Based on the error distribution analysis:

**¹H NMR Confidence Levels:**
- High confidence (≤0.3 ppm error): ~{(np.abs(test_analysis['h_errors']) <= 0.3).mean()*100:.1f}% of predictions
- Medium confidence (0.3-0.5 ppm error): ~{((np.abs(test_analysis['h_errors']) > 0.3) & (np.abs(test_analysis['h_errors']) <= 0.5)).mean()*100:.1f}% of predictions
- Low confidence (>0.5 ppm error): ~{(np.abs(test_analysis['h_errors']) > 0.5).mean()*100:.1f}% of predictions

**¹³C NMR Confidence Levels:**
- High confidence (≤5 ppm error): ~{(np.abs(test_analysis['c_errors']) <= 5).mean()*100:.1f}% of predictions
- Medium confidence (5-15 ppm error): ~{((np.abs(test_analysis['c_errors']) > 5) & (np.abs(test_analysis['c_errors']) <= 15)).mean()*100:.1f}% of predictions
- Low confidence (>15 ppm error): ~{(np.abs(test_analysis['c_errors']) > 15).mean()*100:.1f}% of predictions

## Recommendations for Improvement

Based on the current performance:
"""
    
    # Add specific recommendations based on performance
    if test_analysis['h_nmr']['mae'] > 0.5:
        report_content += """
1. **¹H NMR accuracy needs improvement:**
   - Consider unfreezing ChemBERTa for fine-tuning
   - Increase model capacity (hidden dimensions)
   - Add attention mechanisms for better peak-structure correlation
   - Implement data augmentation strategies
"""
    
    if test_analysis['c_nmr']['mae'] > 10:
        report_content += """
2. **¹³C NMR accuracy needs improvement:**
   - The high error suggests the model struggles with carbon chemical environment prediction
   - Consider adding explicit functional group encoding
   - Implement separate heads for different carbon types (sp³, sp², aromatic)
   - Use a larger or chemistry-specific pre-trained model
"""
    
    if test_analysis['h_nmr']['r2'] < 0.8 or test_analysis['c_nmr']['r2'] < 0.8:
        report_content += """
3. **Low correlation scores indicate systematic issues:**
   - Check for data quality issues or outliers
   - Consider implementing ensemble methods
   - Add regularization to prevent overfitting
   - Investigate if certain molecular structures are poorly predicted
"""
    
    report_content += """
## Usage Guidelines

Based on the model's performance, here are recommended use cases:

**Suitable for:**
"""
    if test_analysis['h_nmr']['mae'] < 0.3:
        report_content += "- Rapid ¹H NMR chemical shift estimation for drug discovery\n"
    if test_analysis['c_nmr']['mae'] < 10:
        report_content += "- Initial ¹³C NMR predictions for structure verification\n"
    
    report_content += """- High-throughput screening where approximate NMR values are sufficient
- Educational purposes and teaching NMR-structure relationships

**Not recommended for:**
- Final structure elucidation requiring high precision
- Publication-quality NMR predictions without experimental validation
- Molecules significantly different from the training set

## Files Generated
- `best_model.pt`: Best model checkpoint
- `error_analysis.png`: Detailed error distribution plots
- `nmr_training_results.png`: Training progress plots
- `final_results.json`: All numerical results in JSON format
- `training_report.md`: This report
"""
    
    # Save report
    with open(save_dir / 'training_report.md', 'w') as f:
        f.write(report_content)
    



def train_epoch_nmr_only(model, dataloader, optimizer, loss_fn, device, accumulation_steps=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    h_losses = []
    c_losses = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Prepare NMR targets
        nmr_targets = torch.stack([
            batch['nmr_features']['h_shifts'],
            batch['nmr_features']['c_shifts']
        ], dim=-1).to(device)
        
        nmr_mask = torch.stack([
            batch['nmr_features']['h_mask'],
            batch['nmr_features']['c_mask']
        ], dim=-1).to(device)
        
        # Forward pass
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute loss
        loss_dict = loss_fn(
            predictions,
            {'nmr_shifts': nmr_targets},
            {'nmr_mask': nmr_mask}
        )
        
        loss = loss_dict['total_loss'] / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Track losses
        total_loss += loss_dict['total_loss'].item()
        h_losses.append(loss_dict['h_nmr_loss'].item())
        c_losses.append(loss_dict['c_nmr_loss'].item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total_loss'].item():.4f}",
            'H_loss': f"{loss_dict['h_nmr_loss'].item():.4f}",
            'C_loss': f"{loss_dict['c_nmr_loss'].item():.4f}"
        })
    
    return {
        'total_loss': total_loss / len(dataloader),
        'h_nmr_loss': np.mean(h_losses),
        'c_nmr_loss': np.mean(c_losses)
    }


def validate_epoch_nmr_only(model, dataloader, loss_fn, device):
    """Validate for one epoch - NMR only"""
    model.eval()
    total_loss = 0
    h_losses = []
    c_losses = []
    h_maes = []
    c_maes = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Prepare NMR targets
            nmr_targets = torch.stack([
                batch['nmr_features']['h_shifts'],
                batch['nmr_features']['c_shifts']
            ], dim=-1).to(device)
            
            nmr_mask = torch.stack([
                batch['nmr_features']['h_mask'],
                batch['nmr_features']['c_mask']
            ], dim=-1).to(device)
            
            # Forward pass
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss_dict = loss_fn(
                predictions,
                {'nmr_shifts': nmr_targets},
                {'nmr_mask': nmr_mask}
            )
            
            # Track losses
            total_loss += loss_dict['total_loss'].item()
            h_losses.append(loss_dict['h_nmr_loss'].item())
            c_losses.append(loss_dict['c_nmr_loss'].item())
            
            # Compute MAE for better interpretability
            pred_shifts = predictions['nmr_shifts']
            h_mae = torch.abs(pred_shifts[:, :, 0] - nmr_targets[:, :, 0])
            c_mae = torch.abs(pred_shifts[:, :, 1] - nmr_targets[:, :, 1])
            
            # Apply masks and compute mean
            h_mae_masked = (h_mae * nmr_mask[:, :, 0]).sum() / (nmr_mask[:, :, 0].sum() + 1e-6)
            c_mae_masked = (c_mae * nmr_mask[:, :, 1]).sum() / (nmr_mask[:, :, 1].sum() + 1e-6)
            
            h_maes.append(h_mae_masked.item())
            c_maes.append(c_mae_masked.item())
    
    # Denormalize MAE values for interpretability
    h_mae_ppm = np.mean(h_maes) * 2.5  # Assuming std=2.5 for H NMR
    c_mae_ppm = np.mean(c_maes) * 50.0  # Assuming std=50.0 for C NMR
    
    return {
        'total_loss': total_loss / len(dataloader),
        'h_nmr_loss': np.mean(h_losses),
        'c_nmr_loss': np.mean(c_losses),
        'h_nmr_mae': np.mean(h_maes),
        'c_nmr_mae': np.mean(c_maes),
        'h_nmr_mae_ppm': h_mae_ppm,
        'c_nmr_mae_ppm': c_mae_ppm
    }


def plot_nmr_results(history, save_dir):
    """Plot training results specific to NMR prediction"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax.text(best_epoch, ax.get_ylim()[1]*0.9, f'Best: {best_epoch}', ha='center')
    
    # H NMR MAE
    ax = axes[0, 1]
    ax.plot(epochs, history['val_h_mae_ppm'], 'g-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (ppm)')
    ax.set_title('¹H NMR Prediction Error')
    ax.grid(True)
    ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Target: 0.3 ppm')
    ax.legend()
    
    # C NMR MAE
    ax = axes[1, 0]
    ax.plot(epochs, history['val_c_mae_ppm'], 'm-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (ppm)')
    ax.set_title('¹³C NMR Prediction Error')
    ax.grid(True)
    ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='Target: 5 ppm')
    ax.legend()
    
    # Learning rate
    ax = axes[1, 1]
    ax.semilogy(epochs, history['learning_rate'], 'k-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'nmr_training_results.png', dpi=150)
    plt.close()


def main():
    """Main training function for SMILES-NMR correlation with improvements"""
    # Load configuration
    config = Config.from_yaml('config_smiles_nmr.yaml')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create results directory
    results_dir = Path('results_smiles_nmr')
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, dataset = create_data_loaders(config)
    logger.info(f"Data loaded - Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating SMILES-to-NMR model...")
    model = EnhancedSMILEStoNMRModel(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {trainable_params:,} trainable parameters "
                f"(out of {total_params:,} total)")
    
    # Create loss function
    loss_fn = AdaptiveLoss(config)
    
    # Create optimizer with different learning rates
    optimizer_params = [
        {'params': model.chemberta.parameters(), 'lr': config.training.learning_rate * 0.1},
        {'params': model.projection.parameters(), 'lr': config.training.learning_rate},
        {'params': model.attention_pool.parameters(), 'lr': config.training.learning_rate},
        {'params': model.nmr_predictor.parameters(), 'lr': config.training.learning_rate},
    ]
    
    optimizer = AdamW(optimizer_params, weight_decay=config.training.weight_decay)
    
    # Create scheduler with more aggressive reduction
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.training.scheduler_patience,
        min_lr=1e-7,
        verbose=True
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config.training.early_stopping_patience, min_delta=0.0001)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_h_mae_ppm': [],
        'val_c_mae_ppm': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    # Training loop
    logger.info("Starting SMILES-to-NMR training...")
    
    for epoch in range(1, config.training.num_epochs + 1):
        # Train with gradient accumulation
        train_metrics = train_epoch_nmr_only(
            model, train_loader, optimizer, loss_fn, device, 
            accumulation_steps=config.training.accumulation_steps
        )
        
        # Validate
        val_metrics = validate_epoch_nmr_only(
            model, val_loader, loss_fn, device
        )
        
        # Get learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['val_h_mae_ppm'].append(val_metrics.get('h_nmr_mae_ppm', 0))
        history['val_c_mae_ppm'].append(val_metrics.get('c_nmr_mae_ppm', 0))
        history['learning_rate'].append(current_lr)
        
        # Log results
        logger.info(f"Epoch {epoch:3d} | "
                   f"Train Loss: {train_metrics['total_loss']:.4f} | "
                   f"Val Loss: {val_metrics['total_loss']:.4f} | "
                   f"H MAE: {val_metrics.get('h_nmr_mae_ppm', 0):.2f} ppm | "
                   f"C MAE: {val_metrics.get('c_nmr_mae_ppm', 0):.2f} ppm | "
                   f"LR: {current_lr:.2e}")
        
        # Step scheduler
        scheduler.step(val_metrics['total_loss'])
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, results_dir / 'best_model.pt')
            logger.info(f"Saved best model (val loss: {best_val_loss:.4f})")
        
        # Check early stopping
        if early_stopping(val_metrics['total_loss'], epoch):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Best model was at epoch {early_stopping.best_epoch}")
            break
        
        # Plot results every N epochs
        if epoch % config.logging.plot_every_n_epochs == 0:
            plot_nmr_results(history, results_dir)
        
        # Save checkpoint
        if epoch % config.training.save_every_n_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, results_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Stop if learning rate is too small
        if current_lr <= 1e-7:
            logger.info("Learning rate too small, stopping training")
            break
    
    # Calculate training time
    training_time = time.time() - training_start_time
    
    # Load best model for final evaluation
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load(results_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Detailed analysis on validation set
    logger.info("Analyzing validation set performance...")
    val_analysis = analyze_predictions(model, val_loader, device, num_samples=1000)
    
    # Final evaluation on test set
    logger.info("Final test evaluation...")
    test_metrics = validate_epoch_nmr_only(
        model, test_loader, loss_fn, device
    )
    
    # Detailed analysis on test set
    logger.info("Analyzing test set performance...")
    test_analysis = analyze_predictions(model, test_loader, device, num_samples=1000)
    
    # Create error analysis plots
    create_error_plots(test_analysis, results_dir)
    
    # Generate final report
    logger.info("Generating final report...")
    report = generate_final_report(
        config, history, test_analysis, val_analysis, training_time, results_dir
    )
    
    # Save final results
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump({
            'best_val_loss': best_val_loss,
            'best_epoch': early_stopping.best_epoch,
            'test_metrics': test_metrics,
            'test_analysis': {
                'h_nmr': test_analysis['h_nmr'],
                'c_nmr': test_analysis['c_nmr']
            },
            'val_analysis': {
                'h_nmr': val_analysis['h_nmr'],
                'c_nmr': val_analysis['c_nmr']
            },
            'history': history,
            'training_time_hours': training_time / 3600
        }, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Best epoch: {early_stopping.best_epoch}")
    logger.info(f"Training time: {training_time/3600:.2f} hours")
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  ¹H NMR MAE: {test_analysis['h_nmr']['mae']:.3f} ppm (median: {test_analysis['h_nmr']['median_abs_error']:.3f})")
    logger.info(f"  ¹³C NMR MAE: {test_analysis['c_nmr']['mae']:.3f} ppm (median: {test_analysis['c_nmr']['median_abs_error']:.3f})")
    logger.info(f"  ¹H NMR R²: {test_analysis['h_nmr']['r2']:.3f}")
    logger.info(f"  ¹³C NMR R²: {test_analysis['c_nmr']['r2']:.3f}")
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()