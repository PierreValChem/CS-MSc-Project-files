"""
Training script optimized for SMILES-to-NMR prediction
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

from config import Config
from nmr_dataset import create_data_loaders
from simple_smiles_nmr_model import SMILEStoNMRModel, SimplifiedNMRLoss
from training_utils import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch_nmr_only(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch - NMR only"""
    model.train()
    total_loss = 0
    h_losses = []
    c_losses = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
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
        optimizer.zero_grad()
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute loss
        loss_dict = loss_fn(
            predictions,
            {'nmr_shifts': nmr_targets},
            {'nmr_mask': nmr_mask}
        )
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        h_losses.append(loss_dict['h_nmr_loss'].item())
        c_losses.append(loss_dict['c_nmr_loss'].item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
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
    
    # H NMR MAE
    ax = axes[0, 1]
    ax.plot(epochs, history['val_h_mae_ppm'], 'g-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (ppm)')
    ax.set_title('¹H NMR Prediction Error')
    ax.grid(True)
    
    # C NMR MAE
    ax = axes[1, 0]
    ax.plot(epochs, history['val_c_mae_ppm'], 'm-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (ppm)')
    ax.set_title('¹³C NMR Prediction Error')
    ax.grid(True)
    
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
    """Main training function for SMILES-NMR correlation"""
    # Load configuration
    config = Config.from_yaml('config_smiles_nmr.yaml')  # Use the focused config
    
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
    model = SMILEStoNMRModel(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {trainable_params:,} trainable parameters "
                f"(out of {total_params:,} total)")
    
    # Create loss function
    loss_fn = SimplifiedNMRLoss(config)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.training.scheduler_patience,
        min_lr=1e-7
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_h_mae_ppm': [],
        'val_c_mae_ppm': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    logger.info("Starting SMILES-to-NMR training...")
    
    for epoch in range(1, config.training.num_epochs + 1):
        # Train
        train_metrics = train_epoch_nmr_only(
            model, train_loader, optimizer, loss_fn, device
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
    
    # Final evaluation on test set
    logger.info("Final test evaluation...")
    test_metrics = validate_epoch_nmr_only(
        model, test_loader, loss_fn, device
    )
    logger.info(f"Test Loss: {test_metrics['total_loss']:.4f} | "
                f"H MAE: {test_metrics.get('h_nmr_mae_ppm', 0):.2f} ppm | "
                f"C MAE: {test_metrics.get('c_nmr_mae_ppm', 0):.2f} ppm")
    
    # Save final results
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump({
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'history': history
        }, f, indent=2)
    
    logger.info(f"Training complete! Results saved to {results_dir}")


if __name__ == "__main__":
    main()