"""
Train SMILES-to-NMR model using 3D coordinates
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging

from config import Config
from nmr_dataset import create_data_loaders
from nmr_model_with_3d_coords import SMILESNMRModelWith3D, SpatialAwareLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch_with_3d(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch using 3D coordinates"""
    model.train()
    total_loss = 0
    h_losses = []
    c_losses = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move all data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        coords = batch['coords'].to(device)  # 3D coordinates
        atom_types = batch['atom_types'].to(device)
        atom_mask = batch['atom_mask'].to(device)
        
        # NMR targets
        nmr_targets = torch.stack([
            batch['nmr_features']['h_shifts'],
            batch['nmr_features']['c_shifts']
        ], dim=-1).to(device)
        
        nmr_mask = torch.stack([
            batch['nmr_features']['h_mask'],
            batch['nmr_features']['c_mask']
        ], dim=-1).to(device)
        
        # Forward pass with 3D coordinates
        predictions = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coords=coords,
            atom_types=atom_types,
            atom_mask=atom_mask
        )
        
        # Compute loss
        loss_dict = loss_fn(
            predictions,
            {'nmr_shifts': nmr_targets},
            {'nmr_mask': nmr_mask}
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        total_loss += loss_dict['total_loss'].item()
        h_losses.append(loss_dict['h_nmr_loss'].item())
        c_losses.append(loss_dict['c_nmr_loss'].item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total_loss'].item():.4f}",
            'H': f"{loss_dict['h_nmr_loss'].item():.4f}",
            'C': f"{loss_dict['c_nmr_loss'].item():.4f}"
        })
    
    return {
        'total_loss': total_loss / len(dataloader),
        'h_nmr_loss': np.mean(h_losses),
        'c_nmr_loss': np.mean(c_losses)
    }


def validate_with_3d(model, dataloader, loss_fn, device):
    """Validate using 3D coordinates"""
    model.eval()
    total_loss = 0
    h_maes = []
    c_maes = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coords = batch['coords'].to(device)
            atom_types = batch['atom_types'].to(device)
            atom_mask = batch['atom_mask'].to(device)
            
            # NMR targets
            nmr_targets = torch.stack([
                batch['nmr_features']['h_shifts'],
                batch['nmr_features']['c_shifts']
            ], dim=-1).to(device)
            
            nmr_mask = torch.stack([
                batch['nmr_features']['h_mask'],
                batch['nmr_features']['c_mask']
            ], dim=-1).to(device)
            
            # Forward pass
            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                coords=coords,
                atom_types=atom_types,
                atom_mask=atom_mask
            )
            
            # Compute loss
            loss_dict = loss_fn(
                predictions,
                {'nmr_shifts': nmr_targets},
                {'nmr_mask': nmr_mask}
            )
            
            total_loss += loss_dict['total_loss'].item()
            
            # Calculate MAE
            pred_shifts = predictions['nmr_shifts']
            h_mae = torch.abs(pred_shifts[:, :, 0] - nmr_targets[:, :, 0])
            c_mae = torch.abs(pred_shifts[:, :, 1] - nmr_targets[:, :, 1])
            
            h_mae_masked = (h_mae * nmr_mask[:, :, 0]).sum() / (nmr_mask[:, :, 0].sum() + 1e-6)
            c_mae_masked = (c_mae * nmr_mask[:, :, 1]).sum() / (nmr_mask[:, :, 1].sum() + 1e-6)
            
            h_maes.append(h_mae_masked.item())
            c_maes.append(c_mae_masked.item())
    
    # Denormalize MAE
    h_mae_ppm = np.mean(h_maes) * 2.07
    c_mae_ppm = np.mean(c_maes) * 50.26
    
    return {
        'total_loss': total_loss / len(dataloader),
        'h_mae_ppm': h_mae_ppm,
        'c_mae_ppm': c_mae_ppm
    }


def main():
    """Main training function with 3D coordinates"""
    # Configuration
    config = Config.from_yaml('config_smiles_nmr_improved.yaml')
    
    # Adjust config for 3D model
    config.model.hidden_dim = 384  # Match your checkpoint
    config.training.learning_rate = 5e-5
    config.training.num_epochs = 100
    config.data.batch_size = 16
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create results directory
    results_dir = Path('results_3d_nmr')
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, _ = create_data_loaders(config)
    
    # Create model
    logger.info("Creating 3D-aware NMR model...")
    model = SMILESNMRModelWith3D(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {trainable_params:,} trainable (out of {total_params:,})")
    
    # Loss and optimizer
    loss_fn = SpatialAwareLoss(config)
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'h_mae': [], 'c_mae': []}
    
    for epoch in range(1, config.training.num_epochs + 1):
        # Train
        train_metrics = train_epoch_with_3d(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        val_metrics = validate_with_3d(model, val_loader, loss_fn, device)
        
        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['h_mae'].append(val_metrics['h_mae_ppm'])
        history['c_mae'].append(val_metrics['c_mae_ppm'])
        
        # Log results
        logger.info(f"Epoch {epoch:3d} | "
                   f"Train: {train_metrics['total_loss']:.4f} | "
                   f"Val: {val_metrics['total_loss']:.4f} | "
                   f"H MAE: {val_metrics['h_mae_ppm']:.2f} ppm | "
                   f"C MAE: {val_metrics['c_mae_ppm']:.2f} ppm")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, results_dir / 'best_3d_model.pt')
            logger.info(f"Saved best model (val loss: {best_val_loss:.4f})")
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping
        if len(history['val_loss']) > 20:
            recent_losses = history['val_loss'][-10:]
            if min(recent_losses) > best_val_loss + 0.001:
                logger.info("Early stopping triggered")
                break
    
    logger.info("Training complete!")
    logger.info(f"Best H MAE: {min(history['h_mae']):.2f} ppm")
    logger.info(f"Best C MAE: {min(history['c_mae']):.2f} ppm")


if __name__ == "__main__":
    main()