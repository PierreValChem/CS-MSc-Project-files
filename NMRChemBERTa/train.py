"""
Enhanced training script with comprehensive reporting and visualization
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

from config import Config
from nmr_dataset import create_data_loaders
from nmr_chemberta_model import NMRChemBERTa
from training_utils import MultiTaskLoss, compute_metrics
from hardware_utils import setup_hardware, get_hardware_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ComprehensiveTrainer:
    """Enhanced trainer with comprehensive metrics tracking and visualization"""
    
    def __init__(self, config, save_dir='results'):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.save_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        self.checkpoints_dir = self.save_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Initialize comprehensive history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_nmr_h_mae': [],
            'val_nmr_h_mae': [],
            'test_nmr_h_mae': [],
            'train_nmr_c_mae': [],
            'val_nmr_c_mae': [],
            'test_nmr_c_mae': [],
            'train_position_rmsd': [],
            'val_position_rmsd': [],
            'test_position_rmsd': [],
            'train_atom_accuracy': [],
            'val_atom_accuracy': [],
            'test_atom_accuracy': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'best_nmr_mae': float('inf'),
            'best_atom_accuracy': 0.0
        }
    
    def update_history(self, epoch, train_metrics, val_metrics, test_metrics, lr, epoch_time):
        """Update training history with all metrics"""
        self.history['epoch'].append(epoch)
        self.history['learning_rate'].append(lr)
        self.history['epoch_time'].append(epoch_time)
        
        # Update losses
        self.history['train_loss'].append(train_metrics.get('total_loss', 0))
        self.history['val_loss'].append(val_metrics.get('total_loss', 0))
        if test_metrics:
            self.history['test_loss'].append(test_metrics.get('total_loss', 0))
        
        # Update NMR metrics
        self.history['train_nmr_h_mae'].append(train_metrics.get('nmr_h_mae_mean', 0))
        self.history['val_nmr_h_mae'].append(val_metrics.get('nmr_h_mae_mean', 0))
        if test_metrics:
            self.history['test_nmr_h_mae'].append(test_metrics.get('nmr_h_mae_mean', 0))
        
        self.history['train_nmr_c_mae'].append(train_metrics.get('nmr_c_mae_mean', 0))
        self.history['val_nmr_c_mae'].append(val_metrics.get('nmr_c_mae_mean', 0))
        if test_metrics:
            self.history['test_nmr_c_mae'].append(test_metrics.get('nmr_c_mae_mean', 0))
        
        # Update other metrics
        self.history['train_position_rmsd'].append(train_metrics.get('position_rmsd_mean', 0))
        self.history['val_position_rmsd'].append(val_metrics.get('position_rmsd_mean', 0))
        if test_metrics:
            self.history['test_position_rmsd'].append(test_metrics.get('position_rmsd_mean', 0))
        
        self.history['train_atom_accuracy'].append(train_metrics.get('atom_type_accuracy_mean', 0))
        self.history['val_atom_accuracy'].append(val_metrics.get('atom_type_accuracy_mean', 0))
        if test_metrics:
            self.history['test_atom_accuracy'].append(test_metrics.get('atom_type_accuracy_mean', 0))
        
        # Update best metrics
        if val_metrics['total_loss'] < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = val_metrics['total_loss']
            self.best_metrics['best_epoch'] = epoch
            self.best_metrics['best_nmr_mae'] = (val_metrics.get('nmr_h_mae_mean', 0) + 
                                                  val_metrics.get('nmr_c_mae_mean', 0)) / 2
            self.best_metrics['best_atom_accuracy'] = val_metrics.get('atom_type_accuracy_mean', 0)
    
    def plot_comprehensive_results(self):
        """Create comprehensive visualization of all training metrics"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a 4x3 grid
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        epochs = self.history['epoch']
        
        # 1. Loss curves (larger plot)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        if self.history['test_loss']:
            ax1.plot(epochs, self.history['test_loss'], 'g-', label='Test', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Loss Curves Over Training', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = self.best_metrics['best_epoch']
        ax1.axvline(x=best_epoch, color='k', linestyle='--', alpha=0.5)
        ax1.text(best_epoch, ax1.get_ylim()[1]*0.9, f'Best: {best_epoch}', ha='center')
        
        # 2. NMR H MAE
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(epochs, self.history['train_nmr_h_mae'], 'b-', label='Train')
        ax2.plot(epochs, self.history['val_nmr_h_mae'], 'r-', label='Val')
        if self.history['test_nmr_h_mae']:
            ax2.plot(epochs, self.history['test_nmr_h_mae'], 'g-', label='Test')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (ppm)')
        ax2.set_title('¹H NMR Mean Absolute Error', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. NMR C MAE
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(epochs, self.history['train_nmr_c_mae'], 'b-', label='Train')
        ax3.plot(epochs, self.history['val_nmr_c_mae'], 'r-', label='Val')
        if self.history['test_nmr_c_mae']:
            ax3.plot(epochs, self.history['test_nmr_c_mae'], 'g-', label='Test')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MAE (ppm)')
        ax3.set_title('¹³C NMR Mean Absolute Error', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Combined NMR Performance
        ax4 = fig.add_subplot(gs[1, 2])
        combined_train = [(h + c) / 2 for h, c in zip(self.history['train_nmr_h_mae'], 
                                                       self.history['train_nmr_c_mae'])]
        combined_val = [(h + c) / 2 for h, c in zip(self.history['val_nmr_h_mae'], 
                                                     self.history['val_nmr_c_mae'])]
        ax4.plot(epochs, combined_train, 'b-', label='Train')
        ax4.plot(epochs, combined_val, 'r-', label='Val')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MAE (ppm)')
        ax4.set_title('Combined NMR Performance', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Atom Type Accuracy
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, [acc * 100 for acc in self.history['train_atom_accuracy']], 'b-', label='Train')
        ax5.plot(epochs, [acc * 100 for acc in self.history['val_atom_accuracy']], 'r-', label='Val')
        if self.history['test_atom_accuracy']:
            ax5.plot(epochs, [acc * 100 for acc in self.history['test_atom_accuracy']], 'g-', label='Test')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Accuracy (%)')
        ax5.set_title('Atom Type Prediction Accuracy', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Position RMSD
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(epochs, self.history['train_position_rmsd'], 'b-', label='Train')
        ax6.plot(epochs, self.history['val_position_rmsd'], 'r-', label='Val')
        if self.history['test_position_rmsd']:
            ax6.plot(epochs, self.history['test_position_rmsd'], 'g-', label='Test')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('RMSD (Å)')
        ax6.set_title('3D Position Prediction RMSD', fontsize=12)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Learning Rate
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.semilogy(epochs, self.history['learning_rate'], 'k-')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Learning Rate')
        ax7.set_title('Learning Rate Schedule', fontsize=12)
        ax7.grid(True, alpha=0.3)
        
        # 8. Training Time
        ax8 = fig.add_subplot(gs[3, 0])
        ax8.plot(epochs, self.history['epoch_time'], 'purple')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Time (seconds)')
        ax8.set_title('Training Time per Epoch', fontsize=12)
        ax8.grid(True, alpha=0.3)
        
        # 9. Final Performance Summary (text)
        ax9 = fig.add_subplot(gs[3, 1:])
        ax9.axis('off')
        
        # Create summary text
        summary_text = f"""
        FINAL MODEL PERFORMANCE SUMMARY
        ==============================
        
        Best Epoch: {self.best_metrics['best_epoch']}
        Best Validation Loss: {self.best_metrics['best_val_loss']:.4f}
        
        Final Performance (Epoch {epochs[-1]}):
        • Validation Loss: {self.history['val_loss'][-1]:.4f}
        • ¹H NMR MAE: {self.history['val_nmr_h_mae'][-1]:.3f} ppm
        • ¹³C NMR MAE: {self.history['val_nmr_c_mae'][-1]:.3f} ppm
        • Atom Type Accuracy: {self.history['val_atom_accuracy'][-1]*100:.1f}%
        • Position RMSD: {self.history['val_position_rmsd'][-1]:.3f} Å
        
        Training Summary:
        • Total Epochs: {len(epochs)}
        • Total Training Time: {sum(self.history['epoch_time'])/60:.1f} minutes
        • Average Time per Epoch: {np.mean(self.history['epoch_time']):.1f} seconds
        """
        
        if self.history['test_loss']:
            test_summary = f"""
        
        Test Set Performance:
        • Test Loss: {self.history['test_loss'][-1]:.4f}
        • ¹H NMR MAE: {self.history['test_nmr_h_mae'][-1]:.3f} ppm
        • ¹³C NMR MAE: {self.history['test_nmr_c_mae'][-1]:.3f} ppm
        • Atom Type Accuracy: {self.history['test_atom_accuracy'][-1]*100:.1f}%
        • Position RMSD: {self.history['test_position_rmsd'][-1]:.3f} Å
        """
            summary_text += test_summary
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top')
        
        plt.suptitle('NMR-ChemBERTa Training Results', fontsize=16, fontweight='bold')
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.plots_dir / f'comprehensive_results_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save individual plots for better visibility
        self._save_individual_plots(epochs)
    
    def _save_individual_plots(self, epochs):
        """Save individual plots for each metric"""
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        if self.history['test_loss']:
            plt.plot(epochs, self.history['test_loss'], 'g-', label='Test', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress - Loss', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # NMR accuracy plot
        plt.figure(figsize=(10, 6))
        combined_val = [(h + c) / 2 for h, c in zip(self.history['val_nmr_h_mae'], 
                                                     self.history['val_nmr_c_mae'])]
        plt.plot(epochs, combined_val, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MAE (ppm)')
        plt.title('NMR Prediction Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / 'nmr_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all results to files"""
        # Save history as JSON
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(self.history)
        df.to_csv(self.save_dir / 'training_history.csv', index=False)
        
        # Save best metrics
        with open(self.save_dir / 'best_metrics.json', 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
        
        # Create final report
        self._create_final_report()
    
    def _create_final_report(self):
        """Create a comprehensive final report"""
        report = f"""
# NMR-ChemBERTa Training Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Training Configuration
- Model: {self.config.model.chemberta_name}
- Hidden Dimension: {self.config.model.hidden_dim}
- Batch Size: {self.config.training.batch_size}
- Learning Rate: {self.config.training.learning_rate}
- Epochs: {self.config.training.num_epochs}

## Dataset Splits
- Train: {self.config.data.train_split * 100:.0f}%
- Validation: {self.config.data.val_split * 100:.0f}%
- Test: {self.config.data.test_split * 100:.0f}%

## Best Performance
- Best Epoch: {self.best_metrics['best_epoch']}
- Best Validation Loss: {self.best_metrics['best_val_loss']:.4f}
- Best Combined NMR MAE: {self.best_metrics['best_nmr_mae']:.3f} ppm
- Best Atom Type Accuracy: {self.best_metrics['best_atom_accuracy']*100:.1f}%

## Final Results (Epoch {self.history['epoch'][-1]})

### Validation Set
- Loss: {self.history['val_loss'][-1]:.4f}
- ¹H NMR MAE: {self.history['val_nmr_h_mae'][-1]:.3f} ppm
- ¹³C NMR MAE: {self.history['val_nmr_c_mae'][-1]:.3f} ppm
- Atom Type Accuracy: {self.history['val_atom_accuracy'][-1]*100:.1f}%
- Position RMSD: {self.history['val_position_rmsd'][-1]:.3f} Å
"""
        
        if self.history['test_loss']:
            report += f"""
### Test Set
- Loss: {self.history['test_loss'][-1]:.4f}
- ¹H NMR MAE: {self.history['test_nmr_h_mae'][-1]:.3f} ppm
- ¹³C NMR MAE: {self.history['test_nmr_c_mae'][-1]:.3f} ppm
- Atom Type Accuracy: {self.history['test_atom_accuracy'][-1]*100:.1f}%
- Position RMSD: {self.history['test_position_rmsd'][-1]:.3f} Å
"""
        
        report += f"""
## Training Summary
- Total Training Time: {sum(self.history['epoch_time'])/60:.1f} minutes
- Average Time per Epoch: {np.mean(self.history['epoch_time']):.1f} seconds

## Notes
- NMR values are normalized using dataset statistics
- Best model saved at: checkpoints/best_model.pt
- All plots saved in: plots/
"""
        
        with open(self.save_dir / 'training_report.md', 'w') as f:
            f.write(report)


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None, use_amp=False):
    """Train for one epoch with proper gradient handling"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass with or without AMP
        if use_amp and scaler is not None:
            with autocast():
                predictions = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    atom_types=batch['atom_types'],
                    h_shifts=batch['h_shifts'],
                    c_shifts=batch['c_shifts'],
                    positions=batch['positions'],
                    h_mask=batch['h_mask'],
                    c_mask=batch['c_mask'],
                    position_mask=batch['position_mask']
                )
                loss, loss_components = loss_fn(predictions, batch)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                atom_types=batch['atom_types'],
                h_shifts=batch['h_shifts'],
                c_shifts=batch['c_shifts'],
                positions=batch['positions'],
                h_mask=batch['h_mask'],
                c_mask=batch['c_mask'],
                position_mask=batch['position_mask']
            )
            loss, loss_components = loss_fn(predictions, batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions and targets
        all_predictions.append({k: v.detach().cpu() for k, v in predictions.items()})
        all_targets.append({
            'h_shifts': batch['h_shifts'].detach().cpu(),
            'c_shifts': batch['c_shifts'].detach().cpu(),
            'positions': batch['positions'].detach().cpu(),
            'atom_types': batch['atom_types'].detach().cpu(),
            'h_mask': batch['h_mask'].detach().cpu(),
            'c_mask': batch['c_mask'].detach().cpu(),
            'position_mask': batch['position_mask'].detach().cpu()
        })
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['total_loss'] = total_loss / len(dataloader)
    
    return metrics


def validate_epoch(model, dataloader, loss_fn, device, use_amp=False):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            if use_amp:
                with autocast():
                    predictions = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        atom_types=batch['atom_types'],
                        h_shifts=batch['h_shifts'],
                        c_shifts=batch['c_shifts'],
                        positions=batch['positions'],
                        h_mask=batch['h_mask'],
                        c_mask=batch['c_mask'],
                        position_mask=batch['position_mask']
                    )
                    loss, _ = loss_fn(predictions, batch)
            else:
                predictions = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    atom_types=batch['atom_types'],
                    h_shifts=batch['h_shifts'],
                    c_shifts=batch['c_shifts'],
                    positions=batch['positions'],
                    h_mask=batch['h_mask'],
                    c_mask=batch['c_mask'],
                    position_mask=batch['position_mask']
                )
                loss, _ = loss_fn(predictions, batch)
            
            total_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append({k: v.cpu() for k, v in predictions.items()})
            all_targets.append({
                'h_shifts': batch['h_shifts'].cpu(),
                'c_shifts': batch['c_shifts'].cpu(),
                'positions': batch['positions'].cpu(),
                'atom_types': batch['atom_types'].cpu(),
                'h_mask': batch['h_mask'].cpu(),
                'c_mask': batch['c_mask'].cpu(),
                'position_mask': batch['position_mask'].cpu()
            })
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['total_loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    """Main training function with enhanced reporting"""
    # Load configuration
    config = Config()
    
    # Setup hardware
    device = setup_hardware(config)
    hardware_config = get_hardware_config()
    
    # Log configuration
    logger.info(f"Training configuration: {config.training}")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, dataset = create_data_loaders(config)
    logger.info(f"Data loaded - Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Initialize model
    logger.info("Creating model...")
    model = NMRChemBERTa(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {trainable_params:,} trainable parameters "
                f"(out of {total_params:,} total)")
    
    # Initialize loss function
    loss_fn = MultiTaskLoss(config)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.training.scheduler_patience,
        min_lr=1e-8,
        verbose=True
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if config.training.use_amp and device.type == 'cuda' else None
    use_amp = config.training.use_amp and device.type == 'cuda'
    
    # Initialize comprehensive trainer
    trainer = ComprehensiveTrainer(config)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config.training.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, use_amp
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, use_amp
        )
        
        # Test periodically
        test_metrics = None
        if test_loader and (epoch % 10 == 0 or epoch == config.training.num_epochs):
            test_metrics = validate_epoch(
                model, test_loader, loss_fn, device, use_amp
            )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Update trainer history
        trainer.update_history(epoch, train_metrics, val_metrics, test_metrics, 
                              current_lr, epoch_time)
        
        # Log epoch results
        logger.info(f"Epoch {epoch:3d} | "
                   f"Train Loss: {train_metrics['total_loss']:.6f} | "
                   f"Val Loss: {val_metrics['total_loss']:.6f} | "
                   f"LR: {current_lr:.2e} | "
                   f"Time: {epoch_time:.1f}s")
        
        # Additional detailed logging every 10 epochs
        if epoch % 10 == 0:
            logger.info(f"  NMR Performance - "
                       f"H MAE: {val_metrics.get('nmr_h_mae_mean', 0):.3f} ppm, "
                       f"C MAE: {val_metrics.get('nmr_c_mae_mean', 0):.3f} ppm")
            logger.info(f"  Other Tasks - "
                       f"Atom Acc: {val_metrics.get('atom_type_accuracy_mean', 0)*100:.1f}%, "
                       f"Pos RMSD: {val_metrics.get('position_rmsd_mean', 0):.3f} Å")
        
        # Step scheduler
        scheduler.step(val_metrics['total_loss'])
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                }
            }
            torch.save(checkpoint, trainer.checkpoints_dir / 'best_model.pt')
            logger.info(f"Saved best model (val loss: {best_val_loss:.6f})")
        
        # Save checkpoint every 25 epochs
        if epoch % 25 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': trainer.history
            }
            torch.save(checkpoint, trainer.checkpoints_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Early stopping check
        if current_lr <= 1e-8 and epoch > config.training.num_epochs // 2:
            logger.info("Learning rate too small, stopping training")
            break
    
    # Final test evaluation
    if test_loader:
        logger.info("Running final test evaluation...")
        test_metrics = validate_epoch(model, test_loader, loss_fn, device, use_amp)
        trainer.update_history(epoch + 1, train_metrics, val_metrics, test_metrics,
                              current_lr, 0)
        logger.info(f"Final Test Loss: {test_metrics['total_loss']:.6f}")
    
    # Generate comprehensive plots and reports
    logger.info("Generating final reports and visualizations...")
    trainer.plot_comprehensive_results()
    trainer.save_results()
    
    logger.info("Training completed! Results saved to 'results/' directory")
    logger.info(f"Best validation loss: {trainer.best_metrics['best_val_loss']:.6f} "
                f"at epoch {trainer.best_metrics['best_epoch']}")
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*60)
    print(f"Best Epoch: {trainer.best_metrics['best_epoch']}")
    print(f"Best Val Loss: {trainer.best_metrics['best_val_loss']:.4f}")
    print(f"Best NMR MAE: {trainer.best_metrics['best_nmr_mae']:.3f} ppm")
    print(f"Best Atom Accuracy: {trainer.best_metrics['best_atom_accuracy']*100:.1f}%")
    print(f"\nResults saved to: {trainer.save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()