import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

class EnhancedMetricsTracker:
    """Enhanced metrics tracking with comprehensive visualization"""
    
    def __init__(self, save_dir='metrics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize all metrics
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_token_accuracy': [],
            'val_token_accuracy': [],
            'test_token_accuracy': [],
            'train_exact_match': [],
            'val_exact_match': [],
            'test_exact_match': [],
            'learning_rate': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'test_perplexity': [],
            'train_mean_tanimoto': [],
            'val_mean_tanimoto': [],
            'test_mean_tanimoto': [],
            'train_validity_rate': [],
            'val_validity_rate': [],
            'test_validity_rate': []
        }
        
        self.predictions_history = []
        self.roc_data = []
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def update(self, epoch, **kwargs):
        """Update metrics for current epoch"""
        # Only append epoch if it's a new one
        if not self.metrics['epoch'] or self.metrics['epoch'][-1] != epoch:
            self.metrics['epoch'].append(epoch)
        
        # Update all provided metrics
        for key, value in kwargs.items():
            if key in self.metrics and value is not None:
                # For test metrics, we might need to pad with None
                if 'test' in key and len(self.metrics[key]) < len(self.metrics['epoch']) - 1:
                    # Pad with None for missing epochs
                    self.metrics[key].extend([None] * (len(self.metrics['epoch']) - 1 - len(self.metrics[key])))
                self.metrics[key].append(value)
            elif key == 'predictions':
                self.predictions_history.append({
                    'epoch': epoch,
                    'predictions': value
                })
    
    def calculate_perplexity(self, loss):
        """Calculate perplexity from loss"""
        return np.exp(loss)
    
    def plot_comprehensive_training_curves(self):
        """Create comprehensive visualization of all training metrics"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Get the actual epochs we have data for
        epochs = self.metrics['epoch']
        
        # Helper function to get valid test data points
        def get_test_data_points(metric_name):
            """Extract non-None test data points with their corresponding epochs"""
            test_data = []
            test_epochs = []
            for i, value in enumerate(self.metrics[metric_name]):
                if value is not None and i < len(epochs):
                    test_data.append(value)
                    test_epochs.append(epochs[i])
            return test_epochs, test_data
        
        # 1. Loss curves (train, val, test)
        ax1 = fig.add_subplot(gs[0, 0])
        
        if self.metrics['train_loss']:
            ax1.plot(epochs[:len(self.metrics['train_loss'])], 
                    self.metrics['train_loss'], 'b-', label='Train', linewidth=2)
        if self.metrics['val_loss']:
            ax1.plot(epochs[:len(self.metrics['val_loss'])], 
                    self.metrics['val_loss'], 'r-', label='Validation', linewidth=2)
        
        # Handle test loss properly
        if self.metrics['test_loss']:
            test_epochs, test_losses = get_test_data_points('test_loss')
            if test_epochs:
                ax1.plot(test_epochs, test_losses, 'g-', label='Test', linewidth=2, marker='o')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training, Validation & Test Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Token Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        
        if self.metrics['train_token_accuracy']:
            ax2.plot(epochs[:len(self.metrics['train_token_accuracy'])], 
                    self.metrics['train_token_accuracy'], 'b-', label='Train', linewidth=2)
        if self.metrics['val_token_accuracy']:
            ax2.plot(epochs[:len(self.metrics['val_token_accuracy'])], 
                    self.metrics['val_token_accuracy'], 'r-', label='Validation', linewidth=2)
        
        # Handle test token accuracy properly
        if self.metrics['test_token_accuracy']:
            test_epochs, test_acc = get_test_data_points('test_token_accuracy')
            if test_epochs:
                ax2.plot(test_epochs, test_acc, 'g-', label='Test', linewidth=2, marker='o')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Token Accuracy (%)')
        ax2.set_title('Token-level Prediction Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Exact Match Accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        
        if self.metrics['train_exact_match']:
            ax3.plot(epochs[:len(self.metrics['train_exact_match'])], 
                    self.metrics['train_exact_match'], 'b-', label='Train', linewidth=2)
        if self.metrics['val_exact_match']:
            ax3.plot(epochs[:len(self.metrics['val_exact_match'])], 
                    self.metrics['val_exact_match'], 'r-', label='Validation', linewidth=2)
        
        # Handle test exact match properly
        if self.metrics['test_exact_match']:
            test_epochs, test_em = get_test_data_points('test_exact_match')
            if test_epochs:
                ax3.plot(test_epochs, test_em, 'g-', label='Test', linewidth=2, marker='o')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Exact Match (%)')
        ax3.set_title('SMILES Exact Match Accuracy', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Combined Accuracy Plot (Token + Exact Match)
        ax4 = fig.add_subplot(gs[1, :2])
        
        if self.metrics['val_token_accuracy'] and self.metrics['val_exact_match']:
            min_len = min(len(self.metrics['val_token_accuracy']), len(self.metrics['val_exact_match']))
            ax4.plot(epochs[:min_len], 
                    self.metrics['val_token_accuracy'][:min_len], 'b-', label='Token Accuracy', linewidth=2)
            ax4.plot(epochs[:min_len], 
                    self.metrics['val_exact_match'][:min_len], 'r-', label='Exact Match', linewidth=2)
            
            # Fill between
            ax4.fill_between(epochs[:min_len],
                        self.metrics['val_exact_match'][:min_len],
                        self.metrics['val_token_accuracy'][:min_len],
                        alpha=0.3, color='gray', label='Gap')
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Validation Accuracy: Token vs Exact Match', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Learning Rate Schedule
        ax5 = fig.add_subplot(gs[1, 2])
        
        if self.metrics['learning_rate']:
            ax5.plot(epochs[:len(self.metrics['learning_rate'])], 
                    self.metrics['learning_rate'], 'g-', linewidth=2)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Learning Rate')
            ax5.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax5.set_yscale('log')
            ax5.grid(True, alpha=0.3)
        
        # 6. Perplexity
        ax6 = fig.add_subplot(gs[2, 0])
        
        if self.metrics['train_loss']:
            train_perp = [self.calculate_perplexity(loss) for loss in self.metrics['train_loss']]
            ax6.plot(epochs[:len(train_perp)], train_perp, 'b-', label='Train', linewidth=2)
        if self.metrics['val_loss']:
            val_perp = [self.calculate_perplexity(loss) for loss in self.metrics['val_loss']]
            ax6.plot(epochs[:len(val_perp)], val_perp, 'r-', label='Validation', linewidth=2)
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Perplexity')
        ax6.set_title('Model Perplexity', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
        
        # 7. Loss Difference (Overfitting Indicator)
        ax7 = fig.add_subplot(gs[2, 1])
        
        if self.metrics['train_loss'] and self.metrics['val_loss']:
            min_len = min(len(self.metrics['train_loss']), len(self.metrics['val_loss']))
            loss_diff = [self.metrics['val_loss'][i] - self.metrics['train_loss'][i] 
                        for i in range(min_len)]
            ax7.plot(epochs[:min_len], loss_diff, 'purple', linewidth=2)
            ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax7.fill_between(epochs[:min_len], 0, loss_diff, 
                        where=[d > 0 for d in loss_diff], 
                        color='red', alpha=0.3, label='Overfitting')
            ax7.fill_between(epochs[:min_len], 0, loss_diff, 
                        where=[d <= 0 for d in loss_diff], 
                        color='green', alpha=0.3, label='Underfitting')
        
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Val Loss - Train Loss')
        ax7.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary Statistics Box
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        # 9. Tanimoto Metrics
        self.plot_tanimoto_metrics()
        
        # Calculate statistics
        stats_text = "Summary Statistics\n" + "="*25 + "\n\n"
        
        if self.metrics['val_loss']:
            best_epoch = np.argmin(self.metrics['val_loss']) + 1
            stats_text += f"Best Epoch (by val loss): {best_epoch}\n"
            stats_text += f"Best Val Loss: {min(self.metrics['val_loss']):.4f}\n"
        
        if self.metrics['val_exact_match']:
            best_exact_epoch = np.argmax(self.metrics['val_exact_match']) + 1
            stats_text += f"Best Exact Match Epoch: {best_exact_epoch}\n"
            stats_text += f"Best Val Exact Match: {max(self.metrics['val_exact_match']):.2f}%\n"
        
        if self.metrics['val_token_accuracy']:
            stats_text += f"Best Val Token Acc: {max(self.metrics['val_token_accuracy']):.2f}%\n"
        
        if self.metrics['test_exact_match']:
            stats_text += f"\nFinal Test Exact Match: {self.metrics['test_exact_match'][-1]:.2f}%\n"
        
        if self.metrics['test_token_accuracy']:
            stats_text += f"Final Test Token Acc: {self.metrics['test_token_accuracy'][-1]:.2f}%\n"
        
        ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Comprehensive Training Metrics Overview', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'comprehensive_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to CSV - handle different lengths
        self._save_metrics_to_csv()
    
    def _save_metrics_to_csv(self):
        """Save metrics to CSV, handling different lengths properly"""
        # Find the maximum length
        max_len = max(len(v) for v in self.metrics.values() if isinstance(v, list))
        
        # Create a dictionary with padded values
        padded_metrics = {}
        for key, values in self.metrics.items():
            if isinstance(values, list):
                # Pad with None to match max length
                padded_values = values + [None] * (max_len - len(values))
                padded_metrics[key] = padded_values
            else:
                padded_metrics[key] = values
        
        # Create DataFrame and save
        metrics_df = pd.DataFrame(padded_metrics)
        metrics_df.to_csv(self.save_dir / 'all_training_metrics.csv', index=False)

    def plot_roc_curves_for_sequence(self, y_true_sequences, y_pred_sequences, epoch, dataset_name='Validation'):
        """Plot ROC curves for sequence-to-sequence models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Flatten sequences for overall ROC
        y_true_flat = y_true_sequences.flatten()
        y_pred_flat = y_pred_sequences.flatten()
        
        # 1. Overall Token-level ROC
        ax = axes[0, 0]
        fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Overall Token-level ROC - {dataset_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Position-wise ROC (first 10 positions)
        ax = axes[0, 1]
        max_positions = min(10, y_true_sequences.shape[1])
        
        for pos in range(max_positions):
            if pos < y_true_sequences.shape[1]:
                y_true_pos = y_true_sequences[:, pos]
                y_pred_pos = y_pred_sequences[:, pos]
                
                # Skip if all values are the same
                if len(np.unique(y_true_pos)) > 1:
                    fpr, tpr, _ = roc_curve(y_true_pos, y_pred_pos)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, linewidth=1.5, label=f'Pos {pos+1} (AUC={roc_auc:.3f})')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Position-wise ROC (First 10 Positions)', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax = axes[1, 0]
        precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_flat)
        avg_precision = average_precision_score(y_true_flat, y_pred_flat)
        
        ax.plot(recall, precision, 'g-', linewidth=2, 
                label=f'PR Curve (AP = {avg_precision:.4f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {dataset_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. ROC AUC by Sequence Length
        ax = axes[1, 1]
        
        # Group sequences by length and calculate AUC for each group
        seq_lengths = (y_true_sequences != 0).sum(axis=1)  # Assuming 0 is padding
        unique_lengths = np.unique(seq_lengths)
        
        length_aucs = []
        length_counts = []
        
        for length in unique_lengths[:20]:  # Limit to first 20 lengths
            mask = seq_lengths == length
            if mask.sum() > 10:  # Only if we have enough samples
                y_true_len = y_true_sequences[mask].flatten()
                y_pred_len = y_pred_sequences[mask].flatten()
                
                if len(np.unique(y_true_len)) > 1:
                    fpr, tpr, _ = roc_curve(y_true_len, y_pred_len)
                    length_auc = auc(fpr, tpr)
                    length_aucs.append(length_auc)
                    length_counts.append(length)
        
        if length_aucs:
            ax.bar(range(len(length_aucs)), length_aucs, color='skyblue', edgecolor='black')
            ax.set_xticks(range(len(length_aucs)))
            ax.set_xticklabels([str(l) for l in length_counts], rotation=45)
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('ROC AUC')
            ax.set_title('ROC AUC by Sequence Length', fontweight='bold')
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'ROC Analysis - {dataset_name} Set (Epoch {epoch})', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'roc_analysis_{dataset_name.lower()}_epoch_{epoch}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store ROC data
        self.roc_data.append({
            'epoch': epoch,
            'dataset': dataset_name,
            'overall_auc': roc_auc,
            'avg_precision': avg_precision
        })
        
        return roc_auc
    
    def plot_tanimoto_metrics(self):
        """Create visualization for Tanimoto similarity metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.metrics['epoch']
        
        # Helper function to get valid test data points
        def get_test_data_points(metric_name):
            """Extract non-None test data points with their corresponding epochs"""
            test_data = []
            test_epochs = []
            for i, value in enumerate(self.metrics[metric_name]):
                if value is not None and i < len(epochs):
                    test_data.append(value)
                    test_epochs.append(epochs[i])
            return test_epochs, test_data
        
        # 1. Mean Tanimoto Similarity over time
        ax = axes[0, 0]
        if self.metrics['train_mean_tanimoto']:
            ax.plot(epochs[:len(self.metrics['train_mean_tanimoto'])], 
                    self.metrics['train_mean_tanimoto'], 'b-', label='Train', linewidth=2)
        if self.metrics['val_mean_tanimoto']:
            ax.plot(epochs[:len(self.metrics['val_mean_tanimoto'])], 
                    self.metrics['val_mean_tanimoto'], 'r-', label='Validation', linewidth=2)
        
        # Handle test tanimoto properly
        if self.metrics['test_mean_tanimoto']:
            test_epochs, test_tanimoto = get_test_data_points('test_mean_tanimoto')
            if test_epochs:
                ax.plot(test_epochs, test_tanimoto, 'g-', label='Test', linewidth=2, marker='o')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Tanimoto Similarity')
        ax.set_title('Molecular Similarity Progress', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Validity Rate over time
        ax = axes[0, 1]
        
        # Train validity rate
        if self.metrics['train_validity_rate']:
            train_validity_values = []
            train_validity_epochs = []
            for i, r in enumerate(self.metrics['train_validity_rate']):
                if r is not None and i < len(epochs):
                    train_validity_values.append(r * 100)
                    train_validity_epochs.append(epochs[i])
            if train_validity_values:
                ax.plot(train_validity_epochs, train_validity_values, 
                        'b-', label='Train', linewidth=2)
        
        # Val validity rate
        if self.metrics['val_validity_rate']:
            val_validity_values = []
            val_validity_epochs = []
            for i, r in enumerate(self.metrics['val_validity_rate']):
                if r is not None and i < len(epochs):
                    val_validity_values.append(r * 100)
                    val_validity_epochs.append(epochs[i])
            if val_validity_values:
                ax.plot(val_validity_epochs, val_validity_values, 
                        'r-', label='Validation', linewidth=2)
        
        # Test validity rate - FIXED
        if self.metrics['test_validity_rate']:
            test_epochs, test_validity = get_test_data_points('test_validity_rate')
            if test_epochs:
                # Convert to percentages
                test_validity_pct = [r * 100 for r in test_validity]
                ax.plot(test_epochs, test_validity_pct, 
                        'g-', label='Test', linewidth=2, marker='o')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validity Rate (%)')
        ax.set_title('SMILES Validity Rate', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Combined view: Exact Match vs Tanimoto
        ax = axes[1, 0]
        if self.metrics['val_exact_match'] and self.metrics['val_mean_tanimoto']:
            # Find the minimum length where both metrics have values
            min_len = min(len(self.metrics['val_exact_match']), 
                        len(self.metrics['val_mean_tanimoto']))
            
            # Only plot where we have both values
            valid_epochs = []
            valid_exact_match = []
            valid_tanimoto = []
            
            for i in range(min_len):
                if (self.metrics['val_exact_match'][i] is not None and 
                    self.metrics['val_mean_tanimoto'][i] is not None):
                    valid_epochs.append(epochs[i])
                    valid_exact_match.append(self.metrics['val_exact_match'][i])
                    valid_tanimoto.append(self.metrics['val_mean_tanimoto'][i])
            
            if valid_epochs:
                ax.plot(valid_epochs, valid_exact_match, 
                        'r-', label='Exact Match (%)', linewidth=2)
                
                # Plot Tanimoto on secondary y-axis
                ax2 = ax.twinx()
                ax2.plot(valid_epochs, valid_tanimoto, 
                        'b-', label='Tanimoto Similarity', linewidth=2)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Exact Match (%)', color='r')
                ax2.set_ylabel('Tanimoto Similarity', color='b')
                ax.tick_params(axis='y', labelcolor='r')
                ax2.tick_params(axis='y', labelcolor='b')
                ax.set_title('Exact Match vs Molecular Similarity', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Combined legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = "Molecular Generation Quality\n" + "="*30 + "\n\n"
        
        # Calculate statistics only from non-None values
        if self.metrics['val_mean_tanimoto']:
            valid_tanimoto_values = [v for v in self.metrics['val_mean_tanimoto'] if v is not None]
            if valid_tanimoto_values:
                best_tanimoto = max(valid_tanimoto_values)
                best_epoch = self.metrics['val_mean_tanimoto'].index(best_tanimoto) + 1
                final_tanimoto = valid_tanimoto_values[-1]
                
                stats_text += f"Best Val Tanimoto: {best_tanimoto:.4f} (Epoch {best_epoch})\n"
                stats_text += f"Final Val Tanimoto: {final_tanimoto:.4f}\n"
        
        if self.metrics['val_validity_rate']:
            valid_validity_values = [v for v in self.metrics['val_validity_rate'] if v is not None]
            if valid_validity_values:
                best_validity = max(valid_validity_values)
                final_validity = valid_validity_values[-1]
                
                stats_text += f"Best Val Validity: {best_validity:.2%}\n"
                stats_text += f"Final Val Validity: {final_validity:.2%}\n"
        
        # Get the last non-None test values
        if self.metrics['test_mean_tanimoto']:
            test_tanimoto_values = [v for v in self.metrics['test_mean_tanimoto'] if v is not None]
            if test_tanimoto_values:
                stats_text += f"\nFinal Test Tanimoto: {test_tanimoto_values[-1]:.4f}\n"
        
        if self.metrics['test_validity_rate']:
            test_validity_values = [v for v in self.metrics['test_validity_rate'] if v is not None]
            if test_validity_values:
                stats_text += f"Final Test Validity: {test_validity_values[-1]:.2%}\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle('Molecular Generation Quality Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'tanimoto_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_final_summary(self):
        """Create a final summary visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Best metrics over time
        ax = axes[0, 0]
        epochs = self.metrics['epoch']
        
        if self.metrics['val_loss']:
            best_loss = [min(self.metrics['val_loss'][:i+1]) for i in range(len(self.metrics['val_loss']))]
            ax.plot(epochs[:len(best_loss)], best_loss, 'b-', label='Best Val Loss', linewidth=2)
        
        if self.metrics['val_exact_match']:
            best_exact = [max(self.metrics['val_exact_match'][:i+1]) for i in range(len(self.metrics['val_exact_match']))]
            ax2 = ax.twinx()
            ax2.plot(epochs[:len(best_exact)], best_exact, 'r-', label='Best Exact Match', linewidth=2)
            ax2.set_ylabel('Best Exact Match (%)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Best Val Loss', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_title('Best Metrics Progress', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Train vs Val vs Test comparison (final epoch)
        ax = axes[0, 1]
        
        if all(self.metrics[f'{split}_exact_match'] for split in ['train', 'val', 'test']):
            final_metrics = {
                'Train': self.metrics['train_exact_match'][-1],
                'Val': self.metrics['val_exact_match'][-1],
                'Test': self.metrics['test_exact_match'][-1]
            }
            
            bars = ax.bar(final_metrics.keys(), final_metrics.values(), 
                          color=['blue', 'red', 'green'], edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
            
            ax.set_ylabel('Exact Match Accuracy (%)')
            ax.set_title('Final Exact Match Comparison', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 3. Learning curve efficiency
        ax = axes[1, 0]
        
        if self.metrics['val_exact_match']:
            improvements = [0] + [self.metrics['val_exact_match'][i] - self.metrics['val_exact_match'][i-1] 
                                 for i in range(1, len(self.metrics['val_exact_match']))]
            
            ax.plot(epochs[:len(improvements)], improvements, 'purple', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.fill_between(epochs[:len(improvements)], 0, improvements,
                          where=[i > 0 for i in improvements],
                          color='green', alpha=0.3, label='Improvement')
            ax.fill_between(epochs[:len(improvements)], 0, improvements,
                          where=[i <= 0 for i in improvements],
                          color='red', alpha=0.3, label='Degradation')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Exact Match Change (%)')
            ax.set_title('Learning Efficiency', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary = self._generate_training_summary()
        ax.text(0.1, 0.9, summary, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Training Summary Report', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_summary_report.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_training_summary(self):
        """Generate text summary of training"""
        summary = "Training Summary\n" + "="*30 + "\n\n"
        
        # Training duration
        total_epochs = len(self.metrics['epoch'])
        summary += f"Total Epochs: {total_epochs}\n\n"
        
        # Best metrics
        if self.metrics['val_loss']:
            best_loss_epoch = np.argmin(self.metrics['val_loss']) + 1
            summary += f"Best Val Loss: {min(self.metrics['val_loss']):.4f} (Epoch {best_loss_epoch})\n"
        
        if self.metrics['val_exact_match']:
            best_exact_epoch = np.argmax(self.metrics['val_exact_match']) + 1
            summary += f"Best Val Exact Match: {max(self.metrics['val_exact_match']):.2f}% (Epoch {best_exact_epoch})\n"
        
        if self.metrics['val_token_accuracy']:
            summary += f"Best Val Token Acc: {max(self.metrics['val_token_accuracy']):.2f}%\n"
        
        # Final test results
        summary += "\nFinal Test Results:\n"
        if self.metrics['test_loss']:
            summary += f"  Loss: {self.metrics['test_loss'][-1]:.4f}\n"
        if self.metrics['test_token_accuracy']:
            summary += f"  Token Accuracy: {self.metrics['test_token_accuracy'][-1]:.2f}%\n"
        if self.metrics['test_exact_match']:
            summary += f"  Exact Match: {self.metrics['test_exact_match'][-1]:.2f}%\n"
        
        # Training efficiency
        if self.metrics['val_exact_match'] and len(self.metrics['val_exact_match']) > 1:
            total_improvement = self.metrics['val_exact_match'][-1] - self.metrics['val_exact_match'][0]
            summary += f"\nTotal Improvement: {total_improvement:.2f}%\n"
            summary += f"Avg Improvement/Epoch: {total_improvement/total_epochs:.2f}%\n"
        
        return summary
    
    def save_final_report(self):
        """Save comprehensive final report"""
        report = {
            'training_summary': self._generate_training_summary(),
            'final_metrics': {
                'train': {
                    'loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
                    'token_accuracy': self.metrics['train_token_accuracy'][-1] if self.metrics['train_token_accuracy'] else None,
                    'exact_match': self.metrics['train_exact_match'][-1] if self.metrics['train_exact_match'] else None
                },
                'validation': {
                    'loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
                    'token_accuracy': self.metrics['val_token_accuracy'][-1] if self.metrics['val_token_accuracy'] else None,
                    'exact_match': self.metrics['val_exact_match'][-1] if self.metrics['val_exact_match'] else None
                },
                'test': {
                    'loss': self.metrics['test_loss'][-1] if self.metrics['test_loss'] else None,
                    'token_accuracy': self.metrics['test_token_accuracy'][-1] if self.metrics['test_token_accuracy'] else None,
                    'exact_match': self.metrics['test_exact_match'][-1] if self.metrics['test_exact_match'] else None
                }
            },
            'best_metrics': {
                'val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else None,
                'val_exact_match': max(self.metrics['val_exact_match']) if self.metrics['val_exact_match'] else None,
                'val_token_accuracy': max(self.metrics['val_token_accuracy']) if self.metrics['val_token_accuracy'] else None
            },
            'roc_data': self.roc_data,
            'total_epochs': len(self.metrics['epoch']),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.save_dir / 'final_training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create all final visualizations
        self.plot_comprehensive_training_curves()
        self.plot_final_summary()
        