#!/usr/bin/env python3
"""
NMR to SMILES Prediction using ChemBERTa with Enhanced Visualization
Trains on complete/perfect data only with detailed performance tracking
"""

import sys
import os

# Add root path to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f"ROOT_DIR: {ROOT_DIR}")  # Debug print
sys.path.insert(0, ROOT_DIR)


import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from transformers import (
    RobertaTokenizer, 
    RobertaModel,
    RobertaConfig,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from enhanced_metrics_module import EnhancedMetricsTracker
from tanimoto_metrics import TanimotoCalculator
from smiles_tokenizer import SMILESTokenizer, create_tokenizer
import logging
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

try:
    from comprehensive_reward_system import (
        create_comprehensive_system, 
        ComprehensiveMoleculeValidator,
        EnhancedRewardAwareLoss
    )
    COMPREHENSIVE_REWARDS_AVAILABLE = True
    print("Comprehensive reward system loaded successfully")
except ImportError as e:
    COMPREHENSIVE_REWARDS_AVAILABLE = False
    print(f"Comprehensive reward system not available: {e}")

try:
    from validity_constraints import (
        MoleculeValidator, ValidityAwareLoss, EnhancedNMRToSMILES,
        ValidatedSMILESGenerator)
    VALIDITY_AVAILABLE = True
except ImportError:
    VALIDITY_AVAILABLE = False
    logger.warning("Validity constraints module not found. Running without validity checking.")


warnings.filterwarnings('ignore')


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self, save_dir='metrics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_exact_match': [],
            'val_exact_match': [],
            'learning_rate': [],
            'epoch': []
        }
        
        self.predictions_history = []
        
    def update(self, epoch, train_loss=None, val_loss=None, 
               train_accuracy=None, val_accuracy=None,
               train_exact_match=None, val_exact_match=None,
               learning_rate=None, predictions=None):
        """Update metrics for current epoch"""
        self.metrics['epoch'].append(epoch)
        
        if train_loss is not None:
            self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        if train_accuracy is not None:
            self.metrics['train_accuracy'].append(train_accuracy)
        if val_accuracy is not None:
            self.metrics['val_accuracy'].append(val_accuracy)
        if train_exact_match is not None:
            self.metrics['train_exact_match'].append(train_exact_match)
        if val_exact_match is not None:
            self.metrics['val_exact_match'].append(val_exact_match)
        if learning_rate is not None:
            self.metrics['learning_rate'].append(learning_rate)
        
        if predictions is not None:
            self.predictions_history.append({
                'epoch': epoch,
                'predictions': predictions
            })
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        ax = axes[0, 0]
        if self.metrics['train_loss']:
            ax.plot(self.metrics['epoch'][:len(self.metrics['train_loss'])], 
                   self.metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.metrics['val_loss']:
            ax.plot(self.metrics['epoch'][:len(self.metrics['val_loss'])], 
                   self.metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax = axes[0, 1]
        if self.metrics['train_accuracy']:
            ax.plot(self.metrics['epoch'][:len(self.metrics['train_accuracy'])], 
                   self.metrics['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        if self.metrics['val_accuracy']:
            ax.plot(self.metrics['epoch'][:len(self.metrics['val_accuracy'])], 
                   self.metrics['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Token Accuracy (%)', fontsize=12)
        ax.set_title('Token-level Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Exact match accuracy
        ax = axes[1, 0]
        if self.metrics['train_exact_match']:
            ax.plot(self.metrics['epoch'][:len(self.metrics['train_exact_match'])], 
                   self.metrics['train_exact_match'], 'b-', label='Train Exact Match', linewidth=2)
        if self.metrics['val_exact_match']:
            ax.plot(self.metrics['epoch'][:len(self.metrics['val_exact_match'])], 
                   self.metrics['val_exact_match'], 'r-', label='Val Exact Match', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Exact Match Accuracy (%)', fontsize=12)
        ax.set_title('SMILES Exact Match Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 1]
        if self.metrics['learning_rate']:
            ax.plot(self.metrics['epoch'][:len(self.metrics['learning_rate'])], 
                   self.metrics['learning_rate'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics as CSV for later analysis
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(self.save_dir / 'training_metrics.csv', index=False)
    
    def plot_roc_curve(self, y_true, y_scores, epoch, dataset_name='Validation'):
        """Plot ROC curve for multi-class prediction"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # For token-level ROC, we need to handle this differently
        # This is a simplified version - you might want to enhance this
        # by calculating ROC for each token position or for the entire sequence
        
        # Convert to binary classification problem (correct token vs incorrect)
        fpr, tpr, _ = roc_curve(y_true.flatten(), y_scores.flatten())
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {dataset_name} Set (Epoch {epoch})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'roc_curve_{dataset_name.lower()}_epoch_{epoch}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return roc_auc
    
    def plot_confusion_matrix(self, predictions, epoch, dataset_name='Test'):
        """Plot confusion matrix for exact match predictions"""
        correct = []
        predicted = []
        
        for pred in predictions:
            correct.append(1 if pred['predicted'] == pred['true'] else 0)
            predicted.append(1 if pred['predicted'] == pred['true'] else 0)
        
        # This is simplified - you might want a more detailed confusion matrix
        cm = confusion_matrix([1]*len(correct), correct)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Incorrect', 'Correct'],
                   yticklabels=['Predictions'])
        
        ax.set_title(f'Exact Match Results - {dataset_name} Set (Epoch {epoch})', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'confusion_matrix_{dataset_name.lower()}_epoch_{epoch}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_summary_report(self):
        """Save a summary report of the training"""
        report = {
            'final_train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
            'final_val_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
            'final_train_accuracy': self.metrics['train_accuracy'][-1] if self.metrics['train_accuracy'] else None,
            'final_val_accuracy': self.metrics['val_accuracy'][-1] if self.metrics['val_accuracy'] else None,
            'final_train_exact_match': self.metrics['train_exact_match'][-1] if self.metrics['train_exact_match'] else None,
            'final_val_exact_match': self.metrics['val_exact_match'][-1] if self.metrics['val_exact_match'] else None,
            'best_val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else None,
            'best_val_accuracy': max(self.metrics['val_accuracy']) if self.metrics['val_accuracy'] else None,
            'best_val_exact_match': max(self.metrics['val_exact_match']) if self.metrics['val_exact_match'] else None,
            'total_epochs': len(self.metrics['epoch'])
        }
        
        with open(self.save_dir / 'training_summary.json', 'w') as f:
            json.dump(report, f, indent=2)

class NMRDataParser:
    """Parse NMR data from .nmredata files"""
    
    def __init__(self):
        self.h_pattern = re.compile(r'([\d.]+|NULL),\s*(\w+|null),.*,\s*([-\d]+)')
        self.c_pattern = re.compile(r'([\d.]+|NULL),\s*(\w+|null),\s*([-\d]+)')
    
    def parse_file(self, filepath: str) -> Optional[Dict]:
        """Parse a single .nmredata file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = {}
            
            # Extract NP_MRD_ID
            id_match = re.search(r'>  <NP_MRD_ID>\n(.+)', content)
            if id_match:
                data['id'] = id_match.group(1).strip()
            
            # Extract canonical SMILES
            smiles_match = re.search(r'>  <Canonical_SMILES>\n(.+)', content)
            if smiles_match:
                data['canonical_smiles'] = smiles_match.group(1).strip()
            
            # Extract atom counts
            h_atoms_match = re.search(r'H_atoms: (\d+)', content)
            c_atoms_match = re.search(r'C_atoms: (\d+)', content)
            if h_atoms_match:
                data['h_atoms'] = int(h_atoms_match.group(1))
            if c_atoms_match:
                data['c_atoms'] = int(c_atoms_match.group(1))
            
            # Extract 1H NMR data
            h_nmr_match = re.search(r'>  <NMREDATA_1D_1H>\n([\s\S]*?)(?=\n>|\n\$\$\$\$)', content)
            if h_nmr_match:
                h_peaks = self._parse_peaks(h_nmr_match.group(1), self.h_pattern)
                data['h_peaks'] = h_peaks
            
            # Extract 13C NMR data
            c_nmr_match = re.search(r'>  <NMREDATA_1D_13C>\n([\s\S]*?)(?=\n>|\n\$\$\$\$)', content)
            if c_nmr_match:
                c_peaks = self._parse_peaks(c_nmr_match.group(1), self.c_pattern)
                data['c_peaks'] = c_peaks
            
            if all(key in data for key in ['canonical_smiles', 'h_peaks', 'c_peaks', 'h_atoms', 'c_atoms']):
                return data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None
    
    def _parse_peaks(self, peak_text: str, pattern) -> List[Dict]:
        """Parse individual peaks from NMR data"""
        peaks = []
        for line in peak_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            match = pattern.match(line)
            if match:
                shift = match.group(1)
                multiplicity = match.group(2)
                atom_num = int(match.group(3))
                
                if shift == 'NULL' or shift == 'null' or atom_num == -1:
                    peaks.append({
                        'shift': None,
                        'multiplicity': 'null',
                        'atom_num': -1,
                        'is_padding': True
                    })
                else:
                    try:
                        shift_val = float(shift)
                    except:
                        shift_val = None
                        
                    peaks.append({
                        'shift': shift_val,
                        'multiplicity': multiplicity if multiplicity else 'unknown',
                        'atom_num': atom_num,
                        'is_padding': False
                    })
                
        return peaks

class NMRFeatureExtractor:
    """Extract and normalize features from NMR data"""
    
    def __init__(self, max_h_peaks=150, max_c_peaks=100):
        self.max_h_peaks = max_h_peaks
        self.max_c_peaks = max_c_peaks
        self.h_scaler = StandardScaler()
        self.c_scaler = StandardScaler()
        self.multiplicity_map = {
            's': 1, 'd': 2, 't': 3, 'q': 4, 'p': 5, 
            'm': 6, 'dd': 7, 'dt': 8, 'td': 9, 'dq': 10,
            'ddd': 11, 'dtd': 12, 'tdd': 13, 'qtd': 14,
            'null': 0, 'unknown': 0
        }
    
    def fit(self, data_list: List[Dict]):
        """Fit scalers on training data"""
        all_h_shifts = []
        all_c_shifts = []
        
        for data in data_list:
            if 'h_peaks' in data:
                for peak in data['h_peaks']:
                    if peak.get('shift') is not None and not peak.get('is_padding', False):
                        all_h_shifts.append(peak['shift'])
            if 'c_peaks' in data:
                for peak in data['c_peaks']:
                    if peak.get('shift') is not None and not peak.get('is_padding', False):
                        all_c_shifts.append(peak['shift'])
        
        if all_h_shifts:
            self.h_scaler.fit(np.array(all_h_shifts).reshape(-1, 1))
        if all_c_shifts:
            self.c_scaler.fit(np.array(all_c_shifts).reshape(-1, 1))
    
    def extract_features(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Extract features from NMR data"""
        h_features = self._process_nmr_peaks(
            data.get('h_peaks', []), 
            self.h_scaler, 
            self.max_h_peaks
        )
        
        c_features = self._process_nmr_peaks(
            data.get('c_peaks', []), 
            self.c_scaler, 
            self.max_c_peaks
        )
        
        h_atoms = data.get('h_atoms', 0) or 0
        c_atoms = data.get('c_atoms', 0) or 0
        h_real_peaks = len([p for p in data.get('h_peaks', []) if not p.get('is_padding', False)])
        c_real_peaks = len([p for p in data.get('c_peaks', []) if not p.get('is_padding', False)])
        
        global_features = torch.tensor([
            float(h_atoms),
            float(c_atoms),
            float(h_real_peaks),
            float(c_real_peaks)
        ], dtype=torch.float32)
        
        return {
            'h_features': h_features,
            'c_features': c_features,
            'global_features': global_features
        }
    
    def _process_nmr_peaks(self, peaks: List[Dict], scaler, max_peaks: int) -> torch.Tensor:
        """Process NMR peaks into feature tensor"""
        features = []
        
        for peak in peaks[:max_peaks]:
            if peak['is_padding']:
                features.append([0.0, 0.0, 0.0, 0.0])
            else:
                if peak['shift'] is not None and hasattr(scaler, 'n_features_in_'):
                    try:
                        shift_norm = scaler.transform([[peak['shift']]])[0][0]
                    except:
                        shift_norm = 0.0
                else:
                    shift_norm = 0.0
                
                mult = peak.get('multiplicity', 'unknown')
                if mult is None:
                    mult = 'unknown'
                mult_encoded = self.multiplicity_map.get(mult.lower() if isinstance(mult, str) else 'unknown', 0) / 14.0
                
                atom_num = peak.get('atom_num', 0)
                if atom_num is None or atom_num < 0:
                    atom_num = 0
                atom_norm = min(atom_num / 200.0, 1.0)
                
                features.append([shift_norm, mult_encoded, atom_norm, 1.0])
        
        while len(features) < max_peaks:
            features.append([0.0, 0.0, 0.0, 0.0])
        
        return torch.tensor(features[:max_peaks], dtype=torch.float32)

class NMRDataset(Dataset):
    """PyTorch Dataset for NMR data"""
    
    def __init__(self, data_list: List[Dict], feature_extractor: NMRFeatureExtractor, 
                 tokenizer: RobertaTokenizer, max_smiles_length: int = 256):
        self.data = data_list
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_smiles_length = max_smiles_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            data = self.data[idx]
            
            features = self.feature_extractor.extract_features(data)
            
            smiles = data.get('canonical_smiles', '')
            if not smiles:
                smiles = 'C'
            
            smiles_tokens = self.tokenizer(
                smiles,
                padding='max_length',
                truncation=True,
                max_length=self.max_smiles_length,
                return_tensors='pt'
            )
            
            # Include H and C atom counts
            h_atoms = data.get('h_atoms', 0)
            c_atoms = data.get('c_atoms', 0)
            
            return {
                'h_features': features['h_features'],
                'c_features': features['c_features'],
                'global_features': features['global_features'],
                'smiles_ids': smiles_tokens['input_ids'].squeeze(),
                'smiles_mask': smiles_tokens['attention_mask'].squeeze(),
                'id': data.get('id', f'unknown_{idx}'),
                'h_atoms': h_atoms,  # Add H count
                'c_atoms': c_atoms   # Add C count
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            raise

class NMREncoder(nn.Module):
    """Encode NMR data into embeddings using pretrained ChemBERTa"""
    
    def __init__(self, h_input_dim=4, c_input_dim=4, hidden_dim=768, num_layers=3, 
                 use_pretrained_chemberta=True, freeze_layers=6):
        super().__init__()
        
        # IMPORTANT: ChemBERTa requires exactly 768 dimensions
        if use_pretrained_chemberta:
            self.hidden_dim = 768  # Force to 768 for ChemBERTa
        else:
            self.hidden_dim = hidden_dim
            
        self.use_pretrained = use_pretrained_chemberta
        
        # Project NMR features to the correct dimension
        self.h_projection = nn.Linear(h_input_dim, self.hidden_dim)
        self.c_projection = nn.Linear(c_input_dim, self.hidden_dim)
        
        if use_pretrained_chemberta:
            # Load pretrained ChemBERTa model
            logger.info("Loading pretrained ChemBERTa model...")
            try:
                # Load the config first to check dimensions
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
                logger.info(f"ChemBERTa config - hidden_size: {config.hidden_size}")
                
                # Load the model
                self.chemberta = RobertaModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
                
                # Verify the hidden size
                actual_hidden_size = self.chemberta.config.hidden_size
                if actual_hidden_size != 768:
                    logger.warning(f"ChemBERTa hidden size is {actual_hidden_size}, not 768!")
                    self.hidden_dim = actual_hidden_size
                    # Recreate projections with correct size
                    self.h_projection = nn.Linear(h_input_dim, actual_hidden_size)
                    self.c_projection = nn.Linear(c_input_dim, actual_hidden_size)
                
                # Freeze lower layers of ChemBERTa
                if freeze_layers > 0:
                    # Freeze embeddings
                    for param in self.chemberta.embeddings.parameters():
                        param.requires_grad = False
                    
                    # Freeze specified number of layers
                    for layer_idx in range(min(freeze_layers, len(self.chemberta.encoder.layer))):
                        for param in self.chemberta.encoder.layer[layer_idx].parameters():
                            param.requires_grad = False
                    
                    logger.info(f"ChemBERTa loaded. Froze embeddings and first {freeze_layers} layers.")
                    logger.info(f"Fine-tuning remaining {len(self.chemberta.encoder.layer) - freeze_layers} layers.")
                else:
                    logger.info("ChemBERTa loaded. All layers will be fine-tuned.")
                    
            except Exception as e:
                logger.error(f"Failed to load ChemBERTa: {e}")
                logger.warning("Falling back to custom transformers")
                self.use_pretrained = False
        
        if not self.use_pretrained:
            # Fallback to custom transformers
            self.h_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8 if self.hidden_dim == 256 else 12,  # Adjust heads based on dim
                    dim_feedforward=512 if self.hidden_dim == 256 else 2048,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=num_layers
            )
            
            self.c_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8 if self.hidden_dim == 256 else 12,
                    dim_feedforward=512 if self.hidden_dim == 256 else 2048,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=num_layers
            )
        
        # Global projection uses the determined hidden dim
        self.global_projection = nn.Linear(4, self.hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def forward(self, h_features, c_features, global_features):
        batch_size = h_features.size(0)
        seq_len_h = h_features.size(1)
        seq_len_c = c_features.size(1)
        
        # Project NMR features to correct dimension
        h_emb = self.h_projection(h_features)  # [batch, seq_len, hidden_dim]
        c_emb = self.c_projection(c_features)  # [batch, seq_len, hidden_dim]
        
        if self.use_pretrained:
            # Use ChemBERTa for encoding
            # Create attention masks (1 for real data, 0 for padding)
            h_mask = (h_features.sum(dim=-1) != 0).long()  # [batch, seq_len]
            c_mask = (c_features.sum(dim=-1) != 0).long()  # [batch, seq_len]
            
            # ChemBERTa expects inputs_embeds instead of input_ids
            # We pass our projected embeddings directly
            h_outputs = self.chemberta(
                inputs_embeds=h_emb,
                attention_mask=h_mask,
                return_dict=True
            )
            h_encoded = h_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            
            c_outputs = self.chemberta(
                inputs_embeds=c_emb,
                attention_mask=c_mask,
                return_dict=True
            )
            c_encoded = c_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            
            # Pool the sequences (using masked mean)
            h_mask_expanded = h_mask.unsqueeze(-1).expand_as(h_encoded).float()
            h_sum = (h_encoded * h_mask_expanded).sum(dim=1)
            h_pooled = h_sum / h_mask_expanded.sum(dim=1).clamp(min=1e-9)
            
            c_mask_expanded = c_mask.unsqueeze(-1).expand_as(c_encoded).float()
            c_sum = (c_encoded * c_mask_expanded).sum(dim=1)
            c_pooled = c_sum / c_mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            # Use custom transformers
            # Create padding mask for transformers
            h_mask = (h_features.sum(dim=-1) != 0).float()  # [batch, seq_len]
            c_mask = (c_features.sum(dim=-1) != 0).float()  # [batch, seq_len]
            
            # Apply transformers
            h_encoded = self.h_transformer(h_emb, src_key_padding_mask=~h_mask.bool())
            c_encoded = self.c_transformer(c_emb, src_key_padding_mask=~c_mask.bool())
            
            # Pool with masking
            h_mask_expanded = h_mask.unsqueeze(-1).expand_as(h_encoded)
            h_pooled = (h_encoded * h_mask_expanded).sum(dim=1) / h_mask_expanded.sum(dim=1).clamp(min=1e-9)
            
            c_mask_expanded = c_mask.unsqueeze(-1).expand_as(c_encoded)
            c_pooled = (c_encoded * c_mask_expanded).sum(dim=1) / c_mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        # Global features
        global_emb = self.global_projection(global_features)
        
        # Combine all features
        combined = torch.cat([h_pooled, c_pooled, global_emb], dim=-1)
        fused = self.fusion(combined)
        
        return fused


class NMRToSMILES(nn.Module):
    """Complete model for NMR to SMILES prediction using ChemBERTa encoder + Custom tokenizer decoder"""
    
    def __init__(self, nmr_encoder: NMREncoder, vocab_size: int, 
                 hidden_dim: int = 768, num_decoder_layers: int = 6):
        super().__init__()
        self.nmr_encoder = nmr_encoder
        self.hidden_dim = hidden_dim  # Should be 768 for ChemBERTa
        
        # IMPORTANT: Using CUSTOM tokenizer vocabulary for decoder
        logger.info(f"Initializing decoder with CUSTOM vocabulary size: {vocab_size}")
        logger.info("ChemBERTa is used ONLY for encoding NMR features")
        logger.info("SMILES generation uses CUSTOM tokenizer vocabulary")
        
        # Standard transformer decoder with CUSTOM vocabulary
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=12,  # 12 heads for 768 dim (768/12 = 64 dim per head)
                dim_feedforward=3072,  # Standard for BERT/RoBERTa
                dropout=0.1,
                activation='gelu',
                batch_first=True  # Important for consistency
            ),
            num_layers=num_decoder_layers
        )
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.1)
        self.vocab_size = vocab_size
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training"""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.constant_(self.output_projection.bias, 0)
    
    def forward(self, h_features, c_features, global_features, 
                target_ids=None, target_mask=None, tokenizer=None):
        # Encode NMR features using ChemBERTa encoder
        nmr_encoding = self.nmr_encoder(h_features, c_features, global_features)
        nmr_encoding = nmr_encoding.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        if target_ids is not None:
            # Training mode
            seq_len = target_ids.size(1)
            batch_size = target_ids.size(0)
            
            # Create position ids
            positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0).expand(batch_size, -1)
            
            # Embed tokens and positions
            token_emb = self.token_embedding(target_ids)
            pos_emb = self.position_embedding(positions)
            target_emb = self.dropout(token_emb + pos_emb)
            
            # Create causal mask
            causal_mask = self._generate_square_subsequent_mask(seq_len).to(target_ids.device)
            
            # Prepare padding mask
            tgt_key_padding_mask = None
            if target_mask is not None:
                tgt_key_padding_mask = ~target_mask.bool()
            
            # Decode
            decoded = self.decoder(
                target_emb,
                nmr_encoding,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            # Project to vocabulary
            output = self.output_projection(decoded)
            
            return output
        else:
            # Generation mode
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided for generation mode")
            return self._generate(nmr_encoding, tokenizer)
    
    def _generate(self, nmr_encoding, tokenizer, max_length=256, temperature=0.8):
        """Clean generation method without debug spam"""
        device = nmr_encoding.device
        batch_size = nmr_encoding.size(0)
        
        # Get token IDs (no debug prints)
        bos_id = tokenizer.sos_token_id  
        eos_id = tokenizer.eos_token_id  
        pad_id = tokenizer.pad_token_id  
        unk_id = tokenizer.unk_token_id  
        
        # Start with SOS token
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_length - 1):
            if finished.all():
                break
                
            seq_len = generated.size(1)
            
            # Create positions
            max_positions = self.position_embedding.num_embeddings - 1
            positions = torch.arange(seq_len, device=device).clamp(0, max_positions)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            
            try:
                # Forward pass
                token_emb = self.token_embedding(generated)
                pos_emb = self.position_embedding(positions)
                target_emb = self.dropout(token_emb + pos_emb)
                
                # Causal mask
                causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)
                
                # Decode
                decoded = self.decoder(target_emb, nmr_encoding, tgt_mask=causal_mask)
                
                # Get logits
                logits = self.output_projection(decoded[:, -1, :])
                
                # Prevent bad tokens
                logits[:, pad_id] = -float('inf')
                logits[:, unk_id] = -float('inf')
                
                # Temperature scaling
                if temperature > 0:
                    logits = logits / max(temperature, 0.1)
                
                # Top-k sampling
                vocab_size = logits.size(-1)
                k = min(max(10, vocab_size // 4), 50)
                
                if k < vocab_size:
                    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
                    logits_filtered = torch.full_like(logits, -float('inf'))
                    logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
                    logits = logits_filtered
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    # Fallback to common tokens
                    common_tokens = [5, 10, 13, 26, 30]  # C, c, O, N, S
                    next_token = torch.tensor([[common_tokens[step % len(common_tokens)]]], 
                                            dtype=torch.long, device=device)
                else:
                    if temperature > 0:
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Update sequences
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, pad_id),
                    next_token
                )
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                finished = finished | (next_token.squeeze(-1) == eos_id)
                
            except Exception as e:
                # Silent fallback - just stop generation
                break
        
        return generated


    # ALSO ADD THIS DEBUG METHOD TO YOUR MODEL
    def debug_model_tokenizer_compatibility(self, tokenizer):
        """Debug the model-tokenizer compatibility"""
        print("\n" + "="*60)
        print("MODEL-TOKENIZER COMPATIBILITY CHECK")
        print("="*60)
        
        print(f"Model vocab size: {self.vocab_size}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        
        if self.vocab_size != tokenizer.vocab_size:
            print("❌ CRITICAL: Vocab size mismatch!")
            return False
        
        # Test embedding bounds
        print(f"Token embedding: {self.token_embedding.num_embeddings} tokens")
        print(f"Position embedding: {self.position_embedding.num_embeddings} positions")
        print(f"Output projection: {self.output_projection.out_features} classes")
        
        # Test special tokens
        special_ids = [tokenizer.pad_token_id, tokenizer.sos_token_id, 
                    tokenizer.eos_token_id, tokenizer.unk_token_id]
        print(f"Special token IDs: {special_ids}")
        
        max_special = max(special_ids)
        if max_special >= self.vocab_size:
            print(f"❌ CRITICAL: Special token ID {max_special} >= vocab size {self.vocab_size}")
            return False
        
        # Test a forward pass with special tokens
        try:
            device = next(self.parameters()).device
            test_ids = torch.tensor([special_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                # Test embeddings
                token_emb = self.token_embedding(test_ids)
                print(f"✅ Token embedding test passed: {token_emb.shape}")
                
                # Test output projection
                logits = self.output_projection(token_emb)
                print(f"✅ Output projection test passed: {logits.shape}")
                
                # Test softmax
                probs = torch.softmax(logits, dim=-1)
                print(f"✅ Softmax test passed: {probs.shape}, sum={probs.sum():.3f}")
            
            print("✅ All compatibility tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Compatibility test failed: {e}")
            return False

    
    def _generate_valid_forced(self, nmr_encoding, tokenizer, max_length=256, temperature=0.8, 
                              max_attempts=50, beam_size=5):
        """Generate valid SMILES with multiple attempts"""
        device = nmr_encoding.device
        batch_size = nmr_encoding.size(0)
        
        # Try simple generation first
        for attempt in range(min(5, max_attempts)):
            generated = self._generate(nmr_encoding, tokenizer, max_length, temperature)
            smiles = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
            
            # Try to validate
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return generated
            except:
                pass
        
        # If simple generation fails, try beam search
        best_valid_smiles = None
        best_score = float('-inf')
        
        # Get special token IDs from custom tokenizer
        if hasattr(tokenizer, 'pad_token_id'):
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = tokenizer.token_to_id.get('<pad>', 0)
            
        if hasattr(tokenizer, 'eos_token_id'):
            eos_token_id = tokenizer.eos_token_id
        else:
            eos_token_id = tokenizer.token_to_id.get('<eos>', 2)
            
        if hasattr(tokenizer, 'bos_token_id'):
            bos_token_id = tokenizer.bos_token_id
        else:
            bos_token_id = tokenizer.token_to_id.get('<bos>', 1)
        
        # Beam search implementation
        for beam_idx in range(min(3, beam_size)):
            beams = [{
                'tokens': torch.tensor([bos_token_id], dtype=torch.long, device=device),
                'score': 0.0,
                'complete': False
            }]
            
            for step in range(min(50, max_length)):
                new_beams = []
                
                for beam in beams[:5]:  # Limit beams per step
                    if beam['complete'] or len(beam['tokens']) >= max_length - 1:
                        new_beams.append(beam)
                        continue
                    
                    # Get next token probabilities
                    try:
                        seq_len = beam['tokens'].size(0)
                        positions = torch.arange(seq_len, device=device).unsqueeze(0)
                        
                        token_emb = self.token_embedding(beam['tokens'].unsqueeze(0))
                        pos_emb = self.position_embedding(positions)
                        target_emb = self.dropout(token_emb + pos_emb)
                        
                        decoded = self.decoder(target_emb, nmr_encoding)
                        
                        logits = self.output_projection(decoded[0, -1, :]) / temperature
                        probs = F.softmax(logits, dim=-1)
                        
                        # Sample top k tokens
                        k = min(5, probs.size(-1))
                        top_probs, top_indices = torch.topk(probs, k)
                        
                        for idx in range(k):
                            token_id = top_indices[idx].item()
                            token_prob = top_probs[idx].item()
                            
                            # Skip padding token
                            if token_id == pad_token_id:
                                continue
                            
                            new_tokens = torch.cat([beam['tokens'], torch.tensor([token_id], device=device)])
                            
                            # Check if complete
                            is_complete = (token_id == eos_token_id)
                            
                            new_beams.append({
                                'tokens': new_tokens,
                                'score': beam['score'] + np.log(token_prob),
                                'complete': is_complete
                            })
                    
                    except Exception as e:
                        logger.debug(f"Error in beam search: {e}")
                        continue
                
                # Keep top beams
                beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:beam_size]
                
                if not beams or all(b['complete'] for b in beams):
                    break
            
            # Check validity of completed beams
            for beam in beams:
                if beam['complete']:
                    smiles = tokenizer.decode(beam['tokens'].tolist(), skip_special_tokens=True)
                    try:
                        from rdkit import Chem
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None and beam['score'] > best_score:
                            best_score = beam['score']
                            best_valid_smiles = beam['tokens'].unsqueeze(0)
                    except:
                        pass
        
        # Return best valid SMILES or last attempt
        if best_valid_smiles is not None:
            return best_valid_smiles
        
        # Final fallback
        return self._generate(nmr_encoding, tokenizer, max_length, temperature * 0.5)
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal attention mask"""
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.float32), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _is_valid_partial_smiles(self, partial_smiles):
        """Quick checks for partial SMILES validity"""
        if not partial_smiles:
            return True
        
        # Check balanced parentheses
        if partial_smiles.count(')') > partial_smiles.count('('):
            return False
        if partial_smiles.count(']') > partial_smiles.count('['):
            return False
        
        return True

class ComprehensiveTrainer:
    """Trainer with comprehensive reward/penalty system for proper molecule generation"""
    
    def __init__(self, model, tokenizer, metrics_tracker, device='cuda', use_comprehensive_rewards=False):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.metrics_tracker = metrics_tracker
        self.use_comprehensive = False
        self.validator = None
        
        # Initialize comprehensive reward/penalty system
        if use_comprehensive_rewards:
            if 'COMPREHENSIVE_REWARDS_AVAILABLE' in globals() and COMPREHENSIVE_REWARDS_AVAILABLE:
                try:
                    from comprehensive_reward_system import create_comprehensive_system
                    self.validator, self.criterion = create_comprehensive_system(model, tokenizer, device)
                    self.use_comprehensive = True
                    logger.info("Using comprehensive reward/penalty system for proper molecule generation")
                    logger.info("Reward priorities: H/C matching > Validity > Structure > Token accuracy")
                    logger.info("Token accuracy is fully integrated with synergy bonuses")
                except Exception as e:
                    logger.warning(f"Failed to initialize comprehensive reward system: {e}")
                    logger.warning("Falling back to standard loss function")
                    self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                    self.use_comprehensive = False
            else:
                try:
                    from comprehensive_reward_system import create_comprehensive_system
                    self.validator, self.criterion = create_comprehensive_system(model, tokenizer, device)
                    self.use_comprehensive = True
                    logger.info("Loaded comprehensive reward system successfully")
                except ImportError:
                    logger.warning("Comprehensive rewards requested but module not available")
                    logger.warning("Please ensure comprehensive_reward_system.py is in the project directory")
                    self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                    self.use_comprehensive = False
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            self.use_comprehensive = False
    
    def _generate_predictions_clean(self, h_features, c_features, global_features, sample_size=3):
        """Clean prediction generation - only generate for a few samples"""
        predictions = []
        
        # Only generate for first few samples to verify it works
        for i in range(min(sample_size, len(h_features))):
            try:
                with torch.no_grad():
                    generated = self.model(
                        h_features[i:i+1], 
                        c_features[i:i+1], 
                        global_features[i:i+1], 
                        tokenizer=self.tokenizer
                    )
                    
                    if generated is None or generated.size(0) == 0:
                        pred_smiles = ""
                    else:
                        token_ids = generated[0].tolist()
                        pred_smiles = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                        
            except Exception:
                pred_smiles = ""
            
            predictions.append(pred_smiles)
        
        # Fill rest with empty strings for speed
        while len(predictions) < len(h_features):
            predictions.append("")
        
        return predictions
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        exact_matches = 0
        total_samples = 0
        
        # Comprehensive metrics tracking
        comprehensive_metrics = {
            'empty_predictions': 0,
            'invalid_smiles': 0,
            'invalid_brackets': 0,
            'invalid_valency': 0,
            'perfect_hc_matches': 0,
            'good_hc_matches': 0,
            'valid_molecules': 0,
            'total_reward': 0,
            'total_penalty': 0,
            'avg_score': 0,
            'synergy_bonuses': 0,
            'token_accuracy_scores': []
        }
        
        progress_bar = tqdm(dataloader, desc='Training')
        for batch_idx, batch in enumerate(progress_bar):
            h_features = batch['h_features'].to(self.device)
            c_features = batch['c_features'].to(self.device)
            global_features = batch['global_features'].to(self.device)
            smiles_ids = batch['smiles_ids'].to(self.device)
            smiles_mask = batch['smiles_mask'].to(self.device)
            
            # Get expected atom counts from global features
            expected_h = global_features[:, 0].cpu().numpy().astype(int)
            expected_c = global_features[:, 1].cpu().numpy().astype(int)
            
            outputs = self.model(
                h_features, c_features, global_features,
                smiles_ids[:, :-1], smiles_mask[:, :-1]
            )
            
            # Generate predictions for comprehensive evaluation
            predictions = None
            nmr_data = None
            
            if self.use_comprehensive:
                if batch_idx == 0:  # Only first batch for verification
                    print("Generating sample predictions...")
                    predictions = self._generate_predictions_clean(
                        h_features, c_features, global_features, sample_size=3
                    )
                    # Quick check - print first prediction only
                    if predictions[0]:
                        print(f"Sample prediction: {predictions[0][:50]}...")
                        print("✅ Generation working!")
                    else:
                        print("⚠️ Empty prediction - check model")
                else:
                    # Skip generation for speed
                    predictions = [""] * len(h_features)
                
                nmr_data = {
                    'h_atoms': expected_h,
                    'c_atoms': expected_c
                }
                
            # Calculate loss with comprehensive rewards/penalties
            if self.use_comprehensive and predictions is not None:
                loss_output = self.criterion(
                    outputs,
                    smiles_ids[:, 1:],
                    predictions=predictions,
                    nmr_data=nmr_data
                )
                
                if isinstance(loss_output, tuple):
                    loss, loss_components = loss_output
                    
                    # Update comprehensive metrics
                    batch_stats = loss_components.get('batch_stats', {})
                    for key in comprehensive_metrics:
                        if key in batch_stats:
                            if isinstance(batch_stats[key], list):
                                comprehensive_metrics[key].extend(batch_stats[key])
                            else:
                                comprehensive_metrics[key] += batch_stats[key]
                    
                    # Track average scores
                    comprehensive_metrics['avg_score'] += loss_components.get('molecule_reward_penalty_sum', 0)
                else:
                    loss = loss_output
            else:
                # Standard loss
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    smiles_ids[:, 1:].reshape(-1)
                )
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(
                outputs, 
                smiles_ids[:, 1:], 
                smiles_mask[:, 1:]
            )
            
            # Check for exact matches
            with torch.no_grad():
                if predictions is None:
                    predictions = self._generate_predictions_safe(
                        h_features, c_features, global_features, debug_level=0
                    )
                
                for i, pred_smiles in enumerate(predictions):
                    true_smiles = self.tokenizer.decode(smiles_ids[i], skip_special_tokens=True)
                    if pred_smiles == true_smiles:
                        exact_matches += 1
                
                total_samples += len(predictions)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_accuracy += accuracy
            
            # Update progress bar with comprehensive metrics
            current_lr = scheduler.get_last_lr()[0]
            progress_dict = {
                'loss': loss.item(),
                'acc': f'{accuracy:.2f}%',
                'lr': f'{current_lr:.2e}'
            }
            
            # Add comprehensive metrics if available
            if self.use_comprehensive and total_samples > 0:
                empty_rate = (comprehensive_metrics['empty_predictions'] / total_samples) * 100
                valid_rate = (comprehensive_metrics['valid_molecules'] / total_samples) * 100
                perfect_hc_rate = (comprehensive_metrics['perfect_hc_matches'] / total_samples) * 100
                
                # Get metrics from loss components
                if isinstance(loss_output, tuple) and len(loss_output) > 1:
                    loss_components = loss_output[1]
                    token_acc = loss_components.get('avg_token_accuracy', 0)
                    synergy_rate = loss_components.get('synergy_rate', 0)
                    
                    progress_dict.update({
                        'empty': f'{empty_rate:.1f}%',
                        'valid': f'{valid_rate:.1f}%',
                        'H/C': f'{perfect_hc_rate:.1f}%',
                        'tok_acc': f'{token_acc:.1f}%',
                        'synergy': f'{synergy_rate:.1f}%'
                    })
            
            progress_bar.set_postfix(progress_dict)
        
        # Calculate final metrics
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        exact_match_rate = (exact_matches / total_samples) * 100 if total_samples > 0 else 0
        
        # Comprehensive metrics summary
        enhanced_metrics = {}
        if self.use_comprehensive and total_samples > 0:
            avg_token_accuracy = (sum(comprehensive_metrics['token_accuracy_scores']) / 
                                len(comprehensive_metrics['token_accuracy_scores'])) if comprehensive_metrics['token_accuracy_scores'] else 0
            
            enhanced_metrics = {
                'empty_prediction_rate': (comprehensive_metrics['empty_predictions'] / total_samples) * 100,
                'invalid_smiles_rate': (comprehensive_metrics['invalid_smiles'] / total_samples) * 100,
                'invalid_brackets_rate': (comprehensive_metrics['invalid_brackets'] / total_samples) * 100,
                'invalid_valency_rate': (comprehensive_metrics['invalid_valency'] / total_samples) * 100,
                'perfect_hc_match_rate': (comprehensive_metrics['perfect_hc_matches'] / total_samples) * 100,
                'good_hc_match_rate': (comprehensive_metrics['good_hc_matches'] / total_samples) * 100,
                'validity_rate': (comprehensive_metrics['valid_molecules'] / total_samples) * 100,
                'avg_reward_score': comprehensive_metrics['avg_score'] / len(dataloader),
                'avg_token_accuracy': avg_token_accuracy,
                'synergy_bonus_rate': (comprehensive_metrics['synergy_bonuses'] / total_samples) * 100,
            }
            
            logger.info(f"  Comprehensive Training Metrics:")
            logger.info(f"    Empty Predictions: {enhanced_metrics['empty_prediction_rate']:.1f}%")
            logger.info(f"    Invalid SMILES: {enhanced_metrics['invalid_smiles_rate']:.1f}%")
            logger.info(f"    Perfect H/C Match: {enhanced_metrics['perfect_hc_match_rate']:.1f}%")
            logger.info(f"    Valid Molecules: {enhanced_metrics['validity_rate']:.1f}%")
            logger.info(f"    Average Token Accuracy: {enhanced_metrics['avg_token_accuracy']:.1f}%")
            logger.info(f"    Synergy Bonus Rate: {enhanced_metrics['synergy_bonus_rate']:.1f}%")
            logger.info(f"    Average Reward Score: {enhanced_metrics['avg_reward_score']:.2f}")
        
        if enhanced_metrics:
            return avg_loss, avg_accuracy, exact_match_rate, enhanced_metrics
        else:
            return avg_loss, avg_accuracy, exact_match_rate
    
    def calculate_accuracy(self, outputs, targets, mask=None):
        """Calculate token-level accuracy"""
        predictions = outputs.argmax(dim=-1)
        
        if mask is not None:
            correct = (predictions == targets) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        else:
            correct = predictions == targets
            accuracy = correct.float().mean()
        
        return accuracy.item() * 100
    
    def evaluate(self, dataloader, return_predictions=True, calculate_roc=False, calculate_tanimoto=True):
        """Enhanced evaluate function with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        predictions = []
        
        # Comprehensive metrics tracking
        comprehensive_metrics = {
            'empty_predictions': 0,
            'invalid_smiles': 0,
            'invalid_brackets': 0,
            'invalid_valency': 0,
            'perfect_hc_matches': 0,
            'good_hc_matches': 0,
            'valid_molecules': 0,
            'total_samples': 0,
            'total_reward': 0,
            'total_penalty': 0,
            'detailed_evaluations': [],
            'token_accuracy_scores': [],
            'synergy_bonuses': 0
        }
        
        # For ROC curves
        all_probs = []
        all_targets = []
        all_predictions = []
        
        # For Tanimoto similarity
        if calculate_tanimoto:
            try:
                from tanimoto_metrics import TanimotoCalculator
                tanimoto_calculator = TanimotoCalculator()
                generated_smiles_list = []
                true_smiles_list = []
            except ImportError:
                logger.warning("TanimotoCalculator not available")
                calculate_tanimoto = False
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                h_features = batch['h_features'].to(self.device)
                c_features = batch['c_features'].to(self.device)
                global_features = batch['global_features'].to(self.device)
                smiles_ids = batch['smiles_ids'].to(self.device)
                smiles_mask = batch['smiles_mask'].to(self.device)
                
                # Get expected atom counts
                expected_h = global_features[:, 0].cpu().numpy().astype(int)
                expected_c = global_features[:, 1].cpu().numpy().astype(int)
                
                outputs = self.model(
                    h_features, c_features, global_features,
                    smiles_ids[:, :-1], smiles_mask[:, :-1]
                )
                
                # Calculate loss (simplified for evaluation)
                if self.use_comprehensive:
                    if hasattr(self.criterion, 'base_criterion'):
                        loss = self.criterion.base_criterion(
                            outputs.reshape(-1, outputs.size(-1)),
                            smiles_ids[:, 1:].reshape(-1)
                        )
                    else:
                        loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                            outputs.reshape(-1, outputs.size(-1)),
                            smiles_ids[:, 1:].reshape(-1)
                        )
                else:
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        smiles_ids[:, 1:].reshape(-1)
                    )
                
                total_loss += loss.item()
                
                # Calculate accuracy
                accuracy = self.calculate_accuracy(
                    outputs, 
                    smiles_ids[:, 1:], 
                    smiles_mask[:, 1:]
                )
                total_accuracy += accuracy
                
                # ROC data collection
                if calculate_roc:
                    probs = torch.softmax(outputs, dim=-1)
                    batch_size, seq_len, vocab_size = probs.shape
                    
                    correct_token_probs = torch.gather(
                        probs, dim=2, index=smiles_ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)
                    
                    predictions_correct = (outputs.argmax(dim=-1) == smiles_ids[:, 1:]).float()
                    
                    all_probs.append(correct_token_probs.cpu().numpy())
                    all_targets.append(smiles_mask[:, 1:].cpu().numpy())
                    all_predictions.append(predictions_correct.cpu().numpy())
                
                # Generate predictions with comprehensive evaluation
                if return_predictions or calculate_tanimoto or self.use_comprehensive:
                    batch_predictions = self._generate_predictions_clean(
                        h_features, c_features, global_features, sample_size=len(h_features)
                    )
                    
                    # Process predictions with comprehensive evaluation
                    for i, pred_smiles in enumerate(batch_predictions):
                        true_smiles = self.tokenizer.decode(smiles_ids[i], skip_special_tokens=True)
                        
                        if return_predictions:
                            prediction_entry = {
                                'id': batch['id'][i],
                                'predicted': pred_smiles,
                                'true': true_smiles,
                                'expected_h': int(expected_h[i]),
                                'expected_c': int(expected_c[i])
                            }
                            
                            # Add comprehensive evaluation if available
                            if self.use_comprehensive and self.validator:
                                evaluation = self.validator.comprehensive_evaluation(
                                    pred_smiles, int(expected_h[i]), int(expected_c[i])
                                )
                                
                                # Calculate token accuracy for this sample
                                sample_outputs = outputs[i]
                                sample_targets = smiles_ids[i, 1:]
                                predictions_tokens = sample_outputs.argmax(dim=-1)
                                correct_tokens = (predictions_tokens == sample_targets).float()
                                sample_token_accuracy = correct_tokens.mean().item()
                                
                                # Calculate synergy bonus
                                if hasattr(self.criterion, '_calculate_synergy_bonus'):
                                    synergy_bonus = self.criterion._calculate_synergy_bonus(evaluation, sample_token_accuracy)
                                else:
                                    synergy_bonus = 0.0
                                
                                # Add all evaluation details
                                evaluation['token_accuracy'] = sample_token_accuracy
                                evaluation['synergy_bonus'] = synergy_bonus
                                evaluation['combined_score'] = evaluation['total_score'] + synergy_bonus
                                
                                prediction_entry['comprehensive_eval'] = evaluation
                                prediction_entry['token_accuracy'] = sample_token_accuracy
                                prediction_entry['synergy_bonus'] = synergy_bonus
                                prediction_entry['combined_score'] = evaluation['combined_score']
                                
                                comprehensive_metrics['detailed_evaluations'].append(evaluation)
                            
                            predictions.append(prediction_entry)
                        
                        if calculate_tanimoto:
                            generated_smiles_list.append(pred_smiles)
                            true_smiles_list.append(true_smiles)
                    
                    # Update comprehensive metrics
                    if self.use_comprehensive and self.validator:
                        for i, (pred_smiles, exp_h, exp_c) in enumerate(zip(batch_predictions, expected_h, expected_c)):
                            evaluation = self.validator.comprehensive_evaluation(pred_smiles, int(exp_h), int(exp_c))
                            
                            # Calculate token accuracy for this sample
                            sample_outputs = outputs[i]
                            sample_targets = smiles_ids[i, 1:]
                            predictions_tokens = sample_outputs.argmax(dim=-1)
                            correct_tokens = (predictions_tokens == sample_targets).float()
                            sample_token_accuracy = correct_tokens.mean().item()
                            
                            # Calculate synergy bonus
                            if hasattr(self.criterion, '_calculate_synergy_bonus'):
                                synergy_bonus = self.criterion._calculate_synergy_bonus(evaluation, sample_token_accuracy)
                                if synergy_bonus > 0:
                                    comprehensive_metrics['synergy_bonuses'] += 1
                            
                            # Update metrics
                            if 'empty_prediction' in evaluation['penalties']:
                                comprehensive_metrics['empty_predictions'] += 1
                            if 'invalid_smiles' in evaluation['penalties']:
                                comprehensive_metrics['invalid_smiles'] += 1
                            if 'invalid_brackets' in evaluation['penalties']:
                                comprehensive_metrics['invalid_brackets'] += 1
                            if 'invalid_valency' in evaluation['penalties']:
                                comprehensive_metrics['invalid_valency'] += 1
                            if 'perfect_hc_match' in evaluation['rewards']:
                                comprehensive_metrics['perfect_hc_matches'] += 1
                            elif 'good_hc_match' in evaluation['rewards']:
                                comprehensive_metrics['good_hc_matches'] += 1
                            if 'valid_molecule' in evaluation['rewards']:
                                comprehensive_metrics['valid_molecules'] += 1
                            
                            comprehensive_metrics['total_reward'] += sum(evaluation['rewards'].values())
                            comprehensive_metrics['total_penalty'] += sum(evaluation['penalties'].values())
                            comprehensive_metrics['token_accuracy_scores'].append(sample_token_accuracy)
                        
                        comprehensive_metrics['total_samples'] += len(batch_predictions)
        
        # Calculate final metrics
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        
        # Calculate exact match accuracy
        exact_matches = sum(1 for p in predictions if p['predicted'] == p['true'])
        exact_match_rate = (exact_matches / len(predictions)) * 100 if predictions else 0
        
        # Enhanced results
        results = {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'exact_match_rate': exact_match_rate,
            'predictions': predictions if return_predictions else None,
        }
        
        # Add comprehensive metrics
        if self.use_comprehensive and comprehensive_metrics['total_samples'] > 0:
            total_samples = comprehensive_metrics['total_samples']
            token_accuracy_scores = comprehensive_metrics.get('token_accuracy_scores', [])
            avg_token_accuracy = sum(token_accuracy_scores) / len(token_accuracy_scores) if token_accuracy_scores else 0
            synergy_bonuses = comprehensive_metrics.get('synergy_bonuses', 0)
            
            enhanced_metrics = {
                'empty_prediction_rate': (comprehensive_metrics['empty_predictions'] / total_samples) * 100,
                'invalid_smiles_rate': (comprehensive_metrics['invalid_smiles'] / total_samples) * 100,
                'invalid_brackets_rate': (comprehensive_metrics['invalid_brackets'] / total_samples) * 100,
                'invalid_valency_rate': (comprehensive_metrics['invalid_valency'] / total_samples) * 100,
                'perfect_hc_match_rate': (comprehensive_metrics['perfect_hc_matches'] / total_samples) * 100,
                'good_hc_match_rate': (comprehensive_metrics['good_hc_matches'] / total_samples) * 100,
                'validity_rate': (comprehensive_metrics['valid_molecules'] / total_samples) * 100,
                'avg_reward': comprehensive_metrics['total_reward'] / total_samples,
                'avg_penalty': comprehensive_metrics['total_penalty'] / total_samples,
                'avg_total_score': (comprehensive_metrics['total_reward'] + comprehensive_metrics['total_penalty']) / total_samples,
                'avg_token_accuracy': avg_token_accuracy * 100,
                'synergy_bonus_rate': (synergy_bonuses / total_samples) * 100,
            }
            
            results['comprehensive_metrics'] = enhanced_metrics
            
            logger.info(f"Comprehensive Evaluation Metrics:")
            logger.info(f"  Empty Predictions: {enhanced_metrics['empty_prediction_rate']:.1f}%")
            logger.info(f"  Invalid SMILES: {enhanced_metrics['invalid_smiles_rate']:.1f}%")
            logger.info(f"  Perfect H/C Match: {enhanced_metrics['perfect_hc_match_rate']:.1f}%")
            logger.info(f"  Valid Molecules: {enhanced_metrics['validity_rate']:.1f}%")
            logger.info(f"  Average Token Accuracy: {enhanced_metrics['avg_token_accuracy']:.1f}%")
            logger.info(f"  Synergy Bonus Rate: {enhanced_metrics['synergy_bonus_rate']:.1f}%")
            logger.info(f"  Average Total Score: {enhanced_metrics['avg_total_score']:.2f}")
        
        # Calculate Tanimoto similarities
        if calculate_tanimoto and generated_smiles_list:
            try:
                tanimoto_stats = tanimoto_calculator.calculate_batch_similarities(
                    generated_smiles_list, true_smiles_list
                )
                results['tanimoto_stats'] = tanimoto_stats
                
                logger.info(f"Tanimoto Similarity - Mean: {tanimoto_stats['mean_tanimoto']:.4f}")
            except Exception as e:
                logger.warning(f"Tanimoto calculation failed: {e}")
        
        if calculate_roc and all_probs:
            results['probs'] = np.concatenate(all_probs, axis=0)
            results['targets'] = np.concatenate(all_targets, axis=0)
            results['predictions_binary'] = np.concatenate(all_predictions, axis=0)
        
        return results

# Replace the data loading section in your main() function with this:

def main(model_config=None):
    """
    Main training function with directory-aware outputs
    
    Args:
        model_config: Optional dict with model-specific configuration
    """
    # Base configuration
    base_config = {
        'data_dir': 'C:\\Users\\pierr\\Desktop\\CS MSc Project files\\peaklist\\fakesmall',
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'hidden_dim': 256,
        'num_encoder_layers': 3,
        'num_decoder_layers': 6,
        'max_h_peaks': 150,
        'max_c_peaks': 100,
        'max_smiles_length': 256,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gradient_clip': 1.0,
        'warmup_ratio': 0.1,
        'dropout': 0.1,
        'save_metrics_every': 5,
        'calculate_roc_every': 10,
        'rebuild_vocab': False  # Default to not rebuilding
    }
    
    # Override with model-specific config if provided
    config = base_config.copy()
    
    if model_config:
        config.update(model_config)
        
        # Handle comprehensive rewards configuration
        if config.get('use_comprehensive_rewards', False):
            # Override training parameters for comprehensive rewards
            config['learning_rate'] = config.get('learning_rate', 3e-6)  # Lower LR
            config['batch_size'] = config.get('batch_size', 8)  # Smaller batch
            config['gradient_clip'] = config.get('gradient_clip', 0.25)  # Stronger clipping
            config['warmup_ratio'] = config.get('warmup_ratio', 0.4)  # Longer warmup
            
            logger.info("Using comprehensive reward system configuration:")
            logger.info(f"  Learning rate: {config['learning_rate']}")
            logger.info(f"  Batch size: {config['batch_size']}")
            logger.info(f"  Gradient clip: {config['gradient_clip']}")
            logger.info(f"  Warmup ratio: {config['warmup_ratio']}")
        
        # Set up model-specific paths - ALL outputs go to current directory
        model_name = config.get('model_name', 'default_model')
        
        # Current directory is the model's directory
        output_base = Path('.')
        
        # Create subdirectories for organization
        config['metrics_dir'] = output_base / 'training_metrics'
        config['models_dir'] = output_base / 'saved_models'
        config['predictions_dir'] = output_base / 'predictions'
        config['logs_dir'] = output_base / 'logs'
        config['vocab_dir'] = output_base / 'vocab'
        
        # Create directories if they don't exist
        for dir_path in [config['metrics_dir'], config['models_dir'], 
                        config['predictions_dir'], config['logs_dir'], config['vocab_dir']]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Set up logging to model-specific file
        log_file = config['logs_dir'] / f'training_{model_name}.log'
        
        # Clear existing handlers and set up new ones
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"Running model: {model_name}")
        logger.info(f"Output directory: {output_base.absolute()}")
    else:
        # Default configuration for standalone runs
        config['metrics_dir'] = Path('training_metrics')
        config['models_dir'] = Path('saved_models')
        config['predictions_dir'] = Path('predictions')
        config['logs_dir'] = Path('logs')
        config['vocab_dir'] = Path('vocab')
        
        # Create directories
        for dir_path in [config['metrics_dir'], config['models_dir'], 
                        config['predictions_dir'], config['logs_dir'], config['vocab_dir']]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Using device: {config['device']}")
    
    # Initialize enhanced metrics tracker
    metrics_tracker = EnhancedMetricsTracker(save_dir=config['metrics_dir'])
    
    # Load and parse data
    logger.info("Loading NMR data from complete compounds directory...")
    parser = NMRDataParser()
    all_data = []
    
    data_path = Path(config['data_dir'])
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path.absolute()}")
        return
    
    # List all nmredata files
    nmredata_files = list(data_path.glob('*.nmredata'))
    logger.info(f"Found {len(nmredata_files)} .nmredata files in complete compounds directory")
    
    # Parse files
    for file_path in tqdm(nmredata_files, desc='Parsing complete compound files'):
        data = parser.parse_file(str(file_path))
        if data:
            all_data.append(data)
    
    logger.info(f"Successfully parsed {len(all_data)} complete compounds")
    
    if len(all_data) == 0:
        logger.error("No data found! Check if the files were copied correctly.")
        return
    
        # Filter SMILES if validation is enabled
    if config.get('use_molecule_validation', False):
        logger.info("Filtering training data for valid SMILES...")

        try:
            from validity_constrains_backup import filter_valid_smiles

            # Split before filtering so we can track effects on each set
            train_val_data, test_data = train_test_split(
                all_data,
                test_size=config['test_split'], 
                random_state=42
            )

            train_data, val_data = train_test_split(
                train_val_data, 
                test_size=config['val_split']/(1-config['test_split']),
                random_state=42
            )

            original_train_len = len(train_data)
            original_val_len = len(val_data)
            original_test_len = len(test_data)

            train_data = filter_valid_smiles(train_data, logger)
            val_data = filter_valid_smiles(val_data, logger)
            test_data = filter_valid_smiles(test_data, logger)

            logger.info(f"Train data: {original_train_len} -> {len(train_data)}")
            logger.info(f"Val data: {original_val_len} -> {len(val_data)}")
            logger.info(f"Test data: {original_test_len} -> {len(test_data)}")

            if len(train_data) < 10:
                logger.error("Too few valid training samples! Check your data.")
                return

        except ImportError:
            logger.warning("Could not import validity filter. Proceeding with unfiltered data.")
            # Fallback to normal split
            train_val_data, test_data = train_test_split(
                all_data,
                test_size=config['test_split'], 
                random_state=42
            )
            train_data, val_data = train_test_split(
                train_val_data, 
                test_size=config['val_split']/(1-config['test_split']),
                random_state=42
            )
    else:
        # Default split without filtering
        train_val_data, test_data = train_test_split(
            all_data,
            test_size=config['test_split'], 
            random_state=42
        )

        train_data, val_data = train_test_split(
            train_val_data, 
            test_size=config['val_split']/(1-config['test_split']),
            random_state=42
        )

    
    # Split data into train/val/test (80/10/10)
    train_val_data, test_data = train_test_split(
        all_data,
        test_size=config['test_split'], 
        random_state=42
    )
    
    train_data, val_data = train_test_split(
        train_val_data, 
        test_size=config['val_split']/(1-config['test_split']),
        random_state=42
    )
    
    logger.info(f"Dataset splits - Train: {len(train_data)} ({len(train_data)/len(all_data)*100:.1f}%), "
                f"Val: {len(val_data)} ({len(val_data)/len(all_data)*100:.1f}%), "
                f"Test: {len(test_data)} ({len(test_data)/len(all_data)*100:.1f}%)")
    
    # Initialize SMILES tokenizer
    logger.info("Initializing SMILES tokenizer...")
    
    # Check if vocabulary already exists
    vocab_path = config['vocab_dir'] / 'smiles_tokenizer_vocab.json'
    if vocab_path.exists() and not config.get('rebuild_vocab', False):
        logger.info(f"Loading existing vocabulary from {vocab_path}")
        tokenizer = SMILESTokenizer(vocab_file=str(vocab_path))
    else:
        logger.info("Building new vocabulary from training data...")
        train_smiles = [data['canonical_smiles'] for data in train_data]
        tokenizer = create_tokenizer(
            train_smiles, 
            save_path=str(vocab_path),
            min_freq=1  # Use 1 for small dataset, 2+ for full 45k dataset
        )

    logger.info(f"Tokenizer initialized with vocabulary size: {tokenizer.vocab_size}")

    # Optional: Add tokenizer analysis
    logger.info("\nExample tokenizations:")
    for i in range(min(3, len(train_data))):
        smiles = train_data[i]['canonical_smiles']
        tokens = tokenizer.tokenize(smiles)
        encoded = tokenizer.encode(smiles, max_length=config['max_smiles_length'])
        decoded = tokenizer.decode(encoded['input_ids'])
        
        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Original: {smiles}")
        logger.info(f"  Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        logger.info(f"  Decoded: {decoded}")
        logger.info(f"  Match: {smiles == decoded}")
    
    # Initialize feature extractor
    feature_extractor = NMRFeatureExtractor(
        max_h_peaks=config['max_h_peaks'],
        max_c_peaks=config['max_c_peaks']
    )
    feature_extractor.fit(train_data)
    
    # Create datasets
    train_dataset = NMRDataset(train_data, feature_extractor, tokenizer, config['max_smiles_length'])
    val_dataset = NMRDataset(val_data, feature_extractor, tokenizer, config['max_smiles_length'])
    test_dataset = NMRDataset(test_data, feature_extractor, tokenizer, config['max_smiles_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
   # Initialize model with ChemBERTa encoder and custom tokenizer decoder
    logger.info("="*60)
    logger.info("MODEL INITIALIZATION")
    logger.info("="*60)
    
    use_chemberta = config.get('use_pretrained_chemberta', True)
    
    if use_chemberta:
        # ChemBERTa-77M uses 384 hidden dimension (not 768!)
        logger.info("Checking ChemBERTa model configuration...")
        try:
            from transformers import AutoConfig
            chemberta_config = AutoConfig.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
            chemberta_hidden_dim = chemberta_config.hidden_size
            logger.info(f"ChemBERTa-77M hidden dimension: {chemberta_hidden_dim}")
            config['hidden_dim'] = chemberta_hidden_dim  # Set to actual ChemBERTa dim
        except Exception as e:
            logger.warning(f"Could not load ChemBERTa config: {e}")
            config['hidden_dim'] = 384  # Default for ChemBERTa-77M
            
        logger.info("Initializing model with:")
        logger.info("  - ChemBERTa-77M encoder (pretrained on 77M molecules)")
        logger.info("  - Custom decoder with YOUR tokenizer vocabulary")
        logger.info(f"  - Hidden dimension: {config['hidden_dim']} (ChemBERTa-77M size)")
        logger.info(f"  - Decoder vocabulary size: {tokenizer.vocab_size}")
    else:
        logger.info("Initializing model without ChemBERTa (custom transformers)")
        # Keep original hidden_dim for custom model
    
    # Create NMR encoder with ChemBERTa
    nmr_encoder = NMREncoder(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_encoder_layers'],
        use_pretrained_chemberta=use_chemberta,
        freeze_layers=config.get('freeze_chemberta_layers', 6)
    )
    
    # Create full model
    model = NMRToSMILES(
        nmr_encoder=nmr_encoder,
        vocab_size=tokenizer.vocab_size,  # YOUR CUSTOM VOCAB SIZE
        hidden_dim=config['hidden_dim'],
        num_decoder_layers=config['num_decoder_layers']
    )
    
    # Initialize trainer with metrics tracker
    use_validity = config.get('use_molecule_validation', False)
    trainer = ComprehensiveTrainer(model, tokenizer, metrics_tracker, device=config['device'], use_comprehensive_rewards=config.get('use_comprehensive_rewards', False))
    
    # Setup optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['warmup_ratio'] * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    best_val_exact_match = 0
    
    # Get model name for saving
    model_name = config.get('model_name', 'nmr_to_smiles')
    
    # Test every N epochs to track progress
    test_every_n_epochs = 5
    
    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_result = trainer.train_epoch(train_loader, optimizer, scheduler)
        if isinstance(train_result, tuple) and len(train_result) == 4:
            train_loss, train_accuracy, train_exact_match, train_enhanced_metrics = train_result
        else:
            train_loss, train_accuracy, train_exact_match = train_result
            train_enhanced_metrics = None
        logger.info(f"Train loss: {train_loss:.4f}, Token Acc: {train_accuracy:.2f}%, "
                   f"Exact Match: {train_exact_match:.2f}%")
        
        # Evaluate on validation set
        calculate_roc = (epoch + 1) % config['calculate_roc_every'] == 0
        val_results = trainer.evaluate(val_loader, calculate_roc=calculate_roc, calculate_tanimoto=True)

        logger.info(f"Val loss: {val_results['loss']:.4f}, Token Acc: {val_results['accuracy']:.2f}%, "
                f"Exact Match: {val_results['exact_match_rate']:.2f}%")

        # Add Tanimoto logging
        if 'tanimoto_stats' in val_results:
            logger.info(f"Val Tanimoto: {val_results['tanimoto_stats']['mean_tanimoto']:.4f}, "
                    f"Validity: {val_results['tanimoto_stats']['validity_rate']:.2%}")
        
        # Optionally evaluate on test set periodically
        test_results = None
        if (epoch + 1) % test_every_n_epochs == 0:
            logger.info("Running periodic test evaluation...")
            test_results = trainer.evaluate(test_loader, return_predictions=False, calculate_roc=False)
            logger.info(f"Test loss: {test_results['loss']:.4f}, Token Acc: {test_results['accuracy']:.2f}%, "
                       f"Exact Match: {test_results['exact_match_rate']:.2f}%")
        
        # Update metrics tracker
        current_lr = scheduler.get_last_lr()[0]
        
        update_dict = {
            'train_loss': train_loss,
            'val_loss': val_results['loss'],
            'train_token_accuracy': train_accuracy,
            'val_token_accuracy': val_results['accuracy'],
            'train_exact_match': train_exact_match,
            'val_exact_match': val_results['exact_match_rate'],
            'learning_rate': current_lr
        }
        
        # Add test metrics if available
        if test_results:
            update_dict.update({
                'test_loss': test_results['loss'],
                'test_token_accuracy': test_results['accuracy'],
                'test_exact_match': test_results['exact_match_rate']
            })
        
        # Add Tanimoto metrics if available
        if 'tanimoto_stats' in val_results:
            update_dict['val_mean_tanimoto'] = val_results['tanimoto_stats']['mean_tanimoto']
            update_dict['val_validity_rate'] = val_results['tanimoto_stats']['validity_rate']

        metrics_tracker.update(epoch=epoch + 1, **update_dict)

        # Generate ROC curves if calculated
        if calculate_roc and 'probs' in val_results and 'targets' in val_results:
            metrics_tracker.plot_roc_curves_for_sequence(
                val_results['targets'], 
                val_results['probs'], 
                epoch + 1, 
                'Validation'
            )
        
        # Save best model based on exact match rate
        if val_results['exact_match_rate'] > best_val_exact_match:
            best_val_exact_match = val_results['exact_match_rate']
            model_save_path = config['models_dir'] / f'best_{model_name}_exact_match.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_results['loss'],
                'val_exact_match': val_results['exact_match_rate'],
                'config': config,
                'vocab_size': tokenizer.vocab_size
            }, model_save_path)
            logger.info(f"Saved best model to {model_save_path}")
        
        # Save based on loss too
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            loss_model_path = config['models_dir'] / f'best_{model_name}_loss.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_results['loss'],
                'val_exact_match': val_results['exact_match_rate'],
                'config': config,
                'vocab_size': tokenizer.vocab_size
            }, loss_model_path)
        
        # Plot metrics periodically
        if (epoch + 1) % config['save_metrics_every'] == 0:
            metrics_tracker.plot_comprehensive_training_curves()
            logger.info(f"Saved training curves to {metrics_tracker.save_dir}")
        
        # Save sample predictions
        if (epoch + 1) % 10 == 0:
            pred_path = config['predictions_dir'] / f'predictions_epoch_{epoch+1}.json'
            with open(pred_path, 'w') as f:
                json.dump(val_results['predictions'][:50], f, indent=2)
    
    # Final evaluation on test set
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*60)
    
    # Load best model for final test
    logger.info("Loading best model for final evaluation...")
    best_model_path = config['models_dir'] / f'best_{model_name}_exact_match.pt'
    try:
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {checkpoint['epoch']} with val exact match {checkpoint['val_exact_match']:.2f}%")
        else:
            logger.warning(f"Best model file not found at {best_model_path}, using current model")
    except Exception as e:
        logger.warning(f"Error loading best model: {e}, using current model")
    
    # Final test evaluation with full metrics and ROC
    logger.info("Running final test evaluation...")
    final_test_results = trainer.evaluate(test_loader, return_predictions=True, calculate_roc=True, calculate_tanimoto=True)
    
    logger.info(f"\nFinal Test Results:")
    logger.info(f"  Loss: {final_test_results['loss']:.4f}")
    logger.info(f"  Token Accuracy: {final_test_results['accuracy']:.2f}%")
    logger.info(f"  Exact Match: {final_test_results['exact_match_rate']:.2f}%")
    
    if 'tanimoto_stats' in final_test_results:
        logger.info(f"  Tanimoto Similarity: {final_test_results['tanimoto_stats']['mean_tanimoto']:.4f}")
        logger.info(f"  Validity Rate: {final_test_results['tanimoto_stats']['validity_rate']:.2%}")
    
    # Update metrics tracker with final test results
    metrics_tracker.update(
        epoch=config['num_epochs'],
        test_loss=final_test_results['loss'],
        test_token_accuracy=final_test_results['accuracy'],
        test_exact_match=final_test_results['exact_match_rate']
    )
    
    # Add Tanimoto metrics to final update if available
    if 'tanimoto_stats' in final_test_results:
        metrics_tracker.update(
            epoch=config['num_epochs'],
            test_mean_tanimoto=final_test_results['tanimoto_stats']['mean_tanimoto'],
            test_validity_rate=final_test_results['tanimoto_stats']['validity_rate']
        )
    
    # Generate final ROC curves for test set
    if 'probs' in final_test_results and 'targets' in final_test_results:
        logger.info("Generating final ROC curves...")
        metrics_tracker.plot_roc_curves_for_sequence(
            final_test_results['targets'],
            final_test_results['probs'],
            config['num_epochs'],
            'Test'
        )
    
    # Generate all final visualizations and reports
    logger.info("Generating final visualizations and reports...")
    metrics_tracker.plot_comprehensive_training_curves()
    metrics_tracker.plot_final_summary()
    metrics_tracker.save_final_report()
    
    # Save test predictions
    if final_test_results.get('predictions'):
        final_pred_path = config['predictions_dir'] / 'test_predictions_final.json'
        with open(final_pred_path, 'w') as f:
            json.dump(final_test_results['predictions'], f, indent=2)
        
        # Save some example predictions for analysis
        logger.info("\nExample Final Test Predictions:")
        for i in range(min(5, len(final_test_results['predictions']))):
            pred = final_test_results['predictions'][i]
            logger.info(f"\nCompound {pred['id']}:")
            logger.info(f"  True:      {pred['true']}")
            logger.info(f"  Predicted: {pred['predicted']}")
            logger.info(f"  Match:     {'YES' if pred['true'] == pred['predicted'] else 'NO'}")
    
    # Save feature extractor
    feature_extractor_path = config['models_dir'] / 'feature_extractor.pkl'
    with open(feature_extractor_path, 'wb') as f:
        pickle.dump(feature_extractor, f)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - FINAL RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Training Epochs: {config['num_epochs']}")
    logger.info(f"Best Validation Exact Match: {best_val_exact_match:.2f}%")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"\nFinal Test Set Performance:")
    logger.info(f"  - Test Loss: {final_test_results['loss']:.4f}")
    logger.info(f"  - Test Token Accuracy: {final_test_results['accuracy']:.2f}%")
    logger.info(f"  - Test Exact Match: {final_test_results['exact_match_rate']:.2f}%")
    if 'tanimoto_stats' in final_test_results:
        logger.info(f"  - Test Tanimoto Similarity: {final_test_results['tanimoto_stats']['mean_tanimoto']:.4f}")
        logger.info(f"  - Test Validity Rate: {final_test_results['tanimoto_stats']['validity_rate']:.2%}")
    logger.info(f"\nAll metrics and visualizations saved to: {metrics_tracker.save_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()