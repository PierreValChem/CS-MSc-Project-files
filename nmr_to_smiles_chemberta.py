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
            
            return {
                'h_features': features['h_features'],
                'c_features': features['c_features'],
                'global_features': features['global_features'],
                'smiles_ids': smiles_tokens['input_ids'].squeeze(),
                'smiles_mask': smiles_tokens['attention_mask'].squeeze(),
                'id': data.get('id', f'unknown_{idx}')
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            raise

class NMREncoder(nn.Module):
    """Encode NMR data into embeddings"""
    
    def __init__(self, h_input_dim=4, c_input_dim=4, hidden_dim=256, num_layers=3):
        super().__init__()
        
        self.h_projection = nn.Linear(h_input_dim, hidden_dim)
        self.h_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=num_layers
        )
        
        self.c_projection = nn.Linear(c_input_dim, hidden_dim)
        self.c_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=num_layers
        )
        
        self.global_projection = nn.Linear(4, hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, h_features, c_features, global_features):
        h_emb = self.h_projection(h_features)
        c_emb = self.c_projection(c_features)
        
        h_emb = h_emb.transpose(0, 1)
        c_emb = c_emb.transpose(0, 1)
        
        h_encoded = self.h_transformer(h_emb)
        c_encoded = self.c_transformer(c_emb)
        
        h_pooled = h_encoded.mean(dim=0)
        c_pooled = c_encoded.mean(dim=0)
        
        global_emb = self.global_projection(global_features)
        
        combined = torch.cat([h_pooled, c_pooled, global_emb], dim=-1)
        fused = self.fusion(combined)
        
        return fused

class NMRToSMILES(nn.Module):
    """Complete model for NMR to SMILES prediction"""
    
    def __init__(self, nmr_encoder: NMREncoder, vocab_size: int, 
                 hidden_dim: int = 256, num_decoder_layers: int = 6):
        super().__init__()
        self.nmr_encoder = nmr_encoder
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=num_decoder_layers
        )
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.1)
        self.vocab_size = vocab_size
    
    def forward(self, h_features, c_features, global_features, 
                target_ids=None, target_mask=None):
        nmr_encoding = self.nmr_encoder(h_features, c_features, global_features)
        nmr_encoding = nmr_encoding.unsqueeze(0)
        
        if target_ids is not None:
            seq_len = target_ids.size(1)
            positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0)
            
            token_emb = self.token_embedding(target_ids)
            pos_emb = self.position_embedding(positions)
            target_emb = self.dropout(token_emb + pos_emb)
            
            causal_mask = self._generate_square_subsequent_mask(seq_len).to(target_ids.device)
            
            target_emb = target_emb.transpose(0, 1)
            
            tgt_padding_mask = None
            if target_mask is not None:
                tgt_padding_mask = ~target_mask.bool()
            
            decoded = self.decoder(
                target_emb,
                nmr_encoding,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            output = self.output_projection(decoded.transpose(0, 1))
            
            return output
        else:
            return self._generate(nmr_encoding)
    
    def _generate(self, nmr_encoding, tokenizer, max_length=256, temperature=1.0):
            """Generate SMILES autoregressively"""
            device = nmr_encoding.device
            batch_size = nmr_encoding.size(1)
            
            # Start with BOS token
            generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                positions = torch.arange(generated.size(1), device=device).unsqueeze(0)
                token_emb = self.token_embedding(generated)
                pos_emb = self.position_embedding(positions)
                target_emb = self.dropout(token_emb + pos_emb)
                
                target_emb = target_emb.transpose(0, 1)
                decoded = self.decoder(target_emb, nmr_encoding)
                
                logits = self.output_projection(decoded[-1]) / temperature
                probs = torch.softmax(logits, dim=-1)
                
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS token
                if tokenizer and hasattr(tokenizer, 'eos_token_id'):
                    if (next_token == tokenizer.eos_token_id).all():
                        break
                elif (next_token == 2).all():  # Common EOS token ID
                    break
        
            return generated
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal attention mask"""
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.float32), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _generate_valid_forced(self, nmr_encoding, tokenizer, max_length=256, temperature=0.8, 
                            max_attempts=50, beam_size=5):
        """
        Generate SMILES with forced validity using multiple strategies
        """
        device = nmr_encoding.device
        batch_size = nmr_encoding.size(1)
        
        # Try to import RDKit for validation
        try:
            from rdkit import Chem
            rdkit_available = True
        except ImportError:
            rdkit_available = False
            logger.warning("RDKit not available for validity checking")
        
        # Strategy 1: Try simple generation first with validity check
        for attempt in range(min(5, max_attempts)):
            generated = self._generate(nmr_encoding, tokenizer, max_length, temperature)
            smiles = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            if rdkit_available:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return generated
        
        # Strategy 2: Constrained beam search with validity checking
        best_valid_smiles = None
        best_score = float('-inf')
        
        # Get special token IDs from tokenizer
        pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
        eos_token_id = getattr(tokenizer, 'eos_token_id', 2)
        
        for beam_idx in range(min(3, beam_size)):  # Limit beam search attempts
            beams = [{
                'tokens': torch.zeros((1,), dtype=torch.long, device=device),
                'score': 0.0,
                'smiles': ''
            }]
            
            for step in range(min(50, max_length)):  # Limit steps to avoid hanging
                new_beams = []
                
                for beam in beams[:5]:  # Limit beams per step
                    if len(beam['tokens']) >= max_length - 1:
                        continue
                    
                    # Get next token probabilities
                    try:
                        positions = torch.arange(beam['tokens'].size(0), device=device).unsqueeze(0)
                        token_emb = self.token_embedding(beam['tokens'].unsqueeze(0))
                        pos_emb = self.position_embedding(positions)
                        target_emb = self.dropout(token_emb + pos_emb)
                        
                        target_emb = target_emb.transpose(0, 1)
                        decoded = self.decoder(target_emb, nmr_encoding)
                        
                        logits = self.output_projection(decoded[-1]) / temperature
                        probs = torch.softmax(logits, dim=-1)
                        
                        # Sample top k tokens
                        k = min(5, probs.size(-1))
                        top_probs, top_indices = torch.topk(probs[0], k)
                        
                        for idx in range(k):
                            token_id = top_indices[idx].item()
                            token_prob = top_probs[idx].item()
                            
                            # Skip if it's padding token
                            if token_id == pad_token_id:
                                continue
                            
                            new_tokens = torch.cat([beam['tokens'], torch.tensor([token_id], device=device)])
                            
                            # Decode to check validity
                            new_smiles = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
                            
                            # Check if it's the end token
                            if token_id == eos_token_id:
                                if rdkit_available:
                                    mol = Chem.MolFromSmiles(new_smiles)
                                    if mol is not None and beam['score'] + np.log(token_prob) > best_score:
                                        best_score = beam['score'] + np.log(token_prob)
                                        best_valid_smiles = new_smiles
                                continue
                            
                            # Quick validity checks for partial SMILES
                            if self._is_valid_partial_smiles(new_smiles):
                                new_beams.append({
                                    'tokens': new_tokens,
                                    'score': beam['score'] + np.log(token_prob),
                                    'smiles': new_smiles
                                })
                    except Exception as e:
                        logger.debug(f"Error in beam search: {e}")
                        continue
                
                # Keep only top beams
                beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:3]
                
                if not beams:
                    break
        
        # If we found a valid SMILES, convert it back to tokens
        if best_valid_smiles is not None:
            # Use the tokenizer's encode method properly
            encoded = tokenizer.encode(best_valid_smiles, max_length=max_length)
            if isinstance(encoded, dict):
                # Handle dict output from SMILESTokenizer
                return torch.tensor(encoded['input_ids'], device=device).unsqueeze(0)
            else:
                # Handle list output
                return torch.tensor([encoded], device=device)
        
        # Strategy 3: Try more simple generations
        for _ in range(max_attempts - 5):
            generated = self._generate(nmr_encoding, tokenizer, max_length, temperature * 0.8)
            smiles = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            if rdkit_available:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return generated
        
        # Final fallback: Return a simple valid default
        default_smiles = "C"  # Methane
        encoded = tokenizer.encode(default_smiles, max_length=max_length)
        if isinstance(encoded, dict):
            return torch.tensor(encoded['input_ids'], device=device).unsqueeze(0)
        else:
            return torch.tensor([encoded], device=device)

    # Also update the _is_valid_partial_smiles method to be more lenient:
    def _is_valid_partial_smiles(self, partial_smiles):
        """Quick checks for partial SMILES validity - more lenient"""
        if not partial_smiles:
            return True
        
        # Don't check empty or very short strings
        if len(partial_smiles) < 2:
            return True
        
        # Check balanced parentheses
        open_paren = partial_smiles.count('(')
        close_paren = partial_smiles.count(')')
        if close_paren > open_paren:
            return False
        
        # Check balanced brackets
        open_bracket = partial_smiles.count('[')
        close_bracket = partial_smiles.count(']')
        if close_bracket > open_bracket:
            return False
        
        # Don't check for invalid patterns in partial SMILES
        # as they might be completed later
        
        return True
    
    def _is_valid_smiles(self, smiles):
        """Check if SMILES is valid using RDKit"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    
    # Modify the forward method to accept tokenizer for generation
    def forward(self, h_features, c_features, global_features, 
                target_ids=None, target_mask=None, tokenizer=None):
        nmr_encoding = self.nmr_encoder(h_features, c_features, global_features)
        nmr_encoding = nmr_encoding.unsqueeze(0)
        
        if target_ids is not None:
            # Training mode - normal forward pass
            seq_len = target_ids.size(1)
            positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0)
            
            token_emb = self.token_embedding(target_ids)
            pos_emb = self.position_embedding(positions)
            target_emb = self.dropout(token_emb + pos_emb)
            
            causal_mask = self._generate_square_subsequent_mask(seq_len).to(target_ids.device)
            
            target_emb = target_emb.transpose(0, 1)
            
            tgt_padding_mask = None
            if target_mask is not None:
                tgt_padding_mask = ~target_mask.bool()
            
            decoded = self.decoder(
                target_emb,
                nmr_encoding,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            output = self.output_projection(decoded.transpose(0, 1))
            
            return output
        else:
            # Generation mode - need tokenizer
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided for generation mode")
            return self._generate(nmr_encoding, tokenizer)

class ComprehensiveTrainer:
    """Trainer with comprehensive reward/penalty system for proper molecule generation"""
    
    def __init__(self, model, tokenizer, metrics_tracker, device='cuda', use_comprehensive_rewards=False):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.metrics_tracker = metrics_tracker
        
        # Initialize comprehensive reward/penalty system
        if use_comprehensive_rewards:
            try:
                from comprehensive_reward_system import create_comprehensive_system
                self.validator, self.criterion = create_comprehensive_system(model, tokenizer, device)
                self.use_comprehensive = True
                logger.info("Using comprehensive reward/penalty system for proper molecule generation")
                logger.info("Reward priorities: H/C matching > Validity > Structure > Basic syntax")
            except ImportError:
                logger.warning("Comprehensive reward system not available. Using standard loss.")
                self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                self.use_comprehensive = False
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            self.use_comprehensive = False
    
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
        for batch in progress_bar:
            h_features = batch['h_features'].to(self.device)
            c_features = batch['c_features'].to(self.device)
            global_features = batch['global_features'].to(self.device)
            smiles_ids = batch['smiles_ids'].to(self.device)
            smiles_mask = batch['smiles_mask'].to(self.device)
            
            # Get expected atom counts from global features
            expected_h = global_features[:, 0].cpu().numpy()
            expected_c = global_features[:, 1].cpu().numpy()
            
            outputs = self.model(
                h_features, c_features, global_features,
                smiles_ids[:, :-1], smiles_mask[:, :-1]
            )
            
            # Generate predictions for comprehensive evaluation
            predictions = None
            nmr_data = None
            
            if self.use_comprehensive:
                with torch.no_grad():
                    # Generate predictions for each sample
                    predictions = []
                    for i in range(len(h_features)):
                        try:
                            generated = self.model(
                                h_features[i:i+1], 
                                c_features[i:i+1], 
                                global_features[i:i+1], 
                                tokenizer=self.tokenizer
                            )
                            pred_smiles = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                        except Exception as e:
                            pred_smiles = ""  # Handle generation errors
                        
                        predictions.append(pred_smiles)
                
                # Prepare NMR data for comprehensive evaluation
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
                            comprehensive_metrics[key] += batch_stats[key]
                    
                    # Track average scores and token accuracy
                    comprehensive_metrics['avg_score'] += loss_components.get('molecule_reward_penalty_sum', 0)
                    if 'avg_token_accuracy' in loss_components:
                        comprehensive_metrics['token_accuracy_scores'].append(loss_components['avg_token_accuracy'])
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
                    # Generate for exact match checking
                    predictions = []
                    for i in range(len(h_features)):
                        try:
                            generated = self.model(
                                h_features[i:i+1], 
                                c_features[i:i+1], 
                                global_features[i:i+1], 
                                tokenizer=self.tokenizer
                            )
                            pred_smiles = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                        except:
                            pred_smiles = ""
                        predictions.append(pred_smiles)
                
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
            
            # Update progress bar with comprehensive metrics including token accuracy
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
                
                # Get token accuracy from loss components if available
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
                else:
                    progress_dict.update({
                        'empty': f'{empty_rate:.1f}%',
                        'valid': f'{valid_rate:.1f}%',
                        'H/C': f'{perfect_hc_rate:.1f}%'
                    })
            
            progress_bar.set_postfix(progress_dict)
        
        # Calculate final metrics
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        exact_match_rate = (exact_matches / total_samples) * 100 if total_samples > 0 else 0
        
        # Comprehensive metrics summary including token accuracy synergy
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
                'token_accuracy_integration': True  # Flag to show token accuracy is integrated
            }
            
            logger.info(f"  Comprehensive Training Metrics (All Rewards Integrated):")
            logger.info(f"    Empty Predictions: {enhanced_metrics['empty_prediction_rate']:.1f}%")
            logger.info(f"    Invalid SMILES: {enhanced_metrics['invalid_smiles_rate']:.1f}%")
            logger.info(f"    Invalid Brackets: {enhanced_metrics['invalid_brackets_rate']:.1f}%")
            logger.info(f"    Invalid Valency: {enhanced_metrics['invalid_valency_rate']:.1f}%")
            logger.info(f"    Perfect H/C Match: {enhanced_metrics['perfect_hc_match_rate']:.1f}%")
            logger.info(f"    Good H/C Match: {enhanced_metrics['good_hc_match_rate']:.1f}%")
            logger.info(f"    Valid Molecules: {enhanced_metrics['validity_rate']:.1f}%")
            logger.info(f"    Average Token Accuracy: {enhanced_metrics['avg_token_accuracy']:.1f}%")
            logger.info(f"    Average Reward Score: {enhanced_metrics['avg_reward_score']:.2f}")
            logger.info(f"    Token Accuracy: Integrated with synergy bonuses [Active]")
        
        if self.use_comprehensive and enhanced_metrics:
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
                expected_h = global_features[:, 0].cpu().numpy()
                expected_c = global_features[:, 1].cpu().numpy()
                
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
                
                # Generate predictions with comprehensive evaluation including token accuracy
                if return_predictions or calculate_tanimoto or self.use_comprehensive:
                    batch_predictions = []
                    
                    for i in range(len(h_features)):
                        try:
                            generated = self.model(
                                h_features[i:i+1], 
                                c_features[i:i+1], 
                                global_features[i:i+1], 
                                tokenizer=self.tokenizer
                            )
                            pred_smiles = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                        except:
                            pred_smiles = ""
                        
                        batch_predictions.append(pred_smiles)
                    
                    # Process predictions with comprehensive evaluation including token accuracy
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
                            if self.use_comprehensive:
                                evaluation = self.validator.comprehensive_evaluation(
                                    pred_smiles, int(expected_h[i]), int(expected_c[i])
                                )
                                
                                # Calculate token accuracy for this sample
                                sample_outputs = outputs[i]
                                sample_targets = smiles_ids[i, 1:]  # Skip first token for targets
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
                    
                    # Update comprehensive metrics including token accuracy synergy
                    if self.use_comprehensive:
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
                            
                            # Update standard metrics
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
        
        # Add comprehensive metrics including token accuracy integration
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
                'token_accuracy_integrated': True  # Flag showing integration
            }
            
            results['comprehensive_metrics'] = enhanced_metrics
            
            logger.info(f"Comprehensive Evaluation Metrics (All Rewards Integrated):")
            logger.info(f"  Empty Predictions: {enhanced_metrics['empty_prediction_rate']:.1f}%")
            logger.info(f"  Invalid SMILES: {enhanced_metrics['invalid_smiles_rate']:.1f}%")
            logger.info(f"  Invalid Brackets: {enhanced_metrics['invalid_brackets_rate']:.1f}%")
            logger.info(f"  Invalid Valency: {enhanced_metrics['invalid_valency_rate']:.1f}%")
            logger.info(f"  Perfect H/C Match: {enhanced_metrics['perfect_hc_match_rate']:.1f}%")
            logger.info(f"  Good H/C Match: {enhanced_metrics['good_hc_match_rate']:.1f}%")
            logger.info(f"  Valid Molecules: {enhanced_metrics['validity_rate']:.1f}%")
            logger.info(f"  Average Token Accuracy: {enhanced_metrics['avg_token_accuracy']:.1f}%")
            logger.info(f"  Synergy Bonus Rate: {enhanced_metrics['synergy_bonus_rate']:.1f}%")
            logger.info(f"  Average Total Score: {enhanced_metrics['avg_total_score']:.2f}")
            logger.info(f"  Token Accuracy Integration: ✓ Active with synergy bonuses")
        
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
    
    # Setup directories and logging
    if model_config:
        config.update(model_config)
        
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
    
    # Initialize model
    logger.info("Initializing model...")
    nmr_encoder = NMREncoder(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_encoder_layers']
    )
    
    model = NMRToSMILES(
        nmr_encoder=nmr_encoder,
        vocab_size=tokenizer.vocab_size,
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
    test_every_n_epochs = 10
    
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