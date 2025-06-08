"""
Enhanced SMILES-to-NMR model with improved architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional
import logging
import math

logger = logging.getLogger(__name__)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for better sequence aggregation"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=0.1,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, hidden_states, attention_mask):
        batch_size = hidden_states.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # Create key padding mask
        key_padding_mask = ~attention_mask.bool()
        
        # Apply attention
        attended, _ = self.attention(
            query, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask
        )
        
        return attended.squeeze(1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for atom positions"""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EnhancedSMILEStoNMRModel(nn.Module):
    """
    Enhanced model with better architecture for NMR prediction
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_atoms = config.model.max_atoms
        
        # Load pre-trained ChemBERTa
        self.chemberta = AutoModel.from_pretrained(config.model.chemberta_name)
        self.chemberta_dim = self.chemberta.config.hidden_size
        
        # Projection to match our hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(self.chemberta_dim, config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout)
        )
        
        # Multi-head attention pooling
        self.attention_pool = MultiHeadAttentionPooling(
            config.model.hidden_dim, 
            num_heads=config.model.num_attention_heads
        )
        
        # Positional encoding for atoms
        self.positional_encoding = PositionalEncoding(config.model.hidden_dim)
        
        # Atom-level transformer for better per-atom predictions
        self.atom_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.model.hidden_dim,
                nhead=config.model.num_attention_heads,
                dim_feedforward=config.model.hidden_dim * 4,
                dropout=config.model.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=3
        )
        
        # Separate prediction heads for H and C NMR
        self.h_nmr_predictor = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.nmr_hidden_dim),
            nn.LayerNorm(config.model.nmr_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.nmr_hidden_dim, config.model.nmr_hidden_dim // 2),
            nn.LayerNorm(config.model.nmr_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.nmr_hidden_dim // 2, 1)
        )
        
        self.c_nmr_predictor = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.nmr_hidden_dim),
            nn.LayerNorm(config.model.nmr_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.nmr_hidden_dim, config.model.nmr_hidden_dim // 2),
            nn.LayerNorm(config.model.nmr_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.nmr_hidden_dim // 2, 1)
        )
        
        # Global context for better predictions
        self.global_context = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim),
            nn.GELU()
        )
        
        # Initialize weights
        self._init_weights()
        
        # Optionally freeze ChemBERTa
        if config.model.freeze_chemberta:
            for param in self.chemberta.parameters():
                param.requires_grad = False
            logger.info("ChemBERTa parameters frozen")
    
    def _init_weights(self):
        """Initialize weights with careful initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization scaled down
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_normal_(module.in_proj_weight, gain=0.5)
                nn.init.xavier_normal_(module.out_proj.weight, gain=0.5)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with better feature extraction
        """
        batch_size = input_ids.size(0)
        
        # Get ChemBERTa embeddings
        chemberta_output = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use multiple layers for better representation
        hidden_states = chemberta_output.last_hidden_state
        
        # Also use intermediate layers
        if hasattr(chemberta_output, 'hidden_states'):
            # Average last 4 layers for better features
            last_layers = chemberta_output.hidden_states[-4:]
            hidden_states = torch.stack(last_layers, dim=0).mean(dim=0)
        
        # Project to our dimension
        hidden_states = self.projection(hidden_states)
        
        # Get global molecular representation
        global_repr = self.attention_pool(hidden_states, attention_mask)
        global_context = self.global_context(global_repr)
        
        # Create atom-level representations
        # Expand global context to all atom positions
        atom_features = global_context.unsqueeze(1).expand(-1, self.max_atoms, -1)
        
        # Add positional encoding
        atom_features = self.positional_encoding(atom_features)
        
        # Apply atom transformer for inter-atom reasoning
        atom_features = self.atom_transformer(atom_features)
        
        # Add residual connection with global context
        atom_features = atom_features + global_context.unsqueeze(1)
        
        # Predict NMR shifts separately for H and C
        h_shifts = self.h_nmr_predictor(atom_features).squeeze(-1)  # (batch, max_atoms)
        c_shifts = self.c_nmr_predictor(atom_features).squeeze(-1)  # (batch, max_atoms)
        
        # Stack to create (batch, max_atoms, 2)
        nmr_shifts = torch.stack([h_shifts, c_shifts], dim=-1)
        
        return {
            'nmr_shifts': nmr_shifts,
            'atom_features': atom_features,  # For visualization
            'global_representation': global_repr  # For analysis
        }


class AdaptiveLoss(nn.Module):
    """Adaptive loss that focuses on hard examples"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_loss = nn.HuberLoss(reduction='none', delta=1.0)
        
        # Learnable temperature for adaptive weighting
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, predictions: Dict, targets: Dict, masks: Dict) -> Dict:
        pred_shifts = predictions['nmr_shifts']
        target_shifts = targets['nmr_shifts']
        nmr_mask = masks['nmr_mask']
        
        # Compute base loss
        loss = self.base_loss(pred_shifts, target_shifts)
        
        # Adaptive weighting based on error magnitude
        with torch.no_grad():
            errors = torch.abs(pred_shifts - target_shifts)
            weights = F.softmax(errors / self.temperature, dim=1)
            weights = weights * nmr_mask  # Apply mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        
        # Apply adaptive weights
        weighted_loss = loss * weights.detach()
        
        # Standard weighting by shift magnitude
        if self.config.training.nmr_loss_reduction == 'weighted':
            magnitude_weights = 1.0 + torch.abs(target_shifts) / 50.0
            weighted_loss = weighted_loss * magnitude_weights
        
        # Apply mask and compute mean
        masked_loss = weighted_loss * nmr_mask
        num_valid = nmr_mask.sum()
        
        if num_valid > 0:
            total_loss = masked_loss.sum() / num_valid
        else:
            total_loss = torch.tensor(0.0, device=pred_shifts.device)
        
        # Separate losses for monitoring
        h_loss = masked_loss[:, :, 0].sum() / (nmr_mask[:, :, 0].sum() + 1e-6)
        c_loss = masked_loss[:, :, 1].sum() / (nmr_mask[:, :, 1].sum() + 1e-6)
        
        return {
            'total_loss': total_loss,
            'nmr_loss': total_loss,
            'h_nmr_loss': h_loss,
            'c_nmr_loss': c_loss,
            'temperature': self.temperature.item()
        }