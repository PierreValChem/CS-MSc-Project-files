"""
Simplified NMR-ChemBERTa model focusing only on SMILES to NMR prediction
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SMILEStoNMRModel(nn.Module):
    """
    Simplified model that predicts NMR chemical shifts directly from SMILES
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_atoms = config.model.max_atoms
        
        # Load pre-trained ChemBERTa
        self.chemberta = AutoModel.from_pretrained(config.model.chemberta_name)
        self.chemberta_dim = self.chemberta.config.hidden_size
        
        # Projection to match our hidden dimension
        self.projection = nn.Linear(self.chemberta_dim, config.model.hidden_dim)
        
        # Attention pooling to aggregate token representations
        self.attention_pool = nn.Sequential(
            nn.Linear(config.model.hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # NMR prediction head - predicts per-atom NMR shifts
        self.nmr_predictor = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim * 2),
            nn.LayerNorm(config.model.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_dim * 2, config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_dim, self.max_atoms * 2)  # H and C shifts for each atom
        )
        
        # Initialize weights
        self._init_weights()
        
        # Optionally freeze ChemBERTa
        if config.model.freeze_chemberta:
            for param in self.chemberta.parameters():
                param.requires_grad = False
            logger.info("ChemBERTa parameters frozen")
    
    def _init_weights(self):
        """Initialize weights with small values"""
        for module in [self.projection, self.attention_pool, self.nmr_predictor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass - only uses SMILES input
        
        Args:
            input_ids: Tokenized SMILES
            attention_mask: Attention mask for SMILES
            **kwargs: Other inputs (ignored for compatibility)
            
        Returns:
            Dictionary with 'nmr_shifts' predictions
        """
        # Get ChemBERTa embeddings
        chemberta_output = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use last hidden states
        hidden_states = chemberta_output.last_hidden_state  # (batch, seq_len, chemberta_dim)
        
        # Project to our dimension
        hidden_states = self.projection(hidden_states)  # (batch, seq_len, hidden_dim)
        
        # Attention pooling - create a single representation
        attention_weights = self.attention_pool(hidden_states)  # (batch, seq_len, 1)
        attention_weights = attention_weights * attention_mask.unsqueeze(-1)  # Mask padding
        
        # Weighted sum
        pooled = torch.sum(hidden_states * attention_weights, dim=1)  # (batch, hidden_dim)
        
        # Predict NMR shifts
        nmr_predictions = self.nmr_predictor(pooled)  # (batch, max_atoms * 2)
        
        # Reshape to (batch, max_atoms, 2)
        nmr_shifts = nmr_predictions.view(-1, self.max_atoms, 2)
        
        return {
            'nmr_shifts': nmr_shifts,
            'smiles_representation': pooled  # For visualization/analysis
        }


class SimplifiedNMRLoss(nn.Module):
    """Simplified loss focusing only on NMR prediction"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.training.nmr_loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none', delta=1.0)
        elif config.training.nmr_loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            self.base_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions: Dict, targets: Dict, masks: Dict) -> Dict:
        """
        Compute NMR-only loss
        """
        pred_shifts = predictions['nmr_shifts']
        target_shifts = targets['nmr_shifts']
        nmr_mask = masks['nmr_mask']
        
        # Compute loss
        loss = self.base_loss(pred_shifts, target_shifts)
        
        # Weight by chemical shift magnitude (optional)
        if self.config.training.nmr_loss_reduction == 'weighted':
            # Higher weight for larger chemical shifts
            weights = 1.0 + torch.abs(target_shifts) / 50.0
            loss = loss * weights
        
        # Apply mask
        masked_loss = loss * nmr_mask
        num_valid = nmr_mask.sum()
        
        if num_valid > 0:
            total_loss = masked_loss.sum() / num_valid
        else:
            total_loss = torch.tensor(0.0, device=pred_shifts.device)
        
        # Separate H and C losses for monitoring
        h_loss = masked_loss[:, :, 0].sum() / (nmr_mask[:, :, 0].sum() + 1e-6)
        c_loss = masked_loss[:, :, 1].sum() / (nmr_mask[:, :, 1].sum() + 1e-6)
        
        return {
            'total_loss': total_loss,
            'nmr_loss': total_loss,
            'h_nmr_loss': h_loss,
            'c_nmr_loss': c_loss
        }


def create_smiles_nmr_model(config):
    """Factory function to create the appropriate model"""
    if hasattr(config.training, 'position_loss_weight') and config.training.position_loss_weight == 0:
        # Use simplified model if only doing NMR prediction
        logger.info("Creating simplified SMILES-to-NMR model")
        return SMILEStoNMRModel(config)
    else:
        # Use full model
        from nmr_chemberta_model import NMRChemBERTa
        logger.info("Creating full NMR-ChemBERTa model")
        return NMRChemBERTa(config)