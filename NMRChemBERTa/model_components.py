"""
Modular components for NMR-ChemBERTa model
Separated for better maintainability and testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for molecular coordinates"""
    
    def __init__(self, d_model: int, max_atoms: int = 200):
        super().__init__()
        self.d_model = d_model
        
        # Learnable positional encoding based on 3D coordinates
        self.coord_proj = nn.Linear(3, d_model // 4)
        self.distance_proj = nn.Linear(1, d_model // 4)
        self.angle_proj = nn.Linear(1, d_model // 4)
        self.dihedral_proj = nn.Linear(1, d_model // 4)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch_size, max_atoms, 3)
            atom_mask: (batch_size, max_atoms)
        Returns:
            pos_encoding: (batch_size, max_atoms, d_model)
        """
        batch_size, max_atoms, _ = coords.shape
        
        # Direct coordinate encoding
        coord_enc = self.coord_proj(coords)  # (B, N, d/4)
        
        # Calculate pairwise distances
        distances = torch.cdist(coords, coords)  # (B, N, N)
        
        # Get nearest neighbor distances (excluding self)
        distances_masked = distances + (1 - atom_mask.unsqueeze(-1)) * 1e6
        nearest_dist, _ = torch.min(
            distances_masked + torch.eye(max_atoms, device=coords.device) * 1e6, 
            dim=-1
        )
        dist_enc = self.distance_proj(nearest_dist.unsqueeze(-1))  # (B, N, d/4)
        
        # Placeholder for angles and dihedrals (can be enhanced)
        angles = torch.zeros(batch_size, max_atoms, 1, device=coords.device)
        dihedrals = torch.zeros(batch_size, max_atoms, 1, device=coords.device)
        
        angle_enc = self.angle_proj(angles)  # (B, N, d/4)
        dihedral_enc = self.dihedral_proj(dihedrals)  # (B, N, d/4)
        
        # Concatenate all encodings
        pos_encoding = torch.cat([coord_enc, dist_enc, angle_enc, dihedral_enc], dim=-1)
        
        # Apply layer norm and mask
        pos_encoding = self.layer_norm(pos_encoding)
        return pos_encoding * atom_mask.unsqueeze(-1)


class NMREncoder(nn.Module):
    """Encoder for NMR spectroscopic data"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Separate encoders for H and C NMR
        self.h_shift_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        self.c_shift_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Combine H and C features
        self.nmr_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, h_shifts: torch.Tensor, c_shifts: torch.Tensor,
                h_mask: torch.Tensor, c_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_shifts: (batch_size, max_atoms) - H NMR chemical shifts
            c_shifts: (batch_size, max_atoms) - C NMR chemical shifts
            h_mask: (batch_size, max_atoms) - mask for H NMR data
            c_mask: (batch_size, max_atoms) - mask for C NMR data
        Returns:
            nmr_features: (batch_size, max_atoms, hidden_dim)
        """
        # Encode chemical shifts
        h_features = self.h_shift_encoder(h_shifts.unsqueeze(-1))  # (B, N, h/2)
        c_features = self.c_shift_encoder(c_shifts.unsqueeze(-1))  # (B, N, h/2)
        
        # Apply masks
        h_features = h_features * h_mask.unsqueeze(-1)
        c_features = c_features * c_mask.unsqueeze(-1)
        
        # Concatenate and fuse
        nmr_features = torch.cat([h_features, c_features], dim=-1)  # (B, N, h)
        nmr_features = self.nmr_fusion(nmr_features)
        
        return nmr_features


class AtomFeatureEncoder(nn.Module):
    """Encoder for atom-level features"""
    
    def __init__(self, num_atom_types: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.atom_embedding = nn.Embedding(
            num_atom_types + 1, hidden_dim, padding_idx=0
        )
        self.atom_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.atom_embedding.weight)
        
    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_types: (batch_size, max_atoms) - atom type indices
        Returns:
            atom_features: (batch_size, max_atoms, hidden_dim)
        """
        # Handle padding index (-1) by converting to 0
        atom_types_safe = torch.where(atom_types >= 0, atom_types, 0)
        
        atom_embeddings = self.atom_embedding(atom_types_safe)
        atom_features = self.atom_encoder(atom_embeddings)
        
        # Mask out padding atoms
        mask = (atom_types >= 0).unsqueeze(-1).float()
        return atom_features * mask


class CrossAttentionLayer(nn.Module):
    """Cross-attention between SMILES tokens and atom features"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross-attention with residual connection
        attn_output, _ = self.multihead_attn(
            query, key, value, key_padding_mask=attn_mask
        )
        query = self.norm1(query + self.dropout1(attn_output))
        
        # FFN with residual connection
        ffn_output = self.ffn(query)
        output = self.norm2(query + self.dropout2(ffn_output))
        
        return output


class TaskHeads(nn.Module):
    """Collection of task-specific prediction heads"""
    
    def __init__(self, hidden_dim: int, num_atom_types: int, dropout: float = 0.1):
        super().__init__()
        
        # NMR chemical shift predictor
        self.nmr_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Predict H and C shifts
        )
        
        # 3D position predictor
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # Predict x, y, z coordinates
        )
        
        # Atom type classifier
        self.atom_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_atom_types)
        )
        
        # SMILES position predictor
        self.smiles_position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, atom_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply all task heads to atom features"""
        return {
            'nmr_shifts': self.nmr_predictor(atom_features),
            'positions': self.position_predictor(atom_features),
            'atom_types': self.atom_classifier(atom_features),
            'smiles_positions': self.smiles_position_predictor(atom_features)
        }


class FeatureFusion(nn.Module):
    """Module for fusing different types of features"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Fuse multiple feature tensors"""
        concatenated = torch.cat(features, dim=-1)
        return self.fusion(concatenated)


class GradientCheckpointWrapper(nn.Module):
    """Wrapper for gradient checkpointing to save memory"""
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self.module, *args, **kwargs, use_reentrant=False
            )
        else:
            return self.module(*args, **kwargs)


def apply_gradient_checkpointing(model: nn.Module, layers_to_checkpoint: list):
    """Apply gradient checkpointing to specific layers"""
    for layer_name in layers_to_checkpoint:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            setattr(model, layer_name, GradientCheckpointWrapper(layer))
    
    return model