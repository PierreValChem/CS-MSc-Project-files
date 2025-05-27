"""
NMR-ChemBERTa Model Architecture
Combines ChemBERTa with 3D coordinates and NMR data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


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
        nearest_dist, _ = torch.min(distances_masked + torch.eye(max_atoms).to(coords.device) * 1e6, dim=-1)
        dist_enc = self.distance_proj(nearest_dist.unsqueeze(-1))  # (B, N, d/4)
        
        # Placeholder for angles and dihedrals (simplified)
        angles = torch.zeros(batch_size, max_atoms, 1).to(coords.device)
        dihedrals = torch.zeros(batch_size, max_atoms, 1).to(coords.device)
        
        angle_enc = self.angle_proj(angles)  # (B, N, d/4)
        dihedral_enc = self.dihedral_proj(dihedrals)  # (B, N, d/4)
        
        # Concatenate all encodings
        pos_encoding = torch.cat([coord_enc, dist_enc, angle_enc, dihedral_enc], dim=-1)
        
        return pos_encoding * atom_mask.unsqueeze(-1)


class NMREncoder(nn.Module):
    """Encoder for NMR spectroscopic data"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Separate encoders for H and C NMR
        self.h_shift_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        self.c_shift_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Combine H and C features
        self.nmr_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
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
    
    def __init__(self, num_atom_types: int, hidden_dim: int):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types + 1, hidden_dim, padding_idx=-1)
        self.atom_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_types: (batch_size, max_atoms) - atom type indices
        Returns:
            atom_features: (batch_size, max_atoms, hidden_dim)
        """
        atom_embeddings = self.atom_embedding(atom_types)
        atom_features = self.atom_encoder(atom_embeddings)
        return atom_features


class CrossAttentionLayer(nn.Module):
    """Cross-attention between SMILES tokens and atom features"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross-attention
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=attn_mask)
        query = self.norm1(query + attn_output)
        
        # FFN
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output


class NMRChemBERTa(nn.Module):
    """Main model combining ChemBERTa with NMR and 3D structural data"""
    
    def __init__(self, 
                 chemberta_name: str = 'seyonec/ChemBERTa-zinc-base-v1',
                 hidden_dim: int = 768,
                 num_atom_types: int = 10,
                 max_atoms: int = 200,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        
        # Load pre-trained ChemBERTa
        self.chemberta = AutoModel.from_pretrained(chemberta_name)
        self.chemberta_config = AutoConfig.from_pretrained(chemberta_name)
        
        # Freeze ChemBERTa layers initially (can be unfrozen during fine-tuning)
        for param in self.chemberta.parameters():
            param.requires_grad = False
        
        # Feature encoders
        self.position_encoder = PositionalEncoding3D(hidden_dim, max_atoms)
        self.nmr_encoder = NMREncoder(hidden_dim)
        self.atom_encoder = AtomFeatureEncoder(num_atom_types, hidden_dim)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention layers
        self.smiles_to_atoms = CrossAttentionLayer(hidden_dim)
        self.atoms_to_smiles = CrossAttentionLayer(hidden_dim)
        
        # Task-specific heads
        self.nmr_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Predict H and C shifts
        )
        
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # Predict x, y, z coordinates
        )
        
        self.atom_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_atom_types)  # Classify atom type
        )
        
        # SMILES position predictor (maps atoms to SMILES token positions)
        self.smiles_position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Predict position in SMILES
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                coords: torch.Tensor,
                atom_types: torch.Tensor,
                atom_mask: torch.Tensor,
                nmr_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model
        
        Args:
            input_ids: (batch_size, seq_len) - tokenized SMILES
            attention_mask: (batch_size, seq_len) - attention mask for SMILES
            coords: (batch_size, max_atoms, 3) - 3D coordinates
            atom_types: (batch_size, max_atoms) - atom type indices
            atom_mask: (batch_size, max_atoms) - mask for valid atoms
            nmr_features: Dict containing NMR data
        
        Returns:
            Dict containing predictions for various tasks
        """
        batch_size = input_ids.shape[0]
        
        # 1. Get ChemBERTa embeddings for SMILES
        chemberta_outputs = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        smiles_embeddings = chemberta_outputs.last_hidden_state  # (B, seq_len, hidden)
        
        # 2. Encode 3D positions
        position_features = self.position_encoder(coords, atom_mask)  # (B, max_atoms, hidden)
        
        # 3. Encode NMR data
        nmr_encoded = self.nmr_encoder(
            nmr_features['h_shifts'],
            nmr_features['c_shifts'],
            nmr_features['h_mask'],
            nmr_features['c_mask']
        )  # (B, max_atoms, hidden)
        
        # 4. Encode atom types
        atom_features = self.atom_encoder(atom_types)  # (B, max_atoms, hidden)
        
        # 5. Fuse atom-level features
        atom_representations = torch.cat([
            position_features,
            nmr_encoded,
            atom_features
        ], dim=-1)  # (B, max_atoms, hidden*3)
        
        atom_representations = self.feature_fusion(atom_representations)  # (B, max_atoms, hidden)
        
        # 6. Cross-attention between SMILES and atoms
        # Create attention mask for atoms (inverted atom_mask for padding)
        atom_attn_mask = ~atom_mask.bool()
        
        # SMILES tokens attend to atoms
        enhanced_smiles = self.smiles_to_atoms(
            smiles_embeddings, 
            atom_representations, 
            atom_representations,
            atom_attn_mask
        )
        
        # Atoms attend to SMILES tokens
        enhanced_atoms = self.atoms_to_smiles(
            atom_representations,
            enhanced_smiles,
            enhanced_smiles,
            ~attention_mask.bool()
        )
        
        # 7. Make predictions
        predictions = {
            'nmr_shifts': self.nmr_predictor(enhanced_atoms),  # (B, max_atoms, 2)
            'positions': self.position_predictor(enhanced_atoms),  # (B, max_atoms, 3)
            'atom_types': self.atom_classifier(enhanced_atoms),  # (B, max_atoms, num_types)
            'smiles_positions': self.smiles_position_predictor(enhanced_atoms),  # (B, max_atoms, 1)
            'atom_representations': enhanced_atoms,  # (B, max_atoms, hidden)
            'smiles_representations': enhanced_smiles  # (B, seq_len, hidden)
        }
        
        return predictions