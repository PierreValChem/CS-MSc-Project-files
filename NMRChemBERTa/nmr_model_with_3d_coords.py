"""
Enhanced SMILES-to-NMR model that properly uses 3D coordinates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict
import math
import logging

logger = logging.getLogger(__name__)


class DistanceMatrix(nn.Module):
    """Compute distance matrix from 3D coordinates"""
    def forward(self, coords):
        # coords: (batch, n_atoms, 3)
        # Compute pairwise distances
        coords_1 = coords.unsqueeze(2)  # (batch, n_atoms, 1, 3)
        coords_2 = coords.unsqueeze(1)  # (batch, 1, n_atoms, 3)
        distances = torch.norm(coords_1 - coords_2, dim=-1)  # (batch, n_atoms, n_atoms)
        return distances


class GaussianDistanceEncoding(nn.Module):
    """Encode distances using Gaussian basis functions"""
    def __init__(self, num_gaussians=50, max_distance=10.0):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.max_distance = max_distance
        
        # Create Gaussian centers
        centers = torch.linspace(0, max_distance, num_gaussians)
        self.register_buffer('centers', centers)
        
        # Learnable widths
        self.widths = nn.Parameter(torch.ones(num_gaussians) * 0.5)
        
    def forward(self, distances):
        # distances: (batch, n_atoms, n_atoms)
        # Expand for broadcasting
        distances = distances.unsqueeze(-1)  # (batch, n_atoms, n_atoms, 1)
        centers = self.centers.view(1, 1, 1, -1)  # (1, 1, 1, num_gaussians)
        widths = self.widths.view(1, 1, 1, -1)
        
        # Compute Gaussian features
        gaussian_features = torch.exp(-(distances - centers)**2 / (2 * widths**2))
        return gaussian_features  # (batch, n_atoms, n_atoms, num_gaussians)


class SpatialTransformer(nn.Module):
    """Transformer that uses 3D spatial information"""
    def __init__(self, hidden_dim, num_heads=8, num_gaussians=50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Distance encoding
        self.distance_encoder = GaussianDistanceEncoding(num_gaussians)
        self.distance_projection = nn.Linear(num_gaussians, num_heads)
        
        # Standard transformer components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x, distances, mask=None):
        # x: (batch, n_atoms, hidden_dim)
        # distances: (batch, n_atoms, n_atoms)
        
        batch_size, n_atoms, _ = x.size()
        
        # Self-attention with distance bias
        residual = x
        x = self.norm1(x)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, n_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, n_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, n_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add distance bias
        distance_features = self.distance_encoder(distances)  # (batch, n_atoms, n_atoms, num_gaussians)
        distance_bias = self.distance_projection(distance_features)  # (batch, n_atoms, n_atoms, num_heads)
        distance_bias = distance_bias.permute(0, 3, 1, 2)  # (batch, num_heads, n_atoms, n_atoms)
        
        scores = scores + distance_bias
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_atoms)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # (batch, num_heads, n_atoms, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_atoms, self.hidden_dim)
        out = self.o_proj(out)
        
        # Residual connection
        x = residual + out
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class CoordinateEncoder(nn.Module):
    """Encode 3D coordinates into features"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # Direct coordinate encoding
        self.coord_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # Relative position encoding
        self.relative_pos_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
    def forward(self, coords):
        # Direct encoding
        coord_features = self.coord_net(coords)
        
        # Relative to center of mass
        center_of_mass = coords.mean(dim=1, keepdim=True)
        relative_coords = coords - center_of_mass
        relative_features = self.relative_pos_net(relative_coords)
        
        return coord_features + relative_features


class SMILESNMRModelWith3D(nn.Module):
    """SMILES-to-NMR model that properly integrates 3D coordinates"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_atoms = config.model.max_atoms
        
        # ChemBERTa for SMILES
        self.chemberta = AutoModel.from_pretrained(config.model.chemberta_name)
        self.chemberta_dim = self.chemberta.config.hidden_size
        
        # Coordinate processing
        self.coord_encoder = CoordinateEncoder(128)
        self.distance_matrix = DistanceMatrix()
        
        # Project ChemBERTa output
        self.smiles_projection = nn.Sequential(
            nn.Linear(self.chemberta_dim, config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout)
        )
        
        # Combine SMILES and spatial features
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.model.hidden_dim + 128, config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout)
        )
        
        # Spatial transformer layers
        self.spatial_layers = nn.ModuleList([
            SpatialTransformer(
                config.model.hidden_dim,
                num_heads=config.model.num_attention_heads
            )
            for _ in range(4)  # Use 4 layers
        ])
        
        # Atom-type specific embeddings
        self.atom_type_embed = nn.Embedding(10, 64)  # 10 atom types
        
        # Final prediction heads with atom-type awareness
        self.h_nmr_net = nn.Sequential(
            nn.Linear(config.model.hidden_dim + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.c_nmr_net = nn.Sequential(
            nn.Linear(config.model.hidden_dim + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Optionally freeze ChemBERTa
        if config.model.freeze_chemberta:
            for param in self.chemberta.parameters():
                param.requires_grad = False
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, coords, atom_types, atom_mask, **kwargs):
        batch_size = input_ids.size(0)
        
        # Process SMILES with ChemBERTa
        chemberta_out = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_features = chemberta_out.last_hidden_state
        
        # Pool to get molecular representation
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (sequence_features * mask_expanded).sum(1) / mask_expanded.sum(1)
        
        # Project and expand to all atoms
        smiles_features = self.smiles_projection(pooled)
        smiles_features = smiles_features.unsqueeze(1).expand(-1, self.max_atoms, -1)
        
        # Process 3D coordinates
        coord_features = self.coord_encoder(coords)
        
        # Compute distance matrix
        distances = self.distance_matrix(coords)
        
        # Combine features
        combined_features = torch.cat([smiles_features, coord_features], dim=-1)
        atom_features = self.feature_fusion(combined_features)
        
        # Apply spatial transformers
        for spatial_layer in self.spatial_layers:
            atom_features = spatial_layer(atom_features, distances, atom_mask)
        
        # Get atom type embeddings
        atom_type_embeds = self.atom_type_embed(atom_types.clamp(min=0, max=9))
        
        # Concatenate with atom features
        final_features = torch.cat([atom_features, atom_type_embeds], dim=-1)
        
        # Predict NMR shifts
        h_shifts = self.h_nmr_net(final_features).squeeze(-1)
        c_shifts = self.c_nmr_net(final_features).squeeze(-1)
        
        # Stack predictions
        nmr_shifts = torch.stack([h_shifts, c_shifts], dim=-1)
        
        return {
            'nmr_shifts': nmr_shifts,
            'atom_features': atom_features,
            'distances': distances
        }


class SpatialAwareLoss(nn.Module):
    """Loss function that considers spatial relationships"""
    def __init__(self, config):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='none', delta=1.0)
        self.config = config
        
    def forward(self, predictions, targets, masks):
        pred_shifts = predictions['nmr_shifts']
        target_shifts = targets['nmr_shifts']
        nmr_mask = masks['nmr_mask']
        
        # Basic prediction loss
        pred_loss = self.huber(pred_shifts, target_shifts)
        
        # Mask and average
        masked_loss = pred_loss * nmr_mask
        num_valid = nmr_mask.sum() + 1e-6
        
        total_loss = masked_loss.sum() / num_valid
        
        # Separate H and C losses
        h_loss = (masked_loss[:, :, 0].sum()) / (nmr_mask[:, :, 0].sum() + 1e-6)
        c_loss = (masked_loss[:, :, 1].sum()) / (nmr_mask[:, :, 1].sum() + 1e-6)
        
        return {
            'total_loss': total_loss,
            'nmr_loss': total_loss,
            'h_nmr_loss': h_loss,
            'c_nmr_loss': c_loss
        }


# Example of how to modify your training script
def create_3d_model(config):
    """Create model that uses 3D coordinates"""
    return SMILESNMRModelWith3D(config)


# In your training loop, make sure to pass coordinates:
# predictions = model(
#     input_ids=batch['input_ids'],
#     attention_mask=batch['attention_mask'],
#     coords=batch['coords'],  # 3D coordinates from your data
#     atom_types=batch['atom_types'],
#     atom_mask=batch['atom_mask']
# )