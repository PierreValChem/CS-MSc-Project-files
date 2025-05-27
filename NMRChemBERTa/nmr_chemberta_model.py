"""
Refactored NMR-ChemBERTa Model
Using modular components for better maintainability
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional
import logging

from model_components import (
    PositionalEncoding3D,
    NMREncoder,
    AtomFeatureEncoder,
    CrossAttentionLayer,
    TaskHeads,
    FeatureFusion,
    apply_gradient_checkpointing
)

logger = logging.getLogger(__name__)


class NMRChemBERTa(nn.Module):
    """
    Main NMR-ChemBERTa model combining:
    - Pre-trained ChemBERTa for SMILES understanding
    - 3D structural information
    - NMR spectroscopic data
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.max_atoms = config.model.max_atoms
        
        # Load pre-trained ChemBERTa
        self._setup_chemberta()
        
        # Initialize feature encoders
        self._setup_encoders()
        
        # Initialize cross-attention layers
        self._setup_cross_attention()
        
        # Initialize task heads
        self._setup_task_heads()
        
        # Apply gradient checkpointing if requested
        if config.hardware.gradient_checkpointing:
            self._apply_gradient_checkpointing()
    
    def _setup_chemberta(self):
        """Initialize ChemBERTa backbone"""
        self.chemberta = AutoModel.from_pretrained(self.config.model.chemberta_name)
        self.chemberta_config = AutoConfig.from_pretrained(self.config.model.chemberta_name)
        
        # Freeze ChemBERTa if requested
        if self.config.model.freeze_chemberta:
            for param in self.chemberta.parameters():
                param.requires_grad = False
            logger.info("ChemBERTa parameters frozen")
        else:
            logger.info("ChemBERTa parameters will be fine-tuned")
    
    def _setup_encoders(self):
        """Initialize feature encoders"""
        self.position_encoder = PositionalEncoding3D(
            self.hidden_dim, 
            self.max_atoms
        )
        
        self.nmr_encoder = NMREncoder(
            self.hidden_dim,
            self.config.model.dropout
        )
        
        self.atom_encoder = AtomFeatureEncoder(
            self.config.model.num_atom_types,
            self.hidden_dim,
            self.config.model.dropout
        )
        
        # Feature fusion layer
        self.feature_fusion = FeatureFusion(
            input_dim=self.hidden_dim * 3,
            output_dim=self.hidden_dim,
            dropout=self.config.model.dropout
        )
    
    def _setup_cross_attention(self):
        """Initialize cross-attention layers"""
        self.smiles_to_atoms = CrossAttentionLayer(
            self.hidden_dim,
            self.config.model.num_attention_heads,
            self.config.model.dropout
        )
        
        self.atoms_to_smiles = CrossAttentionLayer(
            self.hidden_dim,
            self.config.model.num_attention_heads,
            self.config.model.dropout
        )
    
    def _setup_task_heads(self):
        """Initialize task-specific prediction heads"""
        self.task_heads = TaskHeads(
            self.hidden_dim,
            self.config.model.num_atom_types,
            self.config.model.dropout
        )
    
    def _apply_gradient_checkpointing(self):
        """Apply gradient checkpointing to save memory"""
        layers_to_checkpoint = [
            'position_encoder',
            'nmr_encoder',
            'smiles_to_atoms',
            'atoms_to_smiles'
        ]
        apply_gradient_checkpointing(self, layers_to_checkpoint)
        logger.info("Gradient checkpointing applied")
    
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
        # 1. Get ChemBERTa embeddings for SMILES
        smiles_embeddings = self._encode_smiles(input_ids, attention_mask)
        
        # 2. Encode atom-level features
        atom_representations = self._encode_atoms(
            coords, atom_types, atom_mask, nmr_features
        )
        
        # 3. Cross-attention between SMILES and atoms
        enhanced_smiles, enhanced_atoms = self._cross_attention(
            smiles_embeddings, atom_representations, attention_mask, atom_mask
        )
        
        # 4. Make predictions using task heads
        predictions = self.task_heads(enhanced_atoms)
        
        # 5. Add representations to output
        predictions.update({
            'atom_representations': enhanced_atoms,
            'smiles_representations': enhanced_smiles
        })
        
        return predictions
    
    def _encode_smiles(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode SMILES using ChemBERTa"""
        chemberta_outputs = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return chemberta_outputs.last_hidden_state
    
    def _encode_atoms(self, 
                     coords: torch.Tensor,
                     atom_types: torch.Tensor,
                     atom_mask: torch.Tensor,
                     nmr_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode atom-level features"""
        # Encode 3D positions
        position_features = self.position_encoder(coords, atom_mask)
        
        # Encode NMR data
        nmr_encoded = self.nmr_encoder(
            nmr_features['h_shifts'],
            nmr_features['c_shifts'],
            nmr_features['h_mask'],
            nmr_features['c_mask']
        )
        
        # Encode atom types
        atom_features = self.atom_encoder(atom_types)
        
        # Fuse all atom-level features
        atom_representations = self.feature_fusion(
            position_features,
            nmr_encoded,
            atom_features
        )
        
        # Apply atom mask
        return atom_representations * atom_mask.unsqueeze(-1)
    
    def _cross_attention(self,
                        smiles_embeddings: torch.Tensor,
                        atom_representations: torch.Tensor,
                        attention_mask: torch.Tensor,
                        atom_mask: torch.Tensor) -> tuple:
        """Apply cross-attention between SMILES and atoms"""
        # Create attention masks (inverted for padding)
        atom_attn_mask = ~atom_mask.bool()
        smiles_attn_mask = ~attention_mask.bool()
        
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
            smiles_attn_mask
        )
        
        return enhanced_smiles, enhanced_atoms
    
    def unfreeze_chemberta(self):
        """Unfreeze ChemBERTa for fine-tuning"""
        for param in self.chemberta.parameters():
            param.requires_grad = True
        logger.info("ChemBERTa parameters unfrozen for fine-tuning")
    
    def freeze_chemberta(self):
        """Freeze ChemBERTa parameters"""
        for param in self.chemberta.parameters():
            param.requires_grad = False
        logger.info("ChemBERTa parameters frozen")
    
    def get_parameter_count(self):
        """Get number of trainable and total parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }