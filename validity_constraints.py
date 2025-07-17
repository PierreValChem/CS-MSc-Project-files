#!/usr/bin/env python3
"""
Valid Molecule Generation Methods for NMR-to-SMILES Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MoleculeValidator:
    """Validates SMILES strings and provides chemical property checks"""
    
    def __init__(self, max_heavy_atoms=150, max_mol_weight=1500):
        self.max_heavy_atoms = max_heavy_atoms
        self.max_mol_weight = max_mol_weight
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is chemically valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def get_molecule_properties(self, smiles: str) -> Optional[Dict]:
        """Get molecular properties if valid"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'mol_weight': Descriptors.MolWt(mol),
            'num_rings': Descriptors.RingCount(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'logp': Crippen.MolLogP(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol)
        }
    
    def is_reasonable_molecule(self, smiles: str) -> Tuple[bool, str]:
        """Check if molecule has reasonable properties"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES"
        
        # Check various properties
        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy > self.max_heavy_atoms:
            return False, f"Too many heavy atoms ({num_heavy} > {self.max_heavy_atoms})"
        
        mol_weight = Descriptors.MolWt(mol)
        if mol_weight > self.max_mol_weight:
            return False, f"Molecular weight too high ({mol_weight:.1f} > {self.max_mol_weight})"
        
        return True, "Valid"


class ValidatedSMILESGenerator:
    """Enhanced SMILES generation with validity constraints"""
    
    def __init__(self, model, tokenizer, validator: MoleculeValidator):
        self.model = model
        self.tokenizer = tokenizer
        self.validator = validator
        
    def generate_with_beam_search(self, nmr_encoding, beam_size=10, max_length=256, 
                                 temperature=1.0, validity_weight=0.5):
        """
        Beam search with validity scoring
        
        Args:
            nmr_encoding: Encoded NMR features
            beam_size: Number of beams to maintain
            max_length: Maximum sequence length
            temperature: Sampling temperature
            validity_weight: Weight for validity scoring (0-1)
        """
        device = nmr_encoding.device
        batch_size = nmr_encoding.size(1)
        
        # Initialize beams
        beams = [{
            'tokens': torch.zeros((1,), dtype=torch.long, device=device),
            'score': 0.0,
            'complete': False,
            'valid_prefixes': 0  # Track valid chemical prefixes
        } for _ in range(beam_size)]
        
        completed_sequences = []
        
        for step in range(max_length - 1):
            all_candidates = []
            
            for beam in beams:
                if beam['complete']:
                    continue
                
                # Get model predictions
                positions = torch.arange(beam['tokens'].size(0), device=device).unsqueeze(0)
                token_emb = self.model.token_embedding(beam['tokens'].unsqueeze(0))
                pos_emb = self.model.position_embedding(positions)
                target_emb = self.model.dropout(token_emb + pos_emb)
                
                target_emb = target_emb.transpose(0, 1)
                decoded = self.model.decoder(target_emb, nmr_encoding)
                
                logits = self.model.output_projection(decoded[-1]) / temperature
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top k tokens
                top_k_log_probs, top_k_indices = torch.topk(log_probs[0], beam_size)
                
                for k in range(beam_size):
                    token = top_k_indices[k]
                    token_log_prob = top_k_log_probs[k].item()
                    
                    # Create new candidate
                    new_tokens = torch.cat([beam['tokens'], token.unsqueeze(0)])
                    
                    # Calculate validity bonus
                    partial_smiles = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
                    validity_bonus = self._calculate_validity_bonus(partial_smiles)
                    
                    # Combined score
                    new_score = beam['score'] + token_log_prob + validity_weight * validity_bonus
                    
                    candidate = {
                        'tokens': new_tokens,
                        'score': new_score,
                        'complete': token.item() == self.tokenizer.eos_token_id,
                        'valid_prefixes': beam['valid_prefixes'] + (1 if validity_bonus > 0 else 0)
                    }
                    
                    all_candidates.append(candidate)
            
            # Select top beams
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = []
            
            for candidate in all_candidates:
                if candidate['complete']:
                    completed_sequences.append(candidate)
                else:
                    beams.append(candidate)
                
                if len(beams) >= beam_size:
                    break
            
            # If all beams are complete, stop
            if not beams:
                break
        
        # Add remaining beams to completed sequences
        completed_sequences.extend(beams)
        
        # Sort by score and return top valid SMILES
        completed_sequences.sort(key=lambda x: x['score'], reverse=True)
        
        results = []
        for seq in completed_sequences:
            smiles = self.tokenizer.decode(seq['tokens'].tolist(), skip_special_tokens=True)
            if self.validator.is_valid_smiles(smiles):
                results.append({
                    'smiles': smiles,
                    'score': seq['score'],
                    'valid_prefixes': seq['valid_prefixes']
                })
        
        return results
    
    def _calculate_validity_bonus(self, partial_smiles: str) -> float:
        """Calculate bonus for valid chemical prefixes"""
        # Simple heuristic: check if partial SMILES has valid patterns
        # This is a simplified version - you can make it more sophisticated
        
        if not partial_smiles:
            return 0.0
        
        # Check for balanced parentheses
        open_count = partial_smiles.count('(')
        close_count = partial_smiles.count(')')
        if close_count > open_count:
            return -1.0  # Penalty for invalid structure
        
        # Check for balanced brackets
        open_bracket = partial_smiles.count('[')
        close_bracket = partial_smiles.count(']')
        if close_bracket > open_bracket:
            return -1.0
        
        # Bonus for valid element symbols
        valid_elements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        element_count = sum(1 for elem in valid_elements if elem in partial_smiles)
        
        return min(element_count * 0.1, 1.0)


class ValidityAwareLoss(nn.Module):
    """Simplified loss function with strong validity enforcement"""
    
    def __init__(self, base_criterion, validity_weight=1.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.validity_weight = validity_weight
        self.validator = MoleculeValidator()
    
    def forward(self, outputs, targets, predictions=None):
        """
        Calculate loss with validity penalty
        
        Args:
            outputs: Model logits [batch, seq_len, vocab_size]
            targets: Target token ids [batch, seq_len]
            predictions: Optional generated SMILES for validity checking
        """
        # Base cross-entropy loss
        base_loss = self.base_criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        )
        
        # Simple validity penalty
        if predictions is not None:
            invalid_count = 0
            batch_size = len(predictions)
            
            for pred_smiles in predictions:
                # Check if SMILES is valid
                if not self.validator.is_valid_smiles(pred_smiles):
                    invalid_count += 1
            
            # Strong penalty for invalid molecules
            validity_penalty = (invalid_count / batch_size) * 2.0  # 2x penalty
            
            # Combined loss
            total_loss = base_loss + self.validity_weight * validity_penalty
            
            return total_loss, {
                'base_loss': base_loss.item(),
                'validity_penalty': validity_penalty,
                'validity_rate': 1.0 - (invalid_count / batch_size)
            }
        
        return base_loss, {'base_loss': base_loss.item()}

# Also add a function to pre-filter training data for valid SMILES:
def filter_valid_smiles(data_list, logger=None):
    """Filter out invalid SMILES from training data"""
    from rdkit import Chem
    
    valid_data = []
    invalid_count = 0
    
    for data in data_list:
        smiles = data.get('canonical_smiles', '')
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Additional check - ensure it can be converted back
                canonical_smiles = Chem.MolToSmiles(mol)
                if canonical_smiles:
                    data['canonical_smiles'] = canonical_smiles  # Use canonical form
                    valid_data.append(data)
                else:
                    invalid_count += 1
            else:
                invalid_count += 1
        except:
            invalid_count += 1
    
    if logger:
        logger.info(f"Filtered dataset: {len(data_list)} -> {len(valid_data)} "
                   f"({invalid_count} invalid SMILES removed)")
    
    return valid_data


class ConstrainedDecoding:
    """Implement constrained decoding to ensure valid molecules"""
    
    def __init__(self, tokenizer, validator):
        self.tokenizer = tokenizer
        self.validator = validator
        self.forbidden_patterns = self._build_forbidden_patterns()
    
    def _build_forbidden_patterns(self):
        """Build patterns that should never appear in valid SMILES"""
        return [
            # Invalid valence patterns
            'CC(C)(C)(C)C',  # Carbon with 5 bonds
            'O(O)(O)',       # Oxygen with 3 bonds
            'N(N)(N)(N)',    # Nitrogen with 4 bonds (unless charged)
            # Add more patterns as needed
        ]
    
    def get_valid_next_tokens(self, partial_tokens, vocab_size):
        """Get mask of valid next tokens given partial sequence"""
        partial_smiles = self.tokenizer.decode(partial_tokens, skip_special_tokens=True)
        
        # Create mask (1 = valid, 0 = invalid)
        mask = torch.ones(vocab_size)
        
        # Mask out tokens that would create invalid patterns
        for token_id in range(vocab_size):
            token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            
            # Check if adding this token would create invalid SMILES
            test_smiles = partial_smiles + token_str
            
            # Quick checks
            if test_smiles.count(')') > test_smiles.count('('):
                mask[token_id] = 0
            elif test_smiles.count(']') > test_smiles.count('['):
                mask[token_id] = 0
            
        return mask


class EnhancedNMRToSMILES(nn.Module):
    """Enhanced model with validity-aware generation"""
    
    def __init__(self, base_model, validator, constrained_decoding=True):
        super().__init__()
        self.base_model = base_model
        self.validator = validator
        self.constrained_decoding = constrained_decoding
        
        # Copy attributes from base model
        self.nmr_encoder = base_model.nmr_encoder
        self.token_embedding = base_model.token_embedding
        self.position_embedding = base_model.position_embedding
        self.decoder = base_model.decoder
        self.output_projection = base_model.output_projection
        self.hidden_dim = base_model.hidden_dim
        self.dropout = base_model.dropout
    
    def forward(self, h_features, c_features, global_features, 
                target_ids=None, target_mask=None):
        """Forward pass - same as base model"""
        return self.base_model(h_features, c_features, global_features, 
                              target_ids, target_mask)
    
    def generate_valid(self, h_features, c_features, global_features, 
                      max_attempts=10, beam_size=5):
        """Generate valid SMILES with multiple strategies"""
        nmr_encoding = self.nmr_encoder(h_features, c_features, global_features)
        nmr_encoding = nmr_encoding.unsqueeze(0)
        
        # Try beam search first
        generator = ValidatedSMILESGenerator(self, self.base_model.tokenizer, self.validator)
        results = generator.generate_with_beam_search(
            nmr_encoding, beam_size=beam_size
        )
        
        if results:
            return results[0]['smiles']
        
        # Fallback to sampling with validity checks
        for _ in range(max_attempts):
            generated = self._generate(nmr_encoding)
            smiles = self.base_model.tokenizer.decode(generated[0], skip_special_tokens=True)
            
            if self.validator.is_valid_smiles(smiles):
                return smiles
        
        # If all attempts fail, return the best attempt
        return smiles


# Integration into training
def create_validity_aware_trainer(model, tokenizer, device='cuda'):
    """Create trainer with validity awareness"""
    
    # Create validator
    validator = MoleculeValidator()
    
    # Wrap model with enhanced version
    enhanced_model = EnhancedNMRToSMILES(model, validator)
    
    # Create validity-aware loss
    base_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    validity_loss = ValidityAwareLoss(base_criterion, validity_weight=0.1)
    
    return enhanced_model, validity_loss, validator
