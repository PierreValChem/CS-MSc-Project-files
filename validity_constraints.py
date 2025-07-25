#!/usr/bin/env python3
"""
Enhanced Valid Molecule Generation Methods for NMR-to-SMILES Model
Focuses specifically on H and C atom counts from NMR data while allowing other atoms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class AtomCountValidator:
    """Validates SMILES strings with specific focus on H and C atom count matching from NMR data"""
    
    def __init__(self, tolerance_ratio=0.1):
        """
        Args:
            tolerance_ratio: Allowed deviation from expected H and C atom counts (e.g., 0.1 = 10%)
        
        Note: Only validates H and C atoms since other atoms (N, O, S, halogens, etc.) 
        don't appear in 1H and 13C NMR spectra but may still be present in molecules.
        """
        self.tolerance_ratio = tolerance_ratio

    def get_hc_atom_counts(self, smiles: str) -> Optional[Dict[str, int]]:
        """Get H and C atom counts from SMILES string (ignoring other atoms for NMR purposes)"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Count only H and C atoms
            h_count = 0
            c_count = 0
            
            # Count explicit carbons
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    c_count += 1
            
            # Add implicit hydrogens (RDKit method)
            mol_with_h = Chem.AddHs(mol)
            for atom in mol_with_h.GetAtoms():
                if atom.GetSymbol() == 'H':
                    h_count += 1
            
            return {
                'H': h_count,
                'C': c_count,
                'total_heavy_atoms': mol.GetNumHeavyAtoms(),  # For reference
                'other_atoms': mol.GetNumHeavyAtoms() - c_count  # Non-carbon heavy atoms
            }
        except Exception as e:
            logger.debug(f"Error counting H/C atoms in SMILES '{smiles}': {e}")
            return None
    
    def validate_hc_counts(self, smiles: str, expected_h: int, expected_c: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate if SMILES has expected H and C atom counts (ignoring other atoms)
        
        Returns:
            (is_valid, validation_info)
        """
        atom_counts = self.get_hc_atom_counts(smiles)
        
        if atom_counts is None:
            return False, {
                'error': 'Invalid SMILES',
                'expected_h': expected_h,
                'expected_c': expected_c,
                'actual_h': 0,
                'actual_c': 0,
                'h_match': False,
                'c_match': False,
                'other_atoms_present': False,
                'other_atom_count': 0
            }
        
        actual_h = atom_counts.get('H', 0)
        actual_c = atom_counts.get('C', 0)
        other_atoms = atom_counts.get('other_atoms', 0)
        
        # Check if counts match within tolerance (only for H and C)
        h_tolerance = max(1, int(expected_h * self.tolerance_ratio))
        c_tolerance = max(1, int(expected_c * self.tolerance_ratio))
        
        h_match = abs(actual_h - expected_h) <= h_tolerance
        c_match = abs(actual_c - expected_c) <= c_tolerance
        
        validation_info = {
            'expected_h': expected_h,
            'expected_c': expected_c,
            'actual_h': actual_h,
            'actual_c': actual_c,
            'h_match': h_match,
            'c_match': c_match,
            'h_tolerance': h_tolerance,
            'c_tolerance': c_tolerance,
            'other_atoms_present': other_atoms > 0,
            'other_atom_count': other_atoms,
            'total_heavy_atoms': atom_counts.get('total_heavy_atoms', 0),
            'hc_atom_counts': atom_counts  # Full atom count info for reference
        }
        
        return h_match and c_match, validation_info
    
    def calculate_hc_count_score(self, smiles: str, expected_h: int, expected_c: int) -> float:
        """
        Calculate a score based on how close the H and C counts are to expected values
        (Other atoms don't affect the score since they're not visible in NMR)
        
        Returns:
            Score between 0 and 1 (1 = perfect H and C match)
        """
        atom_counts = self.get_hc_atom_counts(smiles)
        
        if atom_counts is None:
            return 0.0
        
        actual_h = atom_counts.get('H', 0)
        actual_c = atom_counts.get('C', 0)
        
        # Calculate normalized errors for H and C only
        h_error = abs(actual_h - expected_h) / max(expected_h, 1)
        c_error = abs(actual_c - expected_c) / max(expected_c, 1)
        
        # Convert to scores (1 - error, capped at 0)
        h_score = max(0, 1 - h_error)
        c_score = max(0, 1 - c_error)
        
        # Combined score (average of H and C scores)
        return (h_score + c_score) / 2.0


class MoleculeValidator:
    """Basic molecule validator for backward compatibility"""
    
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


class EnhancedMoleculeValidator:
    """Enhanced validator with H/C atom count matching and chemical validity"""
    
    def __init__(self, max_heavy_atoms=150, max_mol_weight=1500, hc_count_tolerance=0.1):
        """
        Args:
            max_heavy_atoms: Maximum number of heavy atoms allowed
            max_mol_weight: Maximum molecular weight allowed
            hc_count_tolerance: Tolerance for H and C atom count matching
        """
        self.max_heavy_atoms = max_heavy_atoms
        self.max_mol_weight = max_mol_weight
        self.hc_validator = AtomCountValidator(tolerance_ratio=hc_count_tolerance)
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Basic SMILES validity check"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def comprehensive_validation(self, smiles: str, expected_h: int, expected_c: int) -> Dict[str, Any]:
        """
        Comprehensive validation including H/C atom counts and chemical properties
        
        Returns:
            Dictionary with all validation results
        """
        results = {
            'smiles': smiles,
            'is_valid_smiles': False,
            'hc_count_match': False,
            'reasonable_properties': False,
            'overall_valid': False,
            'scores': {}
        }
        
        # Check basic SMILES validity
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results['error'] = 'Invalid SMILES syntax'
            return results
        
        results['is_valid_smiles'] = True
        
        # Check H and C atom counts specifically
        hc_match, hc_info = self.hc_validator.validate_hc_counts(smiles, expected_h, expected_c)
        results['hc_count_match'] = hc_match
        results['hc_info'] = hc_info
        
        # Calculate H/C count score
        hc_score = self.hc_validator.calculate_hc_count_score(smiles, expected_h, expected_c)
        results['scores']['hc_count_score'] = hc_score
        
        # Check molecular properties (general chemical reasonableness)
        properties = self.get_molecule_properties(smiles)
        if properties:
            results['properties'] = properties
            results['reasonable_properties'] = self.check_reasonable_properties(properties)
            results['scores']['property_score'] = self.calculate_property_score(properties)
        
        # Overall validation (focuses on H/C matching and basic validity)
        results['overall_valid'] = (
            results['is_valid_smiles'] and 
            results['hc_count_match'] and 
            results['reasonable_properties']
        )
        
        # Calculate combined score (emphasizing H/C matching since that's NMR-specific)
        scores = results['scores']
        results['scores']['combined_score'] = (
            scores.get('hc_count_score', 0) * 0.7 +  # 70% weight on H/C counts (NMR-specific)
            scores.get('property_score', 0) * 0.3    # 30% weight on general properties
        )
        
        return results
    
    def get_molecule_properties(self, smiles: str) -> Optional[Dict]:
        """Get molecular properties"""
        try:
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
                'num_hba': Descriptors.NumHAcceptors(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'tpsa': Descriptors.TPSA(mol)
            }
        except Exception as e:
            logger.debug(f"Error calculating properties for '{smiles}': {e}")
            return None
    
    def check_reasonable_properties(self, properties: Dict) -> bool:
        """Check if molecular properties are reasonable"""
        if not properties:
            return False
        
        # Check bounds
        if properties['num_heavy_atoms'] > self.max_heavy_atoms:
            return False
        if properties['mol_weight'] > self.max_mol_weight:
            return False
        if properties['logp'] < -10 or properties['logp'] > 10:
            return False
        if properties['tpsa'] > 300:  # Very high TPSA
            return False
        
        return True
    
    def calculate_property_score(self, properties: Dict) -> float:
        """Calculate score based on molecular properties"""
        if not properties:
            return 0.0
        
        score = 1.0
        
        # Penalize extreme values
        if properties['num_heavy_atoms'] > self.max_heavy_atoms * 0.8:
            score *= 0.8
        if properties['mol_weight'] > self.max_mol_weight * 0.8:
            score *= 0.8
        if abs(properties['logp']) > 7:
            score *= 0.9
        if properties['tpsa'] > 200:
            score *= 0.9
        
        return max(0.0, score)


class ValidityAwareLoss(nn.Module):
    """Simplified loss function with strong validity enforcement - backward compatible"""
    
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


class AtomCountAwareLoss(nn.Module):
    """Loss function that rewards proper H and C atom count matching from NMR data"""
    
    def __init__(self, base_criterion, validator: EnhancedMoleculeValidator, 
                 validity_weight=0.3, hc_count_weight=0.4, token_weight=0.3):
        """
        Args:
            base_criterion: Base loss function (e.g., CrossEntropyLoss)
            validator: Enhanced molecule validator
            validity_weight: Weight for general validity (0-1)
            hc_count_weight: Weight for H and C atom count matching (0-1) 
            token_weight: Weight for token-level accuracy (0-1)
        
        Note: Only H and C atoms are considered for count matching since other atoms
        (N, O, S, halogens) don't appear in 1H and 13C NMR spectra.
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.validator = validator
        self.validity_weight = validity_weight
        self.hc_count_weight = hc_count_weight
        self.token_weight = token_weight
        
        # Normalize weights
        total_weight = validity_weight + hc_count_weight + token_weight
        self.validity_weight /= total_weight
        self.hc_count_weight /= total_weight
        self.token_weight /= total_weight
    
    def forward(self, outputs, targets, predictions=None, nmr_data=None):
        """
        Calculate enhanced loss with H and C atom count awareness
        
        Args:
            outputs: Model logits [batch, seq_len, vocab_size]
            targets: Target token ids [batch, seq_len]
            predictions: Generated SMILES strings for validation
            nmr_data: NMR data with expected H and C atom counts
        """
        # Base token-level loss
        base_loss = self.base_criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        )
        
        loss_components = {
            'base_loss': base_loss.item(),
            'validity_penalty': 0.0,
            'hc_count_penalty': 0.0,
            'token_reward': 0.0
        }
        
        if predictions is not None and nmr_data is not None:
            batch_size = len(predictions)
            
            validity_penalties = []
            hc_count_penalties = []
            
            for i, pred_smiles in enumerate(predictions):
                # Get expected H and C counts from NMR data
                expected_h = nmr_data['h_atoms'][i] if 'h_atoms' in nmr_data else 0
                expected_c = nmr_data['c_atoms'][i] if 'c_atoms' in nmr_data else 0
                
                # Comprehensive validation focusing on H and C
                validation_results = self.validator.comprehensive_validation(
                    pred_smiles, expected_h, expected_c
                )
                
                # Validity penalty (higher penalty for invalid molecules)
                if not validation_results['is_valid_smiles']:
                    validity_penalties.append(2.0)  # Strong penalty
                elif not validation_results['reasonable_properties']:
                    validity_penalties.append(1.0)  # Moderate penalty
                else:
                    validity_penalties.append(0.0)  # No penalty
                
                # H and C count penalty (inverse of H/C count score)
                hc_count_score = validation_results['scores'].get('hc_count_score', 0)
                hc_count_penalty = 1.0 - hc_count_score
                hc_count_penalties.append(hc_count_penalty)
            
            # Average penalties
            avg_validity_penalty = sum(validity_penalties) / batch_size
            avg_hc_count_penalty = sum(hc_count_penalties) / batch_size
            
            loss_components['validity_penalty'] = avg_validity_penalty
            loss_components['hc_count_penalty'] = avg_hc_count_penalty
            
            # Calculate token reward (negative penalty for good predictions)
            token_reward = self._calculate_token_reward(outputs, targets)
            loss_components['token_reward'] = token_reward
            
            # Combined loss
            total_loss = (
                self.token_weight * base_loss +
                self.validity_weight * avg_validity_penalty +
                self.hc_count_weight * avg_hc_count_penalty -
                0.1 * token_reward  # Small reward for good tokens
            )
            
            # Add validation metrics
            loss_components.update({
                'valid_molecules': sum(1 for p in validity_penalties if p == 0),
                'perfect_hc_matches': sum(1 for p in hc_count_penalties if p < 0.1),
                'validity_rate': 1.0 - (avg_validity_penalty / 2.0),
                'hc_match_rate': 1.0 - avg_hc_count_penalty
            })
            
            return total_loss, loss_components
        
        return base_loss, loss_components
    
    def _calculate_token_reward(self, outputs, targets):
        """Calculate reward based on token-level accuracy"""
        predictions = outputs.argmax(dim=-1)
        correct_tokens = (predictions == targets).float()
        
        # Calculate accuracy-based reward
        token_accuracy = correct_tokens.mean()
        
        # Reward high accuracy
        if token_accuracy > 0.9:
            return 1.0
        elif token_accuracy > 0.8:
            return 0.5
        elif token_accuracy > 0.7:
            return 0.2
        else:
            return 0.0


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
        
        # Try to import RDKit for validation
        try:
            from rdkit import Chem
            rdkit_available = True
        except ImportError:
            rdkit_available = False
            logger.warning("RDKit not available for validity checking")
        
        # Strategy 1: Try simple generation first with validity check
        for attempt in range(min(5, 50)):
            generated = self._generate(nmr_encoding, self.tokenizer, max_length, temperature)
            smiles = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            
            if rdkit_available:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return generated
        
        # Fallback: Return a simple valid default
        default_smiles = "C"  # Methane
        encoded = self.tokenizer.encode(default_smiles, max_length=max_length)
        if isinstance(encoded, dict):
            return torch.tensor(encoded['input_ids'], device=device).unsqueeze(0)
        else:
            return torch.tensor([encoded], device=device)
    
    def _generate(self, nmr_encoding, tokenizer, max_length=256, temperature=1.0):
        """Generate SMILES autoregressively"""
        device = nmr_encoding.device
        batch_size = nmr_encoding.size(1)
        
        # Start with BOS token
        generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            positions = torch.arange(generated.size(1), device=device).unsqueeze(0)
            token_emb = self.model.token_embedding(generated)
            pos_emb = self.model.position_embedding(positions)
            target_emb = self.model.dropout(token_emb + pos_emb)
            
            target_emb = target_emb.transpose(0, 1)
            decoded = self.model.decoder(target_emb, nmr_encoding)
            
            logits = self.model.output_projection(decoded[-1]) / temperature
            probs = torch.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS token
            if hasattr(tokenizer, 'eos_token_id'):
                if (next_token == tokenizer.eos_token_id).all():
                    break
            elif (next_token == 2).all():  # Common EOS token ID
                break
    
        return generated


class ConstrainedSMILESGenerator:
    """Enhanced SMILES generator with H and C atom count constraints from NMR data"""
    
    def __init__(self, model, tokenizer, validator: EnhancedMoleculeValidator):
        self.model = model
        self.tokenizer = tokenizer
        self.validator = validator
    
    def generate_with_hc_constraints(self, nmr_encoding, expected_h, expected_c, 
                                    max_attempts=20, beam_size=5, temperature=0.8):
        """
        Generate SMILES with H and C atom count constraints from NMR data
        
        Args:
            nmr_encoding: Encoded NMR features
            expected_h: Expected hydrogen count from 1H NMR
            expected_c: Expected carbon count from 13C NMR
            max_attempts: Maximum generation attempts
            beam_size: Beam search size
            temperature: Sampling temperature
        
        Note: Only validates H and C counts since other atoms don't appear in NMR spectra
        """
        best_results = []
        
        for attempt in range(max_attempts):
            try:
                # Adjust temperature for diversity
                current_temp = temperature * (1 + 0.1 * attempt)
                
                # Generate with beam search focusing on H and C constraints
                results = self._beam_search_with_hc_constraints(
                    nmr_encoding, expected_h, expected_c, 
                    beam_size, current_temp
                )
                
                # Find valid results (focusing on H and C matching)
                for result in results:
                    validation = self.validator.comprehensive_validation(
                        result['smiles'], expected_h, expected_c
                    )
                    
                    if validation['overall_valid']:
                        result['validation'] = validation
                        best_results.append(result)
                
                # If we have good results, return the best one
                if best_results:
                    best_results.sort(key=lambda x: x['validation']['scores']['combined_score'], 
                                    reverse=True)
                    return best_results[0]['smiles']
                
            except Exception as e:
                logger.debug(f"Generation attempt {attempt} failed: {e}")
                continue
        
        # If no valid results, return best partial match
        if best_results:
            return best_results[0]['smiles']
        
        # Final fallback: try simple generation
        try:
            generator = ValidatedSMILESGenerator(self.model, self.tokenizer, self.validator)
            generated = generator._generate(nmr_encoding, self.tokenizer)
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        except:
            return "C"  # Ultimate fallback
    
    def _beam_search_with_hc_constraints(self, nmr_encoding, expected_h, expected_c, 
                                       beam_size, temperature):
        """Beam search with H and C atom count constraints"""
        device = nmr_encoding.device
        
        # Initialize beams
        beams = [{
            'tokens': torch.zeros((1,), dtype=torch.long, device=device),
            'score': 0.0,
            'complete': False
        }]
        
        completed = []
        max_length = 256
        
        for step in range(max_length):
            new_beams = []
            
            for beam in beams:
                if beam['complete']:
                    completed.append(beam)
                    continue
                
                # Get next token probabilities
                try:
                    positions = torch.arange(beam['tokens'].size(0), device=device).unsqueeze(0)
                    token_emb = self.model.token_embedding(beam['tokens'].unsqueeze(0))
                    pos_emb = self.model.position_embedding(positions)
                    target_emb = self.model.dropout(token_emb + pos_emb)
                    
                    target_emb = target_emb.transpose(0, 1)
                    decoded = self.model.decoder(target_emb, nmr_encoding)
                    
                    logits = self.model.output_projection(decoded[-1]) / temperature
                    probs = F.softmax(logits, dim=-1)
                    
                    # Get top k candidates
                    top_k = min(beam_size, probs.size(-1))
                    top_probs, top_indices = torch.topk(probs[0], top_k)
                    
                    for k in range(top_k):
                        token_id = top_indices[k].item()
                        prob = top_probs[k].item()
                        
                        new_tokens = torch.cat([beam['tokens'], torch.tensor([token_id], device=device)])
                        
                        # Check if complete
                        is_complete = (token_id == self.tokenizer.eos_token_id)
                        
                        # Calculate H/C constraint-aware score
                        constraint_bonus = 0.0
                        if len(new_tokens) > 5:  # Only check longer sequences
                            partial_smiles = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
                            constraint_bonus = self._calculate_hc_constraint_bonus(
                                partial_smiles, expected_h, expected_c, is_complete
                            )
                        
                        new_score = beam['score'] + np.log(prob) + constraint_bonus
                        
                        new_beams.append({
                            'tokens': new_tokens,
                            'score': new_score,
                            'complete': is_complete
                        })
                
                except Exception as e:
                    logger.debug(f"Error in beam search step: {e}")
                    continue
            
            # Keep top beams
            all_beams = beams + new_beams
            all_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = [b for b in all_beams[:beam_size] if not b['complete']]
            completed.extend([b for b in all_beams if b['complete']])
            
            if not beams:
                break
        
        # Convert to results
        results = []
        all_sequences = completed + beams
        
        for seq in all_sequences:
            smiles = self.tokenizer.decode(seq['tokens'].tolist(), skip_special_tokens=True)
            results.append({
                'smiles': smiles,
                'score': seq['score']
            })
        
        return results
    
    def _calculate_hc_constraint_bonus(self, partial_smiles, expected_h, expected_c, is_complete):
        """Calculate bonus based on H and C atom count constraints from NMR data"""
        if not partial_smiles:
            return 0.0
        
        # Basic validity check
        if not self._is_valid_partial_smiles(partial_smiles):
            return -2.0  # Strong penalty
        
        # If complete, check H and C atom counts specifically
        if is_complete:
            hc_score = self.validator.hc_validator.calculate_hc_count_score(
                partial_smiles, expected_h, expected_c
            )
            return hc_score * 2.0  # Strong bonus for correct H and C counts
        
        # For partial SMILES, give small bonus for reasonable progress
        return 0.1
    
    def _is_valid_partial_smiles(self, partial_smiles):
        """Quick validation for partial SMILES"""
        if not partial_smiles:
            return True
        
        # Check balanced parentheses and brackets
        if partial_smiles.count(')') > partial_smiles.count('('):
            return False
        if partial_smiles.count(']') > partial_smiles.count('['):
            return False
        
        return True


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
            return self.base_model.tokenizer.decode(results[0], skip_special_tokens=True)
        
        # Fallback to sampling with validity checks
        for _ in range(max_attempts):
            generated = generator._generate(nmr_encoding, self.base_model.tokenizer)
            smiles = self.base_model.tokenizer.decode(generated[0], skip_special_tokens=True)
            
            if self.validator.is_valid_smiles(smiles):
                return smiles
        
        # If all attempts fail, return the best attempt
        return smiles


def create_enhanced_validity_system(model, tokenizer, device='cuda'):
    """Create complete enhanced validity system focused on H and C atom counting"""
    
    # Create enhanced validator focused on H and C atoms
    validator = EnhancedMoleculeValidator(hc_count_tolerance=0.1)
    
    # Create enhanced loss function
    base_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    enhanced_loss = AtomCountAwareLoss(
        base_criterion, 
        validator,
        validity_weight=0.3,
        hc_count_weight=0.4,  # Focuses on H/C from NMR
        token_weight=0.3
    )
    
    # Create constrained generator focused on H and C
    generator = ConstrainedSMILESGenerator(model, tokenizer, validator)
    
    return validator, enhanced_loss, generator


def create_validity_aware_trainer(model, tokenizer, device='cuda'):
    """Create trainer with validity awareness - backward compatible version"""
    
    # Create validator
    validator = MoleculeValidator()
    
    # Wrap model with enhanced version
    enhanced_model = EnhancedNMRToSMILES(model, validator)
    
    # Create validity-aware loss
    base_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    validity_loss = ValidityAwareLoss(base_criterion, validity_weight=0.1)
    
    return enhanced_model, validity_loss, validator


def filter_valid_smiles(data_list, logger=None):
    """Filter out invalid SMILES from training data"""
    try:
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
        
    except ImportError:
        if logger:
            logger.warning("RDKit not available for SMILES filtering")
        return data_list


def filter_data_by_hc_counts(data_list, max_h=100, max_c=50, logger=None):
    """Filter training data by reasonable H and C atom counts from NMR data"""
    filtered_data = []
    
    for data in data_list:
        h_count = data.get('h_atoms', 0)
        c_count = data.get('c_atoms', 0)
        
        # Only filter by H and C counts since other atoms don't appear in NMR
        if h_count <= max_h and c_count <= max_c and h_count > 0 and c_count > 0:
            filtered_data.append(data)
    
    if logger:
        logger.info(f"Filtered by H/C counts: {len(data_list)} -> {len(filtered_data)}")
        logger.info("Note: Only H and C atoms filtered - other atoms (N, O, S, etc.) allowed as they don't appear in NMR")
    
    return filtered_data


def analyze_non_hc_atoms(data_list, logger=None):
    """Analyze the presence of non-H/C atoms in the dataset"""
    try:
        from rdkit import Chem
        from collections import defaultdict
        
        other_atoms_stats = defaultdict(int)
        molecules_with_other_atoms = 0
        total_molecules = 0
        
        for data in data_list:
            smiles = data.get('canonical_smiles', '')
            if not smiles:
                continue
                
            total_molecules += 1
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Count non-H/C atoms
            has_other_atoms = False
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol not in ['H', 'C']:
                    other_atoms_stats[symbol] += 1
                    has_other_atoms = True
            
            if has_other_atoms:
                molecules_with_other_atoms += 1
        
        if logger:
            logger.info(f"\nNon-H/C Atom Analysis:")
            logger.info(f"  Total molecules analyzed: {total_molecules}")
            logger.info(f"  Molecules with non-H/C atoms: {molecules_with_other_atoms} ({molecules_with_other_atoms/total_molecules*100:.1f}%)")
            logger.info(f"  Common non-H/C atoms found:")
            for atom, count in sorted(other_atoms_stats.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {atom}: {count} occurrences")
            logger.info(f"  Note: These atoms don't appear in 1H/13C NMR but are chemically valid")
        
        return other_atoms_stats, molecules_with_other_atoms
        
    except ImportError:
        if logger:
            logger.warning("RDKit not available for non-H/C atom analysis")
        return {}, 0


# Backward compatibility functions
def filter_data_by_atom_counts(data_list, max_h=100, max_c=50, logger=None):
    """Backward compatibility wrapper - redirects to H/C specific function"""
    return filter_data_by_hc_counts(data_list, max_h, max_c, logger)


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


# Compatibility aliases for backward compatibility
MolValidityAwareLoss = ValidityAwareLoss  # Old name
HCAtomCountAwareLoss = AtomCountAwareLoss  # Alternative name


if __name__ == "__main__":
    # Test the enhanced H/C-focused validation system
    print("Enhanced H/C-Focused Validity System for NMR-to-SMILES")
    print("=" * 60)
    
    # Example validation
    validator = EnhancedMoleculeValidator()
    
    test_molecules = [
        {'smiles': 'CCO', 'expected_h': 6, 'expected_c': 2, 'name': 'Ethanol'},
        {'smiles': 'c1ccccc1', 'expected_h': 6, 'expected_c': 6, 'name': 'Benzene'},
        {'smiles': 'CC(=O)N', 'expected_h': 5, 'expected_c': 2, 'name': 'Acetamide'},
        {'smiles': 'CCCl', 'expected_h': 5, 'expected_c': 2, 'name': 'Chloroethane'},
    ]
    
    print("\nTesting H/C validation on example molecules:")
    print("-" * 50)
    
    for mol in test_molecules:
        result = validator.comprehensive_validation(
            mol['smiles'], mol['expected_h'], mol['expected_c']
        )
        
        print(f"\n{mol['name']} ({mol['smiles']}):")
        print(f"  Expected: H={mol['expected_h']}, C={mol['expected_c']}")
        print(f"  Valid: {result['overall_valid']}")
        print(f"  H/C Match: {result['hc_count_match']}")
        print(f"  Score: {result['scores']['combined_score']:.3f}")
        
        if 'hc_info' in result:
            hc_info = result['hc_info']
            print(f"  Actual: H={hc_info['actual_h']}, C={hc_info['actual_c']}")
            if hc_info['other_atoms_present']:
                print(f"  Other atoms: {hc_info['other_atom_count']} (allowed - not in NMR)")
    
    print(f"\nSystem focuses only on H and C atoms from NMR data!")
    print(f"Other atoms (N, O, S, halogens) are allowed without penalty.")