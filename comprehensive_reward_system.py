#!/usr/bin/env python3
"""
Comprehensive Reward/Penalty System for NMR-to-SMILES Training
Priority: Validity > Token Accuracy > Structure > H/C matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
import re

logger = logging.getLogger(__name__)

class ComprehensiveMoleculeValidator:
    """Comprehensive validator with validity as the highest priority"""
    
    def __init__(self, hc_count_tolerance=0.1):
        """
        Args:
            hc_count_tolerance: More lenient tolerance for H/C counts (10% default)
        """
        self.hc_count_tolerance = hc_count_tolerance
        
        # REVISED WEIGHTS: Validity > Token Accuracy > Structure > H/C
        self.weights = {
            # PENALTIES (balanced approach)
            'empty_prediction': -5.0,      # Moderate penalty
            'invalid_smiles': -3.0,        # Reduced to encourage attempts
            'invalid_brackets': -2.0,      # Syntax error
            'invalid_valency': -2.5,       # Chemistry error
            'poor_hc_match': -0.2,         # Very gentle for H/C mismatch
            
            # REWARDS (prioritizing validity)
            'valid_molecule': 15.0,        # HIGHEST - Valid SMILES is crucial
            'valid_syntax': 10.0,          # HIGH - Correct syntax
            'reasonable_structure': 8.0,   # MEDIUM - Good structure
            'minimal_length': 5.0,         # Basic reward
            
            # H/C matching (lower priority)
            'perfect_hc_match': 5.0,       # Reduced from 20.0
            'good_hc_match': 3.0,          # Reduced from 12.0
            'partial_hc_match': 1.5,       # Reduced from 6.0
            
            # Additional rewards
            'valid_brackets': 3.0,         # Syntax correctness
            'reasonable_length': 2.0,      # Appropriate size
            'contains_carbon': 1.0,        # Basic chemistry
        }
    
    def comprehensive_evaluation(self, predicted_smiles: str, expected_h: int, expected_c: int) -> Dict[str, Any]:
        """
        Comprehensive evaluation with validity as top priority
        
        Returns:
            Dictionary with scores, penalties, rewards, and detailed analysis
        """
        evaluation = {
            'predicted_smiles': predicted_smiles,
            'expected_h': expected_h,
            'expected_c': expected_c,
            'scores': {},
            'penalties': {},
            'rewards': {},
            'total_score': 0.0,
            'issues': [],
            'successes': []
        }
        
        # 1. Check for empty prediction
        if not predicted_smiles or predicted_smiles.strip() == "":
            evaluation['penalties']['empty_prediction'] = self.weights['empty_prediction']
            evaluation['issues'].append("Empty prediction")
            evaluation['total_score'] = self.weights['empty_prediction']
            return evaluation
        
        # Small reward for attempting
        evaluation['rewards']['minimal_length'] = self.weights['minimal_length']
        evaluation['successes'].append("Generated non-empty SMILES")
        
        # 2. Check bracket/parentheses validity
        bracket_score, bracket_issues = self._check_bracket_validity(predicted_smiles)
        if bracket_score < 0:
            evaluation['penalties']['invalid_brackets'] = bracket_score
            evaluation['issues'].extend(bracket_issues)
        else:
            evaluation['rewards']['valid_brackets'] = self.weights['valid_brackets']
            evaluation['successes'].append("Valid brackets/parentheses")
        
        # 3. Basic SMILES validity check (HIGHEST PRIORITY)
        mol = None
        try:
            mol = Chem.MolFromSmiles(predicted_smiles)
        except Exception as e:
            evaluation['issues'].append(f"SMILES parsing error: {str(e)}")
        
        if mol is None:
            evaluation['penalties']['invalid_smiles'] = self.weights['invalid_smiles']
            evaluation['issues'].append("Invalid SMILES - cannot parse")
            
            # Still give small reward for containing carbon
            if 'C' in predicted_smiles or 'c' in predicted_smiles:
                evaluation['rewards']['contains_carbon'] = self.weights['contains_carbon']
                evaluation['successes'].append("Contains carbon atoms")
            
            evaluation['total_score'] = sum(evaluation['penalties'].values()) + sum(evaluation['rewards'].values())
            return evaluation
        
        # 4. MAJOR REWARDS for validity
        evaluation['rewards']['valid_molecule'] = self.weights['valid_molecule']
        evaluation['rewards']['valid_syntax'] = self.weights['valid_syntax']
        evaluation['successes'].append("VALID SMILES - Highest priority achieved!")
        evaluation['successes'].append("Correct chemical syntax")
        
        # 5. Check valency
        valency_score, valency_issues = self._check_valency(mol)
        if valency_score < 0:
            evaluation['penalties']['invalid_valency'] = valency_score
            evaluation['issues'].extend(valency_issues)
        else:
            evaluation['rewards']['valid_valency'] = valency_score
            evaluation['successes'].append("Valid atomic valencies")
        
        # 6. Structural reasonableness (MEDIUM PRIORITY)
        structure_score = self._evaluate_structure_reasonableness(mol)
        if structure_score > 0:
            evaluation['rewards']['reasonable_structure'] = self.weights['reasonable_structure']
            evaluation['successes'].append("Reasonable molecular structure")
        
        # 7. Reasonable length reward
        mol_size = mol.GetNumHeavyAtoms()
        if 3 <= mol_size <= 50:  # Wider range for flexibility
            evaluation['rewards']['reasonable_length'] = self.weights['reasonable_length']
            evaluation['successes'].append("Reasonable molecule size")
        
        # 8. H and C atom count analysis (LOWEST PRIORITY - gentle scoring)
        hc_score, hc_details = self._evaluate_hc_counts(mol, expected_h, expected_c)
        evaluation['scores']['hc_match_score'] = hc_score
        evaluation['hc_details'] = hc_details
        
        if hc_score >= 9:  # Perfect or near-perfect match
            evaluation['rewards']['perfect_hc_match'] = self.weights['perfect_hc_match']
            evaluation['successes'].append(f"Perfect H/C match: H={hc_details['actual_h']}/{expected_h}, C={hc_details['actual_c']}/{expected_c}")
        elif hc_score >= 7:  # Good match
            evaluation['rewards']['good_hc_match'] = self.weights['good_hc_match']
            evaluation['successes'].append(f"Good H/C match: H={hc_details['actual_h']}/{expected_h}, C={hc_details['actual_c']}/{expected_c}")
        elif hc_score >= 5:  # Partial match
            evaluation['rewards']['partial_hc_match'] = self.weights['partial_hc_match']
            evaluation['successes'].append(f"Partial H/C match: H={hc_details['actual_h']}/{expected_h}, C={hc_details['actual_c']}/{expected_c}")
        else:  # Poor match - very gentle penalty
            evaluation['penalties']['poor_hc_match'] = self.weights['poor_hc_match']
            evaluation['issues'].append(f"H/C mismatch (low priority): H={hc_details['actual_h']}/{expected_h}, C={hc_details['actual_c']}/{expected_c}")
        
        # Calculate total score
        total_penalties = sum(evaluation['penalties'].values())
        total_rewards = sum(evaluation['rewards'].values())
        evaluation['total_score'] = total_rewards + total_penalties  # penalties are negative
        
        return evaluation
    
    def _check_bracket_validity(self, smiles: str) -> Tuple[float, List[str]]:
        """Check if brackets and parentheses are properly balanced"""
        issues = []
        
        # Check parentheses
        paren_count = 0
        for char in smiles:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count < 0:
                    issues.append("Unmatched closing parenthesis ')'")
                    return self.weights['invalid_brackets'], issues
        
        if paren_count > 0:
            issues.append(f"Unmatched opening parentheses: {paren_count} unclosed '('")
            return self.weights['invalid_brackets'], issues
        
        # Check square brackets
        bracket_count = 0
        for char in smiles:
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count < 0:
                    issues.append("Unmatched closing bracket ']'")
                    return self.weights['invalid_brackets'], issues
        
        if bracket_count > 0:
            issues.append(f"Unmatched opening brackets: {bracket_count} unclosed '['")
            return self.weights['invalid_brackets'], issues
        
        # Check for valid bracket patterns
        if re.search(r'\[\]', smiles):
            issues.append("Empty brackets '[]' found")
            return self.weights['invalid_brackets'] * 0.5, issues
        
        if re.search(r'\(\)', smiles):
            issues.append("Empty parentheses '()' found")
            return self.weights['invalid_brackets'] * 0.5, issues
        
        # All checks passed
        return 1.0, []
    
    def _check_valency(self, mol) -> Tuple[float, List[str]]:
        """Check if all atoms have valid valencies"""
        issues = []
        
        try:
            # RDKit's built-in valency check
            Chem.SanitizeMol(mol)
            
            # Additional explicit valency checks
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                valence = atom.GetTotalValence()
                
                # Define typical valencies for common atoms
                max_valencies = {
                    'C': 4, 'N': 3, 'O': 2, 'S': 6, 'P': 5,
                    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'H': 1
                }
                
                if symbol in max_valencies:
                    if valence > max_valencies[symbol]:
                        issues.append(f"Invalid valency for {symbol}: {valence} > {max_valencies[symbol]}")
                        return self.weights['invalid_valency'], issues
            
            return 2.0, []  # Reward for valid valency
            
        except Exception as e:
            issues.append(f"Valency check failed: {str(e)}")
            return self.weights['invalid_valency'], issues
    
    def _evaluate_hc_counts(self, mol, expected_h: int, expected_c: int) -> Tuple[float, Dict]:
        """Evaluate H and C atom counts with more lenient scoring (lowest priority)"""
        # Count atoms
        c_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        
        # Count hydrogens (explicit + implicit)
        mol_with_h = Chem.AddHs(mol)
        h_count = sum(1 for atom in mol_with_h.GetAtoms() if atom.GetSymbol() == 'H')
        
        details = {
            'actual_h': h_count,
            'actual_c': c_count,
            'expected_h': expected_h,
            'expected_c': expected_c,
            'h_error': abs(h_count - expected_h),
            'c_error': abs(c_count - expected_c),
            'h_error_percent': abs(h_count - expected_h) / max(expected_h, 1) * 100,
            'c_error_percent': abs(c_count - expected_c) / max(expected_c, 1) * 100
        }
        
        # VERY LENIENT SCORING for H/C counts
        h_accuracy = 1.0 - (details['h_error'] / max(expected_h, 1))
        c_accuracy = 1.0 - (details['c_error'] / max(expected_c, 1))
        
        # Lenient scoring thresholds
        if details['h_error'] == 0 and details['c_error'] == 0:
            score = 10.0  # Perfect match
        elif details['h_error'] <= 2 and details['c_error'] <= 1:
            score = 9.0   # Near perfect
        elif details['h_error_percent'] <= 20 and details['c_error_percent'] <= 20:
            score = 7.0   # Good match (within 20%)
        elif details['h_error_percent'] <= 40 and details['c_error_percent'] <= 40:
            score = 5.0   # Partial match (within 40%)
        else:
            # Still give credit for any carbon/hydrogen presence
            score = 3.0 + (h_accuracy + c_accuracy)  # Base score of 3
        
        details['combined_accuracy'] = (h_accuracy + c_accuracy) / 2
        
        return score, details
    
    def _evaluate_structure_reasonableness(self, mol) -> float:
        """Evaluate if the molecular structure is chemically reasonable"""
        try:
            # Basic structural checks
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            
            # Very basic sanity checks
            if num_atoms == 0:
                return 0.0
            
            if num_atoms == 1:
                return 1.0  # Single atom is reasonable
            
            # Check if the molecule is connected (not multiple fragments)
            fragments = Chem.GetMolFrags(mol)
            if len(fragments) > 1:
                return 0.5  # Multiple fragments - somewhat reasonable
            
            # Check molecular weight (avoid extremely large molecules)
            mw = Descriptors.MolWt(mol)
            if mw > 1000:
                return 0.5  # Very large molecule
            
            # Check for reasonable ring systems
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            if num_rings > 10:  # Too many rings
                return 0.5
            
            return 2.0  # Reasonable structure
            
        except Exception:
            return 0.0


class EnhancedRewardAwareLoss(nn.Module):
    """Enhanced loss function with validity-first priority and stable training"""
    
    def __init__(self, base_criterion, validator: ComprehensiveMoleculeValidator,
                 base_loss_weight=0.3, reward_weight=0.4, token_accuracy_weight=0.3):
        """
        Args:
            base_criterion: Base loss function (CrossEntropyLoss)
            validator: Comprehensive molecule validator
            base_loss_weight: Weight for base token loss (0.3)
            reward_weight: Weight for validity/reward system (0.4 - highest)
            token_accuracy_weight: Weight for token accuracy rewards (0.3)
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.validator = validator
        self.base_loss_weight = base_loss_weight
        self.reward_weight = reward_weight
        self.token_accuracy_weight = token_accuracy_weight
        
        # Normalize weights
        total_weight = base_loss_weight + reward_weight + token_accuracy_weight
        self.base_loss_weight /= total_weight
        self.reward_weight /= total_weight
        self.token_accuracy_weight /= total_weight
    
    def forward(self, outputs, targets, predictions=None, nmr_data=None):
        """
        Calculate enhanced loss with validity-first priority
        """
        # Base token-level loss
        base_loss = self.base_criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        )
        
        # Calculate token accuracy rewards (HIGH PRIORITY)
        token_accuracy_reward, token_accuracy_score = self._calculate_token_accuracy_reward(outputs, targets)
        
        loss_components = {
            'base_loss': base_loss.item(),
            'token_accuracy_score': token_accuracy_score,
            'token_accuracy_reward': token_accuracy_reward,
            'reward_penalty_sum': 0.0,
            'detailed_scores': []
        }
        
        if predictions is not None and nmr_data is not None:
            batch_size = len(predictions)
            total_molecule_score = 0.0
            detailed_evaluations = []
            
            # Aggregate statistics
            stats = {
                'empty_predictions': 0,
                'invalid_smiles': 0,
                'invalid_brackets': 0,
                'invalid_valency': 0,
                'perfect_hc_matches': 0,
                'good_hc_matches': 0,
                'valid_molecules': 0,
                'total_reward': 0.0,
                'total_penalty': 0.0,
                'token_accuracy_scores': [],
                'synergy_bonuses': 0
            }
            
            for i, pred_smiles in enumerate(predictions):
                # Get expected counts
                expected_h = nmr_data['h_atoms'][i] if 'h_atoms' in nmr_data else 0
                expected_c = nmr_data['c_atoms'][i] if 'c_atoms' in nmr_data else 0
                
                # Comprehensive evaluation with validity priority
                evaluation = self.validator.comprehensive_evaluation(
                    pred_smiles, expected_h, expected_c
                )
                
                # Calculate token accuracy for this specific sample
                sample_token_accuracy = self._calculate_sample_token_accuracy(outputs[i], targets[i])
                evaluation['token_accuracy'] = sample_token_accuracy
                stats['token_accuracy_scores'].append(sample_token_accuracy)
                
                # Validity-focused synergy bonus
                synergy_bonus = self._calculate_synergy_bonus(evaluation, sample_token_accuracy)
                evaluation['synergy_bonus'] = synergy_bonus
                if synergy_bonus > 0:
                    stats['synergy_bonuses'] += 1
                
                detailed_evaluations.append(evaluation)
                total_molecule_score += evaluation['total_score'] + synergy_bonus
                
                # Update statistics
                if 'empty_prediction' in evaluation['penalties']:
                    stats['empty_predictions'] += 1
                if 'invalid_smiles' in evaluation['penalties']:
                    stats['invalid_smiles'] += 1
                if 'invalid_brackets' in evaluation['penalties']:
                    stats['invalid_brackets'] += 1
                if 'invalid_valency' in evaluation['penalties']:
                    stats['invalid_valency'] += 1
                if 'perfect_hc_match' in evaluation['rewards']:
                    stats['perfect_hc_matches'] += 1
                elif 'good_hc_match' in evaluation['rewards']:
                    stats['good_hc_matches'] += 1
                if 'valid_molecule' in evaluation['rewards']:
                    stats['valid_molecules'] += 1
                
                stats['total_reward'] += sum(evaluation['rewards'].values())
                stats['total_penalty'] += sum(evaluation['penalties'].values())
            
            # Average scores across batch
            avg_molecule_score = total_molecule_score / batch_size
            avg_token_accuracy = sum(stats['token_accuracy_scores']) / batch_size
            
            # LINEAR SCALING for stable training (no exponential)
            molecule_reward_loss = -avg_molecule_score / 20.0  # Scale down rewards
            token_accuracy_loss = -token_accuracy_reward / 10.0  # Scale down token rewards
            
            # Combined loss with stability
            total_loss = (
                self.base_loss_weight * base_loss +
                self.reward_weight * molecule_reward_loss +
                self.token_accuracy_weight * token_accuracy_loss
            )
            
            # CLAMP LOSS to prevent negative values
            total_loss = torch.clamp(total_loss, min=0.01)
            
            # Store detailed information
            loss_components.update({
                'molecule_reward_penalty_sum': avg_molecule_score,
                'molecule_reward_loss': molecule_reward_loss.item() if torch.is_tensor(molecule_reward_loss) else molecule_reward_loss,
                'token_accuracy_loss': token_accuracy_loss,
                'detailed_scores': detailed_evaluations,
                'batch_stats': stats,
                'empty_rate': stats['empty_predictions'] / batch_size * 100,
                'invalid_rate': stats['invalid_smiles'] / batch_size * 100,
                'validity_rate': stats['valid_molecules'] / batch_size * 100,
                'perfect_hc_rate': stats['perfect_hc_matches'] / batch_size * 100,
                'good_hc_rate': stats['good_hc_matches'] / batch_size * 100,
                'avg_token_accuracy': avg_token_accuracy * 100,
                'synergy_rate': stats['synergy_bonuses'] / batch_size * 100,
                'avg_reward': stats['total_reward'] / batch_size,
                'avg_penalty': stats['total_penalty'] / batch_size
            })
            
            return total_loss, loss_components
        
        # If no predictions available, still include token accuracy
        token_accuracy_loss = -token_accuracy_reward / 10.0
        total_loss = (
            (self.base_loss_weight + self.reward_weight) * base_loss +
            self.token_accuracy_weight * token_accuracy_loss
        )
        
        # Clamp to prevent negative
        total_loss = torch.clamp(total_loss, min=0.01)
        
        return total_loss, loss_components
    
    def _calculate_token_accuracy_reward(self, outputs, targets):
        """Calculate token accuracy reward (HIGH PRIORITY)"""
        predictions = outputs.argmax(dim=-1)
        correct_tokens = (predictions == targets).float()
        
        # Calculate accuracy
        token_accuracy = correct_tokens.mean()
        
        # Strong rewards for token accuracy (second priority)
        if token_accuracy > 0.95:
            reward = 10.0
        elif token_accuracy > 0.90:
            reward = 8.0
        elif token_accuracy > 0.85:
            reward = 6.0
        elif token_accuracy > 0.80:
            reward = 4.0
        elif token_accuracy > 0.70:
            reward = 2.0
        elif token_accuracy > 0.60:
            reward = 1.0
        elif token_accuracy > 0.50:
            reward = 0.5
        else:
            reward = 0.0
        
        return reward, token_accuracy.item()
    
    def _calculate_sample_token_accuracy(self, sample_outputs, sample_targets):
        """Calculate token accuracy for a single sample"""
        predictions = sample_outputs.argmax(dim=-1)
        correct_tokens = (predictions == sample_targets).float()
        return correct_tokens.mean().item()
    
    def _calculate_synergy_bonus(self, evaluation, token_accuracy):
        """
        Calculate synergy bonus focusing on validity + token accuracy
        """
        # Check if molecule is valid (highest priority)
        is_valid = 'valid_molecule' in evaluation['rewards']
        has_good_structure = 'reasonable_structure' in evaluation['rewards']
        has_any_hc_match = any(key in evaluation['rewards'] for key in ['perfect_hc_match', 'good_hc_match', 'partial_hc_match'])
        
        # Token accuracy thresholds
        high_token_accuracy = token_accuracy > 0.80
        good_token_accuracy = token_accuracy > 0.70
        decent_token_accuracy = token_accuracy > 0.60
        
        # Validity-focused synergy bonuses
        if is_valid and high_token_accuracy:
            return 8.0  # Strong bonus for valid + high accuracy
        elif is_valid and good_token_accuracy:
            return 5.0  # Good bonus for valid + good accuracy
        elif is_valid and decent_token_accuracy:
            return 3.0  # Moderate bonus
        elif is_valid:
            return 2.0  # Bonus just for validity
        elif high_token_accuracy:
            return 1.0  # Small bonus for high accuracy alone
        
        return 0.0


def create_comprehensive_system(model, tokenizer, device='cuda'):
    """Create comprehensive reward/penalty system with VALIDITY-FIRST priority"""
    
    # Create comprehensive validator with lenient H/C tolerance
    validator = ComprehensiveMoleculeValidator(hc_count_tolerance=0.1)  # 10% tolerance
    
    # Create enhanced loss with validity-first weights
    base_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    enhanced_loss = EnhancedRewardAwareLoss(
        base_criterion,
        validator,
        base_loss_weight=0.3,      # Base token loss
        reward_weight=0.4,         # Highest - validity priority
        token_accuracy_weight=0.3  # High - token accuracy important
    )
    
    return validator, enhanced_loss


# Test the system
if __name__ == "__main__":
    print("Testing Validity-First Reward System")
    print("=" * 60)
    
    validator = ComprehensiveMoleculeValidator()
    
    test_cases = [
        {"smiles": "CCO", "expected_h": 6, "expected_c": 2, "name": "Ethanol (perfect)"},
        {"smiles": "CCCO", "expected_h": 8, "expected_c": 3, "name": "Propanol (perfect)"},
        {"smiles": "CCO", "expected_h": 8, "expected_c": 3, "name": "Ethanol (wrong H/C)"},
        {"smiles": "C", "expected_h": 4, "expected_c": 1, "name": "Methane (perfect)"},
        {"smiles": "", "expected_h": 6, "expected_c": 2, "name": "Empty"},
        {"smiles": "CC(C)C", "expected_h": 10, "expected_c": 4, "name": "Isobutane (perfect)"},
        {"smiles": "CC(X)C", "expected_h": 10, "expected_c": 4, "name": "Invalid SMILES"},
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"SMILES: '{test['smiles']}'")
        print(f"Expected H/C: {test['expected_h']}/{test['expected_c']}")
        
        evaluation = validator.comprehensive_evaluation(
            test['smiles'], test['expected_h'], test['expected_c']
        )
        
        print(f"Total Score: {evaluation['total_score']:.2f}")
        print(f"Rewards: {sum(evaluation['rewards'].values()):.2f}")
        print(f"Penalties: {sum(evaluation['penalties'].values()):.2f}")
        
        # Show validity status prominently
        if 'valid_molecule' in evaluation['rewards']:
            print("✓ VALID MOLECULE")
        else:
            print("✗ INVALID MOLECULE")
        
        if evaluation['successes']:
            print(f"Successes: {', '.join(evaluation['successes'][:2])}")
        if evaluation['issues']:
            print(f"Issues: {', '.join(evaluation['issues'][:2])}")
        
        print("-" * 40)