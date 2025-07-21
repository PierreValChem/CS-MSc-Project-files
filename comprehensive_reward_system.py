#!/usr/bin/env python3
"""
Comprehensive Reward/Penalty System for NMR-to-SMILES Training
Focuses on rewarding proper molecule generation with correct H/C counts and valid chemistry
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
    """Comprehensive validator with detailed rewards and penalties"""
    
    def __init__(self, hc_count_tolerance=0.05):
        """
        Args:
            hc_count_tolerance: Very strict tolerance for H/C counts (5% default)
        """
        self.hc_count_tolerance = hc_count_tolerance
        
        # Define reward/penalty weights (higher = more important)
        self.weights = {
            'empty_prediction': -10.0,      # Severe penalty for empty predictions
            'invalid_smiles': -8.0,         # Severe penalty for invalid SMILES
            'invalid_brackets': -6.0,       # High penalty for bracket/parentheses issues
            'invalid_valency': -7.0,        # High penalty for valency violations
            'perfect_hc_match': 10.0,       # High reward for perfect H/C match
            'good_hc_match': 5.0,           # Good reward for close H/C match
            'poor_hc_match': -4.0,          # Penalty for poor H/C match
            'valid_molecule': 3.0,          # Reward for basic validity
            'reasonable_structure': 2.0,    # Reward for reasonable molecular properties
            'minimal_length': 1.0,          # Small reward for non-empty predictions
        }
    
    def comprehensive_evaluation(self, predicted_smiles: str, expected_h: int, expected_c: int) -> Dict[str, Any]:
        """
        Comprehensive evaluation with detailed rewards and penalties
        
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
        
        # 1. Check for empty prediction (most critical)
        if not predicted_smiles or predicted_smiles.strip() == "":
            evaluation['penalties']['empty_prediction'] = self.weights['empty_prediction']
            evaluation['issues'].append("Empty prediction")
            evaluation['total_score'] = self.weights['empty_prediction']
            return evaluation
        
        # 2. Check bracket/parentheses validity (before RDKit parsing)
        bracket_score, bracket_issues = self._check_bracket_validity(predicted_smiles)
        if bracket_score < 0:
            evaluation['penalties']['invalid_brackets'] = bracket_score
            evaluation['issues'].extend(bracket_issues)
        else:
            evaluation['rewards']['valid_brackets'] = bracket_score
            evaluation['successes'].append("Valid brackets/parentheses")
        
        # 3. Basic SMILES validity check
        mol = None
        try:
            mol = Chem.MolFromSmiles(predicted_smiles)
        except Exception as e:
            evaluation['issues'].append(f"SMILES parsing error: {str(e)}")
        
        if mol is None:
            evaluation['penalties']['invalid_smiles'] = self.weights['invalid_smiles']
            evaluation['issues'].append("Invalid SMILES - cannot parse")
            evaluation['total_score'] = sum(evaluation['penalties'].values()) + sum(evaluation['rewards'].values())
            return evaluation
        
        # 4. Reward basic validity
        evaluation['rewards']['valid_molecule'] = self.weights['valid_molecule']
        evaluation['successes'].append("Valid SMILES structure")
        
        # 5. Check valency (critical for chemical validity)
        valency_score, valency_issues = self._check_valency(mol)
        if valency_score < 0:
            evaluation['penalties']['invalid_valency'] = valency_score
            evaluation['issues'].extend(valency_issues)
        else:
            evaluation['rewards']['valid_valency'] = valency_score
            evaluation['successes'].append("Valid atomic valencies")
        
        # 6. H and C atom count analysis (most important for NMR)
        hc_score, hc_details = self._evaluate_hc_counts(mol, expected_h, expected_c)
        evaluation['scores']['hc_match_score'] = hc_score
        evaluation['hc_details'] = hc_details
        
        if hc_score >= 8:  # Perfect or near-perfect match
            evaluation['rewards']['perfect_hc_match'] = self.weights['perfect_hc_match']
            evaluation['successes'].append(f"Excellent H/C match: H={hc_details['actual_h']}/{expected_h}, C={hc_details['actual_c']}/{expected_c}")
        elif hc_score >= 4:  # Good match
            evaluation['rewards']['good_hc_match'] = self.weights['good_hc_match']
            evaluation['successes'].append(f"Good H/C match: H={hc_details['actual_h']}/{expected_h}, C={hc_details['actual_c']}/{expected_c}")
        else:  # Poor match
            evaluation['penalties']['poor_hc_match'] = self.weights['poor_hc_match']
            evaluation['issues'].append(f"Poor H/C match: H={hc_details['actual_h']}/{expected_h}, C={hc_details['actual_c']}/{expected_c}")
        
        # 7. Structural reasonableness
        structure_score = self._evaluate_structure_reasonableness(mol)
        if structure_score > 0:
            evaluation['rewards']['reasonable_structure'] = structure_score
            evaluation['successes'].append("Reasonable molecular structure")
        
        # 8. Minimal length reward (small incentive for non-empty predictions)
        if len(predicted_smiles) >= 2:
            evaluation['rewards']['minimal_length'] = self.weights['minimal_length']
        
        # Calculate total score
        total_penalties = sum(evaluation['penalties'].values())
        total_rewards = sum(evaluation['rewards'].values())
        evaluation['total_score'] = total_rewards + total_penalties  # penalties are negative
        
        return evaluation
    
    def _check_bracket_validity(self, smiles: str) -> Tuple[float, List[str]]:
        """Check if brackets and parentheses are properly balanced"""
        issues = []
        score = 0.0
        
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
        """Evaluate H and C atom counts with detailed scoring"""
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
        
        # Calculate score based on accuracy
        h_accuracy = 1.0 - (details['h_error'] / max(expected_h, 1))
        c_accuracy = 1.0 - (details['c_error'] / max(expected_c, 1))
        
        # Perfect match bonus
        if details['h_error'] == 0 and details['c_error'] == 0:
            score = 10.0  # Perfect match
        elif details['h_error'] <= 1 and details['c_error'] <= 1:
            score = 8.0   # Near perfect (off by 1)
        elif details['h_error_percent'] <= 10 and details['c_error_percent'] <= 10:
            score = 6.0   # Good match (within 10%)
        elif details['h_error_percent'] <= 25 and details['c_error_percent'] <= 25:
            score = 4.0   # Acceptable match (within 25%)
        else:
            score = 2.0 * (h_accuracy + c_accuracy)  # Poor match
        
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
    """Enhanced loss function with comprehensive reward/penalty system including token accuracy rewards"""
    
    def __init__(self, base_criterion, validator: ComprehensiveMoleculeValidator,
                 base_loss_weight=0.2, reward_weight=0.5, token_accuracy_weight=0.3):
        """
        Args:
            base_criterion: Base loss function (CrossEntropyLoss)
            validator: Comprehensive molecule validator
            base_loss_weight: Weight for base token loss (0.2 = 20%)
            reward_weight: Weight for reward/penalty system (0.5 = 50%)
            token_accuracy_weight: Weight for token accuracy rewards (0.3 = 30%)
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
        Calculate enhanced loss with comprehensive reward/penalty system + token accuracy rewards
        
        Args:
            outputs: Model logits [batch, seq_len, vocab_size]
            targets: Target token ids [batch, seq_len]
            predictions: Generated SMILES strings
            nmr_data: NMR data with expected H and C counts
        """
        # Base token-level loss
        base_loss = self.base_criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        )
        
        # Calculate token accuracy rewards
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
                'synergy_bonuses': 0  # NEW: Track when token accuracy + H/C match both high
            }
            
            for i, pred_smiles in enumerate(predictions):
                # Get expected counts
                expected_h = nmr_data['h_atoms'][i] if 'h_atoms' in nmr_data else 0
                expected_c = nmr_data['c_atoms'][i] if 'c_atoms' in nmr_data else 0
                
                # Comprehensive evaluation
                evaluation = self.validator.comprehensive_evaluation(
                    pred_smiles, expected_h, expected_c
                )
                
                # Calculate token accuracy for this specific sample
                sample_token_accuracy = self._calculate_sample_token_accuracy(outputs[i], targets[i])
                evaluation['token_accuracy'] = sample_token_accuracy
                stats['token_accuracy_scores'].append(sample_token_accuracy)
                
                # SYNERGY BONUS: Reward when both token accuracy and H/C matching are high
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
            
            # Convert rewards to loss terms (negative rewards become positive loss)
            molecule_reward_loss = -avg_molecule_score  # Negative because rewards should decrease loss
            token_accuracy_loss = -token_accuracy_reward  # Token accuracy as loss component
            
            # Combined loss with all three components
            total_loss = (
                self.base_loss_weight * base_loss +
                self.reward_weight * molecule_reward_loss +
                self.token_accuracy_weight * token_accuracy_loss
            )
            
            # Store detailed information
            loss_components.update({
                'molecule_reward_penalty_sum': avg_molecule_score,
                'molecule_reward_loss': molecule_reward_loss,
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
        token_accuracy_loss = -token_accuracy_reward
        total_loss = (
            (self.base_loss_weight + self.reward_weight) * base_loss +  # Redistribute weights
            self.token_accuracy_weight * token_accuracy_loss
        )
        
        return total_loss, loss_components
    
    def _calculate_token_accuracy_reward(self, outputs, targets):
        """Calculate token accuracy reward with graduated bonuses"""
        predictions = outputs.argmax(dim=-1)
        correct_tokens = (predictions == targets).float()
        
        # Calculate accuracy
        token_accuracy = correct_tokens.mean()
        
        # Graduated rewards for token accuracy
        if token_accuracy > 0.95:
            reward = 8.0    # Excellent accuracy
        elif token_accuracy > 0.90:
            reward = 6.0    # Very good accuracy
        elif token_accuracy > 0.85:
            reward = 4.0    # Good accuracy
        elif token_accuracy > 0.80:
            reward = 2.0    # Decent accuracy
        elif token_accuracy > 0.70:
            reward = 1.0    # Basic accuracy
        else:
            reward = 0.0    # Poor accuracy
        
        return reward, token_accuracy.item()
    
    def _calculate_sample_token_accuracy(self, sample_outputs, sample_targets):
        """Calculate token accuracy for a single sample"""
        predictions = sample_outputs.argmax(dim=-1)
        correct_tokens = (predictions == sample_targets).float()
        return correct_tokens.mean().item()
    
    def _calculate_synergy_bonus(self, evaluation, token_accuracy):
        """
        Calculate synergy bonus when both token accuracy and molecular properties are good
        This encourages the model to achieve both good syntax AND good chemistry
        """
        # Check if molecule has good H/C matching
        has_good_hc = ('perfect_hc_match' in evaluation['rewards'] or 
                      'good_hc_match' in evaluation['rewards'])
        
        # Check if molecule is valid
        is_valid = 'valid_molecule' in evaluation['rewards']
        
        # High token accuracy threshold
        high_token_accuracy = token_accuracy > 0.85
        
        # Synergy bonuses
        if has_good_hc and is_valid and high_token_accuracy:
            if 'perfect_hc_match' in evaluation['rewards']:
                return 5.0  # Perfect synergy: perfect H/C + valid + high token accuracy
            else:
                return 3.0  # Good synergy: good H/C + valid + high token accuracy
        elif has_good_hc and high_token_accuracy:
            return 2.0  # Partial synergy: good H/C + high token accuracy
        elif is_valid and high_token_accuracy:
            return 1.0  # Basic synergy: valid + high token accuracy
        
        return 0.0  # No synergy bonus


def create_comprehensive_system(model, tokenizer, device='cuda'):
    """Create comprehensive reward/penalty system with token accuracy integration"""
    
    # Create comprehensive validator with strict H/C tolerance
    validator = ComprehensiveMoleculeValidator(hc_count_tolerance=0.05)  # 5% tolerance
    
    # Create enhanced loss with balanced focus on all components
    base_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    enhanced_loss = EnhancedRewardAwareLoss(
        base_criterion,
        validator,
        base_loss_weight=0.2,      # 20% base token loss
        reward_weight=0.5,         # 50% molecule reward/penalty system
        token_accuracy_weight=0.3  # 30% token accuracy rewards (NEW)
    )
    
    return validator, enhanced_loss


def test_comprehensive_system_with_token_accuracy():
    """Test the comprehensive reward/penalty system including token accuracy"""
    validator = ComprehensiveMoleculeValidator()
    
    # Create mock outputs and targets for token accuracy testing
    import torch
    batch_size, seq_len, vocab_size = 2, 10, 100
    
    # Mock high accuracy case
    outputs_high_acc = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Make outputs match targets for high accuracy
    for i in range(batch_size):
        for j in range(seq_len):
            outputs_high_acc[i, j, targets[i, j]] = 10.0  # High logit for correct token
    
    # Mock low accuracy case
    outputs_low_acc = torch.randn(batch_size, seq_len, vocab_size)
    
    test_cases = [
        # High token accuracy + perfect H/C match
        {
            "smiles": "CCO", 
            "expected_h": 6, 
            "expected_c": 2, 
            "outputs": outputs_high_acc[0:1],
            "targets": targets[0:1],
            "name": "High token accuracy + Perfect H/C"
        },
        # High token accuracy + invalid molecule  
        {
            "smiles": "CCC(C)C(C)", 
            "expected_h": 14, 
            "expected_c": 6,
            "outputs": outputs_high_acc[0:1], 
            "targets": targets[0:1],
            "name": "High token accuracy + Invalid molecule"
        },
        # Low token accuracy + perfect H/C match
        {
            "smiles": "CCO", 
            "expected_h": 6, 
            "expected_c": 2,
            "outputs": outputs_low_acc[0:1],
            "targets": targets[0:1], 
            "name": "Low token accuracy + Perfect H/C"
        },
        # Empty prediction case
        {
            "smiles": "", 
            "expected_h": 6, 
            "expected_c": 2,
            "outputs": outputs_low_acc[0:1],
            "targets": targets[0:1],
            "name": "Empty prediction"
        }
    ]
    
    print("Comprehensive System Test with Token Accuracy Integration")
    print("=" * 70)
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"SMILES: '{test['smiles']}'")
        print(f"Expected H/C: {test['expected_h']}/{test['expected_c']}")
        
        # Test molecular evaluation
        evaluation = validator.comprehensive_evaluation(
            test['smiles'], test['expected_h'], test['expected_c']
        )
        
        # Test token accuracy
        predictions = test['outputs'].argmax(dim=-1)
        correct_tokens = (predictions == test['targets']).float()
        token_accuracy = correct_tokens.mean().item()
        
        # Test synergy bonus
        loss_fn = EnhancedRewardAwareLoss(
            torch.nn.CrossEntropyLoss(), 
            validator,
            base_loss_weight=0.2,
            reward_weight=0.5, 
            token_accuracy_weight=0.3
        )
        synergy_bonus = loss_fn._calculate_synergy_bonus(evaluation, token_accuracy)
        
        print(f"Token Accuracy: {token_accuracy*100:.1f}%")
        print(f"Molecule Score: {evaluation['total_score']:.2f}")
        print(f"Synergy Bonus: {synergy_bonus:.2f}")
        print(f"Combined Benefit: {evaluation['total_score'] + synergy_bonus:.2f}")
        
        if evaluation['issues']:
            print(f"Issues: {', '.join(evaluation['issues'])}")
        if evaluation['successes']:
            print(f"Successes: {', '.join(evaluation['successes'])}")
        
        # Show what the model learns
        if token_accuracy > 0.85 and synergy_bonus > 0:
            print("✓ MODEL LEARNS: High token accuracy + good chemistry = BIG REWARD!")
        elif token_accuracy > 0.85:
            print("→ MODEL LEARNS: Good syntax but need better chemistry")
        elif synergy_bonus > 0:
            print("→ MODEL LEARNS: Good chemistry but need better syntax")
        else:
            print("✗ MODEL LEARNS: Need improvement in both syntax and chemistry")
        
        print("-" * 50)


if __name__ == "__main__":
    test_comprehensive_system_with_token_accuracy()