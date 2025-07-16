# Save this as tanimoto_metrics.py

"""
Tanimoto similarity metrics for molecular generation evaluation
"""

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
import torch
import tqdm

logger = logging.getLogger(__name__)


class TanimotoCalculator:
    """Calculate Tanimoto similarity between generated and true SMILES"""
    
    def __init__(self, fingerprint_type='morgan', radius=2, n_bits=2048):
        """
        Initialize Tanimoto calculator
        
        Args:
            fingerprint_type: Type of fingerprint ('morgan', 'rdkit', 'atom_pair', 'topological')
            radius: Radius for Morgan fingerprint
            n_bits: Number of bits for fingerprint
        """
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits
        
    def get_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Get molecular fingerprint from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            if self.fingerprint_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            elif self.fingerprint_type == 'rdkit':
                fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
            elif self.fingerprint_type == 'atom_pair':
                fp = Chem.AtomPairs.GetAtomPairFingerprintAsBitVect(mol)
            elif self.fingerprint_type == 'topological':
                fp = Chem.AtomPairs.GetTopologicalTorsionFingerprintAsBitVect(mol)
            else:
                raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")
                
            return fp
        except Exception as e:
            logger.debug(f"Error getting fingerprint for {smiles}: {e}")
            return None
    
    def calculate_tanimoto(self, smiles1: str, smiles2: str) -> Optional[float]:
        """Calculate Tanimoto similarity between two SMILES strings"""
        fp1 = self.get_fingerprint(smiles1)
        fp2 = self.get_fingerprint(smiles2)
        
        if fp1 is None or fp2 is None:
            return None
            
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def calculate_batch_similarities(self, 
                                   generated_smiles: List[str], 
                                   true_smiles: List[str]) -> Dict[str, float]:
        """
        Calculate Tanimoto similarities for a batch of SMILES pairs
        
        Returns:
            Dictionary with statistics about the similarities
        """
        similarities = []
        valid_pairs = 0
        invalid_generated = 0
        invalid_true = 0
        
        for gen, true in zip(generated_smiles, true_smiles):
            # Check validity
            gen_valid = self.is_valid_smiles(gen)
            true_valid = self.is_valid_smiles(true)
            
            if not gen_valid:
                invalid_generated += 1
            if not true_valid:
                invalid_true += 1
                
            # Calculate similarity only if both are valid
            if gen_valid and true_valid:
                sim = self.calculate_tanimoto(gen, true)
                if sim is not None and sim > 0:  # Only include non-zero similarities
                    similarities.append(sim)
                    valid_pairs += 1
        
        # Calculate statistics
        if similarities:
            stats = {
                'mean_tanimoto': np.mean(similarities),
                'max_tanimoto': np.max(similarities),
                'min_tanimoto': np.min(similarities),
                'std_tanimoto': np.std(similarities),
                'median_tanimoto': np.median(similarities),
                'valid_pairs': valid_pairs,
                'total_pairs': len(generated_smiles),
                'validity_rate': valid_pairs / len(generated_smiles) if generated_smiles else 0,
                'invalid_generated': invalid_generated,
                'invalid_true': invalid_true,
                'tanimoto_values': similarities  # Keep raw values for distribution analysis
            }
        else:
            stats = {
                'mean_tanimoto': 0.0,
                'max_tanimoto': 0.0,
                'min_tanimoto': 0.0,
                'std_tanimoto': 0.0,
                'median_tanimoto': 0.0,
                'valid_pairs': 0,
                'total_pairs': len(generated_smiles),
                'validity_rate': 0.0,
                'invalid_generated': invalid_generated,
                'invalid_true': invalid_true,
                'tanimoto_values': []
            }
            
        return stats
    
    @staticmethod
    def is_valid_smiles(smiles: str) -> bool:
        """Check if SMILES string is valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def get_similarity_distribution(self, similarities: List[float], 
                                  bins: int = 10) -> Dict[str, List[float]]:
        """Get distribution of similarities for visualization"""
        if not similarities:
            return {'bins': [], 'counts': []}
            
        hist, bin_edges = np.histogram(similarities, bins=bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return {
            'bins': bin_centers.tolist(),
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }


# Integration with your EnhancedTrainer class
def add_tanimoto_to_evaluate(self, dataloader, return_predictions=True, 
                             calculate_roc=False, calculate_tanimoto=True):
    """
    Enhanced evaluate function with Tanimoto similarity calculation
    Add this method to your EnhancedTrainer class or replace the existing evaluate method
    """
    self.model.eval()
    total_loss = 0
    total_accuracy = 0
    predictions = []
    
    # For ROC curves - collect all probabilities and targets
    all_probs = []
    all_targets = []
    all_predictions = []
    
    # For Tanimoto similarity
    tanimoto_calculator = TanimotoCalculator() if calculate_tanimoto else None
    generated_smiles_list = []
    true_smiles_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            h_features = batch['h_features'].to(self.device)
            c_features = batch['c_features'].to(self.device)
            global_features = batch['global_features'].to(self.device)
            smiles_ids = batch['smiles_ids'].to(self.device)
            smiles_mask = batch['smiles_mask'].to(self.device)
            
            outputs = self.model(
                h_features, c_features, global_features,
                smiles_ids[:, :-1], smiles_mask[:, :-1]
            )
            
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                smiles_ids[:, 1:].reshape(-1)
            )
            total_loss += loss.item()
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(
                outputs, 
                smiles_ids[:, 1:], 
                smiles_mask[:, 1:]
            )
            total_accuracy += accuracy
            
            # Collect data for ROC if requested
            if calculate_roc:
                # Get probabilities for correct tokens
                probs = torch.softmax(outputs, dim=-1)
                batch_size, seq_len, vocab_size = probs.shape
                
                # Get the probability of the correct token at each position
                correct_token_probs = torch.gather(
                    probs, 
                    dim=2, 
                    index=smiles_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                
                # Store binary labels (1 if prediction is correct, 0 otherwise)
                predictions_correct = (outputs.argmax(dim=-1) == smiles_ids[:, 1:]).float()
                
                all_probs.append(correct_token_probs.cpu().numpy())
                all_targets.append(smiles_mask[:, 1:].cpu().numpy())
                all_predictions.append(predictions_correct.cpu().numpy())
            
            # Generate predictions
            if return_predictions or calculate_tanimoto:
                generated = self.model(h_features, c_features, global_features)
                
                for i, gen_ids in enumerate(generated):
                    pred_smiles = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    true_smiles = self.tokenizer.decode(smiles_ids[i], skip_special_tokens=True)
                    
                    if return_predictions:
                        predictions.append({
                            'id': batch['id'][i],
                            'predicted': pred_smiles,
                            'true': true_smiles
                        })
                    
                    if calculate_tanimoto:
                        generated_smiles_list.append(pred_smiles)
                        true_smiles_list.append(true_smiles)
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    
    # Calculate exact match accuracy
    exact_matches = sum(1 for p in predictions if p['predicted'] == p['true'])
    exact_match_rate = (exact_matches / len(predictions)) * 100 if predictions else 0
    
    results = {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'exact_match_rate': exact_match_rate,
        'predictions': predictions if return_predictions else None
    }
    
    # Calculate Tanimoto similarities
    if calculate_tanimoto and generated_smiles_list:
        tanimoto_stats = tanimoto_calculator.calculate_batch_similarities(
            generated_smiles_list, true_smiles_list
        )
        results['tanimoto_stats'] = tanimoto_stats
        
        # Log summary
        logger.info(f"Tanimoto Similarity - Mean: {tanimoto_stats['mean_tanimoto']:.4f}, "
                   f"Valid pairs: {tanimoto_stats['valid_pairs']}/{tanimoto_stats['total_pairs']}, "
                   f"Validity rate: {tanimoto_stats['validity_rate']:.2%}")
    
    if calculate_roc and all_probs:
        # Concatenate all batches
        results['probs'] = np.concatenate(all_probs, axis=0)
        results['targets'] = np.concatenate(all_targets, axis=0)
        results['predictions_binary'] = np.concatenate(all_predictions, axis=0)
    
    return results


# Update the metrics tracker update call in your training loop
def update_metrics_with_tanimoto(metrics_tracker, epoch, train_tanimoto_stats=None, 
                                val_tanimoto_stats=None, test_tanimoto_stats=None, **kwargs):
    """
    Extended update function to include Tanimoto metrics
    """
    # Add Tanimoto metrics to kwargs
    if train_tanimoto_stats:
        kwargs['train_mean_tanimoto'] = train_tanimoto_stats['mean_tanimoto']
        kwargs['train_validity_rate'] = train_tanimoto_stats['validity_rate']
    
    if val_tanimoto_stats:
        kwargs['val_mean_tanimoto'] = val_tanimoto_stats['mean_tanimoto']
        kwargs['val_validity_rate'] = val_tanimoto_stats['validity_rate']
    
    if test_tanimoto_stats:
        kwargs['test_mean_tanimoto'] = test_tanimoto_stats['mean_tanimoto']
        kwargs['test_validity_rate'] = test_tanimoto_stats['validity_rate']
    
    # Call original update
    metrics_tracker.update(epoch=epoch, **kwargs)