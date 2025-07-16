#!/usr/bin/env python3
"""
Test individual predictions from the trained model
"""

import torch
import pickle
import json
from pathlib import Path
from transformers import RobertaTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import argparse

from nmr_to_smiles_chemberta import (
    NMRDataParser, 
    NMRFeatureExtractor,
    NMREncoder,
    NMRToSMILES
)

def calculate_similarity_metrics(smiles1, smiles2):
    """Calculate multiple similarity metrics between two SMILES"""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    metrics = {
        'tanimoto': 0.0,
        'dice': 0.0,
        'cosine': 0.0,
        'valid_comparison': False,
        'same_num_atoms': False,
        'same_num_bonds': False,
        'mw_similarity': 0.0
    }
    
    if mol1 is None or mol2 is None:
        return metrics
    
    metrics['valid_comparison'] = True
    
    # Fingerprint similarities
    fp1 = FingerprintMols.FingerprintMol(mol1)
    fp2 = FingerprintMols.FingerprintMol(mol2)
    
    metrics['tanimoto'] = DataStructs.TanimotoSimilarity(fp1, fp2)
    metrics['dice'] = DataStructs.DiceSimilarity(fp1, fp2)
    metrics['cosine'] = DataStructs.CosineSimilarity(fp1, fp2)
    
    # Structural comparisons
    metrics['same_num_atoms'] = mol1.GetNumAtoms() == mol2.GetNumAtoms()
    metrics['same_num_bonds'] = mol1.GetNumBonds() == mol2.GetNumBonds()
    
    # Molecular weight similarity
    mw1 = Descriptors.MolWt(mol1)
    mw2 = Descriptors.MolWt(mol2)
    metrics['mw_similarity'] = 1.0 - abs(mw1 - mw2) / max(mw1, mw2)
    
    # Average similarity across all metrics
    similarity_values = [
        metrics['tanimoto'],
        metrics['dice'],
        metrics['cosine'],
        metrics['mw_similarity']
    ]
    metrics['average_similarity'] = sum(similarity_values) / len(similarity_values)
    
    return metrics

def load_complete_compounds(filepath):
    """Load list of complete compounds from file"""
    complete_compounds = set()
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split(',')
                if parts:
                    compound_id = parts[0].strip()
                    complete_compounds.add(compound_id)
    
    return complete_compounds

def test_predictions(model_path, feature_extractor_path, test_dir, complete_list_file, n_samples=10):
    """Test predictions on complete compounds only"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load complete compounds list
    print(f"Loading complete compounds from: {complete_list_file}")
    complete_compounds = load_complete_compounds(complete_list_file)
    print(f"Found {len(complete_compounds)} complete compounds")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    
    # Load feature extractor
    with open(feature_extractor_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    
    # Initialize model
    nmr_encoder = NMREncoder(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_encoder_layers']
    )
    
    model = NMRToSMILES(
        nmr_encoder=nmr_encoder,
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config['hidden_dim'],
        num_decoder_layers=config['num_decoder_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Parse test files - only complete compounds
    parser = NMRDataParser()
    
    # Find files for complete compounds
    test_path = Path(test_dir)
    tested_compounds = []
    
    print(f"\nSearching for complete compound files in: {test_path}")
    
    for compound_id in list(complete_compounds)[:n_samples]:
        # Try to find file for this compound
        matching_files = list(test_path.glob(f"*{compound_id}*.nmredata"))
        
        if matching_files:
            filepath = matching_files[0]
            data = parser.parse_file(str(filepath))
            
            if data is not None:
                tested_compounds.append((filepath, data))
    
    print(f"\nTesting {len(tested_compounds)} complete compounds:\n")
    results = []
    
    for i, (filepath, data) in enumerate(tested_compounds):
        # Extract features
        features = feature_extractor.extract_features(data)
        
        # Prepare tensors
        h_features = features['h_features'].unsqueeze(0).to(device)
        c_features = features['c_features'].unsqueeze(0).to(device)
        global_features = features['global_features'].unsqueeze(0).to(device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = model(h_features, c_features, global_features)
        
        # Decode
        predicted_smiles = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        true_smiles = data.get('canonical_smiles', '')
        
        # Calculate similarity metrics
        metrics = calculate_similarity_metrics(predicted_smiles, true_smiles)
        
        # Check if valid SMILES
        pred_mol = Chem.MolFromSmiles(predicted_smiles)
        is_valid = pred_mol is not None
        
        # Print results
        print(f"{i+1}. Compound: {data.get('id', 'unknown')}")
        print(f"   True SMILES:      {true_smiles[:80]}{'...' if len(true_smiles) > 80 else ''}")
        print(f"   Predicted SMILES: {predicted_smiles[:80]}{'...' if len(predicted_smiles) > 80 else ''}")
        print(f"   Valid SMILES: {is_valid}")
        print(f"   Exact match: {predicted_smiles == true_smiles}")
        print(f"   SIMILARITY METRICS:")
        print(f"     - Tanimoto:    {metrics['tanimoto']:.3f}")
        print(f"     - Dice:        {metrics['dice']:.3f}")
        print(f"     - Cosine:      {metrics['cosine']:.3f}")
        print(f"     - MW similar:  {metrics['mw_similarity']:.3f}")
        print(f"     - AVERAGE:     {metrics['average_similarity']:.3f}")
        
        if is_valid and pred_mol:
            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol:
                print(f"   Structure comparison:")
                print(f"     - True atoms: {true_mol.GetNumAtoms()}, Pred atoms: {pred_mol.GetNumAtoms()}")
                print(f"     - True bonds: {true_mol.GetNumBonds()}, Pred bonds: {pred_mol.GetNumBonds()}")
                print(f"     - True MW: {Descriptors.MolWt(true_mol):.2f}, Pred MW: {Descriptors.MolWt(pred_mol):.2f}")
        
        print(f"   NMR data: H atoms: {data['h_atoms']}, C atoms: {data['c_atoms']}")
        print()
        
        results.append({
            'id': data.get('id', 'unknown'),
            'true_smiles': true_smiles,
            'predicted_smiles': predicted_smiles,
            'is_valid': is_valid,
            'exact_match': predicted_smiles == true_smiles,
            **metrics
        })
    
    # Summary statistics
    print("\nSUMMARY:")
    print("-" * 50)
    valid_count = sum(1 for r in results if r['is_valid'])
    exact_count = sum(1 for r in results if r['exact_match'])
    
    # Average metrics
    avg_tanimoto = sum(r['tanimoto'] for r in results) / len(results) if results else 0
    avg_dice = sum(r['dice'] for r in results) / len(results) if results else 0
    avg_cosine = sum(r['cosine'] for r in results) / len(results) if results else 0
    avg_mw_sim = sum(r['mw_similarity'] for r in results) / len(results) if results else 0
    avg_overall = sum(r['average_similarity'] for r in results) / len(results) if results else 0
    
    print(f"Total tested: {len(results)}")
    print(f"Valid SMILES: {valid_count}/{len(results)} ({valid_count/len(results)*100:.1f}%)")
    print(f"Exact matches: {exact_count}/{len(results)} ({exact_count/len(results)*100:.1f}%)")
    print(f"\nAVERAGE SIMILARITY METRICS:")
    print(f"  - Tanimoto:          {avg_tanimoto:.3f}")
    print(f"  - Dice:              {avg_dice:.3f}")
    print(f"  - Cosine:            {avg_cosine:.3f}")
    print(f"  - MW similarity:     {avg_mw_sim:.3f}")
    print(f"  - OVERALL AVERAGE:   {avg_overall:.3f}")
    
    # Distribution of similarities
    if results:
        similarities = [r['average_similarity'] for r in results]
        print(f"\nSIMILARITY DISTRIBUTION:")
        print(f"  - Min:     {min(similarities):.3f}")
        print(f"  - Max:     {max(similarities):.3f}")
        print(f"  - Median:  {sorted(similarities)[len(similarities)//2]:.3f}")
        print(f"  - >0.9:    {sum(1 for s in similarities if s > 0.9)}/{len(results)}")
        print(f"  - >0.8:    {sum(1 for s in similarities if s > 0.8)}/{len(results)}")
        print(f"  - >0.7:    {sum(1 for s in similarities if s > 0.7)}/{len(results)}")
    
    # Save results
    with open('complete_compounds_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to complete_compounds_test_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test NMR to SMILES predictions on complete compounds')
    parser.add_argument('--model', default='best_nmr_to_smiles_model.pt', 
                       help='Path to model checkpoint')
    parser.add_argument('--feature-extractor', default='feature_extractor.pkl',
                       help='Path to feature extractor')
    parser.add_argument('--test-dir', required=True,
                       help='Directory containing .nmredata files')
    parser.add_argument('--complete-list', 
                       default=r'C:\Users\pierr\Desktop\CS MSc Project files\peaklist\outputv6\complete_data_compounds.txt',
                       help='Path to complete compounds list')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Number of complete compounds to test')
    
    args = parser.parse_args()
    
    test_predictions(
        args.model,
        args.feature_extractor,
        args.test_dir,
        args.complete_list,
        args.n_samples
    )