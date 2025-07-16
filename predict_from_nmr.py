#!/usr/bin/env python3
"""
Predict SMILES from NMR data using trained model
"""

import torch
import pickle
import json
from pathlib import Path
from transformers import RobertaTokenizer
import argparse
import logging

from nmr_to_smiles_chemberta import (
    NMRDataParser, 
    NMRFeatureExtractor,
    NMREncoder,
    NMRToSMILES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NMRPredictor:
    """Predict SMILES from NMR data"""
    
    def __init__(self, model_path: str, feature_extractor_path: str, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        
        # Load feature extractor
        logger.info("Loading feature extractor...")
        with open(feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)
        
        # Load model checkpoint
        logger.info("Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Initialize model
        nmr_encoder = NMREncoder(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_encoder_layers']
        )
        
        self.model = NMRToSMILES(
            nmr_encoder=nmr_encoder,
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=config['hidden_dim'],
            num_decoder_layers=config['num_decoder_layers']
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
    
    def predict_from_file(self, nmr_file_path: str) -> dict:
        """Predict SMILES from a single NMR file"""
        # Parse file
        parser = NMRDataParser()
        data = parser.parse_file(nmr_file_path)
        
        if data is None:
            raise ValueError(f"Could not parse NMR data from {nmr_file_path}")
        
        # Extract features
        features = self.feature_extractor.extract_features(data)
        
        # Convert to tensors and add batch dimension
        h_features = features['h_features'].unsqueeze(0).to(self.device)
        c_features = features['c_features'].unsqueeze(0).to(self.device)
        global_features = features['global_features'].unsqueeze(0).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = self.model(h_features, c_features, global_features)
        
        # Decode SMILES
        predicted_smiles = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return {
            'id': data.get('id', 'unknown'),
            'predicted_smiles': predicted_smiles,
            'true_smiles': data.get('canonical_smiles', 'unknown'),
            'h_atoms': data['h_atoms'],
            'c_atoms': data['c_atoms'],
            'h_peaks': len([p for p in data['h_peaks'] if not p['is_padding']]),
            'c_peaks': len([p for p in data['c_peaks'] if not p['is_padding']]),
            'is_complete': data.get('is_complete', False)
        }
    
    def predict_from_nmr_data(self, h_peaks: list, c_peaks: list, 
                            h_atoms: int, c_atoms: int) -> str:
        """Predict SMILES from raw NMR data
        
        Args:
            h_peaks: List of dicts with 'shift', 'multiplicity', 'atom_num'
            c_peaks: List of dicts with 'shift', 'multiplicity', 'atom_num'
            h_atoms: Total number of H atoms
            c_atoms: Total number of C atoms
        
        Returns:
            Predicted SMILES string
        """
        # Create data dict
        data = {
            'h_peaks': h_peaks,
            'c_peaks': c_peaks,
            'h_atoms': h_atoms,
            'c_atoms': c_atoms
        }
        
        # Extract features
        features = self.feature_extractor.extract_features(data)
        
        # Convert to tensors and add batch dimension
        h_features = features['h_features'].unsqueeze(0).to(self.device)
        c_features = features['c_features'].unsqueeze(0).to(self.device)
        global_features = features['global_features'].unsqueeze(0).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = self.model(h_features, c_features, global_features)
        
        # Decode SMILES
        predicted_smiles = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return predicted_smiles
    
    def batch_predict(self, file_paths: list) -> list:
        """Predict SMILES for multiple NMR files"""
        predictions = []
        
        for file_path in file_paths:
            try:
                result = self.predict_from_file(file_path)
                predictions.append(result)
                logger.info(f"Predicted for {result['id']}: {result['predicted_smiles']}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                predictions.append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict SMILES from NMR data')
    parser.add_argument('input', help='Input NMR file or directory')
    parser.add_argument('--model', default='best_nmr_to_smiles_model.pt', 
                       help='Path to model checkpoint')
    parser.add_argument('--feature-extractor', default='feature_extractor.pkl',
                       help='Path to feature extractor')
    parser.add_argument('--output', default='predictions.json',
                       help='Output file for predictions')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = NMRPredictor(
        model_path=args.model,
        feature_extractor_path=args.feature_extractor,
        device=args.device
    )
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file prediction
        result = predictor.predict_from_file(str(input_path))
        predictions = [result]
    elif input_path.is_dir():
        # Batch prediction
        nmr_files = list(input_path.glob('*.nmredata'))
        logger.info(f"Found {len(nmr_files)} NMR files")
        predictions = predictor.batch_predict(nmr_files)
    else:
        raise ValueError(f"Input path {input_path} not found")
    
    # Save predictions
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Predictions saved to {args.output}")
    
    # Print summary
    if len(predictions) > 1:
        correct = sum(1 for p in predictions 
                     if 'predicted_smiles' in p and 'true_smiles' in p 
                     and p['predicted_smiles'] == p['true_smiles'])
        total = sum(1 for p in predictions if 'predicted_smiles' in p)
        if total > 0:
            logger.info(f"Exact match accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    main()