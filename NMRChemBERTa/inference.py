"""
Inference script for NMR-ChemBERTa
Use this to make predictions on new molecules
"""

import torch
import numpy as np
from pathlib import Path
import logging

from config import get_default_config
from nmr_chemberta_model import NMRChemBERTa
from data_utils import NMReDataParser
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NMRChemBERTaPredictor:
    """Class for making predictions with trained model"""
    
    def __init__(self, model_path: str, config=None):
        self.config = config or get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = NMRChemBERTa(self.config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer and parser
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.chemberta_name)
        self.parser = NMReDataParser()
    
    def predict_from_file(self, nmredata_file: str):
        """Make predictions from an NMReDATA file"""
        # Parse file
        data = self.parser.parse_nmredata_file(nmredata_file)
        if data is None:
            raise ValueError(f"Failed to parse {nmredata_file}")
        
        # Prepare inputs
        inputs = self._prepare_inputs(data)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(**inputs)
        
        # Process predictions
        results = self._process_predictions(predictions, data)
        
        return results
    
    def predict_from_smiles_and_nmr(self, smiles: str, h_nmr_shifts: list, c_nmr_shifts: list):
        """Make predictions from SMILES and NMR data"""
        # This is a simplified version - you'd need to provide more complete data
        # For full functionality, you need 3D coordinates or a way to generate them
        
        raise NotImplementedError("This method requires initial 3D coordinates. Use predict_from_file instead.")
    
    def _prepare_inputs(self, data):
        """Prepare model inputs from parsed data"""
        # Tokenize SMILES
        tokens = self.tokenizer(
            data['smiles'],
            padding='max_length',
            truncation=True,
            max_length=self.config.model.max_seq_length,
            return_tensors='pt'
        )
        
        # Prepare coordinates and atom data
        coords = data['coords']
        atom_types = data['atom_types']
        num_atoms = data['num_atoms']
        
        # Pad if necessary
        if num_atoms < self.config.model.max_atoms:
            pad_size = self.config.model.max_atoms - num_atoms
            coords = np.vstack([coords, np.zeros((pad_size, 3), dtype=np.float32)])
            atom_types = np.concatenate([atom_types, np.full(pad_size, -1, dtype=np.int64)])
        
        # Create masks
        atom_mask = torch.zeros(self.config.model.max_atoms)
        atom_mask[:num_atoms] = 1.0
        
        # Prepare NMR features
        h_shifts = torch.zeros(self.config.model.max_atoms)
        c_shifts = torch.zeros(self.config.model.max_atoms)
        h_mask = torch.zeros(self.config.model.max_atoms)
        c_mask = torch.zeros(self.config.model.max_atoms)
        
        # Fill NMR data
        for shift, atoms in zip(data['nmr_data']['H']['shifts'], data['nmr_data']['H']['atoms']):
            for atom_idx in atoms:
                if 0 <= atom_idx < self.config.model.max_atoms:
                    h_shifts[atom_idx] = shift
                    h_mask[atom_idx] = 1.0
        
        for shift, atoms in zip(data['nmr_data']['C']['shifts'], data['nmr_data']['C']['atoms']):
            for atom_idx in atoms:
                if 0 <= atom_idx < self.config.model.max_atoms:
                    c_shifts[atom_idx] = shift
                    c_mask[atom_idx] = 1.0
        
        # Create input dictionary
        inputs = {
            'input_ids': tokens['input_ids'].to(self.device),
            'attention_mask': tokens['attention_mask'].to(self.device),
            'coords': torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(self.device),
            'atom_types': torch.tensor(atom_types, dtype=torch.long).unsqueeze(0).to(self.device),
            'atom_mask': atom_mask.unsqueeze(0).to(self.device),
            'nmr_features': {
                'h_shifts': h_shifts.unsqueeze(0).to(self.device),
                'c_shifts': c_shifts.unsqueeze(0).to(self.device),
                'h_mask': h_mask.unsqueeze(0).to(self.device),
                'c_mask': c_mask.unsqueeze(0).to(self.device)
            }
        }
        
        return inputs
    
    def _process_predictions(self, predictions, original_data):
        """Process model predictions into readable format"""
        results = {
            'smiles': original_data['smiles'],
            'num_atoms': original_data['num_atoms'],
            'original_coords': original_data['coords'],
            'predictions': {}
        }
        
        # Extract predictions for valid atoms only
        num_atoms = original_data['num_atoms']
        
        # Predicted 3D coordinates
        if 'positions' in predictions:
            pred_coords = predictions['positions'][0, :num_atoms].cpu().numpy()
            results['predictions']['coordinates'] = pred_coords
            
            # Calculate RMSD from original
            rmsd = np.sqrt(np.mean((pred_coords - original_data['coords'])**2))
            results['predictions']['coordinate_rmsd'] = float(rmsd)
        
        # Predicted NMR shifts
        if 'nmr_shifts' in predictions:
            nmr_preds = predictions['nmr_shifts'][0, :num_atoms].cpu().numpy()
            results['predictions']['h_nmr_shifts'] = nmr_preds[:, 0].tolist()
            results['predictions']['c_nmr_shifts'] = nmr_preds[:, 1].tolist()
        
        # Predicted atom types
        if 'atom_types' in predictions:
            atom_type_logits = predictions['atom_types'][0, :num_atoms]
            atom_type_preds = torch.argmax(atom_type_logits, dim=-1).cpu().numpy()
            results['predictions']['atom_types'] = atom_type_preds.tolist()
            
            # Calculate accuracy
            correct = (atom_type_preds == original_data['atom_types'][:num_atoms]).sum()
            accuracy = correct / num_atoms
            results['predictions']['atom_type_accuracy'] = float(accuracy)
        
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with NMR-ChemBERTa')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to .nmredata file')
    parser.add_argument('--output', type=str, help='Path to save predictions (optional)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = NMRChemBERTaPredictor(args.model)
    
    # Make predictions
    try:
        results = predictor.predict_from_file(args.input)
        
        # Print results
        print(f"\nPredictions for {results['smiles']}:")
        print(f"Number of atoms: {results['num_atoms']}")
        
        if 'coordinate_rmsd' in results['predictions']:
            print(f"Coordinate RMSD: {results['predictions']['coordinate_rmsd']:.3f} Å")
        
        if 'atom_type_accuracy' in results['predictions']:
            print(f"Atom type accuracy: {results['predictions']['atom_type_accuracy']:.2%}")
        
        # Save if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()