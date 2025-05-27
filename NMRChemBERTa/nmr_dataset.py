"""
NMReDATA Dataset loader for ChemBERTa training
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer
import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class NMReDataParser:
    """Parser for NMReDATA files"""
    
    def __init__(self):
        self.atom_symbols = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
        self.atom_to_idx = {atom: idx for idx, atom in enumerate(self.atom_symbols)}
        
    def parse_nmredata_file(self, filepath: str) -> Dict:
        """Parse a single NMReDATA file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract MOL block
        mol_block_end = content.find('>  <')
        if mol_block_end == -1:
            raise ValueError(f"No MOL block found in {filepath}")
        
        mol_block = content[:mol_block_end]
        
        # Parse molecule from MOL block
        mol = Chem.MolFromMolBlock(mol_block)
        if mol is None:
            raise ValueError(f"Could not parse molecule from {filepath}")
        
        # Get 3D coordinates
        conformer = mol.GetConformer()
        coords = []
        atom_types = []
        
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            atom = mol.GetAtomWithIdx(i)
            atom_types.append(self.atom_to_idx.get(atom.GetSymbol(), len(self.atom_symbols)))
        
        coords = np.array(coords, dtype=np.float32)
        atom_types = np.array(atom_types, dtype=np.int64)
        
        # Extract SMILES
        smiles_match = re.search(r'>  <SMILES>\n(.+)\n', content)
        if smiles_match:
            smiles = smiles_match.group(1).strip()
        else:
            # Generate SMILES from molecule if not found
            smiles = Chem.MolToSmiles(mol)
        
        # Parse NMR data
        nmr_data = self._parse_nmr_data(content)
        
        # Create atom to SMILES position mapping
        atom_to_smiles_pos = self._map_atoms_to_smiles_positions(mol, smiles)
        
        return {
            'filepath': filepath,
            'smiles': smiles,
            'mol': mol,
            'coords': coords,
            'atom_types': atom_types,
            'num_atoms': mol.GetNumAtoms(),
            'nmr_data': nmr_data,
            'atom_to_smiles_pos': atom_to_smiles_pos
        }
    
    def _parse_nmr_data(self, content: str) -> Dict:
        """Parse NMR peaks from NMReDATA content"""
        nmr_data = {
            'H': {'shifts': [], 'atoms': [], 'multiplicities': [], 'couplings': []},
            'C': {'shifts': [], 'atoms': [], 'multiplicities': [], 'couplings': []}
        }
        
        # Parse 1H NMR data
        h_match = re.search(r'>  <NMREDATA_1D_1H>\n(.*?)\n\n', content, re.DOTALL)
        if h_match:
            h_data = h_match.group(1).strip()
            nmr_data['H'] = self._parse_peak_list(h_data)
        
        # Parse 13C NMR data
        c_match = re.search(r'>  <NMREDATA_1D_13C>\n(.*?)\n\n', content, re.DOTALL)
        if c_match:
            c_data = c_match.group(1).strip()
            nmr_data['C'] = self._parse_peak_list(c_data)
        
        return nmr_data
    
    def _parse_peak_list(self, peak_data: str) -> Dict:
        """Parse individual peak list"""
        shifts = []
        atoms = []
        multiplicities = []
        couplings = []
        
        for line in peak_data.split('\n'):
            if not line.strip():
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                # Chemical shift
                try:
                    shift = float(parts[0].strip())
                    shifts.append(shift)
                except ValueError:
                    continue
                
                # Multiplicity
                mult = parts[1].strip()
                multiplicities.append(mult)
                
                # J-couplings
                j_values = []
                atom_nums = []
                
                for i, part in enumerate(parts[2:]):
                    part = part.strip()
                    if 'J=' in part:
                        # Extract J-value
                        j_match = re.search(r'J=([\d.]+)', part)
                        if j_match:
                            j_values.append(float(j_match.group(1)))
                    elif part.replace(',', '').replace('-', '').replace(' ', '').isdigit() or '-' in part:
                        # This is atom assignment
                        if '-' in part:
                            # Range like 1-3
                            start, end = map(int, part.split('-'))
                            atom_nums.extend(range(start-1, end))  # Convert to 0-indexed
                        else:
                            # Single atom or comma-separated
                            for atom_str in part.split(','):
                                if atom_str.strip().isdigit():
                                    atom_nums.append(int(atom_str.strip()) - 1)  # Convert to 0-indexed
                
                atoms.append(atom_nums)
                couplings.append(j_values)
        
        return {
            'shifts': shifts,
            'atoms': atoms,
            'multiplicities': multiplicities,
            'couplings': couplings
        }
    
    def _map_atoms_to_smiles_positions(self, mol, smiles: str) -> Dict[int, List[int]]:
        """Map atom indices to their positions in SMILES string"""
        # This is a simplified mapping - in practice, you might want to use
        # RDKit's atom mapping functionality for more accuracy
        
        atom_to_smiles = {}
        
        # Get SMILES with atom mapping
        mol_copy = Chem.Mol(mol)
        for atom in mol_copy.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        
        mapped_smiles = Chem.MolToSmiles(mol_copy)
        
        # Parse positions from mapped SMILES
        # This is simplified - you may need more sophisticated parsing
        for i in range(mol.GetNumAtoms()):
            atom_to_smiles[i] = [i]  # Placeholder - implement proper mapping
        
        return atom_to_smiles


class NMRDataset(Dataset):
    """PyTorch Dataset for NMReDATA files"""
    
    def __init__(self, 
                 data_directory: str,
                 tokenizer_name: str = 'seyonec/ChemBERTa-zinc-base-v1',
                 max_seq_length: int = 512,
                 max_atoms: int = 200):
        """
        Args:
            data_directory: Directory containing .nmredata files
            tokenizer_name: HuggingFace tokenizer to use
            max_seq_length: Maximum SMILES sequence length
            max_atoms: Maximum number of atoms to consider
        """
        self.data_directory = data_directory
        self.max_seq_length = max_seq_length
        self.max_atoms = max_atoms
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize parser
        self.parser = NMReDataParser()
        
        # Find all NMReDATA files
        self.files = glob.glob(os.path.join(data_directory, "*.nmredata"))
        logger.info(f"Found {len(self.files)} NMReDATA files in {data_directory}")
        
        # Pre-parse all files (can be optimized for large datasets)
        self.data = []
        self.failed_files = []
        
        for filepath in self.files[:1000]:  # Limit for testing
            try:
                parsed_data = self.parser.parse_nmredata_file(filepath)
                if parsed_data['num_atoms'] <= self.max_atoms:
                    self.data.append(parsed_data)
                else:
                    logger.warning(f"Skipping {filepath}: too many atoms ({parsed_data['num_atoms']})")
            except Exception as e:
                logger.error(f"Failed to parse {filepath}: {e}")
                self.failed_files.append(filepath)
        
        logger.info(f"Successfully parsed {len(self.data)} files, failed: {len(self.failed_files)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        sample = self.data[idx]
        
        # Tokenize SMILES
        tokens = self.tokenizer(
            sample['smiles'],
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        # Prepare coordinates (pad if necessary)
        coords = sample['coords']
        num_atoms = len(coords)
        
        if num_atoms < self.max_atoms:
            # Pad with zeros
            padding = np.zeros((self.max_atoms - num_atoms, 3), dtype=np.float32)
            coords = np.vstack([coords, padding])
        
        # Prepare atom types (pad if necessary)
        atom_types = sample['atom_types']
        if len(atom_types) < self.max_atoms:
            padding = np.full(self.max_atoms - len(atom_types), -1, dtype=np.int64)
            atom_types = np.concatenate([atom_types, padding])
        
        # Prepare NMR features
        nmr_features = self._prepare_nmr_features(sample['nmr_data'], num_atoms)
        
        # Create atom mask (1 for real atoms, 0 for padding)
        atom_mask = np.zeros(self.max_atoms, dtype=np.float32)
        atom_mask[:num_atoms] = 1.0
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'coords': torch.tensor(coords, dtype=torch.float32),
            'atom_types': torch.tensor(atom_types, dtype=torch.long),
            'atom_mask': torch.tensor(atom_mask, dtype=torch.float32),
            'nmr_features': nmr_features,
            'num_atoms': num_atoms,
            'smiles': sample['smiles']
        }
    
    def _prepare_nmr_features(self, nmr_data: Dict, num_atoms: int) -> Dict[str, torch.Tensor]:
        """Prepare NMR features for the model"""
        # Initialize feature arrays
        h_shifts = np.zeros(self.max_atoms, dtype=np.float32)
        c_shifts = np.zeros(self.max_atoms, dtype=np.float32)
        h_mask = np.zeros(self.max_atoms, dtype=np.float32)
        c_mask = np.zeros(self.max_atoms, dtype=np.float32)
        
        # Fill H NMR data
        for shift, atoms in zip(nmr_data['H']['shifts'], nmr_data['H']['atoms']):
            for atom_idx in atoms:
                if atom_idx < self.max_atoms:
                    h_shifts[atom_idx] = shift
                    h_mask[atom_idx] = 1.0
        
        # Fill C NMR data
        for shift, atoms in zip(nmr_data['C']['shifts'], nmr_data['C']['atoms']):
            for atom_idx in atoms:
                if atom_idx < self.max_atoms:
                    c_shifts[atom_idx] = shift
                    c_mask[atom_idx] = 1.0
        
        return {
            'h_shifts': torch.tensor(h_shifts, dtype=torch.float32),
            'c_shifts': torch.tensor(c_shifts, dtype=torch.float32),
            'h_mask': torch.tensor(h_mask, dtype=torch.float32),
            'c_mask': torch.tensor(c_mask, dtype=torch.float32)
        }


def create_data_loaders(data_directory: str,
                       batch_size: int = 16,
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       num_workers: int = 4):
    """Create train, validation, and test data loaders"""
    
    # Create dataset
    dataset = NMRDataset(data_directory)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "CSV_to_NMRe_output_v3/"
    
    print(f"Testing dataset with directory: {data_dir}")
    
    # Create dataset
    dataset = NMRDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # Test a single sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  SMILES: {sample['smiles']}")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Coordinates shape: {sample['coords'].shape}")
        print(f"  Number of atoms: {sample['num_atoms']}")
        print(f"  H shifts shape: {sample['nmr_features']['h_shifts'].shape}")
        print(f"  C shifts shape: {sample['nmr_features']['c_shifts'].shape}")