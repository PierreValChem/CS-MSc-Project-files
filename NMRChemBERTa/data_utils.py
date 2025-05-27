"""
Data processing utilities separated from main dataset class
"""

import os
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NMReDataParser:
    """Parser for NMReDATA files"""
    
    def __init__(self):
        self.atom_symbols = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
        self.atom_to_idx = {atom: idx + 1 for idx, atom in enumerate(self.atom_symbols)}
        self.atom_to_idx['UNK'] = 0  # Unknown atom type
        
    def parse_nmredata_file(self, filepath: str) -> Optional[Dict]:
        """Parse a single NMReDATA file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract and parse MOL block
            mol_data = self._extract_mol_block(content, filepath)
            if mol_data is None:
                return None
            
            # Extract SMILES
            smiles = self._extract_smiles(content, mol_data['mol'])
            
            # Parse NMR data
            nmr_data = self._parse_nmr_data(content)
            
            # Create atom to SMILES position mapping
            atom_to_smiles_pos = self._map_atoms_to_smiles_positions(
                mol_data['mol'], smiles
            )
            
            return {
                'filepath': filepath,
                'smiles': smiles,
                'mol': mol_data['mol'],
                'coords': mol_data['coords'],
                'atom_types': mol_data['atom_types'],
                'num_atoms': mol_data['num_atoms'],
                'nmr_data': nmr_data,
                'atom_to_smiles_pos': atom_to_smiles_pos
            }
            
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None
    
    def _extract_mol_block(self, content: str, filepath: str) -> Optional[Dict]:
        """Extract and parse MOL block from content"""
        mol_block_end = content.find('>  <')
        if mol_block_end == -1:
            logger.warning(f"No MOL block found in {filepath}")
            return None
        
        mol_block = content[:mol_block_end]
        
        # Parse molecule from MOL block
        mol = Chem.MolFromMolBlock(mol_block)
        if mol is None:
            logger.warning(f"Could not parse molecule from {filepath}")
            return None
        
        # Ensure molecule has 3D coordinates
        if not mol.GetNumConformers():
            logger.warning(f"No conformers found in {filepath}")
            return None
        
        # Get 3D coordinates and atom types
        conformer = mol.GetConformer()
        coords = []
        atom_types = []
        
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            
            atom = mol.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            atom_types.append(self.atom_to_idx.get(symbol, 0))  # 0 for unknown
        
        return {
            'mol': mol,
            'coords': np.array(coords, dtype=np.float32),
            'atom_types': np.array(atom_types, dtype=np.int64),
            'num_atoms': mol.GetNumAtoms()
        }
    
    def _extract_smiles(self, content: str, mol: Chem.Mol) -> str:
        """Extract SMILES from content or generate from molecule"""
        smiles_match = re.search(r'>  <SMILES>\n(.+)\n', content)
        if smiles_match:
            smiles = smiles_match.group(1).strip()
            # Validate SMILES
            test_mol = Chem.MolFromSmiles(smiles)
            if test_mol is not None:
                return smiles
        
        # Generate SMILES from molecule if not found or invalid
        return Chem.MolToSmiles(mol)
    
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
        """Parse individual peak list with robust error handling"""
        shifts = []
        atoms = []
        multiplicities = []
        couplings = []
        
        for line in peak_data.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) < 2:
                    continue
                
                # Chemical shift (first column)
                shift = float(parts[0])
                shifts.append(shift)
                
                # Multiplicity (second column)
                mult = parts[1]
                multiplicities.append(mult)
                
                # Parse atom assignments and J-couplings
                atom_nums = []
                j_values = []
                
                for part in parts[2:]:
                    if 'J=' in part:
                        # Extract J-coupling values
                        j_matches = re.findall(r'J=([\d.]+)', part)
                        j_values.extend([float(j) for j in j_matches])
                    else:
                        # Parse atom assignments
                        atom_nums.extend(self._parse_atom_assignment(part))
                
                atoms.append(atom_nums)
                couplings.append(j_values)
                
            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping malformed peak line: {line} ({e})")
                continue
        
        return {
            'shifts': shifts,
            'atoms': atoms,
            'multiplicities': multiplicities,
            'couplings': couplings
        }
    
    def _parse_atom_assignment(self, assignment_str: str) -> List[int]:
        """Parse atom assignment string (e.g., '1-3', '1,2,3', '5')"""
        atom_nums = []
        assignment_str = assignment_str.replace(' ', '')
        
        if '-' in assignment_str and assignment_str.count('-') == 1:
            # Range like '1-3'
            try:
                start, end = assignment_str.split('-')
                start, end = int(start), int(end)
                atom_nums.extend(range(start - 1, end))  # Convert to 0-indexed
            except ValueError:
                pass
        else:
            # Comma-separated or single number
            for atom_str in assignment_str.split(','):
                atom_str = atom_str.strip()
                if atom_str.isdigit():
                    atom_nums.append(int(atom_str) - 1)  # Convert to 0-indexed
        
        return atom_nums
    
    def _map_atoms_to_smiles_positions(self, mol: Chem.Mol, smiles: str) -> Dict[int, List[int]]:
        """Map atom indices to their positions in SMILES string"""
        # This is a simplified mapping - in practice, more sophisticated
        # mapping using RDKit's atom mapping functionality would be better
        atom_to_smiles = {}
        
        try:
            # Create a copy with atom mapping
            mol_copy = Chem.Mol(mol)
            for atom in mol_copy.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)
            
            # For now, use simple position mapping
            # TODO: Implement proper SMILES-to-atom mapping
            for i in range(mol.GetNumAtoms()):
                atom_to_smiles[i] = [i % len(smiles)]  # Placeholder
                
        except Exception as e:
            logger.debug(f"SMILES mapping failed: {e}")
            # Fallback to simple mapping
            for i in range(mol.GetNumAtoms()):
                atom_to_smiles[i] = [i]
        
        return atom_to_smiles


class DataValidator:
    """Validate parsed data for quality and consistency"""
    
    @staticmethod
    def validate_sample(sample: Dict, max_atoms: int) -> bool:
        """Validate a single data sample"""
        try:
            # Check basic structure
            required_keys = ['smiles', 'coords', 'atom_types', 'num_atoms', 'nmr_data']
            if not all(key in sample for key in required_keys):
                return False
            
            # Check atom count consistency
            if sample['num_atoms'] != len(sample['coords']):
                return False
            
            if sample['num_atoms'] != len(sample['atom_types']):
                return False
            
            # Check reasonable molecule size
            if sample['num_atoms'] > max_atoms or sample['num_atoms'] < 1:
                return False
            
            # Check coordinate validity
            coords = sample['coords']
            if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                return False
            
            # Check SMILES validity
            mol = Chem.MolFromSmiles(sample['smiles'])
            if mol is None:
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def get_data_statistics(samples: List[Dict]) -> Dict:
        """Get statistics about the dataset"""
        if not samples:
            return {}
        
        num_atoms = [s['num_atoms'] for s in samples]
        smiles_lengths = [len(s['smiles']) for s in samples]
        
        # Count NMR data points
        h_peaks = []
        c_peaks = []
        for s in samples:
            h_peaks.append(len(s['nmr_data']['H']['shifts']))
            c_peaks.append(len(s['nmr_data']['C']['shifts']))
        
        return {
            'total_samples': len(samples),
            'atom_count': {
                'mean': np.mean(num_atoms),
                'std': np.std(num_atoms),
                'min': np.min(num_atoms),
                'max': np.max(num_atoms)
            },
            'smiles_length': {
                'mean': np.mean(smiles_lengths),
                'std': np.std(smiles_lengths),
                'min': np.min(smiles_lengths),
                'max': np.max(smiles_lengths)
            },
            'nmr_peaks': {
                'h_mean': np.mean(h_peaks),
                'c_mean': np.mean(c_peaks),
                'h_total': np.sum(h_peaks),
                'c_total': np.sum(c_peaks)
            }
        }


class DataAugmentation:
    """Data augmentation techniques for molecular data"""
    
    @staticmethod
    def add_coordinate_noise(coords: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to coordinates"""
        noise = np.random.normal(0, noise_std, coords.shape)
        return coords + noise.astype(coords.dtype)
    
    @staticmethod
    def rotate_molecule(coords: np.ndarray) -> np.ndarray:
        """Apply random rotation to molecular coordinates"""
        # Generate random rotation matrix
        angles = np.random.uniform(0, 2*np.pi, 3)
        
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        return (coords @ R.T).astype(coords.dtype)
    
    @staticmethod
    def add_nmr_noise(shifts: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """Add noise to NMR chemical shifts"""
        noise = np.random.normal(0, noise_std, shifts.shape)
        return shifts + noise.astype(shifts.dtype)