"""
Refactored NMR Dataset class using modular components
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from data_utils import NMReDataParser, DataValidator, DataAugmentation

logger = logging.getLogger(__name__)


class NMRDataset(Dataset):
    """PyTorch Dataset for NMReDATA files with improved error handling and validation"""
    
    def __init__(self, config, split='train'):
        """
        Args:
            config: Configuration object
            split: 'train', 'val', or 'test'
        """
        self.config = config
        self.split = split
        self.max_seq_length = config.model.max_seq_length
        self.max_atoms = config.model.max_atoms
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.chemberta_name)
        
        # Initialize parser and validator
        self.parser = NMReDataParser()
        self.validator = DataValidator()
        self.augmentation = DataAugmentation()
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"{split.upper()} dataset: {len(self.data)} samples")
        
        # Log statistics
        if self.data:
            stats = self.validator.get_data_statistics(self.data)
            logger.info(f"Dataset statistics: {stats}")
    
    def _load_data(self) -> List[Dict]:
        """Load and parse all data files"""
        # Find all NMReDATA files
        files = glob.glob(os.path.join(self.config.data.data_directory, "*.nmredata"))
        
        if not files:
            raise ValueError(f"No .nmredata files found in {self.config.data.data_directory}")
        
        logger.info(f"Found {len(files)} NMReDATA files")
        
        # Limit files for testing if specified
        if self.config.data.max_files_limit:
            files = files[:self.config.data.max_files_limit]
            logger.info(f"Limited to {len(files)} files for testing")
        
        # Parse files in parallel
        parsed_data = self._parse_files_parallel(files)
        
        # Validate data
        valid_data = []
        for sample in parsed_data:
            if sample and self.validator.validate_sample(sample, self.max_atoms):
                valid_data.append(sample)
        
        logger.info(f"Successfully validated {len(valid_data)}/{len(parsed_data)} samples")
        
        return valid_data
    
    def _parse_files_parallel(self, files: List[str]) -> List[Optional[Dict]]:
        """Parse files in parallel for faster loading"""
        parsed_data = []
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=min(8, len(files))) as executor:
            # Submit all parsing tasks
            future_to_file = {
                executor.submit(self.parser.parse_nmredata_file, filepath): filepath 
                for filepath in files
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_file), 
                             total=len(files), 
                             desc="Parsing files"):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        parsed_data.append(result)
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error parsing {filepath}: {e}")
                    failed_count += 1
        
        logger.info(f"Parsing completed: {len(parsed_data)} success, {failed_count} failed")
        return parsed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample with optional augmentation"""
        sample = self.data[idx]
        
        # Apply data augmentation for training
        if self.split == 'train':
            sample = self._apply_augmentation(sample)
        
        # Tokenize SMILES
        tokens = self._tokenize_smiles(sample['smiles'])
        
        # Prepare molecular features
        mol_features = self._prepare_molecular_features(sample)
        
        # Prepare NMR features
        nmr_features = self._prepare_nmr_features(sample['nmr_data'], sample['num_atoms'])
        
        return {
            **tokens,
            **mol_features,
            'nmr_features': nmr_features,
            'smiles': sample['smiles'],
            'file_id': os.path.basename(sample['filepath'])
        }
    
    def _apply_augmentation(self, sample: Dict) -> Dict:
        """Apply data augmentation techniques"""
        augmented_sample = sample.copy()
        
        # Coordinate augmentation (50% chance)
        if np.random.random() < 0.5:
            # Add noise
            if np.random.random() < 0.3:
                augmented_sample['coords'] = self.augmentation.add_coordinate_noise(
                    sample['coords']
                )
            
            # Random rotation
            if np.random.random() < 0.7:
                augmented_sample['coords'] = self.augmentation.rotate_molecule(
                    augmented_sample['coords']
                )
        
        # NMR noise augmentation (30% chance)
        if np.random.random() < 0.3:
            nmr_data = augmented_sample['nmr_data'].copy()
            
            # Add noise to H shifts
            if nmr_data['H']['shifts']:
                h_shifts = np.array(nmr_data['H']['shifts'])
                h_shifts = self.augmentation.add_nmr_noise(h_shifts)
                nmr_data['H']['shifts'] = h_shifts.tolist()
            
            # Add noise to C shifts
            if nmr_data['C']['shifts']:
                c_shifts = np.array(nmr_data['C']['shifts'])
                c_shifts = self.augmentation.add_nmr_noise(c_shifts)
                nmr_data['C']['shifts'] = c_shifts.tolist()
            
            augmented_sample['nmr_data'] = nmr_data
        
        return augmented_sample
    
    def _tokenize_smiles(self, smiles: str) -> Dict[str, torch.Tensor]:
        """Tokenize SMILES string"""
        tokens = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze()
        }
    
    def _prepare_molecular_features(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Prepare molecular structure features"""
        coords = sample['coords']
        atom_types = sample['atom_types']
        num_atoms = sample['num_atoms']
        
        # Pad coordinates if necessary
        if num_atoms < self.max_atoms:
            padding = np.zeros((self.max_atoms - num_atoms, 3), dtype=np.float32)
            coords = np.vstack([coords, padding])
        else:
            coords = coords[:self.max_atoms]
            num_atoms = min(num_atoms, self.max_atoms)
        
        # Pad atom types if necessary
        if len(atom_types) < self.max_atoms:
            padding = np.full(self.max_atoms - len(atom_types), -1, dtype=np.int64)
            atom_types = np.concatenate([atom_types, padding])
        else:
            atom_types = atom_types[:self.max_atoms]
        
        # Create atom mask (1 for real atoms, 0 for padding)
        atom_mask = np.zeros(self.max_atoms, dtype=np.float32)
        atom_mask[:num_atoms] = 1.0
        
        return {
            'coords': torch.tensor(coords, dtype=torch.float32),
            'atom_types': torch.tensor(atom_types, dtype=torch.long),
            'atom_mask': torch.tensor(atom_mask, dtype=torch.float32),
            'num_atoms': num_atoms
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
                if 0 <= atom_idx < min(self.max_atoms, num_atoms):
                    h_shifts[atom_idx] = shift
                    h_mask[atom_idx] = 1.0
        
        # Fill C NMR data
        for shift, atoms in zip(nmr_data['C']['shifts'], nmr_data['C']['atoms']):
            for atom_idx in atoms:
                if 0 <= atom_idx < min(self.max_atoms, num_atoms):
                    c_shifts[atom_idx] = shift
                    c_mask[atom_idx] = 1.0
        
        return {
            'h_shifts': torch.tensor(h_shifts, dtype=torch.float32),
            'c_shifts': torch.tensor(c_shifts, dtype=torch.float32),
            'h_mask': torch.tensor(h_mask, dtype=torch.float32),
            'c_mask': torch.tensor(c_mask, dtype=torch.float32)
        }
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a specific sample"""
        sample = self.data[idx]
        return {
            'filepath': sample['filepath'],
            'smiles': sample['smiles'],
            'num_atoms': sample['num_atoms'],
            'h_peaks': len(sample['nmr_data']['H']['shifts']),
            'c_peaks': len(sample['nmr_data']['C']['shifts'])
        }


def create_data_loaders(config) -> tuple:
    """Create train, validation, and test data loaders"""
    
    # Create datasets
    full_dataset = NMRDataset(config, split='full')
    
    if len(full_dataset) == 0:
        raise ValueError("No valid samples found in dataset")
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config.data.train_split * total_size)
    val_size = int(config.data.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Ensure minimum sizes
    if train_size < 1 or val_size < 1:
        raise ValueError(f"Dataset too small for splits: {total_size} total samples")
    
    # Random split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Update split information for datasets
    train_dataset.dataset.split = 'train'
    val_dataset.dataset.split = 'val'
    test_dataset.dataset.split = 'test'
    
    # Determine optimal number of workers
    from hardware_utils import get_num_workers
    num_workers = get_num_workers(config.data.batch_size)
    
    logger.info(f"Using {num_workers} workers for data loading")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True  # For stable batch sizes during training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=num_workers > 0
    ) if test_size > 0 else None
    
    logger.info(f"Data splits - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset) if test_dataset else 0}")
    
    return train_loader, val_loader, test_loader, full_dataset


class DatasetAnalyzer:
    """Utility class for analyzing dataset properties"""
    
    def __init__(self, dataset: NMRDataset):
        self.dataset = dataset
    
    def analyze_distribution(self) -> Dict:
        """Analyze the distribution of various properties"""
        if not self.dataset.data:
            return {}
        
        analysis = {}
        
        # Atom count distribution
        atom_counts = [sample['num_atoms'] for sample in self.dataset.data]
        analysis['atom_counts'] = {
            'min': min(atom_counts),
            'max': max(atom_counts),
            'mean': np.mean(atom_counts),
            'std': np.std(atom_counts),
            'percentiles': {
                '25': np.percentile(atom_counts, 25),
                '50': np.percentile(atom_counts, 50),
                '75': np.percentile(atom_counts, 75),
                '95': np.percentile(atom_counts, 95)
            }
        }
        
        # SMILES length distribution
        smiles_lengths = [len(sample['smiles']) for sample in self.dataset.data]
        analysis['smiles_lengths'] = {
            'min': min(smiles_lengths),
            'max': max(smiles_lengths),
            'mean': np.mean(smiles_lengths),
            'std': np.std(smiles_lengths)
        }
        
        # Atom type distribution
        atom_type_counts = {}
        for sample in self.dataset.data:
            for atom_type in sample['atom_types']:
                if atom_type >= 0:  # Skip padding
                    atom_type_counts[atom_type] = atom_type_counts.get(atom_type, 0) + 1
        
        analysis['atom_types'] = atom_type_counts
        
        # NMR peak distribution
        h_peak_counts = []
        c_peak_counts = []
        for sample in self.dataset.data:
            h_peak_counts.append(len(sample['nmr_data']['H']['shifts']))
            c_peak_counts.append(len(sample['nmr_data']['C']['shifts']))
        
        analysis['nmr_peaks'] = {
            'h_peaks': {
                'mean': np.mean(h_peak_counts),
                'std': np.std(h_peak_counts),
                'max': max(h_peak_counts) if h_peak_counts else 0
            },
            'c_peaks': {
                'mean': np.mean(c_peak_counts),
                'std': np.std(c_peak_counts),
                'max': max(c_peak_counts) if c_peak_counts else 0
            }
        }
        
        return analysis
    
    def find_problematic_samples(self) -> List[Dict]:
        """Find samples that might cause issues during training"""
        problematic = []
        
        for i, sample in enumerate(self.dataset.data):
            issues = []
            
            # Check for very large molecules
            if sample['num_atoms'] > self.dataset.max_atoms * 0.9:
                issues.append(f"Large molecule: {sample['num_atoms']} atoms")
            
            # Check for molecules with no NMR data
            if (len(sample['nmr_data']['H']['shifts']) == 0 and 
                len(sample['nmr_data']['C']['shifts']) == 0):
                issues.append("No NMR data")
            
            # Check for very long SMILES
            if len(sample['smiles']) > self.dataset.max_seq_length * 0.9:
                issues.append(f"Long SMILES: {len(sample['smiles'])} chars")
            
            # Check coordinate bounds
            coords = sample['coords']
            if np.any(np.abs(coords) > 100):  # Reasonable coordinate bounds
                issues.append("Large coordinate values")
            
            if issues:
                problematic.append({
                    'index': i,
                    'filepath': sample['filepath'],
                    'issues': issues,
                    'smiles': sample['smiles'][:50] + '...' if len(sample['smiles']) > 50 else sample['smiles']
                })
        
        return problematic


if __name__ == "__main__":
    # Test the refactored dataset
    from config import get_default_config
    
    config = get_default_config()
    config.data.data_directory = "CSV_to_NMRe_output_v3/"
    config.data.max_files_limit = 100  # Limit for testing
    
    print("Testing refactored dataset...")
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader, dataset = create_data_loaders(config)
        
        print(f"Successfully created data loaders:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader) if test_loader else 0}")
        
        # Test a single batch
        batch = next(iter(train_loader))
        print(f"\nSample batch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f"    {subkey}: {subvalue.shape}")
        
        # Analyze dataset
        analyzer = DatasetAnalyzer(dataset)
        analysis = analyzer.analyze_distribution()
        print(f"\nDataset analysis:")
        print(f"  Atom counts: {analysis['atom_counts']['mean']:.1f} ± {analysis['atom_counts']['std']:.1f}")
        print(f"  SMILES lengths: {analysis['smiles_lengths']['mean']:.1f} ± {analysis['smiles_lengths']['std']:.1f}")
        print(f"  H NMR peaks: {analysis['nmr_peaks']['h_peaks']['mean']:.1f}")
        print(f"  C NMR peaks: {analysis['nmr_peaks']['c_peaks']['mean']:.1f}")
        
        # Check for problematic samples
        problems = analyzer.find_problematic_samples()
        if problems:
            print(f"\nFound {len(problems)} potentially problematic samples")
            for prob in problems[:3]:  # Show first 3
                print(f"  {prob['filepath']}: {', '.join(prob['issues'])}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()