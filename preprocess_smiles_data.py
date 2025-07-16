#!/usr/bin/env python3
"""
Preprocess SMILES to remove explicit hydrogens and standardize format
"""

import os
import re
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_smiles(smiles):
    """Clean and standardize SMILES string"""
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Remove explicit hydrogens
        mol = Chem.RemoveHs(mol)
        
        # Get canonical SMILES without explicit H
        clean_smiles = Chem.MolToSmiles(mol, canonical=True)
        
        return clean_smiles
    except:
        return None

def process_nmredata_file(filepath, output_path):
    """Process a single .nmredata file to clean SMILES"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find canonical SMILES section
        smiles_match = re.search(r'(>  <Canonical_SMILES>\n)([^\n]+)', content)
        
        if smiles_match:
            original_smiles = smiles_match.group(2).strip()
            
            # Clean the SMILES
            cleaned_smiles = clean_smiles(original_smiles)
            
            if cleaned_smiles:
                # Replace in content
                content = content.replace(
                    smiles_match.group(0),
                    smiles_match.group(1) + cleaned_smiles
                )
                
                # Write to output
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return True, original_smiles, cleaned_smiles
            else:
                # Copy original if cleaning failed
                shutil.copy2(filepath, output_path)
                return False, original_smiles, None
        else:
            # Copy original if no SMILES found
            shutil.copy2(filepath, output_path)
            return False, None, None
            
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return False, None, None

def compare_smiles(original, cleaned):
    """Compare original and cleaned SMILES"""
    try:
        mol1 = Chem.MolFromSmiles(original)
        mol2 = Chem.MolFromSmiles(cleaned)
        
        if mol1 and mol2:
            # Check if they represent the same molecule
            return (
                mol1.GetNumAtoms() == mol2.GetNumAtoms(),
                len(original),
                len(cleaned),
                original.count('[H]'),
                cleaned.count('[H]')
            )
    except:
        pass
    return False, 0, 0, 0, 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Clean SMILES in NMR data files')
    parser.add_argument('input_dir', help='Input directory with .nmredata files')
    parser.add_argument('output_dir', help='Output directory for cleaned files')
    parser.add_argument('--test', action='store_true', help='Test mode - process only 10 files')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all nmredata files
    files = list(input_path.glob('*.nmredata'))
    if args.test:
        files = files[:10]
    
    logger.info(f"Processing {len(files)} files...")
    
    # Statistics
    stats = {
        'total': len(files),
        'cleaned': 0,
        'failed': 0,
        'no_change': 0,
        'total_h_removed': 0,
        'avg_length_reduction': []
    }
    
    # Process files
    for filepath in tqdm(files, desc="Cleaning SMILES"):
        output_file = output_path / filepath.name
        
        success, original, cleaned = process_nmredata_file(filepath, output_file)
        
        if success and original and cleaned:
            stats['cleaned'] += 1
            
            # Compare
            same_mol, orig_len, clean_len, orig_h, clean_h = compare_smiles(original, cleaned)
            
            if same_mol:
                stats['total_h_removed'] += (orig_h - clean_h)
                stats['avg_length_reduction'].append(orig_len - clean_len)
                
                if original == cleaned:
                    stats['no_change'] += 1
        else:
            stats['failed'] += 1
    
    # Copy complete_data_compounds.txt if it exists
    complete_file = input_path / 'complete_data_compounds.txt'
    if complete_file.exists():
        shutil.copy2(complete_file, output_path / 'complete_data_compounds.txt')
        logger.info("Copied complete_data_compounds.txt")
    
    # Report statistics
    print("\n" + "="*60)
    print("CLEANING SUMMARY:")
    print("="*60)
    print(f"Total files: {stats['total']}")
    print(f"Successfully cleaned: {stats['cleaned']}")
    print(f"No change needed: {stats['no_change']}")
    print(f"Failed to clean: {stats['failed']}")
    
    if stats['avg_length_reduction']:
        avg_reduction = sum(stats['avg_length_reduction']) / len(stats['avg_length_reduction'])
        print(f"\nAverage length reduction: {avg_reduction:.1f} characters")
        print(f"Total [H] removed: {stats['total_h_removed']}")
    
    print(f"\nCleaned files saved to: {output_path}")

if __name__ == "__main__":
    main()