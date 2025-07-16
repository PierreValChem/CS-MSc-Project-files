#!/usr/bin/env python3
"""
Script to copy complete/perfect NMR compound files to a separate directory
This makes it easier to load only high-quality data for machine learning
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_complete_compounds_list(file_path):
    """
    Parse the complete_data_compounds.txt file to get list of compound IDs and filenames
    
    Format expected: NP0122619, NP0122619_Rhaponticin.nmredata
    
    Args:
        file_path: Path to complete_data_compounds.txt
        
    Returns:
        dict: Dictionary mapping compound IDs to expected filenames
    """
    complete_compounds = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Split by comma
                parts = line.split(',')
                if len(parts) >= 2:
                    compound_id = parts[0].strip()
                    filename = parts[1].strip()
                    
                    if compound_id and filename:
                        # Remove .nmredata extension if present to get the base filename
                        if filename.endswith('.nmredata'):
                            filename_base = filename[:-9]  # Remove '.nmredata'
                        else:
                            filename_base = filename
                            
                        complete_compounds[compound_id] = filename_base
                        
        logger.info(f"Loaded {len(complete_compounds)} complete compound entries")
        return complete_compounds
        
    except Exception as e:
        logger.error(f"Error reading complete compounds list: {e}")
        raise

def copy_complete_compounds(source_dir, dest_dir, complete_compounds_file):
    """
    Copy only complete compound .nmredata files to destination directory
    
    Args:
        source_dir: Directory containing all .nmredata files
        dest_dir: Destination directory for complete compounds
        complete_compounds_file: Path to complete_data_compounds.txt
    """
    # Convert to Path objects
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    complete_file_path = Path(complete_compounds_file)
    
    # Validate source directory
    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_path}")
        return
    
    # Validate complete compounds file
    if not complete_file_path.exists():
        # Try looking in the source directory
        complete_file_path = source_path / 'complete_data_compounds.txt'
        if not complete_file_path.exists():
            logger.error(f"Complete compounds file not found: {complete_compounds_file}")
            return
    
    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Destination directory: {dest_path}")
    
    # Parse complete compounds list
    logger.info(f"Reading complete compounds from: {complete_file_path}")
    complete_compounds_dict = parse_complete_compounds_list(complete_file_path)
    
    if not complete_compounds_dict:
        logger.error("No complete compounds found in the list!")
        return
    
    # Find all .nmredata files in source directory
    nmredata_files = list(source_path.glob('*.nmredata'))
    logger.info(f"Found {len(nmredata_files)} .nmredata files in source directory")
    
    # Create a mapping of file stems to full paths for faster lookup
    file_map = {}
    for file_path in nmredata_files:
        # Get the filename without extension
        file_stem = file_path.stem
        file_map[file_stem] = file_path
        
        # Also try just the compound ID part (before first underscore)
        compound_id_only = file_stem.split('_')[0]
        if compound_id_only not in file_map:
            file_map[compound_id_only] = file_path
    
    # Copy matching files
    copied_count = 0
    found_compounds = set()
    missing_compounds = {}
    file_size_total = 0
    
    logger.info("Matching and copying complete compound files...")
    
    with tqdm(total=len(complete_compounds_dict), desc="Copying files") as pbar:
        for compound_id, expected_filename in complete_compounds_dict.items():
            file_found = False
            
            # Try different matching strategies
            # 1. Try exact match with expected filename
            if expected_filename in file_map:
                source_file = file_map[expected_filename]
                file_found = True
            # 2. Try just the compound ID
            elif compound_id in file_map:
                source_file = file_map[compound_id]
                file_found = True
            # 3. Try finding any file that starts with the compound ID
            else:
                for file_stem, file_path in file_map.items():
                    if file_stem.startswith(compound_id):
                        source_file = file_path
                        file_found = True
                        break
            
            if file_found:
                # Copy file to destination
                dest_file = dest_path / source_file.name
                
                try:
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
                    file_size_total += source_file.stat().st_size
                    found_compounds.add(compound_id)
                    
                except Exception as e:
                    logger.error(f"Error copying {source_file.name}: {e}")
            else:
                missing_compounds[compound_id] = expected_filename
            
            pbar.update(1)
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("COPY OPERATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Complete compounds in list: {len(complete_compounds_dict)}")
    logger.info(f"Files successfully copied: {copied_count}")
    logger.info(f"Total size copied: {file_size_total / (1024**2):.2f} MB")
    logger.info(f"Missing compounds: {len(missing_compounds)}")
    
    if missing_compounds:
        logger.warning(f"\nCompounds in complete list but files not found: {len(missing_compounds)}")
        # Save missing compounds to file for investigation
        missing_file = dest_path / 'missing_compounds.txt'
        with open(missing_file, 'w') as f:
            f.write("# Compounds listed as complete but .nmredata files not found\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: compound_id, expected_filename\n\n")
            for compound_id, filename in sorted(missing_compounds.items()):
                f.write(f"{compound_id}, {filename}\n")
        logger.info(f"Missing compounds list saved to: {missing_file}")
        
        # Show first few missing examples
        logger.info("\nFirst few missing compounds:")
        for i, (cid, fname) in enumerate(list(missing_compounds.items())[:5]):
            logger.info(f"  {cid} -> expected: {fname}.nmredata")
    
    # Show some successful matches for verification
    if copied_count > 0:
        logger.info("\nFirst few successful copies:")
        copied_files = list(dest_path.glob('*.nmredata'))[:5]
        for f in copied_files:
            logger.info(f"  ✓ {f.name}")
    
    # Create summary report
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_directory': str(source_path),
        'destination_directory': str(dest_path),
        'complete_compounds_file': str(complete_file_path),
        'total_complete_compounds': len(complete_compounds_dict),
        'files_copied': copied_count,
        'files_missing': len(missing_compounds),
        'total_size_mb': round(file_size_total / (1024**2), 2),
        'success_rate': round((copied_count / len(complete_compounds_dict)) * 100, 2) if complete_compounds_dict else 0
    }
    
    summary_file = dest_path / 'copy_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary report saved to: {summary_file}")
    logger.info("="*60)

def verify_copied_files(dest_dir, complete_compounds_file):
    """
    Verify that all complete compounds were copied successfully
    
    Args:
        dest_dir: Destination directory with copied files
        complete_compounds_file: Path to complete_data_compounds.txt
    """
    dest_path = Path(dest_dir)
    
    if not dest_path.exists():
        logger.error(f"Destination directory does not exist: {dest_path}")
        return
    
    # Parse complete compounds list
    complete_compounds_dict = parse_complete_compounds_list(complete_compounds_file)
    complete_compound_ids = set(complete_compounds_dict.keys())
    
    # Check copied files
    copied_files = list(dest_path.glob('*.nmredata'))
    copied_ids = set()
    
    # Extract compound IDs from copied files
    for f in copied_files:
        # Get compound ID (first part before underscore)
        compound_id = f.stem.split('_')[0]
        copied_ids.add(compound_id)
    
    # Verify
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION RESULTS")
    logger.info("="*60)
    logger.info(f"Complete compounds expected: {len(complete_compound_ids)}")
    logger.info(f"Files found in destination: {len(copied_files)}")
    logger.info(f"Unique compound IDs found: {len(copied_ids)}")
    logger.info(f"Match rate: {len(copied_ids.intersection(complete_compound_ids)) / len(complete_compound_ids) * 100:.2f}%")
    
    missing = complete_compound_ids - copied_ids
    if missing:
        logger.warning(f"Still missing {len(missing)} compounds")
        logger.info("First few missing compound IDs:")
        for cid in list(missing)[:10]:
            logger.info(f"  - {cid}")
    else:
        logger.info("✓ All complete compounds successfully copied!")
    
    extra = copied_ids - complete_compound_ids
    if extra:
        logger.warning(f"Found {len(extra)} extra files not in complete list")
        logger.info("First few extra compound IDs:")
        for cid in list(extra)[:10]:
            logger.info(f"  - {cid}")

def main():
    """Main execution function"""
    
    # Configuration
    SOURCE_DIR = r'C:\Users\pierr\Desktop\CS MSc Project files\peaklist\outputv6'
    DEST_DIR = r'C:\Users\pierr\Desktop\CS MSc Project files\peaklist\complete_compounds_only_explicit_h'
    COMPLETE_COMPOUNDS_FILE = r'C:\Users\pierr\Desktop\CS MSc Project files\peaklist\outputv6\complete_data_compounds.txt'
    
    # You can modify these paths as needed
    print("NMR Complete Compounds Copy Utility")
    print("="*60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print(f"Complete list: {COMPLETE_COMPOUNDS_FILE}")
    print("="*60)
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with copying? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        # Copy files
        copy_complete_compounds(SOURCE_DIR, DEST_DIR, COMPLETE_COMPOUNDS_FILE)
        
        # Verify
        print("\nVerifying copied files...")
        verify_copied_files(DEST_DIR, COMPLETE_COMPOUNDS_FILE)
        
        print("\nProcess complete!")
        
        # Additional useful info
        dest_path = Path(DEST_DIR)
        if dest_path.exists():
            file_count = len(list(dest_path.glob('*.nmredata')))
            total_size = sum(f.stat().st_size for f in dest_path.glob('*.nmredata'))
            print(f"\nYou can now use '{DEST_DIR}' for machine learning")
            print(f"It contains {file_count} complete compound files ({total_size/(1024**3):.2f} GB)")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()

# Alternative: Function for use in other scripts
def get_complete_compounds_directory(source_dir, create_if_missing=True):
    """
    Get or create directory with only complete compounds
    Useful for importing in ML scripts
    
    Args:
        source_dir: Original data directory
        create_if_missing: Whether to create and populate if it doesn't exist
        
    Returns:
        Path: Path to complete compounds directory
    """
    source_path = Path(source_dir)
    complete_dir = source_path.parent / 'complete_compounds_only'
    
    if not complete_dir.exists() and create_if_missing:
        logger.info(f"Creating complete compounds directory: {complete_dir}")
        complete_file = source_path / 'complete_data_compounds.txt'
        
        if complete_file.exists():
            copy_complete_compounds(source_path, complete_dir, complete_file)
        else:
            logger.error("Cannot find complete_data_compounds.txt")
            return None
    
    return complete_dir if complete_dir.exists() else None