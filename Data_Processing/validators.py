"""
Validation functions for NMReDATA files
"""

import os
import glob
import logging
from Data_Processing.utils import setup_logging

logger = setup_logging()


def validate_nmredata_file(file_path):
    """Validate that the created NMReDATA file is properly formatted"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required tags
        required_tags = ['NMREDATA_VERSION', 'NMREDATA_1D_1H']
        missing_tags = []
        for tag in required_tags:
            if tag not in content:
                missing_tags.append(tag)
        
        if missing_tags:
            logger.warning(f"Missing required tags {missing_tags} in {file_path}")
            return False
        
        # Check for MOL block
        lines = content.split('\n')
        if len(lines) < 4:
            logger.error(f"File too short to contain valid MOL block in {file_path}")
            return False
        
        # Check for proper termination
        if '$$$$' not in content:
            logger.warning(f"Missing $$$$ terminator in {file_path}")
            return False
        
        # Check MOL block is not empty
        mol_block_end = content.find('>  <')
        if mol_block_end > 0:
            mol_block = content[:mol_block_end].strip()
            if len(mol_block) < 50:  # Very basic check for minimum content
                logger.warning(f"MOL block appears too short in {file_path}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating {file_path}: {e}")
        return False


def batch_validate(output_directory):
    """Validate all NMReDATA files in the output directory"""
    nmredata_files = glob.glob(os.path.join(output_directory, "*.nmredata"))
    
    logger.info(f"Validating {len(nmredata_files)} NMReDATA files...")
    
    valid_count = 0
    invalid_files = []
    
    for file_path in nmredata_files:
        if validate_nmredata_file(file_path):
            valid_count += 1
        else:
            invalid_files.append(os.path.basename(file_path))
    
    logger.info(f"Validation complete: {valid_count}/{len(nmredata_files)} files are valid")
    
    if invalid_files:
        logger.warning(f"Invalid files: {invalid_files[:10]}...")  # Show first 10
        
        # Save list of invalid files
        invalid_list_path = os.path.join(output_directory, 'invalid_nmredata_files.txt')
        with open(invalid_list_path, 'w') as f:
            for filename in invalid_files:
                f.write(f"{filename}\n")
        logger.info(f"List of invalid files saved to: {invalid_list_path}")
    
    return valid_count, invalid_files


def clean_invalid_files(output_directory):
    """Remove invalid NMReDATA files from output directory"""
    nmredata_files = glob.glob(os.path.join(output_directory, "*.nmredata"))
    
    logger.info(f"Checking {len(nmredata_files)} NMReDATA files for validity...")
    
    removed_count = 0
    
    for file_path in nmredata_files:
        if not validate_nmredata_file(file_path):
            logger.info(f"Removing invalid file: {os.path.basename(file_path)}")
            os.remove(file_path)
            removed_count += 1
    
    logger.info(f"Cleanup complete: Removed {removed_count} invalid files")
    return removed_count