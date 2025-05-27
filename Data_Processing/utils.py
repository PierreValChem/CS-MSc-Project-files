"""
Utility functions for NMReDATA converter
"""

import logging
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pandas as pd
import numpy as np
import psutil


def setup_logging():
    """Set up logging configuration"""
    # Check if logging is already configured
    if logging.getLogger().handlers:
        return logging.getLogger(__name__)
    
    # Create handlers with UTF-8 encoding
    file_handler = logging.FileHandler('nmredata_conversion.log', encoding='utf-8')
    
    # For console handler, we'll avoid Unicode characters on Windows
    console_handler = logging.StreamHandler(sys.stdout)
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
    # Suppress RDKit warnings unless debugging
    RDLogger.DisableLog('rdApp.*')
    
    return logging.getLogger(__name__)


def verify_installation():
    """Verify all required packages are installed correctly"""
    logger = setup_logging()
    
    required_packages = {
        'pandas': pd,
        'rdkit': Chem,
        'numpy': np,
        'psutil': psutil
    }
    
    logger.info("Verifying installation...")
    all_good = True
    
    for package_name, package in required_packages.items():
        try:
            version = getattr(package, '__version__', 'Unknown')
            logger.info(f"✓ {package_name}: {version}")
        except Exception as e:
            logger.error(f"✗ {package_name}: Not found or error - {e}")
            all_good = False
    
    # Check RDKit 3D generation with MMFF
    try:
        mol = Chem.MolFromSmiles('CCO')
        mol_h = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_h)
        
        if result == -1:  # Changed from != 0 to == -1
            logger.error("✗ RDKit 3D embedding failed")
            all_good = False
        else:
            # Test MMFF94s
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol_h):
                    props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94s')
                    if props is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
                        if ff is not None:
                            ff.Minimize()
                            logger.info("✓ RDKit 3D generation with MMFF94s: Working")
                        else:
                            logger.warning("⚠ MMFF94s force field creation failed")
                            # Don't fail on this - it's not critical
                    else:
                        logger.warning("⚠ MMFF94s properties could not be obtained")
                        # Don't fail on this - it's not critical
                else:
                    logger.warning("⚠ MMFF94s not applicable to test molecule")
                    # This is not necessarily a failure
            except Exception as e:
                logger.warning(f"⚠ MMFF test warning: {e}")
                # Don't fail the whole verification for MMFF issues
            
    except Exception as e:
        logger.error(f"✗ RDKit 3D generation: Failed - {e}")
        all_good = False
    
    return all_good  # This line was missing!