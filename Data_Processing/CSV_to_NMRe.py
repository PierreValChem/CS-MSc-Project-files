import pandas as pd
import os
import glob
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import Descriptors
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from collections import defaultdict
import traceback
import sys
import psutil
import signal
import time

# Set up logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('nmredata_conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress RDKit warnings unless debugging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class CSVToNMReDATA:
    def __init__(self, csv_file, txt_directory, output_directory, max_workers=4):
        """
        Initialize the converter with multithreading support
        
        Args:
            csv_file: Path to CSV with columns Natural_Products_Name, NP_MRD_ID, SMILES
            txt_directory: Directory containing txt files named with NP_MRD_ID
            output_directory: Directory to save NMReDATA files
            max_workers: Maximum number of threads for parallel processing
        """
        self.csv_file = csv_file
        self.txt_directory = txt_directory
        self.output_directory = output_directory
        self.max_workers = max_workers
        
        # Validate inputs
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        if not os.path.exists(txt_directory):
            raise FileNotFoundError(f"TXT directory not found: {txt_directory}")
        
        # Thread-safe counters and metrics
        self.lock = Lock()
        self.successful = 0
        self.failed = 0
        self.start_time = None
        
        # Detailed accuracy metrics
        self.metrics = {
            'total_compounds': 0,
            'txt_files_found': 0,
            'txt_files_missing': 0,
            'valid_smiles': 0,
            'invalid_smiles': 0,
            '3d_generation_success': 0,
            '3d_generation_failed': 0,
            'peaks_parsed': 0,
            'empty_peaklists': 0,
            'atom_mapping_success': 0,
            'atom_mapping_failed': 0,
            'consolidation_warnings': 0,
            'nmredata_created': 0,
            'processing_errors': [],
            'memory_issues': 0,
            'timeout_issues': 0
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Check system resources
        self._check_system_resources()
        
        # Load and validate CSV data
        try:
            self.df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_cols = ['Natural_Products_Name', 'NP_MRD_ID', 'SMILES']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Remove duplicates
            original_len = len(self.df)
            self.df = self.df.drop_duplicates(subset=['NP_MRD_ID'])
            if len(self.df) < original_len:
                logger.warning(f"Removed {original_len - len(self.df)} duplicate entries")
            
            # Remove rows with empty SMILES
            self.df = self.df.dropna(subset=['SMILES'])
            self.df = self.df[self.df['SMILES'].str.strip() != '']
            
            self.metrics['total_compounds'] = len(self.df)
            logger.info(f"Loaded {len(self.df)} compounds from CSV")
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
        
        # Pre-cache all txt files for faster lookup
        self.txt_files_cache = self._build_txt_file_cache()
        logger.info(f"Cached {len(self.txt_files_cache)} txt files")
        
        # Set up graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _check_system_resources(self):
        """Check if system has sufficient resources"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 2:
            logger.warning(f"Low memory warning: Only {available_gb:.1f}GB available")
            logger.warning("Consider reducing max_workers or processing in smaller batches")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            logger.warning(f"High CPU usage detected: {cpu_percent}%")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown requested. Finishing current tasks...")
        self.shutdown_requested = True
    
    def _build_txt_file_cache(self):
        """Build a cache of txt files for faster lookup"""
        cache = {}
        all_txt_files = glob.glob(os.path.join(self.txt_directory, "*.txt"))
        
        for file_path in all_txt_files:
            filename = os.path.basename(file_path).lower()
            # Extract all possible ID patterns from filename
            # Store multiple possible keys for flexible matching
            cache[filename] = file_path
            
            # Also store without extension
            filename_no_ext = os.path.splitext(filename)[0]
            cache[filename_no_ext] = file_path
            
        return cache
    
    def find_txt_file(self, np_mrd_id):
        """Find the txt file containing the NP_MRD_ID using cache"""
        np_id_lower = np_mrd_id.lower()
        
        # Try different matching strategies
        for key in self.txt_files_cache:
            if np_id_lower in key:
                return self.txt_files_cache[key]
        
        logger.warning(f"No txt file found for ID {np_mrd_id}")
        return None
    
    def parse_peaklist(self, txt_file):
        """Parse the peaklist from txt file with improved error handling"""
        try:
            # Check file size to avoid memory issues
            file_size = os.path.getsize(txt_file)
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"Large peaklist file ({file_size/1024/1024:.1f}MB): {txt_file}")
            
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"Empty file: {txt_file}")
                return []
            
            peaks = []
            lines = content.split('\n')
            
            # Limit number of lines to prevent memory issues
            if len(lines) > 10000:
                logger.warning(f"Very large peaklist ({len(lines)} lines), processing first 10000")
                lines = lines[:10000]
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle different delimiters
                    if '\t' in line:
                        parts = line.split('\t')
                    else:
                        parts = line.split(',')
                    
                    if len(parts) >= 4:
                        try:
                            element = parts[0].strip().upper()
                            
                            # Validate element
                            if element not in ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'CL', 'BR', 'I']:
                                logger.debug(f"Skipping unknown element '{element}' in line {line_num}")
                                continue
                            
                            atom_number = int(parts[1].strip())
                            
                            # Validate atom number
                            if atom_number < 1 or atom_number > 9999:
                                logger.debug(f"Invalid atom number {atom_number} in line {line_num}")
                                continue
                            
                            chemical_shift = parts[2].strip()
                            if not chemical_shift:
                                continue
                            
                            shift = float(chemical_shift)
                            
                            # Validate chemical shift ranges
                            if element == 'H' and (shift < -5 or shift > 20):
                                logger.warning(f"Unusual H shift {shift} ppm in line {line_num}")
                            elif element == 'C' and (shift < -50 or shift > 250):
                                logger.warning(f"Unusual C shift {shift} ppm in line {line_num}")
                            
                            multiplicity = parts[3].strip() if parts[3].strip() else 's'
                            
                            # Validate multiplicity
                            valid_multiplicities = ['s', 'd', 't', 'q', 'p', 'h', 'm', 'br', 'dd', 'dt', 'td', 'dq', 'qd']
                            if multiplicity.lower() not in valid_multiplicities:
                                logger.debug(f"Unknown multiplicity '{multiplicity}', using 'm'")
                                multiplicity = 'm'
                            
                            coupling_constants = parts[4].strip() if len(parts) > 4 else ""
                            
                            # Parse coupling constants with better error handling
                            coupling = []
                            if coupling_constants:
                                coupling_str = coupling_constants.strip('"\'')
                                if coupling_str:
                                    try:
                                        coupling_parts = coupling_str.split(',')
                                        for cp in coupling_parts:
                                            cp_clean = cp.strip()
                                            if cp_clean:
                                                j_val = float(cp_clean)
                                                if 0 <= j_val <= 50:  # Reasonable J-coupling range
                                                    coupling.append(j_val)
                                                else:
                                                    logger.debug(f"Unusual J-coupling {j_val} Hz")
                                    except ValueError:
                                        logger.debug(f"Could not parse coupling constants: {coupling_str}")
                            
                            peaks.append({
                                'element': element,
                                'atom_number': atom_number,
                                'shift': shift,
                                'multiplicity': multiplicity,
                                'coupling': coupling
                            })
                            
                        except ValueError as e:
                            logger.debug(f"Error parsing line {line_num} in {txt_file}: {e}")
                            continue
                        except Exception as e:
                            logger.debug(f"Unexpected error parsing line {line_num}: {e}")
                            continue
            
            logger.debug(f"Parsed {len(peaks)} peaks from {txt_file}")
            return peaks
            
        except MemoryError:
            logger.error(f"Memory error parsing {txt_file}")
            with self.lock:
                self.metrics['memory_issues'] += 1
            return []
        except Exception as e:
            logger.error(f"Error parsing {txt_file}: {e}")
            return []
    
    def validate_peak_consolidation(self, original_peaks, consolidated_peaks):
        """Validate that peak consolidation is correct and complete"""
        # Check that no peaks were lost
        original_atoms = set()
        for peak in original_peaks:
            original_atoms.add(peak['atom_number'])
        
        consolidated_atoms = set()
        for peak in consolidated_peaks:
            consolidated_atoms.update(peak['atom_numbers'])
        
        if original_atoms != consolidated_atoms:
            missing = original_atoms - consolidated_atoms
            extra = consolidated_atoms - original_atoms
            if missing:
                logger.warning(f"Missing atoms after consolidation: {missing}")
            if extra:
                logger.warning(f"Extra atoms after consolidation: {extra}")
            return False
        
        # Check that equivalent peaks are properly grouped
        shift_groups = defaultdict(list)
        for peak in original_peaks:
            # Group by shift (rounded to 3 decimals), multiplicity, and coupling
            key = (
                round(peak['shift'], 3),
                peak['multiplicity'].strip().lower(),
                tuple(sorted(round(c, 1) for c in peak['coupling'])) if peak['coupling'] else ()
            )
            shift_groups[key].append(peak['atom_number'])
        
        # Verify consolidation matches expected grouping
        for peak in consolidated_peaks:
            key = (
                round(peak['shift'], 3),
                peak['multiplicity'].strip().lower(),
                tuple(sorted(round(c, 1) for c in peak['coupling'])) if peak['coupling'] else ()
            )
            expected_atoms = set(shift_groups.get(key, []))
            actual_atoms = set(peak['atom_numbers'])
            
            if expected_atoms != actual_atoms:
                logger.warning(f"Consolidation mismatch for peak at {peak['shift']} ppm")
                logger.warning(f"Expected atoms: {expected_atoms}, got: {actual_atoms}")
                return False
        
        return True
    
    def consolidate_equivalent_peaks(self, peaks):
        """Enhanced consolidation with validation"""
        if not peaks:
            return []
        
        # Group peaks by very precise criteria
        consolidation_map = defaultdict(list)
        
        for peak in peaks:
            # Use higher precision for grouping (3 decimal places)
            rounded_shift = round(peak['shift'], 3)
            
            # Create a unique key including coupling pattern
            coupling_key = tuple(sorted(round(c, 1) for c in peak['coupling'])) if peak['coupling'] else ()
            
            # Include multiplicity in the key
            key = (rounded_shift, peak['multiplicity'].strip().lower(), coupling_key)
            
            consolidation_map[key].append(peak)
        
        # Build consolidated peaks
        consolidated = []
        for key, peak_group in consolidation_map.items():
            shift, mult, coupling_tuple = key
            
            # Get all atom numbers
            atom_numbers = [p['atom_number'] for p in peak_group]
            
            # Use the original shift from the first peak (not rounded)
            original_shift = peak_group[0]['shift']
            
            consolidated_peak = {
                'element': peak_group[0]['element'],
                'shift': original_shift,
                'multiplicity': peak_group[0]['multiplicity'],
                'coupling': peak_group[0]['coupling'],
                'atom_numbers': sorted(atom_numbers),
                'count': len(atom_numbers)
            }
            
            consolidated.append(consolidated_peak)
        
        # Sort by chemical shift
        consolidated.sort(key=lambda x: x['shift'])
        
        # Validate consolidation
        if not self.validate_peak_consolidation(peaks, consolidated):
            logger.warning("Peak consolidation validation failed - review the results")
            with self.lock:
                self.metrics['consolidation_warnings'] += 1
        
        # Log consolidation for debugging
        if len(peaks) != len(consolidated):
            logger.debug(f"Consolidated {len(peaks)} peaks to {len(consolidated)} unique peaks")
            # Log details of consolidation
            for c_peak in consolidated:
                if c_peak['count'] > 1:
                    logger.debug(f"  {c_peak['shift']:.3f} ppm ({c_peak['multiplicity']}): "
                               f"atoms {c_peak['atom_numbers']} ({c_peak['count']} equivalent)")
        
        return consolidated
    
    def generate_3d_coordinates(self, mol):
        """Enhanced 3D coordinate generation with multiple fallback strategies"""
        try:
            # Add explicit hydrogens to the molecule
            mol_h = Chem.AddHs(mol)
            logger.debug(f"Added hydrogens: {mol_h.GetNumAtoms()} total atoms ({mol.GetNumAtoms()} heavy atoms)")
            
            # Strategy 1: Standard ETKDG embedding
            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                params.maxAttempts = 50  # Increase attempts
                params.numThreads = 0  # Use all available threads
                
                confId = AllChem.EmbedMolecule(mol_h, params)
                
                if confId != -1:
                    # Optimize with MMFF
                    try:
                        # Check if MMFF is applicable
                        if AllChem.MMFFHasAllMoleculeParams(mol_h):
                            props = AllChem.MMFFGetMoleculeProperties(mol_h)
                            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
                            if ff:
                                ff.Minimize(maxIts=200)
                            else:
                                AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
                        else:
                            logger.debug("MMFF not applicable, using UFF")
                            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
                    except Exception as e:
                        logger.debug(f"Force field optimization warning: {e}")
                    
                    logger.debug("Successfully generated 3D coordinates with ETKDG")
                    return mol_h
            except Exception as e:
                logger.debug(f"ETKDG embedding failed: {e}")
            
            # Strategy 2: Try with different random seeds
            logger.debug("Trying alternative random seeds...")
            for seed in [123, 456, 789, 2023, 2024]:
                try:
                    params = AllChem.ETKDGv3()
                    params.randomSeed = seed
                    params.maxAttempts = 30
                    
                    confId = AllChem.EmbedMolecule(mol_h, params)
                    if confId != -1:
                        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
                        logger.debug(f"Success with random seed {seed}")
                        return mol_h
                except:
                    continue
            
            # Strategy 3: Use random coordinates as starting point
            logger.warning("Trying random coordinate embedding...")
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.randomSeed = 42
            params.maxAttempts = 50
            
            confId = AllChem.EmbedMolecule(mol_h, params)
            if confId != -1:
                # More aggressive optimization for random coords
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
                logger.debug("Success with random coordinates")
                return mol_h
            
            # Strategy 4: Try without hydrogens first, then add them
            logger.warning("Trying embedding without hydrogens...")
            mol_no_h = Chem.RemoveHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            
            confId = AllChem.EmbedMolecule(mol_no_h, params)
            if confId != -1:
                # Add hydrogens to the 3D structure
                mol_with_h = Chem.AddHs(mol_no_h, addCoords=True)
                AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=200)
                logger.debug("Success with hydrogen-free embedding")
                return mol_with_h
            
            # Strategy 5: Use 2D coordinates as starting point
            logger.warning("Trying 2D to 3D conversion...")
            AllChem.Compute2DCoords(mol_h)
            params = AllChem.ETKDGv3()
            params.useRandomCoords = False
            params.use2DConstraints = True
            
            confId = AllChem.EmbedMolecule(mol_h, params)
            if confId != -1:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=300)
                logger.debug("Success with 2D to 3D conversion")
                return mol_h
            
            # Strategy 6: Fragment-based approach for large molecules
            if mol.GetNumAtoms() > 50:
                logger.warning("Trying fragment-based embedding for large molecule...")
                # Try to break into smaller fragments
                try:
                    # Use a more relaxed embedding
                    params = AllChem.ETKDGv3()
                    params.pruneRmsThresh = 0.5  # More permissive
                    params.enforceChirality = False  # Relax chirality constraints
                    
                    confId = AllChem.EmbedMolecule(mol_h, params)
                    if confId != -1:
                        AllChem.UFFOptimizeMolecule(mol_h, maxIters=100)
                        logger.debug("Success with relaxed parameters")
                        return mol_h
                except:
                    pass
            
            logger.error("All 3D generation strategies failed")
            return None
                
        except Exception as e:
            logger.error(f"Critical error in 3D coordinate generation: {e}")
            return None
    
    def preprocess_smiles(self, smiles):
        """Preprocess SMILES to improve 3D generation success"""
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Sanitize the molecule
            try:
                Chem.SanitizeMol(mol)
            except:
                # Try to fix common issues
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol is None:
                    return None
                
                # Manually sanitize with error handling
                try:
                    mol.UpdatePropertyCache(strict=False)
                    Chem.SetHybridization(mol)
                    Chem.SetAromaticity(mol)
                except:
                    pass
            
            # Remove problematic features that might cause embedding failures
            # 1. Handle charged atoms
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() != 0:
                    # Try to neutralize if possible
                    if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1:
                        atom.SetFormalCharge(0)
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            
            # 2. Kekulize aromatic systems
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
                Chem.SetAromaticity(mol)
            except:
                pass
            
            # 3. Clean up the molecule
            try:
                mol = Chem.RemoveHs(mol)
                Chem.SanitizeMol(mol)
            except:
                pass
            
            return mol
            
        except Exception as e:
            logger.error(f"Error preprocessing SMILES: {e}")
            return None
    
    def try_chemical_shift_mapping(self, peak_atoms, struct_atoms, mol_3d):
        """Map atoms based on chemical shift order and element type"""
        try:
            mapping = {}
            
            # Separate peaks and structure atoms by element
            element_peaks = defaultdict(list)
            for peak_num, peak_data in peak_atoms.items():
                element_peaks[peak_data['element']].append((peak_num, peak_data))
            
            element_atoms = defaultdict(list)
            for idx, atom_data in enumerate(struct_atoms):
                element_atoms[atom_data['element']].append((idx, atom_data))
            
            # For each element type, map based on chemical shift order
            for element in element_peaks:
                if element not in element_atoms:
                    logger.warning(f"Element {element} in peaks but not in structure")
                    continue
                
                # Get peaks and atoms for this element
                peaks = element_peaks[element]
                atoms = element_atoms[element]
                
                if len(peaks) > len(atoms):
                    logger.warning(f"More {element} peaks ({len(peaks)}) than atoms ({len(atoms)})")
                    continue
                
                # Sort peaks by chemical shift (descending for most elements)
                if element in ['H', 'C']:
                    # Higher shifts typically correspond to more deshielded atoms
                    sorted_peaks = sorted(peaks, key=lambda x: x[1]['shift'], reverse=True)
                else:
                    sorted_peaks = sorted(peaks, key=lambda x: x[1]['shift'])
                
                # For carbons and hydrogens, try to use connectivity information
                if element == 'C' and mol_3d:
                    # Map carbons considering their environment
                    c_mapping = self._map_carbons_by_shift_order(sorted_peaks, atoms, mol_3d)
                    mapping.update(c_mapping)
                elif element == 'H' and mol_3d:
                    # Map hydrogens considering their connected atoms
                    h_mapping = self._map_hydrogens_by_shift_order(sorted_peaks, atoms, mol_3d)
                    mapping.update(h_mapping)
                else:
                    # Simple sequential mapping for other elements
                    for i, (peak_num, peak_data) in enumerate(sorted_peaks):
                        if i < len(atoms):
                            mapping[peak_num] = atoms[i][0]
            
            # Check if we have a reasonable mapping
            if len(mapping) >= len(peak_atoms) * 0.7:  # Accept if 70% mapped
                return mapping
            else:
                return None
            
        except Exception as e:
            logger.error(f"Chemical shift mapping failed: {e}")
            return None
    
    def _map_carbons_by_shift_order(self, sorted_peaks, c_atoms, mol_3d):
        """Map carbons based on shift order and basic environment"""
        mapping = {}
        
        # Categorize carbon atoms by basic environment
        aromatic_carbons = []
        aliphatic_carbons = []
        
        for atom_idx, atom_data in c_atoms:
            atom = mol_3d.GetAtomWithIdx(atom_idx)
            if atom.GetIsAromatic():
                aromatic_carbons.append((atom_idx, atom_data))
            else:
                aliphatic_carbons.append((atom_idx, atom_data))
        
        # Separate peaks by typical shift ranges
        aromatic_peaks = []  # typically > 100 ppm
        aliphatic_peaks = []  # typically < 100 ppm
        
        for peak_num, peak_data in sorted_peaks:
            if peak_data['shift'] > 100:
                aromatic_peaks.append((peak_num, peak_data))
            else:
                aliphatic_peaks.append((peak_num, peak_data))
        
        # Map aromatic carbons to aromatic region peaks
        for i, (peak_num, peak_data) in enumerate(aromatic_peaks):
            if i < len(aromatic_carbons):
                mapping[peak_num] = aromatic_carbons[i][0]
        
        # Map aliphatic carbons to aliphatic region peaks
        for i, (peak_num, peak_data) in enumerate(aliphatic_peaks):
            if i < len(aliphatic_carbons):
                mapping[peak_num] = aliphatic_carbons[i][0]
        
        # Handle any remaining unmapped peaks/atoms
        unmapped_peaks = [p for p in sorted_peaks if p[0] not in mapping]
        unmapped_atoms = [a for a in c_atoms if a[0] not in mapping.values()]
        
        for i, (peak_num, peak_data) in enumerate(unmapped_peaks):
            if i < len(unmapped_atoms):
                mapping[peak_num] = unmapped_atoms[i][0]
        
        return mapping
    
    def _map_hydrogens_by_shift_order(self, sorted_peaks, h_atoms, mol_3d):
        """Map hydrogens based on shift order"""
        mapping = {}
        
        # Categorize hydrogen atoms by their connected atom
        h_by_environment = defaultdict(list)
        
        for atom_idx, atom_data in h_atoms:
            atom = mol_3d.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            
            if neighbors:
                connected = neighbors[0]
                env_key = (connected.GetSymbol(), connected.GetIsAromatic())
                h_by_environment[env_key].append((atom_idx, atom_data))
            else:
                h_by_environment[('Unknown', False)].append((atom_idx, atom_data))
        
        # Separate peaks by typical shift ranges
        aromatic_h = []  # typically > 6 ppm
        heteroatom_h = []  # typically 2-6 ppm
        aliphatic_h = []  # typically < 2 ppm
        
        for peak_num, peak_data in sorted_peaks:
            if peak_data['shift'] > 6:
                aromatic_h.append((peak_num, peak_data))
            elif peak_data['shift'] > 2:
                heteroatom_h.append((peak_num, peak_data))
            else:
                aliphatic_h.append((peak_num, peak_data))
        
        # Map hydrogens by category
        # Aromatic H
        aromatic_h_atoms = h_by_environment.get(('C', True), [])
        for i, (peak_num, peak_data) in enumerate(aromatic_h):
            if i < len(aromatic_h_atoms):
                mapping[peak_num] = aromatic_h_atoms[i][0]
        
        # Heteroatom-connected H
        hetero_h_atoms = []
        for key in [('O', False), ('N', False), ('S', False)]:
            hetero_h_atoms.extend(h_by_environment.get(key, []))
        
        for i, (peak_num, peak_data) in enumerate(heteroatom_h):
            if i < len(hetero_h_atoms):
                mapping[peak_num] = hetero_h_atoms[i][0]
        
        # Aliphatic H
        aliphatic_h_atoms = h_by_environment.get(('C', False), [])
        for i, (peak_num, peak_data) in enumerate(aliphatic_h):
            if i < len(aliphatic_h_atoms):
                mapping[peak_num] = aliphatic_h_atoms[i][0]
        
        # Map any remaining unmapped hydrogens
        unmapped_peaks = [p for p in sorted_peaks if p[0] not in mapping]
        unmapped_atoms = [a for a in h_atoms if a[0] not in mapping.values()]
        
        for i, (peak_num, peak_data) in enumerate(unmapped_peaks):
            if i < len(unmapped_atoms):
                mapping[peak_num] = unmapped_atoms[i][0]
        
        return mapping
    
    def create_atom_mapping(self, peaks, mol_3d):
        """Create mapping between peaklist atom numbers and 3D structure indices"""
        if mol_3d is None:
            return {}
        
        # Get atoms from peaklist
        peak_atoms = {}
        for peak in peaks:
            peak_atoms[peak['atom_number']] = {
                'element': peak['element'],
                'shift': peak['shift']
            }
        
        # Get atoms from 3D structure
        struct_atoms = []
        for i, atom in enumerate(mol_3d.GetAtoms()):
            struct_atoms.append({
                'idx': i,
                'element': atom.GetSymbol(),
                'formal_charge': atom.GetFormalCharge(),
                'neighbors': [n.GetSymbol() for n in atom.GetNeighbors()]
            })
        
        logger.debug(f"Mapping {len(peak_atoms)} peak atoms to {len(struct_atoms)} structure atoms")
        
        # Try mapping strategies in order of preference
        strategies = [
            ("direct", self.try_direct_mapping),
            ("chemical_shift", self.try_chemical_shift_mapping),
            ("heuristic", self.try_heuristic_mapping)
        ]
        
        for strategy_name, strategy_func in strategies:
            if strategy_name == "chemical_shift":
                mapping = strategy_func(peak_atoms, struct_atoms, mol_3d)
            else:
                mapping = strategy_func(peak_atoms, struct_atoms)
            
            if mapping:
                logger.debug(f"Successfully used {strategy_name} mapping strategy")
                return mapping
        
        logger.warning("Could not establish reliable atom mapping")
        return {}
    
    def try_direct_mapping(self, peak_atoms, struct_atoms):
        """Try direct 1:1 mapping assuming peaklist numbers correspond to structure indices"""
        mapping = {}
        
        # Try 0-indexed mapping
        for peak_num, peak_data in peak_atoms.items():
            struct_idx = peak_num - 1
            if 0 <= struct_idx < len(struct_atoms):
                if struct_atoms[struct_idx]['element'] == peak_data['element']:
                    mapping[peak_num] = struct_idx
                else:
                    return None
            else:
                return None
        
        return mapping if len(mapping) == len(peak_atoms) else None
    
    def try_heuristic_mapping(self, peak_atoms, struct_atoms):
        """Use chemical heuristics to map atoms"""
        mapping = {}
        
        # Group by element type
        for element in ['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']:
            peak_nums = [k for k, v in peak_atoms.items() if v['element'] == element]
            struct_indices = [i for i, atom in enumerate(struct_atoms) if atom['element'] == element]
            
            # Sort both by some criteria (e.g., atom number for peaks)
            peak_nums.sort()
            
            # Simple sequential mapping within each element type
            for i, peak_num in enumerate(peak_nums):
                if i < len(struct_indices):
                    mapping[peak_num] = struct_indices[i]
        
        return mapping if len(mapping) >= len(peak_atoms) * 0.8 else None
    
    def create_mol_block(self, mol_3d):
        """Create MOL block with 3D coordinates"""
        if mol_3d is None:
            return ""
        
        try:
            # Ensure we have a conformer
            if mol_3d.GetNumConformers() == 0:
                logger.warning("No conformer found, cannot create MOL block")
                return ""
            
            # Get the MOL block
            mol_block = Chem.MolToMolBlock(mol_3d, confId=0)
            
            return mol_block
        except Exception as e:
            logger.error(f"Error creating MOL block: {e}")
            return ""
    
    def create_nmredata_file(self, row, peaks, mol_3d):
        """Create complete NMReDATA file content"""
        mol_block = self.create_mol_block(mol_3d)
        
        if not mol_block:
            logger.warning(f"Empty MOL block for {row['NP_MRD_ID']}")
        
        # Separate and consolidate peaks
        h_peaks = [p for p in peaks if p['element'] == 'H']
        c_peaks = [p for p in peaks if p['element'] == 'C']
        
        h_peaks_consolidated = self.consolidate_equivalent_peaks(h_peaks)
        c_peaks_consolidated = self.consolidate_equivalent_peaks(c_peaks)
        
        # Build NMReDATA content
        nmredata_content = f"{mol_block}\n"
        
        # Add version and level
        nmredata_content += f""">  <NMREDATA_VERSION>
1.1

>  <NMREDATA_LEVEL>
0

"""
        
        # Add 1H NMR data
        if h_peaks_consolidated:
            nmredata_content += f">  <NMREDATA_1D_1H>\n"
            for peak in h_peaks_consolidated:
                line = f"{peak['shift']}, {peak['multiplicity']}"
                if peak['coupling']:
                    coupling_str = ', '.join([f"J={c}" for c in peak['coupling']])
                    line += f", {coupling_str}"
                
                # Format atom numbers
                if peak['count'] > 1:
                    atom_numbers = sorted(peak['atom_numbers'])
                    is_sequential = all(atom_numbers[i] == atom_numbers[i-1] + 1 
                                      for i in range(1, len(atom_numbers)))
                    
                    if is_sequential and len(atom_numbers) > 2:
                        atom_range = f"{min(atom_numbers)}-{max(atom_numbers)}"
                    else:
                        atom_range = ','.join(map(str, atom_numbers))
                    
                    line += f", {atom_range}, {peak['count']}"
                else:
                    line += f", {peak['atom_numbers'][0]}, 1"
                
                nmredata_content += f"{line}\n"
            nmredata_content += "\n"
        
        # Add 13C NMR data
        if c_peaks_consolidated:
            nmredata_content += f">  <NMREDATA_1D_13C>\n"
            for peak in c_peaks_consolidated:
                line = f"{peak['shift']}, {peak['multiplicity']}"
                if peak['coupling']:
                    coupling_str = ', '.join([f"J={c}" for c in peak['coupling']])
                    line += f", {coupling_str}"
                
                # Format atom numbers
                if peak['count'] > 1:
                    atom_numbers = sorted(peak['atom_numbers'])
                    is_sequential = all(atom_numbers[i] == atom_numbers[i-1] + 1 
                                      for i in range(1, len(atom_numbers)))
                    
                    if is_sequential and len(atom_numbers) > 2:
                        atom_range = f"{min(atom_numbers)}-{max(atom_numbers)}"
                    else:
                        atom_range = ','.join(map(str, atom_numbers))
                    
                    line += f", {atom_range}, {peak['count']}"
                else:
                    line += f", {peak['atom_numbers'][0]}, 1"
                
                nmredata_content += f"{line}\n"
            nmredata_content += "\n"
        
        # Add metadata
        nmredata_content += f""">  <NMREDATA_SOLVENT>
Unknown

>  <NMREDATA_TEMPERATURE>
298

>  <NMREDATA_FREQUENCY>
400

>  <NMREDATA_PULSE_SEQUENCE>
Unknown

>  <NMREDATA_ORIGIN>
{row['Natural_Products_Name']}

>  <NMREDATA_ACQUISITION_DATE>
{datetime.now().strftime('%Y-%m-%d')}

>  <NP_MRD_ID>
{row['NP_MRD_ID']}

>  <SMILES>
{row['SMILES']}

$$$$
"""
        
        return nmredata_content
    
    def process_compound(self, row):
        """Process a single compound - thread-safe method"""
        compound_metrics = {
            'txt_found': False,
            'valid_smiles': False,
            '3d_generated': False,
            'peaks_found': False,
            'atom_mapped': False,
            'consolidation_ok': True
        }
        
        try:
            np_id = row['NP_MRD_ID']
            logger.info(f"Processing {row['Natural_Products_Name']} (ID: {np_id})")
            
            # Find txt file
            txt_file = self.find_txt_file(np_id)
            if not txt_file:
                logger.warning(f"No txt file found for {np_id}")
                with self.lock:
                    self.metrics['txt_files_missing'] += 1
                return False, f"No txt file found for {np_id}", compound_metrics
            
            compound_metrics['txt_found'] = True
            with self.lock:
                self.metrics['txt_files_found'] += 1
            
            # Parse peaklist
            peaks = self.parse_peaklist(txt_file)
            if not peaks:
                logger.warning(f"No peaks found for {np_id}")
                with self.lock:
                    self.metrics['empty_peaklists'] += 1
            else:
                compound_metrics['peaks_found'] = True
                with self.lock:
                    self.metrics['peaks_parsed'] += 1
            
            # Generate molecule from SMILES
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol is None:
                # Try preprocessing
                logger.debug(f"Initial SMILES parsing failed, trying preprocessing for {np_id}")
                mol = self.preprocess_smiles(row['SMILES'])
                if mol is None:
                    logger.error(f"Invalid SMILES for {np_id}: {row['SMILES']}")
                    with self.lock:
                        self.metrics['invalid_smiles'] += 1
                    return False, f"Invalid SMILES for {np_id}", compound_metrics
            
            compound_metrics['valid_smiles'] = True
            with self.lock:
                self.metrics['valid_smiles'] += 1
            
            # Generate 3D coordinates
            mol_3d = self.generate_3d_coordinates(mol)
            if mol_3d is None:
                with self.lock:
                    self.metrics['3d_generation_failed'] += 1
                compound_metrics['3d_generated'] = False
            else:
                compound_metrics['3d_generated'] = True
                with self.lock:
                    self.metrics['3d_generation_success'] += 1
            
            # Create atom mapping
            if mol_3d and peaks:
                atom_mapping = self.create_atom_mapping(peaks, mol_3d)
                if atom_mapping:
                    compound_metrics['atom_mapped'] = True
                    with self.lock:
                        self.metrics['atom_mapping_success'] += 1
                    # Renumber peaks to match structure
                    peaks = self.renumber_peaklist(peaks, atom_mapping)
                else:
                    with self.lock:
                        self.metrics['atom_mapping_failed'] += 1
            
            # Create NMReDATA content
            nmredata_content = self.create_nmredata_file(row, peaks, mol_3d)
            
            # Save to file
            safe_name = "".join(c for c in row['Natural_Products_Name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_filename = f"{np_id}_{safe_name.replace(' ', '_')}.nmredata"
            output_path = os.path.join(self.output_directory, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(nmredata_content)
            
            with self.lock:
                self.metrics['nmredata_created'] += 1
            
            logger.info(f"Successfully created {output_filename}")
            return True, output_filename, compound_metrics
            
        except Exception as e:
            error_msg = f"Error processing {row['NP_MRD_ID']}: {str(e)}"
            logger.error(error_msg)
            with self.lock:
                self.metrics['processing_errors'].append({
                    'id': row['NP_MRD_ID'],
                    'error': str(e)
                })
            return False, error_msg, compound_metrics
    
    def renumber_peaklist(self, peaks, atom_mapping):
        """Renumber peaklist atoms to match 3D structure indices"""
        if not atom_mapping:
            return peaks
        
        renumbered_peaks = []
        for peak in peaks:
            if peak['atom_number'] in atom_mapping:
                new_peak = peak.copy()
                new_peak['atom_number'] = atom_mapping[peak['atom_number']] + 1  # Convert to 1-indexed
                renumbered_peaks.append(new_peak)
            else:
                renumbered_peaks.append(peak)
        
        return renumbered_peaks
    
    def process_all_compounds(self):
        """Process all compounds using multithreading with safety checks"""
        total = len(self.df)
        logger.info(f"Starting processing of {total} compounds with {self.max_workers} workers")
        
        self.start_time = time.time()
        
        # Create checkpoint system
        checkpoint_file = os.path.join(self.output_directory, 'processing_checkpoint.txt')
        processed_ids = set()
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                processed_ids = set(line.strip() for line in f)
            logger.info(f"Resuming from checkpoint: {len(processed_ids)} already processed")
        
        # Filter out already processed compounds
        df_to_process = self.df[~self.df['NP_MRD_ID'].isin(processed_ids)]
        logger.info(f"Processing {len(df_to_process)} remaining compounds")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks in batches to avoid memory issues
            batch_size = 1000
            all_futures = []
            
            for batch_start in range(0, len(df_to_process), batch_size):
                if self.shutdown_requested:
                    break
                
                batch_end = min(batch_start + batch_size, len(df_to_process))
                batch_df = df_to_process.iloc[batch_start:batch_end]
                
                # Submit batch
                batch_futures = {
                    executor.submit(self.process_compound_safe, row): (idx, row) 
                    for idx, row in batch_df.iterrows()
                }
                all_futures.extend(batch_futures.items())
                
                # Process completed tasks in this batch
                for future in as_completed(batch_futures):
                    if self.shutdown_requested:
                        break
                    
                    idx, row = batch_futures[future]
                    try:
                        success, result, compound_metrics = future.result(timeout=300)  # 5 min timeout
                        
                        with self.lock:
                            if success:
                                self.successful += 1
                                # Save checkpoint
                                with open(checkpoint_file, 'a') as f:
                                    f.write(f"{row['NP_MRD_ID']}\n")
                            else:
                                self.failed += 1
                            
                            # Progress update
                            processed = self.successful + self.failed
                            if processed % 10 == 0:
                                elapsed = time.time() - self.start_time
                                rate = processed / elapsed if elapsed > 0 else 0
                                eta = (total - processed) / rate if rate > 0 else 0
                                logger.info(f"Progress: {processed}/{total} ({processed/total*100:.1f}%) "
                                          f"Rate: {rate:.2f}/s ETA: {eta/60:.1f} min")
                                
                                # Check memory usage
                                memory_percent = psutil.virtual_memory().percent
                                if memory_percent > 90:
                                    logger.warning(f"High memory usage: {memory_percent}%")
                    
                    except TimeoutError:
                        logger.error(f"Timeout processing compound {row['NP_MRD_ID']}")
                        with self.lock:
                            self.failed += 1
                            self.metrics['timeout_issues'] += 1
                    except Exception as e:
                        logger.error(f"Unexpected error processing row {idx}: {e}")
                        with self.lock:
                            self.failed += 1
                
                # Garbage collection between batches
                import gc
                gc.collect()
        
        # Clean up checkpoint file if all successful
        if self.failed == 0 and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        # Generate and display accuracy report
        self.generate_accuracy_report()
    
    def process_compound_safe(self, row):
        """Wrapper for process_compound with additional safety checks"""
        try:
            # Check if output file already exists
            np_id = row['NP_MRD_ID']
            safe_name = "".join(c for c in row['Natural_Products_Name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_filename = f"{np_id}_{safe_name.replace(' ', '_')}.nmredata"
            output_path = os.path.join(self.output_directory, output_filename)
            
            if os.path.exists(output_path):
                logger.debug(f"Skipping {np_id} - output file already exists")
                return True, f"Already processed: {output_filename}", {}
            
            return self.process_compound(row)
            
        except Exception as e:
            logger.error(f"Safety wrapper caught error: {e}")
            return False, str(e), {}
    
    def generate_accuracy_report(self):
        """Generate comprehensive accuracy report"""
        # Calculate percentages
        total = self.metrics['total_compounds']
        txt_found_pct = (self.metrics['txt_files_found'] / total * 100) if total > 0 else 0
        valid_smiles_pct = (self.metrics['valid_smiles'] / total * 100) if total > 0 else 0
        gen_3d_pct = (self.metrics['3d_generation_success'] / self.metrics['valid_smiles'] * 100) if self.metrics['valid_smiles'] > 0 else 0
        peaks_parsed_pct = (self.metrics['peaks_parsed'] / self.metrics['txt_files_found'] * 100) if self.metrics['txt_files_found'] > 0 else 0
        mapping_pct = (self.metrics['atom_mapping_success'] / self.metrics['peaks_parsed'] * 100) if self.metrics['peaks_parsed'] > 0 else 0
        overall_success_pct = (self.successful / total * 100) if total > 0 else 0
        
        # Calculate accuracy score (weighted average of key metrics)
        accuracy_score = (
            txt_found_pct * 0.2 +  # 20% weight for finding files
            valid_smiles_pct * 0.2 +  # 20% weight for valid SMILES
            gen_3d_pct * 0.2 +  # 20% weight for 3D generation
            peaks_parsed_pct * 0.1 +  # 10% weight for parsing peaks
            mapping_pct * 0.15 +  # 15% weight for atom mapping
            overall_success_pct * 0.15  # 15% weight for overall success
        ) / 100
        
        # Generate report
        report = f"""
================================================================================
                         NMReDATA CONVERSION ACCURACY REPORT
================================================================================

SUMMARY STATISTICS:
-----------------
Total Compounds Processed:      {total}
Successfully Converted:         {self.successful} ({overall_success_pct:.1f}%)
Failed:                        {self.failed} ({self.failed/total*100:.1f}%)

DETAILED METRICS:
----------------
File Matching:
  - TXT files found:           {self.metrics['txt_files_found']} ({txt_found_pct:.1f}%)
  - TXT files missing:         {self.metrics['txt_files_missing']} ({self.metrics['txt_files_missing']/total*100:.1f}%)

Structure Processing:
  - Valid SMILES:              {self.metrics['valid_smiles']} ({valid_smiles_pct:.1f}%)
  - Invalid SMILES:            {self.metrics['invalid_smiles']} ({self.metrics['invalid_smiles']/total*100:.1f}%)
  - 3D generation success:     {self.metrics['3d_generation_success']} ({gen_3d_pct:.1f}% of valid SMILES)
  - 3D generation failed:      {self.metrics['3d_generation_failed']}

NMR Data Processing:
  - Peaklists parsed:          {self.metrics['peaks_parsed']} ({peaks_parsed_pct:.1f}% of found files)
  - Empty peaklists:           {self.metrics['empty_peaklists']}
  - Atom mapping success:      {self.metrics['atom_mapping_success']} ({mapping_pct:.1f}% of parsed peaks)
  - Atom mapping failed:       {self.metrics['atom_mapping_failed']}
  - Consolidation warnings:    {self.metrics['consolidation_warnings']}

Output:
  - NMReDATA files created:    {self.metrics['nmredata_created']}

ACCURACY SCORE: {accuracy_score:.2%}
--------------
(Weighted average of all metrics)

QUALITY INDICATORS:
------------------
✓ Excellent (>95%): """ + ("✓" if accuracy_score > 0.95 else "✗") + f"""
✓ Good (85-95%):    """ + ("✓" if 0.85 <= accuracy_score <= 0.95 else "✗") + f"""
✓ Fair (70-85%):    """ + ("✓" if 0.70 <= accuracy_score < 0.85 else "✗") + f"""
✗ Poor (<70%):      """ + ("✓" if accuracy_score < 0.70 else "✗") + f"""

COMMON ISSUES:
-------------"""
        
        # Add common issues
        if self.metrics['txt_files_missing'] > 0:
            report += f"\n- Missing TXT files: {self.metrics['txt_files_missing']} compounds lack NMR data files"
        
        if self.metrics['invalid_smiles'] > 0:
            report += f"\n- Invalid SMILES: {self.metrics['invalid_smiles']} compounds have unparseable structures"
        
        if self.metrics['3d_generation_failed'] > 0:
            report += f"\n- 3D generation failures: {self.metrics['3d_generation_failed']} molecules couldn't be embedded"
        
        if self.metrics['atom_mapping_failed'] > 0:
            report += f"\n- Atom mapping issues: {self.metrics['atom_mapping_failed']} compounds have ambiguous NMR assignments"
        
        if self.metrics['consolidation_warnings'] > 0:
            report += f"\n- Peak consolidation warnings: {self.metrics['consolidation_warnings']} files may have incorrect peak grouping"
        
        # Add error samples if any
        if self.metrics['processing_errors']:
            report += f"\n\nSAMPLE ERRORS (first 5):\n"
            for error in self.metrics['processing_errors'][:5]:
                report += f"- {error['id']}: {error['error'][:100]}...\n"
        
        report += """
================================================================================
"""
        
        # Print and save report
        print(report)
        
        # Save detailed report to file
        report_path = os.path.join(self.output_directory, "accuracy_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
            # Add detailed error log
            if self.metrics['processing_errors']:
                f.write("\n\nDETAILED ERROR LOG:\n")
                f.write("="*80 + "\n")
                for error in self.metrics['processing_errors']:
                    f.write(f"\nCompound ID: {error['id']}\n")
                    f.write(f"Error: {error['error']}\n")
                    f.write("-"*40 + "\n")
        
        logger.info(f"Accuracy report saved to: {report_path}")
        
        # Return accuracy score for programmatic use
        return accuracy_score

def main():
    """Main function to run the converter with error handling"""
    try:
        # Configuration
        csv_file = "NP-ID and structure NP0100001-NP0150000.csv"
        txt_directory = "NP-MRD_nmr_peak_lists_NP0100001_NP0150000/NP-MRD_nmr_peak_lists_NP0100001_NP0150000/"
        output_directory = "CSV_to_NMRe_output_v2/"
        
        # Optimal settings for i7-12800H (14 cores: 6P + 8E) with 32GB RAM
        # Use 10-12 workers to leave some cores for system processes
        max_workers = 10
        
        # Check if running in batch mode or specific range
        if len(sys.argv) > 1:
            if sys.argv[1] == '--batch':
                # Process in smaller batches
                batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
                logger.info(f"Running in batch mode with batch size {batch_size}")
                
                df_full = pd.read_csv(csv_file)
                total_batches = (len(df_full) + batch_size - 1) // batch_size
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min((batch_num + 1) * batch_size, len(df_full))
                    
                    logger.info(f"Processing batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx})")
                    
                    # Create temporary CSV for this batch
                    batch_csv = f"batch_{batch_num + 1}.csv"
                    df_full.iloc[start_idx:end_idx].to_csv(batch_csv, index=False)
                    
                    # Process batch
                    converter = CSVToNMReDATA(batch_csv, txt_directory, output_directory, max_workers)
                    converter.process_all_compounds()
                    
                    # Clean up
                    os.remove(batch_csv)
                    
                    # Pause between batches
                    if batch_num < total_batches - 1:
                        logger.info("Pausing for 30 seconds between batches...")
                        time.sleep(30)
            else:
                # Normal processing
                converter = CSVToNMReDATA(csv_file, txt_directory, output_directory, max_workers)
                converter.process_all_compounds()
        else:
            # Normal processing
            converter = CSVToNMReDATA(csv_file, txt_directory, output_directory, max_workers)
            converter.process_all_compounds()
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def verify_installation():
    """Verify all required packages are installed correctly"""
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
        except:
            logger.error(f"✗ {package_name}: Not found or error")
            all_good = False
    
    # Check RDKit 3D generation
    try:
        mol = Chem.MolFromSmiles('CCO')
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h)
        logger.info("✓ RDKit 3D generation: Working")
    except:
        logger.error("✗ RDKit 3D generation: Failed")
        all_good = False
    
    return all_good

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
        if '$$' not in content:
            logger.warning(f"Missing $$ terminator in {file_path}")
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

def analyze_consolidation_accuracy(csv_file, txt_directory, sample_size=10):
    """Analyze the accuracy of peak consolidation on a sample"""
    converter = CSVToNMReDATA(csv_file, txt_directory, "temp/", max_workers=1)
    df = pd.read_csv(csv_file)
    
    # Take a random sample
    sample_df = df.sample(min(sample_size, len(df)))
    
    consolidation_report = []
    
    for _, row in sample_df.iterrows():
        np_id = row['NP_MRD_ID']
        txt_file = converter.find_txt_file(np_id)
        
        if txt_file:
            peaks = converter.parse_peaklist(txt_file)
            h_peaks = [p for p in peaks if p['element'] == 'H']
            c_peaks = [p for p in peaks if p['element'] == 'C']
            
            h_consolidated = converter.consolidate_equivalent_peaks(h_peaks)
            c_consolidated = converter.consolidate_equivalent_peaks(c_peaks)
            
            report = {
                'NP_ID': np_id,
                'H_original': len(h_peaks),
                'H_consolidated': len(h_consolidated),
                'C_original': len(c_peaks),
                'C_consolidated': len(c_consolidated),
                'H_reduction': len(h_peaks) - len(h_consolidated),
                'C_reduction': len(c_peaks) - len(c_consolidated)
            }
            
            consolidation_report.append(report)
            
            # Show details for significant consolidations
            if report['H_reduction'] > 0 or report['C_reduction'] > 0:
                logger.info(f"\nConsolidation for {np_id}:")
                logger.info(f"  1H: {report['H_original']} → {report['H_consolidated']} (reduced by {report['H_reduction']})")
                logger.info(f"  13C: {report['C_original']} → {report['C_consolidated']} (reduced by {report['C_reduction']})")
    
    # Summary statistics
    if consolidation_report:
        avg_h_reduction = sum(r['H_reduction'] for r in consolidation_report) / len(consolidation_report)
        avg_c_reduction = sum(r['C_reduction'] for r in consolidation_report) / len(consolidation_report)
        
        logger.info(f"\nConsolidation Summary:")
        logger.info(f"Average 1H peak reduction: {avg_h_reduction:.2f}")
        logger.info(f"Average 13C peak reduction: {avg_c_reduction:.2f}")
    
    return consolidation_report

def test_chemical_shift_mapping(smiles, peaks_data):
    """Test the chemical shift mapping on a specific compound"""
    # Create a minimal converter instance
    converter = CSVToNMReDATA("dummy.csv", "dummy/", "dummy/", max_workers=1)
    
    # Generate molecule and 3D structure
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.error("Invalid SMILES")
        return
    
    mol_3d = converter.generate_3d_coordinates(mol)
    if not mol_3d:
        logger.error("Could not generate 3D structure")
        return
    
    # Create peak atoms dictionary
    peak_atoms = {}
    for i, (element, shift) in enumerate(peaks_data, 1):
        peak_atoms[i] = {'element': element, 'shift': shift}
    
    # Create structure atoms list
    struct_atoms = []
    for i, atom in enumerate(mol_3d.GetAtoms()):
        struct_atoms.append({
            'idx': i,
            'element': atom.GetSymbol(),
            'formal_charge': atom.GetFormalCharge(),
            'neighbors': [n.GetSymbol() for n in atom.GetNeighbors()]
        })
    
    # Test the mapping
    mapping = converter.try_chemical_shift_mapping(peak_atoms, struct_atoms, mol_3d)
    
    if mapping:
        logger.info("Chemical shift mapping successful:")
        for peak_num, struct_idx in mapping.items():
            peak = peak_atoms[peak_num]
            struct = struct_atoms[struct_idx]
            logger.info(f"  Peak {peak_num} ({peak['element']}, {peak['shift']} ppm) → "
                       f"Atom {struct_idx} ({struct['element']})")
    else:
        logger.warning("Chemical shift mapping failed")
    
    return mapping

def profile_performance(csv_file, txt_directory, num_compounds=100):
    """Profile the performance with different numbers of workers"""
    import time
    
    # Test with different worker counts
    worker_counts = [1, 2, 4, 8]
    results = {}
    
    # Load only a subset of compounds for testing
    df = pd.read_csv(csv_file)
    test_df = df.head(num_compounds)
    test_csv = "test_subset.csv"
    test_df.to_csv(test_csv, index=False)
    
    for workers in worker_counts:
        logger.info(f"\nTesting with {workers} workers...")
        
        start_time = time.time()
        converter = CSVToNMReDATA(test_csv, txt_directory, f"test_output_{workers}/", workers)
        converter.process_all_compounds()
        end_time = time.time()
        
        elapsed = end_time - start_time
        results[workers] = {
            'time': elapsed,
            'successful': converter.successful,
            'failed': converter.failed,
            'compounds_per_second': converter.successful / elapsed if elapsed > 0 else 0
        }
        
        logger.info(f"Completed in {elapsed:.2f} seconds")
        logger.info(f"Rate: {results[workers]['compounds_per_second']:.2f} compounds/second")
    
    # Clean up
    os.remove(test_csv)
    
    # Summary
    logger.info("\nPerformance Summary:")
    logger.info("Workers | Time (s) | Rate (compounds/s)")
    logger.info("--------|----------|------------------")
    for workers, data in results.items():
        logger.info(f"{workers:7d} | {data['time']:8.2f} | {data['compounds_per_second']:17.2f}")
    
    return results

if __name__ == "__main__":
    # Verify installation first
    if not verify_installation():
        logger.error("Installation verification failed. Please check dependencies.")
        sys.exit(1)
    
    # Run the main conversion
    main()
    
    