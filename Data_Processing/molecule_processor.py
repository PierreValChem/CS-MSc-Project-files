"""
Molecule processing module for 3D coordinate generation and atom mapping
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import traceback
from collections import defaultdict

# Import setup_logging first before using it
try:
    from Data_Processing.utils import setup_logging
    logger = setup_logging()
except ImportError:
    # Fallback if utils is not available yet
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class MoleculeProcessor:
    """Handles molecular structure operations including 3D generation and atom mapping"""
    
    def smiles_to_mol(self, smiles):
        """Convert SMILES string to RDKit molecule object"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Sanitize the molecule
            Chem.SanitizeMol(mol)
            return mol
            
        except Exception as e:
            logger.error(f"Error parsing SMILES: {e}")
            return None
    
    def generate_3d_coordinates(self, mol):
        """Robust 3D coordinate generation using MMFF94s force field"""
        try:
            # Add explicit hydrogens to the molecule
            mol_h = Chem.AddHs(mol)
            logger.debug(f"Added hydrogens: {mol_h.GetNumAtoms()} total atoms ({mol.GetNumAtoms()} heavy atoms)")
            
            # Set up embedding parameters for robust generation
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.maxAttempts = 100  # Increased attempts for difficult molecules
            params.numThreads = 0  # Use all available threads
            params.useRandomCoords = False  # Start with distance geometry
            params.enforceChirality = True  # Maintain stereochemistry
            params.useBasicKnowledge = True  # Use basic chemical knowledge
            params.ETversion = 2  # Use version 2 of ETKDG
            
            # Generate initial conformer
            confId = AllChem.EmbedMolecule(mol_h, params)
            
            if confId == -1:
                # If standard embedding fails, try with relaxed parameters
                logger.warning("Standard embedding failed, trying with relaxed parameters...")
                params.useRandomCoords = True
                params.enforceChirality = False
                params.maxAttempts = 200
                
                confId = AllChem.EmbedMolecule(mol_h, params)
                
                if confId == -1:
                    logger.error("Could not generate 3D coordinates even with relaxed parameters")
                    return None
            
            # Always use MMFF94s for optimization (the 's' variant is recommended for accuracy)
            try:
                # Check if MMFF is applicable to this molecule
                if AllChem.MMFFHasAllMoleculeParams(mol_h):
                    # Set up MMFF94s force field
                    props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94s')
                    if props is None:
                        raise ValueError("Could not get MMFF properties")
                    
                    ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
                    if ff is None:
                        raise ValueError("Could not create MMFF force field")
                    
                    # Initial optimization
                    converged = ff.Minimize(maxIts=2000, energyTol=1e-6, forceTol=1e-4)
                    
                    if converged == 0:
                        logger.debug("MMFF94s optimization converged successfully")
                    elif converged == 1:
                        logger.debug("MMFF94s optimization converged with force tolerance")
                    else:
                        logger.warning("MMFF94s optimization did not fully converge, but proceeding")
                    
                    # Additional optimization passes for better geometry
                    for i in range(2):
                        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
                        if ff:
                            ff.Minimize(maxIts=1000)
                    
                    final_energy = ff.CalcEnergy()
                    logger.debug(f"Final MMFF94s energy: {final_energy:.2f} kcal/mol")
                    
                else:
                    # Fallback to UFF if MMFF is not applicable
                    logger.warning("MMFF94s not applicable, using UFF force field")
                    converged = AllChem.UFFOptimizeMolecule(mol_h, maxIters=2000)
                    
                    if converged == 0:
                        logger.debug("UFF optimization converged")
                    else:
                        logger.warning("UFF optimization did not fully converge")
                    
            except Exception as e:
                logger.error(f"Force field optimization failed: {e}")
                # Try basic UFF optimization as last resort
                try:
                    AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
                    logger.warning("Used basic UFF optimization as fallback")
                except:
                    logger.error("All optimization strategies failed")
                    # Return the unoptimized structure rather than None
                    pass
            
            # Validate the generated structure
            if mol_h.GetNumConformers() == 0:
                logger.error("No conformer present after 3D generation")
                return None
            
            # Check for reasonable bond lengths
            conf = mol_h.GetConformer()
            unreasonable_bonds = 0
            for bond in mol_h.GetBonds():
                idx1 = bond.GetBeginAtomIdx()
                idx2 = bond.GetEndAtomIdx()
                dist = AllChem.GetBondLength(conf, idx1, idx2)
                
                # Check if bond length is reasonable (0.5 to 3.0 Angstroms)
                if dist < 0.5 or dist > 3.0:
                    unreasonable_bonds += 1
                    logger.debug(f"Unusual bond length: {dist:.2f} Å between atoms {idx1} and {idx2}")
            
            if unreasonable_bonds > 0:
                logger.warning(f"Found {unreasonable_bonds} bonds with unusual lengths")
            
            logger.debug(f"Successfully generated 3D coordinates for {mol_h.GetNumAtoms()} atoms")
            return mol_h
                
        except Exception as e:
            logger.error(f"Critical error in 3D coordinate generation: {e}")
            logger.error(traceback.format_exc())
            return None
    
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
            
            # Validate MOL block
            if not mol_block or len(mol_block.strip()) < 10:
                logger.warning("Generated MOL block appears to be empty or invalid")
                return ""
            
            return mol_block
        except Exception as e:
            logger.error(f"Error creating MOL block: {e}")
            return ""
    
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
            ("direct", self._try_direct_mapping),
            ("chemical_shift", self._try_chemical_shift_mapping),
            ("heuristic", self._try_heuristic_mapping)
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
    
    def _try_direct_mapping(self, peak_atoms, struct_atoms):
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
    
    def _try_heuristic_mapping(self, peak_atoms, struct_atoms):
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
    
    def _try_chemical_shift_mapping(self, peak_atoms, struct_atoms, mol_3d):
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