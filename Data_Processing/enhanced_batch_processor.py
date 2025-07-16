#!/usr/bin/env python3
"""
Enhanced Batch NMReDATA Processor with Smart File Selection
- Handles multiple files per NP-ID
- Filters out irrelevant 2D NMR data
- Combines separate H and C NMR files
- Selects most comprehensive file when duplicates exist
"""

import os
import sys
import glob
import pandas as pd
import logging
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, inchi
from rdkit import RDLogger
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import traceback

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('nmredata_conversion.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MolecularRepresentationGenerator:
    """Generates all requested molecular representations from SMILES"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_all_representations(self, smiles, mol=None):
        """Generate all molecular representations"""
        representations = {
            'original_smiles': smiles,
            'canonical_smiles': None,
            'cxsmiles': None,
            'inchi': None,
            'inchi_auxinfo': None,
            'atom_mapping': None,
            'csrml': None,
            'cml': None
        }
        
        try:
            # Get RDKit molecule if not provided
            if mol is None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    self.logger.error(f"Failed to parse SMILES: {smiles}")
                    return representations
            
            # 1. Canonical SMILES
            try:
                representations['canonical_smiles'] = Chem.MolToSmiles(mol, canonical=True)
            except Exception as e:
                self.logger.error(f"Error generating canonical SMILES: {e}")
            
            # 2. CXSMILES
            try:
                representations['cxsmiles'] = Chem.MolToCXSmiles(mol)
            except Exception as e:
                self.logger.error(f"Error generating CXSMILES: {e}")
            
            # 3. InChI with AuxInfo
            try:
                inchi_result = inchi.MolToInchi(mol)
                representations['inchi'] = inchi_result
                
                # Get AuxInfo
                inchi_aux = inchi.MolToInchiAndAuxInfo(mol)
                if len(inchi_aux) >= 2:
                    representations['inchi_auxinfo'] = inchi_aux[1]
            except Exception as e:
                self.logger.error(f"Error generating InChI: {e}")
            
            # 4. Atom Mapping
            try:
                atom_mapped_smiles = self._generate_atom_mapped_smiles(mol)
                representations['atom_mapping'] = atom_mapped_smiles
            except Exception as e:
                self.logger.error(f"Error generating atom mapping: {e}")
            
            # 5. CSRML
            try:
                representations['csrml'] = self._generate_csrml(mol)
            except Exception as e:
                self.logger.error(f"Error generating CSRML: {e}")
            
            # 6. CML
            try:
                representations['cml'] = self._generate_cml(mol)
            except Exception as e:
                self.logger.error(f"Error generating CML: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in generate_all_representations: {e}")
        
        return representations
    
    def _generate_atom_mapped_smiles(self, mol):
        """Generate SMILES with atom mapping numbers"""
        mol_copy = Chem.Mol(mol)
        for idx, atom in enumerate(mol_copy.GetAtoms()):
            atom.SetAtomMapNum(idx + 1)
        return Chem.MolToSmiles(mol_copy, canonical=True)
    
    def _generate_csrml(self, mol):
        """Generate CSRML representation"""
        root = ET.Element('molecule')
        
        # Properties
        props = ET.SubElement(root, 'properties')
        ET.SubElement(props, 'formula').text = rdMolDescriptors.CalcMolFormula(mol)
        ET.SubElement(props, 'weight').text = f"{rdMolDescriptors.CalcExactMolWt(mol):.4f}"
        ET.SubElement(props, 'heavy_atoms').text = str(mol.GetNumHeavyAtoms())
        
        # Atoms
        atoms_elem = ET.SubElement(root, 'atoms')
        for idx, atom in enumerate(mol.GetAtoms()):
            atom_elem = ET.SubElement(atoms_elem, 'atom')
            atom_elem.set('id', str(idx))
            atom_elem.set('element', atom.GetSymbol())
            atom_elem.set('charge', str(atom.GetFormalCharge()))
            atom_elem.set('hybridization', str(atom.GetHybridization()))
            atom_elem.set('aromatic', str(atom.GetIsAromatic()).lower())
        
        # Bonds
        bonds_elem = ET.SubElement(root, 'bonds')
        for bond in mol.GetBonds():
            bond_elem = ET.SubElement(bonds_elem, 'bond')
            bond_elem.set('from', str(bond.GetBeginAtomIdx()))
            bond_elem.set('to', str(bond.GetEndAtomIdx()))
            bond_elem.set('order', str(bond.GetBondTypeAsDouble()))
            bond_elem.set('aromatic', str(bond.GetIsAromatic()).lower())
        
        return self._prettify_xml(root)
    
    def _generate_cml(self, mol):
        """Generate CML representation"""
        root = ET.Element('molecule', xmlns="http://www.xml-cml.org/schema")
        root.set('id', 'mol1')
        
        # Atom array
        atom_array = ET.SubElement(root, 'atomArray')
        for idx, atom in enumerate(mol.GetAtoms()):
            atom_elem = ET.SubElement(atom_array, 'atom')
            atom_elem.set('id', f'a{idx + 1}')
            atom_elem.set('elementType', atom.GetSymbol())
            
            if atom.GetFormalCharge() != 0:
                atom_elem.set('formalCharge', str(atom.GetFormalCharge()))
            
            if atom.GetIsAromatic():
                atom_elem.set('aromatic', 'true')
        
        # Bond array
        bond_array = ET.SubElement(root, 'bondArray')
        for bond in mol.GetBonds():
            bond_elem = ET.SubElement(bond_array, 'bond')
            bond_elem.set('id', f'b{bond.GetIdx() + 1}')
            bond_elem.set('atomRefs2', f'a{bond.GetBeginAtomIdx() + 1} a{bond.GetEndAtomIdx() + 1}')
            
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                bond_elem.set('order', 'S')
            elif bond_type == Chem.BondType.DOUBLE:
                bond_elem.set('order', 'D')
            elif bond_type == Chem.BondType.TRIPLE:
                bond_elem.set('order', 'T')
            elif bond_type == Chem.BondType.AROMATIC:
                bond_elem.set('order', 'A')
        
        # Properties
        property_list = ET.SubElement(root, 'propertyList')
        
        formula_prop = ET.SubElement(property_list, 'property')
        formula_prop.set('dictRef', 'cml:formula')
        ET.SubElement(formula_prop, 'scalar').text = rdMolDescriptors.CalcMolFormula(mol)
        
        weight_prop = ET.SubElement(property_list, 'property')
        weight_prop.set('dictRef', 'cml:molwt')
        ET.SubElement(weight_prop, 'scalar').text = f"{rdMolDescriptors.CalcExactMolWt(mol):.4f}"
        
        return self._prettify_xml(root)
    
    def _prettify_xml(self, elem):
        """Pretty print XML"""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        # Remove XML declaration
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        return '\n'.join(lines).strip()


class SimpleMoleculeProcessor:
    """Simple molecule processing for 3D generation"""
    
    def smiles_to_mol(self, smiles):
        """Convert SMILES to molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            Chem.SanitizeMol(mol)
            return mol
        except:
            return None
    
    def generate_3d_coordinates(self, mol):
        """Generate 3D coordinates"""
        try:
            mol_h = Chem.AddHs(mol)
            
            # Try standard embedding
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.maxAttempts = 100
            
            confId = AllChem.EmbedMolecule(mol_h, params)
            
            if confId == -1:
                # Try with random coords
                params.useRandomCoords = True
                confId = AllChem.EmbedMolecule(mol_h, params)
                
                if confId == -1:
                    return None
            
            # Optimize
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol_h):
                    props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94s')
                    ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
                    if ff:
                        ff.Minimize()
                else:
                    AllChem.UFFOptimizeMolecule(mol_h)
            except:
                pass
            
            return mol_h
            
        except Exception as e:
            logger.error(f"Error in 3D generation: {e}")
            return None
    
    def create_mol_block(self, mol_3d):
        """Create MOL block"""
        if mol_3d is None:
            return ""
        
        try:
            if mol_3d.GetNumConformers() == 0:
                return ""
            
            mol_block = Chem.MolToMolBlock(mol_3d, confId=0)
            return mol_block if mol_block else ""
        except:
            return ""


class EnhancedNMRParser:
    """Enhanced NMR parser that handles multiple file formats and selects best data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_2d_nmr_file(self, content):
        """Check if file contains 2D NMR data (should be ignored)"""
        lines = content.split('\n')[:10]  # Check first 10 lines
        for line in lines:
            if 'F2ppm' in line and 'F1ppm' in line:
                return True
        return False
    
    def is_valid_nmr_format(self, content):
        """Check if file has valid NMR format"""
        # Look for atom_id, symbol, chemical_shift pattern
        lines = content.split('\n')
        for line in lines[:20]:  # Check first 20 lines
            if 'atom_id' in line and 'symbol' in line and 'chemical_shift' in line:
                return True
            # Also check for the old format (element,atom_number,shift)
            if re.match(r'^[A-Z],\d+,[\d.]+', line.strip()):
                return True
        return False
    
    def parse_peaklist(self, txt_file):
        """Parse NMR peaks from txt file"""
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            if not content:
                return []
            
            # Check if it's 2D NMR (should be ignored)
            if self.is_2d_nmr_file(content):
                self.logger.debug(f"Ignoring 2D NMR file: {txt_file}")
                return []
            
            peaks = []
            lines = content.split('\n')
            
            # Detect format
            is_new_format = False
            for line in lines[:20]:
                if 'atom_id' in line and 'symbol' in line:
                    is_new_format = True
                    break
            
            if is_new_format:
                peaks = self._parse_new_format(lines)
            else:
                peaks = self._parse_old_format(lines)
            
            return peaks
            
        except Exception as e:
            self.logger.error(f"Error parsing {txt_file}: {e}")
            return []
    
    def _parse_new_format(self, lines):
        """Parse new format: atom_id symbol chemical_shift mult coupling"""
        peaks = []
        header_found = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for header
            if 'atom_id' in line and 'symbol' in line:
                header_found = True
                continue
            
            if not header_found:
                continue
            
            # Parse data line
            parts = line.split()
            if len(parts) >= 3:
                try:
                    atom_id = int(parts[0])
                    element = parts[1].upper()
                    
                    # Skip non-NMR elements
                    if element not in ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'CL', 'BR', 'I']:
                        continue
                    
                    # Only process H and C for NMR
                    if element not in ['H', 'C']:
                        continue
                    
                    shift = float(parts[2])
                    multiplicity = parts[3] if len(parts) > 3 and parts[3] != '.' else 's'
                    
                    # Parse coupling constants
                    coupling = []
                    if len(parts) > 4:
                        coupling_str = ' '.join(parts[4:])
                        # Remove quotes and parse
                        coupling_str = coupling_str.strip('"\'')
                        for val in re.findall(r'[\d.]+', coupling_str):
                            try:
                                coupling.append(float(val))
                            except:
                                pass
                    
                    peaks.append({
                        'element': element,
                        'atom_number': atom_id,
                        'shift': shift,
                        'multiplicity': multiplicity,
                        'coupling': coupling
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error parsing line: {line} - {e}")
                    continue
        
        return peaks
    
    def _parse_old_format(self, lines):
        """Parse old format: element,atom_number,shift,mult,coupling"""
        peaks = []
        
        for line in lines[:10000]:  # Limit lines
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle different delimiters
                parts = line.split('\t') if '\t' in line else line.split(',')
                
                if len(parts) >= 4:
                    try:
                        element = parts[0].strip().upper()
                        if element not in ['H', 'C']:
                            continue
                        
                        atom_number = int(parts[1].strip())
                        if atom_number < 1 or atom_number > 9999:
                            continue
                        
                        shift = float(parts[2].strip())
                        multiplicity = parts[3].strip() if parts[3].strip() else 's'
                        
                        # Parse coupling constants
                        coupling = []
                        if len(parts) > 4 and parts[4].strip():
                            coupling_str = parts[4].strip('"\'').strip()
                            if coupling_str:
                                # Remove J= and Hz
                                cleaned = re.sub(r'[Jj]\s*=\s*', '', coupling_str)
                                cleaned = re.sub(r'\s*[Hh][Zz]', '', cleaned)
                                
                                for part in re.split(r'[,;]', cleaned):
                                    try:
                                        j_val = float(part.strip())
                                        coupling.append(j_val)
                                    except:
                                        continue
                        
                        peaks.append({
                            'element': element,
                            'atom_number': atom_number,
                            'shift': shift,
                            'multiplicity': multiplicity,
                            'coupling': coupling
                        })
                        
                    except:
                        continue
        
        return peaks
    
    def find_all_files_for_id(self, np_id, txt_directory):
        """Find all txt files for a given NP-ID"""
        txt_files = []
        txt_directory = Path(txt_directory)
        np_id_lower = np_id.lower()
        
        # Search for files containing the NP-ID
        for txt_file in txt_directory.glob("*.txt"):
            if np_id_lower in txt_file.name.lower():
                txt_files.append(txt_file)
        
        return txt_files
    
    def select_best_file(self, files_with_peaks):
        """Select the most comprehensive file from multiple options"""
        if not files_with_peaks:
            return None, []
        
        # Score each file based on data quality
        best_score = -1
        best_file = None
        best_peaks = []
        
        for file_path, peaks in files_with_peaks:
            score = 0
            
            # Count H and C peaks
            h_peaks = [p for p in peaks if p['element'] == 'H']
            c_peaks = [p for p in peaks if p['element'] == 'C']
            
            # Score based on number of peaks
            score += len(h_peaks) * 2  # H peaks are valuable
            score += len(c_peaks) * 3  # C peaks are even more valuable
            
            # Bonus for having both H and C
            if h_peaks and c_peaks:
                score += 10
            
            # Bonus for peaks with coupling constants
            peaks_with_coupling = [p for p in peaks if p.get('coupling')]
            score += len(peaks_with_coupling) * 2
            
            # Bonus for peaks with multiplicity info
            peaks_with_mult = [p for p in peaks if p.get('multiplicity') and p['multiplicity'] != 's']
            score += len(peaks_with_mult)
            
            self.logger.debug(f"File {file_path.name} score: {score} (H:{len(h_peaks)}, C:{len(c_peaks)})")
            
            if score > best_score:
                best_score = score
                best_file = file_path
                best_peaks = peaks
        
        return best_file, best_peaks
    
    def merge_h_and_c_files(self, files_with_peaks):
        """Merge separate H and C NMR files"""
        h_only_files = []
        c_only_files = []
        mixed_files = []
        
        for file_path, peaks in files_with_peaks:
            h_peaks = [p for p in peaks if p['element'] == 'H']
            c_peaks = [p for p in peaks if p['element'] == 'C']
            
            if h_peaks and not c_peaks:
                h_only_files.append((file_path, peaks))
            elif c_peaks and not h_peaks:
                c_only_files.append((file_path, peaks))
            elif h_peaks and c_peaks:
                mixed_files.append((file_path, peaks))
        
        # If we have mixed files, use the best one
        if mixed_files:
            return self.select_best_file(mixed_files)
        
        # Otherwise, merge H and C files
        merged_peaks = []
        best_h_file = None
        best_c_file = None
        
        if h_only_files:
            best_h_file, h_peaks = self.select_best_file(h_only_files)
            merged_peaks.extend(h_peaks)
        
        if c_only_files:
            best_c_file, c_peaks = self.select_best_file(c_only_files)
            merged_peaks.extend(c_peaks)
        
        if merged_peaks:
            # Return the H file as primary (or C if no H)
            primary_file = best_h_file if best_h_file else best_c_file
            return primary_file, merged_peaks
        
        return None, []


class RawFormatNMReDataWriter:
    """NMReDATA writer with raw format and molecular representations"""
    
    def __init__(self):
        self.molecule_processor = SimpleMoleculeProcessor()
        self.rep_generator = MolecularRepresentationGenerator()
    
    def create_nmredata_file(self, row, peaks, mol_3d):
        """Create NMReDATA file with raw format"""
        mol_block = self.molecule_processor.create_mol_block(mol_3d)
        
        if not mol_block or len(mol_block.strip()) < 10:
            logger.warning(f"Empty MOL block for {row['NP_MRD_ID']}")
            return None
        
        # Build content
        nmredata_content = f"{mol_block}\n"
        
        # Add version
        nmredata_content += """>  <NMREDATA_VERSION>
1.1

>  <NMREDATA_LEVEL>
0

"""
        
        # Separate peaks
        h_peaks = [p for p in peaks if p['element'] == 'H']
        c_peaks = [p for p in peaks if p['element'] == 'C']
        
        # Add raw NMR data
        nmredata_content += self._add_raw_nmr_data(h_peaks, c_peaks)
        
        # Add metadata
        nmredata_content += self._create_metadata(row)
        
        # Add molecular representations
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol:
                representations = self.rep_generator.generate_all_representations(row['SMILES'], mol)
                nmredata_content = self._add_representations(nmredata_content, representations)
        except Exception as e:
            logger.warning(f"Could not add representations: {e}")
        
        nmredata_content += "$$$$\n"
        
        return nmredata_content
    
    def _add_raw_nmr_data(self, h_peaks, c_peaks):
        """Add raw NMR data"""
        content = ""
        
        # 1H NMR
        if h_peaks:
            content += ">  <NMREDATA_1D_1H>\n"
            for peak in sorted(h_peaks, key=lambda x: (x['atom_number'], -x['shift'])):
                line_parts = [
                    peak['element'],
                    str(peak['atom_number']),
                    f"{peak['shift']:.2f}",
                    peak['multiplicity']
                ]
                
                if peak['coupling']:
                    line_parts.append('"' + ','.join(f"{j:.1f}" for j in peak['coupling']) + '"')
                
                content += ",".join(line_parts) + "\n"
            content += "\n"
        
        # 13C NMR
        if c_peaks:
            content += ">  <NMREDATA_1D_13C>\n"
            for peak in sorted(c_peaks, key=lambda x: (x['atom_number'], -x['shift'])):
                line_parts = [
                    peak['element'],
                    str(peak['atom_number']),
                    f"{peak['shift']:.2f}",
                    peak['multiplicity']
                ]
                
                if peak['coupling']:
                    line_parts.append('"' + ','.join(f"{j:.1f}" for j in peak['coupling']) + '"')
                
                content += ",".join(line_parts) + "\n"
            content += "\n"
        
        return content
    
    def _create_metadata(self, row):
        """Create metadata"""
        return f""">  <NMREDATA_SOLVENT>
Unknown

>  <NMREDATA_TEMPERATURE>
298

>  <NMREDATA_FREQUENCY>
400

>  <NMREDATA_ORIGIN>
{row['Natural_Products_Name']}

>  <NP_MRD_ID>
{row['NP_MRD_ID']}

>  <SMILES>
{row['SMILES']}

"""
    
    def _add_representations(self, content, representations):
        """Add molecular representations"""
        additional = ""
        
        if representations.get('canonical_smiles'):
            additional += f">  <CANONICAL_SMILES>\n{representations['canonical_smiles']}\n\n"
        
        if representations.get('cxsmiles'):
            additional += f">  <CXSMILES>\n{representations['cxsmiles']}\n\n"
        
        if representations.get('inchi'):
            additional += f">  <INCHI>\n{representations['inchi']}\n\n"
        
        if representations.get('inchi_auxinfo'):
            additional += f">  <INCHI_AUXINFO>\n{representations['inchi_auxinfo']}\n\n"
        
        if representations.get('atom_mapping'):
            additional += f">  <ATOM_MAPPED_SMILES>\n{representations['atom_mapping']}\n\n"
        
        if representations.get('csrml'):
            additional += f">  <CSRML>\n{representations['csrml']}\n\n"
        
        if representations.get('cml'):
            additional += f">  <CML>\n{representations['cml']}\n\n"
        
        insert_pos = content.rfind('$$$$')
        if insert_pos > 0:
            return content[:insert_pos] + additional + content[insert_pos:]
        return content + additional


class StandaloneBatchProcessor:
    """Process multiple CSV/folder pairs with enhanced file selection"""
    
    def __init__(self, base_directory, output_directory, max_workers=10):
        self.base_directory = Path(base_directory)
        self.output_directory = Path(output_directory)
        self.max_workers = max_workers
        self.molecule_processor = SimpleMoleculeProcessor()
        self.nmr_parser = EnhancedNMRParser()
        self.nmredata_writer = RawFormatNMReDataWriter()
        
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    def find_csv_folder_pairs(self):
        """Find CSV/Excel files and matching folders"""
        pairs = []
        
        # Find data files
        csv_files = list(self.base_directory.glob("*.csv"))
        excel_files = list(self.base_directory.glob("*.xlsx"))
        all_files = csv_files + excel_files
        
        logger.info(f"Found {len(all_files)} data files")
        
        for data_file in all_files:
            # Extract range from filename
            match = re.search(r'NP(\d+)_NP(\d+)', data_file.stem)
            if match:
                start_num = match.group(1)
                end_num = match.group(2)
                
                # Look for folder
                folder_pattern = f"NP-MRD_nmr_peak_lists_NP{start_num}_NP{end_num}"
                
                nmr_folder = None
                for item in self.base_directory.iterdir():
                    if item.is_dir() and folder_pattern in item.name:
                        nmr_folder = item
                        break
                
                if nmr_folder:
                    pairs.append((data_file, nmr_folder))
                    logger.info(f"Matched: {data_file.name} -> {nmr_folder.name}")
                else:
                    logger.warning(f"No folder for {data_file.name}")
        
        return pairs
    
    def process_single_compound(self, row, txt_directory):
        """Process single compound with enhanced file selection"""
        try:
            np_id = row['NP_MRD_ID']
            logger.info(f"Processing {row['Natural_Products_Name']} (ID: {np_id})")
            
            # Find all txt files for this ID
            txt_files = self.nmr_parser.find_all_files_for_id(np_id, txt_directory)
            
            if not txt_files:
                return False, "No txt files found"
            
            logger.debug(f"Found {len(txt_files)} files for {np_id}")
            
            # Parse all files and filter valid ones
            files_with_peaks = []
            for txt_file in txt_files:
                peaks = self.nmr_parser.parse_peaklist(txt_file)
                if peaks:  # Only keep files with valid peaks
                    files_with_peaks.append((txt_file, peaks))
            
            if not files_with_peaks:
                logger.warning(f"No valid NMR data found for {np_id}")
                return False, "No valid NMR data"
            
            # Select best file or merge H/C files
            if len(files_with_peaks) == 1:
                selected_file, peaks = files_with_peaks[0]
            else:
                # Check if we need to merge H and C files
                selected_file, peaks = self.nmr_parser.merge_h_and_c_files(files_with_peaks)
                
                if not peaks:
                    # If merging didn't work, just select the best single file
                    selected_file, peaks = self.nmr_parser.select_best_file(files_with_peaks)
            
            if not peaks:
                return False, "No peaks after selection"
            
            logger.info(f"Selected file: {selected_file.name} with {len(peaks)} peaks")
            
            # Generate molecule
            mol = self.molecule_processor.smiles_to_mol(row['SMILES'])
            if mol is None:
                return False, "Invalid SMILES"
            
            # Generate 3D
            mol_3d = self.molecule_processor.generate_3d_coordinates(mol)
            if mol_3d is None:
                return False, "3D generation failed"
            
            # Create NMReDATA
            content = self.nmredata_writer.create_nmredata_file(row, peaks, mol_3d)
            if content is None:
                return False, "Content generation failed"
            
            # Save file
            safe_name = "".join(c for c in row['Natural_Products_Name'] if c.isalnum() or c in ' -_').strip()
            filename = f"{np_id}_{safe_name.replace(' ', '_')}.nmredata"
            output_path = self.output_directory / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log summary
            h_peaks = len([p for p in peaks if p['element'] == 'H'])
            c_peaks = len([p for p in peaks if p['element'] == 'C'])
            logger.info(f"Created {filename} (H peaks: {h_peaks}, C peaks: {c_peaks})")
            
            return True, filename
            
        except Exception as e:
            logger.error(f"Error processing {row['NP_MRD_ID']}: {e}")
            logger.error(traceback.format_exc())
            return False, str(e)