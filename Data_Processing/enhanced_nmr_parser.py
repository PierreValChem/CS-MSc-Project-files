#!/usr/bin/env python3
"""
Enhanced NMR Parser for handling multiple file formats and smart file selection
Includes padding logic for atom-peak correspondence
"""

import re
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


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
        lines = content.split('\n')
        for line in lines[:50]:  # Check first 50 lines
            # New format: atom_id symbol chemical_shift mult coupling
            if 'atom_id' in line and 'symbol' in line and 'chemical_shift' in line:
                return True
            # Old format: element,atom_number,shift,mult,coupling
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
            
            # Check if it's valid NMR format
            if not self.is_valid_nmr_format(content):
                self.logger.debug(f"Invalid NMR format: {txt_file}")
                return []
            
            peaks = []
            lines = content.split('\n')
            
            # Detect format
            is_new_format = False
            for line in lines[:50]:
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
                    
                    # Only process H and C for NMR
                    if element not in ['H', 'C']:
                        continue
                    
                    shift = float(parts[2])
                    multiplicity = parts[3] if len(parts) > 3 and parts[3] != '.' else 's'
                    
                    # Parse coupling constants
                    coupling = []
                    if len(parts) > 4:
                        coupling_str = ' '.join(parts[4:])
                        coupling = self._parse_coupling_string(coupling_str)
                    
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
                            coupling = self._parse_coupling_string(parts[4])
                        
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
    
    def _parse_coupling_string(self, coupling_str):
        """Parse coupling constants from string"""
        coupling = []
        if not coupling_str:
            return coupling
        
        # Remove quotes and extra spaces
        coupling_str = coupling_str.strip('"\'').strip()
        
        # Remove J= and Hz
        cleaned = re.sub(r'[Jj]\s*=\s*', '', coupling_str)
        cleaned = re.sub(r'\s*[Hh][Zz]', '', cleaned)
        
        # Extract numbers
        for match in re.findall(r'[\d.]+', cleaned):
            try:
                j_val = float(match)
                if 0 <= j_val <= 50:  # Reasonable range
                    coupling.append(j_val)
            except:
                continue
        
        return coupling
    
    def validate_and_pad_peaks(self, peaks, mol_3d, element_type):
        """
        Validate peaks against molecule atoms and add padding if needed.
        Returns (validated_peaks, is_valid) tuple.
        If peaks > atoms, returns ([], False) - file should be skipped.
        If atoms > peaks, adds null padding with -1 atom numbers.
        """
        # Count atoms of specified element in molecule
        atom_count = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == element_type)
        
        # Get peaks for this element
        element_peaks = [p for p in peaks if p['element'] == element_type]
        peak_count = len(element_peaks)
        
        self.logger.debug(f"{element_type} atoms: {atom_count}, peaks: {peak_count}")
        
        # Case 1: More peaks than atoms - invalid file
        if peak_count > atom_count:
            self.logger.warning(f"More {element_type} peaks ({peak_count}) than atoms ({atom_count}) - skipping file")
            return [], False
        
        # Case 2: Equal peaks and atoms - perfect match
        if peak_count == atom_count:
            return element_peaks, True
        
        # Case 3: More atoms than peaks - add padding
        if peak_count < atom_count:
            padding_needed = atom_count - peak_count
            self.logger.info(f"Adding {padding_needed} padding entries for {element_type}")
            
            # Add null peaks for missing atoms
            padded_peaks = element_peaks.copy()
            
            # Add padding peaks with -1 as special "unknown atom" indicator
            for i in range(padding_needed):
                padding_peak = {
                    'element': element_type,
                    'atom_number': -1,  # Special value: "unknown which atom this corresponds to"
                    'shift': None,  # Null shift value
                    'multiplicity': 'null',
                    'coupling': []
                }
                padded_peaks.append(padding_peak)
            
            return padded_peaks, True
    
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
    
    def select_best_file(self, files_with_peaks, mol_3d):
        """
        Select the most comprehensive file from multiple options.
        Now considers atom-peak correspondence in scoring.
        """
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
            
            # Check atom-peak correspondence
            h_atoms = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'H')
            c_atoms = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'C')
            
            # Skip files with more peaks than atoms
            if len(h_peaks) > h_atoms or len(c_peaks) > c_atoms:
                self.logger.debug(f"Skipping {file_path.name}: more peaks than atoms")
                continue
            
            # Score based on completeness (prefer files with more complete data)
            h_completeness = len(h_peaks) / h_atoms if h_atoms > 0 else 0
            c_completeness = len(c_peaks) / c_atoms if c_atoms > 0 else 0
            
            score += h_completeness * 100  # H completeness worth up to 100 points
            score += c_completeness * 150  # C completeness worth up to 150 points
            
            # Bonus for having both H and C
            if h_peaks and c_peaks:
                score += 50
            
            # Bonus for peaks with coupling constants
            peaks_with_coupling = [p for p in peaks if p.get('coupling')]
            score += len(peaks_with_coupling) * 2
            
            # Bonus for peaks with multiplicity info
            peaks_with_mult = [p for p in peaks if p.get('multiplicity') and p['multiplicity'] not in ['s', 'null']]
            score += len(peaks_with_mult)
            
            self.logger.debug(f"File {file_path.name} score: {score:.1f} "
                            f"(H:{len(h_peaks)}/{h_atoms}, C:{len(c_peaks)}/{c_atoms})")
            
            if score > best_score:
                best_score = score
                best_file = file_path
                best_peaks = peaks
        
        return best_file, best_peaks
    
    def merge_h_and_c_files(self, files_with_peaks, mol_3d):
        """Merge separate H and C NMR files"""
        h_only_files = []
        c_only_files = []
        mixed_files = []
        
        for file_path, peaks in files_with_peaks:
            h_peaks = [p for p in peaks if p['element'] == 'H']
            c_peaks = [p for p in peaks if p['element'] == 'C']
            
            # Check atom counts
            h_atoms = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'H')
            c_atoms = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'C')
            
            # Skip files with too many peaks
            if len(h_peaks) > h_atoms or len(c_peaks) > c_atoms:
                continue
            
            if h_peaks and not c_peaks:
                h_only_files.append((file_path, peaks))
            elif c_peaks and not h_peaks:
                c_only_files.append((file_path, peaks))
            elif h_peaks and c_peaks:
                mixed_files.append((file_path, peaks))
        
        # If we have mixed files, use the best one
        if mixed_files:
            return self.select_best_file(mixed_files, mol_3d)
        
        # Otherwise, merge H and C files
        merged_peaks = []
        best_h_file = None
        best_c_file = None
        
        if h_only_files:
            best_h_file, h_peaks = self.select_best_file(h_only_files, mol_3d)
            merged_peaks.extend(h_peaks)
        
        if c_only_files:
            best_c_file, c_peaks = self.select_best_file(c_only_files, mol_3d)
            merged_peaks.extend(c_peaks)
        
        if merged_peaks:
            # Return the H file as primary (or C if no H)
            primary_file = best_h_file if best_h_file else best_c_file
            return primary_file, merged_peaks
        
        return None, []