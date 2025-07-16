"""
NMR data parsing module - Updated to disable consolidation
"""

import os
import logging
from collections import defaultdict
import re
from utils import setup_logging

logger = setup_logging()


class NMRParser:
    """Handles NMR peak list parsing WITHOUT consolidation"""
    
    def __init__(self):
        # Extended list of valid multiplicities
        self.valid_multiplicities = [
            's', 'd', 't', 'q', 'p', 'h', 'm', 'br',  # Basic multiplicities
            'dd', 'dt', 'td', 'dq', 'qd', 'tt', 'tq', 'qt',  # Double combinations
            'ddd', 'ddt', 'dtd', 'tdd', 'ddq', 'dqd', 'qdd',  # Triple combinations
            'dtq', 'dqt', 'tdq', 'tqd', 'qtd', 'qdt',  # More triple combinations
            'dddd', 'dddt', 'ddtd', 'dtdd', 'tddd',  # Quadruple combinations
            'dddq', 'ddqd', 'dqdd', 'qddd',  # More quadruple combinations
            'bs', 'bd', 'bt', 'bq',  # Broad variations
            'brs', 'brd', 'brt', 'brq',  # Broad variations
            'app', 'appt', 'appd', 'appq',  # Apparent multiplicities
            'complex', 'comp', 'multiplet', 'mult',  # Complex patterns
            'overlap', 'ov', 'ovlp',  # Overlapping signals
            'ABq', 'ABX', 'ABC', 'ABCD', 'ABMX',  # Spin system notations
            'J', 'Japp'  # Coupling-related
        ]
        
        # Create lowercase set for fast lookup
        self.valid_multiplicities_lower = {m.lower() for m in self.valid_multiplicities}
    
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
                            
                            # Normalize and validate multiplicity
                            multiplicity = self._normalize_multiplicity(multiplicity)
                            
                            coupling_constants = parts[4].strip() if len(parts) > 4 else ""
                            
                            # Parse coupling constants with better error handling
                            coupling = self._parse_coupling_constants(coupling_constants)
                            
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
            return []
        except Exception as e:
            logger.error(f"Error parsing {txt_file}: {e}")
            return []
    
    def _normalize_multiplicity(self, multiplicity):
        """Normalize multiplicity notation"""
        mult_lower = multiplicity.lower().strip()
        
        # Direct match
        if mult_lower in self.valid_multiplicities_lower:
            return mult_lower
        
        # Common variations and abbreviations
        multiplicity_map = {
            'singlet': 's',
            'doublet': 'd',
            'triplet': 't',
            'quartet': 'q',
            'quintet': 'p',
            'sextet': 'h',
            'septet': 'h',
            'heptet': 'h',
            'octet': 'o',
            'nonet': 'n',
            'broad': 'br',
            'broad singlet': 'bs',
            'broad doublet': 'bd',
            'apparent': 'app',
            'overlapping': 'ovlp',
            'overlapped': 'ovlp',
            'multiple': 'm',
            'multiplet': 'm',
            'complex multiplet': 'complex',
            'complex pattern': 'complex',
            'unresolved': 'm',
            'not resolved': 'm'
        }
        
        # Check common variations
        if mult_lower in multiplicity_map:
            return multiplicity_map[mult_lower]
        
        # Check for patterns with spaces or dashes
        cleaned = re.sub(r'[\s\-_]+', '', mult_lower)
        if cleaned in self.valid_multiplicities_lower:
            return cleaned
        
        # Check for J-containing patterns (e.g., "d, J=7.5")
        if 'j=' in mult_lower or 'j =' in mult_lower:
            base_mult = mult_lower.split(',')[0].strip()
            if base_mult in self.valid_multiplicities_lower:
                return base_mult
            elif base_mult in multiplicity_map:
                return multiplicity_map[base_mult]
        
        # Default to 'm' for unrecognized patterns
        logger.debug(f"Unknown multiplicity '{multiplicity}', using 'm'")
        return 'm'
    
    def _parse_coupling_constants(self, coupling_str):
        """Parse coupling constants with improved handling"""
        coupling = []
        
        if not coupling_str:
            return coupling
        
        # Remove quotes and extra spaces
        coupling_str = coupling_str.strip('"\'').strip()
        
        if not coupling_str:
            return coupling
        
        try:
            # Handle different formats
            # Format 1: "7.5, 7.5, 1.5"
            # Format 2: "J=7.5, J=7.5, J=1.5"
            # Format 3: "7.5 Hz, 7.5 Hz"
            # Format 4: "J = 7.5 Hz"
            
            # Remove 'Hz' and 'J=' patterns
            cleaned = re.sub(r'[Jj]\s*=\s*', '', coupling_str)
            cleaned = re.sub(r'\s*[Hh][Zz]', '', cleaned)
            
            # Split by comma or semicolon
            parts = re.split(r'[,;]', cleaned)
            
            for part in parts:
                part = part.strip()
                if part:
                    try:
                        j_val = float(part)
                        if 0 <= j_val <= 50:  # Reasonable J-coupling range
                            coupling.append(j_val)
                        else:
                            logger.debug(f"Unusual J-coupling {j_val} Hz")
                            # Still include it for completeness
                            coupling.append(j_val)
                    except ValueError:
                        # Try to extract number from string
                        numbers = re.findall(r'[-+]?\d*\.?\d+', part)
                        for num in numbers:
                            try:
                                j_val = float(num)
                                if 0 <= j_val <= 50:
                                    coupling.append(j_val)
                                else:
                                    coupling.append(j_val)
                            except ValueError:
                                continue
                                
        except Exception as e:
            logger.debug(f"Could not parse coupling constants '{coupling_str}': {e}")
        
        return coupling
    
    def consolidate_equivalent_peaks(self, peaks):
        """
        DEPRECATED: This method consolidates peaks. In the new system, we do NOT consolidate.
        Returns peaks as-is without any consolidation.
        """
        logger.warning("consolidate_equivalent_peaks called - returning peaks without consolidation")
        return peaks
    
    def validate_peak_consolidation(self, original_peaks, consolidated_peaks):
        """
        DEPRECATED: Validation for consolidation. Always returns True in non-consolidation mode.
        """
        return True
    
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
    
    def format_consolidated_peak_info(self, consolidated_peak):
        """Format peak information for logging or display (non-consolidated version)"""
        if 'atom_numbers' in consolidated_peak:
            # Old consolidated format
            atom_nums = consolidated_peak['atom_numbers']
            if len(atom_nums) == 1:
                info = f"{consolidated_peak['shift']:.4f} ppm (atom {atom_nums[0]})"
            else:
                info = f"{consolidated_peak['shift']:.4f} ppm (atoms {atom_nums})"
        else:
            # Individual peak format
            info = f"{consolidated_peak['shift']:.4f} ppm (atom {consolidated_peak['atom_number']})"
        
        info += f", {consolidated_peak.get('multiplicity', 's')}"
        
        if consolidated_peak.get('coupling'):
            j_str = ', '.join(f"J={j:.1f}" for j in consolidated_peak['coupling'])
            info += f", {j_str}"
        
        return info
    
    def validate_nmr_completeness(self, peaks, mol_3d):
        """Validate that NMR data is complete (all atoms have peaks)"""
        from rdkit import Chem
        
        # Count atoms in molecule
        atom_counts = {}
        for atom in mol_3d.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
        
        # Count peaks
        peak_counts = {}
        for peak in peaks:
            element = peak['element']
            peak_counts[element] = peak_counts.get(element, 0) + 1
        
        # Check completeness
        complete = True
        warnings = []
        
        # Check H NMR
        if 'H' in atom_counts:
            h_atoms = atom_counts['H']
            h_peaks = peak_counts.get('H', 0)
            if h_peaks != h_atoms:
                complete = False
                warnings.append(f"H NMR incomplete: {h_peaks} peaks for {h_atoms} H atoms")
        
        # Check C NMR
        if 'C' in atom_counts:
            c_atoms = atom_counts['C']
            c_peaks = peak_counts.get('C', 0)
            if c_peaks != c_atoms:
                complete = False
                warnings.append(f"C NMR incomplete: {c_peaks} peaks for {c_atoms} C atoms")
        
        return complete, warnings