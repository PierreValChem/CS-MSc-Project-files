"""
NMReDATA file writer module
"""

from datetime import datetime
import logging

# Import with fallback
try:
    from Data_Processing.utils import setup_logging
    logger = setup_logging()
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class NMReDataWriter:
    """Handles creation of NMReDATA format files"""
    
    def __init__(self):
        # Import MoleculeProcessor here to avoid circular imports
        from Data_Processing.molecule_processor import MoleculeProcessor
        self.molecule_processor = MoleculeProcessor()
    
    def create_nmredata_file(self, row, h_peaks_consolidated, c_peaks_consolidated, mol_3d):
        """Create complete NMReDATA file content"""
        mol_block = self.molecule_processor.create_mol_block(mol_3d)
        
        # Check if MOL block is empty or invalid
        if not mol_block or len(mol_block.strip()) < 10:
            logger.warning(f"Empty or invalid MOL block for {row['NP_MRD_ID']}")
            return None  # Return None to indicate failure
        
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
                line = self._format_peak_line(peak)
                nmredata_content += f"{line}\n"
            nmredata_content += "\n"
        
        # Add 13C NMR data
        if c_peaks_consolidated:
            nmredata_content += f">  <NMREDATA_1D_13C>\n"
            for peak in c_peaks_consolidated:
                line = self._format_peak_line(peak)
                nmredata_content += f"{line}\n"
            nmredata_content += "\n"
        
        # Add metadata
        nmredata_content += self._create_metadata(row)
        
        return nmredata_content
    
    def _format_peak_line(self, peak):
        """Format a single peak line for NMReDATA with enhanced consolidation support"""
        # Start with chemical shift
        line = f"{peak['shift']}"
        
        # Add multiplicity
        line += f", {peak['multiplicity']}"
        
        # Add coupling constants in a consistent format
        if peak['coupling']:
            # Format all J values consistently
            # Sort in descending order for consistency
            sorted_couplings = sorted(peak['coupling'], reverse=True)
            
            # Group similar J values (within 0.1 Hz tolerance)
            grouped_couplings = self._group_similar_couplings(sorted_couplings)
            
            # Format as J=value
            coupling_strs = []
            for j_value, count in grouped_couplings:
                if count > 1:
                    # Multiple identical couplings (e.g., J=7.5x2)
                    coupling_strs.append(f"J={j_value:.1f}x{count}")
                else:
                    coupling_strs.append(f"J={j_value:.1f}")
            
            line += f", {', '.join(coupling_strs)}"
        
        # Add atom numbers
        atom_numbers = sorted(peak['atom_numbers'])
        
        # Format atom numbers efficiently
        atom_range = self._format_atom_range(atom_numbers)
        line += f", {atom_range}"
        
        # Add total count
        line += f", {peak['count']}"
        
        return line
    
    def _group_similar_couplings(self, couplings, tolerance=0.1):
        """Group similar coupling constants within tolerance"""
        if not couplings:
            return []
        
        grouped = []
        current_value = couplings[0]
        count = 1
        
        for i in range(1, len(couplings)):
            if abs(couplings[i] - current_value) <= tolerance:
                # Similar value, increment count
                count += 1
            else:
                # Different value, save current group
                grouped.append((current_value, count))
                current_value = couplings[i]
                count = 1
        
        # Don't forget the last group
        grouped.append((current_value, count))
        
        return grouped
    
    def _format_atom_range(self, atom_numbers):
        """Format atom numbers as ranges where possible"""
        if not atom_numbers:
            return ""
        
        if len(atom_numbers) == 1:
            return str(atom_numbers[0])
        
        # Find consecutive sequences
        ranges = []
        start = atom_numbers[0]
        end = atom_numbers[0]
        
        for i in range(1, len(atom_numbers)):
            if atom_numbers[i] == end + 1:
                # Consecutive, extend range
                end = atom_numbers[i]
            else:
                # Not consecutive, save current range
                if start == end:
                    ranges.append(str(start))
                elif end - start == 1:
                    # Just two numbers, list them
                    ranges.append(f"{start},{end}")
                else:
                    # Range of 3 or more
                    ranges.append(f"{start}-{end}")
                
                start = atom_numbers[i]
                end = atom_numbers[i]
        
        # Handle the last range
        if start == end:
            ranges.append(str(start))
        elif end - start == 1:
            ranges.append(f"{start},{end}")
        else:
            ranges.append(f"{start}-{end}")
        
        # Combine all ranges
        return ','.join(ranges)
    
    def _create_metadata(self, row):
        """Create metadata section for NMReDATA"""
        metadata = f""">  <NMREDATA_SOLVENT>
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
        return metadata
    
    def format_peak_summary(self, peak):
        """Create a human-readable summary of a consolidated peak"""
        summary = f"{peak['shift']:.3f} ppm: "
        
        # Add atom info
        if peak['count'] == 1:
            summary += f"atom {peak['atom_numbers'][0]}"
        else:
            atom_range = self._format_atom_range(sorted(peak['atom_numbers']))
            summary += f"{peak['count']} atoms ({atom_range})"
        
        # Add multiplicity
        summary += f", {peak['multiplicity']}"
        
        # Add coupling info
        if peak['coupling']:
            grouped = self._group_similar_couplings(sorted(peak['coupling'], reverse=True))
            j_parts = []
            for j_val, count in grouped:
                if count > 1:
                    j_parts.append(f"{j_val:.1f} Hz (×{count})")
                else:
                    j_parts.append(f"{j_val:.1f} Hz")
            summary += f", J = {', '.join(j_parts)}"
        
        return summary