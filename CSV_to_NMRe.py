import pandas as pd
import os
import glob
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import Descriptors
import numpy as np
from datetime import datetime

class CSVToNMReDATA:
    def __init__(self, csv_file, txt_directory, output_directory):
        """
        Initialize the converter
        
        Args:
            csv_file: Path to CSV with columns Natural_Products_Name, NP_MRD_ID, SMILES
            txt_directory: Directory containing txt files named with NP_MRD_ID
            output_directory: Directory to save NMReDATA files
        """
        self.csv_file = "NP-ID and structure NP0100001-NP0150000.csv"
        self.txt_directory = "NP-MRD_nmr_peak_lists_NP0100001_NP0150000/NP-MRD_nmr_peak_lists_NP0100001_NP0150000/"
        self.output_directory = "CSV_to_NMRe_output/"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Load CSV data
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} compounds from CSV")
        print(f"CSV file: {csv_file}")
        print(f"TXT directory: {txt_directory}")
        print(f"Output directory: {output_directory}")
    
    def find_txt_file(self, np_mrd_id):
        """Find the txt file containing the NP_MRD_ID anywhere in the filename"""
        # First try exact pattern (for backwards compatibility)
        pattern = os.path.join(self.txt_directory, f"*{np_mrd_id}*.txt")
        files = glob.glob(pattern)
        
        if len(files) == 1:
            print(f"Found file: {os.path.basename(files[0])}")
            return files[0]
        elif len(files) == 0:
            # Try case-insensitive search
            all_txt_files = glob.glob(os.path.join(self.txt_directory, "*.txt"))
            matching_files = []
            
            for file_path in all_txt_files:
                filename = os.path.basename(file_path)
                # Check if NP_MRD_ID appears anywhere in the filename (case-insensitive)
                if np_mrd_id.lower() in filename.lower():
                    matching_files.append(file_path)
            
            if len(matching_files) == 1:
                print(f"Found file (case-insensitive): {os.path.basename(matching_files[0])}")
                return matching_files[0]
            elif len(matching_files) > 1:
                print(f"Warning: Multiple files found for ID {np_mrd_id}:")
                for file_path in matching_files:
                    print(f"  - {os.path.basename(file_path)}")
                print(f"Using first match: {os.path.basename(matching_files[0])}")
                return matching_files[0]
            else:
                print(f"Warning: No txt file found containing ID {np_mrd_id}")
                # List some available files for debugging
                sample_files = [os.path.basename(f) for f in all_txt_files[:5]]
                print(f"Sample files in directory: {sample_files}")
                return None
        else:
            print(f"Warning: Multiple files found for ID {np_mrd_id}:")
            for file_path in files:
                print(f"  - {os.path.basename(file_path)}")
            print(f"Using first match: {os.path.basename(files[0])}")
            return files[0]
    
    def parse_peaklist(self, txt_file):
        """Parse the peaklist from txt file in format: Element,AtomNumber,ChemicalShift,Multiplicity,CouplingConstants"""
        try:
            with open(txt_file, 'r') as f:
                content = f.read().strip()
            
            peaks = []
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    parts = line.split(',')
                    if len(parts) >= 4:
                        element = parts[0].strip()
                        atom_number = parts[1].strip()
                        chemical_shift = parts[2].strip()
                        multiplicity = parts[3].strip()
                        coupling_constants = parts[4].strip() if len(parts) > 4 else ""
                        
                        # Skip entries with empty chemical shifts (like some O, S atoms)
                        if chemical_shift == "":
                            continue
                            
                        try:
                            shift = float(chemical_shift)
                            
                            # Parse coupling constants if present
                            coupling = []
                            if coupling_constants and coupling_constants != '':
                                # Remove quotes and split by comma
                                coupling_str = coupling_constants.strip('"').strip("'")
                                if coupling_str:
                                    coupling = [float(x.strip()) for x in coupling_str.split(',') if x.strip()]
                            
                            peaks.append({
                                'element': element,
                                'atom_number': int(atom_number),
                                'shift': shift,
                                'multiplicity': multiplicity if multiplicity else 's',
                                'coupling': coupling
                            })
                        except ValueError as e:
                            print(f"Error parsing line '{line}': {e}")
                            continue
            
            return peaks
        except Exception as e:
            print(f"Error parsing {txt_file}: {e}")
            return []
    
    def consolidate_equivalent_peaks(self, peaks):
        """Consolidate peaks with identical chemical shifts and multiplicities regardless of atom numbering"""
        consolidated = {}
        
        for peak in peaks:
            # Create a key based on chemical shift, multiplicity, and coupling pattern
            # Round chemical shift to avoid floating point precision issues
            rounded_shift = round(peak['shift'], 2)
            coupling_tuple = tuple(sorted(peak['coupling'])) if peak['coupling'] else tuple()
            key = (rounded_shift, peak['multiplicity'], coupling_tuple)
            
            if key in consolidated:
                # Add atom number to existing peak
                consolidated[key]['atom_numbers'].append(peak['atom_number'])
                consolidated[key]['count'] += 1
            else:
                # Create new consolidated peak
                consolidated[key] = {
                    'element': peak['element'],
                    'shift': peak['shift'],  # Keep original precision for display
                    'multiplicity': peak['multiplicity'],
                    'coupling': peak['coupling'],
                    'atom_numbers': [peak['atom_number']],
                    'count': 1
                }
        
        # Sort consolidated peaks by chemical shift
        consolidated_list = sorted(consolidated.values(), key=lambda x: x['shift'])
        
        # Print consolidation summary for debugging
        print(f"\nConsolidation summary:")
        for peak in consolidated_list:
            if peak['count'] > 1:
                atom_range = f"{min(peak['atom_numbers'])}-{max(peak['atom_numbers'])}" if len(set(peak['atom_numbers'])) > 1 else str(peak['atom_numbers'][0])
                atoms_str = ','.join(map(str, sorted(peak['atom_numbers'])))
                print(f"  {peak['shift']} ppm, {peak['multiplicity']}: atoms {atoms_str} → {atom_range}, {peak['count']}H")
        
        return consolidated_list
    
    def generate_3d_coordinates(self, mol):
        """Generate 3D coordinates for the molecule including explicit hydrogens"""
        try:
            # Add explicit hydrogens to the molecule
            mol_h = Chem.AddHs(mol)
            print(f"Added hydrogens: {mol_h.GetNumAtoms()} total atoms ({mol.GetNumAtoms()} heavy atoms)")
            
            # Generate 3D conformer
            confId = AllChem.EmbedMolecule(mol_h, randomSeed=42)
            if confId == -1:
                print("Warning: Failed to generate 3D conformer, trying with different parameters")
                confId = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=42)
            
            if confId != -1:
                # Optimize the geometry using MMFF force field
                try:
                    AllChem.MMFFOptimizeMolecule(mol_h, confId=confId)
                except:
                    print("Warning: MMFF optimization failed, using UFF")
                    AllChem.UFFOptimizeMolecule(mol_h, confId=confId)
                
                print(f"Successfully generated 3D coordinates for {mol_h.GetNumAtoms()} atoms")
                return mol_h
            else:
                print("Error: Could not embed molecule in 3D")
                return None
                
        except Exception as e:
            print(f"Error generating 3D coordinates: {e}")
            return None
    
    def create_mol_block(self, mol_3d):
        """Create MOL block with 3D coordinates in CDK format including all atoms"""
        if mol_3d is None:
            return ""
        
        try:
            # Ensure we have a conformer with 3D coordinates
            if mol_3d.GetNumConformers() == 0:
                print("Warning: No conformer found, generating new 3D coordinates")
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_3d)
            
            # Get the MOL block with explicit hydrogens and 3D coordinates
            mol_block = Chem.MolToMolBlock(mol_3d, confId=0)
            
            # Verify that we have the expected number of atoms
            atom_count = mol_3d.GetNumAtoms()
            print(f"MOL block contains {atom_count} atoms (including hydrogens)")
            
            return mol_block
        except Exception as e:
            print(f"Error creating MOL block: {e}")
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
        
        print(f"Peaklist has {len(peak_atoms)} assigned atoms")
        print(f"3D structure has {len(struct_atoms)} total atoms")
        
        # Strategy 1: Direct mapping (if numbering is 0-indexed or 1-indexed)
        mapping = self.try_direct_mapping(peak_atoms, struct_atoms)
        if mapping:
            print("Using direct atom mapping")
            return mapping
        
        # Strategy 2: Chemical shift based mapping
        mapping = self.try_chemical_shift_mapping(peak_atoms, struct_atoms, mol_3d)
        if mapping:
            print("Using chemical shift-based mapping")
            return mapping
        
        # Strategy 3: Element type mapping with heuristics
        mapping = self.try_heuristic_mapping(peak_atoms, struct_atoms)
        if mapping:
            print("Using heuristic-based mapping")
            return mapping
        
        print("Warning: Could not establish reliable atom mapping")
        return {}
    
    def try_direct_mapping(self, peak_atoms, struct_atoms):
        """Try direct 1:1 mapping assuming peaklist numbers correspond to structure indices"""
        mapping = {}
        
        # Try 0-indexed mapping (peaklist atom 1 = structure atom 0)
        for peak_num, peak_data in peak_atoms.items():
            struct_idx = peak_num - 1  # Convert to 0-indexed
            if 0 <= struct_idx < len(struct_atoms):
                if struct_atoms[struct_idx]['element'] == peak_data['element']:
                    mapping[peak_num] = struct_idx
                else:
                    return None  # Mismatch, direct mapping won't work
            else:
                return None  # Out of range
        
        return mapping if len(mapping) == len(peak_atoms) else None
    
    def try_chemical_shift_mapping(self, peak_atoms, struct_atoms, mol_3d):
        """Use chemical shift prediction to map atoms"""
        try:
            from rdkit.Chem import Descriptors
            # This is a simplified approach - in practice you'd use more sophisticated prediction
            mapping = {}
            
            # Group atoms by element type
            h_peaks = {k: v for k, v in peak_atoms.items() if v['element'] == 'H'}
            c_peaks = {k: v for k, v in peak_atoms.items() if v['element'] == 'C'}
            
            h_atoms = [i for i, atom in enumerate(struct_atoms) if atom['element'] == 'H']
            c_atoms = [i for i, atom in enumerate(struct_atoms) if atom['element'] == 'C']
            
            # Simple heuristic mapping based on chemical environment
            # This is very basic - a real implementation would use more sophisticated methods
            
            if len(h_peaks) <= len(h_atoms) and len(c_peaks) <= len(c_atoms):
                # Map carbons first (usually more reliable)
                c_peak_list = sorted(c_peaks.items(), key=lambda x: x[1]['shift'])
                for i, (peak_num, peak_data) in enumerate(c_peak_list):
                    if i < len(c_atoms):
                        mapping[peak_num] = c_atoms[i]
                
                # Map hydrogens
                h_peak_list = sorted(h_peaks.items(), key=lambda x: x[1]['shift'])
                for i, (peak_num, peak_data) in enumerate(h_peak_list):
                    if i < len(h_atoms):
                        mapping[peak_num] = h_atoms[i]
                
                return mapping
            
            return None
            
        except Exception as e:
            print(f"Chemical shift mapping failed: {e}")
            return None
    
    def try_heuristic_mapping(self, peak_atoms, struct_atoms):
        """Use chemical heuristics to map atoms"""
        mapping = {}
        
        # Group by element type
        for element in ['C', 'H', 'N', 'O', 'S']:
            peak_nums = [k for k, v in peak_atoms.items() if v['element'] == element]
            struct_indices = [i for i, atom in enumerate(struct_atoms) if atom['element'] == element]
            
            # Simple sequential mapping within each element type
            for i, peak_num in enumerate(sorted(peak_nums)):
                if i < len(struct_indices):
                    mapping[peak_num] = struct_indices[i]
        
        return mapping if len(mapping) == len(peak_atoms) else None
    
    def renumber_peaklist(self, peaks, atom_mapping):
        """Renumber peaklist atoms to match 3D structure indices"""
        if not atom_mapping:
            print("Warning: No atom mapping available, keeping original numbering")
            return peaks
        
        renumbered_peaks = []
        for peak in peaks:
            if peak['atom_number'] in atom_mapping:
                new_peak = peak.copy()
                new_peak['atom_number'] = atom_mapping[peak['atom_number']] + 1  # Convert to 1-indexed
                renumbered_peaks.append(new_peak)
            else:
                print(f"Warning: No mapping found for atom {peak['atom_number']}")
                renumbered_peaks.append(peak)  # Keep original
        
        return renumbered_peaks
    
    def preserve_original_numbering(self, mol_3d, atom_mapping, peaks):
        """Add original atom numbers as properties to preserve peaklist numbering"""
        if not atom_mapping or mol_3d is None:
            return mol_3d
        
        try:
            # Create reverse mapping (structure_idx -> original_peak_number)
            reverse_mapping = {v: k for k, v in atom_mapping.items()}
            
            # Add original numbering as atom properties
            for atom in mol_3d.GetAtoms():
                idx = atom.GetIdx()
                if idx in reverse_mapping:
                    original_num = reverse_mapping[idx]
                    atom.SetProp("OriginalAtomNumber", str(original_num))
                    atom.SetProp("HasNMRAssignment", "True")
                else:
                    atom.SetProp("HasNMRAssignment", "False")
            
            print("Added original atom numbering as properties")
            return mol_3d
            
        except Exception as e:
            print(f"Error preserving original numbering: {e}")
            return mol_3d
    
    def validate_atom_mapping(self, peaks, mol_3d):
        """Validate that peaklist atom numbers match the 3D structure"""
        if mol_3d is None:
            return False
        
        total_atoms = mol_3d.GetNumAtoms()
        peak_atom_numbers = set()
        
        for peak in peaks:
            peak_atom_numbers.add(peak['atom_number'])
        
        max_peak_atom = max(peak_atom_numbers) if peak_atom_numbers else 0
        
        print(f"3D structure has {total_atoms} atoms")
        print(f"Peaklist references atoms up to number {max_peak_atom}")
        print(f"Peak atom numbers: {sorted(peak_atom_numbers)}")
        
        if max_peak_atom > total_atoms:
            print(f"Warning: Peaklist references atom {max_peak_atom} but structure only has {total_atoms} atoms")
            return False
        
        # Check for hydrogen atoms in the structure
        h_count = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'H')
        c_count = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'C')
        other_count = total_atoms - h_count - c_count
        
        print(f"Atom composition: {h_count}H, {c_count}C, {other_count} others")
        
        return True
    
    def create_nmredata_file(self, row, peaks, mol_3d):
        """Create complete NMReDATA file content"""
        mol_block = self.create_mol_block(mol_3d)
        
        # Separate peaks by nucleus type and consolidate equivalent ones
        h_peaks = [p for p in peaks if p['element'] == 'H']
        c_peaks = [p for p in peaks if p['element'] == 'C']
        
        # Consolidate equivalent peaks
        h_peaks_consolidated = self.consolidate_equivalent_peaks(h_peaks)
        c_peaks_consolidated = self.consolidate_equivalent_peaks(c_peaks)
        
        # NMReDATA header with MOL block
        nmredata_content = f"{mol_block}\n"
        
        # Add version and level
        nmredata_content += f""">  <NMREDATA_VERSION>
1.1

>  <NMREDATA_LEVEL>
0

"""
        
        # Add 1H NMR data (consolidated format)
        if h_peaks_consolidated:
            nmredata_content += f">  <NMREDATA_1D_1H>\n"
            for peak in h_peaks_consolidated:
                line = f"{peak['shift']}, {peak['multiplicity']}"
                if peak['coupling']:
                    coupling_str = ', '.join([f"J={c}" for c in peak['coupling']])
                    line += f", {coupling_str}"
                # Use comma-separated notation for non-sequential atoms or range for sequential
                if peak['count'] > 1:
                    atom_numbers = sorted(peak['atom_numbers'])
                    # Check if atoms are sequential
                    is_sequential = all(atom_numbers[i] == atom_numbers[i-1] + 1 for i in range(1, len(atom_numbers)))
                    
                    if is_sequential and len(atom_numbers) > 2:
                        # Use range notation for sequential atoms (3 or more)
                        atom_range = f"{min(atom_numbers)}-{max(atom_numbers)}"
                    else:
                        # Use comma-separated notation for non-sequential or pairs
                        atom_range = ','.join(map(str, atom_numbers))
                    
                    line += f", {atom_range}, {peak['count']}"
                else:
                    line += f", {peak['atom_numbers'][0]}, {peak['count']}"
                nmredata_content += f"{line}\n"
            nmredata_content += "\n"
        
        # Add 13C NMR data (consolidated format)
        if c_peaks_consolidated:
            nmredata_content += f">  <NMREDATA_1D_13C>\n"
            for peak in c_peaks_consolidated:
                line = f"{peak['shift']}, {peak['multiplicity']}"
                if peak['coupling']:
                    coupling_str = ', '.join([f"J={c}" for c in peak['coupling']])
                    line += f", {coupling_str}"
                # Use comma-separated notation for non-sequential atoms or range for sequential
                if peak['count'] > 1:
                    atom_numbers = sorted(peak['atom_numbers'])
                    # Check if atoms are sequential
                    is_sequential = all(atom_numbers[i] == atom_numbers[i-1] + 1 for i in range(1, len(atom_numbers)))
                    
                    if is_sequential and len(atom_numbers) > 2:
                        # Use range notation for sequential atoms (3 or more)
                        atom_range = f"{min(atom_numbers)}-{max(atom_numbers)}"
                    else:
                        # Use comma-separated notation for non-sequential or pairs
                        atom_range = ','.join(map(str, atom_numbers))
                    
                    line += f", {atom_range}, {peak['count']}"
                else:
                    line += f", {peak['atom_numbers'][0]}, {peak['count']}"
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
    
    def process_all_compounds(self):
        """Process all compounds in the CSV file"""
        successful = 0
        failed = 0
        
        for idx, row in self.df.iterrows():
            try:
                print(f"\nProcessing {row['Natural_Products_Name']} (ID: {row['NP_MRD_ID']})")
                
                # Find corresponding txt file
                txt_file = self.find_txt_file(row['NP_MRD_ID'])
                if not txt_file:
                    failed += 1
                    continue
                
                # Parse peaklist
                peaks = self.parse_peaklist(txt_file)
                if not peaks:
                    print(f"No peaks found for {row['NP_MRD_ID']}")
                
                # Generate molecule from SMILES
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol is None:
                    print(f"Invalid SMILES for {row['NP_MRD_ID']}: {row['SMILES']}")
                    failed += 1
                    continue
                
                # Generate 3D coordinates
                mol_3d = self.generate_3d_coordinates(mol)
                
                # Create atom mapping and renumber peaks if necessary
                if mol_3d and peaks:
                    atom_mapping = self.create_atom_mapping(peaks, mol_3d)
                    
                    # Option 1: Renumber peaks to match structure (recommended for ChemBERTa)
                    peaks = self.renumber_peaklist(peaks, atom_mapping)
                    
                    # Option 2: Preserve original numbering as atom properties
                    mol_3d = self.preserve_original_numbering(mol_3d, atom_mapping, peaks)
                    
                    # Validate the final mapping
                    self.validate_atom_mapping(peaks, mol_3d)
                
                # Create NMReDATA content
                nmredata_content = self.create_nmredata_file(row, peaks, mol_3d)
                
                # Save to file
                output_filename = f"{row['NP_MRD_ID']}_{row['Natural_Products_Name'].replace(' ', '_')}.nmredata"
                output_path = os.path.join(self.output_directory, output_filename)
                
                with open(output_path, 'w') as f:
                    f.write(nmredata_content)
                
                print(f"Successfully created {output_filename}")
                successful += 1
                
            except Exception as e:
                print(f"Error processing {row['NP_MRD_ID']}: {e}")
                failed += 1
        
        print(f"\nProcessing complete:")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

def main():
    """Main function to run the converter"""
    # Update these paths according to your file locations
    csv_file = r"C:/Users/pierr/Desktop/CS MSc Project files/NP-ID and structure NP0100001-NP0150000.csv"
    txt_directory = r"C:/Users/pierr/Desktop/CS MSc Project files/NP-MRD_nmr_peak_lists_NP0100001_NP0150000/NP-MRD_nmr_peak_lists_NP0100001_NP0150000/"
    output_directory = r"C:/Users/pierr/Desktop/CS MSc Project files/CSV_to_NMRe_output/"
    
    # Create converter and process files
    converter = CSVToNMReDATA(csv_file, txt_directory, output_directory)
    converter.process_all_compounds()

if __name__ == "__main__":
    main()

# Additional utility functions

def validate_nmredata_file(file_path):
    """Validate that the created NMReDATA file is properly formatted"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for required tags
        required_tags = ['NMREDATA_VERSION', 'NMREDATA_1D_1H']
        for tag in required_tags:
            if tag not in content:
                print(f"Warning: Missing required tag {tag} in {file_path}")
                return False
        
        # Check for MOL block (should start with atom count)
        lines = content.split('\n')
        if len(lines) < 4:
            print(f"Error: File too short to contain valid MOL block in {file_path}")
            return False
        
        print(f"NMReDATA file {file_path} appears valid")
        return True
        
    except Exception as e:
        print(f"Error validating {file_path}: {e}")
        return False

def batch_validate(output_directory):
    """Validate all NMReDATA files in the output directory"""
    nmredata_files = glob.glob(os.path.join(output_directory, "*.nmredata"))
    
    valid_count = 0
    for file_path in nmredata_files:
        if validate_nmredata_file(file_path):
            valid_count += 1
    
    print(f"Validation complete: {valid_count}/{len(nmredata_files)} files are valid")

# Example usage for testing individual functions
def test_individual_compound():
    """Test function for processing a single compound"""
    # Example data
    test_row = {
        'Natural_Products_Name': 'Test Compound',
        'NP_MRD_ID': 'TEST001',
        'SMILES': 'CC(C)CC(C(=O)O)N'
    }
    
    # Initialize converter (adjust paths as needed)
    converter = CSVToNMReDATA("test.csv", "test_peaklists/", "test_output/")
    
    # Test SMILES processing
    mol = Chem.MolFromSmiles(test_row['SMILES'])
    if mol:
        mol_3d = converter.generate_3d_coordinates(mol)
        mol_block = converter.create_mol_block(mol_3d)
        print("MOL block generated successfully")
        print(f"First few lines:\n{mol_block.split(chr(10))[:5]}")
    else:
        print("Failed to process SMILES")

# Debugging function
# Debugging function
def debug_file_matching(csv_file, txt_directory):
    """Debug function to test file matching between CSV and txt files"""
    df = pd.read_csv(csv_file)
    all_txt_files = glob.glob(os.path.join(txt_directory, "*.txt"))
    
    print(f"Found {len(all_txt_files)} txt files in {txt_directory}")
    print(f"Found {len(df)} entries in CSV file")
    print("\nFile matching test:")
    print("-" * 50)
    
    converter = CSVToNMReDATA(csv_file, txt_directory, "temp/")
    
    matched = 0
    unmatched = 0
    
    for idx, row in df.iterrows():
        np_id = row['NP_MRD_ID']
        found_file = converter.find_txt_file(np_id)
        
        if found_file:
            print(f"✓ {np_id} → {os.path.basename(found_file)}")
            matched += 1
        else:
            print(f"✗ {np_id} → No file found")
            unmatched += 1
            
            # Show similar filenames for debugging
            similar_files = []
            for txt_file in all_txt_files:
                filename = os.path.basename(txt_file)
                # Check for partial matches
                if any(part in filename.lower() for part in np_id.lower().split('_')):
                    similar_files.append(filename)
            
            if similar_files:
                print(f"    Similar files: {similar_files[:3]}")
    
    print(f"\nSummary: {matched} matched, {unmatched} unmatched")
    return matched, unmatched
    """Debug function to test peaklist parsing"""
    converter = CSVToNMReDATA("dummy.csv", "dummy/", "dummy/")
    peaks = converter.parse_peaklist(txt_file_path)
    
    print(f"Parsed {len(peaks)} peaks:")
    for peak in peaks[:5]:  # Show first 5 peaks
        print(f"  {peak['element']}{peak['atom_number']}: {peak['shift']} ppm, {peak['multiplicity']}")
    
    # Test consolidation
    h_peaks = [p for p in peaks if p['element'] == 'H']
    consolidated = converter.consolidate_equivalent_peaks(h_peaks)
    print(f"\nConsolidated to {len(consolidated)} unique 1H peaks")
    
    return peaks, consolidated