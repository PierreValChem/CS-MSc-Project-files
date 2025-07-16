#!/usr/bin/env python3
"""
NMReDATA file writer with support for canonical SMILES and padded peaks
"""

from datetime import datetime
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)


class NMReDataWriter:
    """Handles creation of NMReDATA format files with enhanced molecular representations"""
    
    def __init__(self):
        from molecule_processor import MoleculeProcessor
        self.molecule_processor = MoleculeProcessor()
    
    def create_enhanced_nmredata_file(self, row, h_peaks, c_peaks, mol_3d):
        """
        Create enhanced NMReDATA with all molecular representations
        Now handles padded peaks (with None shifts) properly
        """
        # Create MOL block
        mol_block = self.molecule_processor.create_mol_block(mol_3d)
        if not mol_block or len(mol_block.strip()) < 10:
            logger.warning(f"Empty or invalid MOL block for {row['NP_MRD_ID']}")
            return None
        
        # Build NMReDATA content
        nmredata_content = f"{mol_block}\n"
        
        # Add version and level
        nmredata_content += f""">  <NMREDATA_VERSION>
1.1

>  <NMREDATA_LEVEL>
0

"""
        
        # Add 1H NMR data (including padded peaks)
        if h_peaks:
            nmredata_content += f">  <NMREDATA_1D_1H>\n"
            # Sort real peaks by atom number, padded peaks at the end
            real_h_peaks = [p for p in h_peaks if p['atom_number'] != -1]
            padded_h_peaks = [p for p in h_peaks if p['atom_number'] == -1]
            
            # Add real peaks first (sorted by atom number)
            for peak in sorted(real_h_peaks, key=lambda x: x['atom_number']):
                line = self._format_peak_line(peak)
                nmredata_content += f"{line}\n"
            
            # Add padded peaks at the end
            for peak in padded_h_peaks:
                line = self._format_peak_line(peak)
                nmredata_content += f"{line}\n"
            
            nmredata_content += "\n"
        
        # Add 13C NMR data (including padded peaks)
        if c_peaks:
            nmredata_content += f">  <NMREDATA_1D_13C>\n"
            # Sort real peaks by atom number, padded peaks at the end
            real_c_peaks = [p for p in c_peaks if p['atom_number'] != -1]
            padded_c_peaks = [p for p in c_peaks if p['atom_number'] == -1]
            
            # Add real peaks first (sorted by atom number)
            for peak in sorted(real_c_peaks, key=lambda x: x['atom_number']):
                line = self._format_peak_line(peak)
                nmredata_content += f"{line}\n"
            
            # Add padded peaks at the end
            for peak in padded_c_peaks:
                line = self._format_peak_line(peak)
                nmredata_content += f"{line}\n"
            
            nmredata_content += "\n"
        
        # Add all molecular representations
        
        # Original SMILES (from CSV)
        nmredata_content += f">  <SMILES>\n{row['SMILES']}\n\n"
        
        # Canonical SMILES
        try:
            canonical_smiles = Chem.MolToSmiles(mol_3d, canonical=True)
            nmredata_content += f">  <Canonical_SMILES>\n{canonical_smiles}\n\n"
        except Exception as e:
            logger.warning(f"Could not generate canonical SMILES for {row['NP_MRD_ID']}: {e}")
        
        # Isomeric SMILES (preserves stereochemistry)
        try:
            isomeric_smiles = Chem.MolToSmiles(mol_3d, isomericSmiles=True)
            nmredata_content += f">  <Isomeric_SMILES>\n{isomeric_smiles}\n\n"
        except Exception as e:
            logger.warning(f"Could not generate isomeric SMILES for {row['NP_MRD_ID']}: {e}")
        
        # InChI and InChIKey
        try:
            inchi = Chem.MolToInchi(mol_3d)
            nmredata_content += f">  <InChI>\n{inchi}\n\n"
            
            inchi_key = Chem.MolToInchiKey(mol_3d)
            nmredata_content += f">  <InChIKey>\n{inchi_key}\n\n"
        except Exception as e:
            logger.warning(f"Could not generate InChI for {row['NP_MRD_ID']}: {e}")
        
        # Molecular Formula
        try:
            formula = rdMolDescriptors.CalcMolFormula(mol_3d)
            nmredata_content += f">  <Molecular_Formula>\n{formula}\n\n"
        except:
            pass
        
        # Molecular Weight
        try:
            mw = Descriptors.ExactMolWt(mol_3d)
            nmredata_content += f">  <Molecular_Weight>\n{mw:.4f}\n\n"
        except:
            pass
        
        # Atom and peak counts for validation
        h_atoms = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'H')
        c_atoms = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'C')
        h_peaks_count = len(h_peaks)
        c_peaks_count = len(c_peaks)
        h_real_peaks = len([p for p in h_peaks if p['shift'] is not None])
        c_real_peaks = len([p for p in c_peaks if p['shift'] is not None])
        h_padded = h_peaks_count - h_real_peaks
        c_padded = c_peaks_count - c_real_peaks
        
        # Determine if this is complete data (no padding)
        is_complete = (h_padded == 0 and c_padded == 0)
        
        nmredata_content += f">  <Atom_Peak_Counts>\nH_atoms: {h_atoms}\n"
        nmredata_content += f"H_peaks_total: {h_peaks_count}\n"
        nmredata_content += f"H_peaks_real: {h_real_peaks}\n"
        nmredata_content += f"H_peaks_padded: {h_padded}\n"
        nmredata_content += f"C_atoms: {c_atoms}\n"
        nmredata_content += f"C_peaks_total: {c_peaks_count}\n"
        nmredata_content += f"C_peaks_real: {c_real_peaks}\n"
        nmredata_content += f"C_peaks_padded: {c_padded}\n\n"
        
        # Add explicit data completeness label
        nmredata_content += f">  <DATA_COMPLETENESS>\n"
        if is_complete:
            nmredata_content += "COMPLETE - No padding required\n"
            nmredata_content += "Suitable for: TEST SET\n"
        else:
            nmredata_content += "INCOMPLETE - Padding added\n"
            nmredata_content += "Suitable for: TRAINING SET\n"
        nmredata_content += f"Total_peaks_padded: {h_padded + c_padded}\n\n"
        
        # Additional metadata
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

>  <Natural_Products_Name>
{row['Natural_Products_Name']}

$$$$
"""
        
        return nmredata_content
    
    def _format_peak_line(self, peak):
        """
        Format a single peak line for NMReDATA
        Handles both real peaks and padded (null) peaks
        """
        # Handle padded peaks with null shifts
        if peak['shift'] is None:
            # Format: NULL, null, , -1
            # -1 indicates unknown atom assignment (not a false mapping)
            return f"NULL, null, , -1"
        
        # Normal peak formatting
        line = f"{peak['shift']}, {peak['multiplicity']}"
        
        # Add coupling constants
        if peak.get('coupling'):
            coupling_str = ', '.join([f"J={c}" for c in peak['coupling']])
            line += f", {coupling_str}"
        
        # Add atom number (no count needed since each peak = 1 atom)
        line += f", {peak['atom_number']}"
        
        return line