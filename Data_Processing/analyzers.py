"""
Analysis functions for NMR data and 3D structures
"""

import os
import glob
import pandas as pd
import numpy as np
import random
import logging
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from utils import setup_logging
    logger = setup_logging()
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def pre_check_nmr_availability(csv_file, txt_directory):
    """Standalone function to pre-check NMR data availability"""
    logger.info("Running NMR data availability pre-check...")
    
    # Build txt file cache
    txt_files = glob.glob(os.path.join(txt_directory, "*.txt"))
    txt_cache = {}
    for file_path in txt_files:
        filename = os.path.basename(file_path).lower()
        txt_cache[filename] = file_path
        filename_no_ext = os.path.splitext(filename)[0]
        txt_cache[filename_no_ext] = file_path
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Check each compound
    with_nmr = []
    without_nmr = []
    
    for idx, row in df.iterrows():
        np_id = row['NP_MRD_ID'].lower()
        found = False
        
        for key in txt_cache:
            if np_id in key:
                found = True
                break
        
        if found:
            with_nmr.append(row['NP_MRD_ID'])
        else:
            without_nmr.append(row['NP_MRD_ID'])
    
    # Generate report
    logger.info(f"\nNMR Data Availability Report:")
    logger.info(f"Total compounds: {len(df)}")
    logger.info(f"With NMR data: {len(with_nmr)} ({len(with_nmr)/len(df)*100:.1f}%)")
    logger.info(f"Without NMR data: {len(without_nmr)} ({len(without_nmr)/len(df)*100:.1f}%)")
    
    # Save lists
    with open('compounds_with_nmr.txt', 'w') as f:
        for np_id in with_nmr:
            f.write(f"{np_id}\n")
    
    with open('compounds_without_nmr.txt', 'w') as f:
        for np_id in without_nmr:
            f.write(f"{np_id}\n")
    
    logger.info("\nLists saved to compounds_with_nmr.txt and compounds_without_nmr.txt")
    
    return with_nmr, without_nmr


def analyze_3d_generation_quality(output_directory, sample_size=100):
    """Analyze the quality of 3D structures generated"""
    nmredata_files = glob.glob(os.path.join(output_directory, "*.nmredata"))
    
    if len(nmredata_files) == 0:
        logger.warning("No NMReDATA files found for analysis")
        return
    
    # Sample files
    sample_files = random.sample(nmredata_files, min(sample_size, len(nmredata_files)))
    
    quality_metrics = {
        'total_analyzed': 0,
        'valid_structures': 0,
        'mmff_optimized': 0,
        'uff_optimized': 0,
        'unusual_bond_lengths': 0,
        'energy_values': [],
        'atom_counts': [],
        'bond_length_stats': []
    }
    
    logger.info(f"Analyzing 3D structure quality for {len(sample_files)} files...")
    
    for file_path in sample_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract MOL block
            mol_block_end = content.find('>  <')
            if mol_block_end > 0:
                mol_block = content[:mol_block_end]
                
                # Parse with RDKit
                mol = Chem.MolFromMolBlock(mol_block)
                if mol and mol.GetNumConformers() > 0:
                    quality_metrics['total_analyzed'] += 1
                    quality_metrics['valid_structures'] += 1
                    quality_metrics['atom_counts'].append(mol.GetNumAtoms())
                    
                    # Check bond lengths
                    conf = mol.GetConformer()
                    unusual_bonds = 0
                    bond_lengths = []
                    
                    for bond in mol.GetBonds():
                        idx1 = bond.GetBeginAtomIdx()
                        idx2 = bond.GetEndAtomIdx()
                        dist = AllChem.GetBondLength(conf, idx1, idx2)
                        bond_lengths.append(dist)
                        
                        if dist < 0.5 or dist > 3.0:
                            unusual_bonds += 1
                    
                    if unusual_bonds > 0:
                        quality_metrics['unusual_bond_lengths'] += 1
                    
                    if bond_lengths:
                        quality_metrics['bond_length_stats'].extend(bond_lengths)
                    
                    # Try to calculate energy
                    try:
                        if AllChem.MMFFHasAllMoleculeParams(mol):
                            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
                            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                            if ff:
                                energy = ff.CalcEnergy()
                                quality_metrics['energy_values'].append(energy)
                                quality_metrics['mmff_optimized'] += 1
                        else:
                            quality_metrics['uff_optimized'] += 1
                    except:
                        pass
                        
        except Exception as e:
            logger.debug(f"Error analyzing {file_path}: {e}")
    
    # Generate report
    if quality_metrics['total_analyzed'] > 0:
        avg_energy = np.mean(quality_metrics['energy_values']) if quality_metrics['energy_values'] else 0
        std_energy = np.std(quality_metrics['energy_values']) if quality_metrics['energy_values'] else 0
        
        avg_atoms = np.mean(quality_metrics['atom_counts']) if quality_metrics['atom_counts'] else 0
        avg_bond_length = np.mean(quality_metrics['bond_length_stats']) if quality_metrics['bond_length_stats'] else 0
        std_bond_length = np.std(quality_metrics['bond_length_stats']) if quality_metrics['bond_length_stats'] else 0
        
        logger.info(f"\n3D Structure Quality Report:")
        logger.info(f"Files analyzed: {quality_metrics['total_analyzed']}")
        logger.info(f"Valid structures: {quality_metrics['valid_structures']} ({quality_metrics['valid_structures']/quality_metrics['total_analyzed']*100:.1f}%)")
        logger.info(f"MMFF94s optimized: {quality_metrics['mmff_optimized']} ({quality_metrics['mmff_optimized']/quality_metrics['total_analyzed']*100:.1f}%)")
        logger.info(f"UFF optimized: {quality_metrics['uff_optimized']} ({quality_metrics['uff_optimized']/quality_metrics['total_analyzed']*100:.1f}%)")
        logger.info(f"Structures with unusual bonds: {quality_metrics['unusual_bond_lengths']} ({quality_metrics['unusual_bond_lengths']/quality_metrics['total_analyzed']*100:.1f}%)")
        logger.info(f"Average atom count: {avg_atoms:.1f}")
        logger.info(f"Average bond length: {avg_bond_length:.3f} ± {std_bond_length:.3f} Å")
        
        if quality_metrics['energy_values']:
            logger.info(f"Average MMFF94s energy: {avg_energy:.2f} ± {std_energy:.2f} kcal/mol")
            
            # Check for high energy structures
            high_energy_threshold = avg_energy + 2 * std_energy
            high_energy_count = sum(1 for e in quality_metrics['energy_values'] if e > high_energy_threshold)
            if high_energy_count > 0:
                logger.warning(f"Found {high_energy_count} structures with unusually high energy (>{high_energy_threshold:.1f} kcal/mol)")
    
    return quality_metrics


def analyze_consolidation_patterns(output_directory, sample_size=50):
    """Analyze peak consolidation patterns in generated files"""
    nmredata_files = glob.glob(os.path.join(output_directory, "*.nmredata"))
    
    if len(nmredata_files) == 0:
        logger.warning("No NMReDATA files found for analysis")
        return
    
    # Sample files
    sample_files = random.sample(nmredata_files, min(sample_size, len(nmredata_files)))
    
    consolidation_stats = {
        'h_consolidation_ratios': [],
        'c_consolidation_ratios': [],
        'h_multiplicities': {},
        'c_multiplicities': {},
        'h_shift_ranges': [],
        'c_shift_ranges': [],
        'h_coupling_counts': [],
        'c_coupling_counts': []
    }
    
    logger.info(f"Analyzing consolidation patterns for {len(sample_files)} files...")
    
    for file_path in sample_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract 1H NMR data
            h_start = content.find('>  <NMREDATA_1D_1H>')
            h_end = content.find('>  <', h_start + 1) if h_start > 0 else -1
            
            if h_start > 0 and h_end > h_start:
                h_data = content[h_start:h_end].split('\n')[1:-1]  # Skip tag line and empty line
                h_peaks = len([line for line in h_data if line.strip()])
                h_atoms = 0
                
                for line in h_data:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            # Get shift
                            try:
                                shift = float(parts[0].strip())
                                consolidation_stats['h_shift_ranges'].append(shift)
                            except:
                                continue
                            
                            # Get multiplicity
                            mult = parts[1].strip()
                            if mult not in consolidation_stats['h_multiplicities']:
                                consolidation_stats['h_multiplicities'][mult] = 0
                            consolidation_stats['h_multiplicities'][mult] += 1
                            
                            # Count J-couplings
                            j_count = 0
                            for part in parts[2:]:
                                if 'J=' in part:
                                    j_count += 1
                            consolidation_stats['h_coupling_counts'].append(j_count)
                            
                            # Get atom count (last field)
                            try:
                                atom_count = int(parts[-1].strip())
                                h_atoms += atom_count
                            except:
                                pass
                
                if h_atoms > 0 and h_peaks > 0:
                    consolidation_stats['h_consolidation_ratios'].append(h_peaks / h_atoms)
            
            # Extract 13C NMR data
            c_start = content.find('>  <NMREDATA_1D_13C>')
            c_end = content.find('>  <', c_start + 1) if c_start > 0 else -1
            
            if c_start > 0 and c_end > c_start:
                c_data = content[c_start:c_end].split('\n')[1:-1]
                c_peaks = len([line for line in c_data if line.strip()])
                c_atoms = 0
                
                for line in c_data:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            # Get shift
                            try:
                                shift = float(parts[0].strip())
                                consolidation_stats['c_shift_ranges'].append(shift)
                            except:
                                continue
                            
                            # Get multiplicity
                            mult = parts[1].strip()
                            if mult not in consolidation_stats['c_multiplicities']:
                                consolidation_stats['c_multiplicities'][mult] = 0
                            consolidation_stats['c_multiplicities'][mult] += 1
                            
                            # Count J-couplings
                            j_count = 0
                            for part in parts[2:]:
                                if 'J=' in part:
                                    j_count += 1
                            consolidation_stats['c_coupling_counts'].append(j_count)
                            
                            # Get atom count
                            try:
                                atom_count = int(parts[-1].strip())
                                c_atoms += atom_count
                            except:
                                pass
                
                if c_atoms > 0 and c_peaks > 0:
                    consolidation_stats['c_consolidation_ratios'].append(c_peaks / c_atoms)
                            
        except Exception as e:
            logger.debug(f"Error analyzing {file_path}: {e}")
    
    # Generate report
    if consolidation_stats['h_consolidation_ratios'] or consolidation_stats['c_consolidation_ratios']:
        logger.info(f"\nConsolidation Pattern Analysis:")
        
        if consolidation_stats['h_consolidation_ratios']:
            avg_h_ratio = np.mean(consolidation_stats['h_consolidation_ratios'])
            logger.info(f"Average 1H consolidation ratio: {avg_h_ratio:.3f} (peaks/atoms)")
            logger.info(f"  Min ratio: {min(consolidation_stats['h_consolidation_ratios']):.3f}")
            logger.info(f"  Max ratio: {max(consolidation_stats['h_consolidation_ratios']):.3f}")
        
        if consolidation_stats['c_consolidation_ratios']:
            avg_c_ratio = np.mean(consolidation_stats['c_consolidation_ratios'])
            logger.info(f"Average 13C consolidation ratio: {avg_c_ratio:.3f} (peaks/atoms)")
            logger.info(f"  Min ratio: {min(consolidation_stats['c_consolidation_ratios']):.3f}")
            logger.info(f"  Max ratio: {max(consolidation_stats['c_consolidation_ratios']):.3f}")
        
        logger.info(f"\n1H multiplicity distribution:")
        for mult, count in sorted(consolidation_stats['h_multiplicities'].items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {mult}: {count}")
        
        logger.info(f"\n13C multiplicity distribution:")
        for mult, count in sorted(consolidation_stats['c_multiplicities'].items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {mult}: {count}")
        
        if consolidation_stats['h_shift_ranges']:
            logger.info(f"\n1H chemical shift range: {min(consolidation_stats['h_shift_ranges']):.2f} - {max(consolidation_stats['h_shift_ranges']):.2f} ppm")
            logger.info(f"  Average: {np.mean(consolidation_stats['h_shift_ranges']):.2f} ppm")
        
        if consolidation_stats['c_shift_ranges']:
            logger.info(f"13C chemical shift range: {min(consolidation_stats['c_shift_ranges']):.2f} - {max(consolidation_stats['c_shift_ranges']):.2f} ppm")
            logger.info(f"  Average: {np.mean(consolidation_stats['c_shift_ranges']):.2f} ppm")
        
        if consolidation_stats['h_coupling_counts']:
            avg_h_j = np.mean(consolidation_stats['h_coupling_counts'])
            logger.info(f"\n1H average J-couplings per peak: {avg_h_j:.1f}")
        
        if consolidation_stats['c_coupling_counts']:
            avg_c_j = np.mean(consolidation_stats['c_coupling_counts'])
            logger.info(f"13C average J-couplings per peak: {avg_c_j:.1f}")
    
    return consolidation_stats


def analyze_nmr_coverage(csv_file, txt_directory, output_directory):
    """Comprehensive analysis of NMR data coverage and conversion success"""
    logger.info("Running comprehensive NMR coverage analysis...")
    
    # Load original CSV
    df_original = pd.read_csv(csv_file)
    total_compounds = len(df_original)
    
    # Get NMR availability
    with_nmr, without_nmr = pre_check_nmr_availability(csv_file, txt_directory)
    
    # Check output files
    output_files = glob.glob(os.path.join(output_directory, "*.nmredata"))
    successful_ids = set()
    
    for file_path in output_files:
        filename = os.path.basename(file_path)
        # Extract NP_ID from filename (format: NPID_name.nmredata)
        np_id = filename.split('_')[0]
        successful_ids.add(np_id)
    
    # Calculate statistics
    nmr_available = len(with_nmr)
    nmr_missing = len(without_nmr)
    converted_successfully = len(successful_ids)
    
    # Find compounds that had NMR but failed conversion
    failed_conversion = []
    for np_id in with_nmr:
        if np_id not in successful_ids:
            failed_conversion.append(np_id)
    
    # Generate detailed report
    coverage_pct = (nmr_available / total_compounds) * 100
    conversion_pct = (converted_successfully / nmr_available * 100) if nmr_available > 0 else 0
    overall_success_pct = (converted_successfully / total_compounds) * 100
    
    report = f"""
================================================================================
                        NMR COVERAGE AND CONVERSION ANALYSIS
================================================================================

OVERALL STATISTICS:
------------------
Total compounds in dataset:        {total_compounds}
Compounds with NMR data:           {nmr_available} ({coverage_pct:.1f}%)
Compounds without NMR data:        {nmr_missing} ({100-coverage_pct:.1f}%)
Successfully converted:            {converted_successfully} ({overall_success_pct:.1f}% of total)
Conversion success rate:           {conversion_pct:.1f}% (of compounds with NMR)
Failed conversions:                {len(failed_conversion)}

CONVERSION PIPELINE:
-------------------
Input CSV:                         {total_compounds} compounds
    ↓
After NMR pre-check:              {nmr_available} compounds (-{nmr_missing})
    ↓
After conversion:                 {converted_successfully} compounds (-{len(failed_conversion)})

SUCCESS METRICS:
---------------
NMR data availability:            {coverage_pct:.1f}%
Conversion reliability:           {conversion_pct:.1f}%
End-to-end success:              {overall_success_pct:.1f}%

"""
    
    if failed_conversion:
        report += f"""FAILED CONVERSIONS:
------------------
The following {len(failed_conversion)} compounds had NMR data but failed to convert:
"""
        for i, np_id in enumerate(failed_conversion[:20]):  # Show first 20
            report += f"  - {np_id}\n"
        
        if len(failed_conversion) > 20:
            report += f"  ... and {len(failed_conversion) - 20} more\n"
    
    # Save report
    report_path = os.path.join(output_directory, "nmr_coverage_analysis.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save failed conversion list
    if failed_conversion:
        failed_path = os.path.join(output_directory, "failed_conversions.txt")
        with open(failed_path, 'w') as f:
            for np_id in failed_conversion:
                f.write(f"{np_id}\n")
        logger.info(f"Failed conversion list saved to: {failed_path}")
    
    logger.info(report)
    logger.info(f"Coverage analysis saved to: {report_path}")
    
    return {
        'total_compounds': total_compounds,
        'nmr_available': nmr_available,
        'nmr_missing': nmr_missing,
        'converted_successfully': converted_successfully,
        'failed_conversion': failed_conversion,
        'coverage_pct': coverage_pct,
        'conversion_pct': conversion_pct,
        'overall_success_pct': overall_success_pct
    }


def compare_nmr_formats(original_txt_file, nmredata_file):
    """Compare original NMR data with converted NMReDATA format"""
    logger.info(f"Comparing formats:")
    logger.info(f"  Original: {original_txt_file}")
    logger.info(f"  Converted: {nmredata_file}")
    
    comparison = {
        'original_peaks': 0,
        'converted_peaks': 0,
        'consolidation_ratio': 0,
        'atoms_preserved': True,
        'shifts_preserved': True
    }
    
    # Parse original txt file
    original_peaks = []
    try:
        with open(original_txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            element = parts[0].strip()
                            atom_num = int(parts[1].strip())
                            shift = float(parts[2].strip())
                            original_peaks.append({
                                'element': element,
                                'atom': atom_num,
                                'shift': shift
                            })
                        except:
                            pass
    except Exception as e:
        logger.error(f"Error reading original file: {e}")
        return comparison
    
    comparison['original_peaks'] = len(original_peaks)
    
    # Parse NMReDATA file
    try:
        with open(nmredata_file, 'r') as f:
            content = f.read()
        
        # Extract 1H and 13C data
        converted_peaks = []
        
        # Parse 1H
        h_start = content.find('>  <NMREDATA_1D_1H>')
        h_end = content.find('>  <', h_start + 1) if h_start > 0 else -1
        
        if h_start > 0 and h_end > h_start:
            h_data = content[h_start:h_end].split('\n')[1:-1]
            for line in h_data:
                if line.strip():
                    parts = line.split(',')
                    if parts:
                        try:
                            shift = float(parts[0].strip())
                            converted_peaks.append(('H', shift))
                        except:
                            pass
        
        # Parse 13C
        c_start = content.find('>  <NMREDATA_1D_13C>')
        c_end = content.find('>  <', c_start + 1) if c_start > 0 else -1
        
        if c_start > 0 and c_end > c_start:
            c_data = content[c_start:c_end].split('\n')[1:-1]
            for line in c_data:
                if line.strip():
                    parts = line.split(',')
                    if parts:
                        try:
                            shift = float(parts[0].strip())
                            converted_peaks.append(('C', shift))
                        except:
                            pass
        
        comparison['converted_peaks'] = len(converted_peaks)
        
    except Exception as e:
        logger.error(f"Error reading NMReDATA file: {e}")
        return comparison
    
    # Calculate consolidation ratio
    if comparison['original_peaks'] > 0:
        comparison['consolidation_ratio'] = comparison['converted_peaks'] / comparison['original_peaks']
    
    logger.info(f"  Original peaks: {comparison['original_peaks']}")
    logger.info(f"  Converted peaks: {comparison['converted_peaks']}")
    logger.info(f"  Consolidation ratio: {comparison['consolidation_ratio']:.2f}")
    
    return comparison