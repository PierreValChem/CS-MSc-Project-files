#!/usr/bin/env python3
"""
Core IntegratedBatchProcessor class
Handles the actual processing of NMR data files with padding logic
"""

import os
import re
import logging
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import pandas as pd
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from collections import defaultdict

# Import Data_Processing modules
from molecule_processor import MoleculeProcessor
from nmredata_writer import NMReDataWriter
from utils import setup_logging
from report_generator import ReportGenerator
from enhanced_nmr_parser import EnhancedNMRParser

logger = setup_logging()


class IntegratedBatchProcessor:
    """Process multiple CSV/folder pairs with enhanced file selection and padding"""
    
    def __init__(self, base_directory, output_directory, max_workers=10):
        print(f"Initializing IntegratedBatchProcessor...")
        print(f"Base directory: {base_directory}")
        print(f"Output directory: {output_directory}")
        
        self.base_directory = Path(base_directory)
        self.output_directory = Path(output_directory)
        self.max_workers = max_workers
        
        # Initialize components
        self.molecule_processor = MoleculeProcessor()
        self.nmr_parser = EnhancedNMRParser()
        self.nmredata_writer = NMReDataWriter()
        
        # Thread-safe counters
        self.lock = Lock()
        self.successful = 0
        self.failed = 0
        self.start_time = None
        
        # Metrics
        self.metrics = {
            'total_compounds': 0,
            'txt_files_found': 0,
            'txt_files_missing': 0,
            'multiple_files_found': 0,
            'files_merged': 0,
            'files_skipped_excess_peaks': 0,
            'h_peaks_padded': 0,
            'c_peaks_padded': 0,
            'valid_smiles': 0,
            'invalid_smiles': 0,
            '3d_generation_success': 0,
            '3d_generation_failed': 0,
            'peaks_parsed': 0,
            'empty_peaklists': 0,
            'nmredata_created': 0,
            'processing_errors': [],
            # Padding distribution tracking
            'h_padding_distribution': defaultdict(int),  # {0: count, 1: count, ...}
            'c_padding_distribution': defaultdict(int),  # {0: count, 1: count, ...}
            'compounds_no_padding': 0,  # Perfect matches
            'compounds_with_padding': 0,  # Had some padding
            'complete_compounds': [],  # List of (np_id, filename) for complete data
            'incomplete_compounds': []  # List of (np_id, filename, h_padded, c_padded)
        }
        
        self.output_directory.mkdir(parents=True, exist_ok=True)
        print(f"IntegratedBatchProcessor initialized")
    
    def find_csv_folder_pairs(self, csv_only=False):
        """Find CSV/Excel files and matching folders"""
        pairs = []
        
        # Find data files
        csv_files = list(self.base_directory.glob("*.csv"))
        excel_files = [] if csv_only else list(self.base_directory.glob("*.xlsx"))
        all_files = csv_files + excel_files
        
        logger.info(f"Found {len(csv_files)} CSV files and {len(excel_files)} Excel files")
        print(f"Found {len(csv_files)} CSV files and {len(excel_files)} Excel files")
        
        if not all_files:
            logger.warning("No CSV or Excel files found in the base directory!")
            print("No CSV or Excel files found!")
            return pairs
        
        for data_file in all_files:
            # Extract range from filename
            match = re.search(r'NP(\d+)[_-]NP(\d+)', data_file.stem)
            
            if match:
                start_num = match.group(1)
                end_num = match.group(2)
                
                # Look for matching folder
                target_pattern = f"np{start_num}_np{end_num}".lower()
                
                nmr_folder = None
                
                # Search for folder containing the NP range
                for item in self.base_directory.iterdir():
                    if item.is_dir():
                        if target_pattern in item.name.lower():
                            nmr_folder = item
                            break
                
                if nmr_folder:
                    pairs.append((data_file, nmr_folder))
                    logger.info(f"Matched: {data_file.name} -> {nmr_folder.name}")
                    print(f"✓ Matched: {data_file.name} -> {nmr_folder.name}")
                else:
                    logger.warning(f"No folder found for {data_file.name}")
                    print(f"✗ No folder found for {data_file.name}")
        
        logger.info(f"Found {len(pairs)} matching CSV/folder pairs")
        print(f"\nFound {len(pairs)} matching CSV/folder pairs")
        return pairs
    
    def process_all_pairs(self, csv_only=False):
        """Process all pairs with multithreading"""
        pairs = self.find_csv_folder_pairs(csv_only=csv_only)
        
        if not pairs:
            logger.error("No matching CSV/folder pairs found!")
            print("\n✗ No matching pairs found!")
            self.generate_report()
            return
        
        logger.info(f"\nProcessing {len(pairs)} CSV/folder pairs")
        print(f"\nProcessing {len(pairs)} CSV/folder pairs")
        self.start_time = time.time()
        
        for idx, (data_file, nmr_folder) in enumerate(pairs, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing pair {idx}/{len(pairs)}")
            logger.info(f"Data file: {data_file.name}")
            logger.info(f"NMR folder: {nmr_folder.name}")
            
            print(f"\n{'='*70}")
            print(f"Processing pair {idx}/{len(pairs)}")
            print(f"Data file: {data_file.name}")
            print(f"NMR folder: {nmr_folder.name}")
            
            try:
                # Read data
                if data_file.suffix == '.csv':
                    df = pd.read_csv(data_file)
                else:
                    df = pd.read_excel(data_file)
                
                # Check columns
                required = ['Natural_Products_Name', 'NP_MRD_ID', 'SMILES']
                if not all(col in df.columns for col in required):
                    logger.error(f"Missing required columns in {data_file.name}")
                    print(f"✗ Missing required columns in {data_file.name}")
                    continue
                
                self.metrics['total_compounds'] += len(df)
                print(f"Processing {len(df)} compounds...")
                
                # Process compounds with multithreading
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    
                    for idx, row in df.iterrows():
                        future = executor.submit(self.process_single_compound, row, nmr_folder)
                        futures.append((future, row['NP_MRD_ID']))
                    
                    for future, np_id in futures:
                        try:
                            success, result = future.result(timeout=300)  # 5 min timeout
                            
                            with self.lock:
                                if success:
                                    self.successful += 1
                                else:
                                    self.failed += 1
                                
                                # Progress update
                                total_processed = self.successful + self.failed
                                if total_processed % 100 == 0:
                                    elapsed = time.time() - self.start_time
                                    rate = total_processed / elapsed if elapsed > 0 else 0
                                    remaining = self.metrics['total_compounds'] - total_processed
                                    eta = remaining / rate if rate > 0 else 0
                                    
                                    progress_msg = f"Progress: {total_processed}/{self.metrics['total_compounds']} " \
                                                 f"({total_processed/self.metrics['total_compounds']*100:.1f}%) " \
                                                 f"Rate: {rate:.2f}/s ETA: {eta/60:.1f} min"
                                    logger.info(progress_msg)
                                    print(progress_msg)
                        
                        except Exception as e:
                            logger.error(f"Error processing {np_id}: {e}")
                            with self.lock:
                                self.failed += 1
                
            except Exception as e:
                logger.error(f"Error processing pair: {e}")
                logger.error(traceback.format_exc())
                print(f"✗ Error processing pair: {e}")
        
        # Generate final report
        self.generate_report()
    
    def process_single_compound(self, row, txt_directory):
        """Process single compound with enhanced file selection and padding"""
        try:
            np_id = row['NP_MRD_ID']
            logger.info(f"Processing {row['Natural_Products_Name']} (ID: {np_id})")
            
            # Generate molecule first to check atom counts
            mol = self.molecule_processor.smiles_to_mol(row['SMILES'])
            if mol is None:
                with self.lock:
                    self.metrics['invalid_smiles'] += 1
                return False, "Invalid SMILES"
            
            with self.lock:
                self.metrics['valid_smiles'] += 1
            
            # Generate 3D
            mol_3d = self.molecule_processor.generate_3d_coordinates(mol)
            if mol_3d is None:
                with self.lock:
                    self.metrics['3d_generation_failed'] += 1
                return False, "3D generation failed"
            
            with self.lock:
                self.metrics['3d_generation_success'] += 1
            
            # Find all txt files for this ID
            txt_files = self.nmr_parser.find_all_files_for_id(np_id, txt_directory)
            
            if not txt_files:
                with self.lock:
                    self.metrics['txt_files_missing'] += 1
                return False, "No txt files found"
            
            if len(txt_files) > 1:
                with self.lock:
                    self.metrics['multiple_files_found'] += 1
            
            # Parse all files and filter valid ones
            files_with_peaks = []
            
            for txt_file in txt_files:
                peaks = self.nmr_parser.parse_peaklist(txt_file)
                if peaks:
                    files_with_peaks.append((txt_file, peaks))
            
            if not files_with_peaks:
                logger.warning(f"No valid NMR data found for {np_id}")
                with self.lock:
                    self.metrics['empty_peaklists'] += 1
                return False, "No valid NMR data"
            
            with self.lock:
                self.metrics['txt_files_found'] += 1
            
            # Select best file or merge H/C files (now considering atom counts)
            if len(files_with_peaks) == 1:
                selected_file, peaks = files_with_peaks[0]
            else:
                selected_file, peaks = self.nmr_parser.merge_h_and_c_files(files_with_peaks, mol_3d)
                
                if selected_file and len(files_with_peaks) > 1:
                    with self.lock:
                        self.metrics['files_merged'] += 1
            
            if not peaks:
                return False, "No peaks after selection"
            
            with self.lock:
                self.metrics['peaks_parsed'] += 1
            
            # Validate and pad peaks for H and C separately
            h_peaks_raw = [p for p in peaks if p['element'] == 'H']
            c_peaks_raw = [p for p in peaks if p['element'] == 'C']
            
            # Validate and pad H peaks
            h_peaks, h_valid = self.nmr_parser.validate_and_pad_peaks(peaks, mol_3d, 'H')
            if not h_valid:
                with self.lock:
                    self.metrics['files_skipped_excess_peaks'] += 1
                return False, "Excess H peaks - file skipped"
            
            # Validate and pad C peaks
            c_peaks, c_valid = self.nmr_parser.validate_and_pad_peaks(peaks, mol_3d, 'C')
            if not c_valid:
                with self.lock:
                    self.metrics['files_skipped_excess_peaks'] += 1
                return False, "Excess C peaks - file skipped"
            
            # Create filename early so we can use it for tracking
            safe_name = "".join(c for c in row['Natural_Products_Name'] if c.isalnum() or c in ' -_').strip()
            filename = f"{np_id}_{safe_name.replace(' ', '_')}.nmredata"
            
            # Track padding statistics
            h_atoms = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'H')
            c_atoms = sum(1 for atom in mol_3d.GetAtoms() if atom.GetSymbol() == 'C')
            
            h_padded_count = len(h_peaks) - len(h_peaks_raw)
            c_padded_count = len(c_peaks) - len(c_peaks_raw)
            
            # Update padding distribution
            with self.lock:
                self.metrics['h_padding_distribution'][h_padded_count] += 1
                self.metrics['c_padding_distribution'][c_padded_count] += 1
                
                # Track overall padding statistics
                if h_padded_count == 0 and c_padded_count == 0:
                    self.metrics['compounds_no_padding'] += 1
                    self.metrics['complete_compounds'].append((np_id, filename))
                else:
                    self.metrics['compounds_with_padding'] += 1
                    self.metrics['incomplete_compounds'].append((np_id, filename, h_padded_count, c_padded_count))
                
                # Total padding counts
                self.metrics['h_peaks_padded'] += h_padded_count
                self.metrics['c_peaks_padded'] += c_padded_count
            
            if h_padded_count > 0:
                logger.info(f"Padded {h_padded_count} H peaks for {np_id}")
            
            if c_padded_count > 0:
                logger.info(f"Padded {c_padded_count} C peaks for {np_id}")
            
            # Create enhanced NMReDATA with all molecular representations
            content = self.create_enhanced_nmredata(row, h_peaks, c_peaks, mol_3d)
            if content is None:
                return False, "Content generation failed"
            
            # Save file (using the filename we already created)
            output_path = self.output_directory / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            with self.lock:
                self.metrics['nmredata_created'] += 1
            
            logger.info(f"Created {filename} (H: {len(h_peaks)}/{h_atoms}, C: {len(c_peaks)}/{c_atoms})")
            
            return True, filename
            
        except Exception as e:
            logger.error(f"Error processing {row['NP_MRD_ID']}: {e}")
            logger.error(traceback.format_exc())
            with self.lock:
                self.metrics['processing_errors'].append({
                    'id': row.get('NP_MRD_ID', 'Unknown'),
                    'error': str(e)
                })
            return False, str(e)
    
    def create_enhanced_nmredata(self, row, h_peaks, c_peaks, mol_3d):
        """Create enhanced NMReDATA with all molecular representations including canonical SMILES"""
        return self.nmredata_writer.create_enhanced_nmredata_file(row, h_peaks, c_peaks, mol_3d)
    
    def generate_report(self):
        """Generate processing report with padding histogram"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Calculate percentages
        total = self.metrics['total_compounds']
        success_rate = (self.successful / total * 100) if total > 0 else 0
        
        report = f"""
{'='*70}
PROCESSING COMPLETE
{'='*70}
Total compounds: {total}
Successfully converted: {self.successful} ({success_rate:.1f}%)
Failed: {self.failed}
Processing time: {elapsed/60:.1f} minutes
Average rate: {total/elapsed:.2f} compounds/second

File Selection Statistics:
  - Files with multiple NMR data: {self.metrics['multiple_files_found']}
  - H/C files merged: {self.metrics['files_merged']}
  - Files skipped (excess peaks): {self.metrics['files_skipped_excess_peaks']}

Peak Padding Statistics:
  - Compounds with perfect match (no padding): {self.metrics['compounds_no_padding']} ({self.metrics['compounds_no_padding']/self.successful*100:.1f}%)
  - Compounds requiring padding: {self.metrics['compounds_with_padding']} ({self.metrics['compounds_with_padding']/self.successful*100:.1f}%)
  - Total H peaks padded: {self.metrics['h_peaks_padded']}
  - Total C peaks padded: {self.metrics['c_peaks_padded']}

Processing Statistics:
  - Valid SMILES: {self.metrics['valid_smiles']}
  - Invalid SMILES: {self.metrics['invalid_smiles']}
  - 3D generation success: {self.metrics['3d_generation_success']}
  - 3D generation failed: {self.metrics['3d_generation_failed']}

Output directory: {self.output_directory}
"""
        
        logger.info(report)
        print(report)
        
        # Generate padding histogram
        self.generate_padding_histogram()
        
        # Save detailed report
        report_path = self.output_directory / "processing_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("NMR Processing Report with Padding Analysis\n")
            f.write(f"{'='*70}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"SUMMARY STATISTICS\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Total compounds: {total}\n")
            f.write(f"Successfully converted: {self.successful} ({success_rate:.1f}%)\n")
            f.write(f"Failed: {self.failed}\n")
            f.write(f"Processing time: {elapsed/60:.1f} minutes\n\n")
            
            f.write(f"DATA QUALITY ANALYSIS\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Perfect matches (no padding needed): {self.metrics['compounds_no_padding']} ({self.metrics['compounds_no_padding']/self.successful*100:.1f}%)\n")
            f.write(f"Incomplete data (padding added): {self.metrics['compounds_with_padding']} ({self.metrics['compounds_with_padding']/self.successful*100:.1f}%)\n\n")
            
            # H NMR padding distribution
            f.write(f"H NMR PADDING DISTRIBUTION\n")
            f.write(f"-" * 20 + "\n")
            for peaks_padded in sorted(self.metrics['h_padding_distribution'].keys()):
                count = self.metrics['h_padding_distribution'][peaks_padded]
                percentage = count / self.successful * 100
                f.write(f"{peaks_padded} peaks padded: {count} compounds ({percentage:.1f}%)\n")
            
            f.write(f"\nC NMR PADDING DISTRIBUTION\n")
            f.write(f"-" * 20 + "\n")
            for peaks_padded in sorted(self.metrics['c_padding_distribution'].keys()):
                count = self.metrics['c_padding_distribution'][peaks_padded]
                percentage = count / self.successful * 100
                f.write(f"{peaks_padded} peaks padded: {count} compounds ({percentage:.1f}%)\n")
            
            f.write(f"\nDETAILED METRICS\n")
            f.write(f"-" * 20 + "\n")
            for key, value in self.metrics.items():
                if key not in ['processing_errors', 'h_padding_distribution', 'c_padding_distribution', 'complete_compounds', 'incomplete_compounds']:
                    f.write(f"  {key}: {value}\n")
            
            if self.metrics['processing_errors']:
                f.write(f"\nPROCESSING ERRORS (first 20)\n")
                f.write(f"-" * 20 + "\n")
                for error in self.metrics['processing_errors'][:20]:
                    f.write(f"  - {error['id']}: {error['error']}\n")
            
            # Data quality assessment
            f.write(f"\nDATA QUALITY ASSESSMENT\n")
            f.write(f"-" * 20 + "\n")
            avg_h_padding = self.metrics['h_peaks_padded'] / self.successful if self.successful > 0 else 0
            avg_c_padding = self.metrics['c_peaks_padded'] / self.successful if self.successful > 0 else 0
            
            f.write(f"Average H peaks padded per compound: {avg_h_padding:.2f}\n")
            f.write(f"Average C peaks padded per compound: {avg_c_padding:.2f}\n")
            
            if avg_h_padding < 1 and avg_c_padding < 1:
                f.write("\n✓ EXCELLENT data quality - most compounds have complete NMR data\n")
            elif avg_h_padding < 3 and avg_c_padding < 3:
                f.write("\n✓ GOOD data quality - moderate amount of missing peaks\n")
            else:
                f.write("\n⚠ FAIR data quality - significant amount of missing NMR data\n")
        
        # Create separate lists for complete and incomplete data
        complete_list_path = self.output_directory / "complete_data_compounds.txt"
        incomplete_list_path = self.output_directory / "incomplete_data_compounds.txt"
        
        with open(complete_list_path, 'w', encoding='utf-8') as f:
            f.write("# Compounds with Complete NMR Data (No Padding)\n")
            f.write("# Suitable for TEST SET\n")
            f.write("# Format: NP_MRD_ID, filename\n\n")
            for compound_id, filename in self.metrics.get('complete_compounds', []):
                f.write(f"{compound_id}, {filename}\n")
        
        with open(incomplete_list_path, 'w', encoding='utf-8') as f:
            f.write("# Compounds with Incomplete NMR Data (Padding Added)\n")
            f.write("# Suitable for TRAINING SET\n")
            f.write("# Format: NP_MRD_ID, filename, h_padded, c_padded\n\n")
            for compound_id, filename, h_pad, c_pad in self.metrics.get('incomplete_compounds', []):
                f.write(f"{compound_id}, {filename}, {h_pad}, {c_pad}\n")
        
        logger.info(f"Detailed report saved to: {report_path}")
        logger.info(f"Complete data list saved to: {complete_list_path}")
        logger.info(f"Incomplete data list saved to: {incomplete_list_path}")
        print(f"Detailed report saved to: {report_path}")
        print(f"Complete data list saved to: {complete_list_path}")
        print(f"Incomplete data list saved to: {incomplete_list_path}")
    
    def generate_padding_histogram(self):
        """Generate histogram of padding distribution"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # H NMR padding histogram
            h_data = self.metrics['h_padding_distribution']
            if h_data:
                max_padding = max(max(h_data.keys()), 10)
                x_range = range(0, max_padding + 1)
                h_counts = [h_data.get(i, 0) for i in x_range]
                
                ax1.bar(x_range, h_counts, color='skyblue', edgecolor='navy', alpha=0.7)
                ax1.set_xlabel('Number of H Peaks Padded')
                ax1.set_ylabel('Number of Compounds')
                ax1.set_title('H NMR Padding Distribution')
                ax1.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, count in enumerate(h_counts):
                    if count > 0:
                        ax1.text(i, count + 0.5, str(count), ha='center', va='bottom')
            
            # C NMR padding histogram
            c_data = self.metrics['c_padding_distribution']
            if c_data:
                max_padding = max(max(c_data.keys()), 10)
                x_range = range(0, max_padding + 1)
                c_counts = [c_data.get(i, 0) for i in x_range]
                
                ax2.bar(x_range, c_counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
                ax2.set_xlabel('Number of C Peaks Padded')
                ax2.set_ylabel('Number of Compounds')
                ax2.set_title('13C NMR Padding Distribution')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, count in enumerate(c_counts):
                    if count > 0:
                        ax2.text(i, count + 0.5, str(count), ha='center', va='bottom')
            
            plt.suptitle('NMR Data Completeness Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save histogram
            histogram_path = self.output_directory / "padding_distribution_histogram.png"
            plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Padding distribution histogram saved to: {histogram_path}")
            print(f"Padding distribution histogram saved to: {histogram_path}")
            
        except ImportError:
            logger.warning("matplotlib not installed - skipping histogram generation")
            logger.warning("Install with: pip install matplotlib")
        except Exception as e:
            logger.error(f"Error generating histogram: {e}")