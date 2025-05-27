"""
Core converter class for NMReDATA conversion
"""

import pandas as pd
import os
import glob
import time
import signal
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging

from Data_Processing.molecule_processor import MoleculeProcessor
from Data_Processing.nmr_parser import NMRParser
from Data_Processing.nmredata_writer import NMReDataWriter
from Data_Processing.utils import setup_logging

logger = setup_logging()


class CSVToNMReDATA:
    """Main converter class for processing molecular data to NMReDATA format"""
    
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
        
        # Initialize components
        self.molecule_processor = MoleculeProcessor()
        self.nmr_parser = NMRParser()
        self.nmredata_writer = NMReDataWriter()
        
        # Thread-safe counters and metrics
        self.lock = Lock()
        self.successful = 0
        self.failed = 0
        self.start_time = None
        
        # Detailed accuracy metrics
        self.metrics = {
            'total_compounds': 0,
            'compounds_with_nmr': 0,
            'compounds_without_nmr': 0,
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
            'empty_mol_blocks': 0,
            'processing_errors': [],
            'memory_issues': 0,
            'timeout_issues': 0
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Check system resources
        self._check_system_resources()
        
        # Pre-cache all txt files for faster lookup
        self.txt_files_cache = self._build_txt_file_cache()
        logger.info(f"Cached {len(self.txt_files_cache)} txt files")
        
        # Load and validate CSV data with NMR pre-check
        self.df = self._load_and_precheck_data(csv_file)
        
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
        
        return None
    
    def _load_and_precheck_data(self, csv_file):
        """Load CSV and pre-check for NMR data availability"""
        try:
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_cols = ['Natural_Products_Name', 'NP_MRD_ID', 'SMILES']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Remove duplicates
            original_len = len(df)
            df = df.drop_duplicates(subset=['NP_MRD_ID'])
            if len(df) < original_len:
                logger.warning(f"Removed {original_len - len(df)} duplicate entries")
            
            # Remove rows with empty SMILES
            df = df.dropna(subset=['SMILES'])
            df = df[df['SMILES'].str.strip() != '']
            
            logger.info(f"Loaded {len(df)} compounds from CSV")
            
            # Pre-check for NMR data files
            logger.info("Pre-checking for NMR data availability...")
            compounds_with_nmr = []
            compounds_without_nmr = []
            
            for idx, row in df.iterrows():
                np_id = row['NP_MRD_ID']
                if self.find_txt_file(np_id):
                    compounds_with_nmr.append(idx)
                else:
                    compounds_without_nmr.append(idx)
                    logger.debug(f"No NMR data found for {np_id}")
            
            # Update metrics
            self.metrics['compounds_with_nmr'] = len(compounds_with_nmr)
            self.metrics['compounds_without_nmr'] = len(compounds_without_nmr)
            
            # Log summary
            logger.info(f"Pre-check complete:")
            logger.info(f"  - Compounds with NMR data: {len(compounds_with_nmr)}")
            logger.info(f"  - Compounds without NMR data: {len(compounds_without_nmr)}")
            
            # Save list of compounds without NMR data
            if compounds_without_nmr:
                missing_nmr_file = os.path.join(self.output_directory, 'compounds_without_nmr.txt')
                with open(missing_nmr_file, 'w', encoding='utf-8') as f:
                    f.write("NP_MRD_ID,Natural_Products_Name\n")
                    for idx in compounds_without_nmr:
                        row = df.iloc[idx]
                        f.write(f"{row['NP_MRD_ID']},{row['Natural_Products_Name']}\n")
                logger.info(f"List of compounds without NMR data saved to: {missing_nmr_file}")
            
            # Filter to only compounds with NMR data
            df_with_nmr = df.iloc[compounds_with_nmr].copy()
            self.metrics['total_compounds'] = len(df_with_nmr)
            
            logger.info(f"Processing {len(df_with_nmr)} compounds with NMR data")
            
            return df_with_nmr
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def process_compound(self, row):
        """Process a single compound - thread-safe method"""
        compound_metrics = {
            'txt_found': False,
            'valid_smiles': False,
            '3d_generated': False,
            'peaks_found': False,
            'atom_mapped': False,
            'consolidation_ok': True,
            'mol_block_valid': False
        }
        
        try:
            np_id = row['NP_MRD_ID']
            logger.info(f"Processing {row['Natural_Products_Name']} (ID: {np_id})")
            
            # Find txt file (should exist due to pre-check)
            txt_file = self.find_txt_file(np_id)
            if not txt_file:
                logger.error(f"No txt file found for {np_id} (should have been filtered)")
                with self.lock:
                    self.metrics['txt_files_missing'] += 1
                return False, f"No txt file found for {np_id}", compound_metrics
            
            compound_metrics['txt_found'] = True
            with self.lock:
                self.metrics['txt_files_found'] += 1
            
            # Parse peaklist
            peaks = self.nmr_parser.parse_peaklist(txt_file)
            if not peaks:
                logger.warning(f"No peaks found for {np_id}")
                with self.lock:
                    self.metrics['empty_peaklists'] += 1
            else:
                compound_metrics['peaks_found'] = True
                with self.lock:
                    self.metrics['peaks_parsed'] += 1
            
            # Generate molecule from SMILES
            mol = self.molecule_processor.smiles_to_mol(row['SMILES'])
            if mol is None:
                logger.error(f"Invalid SMILES for {np_id}: {row['SMILES']}")
                with self.lock:
                    self.metrics['invalid_smiles'] += 1
                return False, f"Invalid SMILES for {np_id}", compound_metrics
            
            compound_metrics['valid_smiles'] = True
            with self.lock:
                self.metrics['valid_smiles'] += 1
            
            # Generate 3D coordinates with robust method
            mol_3d = self.molecule_processor.generate_3d_coordinates(mol)
            if mol_3d is None:
                with self.lock:
                    self.metrics['3d_generation_failed'] += 1
                compound_metrics['3d_generated'] = False
                logger.error(f"Failed to generate 3D coordinates for {np_id}")
                return False, f"3D generation failed for {np_id}", compound_metrics
            else:
                compound_metrics['3d_generated'] = True
                with self.lock:
                    self.metrics['3d_generation_success'] += 1
            
            # Create atom mapping
            if mol_3d and peaks:
                atom_mapping = self.molecule_processor.create_atom_mapping(peaks, mol_3d)
                if atom_mapping:
                    compound_metrics['atom_mapped'] = True
                    with self.lock:
                        self.metrics['atom_mapping_success'] += 1
                    # Renumber peaks to match structure
                    peaks = self.nmr_parser.renumber_peaklist(peaks, atom_mapping)
                else:
                    with self.lock:
                        self.metrics['atom_mapping_failed'] += 1
            
            # Consolidate peaks
            h_peaks = [p for p in peaks if p['element'] == 'H']
            c_peaks = [p for p in peaks if p['element'] == 'C']
            
            h_peaks_consolidated = self.nmr_parser.consolidate_equivalent_peaks(h_peaks)
            c_peaks_consolidated = self.nmr_parser.consolidate_equivalent_peaks(c_peaks)
            
            # Check for consolidation warnings
            if not self.nmr_parser.validate_peak_consolidation(h_peaks, h_peaks_consolidated):
                with self.lock:
                    self.metrics['consolidation_warnings'] += 1
                compound_metrics['consolidation_ok'] = False
            
            if not self.nmr_parser.validate_peak_consolidation(c_peaks, c_peaks_consolidated):
                with self.lock:
                    self.metrics['consolidation_warnings'] += 1
                compound_metrics['consolidation_ok'] = False
            
            # Create NMReDATA content
            nmredata_content = self.nmredata_writer.create_nmredata_file(
                row, h_peaks_consolidated, c_peaks_consolidated, mol_3d
            )
            
            # Check if content is valid (not None and has proper MOL block)
            if nmredata_content is None:
                logger.error(f"Failed to create valid NMReDATA for {np_id} - empty MOL block")
                with self.lock:
                    self.metrics['empty_mol_blocks'] += 1
                return False, f"Empty MOL block for {np_id}", compound_metrics
            
            compound_metrics['mol_block_valid'] = True
            
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
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                processed_ids = set(line.strip() for line in f)
            logger.info(f"Resuming from checkpoint: {len(processed_ids)} already processed")
        
        # Filter out already processed compounds
        df_to_process = self.df[~self.df['NP_MRD_ID'].isin(processed_ids)]
        logger.info(f"Processing {len(df_to_process)} remaining compounds")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks in batches to avoid memory issues
            batch_size = 1000
            
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
                                with open(checkpoint_file, 'a', encoding='utf-8') as f:
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
    
    def generate_accuracy_report(self):
        """Generate comprehensive accuracy report"""
        from Data_Processing.report_generator import ReportGenerator
        report_gen = ReportGenerator(self.metrics, self.successful, self.failed, self.output_directory)
        return report_gen.generate_report()