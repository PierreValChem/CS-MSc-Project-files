#!/usr/bin/env python3
"""
NMReDATA Converter - Main Entry Point
Converts CSV molecular data with NMR peak lists to NMReDATA format
"""

import sys
import logging
import traceback
from Data_Processing.converter import CSVToNMReDATA
from Data_Processing.utils import verify_installation, setup_logging
from Data_Processing.molecule_processor import MoleculeProcessor
from Data_Processing.validators import batch_validate, clean_invalid_files
from Data_Processing.analyzers import pre_check_nmr_availability, analyze_3d_generation_quality

# Set up logging
logger = setup_logging()


def main():
    """Main function to run the converter with error handling"""
    try:
        # Configuration
        csv_file = "NP-ID and structure NP0100001-NP0150000.csv"
        txt_directory = "NP-MRD_nmr_peak_lists_NP0100001_NP0150000/NP-MRD_nmr_peak_lists_NP0100001_NP0150000/"
        output_directory = "CSV_to_NMRe_output_v3/"
        
        # Optimal settings for i7-12800H (14 cores: 6P + 8E) with 32GB RAM
        max_workers = 10
        
        # Check if running in batch mode
        if len(sys.argv) > 1:
            if sys.argv[1] == '--batch':
                # Process in smaller batches
                batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
                logger.info(f"Running in batch mode with batch size {batch_size}")
                
                # Initialize converter to get pre-filtered data
                converter = CSVToNMReDATA(csv_file, txt_directory, output_directory, max_workers)
                df_filtered = converter.df  # Already filtered for compounds with NMR
                
                total_batches = (len(df_filtered) + batch_size - 1) // batch_size
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min((batch_num + 1) * batch_size, len(df_filtered))
                    
                    logger.info(f"Processing batch {batch_num + 1}/{total_batches} (rows {start_idx}-{end_idx})")
                    
                    # Create temporary CSV for this batch
                    batch_csv = f"batch_{batch_num + 1}.csv"
                    df_filtered.iloc[start_idx:end_idx].to_csv(batch_csv, index=False)
                    
                    # Process batch
                    batch_converter = CSVToNMReDATA(batch_csv, txt_directory, output_directory, max_workers)
                    batch_converter.process_all_compounds()
                    
                    # Clean up
                    import os
                    os.remove(batch_csv)
                    
                    # Pause between batches
                    if batch_num < total_batches - 1:
                        logger.info("Pausing for 30 seconds between batches...")
                        import time
                        time.sleep(30)
            else:
                # Normal processing
                converter = CSVToNMReDATA(csv_file, txt_directory, output_directory, max_workers)
                converter.process_all_compounds()
        else:
            # Normal processing
            converter = CSVToNMReDATA(csv_file, txt_directory, output_directory, max_workers)
            converter.process_all_compounds()
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Verify installation first
    if not verify_installation():
        logger.error("Installation verification failed. Please check dependencies.")
        sys.exit(1)
    
    # Parse command line arguments
    if '--precheck' in sys.argv:
        # Pre-check NMR availability only
        csv_file = "NP-ID and structure NP0100001-NP0150000.csv"
        txt_directory = "NP-MRD_nmr_peak_lists_NP0100001_NP0150000/NP-MRD_nmr_peak_lists_NP0100001_NP0150000/"
        pre_check_nmr_availability(csv_file, txt_directory)
        sys.exit(0)
    
    elif '--clean' in sys.argv:
        # Clean invalid files
        output_directory = sys.argv[sys.argv.index('--clean') + 1] if len(sys.argv) > sys.argv.index('--clean') + 1 else "CSV_to_NMRe_output_v3/"
        clean_invalid_files(output_directory)
        sys.exit(0)
    
    elif '--validate' in sys.argv:
        # Validate existing files
        output_directory = sys.argv[sys.argv.index('--validate') + 1] if len(sys.argv) > sys.argv.index('--validate') + 1 else "CSV_to_NMRe_output_v3/"
        batch_validate(output_directory)
        sys.exit(0)
    
    elif '--analyze3d' in sys.argv:
        # Analyze 3D quality
        output_directory = sys.argv[sys.argv.index('--analyze3d') + 1] if len(sys.argv) > sys.argv.index('--analyze3d') + 1 else "CSV_to_NMRe_output_v3/"
        analyze_3d_generation_quality(output_directory)
        sys.exit(0)
    
    else:
        # Run the main conversion
        main()