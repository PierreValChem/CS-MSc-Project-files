#!/usr/bin/env python3
"""
Enhanced NMReDATA Batch Processor with Smart File Selection and Padding
Main entry point for processing NMR data with atom-peak correspondence
"""

import os
import sys
import argparse
import logging

# Import the processor class from the module
from integrated_batch_processor import IntegratedBatchProcessor
from utils import setup_logging, verify_installation

# Set up logging
logger = setup_logging()


def main():
    """Main entry point"""
    print("="*70)
    print("Enhanced NMReDATA Batch Processor with Padding Logic")
    print("="*70)
    
    parser = argparse.ArgumentParser(
        description="Process NMR data with smart file selection and atom-peak padding"
    )
    parser.add_argument("base_dir", help="Base directory containing CSV/Excel files and NMR folders")
    parser.add_argument("-o", "--output", help="Output directory", default=None)
    parser.add_argument("-w", "--workers", type=int, default=10, help="Number of worker threads (default: 10)")
    parser.add_argument("--verify", action="store_true", help="Verify installation only")
    parser.add_argument("--csv-only", action="store_true", help="Process only CSV files (skip Excel)")
    
    args = parser.parse_args()
    
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output if args.output else 'Default'}")
    print(f"Worker threads: {args.workers}")
    print("="*70)
    
    if args.verify:
        if verify_installation():
            print("All dependencies are installed correctly!")
        else:
            print("Some dependencies are missing. Please check the log.")
        return
    
    # Set output directory
    output_dir = args.output if args.output else os.path.join(args.base_dir, "outputv6")
    
    # Verify base directory exists
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory '{args.base_dir}' does not exist!")
        sys.exit(1)
    
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*70)
    logger.info("Enhanced NMReDATA Processor with Atom-Peak Padding")
    logger.info("="*70)
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Worker threads: {args.workers}")
    logger.info("Features:")
    logger.info("  - Smart file selection for multiple NMR files")
    logger.info("  - Atom-peak correspondence validation")
    logger.info("  - Automatic padding for missing peaks")
    logger.info("  - Canonical and isomeric SMILES generation")
    logger.info("  - Files with excess peaks are skipped")
    logger.info("Starting processing...")
    
    try:
        # Run processor
        print("\nInitializing processor...")
        processor = IntegratedBatchProcessor(args.base_dir, output_dir, max_workers=args.workers)
        
        print("Starting to process files...")
        processor.process_all_pairs(csv_only=args.csv_only)
        
        print("\nProcessing complete!")
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        logger.info("Processing interrupted by user")
        
    except Exception as e:
        print(f"\nError in processing: {e}")
        logger.error(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()