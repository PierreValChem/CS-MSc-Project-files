#!/usr/bin/env python3
"""
Minimal test script to debug NMR processing
"""

import os
import sys
import glob
from pathlib import Path

print("="*70)
print("NMR PROCESSOR TEST SCRIPT")
print("="*70)

# Get directories from command line or use defaults
if len(sys.argv) >= 3:
    base_dir = sys.argv[1]
    output_dir = sys.argv[2]
else:
    base_dir = r"C:\Users\pierr\Desktop\CS MSc Project files\peaklist"
    output_dir = r"C:\Users\pierr\Desktop\CS MSc Project files\outputv4"

print(f"Base directory: {base_dir}")
print(f"Output directory: {output_dir}")

# Check if directories exist
if not os.path.exists(base_dir):
    print(f"ERROR: Base directory does not exist!")
    sys.exit(1)

print(f"✓ Base directory exists")

# List contents
print(f"\nContents of base directory:")
items = os.listdir(base_dir)
print(f"Total items: {len(items)}")

# Find CSV/Excel files
csv_files = list(Path(base_dir).glob("*.csv"))
excel_files = list(Path(base_dir).glob("*.xlsx"))
print(f"\nFound {len(csv_files)} CSV files")
print(f"Found {len(excel_files)} Excel files")

if csv_files:
    print("\nCSV files:")
    for f in csv_files[:5]:  # Show first 5
        print(f"  - {f.name}")

if excel_files:
    print("\nExcel files:")
    for f in excel_files[:5]:  # Show first 5
        print(f"  - {f.name}")

# Find NMR folders
nmr_folders = []
for item in Path(base_dir).iterdir():
    if item.is_dir() and "nmr_peak_lists" in item.name.lower():
        nmr_folders.append(item)

print(f"\nFound {len(nmr_folders)} NMR folders")
if nmr_folders:
    print("NMR folders:")
    for f in nmr_folders[:5]:  # Show first 5
        print(f"  - {f.name}")

# Try to match pairs
print("\n" + "="*70)
print("ATTEMPTING TO MATCH FILES WITH FOLDERS")
print("="*70)

import re

matched_pairs = 0
for data_file in csv_files + excel_files:
    # Extract range from filename
    match = re.search(r'NP(\d+)_NP(\d+)', data_file.stem)
    if match:
        start_num = match.group(1)
        end_num = match.group(2)
        print(f"\nFile: {data_file.name}")
        print(f"  Range: NP{start_num} to NP{end_num}")
        
        # Look for matching folder
        folder_found = False
        for folder in nmr_folders:
            if f"NP{start_num}_NP{end_num}" in folder.name:
                print(f"  ✓ Matched with: {folder.name}")
                matched_pairs += 1
                folder_found = True
                break
        
        if not folder_found:
            print(f"  ✗ No matching folder found")

print(f"\n" + "="*70)
print(f"SUMMARY: {matched_pairs} matched pairs found")
print("="*70)

# Test imports
print("\nTesting required libraries:")
try:
    import pandas
    print("✓ pandas imported successfully")
except ImportError:
    print("✗ pandas import failed - run: pip install pandas")

try:
    import rdkit
    print("✓ rdkit imported successfully")
except ImportError:
    print("✗ rdkit import failed - run: pip install rdkit-pypi")

try:
    import xml.etree.ElementTree
    print("✓ xml processing available")
except:
    print("✗ xml import failed")

print("\nTest complete!")
input("\nPress Enter to exit...")