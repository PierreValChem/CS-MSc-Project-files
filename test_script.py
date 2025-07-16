#!/usr/bin/env python3
"""Debug version of the integrated processor"""

print("Debug script starting...")

import os
import sys

# Test basic functionality
print(f"Python version: {sys.version}")
print(f"Args: {sys.argv}")

# Parse arguments manually
if len(sys.argv) < 2:
    print("Usage: python debug_integrated_processor.py <base_dir> [-o output_dir]")
    sys.exit(1)

base_dir = sys.argv[1]
output_dir = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "-o" else os.path.join(base_dir, "output")

print(f"Base directory: {base_dir}")
print(f"Output directory: {output_dir}")

# Check if directory exists
if not os.path.exists(base_dir):
    print(f"Error: Directory {base_dir} does not exist!")
    sys.exit(1)

# List contents
print("\nDirectory contents:")
items = os.listdir(base_dir)
excel_files = [f for f in items if f.endswith('.xlsx')]
csv_files = [f for f in items if f.endswith('.csv')]
directories = [f for f in items if os.path.isdir(os.path.join(base_dir, f))]

print(f"Excel files ({len(excel_files)}):")
for f in excel_files[:5]:
    print(f"  - {f}")

print(f"\nCSV files ({len(csv_files)}):")
for f in csv_files[:5]:
    print(f"  - {f}")

print(f"\nDirectories ({len(directories)}):")
for d in directories[:5]:
    print(f"  - {d}")

# Test pattern matching
print("\n" + "="*50)
print("Testing pattern matching...")

import re

for excel_file in excel_files[:3]:
    print(f"\nExcel file: {excel_file}")
    
    # Try to extract NP range
    match = re.search(r'NP(\d+)-NP(\d+)', excel_file)
    if match:
        start = match.group(1)
        end = match.group(2)
        print(f"  Range: NP{start} to NP{end}")
        
        # Look for matching folder
        target_pattern = f"NP{start}_NP{end}".lower()
        print(f"  Looking for folder containing: {target_pattern}")
        
        found = False
        for d in directories:
            if target_pattern in d.lower():
                print(f"  ✓ Found match: {d}")
                found = True
                break
        
        if not found:
            print(f"  ✗ No matching folder found")
            print(f"  Available folders that might match:")
            for d in directories:
                if start in d or end in d:
                    print(f"    - {d}")

print("\nDebug complete.")