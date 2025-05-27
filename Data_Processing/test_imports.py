# test_imports.py
import sys
import os

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

try:
    import Data_Processing.utils as utils
    print("✓ utils imported successfully")
except Exception as e:
    print(f"✗ Error importing utils: {e}")

try:
    from Data_Processing.molecule_processor import MoleculeProcessor
    print("✓ MoleculeProcessor imported successfully")
except Exception as e:
    print(f"✗ Error importing MoleculeProcessor: {e}")

try:
    from Data_Processing.nmr_parser import NMRParser
    print("✓ NMRParser imported successfully")
except Exception as e:
    print(f"✗ Error importing NMRParser: {e}")

try:
    from Data_Processing.converter import CSVToNMReDATA
    print("✓ CSVToNMReDATA imported successfully")
except Exception as e:
    print(f"✗ Error importing CSVToNMReDATA: {e}")