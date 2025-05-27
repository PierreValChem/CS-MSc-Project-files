# check_dependencies.py
import sys
print(f"Python version: {sys.version}")
print("-" * 50)

# Check each dependency individually
dependencies = [
    ('pandas', 'pd'),
    ('numpy', 'np'),
    ('psutil', 'psutil'),
    ('rdkit', 'Chem'),
    ('rdkit.Chem', 'AllChem'),
]

missing = []

for module_name, import_name in dependencies:
    try:
        if module_name == 'rdkit':
            import rdkit
            from rdkit import Chem
            print(f"✓ {module_name}: {rdkit.__version__}")
        elif module_name == 'rdkit.Chem':
            from rdkit.Chem import AllChem
            print(f"✓ {module_name}.AllChem: Available")
        else:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {module_name}: {version}")
    except ImportError as e:
        print(f"✗ {module_name}: NOT INSTALLED - {e}")
        missing.append(module_name)
    except Exception as e:
        print(f"✗ {module_name}: ERROR - {e}")
        missing.append(module_name)

print("-" * 50)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print("\nTo install missing packages, run:")
    for pkg in missing:
        if 'rdkit' in pkg:
            print("conda install -c conda-forge rdkit")
        else:
            print(f"pip install {pkg}")
else:
    print("\nAll dependencies are installed!")

# Test RDKit 3D generation specifically
print("\n" + "-" * 50)
print("Testing RDKit 3D generation:")
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    mol = Chem.MolFromSmiles('CCO')
    print("✓ Created molecule from SMILES")
    
    mol_h = Chem.AddHs(mol)
    print("✓ Added hydrogens")
    
    result = AllChem.EmbedMolecule(mol_h)
    if result == 0:
        print("✓ 3D embedding successful")
    else:
        print("✗ 3D embedding failed")
    
    # Test MMFF
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94s')
        if props:
            print("✓ MMFF94s parameters available")
        else:
            print("✗ MMFF94s parameters not available")
    else:
        print("✗ MMFF not applicable to test molecule")
        
except Exception as e:
    print(f"✗ RDKit 3D generation test failed: {e}")
    import traceback
    traceback.print_exc()