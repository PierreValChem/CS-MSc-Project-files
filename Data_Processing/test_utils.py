# test_utils.py
import sys
sys.path.insert(0, '.')  # Ensure current directory is in path

try:
    from Data_Processing.utils import verify_installation
    print("Successfully imported verify_installation")
    
    result = verify_installation()
    print(f"\nverify_installation() returned: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()