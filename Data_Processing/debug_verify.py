# debug_verify.py
import sys
import os

print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}")  # First 3 paths
print("-" * 50)

# First, test importing utils
try:
    import Data_Processing.utils as utils
    print("✓ utils module imported")
    
    # Test setup_logging
    logger = utils.setup_logging()
    print("✓ Logger created")
    
    # Now test verify_installation with detailed output
    print("\nRunning verify_installation()...")
    result = utils.verify_installation()
    print(f"\nResult: {result}")
    print(f"Result type: {type(result)}")
    
except Exception as e:
    print(f"Error during import/execution: {e}")
    import traceback
    traceback.print_exc()

# Also check the log file
print("\n" + "-" * 50)
print("Checking log file content:")
if os.path.exists('nmredata_conversion.log'):
    with open('nmredata_conversion.log', 'r') as f:
        # Get last 20 lines
        lines = f.readlines()
        print("Last 20 lines of log:")
        for line in lines[-20:]:
            print(line.rstrip())
else:
    print("Log file not found")