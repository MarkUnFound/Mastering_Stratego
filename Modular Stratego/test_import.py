
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Attempting to import environment...")
try:
    import environment
    print("Successfully imported environment")
except ImportError as e:
    print(f"ImportError: {e}")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
except IndentationError as e:
    print(f"IndentationError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
