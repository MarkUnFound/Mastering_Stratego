
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from environment import StrategoEnvironment
    print("Successfully imported StrategoEnvironment")
    
    env = StrategoEnvironment(device='cpu')
    print("Successfully instantiated StrategoEnvironment")
    
    if hasattr(env, 'step'):
        print("StrategoEnvironment has 'step' method")
    else:
        print("ERROR: StrategoEnvironment MISSING 'step' method")
        print("Dir:", dir(env))
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
