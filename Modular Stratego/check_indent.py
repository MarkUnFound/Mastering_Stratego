import sys

filename = "train_dqn.py"

with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

stack = [] # (indent, type)
# types: 'try', 'except', 'finally', 'def', 'class', 'if', 'while', 'for', 'with'

for i, line in enumerate(lines):
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        continue
    
    indent = len(line) - len(line.lstrip())
    
    if stripped.startswith('try:'):
        print(f"Line {i+1}: try at {indent}")
    elif stripped.startswith('except '):
        print(f"Line {i+1}: except at {indent}")
    elif stripped.startswith('except:'):
        print(f"Line {i+1}: except at {indent}")
    elif stripped.startswith('finally:'):
        print(f"Line {i+1}: finally at {indent}")
        
    if i+1 == 2072:
        print(f"--> TARGET LINE {i+1}: {line.rstrip()}")
