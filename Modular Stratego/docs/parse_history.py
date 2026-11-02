"""Show the structure of training_history.json files (standard or line-delimited JSON)."""
import json
import os
import sys


def describe(f):
    print(f"\n--- {f} ---")
    try:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("Keys:", list(data.keys()))
            for key in list(data.keys()):
                val = data[key]
                print(f"  {key}: type={type(val)}, len={len(val) if isinstance(val, list) else 'N/A'}")
    except Exception as e:
        print("Error parsing as standard JSON:", e)
        # Try line by line
        try:
            with open(f, 'r', encoding='utf-8') as file:
                lines = [json.loads(line) for line in file if line.strip()]
            print(f"Parsed as line-by-line JSON, {len(lines)} lines")
            if lines:
                print("Keys of first line:", list(lines[0].keys()))
        except Exception as e2:
            print("Error parsing line-by-line:", e2)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        describe(path)
