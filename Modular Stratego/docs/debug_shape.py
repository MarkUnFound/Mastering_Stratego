"""Print the keys and list lengths of one or more training_history.json files."""
import json
import sys


def check(f):
    data = json.load(open(f, 'r', encoding='utf-8'))
    for k, v in list(data.items()):
        if isinstance(v, list):
            print(f"  {k}: list of len {len(v)}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        check(path)
