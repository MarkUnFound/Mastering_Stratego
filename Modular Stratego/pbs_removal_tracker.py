# PBS Removal Tracker
# Use this file to track & verify PBS legacy code removal progress.
# Run: python pbs_removal_tracker.py

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# ===================================================================
# CRITICAL PBS patterns that MUST NOT appear in Python files
# ===================================================================
CRITICAL_PATTERNS = [
    (r'\bProbabilisticBeliefState\b', 'Direct PBS class usage'),
    (r'\bPBS_EVALUATOR_AVAILABLE\b', 'PBS evaluator availability flag'),
    (r'\bPBSEvaluator\b', 'PBS evaluator class'),
    (r'\bget_uncertainty_map\b', 'PBS-only method (not in HistoryAggregator)'),
    (r'\btrain_pbs_evaluator\b', 'PBS evaluator training call'),
    (r'from pbs import', 'Import from pbs module'),
    (r'from pbs_evaluator import', 'Import from pbs_evaluator'),
    (r'from pbs_visualizer import', 'Import from pbs_visualizer'),
    (r'import pbs_visualizer', 'Import pbs_visualizer'),
]

# ===================================================================
# WARNING patterns — legacy naming that should be updated  
# (non-blocking: aliased during transition, then removed)
# ===================================================================
WARNING_PATTERNS = [
    (r'\buse_pbs\b', 'Legacy naming (should become use_history)'),
    (r'\.pbs\b', 'Legacy attribute (should become .history)'),
    (r'\bpbs_instances\b', 'Legacy naming (should become history_instances)'),
    (r'\breset_pbs\b', 'Legacy method name (should become reset_history)'),
    (r'\benable_pbs\b', 'Legacy method name (should become enable_history)'),
    (r'\bupdate_pbs_batch\b', 'Legacy method name (should become update_history_batch)'),
    (r'\btrain_pbs\b', 'Legacy method name (should become train_history)'),
    (r'\bpbs_eval\b', 'Reference to PBS evaluator'),
    (r'\bplot_pbs_evaluator\b', 'PBS evaluator plotting'),
    (r'\bpbs_accuracy\b', 'PBS accuracy metric'),
    (r'\bpbs_visualizer\b', 'PBS visualizer reference'),
    (r'\bPBS_UPDATE_INTERVAL\b', 'PBS update interval config'),
    (r'\blane_opponent_uses_pbs\b', 'PBS naming in lane tracking'),
    (r'\bopp_uses_pbs\b', 'PBS naming in opponent selection'),
]

# ===================================================================
# Files/dirs that should be DELETED
# ===================================================================
SHOULD_DELETE = [
    'pbs/',
    'pbs_evaluator.py',
    'pbs_visualizer.py',
]

# Excluded dirs/files
EXCLUDE_DIRS = {'__pycache__', '.git', 'dqn_models', 'node_modules', 'league'}
EXCLUDE_FILES = {'pbs_removal_tracker.py'}


def scan_files():
    py_files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            if f.endswith('.py') and f not in EXCLUDE_FILES:
                py_files.append(os.path.join(dirpath, f))
    return py_files


def check_pattern(file_path, pattern, description):
    hits = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for line_num, line in enumerate(fh, 1):
                if re.search(pattern, line):
                    hits.append((line_num, line.rstrip()))
    except Exception:
        pass
    return hits


def main():
    print("=" * 70)
    print("  PBS Legacy Code Removal Tracker")
    print("=" * 70)

    py_files = scan_files()
    print(f"\nScanning {len(py_files)} Python files in {ROOT}\n")

    # --- Check files that should be deleted ---
    print("--- Files/Dirs to Delete ---")
    delete_remaining = 0
    for item in SHOULD_DELETE:
        full_path = os.path.join(ROOT, item)
        exists = os.path.exists(full_path)
        status = "[EXISTS - DELETE ME]" if exists else "[GONE]"
        print(f"  {status} {item}")
        if exists:
            delete_remaining += 1

    # --- Critical patterns ---
    print("\n--- Critical PBS References (MUST be removed) ---")
    critical_total = 0
    critical_files = set()
    for pattern, desc in CRITICAL_PATTERNS:
        for fp in py_files:
            hits = check_pattern(fp, pattern, desc)
            if hits:
                rel = os.path.relpath(fp, ROOT)
                critical_files.add(rel)
                for line_num, line_text in hits:
                    critical_total += 1
                    print(f"  [CRITICAL] {rel}:{line_num} -- {desc}")
                    print(f"             {line_text.strip()}")

    # --- Warning patterns ---
    print("\n--- Legacy PBS Naming (should be renamed) ---")
    warning_total = 0
    warning_files = set()
    for pattern, desc in WARNING_PATTERNS:
        for fp in py_files:
            hits = check_pattern(fp, pattern, desc)
            if hits:
                rel = os.path.relpath(fp, ROOT)
                warning_files.add(rel)
                warning_total += len(hits)

    if warning_total > 0:
        # Summarize by file
        file_counts = {}
        for pattern, desc in WARNING_PATTERNS:
            for fp in py_files:
                hits = check_pattern(fp, pattern, desc)
                if hits:
                    rel = os.path.relpath(fp, ROOT)
                    file_counts[rel] = file_counts.get(rel, 0) + len(hits)
        for f, count in sorted(file_counts.items(), key=lambda x: -x[1]):
            print(f"  [WARN] {f}: {count} legacy references")
    else:
        print("  [CLEAN] No legacy naming found.")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Files to delete:     {delete_remaining}/{len(SHOULD_DELETE)}")
    print(f"  Critical references: {critical_total} in {len(critical_files)} files")
    print(f"  Warning references:  {warning_total} in {len(warning_files)} files")

    total_issues = delete_remaining + critical_total
    if total_issues == 0 and warning_total == 0:
        print("\n  [ALL CLEAN] PBS has been fully removed!")
        return 0
    elif total_issues == 0:
        print(f"\n  [FUNCTIONAL] No critical PBS code remains. {warning_total} legacy names to rename.")
        return 0
    else:
        print(f"\n  [INCOMPLETE] {total_issues} critical issues remain.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
