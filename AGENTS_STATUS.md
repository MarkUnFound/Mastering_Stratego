# Project Status: MARQ Framework Research

## Current Status (2026-04-02)
- **Results Chapter Merged**: Combined DQN-noAAREN branch baseline analysis (Findings 1-6, bottleneck equations, behavioral patterns) with empirical Rainbow DQN validation data (Findings 7-14) into unified results.tex.
- **3-Dataset Comparative Analysis**: Validated against three training datasets.
  - DQN+LSTM 15k (15,302 episodes, self-play, JSON intact)
  - Rainbow+AAREN 9.5k (9,778 episodes, self-play, JSON intact)
  - Rainbow+AAREN 75k (75,047 episodes, heuristic opponents, separate lineage)
- **Limitation Mapping Updated**: All findings status changed from "Proposed" to "Validated" with empirical evidence references.
- **Recommendations Chapter**: Restored from DQN-noAAREN branch, updated to reflect validated empirical results.
- **Appendix B**: Merged 8 comparison training charts into existing appendix.
- **Key Findings**:
    - Rainbow DQN+AAREN (9.5k): 49.88% win rate, 6.69% draw rate, loss-WR correlation = -0.787
    - Vanilla DQN+LSTM (15k): 24.97% win rate, 50.04% draw rate, loss-WR correlation = 0.051
    - Rainbow 75k (extended): 11.66% win rate vs heuristic, loss convergence 3.89 to 2.84, 99.96% flag capture ratio
    - AAREN vs LSTM inference: 24.23% vs 17.74% piece identity accuracy

## Implemented Features
### 1. Data Mining Suite (`architecture_comparison.py`, `full_comparison_analysis.py`)
- **Understanding**: Custom parsers and analyzers for Stratego training histories. Computes rolling win rates, reward trajectories, and loss-winrate correlations.
- **Corrupted Data Handling**: The 37k LSTM run history was zeroed out/corrupted. Used chart inspection for the 37k baseline analysis in the DQN-noAAREN branch. Used the 15k LSTM run as the reliable JSON baseline for empirical comparison.

## Agent Learning
- **Task: Comparative Analysis of DQN Models**
    - **Error**: Rolling win rates initially showed absurd values (e.g., 3000%).
    - **Root Cause**: `wins_p1_history` in the older JSON format was a cumulative counter, but the parser treated it as a binary per-episode indicator.
    - **Fix**: Implemented `cumulative_to_binary` transformation in the `architecture_comparison.py` script.
    - **Learning**: Always verify the "delta" vs "total" nature of historical metric arrays before applying windowed averages.
    - **Data Integrity**: Historical research data (like the 37k run) can suffer from FS corruption or incomplete saves; always verify file byte-content (e.g., checking for null bytes) before parsing as JSON.
- **Task: LaTeX Branch Integration**
    - **Error**: `replace_file_content` tool fails on files with CRLF line endings when target content uses LF.
    - **Learning**: When working with files from different git branches (Windows CRLF vs Unix LF), use `write_to_file` with `Overwrite` for large rewrites instead of fighting line-ending mismatches with `replace_file_content`.
    - **Learning**: When merging content from two branches, establish the "style source" (tone, structure) and the "data source" (empirical findings) clearly before writing. The style source dictates paragraph structure, naming conventions (Findings), and formality level.
