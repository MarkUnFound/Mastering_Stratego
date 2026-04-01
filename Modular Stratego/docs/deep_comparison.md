## In-Depth Architectural Comparison: LSTM vs Rainbow+AAREN

| Metric | Vanilla DQN + LSTM | Rainbow DQN + AAREN |
|---|---|---|
| **Total Episodes** | 15302 | 8000 |
| **Total Steps** | 10306433 | 4184398 |
| **Average Steps/Episode** | 673.5 | 523.0 |
| **Total Wins** | 3821 | 3939 |
| **Total Losses** | 3824 | 3569 |
| **Draws** | 7657 | 492 |
| **Overall Win Rate** | 24.97% | 49.24% |
| **Overall Draw Rate** | 50.04% | 6.15% |
| **Wins by Flag / Depletion** | 718 / 3103 | 397 / 3542 |
| **Losses by Flag / Depletion** | 734 / 3090 | 587 / 2982 |
| **Average Loss (last 100)** | 0.0001 | 3.7991 |
| **Average Q-Value (last 100)** | 0.1038 | 0.8653 |
| **Average Reward (last 100)** | 0.0743 | -20.4774 |
| **Early Avg Reward (first 1K)** | 0.0763 | -13.9346 |
| **Late Avg Reward (last 1K)** | 0.0846 | -20.5152 |
| **Avg Entropy (last 100)** | 0.0000 | 0.0470 |
| **DQN Grad Norm (last 100)** | 0.0001 | 0.0132 |
| **AAREN Accuracy (last 100)** | 0.1779 | 0.2405 |
| **AAREN Loss (last 100)** | 1.5581 | 1.9950 |
| **AAREN Embed Std (last 100)** | 0.1724 | 2.0616 |
