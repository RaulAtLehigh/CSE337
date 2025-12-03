# Experiment Comparison

| ID | Hidden layers | Total hidden units | Best eval return | Time to optimal | Training time (s) |
| --- | --- | --- | --- | --- | --- |
| l1_h64 | 64 | 64 | -0.23 | not reached | 259.4 |
| l1_h128 | 128 | 128 | 1.38 | not reached | 240.7 |
| l1_h256 | 256 | 256 | 1.74 | not reached | 233.0 |
| l1_h512 | 512 | 512 | 1.82 | not reached | 266.0 |
| l2_h64 | 64 → 64 | 128 | 1.44 | not reached | 264.0 |
| l2_h128 | 128 → 128 | 256 | 1.37 | not reached | 303.8 |
| l2_h256 | 256 → 256 | 512 | -1.83 | not reached | 451.8 |
| l2_h512 | 512 → 512 | 1024 | 1.13 | not reached | 3609.0 |
| l3_h64 | 64 → 64 → 64 | 192 | 1.47 | not reached | 298.2 |
| l3_h128 | 128 → 128 → 128 | 384 | 1.50 | not reached | 350.8 |
| l3_h256 | 256 → 256 → 256 | 768 | 2.53 | not reached | 508.3 |
| l3_h512 | 512 → 512 → 512 | 1536 | 2.46 | not reached | 1001.3 |
| l4_h64 | 64 → 64 → 64 → 64 | 256 | 1.46 | not reached | 391.2 |
| l4_h128 | 128 → 128 → 128 → 128 | 512 | 2.23 | not reached | 441.4 |
| l4_h256 | 256 → 256 → 256 → 256 | 1024 | 2.56 | not reached | 652.6 |
| l4_h512 | 512 → 512 → 512 → 512 | 2048 | 2.58 | not reached | 1666.8 |

Each run directory contains `summary.txt`, `reward_curve.png`, `actions_eval.json`, and `best_model.pt`.