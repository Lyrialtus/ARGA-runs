# ARGA runs

This is just a refactoring of the original [ARGA project](https://github.com/khalil-research/ARGA-AAAI23). The context: [ARC Prize 2024](https://www.kaggle.com/competitions/arc-prize-2024).

## Description

- `input` contains the copy of the competition [data](https://www.kaggle.com/competitions/arc-prize-2024/data)
- `main` is the code and the notebook that runs it
- `runs` is the all-code notebook with results of running it on Kaggle:

| tasks | 10s | 30s | 180s |
| :-: | :-: | :-: | :-: |
| t400 | 51 (12.75%) | 58 (14.5%) | 63 (15.75%) |
| e400 | 6 (1.5%) | 6 (1.5%) | too long |

All the tasks with the matching input/output shapes are used (not the original subset of 160 tasks), because it's impossible to manually select tasks in the hidden test set. By the way, the performance on the hidden test set is 0 (0%).
