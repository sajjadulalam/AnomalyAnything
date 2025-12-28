# Severity-Aware Synthetic Anomaly Generation (Course Project)

This repository contains **my own experimental extensions and code** for:
**Advanced Machine Learning for Anomaly Detection (WS 2025/26)**.

## What I implemented (extensions)
1. **Severity-controlled synthetic anomaly generation** (mild/moderate/severe prompts).
2. **Training-regime comparisons** (synthetic vs real vs mixed).
3. **Contaminated training experiments** (inject real anomalies into training).
4. Unified evaluation + plots + result tables under `outputs/`.

> Note: This repo can work **without AnomalyAnything** using a built-in fallback generator
(`src/synthgen.py`). If you have AnomalyAnything installed, you can plug it in in
`src/synthgen.py` (see comments).

## Data layout expected
Place datasets under:
- `data/mvtec/<class_name>/train/good/*.png`
- `data/mvtec/<class_name>/test/good/*.png`
- `data/mvtec/<class_name>/test/<anomaly_type>/*.png`
- `data/mvtec/<class_name>/ground_truth/<anomaly_type>/*.png` (optional)

For VisA, a simple folder layout is supported:
- `data/visa/<class_name>/train/good/*.png`
- `data/visa/<class_name>/test/good/*.png`
- `data/visa/<class_name>/test/<anomaly_type>/*.png`

## Run all experiments
```bash
pip install -r requirements.txt
bash scripts/run_all.sh
