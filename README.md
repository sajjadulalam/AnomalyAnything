UPDATED README (copy-paste into README.md)
# 📌 AnomalyAnything — My Experimental Extensions  
**Advanced Machine Learning for Anomaly Detection (WS 2025/26)**  
by **Sajjadul Alam**

[![Python](https://img.shields.io/badge/python-3.8–3.12-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Stars](https://img.shields.io/github/stars/sajjadulalam/AnomalyAnything.svg?style=social)]()
[![Last Commit](https://img.shields.io/github/last-commit/sajjadulalam/AnomalyAnything.svg)]()

This repository contains **my own experimental extensions, code additions, and evaluation tools**
built on top of the AnomalyAnything research framework.

---

## 🚀 What I Implemented (My Extensions)

### ✅ 1. Severity-Controlled Synthetic Anomaly Generation  
Prompts: **mild**, **moderate**, **severe**

### ✅ 2. Real vs Synthetic vs Mixed Training  
Fully automated comparisons

### ✅ 3. Contaminated Training Robustness  
0–20% injected anomalies inside normal splits

### ✅ 4. Unified Results Pipeline  
Exports:
- AUC/F1/Precision tables
- ROC + PR plots
- Synthetic samples
- PatchCore maps
- Checkpoints

---

## 📁 Dataset Folder Structure

### MVTEC AD


data/mvtec/<class_name>/
train/good/
test/good/
test/<anomaly_type>/
ground_truth/<anomaly_type>/ (optional)


### VisA


data/visa/<class_name>/
train/good/
test/good/
test/<anomaly_type>/


---

## ▶️ Running Experiments

Install requirements:


pip install -r requirements.txt


Run all:


bash scripts/run_all.sh


Or individually:


python -m experiments.run_exp1
python -m experiments.run_exp2
python -m experiments.run_exp3


---

## 🧩 Repository Structure


src/
experiments/
configs/
scripts/
outputs/ # Generated


---

## 📊 Outputs (auto-generated)
- Synthetic anomalies
- PatchCore heatmaps
- AE checkpoints
- ROC + PR curves
- Metrics CSV files

---

## ⚙️ Platform Notes
Windows users:


pip install "numpy<2.0" scikit-learn==1.2.2


---

## 🧾 Citing This Repository

If you use this repo in a project or report, please cite:

```bibtex
@software{sajjadulalam_anomalyanything_2025,
  author       = {Alam, Sajjadul},
  title        = {AnomalyAnything - Promptable Unseen Visual Anomaly Generation Extensions},
  year         = {2025},
  url          = {https://github.com/sajjadulalam/AnomalyAnything},
  note         = {Course Project, Advanced Machine Learning for Anomaly Detection (WS 2025/26)}
}


To cite the upstream paper:

@inproceedings{anomalyanything2024,
  title={AnomalyAnything: Promptable Unseen Visual Anomaly Generation},
  author={Zhang et al.},
  year={2024}
}
