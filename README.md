# 📌 AnomalyAnything — My Experimental Extensions  
**Advanced Machine Learning for Anomaly Detection (WS 2025/26)**  
by **Sajjadul Alam**

This repository contains **my own experimental extensions, code additions, and evaluation tools**
built on top of the AnomalyAnything research framework.

---

## 🚀 What I Implemented (My Extensions)

### ✅ 1. Severity-Controlled Synthetic Anomaly Generation
- Prompt-driven anomaly injection: **mild / moderate / severe**
- Generated per-class datasets
- Works with MVTEC and VisA
- Includes **a fallback generator** in `src/synthgen.py`  
  → You can run the repo without the original AnomalyAnything model

### ✅ 2. Real vs Synthetic vs Mixed Training Regimes
Implemented in `run_exp1.py` and `run_exp2.py`

- **Train only on synthetic anomalies**
- **Train only on real anomalies**
- **Evaluate cross-domain generalization**
- Export **scores + curves**

### ✅ 3. Contaminated Training (Exp3)
Implemented in `run_exp3.py`

- Add **0%, 5%, 10%, 20%** real anomalies into “normal” training
- Tests robustness when training data is impure
- Includes files:
  - Synthetic + real mixed train batches
  - ROC/PR curves
  - CSV summaries

### ✅ 4. Unified Results Pipeline
Stored automatically under `outputs/`:
- `figures/` → ROC + PR + comparison charts  
- `tables/results.csv` → AUC, F1, Precision@K  
- `synthetic_samples/` → Generated anomalies  
- `patchcore_contam_train/` → Contaminated training visualization  
- `checkpoints/` → Saved AE models

---

## 📁 Dataset Folder Structure

### MVTEC AD
data/mvtec/<class_name>/
train/good/.png
test/good/.png
test/<anomaly_type>/.png
ground_truth/<anomaly_type>/.png (optional)

### VisA
data/visa/<class_name>/
train/good/.png
test/good/.png
test/<anomaly_type>/*.png

If VisA is missing, the repo still runs MVTEC.

---

## ▶️ Running All Experiments

### Install requirements
pip install -r requirements.txt

### Run everything
bash scripts/run_all.sh

Or one by one:
python -m experiments.run_exp1
python -m experiments.run_exp2
python -m experiments.run_exp3

---

## 🧩 Repository Structure
src/
synthgen.py # My anomaly synthesizer + fallback mode
datasets_mvtec.py # MVTec loader
datasets_visa.py # VisA loader
train_autoencoder.py # Denoising AE
train_patchcore.py # PatchCore embedding + scoring

experiments/
run_exp1.py # Severity study
run_exp2.py # Real vs synthetic
run_exp3.py # Contaminated training

configs/ # YAML experiment settings
outputs/ # Auto generated
scripts/ # Shortcut runners

---

## 📊 Output Files Generated
### For every dataset / class / severity / seed
- **AUC, F1, Precision** (CSV)
- **ROC + PR curves** (PNG)
- **AE checkpoints**
- **Synthetic sample images**
- **PatchCore heatmaps**
- **Contaminated patch grids**

All results are reproducible.

---

## ⚙️ System Notes
Windows users:
pip install "numpy<2.0" scikit-learn==1.2.2
Use `conda` for easiest dependency solving.

---

## 🧾 Academic & Course Notes
- Repository created for **WS 2025/26 AML for Anomaly Detection**
- Focused on **my own ideas and implementation**
- Built referencing:  
  **AnomalyAnything**, **PatchCore (CVPR 2022)**, **MVTEC AD**, **VisA**

