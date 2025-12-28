#!/usr/bin/env bash
set -e

python -c "import torch; print('torch', torch.__version__)"
echo "Running experiments..."

# Exp1: severity
python -m experiments.run_exp1 --config configs/exp1_severity.yaml

# Exp2: syn vs real
python -m experiments.run_exp2 --config configs/exp2_syn_vs_real.yaml

# Exp3: contamination
python -m experiments.run_exp3 --config configs/exp3_contamination.yaml

echo "Done. Check outputs/exp1, outputs/exp2, outputs/exp3"
