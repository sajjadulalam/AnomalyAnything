import os
from typing import Dict, List, Tuple
import pandas as pd

from .metrics import compute_metrics, curves
from .plots import save_roc_curve, save_pr_curve
from .utils import ensure_dir, write_json

def evaluate_and_save(y_true: List[int], scores: List[float], out_dir: str, prefix: str, save_curves: bool = True) -> Dict[str, float]:
    ensure_dir(out_dir)
    m = compute_metrics(y_true, scores)
    write_json(os.path.join(out_dir, f"{prefix}_metrics.json"), m)

    if save_curves:
        (fpr, tpr), (rec, prec) = curves(y_true, scores)
        save_roc_curve(fpr, tpr, os.path.join(out_dir, f"{prefix}_roc.png"), f"{prefix} ROC")
        save_pr_curve(rec, prec, os.path.join(out_dir, f"{prefix}_pr.png"), f"{prefix} PR")

    return m

def append_result_row(csv_path: str, row: Dict):
    ensure_dir(os.path.dirname(csv_path))
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
