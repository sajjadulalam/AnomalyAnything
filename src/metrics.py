from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

def best_f1_threshold(y_true: List[int], scores: List[float]) -> Tuple[float, float, float, float]:
    y = np.array(y_true)
    s = np.array(scores)
    prec, rec, thr = precision_recall_curve(y, s)
    # precision_recall_curve returns thr with len = len(prec)-1
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    idx = int(np.nanargmax(f1))
    best_f1 = float(f1[idx])
    best_prec = float(prec[idx])
    best_rec = float(rec[idx])
    best_thr = float(thr[idx-1]) if idx > 0 and idx-1 < len(thr) else float(np.median(s))
    return best_thr, best_f1, best_prec, best_rec

def compute_metrics(y_true: List[int], scores: List[float]) -> Dict[str, float]:
    y = np.array(y_true)
    s = np.array(scores)
    auc = float(roc_auc_score(y, s)) if len(set(y_true)) > 1 else float("nan")
    thr, f1, prec, rec = best_f1_threshold(y_true, scores)
    return {
        "auc": auc,
        "threshold": float(thr),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }

def curves(y_true: List[int], scores: List[float]):
    import numpy as np
    y = np.array(y_true)
    s = np.array(scores)
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    return (fpr, tpr), (rec, prec)
