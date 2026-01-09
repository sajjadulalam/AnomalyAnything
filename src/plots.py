import os
from typing import List, Tuple
import matplotlib.pyplot as plt

from .utils import ensure_dir

def save_roc_curve(fpr, tpr, path: str, title: str):
    ensure_dir(os.path.dirname(path))
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def save_pr_curve(rec, prec, path: str, title: str):
    ensure_dir(os.path.dirname(path))
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
