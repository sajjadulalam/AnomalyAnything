import os
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_str() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

@dataclass
class RunPaths:
    root: str
    logs: str
    tables: str
    figures: str
    checkpoints: str

def make_run_paths(out_root: str, exp_name: str, dataset: str, cls: str, tag: str, seed: int) -> RunPaths:
    base = os.path.join(out_root, exp_name, dataset, cls, tag, f"seed{seed}")
    logs = os.path.join(base, "logs")
    tables = os.path.join(base, "tables")
    figures = os.path.join(base, "figures")
    checkpoints = os.path.join(base, "checkpoints")
    for p in [logs, tables, figures, checkpoints]:
        ensure_dir(p)
    return RunPaths(root=base, logs=logs, tables=tables, figures=figures, checkpoints=checkpoints)

class SimpleLogger:
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path))

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
