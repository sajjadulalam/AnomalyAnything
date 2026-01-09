"""
A minimal PatchCore-like baseline:
- Extract embeddings from a pretrained backbone (timm).
- Fit NearestNeighbors on train normal embeddings.
- Score test images by distance to nearest neighbors.

This is not the full official PatchCore, but it is a strong, reproducible baseline
and sufficient for course experiments on severity/training contamination effects.
"""
from typing import List, Tuple
import numpy as np
import torch
import timm
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from .utils import device

def _make_backbone(name: str):
    model = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
    model.eval()
    return model

@torch.no_grad()
def extract_embeddings(backbone_name: str, loader: DataLoader) -> np.ndarray:
    dev = device()
    model = _make_backbone(backbone_name).to(dev)
    embs = []
    for x in loader:
        x = x.to(dev)
        z = model(x)  # [B, D]
        embs.append(z.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)

def fit_nn(train_embs: np.ndarray, k: int) -> NearestNeighbors:
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(train_embs)
    return nn

def score_nn(nn: NearestNeighbors, test_embs: np.ndarray) -> List[float]:
    dists, _ = nn.kneighbors(test_embs, return_distance=True)
    # anomaly score: mean distance to k nearest neighbors
    return dists.mean(axis=1).tolist()
