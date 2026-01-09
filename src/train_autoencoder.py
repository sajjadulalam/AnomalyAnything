import os
from typing import Dict, Tuple, List
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .models_ae import ConvAutoencoder
from .utils import device, set_seed, ensure_dir, SimpleLogger

def train_denoising_ae(paired_dataset, epochs: int, lr: float, batch_size: int, seed: int,
                      ckpt_path: str, log_path: str) -> ConvAutoencoder:
    set_seed(seed)
    dev = device()
    logger = SimpleLogger(log_path)
    loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ConvAutoencoder().to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    logger.log(f"Training Denoising AE | epochs={epochs} lr={lr} batch={batch_size} device={dev}")
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y in tqdm(loader, desc=f"AE epoch {ep}/{epochs}", leave=False):
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad(set_to_none=True)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        logger.log(f"epoch={ep} loss={total/max(1,n):.6f}")

    ensure_dir(os.path.dirname(ckpt_path))
    torch.save(model.state_dict(), ckpt_path)
    logger.log(f"Saved checkpoint: {ckpt_path}")
    return model

@torch.no_grad()
def ae_image_scores(model: ConvAutoencoder, test_loader: DataLoader) -> Tuple[List[float], List[int]]:
    """
    Image-level anomaly score: mean absolute reconstruction error.
    """
    dev = device()
    model.eval().to(dev)
    scores = []
    labels = []
    for x, y in test_loader:
        x = x.to(dev)
        yhat = model(x)
        err = torch.mean(torch.abs(yhat - x), dim=(1,2,3))  # compare reconstruction to input
        scores.extend(err.detach().cpu().tolist())
        labels.extend(y.tolist())
    return scores, labels
