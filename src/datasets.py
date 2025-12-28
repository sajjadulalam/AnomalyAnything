import os
import glob
from typing import Dict, List, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def _list_images(patterns: List[str]) -> List[str]:
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(list(set(files)))

def list_mvtec_classes(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def list_visa_classes(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def load_class_paths(dataset_name: str, root: str, cls: str) -> Dict[str, List[str]]:
    """
    Returns dict with:
      train_normal, test_normal, test_anomaly, test_labels (0/1 aligned with test_* lists)
    """
    if dataset_name.lower() == "mvtec":
        train_normal = _list_images([os.path.join(root, cls, "train", "good", "*.*")])
        test_good = _list_images([os.path.join(root, cls, "test", "good", "*.*")])
        test_anom = _list_images([os.path.join(root, cls, "test", "*", "*.*")])
        test_anom = [p for p in test_anom if os.path.basename(os.path.dirname(p)) != "good"]
        test_paths = test_good + test_anom
        test_labels = [0] * len(test_good) + [1] * len(test_anom)
        return {
            "train_normal": train_normal,
            "test_paths": test_paths,
            "test_labels": test_labels,
        }

    if dataset_name.lower() == "visa":
        train_normal = _list_images([os.path.join(root, cls, "train", "good", "*.*")])
        test_good = _list_images([os.path.join(root, cls, "test", "good", "*.*")])
        test_anom = _list_images([os.path.join(root, cls, "test", "*", "*.*")])
        test_anom = [p for p in test_anom if os.path.basename(os.path.dirname(p)) != "good"]
        test_paths = test_good + test_anom
        test_labels = [0] * len(test_good) + [1] * len(test_anom)
        return {
            "train_normal": train_normal,
            "test_paths": test_paths,
            "test_labels": test_labels,
        }

    raise ValueError(f"Unknown dataset: {dataset_name}")

class ImageListDataset(Dataset):
    def __init__(self, paths: List[str], img_size: int, return_path: bool = False):
        self.paths = paths
        self.return_path = return_path
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        if self.return_path:
            return x, p
        return x

class PairedDenoiseDataset(Dataset):
    """
    For denoising AE: input is synthetic anomalous image, target is clean normal image.
    Pairs are matched by filename (same basename).
    """
    def __init__(self, syn_dir: str, clean_paths: List[str], img_size: int):
        self.clean_paths = clean_paths
        self.syn_map = {}
        for sp in glob.glob(os.path.join(syn_dir, "*.*")):
            self.syn_map[os.path.basename(sp)] = sp

        self.pairs: List[Tuple[str, str]] = []
        for cp in clean_paths:
            bn = os.path.basename(cp)
            if bn in self.syn_map:
                self.pairs.append((self.syn_map[bn], cp))

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        syn_p, clean_p = self.pairs[idx]
        syn = self.tf(Image.open(syn_p).convert("RGB"))
        clean = self.tf(Image.open(clean_p).convert("RGB"))
        return syn, clean
