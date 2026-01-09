import os
from typing import Dict
import yaml
from PIL import Image
from tqdm import tqdm

from .datasets import load_class_paths
from .synthgen import generate_anomaly
from .utils import ensure_dir, append_jsonl

def generate_for_class(dataset_name: str, root: str, cls: str, out_root: str,
                       method: str, prompts: Dict[str, str], img_per_normal: int,
                       seed: int, limit_normals: int, severity: str) -> str:
    paths = load_class_paths(dataset_name, root, cls)
    normals = paths["train_normal"][:limit_normals] if limit_normals else paths["train_normal"]

    out_dir = os.path.join(out_root, dataset_name, cls, severity)
    ensure_dir(out_dir)
    meta_path = os.path.join(out_dir, "metadata.jsonl")

    prompt = prompts[severity]
    for p in tqdm(normals, desc=f"Gen {dataset_name}/{cls}/{severity}"):
        img = Image.open(p).convert("RGB")
        base = os.path.basename(p)
        for i in range(img_per_normal):
            s = seed + i
            syn = generate_anomaly(img, prompt=prompt, method=method, seed=s)
            syn.save(os.path.join(out_dir, base))
            append_jsonl(meta_path, {
                "dataset": dataset_name,
                "class": cls,
                "severity": severity,
                "prompt": prompt,
                "seed": s,
                "source_path": p,
                "out_path": os.path.join(out_dir, base),
                "method": method,
            })
    return out_dir

def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gen = cfg["generator"]
    method = gen.get("method", "fallback")
    out_root = gen.get("out_root", "data/generated")
    prompts = gen["prompts"]
    img_per_normal = int(gen.get("images_per_normal", 1))

    limit_normals = int(cfg["training"].get("limit_normals", 0))
    seeds = cfg["training"]["seeds"]

    for ds in cfg["datasets"]:
        ds_name = ds["name"]
        ds_root = ds["root"]
        classes = ds["classes"]
        if classes == ["all"]:
            # lazy import to avoid filesystem errors if missing
            from .datasets import list_mvtec_classes, list_visa_classes
            classes = list_mvtec_classes(ds_root) if ds_name.lower() == "mvtec" else list_visa_classes(ds_root)

        for cls in classes:
            for severity in gen.get("prompts", {}).keys():
                for seed in seeds[:1]:  # generation once is enough; keep deterministic
                    generate_for_class(ds_name, ds_root, cls, out_root, method, prompts,
                                       img_per_normal, seed, limit_normals, severity)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
