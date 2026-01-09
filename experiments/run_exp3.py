import os
import random
import yaml
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.datasets import load_class_paths, ImageListDataset, PairedDenoiseDataset
from src.utils import make_run_paths, SimpleLogger, set_seed, ensure_dir
from src.generate_anomalies import generate_for_class
from src.train_autoencoder import train_denoising_ae, ae_image_scores
from src.train_patchcore import extract_embeddings, fit_nn, score_nn
from src.evaluate import evaluate_and_save, append_result_row
from src.synthgen import generate_anomaly

def _make_inputs_from_clean(clean_paths, out_dir, prompt, method, seed):
    ensure_dir(out_dir)
    for p in clean_paths:
        img = Image.open(p).convert("RGB")
        syn = generate_anomaly(img, prompt=prompt, method=method, seed=seed)
        syn.save(os.path.join(out_dir, os.path.basename(p)))

def run(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = "outputs"
    exp_name = "exp3"
    gen = cfg["generator"]
    method = gen.get("method", "fallback")
    out_gen = gen.get("out_root", "data/generated")
    severity = gen.get("severity_used", "moderate")
    prompt = gen["prompts"][severity]

    tr = cfg["training"]
    seeds = tr["seeds"]
    ratios = tr["contamination_ratios"]
    limit_normals = int(tr.get("limit_normals", 0))
    limit_real_anoms = int(tr.get("limit_real_anomalies", 0))
    img_size = int(tr.get("img_size", 256))
    batch_size = int(tr.get("batch_size", 16))
    ae_cfg = tr["ae"]
    pc_cfg = tr["patchcore"]
    save_curves = bool(cfg["evaluation"].get("save_curves", True))

    for ds in cfg["datasets"]:
        ds_name = ds["name"]
        ds_root = ds["root"]
        for cls in ds["classes"]:
            paths = load_class_paths(ds_name, ds_root, cls)

            # test
            test_ds = ImageListDataset(paths["test_paths"], img_size=img_size)
            test_loader_x = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            class _L(torch.utils.data.Dataset):
                def __init__(self, base, labels):
                    self.base=base; self.labels=labels
                def __len__(self): return len(self.base)
                def __getitem__(self, i):
                    return self.base[i], int(self.labels[i])
            test_loader = DataLoader(_L(test_ds, paths["test_labels"]), batch_size=batch_size, shuffle=False, num_workers=0)

            # generate synthetic moderate anomalies once
            generate_for_class(ds_name, ds_root, cls, out_gen, method, {severity: prompt},
                               img_per_normal=1, seed=0, limit_normals=limit_normals, severity=severity)
            syn_dir = os.path.join(out_gen, ds_name, cls, severity)

            clean_train = paths["train_normal"][:limit_normals] if limit_normals else paths["train_normal"]
            real_anom_paths = [p for p, y in zip(paths["test_paths"], paths["test_labels"]) if y == 1]
            if limit_real_anoms:
                real_anom_paths = real_anom_paths[:limit_real_anoms]

            for seed in seeds:
                set_seed(seed)
                rng = random.Random(seed)

                for r in ratios:
                    tag = f"contam_{int(r*100)}pct_{severity}"
                    runp = make_run_paths(out_root, exp_name, ds_name, cls, tag, seed)
                    logger = SimpleLogger(os.path.join(runp.logs, "run.log"))
                    logger.log(f"Running Exp3 | {ds_name}/{cls} ratio={r} seed={seed}")

                    # AE: build contaminated input dir by mixing synthetic and "real-like" inputs.
                    # (We maintain pairing with clean targets by always generating inputs from clean images.)
                    contam_dir = os.path.join(runp.root, "contam_inputs")
                    ensure_dir(contam_dir)

                    n = len(clean_train)
                    k_real = int(round(r * n))
                    idxs_real = set(rng.sample(range(n), k=min(k_real, n)))

                    for i, cp in enumerate(clean_train):
                        img = Image.open(cp).convert("RGB")
                        if i in idxs_real:
                            # "real contamination" proxy: stronger corruption
                            pmt = "large crack breaking through the surface"
                        else:
                            pmt = prompt
                        out = generate_anomaly(img, prompt=pmt, method=method, seed=seed + i)
                        out.save(os.path.join(contam_dir, os.path.basename(cp)))

                    paired = PairedDenoiseDataset(syn_dir=contam_dir, clean_paths=clean_train, img_size=img_size)

                    if len(paired) >= 5:
                        ckpt = os.path.join(runp.checkpoints, "ae.pt")
                        model = train_denoising_ae(
                            paired_dataset=paired,
                            epochs=int(ae_cfg["epochs"]),
                            lr=float(ae_cfg["lr"]),
                            batch_size=batch_size,
                            seed=seed,
                            ckpt_path=ckpt,
                            log_path=os.path.join(runp.logs, "ae_train.log"),
                        )
                        scores, ytrue = ae_image_scores(model, test_loader)
                        m = evaluate_and_save(ytrue, scores, runp.figures, prefix="ae", save_curves=save_curves)
                        append_result_row(os.path.join(runp.tables, "results.csv"),
                                          {"experiment": "exp3", "dataset": ds_name, "class": cls,
                                           "model": "ae", "tag": tag, "seed": seed, **m})

                    # PatchCore: contamination violates normal-only assumption (simulate by mixing some corrupted "normals")
                    # Create a "contaminated normal train" by replacing some normal images with synthetic corrupted versions.
                    contam_train_dir = os.path.join(runp.root, "patchcore_contam_train")
                    ensure_dir(contam_train_dir)
                    for i, cp in enumerate(clean_train):
                        img = Image.open(cp).convert("RGB")
                        if i in idxs_real:
                            pmt = "large crack breaking through the surface"
                        else:
                            pmt = "small scratch on the surface"  # mild corruption
                        out = generate_anomaly(img, prompt=pmt, method=method, seed=seed + 999 + i)
                        out.save(os.path.join(contam_train_dir, os.path.basename(cp)))

                    # train embeddings on contaminated images
                    train_ds = ImageListDataset(sorted([os.path.join(contam_train_dir, f) for f in os.listdir(contam_train_dir)]),
                                               img_size=img_size)
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)

                    emb_tr = extract_embeddings(pc_cfg["backbone"], train_loader)
                    nn = fit_nn(emb_tr, k=int(pc_cfg["num_neighbors"]))
                    emb_te = extract_embeddings(pc_cfg["backbone"], test_loader_x)
                    pc_scores = score_nn(nn, emb_te)
                    m2 = evaluate_and_save(paths["test_labels"], pc_scores, runp.figures, prefix="patchcore", save_curves=save_curves)
                    append_result_row(os.path.join(runp.tables, "results.csv"),
                                      {"experiment": "exp3", "dataset": ds_name, "class": cls,
                                       "model": "patchcore", "tag": tag, "seed": seed, **m2})

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
