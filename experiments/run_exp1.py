import os
import yaml
from torch.utils.data import DataLoader
import pandas as pd

from src.datasets import load_class_paths, ImageListDataset, PairedDenoiseDataset, list_mvtec_classes, list_visa_classes
from src.utils import make_run_paths, set_seed, device, SimpleLogger
from src.generate_anomalies import generate_for_class
from src.train_autoencoder import train_denoising_ae, ae_image_scores
from src.train_patchcore import extract_embeddings, fit_nn, score_nn
from src.evaluate import evaluate_and_save, append_result_row

def run(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = "outputs"
    exp_name = "exp1"
    gen = cfg["generator"]
    method = gen.get("method", "fallback")
    out_gen = gen.get("out_root", "data/generated")
    prompts = gen["prompts"]
    img_per_normal = int(gen.get("images_per_normal", 1))

    tr = cfg["training"]
    seeds = tr["seeds"]
    limit_normals = int(tr.get("limit_normals", 0))
    img_size = int(tr.get("img_size", 256))
    batch_size = int(tr.get("batch_size", 16))

    ae_cfg = tr["ae"]
    pc_cfg = tr["patchcore"]

    save_curves = bool(cfg["evaluation"].get("save_curves", True))

    for ds in cfg["datasets"]:
        ds_name = ds["name"]
        ds_root = ds["root"]
        classes = ds["classes"]
        if classes == ["all"]:
            classes = list_mvtec_classes(ds_root) if ds_name.lower() == "mvtec" else list_visa_classes(ds_root)

        for cls in classes:
            # load real test set
            paths = load_class_paths(ds_name, ds_root, cls)
            test_ds = ImageListDataset(paths["test_paths"], img_size=img_size)
            test_loader_x = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            # For AE scoring we need labels aligned; build a loader that yields (x,label)
            import torch
            class _L(torch.utils.data.Dataset):
                def __init__(self, base, labels):
                    self.base=base; self.labels=labels
                def __len__(self): return len(self.base)
                def __getitem__(self, i):
                    return self.base[i], int(self.labels[i])
            test_loader = DataLoader(_L(test_ds, paths["test_labels"]), batch_size=batch_size, shuffle=False, num_workers=0)

            for severity in prompts.keys():
                # generate once (deterministic)
                generate_for_class(ds_name, ds_root, cls, out_gen, method, prompts, img_per_normal,
                                   seed=0, limit_normals=limit_normals, severity=severity)
                syn_dir = os.path.join(out_gen, ds_name, cls, severity)

                for seed in seeds:
                    tag = f"{severity}"
                    runp = make_run_paths(out_root, exp_name, ds_name, cls, tag, seed)
                    logger = SimpleLogger(os.path.join(runp.logs, "run.log"))
                    logger.log(f"Running Exp1 | {ds_name}/{cls} severity={severity} seed={seed}")

                    # AE denoising training: syn -> clean
                    clean_train = paths["train_normal"][:limit_normals] if limit_normals else paths["train_normal"]
                    paired = PairedDenoiseDataset(syn_dir=syn_dir, clean_paths=clean_train, img_size=img_size)

                    if len(paired) < 5:
                        logger.log("Not enough paired samples. Skipping AE.")
                    else:
                        ckpt = os.path.join(runp.checkpoints, "ae.pt")
                        model = train_denoising_ae(
                            paired_dataset=paired,
                            epochs=int(ae_cfg["epochs"]),
                            lr=float(ae_cfg["lr"]),
                            batch_size=batch_size,
                            seed=seed,
                            ckpt_path=ckpt,
                            log_path=os.path.join(runp.logs, "ae_train.log")
                        )
                        scores, ytrue = ae_image_scores(model, test_loader)
                        m = evaluate_and_save(ytrue, scores, runp.figures, prefix="ae", save_curves=save_curves)
                        append_result_row(
                            os.path.join(runp.tables, "results.csv"),
                            {"experiment": "exp1", "dataset": ds_name, "class": cls, "model": "ae",
                             "tag": severity, "seed": seed, **m}
                        )

                    # PatchCore-like baseline
                    train_normals = paths["train_normal"][:limit_normals] if limit_normals else paths["train_normal"]
                    train_ds = ImageListDataset(train_normals, img_size=img_size)
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)

                    emb_tr = extract_embeddings(pc_cfg["backbone"], train_loader)
                    nn = fit_nn(emb_tr, k=int(pc_cfg["num_neighbors"]))
                    emb_te = extract_embeddings(pc_cfg["backbone"], test_loader_x)
                    pc_scores = score_nn(nn, emb_te)
                    m2 = evaluate_and_save(paths["test_labels"], pc_scores, runp.figures, prefix="patchcore", save_curves=save_curves)
                    append_result_row(
                        os.path.join(runp.tables, "results.csv"),
                        {"experiment": "exp1", "dataset": ds_name, "class": cls, "model": "patchcore",
                         "tag": severity, "seed": seed, **m2}
                    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
