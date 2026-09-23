#!/usr/bin/env python3
"""Mechanism A2 (docs/mechanism_plan.md): forward feature similarity + lan truyền distortion.

(a) linear CKA giữa backbone stage k của R50 và stage k của từng target, trên ảnh sạch.
(b) D_k(m) = ||f_k(x_adv) - f_k(x)|| / ||f_k(x)|| tại từng stage của từng model, với ảnh
    adv PNG của run baseline (artifacts/runs/<run>/adv/) — đúng ảnh đã cho metrics.json.
Kèm chỉ số transfer theo từng ảnh (suppression, retained confidence) từ dets.json của run,
lưu riêng để A1/A3/A4 dùng lại.

Ảnh sạch = `inputs` của AttackDataset; ảnh adv = PNG -> pipeline_resize (như eval baseline).
Chạy: python scripts/mech_a2_features.py [--run n300_B50_eps5]
"""
import argparse
import json
import os
import sys
import time

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset
from attack.mechanism import boot_mean, boot_spearman, boot_weights, linear_cka, per_image_transfer
from attack.models import CONTROLLED_PANEL, SURROGATE_KEY, load_model
from attack.preprocess import pipeline_resize, to_batch

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
METHODS = ["MI-FGSM", "M-DI2-FGSM", "OSFD"]
TARGETS = ["r101", "convnext_t", "swin_t"]
CROSS = ["convnext_t", "swin_t"]
N_STAGE = 4
BOOT_SEED = 2026


@torch.no_grad()
def backbone_feats(model, pixels, ds):
    batch, _ = to_batch(model, [pixels], [ds])
    return model.backbone(batch)


def rel_distortion(fa, fc):
    return float((fa - fc).norm() / fc.norm())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="n300_B50_eps5")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    adv_dir = os.path.join(REPO_ROOT, "artifacts/runs", args.run, "adv")
    out_dir = os.path.join(REPO_ROOT, "results/mechanism")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda:0"

    dataset = build_attack_dataset(args.run.split("_")[0])
    img_ids = list(dataset.image_ids)
    n = len(img_ids)
    models = {k: load_model(k, device) for k in CONTROLLED_PANEL}

    cka = {t: np.zeros((n, N_STAGE)) for t in TARGETS}
    dist = {me: {m: np.zeros((n, N_STAGE)) for m in models} for me in METHODS}
    t0 = time.time()
    for i in range(n):
        s = dataset[i]
        x, ds = s["inputs"].to(device), s["data_sample"]
        clean = {m: backbone_feats(models[m], x, ds) for m in models}
        for t in TARGETS:
            for k in range(N_STAGE):
                cka[t][i, k] = linear_cka(clean[SURROGATE_KEY][k], clean[t][k])
        for me in METHODS:
            png = os.path.join(adv_dir, me, f"{img_ids[i]}.png")
            adv = pipeline_resize(torch.from_numpy(mmcv.imread(png)).permute(2, 0, 1).float().to(device),
                                  x.shape[-2:])
            for m in models:
                fa = backbone_feats(models[m], adv, ds)
                for k in range(N_STAGE):
                    dist[me][m][i, k] = rel_distortion(fa[k], clean[m][k])
        if (i + 1) % 50 == 0:
            print(f"[A2] {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    # Chỉ số transfer per-ảnh từ dets.json của run baseline.
    coco = COCO(ANN_FILE)
    with open(os.path.join(REPO_ROOT, "artifacts/runs", args.run, "dets.json")) as f:
        dets = json.load(f)
    transfer = {me: {m: per_image_transfer(coco, dets["clean"][m], dets[me][m], img_ids)
                     for m in models} for me in METHODS}

    idx = boot_weights(n, args.n_boot, BOOT_SEED)
    S = lambda arr: boot_mean(arr, idx)
    res = {"run": args.run, "n_images": n, "n_boot": args.n_boot, "boot_seed": BOOT_SEED,
           "stages": "backbone stage 1..4 (stride 4/8/16/32)",
           "cka_clean": {}, "cka_same_minus_cross": {}, "distortion": {},
           "separation": {}, "separation_emergence_vs_stage1": {}, "corr_distortion_transfer": {},
           "transfer_per_image_mean": {}}
    for t in TARGETS:
        res["cka_clean"][t] = [S(cka[t][:, k]) for k in range(N_STAGE)]
    for c in CROSS:
        res["cka_same_minus_cross"][c] = [S(cka["r101"][:, k] - cka[c][:, k]) for k in range(N_STAGE)]
    for me in METHODS:
        res["distortion"][me] = {m: [S(dist[me][m][:, k]) for k in range(N_STAGE)] for m in models}
        res["separation"][me], res["separation_emergence_vs_stage1"][me] = {}, {}
        for c in CROSS:
            sep = dist[me]["r101"] - dist[me][c]  # [n, 4]
            res["separation"][me][c] = [S(sep[:, k]) for k in range(N_STAGE)]
            res["separation_emergence_vs_stage1"][me][c] = [S(sep[:, k] - sep[:, 0]) for k in range(1, N_STAGE)]
        res["corr_distortion_transfer"][me] = {
            m: {metric: [boot_spearman(dist[me][m][:, k], transfer[me][m][metric], idx) for k in range(N_STAGE)]
                for metric in ("retained", "suppression")} for m in TARGETS}
        res["transfer_per_image_mean"][me] = {
            m: {metric: S(transfer[me][m][metric]) for metric in ("suppression", "retained")} for m in models}

    with open(os.path.join(out_dir, "a2_features.json"), "w") as f:
        json.dump(res, f, indent=2)
    per_image = {"img_ids": img_ids,
                 "transfer": {me: {m: {k: v.tolist() for k, v in d.items()} for m, d in dd.items()}
                              for me, dd in transfer.items()},
                 "cka_clean": {t: v.tolist() for t, v in cka.items()},
                 "distortion": {me: {m: v.tolist() for m, v in dd.items()} for me, dd in dist.items()}}
    with open(os.path.join(out_dir, "a2_per_image.json"), "w") as f:
        json.dump(per_image, f)

    f1 = lambda s: f"{s['point']:.3f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]"
    print("\n=== (a) CKA(R50, target) ảnh sạch, stage 1..4 ===")
    for t in TARGETS:
        print(f"{t:<11}" + "  ".join(f1(s) for s in res["cka_clean"][t]))
    print("\n=== (b) distortion D_k (mean) ===")
    for me in METHODS:
        for m in models:
            print(f"{me:<11}{m:<11}" + "  ".join(f"{s['point']:.3f}" for s in res["distortion"][me][m]))
    print("\n=== separation D_k(R101) - D_k(cross) [CI] ===")
    for me in METHODS:
        for c in CROSS:
            print(f"{me:<11}{c:<11}" + "  ".join(f1(s) for s in res["separation"][me][c]))
    print("\n=== separation_k - separation_1 (k=2..4) [CI] ===")
    for me in METHODS:
        for c in CROSS:
            print(f"{me:<11}{c:<11}" + "  ".join(f1(s) for s in res["separation_emergence_vs_stage1"][me][c]))
    print("\n=== Spearman(D_k, retained confidence) theo target ===")
    for me in METHODS:
        for m in TARGETS:
            print(f"{me:<11}{m:<11}" + "  ".join(f1(s) for s in res["corr_distortion_transfer"][me][m]["retained"]))
    print("\n=== transfer per-ảnh (mean): suppression / retained ===")
    for me in METHODS:
        print(f"{me:<11}" + "  ".join(f"{m}: {res['transfer_per_image_mean'][me][m]['suppression']['point']:.3f}/"
                                       f"{res['transfer_per_image_mean'][me][m]['retained']['point']:.3f}"
                                       for m in models))
    print("saved:", os.path.join(out_dir, "a2_features.json"))


if __name__ == "__main__":
    main()
