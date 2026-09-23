#!/usr/bin/env python3
"""Đường bão hòa theo budget B của OSFD (không phải baseline table chính thức).

Chạy OSFD 1 lần với B_max, chụp noise sau mỗi mốc B trong CHECKPOINTS (noise sau k
step đúng bằng lần chạy steps=k — xem on_step trong attack/methods/core.py), đánh
giá từng mốc trên cả 4 model Controlled Panel qua ẢNH THẬT (uint8 cỡ gốc -> Resize
của pipeline, như scripts/quick_transfer_check.py). Tốn đúng bằng 1 lần chạy B_max.

Chỉ để lập kế hoạch runtime/budget — B primary vẫn là {50, 200} theo
docs/protocol_lock.md cho tới khi có quyết định ghi vào progress_log.

Chạy: python scripts/budget_sweep.py [n_images] [B_max]
"""
import json
import os
import random
import sys
import time

import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from attack.data import build_attack_dataset
from attack.methods.baselines import osfd_attack
from attack.models import CONTROLLED_PANEL, SURROGATE_KEY, load_model
from attack.preprocess import pipeline_resize, predict, to_adv_image
from quick_transfer_check import ANN_FILE, EPSILON, SEED, coco_ap, to_coco_dets

CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200]


def main():
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    b_max = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    checkpoints = [b for b in CHECKPOINTS if b <= b_max]
    device = "cuda:0"
    models = {k: load_model(k, device) for k in CONTROLLED_PANEL}
    surrogate = models[SURROGATE_KEY]
    dataset = build_attack_dataset("n300")
    cat_ids = dataset._mmdet_dataset.cat_ids
    coco_gt = COCO(ANN_FILE)

    conds = ["clean"] + [f"B{b}" for b in checkpoints]
    dets = {c: {k: [] for k in models} for c in conds}
    img_ids, t_attack = [], 0.0

    for i in range(n_images):
        sample = dataset[i]
        img_id, x, ds = sample["img_id"], sample["inputs"].to(device), sample["data_sample"]
        orig = sample["orig_inputs"].to(device)
        img_ids.append(img_id)

        snaps = {}
        def on_step(k, noise):
            if k in checkpoints:
                snaps[k] = noise.clone()

        random.seed(SEED + i); torch.manual_seed(SEED + i)
        t0 = time.time()
        osfd_attack(surrogate, orig, ds, steps=b_max, epsilon=EPSILON, on_step=on_step)
        torch.cuda.synchronize(); t_attack += time.time() - t0

        advs = {"clean": x}
        for b in checkpoints:
            advs[f"B{b}"] = pipeline_resize(to_adv_image(orig, snaps[b]), x.shape[-2:])
        for c, img in advs.items():
            for k, m in models.items():
                dets[c][k] += to_coco_dets(predict(m, img, ds, rescale=True), img_id, cat_ids)
        print(f"[{i + 1}/{n_images}] img_id={img_id} ({time.time() - t0:.1f}s)", flush=True)

    ap = {c: {k: coco_ap(coco_gt, dets[c][k], img_ids)[0] for k in models} for c in conds}
    drop = {c: {k: 100 * (ap["clean"][k] - ap[c][k]) / ap["clean"][k] for k in models}
            for c in conds if c != "clean"}

    keys = list(models)
    print(f"\n=== OSFD relative AP drop (%) theo B (n={n_images}, eps={EPSILON}, ảnh thật) ===")
    print(f"{'B':<6}" + "".join(f"{k:>12}" for k in keys) + f"{'cross-avg':>12}")
    for c in drop:
        cross = (drop[c]["convnext_t"] + drop[c]["swin_t"]) / 2
        print(f"{c[1:]:<6}" + "".join(f"{drop[c][k]:>12.1f}" for k in keys) + f"{cross:>12.1f}")
    print(f"\nattack runtime/ảnh (B={b_max}): {t_attack / n_images:.1f}s")

    out_dir = os.path.join(REPO_ROOT, "results/budget_sweep")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"osfd_n{n_images}_Bmax{b_max}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump({"method": "OSFD", "n_images": n_images, "b_max": b_max, "epsilon": EPSILON,
                   "eval": "image-space uint8", "img_ids": img_ids, "ap": ap,
                   "relative_ap_drop": drop,
                   "attack_runtime_s_per_img": t_attack / n_images}, f, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
