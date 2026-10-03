#!/usr/bin/env python3
"""Chẩn đoán cơ chế pilot R1 (KHÔNG phải tiêu chí — docs/progress_log.md 2026-10-03).

20 ảnh đầu dev100 (seed = SEED + chỉ số dev300, như run_baselines), sau 10 step E0 (OSFD), E1a (SGM
γ = 0.2), E1b (SGM γ = 0.5): ảnh adv uint8 -> Resize pipeline -> backbone R50/R101/ConvNeXt-T/Swin-T,
D_k = ‖f_k(adv) − f_k(clean)‖ / ‖f_k(clean)‖ (định nghĩa A2), báo cáo D_k và D_4/D_1 theo model.

Chạy: python scripts/mech_r1_diag.py   -> results/r1_pilot/dev300/mech_diag.json
"""
import json
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.methods.baselines import osfd_attack
from attack.methods.sgm import sgm_osfd_attack
from attack.models import CONTROLLED_PANEL, SURROGATE_KEY, load_model
from attack.preprocess import pipeline_resize, to_adv_image, to_batch

SEED, N_IMG, STEPS = 0, 20, 10
CONDS = {"E0": (osfd_attack, {}), "E1a": (sgm_osfd_attack, {"gamma": 0.2}),
         "E1b": (sgm_osfd_attack, {"gamma": 0.5})}


@torch.no_grad()
def feats(model, pixels, ds):
    return model.backbone(to_batch(model, [pixels], [ds])[0])


def main():
    dev = "cuda:0"
    models = {k: load_model(k, dev) for k in CONTROLLED_PANEL}
    sur = models[SURROGATE_KEY]
    data = build_attack_dataset("dev300")
    pos = {img_id: i for i, img_id in enumerate(data.image_ids)}
    order = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:N_IMG]
    D = {c: {k: [] for k in models} for c in CONDS}
    for img_id in order:
        i = pos[img_id]
        s = data[i]
        orig, ds, x = s["orig_inputs"].to(dev), s["data_sample"], s["inputs"].to(dev)
        clean = {k: feats(m, x, ds) for k, m in models.items()}
        for c, (fn, kw) in CONDS.items():
            random.seed(SEED + i); torch.manual_seed(SEED + i)
            noise = fn(sur, orig, ds, steps=STEPS, **kw)
            adv = pipeline_resize(to_adv_image(orig, noise), x.shape[-2:])
            for k, m in models.items():
                fa = feats(m, adv, ds)
                D[c][k].append([float((a - b).norm() / b.norm()) for a, b in zip(fa, clean[k])])
        print(f"[diag] img {img_id} xong", flush=True)
    out = {"rule": "progress_log 2026-10-03 (pilot R1) — chẩn đoán, không phải tiêu chí",
           "n_images": N_IMG, "steps": STEPS, "images": order, "D_k_mean": {}, "D4_over_D1_mean": {}}
    for c in CONDS:
        out["D_k_mean"][c] = {k: np.mean(v, axis=0).round(4).tolist() for k, v in D[c].items()}
        out["D4_over_D1_mean"][c] = {k: float(np.mean([r[3] / r[0] for r in v])) for k, v in D[c].items()}
    path = os.path.join(REPO_ROOT, "results/r1_pilot/dev300/mech_diag.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for c in CONDS:
        print(c, {k: v for k, v in out["D_k_mean"][c].items()}, {k: round(v, 2) for k, v in out["D4_over_D1_mean"][c].items()})


if __name__ == "__main__":
    main()
