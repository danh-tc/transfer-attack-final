#!/usr/bin/env python3
"""E21 (progress_log 2026-10-24): δ có thực sự di chuyển khỏi δ_S? Trung bình ảnh dev30 của cos(δ_B, δ_B0) và
‖δ_B − δ_B0‖ / ‖δ_B0‖, B ∈ {1, 4, 12, 36}, cho OTSV50-IS và WP2-IS (δ = PNG − ảnh gốc). Ghi results/otsv/delta_move.json."""
import json
import os
import sys

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids

coco = COCO(os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json"))
ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
art = os.path.join(REPO_ROOT, "artifacts/runs")
rd = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
out = {}
for m in ("C2-OCT-1V-WP2-OTSV50-IS", "C2-OCT-1V-WP2-IS"):
    for b in (1, 4, 12, 36):
        cos, rel = [], []
        for i in ids:
            x = rd(os.path.join(REPO_ROOT, "data/coco/val2017", coco.loadImgs(i)[0]["file_name"]))
            d0 = rd(f"{art}/dev300_B41_eps5_limit100_sur-swin_t/adv/C2-OCT-1V/{i}.png") - x
            d = rd(f"{art}/dev300_B{b}_eps5_limit100/adv/{m}/{i}.png") - x
            cos.append(float((d * d0).sum() / (np.linalg.norm(d) * np.linalg.norm(d0) + 1e-12)))
            rel.append(float(np.linalg.norm(d - d0) / (np.linalg.norm(d0) + 1e-12)))
        out[f"{m}@B{b}"] = {"cos": float(np.mean(cos)), "rel_l2": float(np.mean(rel))}
        print(f"{m} B{b}: cos(δ, δ_S) {np.mean(cos):.3f} | ‖δ − δ_S‖/‖δ_S‖ {np.mean(rel):.3f}")
with open(os.path.join(REPO_ROOT, "results/otsv/delta_move.json"), "w") as f:
    json.dump(out, f, indent=1)
