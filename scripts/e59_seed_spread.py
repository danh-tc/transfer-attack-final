#!/usr/bin/env python3
"""E59 (progress_log 2026-10-30): chênh V100 vs 3090 là phương sai quỹ đạo hay hiệu ứng hệ thống? Quét SEED attack trên CÙNG
3090, RD B36 dev30. Đo: (a) spread DINO-Swin-L drop theo seed; (b) đồng thuận dấu giữa δ của các seed (so E49 same-machine
rerun 0.80, cross-machine chưa đo). Nếu spread theo seed ≈ chênh máy (~±5) -> chênh V100/3090 là phương sai, không tái tạo.
  python scripts/e59_seed_spread.py   (sau khi các seed-run đã attack + eval)"""
import json
import os
import sys
from itertools import combinations

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids

SEEDS = [0, 100, 200, 300]
RUN = lambda s: "dev300_B36_eps5_limit100" + (f"_seed{s}" if s else "")
METHOD = "C2-OCT-1V-WP2-RD"
ART = os.path.join(REPO_ROOT, "artifacts/runs")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e59_seed_spread.json")


def main():
    data = build_attack_dataset("dev300")
    pos = {i: j for j, i in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    # (a) Swin-L drop mỗi seed từ generalization json của run tương ứng
    drops = {}
    for s in SEEDS:
        g = os.path.join(REPO_ROOT, "results/runs", RUN(s), "generalization_e59_dev30.json")
        if os.path.exists(g):
            d = json.load(open(g))
            rel = d.get("relative_ap_drop", d)
            try:
                drops[s] = rel[METHOD]["dino_swin_l"]["point"]
            except Exception:
                drops[s] = None
    # (b) đồng thuận dấu δ giữa các seed (trung bình theo ảnh)
    deltas = {s: {} for s in SEEDS}
    for s in SEEDS:
        for img_id in img_ids:
            p = os.path.join(ART, RUN(s), "adv", METHOD, f"{img_id}.png")
            if os.path.exists(p):
                orig = data[pos[img_id]]["orig_inputs"].float()
                deltas[s][img_id] = torch.sign(torch.from_numpy(mmcv.imread(p)).permute(2, 0, 1).float() - orig)
    agree = {}
    for a, b in combinations(SEEDS, 2):
        common = [i for i in img_ids if i in deltas[a] and i in deltas[b]]
        if common:
            agree[f"{a}-{b}"] = float(np.mean([ (deltas[a][i] == deltas[b][i]).float().mean().item() for i in common ]))
    vals = [v for v in drops.values() if v is not None]
    res = {"swin_l_drop_by_seed": drops,
           "swin_l_spread": {"min": min(vals), "max": max(vals), "range": max(vals) - min(vals), "std": float(np.std(vals))} if vals else {},
           "sign_agree_between_seeds": agree,
           "sign_agree_mean": float(np.mean(list(agree.values()))) if agree else None}
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
