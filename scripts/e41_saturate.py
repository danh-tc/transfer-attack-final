#!/usr/bin/env python3
"""E41 (progress_log 2026-10-28): hậu xử lý PNG (không attack lại) — giả thuyết "bậc 2 / năng lượng" cho DINO-Swin-L.
Từ PNG `C2-OCT-1V-WP2-RD` B36 (E39), sinh trong CÙNG run:
  C2-OCT-1V-WP2-RD-SAT: a = clip(x + ε·sign(δ_RD)); pixel δ = 0 -> dấu Rademacher (seed 2026 + img_id)
  RAND5               : a = clip(x + ε·r), r Rademacher (seed 2026 + img_id + 1) — sàn "năng lượng thuần"
Ghi PNG vào artifacts/runs/<run>/adv/<tên>/ + dòng attack_stats (runtime / backward = 0: không tốn gradient thêm).
  python scripts/e41_saturate.py"""
import json
import os
import sys

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids

RUN, SRC, EPS = "dev300_B36_eps5_limit100", "C2-OCT-1V-WP2-RD", 5.0
ADV = os.path.join(REPO_ROOT, "artifacts/runs", RUN, "adv")
STATS = os.path.join(REPO_ROOT, "results/runs", RUN, "attack_stats.jsonl")


def main():
    data = build_attack_dataset("dev300")
    pos = {i: j for j, i in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    e_src, e_sat, zero_frac = [], [], []
    rows = []
    for img_id in img_ids:
        orig = data[pos[img_id]]["orig_inputs"].float()
        src = torch.from_numpy(mmcv.imread(os.path.join(ADV, SRC, f"{img_id}.png"))).permute(2, 0, 1).float()
        d = src - orig
        g = torch.Generator().manual_seed(2026 + img_id)
        s = torch.sign(d)
        rad = torch.randint(0, 2, d.shape, generator=g).float() * 2 - 1
        zero_frac.append(float((s == 0).float().mean()))
        s = torch.where(s == 0, rad, s)
        g2 = torch.Generator().manual_seed(2026 + img_id + 1)
        r = torch.randint(0, 2, d.shape, generator=g2).float() * 2 - 1
        for name, sg in (("C2-OCT-1V-WP2-RD-SAT", s), ("RAND5", r)):
            a = torch.clamp(orig + EPS * sg, 0.0, 255.0)
            assert float((a - orig).abs().max()) <= EPS + 1e-4
            os.makedirs(os.path.join(ADV, name), exist_ok=True)
            mmcv.imwrite(a.permute(1, 2, 0).numpy().astype(np.uint8), os.path.join(ADV, name, f"{img_id}.png"))
            dd = (a - orig) / 255.0
            rows.append({"method": name, "img_id": img_id, "linf_255": float((a - orig).abs().max()), "l2": float(dd.norm()),
                         "l2_per_pixel_rms": float(dd.pow(2).mean().sqrt()), "runtime_s": 0.0, "backward": 0,
                         "forward_views": 0, "postprocess_of": SRC if name != "RAND5" else None})
            if name != "RAND5":
                e_sat.append(float((a - orig).pow(2).mean()))
        e_src.append(float(d.pow(2).mean()))
    with open(STATS, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[e41] {len(img_ids)} ảnh; tỉ lệ pixel δ = 0 (RD) TB {np.mean(zero_frac):.3f}; năng lượng TB δ² RD {np.mean(e_src):.2f}"
          f" -> SAT {np.mean(e_sat):.2f} (×{np.mean(e_sat) / np.mean(e_src):.3f})")


if __name__ == "__main__":
    main()
