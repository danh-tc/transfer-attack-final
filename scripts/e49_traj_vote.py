#!/usr/bin/env python3
"""E49 (progress_log 2026-10-30): gộp 3 quỹ đạo RD B36 dev30 (3090) — phép dò "phần phá Swin-L chung giữa quỹ đạo".
Nguồn (cùng ô RD, khác số học): E46 (AMP, TF32), E47 (AMP, --no-tf32, run *_fp32), E48 (RD-NOAMP, fp32).
Sinh trong run dev300_B36_eps5_limit100 (hậu xử lý, không attack lại):
  RD-VOTE3: a = clip(x + ε·sign(δ1 + δ2 + δ3)); tổng = 0 -> dấu Rademacher (seed 2026 + img_id)
  RD-MEAN3: a = clip(x + round(mean δ)) (giữ biên độ, mô tả)
  python scripts/e49_traj_vote.py           # sinh PNG + attack_stats + đo đồng thuận dấu
  python scripts/e49_traj_vote.py corr      # tương quan suppression DINO-Swin-L theo ảnh giữa 3 nguồn (sau eval)"""
import json
import os
import sys
from itertools import combinations

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.mechanism import per_image_transfer

RUN, EPS = "dev300_B36_eps5_limit100", 5.0
ART = os.path.join(REPO_ROOT, "artifacts/runs")
SRCS = {"e46": (RUN, "C2-OCT-1V-WP2-RD"), "e47": (RUN + "_fp32", "C2-OCT-1V-WP2-RD"),
        "e48": (RUN, "C2-OCT-1V-WP2-RD-NOAMP")}
STATS = os.path.join(REPO_ROOT, "results/runs", RUN, "attack_stats.jsonl")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e49_traj_vote.json")


def build():
    data = build_attack_dataset("dev300")
    pos = {i: j for j, i in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    agree, ties, rows = [], [], []
    for img_id in img_ids:
        orig = data[pos[img_id]]["orig_inputs"].float()
        ds = [torch.from_numpy(mmcv.imread(os.path.join(ART, r, "adv", m, f"{img_id}.png"))).permute(2, 0, 1).float() - orig
              for r, m in SRCS.values()]
        signs = [torch.sign(d) for d in ds]
        agree.append([float((a == b).float().mean()) for a, b in combinations(signs, 2)])
        tot = sum(ds)
        s = torch.sign(tot)
        ties.append(float((s == 0).float().mean()))
        rad = torch.randint(0, 2, s.shape, generator=torch.Generator().manual_seed(2026 + img_id)).float() * 2 - 1
        s = torch.where(s == 0, rad, s)
        for name, a in (("RD-VOTE3", orig + EPS * s), ("RD-MEAN3", orig + torch.round(tot / 3))):
            a = torch.clamp(a, 0.0, 255.0)
            assert float((a - orig).abs().max()) <= EPS + 1e-4
            os.makedirs(os.path.join(ART, RUN, "adv", name), exist_ok=True)
            mmcv.imwrite(a.permute(1, 2, 0).numpy().astype(np.uint8), os.path.join(ART, RUN, "adv", name, f"{img_id}.png"))
            dd = (a - orig) / 255.0
            rows.append({"method": name, "img_id": img_id, "linf_255": float((a - orig).abs().max()), "l2": float(dd.norm()),
                         "l2_per_pixel_rms": float(dd.pow(2).mean().sqrt()), "runtime_s": 0.0, "backward": 0,
                         "forward_views": 0, "postprocess_of": "3×C2-OCT-1V-WP2-RD (E46/E47/E48)"})
    with open(STATS, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    agree = np.array(agree)
    print(f"[e49] {len(img_ids)} ảnh; đồng thuận dấu theo cặp (e46-e47, e46-e48, e47-e48) TB {agree.mean(0).round(3).tolist()}; "
          f"tỉ lệ pixel hòa {np.mean(ties):.3f}")
    return {"sign_agree_pairs": agree.mean(0).tolist(), "tie_frac": float(np.mean(ties))}


def corr():
    coco = COCO(os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json"))
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    sup = {}
    for key, (r, m) in list(SRCS.items()) + [("vote3", (RUN, "RD-VOTE3")), ("mean3", (RUN, "RD-MEAN3"))]:
        g = json.load(open(os.path.join(ART, r, "gen_dets", "dino_swin_l.json")))
        sup[key] = 100 * per_image_transfer(coco, g["clean"], g[m], img_ids)["suppression"]
    ok = ~np.isnan(sup["e46"])
    res = {"n_images": int(ok.sum()), "mean_sup": {k: float(np.nanmean(v)) for k, v in sup.items()}, "pearson": {}, "spearman": {}}
    rank = lambda x: np.argsort(np.argsort(x))
    for a, b in combinations(SRCS, 2):
        res["pearson"][f"{a}-{b}"] = float(np.corrcoef(sup[a][ok], sup[b][ok])[0, 1])
        res["spearman"][f"{a}-{b}"] = float(np.corrcoef(rank(sup[a][ok]), rank(sup[b][ok]))[0, 1])
    res["per_image_sd_across_runs"] = float(np.mean(np.std([sup[k][ok] for k in SRCS], axis=0)))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out.update({"corr": corr()} if sys.argv[1:] == ["corr"] else {"build": build()})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
