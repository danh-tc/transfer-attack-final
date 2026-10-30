#!/usr/bin/env python3
"""E50 (progress_log 2026-10-30): vai trò của pixel δ NẰM Ở NỀN — che δ_RD B36 dev30 (PNG E46, 3090) theo GT box.
Vùng (khung ảnh gốc, hộp COCO mọi annotation kể cả crowd): IN = hợp các box; OUT = phần còn lại;
RING = hợp các box nới mỗi cạnh 25% cạnh box, trừ IN (ngữ cảnh sát object). FULL = RD nguyên.
Sinh trong run dev300_B36_eps5_limit100 (hậu xử lý, không attack lại): RD-IN / RD-OUT / RD-RING = clip(x + δ·mask).
  python scripts/e50_bg_mask.py        # sinh PNG + attack_stats + tỉ lệ diện tích vùng
  python scripts/e50_bg_mask.py fp     # FP / ảnh (nền | sát object, như err_profile) mọi model panel (sau eval)"""
import json
import os
import sys
from collections import defaultdict

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from scripts.err_profile import SCORE, profile
from scripts.eval_generalization import PANEL_GEN

RUN, SRC = "dev300_B36_eps5_limit100", "C2-OCT-1V-WP2-RD"
ART = os.path.join(REPO_ROOT, "artifacts/runs")
STATS = os.path.join(REPO_ROOT, "results/runs", RUN, "attack_stats.jsonl")
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e50_bg_mask.json")
CELLS = [SRC, "RD-IN", "RD-OUT", "RD-RING"]
CTRL = ["r50", "r101", "convnext_t", "swin_t"]
RING = 0.25


def masks(coco, img_id, h, w):
    inside, grown = np.zeros((h, w), bool), np.zeros((h, w), bool)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        x, y, bw, bh = a["bbox"]
        inside[int(y):int(np.ceil(y + bh)), int(x):int(np.ceil(x + bw))] = True
        x0, y0 = max(0, int(x - RING * bw)), max(0, int(y - RING * bh))
        grown[y0:int(np.ceil(y + bh + RING * bh)), x0:int(np.ceil(x + bw + RING * bw))] = True
    return {"RD-IN": inside, "RD-OUT": ~inside, "RD-RING": grown & ~inside}


def build():
    coco = COCO(ANN)
    data = build_attack_dataset("dev300")
    pos = {i: j for j, i in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    frac, rows = defaultdict(list), []
    for img_id in img_ids:
        orig = data[pos[img_id]]["orig_inputs"].float()
        d = torch.from_numpy(mmcv.imread(os.path.join(ART, RUN, "adv", SRC, f"{img_id}.png"))).permute(2, 0, 1).float() - orig
        for name, m in masks(coco, img_id, *d.shape[1:]).items():
            frac[name].append(float(m.mean()))
            a = torch.clamp(orig + d * torch.from_numpy(m)[None].float(), 0.0, 255.0)
            os.makedirs(os.path.join(ART, RUN, "adv", name), exist_ok=True)
            mmcv.imwrite(a.permute(1, 2, 0).numpy().astype(np.uint8), os.path.join(ART, RUN, "adv", name, f"{img_id}.png"))
            dd = (a - orig) / 255.0
            rows.append({"method": name, "img_id": img_id, "linf_255": float((a - orig).abs().max()), "l2": float(dd.norm()),
                         "l2_per_pixel_rms": float(dd.pow(2).mean().sqrt()), "runtime_s": 0.0, "backward": 0,
                         "forward_views": 0, "postprocess_of": SRC + " (E46)"})
    with open(STATS, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = {k: float(np.mean(v)) for k, v in frac.items()}
    print(f"[e50] {len(img_ids)} ảnh; tỉ lệ diện tích TB {res}")
    return {"area_frac": res}


def fp():
    coco = COCO(ANN)
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    ctrl = json.load(open(os.path.join(ART, RUN, "dets_e50_dev30.json")))
    res = {}
    for k in CTRL + PANEL_GEN:
        src = ctrl if k in CTRL else json.load(open(os.path.join(ART, RUN, "gen_dets", f"{k}.json")))
        res[k] = {}
        for name in ["clean"] + CELLS:
            ds = src["clean"][k] if (k in CTRL and name == "clean") else (src[name][k] if k in CTRL else src[name])
            by = defaultdict(list)
            for x in ds:
                if x["score"] >= SCORE:
                    by[x["image_id"]].append(x)
            p, _ = profile(coco, by, img_ids)
            res[k][name] = {t: round(p["fp_per_img"].get(t, 0.0), 2) for t in ("nền", "sát object", "tổng")}
    for k, v in res.items():
        print(k, " | ".join(f"{n}: nền {s['nền']} / sát {s['sát object']}" for n, s in v.items()))
    return res


if __name__ == "__main__":
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out.update({"fp_per_img": fp()} if sys.argv[1:] == ["fp"] else {"build": build()})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
