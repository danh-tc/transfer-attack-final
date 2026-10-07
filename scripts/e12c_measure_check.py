#!/usr/bin/env python3
"""E12c (quy tắc khóa: progress_log 2026-10-07): cos_HP (E12) đo phần chung hay độ mạnh / bão hòa của δ?
(i) giữa 21 ô: sat, cos_HP, cos_sgn; tương quan riêng | sat trên 17 ô E12. (ii) quỹ đạo OSFD b ∈ {10..50} trên dev30.
  python scripts/e12c_measure_check.py [--skip-traj]
"""
import argparse
import json
import os
import random
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.preprocess import to_adv_image
from scripts.e6_shape_texture import IMG_DIR, TARGETS
from scripts.e10a_wb_dissect import SRC
from scripts.e11_cell_props import ANN_FILE, CELLS as CELLS17, CROSS3, OUT, RUN, drops
from scripts.e12_content_align import cos, parts

VC = ["OSFD-K2", "VC-OSFD", "OSFD-K4", "VC4-OSFD"]
SNAPS = [10, 20, 30, 40, 50]
TRAJ = os.path.join(REPO_ROOT, "artifacts/e12c")
EPS = 5


def measures(d, w):
    return {"sat": float((np.abs(d) >= EPS).mean()), "cos_HP": cos(parts(d)["HP"], w),
            "cos_sgn": cos(parts(np.sign(d))["HP"], w)}


def between_cells(coco, ids):
    X = {i: mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32) for i in ids}
    W = {i: parts(mmcv.imread(os.path.join(SRC["WB-OSFD"], f"{i}.png")).astype(np.float32) - X[i])["HP"] for i in ids}
    M = {}
    for c in CELLS17 + VC:
        r = [measures(mmcv.imread(os.path.join(RUN, "adv", c, f"{i}.png")).astype(np.float32) - X[i], W[i]) for i in ids]
        M[c] = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
    return M, X, W


def trajectory(coco, ids, X, W):
    from mmdet.apis import inference_detector
    from attack.methods.baselines import osfd_attack
    from attack.models import load_surrogate
    from attack.targets import load_any
    split = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    model = load_surrogate("cuda:0")
    ds = AttackDataset(ids)
    for n in range(len(ds)):
        s = ds[n]
        i = s["img_id"]
        if all(os.path.exists(os.path.join(TRAJ, f"b{b}", f"{i}.png")) for b in SNAPS):
            continue
        from scripts.run_baselines import SEED
        random.seed(SEED + split.index(i)); torch.manual_seed(SEED + split.index(i))
        orig = s["orig_inputs"].cuda()

        def on_step(b, noise, orig=orig, i=i):
            if b in SNAPS:
                os.makedirs(os.path.join(TRAJ, f"b{b}"), exist_ok=True)
                adv = to_adv_image(orig, noise)
                mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), os.path.join(TRAJ, f"b{b}", f"{i}.png"))
        osfd_attack(model, orig, s["data_sample"], steps=50, on_step=on_step)
        print(f"[E12c traj] {n + 1}/{len(ds)}", flush=True)
    del model
    torch.cuda.empty_cache()
    cat_ids = sorted(coco.getCatIds())
    out = {b: {} for b in SNAPS}
    for b in SNAPS:
        r = [measures(mmcv.imread(os.path.join(TRAJ, f"b{b}", f"{i}.png")).astype(np.float32) - X[i], W[i]) for i in ids]
        out[b] = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
    for t in TARGETS:
        m = load_any(t)
        dets = {"clean": []} | {b: [] for b in SNAPS}
        for i in ids:
            dets["clean"] += to_coco_dets(inference_detector(m, X[i].astype(np.uint8)), i, cat_ids)
            for b in SNAPS:
                dets[b] += to_coco_dets(inference_detector(m, mmcv.imread(os.path.join(TRAJ, f"b{b}", f"{i}.png"))), i, cat_ids)
        ap = lambda k: CocoImageEval(coco, dets[k], ids).ap(np.ones(len(ids)))[0]
        a0 = ap("clean")
        for b in SNAPS:
            out[b][t] = float(100 * (a0 - ap(b)) / a0)
        del m
        torch.cuda.empty_cache()
        print(f"[E12c eval] {t}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-traj", action="store_true")
    args = p.parse_args()
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    M, X, W = between_cells(coco, ids)
    D = drops()
    res = {"cells": M}
    all_c = CELLS17 + VC
    res["rho_sat_cosHP_21"] = float(spearmanr([M[c]["sat"] for c in all_c], [M[c]["cos_HP"] for c in all_c])[0])
    sat = np.array([M[c]["sat"] for c in CELLS17]); ch = np.array([M[c]["cos_HP"] for c in CELLS17])
    resid = ch - np.polyval(np.polyfit(sat, ch, 1), sat)
    ca3 = np.mean([D[t] for t in CROSS3], 0)
    res["partial_17"] = {"swin_l": float(spearmanr(resid, D["dino_swin_l"])[0]), "ca3": float(spearmanr(resid, ca3)[0]),
                         "r101": float(spearmanr(resid, D["r101"])[0])}
    cs = np.array([M[c]["cos_sgn"] for c in CELLS17])
    res["cos_sgn_17"] = {"swin_l": float(spearmanr(cs, D["dino_swin_l"])[0]), "ca3": float(spearmanr(cs, ca3)[0]),
                         "r101": float(spearmanr(cs, D["r101"])[0])}
    print("=== Giữa ô: sat | cos_HP | cos_sgn ===")
    for c in all_c:
        print(f"{c:<15} {M[c]['sat']:.3f} | {M[c]['cos_HP']:+.4f} | {M[c]['cos_sgn']:+.4f}")
    print(f"ρ(sat, cos_HP) 21 ô = {res['rho_sat_cosHP_21']:+.2f}")
    print("Tương quan riêng cos_HP | sat (17 ô):", {k: round(v, 2) for k, v in res["partial_17"].items()})
    print("cos_sgn (17 ô):", {k: round(v, 2) for k, v in res["cos_sgn_17"].items()})
    if not args.skip_traj:
        T = trajectory(coco, ids, X, W)
        res["trajectory"] = T
        ca3t = [np.mean([T[b][t] for t in CROSS3]) for b in SNAPS]
        res["rho_traj_cosHP_ca3"] = float(spearmanr([T[b]["cos_HP"] for b in SNAPS], ca3t)[0])
        print("\n=== Quỹ đạo OSFD: b | sat | cos_HP | cos_sgn | R50 R101 ConvNeXt Swin-T YOLOX Swin-L | CA3 ===")
        for b, c3 in zip(SNAPS, ca3t):
            v = T[b]
            print(f"{b:>3} {v['sat']:.3f} {v['cos_HP']:+.4f} {v['cos_sgn']:+.4f} | " +
                  " ".join(f"{v[t]:5.1f}" for t in TARGETS) + f" | {c3:.1f}")
        a, b_ = res["partial_17"]["swin_l"], res["rho_traj_cosHP_ca3"]
        res["verdict"] = "GIỮ" if (a >= 0.4 and b_ >= 0) else "LOẠI" if (a < 0.2 and b_ < 0) else "XÁM"
        print(f"\nQUY TẮC: (a) riêng|sat Swin-L = {a:+.2f}; (b) ρ quỹ đạo(cos_HP, CA3) = {b_:+.2f} -> {res['verdict']}")
    with open(os.path.join(OUT, "e12c_measure_check.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
