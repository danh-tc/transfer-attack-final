#!/usr/bin/env python3
"""E12 (quy tắc khóa: progress_log 2026-10-07): nội dung δ của 17 ô (surrogate R50) có chung với δ WB-OSFD (Swin-L)
không, và đa dạng hình học có kéo gần không? CPU, PNG có sẵn, dev30.
cos theo ảnh giữa δ_ô và δ_WB cho HP = δ − G₂*δ (chính) và LP = G₂*δ; mốc 0 = cặp ảnh lệch (i, i+1).
  python scripts/e12_content_align.py
"""
import json
import os
import sys

import cv2
import mmcv
import numpy as np
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from scripts.e6_shape_texture import IMG_DIR
from scripts.e10a_wb_dissect import SRC
from scripts.e11_cell_props import ANN_FILE, CELLS, CROSS3, OUT, RUN, drops

BOOT_SEED, N_BOOT = 2026, 1000


def parts(d):
    lp = cv2.GaussianBlur(d, (0, 0), 2.0)
    return {"HP": d - lp, "LP": lp}


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    X = {i: mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32) for i in ids}
    wb = {i: parts(mmcv.imread(os.path.join(SRC["WB-OSFD"], f"{i}.png")).astype(np.float32) - X[i]) for i in ids}
    C = {c: {k: np.zeros(len(ids)) for k in ("HP", "LP", "HP_null")} for c in CELLS}
    for c in CELLS:
        for n, i in enumerate(ids):
            p = parts(mmcv.imread(os.path.join(RUN, "adv", c, f"{i}.png")).astype(np.float32) - X[i])
            for k in ("HP", "LP"):
                C[c][k][n] = cos(p[k], wb[i][k])
            j = ids[(n + 1) % len(ids)]  # mốc 0: δ_WB của ảnh khác, resize về cùng kích thước
            other = cv2.resize(wb[j]["HP"], (p["HP"].shape[1], p["HP"].shape[0]), interpolation=cv2.INTER_LINEAR)
            C[c]["HP_null"][n] = cos(p["HP"], other)
    rng = np.random.default_rng(BOOT_SEED)
    idx = np.vstack([np.arange(len(ids))[None], rng.integers(0, len(ids), (N_BOOT, len(ids)))])
    mean = {c: {k: v[idx].mean(1) for k, v in C[c].items()} for c in CELLS}  # [1 + N_BOOT]
    ci = lambda v: [float(x) for x in np.percentile(v[1:], [2.5, 97.5])]
    D = drops()
    out = {"cells": CELLS, "cos": {c: {k: {"point": float(v[0]), "ci95": ci(v)} for k, v in mean[c].items()} for c in CELLS}}
    # (a)
    diff = mean["OSFD"]["HP"] - mean["NOISE"]["HP"]
    null = mean["OSFD"]["HP_null"]
    a_ok = ci(diff)[0] > 0 and mean["OSFD"]["HP"][0] > null[0] + 3 * null[1:].std()
    out["a"] = {"OSFD_minus_NOISE_HP": {"point": float(diff[0]), "ci95": ci(diff)},
                "null_OSFD": {"point": float(null[0]), "sd": float(null[1:].std())}, "pass": bool(a_ok)}
    # (b)
    out["b"] = {}
    for k in ("HP", "LP"):
        M = np.array([mean[c][k] for c in CELLS])  # [17, 1 + N_BOOT]
        out["b"][k] = {}
        for t in CROSS3 + ["dino_swin_l", "r101"]:
            r = np.array([spearmanr(M[:, b], D[t])[0] for b in range(M.shape[1])])
            out["b"][k][t] = {"point": float(r[0]), "ci95": ci(r)}
    b = out["b"]["HP"]
    b_ok = b["dino_swin_l"]["point"] >= 0.5 and b["dino_swin_l"]["ci95"][0] > 0 and \
        b["dino_swin_l"]["point"] > np.mean([b[t]["point"] for t in CROSS3])
    out["b"]["pass"] = bool(b_ok)
    out["verdict"] = {(True, True): "(a)∧(b)", (True, False): "(a)∧¬(b)", (False, True): "¬(a), (b)",
                      (False, False): "¬(a)"}[(bool(a_ok), bool(b_ok))]
    with open(os.path.join(OUT, "e12_content_align.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=== cos(δ_ô, δ_WB) trung bình ảnh: HP | LP | HP mốc 0 (ảnh lệch) | drop Swin-L ===")
    for j, c in enumerate(CELLS):
        v = out["cos"][c]
        print(f"{c:<15} {v['HP']['point']:+.4f} | {v['LP']['point']:+.4f} | {v['HP_null']['point']:+.4f} | {D['dino_swin_l'][j]:5.1f}")
    a = out["a"]
    print(f"\n(a) OSFD − NOISE (HP) = {a['OSFD_minus_NOISE_HP']['point']:+.4f} {a['OSFD_minus_NOISE_HP']['ci95']}; "
          f"mốc 0 OSFD {a['null_OSFD']['point']:+.4f} (sd {a['null_OSFD']['sd']:.4f}) -> {'ĐẠT' if a['pass'] else 'không'}")
    for k in ("HP", "LP"):
        print(f"(b) {k}: " + "  ".join(f"{t} {v['point']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]"
                                      for t, v in out["b"][k].items()))
    print("(b) ->", "ĐẠT" if out["b"]["pass"] else "không", "| KẾT LUẬN:", out["verdict"])


if __name__ == "__main__":
    main()
