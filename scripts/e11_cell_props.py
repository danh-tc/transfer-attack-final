#!/usr/bin/env python3
"""E11 (quy tắc khóa: progress_log 2026-10-07): đặc tính δ nào (phía R50 / δ) đi theo transfer GIỮA 17 ô dev30?
P1 = E_surv − E_noise8 trên R50 (từ e2_per_image_dev30.json); P2 = tỉ lệ năng lượng dải thấp của δ (A4);
P3 = năng lượng δ trong GT box / diện tích box (A4); P4 = F_R50(LP2) — phần tác dụng còn lại trên R50 sau LP σ = 2 px.
Spearman giữa ô (đặc tính trung bình ô ↔ drop dev30 đã có), CI bootstrap ảnh trên đặc tính (drop cố định).
  python scripts/e11_cell_props.py [--analyze-only]
"""
import argparse
import json
import os
import sys
import time

import cv2
import mmcv
import numpy as np
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from scripts.e4_target_side import matched
from scripts.e6_shape_texture import IMG_DIR
from scripts.mech_a4_spectral import band_fracs, spatial

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
RUN = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5")
RES = os.path.join(REPO_ROOT, "results/runs/dev300_B50_eps5")
OUT = os.path.join(REPO_ROOT, "results/mechanism")
DETS = os.path.join(REPO_ROOT, "artifacts/e11/dets_r50_lp2.json")
CELLS = ["OSFD", "NOISE", "FLIP", "PIA", "PIA-noT", "CAM-R15", "CAM-R30", "ROT7", "ROT15", "ROT30", "SHR15", "SHR30",
         "ANI10", "ANI25", "SCL25", "PROBE-osfd+cls", "C2-PIA"]
TARGETS = ["convnext_t", "swin_t", "yolox_s", "dino_swin_l", "r101"]
CROSS3 = ["convnext_t", "swin_t", "yolox_s"]
BOOT_SEED, N_BOOT, SCORE = 2026, 1000, 0.3


def drops():
    d = {}
    for tag in ("elem", "e9"):
        for f in (f"metrics_{tag}_dev30.json", f"generalization_{tag}_dev30.json"):
            for m, v in json.load(open(os.path.join(RES, f)))["relative_ap_drop"].items():
                for t, x in v.items():
                    d.setdefault(m, {}).setdefault(t, x["point"] if isinstance(x, dict) else x)
    return {t: np.array([d[c][t] for c in CELLS]) for t in TARGETS}


def run_lp2(coco, ids):
    """R50 (inference_detector, như E10a) trên x + δ và x + LP2(δ) cho từng ô."""
    from mmdet.apis import inference_detector
    from attack.targets import load_any
    cat_ids = sorted(coco.getCatIds())
    model = load_any("r50")
    dets = {f"{c}|{k}": [] for c in CELLS for k in ("ID", "LP2")}
    t0 = time.time()
    for n, i in enumerate(ids):
        x = mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32)
        for c in CELLS:
            d = mmcv.imread(os.path.join(RUN, "adv", c, f"{i}.png")).astype(np.float32) - x
            for k, dd in (("ID", d), ("LP2", cv2.GaussianBlur(d, (0, 0), 2.0))):
                im = np.clip(np.round(x + dd), 0, 255).astype(np.uint8)
                dets[f"{c}|{k}"] += to_coco_dets(inference_detector(model, im), i, cat_ids)
        if (n + 1) % 10 == 0:
            print(f"[E11 r50] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(DETS), exist_ok=True)
    with open(DETS, "w") as f:
        json.dump(dets, f)


def per_image_props(coco, ids):
    """[cell][prop] -> mảng theo ảnh (P1 E2, P2, P3); P4 trả riêng dạng (attack ID, attack LP2) theo ảnh."""
    e2 = json.load(open(os.path.join(OUT, "e2_per_image_dev30.json")))
    assert sorted(int(i) for i in e2["img_ids"]) == sorted(ids), "tập ảnh E2 ≠ dev30"
    clean = matched(coco, json.load(open(os.path.join(REPO_ROOT, "artifacts/e10/dets.json")))["r50"]["clean"], ids)
    lp = json.load(open(DETS))
    P = {}
    for c in CELLS:
        r = e2["retained_r50"][c]
        tn = [k for k in next(iter(r.values())) if k not in ("ID", "noise8")]
        surv = np.array([1 - np.nanmean([r[str(i)][k] for k in tn]) for i in ids])
        noise = np.array([1 - r[str(i)]["noise8"] for i in ids])
        bf, ib = [], []
        for i in ids:
            x = mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32)
            d = (mmcv.imread(os.path.join(RUN, "adv", c, f"{i}.png")).astype(np.float32) - x).transpose(2, 0, 1)
            bf.append(band_fracs(d)[0])
            ib.append(spatial(d, [a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))])[1])
        a = {}
        for k in ("ID", "LP2"):
            cc = matched(coco, lp[f"{c}|{k}"], ids)
            a[k] = np.array([[np.sum([1 - sa / sc for sc, sa in zip(clean[i][1], cc[i][1]) if sc >= SCORE]),
                              np.sum([1 for sc in clean[i][1] if sc >= SCORE])] for i in ids])  # (tổng 1−retained, số GT)
        P[c] = {"P1": surv - noise, "P2": np.array(bf), "P3": np.array(ib), "P4": a}
    return P


def cell_means(P, w):
    """w [n] trọng số ảnh -> {prop: [17]} trung bình ô."""
    m = lambda v: np.nansum(w * v) / np.sum(w * ~np.isnan(v))
    out = {k: np.array([m(P[c][k]) for c in CELLS]) for k in ("P1", "P2", "P3")}
    out["P4"] = np.array([(w @ P[c]["P4"]["LP2"][:, 0]) / max(w @ P[c]["P4"]["ID"][:, 0], 1e-9) for c in CELLS])
    return out


def analyze(coco, ids):
    P = per_image_props(coco, ids)
    D = drops()
    rng = np.random.default_rng(BOOT_SEED)
    Ws = [np.ones(len(ids))] + [np.bincount(r, minlength=len(ids)).astype(float)
                                for r in rng.integers(0, len(ids), (N_BOOT, len(ids)))]
    M = [cell_means(P, w) for w in Ws]
    props = ["P1", "P2", "P3", "P4"]
    out = {"cells": CELLS, "drops": {t: D[t].tolist() for t in TARGETS},
           "cell_props": {k: M[0][k].tolist() for k in props}, "rho": {}, "verdict": {}}
    for k in props:
        out["rho"][k] = {}
        for t in TARGETS:
            r = np.array([spearmanr(m[k], D[t])[0] for m in M])
            out["rho"][k][t] = {"point": float(r[0]), "ci95": [float(x) for x in np.nanpercentile(r[1:], [2.5, 97.5])]}
        ok = [t for t in CROSS3 if out["rho"][k][t]["point"] >= 0.5 and out["rho"][k][t]["ci95"][0] > 0
              and out["rho"][k][t]["point"] > out["rho"][k]["r101"]["point"]]
        out["verdict"][k] = {"targets_pass": ok, "pass": len(ok) >= 2}
    with open(os.path.join(OUT, "e11_cell_props.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=== Đặc tính trung bình ô (P1 bền hình học thuần, P2 dải thấp, P3 in-box, P4 F_R50(LP2)) | drop CA3 / Swin-L ===")
    for j, c in enumerate(CELLS):
        ca3 = np.mean([D[t][j] for t in CROSS3])
        print(f"{c:<15}" + "".join(f"{M[0][k][j]:8.3f}" for k in props) + f"   | {ca3:5.1f} / {D['dino_swin_l'][j]:5.1f}")
    print("\n=== Spearman giữa 17 ô [CI bootstrap ảnh] ===")
    for k in props:
        print(f"{k}: " + "  ".join(f"{t} {v['point']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]"
                                   for t, v in out["rho"][k].items())
              + f"  -> {'ĐẠT' if out['verdict'][k]['pass'] else 'không'} {out['verdict'][k]['targets_pass']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    if not args.analyze_only:
        run_lp2(coco, ids)
    analyze(coco, ids)


if __name__ == "__main__":
    main()
