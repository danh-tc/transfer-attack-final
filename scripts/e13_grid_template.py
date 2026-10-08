#!/usr/bin/env python3
"""E13 (quy tắc khóa: progress_log 2026-10-08): δ có thành phần KHÓA LƯỚI chu kỳ 4 không — ô R50 vs WB-OSFD Swin-L?
CPU, PNG có sẵn, dev30. Khung model = imrescale (1333, 800) bilinear như test pipeline (Mask R-CNN = DINO).
T_p[c, a, b] = mean HP[c, y≡a, x≡b (mod p)] − mean HP[c];  Z_p = (‖T_p‖² / p²) / (mean HP² / N_px);  P4 = Z4 / mean(Z3, Z5, Z7).
  python scripts/e13_grid_template.py
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

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(REPO_ROOT, "data/coco/val2017")
RUNS = os.path.join(REPO_ROOT, "results/runs/dev300_B50_eps5")
HF_ADV = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5/adv")
LOCAL_ADV = os.path.join(REPO_ROOT, "artifacts/local_dev300_20261008/adv")
SRC = {"WB-OSFD": os.path.join(REPO_ROOT, "artifacts/runs/e8_wbosfd_dev30/WB-OSFD"),
       **{c: os.path.join(HF_ADV, c) for c in ("NOISE", "OSFD", "ROT15", "PIA", "PROBE-osfd+cls")},
       **{c: os.path.join(LOCAL_ADV, c) for c in ("SH32", "SH4", "SH2", "SHODD")}}
CELLS = [c for c in SRC if c != "WB-OSFD"]
# CA3 lấy từ file metrics của đúng lần sinh PNG đang đo (SH* hôm nay = stride / stride2; còn lại = grid / e9 / pia).
METRIC_FILES = ["stride2_dev30", "stride_dev30", "grid_dev30", "e9_dev30", "pia_dev30"]
PERIODS = (3, 4, 5, 7, 8)
BOOT_SEED, N_BOOT = 2026, 1000
OUT = os.path.join(REPO_ROOT, "results/mechanism/e13_grid_template.json")


def to_model(img):
    return mmcv.imrescale(img, (1333, 800), interpolation="bilinear", backend="cv2").astype(np.float32)


def template(hp, p):
    h, w = (hp.shape[0] // p) * p, (hp.shape[1] // p) * p
    x = hp[:h, :w].reshape(h // p, p, w // p, p, 3)
    t = x.mean(axis=(0, 2))  # [p, p, 3]
    return t - hp.reshape(-1, 3).mean(0)


def z_score(hp, t, p):
    return float((t ** 2).sum() / p ** 2 / ((hp ** 2).mean() / (hp.shape[0] * hp.shape[1])))


def ca3(cell):
    for tag in METRIC_FILES:
        m = json.load(open(os.path.join(RUNS, f"metrics_{tag}.json"), encoding="utf-8"))
        g = json.load(open(os.path.join(RUNS, f"generalization_{tag}.json"), encoding="utf-8"))
        if cell in m["relative_ap_drop"] and cell in g["relative_ap_drop"]:
            d = lambda src, k: src["relative_ap_drop"][cell][k]["point"]
            return (d(m, "convnext_t") + d(m, "swin_t") + d(g, "yolox_s")) / 3, d(g, "dino_swin_l")
    return None, None


def main():
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    Z = {s: {p: np.zeros(len(ids)) for p in PERIODS} for s in SRC}
    R = {c: np.zeros(len(ids)) for c in CELLS}  # corr(T4_ô, T4_WB)
    for n, i in enumerate(ids):
        x = to_model(mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])))
        T4 = {}
        for s, d in SRC.items():
            dm = to_model(mmcv.imread(os.path.join(d, f"{i}.png"))) - x
            hp = dm - cv2.GaussianBlur(dm, (0, 0), 2.0)
            for p in PERIODS:
                t = template(hp, p)
                Z[s][p][n] = z_score(hp, t, p)
                if p == 4:
                    T4[s] = t.ravel()
        for c in CELLS:
            R[c][n] = float(np.corrcoef(T4[c], T4["WB-OSFD"])[0, 1])
    rng = np.random.default_rng(BOOT_SEED)
    idx = np.vstack([np.arange(len(ids))[None], rng.integers(0, len(ids), (N_BOOT, len(ids)))])
    S = lambda v: {"point": float(v[idx][0].mean()), "ci95": [float(q) for q in np.percentile(v[idx[1:]].mean(1), [2.5, 97.5])]}
    P4 = {s: Z[s][4] / np.mean([Z[s][p] for p in (3, 5, 7)], axis=0) for s in SRC}
    # Đính chính SAU khi thấy kết quả (progress_log): nhiễu trắng cho Z_p ≈ p² (không ≈ 1 như entry khóa ghi) ->
    # bản chuẩn hóa đúng ý định: Zn_p = Z_p / p², P4n = Zn4 / mean(Zn3, Zn5, Zn7). Báo cả hai.
    P4n = {s: (Z[s][4] / 16) / np.mean([Z[s][p] / p ** 2 for p in (3, 5, 7)], axis=0) for s in SRC}
    out = {"Z": {s: {p: S(Z[s][p]) for p in PERIODS} for s in SRC}, "P4": {s: S(P4[s]) for s in SRC}, "P4_norm": {s: S(P4n[s]) for s in SRC},
           "r_T4_vs_WB": {c: S(R[c]) for c in CELLS}}
    z4 = {s: out["Z"][s][4]["point"] for s in SRC}
    p4 = {s: out["P4"][s]["point"] for s in SRC}
    p4n = {s: out["P4_norm"][s]["point"] for s in SRC}
    i_ok = p4["SH32"] >= 3 and p4["NOISE"] >= 3
    i_n, iii_n = p4n["SH32"] >= 3 and p4n["NOISE"] >= 3, p4n["WB-OSFD"] >= 3
    ii_ok_z = (z4["SH4"] >= 0.7 * z4["SH32"] and z4["SH2"] <= 0.5 * z4["SH32"] and z4["SHODD"] <= 0.5 * z4["SH32"])
    ii_ok = i_ok and z4["SH4"] >= 0.7 * z4["SH32"] and z4["SH2"] <= 0.5 * z4["SH32"] and z4["SHODD"] <= 0.5 * z4["SH32"]
    iii_ok = p4["WB-OSFD"] >= 3
    r32 = out["r_T4_vs_WB"]["SH32"]
    iv = None
    if i_ok and iii_ok:
        iv = ("GIỐNG" if r32["point"] >= 0.3 and r32["ci95"][0] > 0 else "KHÁC" if r32["point"] <= 0.1 else "xám")
    cas = {c: ca3(c) for c in CELLS}
    have = [c for c in CELLS if cas[c][0] is not None]
    rho_ca3 = spearmanr([z4[c] for c in have], [cas[c][0] for c in have])[0]
    rho_swl = spearmanr([z4[c] for c in have], [cas[c][1] for c in have])[0]
    out["rules_norm"] = {"i": i_n, "ii": i_n and ii_ok_z, "iii": iii_n}
    out["rules"] = {"i": i_ok, "ii": ii_ok, "iii": iii_ok, "iv": iv,
                    "spearman_Z4_CA3": float(rho_ca3), "spearman_Z4_SwinL": float(rho_swl), "cells_with_CA3": have}
    out["ca3_swinl"] = cas
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    fmt = lambda s: f"{s['point']:7.2f} [{s['ci95'][0]:6.2f}, {s['ci95'][1]:6.2f}]"
    print(f"{'':16}" + "".join(f"{'Z' + str(p):>10}" for p in PERIODS) + f"{'P4':>26}{'r(T4, WB)':>26}   CA3 / SwinL")
    for s in SRC:
        r = fmt(out["r_T4_vs_WB"][s]) if s in R else ""
        ca = f"{cas[s][0]:.1f} / {cas[s][1]:.1f}" if s in cas and cas[s][0] is not None else ""
        print(f"{s:16}" + "".join(f"{out['Z'][s][p]['point']:10.2f}" for p in PERIODS) + f"{fmt(out['P4'][s]):>26}{fmt(out['P4_norm'][s]):>26}{r:>26}   {ca}")
    print("rules (nguyên văn):", out["rules"])
    print("rules (chuẩn hóa p²):", out["rules_norm"])
    print("saved:", OUT)


if __name__ == "__main__":
    main()
