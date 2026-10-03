#!/usr/bin/env python3
"""Phân loại ô ablation OSFD (docs/plan_sota.md, bước 1 — LOCKED).

Δ = drop(OSFD) − drop(ô): dương = bỏ/thay thành phần đó làm OSFD YẾU đi. Trên CrossAvg₃ =
mean(ConvNeXt-T, Swin-T, YOLOX-S), paired bootstrap 1000 mẫu (BOOT_SEED 2026) trên các ảnh đánh giá:
  cận dưới CI >= 2 -> nguyên liệu;  cận trên CI < 2 -> không phải;  còn lại -> mơ hồ.
Bậc 1: --subset dev100. Bậc 2 (ô mơ hồ, đủ dev300): bỏ --subset, chỉ quyết ô được truyền qua --cells.
Báo cáo kèm: từng target, R101, R50, DINO-Swin-L, η = cross₂/R101, η_AP (theo 1 − advAP/cleanAP).

Đầu vào: artifacts/runs/<run>/{dets_<tag>.json, gen_dets/{yolox_s,dino_swin_l}.json}.
Chạy: python scripts/ablation_screen.py --subset dev100 --dets dets_abl_dev100.json --tag dev100
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
BOOT_SEED, N_BOOT, THRESH = 2026, 1000, 2.0
REF = "OSFD"
CELLS = ["ABL-task-loss", "ABL-no-RRB", "ABL-RRB-no-noise", "ABL-RRB-no-rotate",
         "ABL-RRB-no-resize", "ABL-no-MI", "ABL-k5"]
CTRL = ["r50", "r101", "convnext_t", "swin_t"]
GEN = ["yolox_s", "dino_swin_l"]
CROSS3 = ["convnext_t", "swin_t", "yolox_s"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=None)
    p.add_argument("--dets", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--cells", nargs="+", default=CELLS)
    p.add_argument("--run", default="dev300_B50_eps5", help="run artifacts (split = phần trước dấu _)")
    args = p.parse_args()
    ART = os.path.join(REPO_ROOT, "artifacts/runs", args.run)

    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.run.split('_')[0]}.csv"))
    if args.subset:
        keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv")))
        img_ids = [i for i in img_ids if i in keep]
    n = len(img_ids)
    conds = ["clean", REF] + args.cells
    coco = COCO(ANN_FILE)
    with open(os.path.join(ART, args.dets)) as f:
        ctrl = json.load(f)
    dets = {k: {c: ctrl[c][k] for c in conds} for k in CTRL}
    for k in GEN:
        with open(os.path.join(ART, "gen_dets", f"{k}.json")) as f:
            g = json.load(f)
        dets[k] = {c: g[c] for c in conds}

    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    ap = {k: {c: np.array([ev.ap(w) for w in W]) * 100
              for c, ev in ((c, CocoImageEval(coco, dets[k][c], img_ids)) for c in conds)}
          for k in CTRL + GEN}
    print("[ablation] AP xong", flush=True)
    clean = {k: ap[k]["clean"][:, 0] for k in ap}
    drop = {c: {k: 100 * (clean[k] - ap[k][c][:, 0]) / clean[k] for k in ap} for c in conds[1:]}
    cross3 = lambda c: np.mean([drop[c][k] for k in CROSS3], axis=0)
    cross2 = lambda c: (drop[c]["convnext_t"] + drop[c]["swin_t"]) / 2
    S = lambda a: summarize(a[0], a[1:])

    rows, verdict = {}, {}
    for c in conds[1:]:
        rows[c] = {
            "relative_ap_drop": {k: S(drop[c][k]) for k in ap},
            "cross_avg3": S(cross3(c)),
            "eta": S(cross2(c) / drop[c]["r101"]),
            "adv_over_clean_ap": {k: S(100 * ap[k][c][:, 0] / clean[k]) for k in ap},
        }
    delta = {}
    for c in args.cells:
        dd = {k: S(drop[REF][k] - drop[c][k]) for k in ap}
        dd["cross_avg3"] = S(cross3(REF) - cross3(c))
        delta[c] = dd
        lo, hi = dd["cross_avg3"]["ci95"]
        verdict[c] = "nguyên liệu" if lo >= THRESH else ("không phải" if hi < THRESH else "mơ hồ")

    out = {"rule": "docs/plan_sota.md bước 1 (LOCKED)", "subset": args.subset, "n_images": n,
           "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "threshold": THRESH,
           "delta_definition": "drop(OSFD) − drop(ô); dương = bỏ thành phần làm OSFD yếu đi",
           "conditions": rows, "delta_vs_osfd": delta, "verdict": verdict,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_dir = os.path.join(REPO_ROOT, "results/ablation", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "screen.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    keys = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
    print(f"\n=== relative AP drop % ({args.tag}, n={n}) ===\n{'':<20}" + "".join(f"{k:>12}" for k in keys) + f"{'Cross3':>9}{'η':>7}")
    for c in conds[1:]:
        print(f"{c:<20}" + "".join(f"{drop[c][k][0]:>12.1f}" for k in keys)
              + f"{cross3(c)[0]:>9.1f}{rows[c]['eta']['point']:>7.2f}")
    fmt = lambda s: f"{s['point']:+5.1f} [{s['ci95'][0]:+5.1f},{s['ci95'][1]:+5.1f}]"
    print("\n=== Δ = OSFD − ô (dương = thành phần có ích) ===")
    for c in args.cells:
        print(f"{c:<20} Cross3 {fmt(delta[c]['cross_avg3'])}  R101 {fmt(delta[c]['r101'])}  "
              f"DINO-SwinL {fmt(delta[c]['dino_swin_l'])}  -> {verdict[c]}")


if __name__ == "__main__":
    main()
