#!/usr/bin/env python3
"""So cặp ô tùy ý (A − B) trên relative AP drop, paired bootstrap theo ảnh — cùng máy đo với ablation_screen.py.

Dùng cho xác nhận C2-PIA @ dev100 (quy tắc khóa: progress_log 2026-10-04 "Xác nhận C2-PIA trên DINO-Swin-L @ dev100").
Δ = drop(A) − drop(B): dương = A mạnh hơn. Báo cáo 6 model + CrossAvg₃; thêm tập con ảnh mới (--exclude) để mô tả.

  python scripts/pair_screen.py --subset dev100 --dets dets_pia_dev100.json --tag confirm_dev100 \
      --pairs C2-PIA:PROBE-osfd+cls PIA:GEO-C0 C2-PIA:OSFD --exclude dev30
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
BOOT_SEED, N_BOOT = 2026, 1000
CTRL = ["r50", "r101", "convnext_t", "swin_t"]
GEN = ["yolox_s", "dino_swin_l"]
CROSS3 = ["convnext_t", "swin_t", "yolox_s"]


def screen(coco, dets, img_ids, conds, pairs):
    n = len(img_ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    ap = {k: {c: np.array([ev.ap(w) for w in W]) * 100
              for c, ev in ((c, CocoImageEval(coco, dets[k][c], img_ids)) for c in conds)} for k in dets}
    clean = {k: ap[k]["clean"][:, 0] for k in ap}
    drop = {c: {k: 100 * (clean[k] - ap[k][c][:, 0]) / clean[k] for k in ap} for c in conds[1:]}
    cross3 = lambda c: np.mean([drop[c][k] for k in CROSS3], axis=0)
    S = lambda a: summarize(a[0], a[1:])
    rows = {c: dict({k: S(drop[c][k]) for k in ap}, cross_avg3=S(cross3(c))) for c in conds[1:]}
    delta = {}
    for a, b in pairs:
        dd = {k: S(drop[a][k] - drop[b][k]) for k in ap}
        dd["cross_avg3"] = S(cross3(a) - cross3(b))
        delta[f"{a} - {b}"] = dd
    return {"n_images": n, "relative_ap_drop": rows, "delta": delta}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--dets", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--pairs", nargs="+", required=True, help="A:B")
    p.add_argument("--exclude", default=None, help="thêm báo cáo trên subset trừ đi danh sách này (mô tả)")
    p.add_argument("--run", default="dev300_B50_eps5")
    args = p.parse_args()
    ART = os.path.join(REPO_ROOT, "artifacts/runs", args.run)
    pairs = [tuple(x.split(":")) for x in args.pairs]
    conds = ["clean"] + sorted({c for pr in pairs for c in pr})

    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.run.split('_')[0]}.csv"))
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv")))
    img_ids = [i for i in ids if i in keep]
    coco = COCO(ANN_FILE)
    with open(os.path.join(ART, args.dets)) as f:
        ctrl = json.load(f)
    dets = {k: {c: ctrl[c][k] for c in conds} for k in CTRL}
    for k in GEN:
        with open(os.path.join(ART, "gen_dets", f"{k}.json")) as f:
            g = json.load(f)
        dets[k] = {c: g[c] for c in conds}

    out = {"subset": args.subset, "pairs": args.pairs, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "delta_definition": "drop(A) − drop(B); dương = A mạnh hơn",
           "main": screen(coco, dets, img_ids, conds, pairs), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.exclude:
        ex = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.exclude}.csv")))
        out[f"{args.subset}_minus_{args.exclude}"] = screen(coco, dets, [i for i in img_ids if i not in ex], conds, pairs)
    out_dir = os.path.join(REPO_ROOT, "results/ablation", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "screen.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    keys = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
    fmt = lambda s: f"{s['point']:+5.1f} [{s['ci95'][0]:+5.1f},{s['ci95'][1]:+5.1f}]"
    for part in [k for k in out if k == "main" or k.endswith(f"_minus_{args.exclude}")]:
        r = out[part]
        print(f"\n=== {part} (n = {r['n_images']}) relative AP drop % ===\n{'':<18}" + "".join(f"{k:>12}" for k in keys) + f"{'Cross3':>9}")
        for c, v in r["relative_ap_drop"].items():
            print(f"{c:<18}" + "".join(f"{v[k]['point']:>12.1f}" for k in keys) + f"{v['cross_avg3']['point']:>9.1f}")
        for pr, dd in r["delta"].items():
            print(f"{pr:<28} DINO-SwinL {fmt(dd['dino_swin_l'])}  Cross3 {fmt(dd['cross_avg3'])}  R101 {fmt(dd['r101'])}  "
                  f"YOLOX {fmt(dd['yolox_s'])}  ConvNeXt {fmt(dd['convnext_t'])}  Swin-T {fmt(dd['swin_t'])}")


if __name__ == "__main__":
    main()
