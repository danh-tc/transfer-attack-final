#!/usr/bin/env python3
"""Pilot VC @ dev30 (quy tắc khóa: progress_log 2026-10-07): áp quy tắc GO dev100 cho K ∈ {2, 4}.
cos_HP(δ_ô, δ_WB) như E12; drop từ metrics_vc_dev30.json + generalization_vc_dev30.json.
  python scripts/vc_pilot_check.py
"""
import json
import os
import sys

import mmcv
import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from scripts.e6_shape_texture import IMG_DIR
from scripts.e10a_wb_dissect import SRC
from scripts.e12_content_align import cos, parts
from scripts.e11_cell_props import ANN_FILE, CROSS3, OUT, RES, RUN

import argparse
CELLS = ["OSFD", "OSFD-K2", "VC-OSFD", "OSFD-K4", "VC4-OSFD"]
PAIRS = {2: ("VC-OSFD", "OSFD-K2"), 4: ("VC4-OSFD", "OSFD-K4")}
V_CELLS = ["OSFD", "OSFD-V4", "OSFD-V8"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", choices=["vc", "v"], default="vc")
    args = p.parse_args()
    global CELLS
    if args.pilot == "v":
        CELLS = V_CELLS
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    c = {m: [] for m in CELLS}
    for i in ids:
        x = mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32)
        w = parts(mmcv.imread(os.path.join(SRC["WB-OSFD"], f"{i}.png")).astype(np.float32) - x)["HP"]
        for m in CELLS:
            c[m].append(cos(parts(mmcv.imread(os.path.join(RUN, "adv", m, f"{i}.png")).astype(np.float32) - x)["HP"], w))
    drop = {}
    for f in (f"metrics_{args.pilot}_dev30.json", f"generalization_{args.pilot}_dev30.json"):
        for m, v in json.load(open(os.path.join(RES, f)))["relative_ap_drop"].items():
            for t, x in v.items():
                drop.setdefault(m, {}).setdefault(t, x["point"] if isinstance(x, dict) else x)
    out = {"cos_HP": {m: float(np.mean(v)) for m, v in c.items()}, "drop": drop, "rule": {}}
    ca3 = {m: float(np.mean([drop[m][t] for t in CROSS3])) for m in CELLS}
    print("ô          R50   R101  ConvNeXt Swin-T YOLOX  Swin-L | CA3  | cos_HP")
    for m in CELLS:
        d = drop[m]
        print(f"{m:<10}" + "".join(f"{d[t]:6.1f} " for t in ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"])
              + f"| {ca3[m]:.1f} | {out['cos_HP'][m]:+.4f}")
    sat = {}
    if args.pilot == "v":
        for m in CELLS:
            sat[m] = float(np.mean([(np.abs(mmcv.imread(os.path.join(RUN, "adv", m, f"{i}.png")).astype(int)
                                            - mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).astype(int)) >= 5).mean()
                                    for i in ids]))
        for m in CELLS[1:]:
            r1 = out["cos_HP"][m] > out["cos_HP"]["OSFD"] and sat[m] >= 0.88
            r2 = ca3[m] >= ca3["OSFD"] + 2 and drop[m]["dino_swin_l"] >= drop["OSFD"]["dino_swin_l"] + 2
            out["rule"][m] = {"sat": sat[m], "r1": bool(r1), "r2": bool(r2), "GO": bool(r1 and r2)}
            print(f"{m}: (1) cos {out['cos_HP'][m]:+.4f} vs OSFD {out['cos_HP']['OSFD']:+.4f}, sat {sat[m]:.3f} -> {r1}; "
                  f"(2) ΔCA3 {ca3[m] - ca3['OSFD']:+.1f}, ΔSwin-L {drop[m]['dino_swin_l'] - drop['OSFD']['dino_swin_l']:+.1f} -> {r2}"
                  f"  => {'GO dev100' if r1 and r2 else 'không'}")
        PAIRS_ = {}
    else:
        PAIRS_ = PAIRS
    for k, (vc, ctl) in PAIRS_.items():
        r1 = out["cos_HP"][vc] > out["cos_HP"][ctl]
        d_ca3, d_sw = ca3[vc] - ca3[ctl], drop[vc]["dino_swin_l"] - drop[ctl]["dino_swin_l"]
        r2 = d_ca3 >= 2 and d_sw >= 2
        r3 = ca3[vc] >= ca3["OSFD"] - 1
        out["rule"][k] = {"cos_up": r1, "dCA3": d_ca3, "dSwinL": d_sw, "r2": r2, "r3": r3, "GO": bool(r1 and r2 and r3)}
        print(f"K={k}: (1) cos {out['cos_HP'][vc]:+.4f} vs {out['cos_HP'][ctl]:+.4f} -> {r1}; "
              f"(2) ΔCA3 {d_ca3:+.1f}, ΔSwin-L {d_sw:+.1f} -> {r2}; (3) CA3 {ca3[vc]:.1f} vs OSFD−1 {ca3['OSFD'] - 1:.1f} -> {r3}"
              f"  => {'GO dev100' if out['rule'][k]['GO'] else 'không'}")
    with open(os.path.join(OUT, f"{args.pilot}_pilot_dev30.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
