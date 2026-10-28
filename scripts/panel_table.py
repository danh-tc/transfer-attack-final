#!/usr/bin/env python3
"""Bảng panel CHỐT (progress_log 2026-10-28): relative AP drop + ASR trên Controlled (r50 / r101 / convnext_t / swin_t, từ dets của
run_baselines) + PANEL_GEN (gen_dets của eval_generalization), cho các ô ở run khác nhau; paired bootstrap so 1 ô mốc.
ASR = trong các GT target detect được ở ảnh sạch (đúng lớp, IoU ≥ 0.5, score ≥ 0.3), tỉ lệ không còn detect (per_image_transfer).
T8 = 8 target hộp đen (bỏ r50 white-box). Ô dạng TÊN@RUN:FILE_DETS_CONTROLLED.
  python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD \
      --cells OSFD@dev300_B50_eps5_limit100:dets_e40_dev30.json C2-OCT-1V-WP2-RD@dev300_B36_eps5_limit100:dets_e45_dev30.json ...
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices
from attack.mechanism import per_image_transfer
from scripts.eval_generalization import PANEL_GEN

ART = os.path.join(REPO_ROOT, "artifacts/runs")
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
CTRL = ["r50", "r101", "convnext_t", "swin_t"]
T8 = ["r101", "convnext_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
BOOT_SEED, N_BOOT = 2026, 1000


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="dev30")
    p.add_argument("--cells", nargs="+", required=True)
    p.add_argument("--ref", required=True, help="tên ô mốc cho Δ (paired bootstrap)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    with quiet():
        coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, f"data/image_lists/{args.subset}.csv"))
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    B = np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))
    models = CTRL + PANEL_GEN

    def ap(d):
        with quiet():
            ev = CocoImageEval(coco, d, ids)  # khởi tạo 1 lần, tái dùng mọi trọng số bootstrap
        return np.array([ev.ap(w) for w in W])[:, 0] * 100

    drop, asr = {}, {}
    for cell in args.cells:
        name, rest = cell.split("@")
        run, dfile = rest.split(":")
        ctrl = json.load(open(os.path.join(ART, run, dfile)))
        drop[name], asr[name] = {}, {}
        for k in models:
            if k in CTRL:
                c, a = ctrl["clean"][k], ctrl[name][k]
            else:
                g = json.load(open(os.path.join(ART, run, "gen_dets", f"{k}.json")))
                c, a = g["clean"], g[name]
            ac = ap(c)
            drop[name][k] = 100 * (ac - ap(a)) / ac  # [1 + N_BOOT]
            with quiet():
                asr[name][k] = 100 * per_image_transfer(coco, c, a, ids)["suppression"]  # theo ảnh
        drop[name]["T8"] = np.mean([drop[name][k] for k in T8], axis=0)
        asr[name]["T8"] = np.nanmean([asr[name][k] for k in T8], axis=0)

    cols = models + ["T8"]
    res = {"subset": args.subset, "n_images": n, "ref": args.ref, "table": {}, "delta_vs_ref": {}}
    for name in drop:
        res["table"][name] = {k: {"drop": round(float(drop[name][k][0]), 2), "asr": round(float(np.nanmean(asr[name][k])), 2)}
                              for k in cols}
        if name == args.ref:
            continue
        res["delta_vs_ref"][name] = {}
        for k in cols:
            dd = drop[name][k] - drop[args.ref][k]
            da = asr[name][k] - asr[args.ref][k]
            bs = [np.nanmean(da[b]) for b in B]
            res["delta_vs_ref"][name][k] = {
                "drop": [round(float(dd[0]), 2), *np.round(np.percentile(dd[1:], [2.5, 97.5]), 2).tolist()],
                "asr": [round(float(np.nanmean(da)), 2), *np.round(np.nanpercentile(bs, [2.5, 97.5]), 2).tolist()]}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("drop / ASR %".ljust(20), " ".join(c[:11].rjust(13) for c in cols))
    for name, row in res["table"].items():
        print(name[:20].ljust(20), " ".join(f"{row[c]['drop']:5.1f} / {row[c]['asr']:5.1f}" for c in cols))
    for name, row in res["delta_vs_ref"].items():
        print(f"Δ {name} − {args.ref}:")
        for c in cols:
            d, a = row[c]["drop"], row[c]["asr"]
            print(f"  {c:12s} drop {d[0]:+6.1f} [{d[1]:+6.1f}, {d[2]:+6.1f}]   ASR {a[0]:+6.1f} [{a[1]:+6.1f}, {a[2]:+6.1f}]")


if __name__ == "__main__":
    main()
