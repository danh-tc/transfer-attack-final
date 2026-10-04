#!/usr/bin/env python3
"""So sánh ứng viên Repair-Robust với mốc (progress_log 2026-10-04, quy tắc KHÓA).

Δ = drop(ứng viên) − drop(mốc), dương = ứng viên mạnh hơn. 6 model (R50, R101, ConvNeXt-T, Swin-T, YOLOX-S,
DINO-Swin-L), CrossAvg₃ = mean(ConvNeXt-T, Swin-T, YOLOX-S); paired bootstrap 1000 mẫu, BOOT_SEED 2026.
Kèm loại lỗi (recall@0.3, FP/ảnh theo loại) — `probe_error_types.analyse`.

--rule:
  v      probe V (dev100): chọn L_V cho R-a nếu Δ DINO-Swin-L(V − C2) ≥ +2 VÀ ΔCrossAvg₃(V − C2) ≥ −2.
  dose   liều–đáp ứng R-a (dev100, so nền p = 0): đi tiếp nếu ≥ 1 liều có Δ DINO-Swin-L ≥ +3 VÀ ΔCrossAvg₃ ≥ −2
         VÀ ΔR101 ≥ −3; chọn liều DINO-Swin-L cao hơn (chênh < 1 -> liều đầu tiên truyền vào = p 0.5).
  none   chỉ báo cáo.

Chạy: python scripts/rr_screen.py --dets dets_rr_dev100.json --subset dev100 --ref PROBE-osfd+cls \
        --cands PROBE-V --extra OSFD --rule v --tag v_dev100
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
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize
from probe_error_types import analyse

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
BOOT_SEED, N_BOOT = 2026, 1000
CTRL = ["r50", "r101", "convnext_t", "swin_t"]
GEN = ["yolox_s", "dino_swin_l"]
KEYS = CTRL + GEN
CROSS3 = ["convnext_t", "swin_t", "yolox_s"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="dev300_B50_eps5")
    p.add_argument("--subset", default=None)
    p.add_argument("--dets", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--cands", nargs="+", required=True)
    p.add_argument("--extra", nargs="*", default=[], help="điều kiện chỉ báo cáo (vd OSFD)")
    p.add_argument("--rule", choices=["v", "dose", "none"], default="none")
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    art = os.path.join(REPO_ROOT, "artifacts/runs", args.run)
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.run.split('_')[0]}.csv"))
    if args.subset:
        keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv")))
        img_ids = [i for i in img_ids if i in keep]
    n = len(img_ids)
    conds = ["clean", args.ref] + args.cands + args.extra
    coco = COCO(ANN_FILE)
    with open(os.path.join(art, args.dets)) as f:
        ctrl = json.load(f)
    dets = {k: {c: ctrl[c][k] for c in conds} for k in CTRL}
    for k in GEN:
        with open(os.path.join(art, "gen_dets", f"{k}.json")) as f:
            g = json.load(f)
        dets[k] = {c: g[c] for c in conds}

    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    ap = {k: {} for k in KEYS}
    for k in KEYS:
        for c in conds:
            ev = CocoImageEval(coco, dets[k][c], img_ids)
            ap[k][c] = np.array([ev.ap(w) for w in W]) * 100
    clean = {k: ap[k]["clean"][:, 0] for k in KEYS}
    drop = {c: {k: 100 * (clean[k] - ap[k][c][:, 0]) / clean[k] for k in KEYS} for c in conds[1:]}
    for c in conds[1:]:
        drop[c]["cross3"] = np.mean([drop[c][k] for k in CROSS3], axis=0)
    S = lambda a: summarize(a[0], a[1:])
    cols = KEYS + ["cross3"]
    rows = {c: {k: S(drop[c][k]) for k in cols} for c in conds[1:]}
    delta = {c: {k: S(drop[c][k] - drop[args.ref][k]) for k in cols} for c in args.cands + args.extra}
    errors = {k: {c: analyse(coco, dets[k][c], img_ids) for c in conds} for k in ("swin_t", "yolox_s", "dino_swin_l")}

    decision = None
    pt = lambda c, k: delta[c][k]["point"]
    if args.rule == "v":
        c = args.cands[0]
        ok = pt(c, "dino_swin_l") >= 2 and pt(c, "cross3") >= -2
        decision = {"use_vanish_for_Ra": bool(ok),
                    "rule": "Δ DINO-Swin-L(V − C2) ≥ +2 VÀ ΔCrossAvg₃(V − C2) ≥ −2 (điểm)"}
    elif args.rule == "dose":
        ok = [c for c in args.cands if pt(c, "dino_swin_l") >= 3 and pt(c, "cross3") >= -2 and pt(c, "r101") >= -3]
        best = None
        if ok:
            best = ok[0]
            for c in ok[1:]:
                if pt(c, "dino_swin_l") - pt(best, "dino_swin_l") >= 1:
                    best = c
        decision = {"pass": bool(ok), "passing": ok, "chosen": best,
                    "rule": "≥ 1 liều: Δ DINO-Swin-L ≥ +3 VÀ ΔCrossAvg₃ ≥ −2 VÀ ΔR101 ≥ −3 (điểm, so nền)"}

    out = {"run": args.run, "subset": args.subset, "n_images": n, "ref": args.ref, "n_boot": N_BOOT,
           "boot_seed": BOOT_SEED, "delta_definition": "drop(ứng viên) − drop(mốc)",
           "relative_ap_drop": rows, "delta_vs_ref": delta, "error_types": errors, "decision": decision,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_dir = os.path.join(REPO_ROOT, "results/repair_robust", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "screen.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n=== relative AP drop % ({args.tag}, n={n}) ===\n{'':<18}" + "".join(f"{k:>12}" for k in cols))
    for c in conds[1:]:
        print(f"{c:<18}" + "".join(f"{rows[c][k]['point']:>12.1f}" for k in cols))
    fmt = lambda s: f"{s['point']:+5.1f} [{s['ci95'][0]:+5.1f},{s['ci95'][1]:+5.1f}]"
    print(f"\n=== Δ = ứng viên − {args.ref} ===")
    for c in args.cands + args.extra:
        print(f"{c:<18} Cross3 {fmt(delta[c]['cross3'])}  R101 {fmt(delta[c]['r101'])}  "
              f"DINO-SwinL {fmt(delta[c]['dino_swin_l'])}  YOLOX {fmt(delta[c]['yolox_s'])}")
    print("\n=== loại lỗi (recall@0.3 | FP nền/ảnh) ===")
    for k, v in errors.items():
        print(f"{k:<12}" + "  ".join(f"{c}: {r['recall@0.3']:.2f} | {r['fp_per_img'].get('background', 0):.1f}"
                                     for c, r in v.items()))
    print("\nQUYẾT ĐỊNH:", json.dumps(decision, ensure_ascii=False))
    print("saved:", os.path.join(out_dir, "screen.json"))


if __name__ == "__main__":
    main()
