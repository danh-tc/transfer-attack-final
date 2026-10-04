#!/usr/bin/env python3
"""Cổng độ trung thực GRGD (sửa B) — forward-only, TRƯỚC mọi attack (quy tắc khóa: progress_log 2026-10-04
"GRGD-v0 sửa (B)").

dev30, mỗi ảnh V view, seed như run_baselines (SEED + chỉ số dev300; ξ = Random(seed + 1), tâm ngẫu nhiên =
Random(seed + 2)); GRGD và RESIDUAL rút CÙNG H, ξ, nhiễu (cùng seed). Chỉ tính view có N ≥ 2.
  G1: median ρ_pos (GRGD) ≥ 0.75, ρ_pos = D_pair_pos_eff / D_pair_pos_req (req theo d_ij đã rút).
  G2: mean D_pair_pos_eff > 0 VÀ mean D_pair_scale_box > 0 (GRGD).
  G3: mean RMS |u_rel| GRGD / RESIDUAL ∈ [0.9, 1.1].
PASS = cả 3.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods.grgd import grgd_views
from attack.preprocess import resize_to_model

SEED = 0  # = run_baselines.SEED


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--views", type=int, default=10)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results/grgd/fidelity_gate.json"))
    args = p.parse_args()

    split = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    ds = AttackDataset(ids)
    per = {"object": [], "random": []}
    for i in range(len(ds)):
        smp = ds[i]
        d = smp["data_sample"]
        im = resize_to_model(smp["orig_inputs"].cuda(), tuple(d.img_shape))
        seed = SEED + split.index(smp["img_id"])
        for anc in per:
            random.seed(seed); torch.manual_seed(seed)
            st = []
            grgd_views(im, args.views, d, random.Random(seed + 1), random.Random(seed + 2), anchors=anc,
                       with_gt=True, stats=st)
            per[anc] += [dict(x, img_id=smp["img_id"]) for x in st if x["n_edges"] > 0]

    g, r = per["object"], per["random"]
    assert [x["n_objects"] for x in g] == [x["n_objects"] for x in r]
    rho = np.array([x["rho_pos"] for x in g])
    rms_ratio = np.mean([x["rms_rel_px"] for x in g]) / np.mean([x["rms_rel_px"] for x in r])
    mean = lambda rows, k: float(np.mean([x[k] for x in rows if k in x]))
    gates = {
        "G1_median_rho_pos": {"value": float(np.median(rho)), "threshold": ">= 0.75", "pass": bool(np.median(rho) >= 0.75)},
        "G2_pos_eff_and_scale_box_gt_0": {"value": [mean(g, "d_pair_pos_eff"), mean(g, "d_pair_scale_box")],
                                          "pass": bool(mean(g, "d_pair_pos_eff") > 0 and mean(g, "d_pair_scale_box") > 0)},
        "G3_rms_ratio_grgd_over_residual": {"value": float(rms_ratio), "threshold": "[0.9, 1.1]",
                                            "pass": bool(0.9 <= rms_ratio <= 1.1)},
    }
    n = np.array([x["n_objects"] for x in g])
    by_n = {}
    for lo, hi in ((2, 4), (5, 10), (11, 1000)):
        m = (n >= lo) & (n <= hi)
        if m.any():
            by_n[f"{lo}-{hi}"] = {"n_views": int(m.sum()), "median_rho_pos": float(np.median(rho[m])),
                                  "mean_sqrt_2_over_N": float(np.mean(np.sqrt(2 / n[m])))}
    keys = ["n_objects", "rms_rel_px", "disp_rel_px", "d_pair_pos_req", "d_pair_pos", "d_pair_pos_eff", "rho_pos",
            "cos_pair_eff_req", "d_pair_pos_box", "d_pair_scale_req", "d_pair_scale", "d_pair_scale_box",
            "fit_rms_ctrl_px", "raw_rms_ctrl_px", "disp_pia_px"]
    out = {"split": "dev30", "views_per_image": args.views, "n_views_N_ge_2": len(g),
           "pass": all(v["pass"] for v in gates.values()), "gates": gates, "rho_pos_by_N": by_n,
           "describe": {name: {k: mean(rows, k) for k in keys} for name, rows in (("GRGD", g), ("RESIDUAL", r))}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("n_views_N_ge_2", "pass", "gates", "rho_pos_by_N")}, indent=2, ensure_ascii=False))
    for name in ("GRGD", "RESIDUAL"):
        print(name, {k: round(v, 3) for k, v in out["describe"][name].items()})


if __name__ == "__main__":
    main()
