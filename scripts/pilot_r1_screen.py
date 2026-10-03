#!/usr/bin/env python3
"""Sàng lọc pilot R1 @ dev300 (bước 1) — quy tắc LOCKED: docs/progress_log.md 2026-10-03.

Đầu vào: run_baselines.py --split dev300 --methods OSFD OSFD-SGM-0.2 OSFD-SGM-0.5 OSFD-SGM-mix
--out metrics_sgm.json; eval_generalization.py ... --dets dets_sgm.json --out generalization_sgm.json.
CrossAvg₃ = mean(drop ConvNeXt-T, Swin-T, YOLOX-S). E_best = CrossAvg₃ cao nhất (chênh < 0.5 -> ưu tiên
E1 hơn E2, γ = 0.2 hơn 0.5). QUA nếu: ΔCrossAvg₃(E_best − E0) >= 3; E_best − E0 > 0 ở ConvNeXt-T,
Swin-T, YOLOX-S; drop R101(E_best) >= drop R101(E0) − 3 (đều điểm ước lượng). Paired bootstrap 1000
mẫu, BOOT_SEED 2026. E2 − E1a, E2 − E1b, E1a − E1b: chỉ báo cáo.

Output: results/r1_pilot/dev300/{screen.json, <cond>/{metrics.json, runtime.json, config.json}}.
Chạy: python scripts/pilot_r1_screen.py
"""
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
from attack.mechanism import boot_weights, per_image_transfer

RUN = "dev300_B50_eps5"
ART = os.path.join(REPO_ROOT, "artifacts/runs", RUN)
RUN_DIR = os.path.join(REPO_ROOT, "results/runs", RUN)
OUT_DIR = os.path.join(REPO_ROOT, "results/r1_pilot/dev300")
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
BOOT_SEED, N_BOOT = 2026, 1000
COND = {"E0": "OSFD", "E1a": "OSFD-SGM-0.2", "E1b": "OSFD-SGM-0.5", "E2": "OSFD-SGM-mix"}
PREF = ["E1a", "E1b", "E2"]  # thứ tự ưu tiên khi hòa (< 0.5)
CTRL = ["r50", "r101", "convnext_t", "swin_t"]
GEN = ["yolox_s", "dino_swin_l"]
CROSS3 = ["convnext_t", "swin_t", "yolox_s"]
BACKWARD = {"E0": "chuẩn", "E1a": "OSFD-adapted SGM γ = 0.2", "E1b": "OSFD-adapted SGM γ = 0.5",
            "E2": "OSFD-adapted SGM, γ chọn đều mỗi step trong {0.2, 0.5} (numpy RNG seed SEED + i)"}


def main():
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    n = len(img_ids)
    coco = COCO(ANN_FILE)
    with open(os.path.join(ART, "dets_sgm.json")) as f:
        ctrl = json.load(f)
    dets = {k: {c: ctrl[c][k] for c in ["clean"] + list(COND.values())} for k in CTRL}
    for k in GEN:
        with open(os.path.join(ART, "gen_dets", f"{k}.json")) as f:
            g = json.load(f)
        dets[k] = {c: g[c] for c in ["clean"] + list(COND.values())}
    stats = {}
    with open(os.path.join(RUN_DIR, "attack_stats.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            stats[(r["method"], r["img_id"])] = r

    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    ap = {}
    for k in CTRL + GEN:
        ap[k] = {}
        for c, d in dets[k].items():
            ev = CocoImageEval(coco, d, img_ids)
            ap[k][c] = np.array([ev.ap(w) for w in W]) * 100
        print(f"[screen] AP xong: {k}", flush=True)
    D = {e: {k: 100 * (ap[k]["clean"][:, 0] - ap[k][m][:, 0]) / ap[k]["clean"][:, 0] for k in CTRL + GEN}
         for e, m in COND.items()}
    cross3 = lambda e: np.mean([D[e][k] for k in CROSS3], axis=0)
    S = lambda a: summarize(a[0], a[1:])

    def diff(a, b):
        row = {k: S(D[a][k] - D[b][k]) for k in CTRL + GEN}
        row["cross_avg3"] = S(cross3(a) - cross3(b))
        return row
    pairs = [("E1a", "E0"), ("E1b", "E0"), ("E2", "E0"), ("E2", "E1a"), ("E2", "E1b"), ("E1a", "E1b")]
    delta = {f"{a}-{b}": diff(a, b) for a, b in pairs}

    ca = {e: float(cross3(e)[0]) for e in PREF}
    top = max(ca.values())
    best = next(e for e in PREF if top - ca[e] < 0.5)
    d = delta[f"{best}-E0"]
    checks = {
        "cross_avg3_best-E0>=3": d["cross_avg3"]["point"] >= 3,
        "convnext_t_best-E0>0": d["convnext_t"]["point"] > 0,
        "swin_t_best-E0>0": d["swin_t"]["point"] > 0,
        "yolox_s_best-E0>0": d["yolox_s"]["point"] > 0,
        "r101_best>=E0-3": d["r101"]["point"] >= -3,
    }
    passed = all(checks.values())

    idx = boot_weights(n, N_BOOT, BOOT_SEED)
    for e, m in COND.items():
        dd = os.path.join(OUT_DIR, e)
        os.makedirs(dd, exist_ok=True)
        rows = [stats[(m, i)] for i in img_ids]
        pit = {k: per_image_transfer(coco, dets[k]["clean"], dets[k][m], img_ids) for k in CTRL + GEN}
        mean_ci = lambda v: summarize(np.nanmean(v), np.array([np.nanmean(v[r]) for r in idx]))
        runtime = {"runtime_s_per_img_mean": float(np.mean([r["runtime_s"] for r in rows])),
                   "backward_per_img": rows[0]["backward"], "forward_views_per_img": rows[0]["forward_views"]}
        metrics = {
            "condition": e, "method": m, "n_images": n,
            "clean_ap": {k: S(ap[k]["clean"][:, 0]) for k in CTRL + GEN},
            "adv_ap": {k: S(ap[k][m][:, 0]) for k in CTRL + GEN},
            "clean_ap50": {k: S(ap[k]["clean"][:, 1]) for k in CTRL + GEN},
            "adv_ap50": {k: S(ap[k][m][:, 1]) for k in CTRL + GEN},
            "relative_ap_drop": {k: S(D[e][k]) for k in CTRL + GEN},
            "cross_avg3": S(cross3(e)),
            "transfer_gap": {**{f"r101_minus_{k}": S(D[e]["r101"] - D[e][k]) for k in CROSS3},
                             "r101_minus_cross_avg3": S(D[e]["r101"] - cross3(e))},
            "suppression_mean": {k: mean_ci(pit[k]["suppression"]) for k in CTRL + GEN},
            "retained_confidence_mean": {k: mean_ci(pit[k]["retained"]) for k in CTRL + GEN},
            "perturbation": {"linf_255_max": max(r["linf_255"] for r in rows),
                             "l2_mean": float(np.mean([r["l2"] for r in rows]))},
            "compute": runtime, "not_computed": ["ASR", "APloc", "CSR", "LPIPS"],
        }
        config = {"split": "dev300", "surrogate": "Mask R-CNN R50", "epsilon_255": 5.0, "alpha": 1.0,
                  "budget_B": 50, "recipe": "OSFD full (MI μ=1, RRB θ=7 l_s=10 ρ=0.8 s_max=1.1 σ=6, k=3)",
                  "backward": BACKWARD[e], "rule": "docs/progress_log.md 2026-10-03 (pilot R1, LOCKED)"}
        for name, obj in [("metrics.json", metrics), ("runtime.json", runtime), ("config.json", config)]:
            with open(os.path.join(dd, name), "w") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

    out = {"rule": "progress_log 2026-10-03 (pilot R1, LOCKED), bước 1 — sàng lọc dev300",
           "n_images": n, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "relative_ap_drop": {e: {k: S(D[e][k]) for k in CTRL + GEN} for e in COND},
           "cross_avg3_by_condition": {e: S(cross3(e)) for e in COND},
           "E_best": best, "delta": delta, "checks": checks, "pass": passed,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(OUT_DIR, "screen.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    fmt = lambda s: f"{s['point']:+6.1f} [{s['ci95'][0]:+5.1f},{s['ci95'][1]:+5.1f}]"
    keys = CTRL + GEN
    print(f"\n=== relative AP drop (dev300, n={n}) ===\n{'':<5}" + "".join(f"{k:>13}" for k in keys + ["cross_avg3"]))
    for e in COND:
        print(f"{e:<5}" + "".join(f"{D[e][k][0]:>13.1f}" for k in keys) + f"{cross3(e)[0]:>13.1f}")
    print("\n=== Δ (toàn bộ ảnh) ===")
    for name, row in delta.items():
        print(f"{name:<8}" + "  ".join(f"{k} {fmt(row[k])}" for k in ["r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l", "cross_avg3"]))
    print(f"\nE_best = {best}  " + "  ".join(f"{k}: {'✓' if v else '✗'}" for k, v in checks.items()))
    print(f"=> sàng lọc dev300: {'QUA' if passed else 'KHÔNG QUA (NO-GO R1 -> dừng RQ3)'}")


if __name__ == "__main__":
    main()
