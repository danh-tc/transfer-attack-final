#!/usr/bin/env python3
"""Sàng lọc pilot KG @ dev300 (bước 1) — quy tắc LOCKED: docs/progress_log.md 2026-10-03.

Đầu vào (sinh bởi run_baselines.py --split dev300 --methods OSFD OSFD-KG-E1 OSFD-KG-E2
--out metrics_kg.json, rồi eval_generalization.py --run dev300_B50_eps5 --models yolox_s
dino_swin_l --methods ... --dets dets_kg.json):
  artifacts/runs/dev300_B50_eps5/{dets_kg.json, gen_dets/yolox_s.json, gen_dets/dino_swin_l.json}
  results/runs/dev300_B50_eps5/attack_stats.jsonl

Paired bootstrap 1000 mẫu trên ảnh (BOOT_SEED 2026), CÙNG trọng số cho mọi điều kiện/model.
CrossAvg₃ = mean(drop ConvNeXt-T, Swin-T, YOLOX-S). QUA nếu đồng thời:
  ΔCrossAvg₃(E2−E0) >= 3 (điểm); ΔCrossAvg₃(E2−E1) > 0 VÀ cận dưới CI > 0;
  E2−E0 > 0 ở ConvNeXt-T, Swin-T, YOLOX-S (điểm); drop R101(E2) >= drop R101(E0) − 3.
Tập ảnh có thông tin (>= 2 cặp hợp lệ): chỉ báo cáo (bootstrap riêng trên tập con).

Output: results/kg_pilot/dev300/{screen.json, <condition>/{metrics.json, per_image_metrics.csv,
config.json, runtime.json}}.
Chạy: python scripts/pilot_kg_screen.py
"""
import csv
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
OUT_DIR = os.path.join(REPO_ROOT, "results/kg_pilot/dev300")
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
BOOT_SEED, N_BOOT = 2026, 1000
COND = {"E0": "OSFD", "E1": "OSFD-KG-E1", "E2": "OSFD-KG-E2"}
CTRL = ["r50", "r101", "convnext_t", "swin_t"]
GEN = ["yolox_s", "dino_swin_l"]
CROSS3 = ["convnext_t", "swin_t", "yolox_s"]
CONFIG = {
    "E0": {"method": "OSFD full recipe (MI μ=1 + RRB θ=7, l_s=10, ρ=0.8, s_max=1.1, σ=6), k=3"},
    "E1": {"method": "OSFD + view cặp (thay view 2 RRB), cặp = geometry-matched random control",
           "pair_selection": "q ~ p ∝ S (KG); dùng e1 ~ đều trên {r : |log(A_r/A_q)| <= log 2}"},
    "E2": {"method": "OSFD + view cặp (thay view 2 RRB), cặp = KG",
           "pair_selection": "q ~ p ∝ S (PPMI, min support 20)"},
}
COMMON = {"split": "dev300", "surrogate": "Mask R-CNN R50", "epsilon_255": 5.0, "alpha": 1.0,
          "budget_B": 50, "seed_image": "SEED + i (SEED = 0, i = chỉ số dev300)", "margin": 0.15,
          "area_ratio": 2.0, "kg": "data/kg/coco80_vg_relation_matrix.npy",
          "rule": "docs/progress_log.md 2026-10-03 (LOCKED)"}


def main():
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    n = len(img_ids)
    coco = COCO(ANN_FILE)
    dets = {}
    with open(os.path.join(ART, "dets_kg.json")) as f:
        ctrl = json.load(f)
    for k in CTRL:
        dets[k] = {c: ctrl[c][k] for c in ["clean"] + list(COND.values())}
    for k in GEN:
        with open(os.path.join(ART, "gen_dets", f"{k}.json")) as f:
            dets[k] = json.load(f)
    stats = {}
    with open(os.path.join(RUN_DIR, "attack_stats.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            stats[(r["method"], r["img_id"])] = r  # dòng sau cùng thắng
    n_pairs = np.array([stats[(COND["E2"], i)]["n_pairs"] for i in img_ids])
    informative = n_pairs >= 2

    # AP + bootstrap. Hàng 0 = mẫu gốc. Tập con có thông tin: bootstrap riêng trên tập con.
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    inf_pos = np.flatnonzero(informative)
    Wi = np.zeros((1 + N_BOOT, n))
    Wi[0, inf_pos] = 1
    Wi[1:, inf_pos] = bootstrap_indices(len(inf_pos), N_BOOT, BOOT_SEED)
    ap, ap_inf = {}, {}
    for k in CTRL + GEN:
        ap[k], ap_inf[k] = {}, {}
        for c in dets[k]:
            ev = CocoImageEval(coco, dets[k][c], img_ids)
            ap[k][c] = np.array([ev.ap(w) for w in W]) * 100
            ap_inf[k][c] = np.array([ev.ap(w) for w in Wi]) * 100
        print(f"[screen] AP xong: {k}", flush=True)

    def drops(A):
        return {e: {k: 100 * (A[k]["clean"][:, 0] - A[k][m][:, 0]) / A[k]["clean"][:, 0]
                    for k in CTRL + GEN} for e, m in COND.items()}
    D, Di = drops(ap), drops(ap_inf)
    cross3 = lambda d, e: np.mean([d[e][k] for k in CROSS3], axis=0)
    S = lambda a: summarize(a[0], a[1:])

    def diffs(d):
        out = {}
        for a, b in [("E2", "E0"), ("E2", "E1"), ("E1", "E0")]:
            row = {k: S(d[a][k] - d[b][k]) for k in CTRL + GEN}
            row["cross_avg3"] = S(cross3(d, a) - cross3(d, b))
            out[f"{a}-{b}"] = row
        return out
    delta, delta_inf = diffs(D), diffs(Di)

    d20, d21 = delta["E2-E0"], delta["E2-E1"]
    checks = {
        "cross_avg3_E2-E0>=3": d20["cross_avg3"]["point"] >= 3,
        "cross_avg3_E2-E1>0": d21["cross_avg3"]["point"] > 0,
        "cross_avg3_E2-E1_ci_lo>0": d21["cross_avg3"]["ci95"][0] > 0,
        "convnext_t_E2-E0>0": d20["convnext_t"]["point"] > 0,
        "swin_t_E2-E0>0": d20["swin_t"]["point"] > 0,
        "yolox_s_E2-E0>0": d20["yolox_s"]["point"] > 0,
        "r101_E2>=E0-3": d20["r101"]["point"] >= -3,
    }
    passed = all(checks.values())

    # Per-image + output theo điều kiện.
    idx = boot_weights(n, N_BOOT, BOOT_SEED)  # cùng luồng RNG với bootstrap_indices -> paired
    for e, m in COND.items():
        d = os.path.join(OUT_DIR, e)
        os.makedirs(d, exist_ok=True)
        pit = {k: per_image_transfer(coco, dets[k]["clean"], dets[k][m], img_ids) for k in CTRL + GEN}
        rows = [stats[(m, i)] for i in img_ids]
        with open(os.path.join(d, "per_image_metrics.csv"), "w", newline="") as f:
            cols = ["img_id", "n_pairs", "informative", "linf_255", "l2", "runtime_s", "backward",
                    "forward_views", "forward_no_grad", "frac_steps_E1_ne_E2",
                    "median_abs_log_area_ratio", "same_class_rate_E1", "same_class_rate_E2"]
            cols += [f"{q}_{k}" for k in CTRL + GEN for q in ("suppression", "retained")]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for j, (img_id, r) in enumerate(zip(img_ids, rows)):
                rec = {c: r.get(c) for c in cols if c in r}
                rec.update(img_id=img_id, n_pairs=int(n_pairs[j]), informative=bool(informative[j]),
                           forward_no_grad=r.get("forward_no_grad", 1))
                for k in CTRL + GEN:
                    rec[f"suppression_{k}"] = pit[k]["suppression"][j]
                    rec[f"retained_{k}"] = pit[k]["retained"][j]
                w.writerow(rec)

        def mean_ci(v):
            return summarize(np.nanmean(v), np.array([np.nanmean(v[r]) for r in idx]))
        runtime = {"runtime_s_per_img_mean": float(np.mean([r["runtime_s"] for r in rows])),
                   "runtime_s_total": float(np.sum([r["runtime_s"] for r in rows])),
                   "backward_per_img": rows[0]["backward"], "forward_views_per_img": rows[0]["forward_views"],
                   "forward_no_grad_per_img_mean": float(np.mean([r.get("forward_no_grad", 1) for r in rows]))}
        metrics = {
            "condition": e, "method": m, "n_images": n, "n_informative": int(informative.sum()),
            "clean_ap": {k: S(ap[k]["clean"][:, 0]) for k in CTRL + GEN},
            "adv_ap": {k: S(ap[k][m][:, 0]) for k in CTRL + GEN},
            "clean_ap50": {k: S(ap[k]["clean"][:, 1]) for k in CTRL + GEN},
            "adv_ap50": {k: S(ap[k][m][:, 1]) for k in CTRL + GEN},
            "relative_ap_drop": {k: S(D[e][k]) for k in CTRL + GEN},
            "cross_avg3": S(cross3(D, e)),
            "transfer_gap": {**{f"r101_minus_{k}": S(D[e]["r101"] - D[e][k]) for k in CROSS3},
                             "r101_minus_cross_avg3": S(D[e]["r101"] - cross3(D, e))},
            "suppression_mean": {k: mean_ci(pit[k]["suppression"]) for k in CTRL + GEN},
            "retained_confidence_mean": {k: mean_ci(pit[k]["retained"]) for k in CTRL + GEN},
            "perturbation": {"linf_255_max": max(r["linf_255"] for r in rows),
                             "l2_mean": float(np.mean([r["l2"] for r in rows]))},
            "compute": runtime,
            "not_computed": ["ASR", "APloc", "CSR", "LPIPS"],
        }
        for name, obj in [("metrics.json", metrics), ("runtime.json", runtime),
                          ("config.json", {**COMMON, **CONFIG[e]})]:
            with open(os.path.join(d, name), "w") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

    kg_rows = [stats[(COND["E2"], i)] for i in img_ids]
    has_log = [r for r in kg_rows if r.get("frac_steps_E1_ne_E2") is not None]
    out = {
        "rule": "progress_log 2026-10-03 (LOCKED), bước 1 — sàng lọc dev300",
        "n_images": n, "n_informative": int(informative.sum()),
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "cross_avg3": ["convnext_t", "swin_t", "yolox_s"],
        "relative_ap_drop": {e: {k: S(D[e][k]) for k in CTRL + GEN} for e in COND},
        "cross_avg3_by_condition": {e: S(cross3(D, e)) for e in COND},
        "delta": delta, "checks": checks, "pass": passed,
        "informative_subset_report_only": {
            "relative_ap_drop": {e: {k: S(Di[e][k]) for k in CTRL + GEN} for e in COND},
            "delta": delta_inf},
        "pair_diagnostics": {
            "frac_steps_E1_ne_E2_mean": float(np.mean([r["frac_steps_E1_ne_E2"] for r in has_log])),
            "same_class_rate_E2": float(np.mean([r["same_class_rate_E2"] for r in has_log])),
            "same_class_rate_E1": float(np.mean([r["same_class_rate_E1"] for r in has_log])),
            "n_kg_degenerate": int(sum(r["kg_degenerate"] for r in kg_rows))},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUT_DIR, "screen.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    fmt = lambda s: f"{s['point']:+6.1f} [{s['ci95'][0]:+5.1f},{s['ci95'][1]:+5.1f}]"
    keys = CTRL + GEN + ["cross_avg3"]
    print(f"\n=== relative AP drop (dev300, n={n}) ===")
    print(f"{'':<4}" + "".join(f"{k:>13}" for k in keys))
    for e in COND:
        vals = [D[e][k][0] for k in CTRL + GEN] + [cross3(D, e)[0]]
        print(f"{e:<4}" + "".join(f"{v:>13.1f}" for v in vals))
    for title, dd in [("TOÀN BỘ ẢNH (chính)", delta), (f"ẢNH CÓ THÔNG TIN n={informative.sum()} (báo cáo)", delta_inf)]:
        print(f"\n=== Δ {title} ===")
        for name, row in dd.items():
            print(f"{name:<6}" + "  ".join(f"{k} {fmt(row[k])}" for k in ["r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l", "cross_avg3"]))
    print("\n" + "  ".join(f"{k}: {'✓' if v else '✗'}" for k, v in checks.items()))
    print(f"=> sàng lọc dev300: {'QUA' if passed else 'KHÔNG QUA (NO-GO hướng KG)'}")


if __name__ == "__main__":
    main()
