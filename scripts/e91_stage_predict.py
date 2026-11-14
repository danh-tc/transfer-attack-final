"""E91 — dự đoán bác bỏ được của stage sensitivity trên backbone CHƯA dùng (x101, regnetx). Quy tắc khóa: progress_log
entry "E91" 2026-11-14. Scientific test của nguyên lý (docs/framework_v0.md §4), KHÔNG phải method.

E91a (`ratio`, ≤ 5' GPU): r = ‖g2‖/‖g345‖ (∇ₓ L_OSFD k=3, ảnh SẠCH, không aug, fp32; = e87_diag) trên devC300[60:80]
  (20 ảnh). Đo cả r50 + convnext_t trên cùng ảnh làm kiểm tra hiệu chuẩn (không cổng). TB qua ảnh, CI bootstrap ảnh.
E91b (`stage`, ≤ 15' GPU): mỗi backbone 2 ô — Δ_stage = Transfer(C3–C5) − Transfer(C2–C5); objective L_OSFD thuần,
  geo_osfd_attack OCT 1 view (= E85 OCT_KW), amp, B = 36 bước = 36 backward, ε = 5; CHỈ đổi feat_stages. Seed
  1000·(n+1), n = chỉ số trong danh sách E91b = devC300[80:80+N]. Eval T5C, relative AP drop, paired bootstrap 1000
  (seed 2026) trong tiến trình: Δ_stage từng backbone + hiệu Δ giữa 2 backbone.
  OMP_NUM_THREADS=4 python scripts/e91_stage_predict.py ratio
  OMP_NUM_THREADS=4 python scripts/e91_stage_predict.py stage --n N [--smoke]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, to_coco_dets
from attack.methods.pia import _view_loss, geo_osfd_attack
from attack.models import load_model
from attack.preprocess import extract_features, resize_to_model, to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.run_baselines import _C2OCT

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
NEW = ["x101", "regnetx"]
CALIB = ["r50", "convnext_t"]
T5C = ["fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
STEPS, EPS, BOOT_SEED, NB = 36, 5.0, 2026, 1000
OCT_KW = dict(geo="elem", elem=_C2OCT["elem"], n_views=1, amp=True)
STAGES = {"C25": None, "C35": (1, 2, 3)}
# ngưỡng EXPLORATORY (từ E85–E87, chưa hiệu chuẩn độc lập), khóa trước E91
R_LO, R_HI = 0.02, 0.2
LOW_ABS, MID_MAX, HIGH_MIN, WIDE_HALF = 1.5, 6.0, 4.0, 4.0
OUT_A = os.path.join(REPO_ROOT, "results/mechanism/e91a_ratio_devC.json")
OUT_B = os.path.join(REPO_ROOT, "results/mechanism/e91b_stage_devC.json")


def devc():
    return load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/devC300.csv"))


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def klass(r):
    return "low" if r < R_LO else "mid" if r < R_HI else "high"


def cmd_ratio():
    ids = devc()[60:80]
    out = {"ids": ids, "thresholds_exploratory": [R_LO, R_HI], "models": {}}
    rng = np.random.default_rng(BOOT_SEED)
    bs = [rng.integers(0, len(ids), len(ids)) for _ in range(NB)]
    t0 = time.time()
    for key in NEW + CALIB:
        model = load_model(key, "cuda:0"); model.eval()
        for q in model.parameters():
            q.requires_grad_(False)
        ds = {x["img_id"]: x for x in target_dataset(key, ids)}
        rs = []
        for i in ids:
            orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]
            hw = tuple(dsm.img_shape)
            with torch.no_grad():
                fc = tuple(f.detach() for f in extract_features(model, resize_to_model(orig, hw), dsm))
            g = {}
            for nm, st in (("g2", (0,)), ("g345", (1, 2, 3))):
                x = orig.clone().requires_grad_(True)
                gx, = torch.autograd.grad(_view_loss(fc, 3.0, st)(model, resize_to_model(x, hw), dsm), x)
                g[nm] = gx.double().flatten()
            rs.append(float(g["g2"].norm() / g["g345"].norm()))
        rs = np.array(rs)
        out["models"][key] = {"r_mean": round(float(rs.mean()), 5), "ci95": ci([rs[b].mean() for b in bs]),
                              "r_median": round(float(np.median(rs)), 5), "class": klass(rs.mean()),
                              "role": "new" if key in NEW else "calib"}
        print(f"[e91a] {key:10} r {rs.mean():.4f} {out['models'][key]['ci95']}  class {klass(rs.mean())}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    out["time_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT_A, "w"), indent=1)
    print(f"[e91a] -> {OUT_A}")


def cmd_stage(n, smoke):
    ids = devc()[80:80 + (2 if smoke else n)]
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    from mmdet.apis import inference_detector
    t0 = time.time()
    adv = {}  # (backbone, stage) -> {img_id: uint8 HWC BGR}
    clean = {}
    for key in NEW:
        model = load_model(key, "cuda:0")
        ds = {x["img_id"]: x for x in target_dataset(key, ids)}
        for st, fs in STAGES.items():
            adv[(key, st)] = {}
            for k, i in enumerate(ids):
                orig = ds[i]["orig_inputs"].cuda().float()
                seed_all(1000 * (k + 1))
                d = geo_osfd_attack(model, orig, ds[i]["data_sample"], steps=STEPS, epsilon=EPS,
                                    feat_stages=fs, **OCT_KW)
                a = to_adv_image(orig, d)
                assert float((a - orig).abs().max()) <= EPS + 1e-4
                adv[(key, st)][i] = a.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                if key == NEW[0] and st == "C25":
                    clean[i] = orig.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            print(f"[e91b] gen {key}-{st} xong ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    t_gen = time.time() - t0
    W = np.vstack([np.ones((1, len(ids))), bootstrap_indices(len(ids), 20 if smoke else NB, BOOT_SEED)])
    S = {}
    for tg in T5C:
        m = load(tg)

        def ap(imgs):
            dets = []
            for i in ids:
                dets += to_coco_dets(inference_detector(m, imgs[i]), i, cat_ids)
            ev = CocoImageEval(coco, dets, ids)
            return np.array([ev.ap(w)[0] for w in W]) * 100
        c = ap(clean)
        for cell, imgs in adv.items():
            S[(tg, cell)] = 100 * (c - ap(imgs)) / c
        del m
        torch.cuda.empty_cache()
        print(f"[e91b] eval {tg} xong ({time.time() - t0:.0f}s)", flush=True)
    t_all = time.time() - t0
    if smoke:
        print(f"[smoke] n {len(ids)}  gen {t_gen:.0f}s  total {t_all:.0f}s  -> /ảnh ≈ {t_all / len(ids):.1f}s "
              f"(KHÔNG đọc số)")
        return
    T = lambda key, st: np.mean([S[(tg, (key, st))] for tg in T5C], 0)
    res = {"ids": ids, "n": len(ids), "time_s": {"gen": round(t_gen, 1), "total": round(t_all, 1)},
           "cells_T5C": {f"{k}-{st}": [round(float(T(k, st)[0]), 2), ci(T(k, st)[1:])] for k in NEW for st in STAGES},
           "delta_stage": {}, "per_target_delta": {}}
    D = {k: T(k, "C35") - T(k, "C25") for k in NEW}
    for k in NEW:
        res["delta_stage"][k] = {"point": round(float(D[k][0]), 2), "ci95": ci(D[k][1:])}
        res["per_target_delta"][k] = {tg: round(float((S[(tg, (k, 'C35'))] - S[(tg, (k, 'C25'))])[0]), 2) for tg in T5C}
    dd = D["regnetx"] - D["x101"]
    res["delta_regnetx_minus_x101"] = {"point": round(float(dd[0]), 2), "ci95": ci(dd[1:])}

    # ---- quy tắc khóa (cần E91a) ----
    A = json.load(open(OUT_A))["models"]
    ver = {}
    for k in NEW:
        c, p, (lo, hi) = A[k]["class"], res["delta_stage"][k]["point"], res["delta_stage"][k]["ci95"]
        wide = (hi - lo) / 2 > WIDE_HALF
        if c == "low":
            ok = abs(p) < LOW_ABS
        elif c == "mid":
            ok = 0 < p < MID_MAX
        else:
            ok = p >= HIGH_MIN and lo > 0
        opposite = hi < 0 if c in ("mid", "high") else False
        ver[k] = {"class": c, "r": A[k]["r_mean"], "prediction_met": bool(ok), "ci_too_wide": bool(wide),
                  "strong_opposite": bool(opposite), "effect_lo_gt0": bool(lo > 0)}
    ra, rb = A["regnetx"]["r_mean"], A["x101"]["r_mean"]
    order_ok = None if A["regnetx"]["class"] == A["x101"]["class"] else bool(
        (res["delta_regnetx_minus_x101"]["point"] > 0) == (ra > rb))
    res["verdict_per_backbone"], res["order_prediction_met"] = ver, order_ok
    if any(v["strong_opposite"] for v in ver.values()):
        verdict = "NO-GO"
    elif all(v["class"] == "low" for v in ver.values()):
        verdict = "INCONCLUSIVE (cả hai r thấp: chỉ kiểm lớp low)"
    elif all(v["prediction_met"] for v in ver.values()) and order_ok is not False and any(
            v["class"] != "low" and v["effect_lo_gt0"] for v in ver.values()):
        verdict = "GO (pilot ủng hộ -> confirmation có power)"
    elif any(v["ci_too_wide"] for v in ver.values() if v["class"] != "low"):
        verdict = "INCONCLUSIVE (CI quá rộng)"
    else:
        verdict = "PARTIAL / REPORT (dự đoán đúng một phần)"
    res["verdict"] = verdict
    json.dump(res, open(OUT_B, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("cells_T5C", "delta_stage", "per_target_delta", "delta_regnetx_minus_x101",
                                          "verdict_per_backbone", "order_prediction_met", "verdict", "time_s")},
                     indent=1, ensure_ascii=False))
    print(f"[e91b] -> {OUT_B}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ratio", "stage"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cmd_ratio() if a.cmd == "ratio" else cmd_stage(a.n, a.smoke)


if __name__ == "__main__":
    main()
