#!/usr/bin/env python3
"""E79a-Basin (progress_log 2026-10-31): Transferable Basin Test — PILOT dev30, DIAGNOSTIC, KHÔNG dựng method.
Giả thuyết (second-order, khác E64-E78 first-order): ENS-WP transfer tốt vì δ nằm trong vùng detection-loss ÍT NHẠY với
nhiễu nhỏ (basin phẳng), không chỉ đạt loss cao tại điểm đặc thù surrogate. Nếu đúng -> objective R50-only tối ưu robustness.
Reuse δ = adv_png − clean_jpg (như E64) từ RD-R50/RD-CNX/ENS-WP (E73 PNG dev100). Sinh lân cận δ'=clip(δ+N(0,σ),±ε), đo:
  - target-side retention = APdrop(δ')/APdrop(δ)  (gần 1 = basin phẳng trên target); Gate A kiểm soát base-APdrop bằng TỶ SỐ.
  - Gate B screen: retention QUAN SÁT TỪ R50 (r50 là model eval) có dự báo retention target khác-họ không (corr theo ảnh).
Oracle/target CHỈ để phân tích. Kiểm soát saturation biên: báo mean‖δ'−δ‖ thực sau clip theo cond.
  python scripts/e79_basin.py   -> results/mechanism/e79_basin.json"""
import json
import os
import sys

import cv2
import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from scripts.e64_cnx_drop import load

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
TMP = os.path.join(REPO_ROOT, "artifacts/e79")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e79_basin.json")
FIX = {"RD-R50": ("dev300_B36_eps5_limit100", "C2-OCT-1V-WP2-RD"),
       "RD-CNX": ("dev300_B36_eps5_limit100_sur-convnext_t", "C2-OCT-1V-WP2-RD-CNX"),
       "ENS-WP": ("dev300_B36_eps5_limit100_ens", "ENS-WP")}
CONDS = ["RD-R50", "RD-CNX", "ENS-WP"]
MODELS = ["r50", "dino_swin_l", "swin_t", "r101"]  # r50 = quan sát-được-từ-surrogate (Gate B); swin* = khác họ
CROSS = ["dino_swin_l", "swin_t"]
EPS, SIGMA, K, SEED = 5.0, 2.0, 3, 79


def adv_png(cond, i):
    run, folder = FIX[cond]
    return os.path.join(ART, run, "adv", folder, f"{i}.png")


def build_neighbors(coco, ids):
    """Sinh δ' = clip(δ+N(0,σ),±ε) -> PNG; trả mean‖δ'−δ‖ thực theo cond. k=0 là base (δ). Idempotent."""
    rng = np.random.default_rng(SEED)
    moved = {c: [] for c in CONDS}
    for c in CONDS:
        for i in ids:
            clean = cv2.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32)
            d = cv2.imread(adv_png(c, i)).astype(np.float32) - clean
            for k in range(K + 1):
                outp = os.path.join(TMP, c, f"k{k}", f"{i}.png")
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                if k == 0:
                    dp = np.clip(d, -EPS, EPS)
                else:
                    z = rng.normal(0, SIGMA, size=d.shape)
                    dp = np.clip(d + z, -EPS, EPS)
                    moved[c].append(float(np.abs(dp - np.clip(d, -EPS, EPS)).mean()))
                cv2.imwrite(outp, np.clip(np.round(clean + dp), 0, 255).astype(np.uint8))
    return {c: round(float(np.mean(v)), 3) for c, v in moved.items()}


def main():
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    assert len(ids) == 30
    cat_ids = sorted(coco.getCatIds())
    moved = build_neighbors(coco, ids)
    from mmdet.apis import inference_detector
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        "n_images": 30, "sigma": SIGMA, "K": K, "moved_after_clip": moved, "ap": {}}
    for m in MODELS:
        if m in res["ap"]:
            continue
        model = load(m)

        def ap_img(path, i):  # AP per-ảnh (ids=[i]) để tính retention theo ảnh
            dets = to_coco_dets(inference_detector(model, path), i, cat_ids)
            return float(CocoImageEval(coco, dets, [i]).ap(np.ones(1))[0]) * 100
        clean = {i: ap_img(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]), i) for i in ids}
        res["ap"][m] = {}
        for c in CONDS:
            per = {}  # i -> [AP base, AP k1, k2, k3]
            for i in ids:
                per[str(i)] = [ap_img(os.path.join(TMP, c, f"k{k}", f"{i}.png"), i) for k in range(K + 1)]
            res["ap"][m][c] = {"clean": {str(i): clean[i] for i in ids}, "adv": per}
        del model
        import torch; torch.cuda.empty_cache()
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
        print(f"[e79] {m} xong", flush=True)

    # --- phân tích ---
    def drop(m, c, i, k):  # AP-drop % của ảnh i, cond c, neighbor k (0=base)
        cl = res["ap"][m][c]["clean"][str(i)]
        return 100 * (cl - res["ap"][m][c]["adv"][str(i)][k]) / cl if cl > 1e-6 else np.nan

    def dataset_retention(m, c):  # (mean neighbor APdrop)/(base APdrop), gộp ảnh (robust cho Gate A)
        base = np.nanmean([drop(m, c, i, 0) for i in ids])
        nb = np.nanmean([drop(m, c, i, k) for i in ids for k in range(1, K + 1)])
        return round(float(nb / base), 4) if base > 1e-6 else float("nan"), round(float(base), 2)

    print(f"\n=== E79a Basin (σ={SIGMA}, K={K}; moved-after-clip {moved}) ===")
    print(f"{'model':12}" + "".join(f"{c+'(ret|base)':>22}" for c in CONDS))
    ret = {m: {} for m in MODELS}
    for m in MODELS:
        row = f"{m:12}"
        for c in CONDS:
            r, b = dataset_retention(m, c); ret[m][c] = (r, b)
            row += f"{r:>10.3f} | {b:>7.1f}  "
        print(row)

    print("\n--- Gate A (ENS basin phẳng hơn trên target khác-họ, kiểm soát base qua TỶ SỐ retention) ---")
    gateA = all(ret[m]["ENS-WP"][0] >= max(ret[m]["RD-R50"][0], ret[m]["RD-CNX"][0]) + 0.03 for m in CROSS)
    for m in CROSS:
        print(f"  {m:12} retention ENS {ret[m]['ENS-WP'][0]:.3f} vs RD-R50 {ret[m]['RD-R50'][0]:.3f} / RD-CNX {ret[m]['RD-CNX'][0]:.3f}")
    print(f"  Gate A = {gateA}")

    print("\n--- Gate B screen (retention QUAN SÁT TỪ R50 có dự báo retention target khác-họ không; corr theo ảnh×cond) ---")
    # per-image retention: mean_k drop(i,k>0) / drop(i,0)
    def img_ret(m, c, i):
        b = drop(m, c, i, 0)
        if not (b > 1e-6):
            return np.nan
        return np.nanmean([drop(m, c, i, k) for k in range(1, K + 1)]) / b
    gateB_corr = {}
    for tgt in CROSS:
        xs, ys = [], []
        for c in CONDS:
            for i in ids:
                a, bb = img_ret("r50", c, i), img_ret(tgt, c, i)
                if np.isfinite(a) and np.isfinite(bb):
                    xs.append(a); ys.append(bb)
        corr = round(float(np.corrcoef(xs, ys)[0, 1]), 4) if len(xs) > 3 else 0.0
        gateB_corr[tgt] = {"corr": corr, "n": len(xs)}
        print(f"  corr(r50-retention, {tgt}-retention) = {corr}  (n={len(xs)})")
    gateB = all(abs(gateB_corr[t]["corr"]) >= 0.30 for t in CROSS)
    print(f"  Gate B (|corr| ≥ 0.30 cả hai target) = {gateB}")

    res["verdict"] = {"gateA": bool(gateA), "gateB": bool(gateB),
                      "retention": {m: {c: ret[m][c] for c in CONDS} for m in MODELS}, "gateB_corr": gateB_corr}
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("\nVERDICT:",
          "E79a GO (cả 2 gate) -> E79b confirm dev100" if gateA and gateB else
          "E79a NO-GO -> " + ("Gate A fail (ENS KHÔNG phẳng hơn target sau kiểm soát base)" if not gateA else
                              "Gate A pass nhưng Gate B fail (R50 KHÔNG quan sát được tính chất) -> không xây method") +
          "; đóng hướng neighborhood-stability")


if __name__ == "__main__":
    main()
