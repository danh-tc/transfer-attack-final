"""E82 (bước 2 của docs/plan_paper_lock.md): strict-1 + WP — phép thử CUỐI của recipe ENS-WP trước khi đóng băng.

Nền: E70 strict-1 noWP Swin-L 55.1 > E71 khối-2 noWP 52.5, và WP cho +7.8 ở khối-2. Nhưng WP2 đặt θ' ở CUỐI bước
và dùng ở bước sau, nên khi luân phiên từng bước (steps=1/lời gọi) caller khôi phục θ0 ngay và WP vô hiệu. wp_in /
wp_out (attack/methods/pia.py) giữ θ' RIÊNG cho từng surrogate qua các lượt -> mỗi surrogate dùng θ' từ lượt trước
của CHÍNH nó (đúng ngữ nghĩa WP2 trong chuỗi bước của surrogate đó). Regression test: scripts/wp_carry_unit_test.py.

Ô (B=36, momentum CHUNG, λ R50=6 / CNX=0.2, OCT 1V, L_OSFD C3-C5 + cls RoI, WP ρ0.002 (weight_linear cho CNX), ε=5):
  ENS-WP-B36   khối 2 bước, luân phiên R50/CNX (18 khối, 9/9)          = recipe hiện tại, sinh LẠI trên cùng máy
  ENS-S1WP     luân phiên từng bước (36 lượt, 18/18) + mang θ' qua lượt
Seed (cả hai ô, cùng quy tắc): torch + python random + numpy = 1000·(n+1) + b, b = chỉ số lời gọi (khối hoặc lượt).
Quy tắc phán quyết: progress_log entry "E82" (khóa trước khi chạy).

  python scripts/e82_strict1_wp.py                          # dev30, panel gọn 4 model, boot 100 (sàng lọc)
  python scripts/e82_strict1_wp.py --list dev100 --panel full --boot 300   # xác nhận (chỉ khi GO)
"""
import argparse
import json
import os
import random
import sys
import time

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e81_ens_alloc_rho import kw_for, COMMON, R, C

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
STEPS, RHO, BOOT_SEED = 36, 0.002, 2026
BASE, NEW = "ENS-WP-B36", "ENS-S1WP"
CELLS = {BASE: dict(block=2, carry=False), NEW: dict(block=1, carry=True)}
PANELS = {"small": ["dino_swin_l", "r101", "fcos_r50", "dino_r50"],
          "full": ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l",
                   "dino_swin_l"]}
RES3 = ["r101", "fcos_r50", "dino_r50"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def attack(ids, cond, run_dir):
    block, carry = CELLS[cond]["block"], CELLS[cond]["carry"]
    out_dir = os.path.join(ART, run_dir, "adv", cond)
    if all(os.path.exists(os.path.join(out_dir, f"{i}.png")) for i in ids):
        print(f"[e82] {cond}: PNG đã đủ, bỏ qua sinh", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    models = {R: load_model(R, "cuda:0"), C: load_model(C, "cuda:0")}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in models}
    kw = {k: kw_for(k, RHO) for k in models}
    ncalls = STEPS // block
    print(f"[e82] {cond}: {ncalls} lời gọi × {block} bước, luân phiên {R}/{C}, carry θ'={carry}", flush=True)
    t0 = time.time()
    for n, i in enumerate(ids):
        if os.path.exists(os.path.join(out_dir, f"{i}.png")):
            continue
        orig = ds[R][i]["orig_inputs"].cuda().float()
        delta, mom = None, None
        wp_state = {R: None, C: None}                 # θ' riêng từng surrogate
        for b in range(ncalls):
            key = (R, C)[b % 2]
            seed_all(1000 * (n + 1) + b)
            extra = {}
            if carry:
                out = []
                extra = dict(wp_in=wp_state[key], wp_out=out)
            delta, mom = geo_osfd_attack(models[key], orig, ds[key][i]["data_sample"], steps=block,
                                         init_noise=delta, mom_in=mom, return_state=True,
                                         **kw[key], **COMMON, **extra)
            if carry:
                wp_state[key] = out
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= 5.0 + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), os.path.join(out_dir, f"{i}.png"))
        print(f"[e82 {cond}] {n + 1}/{len(ids)}  {(time.time() - t0) / (n + 1):.1f}s/ảnh", flush=True)
    del models
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="dev30", choices=["dev30", "dev100"])
    ap.add_argument("--panel", default="small", choices=list(PANELS))
    ap.add_argument("--boot", type=int, default=100)
    args = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    tag = gpu.split()[-1].replace("-", "").lower()
    lim = {"dev30": 100, "dev100": 100}[args.list]   # PNG nằm chung run dir limit100 (dev30 ⊂ dev100)
    run_dir = f"dev300_B36_eps5_limit{lim}_ens_e82_{tag}"
    out_json = os.path.join(REPO_ROOT, f"results/mechanism/e82_strict1_wp_{args.list}_{args.panel}.json")
    print(f"[e82] GPU = {gpu} | run dir {run_dir} | mọi ô sinh+eval TRÊN CÙNG máy", flush=True)

    coco = COCO(ANN)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, f"data/image_lists/{args.list}.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == {"dev30": 30, "dev100": 100}[args.list], len(ids)
    for cond in CELLS:
        attack(ids, cond, run_dir)

    panel = PANELS[args.panel]
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, args.boot, BOOT_SEED)])
    from mmdet.apis import inference_detector
    res = {"list": args.list, "n_images": n, "n_boot": args.boot, "gpu": gpu, "steps": STEPS, "rho": RHO,
           "cells": CELLS, "drop": {}, "diff": {}}
    if os.path.exists(out_json):
        old = json.load(open(out_json))
        if old.get("cells") == res["cells"] and old.get("gpu") == gpu and old.get("n_boot") == args.boot:
            res = old
    for key in panel:
        if key in res["drop"]:
            continue
        model = load(key)

        def ap_of(paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            return np.array([CocoImageEval(coco, dets, ids).ap(w) for w in W])[:, 0] * 100
        clean = ap_of([os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        s = {c: 100 * (clean - ap_of([os.path.join(ART, run_dir, "adv", c, f"{i}.png") for i in ids])) / clean
             for c in CELLS}
        res["drop"][key] = {c: summarize(s[c][0], s[c][1:]) for c in CELLS}
        d = s[NEW] - s[BASE]                                  # paired: cùng W, cùng clean
        res["diff"][key] = summarize(d[0], d[1:])
        res.setdefault("_samples", {})[key] = {c: s[c].tolist() for c in CELLS}
        del model
        torch.cuda.empty_cache()
        print(f"{key:12} {BASE} {s[BASE][0]:.1f}  {NEW} {s[NEW][0]:.1f}  Δ {d[0]:+.1f}", flush=True)
        json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)

    # ---- Phán quyết theo quy tắc khóa (progress_log "E82") ----
    S = {k: {c: np.array(v) for c, v in res["_samples"][k].items()} for k in panel}
    r3 = {c: np.mean([S[m][c] for m in RES3], axis=0) for c in CELLS}
    dr3 = r3[NEW] - r3[BASE]
    dsl = res["diff"]["dino_swin_l"]
    sr3 = summarize(dr3[0], dr3[1:])
    print(f"\n=== E82 @ {args.list}, {gpu}, B={STEPS}, boot {args.boot} ===")
    print(f"{'model':12} {BASE:>11} {NEW:>9} {'Δ':>6}  CI(Δ paired)")
    for m in panel:
        print(f"{m:12} {res['drop'][m][BASE]['point']:11.1f} {res['drop'][m][NEW]['point']:9.1f} "
              f"{res['diff'][m]['point']:+6.1f}  {[round(x, 1) for x in res['diff'][m]['ci95']]}")
    print(f"{'RES3':12} {r3[BASE][0]:11.1f} {r3[NEW][0]:9.1f} {dr3[0]:+6.1f}  {[round(x, 1) for x in sr3['ci95']]}")
    if args.panel == "full":
        t7 = {c: np.mean([S[m][c] for m in T7], axis=0) for c in CELLS}
        print(f"{'T7':12} {t7[BASE][0]:11.1f} {t7[NEW][0]:9.1f} {t7[NEW][0] - t7[BASE][0]:+6.1f}")
    go = dsl["point"] >= 4 and dsl["ci95"][0] > 0 and dr3[0] >= -3
    print(f"\n=> {'GO' if go else 'NO-GO'}: ΔSwin-L {dsl['point']:+.1f} CI {[round(x, 1) for x in dsl['ci95']]}, "
          f"ΔRES3 {dr3[0]:+.1f}")
    print("   " + ("GO ở dev30 -> xác nhận dev100 (--list dev100 --panel full --boot 300) trước khi thay ENS-WP"
                   if go and args.list == "dev30" else
                   "GO ở dev100 -> ENS-S1WP thay ENS-WP, đóng băng" if go else
                   "NO-GO -> giữ ENS-WP hiện tại, đóng băng"))
    res["verdict"] = {"go": bool(go), "dSwinL": dsl, "dRES3": sr3}
    json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
