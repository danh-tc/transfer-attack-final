"""E80-ENSB (progress_log 2026-11-09): quét ngân sách B cho ENS-WP — B36 vs B50, **CẢ HAI sinh trên CÙNG máy**.

Bản đầu chỉ chạy B50 rồi so mốc E71 (A4500) → phán quyết LẪN (lệch máy ~8 ≫ tín hiệu kỳ vọng +2.4).
Bản này thêm ô B36 trên chính máy đang chạy → so sánh trong-cùng-máy hợp lệ, và tính **CI hiệu số PAIRED**
trong cùng tiến trình (cùng bộ trọng số bootstrap W) — khắc phục chỗ E73/E74 chỉ lưu point/ci95 nên hậu kỳ
không tính được CI của hiệu số.

Cả hai ô y hệt E71 ENS-WP, CHỈ khác STEPS: luân phiên R50/ConvNeXt theo khối 2 bước, momentum CHUNG
(mom_in/return_state), WP ρ0.002 (weight_linear cho ConvNeXt), λ R50=6 / CNX=0.2, OCT 1V,
L_OSFD C3-C5 + cls RoI, ε=5, seed 1000·(n+1)+b.
  B36 = 18 khối, cân 9 R50 / 9 ConvNeXt (y hệt E71).
  B50 = 25 khối, 13 R50 / 12 ConvNeXt (lệch 1 khối — caveat đã ghi).

Panel GỌN 4 model: dino_swin_l (trục chính) + r101/fcos_r50/dino_r50 (RES3, canh không vỡ họ ResNet).
Quy tắc phán quyết khóa trong progress_log entry "E80-ENSB MỞ RỘNG".
  python scripts/e80_ens_budget.py   -> results/mechanism/e80_ens_budget.json"""
import json
import os
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
from scripts.run_baselines import _C2OCT
from scripts.e64_cnx_drop import load

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e80_ens_budget.json")
# cond -> (B, run dir). Hậu tố _v100 để KHÔNG trộn với artifact Ampere (tên run không tag).
CELLS = {"ENS-WP-B36": (36, "dev300_B36_eps5_limit100_ens_v100"),
         "ENS-WP-B50": (50, "dev300_B50_eps5_limit100_ens_v100")}
PANEL = ["dino_swin_l", "r101", "fcos_r50", "dino_r50"]
RES3 = ["r101", "fcos_r50", "dino_r50"]
BLOCK, BOOT_SEED, N_BOOT = 2, 2026, 100
COMMON = dict(cls_mode="roi", stage_w=(0.0, 1.0, 1.0, 1.0))
# Y HỆT E71 KW_WP — không đổi một tham số nào ngoài STEPS.
KW_WP = {"r50": dict(_C2OCT, cls_weight=6.0, weight_rho=0.002),
         "convnext_t": dict(_C2OCT, cls_weight=0.2, weight_rho=0.002, weight_linear=True)}
# Mốc A4500 (E71 dev30) — CHỈ để đo độ lệch máy, KHÔNG dùng phán quyết ngân sách.
A4500_B36 = {"dino_swin_l": 60.3, "r101": 92.9, "fcos_r50": 94.4, "dino_r50": 93.8}


def png_dir(cond):
    return os.path.join(ART, CELLS[cond][1], "adv", cond)


def attack(ids, cond):
    steps, _ = CELLS[cond]
    out_dir = png_dir(cond)
    if all(os.path.exists(os.path.join(out_dir, f"{i}.png")) for i in ids):
        print(f"[e80] {cond}: PNG đã đủ, bỏ qua sinh", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    models = {"r50": load_model("r50", "cuda:0"), "convnext_t": load_model("convnext_t", "cuda:0")}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in models}
    order = ["r50", "convnext_t"]
    nblocks = steps // BLOCK
    t0 = time.time()
    for n, i in enumerate(ids):
        if os.path.exists(os.path.join(out_dir, f"{i}.png")):
            continue
        orig = ds["r50"][i]["orig_inputs"].cuda().float()
        delta, mom = None, None
        for b in range(nblocks):
            key = order[b % 2]
            torch.manual_seed(1000 * (n + 1) + b)
            delta, mom = geo_osfd_attack(models[key], orig, ds[key][i]["data_sample"], steps=BLOCK,
                                         init_noise=delta, mom_in=mom, return_state=True,
                                         **KW_WP[key], **COMMON)
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= 5.0 + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                     os.path.join(out_dir, f"{i}.png"))
        print(f"[e80 {cond}] {n + 1}/{len(ids)}  {(time.time() - t0) / (n + 1):.1f}s/ảnh", flush=True)
    del models
    torch.cuda.empty_cache()


def main():
    gpu = torch.cuda.get_device_name(0)
    print(f"[e80] GPU = {gpu} | cả hai ô B36/B50 sinh TRÊN CÙNG máy này", flush=True)
    coco = COCO(ANN)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30, len(ids)
    for cond in CELLS:
        attack(ids, cond)

    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    # W[0] = trọng số 1 (điểm), W[1:] = bootstrap DÙNG CHUNG mọi ô/model -> hiệu số là paired.
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    from mmdet.apis import inference_detector
    res = {"n_images": n, "n_boot": N_BOOT, "gpu": gpu, "block": BLOCK,
           "cells": {c: CELLS[c][0] for c in CELLS}, "drop": {}, "diff_b50_minus_b36": {},
           "machine_shift_vs_a4500": {}}
    if os.path.exists(OUT):
        old = json.load(open(OUT))
        if old.get("cells") == res["cells"] and old.get("gpu") == gpu:
            res = old  # resume per-model
    for key in PANEL:
        if key in res["drop"] and all(c in res["drop"][key] for c in CELLS):
            continue
        model = load(key)

        def ap_of(paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            return np.array([CocoImageEval(coco, dets, ids).ap(w) for w in W])[:, 0] * 100
        clean = ap_of([os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        s = {}
        for cond in CELLS:
            s[cond] = 100 * (clean - ap_of([os.path.join(png_dir(cond), f"{i}.png") for i in ids])) / clean
        res["drop"][key] = {c: summarize(s[c][0], s[c][1:]) for c in CELLS}
        d = s["ENS-WP-B50"] - s["ENS-WP-B36"]          # paired: cùng W, cùng clean
        res["diff_b50_minus_b36"][key] = summarize(d[0], d[1:])
        res["machine_shift_vs_a4500"][key] = round(float(s["ENS-WP-B36"][0]) - A4500_B36[key], 2)
        del model
        torch.cuda.empty_cache()
        print(f"{key:12} B36 {s['ENS-WP-B36'][0]:5.1f} | B50 {s['ENS-WP-B50'][0]:5.1f} | "
              f"Δ {d[0]:+5.1f} CI {[round(x, 1) for x in res['diff_b50_minus_b36'][key]['ci95']]} | "
              f"lệch máy B36 vs A4500 {res['machine_shift_vs_a4500'][key]:+.1f}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)

    # ---- Phán quyết theo quy tắc khóa (progress_log "E80-ENSB MỞ RỘNG") ----
    sl = res["diff_b50_minus_b36"]["dino_swin_l"]
    r3 = {c: float(np.mean([res["drop"][m][c]["point"] for m in RES3])) for c in CELLS}
    print(f"\n=== E80-ENSB @ dev30, {gpu}, trong-cùng-máy, boot {N_BOOT} ===")
    print(f"{'model':12} {'B36':>6} {'B50':>6} {'Δ':>6}  CI(Δ paired)")
    for m in PANEL:
        dd = res["diff_b50_minus_b36"][m]
        print(f"{m:12} {res['drop'][m]['ENS-WP-B36']['point']:6.1f} "
              f"{res['drop'][m]['ENS-WP-B50']['point']:6.1f} {dd['point']:+6.1f}  "
              f"{[round(x, 1) for x in dd['ci95']]}")
    print(f"RES3         {r3['ENS-WP-B36']:6.1f} {r3['ENS-WP-B50']:6.1f} "
          f"{r3['ENS-WP-B50'] - r3['ENS-WP-B36']:+6.1f}")
    print(f"\nlệch máy (B36@{gpu.split()[1]} − B36@A4500): "
          + ", ".join(f"{m} {res['machine_shift_vs_a4500'][m]:+.1f}" for m in PANEL))
    ok_res3 = r3["ENS-WP-B50"] >= r3["ENS-WP-B36"] - 2
    if sl["point"] >= 4 and sl["ci95"][0] > 0 and ok_res3:
        v = "NGÂN SÁCH LÀ ĐÒN BẨY => xác nhận dev100 trước khi vào bảng"
    elif sl["point"] <= -4 and sl["ci95"][1] < 0:
        v = "HẠI => tăng B làm xấu ensemble, đóng dứt"
    else:
        v = ("TRUNG TÍNH => không đủ để theo; đóng lever-B ở mức dev30, dồn sức sang strict-1+WP. "
             "(n=30 -> null YẾU, không đọc là 'B chắc chắn không giúp')")
    print(f"\n=> PHÁN QUYẾT: {v}")
    if not ok_res3:
        print(f"   (RES3 vỡ: {r3['ENS-WP-B50']:.1f} < {r3['ENS-WP-B36'] - 2:.1f})")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
