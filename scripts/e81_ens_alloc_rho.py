"""E81 (progress_log 2026-11-09): hai trục KHÔNG CẦN SỬA CORE cho Swin-L — phân bổ ngân sách giữa 2 surrogate
+ liều ρ của WP. dev30, B36, tất cả sinh/eval TRÊN CÙNG máy, CI hiệu số PAIRED.

Nền: E80 đóng lever ngân sách cho Swin-L (ΔSwin-L +2.1, CI chứa 0) nhưng ENS-WP đã gần trần họ ResNet
(91-95) còn Swin-L mới 60.3 -> phần còn lại phải lấy từ Swin-L.

Trục 1 PHÂN BỔ: ENS-WP chia 9 khối R50 / 9 khối ConvNeXt (1:1), chưa từng thử khác. E68 — ConvNeXt là bên
  mang Swin-L -> dồn khối sang ConvNeXt mua Swin-L ở CÙNG B=36.
  ⚠️ PRIOR YẾU khai báo trước: trục bị CHẶN TRÊN bởi ConvNeXt-thuần (RD-CNX dev30 A4500 Swin-L 62.8);
  ENS 1:1 = 60.3 < 62.8 nên trên Swin-L ensemble đang NỘI SUY -> gain kỳ vọng <= +2.5 < ngưỡng +4.
  Vì vậy chỉ chạy 1 ô ở ĐIỂM CỰC (3 R50 / 15 CNX): cực không đạt => cả trục đóng.
Trục 2 LIỀU ρ: ρ=0.002 bê từ single-surrogate R50, chưa dò. E66 — bật WP = +9.9 Swin-L trên RD-CNX, gần
  miễn phí ở CNN -> trục có năng suất. Quét 0.005 / 0.010, kiểm ĐƠN ĐIỆU theo liều (như E57).

Mọi ô: B=36, BLOCK=2 (18 khối), momentum CHUNG (mom_in/return_state), λ R50=6 / CNX=0.2, OCT 1V,
L_OSFD C3-C5 + cls RoI, ε=5, seed 1000·(n+1)+b. CHỈ đổi đúng thứ ghi trong CELLS.
Quy tắc phán quyết: progress_log entry "E81".
  python scripts/e81_ens_alloc_rho.py   -> results/mechanism/e81_ens_alloc_rho.json"""
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
RUN = "dev300_B36_eps5_limit100_ens_v100"   # dùng lại run dir của E80 (cùng B36, cùng máy)
OUT = os.path.join(REPO_ROOT, "results/mechanism/e81_ens_alloc_rho.json")
BASE = "ENS-WP-B36"                          # mốc: PNG đã sinh ở E80 trên CHÍNH máy này
R, C = "r50", "convnext_t"
# cond -> (mẫu luân phiên khối, ρ). 18 khối, khối thứ b dùng pattern[b % len(pattern)].
CELLS = {
    BASE:         ([R, C],                      0.002),   # 9 R50 / 9 CNX  (= ENS-WP gốc)
    "ENS-A315":   ([R, C, C, C, C, C],          0.002),   # 3 R50 / 15 CNX (điểm cực trục phân bổ)
    "ENS-RHO5":   ([R, C],                      0.005),
    "ENS-RHO10":  ([R, C],                      0.010),
}
PANEL = ["dino_swin_l", "r101", "fcos_r50", "dino_r50"]
RES3 = ["r101", "fcos_r50", "dino_r50"]
STEPS, BLOCK, BOOT_SEED, N_BOOT = 36, 2, 2026, 100
COMMON = dict(cls_mode="roi", stage_w=(0.0, 1.0, 1.0, 1.0))


def kw_for(key, rho):
    """Y HỆT E71 KW_WP, chỉ thay ρ."""
    if key == R:
        return dict(_C2OCT, cls_weight=6.0, weight_rho=rho)
    return dict(_C2OCT, cls_weight=0.2, weight_rho=rho, weight_linear=True)


def png_dir(cond):
    return os.path.join(ART, RUN, "adv", cond)


def attack(ids, cond):
    pattern, rho = CELLS[cond]
    out_dir = png_dir(cond)
    if all(os.path.exists(os.path.join(out_dir, f"{i}.png")) for i in ids):
        print(f"[e81] {cond}: PNG đã đủ, bỏ qua sinh", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    models = {R: load_model(R, "cuda:0"), C: load_model(C, "cuda:0")}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in models}
    kw = {k: kw_for(k, rho) for k in models}
    nblocks = STEPS // BLOCK
    nR = sum(1 for b in range(nblocks) if pattern[b % len(pattern)] == R)
    print(f"[e81] {cond}: {nR} khối {R} / {nblocks - nR} khối {C}, ρ={rho}", flush=True)
    t0 = time.time()
    for n, i in enumerate(ids):
        if os.path.exists(os.path.join(out_dir, f"{i}.png")):
            continue
        orig = ds[R][i]["orig_inputs"].cuda().float()
        delta, mom = None, None
        for b in range(nblocks):
            key = pattern[b % len(pattern)]
            torch.manual_seed(1000 * (n + 1) + b)
            delta, mom = geo_osfd_attack(models[key], orig, ds[key][i]["data_sample"], steps=BLOCK,
                                         init_noise=delta, mom_in=mom, return_state=True,
                                         **kw[key], **COMMON)
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= 5.0 + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                     os.path.join(out_dir, f"{i}.png"))
        print(f"[e81 {cond}] {n + 1}/{len(ids)}  {(time.time() - t0) / (n + 1):.1f}s/ảnh", flush=True)
    del models
    torch.cuda.empty_cache()


def main():
    gpu = torch.cuda.get_device_name(0)
    print(f"[e81] GPU = {gpu} | mọi ô sinh+eval TRÊN CÙNG máy", flush=True)
    coco = COCO(ANN)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30, len(ids)
    for cond in CELLS:
        attack(ids, cond)

    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    from mmdet.apis import inference_detector
    res = {"n_images": n, "n_boot": N_BOOT, "gpu": gpu, "steps": STEPS, "block": BLOCK,
           "cells": {c: {"pattern": CELLS[c][0], "rho": CELLS[c][1]} for c in CELLS},
           "drop": {}, "diff_vs_base": {}}
    if os.path.exists(OUT):
        old = json.load(open(OUT))
        if old.get("cells") == res["cells"] and old.get("gpu") == gpu:
            res = old
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
        s = {c: 100 * (clean - ap_of([os.path.join(png_dir(c), f"{i}.png") for i in ids])) / clean
             for c in CELLS}
        res["drop"][key] = {c: summarize(s[c][0], s[c][1:]) for c in CELLS}
        res["diff_vs_base"][key] = {}
        for c in CELLS:
            if c == BASE:
                continue
            d = s[c] - s[BASE]                      # paired: cùng W, cùng clean
            res["diff_vs_base"][key][c] = summarize(d[0], d[1:])
        del model
        torch.cuda.empty_cache()
        print(f"{key:12} " + "  ".join(f"{c} {s[c][0]:.1f}" for c in CELLS), flush=True)
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)

    # ---- Phán quyết theo quy tắc khóa (progress_log "E81") ----
    r3 = {c: float(np.mean([res["drop"][m][c]["point"] for m in RES3])) for c in CELLS}
    print(f"\n=== E81 @ dev30, {gpu}, B={STEPS}, boot {N_BOOT} ===")
    hdr = f"{'ô':12} {'Swin-L':>7} {'ΔSwin-L':>8} {'CI(Δ paired)':>16} {'RES3':>6} {'ΔRES3':>6}"
    print(hdr)
    for c in CELLS:
        sl = res["drop"]["dino_swin_l"][c]["point"]
        if c == BASE:
            print(f"{c:12} {sl:7.1f} {'(mốc)':>8} {'':>16} {r3[c]:6.1f} {'':>6}")
        else:
            dd = res["diff_vs_base"]["dino_swin_l"][c]
            print(f"{c:12} {sl:7.1f} {dd['point']:+8.1f} "
                  f"{str([round(x, 1) for x in dd['ci95']]):>16} {r3[c]:6.1f} "
                  f"{r3[c] - r3[BASE]:+6.1f}")
    print("\nchi tiết từng target (điểm):")
    print(f"{'model':12} " + " ".join(f"{c:>12}" for c in CELLS))
    for m in PANEL:
        print(f"{m:12} " + " ".join(f"{res['drop'][m][c]['point']:12.1f}" for c in CELLS))

    win = []
    for c in CELLS:
        if c == BASE:
            continue
        dd = res["diff_vs_base"]["dino_swin_l"][c]
        if dd["point"] >= 4 and dd["ci95"][0] > 0:
            win.append((c, dd["point"], r3[c] - r3[BASE]))
    print()
    if any(r3[c] < r3[BASE] - 10 for c in ("ENS-RHO5", "ENS-RHO10")):
        bad = [c for c in ("ENS-RHO5", "ENS-RHO10") if r3[c] < r3[BASE] - 10]
        print(f"=> ρ QUÁ LIỀU ở {bad}: RES3 sụt > 10 => ρ phá surrogate, đọc ô ρ nhỏ hơn làm trần trục")
    if not win:
        print("=> KHÔNG / TRUNG TÍNH: không ô nào ΔSwin-L >= +4 với CI loại 0")
        print("   => đóng CẢ HAI trục không-sửa-core; lever Swin-L còn lại = strict-1 + WP (sửa core)")
        print("   (n=30, boot 100 -> null YẾU)")
    else:
        for c, d, dr3 in win:
            tag = "CÓ ĐÒN BẨY" if dr3 >= -3 else "ĐÁNH ĐỔI (RES3 tụt > 3)"
            print(f"=> {tag}: {c} ΔSwin-L {d:+.1f}, ΔRES3 {dr3:+.1f} => xác nhận dev100 trước khi claim")
    mono = [res["drop"]["dino_swin_l"][c]["point"] for c in (BASE, "ENS-RHO5", "ENS-RHO10")]
    print(f"\nđơn điệu theo ρ (0.002 -> 0.005 -> 0.010) trên Swin-L: "
          f"{mono[0]:.1f} -> {mono[1]:.1f} -> {mono[2]:.1f}"
          + ("  [ĐƠN ĐIỆU TĂNG]" if mono[0] < mono[1] < mono[2] else "  [không đơn điệu]"))
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
