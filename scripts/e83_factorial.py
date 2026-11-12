"""E83 — bước 3a docs/plan_paper_lock.md: factorial giai đoạn 1 (PRIMARY endpoint của paper) @ dev100, cùng máy.

Ô (Mask R-CNN, cùng B=36, ε=5, OCT 1 view, amp, KHÔNG WP trừ ô nối RD):
  {R50, CNX} × {FEAT, CLS, FC}
    FEAT = L_OSFD C3–C5 (feat_stages=(1,2,3) — đặt TƯỜNG MINH: ô RD-CNX-FEAT của E66 đi nhánh _view_loss nên bỏ
           qua stage_w và thực tế dùng C2–C5; ở đây sửa cho đúng định nghĩa khóa "feature C3–C5")
    CLS  = chỉ loss phân loại RoI (cls_mode="roi", feature=False); MI chuẩn hóa mean|g| rồi sign -> không có λ
    FC   = L_OSFD C3–C5 + λ·L_cls RoI, λ theo quy tắc rd_calib (R50 6.0, CNX 0.2 — giá trị đã khóa E39/E63)
  Secondary: FC với λ×0.5 và λ×2 (4 ô).
  Ô nối E65 (tách augmentation/stage vs objective): OSFD-R50, OSFD-CNX (recipe gốc RRB + C2–C5, cùng B=36),
    RD-R50, RD-CNX (= FC + WP ρ0.002; CNX weight_linear).
Seed: torch + python random + numpy = 1000·(n+1), n = chỉ số ảnh trong dev100 (plan 3.6).
Eval: panel 9 model, paired bootstrap 1000 (BOOT_SEED 2026), mọi hiệu số / interaction tính TRONG tiến trình này.
Quy tắc phán quyết: progress_log entry "E83" (khóa + commit TRƯỚC khi chạy).

  python scripts/e83_factorial.py --gen-only --cells R50-FEAT,R50-CLS   # 1 worker sinh (chạy song song được)
  python scripts/e83_factorial.py                                         # sinh phần còn thiếu + eval + phán quyết
  python scripts/e83_factorial.py --list dev5 --boot 20 --tag smoke       # smoke test (KHÔNG đọc số)
"""
import argparse
import gzip
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
from attack.methods.baselines import osfd_attack
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.run_baselines import _C2OCT

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
STEPS, EPS, RHO, BOOT_SEED = 36, 5.0, 0.002, 2026
R, C = "r50", "convnext_t"
LAM = {R: 6.0, C: 0.2}                      # rd_calib (E39 R50, E63 CNX) — không tune
SHORT = {R: "R50", C: "CNX"}
FEAT_KW = dict(_C2OCT, cls_weight=0.0, skip_mask=False, feat_stages=(1, 2, 3))   # L_OSFD C3–C5


def _cls_kw(sur):
    return dict(_C2OCT, cls_weight=LAM[sur], cls_mode="roi", feature=False)


def _fc_kw(sur, mult=1.0):
    return dict(_C2OCT, cls_weight=LAM[sur] * mult, cls_mode="roi", stage_w=(0.0, 1.0, 1.0, 1.0))


def _rd_kw(sur):
    kw = dict(_fc_kw(sur), weight_rho=RHO)
    if sur == C:
        kw["weight_linear"] = True
    return kw


# cell -> (surrogate, hàm attack, kwargs, vai trò)
CELLS = {}
for _s in (R, C):
    p = SHORT[_s]
    CELLS[f"{p}-FEAT"] = (_s, geo_osfd_attack, FEAT_KW, "primary")
    CELLS[f"{p}-CLS"] = (_s, geo_osfd_attack, _cls_kw(_s), "primary")
    CELLS[f"{p}-FC"] = (_s, geo_osfd_attack, _fc_kw(_s), "secondary")
    CELLS[f"{p}-FC-L05"] = (_s, geo_osfd_attack, _fc_kw(_s, 0.5), "sensitivity")
    CELLS[f"{p}-FC-L2"] = (_s, geo_osfd_attack, _fc_kw(_s, 2.0), "sensitivity")
    CELLS[f"OSFD-{p}"] = (_s, osfd_attack, {}, "bridge")
    CELLS[f"RD-{p}"] = (_s, geo_osfd_attack, _rd_kw(_s), "bridge")

PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
PRIMARY_TARGETS = ["swin_t", "dino_swin_l"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]


def contrasts():
    """tên -> {cell: hệ số}. I là primary; còn lại secondary / mô tả."""
    def inter(a, b):  # (CNX_a − CNX_b) − (R50_a − R50_b)
        return {f"CNX-{a}": 1, f"CNX-{b}": -1, f"R50-{a}": -1, f"R50-{b}": 1}
    d = {"I (CLS−FEAT)": inter("CLS", "FEAT"),
         "I_FC (FC−FEAT)": inter("FC", "FEAT"),
         "I_E65 (RD−OSFD)": {"RD-CNX": 1, "OSFD-CNX": -1, "RD-R50": -1, "OSFD-R50": 1}}
    for p in ("R50", "CNX"):
        d[f"{p}: CLS−FEAT"] = {f"{p}-CLS": 1, f"{p}-FEAT": -1}
        d[f"{p}: FC−FEAT"] = {f"{p}-FC": 1, f"{p}-FEAT": -1}
        d[f"{p}: FEAT−OSFD (aug+stage)"] = {f"{p}-FEAT": 1, f"OSFD-{p}": -1}
        d[f"{p}: RD−FC (WP)"] = {f"RD-{p}": 1, f"{p}-FC": -1}
    return d


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def adv_dir(run_dir, cell):
    return os.path.join(ART, run_dir, "adv", cell)


def generate(ids, cell, run_dir):
    sur, fn, kw, _ = CELLS[cell]
    out = adv_dir(run_dir, cell)
    if all(os.path.exists(os.path.join(out, f"{i}.png")) for i in ids):
        print(f"[e83] {cell}: PNG đã đủ, bỏ qua sinh", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    model = load_model(sur, "cuda:0")
    ds = {s["img_id"]: s for s in target_dataset(sur, ids)}
    t0, done = time.time(), 0
    for n, i in enumerate(ids):
        p = os.path.join(out, f"{i}.png")
        if os.path.exists(p):
            continue
        orig = ds[i]["orig_inputs"].cuda().float()
        seed_all(1000 * (n + 1))
        delta = fn(model, orig, ds[i]["data_sample"], steps=STEPS, epsilon=EPS, **kw)
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= EPS + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), p + ".tmp.png")
        os.replace(p + ".tmp.png", p)              # ghi nguyên tử: PNG dở không bị coi là xong khi resume
        done += 1
        print(f"[e83 {cell}] {n + 1}/{len(ids)}  {(time.time() - t0) / done:.1f}s/ảnh", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8")       # tmux/SSH locale ASCII làm vỡ print '−', 'Δ'
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="dev100", choices=["dev5", "dev10", "dev30", "dev100"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cells", default=None, help="danh sách ô, phẩy ngăn (mặc định: tất cả)")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--tag", default="", help="hậu tố run dir / json (vd 'smoke')")
    args = ap.parse_args()
    cells = args.cells.split(",") if args.cells else list(CELLS)
    assert all(c in CELLS for c in cells), cells
    gpu = torch.cuda.get_device_name(0)
    mtag = gpu.split()[-1].replace("-", "").lower()
    sfx = f"_{args.tag}" if args.tag else ""
    run_dir = f"e83_factorial_B{STEPS}_eps5_{mtag}{sfx}"
    out_json = os.path.join(REPO_ROOT, f"results/mechanism/e83_factorial_{args.list}{sfx}.json")
    print(f"[e83] GPU = {gpu} | run dir {run_dir} | ô {cells}", flush=True)

    keep = set(load_image_ids(os.path.join(REPO_ROOT, f"data/image_lists/{args.list}.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")) if i in keep]
    assert len(ids) == int(args.list[3:]), len(ids)
    for c in cells:
        generate(ids, c, run_dir)
    if args.gen_only:
        return
    assert set(cells) == set(CELLS), "eval cần đủ mọi ô"

    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, args.boot, BOOT_SEED)])   # hàng 0 = mẫu gốc
    npz = out_json.replace(".json", "_samples.npz")
    S = dict(np.load(npz)) if os.path.exists(npz) else {}       # "model|cell" -> [1+boot] drop %
    meta = {"list": args.list, "n_images": n, "n_boot": args.boot, "boot_seed": BOOT_SEED, "gpu": gpu,
            "steps": STEPS, "eps": EPS, "lambda": LAM, "run_dir": run_dir,
            "cells": {c: {"surrogate": v[0], "fn": v[1].__name__, "role": v[3],
                          "kw": {k: (list(x) if isinstance(x, tuple) else x) for k, x in v[2].items()}}
                      for c, v in CELLS.items()}}
    from mmdet.apis import inference_detector
    dets_dir = os.path.join(ART, run_dir, "dets")
    for key in PANEL:
        if all(f"{key}|{c}" in S for c in CELLS):
            continue
        model = load(key)
        os.makedirs(os.path.join(dets_dir, key), exist_ok=True)

        def ap_of(name, paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            with gzip.open(os.path.join(dets_dir, key, f"{name}.json.gz"), "wt") as f:   # cho 3c (outcome theo ảnh)
                json.dump(dets, f)
            ev = CocoImageEval(coco, dets, ids)
            return np.array([ev.ap(w)[0] for w in W]) * 100
        clean = ap_of("clean", [os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        for c in CELLS:
            adv = ap_of(c, [os.path.join(adv_dir(run_dir, c), f"{i}.png") for i in ids])
            S[f"{key}|{c}"] = (100 * (clean - adv) / clean).astype(np.float64)
        del model
        torch.cuda.empty_cache()
        np.savez(npz, **S)                                       # incremental -> resume theo model
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in CELLS), flush=True)

    # ---- bảng + contrast (paired: cùng W, cùng clean) ----
    groups = {m: [m] for m in PANEL}
    groups["PRIMARY(Swin-T,Swin-L)"] = PRIMARY_TARGETS
    groups["T7"] = T7
    G = {g: {c: np.mean([S[f"{m}|{c}"] for m in ms], axis=0) for c in CELLS} for g, ms in groups.items()}
    res = dict(meta, drop={}, contrast={})
    for g in groups:
        res["drop"][g] = {c: summarize(G[g][c][0], G[g][c][1:]) for c in CELLS}
        res["contrast"][g] = {}
        for name, coef in contrasts().items():
            v = sum(w * G[g][c] for c, w in coef.items())
            res["contrast"][g][name] = summarize(v[0], v[1:])
    fmt = lambda s: f"{s['point']:+6.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]"
    print(f"\n=== E83 @ {args.list}, {gpu}, B={STEPS}, boot {args.boot} — relative AP drop % ===")
    print(f"{'cell':12}" + "".join(f"{g[:10]:>11}" for g in groups))
    for c in CELLS:
        print(f"{c:12}" + "".join(f"{res['drop'][g][c]['point']:11.1f}" for g in groups))
    print("\n--- contrast (điểm [CI95 paired]) ---")
    for name in contrasts():
        print(f"{name:26}" + " | ".join(f"{g[:8]} {fmt(res['contrast'][g][name])}"
                                        for g in ["PRIMARY(Swin-T,Swin-L)", "dino_swin_l", "swin_t", "T7"]))
    print("\nI (CLS−FEAT) mọi target:")
    for m in PANEL:
        tagm = " (G0: surrogate của 1 nhánh)" if m in (R, C) else ""
        print(f"  {m:12} {fmt(res['contrast'][m]['I (CLS−FEAT)'])}{tagm}")

    # ---- phán quyết theo quy tắc khóa (progress_log "E83") ----
    I = res["contrast"]["PRIMARY(Swin-T,Swin-L)"]["I (CLS−FEAT)"]
    holds = I["ci95"][0] > 0
    res["verdict"] = {"I_primary": I, "claim_2_1_holds": bool(holds)}
    print(f"\n=> I trên TB(Swin-T, DINO-Swin-L) = {fmt(I)}")
    print("   " + ("cận dưới CI > 0 -> claim 2.1 GIỮ; đi tiếp 3b" if holds else
                   "CI chứa 0 (hoặc âm) -> ĐÁNH GIÁ LẠI toàn bộ central claim (plan 3a), không chạy tiếp 3b mặc định"))
    json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)
    print(f"[e83] -> {out_json}")


if __name__ == "__main__":
    main()
