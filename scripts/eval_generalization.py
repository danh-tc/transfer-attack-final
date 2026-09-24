#!/usr/bin/env python3
"""Generalization Panel (idea.md §6): đánh giá ảnh adv ĐÃ SINH của 1 run baseline
(surrogate R50, Controlled Panel) trên các detector khác kiến trúc — không attack lại.

Mỗi model đọc FILE ảnh (ảnh gốc COCO / PNG adv) qua ĐÚNG test pipeline của config riêng
(mmdet.apis.inference_detector) — các model này khác pipeline với Controlled Panel (YOLOX
Resize 640 + Pad 114, FCOS caffe normalize, ...), nên không dùng pipeline_resize.
Detection lưu artifacts/runs/<run>/gen_dets/<model>.json (resume: có file thì bỏ qua).

Metrics (results/runs/<run>/generalization_metrics.json): AP/AP50, relative AP drop, paired
bootstrap 95% CI (cùng BOOT_SEED + thứ tự ảnh như run_baselines.py -> paired với cả
Controlled Panel, tính lại từ dets.json của run), kèm:
- method_diff: OSFD − M-DI² (và các cặp khác) trên từng model.
- headroom_vs_same_family: drop(method, R101) − drop(method, model) — chỗ trống so với
  mức cùng họ của Controlled Panel.

Chạy: python scripts/eval_generalization.py [--run n300_B50_eps5] [--models ...]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets

CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs")
CKPT = os.path.join(REPO_ROOT, "checkpoints")
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(REPO_ROOT, "data/coco/val2017")
BOOT_SEED = 2026  # = run_baselines.py
METHODS = ["MI-FGSM", "M-DI2-FGSM", "OSFD"]
REF_SAME_FAMILY = "r101"

# docs/protocol_lock.md — Generalization Panel. (config, glob checkpoint)
GEN_PANEL = {
    "fcos_r50": ("fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py", "fcos_center-normbbox*"),
    "detr_r50": ("detr/detr_r50_8xb2-150e_coco.py", "detr_r50_8xb2-150e*"),
    "yolox_s": ("yolox/yolox_s_8xb8-300e_coco.py", "yolox_s_8x8*"),
    "yolox_l": ("yolox/yolox_l_8xb8-300e_coco.py", "yolox_l_8x8*"),
    "dino_swin_l": ("dino/dino-5scale_swin-l_8xb2-36e_coco.py", "dino-5scale_swin-l*"),
    # Chẩn đoán post hoc (progress_log 2026-09-23), KHÔNG thuộc panel định trước: tách hiệu ứng
    # kiểu detector DINO khỏi backbone Swin-L. Chỉ bản 12e có checkpoint chính thức.
    "dino_r50": ("dino/dino-4scale_r50_8xb2-12e_coco.py", "dino-4scale_r50_8xb2-12e*"),
    # Chẩn đoán post hoc: config GIỐNG HỆT Swin-T của Controlled Panel, chỉ khác depths
    # [2,2,18,2] (capacity) — cùng pretrain IN-1k 224, cùng lịch 3x.
    "mrcnn_swin_s": ("swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py", "mask_rcnn_swin-s-p4-w7*"),
}
DIAG_ONLY = {"dino_r50", "mrcnn_swin_s"}


def predict_model(key, img_ids, coco, adv_dir, out_path, device, methods):
    """Resume theo điều kiện: file đã có thì chỉ predict điều kiện (clean/method) còn thiếu."""
    dets = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            dets = json.load(f)
    todo = [c for c in ["clean"] + methods if c not in dets]
    if not todo:
        return
    from mmdet.apis import inference_detector, init_detector
    cfg, pat = GEN_PANEL[key]
    ckpts = glob.glob(os.path.join(CKPT, pat + ".pth"))
    assert len(ckpts) == 1, f"{key}: checkpoint {ckpts}"
    model = init_detector(os.path.join(CFG, cfg), ckpts[0], device=device)
    cat_ids = sorted(coco.getCatIds())  # thứ tự 80 lớp COCO = thứ tự label mmdet
    dets.update({c: [] for c in todo})
    t0 = time.time()
    for i, img_id in enumerate(img_ids):
        paths = {"clean": os.path.join(IMG_DIR, coco.loadImgs(img_id)[0]["file_name"])}
        paths.update({m: os.path.join(adv_dir, m, f"{img_id}.png") for m in methods})
        for c in todo:
            p = paths[c]
            dets[c] += to_coco_dets(inference_detector(model, p), img_id, cat_ids)
        if (i + 1) % 50 == 0:
            print(f"[gen {key}] {i + 1}/{len(img_ids)} ({time.time() - t0:.0f}s)", flush=True)
    with open(out_path, "w") as f:
        json.dump(dets, f)
    del model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="n300_B50_eps5")
    p.add_argument("--models", nargs="+", default=[k for k in GEN_PANEL if k not in DIAG_ONLY],
                   choices=list(GEN_PANEL))  # mặc định = panel định trước; dino_r50 chỉ chạy khi chỉ định
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ctrl-refs", nargs="*", default=[],
                   help="thêm model Controlled Panel (vd swin_t) từ dets.json vào so sánh model_diff")
    p.add_argument("--out", default="generalization_metrics.json", help="tên file trong results/runs/<run>/")
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--dets", default="dets.json", help="dets Controlled Panel của run (run_baselines --out)")
    args = p.parse_args()
    methods = args.methods
    split = args.run.split("_")[0]
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{split}.csv"))
    art = os.path.join(REPO_ROOT, "artifacts/runs", args.run)
    run_dir = os.path.join(REPO_ROOT, "results/runs", args.run)
    gen_dir = os.path.join(art, "gen_dets")
    os.makedirs(gen_dir, exist_ok=True)
    coco = COCO(ANN_FILE)

    for key in args.models:
        out = os.path.join(gen_dir, f"{key}.json")
        predict_model(key, img_ids, coco, os.path.join(art, "adv"), out, args.device, methods)

    # Gom dets: Generalization Panel + Controlled Panel (để so paired với R101).
    all_dets = {}
    for key in args.models:
        with open(os.path.join(gen_dir, f"{key}.json")) as f:
            all_dets[key] = json.load(f)
    with open(os.path.join(art, args.dets)) as f:
        ctrl = json.load(f)
    for ref in [REF_SAME_FAMILY] + args.ctrl_refs:
        all_dets[ref] = {c: ctrl[c][ref] for c in ["clean"] + methods}

    n = len(img_ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, args.n_boot, BOOT_SEED)])
    ap = {}
    for key, d in all_dets.items():
        ap[key] = {}
        for c in ["clean"] + methods:
            ev = CocoImageEval(coco, d[c], img_ids)
            ap[key][c] = np.array([ev.ap(w) for w in W]) * 100  # [1+n_boot, 2]
        print(f"[gen] AP xong: {key}", flush=True)

    S = lambda a: summarize(a[0], a[1:])
    drop = {k: {m: 100 * (ap[k]["clean"][:, 0] - ap[k][m][:, 0]) / ap[k]["clean"][:, 0] for m in methods}
            for k in ap}
    gen_keys = list(args.models) + list(args.ctrl_refs)
    out = {"run": args.run, "n_images": n, "n_boot": args.n_boot, "boot_seed": BOOT_SEED,
           "eval": "inference_detector trên file ảnh (ảnh gốc COCO / PNG adv), test pipeline riêng từng model",
           "ap": {k: {c: S(ap[k][c][:, 0]) for c in ap[k]} for k in ap},
           "ap50": {k: {c: S(ap[k][c][:, 1]) for c in ap[k]} for k in ap},
           "relative_ap_drop": {m: {k: S(drop[k][m]) for k in ap} for m in methods},
           "method_diff": {f"{a} - {b}": {k: S(drop[k][a] - drop[k][b]) for k in gen_keys}
                           for a in methods for b in methods if a != b and methods.index(a) > methods.index(b)},
           "headroom_vs_same_family": {m: {k: S(drop[REF_SAME_FAMILY][m] - drop[k][m]) for k in gen_keys}
                                       for m in methods},
           "model_diff": {m: {f"{a} - {b}": S(drop[a][m] - drop[b][m]) for a in gen_keys for b in gen_keys if a < b}
                          for m in methods},
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(run_dir, args.out), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    fmt = lambda s: f"{s['point']:5.1f} [{s['ci95'][0]:5.1f},{s['ci95'][1]:5.1f}]"
    cols = [REF_SAME_FAMILY] + gen_keys
    print(f"\n=== clean AP (n={n}) ===\n" + "  ".join(f"{k}: {out['ap'][k]['clean']['point']:.1f}" for k in cols))
    print("\n=== relative AP drop % [95% CI] ===")
    print(f"{'':<12}" + "".join(f"{k:>21}" for k in cols))
    for m in methods:
        print(f"{m:<12}" + "".join(f"{fmt(out['relative_ap_drop'][m][k]):>21}" for k in cols))
    print("\n=== headroom = drop(R101) − drop(model) ===")
    for m in methods:
        print(f"{m:<12}" + "".join(f"{fmt(out['headroom_vs_same_family'][m][k]):>21}" for k in gen_keys))
    print(f"\n=== method_diff ===\n{'':<22}" + "".join(f"{k:>21}" for k in gen_keys))
    for name, row in out["method_diff"].items():
        print(f"{name:<22}" + "".join(f"{fmt(row[k]):>21}" for k in gen_keys))
    print("\n=== model_diff (OSFD) ===")
    for k, v in out["model_diff"]["OSFD"].items():
        print(f"  {k:<28} {fmt(v)}")
    print("saved:", os.path.join(run_dir, args.out))


if __name__ == "__main__":
    main()
