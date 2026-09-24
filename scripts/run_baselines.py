#!/usr/bin/env python3
"""Baseline table chính thức (idea.md §8-10): surrogate R50 -> 4 model Controlled Panel.

2 giai đoạn, đều resume được (chạy lại lệnh y hệt sau khi bị ngắt):
1. attack: mỗi (method, ảnh) sinh ảnh adversarial THẬT (cỡ gốc, uint8) -> PNG lossless
   ở artifacts/runs/<run>/adv/<method>/<img_id>.png (gitignored), ghi thống kê từng ảnh
   (L_inf, L2, runtime, số backward/forward-view) vào results/runs/<run>/attack_stats.jsonl.
   Ảnh đã có PNG + dòng stats thì bỏ qua.
2. eval: đọc PNG từ đĩa -> Resize đúng test pipeline -> predict trên 4 model -> COCO bbox
   AP/AP50 + paired bootstrap 95% CI (resample ảnh, chung cho mọi điều kiện/model) ->
   results/runs/<run>/metrics.json. Detection thô lưu artifacts/runs/<run>/dets.json.

Hyperparameter baseline: docs/protocol_lock.md ("Baseline hyperparameter").

Chạy (trong tmux):
  python scripts/run_baselines.py --split n300 --budget 50
  python scripts/run_baselines.py --split n300 --budget 50 --stage eval   # chỉ eval lại
"""
import argparse
import functools
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

from attack.data import build_attack_dataset
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.methods.baselines import di_fgsm_attack, mi_fgsm_attack, osfd_attack
from attack.models import CONTROLLED_PANEL, SURROGATE_KEY, load_model
from attack.preprocess import pipeline_resize, predict, to_adv_image

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
SEED = 0
BOOT_SEED = 2026
SAME_FAMILY, CROSS_CNN, CROSS_TR = "r101", "convnext_t", "swin_t"

# name -> (hàm attack, số view forward qua surrogate mỗi backward)
METHODS = {
    "MI-FGSM": (mi_fgsm_attack, 1),
    "M-DI2-FGSM": (di_fgsm_attack, 1),
    "OSFD": (osfd_attack, 2),  # RRB: 2 view / 1 backward (protocol_lock.md, đơn vị B)
    # Pilot A′1 (progress_log 2026-09-24) — không thuộc baseline table, chỉ chạy khi chỉ định.
    "OSFD-W1": (functools.partial(osfd_attack, spatial_weight="box"), 2),
    "OSFD-W2": (functools.partial(osfd_attack, spatial_weight="box_ring"), 2),
    # Pilot P2 (progress_log 2026-09-24).
    "OSFD-P2a": (functools.partial(osfd_attack, channel_concave="median"), 2),
    "OSFD-P2b": (functools.partial(osfd_attack, channel_concave="2median"), 2),
}
BASELINES = ["MI-FGSM", "M-DI2-FGSM", "OSFD"]


def load_stats(path):
    stats = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                stats[(r["method"], r["img_id"])] = r
    return stats


def img_tensor(path, device):
    return torch.from_numpy(mmcv.imread(path)).permute(2, 0, 1).float().to(device)


def stage_attack(args, dataset, surrogate, adv_dir, stats_path, device):
    stats = load_stats(stats_path)
    for name in args.methods:
        fn, views = METHODS[name]
        os.makedirs(os.path.join(adv_dir, name), exist_ok=True)
        t_start, n_done = time.time(), 0
        for i in range(len(dataset)):
            img_id = dataset.image_ids[i]
            png = os.path.join(adv_dir, name, f"{img_id}.png")
            if os.path.exists(png) and (name, img_id) in stats:
                continue
            sample = dataset[i]
            orig, ds = sample["orig_inputs"].to(device), sample["data_sample"]
            random.seed(SEED + i); torch.manual_seed(SEED + i)
            t0 = time.time()
            noise = fn(surrogate, orig, ds, steps=args.budget, epsilon=args.epsilon)
            torch.cuda.synchronize()
            runtime = time.time() - t0
            adv = to_adv_image(orig, noise)
            delta = (adv - orig) / 255.0
            linf = float((adv - orig).abs().max())
            assert linf <= args.epsilon + 1e-4, f"{name} img {img_id}: L_inf {linf}"
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png)
            rec = {"method": name, "img_id": img_id, "linf_255": linf,
                   "l2": float(delta.norm()), "l2_per_pixel_rms": float(delta.pow(2).mean().sqrt()),
                   "runtime_s": runtime, "backward": args.budget,
                   "forward_views": args.budget * views}
            with open(stats_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            stats[(name, img_id)] = rec
            n_done += 1
            if n_done % 10 == 0 or i == len(dataset) - 1:
                el = time.time() - t_start
                print(f"[attack {name}] {i + 1}/{len(dataset)}  {el / n_done:.1f}s/ảnh  "
                      f"còn ~{(len(dataset) - i - 1) * el / n_done / 60:.0f} phút", flush=True)
    return stats


def dets_name(out):
    """metrics.json -> dets.json (mặc định, như cũ); metrics_X.json -> dets_X.json."""
    return "dets.json" if out == "metrics.json" else out.replace("metrics", "dets", 1)


def stage_eval(args, dataset, models, adv_dir, stats, run_dir, art_dir, device):
    cat_ids = dataset._mmdet_dataset.cat_ids
    conds = ["clean"] + list(args.methods)
    dets = {c: {k: [] for k in models} for c in conds}
    t0 = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]
        img_id, x, ds = sample["img_id"], sample["inputs"].to(device), sample["data_sample"]
        imgs = {"clean": x}
        for name in args.methods:
            adv = img_tensor(os.path.join(adv_dir, name, f"{img_id}.png"), device)
            imgs[name] = pipeline_resize(adv, x.shape[-2:])
        for c, img in imgs.items():
            for k, m in models.items():
                dets[c][k] += to_coco_dets(predict(m, img, ds, rescale=True), img_id, cat_ids)
        if (i + 1) % 50 == 0:
            print(f"[eval] predict {i + 1}/{len(dataset)} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(art_dir, dets_name(args.out)), "w") as f:
        json.dump(dets, f)

    coco_gt = COCO(ANN_FILE)
    img_ids = list(dataset.image_ids)
    evals = {c: {k: CocoImageEval(coco_gt, dets[c][k], img_ids) for k in models} for c in conds}
    print("[eval] COCOeval xong (weighted AP khớp pycocotools)", flush=True)

    n = len(img_ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, args.n_boot, BOOT_SEED)])  # hàng 0 = mẫu gốc
    ap = {c: {k: np.array([evals[c][k].ap(w) for w in W]) * 100 for k in models} for c in conds}
    print(f"[eval] bootstrap {args.n_boot} mẫu xong ({time.time() - t0:.0f}s)", flush=True)

    def S(arr):  # arr[0] = điểm trên mẫu gốc, arr[1:] = bootstrap
        return summarize(arr[0], arr[1:])

    metrics = {"ap": {}, "ap50": {}, "relative_ap_drop": {}, "cross_avg": {}, "transfer_gap": {},
               "method_diff_cross_avg": {}, "perturbation": {}, "compute": {}}
    drop = {}
    for c in conds:
        metrics["ap"][c] = {k: S(ap[c][k][:, 0]) for k in models}
        metrics["ap50"][c] = {k: S(ap[c][k][:, 1]) for k in models}
    for name in args.methods:
        drop[name] = {k: 100 * (ap["clean"][k][:, 0] - ap[name][k][:, 0]) / ap["clean"][k][:, 0]
                      for k in models}
        metrics["relative_ap_drop"][name] = {k: S(drop[name][k]) for k in models}
        cross = (drop[name][CROSS_CNN] + drop[name][CROSS_TR]) / 2
        metrics["cross_avg"][name] = S(cross)
        metrics["transfer_gap"][name] = {
            "same_minus_cross_cnn": S(drop[name][SAME_FAMILY] - drop[name][CROSS_CNN]),
            "same_minus_cnn_to_transformer": S(drop[name][SAME_FAMILY] - drop[name][CROSS_TR]),
            "same_minus_cross_avg": S(drop[name][SAME_FAMILY] - cross),
        }
        rows = [stats[(name, i)] for i in img_ids]
        metrics["perturbation"][name] = {
            "linf_255_max": max(r["linf_255"] for r in rows),
            "l2_mean": float(np.mean([r["l2"] for r in rows])),
            "l2_per_pixel_rms_mean": float(np.mean([r["l2_per_pixel_rms"] for r in rows])),
        }
        metrics["compute"][name] = {
            "runtime_s_per_img": float(np.mean([r["runtime_s"] for r in rows])),
            "backward_per_img": rows[0]["backward"], "forward_views_per_img": rows[0]["forward_views"],
        }
    for a in args.methods:
        for b in args.methods:
            if a < b:
                ca = (drop[a][CROSS_CNN] + drop[a][CROSS_TR]) / 2
                cb = (drop[b][CROSS_CNN] + drop[b][CROSS_TR]) / 2
                metrics["method_diff_cross_avg"][f"{a} - {b}"] = S(ca - cb)

    out = {
        "run": os.path.basename(run_dir), "split": args.split, "n_images": n,
        "budget_B": args.budget, "epsilon_255": args.epsilon, "surrogate": SURROGATE_KEY,
        "targets": {"same_family": SAME_FAMILY, "cross_cnn": CROSS_CNN, "cnn_to_transformer": CROSS_TR},
        "eval": "PNG uint8 cỡ gốc -> Resize test pipeline; bbox, area=all, maxDets=100",
        "bootstrap": {"n_boot": args.n_boot, "seed": BOOT_SEED, "unit": "image, paired across conditions/models",
                      "ci": "percentile 2.5/97.5", "p_le_0": "tỉ lệ mẫu bootstrap <= 0"},
        "not_computed": ["ASR", "APloc", "CSR", "LPIPS (chưa cài lpips)"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
    }
    with open(os.path.join(run_dir, args.out), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print_table(out, list(models), args.methods)
    print("saved:", os.path.join(run_dir, args.out))


def print_table(m, keys, methods):
    fmt = lambda s: f"{s['point']:5.1f} [{s['ci95'][0]:5.1f},{s['ci95'][1]:5.1f}]"
    print(f"\n=== bbox AP (n={m['n_images']}, B={m['budget_B']}, eps={m['epsilon_255']}) ===")
    print(f"{'':<12}" + "".join(f"{k:>21}" for k in keys))
    for c, row in m["ap"].items():
        print(f"{c:<12}" + "".join(f"{fmt(row[k]):>21}" for k in keys))
    print("\n=== relative AP drop % [95% CI] ===")
    for c, row in m["relative_ap_drop"].items():
        print(f"{c:<12}" + "".join(f"{fmt(row[k]):>21}" for k in keys))
    print("\n=== CrossAvg / TransferGap (same-family drop − cross drop) ===")
    for c in methods:
        g = m["transfer_gap"][c]
        print(f"{c:<12} CrossAvg {fmt(m['cross_avg'][c])}  gap→ConvNeXt {fmt(g['same_minus_cross_cnn'])}"
              f"  gap→Swin {fmt(g['same_minus_cnn_to_transformer'])}  gap→avg {fmt(g['same_minus_cross_avg'])}")
    print("\n=== runtime ===", {c: round(v["runtime_s_per_img"], 1) for c, v in m["compute"].items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="n300")
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--epsilon", type=float, default=5.0)
    p.add_argument("--methods", nargs="+", default=BASELINES, choices=list(METHODS))
    p.add_argument("--out", default="metrics.json", help="tên file trong results/runs/<run>/")
    p.add_argument("--stage", choices=["all", "attack", "eval"], default="all")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None, help="chỉ để test nhanh, KHÔNG dùng cho bảng chính thức")
    args = p.parse_args()

    run = f"{args.split}_B{args.budget}_eps{args.epsilon:g}" + (f"_limit{args.limit}" if args.limit else "")
    run_dir = os.path.join(REPO_ROOT, "results/runs", run)
    art_dir = os.path.join(REPO_ROOT, "artifacts/runs", run)
    adv_dir = os.path.join(art_dir, "adv")
    os.makedirs(run_dir, exist_ok=True); os.makedirs(adv_dir, exist_ok=True)
    stats_path = os.path.join(run_dir, "attack_stats.jsonl")
    device = "cuda:0"

    dataset = build_attack_dataset(args.split)
    if args.limit:
        dataset.image_ids = dataset.image_ids[:args.limit]
        dataset._indices = dataset._indices[:args.limit]
    print(f"[run] {run}: {len(dataset)} ảnh, methods={args.methods}", flush=True)

    surrogate = load_model(SURROGATE_KEY, device)
    stats = load_stats(stats_path)
    if args.stage in ("all", "attack"):
        stats = stage_attack(args, dataset, surrogate, adv_dir, stats_path, device)
    if args.stage in ("all", "eval"):
        models = {k: surrogate if k == SURROGATE_KEY else load_model(k, device) for k in CONTROLLED_PANEL}
        stage_eval(args, dataset, models, adv_dir, stats, run_dir, art_dir, device)


if __name__ == "__main__":
    main()
