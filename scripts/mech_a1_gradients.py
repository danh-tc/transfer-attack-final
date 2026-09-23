#!/usr/bin/env python3
"""Mechanism A1 (docs/mechanism_plan.md): input-gradient alignment.

g_m = ∇ₓ L_task(m, x) cho 4 model, x = ẢNH GỐC (qua resize_to_model, như attack),
L_task = compute_gt_loss (tổng 5 loss Mask R-CNN — đúng loss của MI/M-DI²). Chỉ số với
surrogate R50: cosine(g_R50, g_t) và sign agreement = tỉ lệ pixel sign(g_R50) == sign(g_t)
(attack đi theo sign). Mốc trên (trần nhiễu): R50 vs R50 khác seed sampler.

Cách hiểu tiêu chí (chốt TRƯỚC khi chạy, 2026-09-23): "align" phải đạt với CẢ cosine VÀ
sign agreement: (i) align(R50,R101) − align(R50,cross) có CI > 0 cho cả ConvNeXt và Swin;
(ii) Spearman(align(R50,t) per-ảnh, suppression(method,t) per-ảnh) có CI > 0 cho ít nhất
1 target khác họ, với method = OSFD VÀ M-DI² (per-ảnh transfer từ a2_per_image.json).
Retained confidence báo cáo song song (ceiling rule).

--trajectory: mở rộng, đo lại alignment tại x_adv sau 10/25/50 step của M-DI² và OSFD
(chạy lại attack với cùng seed như run baseline, chụp δ qua on_step). Mô tả, không thuộc
tiêu chí.

Chạy: python scripts/mech_a1_gradients.py [--trajectory]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset
from attack.mechanism import boot_mean, boot_spearman, boot_weights
from attack.methods.baselines import di_fgsm_attack, osfd_attack
from attack.models import CONTROLLED_PANEL, SURROGATE_KEY, load_model
from attack.preprocess import compute_gt_loss, resize_to_model

TARGETS = ["r101", "convnext_t", "swin_t"]
CROSS = ["convnext_t", "swin_t"]
METHODS_CRIT = ["M-DI2-FGSM", "OSFD"]
BOOT_SEED = 2026
ATTACK_SEED = 0  # = SEED của scripts/run_baselines.py (seed ảnh i = ATTACK_SEED + i)
TRAJ_STEPS = [10, 25, 50]
TRAJ_FNS = {"M-DI2-FGSM": di_fgsm_attack, "OSFD": osfd_attack}


def input_grad(model, orig, ds, seed):
    torch.manual_seed(seed)
    x = orig.clone().requires_grad_(True)
    loss = compute_gt_loss(model, resize_to_model(x, tuple(ds.img_shape)), ds)
    return torch.autograd.grad(loss, x)[0].flatten()


def align(ga, gb):
    cos = float(torch.nn.functional.cosine_similarity(ga, gb, dim=0))
    sign = float((torch.sign(ga) == torch.sign(gb)).float().mean())
    return cos, sign


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--trajectory", action="store_true")
    args = p.parse_args()
    out_dir = os.path.join(REPO_ROOT, "results/mechanism")
    device = "cuda:0"

    dataset = build_attack_dataset("n300")
    img_ids = list(dataset.image_ids)
    n = len(img_ids)
    models = {k: load_model(k, device) for k in CONTROLLED_PANEL}
    pairs = TARGETS + ["r50_reseed"]
    A = {mt: {pr: np.zeros(n) for pr in pairs} for mt in ("cosine", "sign")}
    traj = {me: {s: {mt: {t: np.zeros(n) for t in TARGETS} for mt in ("cosine", "sign")}
                 for s in TRAJ_STEPS} for me in TRAJ_FNS} if args.trajectory else None

    t0 = time.time()
    for i in range(n):
        s = dataset[i]
        orig, ds = s["orig_inputs"].to(device), s["data_sample"]
        g = {m: input_grad(models[m], orig, ds, seed=i) for m in models}
        g["r50_reseed"] = input_grad(models[SURROGATE_KEY], orig, ds, seed=i + 10_000)
        for pr in pairs:
            A["cosine"][pr][i], A["sign"][pr][i] = align(g[SURROGATE_KEY], g[pr])

        if args.trajectory:
            for me, fn in TRAJ_FNS.items():
                snaps = {}
                def on_step(k, noise):
                    if k in TRAJ_STEPS:
                        snaps[k] = noise.clone()
                random.seed(ATTACK_SEED + i); torch.manual_seed(ATTACK_SEED + i)
                fn(models[SURROGATE_KEY], orig, ds, steps=max(TRAJ_STEPS), on_step=on_step)
                for k, noise in snaps.items():
                    xa = torch.clamp(orig + noise, 0, 255)
                    ga = {m: input_grad(models[m], xa, ds, seed=i) for m in models}
                    for t in TARGETS:
                        c, sg = align(ga[SURROGATE_KEY], ga[t])
                        traj[me][k]["cosine"][t][i], traj[me][k]["sign"][t][i] = c, sg
        if (i + 1) % 25 == 0:
            print(f"[A1] {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(out_dir, "a2_per_image.json")) as f:
        transfer = json.load(f)["transfer"]
    assert json.load(open(os.path.join(out_dir, "a2_per_image.json")))["img_ids"] == img_ids

    idx = boot_weights(n, args.n_boot, BOOT_SEED)
    res = {"n_images": n, "n_boot": args.n_boot, "boot_seed": BOOT_SEED, "loss": "compute_gt_loss",
           "alignment": {}, "same_minus_cross": {}, "corr_align_transfer": {}, "criterion": {}}
    for mt in ("cosine", "sign"):
        res["alignment"][mt] = {pr: boot_mean(A[mt][pr], idx) for pr in pairs}
        res["same_minus_cross"][mt] = {c: boot_mean(A[mt]["r101"] - A[mt][c], idx) for c in CROSS}
        res["corr_align_transfer"][mt] = {
            me: {t: {q: boot_spearman(A[mt][t], np.array(transfer[me][t][q], dtype=float), idx)
                     for q in ("suppression", "retained")} for t in TARGETS}
            for me in ["MI-FGSM"] + METHODS_CRIT}
    # Đánh giá tiêu chí đã chốt (trong docstring).
    for mt in ("cosine", "sign"):
        c1 = all(res["same_minus_cross"][mt][c]["ci95"][0] > 0 for c in CROSS)
        c2 = {me: any(res["corr_align_transfer"][mt][me][c]["suppression"]["ci95"][0] > 0 for c in CROSS)
              for me in METHODS_CRIT}
        res["criterion"][mt] = {"same_gt_cross_both": c1, "corr_pos_any_cross": c2,
                                "pass": bool(c1 and all(c2.values()))}
    res["criterion"]["A1_pass"] = res["criterion"]["cosine"]["pass"] and res["criterion"]["sign"]["pass"]
    if args.trajectory:
        res["trajectory"] = {me: {str(k): {mt: {t: boot_mean(traj[me][k][mt][t], idx) for t in TARGETS}
                                           for mt in ("cosine", "sign")} for k in TRAJ_STEPS} for me in TRAJ_FNS}

    name = "a1_gradients_traj.json" if args.trajectory else "a1_gradients.json"
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, name.replace(".json", "_per_image.json")), "w") as f:
        json.dump({"img_ids": img_ids, "alignment": {mt: {pr: v.tolist() for pr, v in d.items()}
                                                      for mt, d in A.items()}}, f)

    f3 = lambda s: f"{s['point']:.4f} [{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}]"
    for mt in ("cosine", "sign"):
        print(f"\n=== {mt}: align(R50, ·) ===")
        for pr in pairs:
            print(f"  {pr:<11} {f3(res['alignment'][mt][pr])}")
        for c in CROSS:
            print(f"  R101 − {c:<11} {f3(res['same_minus_cross'][mt][c])}")
        print(f"  Spearman(align, suppression) / (align, retained):")
        for me in res["corr_align_transfer"][mt]:
            for t in TARGETS:
                d = res["corr_align_transfer"][mt][me][t]
                print(f"    {me:<11}{t:<11} {f3(d['suppression'])}   {f3(d['retained'])}")
        print(f"  criterion: {res['criterion'][mt]}")
    print(f"\nA1_pass = {res['criterion']['A1_pass']}")
    if args.trajectory:
        print("\n=== trajectory: sign agreement R50 vs target tại step k ===")
        for me in TRAJ_FNS:
            for k in TRAJ_STEPS:
                print(f"  {me:<11} k={k:<3}" + "  ".join(
                    f"{t}: {res['trajectory'][me][str(k)]['sign'][t]['point']:.4f}" for t in TARGETS))
    print("saved:", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
