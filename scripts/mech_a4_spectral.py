#!/usr/bin/env python3
"""Mechanism A4 (docs/mechanism_plan.md): phổ tần + độ tập trung của gradient và perturbation.

Đại lượng (không gian ảnh gốc, n300, run `n300_B50_eps5`):
- g_m = ∇ₓ L_task(m, x) trên ảnh sạch cho R50/R101/ConvNeXt-T/Swin-T (seed ảnh i, như A1).
- δ_method = PNG adv − ảnh gốc (MI-FGSM, M-DI2-FGSM, OSFD).
- Phân bố năng lượng theo 3 dải tần: FFT 2D từng kênh, cộng năng lượng 3 kênh; tần số
  radial f = sqrt(fx² + fy²) (cycles/pixel, np.fft.fftfreq). Dải CỐ ĐỊNH chia đều [0, Nyquist=0.5]:
  thấp [0, 1/6), trung [1/6, 1/3), cao [1/3, ∞) (góc phổ > 0.5 vào dải cao). Chuẩn hóa tổng = 1.
- Tập trung không gian (mô tả): tỉ lệ năng lượng ở top-5% pixel; tỉ lệ năng lượng trong GT
  box (không crowd) chia tỉ lệ diện tích box.

Tiêu chí vận hành (chốt + commit TRƯỚC khi chạy, 2026-09-23):
(i) dist_t = L1 giữa vector 3 dải của g_R50 và g_t (per-ảnh). "Lệch phổ": dist_cross − dist_R101
    có CI 95% > 0 cho CẢ ConvNeXt-T và Swin-T.
(ii) match(method, t) = cosine giữa vector 3 dải của δ_method và của g_t (per-ảnh).
    Spearman(match, suppression(method, t)) có CI > 0 cho ≥ 1 target khác họ, với method =
    OSFD VÀ M-DI2-FGSM (suppression per-ảnh từ a2_per_image.json).
A4 đạt nếu (i) và (ii). Retained confidence báo cáo song song (ceiling rule).

Chạy: python scripts/mech_a4_spectral.py
"""
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

from attack.data import build_attack_dataset
from attack.mechanism import boot_mean, boot_spearman, boot_weights
from attack.models import CONTROLLED_PANEL, SURROGATE_KEY, load_model
from attack.preprocess import compute_gt_loss, resize_to_model

RUN = "n300_B50_eps5"
METHODS = ["MI-FGSM", "M-DI2-FGSM", "OSFD"]
METHODS_CRIT = ["M-DI2-FGSM", "OSFD"]
TARGETS = ["r101", "convnext_t", "swin_t"]
CROSS = ["convnext_t", "swin_t"]
BANDS = [0.0, 1 / 6, 1 / 3, np.inf]
BOOT_SEED = 2026
N_BOOT = 1000


def band_fracs(t: np.ndarray) -> np.ndarray:
    """t [C,H,W] -> tỉ lệ năng lượng 3 dải tần."""
    power = (np.abs(np.fft.fft2(t, axes=(-2, -1))) ** 2).sum(0)
    fy = np.fft.fftfreq(t.shape[-2])[:, None]
    fx = np.fft.fftfreq(t.shape[-1])[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    e = np.array([power[(f >= lo) & (f < hi)].sum() for lo, hi in zip(BANDS[:-1], BANDS[1:])])
    return e / e.sum()


def spatial(t: np.ndarray, boxes) -> tuple:
    """t [C,H,W] -> (tỉ lệ năng lượng top-5% pixel, năng lượng-trong-box / diện-tích-box)."""
    e = (t ** 2).sum(0)
    flat = np.sort(e.ravel())[::-1]
    top5 = flat[: max(1, int(0.05 * flat.size))].sum() / flat.sum()
    mask = np.zeros(e.shape, bool)
    for x, y, w, h in boxes:
        mask[int(y): int(np.ceil(y + h)), int(x): int(np.ceil(x + w))] = True
    area = mask.mean()
    in_box = (e[mask].sum() / e.sum() / area) if area > 0 else np.nan
    return top5, in_box


def main():
    device = "cuda:0"
    out_dir = os.path.join(REPO_ROOT, "results/mechanism")
    adv_dir = os.path.join(REPO_ROOT, "artifacts/runs", RUN, "adv")
    dataset = build_attack_dataset("n300")
    img_ids = list(dataset.image_ids)
    n = len(img_ids)
    coco = COCO(os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json"))
    models = {k: load_model(k, device) for k in CONTROLLED_PANEL}

    bf = {k: np.zeros((n, 3)) for k in list(models) + METHODS}
    sp = {k: np.zeros((n, 2)) for k in list(models) + METHODS}
    t0 = time.time()
    for i in range(n):
        s = dataset[i]
        orig, ds = s["orig_inputs"].to(device), s["data_sample"]
        boxes = [a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=img_ids[i], iscrowd=False))]
        for m, mod in models.items():
            torch.manual_seed(i)
            x = orig.clone().requires_grad_(True)
            g = torch.autograd.grad(compute_gt_loss(mod, resize_to_model(x, tuple(ds.img_shape)), ds), x)[0]
            g = g.double().cpu().numpy()
            bf[m][i], sp[m][i] = band_fracs(g), spatial(g, boxes)
        o = orig.cpu().numpy().astype(np.float64)
        for me in METHODS:
            adv = mmcv.imread(os.path.join(adv_dir, me, f"{img_ids[i]}.png")).transpose(2, 0, 1).astype(np.float64)
            d = adv - o
            bf[me][i], sp[me][i] = band_fracs(d), spatial(d, boxes)
        if (i + 1) % 50 == 0:
            print(f"[A4] {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(out_dir, "a2_per_image.json")) as f:
        a2 = json.load(f)
    assert a2["img_ids"] == img_ids
    transfer = a2["transfer"]
    idx = boot_weights(n, N_BOOT, BOOT_SEED)

    dist = {t: np.abs(bf[SURROGATE_KEY] - bf[t]).sum(1) for t in TARGETS}
    cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    match = {me: {t: cos(bf[me], bf[t]) for t in TARGETS} for me in METHODS}
    res = {"n_images": n, "bands_cycles_per_pixel": ["[0,1/6)", "[1/6,1/3)", "[1/3,inf)"],
           "band_fracs_mean": {k: [boot_mean(v[:, b], idx) for b in range(3)] for k, v in bf.items()},
           "spatial_mean": {k: {"top5_share": boot_mean(v[:, 0], idx), "in_box_ratio": boot_mean(v[:, 1], idx)}
                            for k, v in sp.items()},
           "spectral_dist_r50": {t: boot_mean(dist[t], idx) for t in TARGETS},
           "dist_cross_minus_r101": {c: boot_mean(dist[c] - dist["r101"], idx) for c in CROSS},
           "corr_match_transfer": {me: {t: {q: boot_spearman(match[me][t], np.array(transfer[me][t][q], dtype=float), idx)
                                            for q in ("suppression", "retained")} for t in TARGETS} for me in METHODS}}
    c1 = all(res["dist_cross_minus_r101"][c]["ci95"][0] > 0 for c in CROSS)
    c2 = {me: any(res["corr_match_transfer"][me][c]["suppression"]["ci95"][0] > 0 for c in CROSS) for me in METHODS_CRIT}
    res["criterion"] = {"spectral_shift_both_cross": c1, "match_corr_pos_any_cross": c2,
                        "A4_pass": bool(c1 and all(c2.values()))}
    with open(os.path.join(out_dir, "a4_spectral.json"), "w") as f:
        json.dump(res, f, indent=2)

    f3 = lambda s: f"{s['point']:.4f} [{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}]"
    print("\n=== band energy fractions (low / mid / high) ===")
    for k in bf:
        print(f"  {k:<11} " + "  ".join(f"{s['point']:.4f}" for s in res["band_fracs_mean"][k]))
    print("\n=== spatial: top-5% share / in-box ratio ===")
    for k in sp:
        print(f"  {k:<11} {res['spatial_mean'][k]['top5_share']['point']:.3f}  {res['spatial_mean'][k]['in_box_ratio']['point']:.3f}")
    print("\n=== L1 spectral distance g_R50 vs g_t ===")
    for t in TARGETS:
        print(f"  {t:<11} {f3(res['spectral_dist_r50'][t])}")
    for c in CROSS:
        print(f"  {c} − r101: {f3(res['dist_cross_minus_r101'][c])}")
    print("\n=== Spearman(match δ↔g_t, suppression) / (match, retained) ===")
    for me in METHODS:
        for t in TARGETS:
            d = res["corr_match_transfer"][me][t]
            print(f"  {me:<11}{t:<11} {f3(d['suppression'])}   {f3(d['retained'])}")
    print(f"\ncriterion: {res['criterion']}")
    print("saved:", os.path.join(out_dir, "a4_spectral.json"))


if __name__ == "__main__":
    main()
