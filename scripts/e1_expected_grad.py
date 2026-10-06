#!/usr/bin/env python3
"""E1 (quy tắc khóa: progress_log 2026-10-06): gradient KỲ VỌNG của surrogate qua đa dạng hình học có tiến gần gradient
target không, và độ gần đó có giải thích thứ hạng transfer giữa các ô không.

Pha 1 (mỗi target nạp 1 lần): g_t = ∇ₓ loss GT của target theo ảnh gốc (objective cls | task), lưu CPU fp16.
Pha 2 (R50): mỗi ô T, 16 view rút bằng ĐÚNG sampler của ô attack (nhiễu σ = 6, đích feature co-transform, loss OSFD k = 3),
gradient từng view theo ảnh gốc; ḡ_T^K = trung bình K view đầu; A = cos(ḡ_T^K, g_t), trùng dấu.
  python scripts/e1_expected_grad.py                 -> results/mechanism/e1_expected_grad.json (+ per-image)
  python scripts/e1_expected_grad.py --analyze-only  # đọc lại per-image + drop dev30 (screen.json) -> Spearman
"""
import argparse
import functools
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.mechanism import boot_weights
from attack.methods.camgeo import cam_views
from attack.methods.elem import elem_views
from attack.methods.geopool import flip_views
from attack.methods.pia import pia_views
from attack.methods.rrb import rrb_views
from attack.models import load_surrogate
from attack.preprocess import extract_features, resize_to_model
from attack.targets import load_any, target_dataset, target_grads

SEED, BOOT_SEED, N_BOOT = 0, 2026, 1000
K_FEAT, SIGMA = 3.0, 6.0
KS = (1, 4, 16)
NV = 16
TARGETS = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
CROSS = ["convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
OBJ = ("cls", "task")
OUT = os.path.join(REPO_ROOT, "results/mechanism")


def _rrb(img, ds, comp, n, co):
    out = []
    while len(out) < n:
        for v, _, c in rrb_views(img, ds, with_gt=True, companion=comp, sigma=SIGMA):
            out.append((v, c if co else None))
    return out[:n]


# ô -> hàm (img, ds, comp, n) -> [(view, companion_warp | None)]; None = so với feature sạch KHÔNG biến đổi (OSFD gốc).
def _pair(triples):
    return [(v, c) for v, c, _ in triples]


CELLS = {
    "NONE": lambda img, ds, comp, n: [(img, comp)],
    "NOISE": lambda img, ds, comp, n: _pair(flip_views(img, n, 0.0, SIGMA, companion=comp)),
    "FLIP": lambda img, ds, comp, n: _pair(flip_views(img, n, 0.5, SIGMA, companion=comp)),
    "OSFD": lambda img, ds, comp, n: _rrb(img, ds, comp, n, co=False),
    "GEO-C0": lambda img, ds, comp, n: _rrb(img, ds, comp, n, co=True),
    "PIA": lambda img, ds, comp, n: _pair(pia_views(img, n, sigma=SIGMA, companion=comp)),
    "PIA-noT": lambda img, ds, comp, n: _pair(pia_views(img, n, sigma=SIGMA, companion=comp, drop="T")),
    "PIA-aff": lambda img, ds, comp, n: _pair(pia_views(img, n, pi=(0.0, 1.0, 0.0, 0.0), sigma=SIGMA, companion=comp)),
    "PIA-d0.3": lambda img, ds, comp, n: _pair(pia_views(img, n, delta_max=0.3, sigma=SIGMA, companion=comp)),
    "CAM-R15": lambda img, ds, comp, n: _pair(cam_views(img, n, 15.0, 0.0, SIGMA, companion=comp)),
    "CAM-R30": lambda img, ds, comp, n: _pair(cam_views(img, n, 30.0, 0.0, SIGMA, companion=comp)),
    "LENS": lambda img, ds, comp, n: _pair(cam_views(img, n, 0.0, 0.25, SIGMA, companion=comp)),
}
ELEM = {"ROT7": {"rot": 7.0}, "ROT15": {"rot": 15.0}, "ROT30": {"rot": 30.0}, "SHR15": {"shear": 0.15},
        "SHR30": {"shear": 0.30}, "ANI10": {"aniso": 0.10}, "ANI25": {"aniso": 0.25}, "SCL25": {"scale": 0.25}}
for _c, _s in ELEM.items():
    CELLS[_c] = functools.partial(lambda s, img, ds, comp, n: _pair(elem_views(img, n, s, SIGMA, companion=comp)), _s)


def view_grads(model, orig, ds, clean, cell, feats_clean):
    """Gradient (theo ảnh gốc) của từng view, list [C,h0,w0] (GPU)."""
    x = orig.clone().requires_grad_(True)
    img = resize_to_model(x, tuple(ds.img_shape))
    views = CELLS[cell](img, ds, clean, NV)
    grads = []
    for i, (v, c) in enumerate(views):
        with torch.no_grad():
            fc = feats_clean if c is None else [f.detach() for f in extract_features(model, c, ds)]
        loss = sum(F.mse_loss(K_FEAT * a, b) for a, b in zip(fc, extract_features(model, v, ds)))
        grads.append(torch.autograd.grad(loss, x, retain_graph=i < len(views) - 1)[0].detach())
    return grads


def cos(a, b):
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def sign_agree(a, b):
    return float((torch.sign(a) == torch.sign(b)).float().mean())


def run(args):
    dev = "cuda:0"
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv"))
    split_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    seed_idx = {i: split_ids.index(i) for i in ids}
    ids = ids[:args.limit] if args.limit else ids
    cells = args.cells or list(CELLS)

    # Pha 1: gradient target.
    tg = {}
    for t in TARGETS:
        model = load_surrogate(dev) if t == "r50" else load_any(t, dev)
        tds = target_dataset(t, ids)
        tg[t] = {o: [] for o in OBJ}
        t0 = time.time()
        for i in range(len(tds)):
            smp = tds[i]
            torch.manual_seed(SEED + seed_idx[smp["img_id"]])  # sampler RPN/ROI, nhiễu dn
            g, kc, kt = target_grads(model, t, smp["orig_inputs"].to(dev), smp["data_sample"], OBJ)
            for o in OBJ:
                tg[t][o].append(g[o].half().cpu())
        print(f"[E1 target] {t}: {len(tds)} ảnh, {time.time() - t0:.0f}s; cls={kc}; task={kt}", flush=True)
        del model
        torch.cuda.empty_cache()

    # Pha 2: surrogate, mọi ô.
    model = load_surrogate(dev)
    ds_all = AttackDataset(ids)
    per = {c: {str(k): {t: {o: {"cos": [], "sign": []} for o in OBJ} for t in TARGETS} for k in KS} for c in cells}
    self_cos = {c: [] for c in cells}  # cos giữa 2 nửa 8 view (độ ổn định ước lượng kỳ vọng)
    t0 = time.time()
    for i in range(len(ds_all)):
        smp = ds_all[i]
        ds, orig = smp["data_sample"], smp["orig_inputs"].to(dev)
        clean = resize_to_model(orig, tuple(ds.img_shape))
        with torch.no_grad():
            feats_clean = [f.detach() for f in extract_features(model, clean, ds)]
        for ci, c in enumerate(cells):
            seed = SEED + seed_idx[smp["img_id"]] + 1000 * (ci + 1)
            random.seed(seed); torch.manual_seed(seed)
            gs = view_grads(model, orig, ds, clean, c, feats_clean)
            csum = torch.cumsum(torch.stack(gs), 0)
            for k in KS:
                gbar = csum[min(k, len(gs)) - 1]
                for t in TARGETS:
                    for o in OBJ:
                        gt = tg[t][o][i].to(dev).float()
                        per[c][str(k)][t][o]["cos"].append(cos(gbar, gt))
                        per[c][str(k)][t][o]["sign"].append(sign_agree(gbar, gt))
            half = len(gs) // 2
            self_cos[c].append(cos(csum[half - 1], csum[-1] - csum[half - 1]) if half > 0 else 1.0)
        print(f"[E1 surrogate] {i + 1}/{len(ds_all)} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e1_per_image.json"), "w") as f:
        json.dump({"img_ids": ids, "per": per, "self_cos_halves": self_cos}, f)


def load_drops(subset="dev30"):
    """drop dev30 mỗi ô (điểm ước lượng) từ results/runs/dev300_B50_eps5/{metrics,generalization}_*_<subset>.json — file gốc,
    chung định dạng (screen.json cũ có 2 định dạng). Ô trùng nhiều file: lấy file mới nhất (theo created)."""
    run = os.path.join(REPO_ROOT, "results/runs/dev300_B50_eps5")
    drops = {}
    for kind in ("metrics", "generalization"):
        for p in sorted(glob.glob(os.path.join(run, f"{kind}_*_{subset}.json"))):
            d = json.load(open(p))
            for cell, row in d["relative_ap_drop"].items():
                drops.setdefault((kind, cell), []).append((d.get("created", ""), {m: v["point"] for m, v in row.items()}))
    out = {}
    for (kind, cell), v in drops.items():
        out.setdefault(cell, {}).update(sorted(v, key=lambda r: r[0])[-1][1])
    return {c: r for c, r in out.items() if all(t in r for t in TARGETS)}


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y)[0])


def analyze(args):
    d = json.load(open(os.path.join(OUT, "e1_per_image.json")))
    per, n = d["per"], len(d["img_ids"])
    idx = boot_weights(n, N_BOOT, BOOT_SEED)
    drops = load_drops(args.subset)
    cells = [c for c in per if c in drops]
    out = {"n_images": n, "cells_with_drop": cells, "mean": {}, "q1_vs_noise": {}, "q2_spearman": {},
           "self_cos_halves": {c: float(np.mean(v)) for c, v in d["self_cos_halves"].items()}, "drops": drops}

    def M(c, k, t, o, key="cos"):
        return np.array(per[c][str(k)][t][o][key])

    for c in per:
        out["mean"][c] = {str(k): {t: {o: {"cos": float(M(c, k, t, o).mean()), "sign": float(M(c, k, t, o, "sign").mean())}
                                       for o in OBJ} for t in TARGETS} for k in KS}
    for c in per:
        if c in ("NOISE", "NONE"):
            continue
        out["q1_vs_noise"][c] = {}
        for t in TARGETS:
            for o in OBJ:
                diff = M(c, 16, t, o) - M("NOISE", 16, t, o)
                bs = np.array([diff[r].mean() for r in idx])
                out["q1_vs_noise"][c][f"{t}/{o}"] = {"point": float(diff.mean()),
                                                     "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}
    verdict = {}
    for o in OBJ:
        for k in KS:
            for t in TARGETS:
                y = np.array([drops[c][t] for c in cells])
                A = np.stack([M(c, k, t, o) for c in cells])  # [cells, n]
                pt = spearman(A.mean(1), y)
                bs = np.array([spearman(A[:, r].mean(1), y) for r in idx])
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                out["q2_spearman"][f"{o}/K{k}/{t}"] = {"rho": pt, "ci95": [float(lo), float(hi)]}
                if k == 16 and t in CROSS:
                    verdict.setdefault(o, []).append((t, bool(pt >= 0.5 and lo > 0)))
    out["q2_verdict"] = {o: {"targets_ok": [t for t, ok in v if ok],
                             "supported": bool(sum(ok for _, ok in v) >= 3 and dict(v)["dino_swin_l"])} for o, v in verdict.items()}
    with open(os.path.join(OUT, "e1_expected_grad.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"=== E1 (n={n}) mean cos(ḡ_T^16, g_t), objective cls  [×100] ===")
    print(f"{'ô':<10}" + "".join(f"{t:>12}" for t in TARGETS) + f"{'ổn định':>9}")
    for c in per:
        print(f"{c:<10}" + "".join(f"{100 * out['mean'][c]['16'][t]['cls']['cos']:>12.2f}" for t in TARGETS)
              + f"{out['self_cos_halves'][c]:>9.2f}")
    print(f"\n=== Q2 Spearman giữa {len(cells)} ô (alignment vs drop dev30) ===")
    for o in OBJ:
        for k in KS:
            print(f"{o:<5} K={k:<3}" + "".join(
                f"  {t}: {out['q2_spearman'][f'{o}/K{k}/{t}']['rho']:+.2f} [{out['q2_spearman'][f'{o}/K{k}/{t}']['ci95'][0]:+.2f},"
                f"{out['q2_spearman'][f'{o}/K{k}/{t}']['ci95'][1]:+.2f}]" for t in CROSS))
    print("verdict:", out["q2_verdict"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="dev30")
    p.add_argument("--cells", nargs="*", default=None, choices=list(CELLS))
    p.add_argument("--limit", type=int, default=None, help="chỉ để test")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    if not args.analyze_only:
        run(args)
    if not args.limit:
        analyze(args)


if __name__ == "__main__":
    main()
