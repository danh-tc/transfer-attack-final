#!/usr/bin/env python3
"""E14c (quy tắc khóa: progress_log 2026-10-10): hướng hiệu quả của WB ở token object bb3 = đẩy về phía nền?
u_g = μ_nền − μ_g (token SẠCH bb3, stride 16; token gán cho GT box nhỏ nhất chứa tâm ô). Mô tả s, e_u trên Swin-L và
trên C4 của R50; thao tác khung `noise_3` (chỉ token object): in, U, Uperp, synthWB, synthT. Chỉ forward, dev30.
  python scripts/e14c_bgshift.py                -> artifacts/e14/dets_c.json + results/mechanism/e14c_bgshift.json
  python scripts/e14c_bgshift.py --analyze-only
"""
import argparse
import glob
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

from attack.data import build_attack_dataset, load_image_ids
from attack.preprocess import pipeline_resize, to_batch
from scripts.e4_target_side import matched
from scripts.e14_patching import ANN_FILE, CFG, WB, head, to_coco
from scripts.e14b_direction import SRC, STRIDE

CONDS_WB = ["in", "U", "Uperp", "synthWB", "synthT"]
CONDS = ["in", "U", "Uperp"]
R50_SRC = ["OSFD", "OCT", "C2-PIA", "NOISE"]
DETS = os.path.join(REPO_ROOT, "artifacts/e14/dets_c.json")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e14c_bgshift.json")
BOOT_SEED, N_BOOT, SCORE = 2026, 1000, 0.3


def assign(ds, h, w, dev):
    """[h, w] long: chỉ số GT box nhỏ nhất chứa tâm ô, −1 = nền."""
    gi = ds.gt_instances
    b = gi.bboxes.tensor if hasattr(gi.bboxes, "tensor") else gi.bboxes
    if "ignore_flags" in gi:
        b = b[~gi.ignore_flags.bool()]
    b = b.float().to(dev)
    yy = (torch.arange(h, device=dev).float() + 0.5) * STRIDE
    xx = (torch.arange(w, device=dev).float() + 0.5) * STRIDE
    lab = torch.full((h, w), -1, dtype=torch.long, device=dev)
    best = torch.full((h, w), float("inf"), device=dev)
    for g, (x1, y1, x2, y2) in enumerate(b):
        inside = ((yy >= y1) & (yy < y2))[:, None] & ((xx >= x1) & (xx < x2))[None]
        area = float((x2 - x1) * (y2 - y1))
        upd = inside & (area < best)
        lab[upd], best[upd] = g, area
    return lab


def bg_dirs(c3, lab):
    """-> U [C, h, w]: u_g tại mỗi token object (0 ở nền / khi không có nền)."""
    C = c3.shape[0]
    U = torch.zeros_like(c3)
    bg = lab < 0
    if bg.sum() == 0 or (~bg).sum() == 0:
        return U, False
    mu_bg = c3[:, bg].mean(1)
    for g in lab[~bg].unique():
        m = lab == g
        U[:, m] = (mu_bg - c3[:, m].mean(1))[:, None]
    return U, True


def dir_stats(d, U, obj):
    """s = mean_t (Δ_t·u)/‖u‖², e_u = Σ(Δ_t·û)² / Σ‖Δ_t‖² trên token object."""
    dt, ut = d[:, obj].double(), U[:, obj].double()
    un2 = (ut ** 2).sum(0)
    dot = (dt * ut).sum(0)
    return float((dot / un2).mean()), float((dot ** 2 / un2).sum() / (dt ** 2).sum())


@torch.no_grad()
def run(coco, ids):
    from mmdet.apis import init_detector
    from attack.models import load_surrogate
    dev = "cuda:0"
    model = init_detector(CFG, glob.glob(os.path.join(REPO_ROOT, "checkpoints", "dino-5scale_swin-l*.pth"))[0], device=dev)
    r50 = load_surrogate(dev)
    cat_ids = sorted(coco.getCatIds())
    data = build_attack_dataset("dev300")
    pos = {i: k for k, i in enumerate(data.image_ids)}
    keys = [f"{WB}|{c}" for c in CONDS_WB] + [f"{s}|{c}" for s in SRC if s != WB for c in CONDS]
    dets = {"clean": []}
    dets.update({k: [] for k in keys})
    stats = {"swinl": {s: [] for s in SRC}, "r50": {s: [] for s in R50_SRC}}
    t0 = time.time()
    for n, img_id in enumerate(ids):
        smp = data[pos[img_id]]
        x, ds = smp["inputs"].to(dev), smp["data_sample"]
        xb, dss = to_batch(model, [x], [ds])
        c = list(model.backbone(xb))
        dets["clean"] += to_coco(head(model, c, dss), img_id, cat_ids)
        c3 = c[2][0]
        lab = assign(ds, *c3.shape[1:], dev)
        obj = lab >= 0
        U, ok = bg_dirs(c3, lab)
        uh = U / U.norm(dim=0, keepdim=True).clamp_min(1e-12)
        # R50 C4
        r_c4 = r50.backbone(to_batch(r50, [x], [ds])[0])[2][0]
        r_lab = assign(ds, *r_c4.shape[1:], dev)
        rU, r_ok = bg_dirs(r_c4, r_lab)
        imgs = {s: pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(root, f"{img_id}.png")))
                                   .permute(2, 0, 1).float().to(dev), x.shape[-2:]) for s, root in SRC.items()}
        D3 = {s: model.backbone(to_batch(model, [a], [ds])[0])[2][0] - c3 for s, a in imgs.items()}
        for s, d in D3.items():
            if ok:
                stats["swinl"][s].append(dir_stats(d, U, obj))
            din = d * obj
            par = (din * uh).sum(0, keepdim=True) * uh
            parts = {"in": din, "U": par, "Uperp": din - par}
            if s == WB:
                for name, ref in (("synthWB", WB), ("synthT", "OSFD")):
                    tgt = float((D3[ref] * obj).norm())
                    parts[name] = U * (tgt / max(float(U.norm()), 1e-12))
            for cond, t in parts.items():
                dets[f"{s}|{cond}"] += to_coco(head(model, [c[0], c[1], (c3 + t)[None], c[3]], dss), img_id, cat_ids)
        if r_ok:
            for s in R50_SRC:
                ra = r50.backbone(to_batch(r50, [imgs[s]], [ds])[0])[2][0]
                stats["r50"][s].append(dir_stats(ra - r_c4, rU, r_lab >= 0))
        if (n + 1) % 5 == 0:
            print(f"[E14c] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(DETS), exist_ok=True)
    with open(DETS, "w") as f:
        json.dump({"dets": dets, "stats": stats}, f)


def analyze(coco, ids):
    blob = json.load(open(DETS))
    dets, stats = blob["dets"], blob["stats"]
    n = len(ids)
    rng = np.random.default_rng(BOOT_SEED)
    W = np.vstack([np.ones((1, n)), [np.bincount(r, minlength=n) for r in rng.integers(0, n, (N_BOOT, n))]])
    cl = matched(coco, dets["clean"], ids)
    rows = [(k, j) for k, i in enumerate(ids) for j, sc in enumerate(cl[i][1]) if sc >= SCORE]
    w = W[:, np.array([r[0] for r in rows])]

    def D(key):
        mm = matched(coco, dets[key], ids)
        y = np.array([mm[ids[k]][1][j] < SCORE for k, j in rows], dtype=float)
        return (w * y).sum(1) / w.sum(1)

    ci = lambda v: [float(x) for x in np.percentile(v[1:], [2.5, 97.5])]
    out = {"n_images": n, "n_gt_detected_clean": len(rows),
           "D": {k: {"point": float(v[0]), "ci95": ci(v)} for k, v in ((k, D(k)) for k in dets if k != "clean")},
           "dir": {side: {s: {"s": float(np.mean([r[0] for r in v])), "e_u": float(np.mean([r[1] for r in v])), "n_img": len(v)}
                          for s, v in d.items()} for side, d in stats.items()}}
    p = lambda k: out["D"][k]["point"]
    sw = out["dir"]["swinl"]
    d_in = p(f"{WB}|in")
    out["verdict"] = {"B1": bool(sw[WB]["s"] > 0 and sw[WB]["s"] >= 2 * sw["NOISE"]["s"] and sw[WB]["s"] >= 2 * sw["OSFD"]["s"]),
                      "B2": bool(p(f"{WB}|U") >= 0.5 * d_in), "B3": bool(p(f"{WB}|synthT") >= 0.7 * d_in)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"GT phát hiện ở sạch: {len(rows)}")
    for k, v in out["D"].items():
        print(f"{k:<20}{v['point']:7.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    for side, d in out["dir"].items():
        dim = 768 if side == "swinl" else 1024
        print(f"\n=== hướng nền trên {side}: s (tỉ lệ khoảng cách object→nền) | e_u (×{dim} = bội so ngẫu nhiên) ===")
        for s, v in d.items():
            print(f"{s:<9} s {v['s']:+.4f}   e_u {v['e_u']:.5f} (×{v['e_u'] * dim:.1f})   n_img {v['n_img']}")
    print("\nVERDICT", out["verdict"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    ids = ids[:args.limit] if args.limit else ids
    if not args.analyze_only:
        run(coco, ids)
    analyze(coco, ids)


if __name__ == "__main__":
    main()
