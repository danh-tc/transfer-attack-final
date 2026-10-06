#!/usr/bin/env python3
"""E8 (quy tắc khóa: progress_log 2026-10-06): TRẦN — trong L∞ ε = 5/255, nhiễu QUY MÔ LỚN có hạ được object lớn của
DINO-Swin-L không? White-box trên CHÍNH Swin-L (chỉ đo trần, không phải method; không vi phạm threat model vì không dùng
sinh attack transfer).
δ = upsample_bilinear(z), z ở lưới thô (h0/s, w0/s) của ảnh gốc, s ∈ {1 (pixel), 16, 32}; |z| ≤ ε ⇒ |δ| ≤ ε.
Loss = số hạng phân loại không-dn của DINO (attack.targets, objective 'cls'), ascent; MI μ = 1, α = 1 (trên z), B bước.
  python scripts/e8_whitebox_lowfreq.py --scales 1 16 32 --budget 20
"""
import argparse
import json
import os
import sys
import time

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.preprocess import to_adv_image
from attack.targets import load_any, target_dataset, target_grads

ART = os.path.join(REPO_ROOT, "artifacts/e8")
SEED = 0


def up(z, size):
    return F.interpolate(z[None], size=size, mode="bilinear", align_corners=True)[0]


def attack(model, orig, ds, s, steps, eps, alpha=1.0, mu=1.0):
    h0, w0 = orig.shape[-2:]
    zs = (max(2, round(h0 / s)), max(2, round(w0 / s))) if s > 1 else (h0, w0)
    z = torch.zeros((3, *zs), device=orig.device)
    acc = torch.zeros_like(z)
    for _ in range(steps):
        zz = z.clone().requires_grad_(True)
        d = up(zz, (h0, w0)) if s > 1 else zz
        g_img, _, _ = target_grads(model, "dino_swin_l", torch.clamp(orig + d.detach(), 0, 255), ds, ("cls",))
        g = torch.autograd.grad(d, zz, grad_outputs=g_img["cls"])[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = mu * acc + g
        z = torch.clamp(z + alpha * torch.sign(acc), -eps, eps)
    return up(z, (h0, w0)) if s > 1 else z


def evaluate(ids, conds):
    """Swin-L predict (inference_detector) trên ảnh sạch + PNG từng điều kiện; suppression / retained theo kích thước,
    bootstrap cụm theo ảnh. Mốc transfer: PNG dev30 của run dev300 (OSFD, PIA-noT, NOISE)."""
    from mmdet.apis import inference_detector
    from pycocotools.coco import COCO
    from attack.evaluation import CocoImageEval, to_coco_dets
    from scripts.e4_target_side import SIZES, matched
    coco = COCO(os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json"))
    cat_ids = sorted(coco.getCatIds())
    model = load_any("dino_swin_l")
    run = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5/adv")
    paths = {c: (lambda i, c=c: os.path.join(ART, c, f"{i}.png")) for c in conds}
    for c in ("OSFD", "PIA-noT", "NOISE"):
        paths[c] = lambda i, c=c: os.path.join(run, c, f"{i}.png")
    paths["clean"] = lambda i: os.path.join(REPO_ROOT, "data/coco/val2017", coco.loadImgs(i)[0]["file_name"])
    dets = {c: [] for c in paths}
    for i in ids:
        for c, pth in paths.items():
            dets[c] += to_coco_dets(inference_detector(model, pth(i)), i, cat_ids)
    area = {a["id"]: a["area"] for a in coco.loadAnns(coco.getAnnIds(imgIds=ids, iscrowd=False))}
    zone = lambda ar: next(n for n, lo, hi in SIZES if lo <= ar < hi)
    cl = matched(coco, dets["clean"], ids)
    rng = np.random.default_rng(2026)
    W = np.vstack([np.ones((1, len(ids))), [np.bincount(r, minlength=len(ids)) for r in rng.integers(0, len(ids), (1000, len(ids)))]])
    pos = {i: k for k, i in enumerate(ids)}
    ap_clean = CocoImageEval(coco, dets["clean"], ids).ap(np.ones(len(ids)))[0]
    out = {}
    for c in paths:
        if c == "clean":
            continue
        cc = matched(coco, dets[c], ids)
        rec = [(pos[i], zone(area[a]), float(sa < 0.3)) for i in ids for a, sc, sa in zip(cl[i][0], cl[i][1], cc[i][1]) if sc >= 0.3]
        ap = CocoImageEval(coco, dets[c], ids).ap(np.ones(len(ids)))[0]
        out[c] = {"ap_drop": 100 * (ap_clean - ap) / ap_clean}
        for z, _, _ in SIZES:
            r = [x for x in rec if x[1] == z]
            img = np.array([x[0] for x in r]); y = np.array([x[2] for x in r]); w = W[:, img]
            v = (w * y).sum(1) / np.maximum(w.sum(1), 1e-9)
            out[c][z] = {"n": len(r), "sup": float(v[0]), "ci95": [float(x) for x in np.percentile(v[1:], [2.5, 97.5])]}
    with open(os.path.join(REPO_ROOT, "results/mechanism/e8_whitebox_lowfreq.json"), "w") as f:
        json.dump({"ids": len(ids), "results": out}, f, indent=2)
    print("=== E8: DINO-Swin-L suppression theo kích thước [CI] | AP drop % ===")
    for c, r in out.items():
        print(f"{c:<9} AP drop {r['ap_drop']:5.1f}  " + "  ".join(
            f"{z}: {r[z]['sup']:.2f} [{r[z]['ci95'][0]:.2f},{r[z]['ci95'][1]:.2f}] (n={r[z]['n']})" for z, _, _ in SIZES))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--osfd", action="store_true", help="chỉ ô WB-OSFD (OSFD đúng công thức, surrogate = DINO-Swin-L)")
    p.add_argument("--subset", default="dev30")
    p.add_argument("--scales", nargs="+", type=int, default=[1, 16, 32])
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv"))
    split = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    ids = ids[:args.limit] if args.limit else ids
    if args.eval_only:
        return evaluate(ids, [f"WB-s{s}" for s in args.scales])
    model = load_any("dino_swin_l")
    tds = target_dataset("dino_swin_l", ids)
    stats = []
    if args.osfd:
        # WB-OSFD: ĐÚNG công thức OSFD (loss feature backbone k = 3, MI μ = 1, RRB, B bước) với surrogate = DINO-Swin-L.
        # Swin checkpoint khi input requires_grad (kể cả eval) -> không tương thích autograd.grad; chỉ backbone nên tắt được.
        from attack.methods.baselines import osfd_attack
        import random
        for m in model.modules():
            if getattr(m, "with_cp", False):
                m.with_cp = False
        name = "WB-OSFD"
        os.makedirs(os.path.join(ART, name), exist_ok=True)
        t0 = time.time()
        for i in range(len(tds)):
            smp = tds[i]
            png = os.path.join(ART, name, f"{smp['img_id']}.png")
            if os.path.exists(png):
                continue
            seed = SEED + split.index(smp["img_id"])
            random.seed(seed); torch.manual_seed(seed)
            orig = smp["orig_inputs"].cuda()
            d = osfd_attack(model, orig, smp["data_sample"], steps=args.budget, epsilon=args.epsilon)
            adv = to_adv_image(orig, d)
            assert float((adv - orig).abs().max()) <= args.epsilon + 1e-4
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png)
            if (i + 1) % 5 == 0:
                print(f"[E8 {name}] {i + 1}/{len(tds)} ({time.time() - t0:.0f}s) peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        del model
        torch.cuda.empty_cache()
        return evaluate(ids, [name])
    for s in args.scales:
        name = f"WB-s{s}"
        os.makedirs(os.path.join(ART, name), exist_ok=True)
        t0 = time.time()
        for i in range(len(tds)):
            smp = tds[i]
            png = os.path.join(ART, name, f"{smp['img_id']}.png")
            if os.path.exists(png):
                continue
            torch.manual_seed(SEED + split.index(smp["img_id"]))
            orig = smp["orig_inputs"].cuda()
            d = attack(model, orig, smp["data_sample"], s, args.budget, args.epsilon)
            adv = to_adv_image(orig, d)
            assert float((adv - orig).abs().max()) <= args.epsilon + 1e-4
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png)
            stats.append({"cond": name, "img_id": smp["img_id"], "linf": float((adv - orig).abs().max()),
                          "l2_rms": float(((adv - orig) / 255).pow(2).mean().sqrt())})
            if (i + 1) % 10 == 0:
                print(f"[E8 {name}] {i + 1}/{len(tds)} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(ART, "stats.jsonl"), "a") as f:
        for r in stats:
            f.write(json.dumps(r) + "\n")
    del model
    torch.cuda.empty_cache()
    evaluate(ids, [f"WB-s{s}" for s in args.scales])


if __name__ == "__main__":
    main()
