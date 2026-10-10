#!/usr/bin/env python3
"""E14b (quy tắc khóa: progress_log 2026-10-10): hướng distortion bb3 (stride 16) của WB khác R50 ở đâu — không gian
(object / nền) hay không gian con kênh (32 PC trội của token sạch / phần còn lại)? Khung `noise_3` của E14: sạch mọi
stage, riêng bb3 = c₃ + T(Δ₃). Chỉ forward, dev30.
  python scripts/e14b_direction.py                -> artifacts/e14/dets_b.json + results/mechanism/e14b_direction.json
  python scripts/e14b_direction.py --analyze-only
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
from scripts.e14_patching import ANN_FILE, CFG, SRC as SRC14, WB, head, to_coco

SRC = dict(SRC14, NOISE=os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5/adv/NOISE"))
CONDS = ["n3", "in", "out", "P32", "Q32"]
K_PC, STRIDE = 32, 16
DETS = os.path.join(REPO_ROOT, "artifacts/e14/dets_b.json")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e14b_direction.json")
BOOT_SEED, N_BOOT, SCORE = 2026, 1000, 0.3


def box_mask(ds, h, w, dev):
    """[h, w] bool: ô lưới stride 16 có tâm nằm trong ≥ 1 GT box (khung model, bỏ ignore)."""
    gi = ds.gt_instances
    b = gi.bboxes.tensor if hasattr(gi.bboxes, "tensor") else gi.bboxes
    if "ignore_flags" in gi:
        b = b[~gi.ignore_flags.bool()]
    b = b.float().to(dev)
    yy = (torch.arange(h, device=dev).float() + 0.5) * STRIDE
    xx = (torch.arange(w, device=dev).float() + 0.5) * STRIDE
    m = torch.zeros(h, w, dtype=torch.bool, device=dev)
    for x1, y1, x2, y2 in b:
        m |= ((yy >= y1) & (yy < y2))[:, None] & ((xx >= x1) & (xx < x2))[None]
    return m


@torch.no_grad()
def run(coco, ids):
    from mmdet.apis import init_detector
    dev = "cuda:0"
    model = init_detector(CFG, glob.glob(os.path.join(REPO_ROOT, "checkpoints", "dino-5scale_swin-l*.pth"))[0], device=dev)
    cat_ids = sorted(coco.getCatIds())
    data = build_attack_dataset("dev300")
    pos = {i: k for k, i in enumerate(data.image_ids)}
    dets = {"clean": []}
    dets.update({f"{s}|{c}": [] for s in SRC for c in CONDS})
    stats = {s: [] for s in SRC}  # mỗi ảnh: dict mô tả
    t0 = time.time()
    for n, img_id in enumerate(ids):
        smp = data[pos[img_id]]
        x, ds = smp["inputs"].to(dev), smp["data_sample"]
        xb, dss = to_batch(model, [x], [ds])
        c = list(model.backbone(xb))
        dets["clean"] += to_coco(head(model, c, dss), img_id, cat_ids)
        c3 = c[2][0]                                            # [768, h, w]
        C, h, w = c3.shape
        m = box_mask(ds, h, w, dev)
        tok = c3.reshape(C, -1).T.double()                      # [hw, 768]
        U = torch.linalg.eigh(torch.cov(tok.T))[1][:, -K_PC:]   # 32 PC đầu (eigh tăng dần)
        proj = lambda d: ((d.reshape(C, -1).T.double() @ U) @ U.T).T.reshape(C, h, w).float()
        D3 = {}
        for s, root in SRC.items():
            a = pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(root, f"{img_id}.png")))
                                .permute(2, 0, 1).float().to(dev), x.shape[-2:])
            D3[s] = model.backbone(to_batch(model, [a], [ds])[0])[2][0] - c3
        for s in SRC:
            d = D3[s]
            parts = {"n3": d, "in": d * m, "out": d * ~m, "P32": proj(d)}
            parts["Q32"] = d - parts["P32"]
            for cond, t in parts.items():
                f = [c[0], c[1], (c3 + t)[None], c[3]]
                dets[f"{s}|{cond}"] += to_coco(head(model, f, dss), img_id, cat_ids)
            e = float((d.double() ** 2).sum())
            cosw = lambda u, v: float((u.double() * v.double()).sum() / (u.double().norm() * v.double().norm() + 1e-12))
            dw = D3[WB]
            stats[s].append({"e_in": float((d.double() ** 2 * m).sum()) / e, "tok_in": float(m.float().mean()),
                             "e_P32": float((parts["P32"].double() ** 2).sum()) / e,
                             "cos_WB": cosw(d, dw), "cos_WB_in": cosw(d * m, dw * m), "cos_WB_out": cosw(d * ~m, dw * ~m)})
        if (n + 1) % 5 == 0:
            print(f"[E14b] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
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
    Ds = {k: D(k) for k in dets if k != "clean"}
    out = {"n_images": n, "n_gt_detected_clean": len(rows),
           "D": {k: {"point": float(v[0]), "ci95": ci(v)} for k, v in Ds.items()},
           "stats": {s: {k: float(np.mean([r[k] for r in stats[s]])) for k in stats[s][0]} for s in SRC}}
    p = lambda s, c: out["D"][f"{s}|{c}"]["point"]
    n3 = p(WB, "n3")
    st = out["stats"]
    verdict = {}
    for name, (A, B), key in (("H-S", ("in", "out"), "e_in"), ("H-P", ("P32", "Q32"), "e_P32")):
        frac = lambda s, part: st[s][key] if part == A else 1 - st[s][key]
        sens = A if p(WB, A) >= 0.7 * n3 and p(WB, B) <= 0.3 * n3 else B if p(WB, B) >= 0.7 * n3 and p(WB, A) <= 0.3 * n3 else None
        v = {"sensitive_part": sens}
        if sens:
            fw, fo = frac(WB, sens), frac("OSFD", sens)
            v.update(frac_WB=fw, frac_OSFD=fo,
                     supported=bool(fo <= fw - 0.10) if name == "H-S" else bool(fo <= 0.6 * fw))
        verdict[name] = v
    out["verdict"] = verdict
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"GT phát hiện ở sạch: {len(rows)}")
    print(f"\n{'nguồn':<9}" + "".join(f"{c:>18}" for c in CONDS))
    for s in SRC:
        print(f"{s:<9}" + "".join(f"{out['D'][f'{s}|{c}']['point']:7.3f} [{out['D'][f'{s}|{c}']['ci95'][0]:.2f},{out['D'][f'{s}|{c}']['ci95'][1]:.2f}]"
                                  for c in CONDS))
    print(f"\n{'mô tả':<9}" + "".join(f"{k:>11}" for k in st[WB]))
    for s in SRC:
        print(f"{s:<9}" + "".join(f"{v:11.3f}" for v in st[s].values()))
    print("\nVERDICT", json.dumps(verdict, ensure_ascii=False))


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
