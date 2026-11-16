"""T0 (progress_log 2026-11-16 22:05, KHÓA): gradient cls theo object đã bị giết vs còn sống trên CNX (white-box), dọc
quỹ đạo RD-CNX B50 đóng băng. Chỉ đo, không sinh attack mới.
  python scripts/t0_saturation.py --n 30
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from mmdet.structures.bbox import bbox2roi

from attack.data import load_image_ids
from attack.models import load_model
from attack.preprocess import resize_to_model, to_batch
from attack.targets import target_dataset
from scripts.e88_ens_baselines import C, seed_all
from scripts.run_baselines import METHODS

EPS, SNAPS, KILL, ALIVE, BOOT_SEED = 5.0, (0, 10, 25, 50), 0.1, 0.5, 2026
OUT = os.path.join(REPO_ROOT, "results/mechanism/t0_saturation_dev30.json")


def per_object(model, orig, delta, dsm):
    """-> p_y [n_obj], g [n_obj] (‖∇ₓ −log p_y‖₂ theo ảnh gốc)."""
    gt = dsm.gt_instances
    boxes, labels = gt.bboxes.tensor if hasattr(gt.bboxes, "tensor") else gt.bboxes, gt.labels
    ps, gs = [], []
    for j in range(len(labels)):
        x = (orig + delta).clone().requires_grad_(True)
        inputs, dss = to_batch(model, [resize_to_model(x, tuple(dsm.img_shape))], [dsm])
        feats = model.extract_feat(inputs)
        rois = bbox2roi([boxes[j:j + 1].to(inputs.device).float()])
        cls = model.roi_head._bbox_forward(feats, rois)["cls_score"]
        logp = F.log_softmax(cls, -1)[0, int(labels[j])]
        (-logp).backward()
        ps.append(float(logp.exp())); gs.append(float(x.grad.norm()))
    return np.array(ps), np.array(gs)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:args.n]
    model = load_model(C, "cuda:0")
    ds = {s["img_id"]: s for s in target_dataset(C, ids)}
    fn, _ = METHODS["C2-OCT-1V-WP2-RD-CNX"]
    rec = []   # mỗi object: img idx, p[snap], g[snap]
    for n, i in enumerate(ids):
        orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]
        if len(dsm.gt_instances.labels) == 0:
            continue
        snaps = {0: torch.zeros_like(orig)}
        seed_all(1000 * (n + 1))
        final = fn(model, orig, dsm, steps=50, epsilon=EPS,
                   on_step=lambda k, nz: snaps.__setitem__(k, nz.detach().clone()) if k in SNAPS else None)
        snaps[50] = final.detach().clone()
        model.eval()
        req = [q.requires_grad for q in model.parameters()]
        for q in model.parameters():
            q.requires_grad_(False)
        P, G = [], []
        for s in SNAPS:
            p_, g_ = per_object(model, orig, snaps[s], dsm)
            P.append(p_); G.append(g_)
        for q, r in zip(model.parameters(), req):
            q.requires_grad_(r)
        P, G = np.array(P).T, np.array(G).T   # [n_obj, n_snap]
        for j in range(len(P)):
            rec.append({"img": n, "p": P[j].tolist(), "g": G[j].tolist()})
        print(f"[t0] {n + 1}/{len(ids)} obj {len(P)}  p@50 {np.round(P[:, -1], 2).tolist()}", flush=True)
    img = np.array([r["img"] for r in rec]); p = np.array([r["p"] for r in rec]); g = np.array([r["g"] for r in rec])
    rng = np.random.default_rng(BOOT_SEED)
    nimg = len(ids)
    B = [np.arange(nimg)] + [rng.integers(0, nimg, nimg) for _ in range(1000)]
    by = [np.where(img == k)[0] for k in range(nimg)]
    res = {"n_img": nimg, "n_obj": len(rec), "snaps": SNAPS, "per_snap": {}}
    for si, s in enumerate(SNAPS):
        k_, a_ = p[:, si] < KILL, p[:, si] >= ALIVE

        def R(ix):
            kk, aa = ix[k_[ix]], ix[a_[ix]]
            return g[kk, si].mean() / g[aa, si].mean() if len(kk) and len(aa) else np.nan
        v = np.array([R(np.concatenate([by[k] for k in b])) for b in B])
        lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
        res["per_snap"][s] = {"n_killed": int(k_.sum()), "n_alive": int(a_.sum()), "g_killed": float(g[k_, si].mean()) if k_.any() else None,
                              "g_alive": float(g[a_, si].mean()) if a_.any() else None, "R": float(v[0]), "R_ci95": [float(lo), float(hi)]}
        q = res["per_snap"][s]
        print(f"step {s:3d}: killed {q['n_killed']:4d} alive {q['n_alive']:4d}  g_k {q['g_killed']}  g_a {q['g_alive']}  R {v[0]:.3f} [{lo:.3f}, {hi:.3f}]")
    # quỹ đạo: object sống@0 (p ≥ .5) và chết@50 (p < .1)
    tr = (p[:, 0] >= ALIVE) & (p[:, -1] < KILL)
    res["traj_killed_during"] = {"n": int(tr.sum()), "g_mean_by_snap": g[tr].mean(0).tolist() if tr.any() else None,
                                 "g_alive_throughout": g[(p >= ALIVE).all(1)].mean(0).tolist() if (p >= ALIVE).all(1).any() else None}
    print("quỹ đạo (sống@0, chết@50):", res["traj_killed_during"])
    q = res["per_snap"][50]
    res["verdict"] = ("BÃO HÒA ủng hộ -> khóa pilot logit-margin" if q["R_ci95"][1] < 0.3 else
                      "NO-GO -> khép hướng bão hòa" if q["R_ci95"][0] > 0.5 else "UNCLEAR")
    print(f"T0 VERDICT (khóa): {res['verdict']}")
    json.dump({**res, "objects": rec}, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[t0] -> {OUT}")


if __name__ == "__main__":
    main()
