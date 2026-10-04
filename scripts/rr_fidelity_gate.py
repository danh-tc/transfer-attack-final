#!/usr/bin/env python3
"""Cổng độ trung thực của phép sửa R-a (progress_log 2026-10-04, KHÓA) — chỉ forward.

Q_rep(ảnh)  = retained confidence (attack/mechanism.py) của R50 khi C2, C3 thay bằng feature SẠCH
              (p = 1, không RRB) rồi mới qua FPN -> RPN/ROI.
Q_plain     = cùng chỉ số với R50 thường (white-box).
QUA nếu ở CẢ OSFD và C2 (PROBE-osfd+cls), dev100 (PNG run dev300_B50_eps5, seed theo dev300):
  Spearman(Q_rep, retained DINO-Swin-L) ≥ +0.30  VÀ  ρ(Q_rep) − ρ(Q_plain) ≥ +0.10 với cận dưới
  CI 95% paired bootstrap (1000 mẫu, BOOT_SEED 2026) > 0.
Mô tả: mean Q_rep / Q_plain, cosd FPN (trung bình 5 mức) có / không sửa, tương quan với các target khác.

Cần trước: PNG adv (run_baselines --subset dev100) + gen_dets/dino_swin_l.json (eval_generalization).
Chạy: python scripts/rr_fidelity_gate.py [--dets dets_rr_dev100.json]
"""
import argparse
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.evaluation import to_coco_dets
from attack.mechanism import boot_weights, ci, per_image_transfer
from attack.models import load_surrogate
from attack.preprocess import extract_features, pipeline_resize, predict, to_batch
from scipy.stats import spearmanr

RUN = "dev300_B50_eps5"
CONDS = ["OSFD", "PROBE-osfd+cls"]
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
BOOT_SEED, N_BOOT = 2026, 1000
RHO_MIN, DIFF_MIN = 0.30, 0.10


def repair_hook(clean_feats, n=2):
    return lambda m, i, o: tuple(clean_feats[:n]) + tuple(o[n:])


def spearman(a, b, ii):
    x, y = a[ii], b[ii]
    ok = ~(np.isnan(x) | np.isnan(y))
    return spearmanr(x[ok], y[ok])[0] if ok.sum() > 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="dets_rr_dev100.json", help="dets Controlled Panel (mô tả)")
    args = ap.parse_args()
    art = os.path.join(REPO_ROOT, "artifacts/runs", RUN)
    dev300 = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    ids = [i for i in dev300 if i in keep]  # thứ tự như eval (split dev300, subset dev100)
    ds = AttackDataset(ids)
    cat_ids = ds._mmdet_dataset.cat_ids
    model = load_surrogate("cuda:0")
    dets = {"clean": [], **{f"{c}|{v}": [] for c in CONDS for v in ("plain", "rep")}}
    cosd = {f"{c}|{v}": [] for c in CONDS for v in ("plain", "rep")}

    def fpn(img, dsm, hook=None):
        h = model.backbone.register_forward_hook(hook) if hook else None
        try:
            return model.extract_feat(to_batch(model, [img], [dsm])[0])
        finally:
            if h: h.remove()

    with torch.no_grad():
        for i in range(len(ds)):
            smp = ds[i]
            x, dsm = smp["inputs"].cuda(), smp["data_sample"]
            clean_feats = extract_features(model, x, dsm)
            dets["clean"] += to_coco_dets(predict(model, x, dsm), ids[i], cat_ids)
            fpn_clean = fpn(x, dsm)
            for c in CONDS:
                adv = torch.from_numpy(mmcv.imread(os.path.join(art, "adv", c, f"{ids[i]}.png"))).permute(2, 0, 1)
                adv = pipeline_resize(adv.float().cuda(), x.shape[-2:])
                for v, hook in (("plain", None), ("rep", repair_hook(clean_feats))):
                    h = model.backbone.register_forward_hook(hook) if hook else None
                    try:
                        dets[f"{c}|{v}"] += to_coco_dets(predict(model, adv, dsm), ids[i], cat_ids)
                    finally:
                        if h: h.remove()
                    fa = fpn(adv, dsm, hook)
                    cosd[f"{c}|{v}"].append(float(np.mean([
                        1 - torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
                        for a, b in zip(fa, fpn_clean)])))
            if (i + 1) % 25 == 0:
                print(f"[gate] {i + 1}/{len(ds)}", flush=True)

    coco = COCO(ANN)
    q = {k: per_image_transfer(coco, dets["clean"], d, ids)["retained"] for k, d in dets.items() if k != "clean"}
    with open(os.path.join(art, "gen_dets", "dino_swin_l.json")) as f:
        g = json.load(f)
    tgt = {"dino_swin_l": {c: per_image_transfer(coco, g["clean"], g[c], ids)["retained"] for c in CONDS}}
    # mô tả: các target khác nếu đã có dets
    ctrl_path = os.path.join(art, args.dets)
    if os.path.exists(ctrl_path):
        with open(ctrl_path) as f:
            ctrl = json.load(f)
        for k in ("r101", "convnext_t", "swin_t"):
            tgt[k] = {c: per_image_transfer(coco, ctrl["clean"][k], ctrl[c][k], ids)["retained"] for c in CONDS}
    yp = os.path.join(art, "gen_dets", "yolox_s.json")
    if os.path.exists(yp):
        with open(yp) as f:
            y = json.load(f)
        tgt["yolox_s"] = {c: per_image_transfer(coco, y["clean"], y[c], ids)["retained"] for c in CONDS}

    idx = boot_weights(len(ids), N_BOOT, BOOT_SEED)
    allidx = np.arange(len(ids))
    res = {"n_images": len(ids), "criteria": f"rho_rep >= {RHO_MIN} AND diff >= {DIFF_MIN} AND diff CI lo > 0, ở cả {CONDS}",
           "conds": {}}
    passes = []
    for c in CONDS:
        r = {"mean_Q_plain": float(np.nanmean(q[f"{c}|plain"])), "mean_Q_rep": float(np.nanmean(q[f"{c}|rep"])),
             "cosd_fpn_plain": float(np.mean(cosd[f"{c}|plain"])), "cosd_fpn_rep": float(np.mean(cosd[f"{c}|rep"])),
             "spearman": {}}
        for t, tv in tgt.items():
            a, b, y_ = q[f"{c}|rep"], q[f"{c}|plain"], tv[c]
            rr = [spearman(a, y_, ii) for ii in idx]
            rp = [spearman(b, y_, ii) for ii in idx]
            r["spearman"][t] = {
                "rho_rep": ci(spearman(a, y_, allidx), np.array(rr)),
                "rho_plain": ci(spearman(b, y_, allidx), np.array(rp)),
                "diff": ci(spearman(a, y_, allidx) - spearman(b, y_, allidx), np.array(rr) - np.array(rp)),
            }
        s = r["spearman"]["dino_swin_l"]
        ok = (s["rho_rep"]["point"] >= RHO_MIN and s["diff"]["point"] >= DIFF_MIN and s["diff"]["ci95"][0] > 0)
        r["pass"] = bool(ok)
        passes.append(ok)
        res["conds"][c] = r
        print(f"\n== {c} ==  Q_plain {r['mean_Q_plain']:.3f}  Q_rep {r['mean_Q_rep']:.3f}  "
              f"cosd FPN {r['cosd_fpn_plain']:.3f} -> {r['cosd_fpn_rep']:.3f}")
        for t, v in r["spearman"].items():
            f_ = lambda z: f"{z['point']:+.3f} [{z['ci95'][0]:+.3f},{z['ci95'][1]:+.3f}]"
            print(f"  {t:<12} ρ_rep {f_(v['rho_rep'])}  ρ_plain {f_(v['rho_plain'])}  diff {f_(v['diff'])}")
        print(f"  -> {'QUA' if ok else 'TRƯỢT'}")
    res["gate_pass"] = bool(all(passes))
    print(f"\nCỔNG ĐỘ TRUNG THỰC R-a: {'QUA' if res['gate_pass'] else 'TRƯỢT'}")
    out = os.path.join(REPO_ROOT, "results/repair_robust/fidelity_gate_Ra.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
