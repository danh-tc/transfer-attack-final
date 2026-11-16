"""G1 (progress_log 2026-11-17 02:40, KHÓA): cosine input-grad(cls) của CNX vs R50 so với DINO-Swin-L, dọc quỹ đạo
RD-CNX B50 (snapshot bước 0/25/50). Load model tuần tự tránh OOM. dev30.
  python scripts/g1_graddir.py --n 30
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import bootstrap_indices, summarize
from attack.models import load_model
from attack.targets import load_any, target_dataset, target_grads
from scripts.e88_ens_baselines import C, seed_all
from scripts.run_baselines import METHODS

SNAPS = [0, 25, 50]
OUT = os.path.join(REPO_ROOT, "results/mechanism/g1_graddir_dev30.json")


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:ap.parse_args().n]
    # 1) sinh RD-CNX, snapshot δ; lưu x = clean+δ (CPU) + g_cnx tại chỗ
    cnx = load_model(C, "cuda:0")
    ds_c = {s["img_id"]: s for s in target_dataset(C, ids)}
    fn, _ = METHODS["C2-OCT-1V-WP2-RD-CNX"]
    X = {}       # (n,snap) -> x cpu [C,h,w]
    gC = {}      # (n,snap) -> g_cnx cpu
    for n, i in enumerate(ids):
        orig = ds_c[i]["orig_inputs"].cuda().float(); dsm = ds_c[i]["data_sample"]
        snaps = {0: torch.zeros_like(orig)}
        seed_all(1000 * (n + 1))
        d = fn(cnx, orig, dsm, steps=50, epsilon=5.0,
               on_step=lambda k, nz: snaps.__setitem__(k, nz.detach().clone()) if k in SNAPS else None)
        snaps[50] = d.detach().clone()
        for s in SNAPS:
            x = torch.clamp(orig + snaps[s], 0, 255)
            X[(n, s)] = x.cpu()
            gC[(n, s)] = target_grads(cnx, C, x, dsm, objectives=("cls",))[0]["cls"].cpu()
        print(f"[g1 cnx] {n+1}/{len(ids)}", flush=True)
    del cnx; torch.cuda.empty_cache()

    def grads_for(key):
        m = load_model(key, "cuda:0") if key in ("r50",) else load_any(key)
        ds = {s["img_id"]: s for s in target_dataset(key, ids)}
        out = {}
        for n, i in enumerate(ids):
            dsm = ds[i]["data_sample"]
            for s in SNAPS:
                out[(n, s)] = target_grads(m, key, X[(n, s)].cuda().float(), dsm, objectives=("cls",))[0]["cls"].cpu()
            print(f"[g1 {key}] {n+1}/{len(ids)}", flush=True)
        del m; torch.cuda.empty_cache()
        return out
    gR = grads_for("r50")
    gS = grads_for("dino_swin_l")

    # 2) cosine theo ảnh, mỗi snapshot
    W = np.vstack([np.ones((1, len(ids))), bootstrap_indices(len(ids), 1000, 2026)])
    res = {"snaps": SNAPS, "per_snap": {}}
    for s in SNAPS:
        cc = np.array([cos(gC[(n, s)], gS[(n, s)]) for n in range(len(ids))])
        cr = np.array([cos(gR[(n, s)], gS[(n, s)]) for n in range(len(ids))])
        dcos = cc - cr
        ci = lambda v: summarize(float((W[0] @ v) / len(ids)), (W[1:] @ v) / len(ids))
        res["per_snap"][s] = {"cos_cnx_swin": ci(cc), "cos_r50_swin": ci(cr), "dcos": ci(dcos),
                              "ratio": float(cc.mean() / (cr.mean() + 1e-12))}
        q = res["per_snap"][s]
        print(f"step {s:2d}: cos(CNX,Swin) {cc.mean():+.4f}  cos(R50,Swin) {cr.mean():+.4f}  "
              f"Δcos {q['dcos']['point']:+.4f} [{q['dcos']['ci95'][0]:+.4f}, {q['dcos']['ci95'][1]:+.4f}]  ratio {q['ratio']:.2f}", flush=True)
    d50 = res["per_snap"][50]["dcos"]
    res["verdict"] = "XÁC NHẬN E68 dọc quỹ đạo (Δcos>0 bước 50)" if d50["ci95"][0] > 0 else "KHÔNG xác nhận (Δcos CI chứa 0 bước 50)"
    print("G1 VERDICT:", res["verdict"])
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
