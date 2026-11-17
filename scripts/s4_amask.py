"""S4 (progress_log 2026-11-17, KHÓA): A giới hạn trong MASK object có dự báo drop Swin-L không — CPU, dùng g_t đã lưu
của S3 (artifacts/runs/s3_region/grads) + PNG S1 + drop theo (ô, ảnh) của S1.
A_MASK(c, i, t) = Σ_MASK sign δ·g_t / Σ_MASK |g_t|. Chính: TB t ∈ {0.5, 1}; phụ: path (0, 0.5, 1), t=0, và A_ALL (S2).
Ba mức tương quan với drop Swin-L của 6 ô attack: theo ô (6), gộp (ô, ảnh) (60), TRONG từng ảnh (Spearman 6 ô / ảnh, TB).
  python scripts/s4_amask.py
"""
import json
import os
import sys

import mmcv
import numpy as np
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.s3_region import ADV, ANN, GDIR, IMGDIR, regions

ATT = ["RD-ALL", "RD-FEAT", "RD-DET", "OSFD-ALL", "OSFD-FEAT", "OSFD-DET"]
OUT = os.path.join(REPO_ROOT, "results/mechanism/s4_amask_dev10.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s1 = json.load(open(os.path.join(REPO_ROOT, "results/mechanism/s1_signmap_dev10.json"), encoding="utf-8"))
    ids = s1["ids"]
    drop_c = {c: s1["drop"]["dino_swin_l"][c] for c in ATT}
    drop_i = {(p["cell"], p["img"]): p["drop"] for p in s1["sign_vs_drop"]["points"]}
    coco = COCO(ANN)
    A = {}   # (variant, c, i) -> value
    for i in ids:
        clean = mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32)
        m = regions(coco, i, *clean.shape[:2])["MASK"][None]
        g = {"0.0": np.load(os.path.join(GDIR, f"{i}_clean.npy")).astype(np.float32)}
        for c in ATT:
            sd = np.sign(mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean).transpose(2, 0, 1)
            for t in ("0.5", "1.0"):
                g[t] = np.load(os.path.join(GDIR, f"{i}_{c}_t{t}.npy")).astype(np.float32)
            for t in ("0.0", "0.5", "1.0"):
                mm = np.broadcast_to(m, g[t].shape)
                A[("mask", t, c, i)] = float((sd * g[t])[mm].sum() / np.abs(g[t])[mm].sum())
                A[("all", t, c, i)] = float((sd * g[t]).sum() / np.abs(g[t]).sum())
    variants = {"MASK mid(0.5,1) [CHÍNH]": ("mask", ("0.5", "1.0")), "MASK path(0,0.5,1)": ("mask", ("0.0", "0.5", "1.0")),
                "MASK t=0": ("mask", ("0.0",)), "ALL mid(0.5,1) [S2]": ("all", ("0.5", "1.0"))}
    res = {"n": len(ids), "variants": {}}
    print(f"{'biến thể':26}{'ρ ô(6)':>9}{'ρ (ô,ảnh)':>11}{'ρ trong ảnh TB':>16}{'#ảnh ρ>0':>10}")
    for name, (reg, ts) in variants.items():
        a = {(c, i): np.mean([A[(reg, t, c, i)] for t in ts]) for c in ATT for i in ids}
        a_c = {c: np.mean([a[(c, i)] for i in ids]) for c in ATT}
        rc = spearmanr([a_c[c] for c in ATT], [drop_c[c] for c in ATT]).correlation
        keys = [k for k in a if k in drop_i and np.isfinite(drop_i[k])]
        rp = spearmanr([a[k] for k in keys], [drop_i[k] for k in keys]).correlation
        wi = []
        for i in ids:
            ks = [(c, i) for c in ATT if (c, i) in drop_i and np.isfinite(drop_i[(c, i)])]
            dv = [drop_i[k] for k in ks]
            if len(ks) >= 4 and np.std(dv) > 0:
                wi.append(spearmanr([a[k] for k in ks], dv).correlation)
        wi = np.array(wi)
        res["variants"][name] = {"A_cell": a_c, "spearman_cell": rc, "spearman_pooled": rp,
                                 "spearman_within_mean": float(np.nanmean(wi)), "n_img_within": int(len(wi)),
                                 "n_within_pos": int((wi > 0).sum()), "within": wi.tolist()}
        print(f"{name:26}{rc:+9.2f}{rp:+11.2f}{np.nanmean(wi):+16.2f}{f'{(wi > 0).sum()}/{len(wi)}':>10}")
    print("\nA_MASK mid theo ô vs drop Swin-L:")
    for c in ATT:
        print(f"  {c:10} A {res['variants']['MASK mid(0.5,1) [CHÍNH]']['A_cell'][c]:.4f}   drop {drop_c[c]:.1f}")
    p = res["variants"]["MASK mid(0.5,1) [CHÍNH]"]
    v = ("A_MASK DỰ BÁO drop (cả giữa ô lẫn trong ảnh)" if p["spearman_cell"] >= 0.6 and p["spearman_within_mean"] >= 0.3
         else "A_MASK KHÔNG dự báo" if p["spearman_cell"] <= 0.1 and p["spearman_within_mean"] <= 0.1 else "KHÔNG RÕ")
    res["verdict"] = v
    print(f"\n=> {v}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[s4] -> {OUT}")


if __name__ == "__main__":
    main()
