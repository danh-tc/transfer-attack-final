"""A1 (progress_log 2026-11-15): tín hiệu object-level từ ConvNeXt (surrogate, white-box) có dự báo object RES3 bị giết thêm
ở B100 không, vượt metadata? CPU, dets F1b dev100 đã lưu. Matching E77 (score ≥ 0.3, IoU ≥ 0.5 cùng lớp). Không attack mới.
Q1  concordance sống@B50: RES3 vs CNX (log-OR, bootstrap ảnh); trần tham chiếu = cùng target seed1 vs seed2.
Q2  quần thể: object at-risk trên target m VÀ còn sống ở m@B50; y = chết ở m@B100. Logistic L2 (Newton), CV 5-fold nhóm theo
    ẢNH (seed 2026), RES3 gộp (one-hot model trong M). M = log area, điểm sạch target, supercategory, model.
    S = [CNX at-risk, CNX sống@50, điểm CNX@50, điểm sạch CNX, tụt điểm CNX] từ RD-CNX@50 (S1, cùng quỹ đạo) hoặc @50s2 (S2).
Q3  ΔAUC = AUC(M+S) − AUC(M) trên dự báo out-of-fold; CI bootstrap ảnh 1000 seed 2026.
GO (khóa): CI95 ΔAUC(M+S1) > 0 VÀ CI95 ΔAUC(M+S2) > 0. Ngược lại không mở object-aware method.
  python scripts/a1_cnx_signal.py  -> results/mechanism/a1_cnx_signal_dev100.json"""
import gzip
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from scripts.e77_dfc import fate, group
from scripts.f1b_r1a import ANN, ART, GROUPS

RUN = "f1b_r1a_eps5_3090"
OUT = os.path.join(REPO_ROOT, "results/mechanism/a1_cnx_signal_dev100.json")
SEED, N_BOOT, K, LAM = 2026, 1000, 5, 1.0


def auc(y, s):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = np.empty(len(s)); o = np.argsort(s, kind="mergesort"); r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)       # rank trung bình khi trùng
    r = (np.bincount(inv, r) / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logit_fit(X, y, lam=LAM, it=50):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1]); P = lam * np.eye(Z.shape[1]); P[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + P @ w
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + P
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda Xn: 1 / (1 + np.exp(-np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="RES3", choices=["RES3", "SWINL"],
                    help="SWINL: A1-SwinL (progress_log 2026-11-16), y hệt định nghĩa/cổng, target = dino_swin_l")
    args = ap.parse_args()
    TG = GROUPS["RES3"] if args.group == "RES3" else ["dino_swin_l"]
    out = OUT if args.group == "RES3" else OUT.replace("_dev100", "_swinl_dev100")
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    n = len(ids)
    sup = sorted({c["supercategory"] for c in coco.loadCats(coco.getCatIds())})
    supof = {c["id"]: sup.index(c["supercategory"]) for c in coco.loadCats(coco.getCatIds())}
    D = {m: json.load(gzip.open(os.path.join(ART, RUN, "dets", m, "all.json.gz"), "rt")) for m in [*TG, "convnext_t"]}
    G = {m: {c: group(D[m][c]) for c in ("clean", "RD-CNX@50", "RD-CNX@100", "RD-CNX@50s2")} for m in D}
    anns = [(k, a) for k, i in enumerate(ids) for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))]

    def st(m, c, k, a):
        f, s = fate(a["bbox"], a["category_id"], G[m][c].get(ids[k], []))
        return f == "survived", (s or 0.0)

    rows = []
    for k, a in anns:
        cnx_risk, cnx_clean = st("convnext_t", "clean", k, a)
        cs = {c: st("convnext_t", c, k, a) for c in ("RD-CNX@50", "RD-CNX@50s2")}
        for j, m in enumerate(TG):
            risk, sc = st(m, "clean", k, a)
            if not risk:
                continue
            a50, _ = st(m, "RD-CNX@50", k, a); a100, _ = st(m, "RD-CNX@100", k, a); a50s2, _ = st(m, "RD-CNX@50s2", k, a)
            S = {}
            for tag, c in (("S1", "RD-CNX@50"), ("S2", "RD-CNX@50s2")):
                al, s = cs[c]
                S[tag] = [float(cnx_risk), float(al and cnx_risk), s if cnx_risk else 0.0, cnx_clean if cnx_risk else 0.0,
                          (cnx_clean - s) if cnx_risk else 0.0]
            M = [np.log(a["area"] + 1), sc] + [float(supof[a["category_id"]] == t) for t in range(len(sup))] + \
                [float(j == t) for t in range(1, 3)]
            rows.append({"img": k, "model": m, "a50": a50, "a100": a100, "a50s2": a50s2, "cnx_risk": cnx_risk,
                         "cnx_a50": cs["RD-CNX@50"][0] and cnx_risk, "M": M, **S})
    rng = np.random.default_rng(SEED)
    B = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(N_BOOT)]
    img_all = np.array([r["img"] for r in rows])
    by_all = [np.where(img_all == k)[0] for k in range(n)]

    def boot(fn, by):
        v = np.array([fn(np.concatenate([by[k] for k in b])) for b in B])
        lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
        return {"point": float(v[0]), "ci95": [float(lo), float(hi)]}

    res = {"n_images": n, "matching": "E77", "Q1": {}, "Q2": {}}
    # ---- Q1: concordance sống@50 (object at-risk trên cả target lẫn CNX) ----
    def logor(x, y):
        a_ = np.sum(x & y) + .5; b_ = np.sum(x & ~y) + .5; c_ = np.sum(~x & y) + .5; d_ = np.sum(~x & ~y) + .5
        return np.log(a_ * d_ / (b_ * c_))
    A50 = np.array([r["a50"] for r in rows]); C50 = np.array([r["cnx_a50"] for r in rows]); CR = np.array([r["cnx_risk"] for r in rows])
    A50s2 = np.array([r["a50s2"] for r in rows])
    for m in [*TG, "RES3"]:
        sel = np.array([(r["model"] == m or m == "RES3") for r in rows]) & CR
        by = [np.where((img_all == k) & sel)[0] for k in range(n)]
        q = {"n_obj": int(sel.sum()), "P_cnx_alive_given_alive": float(C50[sel & A50].mean()),
             "P_cnx_alive_given_dead": float(C50[sel & ~A50].mean()),
             "logOR_target_vs_CNX": boot(lambda ix: logor(A50[ix], C50[ix]), by),
             "logOR_seed1_vs_seed2_ref": boot(lambda ix: logor(A50[ix], A50s2[ix]), by)}
        res["Q1"][m] = q
        print(f"Q1 {m:9} n={q['n_obj']:4d}  P(CNX sống|sống) {q['P_cnx_alive_given_alive']:.2f}  P(CNX sống|chết) "
              f"{q['P_cnx_alive_given_dead']:.2f}  logOR {q['logOR_target_vs_CNX']['point']:+.2f} "
              f"[{q['logOR_target_vs_CNX']['ci95'][0]:+.2f},{q['logOR_target_vs_CNX']['ci95'][1]:+.2f}]  (trần seed "
              f"{q['logOR_seed1_vs_seed2_ref']['point']:+.2f})")
    # ---- Q2/Q3: dự báo giết thêm B50→B100 trong object còn sống@50 ----
    pop = [r for r in rows if r["a50"]]
    y = np.array([not r["a100"] for r in pop], float)
    img = np.array([r["img"] for r in pop])
    perm = rng.permutation(n); fold = np.empty(n, int); fold[perm] = np.arange(n) % K
    X = {"M": np.array([r["M"] for r in pop])}
    X["M+S1"] = np.c_[X["M"], [r["S1"] for r in pop]]
    X["M+S2"] = np.c_[X["M"], [r["S2"] for r in pop]]
    X["S1only"] = np.array([r["S1"] for r in pop])
    oof = {}
    for name, Xm in X.items():
        p = np.empty(len(pop))
        for f in range(K):
            te = fold[img] == f
            p[te] = logit_fit(Xm[~te], y[~te])(Xm[te])
        oof[name] = p
    by = [np.where(img == k)[0] for k in range(n)]
    res["Q2"] = {"n_pop": len(pop), "n_pos": int(y.sum()), "base_rate": float(y.mean()),
                 "AUC": {k: boot(lambda ix, k=k: auc(y[ix], oof[k][ix]), by) for k in oof},
                 "dAUC_S1": boot(lambda ix: auc(y[ix], oof["M+S1"][ix]) - auc(y[ix], oof["M"][ix]), by),
                 "dAUC_S2": boot(lambda ix: auc(y[ix], oof["M+S2"][ix]) - auc(y[ix], oof["M"][ix]), by),
                 "AUC_cnx_alive50_univariate": boot(lambda ix: auc(y[ix], np.array([r["S1"][1] for r in pop])[ix]), by)}
    q = res["Q2"]
    print(f"\nQ2 quần thể: {q['n_pop']} object sống@50 (RES3 gộp), giết thêm @100: {q['n_pos']} (base rate {q['base_rate']:.2f})")
    for k, v in q["AUC"].items():
        print(f"   AUC {k:7} {v['point']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    for k in ("dAUC_S1", "dAUC_S2", "AUC_cnx_alive50_univariate"):
        print(f"   {k:26} {q[k]['point']:+.3f} [{q[k]['ci95'][0]:+.3f}, {q[k]['ci95'][1]:+.3f}]")
    go = q["dAUC_S1"]["ci95"][0] > 0 and q["dAUC_S2"]["ci95"][0] > 0
    res["verdict"] = "GO (tín hiệu CNX bổ sung, ổn định 2 seed)" if go else "NO-GO (không mở object-aware method)"
    print(f"A1 VERDICT: {res['verdict']}")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[a1] -> {out}")


if __name__ == "__main__":
    main()
