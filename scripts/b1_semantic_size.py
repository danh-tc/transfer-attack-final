"""B1 (progress_log 2026-11-15, CPU): tín hiệu "person" của B0 có sống sau khi kiểm soát kích thước/hình dạng/điểm sạch không,
và CNX white-box có proxy (yếu hơn trên person) không? Giả thuyết hình thành trên dev100 (B0), KIỂM LẶP trên devC[140:260].
Matching E77. Đơn vị: object at-risk trên target (RES3 gộp). Logistic L2 nhỏ (λ=1e-3), bootstrap ẢNH 1000 seed 2026 (fit lại mỗi mẫu).
  H1  y = ENS giết & CNX-B50 sống;  X = [person, log area, log aspect, điểm sạch, model one-hot]  -> hệ số person
  H2  (proxy chỉ-CNX) trên target convnext_t: y = CNX-B50 còn sống;  X = [person, log area, log aspect, điểm sạch] -> hệ số person
Bộ dữ liệu: dev100 (F1b: ENS-WP vs RD-CNX@50) và devC (a_alpha/a_ens_alpha: ENS1 vs B50 [α=1], ENS2 vs A2 [α=2]).
  python scripts/b1_semantic_size.py -> results/mechanism/b1_semantic_size.json"""
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
from scripts.a_alpha import ids_devC

OUT = os.path.join(REPO_ROOT, "results/mechanism/b1_semantic_size.json")
PERSON = 1


def fit(X, y, lam=1e-3, it=60):
    Z = np.c_[np.ones(len(X)), X]
    w = np.zeros(Z.shape[1]); P = lam * np.eye(Z.shape[1]); P[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + P
        s = np.linalg.solve(H, Z.T @ (p - y) + P @ w); w -= s
        if np.abs(s).max() < 1e-8:
            break
    return w


def dets(ds, m):
    if ds == "dev100":
        D = json.load(gzip.open(os.path.join(ART, "f1b_r1a_eps5_3090", "dets", m, "all.json.gz"), "rt"))
        return {"clean": D["clean"], "CNX": D["RD-CNX@50"], "ENS": D["ENS-WP"]}
    D = json.load(gzip.open(os.path.join(ART, "a_alpha_devC_eps5_3090", "dets", m, "all.json.gz"), "rt"))
    E = json.load(gzip.open(os.path.join(ART, "a_alpha_devC_eps5_3090", "dets", m, "ens_alpha.json.gz"), "rt"))
    if ds == "devC-α1":
        return {"clean": D["clean"], "CNX": D["B50"], "ENS": E["ENS1"]}
    return {"clean": D["clean"], "CNX": D["A2"], "ENS": E["ENS2"]}


def rows_for(coco, ids, ds, models, need_ens=True):
    out = []
    for j, m in enumerate(models):
        G = {k: group(v) for k, v in dets(ds, m).items()}
        for k, i in enumerate(ids):
            for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
                f, sc = fate(a["bbox"], a["category_id"], G["clean"].get(i, []))
                if f != "survived":
                    continue
                w, h = max(a["bbox"][2], 1), max(a["bbox"][3], 1)
                cnx = fate(a["bbox"], a["category_id"], G["CNX"].get(i, []))[0] == "survived"
                ens = fate(a["bbox"], a["category_id"], G["ENS"].get(i, []))[0] == "survived"
                out.append({"img": k, "x": [float(a["category_id"] == PERSON), np.log(a["area"] + 1), np.log(w / h), sc]
                            + [float(j == t) for t in range(1, len(models))], "cnx_alive": cnx, "ens_extra": (not ens) and cnx})
    return out


def coef(rows, ykey, n):
    X = np.array([r["x"] for r in rows]); y = np.array([r[ykey] for r in rows], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    sd[0] = 1.0; mu[0] = 0.0                       # giữ person ở thang 0/1 -> hệ số = log-OR
    Xs = (X - mu) / sd
    img = np.array([r["img"] for r in rows]); by = [np.where(img == k)[0] for k in range(n)]
    rng = np.random.default_rng(2026)
    v = [fit(Xs, y)[1]]
    for _ in range(1000):
        ix = np.concatenate([by[q] for q in rng.integers(0, n, n)])
        v.append(fit(Xs[ix], y[ix])[1])
    v = np.array(v); lo, hi = np.percentile(v[1:], [2.5, 97.5])
    return {"logOR_person": float(v[0]), "ci95": [float(lo), float(hi)], "n_obj": len(rows), "base_rate": float(y.mean()),
            "raw_rate_person": float(y[X[:, 0] == 1].mean()), "raw_rate_other": float(y[X[:, 0] == 0].mean())}


def main():
    coco = COCO(ANN)
    sets = {"dev100": load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")),
            "devC-α1": ids_devC(), "devC-α2": ids_devC()}
    res = {}
    for ds, ids in sets.items():
        n = len(ids)
        h1 = coef(rows_for(coco, ids, ds, GROUPS["RES3"]), "ens_extra", n)
        h2 = coef(rows_for(coco, ids, ds, ["convnext_t"]), "cnx_alive", n)
        res[ds] = {"H1_ens_extra_RES3": h1, "H2_cnx_alive_whitebox": h2}
        for name, h in (("H1 ENS-extra RES3", h1), ("H2 CNX sống (proxy)", h2)):
            print(f"{ds:8} {name:20} logOR person {h['logOR_person']:+.2f} [{h['ci95'][0]:+.2f}, {h['ci95'][1]:+.2f}]  "
                  f"(thô: person {100 * h['raw_rate_person']:.1f}% vs khác {100 * h['raw_rate_other']:.1f}%, n={h['n_obj']})")
    rep = res["devC-α1"]["H1_ens_extra_RES3"]["ci95"][0] > 0
    prx = all(res[d]["H2_cnx_alive_whitebox"]["ci95"][0] > 0 for d in ("dev100", "devC-α1"))
    res["verdict"] = {"H1_replicates_devC": bool(rep), "H2_proxy_exists": bool(prx),
                      "go_semantic": bool(rep and prx)}
    print(f"\nH1 lặp lại devC (α=1): {rep} | H2 proxy CNX (dev100 & devC): {prx} -> "
          f"{'GO hướng ngữ nghĩa' if rep and prx else 'NO-GO hướng ngữ nghĩa'}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
