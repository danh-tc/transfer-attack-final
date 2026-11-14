"""E90a — consensus-robust: kiem tra CPU truoc moi GPU (user duyet 2026-11-14). Chi doc dets E88 @ dev100.
Quy tac khoa: docs/progress_log.md entry "E90a" 2026-11-14.

Consensus C cua object (GT khong-crowd duoc target t detect sach, score >= 0.3), tinh tren score SACH:
  C_LOO  = TB score matched tren 6 target T7 con lai (= det cua S0c)
  C_LFO  = TB tren target T7 KHONG chung ho backbone VA KHONG chung ho detector voi t (leave-family-out)
Kiem soat:
  S0C   = conf_t (logit), size, FE target                                  (nhu S0c)
  FULL  = S0C + conf_t^2 + dummy score-bin cua t + logit score sach tren 2 surrogate (r50, convnext_t)
          + log so GT/anh + FE supercategory
Outcome tren target (WEAK3 gop): surv = s_adv >= 0.3; drop = logit(s_clean) - logit(max(s_adv, 0.01)).
Effect C: P10 -> P90 (quantile toan mau, co dinh): AME logistic (diem %) / beta_OLS * (P90-P10) (logit).
MO TA (khong cong): dau hieu diem cuoi H1/H2 tren SURROGATE (white-box) — object surrogate detect sach; outcome
  surv/drop tren chinh surrogate, kiem soat logit score sach surrogate + size + FE; C = C_LFO cua object do
  (tinh tren 7 target T7, khong surrogate nao nam trong T7).
Bootstrap cum theo anh 500, seed 2026, paired giua cell; multiprocessing, 1 luong BLAS/tien trinh.
  OMP_NUM_THREADS=1 python scripts/e90a_consensus.py -> results/mechanism/e90a_consensus_dev100.json
"""
import gzip
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.mechanism import DETECT_SCORE, _matched_scores

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
DETS = os.path.join(REPO_ROOT, "artifacts/runs/e88_ensbase_eps5_v100sxm216gb/dets")
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
WEAK3 = ["yolov3_d53", "yolox_l", "dino_swin_l"]
SURS = ["r50", "convnext_t"]
CELLS = ["ENS-WP", "RD-CNX", "OSFD-PUB"]
PRIMARY = ["ENS-WP", "RD-CNX"]
SUR_OF = {"ENS-WP": ["r50", "convnext_t"], "RD-CNX": ["convnext_t"], "OSFD-PUB": ["r50"]}
# ho (model_registry.md): backbone / detector
BB = {"r50": "resnet", "r101": "resnet", "fcos_r50": "resnet", "dino_r50": "resnet", "yolov3_d53": "darknet",
      "yolox_l": "cspdarknet", "swin_t": "swin", "dino_swin_l": "swin", "convnext_t": "convnext"}
DH = {"r50": "mrcnn", "r101": "mrcnn", "fcos_r50": "fcos", "dino_r50": "dino", "yolov3_d53": "yolo",
      "yolox_l": "yolo", "swin_t": "mrcnn", "dino_swin_l": "dino", "convnext_t": "mrcnn"}
FLOOR, SEED = 0.01, 2026
NB = int(os.environ.get("E90A_NB", 500))  # smoke: E90A_NB=4, KHONG doc so
SURV_MIN = 0.10


def load(m, c):
    with gzip.open(os.path.join(DETS, m, f"{c}.json.gz"), "rt") as f:
        return json.load(f)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def lgc(p):
    return logit(np.clip(p, 0.01, 0.999))


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def fit_logit(X, y, l2=1e-3, it=60):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (y - p) - l2 * b
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def fit_ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def lfo_others(t, pool):
    return [u for u in pool if u != t and BB[u] != BB[t] and DH[u] != DH[t]]


# ---------------- du lieu (global cho worker fork) ----------------
G = {}


def build():
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    models = T7 + SURS
    clean = {m: _matched_scores(coco, load(m, "clean"), ids) for m in models}
    evalm = WEAK3 + SURS
    adv = {(m, c): _matched_scores(coco, load(m, c), ids) for m in evalm for c in CELLS}
    sup_of = {c["id"]: c["supercategory"] for c in coco.loadCats(coco.getCatIds())}
    anns = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    sups = sorted(set(sup_of.values()))

    def rows_for(t, pool_loo, pool_lfo):
        R = {k: [] for k in ("img", "tg", "sc", "size", "C_LOO", "C_LFO", "lngt", "sup", "s_r50", "s_cnx",
                             *[f"sa|{c}" for c in CELLS])}
        for k, i in enumerate(ids):
            cs = clean[t][i]
            idx = np.where(cs >= DETECT_SCORE)[0]
            if len(idx) == 0:
                continue
            R["img"] += [k] * len(idx); R["tg"] += [t] * len(idx); R["sc"] += list(cs[idx])
            R["size"] += [np.log(np.sqrt(max(anns[i][j]["area"], 1.0))) for j in idx]
            R["C_LOO"] += list(np.mean([clean[u][i][idx] for u in pool_loo], 0))
            R["C_LFO"] += list(np.mean([clean[u][i][idx] for u in pool_lfo], 0))
            R["lngt"] += [np.log(len(cs))] * len(idx)
            R["sup"] += [sup_of[anns[i][j]["category_id"]] for j in idx]
            R["s_r50"] += list(clean["r50"][i][idx]); R["s_cnx"] += list(clean["convnext_t"][i][idx])
            for c in CELLS:
                R[f"sa|{c}"] += list(adv[(t, c)][i][idx])
        return R

    def concat(Rs):
        return {k: np.array(sum((R[k] for R in Rs), [])) for k in Rs[0]}

    tgt = concat([rows_for(t, [u for u in T7 if u != t], lfo_others(t, T7)) for t in WEAK3])
    sur = {s: concat([rows_for(s, T7, lfo_others(s, T7))]) for s in SURS}
    G.update(ids=ids, tgt=tgt, sur=sur, sups=sups,
             lfo_pools={t: lfo_others(t, T7) for t in WEAK3 + SURS})


def design(D, sel, spec, cvar, z):
    """tra X (cot 1 = C chuan hoa) cho dong sel. z = dict chuan hoa (mu, sd) co dinh tu toan mau."""
    def zc(k):
        return (D[k][sel] - z[k][0]) / z[k][1]
    cols = [np.ones(len(sel)), zc(cvar), zc("conf"), zc("size")]
    for t in sorted(set(D["tg"])) [1:]:
        cols.append((D["tg"][sel] == t).astype(float))
    if spec == "FULL":
        cols += [zc("conf") ** 2, zc("lr50"), zc("lcnx"), zc("lngt")]
        sb = np.digitize(D["sc"][sel], [0.5, 0.8])
        cols += [(sb == 1).astype(float), (sb == 2).astype(float)]
        for s in G["sups"][1:]:
            cols.append((D["sup"][sel] == s).astype(float))
    X = np.stack(cols, 1)
    keep = X[:, 4:].std(0) > 0 if X.shape[1] > 4 else np.array([], bool)
    return np.concatenate([X[:, :4], X[:, 4:][:, keep]], 1)


def prep(D):
    D["conf"] = lgc(D["sc"]); D["lr50"] = lgc(D["s_r50"]); D["lcnx"] = lgc(D["s_cnx"])
    return D


def effect(D, sel, spec, cvar, ysurv, ydrop, z, q):
    X = design(D, sel, spec, cvar, z)
    bl = fit_logit(X, ysurv[sel]); bo = fit_ols(X, ydrop[sel])
    lo, hi = X.copy(), X.copy()
    lo[:, 1] = (q[0] - z[cvar][0]) / z[cvar][1]; hi[:, 1] = (q[1] - z[cvar][0]) / z[cvar][1]
    p = lambda A: 1 / (1 + np.exp(-np.clip(A @ bl, -30, 30)))
    return float((p(hi) - p(lo)).mean()), float(bo[1] * (q[1] - q[0]) / z[cvar][1])


TGT_SPECS = [("C_LFO", "FULL"), ("C_LOO", "FULL"), ("C_LFO", "S0C"), ("C_LOO", "S0C")]


def one_rep(b):
    """b = mang chi so anh (bootstrap) hoac None (diem)."""
    out = {}
    D = G["tgt"]
    rows = np.arange(len(D["img"])) if b is None else np.concatenate([G["img_rows"][k] for k in b])
    for cvar, spec in TGT_SPECS:
        for c in CELLS:
            s, d = effect(D, rows, spec, cvar, G["Y1"][c], G["Y2"][c], G["z"], G["q"][cvar])
            out[f"tgt|{cvar}|{spec}|{c}"] = (s, d)
    for s_name, Ds in G["sur"].items():
        rr = np.arange(len(Ds["img"])) if b is None else np.concatenate([G["sur_rows"][s_name][k] for k in b])
        for c in CELLS:
            if s_name not in SUR_OF[c]:
                continue
            sv, dv = effect(Ds, rr, "S0C_SUR", "C_LFO", G["SY1"][(s_name, c)], G["SY2"][(s_name, c)],
                            G["zs"][s_name], G["qs"][s_name])
            out[f"sur|{s_name}|{c}"] = (sv, dv)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    build()
    D = prep(G["tgt"])
    for s in SURS:
        prep(G["sur"][s])
    n_img = len(G["ids"])
    G["img_rows"] = [np.where(D["img"] == k)[0] for k in range(n_img)]
    G["sur_rows"] = {s: [np.where(G["sur"][s]["img"] == k)[0] for k in range(n_img)] for s in SURS}
    G["z"] = {k: (D[k].mean(), D[k].std()) for k in ("C_LOO", "C_LFO", "conf", "size", "lr50", "lcnx", "lngt")}
    G["q"] = {k: (np.percentile(D[k], 10), np.percentile(D[k], 90)) for k in ("C_LOO", "C_LFO")}
    G["Y1"] = {c: (D[f"sa|{c}"] >= DETECT_SCORE).astype(float) for c in CELLS}
    G["Y2"] = {c: D["conf"] - logit(np.maximum(D[f"sa|{c}"], FLOOR)) for c in CELLS}
    # surrogate side: conf = logit score sach cua chinh surrogate; FE target suy bien (1 target) -> design bo
    G["zs"], G["qs"], G["SY1"], G["SY2"] = {}, {}, {}, {}
    for s, Ds in G["sur"].items():
        G["zs"][s] = {k: (Ds[k].mean(), Ds[k].std() + 1e-12) for k in ("C_LFO", "conf", "size")}
        G["qs"][s] = (np.percentile(Ds["C_LFO"], 10), np.percentile(Ds["C_LFO"], 90))
        for c in CELLS:
            G["SY1"][(s, c)] = (Ds[f"sa|{c}"] >= DETECT_SCORE).astype(float)
            G["SY2"][(s, c)] = Ds["conf"] - logit(np.maximum(Ds[f"sa|{c}"], FLOOR))

    print(f"[e90a] n_obj target {len(D['img'])}, surrogate { {s: len(G['sur'][s]['img']) for s in SURS} }", flush=True)
    point = one_rep(None)
    rng = np.random.default_rng(SEED)
    bs_idx = [rng.integers(0, n_img, n_img) for _ in range(NB)]
    with Pool(min(32, os.cpu_count())) as pool:
        boots = pool.map(one_rep, bs_idx, chunksize=8)

    res = {"note": "E90a consensus-robust, E88 dev100 (da xem o S0/S0c); cong khoa truoc; surrogate side = mo ta",
           "lfo_pools": G["lfo_pools"], "n_obj_target": int(len(D["img"])),
           "n_obj_surrogate": {s: int(len(G["sur"][s]["img"])) for s in SURS},
           "corr_target": {f"{a}~{b}": round(float(np.corrcoef(D[a], D[b])[0, 1]), 3) for a, b in
                           (("C_LFO", "C_LOO"), ("C_LFO", "conf"), ("C_LFO", "lr50"), ("C_LFO", "lcnx"),
                            ("C_LFO", "size"), ("conf", "lcnx"))},
           "P10_P90": {k: [round(float(x), 3) for x in v] for k, v in G["q"].items()},
           "effects": {}}
    for key in point:
        s_pt, d_pt = point[key]
        res["effects"][key] = {"surv": [round(s_pt, 4), ci([b[key][0] for b in boots])],
                               "drop": [round(d_pt, 4), ci([b[key][1] for b in boots])]}
    for c in PRIMARY:
        key, ref = f"tgt|C_LFO|FULL|{c}", "tgt|C_LFO|FULL|OSFD-PUB"
        res["effects"][f"contrast|{c}-OSFD-PUB|C_LFO|FULL"] = {
            "surv": [round(point[key][0] - point[ref][0], 4), ci([b[key][0] - b[ref][0] for b in boots])],
            "drop": [round(point[key][1] - point[ref][1], 4), ci([b[key][1] - b[ref][1] for b in boots])]}

    E = res["effects"]
    ok_s = all(E[f"tgt|C_LFO|FULL|{c}"]["surv"][0] >= SURV_MIN and E[f"tgt|C_LFO|FULL|{c}"]["surv"][1][0] > 0
               for c in PRIMARY)
    ok_d = all(E[f"tgt|C_LFO|FULL|{c}"]["drop"][1][1] < 0 for c in PRIMARY)
    res["PASS_E90a"] = bool(ok_s and ok_d)
    res["gate"] = {"residual_survival": bool(ok_s), "under_attacked": bool(ok_d)}

    out = os.path.join(REPO_ROOT, "results/mechanism/e90a_consensus_dev100.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"LFO pools {G['lfo_pools']}\ncorr {res['corr_target']}\nP10/P90 {res['P10_P90']}")
    for key, v in E.items():
        print(f"  {key:38} surv {100 * v['surv'][0]:+6.1f} [{100 * v['surv'][1][0]:+.1f}, {100 * v['surv'][1][1]:+.1f}]"
              f"   drop {v['drop'][0]:+5.2f} [{v['drop'][1][0]:+.2f}, {v['drop'][1][1]:+.2f}]")
    print(f"\ngate {res['gate']}  PASS_E90a {res['PASS_E90a']}\n[e90a] -> {out}")


if __name__ == "__main__":
    main()
