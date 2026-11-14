"""S0c — S0 controlled reanalysis (CPU, user duyet 2026-11-14). Chi doc dets E88 @ dev100 (da dung o S0; KHONG dung
anh moi, KHONG devC/devD, KHONG doi method).

Cau hoi: object confidence cao / kich thuoc lon song sot nhieu hon (S0 b) co con la RESIDUAL limitation sau khi kiem
soat dong thoi (conf, size, detectability cheo model) khong; va attack co "danh yeu" cac object do (day logit it hon)
hay chi la hieu ung nguong (bien do can vuot lon hon)?

Object = GT khong-crowd duoc target t detect tren anh SACH (IoU>=0.5, dung lop, score>=0.3) — nhu S0/mechanism_plan.
Covariate (moi object x target):
  conf = logit(s_clean_t)            (clip [0.01, 0.999])
  size = log sqrt(area)
  det  = TB score sach matched cua object tren 6 target T7 CON LAI (leave-one-out; 0 neu khong match)
  + fixed effect target.
Outcome:
  Y1 surv  = s_adv_t >= 0.3
  Y2 drop  = logit(s_clean) - logit(max(s_adv, FLOOR)), FLOOR = 0.01 (floor luu chung lon nhat: yolox_l 0.01)
Effect = thay doi du doan khi covariate X di tu P10 -> P90 (quantile toan mau, co dinh), cac bien khac giu gia tri
quan sat: Y1 = AME logistic (diem %), Y2 = beta_OLS * (P90-P10) (don vi logit).
Bootstrap cum theo anh, 500, seed 2026, cung chi so cho moi cell (paired).
  python scripts/s0c_controlled.py -> results/mechanism/s0c_controlled_dev100.json
"""
import gzip
import json
import os
import sys

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
CELLS = ["ENS-WP", "RD-CNX", "OSFD-PUB"]  # 2 primary + 1 objective khac (tham chieu "chung moi objective")
PRIMARY = ["ENS-WP", "RD-CNX"]
COVS = ["conf", "size", "det"]
FLOOR, SEED, NB = 0.01, 2026, 500
# quy tac khoa (progress_log 2026-11-14 S0c)
SURV_MIN = 0.10  # AME P10->P90 tren surv >= +10 diem %


def load(m, c):
    with gzip.open(os.path.join(DETS, m, f"{c}.json.gz"), "rt") as f:
        return json.load(f)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def design(d, tg, mu, sd):
    """[1, conf, size, det (chuan hoa), dummy target (bo target dau)]"""
    X = [np.ones(len(tg))] + [(d[k] - mu[k]) / sd[k] for k in COVS]
    for t in sorted(set(TG_ALL))[1:]:
        X.append((tg == t).astype(float))
    return np.stack(X, 1)


def fit_logit(X, y, l2=1e-4, it=50):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - p) - l2 * b
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def fit_ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main():
    global TG_ALL
    sys.stdout.reconfigure(encoding="utf-8")
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    clean = {t: _matched_scores(coco, load(t, "clean"), ids) for t in T7}
    adv = {(t, c): _matched_scores(coco, load(t, c), ids) for t in WEAK3 for c in CELLS}
    areas = {i: np.array([a["area"] for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))]) for i in ids}

    rows = {k: [] for k in ("img", "tg", "sc", "conf", "size", "det", *[f"sa|{c}" for c in CELLS])}
    for t in WEAK3:
        others = [u for u in T7 if u != t]
        for k, i in enumerate(ids):
            cs = clean[t][i]
            idx = np.where(cs >= DETECT_SCORE)[0]
            if len(idx) == 0:
                continue
            det = np.mean([clean[u][i][idx] for u in others], 0)
            rows["img"] += [k] * len(idx); rows["tg"] += [t] * len(idx)
            rows["sc"] += list(cs[idx]); rows["det"] += list(det)
            rows["conf"] += list(logit(np.clip(cs[idx], 0.01, 0.999)))
            rows["size"] += list(np.log(np.sqrt(np.maximum(areas[i][idx], 1.0))))
            for c in CELLS:
                rows[f"sa|{c}"] += list(adv[(t, c)][i][idx])
    D = {k: np.array(v) for k, v in rows.items()}
    TG_ALL = list(D["tg"])
    n = len(D["img"])
    mu = {k: D[k].mean() for k in COVS}
    sd = {k: D[k].std() for k in COVS}
    q = {k: (np.percentile(D[k], 10), np.percentile(D[k], 90)) for k in COVS}
    Y1 = {c: (D[f"sa|{c}"] >= DETECT_SCORE).astype(float) for c in CELLS}
    Y2 = {c: logit(np.clip(D["sc"], 0.01, 0.999)) - logit(np.maximum(D[f"sa|{c}"], FLOOR)) for c in CELLS}

    full = effects_idx(np.arange(n), D, mu, sd, q, Y1, Y2)
    rng = np.random.default_rng(SEED)
    img_rows = [np.where(D["img"] == k)[0] for k in range(len(ids))]
    boots = []
    for _ in range(NB):
        b = rng.integers(0, len(ids), len(ids))
        sel = np.concatenate([img_rows[k] for k in b])
        boots.append(effects_idx(sel, D, mu, sd, q, Y1, Y2))
    res = {"note": "S0c controlled reanalysis, E88 dev100, WEAK3 gop; mo ta + quy tac GO E90 khoa truoc",
           "n_objects": int(n), "n_by_target": {t: int((D["tg"] == t).sum()) for t in WEAK3},
           "quantiles_P10_P90": {k: [round(float(v), 3) for v in q[k]] for k in COVS},
           "corr": {f"{a}~{b}": round(float(np.corrcoef(D[a], D[b])[0, 1]), 3)
                    for a, b in (("conf", "size"), ("conf", "det"), ("size", "det"))},
           "effects": {}}
    for c in CELLS:
        res["effects"][c] = {key: {"point": round(full[c][key], 4), "ci95": ci([bb[c][key] for bb in boots])}
                             for key in full[c]}
    # paired: primary - OSFD-PUB (mo ta)
    res["contrast_vs_OSFD-PUB"] = {
        c: {key: {"point": round(full[c][key] - full["OSFD-PUB"][key], 4),
                  "ci95": ci([bb[c][key] - bb["OSFD-PUB"][key] for bb in boots])} for key in full[c]}
        for c in PRIMARY}

    # bang phan tang score bin x size (mo ta)
    sb = np.digitize(D["sc"], [0.5, 0.8])
    sz = np.digitize(D["size"], [np.log(32), np.log(96)])
    lab_s, lab_z = ["[.3,.5)", "[.5,.8)", "[.8,1]"], ["S", "M", "L"]
    res["strata_surv"] = {}
    for c in CELLS:
        tab = {}
        for a in range(3):
            for z in range(3):
                m = (sb == a) & (sz == z)
                tab[f"{lab_s[a]}|{lab_z[z]}"] = [int(m.sum()), round(float(Y1[c][m].mean()), 4) if m.any() else None,
                                                 round(float(Y2[c][m].mean()), 3) if m.any() else None,
                                                 round(float((D[f"sa|{c}"][m] <= FLOOR).mean()), 3) if m.any() else None]
        res["strata_surv"][c] = tab
    res["strata_cols"] = "[n, surv_rate, mean_logit_drop, frac_at_floor]"

    # quy tac khoa
    verdict = {}
    for k in ("conf", "size"):
        ok_s = all(res["effects"][c][f"surv|{k}"]["point"] >= SURV_MIN and res["effects"][c][f"surv|{k}"]["ci95"][0] > 0
                   for c in PRIMARY)
        ok_d = all(res["effects"][c][f"drop|{k}"]["ci95"][1] < 0 for c in PRIMARY)
        thr = all(res["effects"][c][f"drop|{k}"]["ci95"][0] >= 0 for c in PRIMARY)
        verdict[k] = {"residual_survival": ok_s, "under_attacked": ok_d, "threshold_only": ok_s and thr,
                      "GO_E90": ok_s and ok_d}
    res["verdict"] = verdict
    res["GO_E90_any"] = any(v["GO_E90"] for v in verdict.values())

    out = os.path.join(REPO_ROOT, "results/mechanism/s0c_controlled_dev100.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"n_objects {n}  by target {res['n_by_target']}  corr {res['corr']}")
    print(f"P10/P90 {res['quantiles_P10_P90']}")
    for c in CELLS:
        print(f"\n== {c} (WEAK3 gop) ==")
        for key, v in res["effects"][c].items():
            sc = 100 if key.startswith("surv") else 1
            print(f"  {key:10} {sc * v['point']:+7.2f} [{sc * v['ci95'][0]:+.2f}, {sc * v['ci95'][1]:+.2f}]")
    for c in PRIMARY:
        print(f"\n-- {c} - OSFD-PUB --")
        for key, v in res["contrast_vs_OSFD-PUB"][c].items():
            sc = 100 if key.startswith("surv") else 1
            print(f"  {key:10} {sc * v['point']:+7.2f} [{sc * v['ci95'][0]:+.2f}, {sc * v['ci95'][1]:+.2f}]")
    print(f"\nstrata {res['strata_cols']}")
    for c in PRIMARY:
        print(f"  {c}: " + "  ".join(f"{k}:{v}" for k, v in res["strata_surv"][c].items()))
    print(f"\nverdict {json.dumps(verdict)}\nGO_E90_any {res['GO_E90_any']}\n[s0c] -> {out}")


def effects_idx(sel, D, mu, sd, q, Y1, Y2):
    d = {k: D[k][sel] for k in COVS}
    X = design(d, D["tg"][sel], mu, sd)
    out = {}
    for c in CELLS:
        bl = fit_logit(X, Y1[c][sel])
        bo = fit_ols(X, Y2[c][sel])
        e = {}
        for j, k in enumerate(COVS, start=1):
            lo, hi = X.copy(), X.copy()
            lo[:, j] = (q[k][0] - mu[k]) / sd[k]
            hi[:, j] = (q[k][1] - mu[k]) / sd[k]
            e[f"surv|{k}"] = float((1 / (1 + np.exp(-hi @ bl)) - 1 / (1 + np.exp(-lo @ bl))).mean())
            e[f"drop|{k}"] = float(bo[j] * (q[k][1] - q[k][0]) / sd[k])
        out[c] = e
    return out


if __name__ == "__main__":
    main()
