"""E92a — optimization mismatch mức object (CPU-only, chỉ đọc E90b: RD-CNX B50 đóng băng, devC300[:60]).
Quy tắc khóa: docs/progress_log.md entry "E92a" 2026-11-14.

Quan sát = (object, target, khoảng) với khoảng ∈ {10→25, 25→50}, target ∈ {yolox_l, dino_swin_l}.
Object: GT không-crowd, CNX detect sạch ≥ 0.3 (như E90b), target detect sạch ≥ 0.3, VÀ target CÒN detect ở đầu khoảng
(s_tgt(t1) ≥ 0.3: "target-active" — còn phải phá; tránh đọc hiệu ứng trần thành mismatch).
  ΔS = z(t1) − z(t2)       z = logit p_y của RoI head CNX trên box GT (classification logit surrogate)
  ΔT = ℓ(t1) − ℓ(t2)       ℓ = logit(max(score detection matched của target, 0.01))  (target chỉ có score)
  Cùng chiều: dương = attack tiến bộ. stall = ΔT ≤ 0. M = P(ΔS > 0 ∧ ΔT ≤ 0).
Feature CHỈ surrogate (khóa): z(t1), ΔS, logit score detection CNX tại t1, logit score CNX sạch, log √area.
Kèm (không phải surrogate-only, chỉ báo): headroom target ℓ(t1) − logit(0.3), ℓ sạch target.
Bootstrap theo ẢNH 1000, seed 2026.
  python scripts/e92a_mismatch.py -> results/mechanism/e92a_mismatch_devC60.json
"""
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
SRC = os.path.join(REPO_ROOT, "results/mechanism/e90b_traj_devC20.json")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e92a_mismatch_devC60.json")
TGT = ["yolox_l", "dino_swin_l"]
INTV = [("10", "25"), ("25", "50")]
DET, FLOOR, SEED, NB = 0.3, 0.01, 2026, 1000
# ngưỡng chẩn đoán (khóa trước)
STALL_MIN, STALL_LO, IMG_FRAC = 0.25, 0.15, 0.5
AUC_MIN, AUC_LO, DAUC_MIN = 0.65, 0.55, 0.05
MIN_OBS, MIN_IMG = 100, 20
SUR_FEATS = ["z_t1", "dS", "ls_sur_t1", "ls_sur_0", "lsize"]


def lg(p):
    p = np.clip(np.asarray(p, float), FLOOR, 0.999)
    return np.log(p / (1 - p))


def fit_logit(X, y, l2=1e-2, it=60):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, X.T @ (y - p) - l2 * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    # xử lý hòa: rank trung bình
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    R = json.load(open(SRC))
    coco = COCO(ANN)
    rows = []
    for img_s, x in R["raw"].items():
        i = int(img_s)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
        z, ss = x["z"], x["s_sur"]
        for tg in TGT:
            st = x["s_tgt"][tg]
            for j in range(len(anns)):
                if ss["0"][j] < DET or st["0"][j] < DET:
                    continue
                for t1, t2 in INTV:
                    if st[t1][j] < DET:
                        continue
                    rows.append(dict(img=i, tg=tg, iv=f"{t1}-{t2}",
                                     dS=z[t1][j] - z[t2][j], dT=float(lg(st[t1][j]) - lg(st[t2][j])),
                                     z_t1=z[t1][j], ls_sur_t1=float(lg(ss[t1][j])), ls_sur_0=float(lg(ss["0"][j])),
                                     lsize=float(np.log(np.sqrt(max(anns[j]["area"], 1.0)))),
                                     head_t1=float(lg(st[t1][j]) - lg(DET)), lt0=float(lg(st["0"][j]))))
    D = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    n = len(rows)
    imgs = sorted(set(D["img"].tolist()))
    idx_of = {i: np.where(D["img"] == i)[0] for i in imgs}
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(imgs, len(imgs)) for _ in range(NB)]
    stall = (D["dT"] <= 0).astype(float)
    sprog = D["dS"] > 0

    def boot_stat(fn, sel=None):
        sel = np.ones(n, bool) if sel is None else sel
        pt = fn(np.where(sel)[0])
        bs = []
        for b in boots:
            ii = np.concatenate([idx_of[k] for k in b])
            ii = ii[sel[ii]]
            if len(ii):
                bs.append(fn(ii))
        return [round(float(pt), 4), ci(bs)]

    res = {"note": "E92a CPU, dữ liệu E90b (RD-CNX B50, devC300[:60]); chẩn đoán", "n_obs": n, "n_images": len(imgs),
           "descr": {}}
    # ---- mô tả theo khoảng × target ----
    for tg in TGT:
        for t1, t2 in INTV:
            sel = (D["tg"] == tg) & (D["iv"] == f"{t1}-{t2}")
            res["descr"][f"{tg}|{t1}-{t2}"] = {
                "n": int(sel.sum()), "n_img": int(len(set(D["img"][sel].tolist()))),
                "mean_dS": boot_stat(lambda ii: D["dS"][ii].mean(), sel),
                "mean_dT": boot_stat(lambda ii: D["dT"][ii].mean(), sel),
                "M": boot_stat(lambda ii: (sprog[ii] & (stall[ii] == 1)).mean(), sel),
                "frac_dS_pos": round(float(sprog[sel].mean()), 4),
                "stall_given_dSpos": boot_stat(lambda ii: stall[ii][sprog[ii]].mean() if sprog[ii].any() else np.nan, sel)}
    # ---- G1: có hệ thống ----
    g1_stat = boot_stat(lambda ii: stall[ii][sprog[ii]].mean() if sprog[ii].any() else np.nan)
    per_img = []
    for i in imgs:
        ii = idx_of[i]
        if sprog[ii].any():
            per_img.append(bool(stall[ii][sprog[ii]].any()))
    img_frac = float(np.mean(per_img))
    G1 = g1_stat[0] >= STALL_MIN and g1_stat[1][0] > STALL_LO and img_frac >= IMG_FRAC
    # ---- G2: còn sau kiểm soát ΔS + khoảng: stall trong tertile ΔS cao nhất của từng (target, khoảng) ----
    top = np.zeros(n, bool)
    for tg in TGT:
        for t1, t2 in INTV:
            sel = (D["tg"] == tg) & (D["iv"] == f"{t1}-{t2}")
            if sel.sum():
                top |= sel & (D["dS"] >= np.percentile(D["dS"][sel], 200 / 3))
    g2_stat = boot_stat(lambda ii: stall[ii].mean(), top)
    G2 = g2_stat[0] >= STALL_MIN and g2_stat[1][0] > STALL_LO
    # ---- G3: dự báo stall CHỈ bằng feature surrogate (leave-one-image-out) ----
    def design(feats, ii, mu, sd):
        cols = [np.ones(len(ii)), (D["iv"][ii] == "25-50").astype(float), (D["tg"][ii] == "dino_swin_l").astype(float)]
        cols += [(D[f][ii] - mu[f]) / sd[f] for f in feats]
        return np.stack(cols, 1)

    def loio(feats):
        pred = np.zeros(n)
        for i in imgs:
            te = idx_of[i]
            tr = np.where(D["img"] != i)[0]
            mu = {f: D[f][tr].mean() for f in feats}
            sd = {f: D[f][tr].std() + 1e-9 for f in feats}
            b = fit_logit(design(feats, tr, mu, sd), stall[tr])
            pred[te] = design(feats, te, mu, sd) @ b
        return pred

    p_base, p_sur = loio([]), loio(SUR_FEATS)
    p_full = loio(SUR_FEATS + ["head_t1", "lt0"])
    a_base = boot_stat(lambda ii: auc(stall[ii], p_base[ii]))
    a_sur = boot_stat(lambda ii: auc(stall[ii], p_sur[ii]))
    a_diff = boot_stat(lambda ii: auc(stall[ii], p_sur[ii]) - auc(stall[ii], p_base[ii]))
    a_full = boot_stat(lambda ii: auc(stall[ii], p_full[ii]))
    G3 = a_sur[0] >= AUC_MIN and a_sur[1][0] > AUC_LO and a_diff[0] >= DAUC_MIN
    # mô tả: tương quan đơn biến feature surrogate với stall (point-biserial)
    uni = {f: round(float(np.corrcoef(D[f], stall)[0, 1]), 3) for f in SUR_FEATS + ["head_t1", "lt0"]}

    enough = n >= MIN_OBS and len(imgs) >= MIN_IMG
    verdict = "INCONCLUSIVE" if not enough else "GO" if (G1 and G2 and G3) else "NO-GO"
    res.update({"G1_systematic": {"stall_given_dSpos": g1_stat, "img_frac_with_stall": round(img_frac, 3), "pass": bool(G1)},
                "G2_after_control_topTertile_dS": {"stall": g2_stat, "n": int(top.sum()), "pass": bool(G2)},
                "G3_surrogate_only": {"auc_base_interval_target": a_base, "auc_surrogate": a_sur,
                                      "dauc_sur_minus_base": a_diff, "pass": bool(G3)},
                "descr_auc_with_target_headroom": a_full, "descr_univariate_corr_with_stall": uni,
                "enough": bool(enough), "verdict": verdict})
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"n_obs {n}  images {len(imgs)}")
    for k, v in res["descr"].items():
        print(f"  {k:22} n {v['n']:3} ({v['n_img']} ảnh)  ΔS {v['mean_dS'][0]:+.2f} {v['mean_dS'][1]}  "
              f"ΔT {v['mean_dT'][0]:+.2f} {v['mean_dT'][1]}  P(ΔS>0) {v['frac_dS_pos']:.2f}  "
              f"stall|ΔS>0 {v['stall_given_dSpos'][0]:.2f} {v['stall_given_dSpos'][1]}  M {v['M'][0]:.2f}")
    for k in ("G1_systematic", "G2_after_control_topTertile_dS", "G3_surrogate_only"):
        print(f"{k}: {json.dumps(res[k], ensure_ascii=False)}")
    print(f"descr AUC (+ headroom target, KHÔNG surrogate-only): {a_full}\nunivariate corr: {uni}")
    print(f"\nverdict {verdict}\n[e92a] -> {OUT}")


if __name__ == "__main__":
    main()
