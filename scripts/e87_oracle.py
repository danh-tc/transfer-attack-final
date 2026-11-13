"""E87-oracle — CHẨN ĐOÁN HẬU KỲ (CPU, chỉ đọc dets E87; user duyệt 2026-11-13). KHÔNG dùng để chọn/đổi recipe.
Câu hỏi: trên tập perturbation ĐÃ CÓ của E87, chọn α theo từng ảnh (biết trước outcome target = oracle) hơn fixed α=0
bao nhiêu? -> có đáng nghiên cứu routing trên trục C2 không. Đây là *oracle selection headroom* (luôn ≥ 0 do chọn max
trên chính outcome đánh giá), KHÔNG phải headroom của router surrogate-only, KHÔNG phải upper bound của mọi method.

Outcome theo ảnh (định nghĩa khóa, docs/mechanism_plan.md; attack/mechanism.per_image_transfer): trên GT được target
detect ở ảnh SẠCH (IoU ≥ 0.5, đúng lớp, score ≥ 0.3): suppression = tỉ lệ GT đó mất trên ảnh adv; retained = TB
score_adv/score_clean (phụ, liên tục; oracle dùng confidence drop = 1 − retained để "cao = attack mạnh"). Mẫu số giống nhau cho mọi α (cùng dets sạch). Target T5C.
- PRIMARY: image-level CNX oracle-of-2 {α0, α0.5}: mỗi ảnh 1 α chung cho mọi target, tối đa hóa MACRO suppression
  (TB qua các target T5C có ≥ 1 GT sạch được detect). Headroom H = TB_ảnh[max] − TB_ảnh[α0]. Hòa -> α0.
- Phụ: MICRO (trọng số = số GT sạch được detect từng target), oracle-of-3 {α0, α0.5, α1}.
- Exploratory: per-target oracle (mỗi target α riêng); R50 oracle-of-2 {α0, α1} (đối chứng âm tham chiếu, KHÔNG phải
  null đã hiệu chuẩn; hiệu CNX − R50 chỉ là contrast exploratory, không gọi là headroom đã hiệu chỉnh).
- Phân tầng (định nghĩa khóa trước khi xem): m_i = TB qua T5C số GT sạch được detect; nhóm [0,3) | [3,6) | [6,∞).
- Thống kê: paired bootstrap theo ảnh 1000 (seed 2026), MỖI replicate tính lại toàn bộ statistic (gồm quy tắc oracle).
  python scripts/e87_oracle.py -> results/mechanism/e87_oracle_devB100.json"""
import gzip
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.mechanism import DETECT_SCORE, _matched_scores, per_image_transfer

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
DETS = os.path.join(REPO_ROOT, "artifacts/runs/e87_c2w_B36_eps5_v100sxm216gb/dets")
T5C = ["fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
CELLS = ["CNX-A0", "CNX-A05", "CNX-A1", "R50-A0", "R50-A1"]
NB, SEED = 1000, 2026
BINS = [(0, 3), (3, 6), (6, np.inf)]


def load(m, c):
    with gzip.open(os.path.join(DETS, m, f"{c}.json.gz"), "rt") as f:
        return json.load(f)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/devB100.csv"))
    n = len(ids)
    S, R, ND = {}, {}, {}                        # [target][cell] -> [n] suppression / retained; ND[target] -> [n]
    for t in T5C:
        clean = load(t, "clean")
        ND[t] = np.array([int((v >= DETECT_SCORE).sum()) for v in (lambda d: [d[i] for i in ids])(
            _matched_scores(coco, clean, ids))])
        for c in CELLS:
            o = per_image_transfer(coco, clean, load(t, c), ids)
            S[(t, c)], R[(t, c)] = o["suppression"], 1.0 - o["retained"]   # R = confidence drop (cao = attack mạnh)
    nd = np.stack([ND[t] for t in T5C])                       # [T, n]
    valid = (nd > 0).any(0)
    m_i = nd.mean(0)

    def per_img(metric, c, micro):
        X = np.stack([metric[(t, c)] for t in T5C])           # [T, n], NaN nếu target không có GT sạch
        if micro:
            w = nd.astype(float)
            return np.nansum(X * w, 0) / np.where(w.sum(0) > 0, w.sum(0), np.nan)
        return np.nanmean(np.where(nd > 0, X, np.nan), 0)

    def stats(metric, micro, cand, base, idx):
        """headroom image-level oracle trên ảnh idx (tính lại từ đầu), tỉ lệ chọn ≠ base."""
        V = np.stack([per_img(metric, c, micro)[idx] for c in cand])        # [K, n']
        b = V[cand.index(base)]
        best = V.max(0)
        choose_other = (best > b + 1e-12)
        return float(best.mean() - b.mean()), float(choose_other.mean()), best - b

    rng = np.random.default_rng(SEED)
    base_idx = np.where(valid)[0]
    boots = [base_idx[rng.integers(0, len(base_idx), len(base_idx))] for _ in range(NB)]

    def report(metric, micro, cand, base, idx_all=base_idx):
        h, f, g = stats(metric, micro, cand, base, idx_all)
        hb = np.array([stats(metric, micro, cand, base, b)[0] for b in boots])
        fixed = {c: float(np.nanmean(per_img(metric, c, micro)[idx_all])) for c in cand}
        return {"headroom": h, "ci95": [float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))],
                "frac_choose_not_base": f, "fixed_means": fixed,
                "gain_quantiles_p50_p75_p90_max": [float(np.percentile(g, q)) for q in (50, 75, 90, 100)],
                "frac_gain_zero": float((g <= 1e-12).mean())}

    res = {"n_images": n, "n_valid": int(valid.sum()), "targets": T5C, "n_boot": NB, "boot_seed": SEED,
           "zero_denominator_per_target": {t: int((ND[t] == 0).sum()) for t in T5C},
           "note": "CHẨN ĐOÁN HẬU KỲ; oracle selection headroom (target-informed), không phải router khả thi"}
    A2, A3, R2 = ["CNX-A0", "CNX-A05"], ["CNX-A0", "CNX-A05", "CNX-A1"], ["R50-A0", "R50-A1"]
    res["PRIMARY_cnx_oracle2_macro_sup"] = report(S, False, A2, "CNX-A0")
    res["cnx_oracle2_micro_sup"] = report(S, True, A2, "CNX-A0")
    res["cnx_oracle3_macro_sup"] = report(S, False, A3, "CNX-A0")
    res["r50_oracle2_macro_sup (ref)"] = report(S, False, R2, "R50-A0")
    res["cnx_oracle2_macro_confdrop (phụ, liên tục)"] = report(R, False, A2, "CNX-A0")
    res["r50_oracle2_macro_confdrop (ref)"] = report(R, False, R2, "R50-A0")
    # contrast exploratory CNX − R50 (cùng replicate)
    hc = np.array([stats(S, False, A2, "CNX-A0", b)[0] for b in boots])
    hr = np.array([stats(S, False, R2, "R50-A0", b)[0] for b in boots])
    p = res["PRIMARY_cnx_oracle2_macro_sup"]["headroom"] - res["r50_oracle2_macro_sup (ref)"]["headroom"]
    res["contrast_cnx_minus_r50 (exploratory)"] = {"point": p, "ci95": [float(np.percentile(hc - hr, 2.5)),
                                                                      float(np.percentile(hc - hr, 97.5))]}
    # per-target oracle (exploratory) — mỗi target chọn α riêng
    res["per_target_oracle2_sup (exploratory)"] = {}
    for t in T5C:
        ok = np.where(ND[t] > 0)[0]
        a0, a5 = S[(t, "CNX-A0")][ok], S[(t, "CNX-A05")][ok]
        rb = np.random.default_rng(SEED)
        hb = [float(np.maximum(a0, a5)[j].mean() - a0[j].mean()) for j in
              (rb.integers(0, len(ok), len(ok)) for _ in range(NB))]
        res["per_target_oracle2_sup (exploratory)"][t] = {
            "headroom": float(np.maximum(a0, a5).mean() - a0.mean()),
            "ci95": [float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))],
            "frac_choose_a05": float((a5 > a0).mean()), "n": int(len(ok)),
            "fixed_a0": float(a0.mean()), "fixed_a05": float(a5.mean())}
    # phân tầng theo m_i
    res["strata_primary"] = {}
    for lo, hi in BINS:
        idx = np.where(valid & (m_i >= lo) & (m_i < hi))[0]
        if len(idx) == 0:
            continue
        h, f, _ = stats(S, False, A2, "CNX-A0", idx)
        hr_, fr_, _ = stats(S, False, R2, "R50-A0", idx)
        res["strata_primary"][f"[{lo},{hi})"] = {"n": int(len(idx)), "cnx_headroom": h, "cnx_frac_a05": f,
                                                 "r50_headroom_ref": hr_, "r50_frac_a1": fr_}
    out = os.path.join(REPO_ROOT, "results/mechanism/e87_oracle_devB100.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    f = lambda d: (f"H {100 * d['headroom']:+.2f} pt [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}]  "
                   f"chọn≠base {100 * d['frac_choose_not_base']:.0f}%  fixed " +
                   " ".join(f"{k} {100 * v:.1f}" for k, v in d["fixed_means"].items()) +
                   f"  gain p50/p75/p90/max " + "/".join(f"{100 * q:.1f}" for q in d["gain_quantiles_p50_p75_p90_max"]))
    print(f"=== E87 oracle (chẩn đoán hậu kỳ) devB100, n_valid {res['n_valid']}/{n}; mẫu số 0: {res['zero_denominator_per_target']}")
    for k, v in res.items():
        if isinstance(v, dict) and "headroom" in v and "frac_choose_not_base" in v:
            print(f"{k:45} {f(v)}")
    c = res["contrast_cnx_minus_r50 (exploratory)"]
    print(f"{'contrast CNX − R50 (exploratory)':45} {100 * c['point']:+.2f} [{100 * c['ci95'][0]:+.2f}, {100 * c['ci95'][1]:+.2f}]")
    for t, d in res["per_target_oracle2_sup (exploratory)"].items():
        print(f"  per-target {t:12} H {100 * d['headroom']:+.2f} [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}] "
              f"chọn α0.5 {100 * d['frac_choose_a05']:.0f}%  fixed α0 {100 * d['fixed_a0']:.1f} α0.5 {100 * d['fixed_a05']:.1f} (n {d['n']})")
    for k, d in res["strata_primary"].items():
        print(f"  tầng m∈{k:9} n {d['n']:3}  CNX H {100 * d['cnx_headroom']:+.2f} (chọn α0.5 {100 * d['cnx_frac_a05']:.0f}%)  "
              f"R50 ref H {100 * d['r50_headroom_ref']:+.2f} (chọn α1 {100 * d['r50_frac_a1']:.0f}%)")
    print(f"[e87-oracle] -> {out}")


if __name__ == "__main__":
    main()
