"""S0-CPU (muc tieu B, user duyet 2026-11-13) — chan doan MO TA tren artifact E88 @ dev100.
Chi doc dets/npz da luu; KHONG dung devC/devD; KHONG doi method. Cau hoi: co failure pattern nao cac recipe hien
tai CHUA giai quyet, du lon de bien minh method moi?

(a) Headroom AP theo target: oracle tren AP tong hop theo target (KHONG per-image). Moi replicate bootstrap (tu
    e88 _samples.npz, paired) lay max theo tung target roi TB tren T7, tru ENS-WP.
(b) Phan tang that bai theo object cua ENS-WP (va RD-CNX doi chieu). Object = GT khong-crowd duoc target detect tren
    anh SACH (IoU>=0.5, dung lop, score>=0.3); "pha" = mat tren anh adv (score<0.3) — dinh nghia mechanism_plan.
    Tang: kich thuoc COCO (S<32^2, M<96^2, L), score sach, so GT/anh, supercategory. Nhom target: T7, WEAK3.
(c) Overlap object: trong object ENS-WP BO SOT (survivor), ti le RD-CNX / o khac pha duoc, so voi ti le pha vo dieu
    kien. Tham chieu "churn ngau nhien" = cap gan tuong duong OSFD-PUB/OSFD-REF.
Bootstrap theo anh (seed 2026): 1000 cho (a) tu npz; 500 cho ti le object.
  python scripts/s0_failure.py -> results/mechanism/s0_failure_dev100.json
"""
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.mechanism import DETECT_SCORE, _matched_scores

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
DETS = os.path.join(REPO_ROOT, "artifacts/runs/e88_ensbase_eps5_v100sxm216gb/dets")
NPZ = os.path.join(REPO_ROOT, "results/mechanism/e88_ensbase_dev100_samples.npz")
CELLS = ["OSFD-PUB", "OSFD-REF", "RD-CNX", "ENS-WP", "ENS-OSFD-AVG", "ENS-OSFD-GN", "ENS-RD-AVG", "ENS-RD-GN"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
WEAK3 = ["yolov3_d53", "yolox_l", "dino_swin_l"]
SEED, NB_OBJ = 2026, 500


def load(m, c):
    with gzip.open(os.path.join(DETS, m, f"{c}.json.gz"), "rt") as f:
        return json.load(f)


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def size_cat(area):
    return "S" if area < 32 ** 2 else "M" if area < 96 ** 2 else "L"


def score_bin(s):
    return "[.3,.5)" if s < 0.5 else "[.5,.8)" if s < 0.8 else "[.8,1]"


def ngt_bin(n):
    return "1-2" if n <= 2 else "3-7" if n <= 7 else ">=8"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    cat_super = {c["id"]: c["supercategory"] for c in coco.loadCats(coco.getCatIds())}
    res = {"note": "S0-CPU mo ta tren E88 dev100; khong lam bang chung, khong doi method", "n_images": len(ids)}

    # ---------- (a) headroom AP tu npz ----------
    S = dict(np.load(NPZ))
    tt = lambda cells: np.mean([np.max([S[f"{t}|{c}"] for c in cells], 0) for t in T7], 0)
    ens = np.mean([S[f"{t}|ENS-WP"] for t in T7], 0)
    res["a_headroom_T7"] = {}
    for name, cells in (("max(ENS-WP,RD-CNX)", ["ENS-WP", "RD-CNX"]), ("max_8_cells", CELLS)):
        d = tt(cells) - ens
        res["a_headroom_T7"][name] = {"oracle_T7": round(float(tt(cells)[0]), 2),
                                      "minus_ENS-WP": round(float(d[0]), 2), "ci95": ci(d[1:])}

    # ---------- object records ----------
    # per (target): clean matched scores + per-cell adv matched scores, theo tung ann (align voi loadAnns)
    clean_s = {t: _matched_scores(coco, load(t, "clean"), ids) for t in T7}
    adv_s = {(t, c): _matched_scores(coco, load(t, c), ids) for t in T7 for c in CELLS}
    anns = {i: [a for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))] for i in ids}

    # obj[t] = list theo anh: moi phan tu = dict cac mang bool/meta cho cac GT detected-clean cua target t
    obj = {t: [] for t in T7}
    for t in T7:
        for i in ids:
            cs = clean_s[t][i]
            keep = cs >= DETECT_SCORE
            idx = np.where(keep)[0]
            rec = {"supp": {c: (adv_s[(t, c)][i][idx] < DETECT_SCORE) for c in CELLS},
                   "size": np.array([size_cat(anns[i][j]["area"]) for j in idx]),
                   "sbin": np.array([score_bin(cs[j]) for j in idx]),
                   "ngt": ngt_bin(int(keep.sum())),
                   "sup": np.array([cat_super[anns[i][j]["category_id"]] for j in idx])}
            obj[t].append(rec)

    def gather(t, img_idx):
        r = {k: [] for k in ("size", "sbin", "sup")}
        supp = {c: [] for c in CELLS}
        ngt = []
        for k in img_idx:
            rec = obj[t][k]
            n = len(rec["size"])
            if n == 0:
                continue
            for c in CELLS:
                supp[c].append(rec["supp"][c])
            for key in ("size", "sbin", "sup"):
                r[key].append(rec[key])
            ngt.append(np.full(n, rec["ngt"]))
        cat = lambda L: np.concatenate(L) if L else np.array([])
        return {"supp": {c: cat(supp[c]) for c in CELLS}, "ngt": cat(ngt),
                **{k: cat(v) for k, v in r.items()}}

    full = {t: gather(t, range(len(ids))) for t in T7}
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(NB_OBJ)]

    def rate_on(t, mask_fn, cell):
        """ti le supp cua cell tren object thoa mask_fn(data), pooled; + CI bootstrap anh."""
        def r(idx):
            d = gather(t, idx)
            if len(d["supp"][cell]) == 0:
                return np.nan
            m = mask_fn(d)
            return float(d["supp"][cell][m].mean()) if m.any() else np.nan
        pt = r(range(len(ids)))
        bs = np.array([r(b) for b in boots])
        return pt, ci(bs[~np.isnan(bs)])

    # ---------- (b) phan tang that bai ENS-WP (tren WEAK3 gop) ----------
    def pooled_weak(mask_fn, cell):
        # gop object cua WEAK3; bootstrap theo anh (cung idx cho ca 3 target)
        def r(idx):
            num = den = 0
            for t in WEAK3:
                d = gather(t, idx)
                if len(d["supp"][cell]) == 0:
                    continue
                m = mask_fn(d)
                num += d["supp"][cell][m].sum(); den += m.sum()
            return float(num / den) if den else np.nan
        pt = r(range(len(ids)))
        bs = np.array([r(b) for b in boots])
        return round(pt, 4), ci(bs[~np.isnan(bs)])

    res["b_ENS-WP_suppress_rate_WEAK3"] = {"overall": pooled_weak(lambda d: np.ones(len(d["ngt"]), bool), "ENS-WP")}
    for key, vals in (("size", ["S", "M", "L"]), ("sbin", ["[.3,.5)", "[.5,.8)", "[.8,1]"]),
                      ("ngt", ["1-2", "3-7", ">=8"])):
        for v in vals:
            res["b_ENS-WP_suppress_rate_WEAK3"][f"{key}={v}"] = pooled_weak(
                lambda d, key=key, v=v: d[key] == v, "ENS-WP")
    # supercategory: top theo so object
    allsup = np.concatenate([full[t]["sup"] for t in WEAK3])
    top = [s for s, _ in sorted(((s, (allsup == s).sum()) for s in set(allsup)), key=lambda x: -x[1])[:6]]
    for s in top:
        res["b_ENS-WP_suppress_rate_WEAK3"][f"super={s}"] = pooled_weak(lambda d, s=s: d["sup"] == s, "ENS-WP")
    # RD-CNX doi chieu cung tang size
    res["b_RD-CNX_suppress_rate_WEAK3"] = {"overall": pooled_weak(lambda d: np.ones(len(d["ngt"]), bool), "RD-CNX")}
    for v in ["S", "M", "L"]:
        res["b_RD-CNX_suppress_rate_WEAK3"][f"size={v}"] = pooled_weak(lambda d, v=v: d["size"] == v, "RD-CNX")

    # ---------- (c) overlap: survivor cua ENS-WP ----------
    def cond_rate(t_list, surv_cell, test_cell):
        """P(test pha | surv_cell KHONG pha) va P(test pha) vo dieu kien, pooled tren t_list, CI bootstrap anh."""
        def r(idx, cond):
            num = den = 0
            for t in t_list:
                d = gather(t, idx)
                if len(d["supp"][surv_cell]) == 0:
                    continue
                base = ~d["supp"][surv_cell] if cond else np.ones(len(d["supp"][surv_cell]), bool)
                num += d["supp"][test_cell][base].sum(); den += base.sum()
            return float(num / den) if den else np.nan
        pc, pu = r(range(len(ids)), True), r(range(len(ids)), False)
        bc = np.array([r(b, True) for b in boots])
        return {"P_test_given_surv_survived": round(pc, 4), "ci95": ci(bc[~np.isnan(bc)]),
                "P_test_uncond": round(pu, 4)}

    res["c_overlap_WEAK3"] = {
        "RD-CNX_on_ENS-WP_survivors": cond_rate(WEAK3, "ENS-WP", "RD-CNX"),
        "ref_churn_OSFD-REF_on_OSFD-PUB_survivors": cond_rate(WEAK3, "OSFD-PUB", "OSFD-REF"),
        "ENS-WP_on_RD-CNX_survivors": cond_rate(WEAK3, "RD-CNX", "ENS-WP"),
    }
    res["c_overlap_dino_swin_l"] = {
        "RD-CNX_on_ENS-WP_survivors": cond_rate(["dino_swin_l"], "ENS-WP", "RD-CNX"),
        "ref_churn_OSFD-REF_on_OSFD-PUB_survivors": cond_rate(["dino_swin_l"], "OSFD-PUB", "OSFD-REF"),
    }
    # union target-informed (khong kha thi) de dinh co: P(it nhat 1 trong {ENS-WP,RD-CNX} pha)
    def union_rate(t_list, cells):
        def r(idx):
            num = den = 0
            for t in t_list:
                d = gather(t, idx)
                if len(d["supp"][cells[0]]) == 0:
                    continue
                u = np.any([d["supp"][c] for c in cells], 0)
                num += u.sum(); den += len(u)
            return float(num / den) if den else np.nan
        pt = r(range(len(ids)))
        return round(pt, 4), ci(np.array([r(b) for b in boots]))
    res["c_union_informed_WEAK3"] = {
        "ENS-WP_only": pooled_weak(lambda d: np.ones(len(d["ngt"]), bool), "ENS-WP"),
        "RD-CNX_only": pooled_weak(lambda d: np.ones(len(d["ngt"]), bool), "RD-CNX"),
        "union_ENS-WP_or_RD-CNX": union_rate(WEAK3, ["ENS-WP", "RD-CNX"])}

    out = os.path.join(REPO_ROOT, "results/mechanism/s0_failure_dev100.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    f2 = lambda x: f"{100 * x[0]:.1f} [{100 * x[1][0]:.1f},{100 * x[1][1]:.1f}]"
    print("\n=== (a) headroom AP T7 (oracle theo target) ===")
    for k, v in res["a_headroom_T7"].items():
        print(f"  {k:22} oracle {v['oracle_T7']:.1f}  -ENS-WP {v['minus_ENS-WP']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]")
    print("\n=== (b) ENS-WP suppress rate tren WEAK3 (yolov3,yolox_l,swin_l) ===")
    for k, v in res["b_ENS-WP_suppress_rate_WEAK3"].items():
        print(f"  {k:16} {f2(v)}")
    print("  --- RD-CNX doi chieu ---")
    for k, v in res["b_RD-CNX_suppress_rate_WEAK3"].items():
        print(f"  {k:16} {f2(v)}")
    print("\n=== (c) overlap tren WEAK3 ===")
    for k, v in res["c_overlap_WEAK3"].items():
        print(f"  {k:42} P(pha|sot) {100*v['P_test_given_surv_survived']:.1f} [{100*v['ci95'][0]:.1f},{100*v['ci95'][1]:.1f}]  (vo dk {100*v['P_test_uncond']:.1f})")
    print("  --- dino_swin_l ---")
    for k, v in res["c_overlap_dino_swin_l"].items():
        print(f"  {k:42} P(pha|sot) {100*v['P_test_given_surv_survived']:.1f} [{100*v['ci95'][0]:.1f},{100*v['ci95'][1]:.1f}]  (vo dk {100*v['P_test_uncond']:.1f})")
    print("  --- union informed WEAK3 ---")
    for k, v in res["c_union_informed_WEAK3"].items():
        print(f"  {k:26} {f2(v)}")
    print(f"\n[s0] -> {out}")


if __name__ == "__main__":
    main()
