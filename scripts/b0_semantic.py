"""B0 (khám phá, CPU, KHÔNG gate): object RES3 mà ENS-WP giết nhưng RD-CNX@50 không giết có pattern NGỮ NGHĨA không?
Dets F1b dev100 đã lưu; matching E77 (score ≥ 0.3, IoU ≥ 0.5 cùng lớp). Đối chứng seed-null: object @50s2 giết mà @50 không.
Báo: (1) kiểu thất bại của các kill thêm (cls / loc / suppressed); (2) với cls-fail: lớp nhầm cùng supercategory hay khác;
(3) làm giàu theo supercategory (tỉ lệ kill-thêm trong supercat / toàn bộ), CI bootstrap ảnh 1000 seed 2026.
  python scripts/b0_semantic.py -> results/mechanism/b0_semantic_dev100.json"""
import gzip
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.mechanism import _iou_xywh
from scripts.e77_dfc import fate, group
from scripts.f1b_r1a import ANN, ART, GROUPS

RUN = "f1b_r1a_eps5_3090"
OUT = os.path.join(REPO_ROOT, "results/mechanism/b0_semantic_dev100.json")
PAIRS = {"ENS-WP vs CNX@50": ("ENS-WP", "RD-CNX@50"), "seed-null 50s2 vs 50": ("RD-CNX@50s2", "RD-CNX@50"),
         "RD-R50 vs CNX@50": ("RD-R50", "RD-CNX@50")}


def main():
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    n = len(ids)
    sup = {c["id"]: c["supercategory"] for c in coco.loadCats(coco.getCatIds())}
    rows = []
    for m in [*GROUPS["RES3"], "dino_swin_l"]:
        D = json.load(gzip.open(os.path.join(ART, RUN, "dets", m, "all.json.gz"), "rt"))
        G = {c: group(D[c]) for c in ("clean", "RD-CNX@50", "RD-CNX@50s2", "ENS-WP", "RD-R50")}
        for k, i in enumerate(ids):
            for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
                if fate(a["bbox"], a["category_id"], G["clean"].get(i, []))[0] != "survived":
                    continue
                r = {"img": k, "model": m, "sup": sup[a["category_id"]], "area": a["area"]}
                for c in ("RD-CNX@50", "RD-CNX@50s2", "ENS-WP", "RD-R50"):
                    di = G[c].get(i, [])
                    f, _ = fate(a["bbox"], a["category_id"], di)
                    r[c] = f
                    if f == "cls_fail":  # lớp nhầm: det IoU ≥ 0.5 điểm cao nhất
                        B = np.array([d["bbox"] for d in di]).reshape(-1, 4)
                        iou = _iou_xywh(np.array(a["bbox"]), B)
                        j = max((q for q in range(len(di)) if iou[q] >= 0.5), key=lambda q: di[q]["score"])
                        r[c + "|pred_sup_same"] = sup[di[j]["category_id"]] == r["sup"]
                rows.append(r)
    rng = np.random.default_rng(2026)
    B = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(1000)]
    res = {"pairs": {}}
    for tgt in ("RES3", "dino_swin_l"):
        R = [r for r in rows if (r["model"] in GROUPS["RES3"]) == (tgt == "RES3") and (tgt == "RES3" or r["model"] == tgt)]
        img = np.array([r["img"] for r in R]); by = [np.where(img == k)[0] for k in range(n)]
        sups = sorted({r["sup"] for r in R})
        for name, (a, b) in PAIRS.items():
            extra = np.array([r[a] != "survived" and r[b] == "survived" for r in R])
            ftype = {t: int(sum(1 for r, e in zip(R, extra) if e and r[a] == t)) for t in ("cls_fail", "loc_fail", "suppressed")}
            same = [r[a + "|pred_sup_same"] for r, e in zip(R, extra) if e and r[a] == "cls_fail"]
            base = extra.mean()
            enr = {}
            for s in sups:
                ms = np.array([r["sup"] == s for r in R])
                if ms.sum() < 30:
                    continue
                v = []
                for bb in B:
                    ix = np.concatenate([by[q] for q in bb])
                    e, mm = extra[ix], ms[ix]
                    v.append(e[mm].mean() - e.mean() if mm.any() else np.nan)
                v = np.array(v); lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
                enr[s] = {"n": int(ms.sum()), "rate": float(extra[ms].mean()), "diff_vs_all": [float(v[0]), float(lo), float(hi)]}
            key = f"{tgt} | {name}"
            res["pairs"][key] = {"n_obj": len(R), "extra_rate": float(base), "fail_type": ftype,
                                 "cls_fail_same_supercat": [int(sum(same)), len(same)], "enrichment": enr}
            print(f"\n{key}: n={len(R)} kill-thêm {100 * base:.1f}%  kiểu {ftype}  cls-fail cùng supercat {sum(same)}/{len(same)}")
            for s, e in sorted(enr.items(), key=lambda x: -x[1]["diff_vs_all"][0]):
                d = e["diff_vs_all"]
                flag = " *" if d[1] > 0 or d[2] < 0 else ""
                print(f"   {s:12} n={e['n']:4d} rate {100 * e['rate']:5.1f}%  Δ {100 * d[0]:+5.1f} [{100 * d[1]:+.1f},{100 * d[2]:+.1f}]{flag}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[b0] -> {OUT}")


if __name__ == "__main__":
    main()
