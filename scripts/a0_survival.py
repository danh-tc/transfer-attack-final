"""A0 (progress_log 2026-11-15): object-level survival RD-CNX B50 → B100 vs seed-null B50 ↔ B50s2. CPU, dets F1b đã lưu.
Matching cố định theo E77 (scripts/e77_dfc.py): det score ≥ 0.3; object at-risk = GT (non-crowd) mà dets SẠCH có box IoU ≥ 0.5
cùng lớp; object "sống" dưới attack ⇔ vẫn có det IoU ≥ 0.5 cùng lớp. Bootstrap theo ẢNH (1000, seed 2026), không theo object.
Báo riêng với COCO AP; không suy object "giết thêm" là nguyên nhân của ΔAP. Mô tả, KHÔNG dùng để chọn method bằng thông tin target.
  python scripts/a0_survival.py   -> results/mechanism/a0_survival_dev100.json"""
import gzip
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from scripts.e77_dfc import fate, group, SCORE_THR
from scripts.f1b_r1a import ANN, ART, PANEL, GROUPS

RUN = "f1b_r1a_eps5_3090"
OUT = os.path.join(REPO_ROOT, "results/mechanism/a0_survival_dev100.json")
CELLS = ["RD-CNX@36", "RD-CNX@50", "RD-CNX@75", "RD-CNX@100", "RD-CNX@50s2"]
SIZE = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, 1e12)]
SCORE = [("s<0.7", 0.3, 0.7), ("0.7–0.9", 0.7, 0.9), ("≥0.9", 0.9, 1.01)]
N_BOOT, SEED = 1000, 2026


def objects(coco, ids, dets):
    """-> list dict per at-risk object: img index, area, clean score, alive[cell]."""
    cl = group(dets["clean"])
    adv = {c: group(dets[c]) for c in CELLS}
    out = []
    for k, i in enumerate(ids):
        for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
            f, sc = fate(a["bbox"], a["category_id"], cl.get(i, []))
            if f != "survived":
                continue
            o = {"img": k, "area": a["area"], "score": sc}
            for c in CELLS:
                o[c] = fate(a["bbox"], a["category_id"], adv[c].get(i, []))[0] == "survived"
            out.append(o)
    return out


def main():
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    n = len(ids)
    rng = np.random.default_rng(SEED)
    B = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(N_BOOT)]
    objs = {}
    for key in PANEL:
        dets = json.load(gzip.open(os.path.join(ART, RUN, "dets", key, "all.json.gz"), "rt"))
        objs[key] = objects(coco, ids, dets)

    def stat(olist, fn):
        """fn(mask arrays) -> scalar per resample; resample theo ảnh (mỗi ảnh lấy toàn bộ object của nó)."""
        img = np.array([o["img"] for o in olist])
        A = {c: np.array([o[c] for o in olist]) for c in CELLS}
        by = [np.where(img == k)[0] for k in range(n)]
        vals = []
        for b in B:
            idx = np.concatenate([by[k] for k in b]) if len(olist) else np.array([], int)
            vals.append(fn({c: A[c][idx] for c in CELLS}) if len(idx) else np.nan)
        v = np.array(vals)
        lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
        return {"point": float(v[0]), "ci95": [float(lo), float(hi)]}

    K = lambda c: (lambda A: 1 - A[c].mean())
    net = lambda a, b: (lambda A: A[a].mean() - A[b].mean())          # = K(b) − K(a): tỉ lệ object bị giết thêm khi a→b
    newk = lambda a, b: (lambda A: (A[a] & ~A[b]).mean())
    rev = lambda a, b: (lambda A: (~A[a] & A[b]).mean())
    res = {"score_thr": SCORE_THR, "matching": "E77", "n_images": n, "per_target": {}, "strata": {}}
    targets = list(GROUPS["RES3"]) + ["dino_swin_l", "swin_t", "yolov3_d53", "yolox_l", "convnext_t"]
    pooled = {"RES3": [dict(o, img=o["img"]) for m in GROUPS["RES3"] for o in objs[m]]}
    print(f"{'target':12} {'n_obj':>6} | K50   K100  | net50→100           | seed-null net        | churn100 churnSN")
    for t, ol in [(m, objs[m]) for m in targets] + list(pooled.items()):
        r = {"n_obj": len(ol), "K": {c: stat(ol, K(c)) for c in CELLS},
             "net": {f"{a}->{b}": stat(ol, net(a, b)) for a, b in
                     (("RD-CNX@36", "RD-CNX@50"), ("RD-CNX@50", "RD-CNX@75"), ("RD-CNX@75", "RD-CNX@100"),
                      ("RD-CNX@50", "RD-CNX@100"), ("RD-CNX@50", "RD-CNX@50s2"))},
             "new_kill_50_100": stat(ol, newk("RD-CNX@50", "RD-CNX@100")), "revived_50_100": stat(ol, rev("RD-CNX@50", "RD-CNX@100")),
             "new_kill_seed": stat(ol, newk("RD-CNX@50", "RD-CNX@50s2")), "revived_seed": stat(ol, rev("RD-CNX@50", "RD-CNX@50s2"))}
        res["per_target"][t] = r
        f = lambda s: f"{100 * s['point']:+5.1f} [{100 * s['ci95'][0]:+.1f},{100 * s['ci95'][1]:+.1f}]"
        ch = r["new_kill_50_100"]["point"] + r["revived_50_100"]["point"]
        chs = r["new_kill_seed"]["point"] + r["revived_seed"]["point"]
        print(f"{t:12} {len(ol):6d} | {100 * r['K']['RD-CNX@50']['point']:5.1f} {100 * r['K']['RD-CNX@100']['point']:5.1f} | "
              f"{f(r['net']['RD-CNX@50->RD-CNX@100']):19} | {f(r['net']['RD-CNX@50->RD-CNX@50s2']):20} | {100 * ch:6.1f} {100 * chs:6.1f}")
    # positive control (khóa): trên RES3 gộp, net 36→50 > net 75→100, net 50→100 CI > 0, seed-null net CI chứa 0
    p = res["per_target"]["RES3"]["net"]
    pc = (p["RD-CNX@36->RD-CNX@50"]["point"] > p["RD-CNX@75->RD-CNX@100"]["point"]
          and p["RD-CNX@50->RD-CNX@100"]["ci95"][0] > 0
          and p["RD-CNX@50->RD-CNX@50s2"]["ci95"][0] <= 0 <= p["RD-CNX@50->RD-CNX@50s2"]["ci95"][1])
    res["positive_control"] = {"pass": bool(pc), "segments": {k: v for k, v in p.items()}}
    print(f"\nPOSITIVE CONTROL (RES3 gộp): {'PASS' if pc else 'FAIL'}  segs " +
          "  ".join(f"{k.replace('RD-CNX@', '')} {100 * v['point']:+.1f}" for k, v in p.items()))
    # strata (mô tả): theo kích thước COCO và điểm sạch; share net gain vs share object; seed-null cùng strata
    for t in ("RES3", "dino_swin_l"):
        ol = pooled["RES3"] if t == "RES3" else objs[t]
        tot = sum(o["RD-CNX@50"] and not o["RD-CNX@100"] for o in ol) - sum(o["RD-CNX@100"] and not o["RD-CNX@50"] for o in ol)
        res["strata"][t] = {}
        for axis, bins, field in (("size", SIZE, "area"), ("clean_score", SCORE, "score")):
            for name, lo, hi in bins:
                sub = [o for o in ol if lo <= o[field] < hi]
                if not sub:
                    continue
                cnt = sum(o["RD-CNX@50"] and not o["RD-CNX@100"] for o in sub) - sum(o["RD-CNX@100"] and not o["RD-CNX@50"] for o in sub)
                s = {"n_obj": len(sub), "share_obj": len(sub) / len(ol), "share_net_gain": cnt / tot if tot else None,
                     "K50": stat(sub, K("RD-CNX@50")), "net_50_100": stat(sub, net("RD-CNX@50", "RD-CNX@100")),
                     "net_seed": stat(sub, net("RD-CNX@50", "RD-CNX@50s2"))}
                res["strata"][t][f"{axis}:{name}"] = s
                print(f"{t:11} {axis:11} {name:8} n={len(sub):4d} obj {100 * s['share_obj']:4.0f}% | gain "
                      f"{100 * (s['share_net_gain'] or 0):5.0f}% | K50 {100 * s['K50']['point']:5.1f} | net "
                      f"{100 * s['net_50_100']['point']:+5.1f} [{100 * s['net_50_100']['ci95'][0]:+.1f},{100 * s['net_50_100']['ci95'][1]:+.1f}]"
                      f" | seed {100 * s['net_seed']['point']:+5.1f} [{100 * s['net_seed']['ci95'][0]:+.1f},{100 * s['net_seed']['ci95'][1]:+.1f}]")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[a0] -> {OUT}")


if __name__ == "__main__":
    main()
