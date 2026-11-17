"""S6 — hành vi vùng con + oracle trên dev30 (progress_log KHÓA S6, quy tắc 5–6). Cần PNG 6 attack + 5 WB dev30.
Biến thể (áp ĐỒNG THỜI cho mọi instance không-crowd của ảnh): full; keep_X (δ chỉ trong X); rnd_X (dấu ngẫu nhiên
trong X, giữ |δ|), X ∈ BANDS ∪ VERT. Oracle (chỉ chẩn đoán): O1-{hot,rand}-k, O2-{hot,rand}-k, k ∈ {5,10,25,50}% pixel-kênh
MASK (hot = top-k |g_t sạch| DINO-Swin-L). Outcome Swin-L: kill (score_adv < 0.3) + harm của object clean-TP.
  python scripts/s6_behav.py
"""
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from attack.targets import load_any, target_dataset, target_grads
from scripts.e64_cnx_drop import load
from scripts.s3_region import ADV, ANN, GDIR, IMGDIR
from scripts.s5_dev30 import match_score
from scripts.s6_common import BANDS, VERT, inst_subregions

ATT = ["RD-ALL", "RD-FEAT", "RD-DET", "OSFD-ALL", "OSFD-FEAT", "OSFD-DET"]
WBS = ["WB-0", "WB-r1", "WB-r2", "WB-r3", "WB-r4"]
CELLS = ATT + WBS
SUBS = BANDS + VERT
VARS = ["full"] + [f"keep_{x}" for x in SUBS] + [f"rnd_{x}" for x in SUBS]
KS = (5, 10, 25, 50)
ORACLE = [f"O{o}-{s}-{k}" for o in (1, 2) for s in ("hot", "rand") for k in KS]
OUT = os.path.join(REPO_ROOT, "results/mechanism/s6_behav_dev30.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    U = {}
    for i in ids:
        h, w = clean[i].shape[:2]
        u = {x: np.zeros((h, w), bool) for x in SUBS + ["MASK"]}
        for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
            R = inst_subregions(coco, a, h, w)
            if R is not None:
                for x in u:
                    u[x] |= R[x]
        U[i] = u
    # g_t sạch dev30 (cache chung với s6_gen --part grad; cùng seed)
    if not all(os.path.exists(os.path.join(GDIR, f"{i}_clean.npy")) for i in ids):
        model = load_any("dino_swin_l"); ds = {s["img_id"]: s for s in target_dataset("dino_swin_l", ids)}
        for n, i in enumerate(ids):
            p = os.path.join(GDIR, f"{i}_clean.npy")
            if not os.path.exists(p):
                torch.manual_seed(1000 * (n + 1))
                g = target_grads(model, "dino_swin_l", ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"],
                                 objectives=("cls",))[0]["cls"].float().cpu().numpy()
                np.save(p, g.astype(np.float16))
        del model; torch.cuda.empty_cache()
    D = {c: {i: mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i] for i in ids} for c in CELLS}
    SG = {i: np.random.default_rng(42 + i).choice([-1, 1], size=clean[i].shape) for i in ids}
    SEL = {}
    for i in ids:
        g = np.abs(np.load(os.path.join(GDIR, f"{i}_clean.npy")).astype(np.float32)).transpose(1, 2, 0)
        mk = np.broadcast_to(U[i]["MASK"][:, :, None], g.shape)
        idx = np.flatnonzero(mk); order = idx[np.argsort(-g.reshape(-1)[idx], kind="stable")]
        rp = np.random.default_rng(7 + i).permutation(idx)
        for k in KS:
            n = max(1, int(len(idx) * k / 100))
            SEL[(i, "hot", k)], SEL[(i, "rand", k)] = order[:n], rp[:n]

    def make(c, i, v):
        d = D[c][i]
        if v == "full":
            dd = d
        elif v.startswith("keep_"):
            dd = d * U[i][v[5:]][:, :, None]
        elif v.startswith("rnd_"):
            dd = np.where(U[i][v[4:]][:, :, None], np.abs(d) * SG[i], d)
        else:
            o, s, k = v.split("-"); sel = SEL[(i, s, int(k))]
            wb = D["WB-0"][i].reshape(-1)
            if o == "O1":
                dd = np.zeros(d.size, np.int32); dd[sel] = wb[sel]
            else:
                dd = D["RD-ALL"][i].reshape(-1).copy(); dd[sel] = np.abs(dd[sel]) * np.sign(wb[sel])
            dd = dd.reshape(d.shape)
        return np.clip(clean[i] + dd, 0, 255).astype(np.uint8)

    from mmdet.apis import inference_detector
    model = load("dino_swin_l")
    res = {"n_img": len(ids), "kill": {}, "harm": {}, "area_frac_of_mask": {}}
    with torch.no_grad():
        tp = {}
        for i in ids:
            dts = to_coco_dets(inference_detector(model, clean[i].astype(np.uint8)), i, cat)
            tp[i] = [(a, s) for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
                     for s in [match_score(a["bbox"], a["category_id"], dts)] if s >= 0.3]
        ntp = sum(len(v) for v in tp.values())
        jobs = [(c, v) for c in CELLS for v in VARS] + [("ORACLE", v) for v in ORACLE]
        for n, (c, v) in enumerate(jobs):
            k = 0; hs = []
            for i in ids:
                dts = to_coco_dets(inference_detector(model, make(c if c != "ORACLE" else "WB-0", i, v)), i, cat)
                for a, s in tp[i]:
                    sa = match_score(a["bbox"], a["category_id"], dts)
                    k += sa < 0.3; hs.append(1 - sa / s)
            res["kill"].setdefault(c, {})[v] = 100 * k / ntp
            res["harm"].setdefault(c, {})[v] = float(np.mean(hs))
            if n % 20 == 0:
                print(f"[s6 behav] {n + 1}/{len(jobs)}", flush=True)
    res["n_tp"] = ntp
    K = res["kill"]
    print(f"\nDINO-Swin-L kill % (dev30, {ntp} object clean-TP)")
    print(f"{'':10}" + "".join(f"{v[:10]:>11}" for v in VARS))
    for c in CELLS:
        print(f"{c:10}" + "".join(f"{K[c][v]:11.0f}" for v in VARS))
    print("\nOracle (kill / harm):")
    for v in ORACLE:
        print(f"  {v:14} {K['ORACLE'][v]:5.0f} / {res['harm']['ORACLE'][v]:.3f}")
    print(f"  mốc: WB-0 full {K['WB-0']['full']:.0f}, RD-ALL full {K['RD-ALL']['full']:.0f}")
    verdict = {}
    for name, P in (("BANDS", BANDS), ("VERT", VERT)):
        best = {c: max(P, key=lambda x: K[c][f"keep_{x}"]) for c in CELLS}
        cnt = {x: [sum(best[c] == x for c in WBS), sum(best[c] == x for c in ATT)] for x in P}
        top = max(P, key=lambda x: cnt[x][0] + cnt[x][1])
        verdict[name] = {"best_keep_count[wb,att]": cnt,
                         "verdict": (f"keep {top} mạnh nhất (vững)" if cnt[top][0] >= 4 and cnt[top][1] >= 5
                                     else "KHÔNG vững")}
        print(f"=> {name}: {verdict[name]}")
    res["verdict"] = verdict
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[s6 behav] -> {OUT}")


if __name__ == "__main__":
    main()
