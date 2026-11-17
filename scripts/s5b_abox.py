"""S5-bổ sung (progress_log 2026-11-17, khóa trước khi có số S5): biến thể phụ A_BOX vs A_MASK trên new20 (dev30 \\ dev10).
A_BOX = A trên hợp box GT (MASK ∪ BOXBG), TB t ∈ {0.5, 1}, g_t cache S3/S5. Harm như S5 (forward lại DINO-Swin-L).
  python scripts/s5b_abox.py
"""
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from scripts.e64_cnx_drop import load
from scripts.s3_region import ADV, ANN, GDIR, IMGDIR, regions
from scripts.s5_dev30 import ATT, BOOT_SEED, NBOOT, TS, match_score, within

OUT = os.path.join(REPO_ROOT, "results/mechanism/s5b_abox_dev30.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")); new = ids[10:]
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in new}
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in new}
    A = {}
    for i in new:
        R = regions(coco, i, *clean[i].shape[:2])
        for c in ATT:
            s = np.sign(mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]).transpose(2, 0, 1)
            for r in ("MASK", "BOX"):
                m = np.broadcast_to(R[r][None], s.shape); v = []
                for t in TS:
                    g = np.load(os.path.join(GDIR, f"{i}_{c}_t{t}.npy")).astype(np.float32)
                    v.append(float((s * g)[m].sum() / np.abs(g)[m].sum()))
                A[(r, c, i)] = float(np.mean(v))
    from mmdet.apis import inference_detector
    model = load("dino_swin_l"); harm = {}
    with torch.no_grad():
        for i in new:
            d = to_coco_dets(inference_detector(model, clean[i].astype(np.uint8)), i, cat)
            sc = [(a, s_) for a in gts[i] for s_ in [match_score(a["bbox"], a["category_id"], d)] if s_ >= 0.3]
            for c in ATT:
                d = to_coco_dets(inference_detector(model, mmcv.imread(os.path.join(ADV, c, f"{i}.png"))), i, cat)
                r = [match_score(a["bbox"], a["category_id"], d) / s_ for a, s_ in sc]
                harm[(c, i)] = 1 - float(np.mean(r)) if r else np.nan
    sub = [i for i in new if np.isfinite(harm[(ATT[0], i)])]
    rng = np.random.default_rng(BOOT_SEED); res = {"n_img": len(sub)}
    W = {}
    for r in ("MASK", "BOX"):
        a = {(c, i): A[(r, c, i)] for c in ATT for i in sub}
        W[r] = within(a, harm, sub)
        bs = [np.nanmean(W[r][rng.integers(0, len(sub), len(sub))]) for _ in range(NBOOT)]
        ac = [np.mean([a[(c, i)] for i in sub]) for c in ATT]; hc = [np.nanmean([harm[(c, i)] for i in sub]) for c in ATT]
        res[r] = {"rho_cell_harm": spearmanr(ac, hc).correlation, "rho_within_mean": float(np.nanmean(W[r])),
                  "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "A_cell": dict(zip(ATT, ac))}
        print(f"A_{r:4}: ρ ô vs harm {res[r]['rho_cell_harm']:+.2f} | ρ trong ảnh {res[r]['rho_within_mean']:+.2f} "
              f"[{res[r]['ci95'][0]:+.2f}, {res[r]['ci95'][1]:+.2f}]")
    dd = []
    for _ in range(NBOOT):
        k = rng.integers(0, len(sub), len(sub)); dd.append(np.nanmean(W["MASK"][k]) - np.nanmean(W["BOX"][k]))
    pt = float(np.nanmean(W["MASK"]) - np.nanmean(W["BOX"])); ci = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
    res["diff_mask_minus_box"] = {"point": pt, "ci95": ci}
    res["verdict"] = "MASK > BOX" if ci[0] > 0 else "BOX > MASK" if ci[1] < 0 else "không phân biệt"
    print(f"MASK − BOX (ρ trong ảnh): {pt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] -> {res['verdict']}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
