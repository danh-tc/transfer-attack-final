"""S13 (progress_log 2026-11-17, KHOA): panel MODERN held-out (eval-only) — delta DONG BANG transfer toi detector hien dai.
Khong sinh attack moi; chi inference delta co san (dev30) tren target moi. Target: RTMDet-L (CNN 1-stage), Deformable-DETR-R50,
DDQ-DETR-R50 (2023), Mask2Former-R50 (Transformer head). delta: RD-ALL (RD-CNX), OSFD-FEAT (OSFD baseline), R50-ALL (RD tren CNN thuong).
Do relative AP drop + kill object clean-TP. dev30, xu huong. Held-out: KHONG dung chon/tune gi.
  python scripts/s13_modern.py
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
from attack.evaluation import CocoImageEval, to_coco_dets
from scripts.e77_dfc import fate, group
from scripts.s3_region import ADV, ANN, IMGDIR

MIM = os.path.join(REPO_ROOT, ".venv/lib/python3.10/site-packages/mmdet/.mim/configs")
CKPT = os.path.join(REPO_ROOT, "checkpoints/modern")
TARGETS = {
    "rtmdet_l": (f"{MIM}/rtmdet/rtmdet_l_8xb32-300e_coco.py", f"{CKPT}/rtmdet_l.pth"),
    "deformable_detr": (f"{MIM}/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py", f"{CKPT}/deformable_detr.pth"),
    "ddq_detr": (f"{MIM}/ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py", f"{CKPT}/ddq.pth"),
    "mask2former": (f"{MIM}/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py", f"{CKPT}/mask2former.pth"),
}
DELTAS = ["RD-ALL", "OSFD-FEAT", "R50-ALL"]   # RD-CNX chinh, OSFD baseline, RD tren CNN thuong


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmdet.apis import inference_detector, init_detector
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])) for i in ids}
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    res = {"n_img": len(ids), "clean_ap": {}, "drop": {}, "kill": {}}
    for tname, (cfg, ck) in TARGETS.items():
        if not os.path.exists(ck):
            print(f"[s13] THIEU checkpoint {ck}, bo qua {tname}", flush=True); continue
        model = init_detector(cfg, ck, device="cuda:0")
        with torch.no_grad():
            cl, tp = [], {}
            for i in ids:
                det = to_coco_dets(inference_detector(model, clean[i]), i, cat); cl += det
                g = group(det).get(i, [])
                tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            apc = CocoImageEval(coco, cl, ids).ap(np.ones(len(ids)))[0]
            ntp = sum(len(v) for v in tp.values())
            res["clean_ap"][tname] = apc * 100
            res["drop"][tname], res["kill"][tname] = {}, {}
            for d in DELTAS:
                ad, k = [], 0
                for i in ids:
                    det = to_coco_dets(inference_detector(model, mmcv.imread(os.path.join(ADV, d, f"{i}.png"))), i, cat)
                    ad += det
                    g = group(det).get(i, [])
                    k += sum(fate(x["bbox"], x["category_id"], g)[0] != "survived" for x in tp[i])
                apa = CocoImageEval(coco, ad, ids).ap(np.ones(len(ids)))[0]
                res["drop"][tname][d] = 100 * (apc - apa) / max(apc, 1e-9)
                res["kill"][tname][d] = 100 * k / max(ntp, 1)
            print(f"[s13] {tname:16} clean AP {apc*100:.1f} (n_tp {ntp}) | " +
                  "  ".join(f"{d} drop {res['drop'][tname][d]:.0f}/kill {res['kill'][tname][d]:.0f}" for d in DELTAS), flush=True)
        del model; torch.cuda.empty_cache()

    print(f"\n=== PANEL MODERN held-out (dev30) — relative AP drop % (kill %) ===")
    print(f"{'target':16}{'cleanAP':>8}" + "".join(f"{d:>16}" for d in DELTAS))
    for t in res["drop"]:
        print(f"{t:16}{res['clean_ap'][t]:8.1f}" + "".join(f"{res['drop'][t][d]:8.0f}({res['kill'][t][d]:3.0f})" for d in DELTAS))
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/s13_modern_dev30.json"), "w"), indent=1, ensure_ascii=False)
    print("[s13] -> results/mechanism/s13_modern_dev30.json")


if __name__ == "__main__":
    main()
