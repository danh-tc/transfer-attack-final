"""S10 (progress_log 2026-11-17, KHOA): luoi 2x2 TARGET backbone x head -> tach vai tro + kiem TUONG TAC o DINO-Swin-L.
Probe = delta CNX (RD-ALL chinh; FEAT/DET phu) — khac ho voi ca 4 target. dev30. A_MASK + kill.
Luoi: RoIxR50 = r50 | RoIxSwin = swin_t (va mrcnn_swin_s cho size) | DINOxR50 = dino_r50 | DINOxSwin-L = dino_swin_l.
3 o (dino_r50, swin_t, dino_swin_l) lay tu S8 json; S10 chi tinh target MOI r50, mrcnn_swin_s.
Tuong tac: predicted(DINOxSwin) = RoIxR50 + [RoIxSwin - RoIxR50] + [DINOxR50 - RoIxR50]; interaction = actual - predicted.
  python scripts/s10_grid.py [--n 30]
"""
import argparse
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
from scripts.e77_dfc import fate, group
from scripts.s3_region import ANN, IMGDIR
from scripts.s8_detfeat import ADV, GDIR

CNX = ["RD-ALL", "RD-FEAT", "RD-DET"]
NEW = ["r50"]                           # target moi (RoIxR50); 3 o kia tu S8. swin_s bo (khong co trong registry gradient)
TS = (0.5, 1.0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); args = ap.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:args.n]
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    MK = {}
    for i in ids:
        h, w = clean[i].shape[:2]; m = np.zeros((h, w), bool)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
            m |= coco.annToMask(a).astype(bool)[:h, :w]
        MK[i] = m
    D = {c: {i: (mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]).transpose(2, 0, 1)
             for i in ids} for c in CNX}
    os.makedirs(GDIR, exist_ok=True)
    res = {"n_img": len(ids), "A": {}, "kill": {}}
    from mmdet.apis import inference_detector
    for tgt in NEW:
        gm = load_any(tgt); ds = {s["img_id"]: s for s in target_dataset(tgt, ids)}
        num = {c: {} for c in CNX}; den = {c: {} for c in CNX}
        for n, i in enumerate(ids):
            orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
            m = np.broadcast_to(MK[i][None], D[CNX[0]][i].shape)
            for c in CNX:
                d = torch.from_numpy(D[c][i].astype(np.float32)).cuda(); sd = np.sign(D[c][i]); nn = dd = 0.0
                for t in TS:
                    p = os.path.join(GDIR, f"{tgt}_{i}_{c}_t{t}.npy")
                    if os.path.exists(p):
                        g = np.load(p).astype(np.float32)
                    else:
                        torch.manual_seed(1000 * (n + 1))
                        g = target_grads(gm, tgt, torch.clamp(orig + t * d, 0, 255), dsm, objectives=("cls",))[0]["cls"].float().cpu().numpy()
                        np.save(p + ".tmp.npy", g.astype(np.float16)); os.replace(p + ".tmp.npy", p)
                    nn += (sd * g)[m].sum(); dd += np.abs(g)[m].sum()
                num[c][i] = float(nn); den[c][i] = float(dd)
            print(f"[s10 A {tgt}] {n+1}/{len(ids)}", flush=True)
        del gm; torch.cuda.empty_cache()
        res["A"][tgt] = {c: sum(num[c].values()) / sum(den[c].values()) for c in CNX}
        tm = load(tgt); res["kill"][tgt] = {}
        with torch.no_grad():
            tp = {}
            for i in ids:
                g = group(to_coco_dets(inference_detector(tm, clean[i].astype(np.uint8)), i, cat)).get(i, [])
                tp[i] = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
                         if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values())
            for c in CNX:
                k = 0
                for i in ids:
                    gg = group(to_coco_dets(inference_detector(tm, mmcv.imread(os.path.join(ADV, c, f"{i}.png"))), i, cat)).get(i, [])
                    k += sum(fate(x["bbox"], x["category_id"], gg)[0] != "survived" for x in tp[i])
                res["kill"][tgt][c] = 100 * k / max(ntp, 1)
        del tm; torch.cuda.empty_cache()

    s8 = json.load(open(os.path.join(REPO_ROOT, "results/mechanism/s8_detfeat_dev30.json")))
    for tgt in ("dino_r50", "swin_t", "dino_swin_l"):
        res["A"][tgt] = {c: s8["A"][tgt][c] for c in CNX}; res["kill"][tgt] = {c: s8["kill"][tgt][c] for c in CNX}

    GRID = {"RoIxR50": "r50", "RoIxSwinT": "swin_t", "DINOxR50": "dino_r50", "DINOxSwinL": "dino_swin_l"}
    print(f"\nLuoi 2x2 (probe RD-CNX-ALL), A_MASK (kill%):")
    for g, t in GRID.items():
        print(f"  {g:10} {res['A'][t]['RD-ALL']:.4f} ({res['kill'][t]['RD-ALL']:.0f})")
    res["interaction"] = {}
    for met, lab in (("kill", "kill%"), ("A", "A_MASK")):
        def v(t):
            return res[met][t]["RD-ALL"] if met == "kill" else res["A"][t]["RD-ALL"]
        base = v("r50"); e_bb = v("swin_t") - base; e_head = v("dino_r50") - base
        pred = base + e_bb + e_head; inter = v("dino_swin_l") - pred
        res["interaction"][met] = {"RoIxR50": v("r50"), "backbone_eff": e_bb, "head_eff": e_head,
                                   "predicted_DINOxSwinL": pred, "actual": v("dino_swin_l"), "interaction": inter}
        print(f"\n[{lab}] RoIxR50 {v('r50'):.3f} | backbone eff {e_bb:+.3f} | head eff {e_head:+.3f}")
        print(f"   predicted DINOxSwinL (cong) {pred:.3f} vs actual {v('dino_swin_l'):.3f} -> tuong tac {inter:+.3f}")
    ik = res["interaction"]["kill"]["interaction"]
    res["verdict"] = ("TUONG TAC AM manh (combo kho hon tong) -> stuck = backbone x head" if ik <= -10 else
                      "khong co tuong tac ro (combo ~ cong)" if abs(ik) < 10 else "tuong tac DUONG")
    print(f"\n=> {res['verdict']}")
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/s10_grid_dev30.json"), "w"),
              indent=1, ensure_ascii=False, default=float)
    print("[s10] -> results/mechanism/s10_grid_dev30.json")


if __name__ == "__main__":
    main()
