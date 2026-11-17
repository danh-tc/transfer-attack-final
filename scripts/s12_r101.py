"""S12 (progress_log 2026-11-17, KHOA): RD recipe (loss manh) tren CNN THUONG TO HON (ResNet-101) -> dong hieu ung voi CNX?
Surrogate r101, dev30. Cells: R101-FEAT (feature C3-C5) | R101-DET (RoI cls only) | R101-ALL (feature+RoI, lam=6.0 nhu R50,
chua calib rieng — ghi ro). Cung OCT+MI+WP+B50+eps5+seed S1. So voi R50 (S8) va ConvNeXt (S8) tren thang target.
  python scripts/s12_r101.py
"""
import functools
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import scripts.s1_signmap as s1
from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import load_any, target_dataset, target_grads
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.run_baselines import _C2OCT
from scripts.s3_region import ANN, IMGDIR
from scripts.s8_detfeat import ADV, GDIR

WP = dict(weight_rho=0.002)
CELLS = {
    "R101-FEAT": functools.partial(geo_osfd_attack, **dict(_C2OCT, cls_weight=0.0, skip_mask=False, feat_stages=(1, 2, 3)), **WP),
    "R101-DET": functools.partial(geo_osfd_attack, **dict(_C2OCT, cls_weight=6.0), cls_mode="roi", feature=False, **WP),
    "R101-ALL": functools.partial(geo_osfd_attack, **dict(_C2OCT, cls_weight=6.0), cls_mode="roi", stage_w=(0.0, 1.0, 1.0, 1.0), **WP),
}
TARGETS = ["dino_r50", "swin_t", "dino_swin_l"]
TS = (0.5, 1.0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    MK = {}
    for i in ids:
        h, w = clean[i].shape[:2]; m = np.zeros((h, w), bool)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
            m |= coco.annToMask(a).astype(bool)[:h, :w]
        MK[i] = m
    # sinh δ R101
    todo = [c for c in CELLS if not all(os.path.exists(os.path.join(ADV, c, f"{i}.png")) for i in ids)]
    if todo:
        m = load_model("r101", "cuda:0"); ds = {s["img_id"]: s for s in target_dataset("r101", ids)}
        for c in todo:
            os.makedirs(os.path.join(ADV, c), exist_ok=True)
            for n, i in enumerate(ids):
                p = os.path.join(ADV, c, f"{i}.png")
                if os.path.exists(p):
                    continue
                orig = ds[i]["orig_inputs"].cuda().float(); s1.seed_all(1000 * (n + 1))
                d = CELLS[c](m, orig, ds[i]["data_sample"], steps=50, epsilon=5.0)
                a = to_adv_image(orig, d)
                mmcv.imwrite(a.permute(1, 2, 0).cpu().numpy().astype(np.uint8), p + ".tmp.png"); os.replace(p + ".tmp.png", p)
                print(f"[s12 gen {c}] {n+1}/30", flush=True)
        del m; torch.cuda.empty_cache()

    D = {c: {i: (mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]).transpose(2, 0, 1) for i in ids} for c in CELLS}
    res = {"n_img": len(ids), "A": {}, "kill": {}}
    from mmdet.apis import inference_detector
    for tgt in TARGETS:
        gm = load_any(tgt); ds = {s["img_id"]: s for s in target_dataset(tgt, ids)}
        num = {c: {} for c in CELLS}; den = {c: {} for c in CELLS}
        for n, i in enumerate(ids):
            orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
            m = np.broadcast_to(MK[i][None], D[list(CELLS)[0]][i].shape)
            for c in CELLS:
                dd = torch.from_numpy(D[c][i].astype(np.float32)).cuda(); sd = np.sign(D[c][i]); nn = de = 0.0
                for t in TS:
                    p = os.path.join(GDIR, f"{tgt}_{i}_{c}_t{t}.npy")
                    if os.path.exists(p):
                        g = np.load(p).astype(np.float32)
                    else:
                        torch.manual_seed(1000 * (n + 1))
                        g = target_grads(gm, tgt, torch.clamp(orig + t * dd, 0, 255), dsm, objectives=("cls",))[0]["cls"].float().cpu().numpy()
                        np.save(p + ".tmp.npy", g.astype(np.float16)); os.replace(p + ".tmp.npy", p)
                    nn += (sd * g)[m].sum(); de += np.abs(g)[m].sum()
                num[c][i] = float(nn); den[c][i] = float(de)
            print(f"[s12 A {tgt}] {n+1}/30", flush=True)
        del gm; torch.cuda.empty_cache()
        res["A"][tgt] = {c: sum(num[c].values()) / sum(den[c].values()) for c in CELLS}
        tm = load(tgt); res["kill"][tgt] = {}
        with torch.no_grad():
            tp = {}
            for i in ids:
                g = group(to_coco_dets(inference_detector(tm, clean[i].astype(np.uint8)), i, cat)).get(i, [])
                tp[i] = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values())
            for c in CELLS:
                k = 0
                for i in ids:
                    gg = group(to_coco_dets(inference_detector(tm, mmcv.imread(os.path.join(ADV, c, f"{i}.png"))), i, cat)).get(i, [])
                    k += sum(fate(x["bbox"], x["category_id"], gg)[0] != "survived" for x in tp[i])
                res["kill"][tgt][c] = 100 * k / max(ntp, 1)
        del tm; torch.cuda.empty_cache()

    print(f"\nR101 (CNN thuong, 43M) RD loss — A_MASK (kill%):")
    print(f"{'':10}" + "".join(f"{t[:11]:>16}" for t in TARGETS))
    for c in CELLS:
        print(f"{c:10}" + "".join(f"{res['A'][t][c]:9.4f}({res['kill'][t][c]:3.0f})" for t in TARGETS))
    # so sanh 3 backbone tren Swin-L (DET & ALL) tu S8 + S12
    s8 = json.load(open(os.path.join(REPO_ROOT, "results/mechanism/s8_detfeat_dev30.json")))
    print(f"\nTren DINO-Swin-L, kill% theo backbone (CNN thuong R50/R101 vs ConvNeXt):")
    print(f"  {'':8}{'FEAT':>7}{'DET(RoI)':>10}{'ALL':>7}")
    rows = {"R50": ("R50-FEAT", "R50-DET", "R50-ALL"), "R101": ("R101-FEAT", "R101-DET", "R101-ALL"),
            "ConvNeXt": ("RD-FEAT", "RD-DET", "RD-ALL")}
    getk = lambda c: res["kill"]["dino_swin_l"].get(c) if c in res["kill"]["dino_swin_l"] else s8["kill"]["dino_swin_l"][c]
    for b, (f, dd, a) in rows.items():
        print(f"  {b:8}{getk(f):7.0f}{getk(dd):10.0f}{getk(a):7.0f}")
    res["swinl_by_backbone"] = {b: {k: getk(c) for k, c in zip(("FEAT", "DET", "ALL"), cs)} for b, cs in rows.items()}
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/s12_r101_dev30.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print("[s12] -> results/mechanism/s12_r101_dev30.json")


if __name__ == "__main__":
    main()
