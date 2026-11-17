"""S15 (progress_log 2026-11-17): OSFD GOC (baselines.osfd_attack, khong phai geo xap xi) tren ConvNeXt (fair same-surrogate
vs RD-CNX) + R50 (baseline bang dong bang). Sinh dev30 B50 eps5 seed S1, eval full panel (nhu S14), so voi RD-CNX (json S14).
  python scripts/s15_osfd_real.py
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

import scripts.s1_signmap as s1
from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.methods.baselines import osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.s3_region import ADV, ANN, IMGDIR
from scripts.s13_modern import TARGETS as MODERN
from scripts.s14_fullpanel import E64

CELLS = {"OSFDreal-CNX": "convnext_t", "OSFDreal-R50": "r50"}


def gen(ids):
    for cell, sur in CELLS.items():
        out = os.path.join(ADV, cell)
        if all(os.path.exists(os.path.join(out, f"{i}.png")) for i in ids):
            continue
        os.makedirs(out, exist_ok=True)
        m = load_model(sur, "cuda:0"); ds = {s["img_id"]: s for s in target_dataset(sur, ids)}
        for n, i in enumerate(ids):
            p = os.path.join(out, f"{i}.png")
            if os.path.exists(p):
                continue
            orig = ds[i]["orig_inputs"].cuda().float(); s1.seed_all(1000 * (n + 1))
            d = osfd_attack(m, orig, ds[i]["data_sample"], steps=50, epsilon=5.0)
            a = to_adv_image(orig, d)
            mmcv.imwrite(a.permute(1, 2, 0).cpu().numpy().astype(np.uint8), p + ".tmp.png"); os.replace(p + ".tmp.png", p)
            print(f"[s15 gen {cell}] {n+1}/{len(ids)}", flush=True)
        del m; torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmdet.apis import inference_detector, init_detector
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    gen(ids)
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])) for i in ids}
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    D = list(CELLS)

    def ev(model):
        with torch.no_grad():
            cl, tp = [], {}
            for i in ids:
                det = to_coco_dets(inference_detector(model, clean[i]), i, cat); cl += det
                g = group(det).get(i, [])
                tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            apc = CocoImageEval(coco, cl, ids).ap(np.ones(len(ids)))[0]; ntp = sum(len(v) for v in tp.values())
            o = {"drop": {}, "kill": {}}
            for d in D:
                ad, k = [], 0
                for i in ids:
                    det = to_coco_dets(inference_detector(model, mmcv.imread(os.path.join(ADV, d, f"{i}.png"))), i, cat); ad += det
                    g = group(det).get(i, [])
                    k += sum(fate(x["bbox"], x["category_id"], g)[0] != "survived" for x in tp[i])
                apa = CocoImageEval(coco, ad, ids).ap(np.ones(len(ids)))[0]
                o["drop"][d] = 100 * (apc - apa) / max(apc, 1e-9); o["kill"][d] = 100 * k / max(ntp, 1)
            return o
    res = {}
    for t in E64:
        m = load(t); res[t] = ev(m); del m; torch.cuda.empty_cache()
        print(f"[s15] {t:16} " + " ".join(f"{d} {res[t]['drop'][d]:.0f}" for d in D), flush=True)
    for t, (cfg, ck) in MODERN.items():
        if not os.path.exists(ck):
            continue
        m = init_detector(cfg, ck, device="cuda:0"); res[t] = ev(m); del m; torch.cuda.empty_cache()
        print(f"[s15] {t:16} " + " ".join(f"{d} {res[t]['drop'][d]:.0f}" for d in D), flush=True)

    s14 = json.load(open(os.path.join(REPO_ROOT, "results/mechanism/s14_fullpanel_dev30.json")))
    order = E64 + [t for t in MODERN if t in res]
    print(f"\n=== OSFD GOC vs RD-CNX (dev30, relative AP drop %) ===")
    print(f"{'target':18}{'RD-CNX':>8}{'OSFDreal-CNX':>14}{'OSFDreal-R50':>14}{'OSFD-style(S14)':>16}")
    for t in order:
        rd = s14[t]["drop"]["RD-ALL"]; ofs = s14[t]["drop"]["OSFD-FEAT"]
        print(f"{t:18}{rd:8.0f}{res[t]['drop']['OSFDreal-CNX']:14.0f}{res[t]['drop']['OSFDreal-R50']:14.0f}{ofs:16.0f}")
    rd_m = np.mean([s14[t]["drop"]["RD-ALL"] for t in order if t != "convnext_t"])
    oc_m = np.mean([res[t]["drop"]["OSFDreal-CNX"] for t in order if t != "convnext_t"])
    or_m = np.mean([res[t]["drop"]["OSFDreal-R50"] for t in order if t != "r50"])
    print(f"\nmean (loại WB): RD-CNX {rd_m:.1f} | OSFDreal-CNX {oc_m:.1f} | OSFDreal-R50 {or_m:.1f}")
    print("Swin-L: " + " | ".join(f"{t} RD {s14[t]['drop']['RD-ALL']:.0f} vs OSFDr-CNX {res[t]['drop']['OSFDreal-CNX']:.0f}"
                                   for t in ("dino_swin_l", "ddq_swinl")))
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/s15_osfd_real_dev30.json"), "w"), indent=1, ensure_ascii=False)
    print("[s15] -> results/mechanism/s15_osfd_real_dev30.json")


if __name__ == "__main__":
    main()
