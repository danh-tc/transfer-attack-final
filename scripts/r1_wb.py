"""R1-WB (progress_log 2026-11-17 00:45 + 01:05, KHÓA): white-box DINO-Swin-L oracle.
Sinh δ WB (loss cls không-dn, MI μ1, sign ascent, B20, ε5), 10 ảnh; rồi (a) region ablation keep/rem/rnd trên Swin-L,
(b) đồng thuận DẤU trong IN giữa δ_WB và 4 surrogate transfer (PNG F1b).
  python scripts/r1_wb.py --n 10
"""
import argparse
import itertools
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
from attack.preprocess import to_adv_image
from attack.targets import load_any, target_dataset, target_grads
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.r1_region import ADV, ATT, CELLS, IMGDIR, apply_cell, region_masks

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
WBDIR = os.path.join(REPO_ROOT, "artifacts/runs/r1_wb_swinl/adv/WB")
EPS, ALPHA, STEPS, MU = 5.0, 1.0, 20, 1.0
OUT = os.path.join(REPO_ROOT, "results/mechanism/r1_wb_dev10.json")


def gen_wb(ids):
    os.makedirs(WBDIR, exist_ok=True)
    if all(os.path.exists(os.path.join(WBDIR, f"{i}.png")) for i in ids):
        print("[wb] PNG đủ", flush=True); return
    model = load_any("dino_swin_l")
    ds = {s["img_id"]: s for s in target_dataset("dino_swin_l", ids)}
    for n, i in enumerate(ids):
        if os.path.exists(os.path.join(WBDIR, f"{i}.png")):
            continue
        torch.manual_seed(1000 * (n + 1))
        orig = ds[i]["orig_inputs"].cuda().float()
        dsm = ds[i]["data_sample"]
        delta = torch.zeros_like(orig); g = torch.zeros_like(orig)
        for _ in range(STEPS):
            x = torch.clamp(orig + delta, 0, 255)
            grad = target_grads(model, "dino_swin_l", x, dsm, objectives=("cls",))[0]["cls"]
            g = MU * g + grad / (grad.abs().mean() + 1e-12)
            delta = torch.clamp(delta + ALPHA * torch.sign(g), -EPS, EPS)
            delta = torch.clamp(orig + delta, 0, 255) - orig
        adv = to_adv_image(orig, delta)
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), os.path.join(WBDIR, f"{i}.png"))
        print(f"[wb] {n + 1}/{len(ids)}", flush=True)
    del model; torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:ap.parse_args().n]
    coco = COCO(ANN); cat_ids = sorted(coco.getCatIds())
    gen_wb(ids)
    from mmdet.apis import inference_detector

    # (a) region ablation của δ_WB trên dino_swin_l
    clean_arr, M, gts, dWB = {}, {}, {}, {}
    for i in ids:
        c = mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]))
        clean_arr[i] = c; h, w = c.shape[:2]
        gts[i] = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
        M[i] = region_masks(gts[i], h, w)
        dWB[i] = mmcv.imread(os.path.join(WBDIR, f"{i}.png")).astype(np.int32) - c.astype(np.int32)
    model = load("dino_swin_l")
    clean_grp, tp = {}, {}
    with torch.no_grad():
        for i in ids:
            clean_grp[i] = group(to_coco_dets(inference_detector(model, clean_arr[i]), i, cat_ids)).get(i, [])
            tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], clean_grp[i])[0] == "survived"]
    n_tp = sum(len(v) for v in tp.values())
    rng = np.random.default_rng(42); kills = {}
    with torch.no_grad():
        for cell in CELLS:
            k = 0
            for i in ids:
                if not tp[i]:
                    continue
                img = apply_cell(clean_arr[i], dWB[i], cell, M[i], rng)
                g = group(to_coco_dets(inference_detector(model, img), i, cat_ids)).get(i, [])
                k += sum(fate(a["bbox"], a["category_id"], g)[0] != "survived" for a in tp[i])
            kills[cell] = k
    del model; torch.cuda.empty_cache()
    kr = {c: kills[c] / max(n_tp, 1) for c in CELLS}
    share = {c: (kills[c] / kills["full"] if kills["full"] else 0.0) for c in CELLS}
    print(f"\nWB region (n_tp {n_tp}) kill%: " + " ".join(f"{c.replace('_','')}={kr[c]*100:.0f}" for c in CELLS))
    print(f"WB share keep: IN {share['keep_IN']:.2f}  RING {share['keep_RING']:.2f}  FAR {share['keep_FAR']:.2f}")

    # (b) đồng thuận dấu trong IN: WB vs mỗi surrogate
    SWB, Ssur = [], {a: [] for a in ATT}
    for i in ids:
        inm = M[i]["IN"].astype(bool)
        SWB.append(np.sign(dWB[i])[inm])
        for a in ATT:
            d = mmcv.imread(os.path.join(ADV, ATT[a][0], ATT[a][1], f"{i}.png")).astype(np.int32) - clean_arr[i].astype(np.int32)
            Ssur[a].append(np.sign(d)[inm])
    swb = np.concatenate(SWB)
    agree = {}
    for a in ATT:
        s = np.concatenate(Ssur[a]); nz = (swb != 0) & (s != 0)
        agree[a] = float((swb[nz] == s[nz]).mean())
    print("\nĐồng thuận DẤU trong IN, WB vs surrogate:")
    for a in ATT:
        print(f"   WB vs {a:10} {agree[a]:.3f}")
    d_cnx = agree["RD-CNX@50"] - agree["RD-R50"]; d_ens = agree["ENS-WP"] - agree["OSFD-PUB"]
    verdict = ("TRÙNG BẢN ĐỒ DẤU (CNX gần WB hơn R50)" if d_cnx >= 0.03 and d_ens >= 0.03
               else "BÁC: dấu WB không trùng surrogate nào" if max(agree.values()) <= 0.52
               else "KHÔNG RÕ")
    print(f"Δ(RD-CNX − RD-R50) {d_cnx:+.3f} | Δ(ENS − OSFD) {d_ens:+.3f} -> {verdict}")
    json.dump({"n": len(ids), "n_tp": n_tp, "wb_kill_rate": kr, "wb_share": share,
               "sign_agree_wb_vs_surrogate": agree, "d_cnx_minus_r50": d_cnx, "d_ens_minus_osfd": d_ens,
               "verdict": verdict}, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[r1-wb] -> {OUT}")


if __name__ == "__main__":
    main()
