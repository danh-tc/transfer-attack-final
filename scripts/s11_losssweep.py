"""S11 (progress_log 2026-11-17, KHOA): quet VI TRI LOSS nong->sau tren pipeline surrogate (OCT+MI+WP giu nguyen),
xem transfer DINO-Swin-L tang va co danh doi tren backbone CNN khong. Surrogate ConvNeXt, dev30.
Vi tri loss: L_feat (backbone C3-C5 = RD-FEAT, co) -> L_rpn (RPN objectness, MOI) -> L_roi (RoI cls = RD-DET, co).
Target trai 2 phia: r101 (CNN RoI de) | dino_r50 (DINO-CNN) | swin_t (Swin RoI) | dino_swin_l (kho).
Do kill + A_MASK (= TB t{0.5,1} Sum_MASK sign d*g_t / Sum_MASK |g_t|). g_t cache GDIR (dung chung S8).
  python scripts/s11_losssweep.py
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

# ten delta -> (co san? / ham sinh). FEAT=RD-FEAT, DET=RD-DET da co o S1. RPN moi.
RPN_FN = functools.partial(geo_osfd_attack, **dict(_C2OCT, cls_weight=0.2), cls_mode="rpn", feature=False,
                           weight_rho=0.002, weight_linear=True)
POS = [("L_feat", "RD-FEAT"), ("L_rpn", "RD-RPN"), ("L_roi", "RD-DET")]  # nong -> sau
TARGETS = ["r101", "dino_r50", "swin_t", "dino_swin_l"]
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
    # sinh RD-RPN (CNX)
    out = os.path.join(ADV, "RD-RPN")
    if not all(os.path.exists(os.path.join(out, f"{i}.png")) for i in ids):
        os.makedirs(out, exist_ok=True)
        m = load_model("convnext_t", "cuda:0"); ds = {s["img_id"]: s for s in target_dataset("convnext_t", ids)}
        for n, i in enumerate(ids):
            p = os.path.join(out, f"{i}.png")
            if os.path.exists(p):
                continue
            orig = ds[i]["orig_inputs"].cuda().float(); s1.seed_all(1000 * (n + 1))
            d = RPN_FN(m, orig, ds[i]["data_sample"], steps=50, epsilon=5.0)
            a = to_adv_image(orig, d)
            mmcv.imwrite(a.permute(1, 2, 0).cpu().numpy().astype(np.uint8), p + ".tmp.png"); os.replace(p + ".tmp.png", p)
            print(f"[s11 gen RPN] {n+1}/30", flush=True)
        del m; torch.cuda.empty_cache()

    names = [d for _, d in POS]
    D = {c: {i: (mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]).transpose(2, 0, 1)
             for i in ids} for c in names}
    os.makedirs(GDIR, exist_ok=True)
    res = {"n_img": len(ids), "pos": [p for p, _ in POS], "A": {}, "kill": {}}
    from mmdet.apis import inference_detector
    for tgt in TARGETS:
        gm = load_any(tgt); ds = {s["img_id"]: s for s in target_dataset(tgt, ids)}
        num = {c: {} for c in names}; den = {c: {} for c in names}
        for n, i in enumerate(ids):
            orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
            m = np.broadcast_to(MK[i][None], D[names[0]][i].shape)
            for c in names:
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
            print(f"[s11 A {tgt}] {n+1}/30", flush=True)
        del gm; torch.cuda.empty_cache()
        res["A"][tgt] = {c: sum(num[c].values()) / sum(den[c].values()) for c in names}
        tm = load(tgt); res["kill"][tgt] = {}
        with torch.no_grad():
            tp = {}
            for i in ids:
                g = group(to_coco_dets(inference_detector(tm, clean[i].astype(np.uint8)), i, cat)).get(i, [])
                tp[i] = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
                         if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values())
            for c in names:
                k = 0
                for i in ids:
                    gg = group(to_coco_dets(inference_detector(tm, mmcv.imread(os.path.join(ADV, c, f"{i}.png"))), i, cat)).get(i, [])
                    k += sum(fate(x["bbox"], x["category_id"], gg)[0] != "survived" for x in tp[i])
                res["kill"][tgt][c] = 100 * k / max(ntp, 1)
        del tm; torch.cuda.empty_cache()

    print(f"\nQUET VI TRI LOSS (CNX surrogate) — A_MASK (kill%):")
    print(f"{'vi tri':8}" + "".join(f"{t[:11]:>18}" for t in TARGETS))
    for pos, c in POS:
        print(f"{pos:8}" + "".join(f"{res['A'][t][c]:9.4f}({res['kill'][t][c]:3.0f})" for t in TARGETS))
    # xu huong nong->sau
    print("\nxu huong kill nong->sau (L_feat -> L_rpn -> L_roi):")
    for t in TARGETS:
        seq = [res["kill"][t][c] for _, c in POS]
        print(f"  {t:12} {seq[0]:.0f} -> {seq[1]:.0f} -> {seq[2]:.0f}  (sau - nong {seq[2]-seq[0]:+.0f})")
    res["trend_kill_deep_minus_shallow"] = {t: res["kill"][t]["RD-DET"] - res["kill"][t]["RD-FEAT"] for t in TARGETS}
    hard = res["trend_kill_deep_minus_shallow"]["dino_swin_l"]
    cnn = np.mean([res["trend_kill_deep_minus_shallow"][t] for t in ("r101", "dino_r50")])
    res["verdict"] = (f"loss sau GIUP Swin-L (+{hard:.0f}) " + ("co" if cnn < -3 else "KHONG") + f" danh doi CNN (TB {cnn:+.0f})")
    print(f"\n=> {res['verdict']}")
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/s11_losssweep_dev30.json"), "w"),
              indent=1, ensure_ascii=False, default=float)
    print("[s11] -> results/mechanism/s11_losssweep_dev30.json")


if __name__ == "__main__":
    main()
