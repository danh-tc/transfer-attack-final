"""S8 (progress_log 2026-11-17, KHOA): tach vai tro DETECTOR vs FEATURE + truc surrogate (CNX/R50) tren THANG TARGET
de-kho (DINO-R50 ~92% -> MaskRCNN-Swin-T ~75% -> DINO-Swin-L ~30%). dev30. Giai thich vi sao stuck o DINO-Swin-L.
6 delta = {CNX, R50} x {ALL, FEAT, DET}; CNX da co (S1), R50 sinh moi (cung cong thuc/seed S1, lam=6.0, khong weight_linear).
Moi target: g_t = grad cls (cache rieng target), A_MASK = TB t{0.5,1} Sum_MASK sign d*g_t / Sum_MASK |g_t|; + kill object clean-TP.
  python scripts/s8_detfeat.py
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
from scripts.run_baselines import METHODS, _C2OCT

ADV = os.path.join(REPO_ROOT, "artifacts/runs/s1_signmap/adv")
GDIR = os.path.join(REPO_ROOT, "artifacts/runs/s8_grads")
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
WP = dict(weight_rho=0.002)
# R50 cells (lam 6.0 theo rd_calib E83, KHONG weight_linear) — CNX cells da sinh o S1
R50_ATT = {
    "R50-ALL": functools.partial(geo_osfd_attack, **dict(_C2OCT, cls_weight=6.0), cls_mode="roi",
                                 stage_w=(0.0, 1.0, 1.0, 1.0), **WP),
    "R50-FEAT": functools.partial(geo_osfd_attack, **dict(_C2OCT, cls_weight=0.0, skip_mask=False,
                                                          feat_stages=(1, 2, 3)), **WP),
    "R50-DET": functools.partial(geo_osfd_attack, **dict(_C2OCT, cls_weight=6.0), cls_mode="roi", feature=False, **WP),
}
CNX = ["RD-ALL", "RD-FEAT", "RD-DET"]   # = CNX-{ALL,FEAT,DET} (S1)
DELTAS = CNX + list(R50_ATT)
TARGETS = ["dino_r50", "swin_t", "dino_swin_l"]   # de -> kho
TS = (0.5, 1.0)


def gen_r50(ids):
    todo = [c for c in R50_ATT if not all(os.path.exists(os.path.join(ADV, c, f"{i}.png")) for i in ids)]
    if not todo:
        return
    model = load_model("r50", "cuda:0"); ds = {s["img_id"]: s for s in target_dataset("r50", ids)}
    for c in todo:
        os.makedirs(os.path.join(ADV, c), exist_ok=True)
        for n, i in enumerate(ids):
            p = os.path.join(ADV, c, f"{i}.png")
            if os.path.exists(p):
                continue
            orig = ds[i]["orig_inputs"].cuda().float()
            s1.seed_all(1000 * (n + 1))
            d = R50_ATT[c](model, orig, ds[i]["data_sample"], steps=50, epsilon=5.0)
            a = to_adv_image(orig, d)
            assert float((a - orig).abs().max()) <= 5.0 + 1e-4
            mmcv.imwrite(a.permute(1, 2, 0).cpu().numpy().astype(np.uint8), p + ".tmp.png"); os.replace(p + ".tmp.png", p)
            print(f"[s8 gen {c}] {n+1}/{len(ids)}", flush=True)
    del model; torch.cuda.empty_cache()


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
    gen_r50(ids)
    os.makedirs(GDIR, exist_ok=True)
    D = {c: {i: (mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]).transpose(2, 0, 1)
             for i in ids} for c in DELTAS}

    res = {"n_img": len(ids), "targets": TARGETS, "A": {}, "kill": {}}
    from mmdet.apis import inference_detector
    for tgt in TARGETS:
        # A_MASK: g_t theo target (cache GDIR/<tgt>_...)
        gmodel = load_any(tgt)
        ds = {s["img_id"]: s for s in target_dataset(tgt, ids)}
        accnum = {c: {} for c in DELTAS}; accden = {c: {} for c in DELTAS}
        for n, i in enumerate(ids):
            orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
            m = np.broadcast_to(MK[i][None], D[DELTAS[0]][i].shape)
            for c in DELTAS:
                d = torch.from_numpy(D[c][i].astype(np.float32)).cuda()
                sd = np.sign(D[c][i])
                num = den = 0.0
                for t in TS:
                    p = os.path.join(GDIR, f"{tgt}_{i}_{c}_t{t}.npy")
                    if os.path.exists(p):
                        g = np.load(p).astype(np.float32)
                    else:
                        torch.manual_seed(1000 * (n + 1))
                        g = target_grads(gmodel, tgt, torch.clamp(orig + t * d, 0, 255), dsm,
                                         objectives=("cls",))[0]["cls"].float().cpu().numpy()
                        np.save(p + ".tmp.npy", g.astype(np.float16)); os.replace(p + ".tmp.npy", p)
                    num += (sd * g)[m].sum(); den += np.abs(g)[m].sum()
                accnum[c][i] = float(num); accden[c][i] = float(den)
            print(f"[s8 A {tgt}] {n+1}/{len(ids)}", flush=True)
        del gmodel; torch.cuda.empty_cache()
        res["A"][tgt] = {c: sum(accnum[c].values()) / sum(accden[c].values()) for c in DELTAS}
        # kill
        tmodel = load(tgt); res["kill"][tgt] = {}
        with torch.no_grad():
            tp = {}
            for i in ids:
                g = group(to_coco_dets(inference_detector(tmodel, clean[i].astype(np.uint8)), i, cat)).get(i, [])
                tp[i] = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
                         if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values())
            for c in DELTAS:
                k = 0
                for i in ids:
                    gg = group(to_coco_dets(inference_detector(tmodel, mmcv.imread(os.path.join(ADV, c, f"{i}.png"))), i, cat)).get(i, [])
                    k += sum(fate(x["bbox"], x["category_id"], gg)[0] != "survived" for x in tp[i])
                res["kill"][tgt][c] = 100 * k / max(ntp, 1)
        del tmodel; torch.cuda.empty_cache()

    print(f"\n{'':9}" + "".join(f"{t[:11]:>22}" for t in TARGETS))
    print(f"{'delta':9}" + "".join(f"{'A_MASK':>11}{'kill%':>11}" for _ in TARGETS))
    for c in DELTAS:
        print(f"{c:9}" + "".join(f"{res['A'][t][c]:11.4f}{res['kill'][t][c]:11.0f}" for t in TARGETS))

    # phan tich (CNX cells ten RD-*, R50 cells ten R50-*)
    A = res["A"]
    def amap(sur, br):
        return {"CNX": {"ALL": "RD-ALL", "FEAT": "RD-FEAT", "DET": "RD-DET"}, "R50": {"ALL": "R50-ALL", "FEAT": "R50-FEAT", "DET": "R50-DET"}}[sur][br]
    res["complement_ALL_minus_maxbranch"] = {
        sur: {t: A[t][amap(sur, "ALL")] - max(A[t][amap(sur, "FEAT")], A[t][amap(sur, "DET")]) for t in TARGETS}
        for sur in ("CNX", "R50")}
    res["det_minus_feat"] = {sur: {t: A[t][amap(sur, "DET")] - A[t][amap(sur, "FEAT")] for t in TARGETS}
                             for sur in ("CNX", "R50")}
    res["cnx_minus_r50_ALL"] = {t: A[t]["RD-ALL"] - A[t]["R50-ALL"] for t in TARGETS}
    print("\nDET - FEAT (A_MASK):")
    for sur in ("CNX", "R50"):
        print(f"  {sur}: " + "  ".join(f"{t[:9]} {res['det_minus_feat'][sur][t]:+.4f}" for t in TARGETS))
    print("CNX - R50 (ALL):  " + "  ".join(f"{t[:9]} {res['cnx_minus_r50_ALL'][t]:+.4f}" for t in TARGETS))
    print("\nA_MASK CNX-ALL theo thang:  " + "  ".join(f"{t[:9]} {A[t]['RD-ALL']:.4f}" for t in TARGETS))
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/s8_detfeat_dev30.json"), "w"),
              indent=1, ensure_ascii=False, default=float)
    print("\n[s8] -> results/mechanism/s8_detfeat_dev30.json")


if __name__ == "__main__":
    main()
