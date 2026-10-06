#!/usr/bin/env python3
"""E2 (quy tắc khóa: progress_log 2026-10-06): hiệu ứng của δ trên surrogate có SỐNG SÓT qua biến đổi hình học không, và
độ sống sót đó có dự đoán transfer sang DINO-Swin-L không. Chỉ forward.

T' cố định áp CÙNG LÚC lên ảnh sạch và adv ở khung R50 (sau Resize của test pipeline): ID, xoay +10°, trượt ngang 0.15,
phối cảnh yaw 15° (căn tâm), tịnh tiến (+7, +5) px. R50 predict (khung model) trên T'(sạch) -> detection tham chiếu
(score ≥ 0.3); trên T'(adv) -> score khớp (cùng lớp, IoU ≥ 0.5) -> retained_T'. E_id = 1 − retained_ID,
E_surv = 1 − mean_{T' ≠ ID} retained_T'.
  python scripts/e2_survival.py --run n300_B50_eps5 --methods MI-FGSM M-DI2-FGSM OSFD       # (a) theo ảnh
  python scripts/e2_survival.py --run dev300_B50_eps5 --subset dev30 --methods ... --tag dev30  # (b) giữa ô
"""
import argparse
import json
import math
import os
import sys
import time

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.mechanism import boot_spearman, boot_weights, per_image_transfer
from attack.methods.camgeo import rotation_homography
from attack.methods.elem import linear_homography
from attack.methods.pia import warp, warp_grid
from attack.models import load_surrogate
from attack.preprocess import pipeline_resize, predict

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
BOOT_SEED, N_BOOT, SCORE = 2026, 1000, 0.3
OUT = os.path.join(REPO_ROOT, "results/mechanism")


def transforms(h, w):
    t = math.radians(10.0)
    T = torch.eye(3, dtype=torch.float64)
    T[0, 2], T[1, 2] = -7.0, -5.0  # pixel ra q lấy mẫu ở q − (7, 5) -> nội dung dời (+7, +5)
    return {"ID": None,
            "rot10": linear_homography(torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]],
                                                    dtype=torch.float64), h, w),
            "shear15": linear_homography(torch.tensor([[1.0, 0.15], [0.0, 1.0]], dtype=torch.float64), h, w),
            "persp15": rotation_homography(math.radians(15.0), 0.0, h, w),
            "trans7_5": T}


def iou(a, b):
    """a [N,4], b [M,4] xyxy -> [N,M]."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).clamp(min=0).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def retained(ref, adv):
    keep = ref.scores >= SCORE
    if not keep.any():
        return np.nan
    rb, rs, rl = ref.bboxes[keep], ref.scores[keep], ref.labels[keep]
    if len(adv.bboxes) == 0:
        return 0.0
    m = (iou(rb, adv.bboxes) >= 0.5) & (rl[:, None] == adv.labels[None])
    sa = torch.where(m, adv.scores[None].expand_as(m), torch.zeros_like(m, dtype=adv.scores.dtype)).max(1).values
    return float((sa / rs).mean())


def run(args):
    dev = "cuda:0"
    split = args.run.split("_")[0]
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{split}.csv"))
    if args.subset:
        keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv")))
        ids = [i for i in ids if i in keep]
    ids = ids[:args.limit] if args.limit else ids
    adv_dir = os.path.join(REPO_ROOT, "artifacts/runs", args.run, "adv")
    model = load_surrogate(dev)
    ds_all = AttackDataset(ids)
    res = {m: {} for m in args.methods}
    t0 = time.time()
    for i in range(len(ds_all)):
        smp = ds_all[i]
        x, ds = smp["inputs"].to(dev), smp["data_sample"]
        h, w = x.shape[-2:]
        Ts = transforms(h, w)
        grids = {k: (warp_grid(H, h, w, dev) if H is not None else None) for k, H in Ts.items()}
        # Đối chứng KHÔNG hình học: nhiễu Gaussian σ = 8 cố định theo ảnh (cùng nhiễu cho sạch và adv).
        gen = torch.Generator(device=dev).manual_seed(BOOT_SEED + smp["img_id"])
        grids["noise8"] = 8.0 * torch.randn(x.shape, generator=gen, device=dev)
        tv = lambda img, g: img if g is None else (torch.clamp(img + g, 0, 255) if g.dim() == 3 else warp(img, g))
        refs = {k: predict(model, tv(x, g), ds, rescale=False).pred_instances for k, g in grids.items()}
        for m in args.methods:
            png = torch.from_numpy(mmcv.imread(os.path.join(adv_dir, m, f"{smp['img_id']}.png"))).permute(2, 0, 1).float().to(dev)
            a = pipeline_resize(png, (h, w))
            res[m][smp["img_id"]] = {k: retained(refs[k], predict(model, tv(a, g), ds, rescale=False).pred_instances)
                                    for k, g in grids.items()}
        if (i + 1) % 25 == 0 or i == len(ds_all) - 1:
            print(f"[E2] {i + 1}/{len(ds_all)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"e2_per_image_{args.tag}.json"), "w") as f:
        json.dump({"run": args.run, "img_ids": ids, "retained_r50": res}, f)


def analyze(args):
    d = json.load(open(os.path.join(OUT, f"e2_per_image_{args.tag}.json")))
    ids, res = d["img_ids"], d["retained_r50"]
    tnames = [k for k in next(iter(next(iter(res.values())).values())) if k not in ("ID", "noise8")]
    E = {m: {"id": np.array([1 - res[m][str(i)]["ID"] for i in ids]),
             "noise8": np.array([1 - res[m][str(i)]["noise8"] for i in ids]),
             "surv": np.array([1 - np.nanmean([res[m][str(i)][k] for k in tnames]) for i in ids]),
             **{k: np.array([1 - res[m][str(i)][k] for i in ids]) for k in tnames}} for m in res}
    out = {"run": d["run"], "n_images": len(ids), "transforms": tnames,
           "mean_E": {m: {k: float(np.nanmean(v)) for k, v in E[m].items()} for m in E}}
    if args.targets:
        coco = COCO(ANN_FILE)
        art = os.path.join(REPO_ROOT, "artifacts/runs", d["run"])
        ctrl = json.load(open(os.path.join(art, args.dets)))
        idx = boot_weights(len(ids), N_BOOT, BOOT_SEED)
        out["spearman"] = {}
        for t in args.targets:
            gen = os.path.join(art, "gen_dets", f"{t}.json")
            dets = json.load(open(gen)) if os.path.exists(gen) else {c: ctrl[c][t] for c in ["clean"] + list(res)}
            for m in res:
                ret_t = per_image_transfer(coco, dets["clean"], dets[m], ids)["retained"]
                r_id = boot_spearman(E[m]["id"], ret_t, idx)
                r_sv = boot_spearman(E[m]["surv"], ret_t, idx)
                # hiệu ρ(E_surv) − ρ(E_id), paired cùng chỉ số resample
                from scipy.stats import spearmanr

                def rho(a, ii):
                    x, y = a[ii], ret_t[ii]
                    ok = ~(np.isnan(x) | np.isnan(y))
                    return spearmanr(x[ok], y[ok])[0]
                diff = np.array([rho(E[m]["surv"], r) - rho(E[m]["id"], r) for r in idx])
                pt = rho(E[m]["surv"], np.arange(len(ids))) - rho(E[m]["id"], np.arange(len(ids)))
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                dn = np.array([rho(E[m]["surv"], r) - rho(E[m]["noise8"], r) for r in idx])
                ptn = rho(E[m]["surv"], np.arange(len(ids))) - rho(E[m]["noise8"], np.arange(len(ids)))
                out.setdefault("spearman", {})
                out["spearman"][f"{m}/{t}"] = {"rho_E_id": r_id, "rho_E_surv": r_sv,
                                               "rho_E_noise8": boot_spearman(E[m]["noise8"], ret_t, idx),
                                               "surv_minus_noise8": {"point": float(ptn),
                                                                     "ci95": [float(x) for x in np.nanpercentile(dn, [2.5, 97.5])]},
                                               "surv_minus_id": {"point": float(pt), "ci95": [float(lo), float(hi)]},
                                               "per_transform": {k: boot_spearman(E[m][k], ret_t, idx)["point"] for k in tnames}}
    with open(os.path.join(OUT, f"e2_survival_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"=== E2 {args.tag}: mean E (1 − retained R50) ===")
    for m, v in out["mean_E"].items():
        print(f"{m:<14}" + "  ".join(f"{k}={x:.3f}" for k, x in v.items()))
    for k, v in out.get("spearman", {}).items():
        s = v["surv_minus_id"]
        sn = v["surv_minus_noise8"]
        print(f"{k:<28} ρ(E_id)={v['rho_E_id']['point']:+.2f}  ρ(E_noise8)={v['rho_E_noise8']['point']:+.2f}  "
              f"ρ(E_surv)={v['rho_E_surv']['point']:+.2f}  surv−id={s['point']:+.2f} [{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}]  "
              f"surv−noise={sn['point']:+.2f} [{sn['ci95'][0]:+.2f},{sn['ci95'][1]:+.2f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="n300_B50_eps5")
    p.add_argument("--subset", default=None)
    p.add_argument("--methods", nargs="+", default=["MI-FGSM", "M-DI2-FGSM", "OSFD"])
    p.add_argument("--targets", nargs="*", default=["dino_swin_l", "convnext_t", "swin_t", "r101"])
    p.add_argument("--dets", default="dets.json")
    p.add_argument("--tag", default="n300")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    if not args.analyze_only:
        run(args)
    analyze(args)


if __name__ == "__main__":
    main()
