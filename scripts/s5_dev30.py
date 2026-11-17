"""S5 (progress_log 2026-11-17, KHÓA): xác nhận A_MASK vs A_ALL trên 20 ảnh MỚI (dev30 \\ dev10) với outcome LIÊN TỤC
theo object. Sinh 6 ô attack S1 cho ảnh 11–30 (cùng seed 1000·(n+1), n = chỉ số trong dev30), g_t DINO-Swin-L
(t ∈ {0, 0.5, 1}, cache như S3), eval DINO-Swin-L.
Outcome theo ảnh: GT không-crowd có score sạch ≥ 0.3 (det cùng lớp IoU ≥ 0.5, max score) = clean-TP;
  harm = 1 − TB(score_adv / score_clean) (score_adv = 0 nếu không còn det khớp); kill = tỉ lệ score_adv < 0.3.
A_R(c, i) = Σ_R sign δ·g_t / Σ_R |g_t|, TB t ∈ {0.5, 1}; R ∈ {MASK, ALL}.
Tương quan với harm: theo ô (6, harm TB ảnh), trong-ảnh (Spearman 6 ô / ảnh, TB), CI bootstrap theo ảnh cho
ρ trong-ảnh và cho hiệu số MASK − ALL (paired).
  python scripts/s5_dev30.py
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

import scripts.s1_signmap as s1
from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.targets import load_any, target_dataset, target_grads
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import _iou_xywh
from scripts.s3_region import ADV, ANN, GDIR, IMGDIR, regions

ATT = ["RD-ALL", "RD-FEAT", "RD-DET", "OSFD-ALL", "OSFD-FEAT", "OSFD-DET"]
OUT = os.path.join(REPO_ROOT, "results/mechanism/s5_dev30.json")
TS = ("0.5", "1.0")
NBOOT, BOOT_SEED = 1000, 2026


def match_score(gt, cat, dets):
    if not dets:
        return 0.0
    B = np.array([d["bbox"] for d in dets]).reshape(-1, 4)
    ok = (_iou_xywh(np.array(gt), B) >= 0.5) & (np.array([d["category_id"] for d in dets]) == cat)
    return float(np.array([d["score"] for d in dets])[ok].max()) if ok.any() else 0.0


def within(a, h, imgs):
    r = []
    for i in imgs:
        x = [a[(c, i)] for c in ATT]; y = [h[(c, i)] for c in ATT]
        r.append(spearmanr(x, y).correlation if np.std(y) > 0 and np.std(x) > 0 else np.nan)
    return np.array(r)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    assert ids[:10] == load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    new = ids[10:]
    for c in ATT:
        s1.gen_attack(ids, c, os.path.join(ADV, c))
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    MASK = {i: regions(coco, i, *clean[i].shape[:2])["MASK"] for i in ids}
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    sd = {(c, i): np.sign(mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]) for c in ATT for i in ids}

    # ---- g_t (cache S3) ----
    os.makedirs(GDIR, exist_ok=True)
    model = load_any("dino_swin_l")
    ds = {s["img_id"]: s for s in target_dataset("dino_swin_l", ids)}
    A = {}
    for n, i in enumerate(ids):
        orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
        m = np.broadcast_to(MASK[i][None], (3,) + MASK[i].shape)
        for c in ATT:
            d =torch.from_numpy((mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i])
                                 .transpose(2, 0, 1).astype(np.float32)).cuda()
            s = sd[(c, i)].transpose(2, 0, 1)
            vals = {"MASK": [], "ALL": []}
            for t in TS:
                p = os.path.join(GDIR, f"{i}_{c}_t{t}.npy")
                if os.path.exists(p):
                    g = np.load(p).astype(np.float32)
                else:
                    torch.manual_seed(1000 * (n + 1))
                    g = target_grads(model, "dino_swin_l", torch.clamp(orig + float(t) * d, 0, 255), dsm,
                                     objectives=("cls",))[0]["cls"].float().cpu().numpy()
                    np.save(p, g.astype(np.float16))
                vals["MASK"].append(float((s * g)[m].sum() / np.abs(g)[m].sum()))
                vals["ALL"].append(float((s * g).sum() / np.abs(g).sum()))
            for r in vals:
                A[(r, c, i)] = float(np.mean(vals[r]))
        print(f"[s5 grad] {n + 1}/{len(ids)}", flush=True)
    del model; torch.cuda.empty_cache()

    # ---- outcome liên tục ----
    from mmdet.apis import inference_detector
    model = load("dino_swin_l")
    harm, kill, nobj, drop = {}, {}, {}, {}
    with torch.no_grad():
        cl, sc = [], {}
        for i in ids:
            d = to_coco_dets(inference_detector(model, clean[i].astype(np.uint8)), i, cat); cl += d
            sc[i] = [(a, s_) for a in gts[i] for s_ in [match_score(a["bbox"], a["category_id"], d)] if s_ >= 0.3]
        evc = CocoImageEval(coco, cl, ids)
        for c in ATT:
            ad = []
            for i in ids:
                d = to_coco_dets(inference_detector(model, mmcv.imread(os.path.join(ADV, c, f"{i}.png"))), i, cat); ad += d
                r = [match_score(a["bbox"], a["category_id"], d) / s_ for a, s_ in sc[i]]
                harm[(c, i)] = 1 - float(np.mean(r)) if r else np.nan
                kill[(c, i)] = float(np.mean([x * s_ < 0.3 for x, (a, s_) in zip(r, sc[i])])) if r else np.nan
                nobj[i] = len(r)
            eva = CocoImageEval(coco, ad, ids)
            for name, sub in (("new20", new), ("all30", ids)):
                w = np.array([1.0 if i in sub else 0.0 for i in ids])
                drop[(c, name)] = 100 * (evc.ap(w)[0] - eva.ap(w)[0]) / evc.ap(w)[0]
    del model; torch.cuda.empty_cache()

    # ---- phân tích ----
    res = {"ids_new": new, "n_obj": nobj, "sets": {}}
    rng = np.random.default_rng(BOOT_SEED)
    for name, sub in (("new20", new), ("all30", ids)):
        sub = [i for i in sub if np.isfinite(harm[(ATT[0], i)])]
        out = {"n_img": len(sub), "cell": {}}
        print(f"\n=== {name} ({len(sub)} ảnh có clean-TP) ===")
        print(f"{'ô':10}{'A_MASK':>9}{'A_ALL':>9}{'harm':>8}{'kill':>8}{'AP drop':>9}")
        for c in ATT:
            out["cell"][c] = {k: float(np.mean([D[(r, c, i)] if r else D[(c, i)] for i in sub]))
                              for k, D, r in (("A_MASK", A, "MASK"), ("A_ALL", A, "ALL"), ("harm", harm, None),
                                              ("kill", kill, None))}
            out["cell"][c]["ap_drop"] = drop[(c, name)]
            e = out["cell"][c]
            print(f"{c:10}{e['A_MASK']:9.4f}{e['A_ALL']:9.4f}{e['harm']:8.3f}{e['kill']:8.3f}{e['ap_drop']:9.1f}")
        for r in ("MASK", "ALL"):
            a = {(c, i): A[(r, c, i)] for c in ATT for i in sub}
            wi = within(a, harm, sub)
            bs = [np.nanmean(wi[rng.integers(0, len(sub), len(sub))]) for _ in range(NBOOT)]
            out[r] = {"rho_cell_harm": spearmanr([out["cell"][c][f"A_{r}"] for c in ATT], [out["cell"][c]["harm"] for c in ATT]).correlation,
                      "rho_cell_apdrop": spearmanr([out["cell"][c][f"A_{r}"] for c in ATT], [out["cell"][c]["ap_drop"] for c in ATT]).correlation,
                      "rho_within_mean": float(np.nanmean(wi)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      "n_pos": int((wi > 0).sum()), "within": wi.tolist()}
            o = out[r]
            print(f"A_{r:4}: ρ ô vs harm {o['rho_cell_harm']:+.2f} | vs AP drop {o['rho_cell_apdrop']:+.2f} | "
                  f"ρ trong ảnh TB {o['rho_within_mean']:+.2f} [{o['ci95'][0]:+.2f}, {o['ci95'][1]:+.2f}] ({o['n_pos']}/{len(sub)} > 0)")
        wm = np.array(out["MASK"]["within"]); wa = np.array(out["ALL"]["within"])
        idx = [rng.integers(0, len(sub), len(sub)) for _ in range(NBOOT)]
        dd = [np.nanmean(wm[k]) - np.nanmean(wa[k]) for k in idx]
        out["diff_mask_minus_all"] = {"point": float(np.nanmean(wm) - np.nanmean(wa)),
                                      "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
        print(f"MASK − ALL (ρ trong ảnh): {out['diff_mask_minus_all']['point']:+.2f} "
              f"[{out['diff_mask_minus_all']['ci95'][0]:+.2f}, {out['diff_mask_minus_all']['ci95'][1]:+.2f}]")
        res["sets"][name] = out

    p = res["sets"]["new20"]
    verdict = {}
    for r in ("MASK", "ALL"):
        o = p[r]
        verdict[r] = ("DỰ BÁO" if o["rho_cell_harm"] >= 0.6 and o["rho_within_mean"] >= 0.3 and o["ci95"][0] > 0 else
                      "KHÔNG dự báo" if o["rho_cell_harm"] <= 0.1 and o["rho_within_mean"] <= 0.1 else "KHÔNG RÕ")
    d = p["diff_mask_minus_all"]
    verdict["MASK_vs_ALL"] = ("MASK > ALL" if d["ci95"][0] > 0 else "ALL > MASK" if d["ci95"][1] < 0 else "không phân biệt")
    res["verdict"] = verdict
    print(f"\n=> (new20) A_MASK: {verdict['MASK']} | A_ALL: {verdict['ALL']} | {verdict['MASK_vs_ALL']}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"[s5] -> {OUT}")


if __name__ == "__main__":
    main()
