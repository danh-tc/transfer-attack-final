#!/usr/bin/env python3
"""E77-DFC (progress_log 2026-10-31): Detection Failure Consistency — DIAGNOSTIC output-space, gate-first, KHÔNG dựng method.
RQ (giả thuyết MỚI): các backbone có gradient-geometry rất khác (E64-E76) nhưng có thể chia sẻ CƠ CHẾ THẤT BẠI ở mức detection.
E77 phân rã AP-drop thành LOẠI thất bại cho RD-R50 / RD-CNX / ENS-WP × nhiều target, xem có một LOẠI mà attack transfer-mạnh
(RD-CNX/ENS) gây ra trên Swin mà RD-R50 (yếu) KHÔNG — tức một hành vi đầu ra cụ thể để nhắm bằng loss R50-only.
Forward-only trên PNG dev100 ĐÃ CÓ (E73), không sinh attack. Khác E64-E76: output-space, không cosine input-gradient.
Phân loại số phận mỗi object clean-TP dưới attack: survived / cls-fail (box đúng chỗ sai lớp) / loc-fail (box lệch, đúng lớp) /
suppressed (mất hẳn); + false-positive/ảnh (ảo giác), + erosion điểm của object sống. GATE theo COMPOSITION chuẩn-hoá (shape),
KHÔNG theo độ lớn (tránh tautology "attack mạnh hơn => mọi chiều cao hơn").
  python scripts/e77_dfc.py   -> results/mechanism/e77_dfc.json"""
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from attack.mechanism import _iou_xywh
from scripts.e64_cnx_drop import load

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e77_dfc.json")
# cond -> (run dir, adv folder)  — PNG dev100 từ E73
FIX = {"RD-R50": ("dev300_B36_eps5_limit100", "C2-OCT-1V-WP2-RD"),
       "RD-CNX": ("dev300_B36_eps5_limit100_sur-convnext_t", "C2-OCT-1V-WP2-RD-CNX"),
       "ENS-WP": ("dev300_B36_eps5_limit100_ens", "ENS-WP")}
CONDS = ["RD-R50", "RD-CNX", "ENS-WP"]
TARGETS = ["dino_swin_l", "swin_t", "convnext_t", "r101", "dino_r50", "fcos_r50"]
TRANS = ["dino_swin_l", "swin_t"]      # họ Transformer: RD-CNX/ENS mạnh, RD-R50 yếu
RESNET = ["r101", "dino_r50", "fcos_r50"]
SCORE_THR = 0.3


def png(cond, i):
    run, folder = FIX[cond]
    return os.path.join(ART, run, "adv", folder, f"{i}.png")


def group(dets):
    by = {}
    for d in dets:
        if d["score"] >= SCORE_THR:
            by.setdefault(d["image_id"], []).append(d)
    return by


def fate(gt, cat, dets_img):
    """Số phận 1 object clean-TP (gt xywh, cat) dưới 1 điều kiện."""
    if not dets_img:
        return "suppressed", None
    B = np.array([d["bbox"] for d in dets_img]).reshape(-1, 4)
    C = np.array([d["category_id"] for d in dets_img])
    S = np.array([d["score"] for d in dets_img])
    iou = _iou_xywh(np.array(gt), B)
    hi = iou >= 0.5
    same = C == cat
    if (hi & same).any():
        return "survived", float(S[hi & same].max())
    if hi.any():
        return "cls_fail", None
    if ((iou >= 0.1) & same).any():
        return "loc_fail", None
    return "suppressed", None


def main():
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    assert len(ids) == 100
    cat_ids = sorted(coco.getCatIds())
    from mmdet.apis import inference_detector
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"n_images": 100, "score_thr": SCORE_THR, "profile": {}}
    for t in TARGETS:
        if t in res["profile"]:
            continue
        model = load(t)
        # clean dets -> xác định object clean-TP ("at-risk") + điểm sạch của chúng
        cleang = group([d for i in ids for d in to_coco_dets(inference_detector(model, os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])), i, cat_ids)])
        atrisk = {}  # img -> list (gt_xywh, cat, clean_score)
        for i in ids:
            for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
                f, sc = fate(a["bbox"], a["category_id"], cleang.get(i, []))
                if f == "survived":
                    atrisk.setdefault(i, []).append((a["bbox"], a["category_id"], sc))
        prof = {}
        for c in CONDS:
            advg = group([d for i in ids for d in to_coco_dets(inference_detector(model, png(c, i)), i, cat_ids)])
            cnt = {"survived": 0, "cls_fail": 0, "loc_fail": 0, "suppressed": 0}
            nrisk = 0; fp = 0; drop_sum = 0.0; surv = 0
            for i in ids:
                di = advg.get(i, [])
                for gt, cat, csc in atrisk.get(i, []):
                    nrisk += 1
                    f, sc = fate(gt, cat, di)
                    cnt[f] += 1
                    if f == "survived":
                        surv += 1; drop_sum += (csc - sc)
                # false positive: det không khớp GT nào (IoU<0.1 mọi GT)
                anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
                G = np.array([a["bbox"] for a in anns]).reshape(-1, 4)
                for d in di:
                    if len(anns) == 0 or _iou_xywh(np.array(d["bbox"]), G).max() < 0.1:
                        fp += 1
            fails = nrisk - cnt["survived"]
            comp = {k: (cnt[k] / fails if fails else 0.0) for k in ("cls_fail", "loc_fail", "suppressed")}
            prof[c] = {"n_atrisk": nrisk, "frac": {k: round(cnt[k] / nrisk, 4) for k in cnt},
                       "fail_comp": {k: round(v, 4) for k, v in comp.items()},
                       "fp_per_img": round(fp / len(ids), 3),
                       "score_erosion_surv": round(drop_sum / surv, 4) if surv else 0.0}
        res["profile"][t] = prof
        del model
        import torch; torch.cuda.empty_cache()
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
        print(t, {c: {"supp": prof[c]["frac"]["suppressed"], "fp": prof[c]["fp_per_img"]} for c in CONDS}, flush=True)

    # ---- bảng + GO/NO-GO (khóa trước) ----
    print("\n=== E77-DFC: profile loại-thất-bại (frac theo at-risk objects) ===")
    for t in TARGETS:
        print(f"\n{t}  (at-risk n={res['profile'][t]['RD-R50']['n_atrisk']})")
        print(f"  {'cond':8}{'surv':>7}{'cls':>7}{'loc':>7}{'supp':>7}  | {'FP/img':>7}{'erode':>7}  | comp[cls,loc,supp]")
        for c in CONDS:
            p = res["profile"][t][c]; fr = p["frac"]; fc = p["fail_comp"]
            print(f"  {c:8}{fr['survived']:7.3f}{fr['cls_fail']:7.3f}{fr['loc_fail']:7.3f}{fr['suppressed']:7.3f}"
                  f"  | {p['fp_per_img']:7.2f}{p['score_erosion_surv']:7.3f}  | "
                  f"[{fc['cls_fail']:.2f},{fc['loc_fail']:.2f},{fc['suppressed']:.2f}]")

    print("\n--- GO/NO-GO (khóa trước khi đọc) — COMPOSITION chuẩn-hoá, không độ lớn ---")
    # một LOẠI thất bại d: comp[d] của RD-CNX và ENS đều ≥ comp[d] của RD-R50 + 0.20, NHẤT QUÁN trên cả 2 target Transformer
    go_dims = []
    for d in ("cls_fail", "loc_fail", "suppressed"):
        ok = all(res["profile"][t]["RD-CNX"]["fail_comp"][d] >= res["profile"][t]["RD-R50"]["fail_comp"][d] + 0.20
                 and res["profile"][t]["ENS-WP"]["fail_comp"][d] >= res["profile"][t]["RD-R50"]["fail_comp"][d] + 0.20
                 for t in TRANS)
        deltas = [round(res["profile"][t]["ENS-WP"]["fail_comp"][d] - res["profile"][t]["RD-R50"]["fail_comp"][d], 3) for t in TRANS]
        print(f"dim {d:10} ENS−RD-R50 comp (Transformer {TRANS}) = {deltas}  qualifies={ok}")
        if ok:
            go_dims.append(d)
    # FP như chiều ảo-giác riêng (tỷ số, không chuẩn-hoá composition)
    fp_ratio = [round(res["profile"][t]["ENS-WP"]["fp_per_img"] / max(res["profile"][t]["RD-R50"]["fp_per_img"], 1e-6), 2) for t in TRANS]
    print(f"FP ENS/RD-R50 (Transformer) = {fp_ratio}")
    print("\nVERDICT:",
          "E77 GO — có loại thất bại cross-family cụ thể: " + ",".join(go_dims) +
          " (RD-CNX/ENS gây, RD-R50 không) -> mục tiêu output để thiết kế loss R50-only (E78)"
          if go_dims else
          "E77 NO-GO — profile thất bại khác nhau CHỦ YẾU về độ lớn, không về LOẠI; không có cơ chế thất bại cross-family riêng để nhắm -> đóng DFC")


if __name__ == "__main__":
    main()
