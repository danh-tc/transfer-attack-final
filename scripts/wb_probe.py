#!/usr/bin/env python3
"""White-box DINO-Swin-L vs surrogate R50 trên dev10 (progress_log 2026-10-18) — MÔ TẢ, sinh manh mối, không phải method.

Câu hỏi: (1) khi attack THẬT SỰ phá được DINO-Swin-L (white-box), nó phá theo kiểu nào (xóa object lớn? box giả?),
so với C2-OCT-1V / OSFD từ R50; (2) chiều ngược: δ sinh trên DINO-Swin-L có hại R50 / CNN không (bất đối xứng?).
Ô white-box (surrogate = DINO-Swin-L, NGOÀI threat model, chỉ phân tích):
- WB-OSFD: đúng công thức OSFD (k = 3, MI, RRB), B = 50, AMP fp16 (V100 16 GB; E8 dùng fp32 trên 3090).
- WB-CLS: số hạng phân loại không-dn của DINO (như E8 WB-s1), ascent, MI, B = 50, checkpointing.
Mốc từ R50 (PNG có sẵn, cùng ảnh): OSFD B50 (`dev300_B50_eps5_limit100`), C2-OCT-1V B41 (`dev300_B41_eps5_limit100`).
Eval: 6 model, AP drop (10 ảnh, không CI) + hồ sơ lỗi (scripts/err_profile.profile).

Chạy: python scripts/wb_probe.py [--budget 50] [--subset dev10]
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import build_attack_dataset, load_image_ids  # noqa: E402
from attack.evaluation import CocoImageEval, to_coco_dets  # noqa: E402
from attack.models import load_model  # noqa: E402
from attack.preprocess import pipeline_resize, predict, to_adv_image  # noqa: E402
from attack.targets import load_any, target_dataset  # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts/wb_probe")
SEED = 0
MODELS = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
REF = {"OSFD": "artifacts/runs/dev300_B50_eps5_limit100/adv/OSFD",
       "C2-OCT-1V": "artifacts/runs/dev300_B41_eps5_limit100/adv/C2-OCT-1V"}


def gen(ids, split, budget):
    from attack.methods.baselines import osfd_attack
    from scripts.e8_whitebox_lowfreq import attack as cls_attack
    model = load_any("dino_swin_l")
    tds = target_dataset("dino_swin_l", ids)
    for cell in ("WB-CLS", "WB-OSFD"):
        os.makedirs(os.path.join(ART, cell), exist_ok=True)
        if cell == "WB-OSFD":  # như E8: checkpointing không tương thích autograd.grad
            for m in model.modules():
                if getattr(m, "with_cp", False):
                    m.with_cp = False
        t0 = time.time()
        for i in range(len(tds)):
            smp = tds[i]
            png = os.path.join(ART, cell, f"{smp['img_id']}.png")
            if os.path.exists(png):
                continue
            seed = SEED + split.index(smp["img_id"])
            random.seed(seed); torch.manual_seed(seed)
            orig = smp["orig_inputs"].cuda()
            if cell == "WB-CLS":
                d = cls_attack(model, orig, smp["data_sample"], 1, budget, 5.0)
            else:
                model.eval()
                with torch.autocast("cuda", dtype=torch.float16):
                    d = osfd_attack(model, orig, smp["data_sample"], steps=budget, epsilon=5.0)
            adv = to_adv_image(orig, d.float())
            assert float((adv - orig).abs().max()) <= 5.0 + 1e-4
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png)
            print(f"[wb {cell}] {i + 1}/{len(tds)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()


def evaluate(ids, conds):
    from mmdet.apis import inference_detector
    from pycocotools.coco import COCO
    coco = COCO(os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json"))
    cat_ids = sorted(coco.getCatIds())
    path = {c: (lambda i, c=c: os.path.join(ART, c, f"{i}.png")) for c in conds if c.startswith("WB")}
    for c in conds:
        if c in REF:
            path[c] = lambda i, c=c: os.path.join(REPO_ROOT, REF[c], f"{i}.png")
    dataset = build_attack_dataset("dev300")
    pos = {img: j for j, img in enumerate(dataset.image_ids)}
    dets = {m: {c: [] for c in ["clean"] + conds} for m in MODELS}
    for m in MODELS:
        model = load_any(m)
        for i in ids:
            if m in ("yolox_s", "dino_swin_l"):
                fn = os.path.join(REPO_ROOT, "data/coco/val2017", coco.loadImgs(i)[0]["file_name"])
                dets[m]["clean"] += to_coco_dets(inference_detector(model, fn), i, cat_ids)
                for c in conds:
                    dets[m][c] += to_coco_dets(inference_detector(model, path[c](i)), i, cat_ids)
            else:
                smp = dataset[pos[i]]
                x, ds = smp["inputs"].cuda(), smp["data_sample"]
                dets[m]["clean"] += to_coco_dets(predict(model, x, ds, rescale=True), i, cat_ids)
                for c in conds:
                    adv = torch.from_numpy(mmcv.imread(path[c](i))).permute(2, 0, 1).float().cuda()
                    dets[m][c] += to_coco_dets(predict(model, pipeline_resize(adv, x.shape[-2:]), ds, rescale=True), i, cat_ids)
        del model
        torch.cuda.empty_cache()
        print(f"[eval] {m} xong", flush=True)
    json.dump(dets, open(os.path.join(ART, "dets.json"), "w"))
    return coco, dets


def report(coco, dets, ids, conds):
    from scripts.err_profile import profile
    by = lambda d: {k: [x for x in d if x["image_id"] == k and x["score"] >= 0.3] for k in ids}
    n = len(ids)
    out = {}
    print(f"\n=== AP drop % ({n} ảnh, không CI) ===\n{'cond':<11}" + "".join(f"{m:>12}" for m in MODELS))
    for c in conds:
        row = {}
        for m in MODELS:
            ap0 = CocoImageEval(coco, dets[m]["clean"], ids).ap(np.ones(n))[0]
            ap = CocoImageEval(coco, dets[m][c], ids).ap(np.ones(n))[0]
            row[m] = 100 * (ap0 - ap) / ap0
        out[c] = {"ap_drop": row}
        print(f"{c:<11}" + "".join(f"{row[m]:12.1f}" for m in MODELS))
    print(f"\n=== Hồ sơ lỗi: recall nhỏ / vừa / lớn | conf | FP / ảnh | FP score ≥ 0.5 ===")
    for m in ("dino_swin_l", "swin_t", "r50"):
        clean_prof, clean_hits = profile(coco, by(dets[m]["clean"]), ids)
        print(f"{m}  sạch: " + "/".join(f"{clean_prof['recall'][b]:.2f}" for b in ("nhỏ", "vừa", "lớn"))
              + f" | – | {clean_prof['fp_per_img'].get('tổng', 0):.1f} | {clean_prof['fp_per_img'].get('score ≥0.5', 0):.1f}")
        for c in conds:
            r, _ = profile(coco, by(dets[m][c]), ids, clean_hits)
            out[c].setdefault("profile", {})[m] = r
            print(f"{m}  {c:<10}" + "/".join(f"{r['recall'][b]:.2f}" for b in ("nhỏ", "vừa", "lớn"))
                  + f" | {r['conf_ratio'] or 0:.2f} | {r['fp_per_img'].get('tổng', 0):.1f} | {r['fp_per_img'].get('score ≥0.5', 0):.1f}")
    json.dump(out, open(os.path.join(REPO_ROOT, "results/mechanism/wb_probe_dev10.json"), "w"), indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--subset", default="dev10")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv"))
    split = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    conds = ["OSFD", "C2-OCT-1V", "WB-OSFD", "WB-CLS"]
    if not args.eval_only:
        gen(ids, split, args.budget)
    coco, dets = evaluate(ids, conds)
    report(coco, dets, ids, conds)


if __name__ == "__main__":
    main()
