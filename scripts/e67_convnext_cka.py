#!/usr/bin/env python3
"""E67 (progress_log 2026-10-31): vì sao surrogate ConvNeXt bắc cầu Transformer mà tụt ResNet? — cơ chế biểu diễn, forward-only.
Giả thuyết: ConvNeXt (CNN macro-design kiểu Swin: kernel 7x7 depthwise, LayerNorm, patchify) có BIỂU DIỄN gần Swin hơn gần
ResNet -> δ tối ưu trên ConvNeXt rơi vào subspace Transformer chia sẻ (transfer mạnh sang Swin-L) nhưng lệch subspace ResNet.
Đo trên ảnh SẠCH dev30, cùng một pixel đầu vào (resize về khung ConvNeXt) qua backbone của 5 model -> cùng lưới stride 4/8/16/32:
  linear CKA từng cặp model, từng stage (1..4). Trục chính: CKA(ConvNeXt, {swin_t, dino_swin_l}) vs CKA(ConvNeXt, {r50, r101}).
Kèm: liên hệ CKA(ConvNeXt, target_backbone) với transfer δ_CNX (drop E63) và δ_R50 (drop E62) trên 4 model có backbone trong bộ.
  python scripts/e67_convnext_cka.py   -> results/mechanism/e67_convnext_cka.json"""
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.mechanism import linear_cka
from attack.models import CONTROLLED_PANEL, load_model
from attack.preprocess import extract_features, resize_to_model

OUT = os.path.join(REPO_ROOT, "results/mechanism/e67_convnext_cka.json")
MODELS = ["r50", "r101", "convnext_t", "swin_t", "dino_swin_l"]
ANCHOR = "convnext_t"
N_STAGE = 4
BOOT_SEED, N_BOOT = 2026, 1000


def load_any(key, device):
    if key in CONTROLLED_PANEL:
        return load_model(key, device)
    from mmdet.apis import init_detector
    import glob
    from scripts.eval_generalization import GEN_PANEL
    cfg, pat = GEN_PANEL[key]
    ckpt = glob.glob(os.path.join(REPO_ROOT, "checkpoints", pat + ".pth"))[0]
    return init_detector(os.path.join(REPO_ROOT, "third_party/mmdetection/configs", cfg), ckpt, device="cuda:0")


@torch.no_grad()
def main():
    dev = "cuda:0"
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    data = build_attack_dataset("dev300")
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    pos = {iid: k for k, iid in enumerate(data.image_ids)}
    by_id = {iid: data[pos[iid]] for iid in ids}
    models = {m: load_any(m, dev) for m in MODELS}
    pairs = [(a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:]]
    # cka[stage][pair] = list theo ảnh
    cka = [{f"{a}|{b}": [] for a, b in pairs} for _ in range(N_STAGE)]
    for n, iid in enumerate(ids):
        s = by_id[iid]
        orig = s["orig_inputs"].to(dev).float()
        # cùng một pixel đầu vào cho mọi model: khung ConvNeXt (anchor) -> lưới stage trùng nhau.
        px = resize_to_model(orig, tuple(s["data_sample"].img_shape))
        feats = {}
        for m in MODELS:
            f = extract_features(models[m], px, s["data_sample"])
            feats[m] = [x.float() for x in f[:N_STAGE]]
        for a, b in pairs:
            for st in range(N_STAGE):
                fa, fb = feats[a][st], feats[b][st]
                if fa.shape[-2:] != fb.shape[-2:]:  # lệch lưới (model resize khác) -> nội suy fb về lưới fa
                    fb = torch.nn.functional.interpolate(fb, size=fa.shape[-2:], mode="bilinear", align_corners=False)
                cka[st][f"{a}|{b}"].append(linear_cka(fa, fb))
        del feats
        torch.cuda.empty_cache()
        print(f"[e67] {n + 1}/30", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.integers(0, 30, size=(N_BOOT, 30))

    def summ(v):
        v = np.array(v)
        bs = v[boot].mean(1)
        return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

    res = {"n_images": 30, "anchor": ANCHOR, "note": "CKA backbone trên ảnh sạch, cùng pixel đầu vào; [point, ci_lo, ci_hi]",
           "cka_by_stage": [{p: summ(cka[st][p]) for p in cka[st]} for st in range(N_STAGE)]}
    # Trục chính: ConvNeXt gần Transformer hay ResNet? (stage sâu 3-4, trung bình)
    def anchored(target, st):
        p = f"{ANCHOR}|{target}" if f"{ANCHOR}|{target}" in cka[st] else f"{target}|{ANCHOR}"
        return np.array(cka[st][p])
    deep = {t: [summ(np.mean([anchored(t, st) for st in (2, 3)], 0))] for t in MODELS if t != ANCHOR}
    res["convnext_vs_target_deep"] = {t: v[0] for t, v in deep.items()}
    # Hiệu (ConvNeXt~Swin) − (ConvNeXt~ResNet), stage sâu, paired theo ảnh
    swin = np.mean([anchored("swin_t", st) for st in (2, 3)], 0)
    swinl = np.mean([anchored("dino_swin_l", st) for st in (2, 3)], 0)
    r50 = np.mean([anchored("r50", st) for st in (2, 3)], 0)
    r101 = np.mean([anchored("r101", st) for st in (2, 3)], 0)
    diff = (swin + swinl) / 2 - (r50 + r101) / 2
    res["swin_minus_resnet_deep"] = summ(diff)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"convnext_vs_target_deep": res["convnext_vs_target_deep"],
                      "swin_minus_resnet_deep": res["swin_minus_resnet_deep"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
