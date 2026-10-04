#!/usr/bin/env python3
"""DINO-Swin-L làm failure case: PIA phá được Swin-L qua đâu? (quy tắc khóa: progress_log 2026-10-04 "Pivot (user):
DINO-Swin-L làm failure case"). Chỉ forward, PNG có sẵn, dev30.

Ảnh sạch vs adv qua cùng test pipeline (như mech_swinl.py): backbone stage 1–4, ChannelMapper 5 mức, memory
encoder (toàn bộ + tách 5 mức theo spatial_shapes). cosd = 1 − cos, rel = ‖a − c‖/‖c‖ mỗi ảnh.
A_enc(ảnh) = cosd_encoder / cosd_backbone stage 4. Cặp chính P1 = C2-PIA vs C2, P2 = PIA vs GEO-C0.

  python scripts/mech_swinl_geo.py   -> results/ablation/mech/swinl_geo.json
"""
import glob
import json
import os
import sys

import mmcv
import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.evaluation import bootstrap_indices
from attack.preprocess import pipeline_resize, to_batch

RUN = os.path.join(REPO_ROOT, "results/runs/dev300_B50_eps5")
ART = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5")
CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py")
C2 = "PROBE-osfd+cls"
CONDS = ["OSFD", "GEO-C0", "PIA", "PIA-aff", "PIA-d0.3", C2, "C2-PIA", "MORGD", "SMOOTH", "C2-MORGD", "C2-SMOOTH"]
PAIRS = {"P1": ("C2-PIA", C2), "P2": ("PIA", "GEO-C0")}
GEN = ["pia", "pia2", "morgd", "morgd0"]  # generalization_<tag>_dev30.json chứa AP drop của các ô trên
BOOT_SEED, N_BOOT = 2026, 1000


@torch.no_grad()
def levels(model, pix, ds):
    """-> {tầng: tensor} : bb1..bb4, neck1..neck5, enc (toàn bộ), enc1..enc5."""
    x, dss = to_batch(model, [pix], [ds])
    bb = model.backbone(x)
    neck = model.neck(bb)
    enc_in, _ = model.pre_transformer(neck, dss)
    mem = model.forward_encoder(**enc_in)["memory"][0]
    out = {f"bb{i + 1}": f for i, f in enumerate(bb)}
    out.update({f"neck{i + 1}": f for i, f in enumerate(neck)})
    out["enc"] = mem
    sizes = (enc_in["spatial_shapes"][:, 0] * enc_in["spatial_shapes"][:, 1]).tolist()
    out.update({f"enc{i + 1}": m for i, m in enumerate(torch.split(mem, sizes, dim=0))})
    return out


def dist(a, c):
    a, c = a.flatten().double(), c.flatten().double()
    return float(1 - torch.dot(a, c) / (a.norm() * c.norm())), float((a - c).norm() / c.norm())


def boot_diff(a, b, W):
    """Δ = mean(a) − mean(b), paired, W [1 + n_boot, n] (hàng 0 = mẫu gốc)."""
    d = (W @ (a - b)) / W.sum(1)
    return {"delta": float(d[0]), "ci95": [float(np.percentile(d[1:], 2.5)), float(np.percentile(d[1:], 97.5))],
            "rel": float(a.mean() / b.mean())}


def main():
    dev = "cuda:0"
    from mmdet.apis import init_detector
    model = init_detector(CFG, glob.glob(os.path.join(REPO_ROOT, "checkpoints", "dino-5scale_swin-l*.pth"))[0], device=dev)
    data = build_attack_dataset("dev300")
    pos = {img_id: i for i, img_id in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    rec = {c: {} for c in CONDS}  # rec[c][tầng] = [n, 2] (cosd, rel)
    for j, img_id in enumerate(img_ids):
        s = data[pos[img_id]]
        x, ds = s["inputs"].to(dev), s["data_sample"]
        clean = levels(model, x, ds)
        for c in CONDS:
            a = pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(ART, "adv", c, f"{img_id}.png")))
                                .permute(2, 0, 1).float().to(dev), x.shape[-2:])
            for lvl, fa in levels(model, a, ds).items():
                rec[c].setdefault(lvl, []).append(dist(fa, clean[lvl]))
        if (j + 1) % 10 == 0:
            print(f"[swinl_geo] {j + 1}/{len(img_ids)}", flush=True)

    n = len(img_ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    cosd = {c: {l: np.array([v[0] for v in vs]) for l, vs in rec[c].items()} for c in CONDS}
    for c in CONDS:
        cosd[c]["A_enc"] = cosd[c]["enc"] / cosd[c]["bb4"]
        for i in range(1, 6):
            cosd[c][f"A{i}"] = cosd[c][f"enc{i}"] / cosd[c][f"neck{i}"]
    lv = list(cosd[CONDS[0]].keys())

    drop = {}
    for tag in GEN:
        with open(os.path.join(RUN, f"generalization_{tag}_dev30.json")) as f:
            for c, v in json.load(f)["relative_ap_drop"].items():
                drop.setdefault(c, v["dino_swin_l"]["point"])

    out = {"n_images": n, "conds": CONDS, "pairs": {k: list(v) for k, v in PAIRS.items()},
           "dino_swin_l_drop": {c: drop[c] for c in CONDS},
           "mean_cosd": {c: {l: float(cosd[c][l].mean()) for l in lv} for c in CONDS},
           "mean_rel": {c: {l: float(np.mean([v[1] for v in rec[c][l]])) for l in rec[c]} for c in CONDS},
           "pair_tests": {}, "verdict": {}}
    for p, (a, b) in PAIRS.items():
        t = {l: boot_diff(cosd[a][l], cosd[b][l], W) for l in lv}
        shallow = [s for s in ("bb1", "bb2") if t[s]["ci95"][0] > 0 and t[s]["rel"] >= 1.25 and t[s]["rel"] >= t["bb4"]["rel"]]
        out["pair_tests"][p] = t
        out["verdict"][p] = {"SHALLOW_SPECIFIC": bool(shallow), "shallow_stages": shallow,
                             "ENCODER": bool(t["A_enc"]["ci95"][0] > 0)}
    sh = all(v["SHALLOW_SPECIFIC"] for v in out["verdict"].values())
    en = all(v["ENCODER"] for v in out["verdict"].values())
    out["verdict"]["conclusion"] = ("SHALLOW-SPECIFIC ở cả P1, P2 -> ủng hộ tiền đề đa scale" if sh else
                                    "ENCODER ở cả P1, P2, không SHALLOW -> một phần, tiền đề đa scale không được ủng hộ" if en else
                                    "không SHALLOW, không ENCODER ở cả hai cặp -> bác giả thuyết")
    dvec = np.array([drop[c] for c in CONDS])
    out["across_conds_spearman_vs_swinl_drop"] = {l: float(spearmanr([out["mean_cosd"][c][l] for c in CONDS], dvec)[0]) for l in lv}

    path = os.path.join(REPO_ROOT, "results/ablation/mech/swinl_geo.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    show = ["bb1", "bb2", "bb3", "bb4", "neck1", "neck5", "enc", "enc1", "enc2", "enc3", "enc4", "enc5", "A_enc"]
    print(f"{'ô':<16}{'drop':>6} " + " ".join(f"{l:>7}" for l in show))
    for c in CONDS:
        print(f"{c:<16}{drop[c]:6.1f} " + " ".join(f"{out['mean_cosd'][c][l]:7.3f}" for l in show))
    print("ρ giữa ô vs drop  " + " ".join(f"{out['across_conds_spearman_vs_swinl_drop'][l]:+7.2f}" for l in show))
    for p, t in out["pair_tests"].items():
        print(f"\n{p} {PAIRS[p][0]} − {PAIRS[p][1]}: " + "  ".join(
            f"{l} Δ{t[l]['delta']:+.3f} [{t[l]['ci95'][0]:+.3f}, {t[l]['ci95'][1]:+.3f}] ×{t[l]['rel']:.2f}" for l in show))
        print(f"   {out['verdict'][p]}")
    print("\nKẾT LUẬN:", out["verdict"]["conclusion"])


if __name__ == "__main__":
    main()
