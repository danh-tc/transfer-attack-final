#!/usr/bin/env python3
"""Sanity gate (bước 0) của pilot KG — quy tắc LOCKED: docs/progress_log.md 2026-10-03.
Chỉ kiểm tính đúng kỹ thuật + tính khả thi của phép thử; KHÔNG đo transfer.

20 ảnh có thông tin (>= 2 cặp hợp lệ) đầu tiên theo thứ tự dev100.csv. Seed ảnh = SEED + chỉ số
trong dev300 (đúng seed mà lần sàng lọc dev300 sẽ dùng -> chuỗi cặp kiểm ở đây = chuỗi thật).
- S1: crop chứa trọn 2 GT box (mọi cặp); feature crop đúng lưới stride 4/8/16/32; tại δ = 0,
      không nhiễu, loss view cặp = (k−1)²·mean F_cln² (sai số tương đối < 1e-5).
- S2: tỉ lệ step có cặp E1 ≠ cặp E2 (gộp 20 ảnh × 50 step, chuỗi THẬT của lần chạy) >= 0.5.
- S3: trung vị |log(A_E1/A_E2)| (gộp các step) <= log 1.5.
- S4: sign agreement(∇E2, ∇E0) tại step 0, trung bình 20 ảnh < 0.95. Sau 1 step MI, δ = α·sign(g)
      nên so dấu δ sau 1 step = so dấu gradient.
Chẩn đoán (không thuộc quy tắc): tỉ lệ cặp cùng class của E1/E2; P(E1≠E2) theo ảnh; chuỗi (q, e1)
của lần chạy E1 và E2 trùng nhau; sàn nhiễu sign agreement E0 vs E0 khác seed; E1 vs E2 ở step 0.

Chạy: python scripts/pilot_kg_gate.py   -> results/kg_pilot/dev300/sanity_gate.json
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.losses.osfd import make_osfd_loss_fn
from attack.methods.baselines import osfd_attack
from attack.methods.kg_pair import PairSampler, crop_resize, kg_osfd_attack
from attack.models import load_surrogate
from attack.preprocess import extract_features, resize_to_model

SEED = 0  # = run_baselines.py
N_IMG, STEPS, K = 20, 50, 3.0
EPS = 5.0


def seeded(seed):
    random.seed(seed); torch.manual_seed(seed)


def main():
    dev = "cuda:0"
    model = load_surrogate(dev)
    data = build_attack_dataset("dev300")
    order = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    pos = {img_id: i for i, img_id in enumerate(data.image_ids)}

    s1_contain, s1_shape, s1_rel = True, True, 0.0
    steps_log, per_img, sign_e2_e0, sign_floor, sign_e2_e1, seq_equal = [], [], [], [], [], True
    picked = []
    t0 = time.time()
    for img_id in order:
        if len(picked) == N_IMG:
            break
        i = pos[img_id]
        s = data[i]
        ds = s["data_sample"]
        sampler = PairSampler(ds, seed=SEED + i)
        if sampler.n_pairs < 2:
            continue
        picked.append(img_id)
        orig = s["orig_inputs"].to(dev)
        clean = resize_to_model(orig, tuple(ds.img_shape))

        # S1 — crop chứa trọn 2 box, mọi cặp.
        b = sampler.boxes
        for c, (a, bb) in zip(sampler.crop, zip(sampler.I, sampler.J)):
            for box in (b[a], b[bb]):
                s1_contain &= bool(c[0] <= box[0] + 1e-4 and c[1] <= box[1] + 1e-4
                                   and c[2] >= box[2] - 1e-4 and c[3] >= box[3] - 1e-4)
        # S1 — 1 cặp cố định (cặp 0): lưới feature + loss tại δ = 0 không nhiễu.
        crop_clean = crop_resize(clean, sampler.crop[0])
        with torch.no_grad():
            feats = extract_features(model, crop_clean, ds)
            h, w = crop_clean.shape[-2:]
            ph, pw = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
            for f, stride in zip(feats, (4, 8, 16, 32)):
                s1_shape &= tuple(f.shape[-2:]) == (ph // stride, pw // stride)
            loss = make_osfd_loss_fn(model, crop_clean, ds, k=K)(
                model, crop_resize(resize_to_model(orig, tuple(ds.img_shape)), sampler.crop[0]), ds)
            expect = sum((K - 1) ** 2 * f.pow(2).mean() for f in feats)
            s1_rel = max(s1_rel, abs(loss - expect).item() / expect.item())

        # S2/S3 — chuỗi THẬT của 50 step, chạy cả E1 và E2 (cùng seed như dev300).
        logs = {}
        for mode in ("random_matched", "kg"):
            info = {}
            seeded(SEED + i)
            noise = kg_osfd_attack(model, orig, ds, steps=STEPS, mode=mode, pair_seed=SEED + i,
                                   epsilon=EPS, info=info)
            assert float(noise.abs().max()) <= EPS + 1e-4
            logs[mode] = info["sampler"].log
        seq = [(r["q"], r["e1"]) for r in logs["kg"]]
        seq_equal &= seq == [(r["q"], r["e1"]) for r in logs["random_matched"]]
        steps_log += logs["kg"]
        per_img.append({
            "img_id": img_id, "dev300_index": i, "n_pairs": sampler.n_pairs,
            "kg_degenerate": sampler.kg_degenerate,
            "p_E1_ne_E2": float(np.mean([r["q"] != r["e1"] for r in logs["kg"]])),
            "median_abs_log_area_ratio": float(np.median([r["abs_log_area_ratio"] for r in logs["kg"]])),
            "same_class_E2": float(np.mean([r["same_class_q"] for r in logs["kg"]])),
            "same_class_E1": float(np.mean([r["same_class_e1"] for r in logs["kg"]])),
        })

        # S4 — dấu δ sau 1 step (= dấu gradient step 0).
        def one_step(fn, seed, **kw):
            seeded(seed)
            return torch.sign(fn(model, orig, ds, steps=1, epsilon=EPS, **kw))
        d_e0 = one_step(osfd_attack, SEED + i)
        d_e2 = one_step(kg_osfd_attack, SEED + i, mode="kg", pair_seed=SEED + i)
        d_e1 = one_step(kg_osfd_attack, SEED + i, mode="random_matched", pair_seed=SEED + i)
        d_e0b = one_step(osfd_attack, SEED + i + 100000)
        agree = lambda a, b: float((a == b).float().mean())
        sign_e2_e0.append(agree(d_e2, d_e0)); sign_floor.append(agree(d_e0, d_e0b))
        sign_e2_e1.append(agree(d_e2, d_e1))
        r = per_img[-1]
        print(f"[gate] {len(picked)}/{N_IMG} img {img_id} (dev300 #{i}, {sampler.n_pairs} cặp)  "
              f"P(E1≠E2) {r['p_E1_ne_E2']:.2f}  |logA| med {r['median_abs_log_area_ratio']:.2f}  "
              f"sign E2~E0 {sign_e2_e0[-1]:.3f} (sàn E0~E0' {sign_floor[-1]:.3f})  "
              f"{time.time() - t0:.0f}s", flush=True)

    s2 = float(np.mean([r["q"] != r["e1"] for r in steps_log]))
    s3 = float(np.median([r["abs_log_area_ratio"] for r in steps_log]))
    s4 = float(np.mean(sign_e2_e0))
    checks = {
        "S1_crop_contains_both_boxes": s1_contain,
        "S1_feature_grid_ok": s1_shape,
        "S1_loss_at_zero_max_rel_diff": s1_rel, "S1_loss_ok": s1_rel < 1e-5,
        "S2_frac_steps_E1_ne_E2": s2, "S2_pass": s2 >= 0.5,
        "S3_median_abs_log_area_ratio": s3, "S3_threshold": math.log(1.5), "S3_pass": s3 <= math.log(1.5),
        "S4_sign_agreement_E2_vs_E0_step0": s4, "S4_pass": s4 < 0.95,
    }
    checks["S1_pass"] = s1_contain and s1_shape and checks["S1_loss_ok"]
    checks["all_pass"] = all(checks[k] for k in ("S1_pass", "S2_pass", "S3_pass", "S4_pass"))
    out = {
        "rule": "progress_log 2026-10-03 (pilot KG, LOCKED)", "n_images": len(picked), "steps": STEPS,
        "images": picked, "checks": checks,
        "diagnostics": {
            "pair_sequence_identical_E1_E2_runs": seq_equal,
            "same_class_rate_E2": float(np.mean([r["same_class_q"] for r in steps_log])),
            "same_class_rate_E1": float(np.mean([r["same_class_e1"] for r in steps_log])),
            "sign_agreement_floor_E0_vs_E0_other_seed": float(np.mean(sign_floor)),
            "sign_agreement_E2_vs_E1_step0": float(np.mean(sign_e2_e1)),
            "per_image": per_img,
        },
    }
    path = os.path.join(REPO_ROOT, "results/kg_pilot/dev300/sanity_gate.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({"checks": checks, **{k: v for k, v in out["diagnostics"].items() if k != "per_image"}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
