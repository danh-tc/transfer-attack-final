#!/usr/bin/env python3
"""E78-SMR (progress_log 2026-10-31): ENS-WP Momentum Attribution — DIAGNOSTIC, KHÔNG dựng method.
RQ: ENS-WP dùng phần BỔ SUNG nào của ConvNeXt, LÚC NÀO, và phần hiệu quả đó có recover được bằng quy tắc CHỈ-R50
(optimization dynamics) không? Replay ĐÚNG quỹ đạo ENS-WP (18 block xen kẽ R50/ConvNeXt, momentum chung — config e71),
tại mỗi block ConvNeXt tính COUNTERFACTUAL: "R50 sẽ update thế nào tại ĐÚNG state (delta,mom) này?" (cùng seed/augmentation).
Đo: rotation cos(u_C,u_R), thành phần TRỰC GIAO mới ‖u_C⊥u_R‖/‖u_R‖ (thông tin bổ sung), mag-ratio, theo thời điểm;
và tín hiệu CHỈ-R50 (độ tự-lặp các update R50 trước) có DỰ BÁO đóng góp ConvNeXt không (gate single-surrogate).
Counterfactual chỉ đo hình học update cục bộ — KHÔNG suy AP-drop. Tự validate: final delta ảnh 0 khớp PNG ENS-WP (±1).
  python scripts/e78_momentum_attr.py   -> results/mechanism/e78_momentum_attr.json"""
import json
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.targets import target_dataset
from scripts.e71_ensemble_wp import COMMON, KW_WP, BLOCK, STEPS

OUT = os.path.join(REPO_ROOT, "results/mechanism/e78_momentum_attr.json")
ORDER = ["r50", "convnext_t"]
NBLOCK = STEPS // BLOCK  # 18
HISTK = 3  # số update R50 gần nhất để tính tự-lặp


def flat(a):
    return a.detach().flatten().double()


def cosf(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def run_block(models, key, orig, ds_s, delta_in, mom_in, seed):
    """1 block (BLOCK bước) từ (delta_in, mom_in). Seed CẢ torch LẪN python-random (OCT elem dùng random) ->
    quỹ đạo trung thực + tái lập được; counterfactual cùng seed = cùng augmentation (so sánh sạch)."""
    torch.manual_seed(seed); random.seed(seed)
    return geo_osfd_attack(models[key], orig, ds_s["data_sample"], steps=BLOCK,
                           init_noise=delta_in, mom_in=mom_in, return_state=True, **KW_WP[key], **COMMON)


def main():
    dev = "cuda:0"
    # PNG ENS-WP trên đĩa do E73 sinh theo thứ tự dev100 (seed = 1000*(n_dev100+1)+b) -> replay PHẢI theo dev100.
    # Phân tích 30 ảnh ĐẦU của dev100 (n = vị trí dev100, khớp seed đã sinh).
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    assert len(ids) == 30
    models = {k: load_model(k, dev) for k in ORDER}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in ORDER}

    rec = {"block": [], "frac_t": [], "rotation": [], "orth_frac": [], "mag_ratio": [], "hist_selfsim": []}
    vdiag = []
    for n, iid in enumerate(ids):
        orig = ds["r50"][iid]["orig_inputs"].to(dev).float()
        # --- PHA 1: replay SẠCH (đúng e71, KHÔNG xen counterfactual -> không desync RNG), ghi state mỗi block ---
        delta, mom = None, None
        states = []  # (b, key, d_before, mom_before (chỉ CNX), seed, u_cpu)
        for b in range(NBLOCK):
            key = ORDER[b % 2]
            d_before = torch.zeros_like(orig) if delta is None else delta.clone()
            mom_before = None if (mom is None or key == "r50") else mom.clone()
            seed = 1000 * (n + 1) + b
            delta, mom = run_block(models, key, orig, ds[key][iid], delta, mom, seed)
            states.append((b, key, d_before if key != "r50" else None, mom_before, seed, flat(delta - d_before).cpu()))
        if n < 3:  # SANITY: quỹ đạo là ENS-WP hợp lệ (ε-ball + có nhiễu thực). KHÔNG khớp PNG cũ (bản gốc dùng
            # random KHÔNG seed per-block -> không tái lập giữa process); replay này là mẫu ENS-WP trung thực, đã seed.
            d = delta.abs()
            mad = float(d.mean()); mx = float(d.max())
            vdiag.append({"iid": iid, "mean_abs_delta": round(mad, 3), "max_abs_delta": round(mx, 3)})
            assert mx <= 5.0 + 1e-3 and mad > 1.0, f"delta bất thường (mean {mad}, max {mx}) -> dừng"
            print(f"[validate {iid}] ENS-WP replay: mean|δ|={mad:.2f} max|δ|={mx:.2f} (ε-ball OK)", flush=True)
        # --- PHA 2: counterfactual R50 tại ĐÚNG (d_before, mom_before, seed) từng block ConvNeXt (không đụng quỹ đạo thật) ---
        r50_u = []
        for (b, key, d_before, mom_before, seed, u) in states:
            if key == "r50":
                r50_u.append(u)
                continue
            cf_delta, _ = run_block(models, "r50", orig, ds["r50"][iid], d_before, mom_before, seed)
            uR = flat(cf_delta - d_before).cpu(); uC = u
            proj = (uC @ uR) / (uR @ uR + 1e-12) * uR
            hist = float("nan")
            if len(r50_u) >= 2:
                ru = r50_u[-HISTK:]
                hist = float(np.mean([cosf(ru[i], ru[j]) for i in range(len(ru)) for j in range(i + 1, len(ru))]))
            rec["block"].append(b); rec["frac_t"].append(b / (NBLOCK - 1))
            rec["rotation"].append(cosf(uC, uR))
            rec["orth_frac"].append(float((uC - proj).norm() / (uR.norm() + 1e-12)))
            rec["mag_ratio"].append(float(uC.norm() / (uR.norm() + 1e-12)))
            rec["hist_selfsim"].append(hist)
        print(f"[e78] {n + 1}/30", flush=True)

    R = {k: np.array(v, float) for k, v in rec.items()}

    def mean_by_phase(metric):
        out = {}
        for ph, sel in [("early", R["frac_t"] < 1 / 3), ("mid", (R["frac_t"] >= 1 / 3) & (R["frac_t"] < 2 / 3)), ("late", R["frac_t"] >= 2 / 3)]:
            out[ph] = round(float(np.nanmean(R[metric][sel])), 4)
        return out

    valid = ~np.isnan(R["hist_selfsim"])

    def pearson(a, b):
        a, b = a[valid], b[valid]
        if len(a) < 3:
            return 0.0
        return round(float(np.corrcoef(a, b)[0, 1]), 4)

    res = {"n_images": 30, "n_cnx_blocks": int(len(R["block"])), "validation": vdiag,
           "overall": {m: round(float(np.nanmean(R[m])), 4) for m in ("rotation", "orth_frac", "mag_ratio")},
           "by_phase": {m: mean_by_phase(m) for m in ("rotation", "orth_frac", "mag_ratio")},
           "corr_hist_selfsim": {"vs_orth_frac": pearson(R["hist_selfsim"], R["orth_frac"]),
                                 "vs_rotation": pearson(R["hist_selfsim"], R["rotation"]),
                                 "n_valid": int(valid.sum())}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)

    print("\n=== E78 Momentum Attribution (ConvNeXt block vs counterfactual R50) ===")
    print("overall:", res["overall"])
    for m in ("rotation", "orth_frac", "mag_ratio"):
        print(f"  {m:10} by phase: {res['by_phase'][m]}")
    print("corr(R50 hist self-sim, ConvNeXt orth_frac) =", res["corr_hist_selfsim"]["vs_orth_frac"],
          "| vs rotation =", res["corr_hist_selfsim"]["vs_rotation"], f"(n={res['corr_hist_selfsim']['n_valid']})")

    print("\n--- GO/NO-GO (khóa trước khi đọc) ---")
    mech = (res["overall"]["rotation"] < 0.70) and (res["overall"]["orth_frac"] >= 0.30)
    single = abs(res["corr_hist_selfsim"]["vs_orth_frac"]) >= 0.40
    print(f"Mechanistic GO (rotation<0.70 & orth_frac≥0.30): {mech}  [rot {res['overall']['rotation']}, orth {res['overall']['orth_frac']}]")
    print(f"Single-surrogate GO (|corr hist->orth| ≥ 0.40): {single}  [corr {res['corr_hist_selfsim']['vs_orth_frac']}]")
    print("\nVERDICT:",
          "E78 FULL GO — ConvNeXt bổ sung hướng mới (không chỉ magnitude) VÀ tín hiệu R50-only dự báo được -> đáng xây controller SMR (E79, kiểm causal)"
          if mech and single else
          "E78 MECHANISTIC-ONLY (ConvNeXt đóng góp hướng mới nhưng R50-history KHÔNG dự báo được) -> KHÔNG xây SMR; dùng attribution cho paper cơ chế"
          if mech else
          "E78 NO-GO — ConvNeXt chủ yếu đổi magnitude, không có đóng góp hướng tách bạch")


if __name__ == "__main__":
    main()
