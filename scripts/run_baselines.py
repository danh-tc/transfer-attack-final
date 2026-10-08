#!/usr/bin/env python3
"""Baseline table chính thức (idea.md §8-10): surrogate R50 -> 4 model Controlled Panel.

2 giai đoạn, đều resume được (chạy lại lệnh y hệt sau khi bị ngắt):
1. attack: mỗi (method, ảnh) sinh ảnh adversarial THẬT (cỡ gốc, uint8) -> PNG lossless
   ở artifacts/runs/<run>/adv/<method>/<img_id>.png (gitignored), ghi thống kê từng ảnh
   (L_inf, L2, runtime, số backward/forward-view) vào results/runs/<run>/attack_stats.jsonl.
   Ảnh đã có PNG + dòng stats thì bỏ qua.
2. eval: đọc PNG từ đĩa -> Resize đúng test pipeline -> predict trên 4 model -> COCO bbox
   AP/AP50 + paired bootstrap 95% CI (resample ảnh, chung cho mọi điều kiện/model) ->
   results/runs/<run>/metrics.json. Detection thô lưu artifacts/runs/<run>/dets.json.

Hyperparameter baseline: docs/protocol_lock.md ("Baseline hyperparameter").

Chạy (trong tmux):
  python scripts/run_baselines.py --split n300 --budget 50
  python scripts/run_baselines.py --split n300 --budget 50 --stage eval   # chỉ eval lại
"""
import argparse
import functools
import json
import os
import random
import sys
import time

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.methods.baselines import di_fgsm_attack, mi_fgsm_attack, osfd_attack
from attack.methods.kg_pair import kg_osfd_attack
from attack.methods.pia import geo_osfd_attack
from attack.methods.repair import repair_robust_attack
from attack.methods.sgm import GAMMAS, sgm_osfd_attack
from attack.models import CONTROLLED_PANEL, SURROGATE_KEY, load_model
from attack.preprocess import pipeline_resize, predict, to_adv_image

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
SEED = 0
BOOT_SEED = 2026
SAME_FAMILY, CROSS_CNN, CROSS_TR = "r101", "convnext_t", "swin_t"

# name -> (hàm attack, số view forward qua surrogate mỗi backward)
METHODS = {
    "MI-FGSM": (mi_fgsm_attack, 1),
    "M-DI2-FGSM": (di_fgsm_attack, 1),
    "OSFD": (osfd_attack, 2),  # RRB: 2 view / 1 backward (protocol_lock.md, đơn vị B)
    # Pilot VC (progress_log 2026-10-07): 2 backward / step (RRB độc lập), 25 bước, B = 50 giữ nguyên.
    "OSFD-K2": (functools.partial(osfd_attack, vc_k=2, combine="sum"), 2),
    "VC-OSFD": (functools.partial(osfd_attack, vc_k=2, combine="consensus"), 2),
    # K = 4: 12 bước × 4 backward = 48 (≤ B, bất lợi nhẹ cho ô mới).
    # Pilot V (progress_log 2026-10-07): r lần rút RRB độc lập gộp trong 1 backward, 50 bước (B = 50), view forward = 2r/bước.
    "OSFD-V4": (functools.partial(osfd_attack, rrb_draws=2), 4),
    "OSFD-V8": (functools.partial(osfd_attack, rrb_draws=4), 8),
    "OSFD-K4": (functools.partial(osfd_attack, vc_k=4, combine="sum"), 2),
    "VC4-OSFD": (functools.partial(osfd_attack, vc_k=4, combine="consensus"), 2),
    # Pilot A′1 (progress_log 2026-09-24) — không thuộc baseline table, chỉ chạy khi chỉ định.
    "OSFD-W1": (functools.partial(osfd_attack, spatial_weight="box"), 2),
    "OSFD-W2": (functools.partial(osfd_attack, spatial_weight="box_ring"), 2),
    # Pilot P2 (progress_log 2026-09-24).
    "OSFD-P2a": (functools.partial(osfd_attack, channel_concave="median"), 2),
    "OSFD-P2b": (functools.partial(osfd_attack, channel_concave="2median"), 2),
    # Pilot KG (progress_log 2026-10-03): view 2 của RRB -> crop cặp object. E1 = geometry-matched
    # random control, E2 = KG. Nhận thêm pair_seed = SEED + i (cùng seed ảnh).
    "OSFD-KG-E1": (functools.partial(kg_osfd_attack, mode="random_matched"), 2),
    "OSFD-KG-E2": (functools.partial(kg_osfd_attack, mode="kg"), 2),
}
KG_METHODS = {"OSFD-KG-E1", "OSFD-KG-E2"}
# Pilot R1 (progress_log 2026-10-03): OSFD-adapted SGM (decay chỉ trong nhánh residual). E2 (mix) nhận
# gamma_seed = SEED + i, chọn đều γ ∈ {0.2, 0.5} mỗi step.
METHODS.update({
    "OSFD-SGM-0.2": (functools.partial(sgm_osfd_attack, gamma=0.2), 2),
    "OSFD-SGM-0.5": (functools.partial(sgm_osfd_attack, gamma=0.5), 2),
    "OSFD-SGM-mix": (functools.partial(sgm_osfd_attack, gamma_choices=GAMMAS), 2),
})
SGM_MIX = {"OSFD-SGM-mix"}
# Ablation OSFD (docs/plan_sota.md, bước 1): bỏ/thay đúng 1 thành phần của OSFD full recipe.
METHODS.update({
    "ABL-task-loss": (functools.partial(osfd_attack, loss="task"), 2),
    "ABL-no-RRB": (functools.partial(osfd_attack, use_rrb=False), 1),
    "ABL-RRB-no-noise": (functools.partial(osfd_attack, rrb_noise=False), 2),
    "ABL-RRB-no-rotate": (functools.partial(osfd_attack, rrb_rotate=False), 2),
    "ABL-RRB-no-resize": (functools.partial(osfd_attack, rrb_resize=False), 2),
    "ABL-no-MI": (functools.partial(osfd_attack, momentum=0.0), 2),
    "ABL-k5": (functools.partial(osfd_attack, k=5.0), 2),
    # Ablation thành phần task loss (progress_log 2026-10-03, sau probe loại lỗi).
    "ABL-task-cls": (functools.partial(osfd_attack, loss="task", task_terms=("loss_rpn_cls", "loss_cls")), 2),
    "ABL-task-box": (functools.partial(osfd_attack, loss="task", task_terms=("loss_rpn_bbox", "loss_bbox")), 2),
    "ABL-task-mask": (functools.partial(osfd_attack, loss="task", task_terms=("loss_mask",)), 2),
    # Probe bổ trợ (progress_log 2026-10-03): OSFD + 4·(rpn_cls + cls), λ = 4 khóa trước theo tỉ lệ gradient.
    "PROBE-osfd+cls": (functools.partial(osfd_attack, loss="feature+cls", cls_weight=4.0), 2),
    # Nhánh S1 (progress_log 2026-10-03): C2 + lọc thông thấp gradient Gaussian, f0 khóa trước.
    "S1-lp6": (functools.partial(osfd_attack, loss="feature+cls", cls_weight=4.0, grad_lowpass_f0=1 / 6), 2),
    "S1-lp12": (functools.partial(osfd_attack, loss="feature+cls", cls_weight=4.0, grad_lowpass_f0=1 / 12), 2),
})
# Repair-Robust (progress_log 2026-10-04): probe V = C2 với L_V (chỉ đẩy GT về nền), λ_V = 0.3.
# Nhận repair_seed = SEED + i, info (bật/tắt sửa mỗi step, forward no_grad).
METHODS.update({
    "PROBE-V": (functools.partial(repair_robust_attack, cls="vanish", cls_weight=0.3, repair_p=0.0), 2),
})
# R-a: C2, C3 thay bằng sạch trước FPN với xác suất p mỗi step; thành phần phân loại theo quy tắc chọn của
# probe V (chỉ chạy biến thể được chọn). λ thừa kế: C2 = 4, V = 0.3.
METHODS.update({
    f"RA-{c}-p{p}": (functools.partial(repair_robust_attack, cls=cls, cls_weight=w, repair_p=p), 2)
    for c, cls, w in (("c2", "c2", 4.0), ("v", "vanish", 0.3)) for p in (0.5, 1.0)
})
RR_METHODS = {"PROBE-V"} | {m for m in METHODS if m.startswith("RA-")}
# Pilot PIA @ dev30 (progress_log 2026-10-04): loss feature OSFD + MI, feature sạch co-transform theo view.
# PIA-N20: 20 view/step = 4 backward × 5 view (B thực = 4 × budget).
METHODS.update({
    "GEO-C0": (functools.partial(geo_osfd_attack, geo="rrb"), 2),
    "PIA": (functools.partial(geo_osfd_attack, geo="pia"), 2),
    "PIA-d0.3": (functools.partial(geo_osfd_attack, geo="pia", delta_max=0.3), 2),
    "PIA-aff": (functools.partial(geo_osfd_attack, geo="pia", pi=(0.0, 1.0, 0.0, 0.0)), 2),
    "PIA-N20": (functools.partial(geo_osfd_attack, geo="pia", n_views=5, backward_per_step=4), 20),
    # Bổ sung (progress_log 2026-10-04): C2 với RRB -> PIA; PIA không co-transform feature sạch.
    "C2-PIA": (functools.partial(geo_osfd_attack, geo="pia", cls_weight=4.0), 2),
    "PIA-noCT": (functools.partial(geo_osfd_attack, geo="pia", co_transform=False), 2),
    # Pilot MORGD (progress_log 2026-10-04): khung C2-PIA, phép biến đổi = warp mượt neo theo object / ngẫu nhiên.
    "C2-MORGD": (functools.partial(geo_osfd_attack, geo="morgd", cls_weight=4.0), 2),
    "C2-SMOOTH": (functools.partial(geo_osfd_attack, geo="smooth", cls_weight=4.0), 2),
    # Bản thuần (không cls), khung như ô PIA — baseline so RRB.
    "MORGD": (functools.partial(geo_osfd_attack, geo="morgd"), 2),
    "SMOOTH": (functools.partial(geo_osfd_attack, geo="smooth"), 2),
    # RaPA (progress_log 2026-10-04): DropConnect BN backbone, p = 0.05, S = 1, cùng mask sạch/adv mỗi view.
    "MORGD-RaPA": (functools.partial(geo_osfd_attack, geo="morgd", rapa_p=0.05), 2),
    # Pilot GRGD-v0 (progress_log 2026-10-04): khung C2-PIA + T_relative sau PIA; neo tâm GT | tâm ngẫu nhiên.
    "C2-GRGD": (functools.partial(geo_osfd_attack, geo="grgd", cls_weight=4.0), 2),
    "C2-PIA-RESIDUAL": (functools.partial(geo_osfd_attack, geo="grgd_random", cls_weight=4.0), 2),
    # Cải tiến MORGD (progress_log 2026-10-04): MORGD-v2 phân vùng theo box, xếp chồng sau PIA | đối chứng | không PIA.
    "C2-PIA-MORGD2": (functools.partial(geo_osfd_attack, geo="pia_morgd2", cls_weight=4.0), 2),
    "C2-PIA-SMOOTH2": (functools.partial(geo_osfd_attack, geo="pia_smooth2", cls_weight=4.0), 2),
    "C2-MORGD2": (functools.partial(geo_osfd_attack, geo="morgd2", cls_weight=4.0), 2),
    # Giai đoạn A (progress_log 2026-10-04): PIA bỏ 1 thành phần homography (khung loss feature thuần).
    **{f"PIA-no{f}": (functools.partial(geo_osfd_attack, geo="pia", pia_drop=f), 2) for f in "TRSKP"},
    # Giai đoạn B (progress_log 2026-10-04): đa dạng mô hình camera (khung loss feature thuần).
    "CAM-R15": (functools.partial(geo_osfd_attack, geo="cam", cam_theta=15.0), 2),
    "CAM-R30": (functools.partial(geo_osfd_attack, geo="cam", cam_theta=30.0), 2),
    "LENS": (functools.partial(geo_osfd_attack, geo="cam", cam_theta=0.0, cam_kappa=0.25), 2),
    "CAM-R15+LENS": (functools.partial(geo_osfd_attack, geo="cam", cam_theta=15.0, cam_kappa=0.25), 2),
    # Quy luật "ngoài augmentation lúc train" + POOL (progress_log 2026-10-04).
    "NOISE": (functools.partial(geo_osfd_attack, geo="noise"), 2),
    "FLIP": (functools.partial(geo_osfd_attack, geo="flip"), 2),
    "POOL": (functools.partial(geo_osfd_attack, geo="pool"), 2),
})
# Quét liều phép biến đổi cơ bản quanh tâm (progress_log 2026-10-05): xoay (độ), trượt k, ln k lệch tỉ lệ, ln s co giãn.
ELEM_CELLS = {"ROT7": {"rot": 7.0}, "ROT15": {"rot": 15.0}, "ROT30": {"rot": 30.0},
              "SHR15": {"shear": 0.15}, "SHR30": {"shear": 0.30},
              "ANI10": {"aniso": 0.10}, "ANI25": {"aniso": 0.25}, "SCL25": {"scale": 0.25},
              # Kiểm "pha lưới" (progress_log 2026-10-07): chỉ tịnh tiến, bội 32 | lẻ | dưới 1 px.
              "SH32": {"shift": "s32"}, "SHODD": {"shift": "odd"}, "SHSUB": {"shift": "sub"},
              # Stride quyết định (progress_log 2026-10-08): bội 4 / 8 / 16 trong ±64.
              "SH4": {"shift": "s4"}, "SH8": {"shift": "s8"}, "SH16": {"shift": "s16"},
              "SH2": {"shift": "s2"}}  # tách stride 2 vs 4 (progress_log 2026-10-08)
METHODS.update({c: (functools.partial(geo_osfd_attack, geo="elem", elem=s), 2) for c, s in ELEM_CELLS.items()})
GEO_METHODS = {"GEO-C0", "PIA", "PIA-d0.3", "PIA-aff", "PIA-N20", "C2-PIA", "PIA-noCT", "C2-MORGD", "C2-SMOOTH", "MORGD", "SMOOTH", "MORGD-RaPA",
               "C2-GRGD", "C2-PIA-RESIDUAL", "C2-PIA-MORGD2", "C2-PIA-SMOOTH2", "C2-MORGD2",
               "PIA-noT", "PIA-noR", "PIA-noS", "PIA-noK", "PIA-noP", "CAM-R15", "CAM-R30", "LENS", "CAM-R15+LENS",
               "NOISE", "FLIP", "POOL"} | set(ELEM_CELLS)
BACKWARD_PER_STEP = {"PIA-N20": 4}
BASELINES = ["MI-FGSM", "M-DI2-FGSM", "OSFD"]


def load_stats(path):
    stats = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                stats[(r["method"], r["img_id"])] = r
    return stats


def img_tensor(path, device):
    return torch.from_numpy(mmcv.imread(path)).permute(2, 0, 1).float().to(device)


def stage_attack(args, dataset, surrogate, adv_dir, stats_path, device):
    stats = load_stats(stats_path)
    for name in args.methods:
        fn, views = METHODS[name]
        os.makedirs(os.path.join(adv_dir, name), exist_ok=True)
        t_start, n_done = time.time(), 0
        for i in range(len(dataset)):
            img_id = dataset.image_ids[i]
            png = os.path.join(adv_dir, name, f"{img_id}.png")
            if os.path.exists(png) and (name, img_id) in stats:
                continue
            sample = dataset[i]
            orig, ds = sample["orig_inputs"].to(device), sample["data_sample"]
            seed = SEED + dataset.seed_index[i]  # chỉ số trong split đầy đủ (giữ seed khi chạy --subset)
            random.seed(seed); torch.manual_seed(seed)
            t0 = time.time()
            info = {}
            kw = {"pair_seed": seed, "info": info} if name in KG_METHODS else {}
            if name in SGM_MIX:
                kw = {"gamma_seed": seed, "info": info}
            if name in RR_METHODS:
                kw = {"repair_seed": seed, "info": info}
            if name in GEO_METHODS:
                kw = {"info": info}
            noise = fn(surrogate, orig, ds, steps=args.budget, epsilon=args.epsilon, **kw)
            torch.cuda.synchronize()
            runtime = time.time() - t0
            adv = to_adv_image(orig, noise)
            delta = (adv - orig) / 255.0
            linf = float((adv - orig).abs().max())
            assert linf <= args.epsilon + 1e-4, f"{name} img {img_id}: L_inf {linf}"
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png)
            rec = {"method": name, "img_id": img_id, "linf_255": linf,
                   "l2": float(delta.norm()), "l2_per_pixel_rms": float(delta.pow(2).mean().sqrt()),
                   "runtime_s": runtime, "backward": args.budget * BACKWARD_PER_STEP.get(name, 1),
                   "forward_views": args.budget * views}
            if name in GEO_METHODS:
                rec["forward_no_grad"] = info["forward_no_grad"]
                if info["modes"]:
                    rec["mode_frac"] = {m: info["modes"].count(m) / len(info["modes"]) for m in (4, 3, 2, 1)}
                if "rapa_masked_frac" in info:
                    rec["rapa_masked_frac"] = info["rapa_masked_frac"]
                if info["disp"]:
                    rec["mean_disp_px"] = sum(info["disp"]) / len(info["disp"])
                if info["grgd"]:
                    g = info["grgd"]
                    rel = [v for v in g if v["n_edges"] > 0]
                    rec["grgd"] = {"frac_views_relative": len(rel) / len(g),
                                   "n_objects": float(np.mean([v["n_objects"] for v in g])),
                                   "disp_pia_px": float(np.mean([v["disp_pia_px"] for v in g]))}
                    for key in set().union(*[v.keys() for v in rel]) - {"n_objects", "disp_pia_px"}:
                        rec["grgd"][key] = float(np.mean([v[key] for v in rel if key in v]))
                if info["morgd2"]:
                    g = info["morgd2"]
                    mv = [v for v in g if v["moved"]]
                    fid = [r for v in mv for r, _ in v.get("fid", [])]
                    rec["morgd2"] = {"frac_views_moved": len(mv) / len(g),
                                     "rms_u_px": float(np.mean([v["rms_u_px"] for v in mv])) if mv else 0.0,
                                     "d_pair_pos_box": float(np.mean([v["d_pair_pos_box"] for v in mv if "d_pair_pos_box" in v])) if mv else None,
                                     "median_ratio_pos": float(np.median(fid)) if fid else None}
                if info["pia_factors"]:
                    rec["pia_factors"] = {k: float(np.mean([v[k] for v in info["pia_factors"]])) for k in info["pia_factors"][0]}
                if info["gt_kept"]:
                    rec["gt_kept_frac"] = sum(a for a, _ in info["gt_kept"]) / max(1, sum(b for _, b in info["gt_kept"]))
            if name in KG_METHODS:
                sp, log = info["sampler"], info["sampler"].log
                rec.update({
                    "forward_no_grad": info["forward_no_grad"], "n_pairs": sp.n_pairs,
                    "kg_degenerate": bool(sp.n_pairs > 0 and sp.kg_degenerate),
                    "frac_steps_E1_ne_E2": float(np.mean([r["q"] != r["e1"] for r in log])) if log else None,
                    "median_abs_log_area_ratio": float(np.median([r["abs_log_area_ratio"] for r in log])) if log else None,
                    "same_class_rate_E2": float(np.mean([r["same_class_q"] for r in log])) if log else None,
                    "same_class_rate_E1": float(np.mean([r["same_class_e1"] for r in log])) if log else None,
                })
            if name in RR_METHODS:
                rec["forward_no_grad"] = info["forward_no_grad"]
                rec["frac_steps_repair"] = float(np.mean(info["repair_on"]))
            if name in SGM_MIX:
                rec["frac_steps_gamma_0.2"] = float(np.mean([g == 0.2 for g in info["gammas"]]))
            with open(stats_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            stats[(name, img_id)] = rec
            n_done += 1
            if n_done % 10 == 0 or i == len(dataset) - 1:
                el = time.time() - t_start
                print(f"[attack {name}] {i + 1}/{len(dataset)}  {el / n_done:.1f}s/ảnh  "
                      f"còn ~{(len(dataset) - i - 1) * el / n_done / 60:.0f} phút", flush=True)
    return stats


def dets_name(out):
    """metrics.json -> dets.json (mặc định, như cũ); metrics_X.json -> dets_X.json."""
    return "dets.json" if out == "metrics.json" else out.replace("metrics", "dets", 1)


def stage_eval(args, dataset, models, adv_dir, stats, run_dir, art_dir, device):
    cat_ids = dataset._mmdet_dataset.cat_ids
    conds = ["clean"] + list(args.methods)
    dets = {c: {k: [] for k in models} for c in conds}
    t0 = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]
        img_id, x, ds = sample["img_id"], sample["inputs"].to(device), sample["data_sample"]
        imgs = {"clean": x}
        for name in args.methods:
            adv = img_tensor(os.path.join(adv_dir, name, f"{img_id}.png"), device)
            imgs[name] = pipeline_resize(adv, x.shape[-2:])
        for c, img in imgs.items():
            for k, m in models.items():
                dets[c][k] += to_coco_dets(predict(m, img, ds, rescale=True), img_id, cat_ids)
        if (i + 1) % 50 == 0:
            print(f"[eval] predict {i + 1}/{len(dataset)} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(art_dir, dets_name(args.out)), "w") as f:
        json.dump(dets, f)

    coco_gt = COCO(ANN_FILE)
    img_ids = list(dataset.image_ids)
    evals = {c: {k: CocoImageEval(coco_gt, dets[c][k], img_ids) for k in models} for c in conds}
    print("[eval] COCOeval xong (weighted AP khớp pycocotools)", flush=True)

    n = len(img_ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, args.n_boot, BOOT_SEED)])  # hàng 0 = mẫu gốc
    ap = {c: {k: np.array([evals[c][k].ap(w) for w in W]) * 100 for k in models} for c in conds}
    print(f"[eval] bootstrap {args.n_boot} mẫu xong ({time.time() - t0:.0f}s)", flush=True)

    def S(arr):  # arr[0] = điểm trên mẫu gốc, arr[1:] = bootstrap
        return summarize(arr[0], arr[1:])

    metrics = {"ap": {}, "ap50": {}, "relative_ap_drop": {}, "cross_avg": {}, "transfer_gap": {},
               "method_diff_cross_avg": {}, "perturbation": {}, "compute": {}}
    drop = {}
    for c in conds:
        metrics["ap"][c] = {k: S(ap[c][k][:, 0]) for k in models}
        metrics["ap50"][c] = {k: S(ap[c][k][:, 1]) for k in models}
    for name in args.methods:
        drop[name] = {k: 100 * (ap["clean"][k][:, 0] - ap[name][k][:, 0]) / ap["clean"][k][:, 0]
                      for k in models}
        metrics["relative_ap_drop"][name] = {k: S(drop[name][k]) for k in models}
        cross = (drop[name][CROSS_CNN] + drop[name][CROSS_TR]) / 2
        metrics["cross_avg"][name] = S(cross)
        metrics["transfer_gap"][name] = {
            "same_minus_cross_cnn": S(drop[name][SAME_FAMILY] - drop[name][CROSS_CNN]),
            "same_minus_cnn_to_transformer": S(drop[name][SAME_FAMILY] - drop[name][CROSS_TR]),
            "same_minus_cross_avg": S(drop[name][SAME_FAMILY] - cross),
        }
        rows = [stats[(name, i)] for i in img_ids]
        metrics["perturbation"][name] = {
            "linf_255_max": max(r["linf_255"] for r in rows),
            "l2_mean": float(np.mean([r["l2"] for r in rows])),
            "l2_per_pixel_rms_mean": float(np.mean([r["l2_per_pixel_rms"] for r in rows])),
        }
        metrics["compute"][name] = {
            "runtime_s_per_img": float(np.mean([r["runtime_s"] for r in rows])),
            "backward_per_img": rows[0]["backward"], "forward_views_per_img": rows[0]["forward_views"],
        }
    for a in args.methods:
        for b in args.methods:
            if a < b:
                ca = (drop[a][CROSS_CNN] + drop[a][CROSS_TR]) / 2
                cb = (drop[b][CROSS_CNN] + drop[b][CROSS_TR]) / 2
                metrics["method_diff_cross_avg"][f"{a} - {b}"] = S(ca - cb)

    out = {
        "run": os.path.basename(run_dir), "split": args.split, "n_images": n,
        "budget_B": args.budget, "epsilon_255": args.epsilon, "surrogate": SURROGATE_KEY,
        "targets": {"same_family": SAME_FAMILY, "cross_cnn": CROSS_CNN, "cnn_to_transformer": CROSS_TR},
        "eval": "PNG uint8 cỡ gốc -> Resize test pipeline; bbox, area=all, maxDets=100",
        "bootstrap": {"n_boot": args.n_boot, "seed": BOOT_SEED, "unit": "image, paired across conditions/models",
                      "ci": "percentile 2.5/97.5", "p_le_0": "tỉ lệ mẫu bootstrap <= 0"},
        "not_computed": ["ASR", "APloc", "CSR", "LPIPS (chưa cài lpips)"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
    }
    with open(os.path.join(run_dir, args.out), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print_table(out, list(models), args.methods)
    print("saved:", os.path.join(run_dir, args.out))


def print_table(m, keys, methods):
    fmt = lambda s: f"{s['point']:5.1f} [{s['ci95'][0]:5.1f},{s['ci95'][1]:5.1f}]"
    print(f"\n=== bbox AP (n={m['n_images']}, B={m['budget_B']}, eps={m['epsilon_255']}) ===")
    print(f"{'':<12}" + "".join(f"{k:>21}" for k in keys))
    for c, row in m["ap"].items():
        print(f"{c:<12}" + "".join(f"{fmt(row[k]):>21}" for k in keys))
    print("\n=== relative AP drop % [95% CI] ===")
    for c, row in m["relative_ap_drop"].items():
        print(f"{c:<12}" + "".join(f"{fmt(row[k]):>21}" for k in keys))
    print("\n=== CrossAvg / TransferGap (same-family drop − cross drop) ===")
    for c in methods:
        g = m["transfer_gap"][c]
        print(f"{c:<12} CrossAvg {fmt(m['cross_avg'][c])}  gap→ConvNeXt {fmt(g['same_minus_cross_cnn'])}"
              f"  gap→Swin {fmt(g['same_minus_cnn_to_transformer'])}  gap→avg {fmt(g['same_minus_cross_avg'])}")
    print("\n=== runtime ===", {c: round(v["runtime_s_per_img"], 1) for c, v in m["compute"].items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="n300")
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--epsilon", type=float, default=5.0)
    p.add_argument("--methods", nargs="+", default=BASELINES, choices=list(METHODS))
    p.add_argument("--out", default="metrics.json", help="tên file trong results/runs/<run>/")
    p.add_argument("--stage", choices=["all", "attack", "eval"], default="all")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None, help="chỉ để test nhanh, KHÔNG dùng cho bảng chính thức")
    p.add_argument("--subset", default=None,
                   help="chỉ chạy các ảnh của data/image_lists/<subset>.csv bên trong split (vd dev100 trong "
                        "dev300); seed vẫn theo chỉ số trong split -> nối thêm ảnh sau không phải chạy lại")
    args = p.parse_args()

    run = f"{args.split}_B{args.budget}_eps{args.epsilon:g}" + (f"_limit{args.limit}" if args.limit else "")
    run_dir = os.path.join(REPO_ROOT, "results/runs", run)
    art_dir = os.path.join(REPO_ROOT, "artifacts/runs", run)
    adv_dir = os.path.join(art_dir, "adv")
    os.makedirs(run_dir, exist_ok=True); os.makedirs(adv_dir, exist_ok=True)
    stats_path = os.path.join(run_dir, "attack_stats.jsonl")
    device = "cuda:0"

    dataset = build_attack_dataset(args.split)
    dataset.seed_index = list(range(len(dataset)))
    if args.subset:
        keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv")))
        pos = [j for j, img_id in enumerate(dataset.image_ids) if img_id in keep]
        assert len(pos) == len(keep), f"{args.subset} không nằm trọn trong {args.split}"
        dataset.image_ids = [dataset.image_ids[j] for j in pos]
        dataset._indices = [dataset._indices[j] for j in pos]
        dataset.seed_index = pos
    if args.limit:
        dataset.image_ids = dataset.image_ids[:args.limit]
        dataset._indices = dataset._indices[:args.limit]
        dataset.seed_index = dataset.seed_index[:args.limit]
    print(f"[run] {run}: {len(dataset)} ảnh{' (subset ' + args.subset + ')' if args.subset else ''}, "
          f"methods={args.methods}", flush=True)

    surrogate = load_model(SURROGATE_KEY, device)
    stats = load_stats(stats_path)
    if args.stage in ("all", "attack"):
        stats = stage_attack(args, dataset, surrogate, adv_dir, stats_path, device)
    if args.stage in ("all", "eval"):
        models = {k: surrogate if k == SURROGATE_KEY else load_model(k, device) for k in CONTROLLED_PANEL}
        stage_eval(args, dataset, models, adv_dir, stats, run_dir, art_dir, device)


if __name__ == "__main__":
    main()
