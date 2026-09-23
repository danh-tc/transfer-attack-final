#!/usr/bin/env python3
"""Mechanism A3 (docs/mechanism_plan.md): attack OSFD chỉ trên 1 stage backbone của R50.

Can thiệp: OSFD full recipe (MI + RRB, B=50, ε=5, k=3, cùng seed ảnh như run baseline)
nhưng loss chỉ lấy stage k ∈ {1,2,3,4} (1-based; stride 4/8/16/32). 100 ảnh đầu của n300.
Tham chiếu "all" = OSFD gốc (mọi stage) — dùng lại PNG của run `n300_B50_eps5` cho cùng ảnh.
Eval như baseline: ảnh uint8 cỡ gốc (PNG) -> Resize test pipeline -> predict; COCO AP;
paired bootstrap 1000 mẫu trên 100 ảnh.

Target lõi (tiêu chí): R101 (cùng họ), ConvNeXt-T, Swin-T. Mô tả thêm (KHÔNG tính tiêu chí,
theo user 2026-09-23): Mask R-CNN Swin-S, DINO-Swin-L (đọc file qua pipeline riêng).

Tiêu chí vận hành (chốt + commit TRƯỚC khi chạy, 2026-09-23):
- T_k(t) = relative AP drop của attack stage k trên target t.
- Với cặp stage (i, j) và target khác họ c ∈ {ConvNeXt-T, Swin-T}:
  Δ_R101 = T_i(R101) − T_j(R101), Δ_c = T_i(c) − T_j(c), interaction = Δ_c − Δ_R101.
- **A3 đạt ("stage-aware" có bằng chứng)** nếu tồn tại (i, j, c) mà Δ_R101 và Δ_c trái dấu
  (điểm ước lượng — thứ hạng stage ĐẢO giữa cùng họ và khác họ) VÀ CI 95% của interaction
  loại trừ 0.
- A3 chỉ định nghĩa được cho OSFD (loss feature tách theo stage); task loss của M-DI² không
  tách stage → yêu cầu "cả OSFD và M-DI²" của quy tắc chung không áp dụng được cho A3.
- Mô tả kèm: T_k − T_all (stage đơn có hơn OSFD gốc không), alignment (cosine, sign) giữa
  ∇ₓ(OSFD-loss stage k) tại ảnh sạch và ∇ₓ L_task của target.

Chạy: python scripts/mech_a3_stage_attack.py
"""
import argparse
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
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from attack.data import build_attack_dataset
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.losses.osfd import make_osfd_loss_fn
from attack.methods.baselines import osfd_attack
from attack.models import CONTROLLED_PANEL, SURROGATE_KEY, load_model
from attack.preprocess import compute_gt_loss, pipeline_resize, predict, resize_to_model, to_adv_image
from eval_generalization import GEN_PANEL, CFG, CKPT

N_IMG = 100
EPS = 5.0
B = 50
SEED = 0          # = run_baselines.py (seed ảnh i = SEED + i)
BOOT_SEED = 2026
STAGES = [1, 2, 3, 4]
CORE = ["r101", "convnext_t", "swin_t"]
CROSS = ["convnext_t", "swin_t"]
DESC = ["mrcnn_swin_s", "dino_swin_l"]
BASE_RUN = "n300_B50_eps5"


def img_tensor(path, device):
    return torch.from_numpy(mmcv.imread(path)).permute(2, 0, 1).float().to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    device = "cuda:0"
    art = os.path.join(REPO_ROOT, "artifacts/runs/a3_stage_n100_B50_eps5")
    base_art = os.path.join(REPO_ROOT, "artifacts/runs", BASE_RUN)
    out_dir = os.path.join(REPO_ROOT, "results/mechanism")
    conds = [f"stage{k}" for k in STAGES]

    dataset = build_attack_dataset("n300")
    dataset.image_ids, dataset._indices = dataset.image_ids[:N_IMG], dataset._indices[:N_IMG]
    img_ids = list(dataset.image_ids)
    models = {k: load_model(k, device) for k in CONTROLLED_PANEL}
    surrogate = models[SURROGATE_KEY]

    # ---- 1. Attack theo stage (resume: có PNG thì bỏ qua) ----
    t0 = time.time()
    for k in STAGES:
        os.makedirs(os.path.join(art, f"stage{k}"), exist_ok=True)
        for i in range(N_IMG):
            png = os.path.join(art, f"stage{k}", f"{img_ids[i]}.png")
            if os.path.exists(png):
                continue
            s = dataset[i]
            orig, ds = s["orig_inputs"].to(device), s["data_sample"]
            random.seed(SEED + i); torch.manual_seed(SEED + i)
            noise = osfd_attack(surrogate, orig, ds, steps=B, epsilon=EPS, stages=[k - 1])
            adv = to_adv_image(orig, noise)
            assert (adv - orig).abs().max() <= EPS + 1e-4
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png)
            if (i + 1) % 25 == 0:
                print(f"[A3 attack stage{k}] {i + 1}/{N_IMG} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2. Predict: target lõi + surrogate (pipeline_resize), mô tả (file) ----
    coco = COCO(os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json"))
    cat_ids = dataset._mmdet_dataset.cat_ids
    all_conds = ["clean", "all"] + conds
    dets = {c: {m: [] for m in list(models) + DESC} for c in all_conds}
    align = {mt: {c: {t: np.zeros(N_IMG) for t in CORE} for c in conds} for mt in ("cosine", "sign")}
    for i in range(N_IMG):
        s = dataset[i]
        img_id, x, ds, orig = s["img_id"], s["inputs"].to(device), s["data_sample"], s["orig_inputs"].to(device)
        imgs = {"clean": x, "all": pipeline_resize(img_tensor(os.path.join(base_art, "adv/OSFD", f"{img_id}.png"), device), x.shape[-2:])}
        for c in conds:
            imgs[c] = pipeline_resize(img_tensor(os.path.join(art, c, f"{img_id}.png"), device), x.shape[-2:])
        for c, im in imgs.items():
            for m, mod in models.items():
                dets[c][m] += to_coco_dets(predict(mod, im, ds, rescale=True), img_id, cat_ids)
        # Alignment (mô tả): ∇ OSFD stage k (R50, ảnh sạch) vs ∇ L_task target.
        g_t = {}
        for t in CORE:
            torch.manual_seed(i)
            xg = orig.clone().requires_grad_(True)
            g_t[t] = torch.autograd.grad(compute_gt_loss(models[t], resize_to_model(xg, tuple(ds.img_shape)), ds), xg)[0].flatten()
        clean_px = resize_to_model(orig, tuple(ds.img_shape))
        for k in STAGES:
            fn = make_osfd_loss_fn(surrogate, clean_px, ds, stages=[k - 1])
            xg = orig.clone().requires_grad_(True)
            g_s = torch.autograd.grad(fn(surrogate, resize_to_model(xg, tuple(ds.img_shape)), ds), xg)[0].flatten()
            for t in CORE:
                align["cosine"][f"stage{k}"][t][i] = float(torch.nn.functional.cosine_similarity(g_s, g_t[t], dim=0))
                align["sign"][f"stage{k}"][t][i] = float((torch.sign(g_s) == torch.sign(g_t[t])).float().mean())
        if (i + 1) % 25 == 0:
            print(f"[A3 eval core] {i + 1}/{N_IMG}", flush=True)

    # Mô tả: clean + all lấy từ gen_dets của run baseline; stage k predict mới.
    from mmdet.apis import inference_detector, init_detector
    import glob
    idset = set(img_ids)
    for key in DESC:
        with open(os.path.join(base_art, "gen_dets", f"{key}.json")) as f:
            gd = json.load(f)
        dets["clean"][key] = [d for d in gd["clean"] if d["image_id"] in idset]
        dets["all"][key] = [d for d in gd["OSFD"] if d["image_id"] in idset]
        cfg, pat = GEN_PANEL[key]
        mod = init_detector(os.path.join(CFG, cfg), glob.glob(os.path.join(CKPT, pat + ".pth"))[0], device=device)
        cids = sorted(coco.getCatIds())
        for c in conds:
            for img_id in img_ids:
                dets[c][key] += to_coco_dets(inference_detector(mod, os.path.join(art, c, f"{img_id}.png")), img_id, cids)
        del mod
        print(f"[A3 eval desc] {key} xong", flush=True)
    with open(os.path.join(art, "dets.json"), "w") as f:
        json.dump(dets, f)

    # ---- 3. AP + paired bootstrap ----
    W = np.vstack([np.ones((1, N_IMG)), bootstrap_indices(N_IMG, args.n_boot, BOOT_SEED)])
    mods = list(models) + DESC
    ap = {c: {} for c in all_conds}
    for c in all_conds:
        for m in mods:
            ev = CocoImageEval(coco, dets[c][m], img_ids)
            ap[c][m] = np.array([ev.ap(w)[0] for w in W]) * 100
    drop = {c: {m: 100 * (ap["clean"][m] - ap[c][m]) / ap["clean"][m] for m in mods} for c in all_conds if c != "clean"}
    S = lambda a: summarize(a[0], a[1:])

    res = {"n_images": N_IMG, "budget_B": B, "epsilon_255": EPS, "n_boot": args.n_boot,
           "clean_ap": {m: S(ap["clean"][m]) for m in mods},
           "relative_ap_drop": {c: {m: S(drop[c][m]) for m in mods} for c in drop},
           "stage_minus_all": {c: {m: S(drop[c][m] - drop["all"][m]) for m in mods} for c in conds},
           "interaction": {}, "criterion": {"reversals": []},
           "alignment": {mt: {c: {t: summarize(np.mean(align[mt][c][t]), np.array(
               [np.mean(align[mt][c][t][r]) for r in np.random.default_rng(BOOT_SEED).integers(0, N_IMG, (args.n_boot, N_IMG))]))
               for t in CORE} for c in conds} for mt in align}}
    for a in range(len(conds)):
        for b in range(a + 1, len(conds)):
            ci_, cj = conds[a], conds[b]
            dR = drop[ci_]["r101"] - drop[cj]["r101"]
            for c in CROSS:
                dC = drop[ci_][c] - drop[cj][c]
                inter = S(dC - dR)
                key = f"{ci_} vs {cj} | {c}"
                res["interaction"][key] = {"delta_r101": S(dR), "delta_cross": S(dC), "interaction": inter}
                reversed_ = np.sign(dR[0]) != np.sign(dC[0])
                excl = inter["ci95"][0] > 0 or inter["ci95"][1] < 0
                if reversed_ and excl:
                    res["criterion"]["reversals"].append(key)
    res["criterion"]["A3_pass"] = bool(res["criterion"]["reversals"])
    with open(os.path.join(out_dir, "a3_stage_attack.json"), "w") as f:
        json.dump(res, f, indent=2)

    fmt = lambda s: f"{s['point']:5.1f} [{s['ci95'][0]:5.1f},{s['ci95'][1]:5.1f}]"
    print(f"\n=== relative AP drop % (n={N_IMG}, B={B}) — lõi: r101/convnext_t/swin_t; mô tả: {DESC} ===")
    print(f"{'':<8}" + "".join(f"{m:>21}" for m in mods))
    for c in ["all"] + conds:
        print(f"{c:<8}" + "".join(f"{fmt(res['relative_ap_drop'][c][m]):>21}" for m in mods))
    print("\n=== interaction (Δ_cross − Δ_R101) ===")
    for k, v in res["interaction"].items():
        print(f"  {k:<32} ΔR101 {fmt(v['delta_r101'])}  Δcross {fmt(v['delta_cross'])}  inter {fmt(v['interaction'])}")
    print("\n=== alignment sign (∇ OSFD stage k vs ∇ task target) ===")
    for c in conds:
        print(f"  {c}: " + "  ".join(f"{t} {res['alignment']['sign'][c][t]['point']:.4f}" for t in CORE))
    print(f"\ncriterion: {res['criterion']}")
    print("saved:", os.path.join(out_dir, "a3_stage_attack.json"))


if __name__ == "__main__":
    main()
