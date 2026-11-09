"""E80-ENSB (progress_log 2026-11-09): quét ngân sách B cho ENS-WP — B=50 thay vì B=36, mọi thứ khác GIỮ NGUYÊN E71.

⚠️ MÁY NÀY LÀ V100 (Volta), mốc so sánh (E71 ENS-WP B36 dev30, Swin-L 60.3) sinh trên A4500 (Ampere).
CLAUDE.md khóa: KHÔNG ghép số Swin-L giữa V100 và Ampere (lệch ~8 điểm, kernel fp16 Volta lúc SINH δ, E59/E60/E61).
User chọn chạy 1 ô (không sinh lại mốc B36 trên V100) để test nhanh -> cách đọc HỢP LỆ DUY NHẤT là MỘT CHIỀU:
lệch V100 kéo Swin-L LÊN, nên B50@V100 < 60.3 = bằng chứng NGƯỢC cho đòn bẩy ngân sách; B50@V100 > 60.3 = LẪN
(không tách được ngân sách khỏi máy). Xem quy tắc khóa trong progress_log entry E80-ENSB.

Cấu hình y hệt E71 ENS-WP, chỉ đổi STEPS 36 -> 50: luân phiên R50/ConvNeXt theo khối 2 bước, momentum CHUNG,
WP ρ0.002 (weight_linear cho ConvNeXt), λ R50=6 / CNX=0.2, OCT 1V, L_OSFD C3-C5 + cls RoI.
50 / 2 = 25 khối -> 13 khối R50 / 12 khối ConvNeXt (lệch 1 khối; E71 B36 cân 9/9) — ghi làm caveat.

Panel GỌN 4 model (user chọn): dino_swin_l (trục chính) + r101/fcos_r50/dino_r50 (RES3, canh không vỡ họ ResNet).
boot 100 (sàng lọc, không phải mức bảng paper 300).
  python scripts/e80_ens_budget.py   -> results/mechanism/e80_ens_budget.json"""
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.run_baselines import _C2OCT
from scripts.e64_cnx_drop import load

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
RUN = "dev300_B50_eps5_limit100_ens_v100"
OUT = os.path.join(REPO_ROOT, "results/mechanism/e80_ens_budget.json")
COND = "ENS-WP-B50"
PANEL = ["dino_swin_l", "r101", "fcos_r50", "dino_r50"]
BLOCK, STEPS, BOOT_SEED, N_BOOT = 2, 50, 2026, 100
COMMON = dict(cls_mode="roi", stage_w=(0.0, 1.0, 1.0, 1.0))
# Y HỆT E71 KW_WP — không đổi một tham số nào ngoài STEPS.
KW_WP = {"r50": dict(_C2OCT, cls_weight=6.0, weight_rho=0.002),
         "convnext_t": dict(_C2OCT, cls_weight=0.2, weight_rho=0.002, weight_linear=True)}


def attack(ids):
    out_dir = os.path.join(ART, RUN, "adv", COND)
    if all(os.path.exists(os.path.join(out_dir, f"{i}.png")) for i in ids):
        print(f"[e80] {COND}: PNG đã đủ, bỏ qua sinh", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    models = {"r50": load_model("r50", "cuda:0"), "convnext_t": load_model("convnext_t", "cuda:0")}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in models}
    order = ["r50", "convnext_t"]
    nblocks = STEPS // BLOCK
    import time
    t0 = time.time()
    for n, i in enumerate(ids):
        if os.path.exists(os.path.join(out_dir, f"{i}.png")):
            continue
        orig = ds["r50"][i]["orig_inputs"].cuda().float()
        delta, mom = None, None
        for b in range(nblocks):
            key = order[b % 2]
            torch.manual_seed(1000 * (n + 1) + b)
            delta, mom = geo_osfd_attack(models[key], orig, ds[key][i]["data_sample"], steps=BLOCK,
                                         init_noise=delta, mom_in=mom, return_state=True,
                                         **KW_WP[key], **COMMON)
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= 5.0 + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                     os.path.join(out_dir, f"{i}.png"))
        print(f"[e80 {COND}] {n + 1}/{len(ids)}  {(time.time() - t0) / (n + 1):.1f}s/ảnh", flush=True)
    del models
    torch.cuda.empty_cache()


def main():
    print(f"[e80] GPU = {torch.cuda.get_device_name(0)}  (mốc E71 B36 sinh trên A4500 — xem caveat một chiều)",
          flush=True)
    coco = COCO(ANN)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30, len(ids)
    attack(ids)

    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    from mmdet.apis import inference_detector
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        "n_images": n, "n_boot": N_BOOT, "gpu": torch.cuda.get_device_name(0),
        "steps": STEPS, "block": BLOCK, "drop": {}}
    for key in PANEL:
        if key in res["drop"]:
            continue
        model = load(key)

        def ap_of(paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            return np.array([CocoImageEval(coco, dets, ids).ap(w) for w in W])[:, 0] * 100
        clean = ap_of([os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        adv = ap_of([os.path.join(ART, RUN, "adv", COND, f"{i}.png") for i in ids])
        s = 100 * (clean - adv) / clean
        res["drop"][key] = summarize(s[0], s[1:])
        del model
        torch.cuda.empty_cache()
        print(f"{key}  {COND} drop = {res['drop'][key]['point']:.1f} "
              f"CI {[round(x, 1) for x in res['drop'][key]['ci95']]}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)

    # Phán quyết theo quy tắc MỘT CHIỀU đã khóa (progress_log E80-ENSB).
    sl = res["drop"]["dino_swin_l"]["point"]
    res3 = [res["drop"][m]["point"] for m in ("r101", "fcos_r50", "dino_r50")]
    print(f"\n=== E80-ENSB @ dev30, {torch.cuda.get_device_name(0)}, B={STEPS} ===")
    print(f"dino_swin_l {sl:.1f}  | RES3 {np.mean(res3):.1f}  (r101 {res3[0]:.1f} fcos {res3[1]:.1f} "
          f"dino_r50 {res3[2]:.1f})")
    print(f"mốc A4500 E71 B36: Swin-L 60.3, RES3 93.7  — LỆCH MÁY, chỉ đọc một chiều")
    if sl < 60.3 - 4:
        print("=> NGƯỢC: thấp hơn mốc ≥4 DÙ lệch V100 kéo lên => bằng chứng ngược cho đòn bẩy ngân sách")
    elif sl < 60.3:
        print("=> NGHIÊNG NGƯỢC: thấp hơn mốc nhưng <4 => yếu, vô kết luận")
    else:
        print("=> LẪN: cao hơn mốc, KHÔNG tách được ngân sách khỏi lệch máy => cần sinh lại B36 trên V100")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
