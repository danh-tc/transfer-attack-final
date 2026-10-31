"""E74 (progress_log 2026-10-31): quét NGÂN SÁCH B cho RD-CNX (ConvNeXt ĐƠN) @ dev100 — tăng B có lấy lại họ ResNet
mà GIỮ Swin-L không? Đây là lever single-surrogate "giữ cả hai" DUY NHẤT còn lại (không model thứ 2, không đổi threat
model, chỉ thêm compute) — E66 ghi "việc tiếp" nhưng chưa chạy. Sinh B50/B75 bằng run_baselines (run.sh); eval cùng đường
inference-PNG, incremental save (resume per-model). Mốc B36/ENS-WP/RD-R50 ĐỌC LẠI từ e73_dev100.json (cùng dev100, cùng eval).
  python scripts/e74_cnx_budget.py   -> results/mechanism/e74_cnx_budget.json"""
import json
import os
import sys

import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from scripts.e64_cnx_drop import load

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e74_cnx_budget.json")
E73 = os.path.join(REPO_ROOT, "results/mechanism/e73_dev100.json")
PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
RES3 = ["r101", "fcos_r50", "dino_r50"]
# cond -> (run dir, tên thư mục adv) — B50 (user dừng B75); B36/ENS/RD-R50 lấy từ e73
NEW = {"RD-CNX-B50": ("dev300_B50_eps5_limit100_sur-convnext_t", "C2-OCT-1V-WP2-RD-CNX")}
BOOT_SEED, N_BOOT = 2026, 300


def png(cond, i):
    run, folder = NEW[cond]
    return os.path.join(ART, run, "adv", folder, f"{i}.png")


def main():
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    assert len(ids) == 100
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    from mmdet.apis import inference_detector
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"n_images": n, "n_boot": N_BOOT, "drop": {}}
    conds = list(NEW)
    for key in PANEL:
        if key in res["drop"] and all(c in res["drop"][key] for c in conds):
            continue
        model = load(key)
        res["drop"].setdefault(key, {})

        def ap_of(paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            return np.array([CocoImageEval(coco, dets, ids).ap(w) for w in W])[:, 0] * 100
        clean = ap_of([os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        for c in conds:
            if c in res["drop"][key]:
                continue
            adv = ap_of([png(c, i) for i in ids])
            res["drop"][key][c] = summarize(*(lambda a: (a[0], a[1:]))(100 * (clean - adv) / clean))
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)  # incremental -> resume per-model
        print(key, {c: round(res["drop"][key][c]["point"], 1) for c in conds}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)

    # Gộp mốc B36/ENS-WP/RD-R50 từ E73 (cùng dev100, cùng đường eval)
    e73 = json.load(open(E73))["drop"]
    pt = lambda c, m: (res["drop"][m][c]["point"] if c in NEW else e73[m][c]["point"])
    t7 = lambda c: sum(pt(c, m) for m in T7) / 7
    ROWS = ["RD-R50", "RD-CNX", "RD-CNX-B50", "ENS-WP"]  # RD-CNX = B36 từ e73; B75 user đã dừng
    print("\n" + " " * 11 + " ".join(m[:9].rjust(9) for m in PANEL) + "    T7")
    for c in ROWS:
        print(f"{c:11}" + " ".join(f"{pt(c, m):9.1f}" for m in PANEL) + f" {t7(c):6.1f}")

    print("\n--- lever single-surrogate: B có lấy lại ResNet & giữ Swin-L? ---")
    for c in ("RD-CNX", "RD-CNX-B50"):
        res3 = np.mean([pt(c, m) for m in RES3])
        print(f"{c:11} RES3avg {res3:5.1f} (R101 {pt(c,'r101'):.1f} FCOS {pt(c,'fcos_r50'):.1f} DINO-R50 {pt(c,'dino_r50'):.1f})"
              f"  SwinL {pt(c,'dino_swin_l'):5.1f}  T7 {t7(c):5.1f}")
    print(f"{'RD-R50':11} RES3avg {np.mean([pt('RD-R50', m) for m in RES3]):5.1f}  (mốc ResNet cần đạt)")
    print(f"{'ENS-WP':11} T7 {t7('ENS-WP'):.1f} SwinL {pt('ENS-WP','dino_swin_l'):.1f}  (mốc giữ-cả-hai B36)")

    # Quy tắc khóa
    swin_floor = pt("RD-CNX", "dino_swin_l") - 5
    rd_r50 = {m: pt("RD-R50", m) for m in RES3}
    viable = []
    for c in ("RD-CNX-B50",):
        ok_res = all(pt(c, m) >= rd_r50[m] - 3 for m in RES3)
        ok_swin = pt(c, "dino_swin_l") >= swin_floor
        ok_t7 = t7(c) >= t7("ENS-WP") - 2
        print(f"[rule] {c}: RES3≥RD-R50−3 {ok_res}  SwinL≥RD-CNX−5 {ok_swin}  T7≥ENS−2 {ok_t7}")
        if ok_res and ok_swin and ok_t7:
            viable.append(c)
    gain = np.mean([pt("RD-CNX-B50", m) for m in RES3]) - np.mean([pt("RD-CNX", m) for m in RES3])
    print(f"\nResNet gain B36->B50 = {gain:+.1f}  (|< +2 và vẫn xa RD-R50| => HƯỚNG, không phải ngân sách; B75 user dừng)")
    print("VERDICT:", "SINGLE-SURROGATE GIỮ-CẢ-HAI VIABLE @ " + ",".join(viable) if viable
          else "B50 KHÔNG đủ -> hướng thiên về DIRECTION; cần B75 để chốt trần HOẶC nguồn-hướng-thứ-2")


if __name__ == "__main__":
    main()
