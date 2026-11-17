"""S6 sinh dữ liệu (progress_log 2026-11-17, KHÓA S6). Cùng công thức + seed S1 (chỉ số n trong dev100; dev30 = dev100[:30]).
  --part att  : 6 ô attack S1 cho dev100
  --part wb   : WB-0/r1/r2/r3/r4 cho dev30 TRƯỚC (cho phần hành vi), rồi WB-0/r1/r2 cho dev100
  --part grad : g_t DINO-Swin-L (cache S3): ảnh sạch + t ∈ {0.5, 1} cho 6 ô attack + WB-0/r1/r2, mọi ảnh dev100
"""
import argparse
import os
import sys

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import scripts.s1_signmap as s1
from attack.data import load_image_ids
from attack.targets import load_any, target_dataset, target_grads
from scripts.s3_region import ADV, GDIR

ATT = ["RD-ALL", "RD-FEAT", "RD-DET", "OSFD-ALL", "OSFD-FEAT", "OSFD-DET"]
WB3 = {"WB-0": None, "WB-r1": 1, "WB-r2": 2}
WB5 = dict(WB3, **{"WB-r3": 3, "WB-r4": 4})
GCELLS = ATT + list(WB3)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--part", choices=["att", "wb", "grad"], required=True)
    part = ap.parse_args().part
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    if part == "att":
        for c in ATT:
            s1.gen_attack(ids, c, os.path.join(ADV, c))
    elif part == "wb":
        s1.WB = WB5; s1.gen_wb(ids[:30], {c: os.path.join(ADV, c) for c in WB5})
        s1.WB = WB3; s1.gen_wb(ids, {c: os.path.join(ADV, c) for c in WB3})
    else:
        os.makedirs(GDIR, exist_ok=True)
        model = load_any("dino_swin_l")
        ds = {s["img_id"]: s for s in target_dataset("dino_swin_l", ids)}
        for n, i in enumerate(ids):
            orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
            clean = orig.permute(1, 2, 0).cpu().numpy().astype(np.int32)
            jobs = [("clean", orig)]
            for c in GCELLS:
                d = torch.from_numpy((mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean)
                                     .transpose(2, 0, 1).astype(np.float32)).cuda()
                jobs += [(f"{c}_t{t}", torch.clamp(orig + t * d, 0, 255)) for t in (0.5, 1.0)]
            for name, x in jobs:
                p = os.path.join(GDIR, f"{i}_{name}.npy")
                if os.path.exists(p):
                    continue
                torch.manual_seed(1000 * (n + 1))
                g = target_grads(model, "dino_swin_l", x, dsm, objectives=("cls",))[0]["cls"].float().cpu().numpy()
                np.save(p, g.astype(np.float16))
            print(f"[s6 grad] {n + 1}/{len(ids)}", flush=True)


if __name__ == "__main__":
    main()
