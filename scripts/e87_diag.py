"""E87 chẩn đoán gradient (MÔ TẢ, hậu kỳ — KHÔNG dùng để đổi α hay cổng GO/NO-GO), devB100.
Với surrogate s ∈ {CNX, R50}, trạng thái ∈ {clean (δ=0), adv (ảnh adv cuối của ô s-A1 E87, PNG)}, không aug, fp32:
  g2 = ∇ₓ L_C2, g345 = ∇ₓ L_C3–C5, g25 = ∇ₓ L_C2–C5  (L_OSFD k=3, feature sạch của ảnh sạch, như _view_loss).
Báo (TB qua ảnh, bootstrap 1000 seed 2026):
  ‖g2‖/‖g345‖, ‖g2‖/‖g25‖, năng lượng ‖g2‖²/‖g25‖², cos(g2, g345), sai số phân rã ‖g25 − (g2 + g345)‖/‖g25‖;
  D(α) = tỉ lệ toạ độ sign(g345) ≠ sign(g345 + α·g2), α ∈ {1, 0.5};
  D_null(α): thay g2 bằng vector Gauss cùng norm (5 lần rút, seed 87·1000 + n); báo D − D_null.
Lưu ý: bước attack là sign của momentum MI (chuẩn hóa mean|g|, có OCT) -> D theo từng gradient chỉ là proxy.
  python scripts/e87_diag.py [--list smoke3] -> results/mechanism/e87_diag_<list>.json"""
import argparse
import json
import os
import sys

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.methods.pia import _view_loss
from attack.models import load_model
from attack.preprocess import extract_features, resize_to_model
from attack.targets import target_dataset
from scripts.e87_c2_weight import ART, STEPS, image_ids

SURS = {"convnext_t": "CNX", "r50": "R50"}
BOOT_SEED, N_NULL = 2026, 5


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="devB100", choices=["devB100", "smoke3"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ids = image_ids(args.list)
    gpu = torch.cuda.get_device_name(0)
    sfx = f"_{args.tag}" if args.tag else ""
    run_dir = f"e87_c2w_B{STEPS}_eps5_{gpu.split()[-1].replace('-', '').lower()}{sfx}"
    M = {}
    for s, p in SURS.items():
        model = load_model(s, "cuda:0"); model.eval()
        for q in model.parameters():
            q.requires_grad_(False)
        ds = {x["img_id"]: x for x in target_dataset(s, ids)}
        for n, i in enumerate(ids):
            orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]
            hw = tuple(dsm.img_shape)
            with torch.no_grad():
                fc = tuple(f.detach() for f in extract_features(model, resize_to_model(orig, hw), dsm))
            png = os.path.join(ART, run_dir, "adv", f"{p}-A1", f"{i}.png")
            adv = torch.from_numpy(mmcv.imread(png)).permute(2, 0, 1).float().cuda()
            assert adv.shape == orig.shape and float((adv - orig).abs().max()) <= 5.0 + 1e-4
            for state, x0 in (("clean", orig), ("adv", adv)):
                g = {}
                for nm, lf in (("g2", _view_loss(fc, 3.0, (0,))), ("g345", _view_loss(fc, 3.0, (1, 2, 3))),
                               ("g25", _view_loss(fc, 3.0))):
                    x = x0.clone().requires_grad_(True)
                    gx, = torch.autograd.grad(lf(model, resize_to_model(x, hw), dsm), x)
                    g[nm] = gx.double().flatten()
                g2, g3, g25 = g["g2"], g["g345"], g["g25"]
                r = {"r_g2_g345": float(g2.norm() / g3.norm()), "r_g2_g25": float(g2.norm() / g25.norm()),
                     "energy_g2": float(g2.norm() ** 2 / g25.norm() ** 2),
                     "cos_g2_g345": float(g2 @ g3 / (g2.norm() * g3.norm())),
                     "decomp_err": float((g25 - (g2 + g3)).norm() / g25.norm())}
                gen = torch.Generator(device=g3.device).manual_seed(87 * 1000 + n)
                s3 = torch.sign(g3)
                for a, tag in ((1.0, "1"), (0.5, "05")):
                    d = float((s3 != torch.sign(g3 + a * g2)).double().mean())
                    dn = np.mean([float((s3 != torch.sign(g3 + a * z * (g2.norm() / z.norm()))).double().mean())
                                  for z in (torch.randn(g3.shape, generator=gen, device=g3.device, dtype=g3.dtype)
                                            for _ in range(N_NULL))])
                    r.update({f"D_a{tag}": d, f"Dnull_a{tag}": float(dn), f"D_minus_null_a{tag}": d - float(dn)})
                for k, v in r.items():
                    M.setdefault((p, state, k), []).append(v)
            print(f"[e87-diag {p}] {n + 1}/{len(ids)}", flush=True)
        del model; torch.cuda.empty_cache()
    boot = np.random.default_rng(BOOT_SEED).integers(0, len(ids), size=(1000, len(ids)))
    res = {"list": args.list, "n_images": len(ids), "gpu": gpu, "run_dir_adv": run_dir,
           "note": "[point, lo95, hi95]; MÔ TẢ hậu kỳ, không làm cổng, không dùng để chọn α"}
    for (p, state, k), v in sorted(M.items()):
        v = np.array(v)
        bm = v[boot].mean(1)
        res.setdefault(p, {}).setdefault(state, {})[k] = [round(float(v.mean()), 5), round(float(np.percentile(bm, 2.5)), 5),
                                                          round(float(np.percentile(bm, 97.5)), 5)]
    keys = ["r_g2_g345", "r_g2_g25", "energy_g2", "cos_g2_g345", "decomp_err", "D_a1", "Dnull_a1", "D_minus_null_a1",
            "D_a05", "Dnull_a05", "D_minus_null_a05"]
    print(f"\n=== E87 chẩn đoán @ {args.list} (n={len(ids)}) — TB [CI95] ===")
    for p in SURS.values():
        for state in ("clean", "adv"):
            print(f"{p:4} {state:5} " + "  ".join(f"{k} {res[p][state][k][0]:+.4f} [{res[p][state][k][1]:+.4f}, "
                                                    f"{res[p][state][k][2]:+.4f}]" for k in keys))
    out = os.path.join(REPO_ROOT, f"results/mechanism/e87_diag_{args.list}{sfx}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[e87-diag] -> {out}")


if __name__ == "__main__":
    main()
