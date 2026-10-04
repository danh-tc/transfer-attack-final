#!/usr/bin/env python3
"""Unit test MORGD-RaPA (progress_log 2026-10-04) — chạy TRƯỚC mọi attack.

1. MORGD (rapa_p=None) trùng tuyệt đối HEAD; rapa_p = 0 trùng tuyệt đối MORGD.
2. Tỉ lệ phần tử bị mask ≈ p = 0.05.
3. Tham số model sau attack không đổi.
4. Cùng mask: trong cùng context, feature 2 lần forward trùng nhau và khác feature không mask.
5. Gradient theo input dưới context == gradient khi gán tham số masked thật (forward + backward cùng masked).
6. Chạy được, L∞ đúng, forward no_grad đúng.
"""
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods import pia
from attack.models import load_surrogate
from attack.preprocess import extract_features, resize_to_model
from scripts.pia_unit_test import check, load_head, seed

DEV = "cuda:0"


def main():
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:2]
    ds = AttackDataset(ids)
    model = load_surrogate(DEV)
    smp = ds[0]
    orig, dsm = smp["orig_inputs"].to(DEV), smp["data_sample"]
    seed(0); pia.geo_osfd_attack(model, orig, dsm, steps=1, geo="morgd")  # khởi động GPU
    before = {k: v.clone() for k, v in model.state_dict().items()}

    # 1
    old = load_head("attack/methods/pia.py", "pia_head")
    out = []
    for mod, kw in ((old, {}), (pia, {}), (pia, {"rapa_p": 0.0})):
        seed(9); out.append(mod.geo_osfd_attack(model, orig, dsm, steps=3, geo="morgd", **kw))
    check(torch.equal(out[0], out[1]), "MORGD (không RaPA) trùng HEAD")
    check(torch.equal(out[1], out[2]), "rapa_p = 0 trùng tuyệt đối MORGD")

    # 2 + 6
    info = {}
    seed(9)
    noise = pia.geo_osfd_attack(model, orig, dsm, steps=3, geo="morgd", rapa_p=0.05, info=info)
    f = info["rapa_masked_frac"]
    check(abs(f - 0.05) < 0.005, f"tỉ lệ mask {f:.4f} ≈ 0.05")
    check(noise.abs().max().item() <= 5.0 + 1e-4 and info["forward_no_grad"] == 6
          and not torch.equal(noise, out[1]), f"MORGD-RaPA chạy được, L∞ {noise.abs().max().item():.1f}, khác MORGD")

    # 3
    after = model.state_dict()
    check(all(torch.equal(before[k], after[k]) for k in before), "tham số model không đổi sau attack")

    # 4
    clean = resize_to_model(orig, tuple(dsm.img_shape))
    torch.manual_seed(123)
    sample, _ = pia.rapa_sampler(model, 0.05, DEV)
    ctx = sample()
    with torch.no_grad():
        with ctx():
            a = extract_features(model, clean, dsm)
        with ctx():
            b = extract_features(model, clean, dsm)
        c = extract_features(model, clean, dsm)
    check(all(torch.equal(x, y) for x, y in zip(a, b)) and not torch.equal(a[-1], c[-1]),
          "cùng context -> cùng mask; khác feature không mask")

    # 5
    x = (clean + 2.0).requires_grad_(True)
    with ctx():
        la = sum(f_.pow(2).mean() for f_ in extract_features(model, x, dsm))
    ga = torch.autograd.grad(la, x)[0]
    # gán tham số masked thật rồi tính lại
    names = [n for n, m in model.backbone.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    with torch.no_grad(), ctx():
        snap = {n: (model.backbone.get_submodule(n).weight.detach().clone(),
                    model.backbone.get_submodule(n).bias.detach().clone()) for n in names}
    orig_p = {n: (model.backbone.get_submodule(n).weight.data.clone(),
                  model.backbone.get_submodule(n).bias.data.clone()) for n in names}
    for n in names:
        model.backbone.get_submodule(n).weight.data.copy_(snap[n][0])
        model.backbone.get_submodule(n).bias.data.copy_(snap[n][1])
    x2 = (clean + 2.0).requires_grad_(True)
    lb = sum(f_.pow(2).mean() for f_ in extract_features(model, x2, dsm))
    gb = torch.autograd.grad(lb, x2)[0]
    for n in names:
        model.backbone.get_submodule(n).weight.data.copy_(orig_p[n][0])
        model.backbone.get_submodule(n).bias.data.copy_(orig_p[n][1])
    rel = ((ga - gb).norm() / gb.norm()).item()
    check(rel < 1e-4 and abs(la.item() - lb.item()) <= 1e-5 * abs(lb.item()),
          f"gradient dưới context == tham số masked thật (lệch tương đối {rel:.1e})")
    check(all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "khôi phục tham số đúng")
    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()
