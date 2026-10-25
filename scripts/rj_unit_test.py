#!/usr/bin/env python3
"""Unit test RJ (E34): jitter thang RoI trước RoIAlign. R50 + amp KHÔNG tất định giữa các lần chạy, nên kiểm transform trên
MỘT tensor rois CỐ ĐỊNH (bắt 1 lần từ run thật), đưa qua chính fwd của roi_jitter với `orig` là stub ghi rois nó nhận:
(1) r = 0 no-op: stub nhận rois y hệt input; (2) r = 0.3: tâm giữ, |log s| ≤ 0.3, có phương sai; (3) attack đầy đủ r = 0.3
chạy được, L∞ = 5 hữu hạn. Chạy: python scripts/rj_unit_test.py"""
import math
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import AttackDataset, load_image_ids
from attack.methods.pia import geo_osfd_attack
from attack.methods.repair import roi_jitter
from attack.models import load_model

ok = True
m = load_model("r50", "cuda:0")
dset = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")))
_C2OCT = dict(geo="elem", elem={"scale": math.log(2) / 2, "shift": "ph8"}, n_views=1, cls_weight=4.0,
              skip_mask=True, amp=True)
s = dset[0]; o = s["orig_inputs"].cuda(); d = s["data_sample"]
ext = m.roi_head.bbox_roi_extractor
orig_fwd = ext.forward

# Bắt 1 tensor rois thật từ 1 bước attack
grab = {}
def spy(feats, rois, rsf=None):
    grab.setdefault("rois", rois.detach().clone())
    return orig_fwd(feats, rois, rsf)
ext.forward = spy
try:
    torch.manual_seed(0); geo_osfd_attack(m, o, d, steps=1, epsilon=5, **_C2OCT)
finally:
    ext.forward = orig_fwd
rois_real = grab["rois"]
print(f"[info] rois bắt được: {tuple(rois_real.shape)}")


def jittered_rois(r):
    """Đưa rois_real qua fwd của roi_jitter(r); orig = stub ghi lại rois mà extractor gốc SẼ nhận (sau jitter)."""
    cap = {}
    def stub(feats, rois, rsf=None):
        cap["r"] = rois.detach().clone()
        return torch.zeros(len(rois), 256, 7, 7, device=rois.device)  # dummy, không dùng
    ext.forward = stub
    gen = torch.Generator(device="cuda:0").manual_seed(23)
    try:
        with roi_jitter(m, r, gen):      # orig = stub
            ext.forward(None, rois_real)  # fwd jitter -> stub
    finally:
        ext.forward = orig_fwd
    return cap["r"]


r0 = jittered_rois(0.0)
c = torch.equal(r0, rois_real)
print(f"[1] r=0 no-op: rois y hệt input: {c}"); ok &= c

r3 = jittered_rois(0.3)
cx_p, cy_p = (rois_real[:, 1] + rois_real[:, 3]) / 2, (rois_real[:, 2] + rois_real[:, 4]) / 2
cx_3, cy_3 = (r3[:, 1] + r3[:, 3]) / 2, (r3[:, 2] + r3[:, 4]) / 2
sw = (r3[:, 3] - r3[:, 1]) / (rois_real[:, 3] - rois_real[:, 1]).clamp_min(1e-6)
sh = (r3[:, 4] - r3[:, 2]) / (rois_real[:, 4] - rois_real[:, 2]).clamp_min(1e-6)
batch_ok = torch.equal(r3[:, 0], rois_real[:, 0])  # cột batch_idx giữ nguyên
center_ok = torch.allclose(cx_p, cx_3, atol=1e-2) and torch.allclose(cy_p, cy_3, atol=1e-2)
scale_ok = bool((sw.log().abs() <= 0.3 + 1e-3).all()) and float(sw.std()) > 1e-3
iso_ok = torch.allclose(sw, sh, atol=1e-3)  # co giãn đẳng hướng (cùng s cho w và h)
c = batch_ok and center_ok and scale_ok and iso_ok and not torch.equal(r3, rois_real)
print(f"[2] r=0.3: batch_idx giữ={batch_ok}, tâm giữ={center_ok}, |log s|≤0.3&var={scale_ok}, đẳng hướng={iso_ok}: {c}"); ok &= c

torch.manual_seed(0); n = geo_osfd_attack(m, o, d, steps=8, epsilon=5, **_C2OCT, roi_jit=0.3)
c = abs(float(n.abs().max()) - 5.0) < 1e-4 and bool(torch.isfinite(n).all())
print(f"[3] attack r=0.3: L∞={float(n.abs().max()):.1f}, hữu hạn: {c}"); ok &= c
print("RJ unit test", "QUA" if ok else "TRƯỢT")
sys.exit(0 if ok else 1)
