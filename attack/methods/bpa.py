"""Làm mượt đường lan ngược ReLU của backbone surrogate (họ LinBP / BPA; progress_log 2026-10-21).

Forward GIỮ NGUYÊN ReLU; backward thay 1[x > 0] bằng đạo hàm của x·σ(βx) (β = 1.702 ≈ GELU — activation
của Swin) -> đổi hướng gradient, không đổi giá trị loss. Chỉ ReLU trong backbone (stem + Bottleneck),
`stages` = chỉ số 0-based của res_layers được làm mượt (None = mọi stage + stem).
"""
import contextlib

import torch
import torch.nn as nn


class _SmoothReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        s = torch.sigmoid(ctx.beta * x.float())
        return (g.float() * (s * (1 + ctx.beta * x.float() * (1 - s)))).to(g.dtype), None


@contextlib.contextmanager
def smooth_relu_backward(model, beta: float = 1.702, stages=None):
    bb = model.backbone
    mods = [m for i, layer in enumerate(bb.res_layers) if stages is None or i in stages
            for m in getattr(bb, layer).modules() if isinstance(m, nn.ReLU)]
    if stages is None and hasattr(bb, "relu"):
        mods.append(bb.relu)
    for m in mods:
        m.forward = lambda x, b=beta: _SmoothReLU.apply(x, b)
    try:
        yield len(mods)
    finally:
        for m in mods:
            del m.forward  # về lại nn.ReLU.forward của class


def bpa_attack(model, *args, attack_fn, beta: float = 1.702, stages=None, **kw):
    """attack_fn (vd geo_osfd_attack đã gắn cấu hình C2-OCT-1V) chạy với backward ReLU backbone làm mượt."""
    with smooth_relu_backward(model, beta, stages):
        return attack_fn(model, *args, **kw)
