"""Pilot R1 (quy tắc LOCKED: docs/progress_log.md 2026-10-03): OSFD-adapted SGM — "stage-balanced SGM".

KHÔNG phải port SGM nguyên bản (Wu et al., ICLR 2020; csdongxian/skip-connections-matter/utils_sgm.py).
Bản gốc với ResNet-50 nhân √γ ở CẢ 3 lần dùng module relu của mỗi Bottleneck (trừ block đầu mỗi stage)
+ relu stem + chuẩn hóa std gradient mỗi block: tỉ lệ residual/skip = γ, phần còn lại là hằng số vô
hướng — vô hại khi loss vào ở đầu ra mạng, nhưng với OSFD (loss ở 4 stage) làm số hạng stage sâu bị
triệt theo độ sâu. Bản này chỉ giữ phần cốt lõi: nhân √γ ở 2 ReLU TRONG nhánh residual (sau norm1,
sau norm2) của mọi Bottleneck trừ block đầu mỗi stage -> residual × γ, skip × 1. Forward không đổi.

Cài bằng forward hook trên norm1/norm2: chèn GradScale (forward = bản sao, backward × √γ) trước ReLU
inplace — nhân hằng số giao hoán với mặt nạ ReLU nên tương đương hook ở chính ReLU.
"""
import math
from typing import List, Optional, Sequence

import numpy as np
import torch

from attack.methods.core import run_iterative_attack
from attack.methods.rrb import rrb_views
from attack.losses.osfd import make_osfd_loss_fn
from attack.preprocess import resize_to_model

GAMMAS = (0.2, 0.5)  # khóa: E1a, E1b; E2 lấy mẫu đều trong tập này


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()  # bản sao: ReLU inplace phía sau không sửa output của Function

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


class _GradStdNorm(torch.autograd.Function):
    """Chỉ cho bản tham chiếu nguyên văn (unit test): backward chia cho std."""
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return g / g.std()


def residual_blocks(backbone) -> List:
    """Bottleneck không phải block đầu mỗi stage (block đầu có downsample — code gốc bỏ qua '0.relu')."""
    blocks = []
    for name in backbone.res_layers:
        blocks += list(getattr(backbone, name))[1:]
    return blocks


class StageBalancedSGM:
    """Hook có γ thay đổi được (E2 đổi γ mỗi step). Dùng như context manager; ra khỏi thì gỡ hook."""

    def __init__(self, backbone, gamma: float = 1.0):
        self.backbone, self.gamma, self._handles = backbone, gamma, []

    def _hook(self, module, inputs, output):
        if not torch.is_grad_enabled():
            return output
        return _GradScale.apply(output, math.sqrt(self.gamma))

    def __enter__(self):
        for blk in residual_blocks(self.backbone):
            self._handles += [blk.norm1.register_forward_hook(self._hook),
                              blk.norm2.register_forward_hook(self._hook)]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


class LiteralSGM:
    """Bản port NGUYÊN VĂN code gốc (chỉ để unit test hướng SGM với loss 1 điểm vào): √γ ở cả 3 lần dùng
    relu của block không-đầu + relu stem; chuẩn hóa std gradient ở đầu ra MỌI block."""

    def __init__(self, backbone, gamma: float):
        self.backbone, self.s, self._handles = backbone, math.sqrt(gamma), []

    def __enter__(self):
        scale = lambda m, i, o: _GradScale.apply(o, self.s)
        norm = lambda m, i, o: _GradStdNorm.apply(o)
        self._handles.append(self.backbone.relu.register_forward_hook(scale))
        for blk in residual_blocks(self.backbone):
            self._handles.append(blk.relu.register_forward_hook(scale))
        for name in self.backbone.res_layers:
            for blk in getattr(self.backbone, name):
                self._handles.append(blk.register_forward_hook(norm))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


def sgm_osfd_attack(model, orig_pixels, data_sample, steps: int, gamma: Optional[float] = None,
                    gamma_choices: Optional[Sequence[float]] = None, gamma_seed: Optional[int] = None,
                    epsilon: float = 5.0, alpha: float = 1.0, k: float = 3.0, momentum: float = 1.0,
                    rrb_theta: float = 7.0, rrb_l_s: int = 10, rrb_rho: float = 0.8,
                    rrb_s_max: float = 1.1, rrb_sigma: float = 6.0, info: Optional[dict] = None,
                    on_step=None):
    """OSFD full recipe (MI + RRB 2 view + loss feature backbone) với backward stage-balanced SGM.
    gamma: γ cố định (E1a/E1b). gamma_choices + gamma_seed: mỗi step chọn đều 1 γ (E2), numpy RNG riêng
    (không đụng random/torch toàn cục mà RRB dùng)."""
    assert (gamma is None) != (gamma_choices is None)
    clean_pixels = resize_to_model(orig_pixels, tuple(data_sample.img_shape))
    loss_fn = make_osfd_loss_fn(model, clean_pixels, data_sample, k=k)  # no_grad -> hook không tác dụng
    rng = np.random.default_rng(gamma_seed) if gamma_choices is not None else None
    used = []
    with StageBalancedSGM(model.backbone, gamma if gamma is not None else 1.0) as sgm:
        def views_fn(img, ds, step_idx, total_steps):
            if rng is not None:
                sgm.gamma = float(rng.choice(gamma_choices))
            used.append(sgm.gamma)
            return [(v, ds) for v in rrb_views(img, ds, theta=rrb_theta, l_s=rrb_l_s, rho=rrb_rho,
                                               s_max=rrb_s_max, sigma=rrb_sigma)]
        noise = run_iterative_attack(model, orig_pixels, data_sample, loss_fn=loss_fn, steps=steps,
                                     epsilon=epsilon, alpha=alpha, momentum=momentum,
                                     views_fn=views_fn, on_step=on_step)
    if info is not None:
        info["gammas"] = used
    return noise
