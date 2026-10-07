"""3 trong 4 baseline đã khoá ở idea.md §8: MI-FGSM, DI-FGSM, OSFD.
(AugTrans tạm bỏ khỏi plan — idea.md §8.)

Hyperparameter mặc định:
- epsilon, alpha: docs/protocol_lock.md (epsilon primary = 5/255 -> 5.0 trong
  pixel-space [0,255] dùng ở đây; alpha=1.0 theo config mặc định của
  ref-repo/OSFD-main/config/default.py).
- momentum (MI-FGSM): 1.0, theo paper gốc Dong et al. CVPR'18 (giá trị mu phổ
  biến nhất trong literature; ref-repo cũng để default 1.0).
- DI-FGSM = M-DI²-FGSM (MI + DI, momentum 1.0 — chốt 2026-09-23, cùng có MI như
  OSFD cho so sánh công bằng); prob/scale 1.0 / 1.1 theo ref-repo/OSFD-main/
  config/default.py (cùng nguồn config với OSFD).
- k (OSFD): 3.0, theo docs/protocol_lock.md / config/default.py của ref-repo.
- OSFD full recipe (momentum, RRB theta/l_s/rho/s_max/sigma): khớp
  ref-repo/OSFD-main/config/base.yaml — đây là config thực nghiệm tác giả
  OSFD đã dùng để chạy paper, không phải default __init__ khiêm tốn hơn của
  từng class riêng lẻ.

orig_pixels: ảnh GỐC (AttackDataset `orig_inputs`) — noise trả về cùng shape, ở
không gian ảnh gốc; ảnh adversarial thật = attack.preprocess.to_adv_image(orig, noise).

steps truyền vào PHẢI bằng B (gradient-evaluation budget) — cả 3 method đều
1 backward/iteration nên steps=B trực tiếp (docs/protocol_lock.md); RRB
batch-doubling bên trong osfd_attack() vẫn tính là 1 backward/step (xem
attack/methods/core.py).

osfd_attack() mặc định dùng FULL RECIPE của paper (MI + RRB + OSFD-loss,
khớp base.yaml) — truyền use_rrb=False, momentum=0.0 nếu muốn bản OSFD-loss
đơn lẻ (ablation: tách riêng đóng góp của loss so với input-diversity).
"""
import torch
from mmdet.structures import DetDataSample

from attack.losses.osfd import make_osfd_loss_fn
from attack.methods.core import run_iterative_attack
from attack.methods.diversity import input_diversity_with_boxes
from attack.methods.rrb import rrb_views
from attack.preprocess import (compute_gt_loss, extract_features, make_feature_plus_gt_loss,
                               make_gt_loss, resize_to_model)

EPSILON_PRIMARY = 5.0  # 5/255 quy về pixel-space [0,255]
EPSILON_SECONDARY = 8.0
ALPHA_DEFAULT = 1.0


def mi_fgsm_attack(model, orig_pixels, data_sample: DetDataSample, steps: int,
                   epsilon: float = EPSILON_PRIMARY, alpha: float = ALPHA_DEFAULT,
                   momentum: float = 1.0, on_step=None):
    return run_iterative_attack(
        model, orig_pixels, data_sample, loss_fn=compute_gt_loss,
        steps=steps, epsilon=epsilon, alpha=alpha, momentum=momentum, on_step=on_step)


def di_fgsm_attack(model, orig_pixels, data_sample: DetDataSample, steps: int,
                   epsilon: float = EPSILON_PRIMARY, alpha: float = ALPHA_DEFAULT,
                   momentum: float = 1.0, prob: float = 1.0, scale: float = 1.1,
                   on_step=None):
    # input_diversity_with_boxes co-transform GT box khớp ảnh đã resize+pad —
    # bắt buộc vì compute_gt_loss cần GT đúng vị trí (xem attack/methods/
    # diversity.py, cùng loại bug đã fix cho AugTrans).
    views_fn = lambda img, ds, k, k_max: [input_diversity_with_boxes(img, ds, prob=prob, scale=scale)]
    return run_iterative_attack(
        model, orig_pixels, data_sample, loss_fn=compute_gt_loss,
        steps=steps, epsilon=epsilon, alpha=alpha, momentum=momentum, views_fn=views_fn,
        on_step=on_step)


def osfd_attack(model, orig_pixels, data_sample: DetDataSample, steps: int,
                epsilon: float = EPSILON_PRIMARY, alpha: float = ALPHA_DEFAULT,
                k: float = 3.0, use_rrb: bool = True, momentum: float = 1.0,
                rrb_theta: float = 7.0, rrb_l_s: int = 10, rrb_rho: float = 0.8,
                rrb_s_max: float = 1.1, rrb_sigma: float = 6.0, on_step=None,
                stages=None, spatial_weight=None, channel_concave=None,
                loss: str = "feature", rrb_rotate: bool = True, rrb_resize: bool = True,
                rrb_noise: bool = True, task_terms=None, cls_weight: float = 0.0,
                grad_lowpass_f0=None, vc_k=None, combine="sum", rrb_draws: int = 1):
    # Feature sạch lấy từ CÙNG phép resize khả vi dùng cho adv (core.py) — tại δ=0
    # adv trùng tuyệt đối clean, loss chỉ đo phần lệch do δ gây ra.
    # spatial_weight: pilot A′1 ("box" / "box_ring"); channel_concave: pilot P2
    # ("median" / "2median") — attack/losses/osfd.py.
    # Ablation OSFD (docs/plan_sota.md): loss="task" thay loss feature bằng task loss GT
    # (RRB khi đó co-transform GT box + mask); rrb_rotate/resize/noise tắt từng phần RRB.
    assert loss in ("feature", "task", "feature+cls")
    if loss == "feature+cls":
        # Probe: L_OSFD + cls_weight · (loss_rpn_cls + loss_cls), 1 forward/view (progress_log 2026-10-03).
        clean_pixels = resize_to_model(orig_pixels, tuple(data_sample.img_shape))
        with torch.no_grad():
            feats_clean = tuple(f.detach() for f in extract_features(model, clean_pixels, data_sample))
        loss_fn = make_feature_plus_gt_loss(feats_clean, k, ("loss_rpn_cls", "loss_cls"), cls_weight)
    elif loss == "task":
        # task_terms: None = đủ 5 loss Mask R-CNN; tuple key = chỉ các thành phần đó.
        loss_fn = compute_gt_loss if task_terms is None else make_gt_loss(task_terms)
    else:
        clean_pixels = resize_to_model(orig_pixels, tuple(data_sample.img_shape))
        loss_fn = make_osfd_loss_fn(model, clean_pixels, data_sample, k=k, stages=stages,
                                    spatial_weight=spatial_weight, channel_concave=channel_concave)
    views_fn = None
    if use_rrb:
        rrb_kw = dict(theta=rrb_theta, l_s=rrb_l_s, rho=rrb_rho, s_max=rrb_s_max, sigma=rrb_sigma,
                      use_rotate=rrb_rotate, use_resize=rrb_resize, use_noise=rrb_noise)
        if loss in ("task", "feature+cls"):
            views_fn = lambda img, ds, k, k_max: rrb_views(img, ds, with_gt=True, **rrb_kw)
        else:
            # Loss feature không phụ thuộc gt_instances -> trả lại data_sample gốc nguyên vẹn.
            # rrb_draws > 1 (pilot V, progress_log 2026-10-07): gộp nhiều lần rút RRB độc lập vào CÙNG 1 backward.
            views_fn = lambda img, ds, k, k_max: [(view, ds) for _ in range(rrb_draws)
                                                  for view in rrb_views(img, ds, **rrb_kw)]
    grad_transform = None
    if grad_lowpass_f0 is not None:
        grad_transform = lambda g: gaussian_lowpass(g, grad_lowpass_f0)
    # Pilot VC (progress_log 2026-10-07): vc_k backward / step, mỗi backward 1 lần rút RRB độc lập; giữ B = steps
    # backward tổng -> steps // vc_k bước cập nhật; combine "sum" (đối chứng OSFD-K2) | "consensus" (VC-OSFD).
    bps = 1
    if vc_k is not None:
        steps, bps = steps // vc_k, vc_k
    return run_iterative_attack(
        model, orig_pixels, data_sample, loss_fn=loss_fn,
        steps=steps, epsilon=epsilon, alpha=alpha,
        momentum=momentum, views_fn=views_fn, on_step=on_step, grad_transform=grad_transform,
        backward_per_step=bps, combine=combine)


def gaussian_lowpass(g: torch.Tensor, f0: float) -> torch.Tensor:
    """Lọc thông thấp Gaussian trong miền tần số: H(f) = exp(−f²/(2·f0²)), f = bán kính (cycles/pixel)
    trên 2 trục cuối; g [C,H,W] ở không gian ảnh gốc. Nhánh S1 (progress_log 2026-10-03)."""
    h, w = g.shape[-2:]
    fy = torch.fft.fftfreq(h, device=g.device)[:, None]
    fx = torch.fft.fftfreq(w, device=g.device)[None, :]
    H = torch.exp(-(fx ** 2 + fy ** 2) / (2 * f0 ** 2))
    return torch.fft.ifft2(torch.fft.fft2(g) * H).real
