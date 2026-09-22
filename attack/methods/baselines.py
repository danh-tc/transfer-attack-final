"""3 trong 4 baseline đã khoá ở idea.md §8: MI-FGSM, DI-FGSM, OSFD.
(AugTrans chưa port — không có ref code, phải viết từ papers/AugTrans.pdf.)

Hyperparameter mặc định:
- epsilon, alpha: docs/protocol_lock.md (epsilon primary = 5/255 -> 5.0 trong
  pixel-space [0,255] dùng ở đây; alpha=1.0 theo config mặc định của
  ref-repo/OSFD-main/config/default.py).
- momentum (MI-FGSM): 1.0, theo paper gốc Dong et al. CVPR'18 (giá trị mu phổ
  biến nhất trong literature; ref-repo cũng để default 1.0).
- prob/scale (DI-FGSM): 0.7 / 1.1, theo attack/base/DI.py.__init__ của ref-repo.
- k (OSFD): 3.0, theo docs/protocol_lock.md / config/default.py của ref-repo.
- OSFD full recipe (momentum, RRB theta/l_s/rho/s_max/sigma): khớp
  ref-repo/OSFD-main/config/base.yaml — đây là config thực nghiệm tác giả
  OSFD đã dùng để chạy paper, không phải default __init__ khiêm tốn hơn của
  từng class riêng lẻ.

steps truyền vào PHẢI bằng B (gradient-evaluation budget) — cả 3 method đều
1 backward/iteration nên steps=B trực tiếp (docs/protocol_lock.md); RRB
batch-doubling bên trong osfd_attack() vẫn tính là 1 backward/step (xem
attack/methods/core.py).

osfd_attack() mặc định dùng FULL RECIPE của paper (MI + RRB + OSFD-loss,
khớp base.yaml) — truyền use_rrb=False, momentum=0.0 nếu muốn bản OSFD-loss
đơn lẻ (ablation: tách riêng đóng góp của loss so với input-diversity).
"""
from mmdet.structures import DetDataSample

from attack.losses.osfd import make_osfd_loss_fn
from attack.methods.core import run_iterative_attack
from attack.methods.diversity import input_diversity
from attack.methods.rrb import rrb_views
from attack.preprocess import compute_gt_loss

EPSILON_PRIMARY = 5.0  # 5/255 quy về pixel-space [0,255]
EPSILON_SECONDARY = 8.0
ALPHA_DEFAULT = 1.0


def mi_fgsm_attack(model, clean_pixels, data_sample: DetDataSample, steps: int,
                   epsilon: float = EPSILON_PRIMARY, alpha: float = ALPHA_DEFAULT,
                   momentum: float = 1.0):
    return run_iterative_attack(
        model, clean_pixels, data_sample, loss_fn=compute_gt_loss,
        steps=steps, epsilon=epsilon, alpha=alpha, momentum=momentum)


def di_fgsm_attack(model, clean_pixels, data_sample: DetDataSample, steps: int,
                   epsilon: float = EPSILON_PRIMARY, alpha: float = ALPHA_DEFAULT,
                   prob: float = 0.7, scale: float = 1.1):
    views_fn = lambda img, ds: [input_diversity(img, prob=prob, scale=scale)]
    return run_iterative_attack(
        model, clean_pixels, data_sample, loss_fn=compute_gt_loss,
        steps=steps, epsilon=epsilon, alpha=alpha, views_fn=views_fn)


def osfd_attack(model, clean_pixels, data_sample: DetDataSample, steps: int,
                epsilon: float = EPSILON_PRIMARY, alpha: float = ALPHA_DEFAULT,
                k: float = 3.0, use_rrb: bool = True, momentum: float = 1.0,
                rrb_theta: float = 7.0, rrb_l_s: int = 10, rrb_rho: float = 0.8,
                rrb_s_max: float = 1.1, rrb_sigma: float = 6.0):
    loss_fn = make_osfd_loss_fn(model, clean_pixels, data_sample, k=k)
    views_fn = None
    if use_rrb:
        views_fn = lambda img, ds: rrb_views(
            img, ds, theta=rrb_theta, l_s=rrb_l_s, rho=rrb_rho,
            s_max=rrb_s_max, sigma=rrb_sigma)
    return run_iterative_attack(
        model, clean_pixels, data_sample, loss_fn=loss_fn,
        steps=steps, epsilon=epsilon, alpha=alpha,
        momentum=momentum, views_fn=views_fn)
