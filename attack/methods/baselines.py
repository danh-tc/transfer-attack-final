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

steps truyền vào PHẢI bằng B (gradient-evaluation budget) — cả 3 method đều
1 backward/iteration nên steps=B trực tiếp (docs/protocol_lock.md).

Quyết định CHƯA chốt, cần bàn trước khi chạy baseline table thật: OSFD trong
ref-repo gốc (base.yaml) chạy full recipe MI+RRB+OSFD-loss, không phải OSFD-loss
đơn lẻ. Ở đây osfd_attack() mới port phần loss (OSFD-loss) ghép với update rule
IFGSM thuần (momentum=0, không diversity) — CHƯA gồm RRB (attack/base/RRB.py,
chưa port) hay MI. Nếu muốn tái tạo đúng "OSFD" như paper gốc định nghĩa, cần
port thêm RRB và đổi osfd_attack() sang momentum=1.0 (MI) trước khi chạy
baseline table chính thức.
"""
from mmdet.structures import DetDataSample

from attack.losses.osfd import make_osfd_loss_fn
from attack.methods.core import run_iterative_attack
from attack.methods.diversity import input_diversity
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
    diversity_fn = lambda img: input_diversity(img, prob=prob, scale=scale)
    return run_iterative_attack(
        model, clean_pixels, data_sample, loss_fn=compute_gt_loss,
        steps=steps, epsilon=epsilon, alpha=alpha, diversity_fn=diversity_fn)


def osfd_attack(model, clean_pixels, data_sample: DetDataSample, steps: int,
                epsilon: float = EPSILON_PRIMARY, alpha: float = ALPHA_DEFAULT,
                k: float = 3.0):
    loss_fn = make_osfd_loss_fn(model, clean_pixels, data_sample, k=k)
    return run_iterative_attack(
        model, clean_pixels, data_sample, loss_fn=loss_fn,
        steps=steps, epsilon=epsilon, alpha=alpha)
