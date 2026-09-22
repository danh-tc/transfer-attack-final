"""Attack core: engine chung cho IFGSM / MI-FGSM / DI-FGSM (sign-gradient iterative attack).

Port từ ref-repo/OSFD-main/attack/base/{IFGSM,MI,DI}.py, gộp lại thành 1 hàm
thay vì hệ thống registry + pipeline 4-giai-đoạn của bản gốc — bản gốc cần
kiểu đó để tổ hợp tự do nhiều method + nhiều buffer/cache cho nhiều thí
nghiệm khác nhau trong paper OSFD; ở đây chỉ cần 3 biến thể cố định
(IFGSM/MI/DI) nên không cần abstraction đó.

Quy ước:
- Không gian pixel [0,255] (KHÔNG normalize) — normalize xảy ra bên trong
  model.data_preprocessor, xem attack/preprocess.py.
- epsilon, alpha tính bằng đơn vị pixel tuyệt đối (vd epsilon=5 ứng với
  epsilon=5/255 trong idea.md/protocol_lock.md, vì pixel ở đây scale
  0-255 chứ không phải 0-1).
- Cả 3 biến thể đều là GRADIENT ASCENT trên loss_fn (loss càng cao càng tốt
  cho attacker, đúng untargeted threat model idea.md §3) — kể cả khi ghép
  với loss của OSFD (xem attack/losses/osfd.py để hiểu vì sao ascent vẫn
  đúng hướng cho loss đó).
"""
from typing import Callable, Optional

import torch
from mmdet.structures import DetDataSample

LossFn = Callable[[torch.nn.Module, torch.Tensor, DetDataSample], torch.Tensor]
DiversityFn = Callable[[torch.Tensor], torch.Tensor]


def run_iterative_attack(
    model: torch.nn.Module,
    clean_pixels: torch.Tensor,
    data_sample: DetDataSample,
    loss_fn: LossFn,
    steps: int,
    epsilon: float,
    alpha: float = 1.0,
    momentum: float = 0.0,
    diversity_fn: Optional[DiversityFn] = None,
) -> torch.Tensor:
    """Trả về noise tensor cuối cùng (cùng shape clean_pixels), đã clamp [-epsilon, epsilon].

    momentum=0.0  -> IFGSM thuần (không tích lũy động lượng).
    momentum>0.0  -> MI-FGSM (khớp ref: chuẩn hoá gradient bằng mean(abs(.)) trước khi
                     tích lũy động lượng — attack/base/MI.py).
    diversity_fn  -> DI-FGSM nếu truyền attack.methods.diversity.input_diversity
                     (áp dụng lên ảnh trước khi forward, KHÔNG áp dụng lên noise lưu lại).

    steps PHẢI bằng B (gradient-evaluation budget, docs/protocol_lock.md) — mỗi step
    đúng 1 lần .backward() qua surrogate (qua torch.autograd.grad), không hơn.
    """
    noise = torch.zeros_like(clean_pixels)
    accumulated_grad = torch.zeros_like(clean_pixels)

    for _ in range(steps):
        noise = noise.detach().requires_grad_(True)
        adv_pixels = torch.clamp(clean_pixels + noise, min=0.0, max=255.0)
        forward_input = diversity_fn(adv_pixels) if diversity_fn is not None else adv_pixels

        loss = loss_fn(model, forward_input, data_sample)
        grad = torch.autograd.grad(loss, noise)[0]

        if momentum > 0.0:
            grad = grad / (grad.abs().mean() + 1e-12)
            accumulated_grad = momentum * accumulated_grad + grad
            update_direction = accumulated_grad
        else:
            update_direction = grad

        with torch.no_grad():
            noise = noise + alpha * torch.sign(update_direction)
            noise = torch.clamp(noise, min=-epsilon, max=epsilon)

    return noise.detach()
