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
- `views_fn` (thay cho "diversity_fn" đơn giản trước đây): trả về 1 LIST cặp
  (ảnh, data_sample) từ 1 ảnh adv — length 1 cho DI-FGSM, length 2 cho RRB
  (batch-doubling, xem attack/methods/rrb.py), length N_EOT cho AugTrans
  (xem attack/methods/augtrans.py). MỖI VIEW MANG data_sample RIÊNG vì biến
  đổi hình học (rotate/resize) làm object DỊCH CHUYỂN trong ảnh — nếu loss
  phụ thuộc GT (vd AugTrans/DI-FGSM dùng compute_gt_loss) mà vẫn dùng
  data_sample gốc (box CHƯA biến đổi) trong khi ảnh ĐÃ biến đổi, loss tính
  sai vị trí hoàn toàn, gradient gần như vô nghĩa — đây từng là bug thật
  (xem docs/progress_log.md, attack/methods/box_transforms.py). Với
  view không đổi box (RRB — loss OSFD không phụ thuộc GT), cứ trả lại
  data_sample gốc nguyên vẹn. Loss của mỗi view được CỘNG lại trước 1 lần
  .backward() duy nhất — khớp đúng định nghĩa B trong protocol_lock.md
  ("concat 2 view rồi backward 1 lần tính là B=1"): cộng 2 loss riêng rồi
  backward 1 lần tương đương toán học với batch 2 ảnh rồi backward 1 lần
  (autograd cộng gradient qua cả 2 đường như nhau), đơn giản hơn nhiều so
  với phải ghép batch thật qua model.data_preprocessor. Nhận thêm
  (step_idx, total_steps) vì AugTrans cần biết tiến độ curriculum
  (Algorithm 1, θ_max(k) phụ thuộc k/K_max) — DI/RRB không dùng, bỏ qua.
"""
from typing import Callable, List, Optional, Tuple

import torch
from mmdet.structures import DetDataSample

LossFn = Callable[[torch.nn.Module, torch.Tensor, DetDataSample], torch.Tensor]
ViewsFn = Callable[[torch.Tensor, DetDataSample, int, int], List[Tuple[torch.Tensor, DetDataSample]]]


def run_iterative_attack(
    model: torch.nn.Module,
    clean_pixels: torch.Tensor,
    data_sample: DetDataSample,
    loss_fn: LossFn,
    steps: int,
    epsilon: float,
    alpha: float = 1.0,
    momentum: float = 0.0,
    views_fn: Optional[ViewsFn] = None,
) -> torch.Tensor:
    """Trả về noise tensor cuối cùng (cùng shape clean_pixels), đã clamp [-epsilon, epsilon].

    momentum=0.0  -> IFGSM thuần (không tích lũy động lượng).
    momentum>0.0  -> MI-FGSM (khớp ref: chuẩn hoá gradient bằng mean(abs(.)) trước khi
                     tích lũy động lượng — attack/base/MI.py).
    views_fn      -> DI-FGSM nếu truyền `lambda img, ds: [input_diversity(img)]`, hoặc
                     RRB nếu truyền `attack.methods.rrb.rrb_views` (2 view/step).

    steps PHẢI bằng B (gradient-evaluation budget, docs/protocol_lock.md) — mỗi step
    đúng 1 lần .backward() qua surrogate (qua torch.autograd.grad), không hơn, bất kể
    views_fn sinh ra bao nhiêu view.
    """
    noise = torch.zeros_like(clean_pixels)
    accumulated_grad = torch.zeros_like(clean_pixels)

    for step_idx in range(steps):
        noise = noise.detach().requires_grad_(True)
        adv_pixels = torch.clamp(clean_pixels + noise, min=0.0, max=255.0)
        views = (views_fn(adv_pixels, data_sample, step_idx, steps)
                if views_fn is not None else [(adv_pixels, data_sample)])

        loss = None
        for view_img, view_data_sample in views:
            term = loss_fn(model, view_img, view_data_sample)
            loss = term if loss is None else loss + term
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
