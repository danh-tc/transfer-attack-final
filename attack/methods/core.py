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
from typing import Callable, List, Optional

import torch
from mmdet.structures import DetDataSample

from attack.preprocess import resize_to_model

LossFn = Callable[[torch.nn.Module, torch.Tensor, DetDataSample], torch.Tensor]
# Mỗi view: (ảnh, data_sample) hoặc (ảnh, data_sample, loss_fn riêng của view).
ViewsFn = Callable[[torch.Tensor, DetDataSample, int, int], List[tuple]]


def run_iterative_attack(
    model: torch.nn.Module,
    orig_pixels: torch.Tensor,
    data_sample: DetDataSample,
    loss_fn: LossFn,
    steps: int,
    epsilon: float,
    alpha: float = 1.0,
    momentum: float = 0.0,
    views_fn: Optional[ViewsFn] = None,
    on_step: Optional[Callable[[int, torch.Tensor], None]] = None,
    grad_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    backward_per_step: int = 1,
    combine: str = "sum",
) -> torch.Tensor:
    """Trả về noise tensor cuối cùng (cùng shape orig_pixels), đã clamp [-epsilon, epsilon].

    orig_pixels: ảnh GỐC [C,ori_h,ori_w] (AttackDataset `orig_inputs`). δ được tối ưu ở
    CHÍNH không gian này, mỗi step ảnh adv được resize KHẢ VI về khung model
    (data_sample.img_shape) trước khi vào surrogate — ảnh adversarial thật (cỡ gốc,
    uint8, attack.preprocess.to_adv_image) giữ được toàn bộ δ. Bản trước tối ưu δ ở
    khung đã resize (upsample ~1.87x) nên khi đưa về ảnh thật mất ~64% biên độ
    (xem docs/progress_log.md 2026-09-23).

    momentum=0.0  -> IFGSM thuần (không tích lũy động lượng).
    momentum>0.0  -> MI-FGSM (khớp ref: chuẩn hoá gradient bằng mean(abs(.)) trước khi
                     tích lũy động lượng — attack/base/MI.py).
    views_fn      -> DI-FGSM nếu truyền `lambda img, ds: [input_diversity(img)]`, hoặc
                     RRB nếu truyền `attack.methods.rrb.rrb_views` (2 view/step).

    steps PHẢI bằng B (gradient-evaluation budget, docs/protocol_lock.md) — mỗi step
    đúng 1 lần .backward() qua surrogate (qua torch.autograd.grad), không hơn, bất kể
    views_fn sinh ra bao nhiêu view.

    backward_per_step > 1 (chỉ ô mô tả PIA-N20, progress_log 2026-10-04): mỗi step gọi views_fn
    + backward lặp lại bấy nhiêu lần và CỘNG gradient (cụm view không vừa 1 graph) — B thực
    = steps × backward_per_step. Mặc định 1 trùng tuyệt đối bản cũ.

    combine (pilot VC, progress_log 2026-10-07) — cách gộp gradient của backward_per_step lần backward:
    "sum" (mặc định, như cũ); "consensus" = trung bình, chỉ giữ pixel mà MỌI gradient cùng dấu (0 ở chỗ còn lại).
    """
    assert combine in ("sum", "consensus")
    model_size = tuple(data_sample.img_shape)
    noise = torch.zeros_like(orig_pixels)
    accumulated_grad = torch.zeros_like(orig_pixels)

    for step_idx in range(steps):
        noise = noise.detach().requires_grad_(True)
        grad = None
        gs = []
        for _ in range(backward_per_step):
            adv_orig = torch.clamp(orig_pixels + noise, min=0.0, max=255.0)
            adv_pixels = resize_to_model(adv_orig, model_size)
            views = (views_fn(adv_pixels, data_sample, step_idx, steps)
                    if views_fn is not None else [(adv_pixels, data_sample)])

            loss = None
            for view in views:
                # Phần tử thứ 3 (tùy chọn) = loss riêng của view — pilot KG: view cặp so với
                # feature sạch của CHÍNH vùng crop (attack/methods/kg_pair.py).
                view_img, view_data_sample = view[0], view[1]
                view_loss_fn = view[2] if len(view) > 2 else loss_fn
                term = view_loss_fn(model, view_img, view_data_sample)
                loss = term if loss is None else loss + term
            g = torch.autograd.grad(loss, noise)[0]
            grad = g if grad is None else grad + g
            if combine == "consensus":
                gs.append(g)
        if combine == "consensus":
            s = torch.stack([torch.sign(x) for x in gs])
            agree = (s == s[0]).all(0) & (s[0] != 0)
            grad = grad / len(gs) * agree
        if grad_transform is not None:  # vd lọc thông thấp gradient (nhánh S1, progress_log 2026-10-03)
            grad = grad_transform(grad)

        if momentum > 0.0:
            grad = grad / (grad.abs().mean() + 1e-12)
            accumulated_grad = momentum * accumulated_grad + grad
            update_direction = accumulated_grad
        else:
            update_direction = grad

        with torch.no_grad():
            noise = noise + alpha * torch.sign(update_direction)
            noise = torch.clamp(noise, min=-epsilon, max=epsilon)
        if on_step is not None:
            # (số backward đã dùng, noise hiện tại) — noise sau k step đúng bằng kết quả
            # của lần chạy steps=k (không gì trong vòng lặp phụ thuộc `steps`, trừ
            # views_fn của AugTrans), nên 1 lần chạy B lớn cho cả đường cong theo B.
            on_step(step_idx + 1, noise.detach())

    return noise.detach()
