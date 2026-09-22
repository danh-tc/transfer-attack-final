"""Lớp glue khả vi (differentiable) giữa pixel tensor [0,255] và forward của mmdet v3.

Khác biệt cốt lõi so với mmdet v2 (xem ref-repo/OSFD-main/attack/utils/mmdet.py):
trong v3, normalize (mean/std) + pad + bgr2rgb nằm trong `model.data_preprocessor`
— một nn.Module thật sự, KHÔNG có torch.no_grad()/detach() nào bên trong (đã tự
verify bằng cách đọc source mmengine ImgDataPreprocessor.forward). Nghĩa là có thể
cộng noise adversarial thẳng vào tensor pixel THÔ (trước data_preprocessor) và
gradient sẽ lan truyền ngược tự nhiên qua toàn bộ normalize -> backbone -> head,
không cần tự denormalize/renormalize thủ công như v2.
"""
from typing import List, Tuple

import torch
from mmdet.structures import DetDataSample


def to_batch(model, pixel_tensors: List[torch.Tensor],
            data_samples: List[DetDataSample], training: bool = False
            ) -> Tuple[torch.Tensor, List[DetDataSample]]:
    """Chạy model.data_preprocessor trên 1 batch tensor pixel [0,255] (CHW mỗi ảnh).

    Trả về (batch_inputs đã normalize+pad [N,C,H,W], data_samples đã gắn
    batch_input_shape/pad_shape) — sẵn sàng feed vào model.loss/.predict/.extract_feat.
    Khả vi: nếu pixel_tensors[i].requires_grad, gradient lan được tới đó qua backward().
    """
    data = {"inputs": list(pixel_tensors), "data_samples": list(data_samples)}
    processed = model.data_preprocessor(data, training)
    return processed["inputs"], processed["data_samples"]


def compute_gt_loss(model, pixel_tensor: torch.Tensor, data_sample: DetDataSample
                    ) -> torch.Tensor:
    """Loss GT-assisted (task loss thật của detector) cho 1 ảnh — dùng cho các
    method cần tối đa hoá loss detection chuẩn (MI-FGSM/DI-FGSM, idea.md §3
    primary setting: GT-assisted).

    pixel_tensor: [C,H,W], giá trị kỳ vọng trong khoảng ~[0,255] (không bắt buộc
    clamp cứng ở đây — clamp epsilon-ball là việc của attack method, không phải
    của hàm loss).
    """
    batch_inputs, batch_data_samples = to_batch(model, [pixel_tensor], [data_sample])
    losses = model.loss(batch_inputs, batch_data_samples)
    return sum_loss_dict(losses)


def sum_loss_dict(losses: dict) -> torch.Tensor:
    """Cộng dồn mọi giá trị có chữ 'loss' trong tên key — khớp quy ước mmdet
    (loss dict có thể pha trộn tensor scalar và list[tensor], vd multi-level RPN loss)."""
    total = None
    for name, value in losses.items():
        if "loss" not in name:
            continue
        if isinstance(value, (list, tuple)):
            for v in value:
                total = v if total is None else total + v
        else:
            total = value if total is None else total + value
    if total is None:
        raise ValueError(f"Không tìm thấy key nào chứa 'loss' trong: {list(losses)}")
    return total


def extract_features(model, pixel_tensor: torch.Tensor, data_sample: DetDataSample
                     ) -> Tuple[torch.Tensor, ...]:
    """Multi-stage backbone+neck feature (tuple of [1,C,H,W]) — dùng cho OSFD
    (feature-disruption loss, không cần GT)."""
    batch_inputs, _ = to_batch(model, [pixel_tensor], [data_sample])
    return model.extract_feat(batch_inputs)


@torch.no_grad()
def predict(model, pixel_tensor: torch.Tensor, data_sample: DetDataSample,
           rescale: bool = True) -> DetDataSample:
    """Chạy inference đầy đủ (có post-process: NMS, rescale về ori_shape) — dùng
    để eval mAP (clean/adversarial) hoặc lấy pseudo-label cho secondary threat
    model (surrogate-prediction-only, idea.md §3)."""
    batch_inputs, batch_data_samples = to_batch(model, [pixel_tensor], [data_sample])
    results = model.predict(batch_inputs, batch_data_samples, rescale=rescale)
    return results[0]
