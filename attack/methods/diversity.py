"""DI-FGSM input diversity: random resize + random pad, khả vi.

Port từ ref-repo/OSFD-main/attack/base/DI.py — có sửa 2 chỗ:

1. Bản gốc chỉ dùng `imgs.shape[2]` (chiều cao) làm kích thước mục tiêu cho
   CẢ 2 chiều khi resize/pad, tức là ép ảnh về hình vuông. DI-FGSM gốc (Xie
   et al., CVPR'19) thiết kế cho ảnh classification vốn đã vuông sẵn
   (ImageNet crop) nên không gặp vấn đề; nhưng ảnh detection sau
   Resize(keep_ratio=True) (vd ~1196x800) không vuông — ép vuông sẽ méo tỉ
   lệ khung hình nghiêm trọng. Ở đây xử lý H/W độc lập.
2. **Co-transform GT box theo đúng biến đổi ảnh.** Khi ghép DI với loss phụ
   thuộc GT (compute_gt_loss, dùng cho di_fgsm_attack — attack/methods/
   baselines.py), nếu box vẫn giữ nguyên toạ độ gốc trong khi ảnh đã bị
   resize+pad, model.loss() tính loss trên vị trí SAI hoàn toàn, gradient
   gần như vô nghĩa. Đây từng là bug thật với AugTrans (xem docs/
   progress_log.md, attack/methods/box_transforms.py) — DI-FGSM dùng cùng
   pattern "resize/pad ảnh + loss phụ thuộc GT" nên mắc lỗi tương tự, sửa
   luôn ở đây cho nhất quán.
"""
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample


def input_diversity(img: torch.Tensor, prob: float = 0.7, scale: float = 1.1) -> torch.Tensor:
    """img: [C,H,W]. Trả về ảnh đã biến đổi (hoặc nguyên bản nếu không trúng prob).
    Bản KHÔNG co-transform box — chỉ dùng khi loss không phụ thuộc GT (hiện
    không có baseline nào dùng bản này; giữ lại vì đơn giản/dễ test riêng)."""
    if random.random() >= prob:
        return img
    img2, _ = _apply(img, scale)
    return img2


def input_diversity_with_boxes(img: torch.Tensor, data_sample: DetDataSample,
                               prob: float = 0.7, scale: float = 1.1
                               ) -> Tuple[torch.Tensor, DetDataSample]:
    """Như input_diversity, nhưng đồng thời trả về data_sample đã co-transform
    gt_instances.bboxes khớp đúng ảnh đã biến đổi — dùng cho di_fgsm_attack."""
    if random.random() >= prob:
        return img, data_sample
    img2, box_transform = _apply(img, scale)

    import copy
    ds2 = copy.deepcopy(data_sample)
    boxes = ds2.gt_instances.bboxes
    boxes = boxes.tensor if hasattr(boxes, "tensor") else boxes
    ds2.gt_instances.bboxes = box_transform(boxes.to(img.device))
    return img2, ds2


def _apply(img: torch.Tensor, scale: float):
    """Thực hiện resize-up -> pad -> resize-down, trả về (ảnh mới, hàm biến
    đổi box tương ứng). Tách hàm để dùng chung cho cả 2 API ở trên."""
    x = img.unsqueeze(0)
    _, _, h, w = x.shape
    new_h = random.randint(h, int(scale * h))
    new_w = random.randint(w, int(scale * w))
    rescaled = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

    canvas_h, canvas_w = int(scale * h), int(scale * w)
    rem_h = canvas_h - new_h
    rem_w = canvas_w - new_w
    pad_top = random.randint(0, rem_h)
    pad_left = random.randint(0, rem_w)
    padded = F.pad(rescaled, (pad_left, rem_w - pad_left, pad_top, rem_h - pad_top),
                   mode="constant", value=0.0)

    resized_back = F.interpolate(padded, size=(h, w), mode="bilinear", align_corners=True)
    img_out = resized_back.squeeze(0)

    def box_transform(boxes: torch.Tensor) -> torch.Tensor:
        if len(boxes) == 0:
            return boxes
        b = boxes.clone().float()
        # (1) resize gốc (h,w) -> (new_h,new_w)
        sx1, sy1 = new_w / w, new_h / h
        b[:, 0] *= sx1; b[:, 2] *= sx1
        b[:, 1] *= sy1; b[:, 3] *= sy1
        # (2) dịch theo offset pad trong canvas (canvas_h,canvas_w)
        b[:, 0] += pad_left; b[:, 2] += pad_left
        b[:, 1] += pad_top; b[:, 3] += pad_top
        # (3) resize canvas (canvas_h,canvas_w) -> lại (h,w)
        sx2, sy2 = w / canvas_w, h / canvas_h
        b[:, 0] *= sx2; b[:, 2] *= sx2
        b[:, 1] *= sy2; b[:, 3] *= sy2
        b[:, 0] = b[:, 0].clamp(0, w); b[:, 2] = b[:, 2].clamp(0, w)
        b[:, 1] = b[:, 1].clamp(0, h); b[:, 3] = b[:, 3].clamp(0, h)
        return b

    return img_out, box_transform
