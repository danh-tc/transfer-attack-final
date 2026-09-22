"""DI-FGSM input diversity: random resize + random pad, khả vi.

Port từ ref-repo/OSFD-main/attack/base/DI.py — có sửa 1 chỗ: bản gốc chỉ dùng
`imgs.shape[2]` (chiều cao) làm kích thước mục tiêu cho CẢ 2 chiều khi
resize/pad, tức là ép ảnh về hình vuông. DI-FGSM gốc (Xie et al., CVPR'19)
thiết kế cho ảnh classification vốn đã vuông sẵn (ImageNet crop) nên không
gặp vấn đề; nhưng ảnh detection sau Resize(keep_ratio=True) (vd ~1196x800)
không vuông — ép vuông sẽ méo tỉ lệ khung hình nghiêm trọng, không đúng tinh
thần "input diversity" (thay đổi scale nhẹ, không phá cấu trúc ảnh). Ở đây xử
lý H/W độc lập, giống cách attack/base/RRB.py's adaptive_random_resizing đã
làm đúng trong cùng ref repo.
"""
import random

import torch
import torch.nn.functional as F


def input_diversity(img: torch.Tensor, prob: float = 0.7, scale: float = 1.1) -> torch.Tensor:
    """img: [C,H,W]. Trả về ảnh đã biến đổi (hoặc nguyên bản nếu không trúng prob)."""
    if random.random() >= prob:
        return img

    x = img.unsqueeze(0)
    _, _, h, w = x.shape
    new_h = random.randint(h, int(scale * h))
    new_w = random.randint(w, int(scale * w))
    rescaled = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

    rem_h = int(scale * h) - new_h
    rem_w = int(scale * w) - new_w
    pad_top = random.randint(0, rem_h)
    pad_left = random.randint(0, rem_w)
    padded = F.pad(rescaled, (pad_left, rem_w - pad_left, pad_top, rem_h - pad_top),
                   mode="constant", value=0.0)

    resized_back = F.interpolate(padded, size=(h, w), mode="bilinear", align_corners=True)
    return resized_back.squeeze(0)
