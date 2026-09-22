"""RRB (Random Rotation-Resizing-Blur) — port từ ref-repo/OSFD-main/attack/base/RRB.py.

Đây là phần "input diversity" riêng của paper OSFD (không phải DI-FGSM), dùng
để tạo full recipe "OSFD" đúng như base.yaml gốc định nghĩa: base_attack =
['MI', 'RRB'] + transfer_attack = OSFD. Sinh 2 view/step (batch-doubling):
view 1 = rotate(adv), view 2 = resize(rotate(adv)) — CHAIN chứ không phải 2
biến đổi độc lập trên cùng ảnh gốc (khớp đúng logic vòng lặp `for i in
range(2)` của bản gốc: mỗi vòng ghi đè lên `data_adv_imgs` rồi mới append).
Cả 2 view sau đó cùng bị làm mờ Gaussian.

Sửa 1 lỗi nhỏ so với bản gốc: điểm xoay fallback (khi không chọn GT box) bản
gốc dùng `[H//2, W//2]` như thể là toạ độ (x,y) — trong khi tâm GT box tính
đúng là (x,y) = (cx,cy). Với ảnh không vuông (H≠W, luôn đúng ở đây vì
Resize(keep_ratio=True)), lẫn lộn thứ tự này khiến điểm xoay fallback lệch
khỏi tâm ảnh thật. Ở đây dùng đúng (W//2, H//2).
"""
import random
from typing import List

import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample
from torchvision.transforms.functional import rotate


def _get_gt_boxes_xyxy(data_sample: DetDataSample) -> torch.Tensor:
    boxes = data_sample.gt_instances.bboxes
    return boxes.tensor if hasattr(boxes, "tensor") else boxes


def random_axis_rotation(img: torch.Tensor, gt_boxes: torch.Tensor,
                         max_angle: float = 7.0, max_center_jitter: int = 10) -> torch.Tensor:
    """img: [C,H,W]. Xoay quanh 1 điểm chọn ngẫu nhiên: tâm 1 GT box (có jitter) hoặc tâm ảnh."""
    x = img.unsqueeze(0)
    h, w = x.shape[-2:]
    device, dtype = img.device, img.dtype

    if len(gt_boxes) > 0:
        box_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # (cx,cy) mỗi box
        img_center = torch.tensor([[w / 2, h / 2]], device=device, dtype=box_centers.dtype)
        centers = torch.cat([box_centers, img_center], dim=0)
    else:
        centers = torch.tensor([[w / 2, h / 2]], device=device, dtype=dtype)

    if max_center_jitter > 0:
        centers = centers + torch.randint_like(centers, low=-max_center_jitter, high=max_center_jitter + 1)
    cx, cy = centers[random.randrange(len(centers))]
    angle = random.uniform(-max_angle, max_angle)
    rotated = rotate(x, angle, center=[float(cx), float(cy)])
    return rotated.squeeze(0)


def adaptive_random_resizing(img: torch.Tensor, gt_boxes: torch.Tensor,
                             rho: float = 0.8, s_max: float = 1.1) -> torch.Tensor:
    """img: [C,H,W]. Resize+pad với scale phụ thuộc kích thước 1 GT box ngẫu nhiên
    (box càng lớn thì scale-up càng nhiều, chặn trên bởi s_max) — giữ tỉ lệ H/W
    độc lập (khác lỗi ép-vuông của DI, xem attack/methods/diversity.py)."""
    x = img.unsqueeze(0)
    ori_h, ori_w = x.shape[-2:]

    if len(gt_boxes) > 0:
        box = gt_boxes[random.randrange(len(gt_boxes))]
        box_w = float(box[2] - box[0])
        box_h = float(box[3] - box[1])
    else:
        box_w = box_h = 0.0

    scale_h = min(1 + rho * (box_h / ori_h), s_max)
    scale_w = min(1 + rho * (box_w / ori_w), s_max)
    new_h = random.randint(ori_h, max(ori_h, int(scale_h * ori_h)))
    new_w = random.randint(ori_w, max(ori_w, int(scale_w * ori_w)))
    rescaled = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

    rem_h = int(scale_h * ori_h) - new_h
    rem_w = int(scale_w * ori_w) - new_w
    pad_top = random.randint(0, max(0, rem_h))
    pad_left = random.randint(0, max(0, rem_w))
    padded = F.pad(rescaled, (pad_left, max(0, rem_w) - pad_left, pad_top, max(0, rem_h) - pad_top),
                   mode="constant", value=0.0)
    resized_back = F.interpolate(padded, size=(ori_h, ori_w), mode="bilinear", align_corners=True)
    return resized_back.squeeze(0)


def gaussian_blur(img: torch.Tensor, sigma: float = 6.0) -> torch.Tensor:
    return torch.clamp(img + torch.randn_like(img) * sigma, min=0.0, max=255.0)


def rrb_views(adv_pixels: torch.Tensor, data_sample: DetDataSample,
             prob: float = 1.0, theta: float = 7.0, l_s: int = 10,
             rho: float = 0.8, s_max: float = 1.1, sigma: float = 6.0) -> List[torch.Tensor]:
    """Sinh 2 view theo đúng chain của bản gốc: view1 = rotate(adv) (nếu trúng prob),
    view2 = resize(view1) (nếu trúng prob) — rồi cả 2 cùng bị Gaussian blur.
    Hyperparameter mặc định khớp ref-repo/OSFD-main/config/base.yaml (không phải
    default __init__ của class RRB — base.yaml là config thực nghiệm đã dùng)."""
    # data_sample (và gt_instances bên trong) không tự động theo device của ảnh —
    # AttackDataset trả về data_sample nguyên bản từ pipeline (CPU), chỉ tensor ảnh
    # mới được caller .to(device) thủ công. Phải tự đồng bộ device ở đây.
    gt_boxes = _get_gt_boxes_xyxy(data_sample).to(adv_pixels.device)

    view = adv_pixels
    if random.random() < prob:
        view = random_axis_rotation(view, gt_boxes, max_angle=theta, max_center_jitter=l_s)
    view1 = view

    if random.random() < prob:
        view = adaptive_random_resizing(view, gt_boxes, rho=rho, s_max=s_max)
    view2 = view

    return [gaussian_blur(view1, sigma=sigma), gaussian_blur(view2, sigma=sigma)]
