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

from attack.methods.box_transforms import get_gt, rotate_boxes
from attack.methods.box_transforms import with_gt as gt_with


def _get_gt_boxes_xyxy(data_sample: DetDataSample) -> torch.Tensor:
    boxes = data_sample.gt_instances.bboxes
    return boxes.tensor if hasattr(boxes, "tensor") else boxes


def random_axis_rotation(img: torch.Tensor, gt_boxes: torch.Tensor,
                         max_angle: float = 7.0, max_center_jitter: int = 10) -> torch.Tensor:
    """img: [C,H,W]. Xoay quanh 1 điểm chọn ngẫu nhiên: tâm 1 GT box (có jitter) hoặc tâm ảnh."""
    angle, center = _sample_rotation(img, gt_boxes, max_angle, max_center_jitter)
    return rotate(img.unsqueeze(0), angle, center=center).squeeze(0)


def _sample_rotation(img: torch.Tensor, gt_boxes: torch.Tensor, max_angle: float,
                     max_center_jitter: int):
    """Rút (góc, tâm [cx, cy]) — đúng thứ tự tiêu thụ random/torch như bản gốc."""
    h, w = img.shape[-2:]
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
    return angle, [float(cx), float(cy)]


def adaptive_random_resizing(img: torch.Tensor, gt_boxes: torch.Tensor,
                             rho: float = 0.8, s_max: float = 1.1) -> torch.Tensor:
    """img: [C,H,W]. Resize+pad với scale phụ thuộc kích thước 1 GT box ngẫu nhiên
    (box càng lớn thì scale-up càng nhiều, chặn trên bởi s_max) — giữ tỉ lệ H/W
    độc lập (khác lỗi ép-vuông của DI, xem attack/methods/diversity.py)."""
    return _apply_resize(img, _sample_resize(img.shape[-2:], gt_boxes, rho, s_max))


def _sample_resize(size, gt_boxes: torch.Tensor, rho: float, s_max: float):
    """Rút tham số resize-pad — đúng thứ tự tiêu thụ random như bản gốc."""
    ori_h, ori_w = size

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
    rem_h = max(0, int(scale_h * ori_h) - new_h)
    rem_w = max(0, int(scale_w * ori_w) - new_w)
    pad_top = random.randint(0, rem_h)
    pad_left = random.randint(0, rem_w)
    return dict(ori=(ori_h, ori_w), new=(new_h, new_w), pad=(pad_left, rem_w - pad_left, pad_top, rem_h - pad_top))


def _apply_resize(t: torch.Tensor, p) -> torch.Tensor:
    """Áp resize-pad-resize về cỡ cũ cho tensor [C,H,W] (ảnh) hoặc [N,H,W] (GT mask)."""
    rescaled = F.interpolate(t.unsqueeze(0), size=p["new"], mode="bilinear", align_corners=True)
    padded = F.pad(rescaled, p["pad"], mode="constant", value=0.0)
    return F.interpolate(padded, size=p["ori"], mode="bilinear", align_corners=True).squeeze(0)


def _resize_boxes(boxes: torch.Tensor, p) -> torch.Tensor:
    """Box [N,4] xyxy theo đúng _apply_resize (như box_transform của DI)."""
    if len(boxes) == 0:
        return boxes
    (h, w), (nh, nw), (pl, pr, pt, pb) = p["ori"], p["new"], p["pad"]
    ch, cw = nh + pt + pb, nw + pl + pr
    b = boxes.clone().float()
    b[:, [0, 2]] = (b[:, [0, 2]] * nw / w + pl) * w / cw
    b[:, [1, 3]] = (b[:, [1, 3]] * nh / h + pt) * h / ch
    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)
    return b


def gaussian_blur(img: torch.Tensor, sigma: float = 6.0) -> torch.Tensor:
    return torch.clamp(img + torch.randn_like(img) * sigma, min=0.0, max=255.0)


def rrb_views(adv_pixels: torch.Tensor, data_sample: DetDataSample,
             prob: float = 1.0, theta: float = 7.0, l_s: int = 10,
             rho: float = 0.8, s_max: float = 1.1, sigma: float = 6.0,
             use_rotate: bool = True, use_resize: bool = True, use_noise: bool = True,
             with_gt: bool = False) -> List:
    """Sinh 2 view theo đúng chain của bản gốc: view1 = rotate(adv) (nếu trúng prob),
    view2 = resize(view1) (nếu trúng prob) — rồi cả 2 cùng bị Gaussian blur.
    Hyperparameter mặc định khớp ref-repo/OSFD-main/config/base.yaml (không phải
    default __init__ của class RRB — base.yaml là config thực nghiệm đã dùng).

    Ablation OSFD (docs/plan_sota.md): use_rotate / use_resize / use_noise = False tắt
    từng phần (cờ mặc định trùng tuyệt đối bản gốc, cả thứ tự tiêu thụ RNG). Tắt resize:
    view2 = một lần xoay MỚI của adv (rút lại tham số). with_gt = True (ô task loss):
    trả về [(ảnh, data_sample đã co-transform GT box + mask)] thay vì [ảnh]."""
    # data_sample (và gt_instances bên trong) không tự động theo device của ảnh —
    # AttackDataset trả về data_sample nguyên bản từ pipeline (CPU), chỉ tensor ảnh
    # mới được caller .to(device) thủ công. Phải tự đồng bộ device ở đây.
    gt_boxes = _get_gt_boxes_xyxy(data_sample).to(adv_pixels.device)
    h, w = adv_pixels.shape[-2:]
    boxes = masks = None
    if with_gt:
        boxes, masks = get_gt(data_sample, adv_pixels.device)

    def rotate_view(img, b, m):
        angle, center = _sample_rotation(img, gt_boxes, theta, l_s)
        img = rotate(img.unsqueeze(0), angle, center=center).squeeze(0)
        if with_gt:
            b = rotate_boxes(b, angle, center, h, w)
            m = rotate(m, angle, center=center) if m is not None and len(m) else m
        return img, b, m

    view, b, m = adv_pixels, boxes, masks
    if random.random() < prob and use_rotate:
        view, b, m = rotate_view(view, b, m)
    view1, b1, m1 = view, b, m

    if random.random() < prob:
        if use_resize:
            p = _sample_resize((h, w), gt_boxes, rho, s_max)
            view = _apply_resize(view, p)
            if with_gt:
                b = _resize_boxes(b, p)
                m = _apply_resize(m, p) if m is not None and len(m) else m
        elif use_rotate:
            view, b, m = rotate_view(adv_pixels, boxes, masks)
    view2, b2, m2 = view, b, m

    noisy = (lambda v: gaussian_blur(v, sigma=sigma)) if use_noise else (lambda v: v)
    out = [noisy(view1), noisy(view2)]
    if not with_gt:
        return out
    return [(out[0], gt_with(data_sample, b1, m1)), (out[1], gt_with(data_sample, b2, m2))]
