"""Biến đổi GT box ĐỒNG BỘ với biến đổi ảnh (rotation, resize+crop/pad).

Bắt buộc phải có khi loss tấn công phụ thuộc GT (model.loss() cần box khớp
đúng vị trí object trong ảnh ĐÃ biến đổi — nếu vẫn dùng box gốc trước biến
đổi trong khi ảnh đã xoay/resize, loss tính hoàn toàn sai vị trí, gradient
gần như vô nghĩa). Bug này từng làm AugTrans không suppress được GT dù
noise đã full epsilon-ball — xem docs/progress_log.md. Áp dụng cho CẢ box
lẫn mask (get_gt/with_gt bên dưới).
"""
import copy
import math
from typing import Optional, Tuple

import torch
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks


def get_gt(data_sample: DetDataSample, device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Trả về (boxes [N,4] xyxy float, masks [N,H,W] float 0/1 hoặc None).

    Mask cũng PHẢI co-transform giống box: Mask R-CNN tính loss_mask (~62% tổng
    loss trên ảnh mẫu) từ gt_masks — nếu ảnh xoay/resize mà mask giữ nguyên thì
    target mask lệch khỏi object thật. Mask trả về dạng tensor để caller áp
    CHÍNH phép biến đổi tensor đã dùng cho ảnh (rotate/interpolate/crop/pad),
    đảm bảo khớp tuyệt đối thay vì viết lại công thức riêng như với box.
    """
    gt = data_sample.gt_instances
    boxes = gt.bboxes.tensor if hasattr(gt.bboxes, "tensor") else gt.bboxes
    boxes = boxes.to(device).float()
    masks = None
    if "masks" in gt:
        masks = gt.masks.to_tensor(dtype=torch.float32, device=device)
    return boxes, masks


def with_gt(data_sample: DetDataSample, boxes: torch.Tensor,
            masks: Optional[torch.Tensor]) -> DetDataSample:
    """data_sample MỚI (deepcopy) mang box/mask đã biến đổi. Mask threshold 0.5
    (sau interpolate bilinear) rồi đóng gói lại BitmapMasks — kiểu mask head
    của mmdet cần cho crop_and_resize khi tính mask target."""
    ds = copy.deepcopy(data_sample)
    ds.gt_instances.bboxes = boxes
    if masks is not None:
        h, w = masks.shape[-2:]
        ds.gt_instances.masks = BitmapMasks(
            (masks > 0.5).to(torch.uint8).cpu().numpy(), h, w)
    return ds


def rotate_boxes(boxes: torch.Tensor, angle_deg: float, center, img_h: int, img_w: int) -> torch.Tensor:
    """boxes: [N,4] xyxy. Xoay 4 góc mỗi box quanh `center`, lấy AABB (axis-
    aligned bounding box) mới của tứ giác đã xoay, rồi clip vào biên ảnh.

    Quy ước góc khớp đúng torchvision.transforms.functional.rotate — verify
    bằng thực nghiệm (đặt 1 điểm đánh dấu, xoay, đo lại vị trí), không suy
    từ tài liệu vì dễ nhầm chiều dương/âm giữa hệ toạ độ toán học (y lên) và
    hệ toạ độ ảnh (y xuống): điểm lệch (dx,dy) so với center chuyển thành
    (dx*cos(theta) + dy*sin(theta), -dx*sin(theta) + dy*cos(theta)).
    """
    if len(boxes) == 0:
        return boxes
    cx, cy = center
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners_x = torch.stack([x1, x2, x2, x1], dim=1)  # [N,4] — 4 góc mỗi box
    corners_y = torch.stack([y1, y1, y2, y2], dim=1)
    dx = corners_x - cx
    dy = corners_y - cy
    new_x = cx + dx * cos_t + dy * sin_t
    new_y = cy - dx * sin_t + dy * cos_t

    x1n = new_x.min(dim=1).values.clamp(0, img_w)
    x2n = new_x.max(dim=1).values.clamp(0, img_w)
    y1n = new_y.min(dim=1).values.clamp(0, img_h)
    y2n = new_y.max(dim=1).values.clamp(0, img_h)
    return torch.stack([x1n, y1n, x2n, y2n], dim=1)


def scale_shift_boxes(boxes: torch.Tensor, s_w: float, s_h: float,
                      shift_x: float, shift_y: float,
                      img_w: int, img_h: int) -> torch.Tensor:
    """boxes: [N,4] xyxy ở hệ toạ độ ảnh TRƯỚC resize. Trả về box ở hệ toạ độ
    ảnh SAU khi resize theo (s_w,s_h) rồi dịch (shift_x,shift_y).

    Quy ước shift: shift=top/left offset khi CROP (ảnh phóng to rồi cắt bớt,
    shift>0 nghĩa là bỏ đi `shift` pixel phía trên-trái); shift=-pad khi PAD
    (ảnh thu nhỏ rồi đệm thêm, shift<0 nghĩa là thêm pixel phía trên-trái).
    Công thức: coord_final = coord_gốc*scale - shift (khớp cả 2 trường hợp,
    xem attack/methods/augtrans.py._resize_crop_pad để thấy shift được tính
    thế nào khi biến đổi ảnh thật)."""
    if len(boxes) == 0:
        return boxes
    x1 = (boxes[:, 0] * s_w - shift_x).clamp(0, img_w)
    x2 = (boxes[:, 2] * s_w - shift_x).clamp(0, img_w)
    y1 = (boxes[:, 1] * s_h - shift_y).clamp(0, img_h)
    y2 = (boxes[:, 3] * s_h - shift_y).clamp(0, img_h)
    return torch.stack([x1, y1, x2, y2], dim=1)
