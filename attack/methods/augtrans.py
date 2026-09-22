"""AugTrans — port từ papers/AugTrans.pdf (Pandey et al., "AugTrans: Boosting
Adversarial Transferability in Object Detection with a Dynamic, Object-Aware
Augmentation Pipeline"). KHÔNG có ref code trong repo — port trực tiếp từ
Algorithm 1, Algorithm 2, Eq (2)-(6) và bảng hyperparameter trong paper.

4 thành phần biến đổi (Section 3.3, Fig. 1), áp dụng TUẦN TỰ cho mỗi EOT
sample: (1) dynamic object-centric rotation (curriculum-scheduled góc quay,
tâm quay chọn ngẫu nhiên giữa tâm ảnh/điểm ngẫu nhiên/tâm 1 GT box), (2)
multi-box aware resizing (scale factor tỉ lệ với kích thước K_obj GT box lớn
nhất), (3) contextual crop/reflective-pad (đưa ảnh về đúng kích thước gốc),
(4) composite noise injection (Gaussian + salt-and-pepper).

Lưu ý về 1 câu mơ hồ trong paper: Section 3.3 + Algorithm 2 + Fig. 1 đều mô
tả pipeline là tổ hợp CỐ ĐỊNH cả 4 bước theo đúng thứ tự cho mỗi EOT sample
("a fresh transformation sequence comprising four complementary components
is assigned to each EOT sample"). Nhưng phần "Hyperparameter Configuration"
(cuối Section 3.5.2) lại viết "we apply 2 sequential transformations per EOT
sample, randomly sampling from the available transformation types" — mâu
thuẫn với phần chính. Ở đây chọn theo mô tả chính + Algorithm 2 (áp dụng đủ
cả 4 bước) vì đây là mô tả nhất quán và có pseudocode cụ thể đi kèm; câu ở
phần hyperparameter không có thuật toán chi tiết nào khác để theo.

Đơn vị: pixel-space [0,255] (khớp quy ước attack/methods/core.py) — paper
gốc dùng [0,1], mọi hằng số pixel-scale (epsilon, step size η, sigma noise)
đã nhân 255 khi đưa vào đây.
"""
import copy
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample
from torchvision.transforms.functional import rotate

from attack.methods.box_transforms import rotate_boxes
from attack.methods.core import run_iterative_attack
from attack.preprocess import to_batch

# --- Hyperparameter mặc định, khớp Section 4.1 "Hyperparameters" + Algorithm 1 của paper ---
N_EOT_DEFAULT = 10
EPSILON_PRIMARY = 5.0  # 5/255 -> pixel-space [0,255], khớp docs/protocol_lock.md
ALPHA_MARGIN = 2.0  # xem docstring augtrans_attack() — vì sao KHÔNG dùng alpha=0.0004 nguyên gốc
THETA_BASE_DEFAULT = 8.0  # độ, Algorithm 1
CURRICULUM_C_DEFAULT = 0.5  # Algorithm 1
K_OBJ_DEFAULT = 3  # Section 3.3.3
RHO_RANGE_DEFAULT = (-0.2, 0.5)  # Eq (3)
ZETA_AR_DEFAULT = 0.1  # Eq (4)
NOISE_SIGMA_DEFAULT = 0.02 * 255  # Eq (5), scale [0,1] -> [0,255]
NOISE_P_SP_DEFAULT = 0.01  # Eq (5)
ALPHA_CLS, ALPHA_BOX, ALPHA_OBJ, ALPHA_RPN_BOX = 1.0, 1.0, 2.0, 1.0  # Eq (6)
ALPHA_MASK = 1.0  # KHÔNG có trong Eq (6) gốc — xem docstring augtrans_loss()
GAMMA_CLS, GAMMA_OBJ = 0.8, 0.8  # Eq (6), "gradient regularization"


def _get_gt_boxes_xyxy(data_sample: DetDataSample, device) -> torch.Tensor:
    boxes = data_sample.gt_instances.bboxes
    boxes = boxes.tensor if hasattr(boxes, "tensor") else boxes
    return boxes.to(device)


# --- (1) Dynamic object-centric rotation — Section 3.3.2, Eq (2) ---

def _curriculum_theta_max(step_idx: int, total_steps: int,
                          theta_base: float, c: float) -> float:
    """Eq (2): theta_max(k) = theta_base * (1 + c * k/K_max)."""
    progress = step_idx / max(1, total_steps)
    return theta_base * (1.0 + c * progress)


def _sample_rotation_center(h: int, w: int, gt_boxes: torch.Tensor) -> Tuple[float, float]:
    """C_rot ~ {C_img, C_rand, C_obj} — 3 lựa chọn đồng xác suất (paper không
    ghi trọng số khác nhau nên dùng đều)."""
    choice = random.randrange(3)
    if choice == 0:  # C_img
        return w / 2.0, h / 2.0
    if choice == 1:  # C_rand
        return random.uniform(0, w), random.uniform(0, h)
    # C_obj — tâm 1 GT box ngẫu nhiên, fallback về tâm ảnh nếu không có GT
    if len(gt_boxes) == 0:
        return w / 2.0, h / 2.0
    box = gt_boxes[random.randrange(len(gt_boxes))]
    return float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)


def _object_centric_rotate(img: torch.Tensor, gt_boxes: torch.Tensor,
                           step_idx: int, total_steps: int,
                           theta_base: float, curriculum_c: float
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trả về (ảnh đã xoay, gt_boxes đã xoay theo — xem attack/methods/box_transforms.py)."""
    h, w = img.shape[-2:]
    theta_max = _curriculum_theta_max(step_idx, total_steps, theta_base, curriculum_c)
    angle = random.uniform(-theta_max, theta_max)
    cx, cy = _sample_rotation_center(h, w, gt_boxes)
    rotated_img = rotate(img.unsqueeze(0), angle, center=[cx, cy]).squeeze(0)
    rotated_boxes = rotate_boxes(gt_boxes, angle, (cx, cy), h, w)
    return rotated_img, rotated_boxes


# --- (2)+(3) Multi-box aware resizing + contextual crop/reflective-pad — Section 3.3.3/3.3.4 ---

def _content_adaptive_scale(gt_boxes: torch.Tensor, img_h: int, img_w: int,
                            k_obj: int, rho_range: Tuple[float, float],
                            zeta_ar: float) -> Tuple[float, float]:
    """Eq (3)+(4): scale factor tỉ lệ với kích thước trung bình của K_obj GT
    box lớn nhất (theo diện tích), cộng jitter tỉ lệ khung hình độc lập."""
    if len(gt_boxes) == 0:
        r_h = r_w = 0.0
    else:
        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        top_idx = torch.argsort(areas, descending=True)[:k_obj]
        top_boxes = gt_boxes[top_idx]
        mean_h = float((top_boxes[:, 3] - top_boxes[:, 1]).mean())
        mean_w = float((top_boxes[:, 2] - top_boxes[:, 0]).mean())
        r_h = mean_h / img_h
        r_w = mean_w / img_w

    rho_h = random.uniform(*rho_range)
    rho_w = random.uniform(*rho_range)
    s_h_base = 1.0 + r_h * rho_h
    s_w_base = 1.0 + r_w * rho_w

    zeta_h = random.uniform(-zeta_ar, zeta_ar)
    zeta_w = random.uniform(-zeta_ar, zeta_ar)
    s_h = max(0.1, s_h_base * (1.0 + zeta_h))
    s_w = max(0.1, s_w_base * (1.0 + zeta_w))
    return s_h, s_w


def _resize_crop_pad(img: torch.Tensor, gt_boxes: torch.Tensor,
                     s_h: float, s_w: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize theo (s_h,s_w) rồi đưa về đúng kích thước gốc: crop ngẫu nhiên
    nếu phóng to, reflection-pad với offset ngẫu nhiên nếu thu nhỏ — xử lý
    H/W độc lập (Section 3.3.4). Trả về (ảnh, gt_boxes) đã cùng biến đổi:
    box *= scale rồi dịch theo đúng offset crop/pad đã dùng cho ảnh."""
    c, h, w = img.shape
    new_h = max(1, round(h * s_h))
    new_w = max(1, round(w * s_w))
    x = F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                      mode="bilinear", align_corners=True).squeeze(0)

    boxes = gt_boxes.clone().float()
    if len(boxes) > 0:
        boxes[:, 0] *= new_w / w; boxes[:, 2] *= new_w / w
        boxes[:, 1] *= new_h / h; boxes[:, 3] *= new_h / h

    if new_h > h:
        top = random.randint(0, new_h - h)
        x = x[:, top:top + h, :]
        if len(boxes) > 0:
            boxes[:, 1] -= top; boxes[:, 3] -= top
    elif new_h < h:
        pad_total = h - new_h
        pad_top = random.randint(0, pad_total)
        x = F.pad(x.unsqueeze(0), (0, 0, pad_top, pad_total - pad_top),
                 mode="reflect").squeeze(0)
        if len(boxes) > 0:
            boxes[:, 1] += pad_top; boxes[:, 3] += pad_top

    if new_w > w:
        left = random.randint(0, new_w - w)
        x = x[:, :, left:left + w]
        if len(boxes) > 0:
            boxes[:, 0] -= left; boxes[:, 2] -= left
    elif new_w < w:
        pad_total = w - new_w
        pad_left = random.randint(0, pad_total)
        x = F.pad(x.unsqueeze(0), (pad_left, pad_total - pad_left, 0, 0),
                 mode="reflect").squeeze(0)
        if len(boxes) > 0:
            boxes[:, 0] += pad_left; boxes[:, 2] += pad_left

    if len(boxes) > 0:
        boxes[:, 0] = boxes[:, 0].clamp(0, w); boxes[:, 2] = boxes[:, 2].clamp(0, w)
        boxes[:, 1] = boxes[:, 1].clamp(0, h); boxes[:, 3] = boxes[:, 3].clamp(0, h)

    return x, boxes


# --- (4) Composite noise injection — Eq (5) ---

def _composite_noise(img: torch.Tensor, sigma: float, p_sp: float) -> torch.Tensor:
    """Eq (5): omega = omega_G + omega_SPN. Salt/pepper roll theo VỊ TRÍ
    KHÔNG gian (1 kênh, broadcast qua C) — khớp định nghĩa salt-and-pepper
    chuẩn (đốm trắng/đen xuất hiện cùng lúc ở mọi channel tại 1 pixel, không
    phải nhiễu độc lập từng channel trông như static màu)."""
    gaussian = torch.randn_like(img) * sigma
    sp_roll = torch.rand(1, *img.shape[-2:], device=img.device, dtype=img.dtype)
    salt = sp_roll < p_sp / 2
    pepper = (sp_roll >= p_sp / 2) & (sp_roll < p_sp)
    out = img + gaussian
    out = torch.where(salt, torch.full_like(img, 255.0), out)
    out = torch.where(pepper, torch.zeros_like(img), out)
    return torch.clamp(out, min=0.0, max=255.0)


# --- Pipeline đầy đủ cho 1 EOT sample ---

def augtrans_transform(img: torch.Tensor, data_sample: DetDataSample,
                       step_idx: int, total_steps: int,
                       theta_base: float = THETA_BASE_DEFAULT,
                       curriculum_c: float = CURRICULUM_C_DEFAULT,
                       k_obj: int = K_OBJ_DEFAULT,
                       rho_range: Tuple[float, float] = RHO_RANGE_DEFAULT,
                       zeta_ar: float = ZETA_AR_DEFAULT,
                       noise_sigma: float = NOISE_SIGMA_DEFAULT,
                       noise_p_sp: float = NOISE_P_SP_DEFAULT
                       ) -> Tuple[torch.Tensor, DetDataSample]:
    """Trả về (ảnh đã biến đổi, data_sample MỚI có gt_instances.bboxes đã
    co-transform khớp đúng ảnh) — bắt buộc vì augtrans_loss() dùng GT
    (xem cảnh báo trong box_transforms.py: box không co-transform => loss
    tính sai vị trí object, đây từng là bug thật khiến AugTrans vô hiệu)."""
    gt_boxes = _get_gt_boxes_xyxy(data_sample, img.device)
    x, gt_boxes = _object_centric_rotate(img, gt_boxes, step_idx, total_steps, theta_base, curriculum_c)
    s_h, s_w = _content_adaptive_scale(gt_boxes, img.shape[-2], img.shape[-1], k_obj, rho_range, zeta_ar)
    x, gt_boxes = _resize_crop_pad(x, gt_boxes, s_h, s_w)
    x = _composite_noise(x, noise_sigma, noise_p_sp)

    new_data_sample = copy.deepcopy(data_sample)
    new_data_sample.gt_instances.bboxes = gt_boxes
    return x, new_data_sample


def augtrans_views_fn(img: torch.Tensor, data_sample: DetDataSample,
                      step_idx: int, total_steps: int, n_eot: int, **kwargs
                      ) -> List[Tuple[torch.Tensor, DetDataSample]]:
    return [augtrans_transform(img, data_sample, step_idx, total_steps, **kwargs)
           for _ in range(n_eot)]


# --- Loss đa thành phần — Eq (6) ---

def _sum_maybe_list(value):
    if isinstance(value, (list, tuple)):
        total = None
        for v in value:
            total = v if total is None else total + v
        return total
    return value


def augtrans_loss(model, pixel_tensor: torch.Tensor, data_sample: DetDataSample,
                  alpha_cls: float = ALPHA_CLS, alpha_box: float = ALPHA_BOX,
                  alpha_obj: float = ALPHA_OBJ, alpha_rpn_box: float = ALPHA_RPN_BOX,
                  alpha_mask: float = ALPHA_MASK,
                  gamma_cls: float = GAMMA_CLS, gamma_obj: float = GAMMA_OBJ) -> torch.Tensor:
    """Eq (6) gốc: L = a_cls*(L_cls)^g_cls + a_box*L_box + a_obj*(L_obj)^g_obj + a_rpn_box*L_rpn_box.

    Map tên loss key thật của mmdet v3 TwoStageDetector.loss() (verify bằng
    cách in losses.keys() thật trên GPU, xem docs/progress_log.md): L_cls ->
    'loss_cls', L_box_reg -> 'loss_bbox' (đầu ROI head, final), L_obj ->
    'loss_rpn_cls' (objectness nhị phân của RPN, list theo từng FPN level,
    cộng lại trước khi lấy mũ), L_rpn_box_reg -> 'loss_rpn_bbox' (cũng list
    theo level).

    **KHÁC nguyên bản paper: có cộng thêm alpha_mask*L_mask (a_mask=1.0 mặc
    định, không có exponent — cùng kiểu linear như L_box/L_rpn_box).** Paper
    AugTrans dùng Faster R-CNN làm source (Section 3.5.1: "For two-stage
    detectors like Faster R-CNN") — detector không có mask head, nên Eq (6)
    4 số hạng đã là TOÀN BỘ task loss của họ. Surrogate của dự án này bị khoá
    là Mask R-CNN (protocol_lock.md, để đồng nhất kiến trúc cả Controlled
    Panel) — có thêm loss_mask, và trên ảnh mẫu verify thật chiếm tới ~62%
    tổng loss (1.68 so với loss_cls=0.52+loss_bbox=0.43+loss_rpn~0.03). Dùng
    nguyên Eq (6) 4 số hạng khiến attack bỏ qua phần lớn nhất tín hiệu tấn
    công khả dụng của surrogate thật — verify bằng thực nghiệm cô lập
    (attack vẫn yếu dù sửa hết bug pipeline/EOT, dù tăng K_max lên tới 200
    khớp scale paper) cho thấy đây là nguyên nhân chính, không phải bug code.
    Quyết định (đã hỏi ý kiến): bổ sung loss_mask để attack tận dụng đúng
    toàn bộ task loss của surrogate THẬT đang dùng, giữ tinh thần "tấn công
    toàn bộ multi-task loss" của paper thay vì áp cứng công thức tính cho
    kiến trúc khác. Xem docs/progress_log.md để biết đầy đủ quá trình debug.
    """
    batch_inputs, batch_data_samples = to_batch(model, [pixel_tensor], [data_sample])
    losses = model.loss(batch_inputs, batch_data_samples)

    l_cls = _sum_maybe_list(losses["loss_cls"])
    l_box = _sum_maybe_list(losses["loss_bbox"])
    l_obj = _sum_maybe_list(losses["loss_rpn_cls"])
    l_rpn_box = _sum_maybe_list(losses["loss_rpn_bbox"])
    l_mask = _sum_maybe_list(losses["loss_mask"]) if "loss_mask" in losses else None

    eps = 1e-6  # tránh pow(0, gamma<1) = grad vô hạn tại đúng 0
    l_cls_scaled = torch.clamp(l_cls, min=eps) ** gamma_cls
    l_obj_scaled = torch.clamp(l_obj, min=eps) ** gamma_obj

    total = (alpha_cls * l_cls_scaled + alpha_box * l_box
            + alpha_obj * l_obj_scaled + alpha_rpn_box * l_rpn_box)
    if l_mask is not None:
        total = total + alpha_mask * l_mask
    return total


# --- Entry point khớp interface các baseline khác (attack/methods/baselines.py) ---

def augtrans_attack(model, clean_pixels: torch.Tensor, data_sample: DetDataSample,
                    budget_B: int, epsilon: float = EPSILON_PRIMARY,
                    alpha: float = None, n_eot: int = N_EOT_DEFAULT,
                    theta_base: float = THETA_BASE_DEFAULT,
                    curriculum_c: float = CURRICULUM_C_DEFAULT) -> torch.Tensor:
    """budget_B: gradient-evaluation budget theo docs/protocol_lock.md.

    KHÁC 3 baseline kia: protocol_lock.md định nghĩa riêng cho AugTrans
    B = K_max * N_EOT (mỗi EOT sample tính là 1 đơn vị B, không giống RRB
    coi cả batch-doubling là 1 B) — nên K_max = B / N_EOT ở đây, KHÔNG
    truyền budget_B thẳng làm số bước lặp như mi_fgsm_attack/di_fgsm_attack.
    B=10 bị loại khỏi so sánh (xem protocol_lock.md) vì K_max=1 không đủ cho
    curriculum hoạt động có ý nghĩa.

    alpha=None (mặc định) -> tự tính = ALPHA_MARGIN * epsilon / K_max, KHÔNG
    dùng thẳng eta=0.0004 nguyên gốc của paper. Lý do: paper chạy K_max=160,
    còn ở protocol B∈{50,200}/N_EOT=10 chỉ cho K_max∈{5,20} — với alpha gốc,
    quãng đường tối đa noise đi được (K_max*alpha) chỉ ~0.51-2.04, KHÔNG BAO
    GIỜ chạm tới epsilon=5 (verify thật: xem docs/progress_log.md, smoke
    test cho L_inf=0.51 dù epsilon=5.0) — trong khi MI-FGSM/DI-FGSM/OSFD
    dùng alpha=1.0 luôn no đủ epsilon-ball trong vài step đầu, dư thừa budget
    còn lại để tinh chỉnh. Dùng nguyên alpha gốc sẽ khiến AugTrans thua thiệt
    một cách GIẢ TẠO (do quy đổi đơn vị B, không phải do method yếu), vi phạm
    nguyên tắc fair comparison của protocol_lock.md. Quyết định: scale alpha
    theo K_max thực tế để AugTrans cũng tận dụng hết budget perturbation như
    3 baseline kia — margin=2x (K_max*alpha=2*epsilon) để có dư địa cho
    trường hợp dấu gradient không nhất quán ở vài step đầu (giống cách
    MI/DI/OSFD có margin dư dả tự nhiên nhờ steps=B lớn hơn K_max nhiều lần).
    """
    k_max = max(1, round(budget_B / n_eot))
    if alpha is None:
        alpha = ALPHA_MARGIN * epsilon / k_max
    views_fn = lambda img, ds, k, kmax: augtrans_views_fn(
        img, ds, k, kmax, n_eot, theta_base=theta_base, curriculum_c=curriculum_c)
    return run_iterative_attack(
        model, clean_pixels, data_sample, loss_fn=augtrans_loss,
        steps=k_max, epsilon=epsilon, alpha=alpha, views_fn=views_fn)
