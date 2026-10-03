"""Pilot KG (quy tắc LOCKED: docs/progress_log.md 2026-10-03): thay view 2 của RRB bằng crop
vùng hợp của 1 cặp GT object; cặp chọn theo Knowledge Graph (E2) hoặc random khớp hình học (E1).

Mỗi step: view1 = rotate(adv) + nhiễu (giữ nguyên RRB, loss OSFD với feature sạch ảnh full),
view2 = T_pair(adv) + nhiễu (loss OSFD với feature sạch của CHÍNH crop đó, k·F_cln(T_pair(x))).
2 view forward, 1 backward -> B không đổi so với OSFD.

Chọn cặp (PairSampler): mỗi step rút q ~ p ∝ S_{c_i c_j} rồi rút e1 đều trong
M(q) = {r : |log(A_r/A_q)| <= log 2}; E2 dùng q, E1 dùng e1. Cả hai điều kiện LUÔN rút cả q và
e1 từ cùng 1 RNG (seed theo ảnh) -> chuỗi (q, e1) giống hệt nhau giữa E1 và E2.

Ngẫu nhiên tách luồng để E1/E2 chỉ khác ở cặp được dùng:
- chọn cặp: numpy Generator(pair_seed);
- nhiễu σ của view cặp: torch.Generator riêng (pair_seed) — crop E1/E2 khác kích thước nên
  nếu dùng RNG torch chung, lượng số ngẫu nhiên tiêu thụ khác nhau sẽ làm lệch RRB view1 các
  step sau;
- view1 (xoay + nhiễu) dùng random/torch toàn cục như RRB, seed SEED + i do caller đặt.
Ảnh 0 cặp: views = rrb_views gốc (= E0).
"""
import math
import os
import random
from functools import lru_cache
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample

from attack.losses.osfd import make_osfd_loss_fn
from attack.methods.core import run_iterative_attack
from attack.methods.rrb import gaussian_blur, random_axis_rotation, rrb_views
from attack.preprocess import resize_to_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KG_MATRIX = os.path.join(REPO_ROOT, "data/kg/coco80_vg_relation_matrix.npy")
MARGIN = 0.15
AREA_RATIO = 2.0
SCALE = (1333, 800)  # Resize của test pipeline (keep_ratio)
MODES = ("kg", "random_matched")


@lru_cache(maxsize=1)
def load_kg() -> np.ndarray:
    return np.load(KG_MATRIX)


def rescale_size(w: int, h: int, scale=SCALE):
    """= mmcv.rescale_size(keep_ratio): cạnh dài <= 1333, cạnh ngắn <= 800."""
    sf = min(max(scale) / max(h, w), min(scale) / min(h, w))
    return int(w * sf + 0.5), int(h * sf + 0.5)


class PairSampler:
    """Cặp hợp lệ = 2 GT instance khác nhau (gt_instances đã bỏ crowd), khung model."""

    def __init__(self, data_sample: DetDataSample, seed: int, S: Optional[np.ndarray] = None):
        S = load_kg() if S is None else S
        boxes = data_sample.gt_instances.bboxes
        boxes = (boxes.tensor if hasattr(boxes, "tensor") else boxes).float().cpu().numpy()
        labels = data_sample.gt_instances.labels.cpu().numpy()
        img_h, img_w = data_sample.img_shape
        n = len(boxes)
        self.I, self.J = np.triu_indices(n, 1)
        self.n_pairs = len(self.I)
        self.boxes, self.labels = boxes, labels
        self.rng = np.random.default_rng(seed)
        self.log = []
        if self.n_pairs == 0:
            return
        b1, b2 = boxes[self.I], boxes[self.J]
        x1 = np.minimum(b1[:, 0], b2[:, 0]); y1 = np.minimum(b1[:, 1], b2[:, 1])
        x2 = np.maximum(b1[:, 2], b2[:, 2]); y2 = np.maximum(b1[:, 3], b2[:, 3])
        uw, uh = x2 - x1, y2 - y1
        ex1, ex2 = np.clip(x1 - MARGIN * uw, 0, img_w), np.clip(x2 + MARGIN * uw, 0, img_w)
        ey1, ey2 = np.clip(y1 - MARGIN * uh, 0, img_h), np.clip(y2 + MARGIN * uh, 0, img_h)
        self.area = (ex2 - ex1) * (ey2 - ey1)
        # Crop nguyên pixel phủ trọn vùng đã nở (floor/ceil) -> luôn chứa trọn 2 box.
        self.crop = np.stack([np.floor(ex1), np.floor(ey1),
                              np.minimum(np.ceil(ex2), img_w), np.minimum(np.ceil(ey2), img_h)],
                             axis=1).astype(int)
        s = S[labels[self.I], labels[self.J]]
        self.kg_degenerate = bool(s.max() == 0)
        self.p = np.full(self.n_pairs, 1 / self.n_pairs) if self.kg_degenerate else s / s.sum()
        la = np.log(self.area)
        self.match = [np.flatnonzero(np.abs(la - la[q]) <= math.log(AREA_RATIO))
                      for q in range(self.n_pairs)]
        self.same_class = labels[self.I] == labels[self.J]

    def draw(self):
        """-> (q, e1). Luôn rút cả hai (chuỗi giống nhau giữa E1 và E2)."""
        q = int(self.rng.choice(self.n_pairs, p=self.p))
        e1 = int(self.rng.choice(self.match[q]))
        self.log.append({
            "q": q, "e1": e1,
            "abs_log_area_ratio": float(abs(math.log(self.area[e1] / self.area[q]))),
            "same_class_q": bool(self.same_class[q]), "same_class_e1": bool(self.same_class[e1]),
        })
        return q, e1


def crop_resize(img: torch.Tensor, crop) -> torch.Tensor:
    """T_pair không nhiễu: crop [C,H,W] theo (x1,y1,x2,y2) rồi resize bilinear khả vi theo
    luật Resize keep_ratio của test pipeline (không chặn hệ số phóng)."""
    x1, y1, x2, y2 = (int(v) for v in crop)
    patch = img[:, y1:y2, x1:x2]
    new_w, new_h = rescale_size(x2 - x1, y2 - y1)
    return F.interpolate(patch.unsqueeze(0), size=(new_h, new_w), mode="bilinear",
                         align_corners=False).squeeze(0)


def kg_osfd_attack(model, orig_pixels, data_sample: DetDataSample, steps: int, mode: str,
                   pair_seed: int, epsilon: float = 5.0, alpha: float = 1.0, k: float = 3.0,
                   momentum: float = 1.0, rrb_theta: float = 7.0, rrb_l_s: int = 10,
                   rrb_rho: float = 0.8, rrb_s_max: float = 1.1, rrb_sigma: float = 6.0,
                   S: Optional[np.ndarray] = None, info: Optional[dict] = None, on_step=None):
    """mode: 'kg' (E2) | 'random_matched' (E1). info (dict, tùy chọn) nhận sampler + số
    forward no_grad (feature sạch theo cặp, cache)."""
    assert mode in MODES, mode
    clean_pixels = resize_to_model(orig_pixels, tuple(data_sample.img_shape))
    base_loss = make_osfd_loss_fn(model, clean_pixels, data_sample, k=k)
    sampler = PairSampler(data_sample, pair_seed, S)
    gen = torch.Generator(device=orig_pixels.device)
    gen.manual_seed(pair_seed)
    pair_loss = {}

    def loss_for(idx):
        if idx not in pair_loss:
            pair_loss[idx] = make_osfd_loss_fn(
                model, crop_resize(clean_pixels, sampler.crop[idx]), data_sample, k=k)
        return pair_loss[idx]

    def views_fn(img, ds, step_idx, total_steps):
        if sampler.n_pairs == 0:
            return [(v, ds) for v in rrb_views(img, ds, theta=rrb_theta, l_s=rrb_l_s,
                                               rho=rrb_rho, s_max=rrb_s_max, sigma=rrb_sigma)]
        gt_boxes = torch.from_numpy(sampler.boxes).to(img.device)
        # Cùng thứ tự tiêu thụ random/torch như rrb_views (prob = 1 nhưng vẫn rút) -> step 0
        # view1 trùng hệt view1 của E0 khi cùng seed ảnh.
        view1 = img
        if random.random() < 1.0:
            view1 = random_axis_rotation(img, gt_boxes, max_angle=rrb_theta,
                                         max_center_jitter=rrb_l_s)
        view1 = gaussian_blur(view1, sigma=rrb_sigma)
        q, e1 = sampler.draw()
        idx = q if mode == "kg" else e1
        view2 = crop_resize(img, sampler.crop[idx])
        view2 = torch.clamp(view2 + torch.randn(view2.shape, generator=gen, device=view2.device)
                            * rrb_sigma, 0.0, 255.0)
        return [(view1, ds), (view2, ds, loss_for(idx))]

    noise = run_iterative_attack(model, orig_pixels, data_sample, loss_fn=base_loss, steps=steps,
                                 epsilon=epsilon, alpha=alpha, momentum=momentum,
                                 views_fn=views_fn, on_step=on_step)
    if info is not None:
        info["sampler"] = sampler
        info["forward_no_grad"] = 1 + len(pair_loss)  # feature sạch ảnh full + mỗi cặp đã dùng
    return noise
