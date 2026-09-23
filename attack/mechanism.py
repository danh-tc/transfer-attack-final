"""Công cụ dùng chung cho Mechanism Stage (docs/mechanism_plan.md).

- linear_cka: tương đồng biểu diễn giữa 2 feature map khác số kênh (A2a).
- per_image_transfer: chỉ số transfer theo từng ảnh — suppression rate + retained
  confidence (định nghĩa khóa ở mechanism_plan.md, mục "Chỉ số transfer theo từng ảnh").
- boot_mean / boot_spearman: paired bootstrap trên ảnh (cùng chỉ số resample cho mọi
  đại lượng trong 1 phép so sánh).
"""
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.stats import spearmanr

DETECT_SCORE = 0.3
MATCH_IOU = 0.5


def linear_cka(fx: torch.Tensor, fy: torch.Tensor) -> float:
    """fx [1,C1,H,W], fy [1,C2,H,W] cùng lưới không gian; mẫu = vị trí (H*W).
    Linear CKA (Kornblith et al. 2019) dạng feature-space, tính float64."""
    x = fx.flatten(2)[0].T.double()  # [N, C1]
    y = fy.flatten(2)[0].T.double()
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    hsic = (y.T @ x).pow(2).sum()
    return float(hsic / ((x.T @ x).norm() * (y.T @ y).norm()))


def _iou_xywh(gt: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """gt [4] xywh, dets [D,4] xywh -> IoU [D]."""
    gx2, gy2 = gt[0] + gt[2], gt[1] + gt[3]
    dx2, dy2 = dets[:, 0] + dets[:, 2], dets[:, 1] + dets[:, 3]
    iw = np.clip(np.minimum(gx2, dx2) - np.maximum(gt[0], dets[:, 0]), 0, None)
    ih = np.clip(np.minimum(gy2, dy2) - np.maximum(gt[1], dets[:, 1]), 0, None)
    inter = iw * ih
    return inter / (gt[2] * gt[3] + dets[:, 2] * dets[:, 3] - inter + 1e-9)


def _matched_scores(coco: COCO, dets: List[dict], img_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    """Mỗi ảnh: score cao nhất của detection đúng lớp có IoU >= 0.5 với từng GT
    (không-crowd), 0 nếu không có. Tọa độ ảnh gốc (dets là predict rescale=True)."""
    by_img = defaultdict(list)
    for d in dets:
        by_img[d["image_id"]].append(d)
    out = {}
    for img_id in img_ids:
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))]
        ds = by_img.get(img_id, [])
        scores = np.zeros(len(anns))
        if ds:
            boxes = np.array([d["bbox"] for d in ds])
            cats = np.array([d["category_id"] for d in ds])
            sc = np.array([d["score"] for d in ds])
            for j, a in enumerate(anns):
                ok = (cats == a["category_id"]) & (_iou_xywh(np.array(a["bbox"]), boxes) >= MATCH_IOU)
                if ok.any():
                    scores[j] = sc[ok].max()
        out[img_id] = scores
    return out


def per_image_transfer(coco: COCO, clean_dets: List[dict], adv_dets: List[dict],
                       img_ids: Sequence[int]) -> Dict[str, np.ndarray]:
    """Trên các GT mà model detect được ở ảnh sạch (score >= 0.3):
    suppression = tỉ lệ GT không còn detect (score_adv < 0.3) trên ảnh adv;
    retained = trung bình score_adv / score_clean. NaN nếu ảnh không có GT nào
    được detect ở ảnh sạch."""
    s_clean = _matched_scores(coco, clean_dets, img_ids)
    s_adv = _matched_scores(coco, adv_dets, img_ids)
    sup, ret = [], []
    for img_id in img_ids:
        keep = s_clean[img_id] >= DETECT_SCORE
        if not keep.any():
            sup.append(np.nan); ret.append(np.nan); continue
        sc, sa = s_clean[img_id][keep], s_adv[img_id][keep]
        sup.append(float(np.mean(sa < DETECT_SCORE)))
        ret.append(float(np.mean(sa / sc)))
    return {"suppression": np.array(sup), "retained": np.array(ret)}


def boot_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n] chỉ số ảnh resample — dùng chung cho mọi đại lượng (paired)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ci(point: float, samples: np.ndarray) -> Dict[str, float]:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)],
            "excludes_0": bool(lo > 0 or hi < 0)}


def boot_mean(values: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    """values [n] (có thể NaN) -> mean + CI 95% theo cùng chỉ số resample idx."""
    return ci(np.nanmean(values), np.array([np.nanmean(values[r]) for r in idx]))


def boot_spearman(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    def rho(ii):
        x, y = a[ii], b[ii]
        ok = ~(np.isnan(x) | np.isnan(y))
        return spearmanr(x[ok], y[ok])[0] if ok.sum() > 2 else np.nan
    out = ci(rho(np.arange(len(a))), np.array([rho(r) for r in idx]))
    out["n"] = int((~(np.isnan(a) | np.isnan(b))).sum())
    return out
