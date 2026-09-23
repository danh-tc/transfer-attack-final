"""COCO bbox AP + paired bootstrap trên ảnh (idea.md §9-10).

Bootstrap resample ẢNH (có lặp). pycocotools tự dedup imgIds nên không resample trực
tiếp được — thay vào đó chạy COCOeval.evaluate() 1 lần trên toàn bộ ảnh để lấy kết
quả match từng (category, ảnh), rồi tái hiện COCOeval.accumulate() với TRỌNG SỐ ảnh
(w_i = số lần ảnh i xuất hiện trong mẫu bootstrap): tp/fp cộng dồn nhân trọng số của
ảnh chứa detection, số GT không-ignore nhân trọng số ảnh chứa GT. Với w = 1 kết quả
phải trùng COCOeval (kiểm tra trong `CocoImageEval.__init__`).

Chỉ tính area='all', maxDets=100 (AP, AP50 — đúng 2 số idea.md §9 dùng).
"""
import contextlib
import io
from typing import Dict, List, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REC_THRS = np.linspace(0.0, 1.00, 101)


def to_coco_dets(result, img_id: int, cat_ids: Sequence[int]) -> List[dict]:
    """DetDataSample (predict rescale=True, tọa độ ảnh gốc) -> list detection COCO."""
    b = result.pred_instances.bboxes.cpu().numpy()
    s = result.pred_instances.scores.cpu().numpy()
    l = result.pred_instances.labels.cpu().numpy()
    return [{"image_id": img_id, "category_id": cat_ids[int(li)],
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(si)}
            for (x1, y1, x2, y2), si, li in zip(b, s, l)]


class CocoImageEval:
    """Kết quả match per (category, ảnh) của 1 bộ detection, cho phép tính AP/AP50
    với trọng số ảnh bất kỳ (bootstrap) mà không chạy lại COCOeval."""

    def __init__(self, coco_gt: COCO, dets: List[dict], img_ids: Sequence[int]):
        self.img_ids = list(img_ids)
        self._order, self._cats = np.arange(len(self.img_ids)), []
        if not dets:  # model không ra detection nào trên toàn bộ ảnh -> AP = 0
            return
        ev = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
        ev.params.imgIds = self.img_ids
        ev.params.areaRng, ev.params.areaRngLbl, ev.params.maxDets = [[0, 1e10]], ["all"], [100]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
        # COCOeval sort imgIds (unique) — map lại theo thứ tự đó.
        eval_img_ids = list(ev.params.imgIds)
        pos = {img_id: i for i, img_id in enumerate(eval_img_ids)}
        self._order = np.array([pos[i] for i in self.img_ids])
        n_img = len(eval_img_ids)

        # Mỗi category: detection đã sort theo score giảm dần (mergesort như COCO),
        # ảnh chứa từng detection, tp/fp [T,D], và số GT không-ignore mỗi ảnh.
        for k in range(len(ev.params.catIds)):
            E = [ev.evalImgs[k * n_img + i] for i in range(n_img)]
            ngt = np.array([0 if e is None else int(np.count_nonzero(e["gtIgnore"] == 0)) for e in E])
            if ngt.sum() == 0:
                continue  # COCO: category không có GT -> precision -1, bị loại khỏi trung bình
            scores, img_idx, dtm, dtig = [], [], [], []
            for i, e in enumerate(E):
                if e is None or len(e["dtScores"]) == 0:
                    continue
                scores.append(np.asarray(e["dtScores"][:100]))
                img_idx.append(np.full(len(scores[-1]), i))
                dtm.append(e["dtMatches"][:, :100])
                dtig.append(e["dtIgnore"][:, :100])
            if scores:
                order = np.argsort(-np.concatenate(scores), kind="mergesort")
                dtm = np.concatenate(dtm, axis=1)[:, order]
                dtig = np.concatenate(dtig, axis=1)[:, order]
                img_idx = np.concatenate(img_idx)[order]
                tps = np.logical_and(dtm, np.logical_not(dtig)).astype(np.float64)
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtig)).astype(np.float64)
            else:
                img_idx, tps, fps = np.zeros(0, int), np.zeros((10, 0)), np.zeros((10, 0))
            self._cats.append((ngt, img_idx, tps, fps))

        # Tự kiểm: w=1 phải trùng COCOeval.accumulate (chính là số summarize() in ra).
        p = ev.eval["precision"][:, :, :, 0, 0]
        ref_ap = float(np.mean(p[p > -1])) if (p > -1).any() else 0.0
        p50 = p[0]
        ref_ap50 = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else 0.0
        ap, ap50 = self.ap(np.ones(len(self.img_ids)))
        assert abs(ap - ref_ap) < 1e-9 and abs(ap50 - ref_ap50) < 1e-9, \
            f"weighted AP lệch COCOeval: {ap} vs {ref_ap}, {ap50} vs {ref_ap50}"

    def ap(self, weights: np.ndarray):
        """weights: [n_img] theo thứ tự `img_ids` truyền vào. Trả (AP, AP50) thang 0-1."""
        w = np.zeros(len(self._order))
        w[self._order] = weights
        precs = []  # [T,R] mỗi category có GT trong mẫu
        for ngt, img_idx, tps, fps in self._cats:
            npig = float((w * ngt).sum())
            if npig == 0:
                continue
            wd = w[img_idx]
            tp = np.cumsum(tps * wd, axis=1)
            fp = np.cumsum(fps * wd, axis=1)
            q = np.zeros((tps.shape[0], len(REC_THRS)))
            if tp.shape[1] > 0:
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                for t in range(tp.shape[0]):
                    idx = np.searchsorted(rc[t], REC_THRS, side="left")
                    ok = idx < tp.shape[1]
                    q[t, ok] = pr[t, idx[ok]]
            precs.append(q)
        if not precs:
            return 0.0, 0.0
        precs = np.stack(precs)  # [K,T,R]
        return float(precs.mean()), float(precs[:, 0].mean())


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n] trọng số (số lần mỗi ảnh được chọn) — DÙNG CHUNG cho mọi điều kiện
    và mọi model để bootstrap là paired."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.float64)


def summarize(point: float, samples: np.ndarray) -> Dict[str, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)],
            "boot_mean": float(samples.mean()), "p_le_0": float(np.mean(samples <= 0))}
