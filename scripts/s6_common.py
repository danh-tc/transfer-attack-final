"""S6 — vùng con theo từng instance (progress_log KHÓA S6).
BANDS: EDGE/MID/CORE theo khoảng cách tới viền (distance transform), 3 phần ĐÚNG đều diện tích (xếp hạng, phá hòa
       bằng nhiễu cố định seed theo ann id). VERT: TOP/MIDV/BOT theo y, cùng cách. PARTS (person): HEAD/TORSO/ARMS/LEGS
       = nhóm keypoint visible gần nhất.
"""
import os

import numpy as np
from pycocotools.coco import COCO
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KP_ANN = os.path.join(REPO_ROOT, "data/coco/annotations/person_keypoints_val2017.json")
BANDS = ["EDGE", "MID", "CORE"]
VERT = ["TOP", "MIDV", "BOT"]
PARTS = ["HEAD", "TORSO", "ARMS", "LEGS"]
PART_KP = {"HEAD": [0, 1, 2, 3, 4], "TORSO": [5, 6, 11, 12], "ARMS": [7, 8, 9, 10], "LEGS": [13, 14, 15, 16]}
MIN_PIX = 30

_kp = None


def kp_coco():
    global _kp
    if _kp is None:
        _kp = COCO(KP_ANN)
    return _kp


def _thirds(m, key, seed):
    ys, xs = np.nonzero(m)
    v = key[ys, xs] + np.random.default_rng(seed).random(len(ys)) * 1e-3
    order = np.argsort(v, kind="stable")
    out = [np.zeros_like(m) for _ in range(3)]
    for k, part in enumerate(np.array_split(order, 3)):
        out[k][ys[part], xs[part]] = True
    return out


def inst_subregions(coco, ann, h, w):
    """-> dict tên vùng -> mask bool [h, w], hoặc None nếu object quá nhỏ. Có 'MASK'."""
    m = coco.annToMask(ann).astype(bool)[:h, :w]
    if m.sum() < MIN_PIX:
        return None
    R = {"MASK": m}
    d = distance_transform_edt(m)
    e, mi, co = _thirds(m, d, ann["id"])          # tăng dần khoảng cách -> EDGE trước
    R.update(EDGE=e, MID=mi, CORE=co)
    yy = np.broadcast_to(np.arange(h)[:, None], (h, w)).astype(float)
    t, mv, b = _thirds(m, yy, ann["id"] + 1)
    R.update(TOP=t, MIDV=mv, BOT=b)
    if ann["category_id"] == 1:
        kc = kp_coco()
        if ann["id"] in kc.anns:
            kp = np.array(kc.anns[ann["id"]]["keypoints"]).reshape(17, 3)
            vis = [(k, g) for g, ks in PART_KP.items() for k in ks if kp[k, 2] > 0]
            groups = sorted({g for _, g in vis})
            if len(groups) >= 2:
                tree = cKDTree(np.array([kp[k, :2] for k, _ in vis]))
                ys, xs = np.nonzero(m)
                _, nn = tree.query(np.stack([xs, ys], 1))
                lab = np.array([g for _, g in vis])[nn]
                for g in PARTS:
                    if g in groups:
                        pm = np.zeros_like(m); sel = lab == g; pm[ys[sel], xs[sel]] = True
                        R[g] = pm
    return R
