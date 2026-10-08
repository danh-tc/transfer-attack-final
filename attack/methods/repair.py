"""Hướng Repair-Robust Attack (quy tắc khóa: docs/progress_log.md entry 2026-10-04).

Giả thuyết: encoder DINO "sửa" distortion vừa phải ở stage sâu nhờ các mức nông (stride 4/8)
gần như còn sạch. Mô phỏng phép sửa đó NGAY TRÊN surrogate R50 và tối ưu δ để tín hiệu phân
loại sống sót qua phép sửa.

- Probe V: C2 với thành phần phân loại thay bằng L_V ("chỉ đẩy GT về nền"):
    L_V = −[ mean_{a∈P_rpn} BCE(o_a, 0) + mean_{r∈P_roi} CE(z_r, nền) ]
  P = MỌI anchor / RoI dương do assigner train_cfg của surrogate gán (không lấy mẫu, không có
  số hạng đẩy mẫu âm thành foreground). RoI: proposal của RPN (train_cfg.rpn_proposal) + GT làm
  proposal (rcnn sampler add_gt_as_proposals=True).
- R-a: sau backbone, thay C2, C3 bằng feature SẠCH của cùng view (cùng phép xoay/resize RRB,
  không nhiễu, no_grad) rồi mới qua FPN -> RPN/ROI. L_OSFD vẫn tính trên backbone adv, KHÔNG
  qua phép sửa. Mỗi step 1 lần Bernoulli(p) (numpy RNG riêng seed SEED + i).

L = L_OSFD + λ·L_cls(head(FPN(R(C_adv)))), L_cls ∈ {C2: loss_rpn_cls + loss_cls, V: L_V}.
Với p = 0 và cls="c2" trùng tuyệt đối loss của C2 (`preprocess.make_feature_plus_gt_loss`).
"""
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures.bbox import bbox2roi
from mmengine.structures import InstanceData

from attack.methods.core import run_iterative_attack
from attack.methods.rrb import rrb_views
from attack.preprocess import extract_features, resize_to_model, sum_loss_dict, to_batch

C2_TERMS = ("loss_rpn_cls", "loss_cls")


def _boxes(b):
    return (b.tensor if hasattr(b, "tensor") else b).float()


def vanish_loss(model, x, batch_data_samples, return_counts: bool = False):
    """L_V trên feature FPN x (tuple các mức) của 1 ảnh. Trả về tensor vô hướng (≤ 0)."""
    gts, gts_ignore, metas = unpack_gt_instances(batch_data_samples)
    gt, gt_ignore, meta = gts[0], gts_ignore[0], metas[0]
    gt_boxes = _boxes(gt.bboxes)
    device = x[0].device
    zero = x[0].new_zeros(())
    n_rpn = n_roi = 0
    l_rpn = l_roi = zero
    if len(gt_boxes) == 0:
        return (zero, (0, 0)) if return_counts else zero
    ign = None  # train_cfg ignore_iof_thr = −1 cho cả rpn và rcnn -> assigner không dùng crowd

    # --- RPN: mọi anchor dương (assigner rpn, labels = 0 như two_stage.loss) ---
    rpn = model.rpn_head
    cls_scores, bbox_preds = rpn(x)
    sizes = [c.shape[-2:] for c in cls_scores]
    anchor_list, valid_list = rpn.get_anchors(sizes, [meta], device)
    anchors, valid = torch.cat(anchor_list[0]), torch.cat(valid_list[0])
    inside = anchor_inside_flags(anchors, valid, meta["img_shape"][:2], rpn.train_cfg["allowed_border"])
    rpn_gt = InstanceData(bboxes=gt_boxes, labels=torch.zeros_like(gt.labels))
    ar = rpn.assigner.assign(InstanceData(priors=anchors[inside]), rpn_gt, ign)
    pos = ar.gt_inds > 0
    logits = torch.cat([c[0].permute(1, 2, 0).reshape(-1) for c in cls_scores])[inside][pos]
    if len(logits):
        n_rpn = len(logits)
        l_rpn = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))

    # --- RoI: proposal (detach, như two_stage) + GT làm proposal; mọi RoI dương ---
    proposals = rpn.predict_by_feat(cls_scores, bbox_preds, batch_img_metas=[meta],
                                    cfg=model.train_cfg.rpn_proposal)[0].bboxes
    roi = model.roi_head
    priors = proposals
    if roi.bbox_sampler.add_gt_as_proposals:
        priors = torch.cat([gt_boxes, proposals.float()])
    ar = roi.bbox_assigner.assign(InstanceData(priors=priors),
                                  InstanceData(bboxes=gt_boxes, labels=gt.labels), ign)
    pos_priors = priors[ar.gt_inds > 0]
    if len(pos_priors):
        n_roi = len(pos_priors)
        cls = roi._bbox_forward(x, bbox2roi([pos_priors]))["cls_score"]
        bg = torch.full((len(cls),), roi.bbox_head.num_classes, dtype=torch.long, device=device)
        l_roi = F.cross_entropy(cls, bg)
    out = -(l_rpn + l_roi)
    return (out, (n_rpn, n_roi)) if return_counts else out


def cls_term(model, batch_inputs, batch_data_samples, cls: str):
    """Thành phần phân loại trên forward hiện tại (hook backbone, nếu có, đã gắn bởi caller)."""
    if cls == "c2":
        losses = model.loss(batch_inputs, batch_data_samples)
        return sum_loss_dict({t: losses[t] for t in C2_TERMS})
    assert cls == "vanish"
    return vanish_loss(model, model.extract_feat(batch_inputs), batch_data_samples)


def make_view_loss(feats_clean, k: float, cls: str, weight: float, repair_feats=None, feat_weight: float = 1.0):
    """loss_fn(model, ảnh view, data_sample view): L_OSFD trên backbone adv + weight·L_cls trên
    nhánh head; repair_feats = (C2, C3) sạch -> thay vào trước FPN (R-a), None = không sửa."""
    def loss_fn(model, pixel_tensor, data_sample):
        cap = {}

        def hook(m, i, o):
            cap["f"] = o
            if repair_feats is not None:
                return tuple(repair_feats) + tuple(o[len(repair_feats):])
            return None
        h = model.backbone.register_forward_hook(hook)
        try:
            batch_inputs, bds = to_batch(model, [pixel_tensor], [data_sample])
            c = cls_term(model, batch_inputs, bds, cls)
        finally:
            h.remove()
        if feat_weight == 0:  # CLS-OCT (progress_log 2026-10-08): chỉ loss cls
            return weight * c
        feat = sum(F.mse_loss(k * fc, fa) for fc, fa in zip(feats_clean, cap["f"]))
        return feat_weight * feat + weight * c
    return loss_fn


def repair_robust_attack(model, orig_pixels, data_sample, steps: int, epsilon: float = 5.0,
                         alpha: float = 1.0, k: float = 3.0, momentum: float = 1.0,
                         cls: str = "c2", cls_weight: float = 4.0, repair_p: float = 0.0,
                         repair_levels: int = 2, repair_seed: int = 0, info=None,
                         rrb_theta: float = 7.0, rrb_l_s: int = 10, rrb_rho: float = 0.8,
                         rrb_s_max: float = 1.1, rrb_sigma: float = 6.0, on_step=None):
    """MI + RRB (co-transform GT) + L_OSFD + cls_weight·L_cls, phép sửa R-a với xác suất repair_p
    mỗi step (thay `repair_levels` mức nông đầu tiên = C2, C3). 1 backward / step, 2 view."""
    model_size = tuple(data_sample.img_shape)
    clean_pixels = resize_to_model(orig_pixels, model_size)
    with torch.no_grad():
        feats_clean = tuple(f.detach() for f in extract_features(model, clean_pixels, data_sample))
    rng = np.random.default_rng(repair_seed)
    rrb_kw = dict(theta=rrb_theta, l_s=rrb_l_s, rho=rrb_rho, s_max=rrb_s_max, sigma=rrb_sigma)
    log = {"repair_on": [], "forward_no_grad": 0}

    def views_fn(img, ds, step_idx, total):
        on = bool(rng.random() < repair_p)  # luôn rút 1 lần / step (định nghĩa khóa)
        log["repair_on"].append(on)
        out = []
        for v, dsv, cv in rrb_views(img, ds, with_gt=True, companion=clean_pixels, **rrb_kw):
            rep = None
            if on:
                with torch.no_grad():
                    rep = tuple(f.detach() for f in extract_features(model, cv, dsv)[:repair_levels])
                log["forward_no_grad"] += 1
            out.append((v, dsv, make_view_loss(feats_clean, k, cls, cls_weight, rep)))
        return out

    noise = run_iterative_attack(model, orig_pixels, data_sample, loss_fn=None, steps=steps,
                                 epsilon=epsilon, alpha=alpha, momentum=momentum,
                                 views_fn=views_fn, on_step=on_step)
    if info is not None:
        info.update(log)
    return noise
