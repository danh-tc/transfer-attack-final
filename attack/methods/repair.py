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
import contextlib
import copy

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
    if cls == "rpn":  # chỉ loss_rpn_cls (progress_log 2026-10-08): bỏ sinh proposal + RoI head; nhãn GT -> 0 như two_stage.loss
        rpn_ds = copy.deepcopy(batch_data_samples)
        for d in rpn_ds:
            d.gt_instances.labels = torch.zeros_like(d.gt_instances.labels)
        return sum_loss_dict({"loss_rpn_cls": model.rpn_head.loss(model.extract_feat(batch_inputs), rpn_ds)["loss_rpn_cls"]})
    assert cls == "vanish"
    return vanish_loss(model, model.extract_feat(batch_inputs), batch_data_samples)


@contextlib.contextmanager
def roi_cls_filter(model, cap=None, stats=None, head=None):
    """C2-OCT-1V-POS / POS-CAP (progress_log 2026-10-18, kiểu DAG): loss_cls của RoI head chỉ trên RoI DƯƠNG (bỏ RoI nền
    = nhãn num_classes); cap: bỏ thêm RoI dương có p(lớp đúng) < cap. avg_factor = số RoI còn giữ (giữ độ lớn trung bình
    / RoI, không vô tình hạ λ). loss_rpn_cls KHÔNG đổi. Thay tạm forward của bbox_head.loss_cls (instance attribute).
    head: head khác (vd model.rpn_head, num_classes = 1, nền = nhãn 1) — dùng cho loss box mồi chỉ trên mẫu dương."""
    head = head if head is not None else model.roi_head.bbox_head
    crit, orig = head.loss_cls, head.loss_cls.forward
    nc = head.num_classes

    def fwd(cls_score, label, weight=None, avg_factor=None, **kw):
        keep = label < nc
        if cap is not None:
            p = torch.softmax(cls_score.float(), -1).gather(1, label.clamp(max=nc)[:, None])[:, 0]
            keep = keep & (p >= cap)
        w = (weight if weight is not None else torch.ones_like(label, dtype=cls_score.dtype)) * keep
        n = max(float(keep.sum()), 1.0)
        if stats is not None:
            stats["kept"] += int(keep.sum()); stats["pos"] += int((label < nc).sum()); stats["total"] += label.numel()
        return orig(cls_score, label, w, avg_factor=n, **kw)
    crit.forward = fwd
    try:
        yield
    finally:
        del crit.forward  # về lại forward của lớp (orig là bound method của lớp)


@contextlib.contextmanager
def roi_size_weight(model, thr_model: float, w_large: float):
    """C2-OCT-1V-WP2-SZC (progress_log 2026-10-22): loss_cls của RoI head nhân w_large ở RoI DƯƠNG gán cho GT có diện tích
    (khung view) ≥ thr_model -> object lớn gần như chỉ còn L_OSFD. Thứ tự RoI = [dương, âm] (StandardRoIHead: priors =
    cat(pos, neg); BBoxHead._get_targets_single cùng thứ tự). avg_factor giữ nguyên (bỏ đóng góp, không dồn sang RoI khác).
    loss_rpn_cls KHÔNG đổi. Thay tạm loss_and_target + forward của bbox_head.loss_cls."""
    head = model.roi_head.bbox_head
    crit, orig_fwd, orig_lat = head.loss_cls, head.loss_cls.forward, head.loss_and_target
    roi_w = {}

    def lat(cls_score, bbox_pred, rois, sampling_results, *a, **kw):
        ws = []
        for res in sampling_results:
            gb = res.pos_gt_bboxes
            gb = (gb.tensor if hasattr(gb, "tensor") else gb).float()
            area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            ws += [torch.where(area >= thr_model, w_large, 1.0), torch.ones(len(res.neg_priors), device=gb.device)]
        roi_w["w"] = torch.cat(ws)
        return orig_lat(cls_score, bbox_pred, rois, sampling_results, *a, **kw)

    def fwd(cls_score, label, weight=None, avg_factor=None, **kw):
        w = roi_w["w"].to(cls_score.dtype)
        w = w * weight if weight is not None else w
        return orig_fwd(cls_score, label, w, avg_factor=avg_factor, **kw)
    head.loss_and_target, crit.forward = lat, fwd
    try:
        yield
    finally:
        del head.loss_and_target, crit.forward  # về lại phương thức của lớp


@contextlib.contextmanager
def roi_margin(model, mode: str, scale: float):
    """C2-OCT-1V-MRG / -MRGF (progress_log 2026-10-20, kiểu CW): thay CE của RoI head bằng margin KHÔNG chặn trên RoI DƯƠNG:
    L = scale · mean_{RoI dương} (max_{c ≠ đúng} z_c − z_đúng); mode "all": c chạy mọi logit khác lớp đúng (kể cả nền),
    "fg": chỉ 80 lớp object (không đẩy về nền). Ascent -> đẩy logit đúng xuống dưới lớp sai mạnh nhất, tiếp tục cả khi đã sai.
    RoI nền bị bỏ (POS: không quan trọng). loss_rpn_cls KHÔNG đổi. Thay tạm forward của bbox_head.loss_cls."""
    assert mode in ("all", "fg")
    head = model.roi_head.bbox_head
    crit, nc = head.loss_cls, head.num_classes

    def fwd(cls_score, label, weight=None, avg_factor=None, **kw):
        keep = label < nc
        z = cls_score.float()[keep]
        if len(z) == 0:
            return cls_score.sum() * 0.0
        t = label[keep]
        zt = z.gather(1, t[:, None])[:, 0]
        other = z.scatter(1, t[:, None], float("-inf"))
        if mode == "fg":
            other = other[:, :nc]
        return scale * (other.max(1).values - zt).mean()
    crit.forward = fwd
    try:
        yield
    finally:
        del crit.forward


FAB_LABEL_OFFSET = 1000  # box mồi đi chung gt_instances (co-transform cùng GT thật), nhãn + offset để tách lại
FAB_JITTER = 32


def split_fab(data_sample):
    """(ds GT thật, ds box mồi với nhãn thật của mồi) từ gt_instances gộp; mồi có nhãn ≥ FAB_LABEL_OFFSET."""
    fake = data_sample.gt_instances.labels >= FAB_LABEL_OFFSET
    real_ds, fab_ds = copy.copy(data_sample), copy.copy(data_sample)
    real_ds.gt_instances = data_sample.gt_instances[~fake]
    fi = data_sample.gt_instances[fake]
    fab_ds.gt_instances = InstanceData(bboxes=fi.bboxes, labels=fi.labels - FAB_LABEL_OFFSET)
    return real_ds, fab_ds


def fab_term(model, x, fab_ds):
    """C2-OCT-1V-FAB (progress_log 2026-10-18): loss DESCENT để R50 phát hiện TỰ TIN box mồi (cỡ object, nhãn cố định):
    loss_rpn_cls (objectness tại mồi) + loss_cls RoI head trên proposal = mồi rung FAB_JITTER lần (σ = 0.1 cạnh).
    Dùng lại feature FPN x của forward đang có (không forward backbone thêm). Không mồi nào còn trong view -> 0."""
    b = fab_ds.gt_instances.bboxes
    b = (b.tensor if hasattr(b, "tensor") else b).float()
    if len(b) == 0:
        return x[0].sum() * 0.0
    rds = copy.copy(fab_ds)
    rds.gt_instances = InstanceData(bboxes=b, labels=torch.zeros_like(fab_ds.gt_instances.labels))
    # CHỈ mẫu dương (anchor / RoI trúng mồi): không thì descent ép MỌI chỗ khác — kể cả object thật — thành nền.
    with roi_cls_filter(model, head=model.rpn_head):
        l_rpn = sum_loss_dict({"loss_rpn_cls": model.rpn_head.loss(x, [rds])["loss_rpn_cls"]})
    wh = (b[:, 2:] - b[:, :2]).repeat(1, 2)
    props = b.repeat_interleave(FAB_JITTER, 0)
    props = props + torch.randn_like(props) * 0.1 * wh.repeat_interleave(FAB_JITTER, 0)
    h, w = fab_ds.img_shape
    props = torch.stack([props[:, 0].clamp(0, w), props[:, 1].clamp(0, h), props[:, 2].clamp(0, w), props[:, 3].clamp(0, h)], 1)
    fds = copy.copy(fab_ds)
    fds.gt_instances = InstanceData(bboxes=b, labels=fab_ds.gt_instances.labels)
    with roi_cls_filter(model):
        l_roi = model.roi_head.loss(x, [InstanceData(bboxes=props.detach())], [fds])["loss_cls"]
    return l_rpn + l_roi


def roi_embed(model, x, boxes):
    """Embedding mức object của Mask R-CNN: RoIAlign 7×7 trên FPN x tại `boxes` [N,4] (khung model) -> 2 fc chung
    (Shared2FCBBoxHead, ReLU) -> [N, 1024] — đúng vector mà bộ phân loại / hồi quy box của RoI head dùng."""
    rois = bbox2roi([boxes])
    ext = model.roi_head.bbox_roi_extractor
    f = ext(x[:ext.num_inputs], rois).flatten(1)
    for fc in model.roi_head.bbox_head.shared_fcs:
        f = F.relu(fc(f))
    return f


def _pooled_mse(target, fa, kernel):
    """MSE trên feature gộp (avg-pool kernel | "gap"), co về giá trị MSE từng điểm (hệ số detach)."""
    pool = (lambda f: f.mean((-2, -1))) if kernel == "gap" else (lambda f: F.avg_pool2d(f, kernel, kernel, ceil_mode=True))
    p = F.mse_loss(pool(target), pool(fa))
    return p * (F.mse_loss(target, fa).detach() / p.detach().clamp_min(1e-12))


def _ln_mse(target, fa):
    """C2-OCT-1V-WP2-LN (progress_log 2026-10-22): MSE giữa LayerNorm theo kênh tại từng vị trí của target và fa (chỉ đo
    thay đổi HƯỚNG của vector feature mỗi token — thứ còn lại sau LN của Transformer; LN(k·F) = LN(F) nên k không tác dụng),
    nhân hệ số detach để GIÁ TRỊ bằng MSE gốc (giữ cân bằng với L_cls như GLB, không tune)."""
    ln = lambda f: F.layer_norm(f.float().permute(0, 2, 3, 1), f.shape[1:2]).permute(0, 3, 1, 2)
    p = F.mse_loss(ln(target), ln(fa))
    return p * (F.mse_loss(target.float(), fa.float()).detach() / p.detach().clamp_min(1e-12))


def _size_weighted_mse(target, fa, stage, data_sample, size_w):
    """C2-OCT-1V-WP2-SZ: mean(w·(target − fa)²) / mean(w); w = 1 + a trong box GT lớn (diện tích quy về ảnh gốc ≥ thr)."""
    a, thr = size_w
    gb = data_sample.gt_instances.bboxes
    gb = (gb.tensor if hasattr(gb, "tensor") else gb).float()
    sx, sy = data_sample.metainfo["scale_factor"]
    w = torch.ones(fa.shape[-2:], device=fa.device, dtype=torch.float32)
    stride = 4 * 2 ** stage  # stride stage ResNet: 4, 8, 16, 32
    for x1, y1, x2, y2 in gb.tolist():
        if (x2 - x1) * (y2 - y1) / (sx * sy) >= thr:
            w[max(0, int(y1 // stride)):int(-(-y2 // stride)), max(0, int(x1 // stride)):int(-(-x2 // stride))] = 1 + a
    sq = (target.float() - fa.float()).pow(2).mean(1)[0]  # [H, W], trung bình kênh
    return (w * sq).sum() / w.sum()


def _box_mask(fa, stage, data_sample):
    """[H, W] bool ở stage `stage`: ô stride 4·2^stage giao ≥ 1 box GT của view (khung view)."""
    gb = data_sample.gt_instances.bboxes
    gb = (gb.tensor if hasattr(gb, "tensor") else gb).float()
    m = torch.zeros(fa.shape[-2:], device=fa.device, dtype=torch.bool)
    stride = 4 * 2 ** stage
    for x1, y1, x2, y2 in gb.tolist():
        m[max(0, int(y1 // stride)):int(-(-y2 // stride)), max(0, int(x1 // stride)):int(-(-x2 // stride))] = True
    return m


def _xbg_mse(target, fbg, fa, stage, data_sample, mode):
    """C2-OCT-1V-WP2-XBG / -XBGS (progress_log 2026-10-23): số hạng feature của 1 stage khi có đích F_R50(x_bg).
    sq = (k·F_sạch − F_adv)², T = mean trong box của (F_bg − F_adv)² (trung bình kênh; F_bg = feature của view ảnh đã xóa
    object, co-transform). "add": MSE gốc − c·T; "swap": phần MSE gốc NGOÀI box − c·T (trong box chỉ còn kéo về F_bg).
    c detach để GIÁ TRỊ c·T bằng phần MSE gốc trong box (giữ cân bằng với L_cls như LN / GLB, không tune). Không box -> MSE gốc."""
    m = _box_mask(fa, stage, data_sample)
    sq = (target.float() - fa.float()).pow(2).mean(1)[0]
    full = sq.mean()
    if not m.any():
        return full
    frac = m.float().mean()
    t = (fbg.float() - fa.float()).pow(2).mean(1)[0][m].mean()
    pull = t * (frac * sq[m].mean()).detach() / t.detach().clamp_min(1e-12)
    base = full if mode == "add" else sq[~m].sum() / sq.numel()
    return base - pull


def _is_reweight(g, n_pairs, beta, alpha, gen):
    """IS (E30, COGO): g [1,C,H,W] gradient output tầng; triệt kênh gradient TƯƠNG QUAN cao (dư thừa = đặc thù surrogate).
    Rút n_pairs cặp kênh, |Pearson corr|; τ = beta·mean; cặp ≥ τ -> cả 2 kênh +1 dư thừa; w = clamp(1 − alpha·count, 0, 1)."""
    c = g.shape[1]
    if n_pairs <= 0 or c < 2:
        return g
    gf = g[0].reshape(c, -1).float()
    gf = gf - gf.mean(1, keepdim=True)
    norm = gf.norm(dim=1) + 1e-12
    ii = torch.randint(0, c, (n_pairs,), generator=gen, device=g.device)
    jj = torch.randint(0, c, (n_pairs,), generator=gen, device=g.device)
    valid = ii != jj
    corr = (gf[ii] * gf[jj]).sum(1).abs() / (norm[ii] * norm[jj])
    corr = corr * valid
    tau = beta * corr.sum() / (valid.sum() + 1e-12)
    red = corr >= tau
    count = torch.zeros(c, device=g.device)
    count.index_add_(0, ii[red & valid], torch.ones(int((red & valid).sum()), device=g.device))
    count.index_add_(0, jj[red & valid], torch.ones(int((red & valid).sum()), device=g.device))
    w = (1 - alpha * count).clamp(0, 1).to(g.dtype)
    return g * w.view(1, c, 1, 1)


@contextlib.contextmanager
def is_backward_hook(model, stages, beta, alpha, n_pairs, gen):
    """Gắn grad-hook lên output các layer backbone trong `stages` ({2:layer3, 3:layer4}) để IS reweight gradient."""
    layers = {2: model.backbone.layer3, 3: model.backbone.layer4}
    handles = []
    def mk_fwd():
        def fwd(m, i, o):
            if o.requires_grad:  # bỏ qua forward no_grad (feature sạch co-transform)
                handles.append(o.register_hook(lambda g: _is_reweight(g, n_pairs, beta, alpha, gen)))
        return fwd
    fh = [layers[s].register_forward_hook(mk_fwd()) for s in stages]
    try:
        yield
    finally:
        for h in fh + handles:
            h.remove()


def _fpa_mse(fc_tgt, fa, gamma: float, p: float):
    """FPA-N (E29): FP layer trong forward của ADV — mỗi bước xác suất p hoán vị vị trí CHỈ F_adv (đổi mỗi pixel với
    1 trong 4 láng giềng) trên tỉ lệ gamma kênh ngẫu nhiên; đích k·F_cln GIỮ NGUYÊN -> MSE(k·F_cln, perm(F_adv)) tạo
    mismatch chéo vị trí, gradient ép F_adv[nb] về k·F_cln -> nơ-ron dùng thông tin ngoài receptive field (giống ViT).
    torch.roll (biên vòng) differentiable. p = 0 -> F.mse_loss thường."""
    if p <= 0 or float(torch.rand(1)) >= p:
        return F.mse_loss(fc_tgt, fa)
    c = fa.shape[1]
    sel = (torch.rand(c, device=fa.device) < gamma)[None, :, None, None]
    dirs = torch.randint(0, 4, fa.shape[-2:], device=fa.device)
    rolls = [torch.roll(fa, 1, 2), torch.roll(fa, -1, 2), torch.roll(fa, 1, 3), torch.roll(fa, -1, 3)]
    fa_p = fa
    for kk in range(4):
        fa_p = torch.where(sel & (dirs[None, None] == kk), rolls[kk], fa_p)
    return F.mse_loss(fc_tgt, fa_p)


def make_view_loss(feats_clean, k: float, cls: str, weight: float, repair_feats=None, feat_weight: float = 1.0,
                   ilpd_gamma=None, ilpd_stage: int = 2, feat_chw=None, roi_ctx=None, fab_weight: float = 0.0,
                   obj_weight: float = 0.0, obj_k: float = 1.0, split: bool = False, stage_pool=None, size_w=None,
                   ln_feat: bool = False, xbg_feats=None, xbg_mode: str = "add", stage_w=None, fpa=None):
    """loss_fn(model, ảnh view, data_sample view): L_OSFD trên backbone adv + weight·L_cls trên
    nhánh head; repair_feats = (C2, C3) sạch -> thay vào trước FPN (R-a), None = không sửa.
    ilpd_gamma = γ (ILPD, NeurIPS 2023; progress_log 2026-10-18): chỉ trên đường head, stage `ilpd_stage` vào FPN =
    F_sạch + γ·(F_adv − F_sạch) (F_sạch = feats_clean co-transform của view); L_OSFD vẫn trên F_adv gốc.
    feat_chw (C2-OCT-1V-FIA, progress_log 2026-10-18): list [C_l] trọng số kênh, trung bình 1 -> L_OSFD từng stage =
    mean(w_c · (k·F_sạch − F_adv)²); w ≡ 1 trùng F.mse_loss. roi_ctx: () -> context manager quanh cls_term (roi_cls_filter).
    split (C2-OCT-1V-DM, progress_log 2026-10-20): trả cặp (feat_weight·L_feat, weight·L_cls) thay cho tổng (core dual).
    stage_pool = (s, kernel | "gap") (C2-OCT-1V-GLB, progress_log 2026-10-21): số hạng L_OSFD của stage s tính trên feature
    gộp avg-pool kernel×kernel (stride = kernel) | toàn cục, nhân hệ số (detach) để GIÁ TRỊ bằng MSE từng điểm -> chỉ đổi hướng.
    ln_feat (C2-OCT-1V-WP2-LN, progress_log 2026-10-22): L_OSFD từng stage trong không gian LayerNorm theo kênh (_ln_mse).
    size_w = (a, thr) (C2-OCT-1V-WP2-SZ, progress_log 2026-10-22): L_OSFD từng stage = mean(w·(k·F_sạch − F_adv)²) / mean(w),
    w = 1 + a trong box GT của view có diện tích quy về ảnh gốc ≥ thr (px²), 1 ngoài -> dồn đẩy feature vào object lớn.
    xbg_feats (C2-OCT-1V-WP2-XBG / -XBGS, progress_log 2026-10-23): feature backbone của view ảnh đã xóa object -> _xbg_mse.
    stage_w (C2-OCT-1V-WP2-C2W, progress_log 2026-10-24): tuple trọng số từng stage của L_OSFD (C2..C5); None ≡ (1, 1, 1, 1)."""
    assert ilpd_gamma is None or (repair_feats is None and feats_clean is not None)

    def loss_fn(model, pixel_tensor, data_sample):
        cap = {}

        def hook(m, i, o):
            cap["f"] = o
            if repair_feats is not None:
                return tuple(repair_feats) + tuple(o[len(repair_feats):])
            if ilpd_gamma is not None:
                o = list(o)
                fc = feats_clean[ilpd_stage].to(o[ilpd_stage].dtype)
                o[ilpd_stage] = fc + ilpd_gamma * (o[ilpd_stage] - fc)
                return tuple(o)
            return None
        h = model.backbone.register_forward_hook(hook)
        hn = model.neck.register_forward_hook(lambda m, i, o: cap.update(x=o)) if fab_weight > 0 or obj_weight > 0 else None
        fab_ds = None
        if fab_weight > 0:
            data_sample, fab_ds = split_fab(data_sample)
        try:
            batch_inputs, bds = to_batch(model, [pixel_tensor], [data_sample])
            with (roi_ctx() if roi_ctx is not None else contextlib.nullcontext()):
                c = cls_term(model, batch_inputs, bds, cls)
        finally:
            h.remove()
            if hn is not None:
                hn.remove()
        if fab_weight > 0:
            fab_ds.set_metainfo(bds[0].metainfo)  # batch_input_shape / pad_shape do to_batch đặt
            c = c - fab_weight / weight * fab_term(model, cap["x"], fab_ds)  # weight·c − fab_weight·L_fab
        if obj_weight > 0:
            # C2-OCT-1V-OBJ (progress_log 2026-10-18): L_obj = MSE(obj_k·e_sạch, e_adv) trên embedding RoI 1024-d tại box GT
            # của view (mức object, không cần nhãn); e_sạch qua FPN của feature backbone sạch co-transform. obj_k = 1: khoảng
            # cách thuần (k = 3 kiểu OSFD cho gradient NGƯỢC L_cls, cos tới −0.56 — đo 5 ảnh dev100).
            gb = data_sample.gt_instances.bboxes
            gb = (gb.tensor if hasattr(gb, "tensor") else gb).float()
            if len(gb):
                with torch.no_grad():
                    e_c = roi_embed(model, model.neck(tuple(feats_clean)), gb)
                e_a = roi_embed(model, cap["x"], gb)
                c = c + obj_weight / weight * F.mse_loss(obj_k * e_c, e_a)  # weight·c + obj_weight·L_obj
        if feat_weight == 0:  # CLS-OCT (progress_log 2026-10-08): chỉ loss cls
            return weight * c
        if feat_chw is not None:
            feat = sum((w.view(1, -1, 1, 1) * (k * fc - fa).pow(2)).mean()
                       for w, fc, fa in zip(feat_chw, feats_clean, cap["f"]))
        elif stage_pool is not None:
            feat = sum(_pooled_mse(k * fc, fa, stage_pool[1]) if s == stage_pool[0] else F.mse_loss(k * fc, fa)
                       for s, (fc, fa) in enumerate(zip(feats_clean, cap["f"])))
        elif ln_feat:
            feat = sum(_ln_mse(k * fc, fa) for fc, fa in zip(feats_clean, cap["f"]))
        elif size_w is not None:
            feat = sum(_size_weighted_mse(k * fc, fa, s, data_sample, size_w)
                       for s, (fc, fa) in enumerate(zip(feats_clean, cap["f"])))
        elif xbg_feats is not None:
            feat = sum(_xbg_mse(k * fc, fb, fa, s, data_sample, xbg_mode)
                       for s, (fc, fb, fa) in enumerate(zip(feats_clean, xbg_feats, cap["f"])))
        elif stage_w is not None:
            feat = sum(w * F.mse_loss(k * fc, fa) for w, fc, fa in zip(stage_w, feats_clean, cap["f"]))
        elif fpa is not None:
            # E29 (progress_log 2026-10-24): FPA-N — hoán vị vị trí feature ở các stage fpa[2], CÙNG mask cho k·fc và fa.
            feat = sum(_fpa_mse(k * fc, fa, fpa[0], fpa[1]) if s in fpa[2] else F.mse_loss(k * fc, fa)
                       for s, (fc, fa) in enumerate(zip(feats_clean, cap["f"])))
        else:
            feat = sum(F.mse_loss(k * fc, fa) for fc, fa in zip(feats_clean, cap["f"]))
        if split:
            return feat_weight * feat, weight * c
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
