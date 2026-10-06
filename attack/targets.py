"""Load target bất kỳ (Controlled + Generalization Panel) và gradient loss GT của target theo ẢNH GỐC — chỉ để PHÂN TÍCH
(E1, progress_log 2026-10-06), không bao giờ dùng sinh attack (threat model idea.md §3).

Ảnh gốc [C,h0,w0] -> resize khả vi về khung của chính target (img_shape theo test pipeline của target) -> data_preprocessor
-> model.loss. DINO chỉ tính loss ở train mode (dn queries): bật train mode nhưng đặt eval mọi Dropout / DropPath / BN, chỉ
lấy số hạng KHÔNG-dn (dn_mask chặn query khớp nhìn dn query -> loss khớp không phụ thuộc nhiễu dn).
"""
import glob
import os

import torch
from mmdet.apis import init_detector

from attack.data import AttackDataset
from attack.models import CONTROLLED_PANEL, load_model
from attack.preprocess import resize_to_model, sum_loss_dict, to_batch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs")
CKPT = os.path.join(REPO_ROOT, "checkpoints")
# = scripts/eval_generalization.py GEN_PANEL (config, glob checkpoint)
GEN = {
    "yolox_s": ("yolox/yolox_s_8xb8-300e_coco.py", "yolox_s_8x8*"),
    "dino_swin_l": ("dino/dino-5scale_swin-l_8xb2-36e_coco.py", "dino-5scale_swin-l*"),
    "dino_r50": ("dino/dino-4scale_r50_8xb2-12e_coco.py", "dino-4scale_r50_8xb2-12e*"),
}


def config_path(key: str) -> str:
    return CONTROLLED_PANEL[key].config_path if key in CONTROLLED_PANEL else os.path.join(CFG, GEN[key][0])


def load_any(key: str, device: str = "cuda:0"):
    if key in CONTROLLED_PANEL:
        return load_model(key, device)
    ck = glob.glob(os.path.join(CKPT, GEN[key][1] + ".pth"))
    assert len(ck) == 1, f"{key}: checkpoint {ck}"
    return init_detector(config_path(key), ck[0], device=device)


def target_dataset(key: str, image_ids):
    """data_sample ở khung của target (GT co-transform qua Resize của chính target)."""
    return AttackDataset(image_ids, config_path=config_path(key))


def kind(key: str) -> str:
    return "dino" if key.startswith("dino") else "yolox" if key.startswith("yolox") else "mrcnn"


def _loss_mode(model, k: str):
    if k != "dino":
        model.eval()
        return
    model.train()
    for m in model.modules():
        name = type(m).__name__
        if "Drop" in name or isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def select_terms(losses: dict, k: str, objective: str) -> dict:
    """objective 'cls' (chính) | 'task' (đủ, không-dn)."""
    keys = [n for n in losses if "loss" in n and "dn_" not in n]
    if objective == "cls":
        if k == "mrcnn":
            keys = [n for n in keys if n in ("loss_rpn_cls", "loss_cls")]
        elif k == "yolox":
            keys = [n for n in keys if n in ("loss_cls", "loss_obj")]
        else:
            keys = [n for n in keys if n.endswith("loss_cls")]
    assert keys, f"không có số hạng loss {objective} trong {list(losses)}"
    return {n: losses[n] for n in keys}


def target_grads(model, key: str, orig: torch.Tensor, ds, objectives=("cls", "task")):
    """-> {objective: ∇ loss theo ảnh gốc [C,h0,w0]}. Mỗi objective 1 forward + .backward() (KHÔNG autograd.grad: Swin-L
    train mode dùng gradient checkpointing, chỉ tương thích .backward(); giữ checkpointing để vừa bộ nhớ). Tham số model
    tạm đóng băng (không tích gradient vào weight). RNG torch đặt lại như nhau trước mỗi forward (sampler Mask R-CNN)."""
    k = kind(key)
    _loss_mode(model, k)
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    state = torch.get_rng_state(), torch.cuda.get_rng_state()
    try:
        out, keys = {}, {}
        for obj in objectives:
            torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1])
            x = orig.clone().requires_grad_(True)
            inputs, dss = to_batch(model, [resize_to_model(x, tuple(ds.img_shape))], [ds])
            losses = model.loss(inputs, dss)
            keys = {o: sorted(select_terms(losses, k, o)) for o in ("cls", "task")}
            sum_loss_dict(select_terms(losses, k, obj)).backward()
            out[obj] = x.grad.detach()
            del losses, inputs
        return out, keys["cls"], keys["task"]
    finally:
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)
        model.eval()
