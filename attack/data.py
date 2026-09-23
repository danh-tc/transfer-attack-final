"""Load ảnh + GT theo danh sách cố định (data/image_lists/n300.csv, n1000.csv).

Dùng lại NGUYÊN VẸN dataset pipeline (Resize, normalize stats, ...) đã khóa trong
config mmdet của surrogate — xem docs/protocol_lock.md: 4 model Controlled Panel
dùng chung test pipeline (Resize scale=(1333,800), keep_ratio=True) và chung
data_preprocessor (mean/std/bgr_to_rgb), nên 1 tensor pixel sinh ra từ đây feed
thẳng được vào cả 4 model, không cần resize riêng noise cho từng target.
"""
import csv
import os
from typing import List

from mmdet.registry import DATASETS
from mmengine.config import Config
from mmengine.registry import init_default_scope

from attack.models import CONTROLLED_PANEL, SURROGATE_KEY

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_image_ids(csv_path: str) -> List[int]:
    """Đọc data/image_lists/{n300,n1000}.csv, trả về list image_id theo đúng thứ tự trong file."""
    ids = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["image_id"]))
    return ids


def _gt_before_resize(pipeline: list) -> list:
    """Đưa LoadAnnotations lên TRƯỚC Resize, và rasterize mask (poly2mask=True).

    Test pipeline gốc của mmdet đặt LoadAnnotations SAU Resize (vì lúc eval GT chỉ
    dùng qua file annotation, ở tọa độ ảnh gốc). Hệ quả: gt_instances.bboxes/masks
    nằm ở tọa độ ảnh GỐC (vd 640x428) trong khi `inputs` đã resize (vd 1196x800) —
    mọi loss dựa trên GT (MI/DI-FGSM, AugTrans) tính sai vị trí. Đây là bug thật,
    xem docs/progress_log.md. Đặt LoadAnnotations trước để Resize co-transform luôn
    box + mask về đúng khung ảnh đã resize.

    poly2mask=True: mask dạng BitmapMasks (numpy) để các biến đổi hình học trong
    attack (DI/AugTrans) co-transform mask bằng đúng phép biến đổi tensor dùng cho
    ảnh, không phải tự xử lý polygon (xem attack/methods/box_transforms.py).
    """
    pipeline = [dict(t) for t in pipeline]
    types = [t["type"] for t in pipeline]
    ann_idx, resize_idx = types.index("LoadAnnotations"), types.index("Resize")
    ann = pipeline.pop(ann_idx)
    ann["poly2mask"] = True
    pipeline.insert(resize_idx if ann_idx > resize_idx else resize_idx - 1, ann)
    return pipeline


class AttackDataset:
    """Dataset trả về (img_id, inputs, data_sample) cho 1 danh sách image_id cố định.

    - `inputs`: Tensor float32 [C,H,W], CHƯA normalize (đúng output của PackDetInputs
      trong mmdet v3 — normalize xảy ra sau, bên trong model.data_preprocessor,
      xem attack/preprocess.py). Đây là không gian pixel để cộng noise adversarial vào.
    - `data_sample`: DetDataSample, đã có gt_instances.bboxes/labels (LoadAnnotations
      nằm sẵn trong test pipeline của các config Controlled Panel).
    """

    def __init__(self, image_ids: List[int], config_path: str = None):
        if config_path is None:
            config_path = CONTROLLED_PANEL[SURROGATE_KEY].config_path
        init_default_scope("mmdet")
        cfg = Config.fromfile(config_path)
        dataset_cfg = cfg.test_dataloader.dataset.copy()
        # Tắt serialize_data: chỉ vài trăm/nghìn ảnh, không cần tối ưu memory kiểu
        # multi-worker DataLoader — giữ self.data_list là list thường cho dễ lọc/reorder.
        dataset_cfg["serialize_data"] = False
        dataset_cfg["pipeline"] = _gt_before_resize(dataset_cfg["pipeline"])
        self._mmdet_dataset = DATASETS.build(dataset_cfg)
        self._mmdet_dataset.full_init()

        id_to_idx = {
            info["img_id"]: idx for idx, info in enumerate(self._mmdet_dataset.data_list)
        }
        missing = [i for i in image_ids if i not in id_to_idx]
        if missing:
            raise ValueError(
                f"{len(missing)} image_id trong danh sách không có trong dataset "
                f"(vd: {missing[:5]}) — kiểm tra data/coco/ đã tải đủ chưa."
            )
        self.image_ids = list(image_ids)
        self._indices = [id_to_idx[i] for i in image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i: int):
        img_id = self.image_ids[i]
        dataset_idx = self._indices[i]
        data_info = self._mmdet_dataset.get_data_info(dataset_idx)
        packed = self._mmdet_dataset.pipeline(data_info)
        return {
            "img_id": img_id,
            "inputs": packed["inputs"].float(),
            "data_sample": packed["data_samples"],
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def build_attack_dataset(split: str = "n300") -> AttackDataset:
    """split: 'n300' hoặc 'n1000' — khớp tên file trong data/image_lists/."""
    csv_path = os.path.join(REPO_ROOT, "data/image_lists", f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} — chạy scripts/generate_image_lists.py trước (hoặc scripts/bootstrap.sh)."
        )
    image_ids = load_image_ids(csv_path)
    return AttackDataset(image_ids)
