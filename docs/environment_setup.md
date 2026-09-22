# Environment Setup (RTX 3090 / RTX 4000 Ada, Linux, GPU thuê)

Bối cảnh: mỗi lần chạy thí nghiệm là một máy GPU thuê mới hoàn toàn (ổ đĩa trống, không có gì được giữ lại giữa các lần thuê). Mọi thứ cần để dựng lại môi trường phải nằm trong repo (git-tracked) — không dựa vào bất cứ gì đã cài thủ công lần trước.

## Cách dùng

```bash
git clone <repo-url> transfer-attack-final
cd transfer-attack-final
bash scripts/setup_env.sh
source .venv/bin/activate
```

Script tự làm hết: kiểm tra GPU/driver, cài apt deps, tạo venv mới (xóa venv cũ nếu có), cài PyTorch/MMCV/MMDetection theo version đã pin, verify bằng import + `torch.cuda.is_available()`, và ghi báo cáo môi trường vào `environment_report.txt` (không commit — xem `.gitignore`).

Chạy 1 lệnh, xong là có venv sẵn sàng.

## Vì sao chọn các thứ này

**`venv` chuẩn thay vì conda** — máy thuê chỉ sống 1 phiên, không cần quản lý nhiều môi trường song song hay CUDA toolkit riêng của conda. Wheel PyTorch cu118 tự mang theo CUDA runtime.

| Package | Version | Ghi chú |
|---|---|---|
| Python | **3.10** (pin cứng) | mặc định Ubuntu 22.04 — base image phổ biến nhất ở GPU rental. Script tự cài qua deadsnakes PPA nếu máy không có sẵn |
| torch / torchvision | 2.1.2 / 0.16.2, build `cu118` | xem mục GPU compatibility bên dưới |
| mmengine | mới nhất qua `mim` | mim tự chọn bản khớp torch/cuda đã cài |
| mmcv | 2.1.0 | cài qua `mim`, không cài qua pip thường (dễ sai bản build) |
| mmpretrain | ≥1.2.0 | **bắt buộc** để dùng được config ConvNeXt/Swin trong MMDetection — các config này đăng ký backbone qua registry của mmpretrain, thiếu là import lỗi dù mmdet đã cài đủ |
| mmdetection | **v3.3.0**, clone + editable install vào `third_party/mmdetection` | pin theo `docs/protocol_lock.md`. Clone source (không chỉ `pip install mmdet`) để có đủ `configs/` cần cho Controlled/Generalization Panel |
| numpy | 1.26.4 (pin `<2`) | mmcv/mmdet ở version trên chưa tương thích đầy đủ numpy 2.x |
| pycocotools, opencv-python-headless | mới nhất | eval COCO mAP, đọc/ghi ảnh; bản `headless` vì server không có màn hình |

## GPU compatibility — RTX 3090 vs RTX 4000 Ada

- **RTX 3090**: kiến trúc Ampere, compute capability **sm_86**. Wheel `cu118` hỗ trợ native, không vấn đề gì.
- **RTX 4000 Ada**: kiến trúc Ada Lovelace, compute capability **sm_89**. Wheel `cu118` của PyTorch 2.1.2 **không build sẵn kernel native sm_89**, nhưng vẫn chạy được nhờ PTX forward-compatibility trong cùng họ CUDA 8.x (JIT compile PTX của sm_86/sm_80 sang sm_89 khi khởi động) — lần chạy đầu có thể chậm hơn vài giây do JIT, các lần sau như bình thường. Đây là cơ chế NVIDIA đảm bảo chính thức, không phải hack.
- Script **không cần đổi gì theo GPU** — cùng 1 pin `cu118` chạy được cả hai. Nếu sau này thuê GPU đời mới hơn nữa (Hopper/Blackwell) mới cần xét lại CUDA tag.

## Checkpoint & dataset — KHÔNG nằm trong setup_env.sh

`setup_env.sh` chỉ dựng môi trường Python. Checkpoint (`.pth`) và ảnh COCO subset (n=300/n=1000) tải riêng ở bước sau (script `scripts/download_checkpoints.sh` — **chưa viết**, sẽ làm khi bắt đầu Baseline-First Stage), vì:
- Checkpoint tải qua `mim download mmdet --config <name> --dest .` — danh sách config chính xác nằm ở `docs/protocol_lock.md`, đổi hyperparameter không kéo theo đổi checkpoint nên tách riêng cho gọn.
- Ảnh COCO subset (n=300/n=1000) phải cố định theo seed (idea.md §4: "Không tune trên 1000 ảnh") — danh sách image_id sẽ được commit vào `data/image_lists/` (git-tracked, nhẹ) để tái lập chính xác giữa các lần thuê máy khác nhau, nhưng ảnh thật (`data/coco/`) thì tải lại mỗi phiên, không commit.

## Persistent storage cho artifact nặng

Checkpoint và ảnh adversarial sinh ra trong lúc chạy thí nghiệm **không sống sót qua lần trả máy** nếu không tự sync ra ngoài. Repo git chỉ giữ: code, config nhẹ, danh sách image_id, và **kết quả số** (`results/runs/*/metrics.json` — nhẹ, chính là baseline table idea.md §8-9). Còn checkpoint/ảnh adversarial dùng cho Mechanism Stage (idea.md §11) sau này cần giữ lại thật thì phải tự quyết định nơi lưu (cloud bucket / scp về máy khác) khi tới lúc đó — chưa cấu hình vì chưa cần ngay ở Baseline-First Stage (chỉ cần metrics số, không cần giữ ảnh adversarial lâu dài).

## Troubleshooting

- **`nvidia-smi` không chạy được** → image máy thuê chưa cài driver NVIDIA, hoặc container không được cấp quyền GPU. Kiểm tra lại cấu hình thuê máy trước khi chạy script.
- **`torch.cuda.is_available()` trả về `False`** sau khi cài xong → driver quá cũ so với CUDA 11.8 (cần driver ≥ 450.80.02), hoặc container không pass-through GPU đúng. Thử `nvidia-smi` xem driver version.
- **Build `mmcv` rất lâu (5–10 phút)** → bình thường, mmcv build custom CUDA ops. Nếu quá 20 phút, kiểm tra `ninja-build` đã cài chưa (script đã cài qua apt).
- **Import config ConvNeXt/Swin báo lỗi thiếu module** → thường do thiếu `mmpretrain`, script đã cài nhưng nếu tự thêm backbone/config mới, kiểm tra lại registry cần package nào.
- **RTX 4000 Ada chạy chậm bất thường ở lần đầu tiên** → PTX JIT compile (xem mục GPU compatibility), bình thường, chỉ xảy ra 1 lần đầu mỗi phiên.
- **`environment_report.txt`** ở root sau khi chạy xong ghi lại đầy đủ `pip freeze` + thông tin GPU — dùng để đối chiếu khi 1 thí nghiệm không tái lập được giữa 2 lần thuê máy khác nhau.

## Khi cần đổi version

Không sửa version trực tiếp trong đầu — nếu cần nâng cấp package, sửa biến ở đầu `scripts/setup_env.sh`, ghi lý do vào `docs/progress_log.md`, rồi mới chạy lại. Nếu đổi cả pin mmdetection version, cập nhật đồng bộ `docs/protocol_lock.md`.
