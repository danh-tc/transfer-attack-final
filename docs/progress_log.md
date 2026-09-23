# Nhật ký tiến độ (Progress Log)

Chỉ thêm mới (append-only). Sau mỗi phiên làm việc có kết quả hoặc quyết định, thêm một entry mới có ngày ở cuối file. Không sửa lại entry cũ, trừ khi sửa lỗi sai sự kiện — nếu phân vân, gạch ngang phần cũ thay vì xóa.

---

## 2026-09-22 — Khóa protocol + dựng cấu trúc dự án

- `idea.md` (research plan đã khóa, baseline-first) đã có sẵn từ trước, kèm `papers/` (OSFD, AugTrans, benchmark taxonomy) và `ref-repo/OSFD-main` (code OSFD gốc, mmdet v2.28.2).
- Quyết định framework: **mmdetection v3.3.0** (không dùng v2 như OSFD ref-repo), vì v3 có sẵn config DINO/ConvNeXt/Swin cần cho Generalization Panel; code attack loop tham khảo OSFD ref-repo cần port sang v3 API.
- Khóa exact config Controlled Panel (surrogate R50 + 3 target R101/ConvNeXt-T/Swin-T, toàn bộ 3x+multi-scale) và Generalization Panel (FCOS-R50, DETR-R50, YOLOX-S, DINO-Swin-**L** thay Swin-T vì không có checkpoint chính thức) — chi tiết đầy đủ ở [protocol_lock.md](protocol_lock.md), [model_registry.md](model_registry.md).
- Khóa định nghĩa B (gradient-evaluation budget) = tổng số `.backward()` calls; AugTrans (dùng EOT, N_EOT=10) normalize bằng `iterations = B/N_EOT`. Loại B=10 khỏi so sánh chéo method vì AugTrans không đủ iteration ở mức này (chỉ 1 iteration, không đủ cho curriculum/EOT hoạt động). Primary budget set chốt: **B ∈ {50, 200}**.
- Dựng cấu trúc dự án theo khuôn mẫu từ `transfer-attack-new` (dự án nghiên cứu tương tự, thư mục khác): `CLAUDE.md` (bootstrap, auto-load mỗi session) + `docs/protocol_lock.md` + `docs/model_registry.md` + `docs/progress_log.md` (file này) + `docs/environment_setup.md` + `scripts/setup_env.sh`.
- Lý do tổ chức thế này: GPU thuê (RTX 3090 / RTX 4000 Ada), mỗi phiên là máy mới hoàn toàn, memory riêng của Claude không tồn tại trên máy GPU nếu phiên chạy trực tiếp trên máy thuê — toàn bộ context bắt buộc phải nằm trong repo (git-tracked), không phụ thuộc memory ngoài repo.
- Chưa cài môi trường thật, chưa tải checkpoint, chưa verify model nào bằng cách chạy thật (mọi AP trong model_registry.md lấy từ README GitHub, chưa tự eval lại).
- Bước tiếp theo: chạy `scripts/setup_env.sh` trên máy GPU thật lần đầu để verify script chạy được (chưa test), sau đó tải 4 checkpoint Controlled Panel + verify clean AP khớp bảng trong model_registry.md.

---

## 2026-09-22 — Chạy `setup_env.sh` lần đầu trên GPU thuê thật (RTX 3090) — 3 lỗi, đã sửa script

- Máy: RTX 3090, driver 570.211.01, 24GB VRAM (GPU thuê thật, không phải RTX 4000 Ada).
- `scripts/setup_env.sh` chạy lần đầu **fail ở bước cài mmdetection editable** (`set -e` dừng script giữa chừng, exit code thật bị pipe `tee` che mất — cần lưu ý lần sau không suy luận "exit code 0 của lệnh có tee" = script thành công, phải đọc log). 3 lỗi phát hiện, đã sửa trực tiếp vào `scripts/setup_env.sh`:
  1. **`pip install -e third_party/mmdetection`** (không có `--no-build-isolation`) → `ModuleNotFoundError: No module named 'torch'`. Nguyên nhân: `setup.py` của mmdet import `torch` ở top-level, nhưng build isolation mặc định của pip tạo venv tạm không có torch (dù venv chính đã cài). Fix: thêm `--no-build-isolation`.
  2. Sau khi thêm `--no-build-isolation`, gặp lỗi mới: `Project ... uses a build backend that is missing the 'build_editable' hook`. Nguyên nhân: `setuptools` mặc định trong venv là **60.2.0** (quá cũ), không hỗ trợ PEP 660 editable install cho package chỉ có `setup.py` (không có `pyproject.toml`) khi dùng với pip rất mới (26.2.1 trong môi trường này). Fix: `pip install -U "setuptools<81"` trước bước cài mmdet (đủ mới để hỗ trợ PEP 660, chưa nhảy sang major version có thể đổi hành vi khác — chưa test setuptools ≥81).
  3. Cài `opencv-python-headless` không pin version → kéo bản mới nhất (5.0.0.93) đòi `numpy>=2`, phá pin `numpy==1.26.4` (torch import lỗi `_ARRAY_API not found` vì mmcv/mmdet build với numpy 1.x ABI). Ngoài ra `mmcv`/`mmengine` tự kéo thêm package `opencv-python` (không phải headless) cũng bản mới nhất, cùng vấn đề. Fix: pin cả `opencv-python` lẫn `opencv-python-headless` về `4.10.0.84` (bản cuối cùng còn hỗ trợ numpy<2), cài cùng lúc với `numpy==1.26.4` để resolver không tự nâng lại, và pin lại lần nữa sau khi cài `pycocotools`/thư viện phụ trợ (phòng bị kéo lại).
  - Xung đột resolver **chấp nhận bỏ qua, không phải lỗi chặn**: `openxlab==0.1.3 requires setuptools~=60.2.0` (openxlab là dependency phụ của mmpretrain, không dùng trong pipeline attack/eval của dự án này) và `requests==2.28.2 requires urllib3<1.27` (không ảnh hưởng phần dùng trong dự án). Không cần xử lý trừ khi sau này thực sự import `openxlab` hoặc gặp lỗi liên quan `requests`.
- Sau khi sửa, verify sạch: `torch 2.1.2+cu118 cuda_available=True` (GPU: NVIDIA GeForce RTX 3090), `mmengine 0.10.7`, `mmcv 2.1.0`, `mmdet 3.3.0`, `mmpretrain 1.2.0`, `numpy 1.26.4`, `cv2 4.10.0` — không còn warning ABI numpy.
- Verify thêm (chưa nằm trong `setup_env.sh` gốc, làm thủ công để bám protocol_lock.md — "việc đầu tiên khi có GPU"): parse **và** build (`MODELS.build`) cả 4 config Controlled Panel bằng `mmdet.registry` + `init_default_scope("mmdet")` — tất cả OK, không lỗi registry ConvNeXt/Swin (mmpretrain resolve đúng):
  - `mask-rcnn_r50_fpn_ms-poly-3x_coco.py` — 44.4M params
  - `mask-rcnn_r101_fpn_ms-poly-3x_coco.py` — 63.4M params
  - `mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` — 48.1M params
  - `mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` — 47.8M params
  - Lưu ý: đây mới là build kiến trúc model (random init), **chưa tải checkpoint, chưa load weight thật, chưa eval clean AP** — cột "Ngày verify" trong `model_registry.md` vẫn để trống, chưa được điền ở bước này.
- `environment_report.txt` đã ghi lại (không commit, xem `.gitignore`).
- Bước tiếp theo: viết `scripts/download_checkpoints.sh` (tải 4 checkpoint Controlled Panel qua `mim download mmdet --config <name> --dest .` theo đúng tên config trong `protocol_lock.md`) + tải/chốt COCO subset n=300 ảnh (ghi `data/image_lists/`), sau đó chạy eval clean thật để điền cột "Ngày verify" trong `model_registry.md`.

---

## 2026-09-22 — Tải checkpoint + COCO val2017, verify 4/4 model Controlled Panel trên GPU thật

Tiếp tục phiên GPU cùng ngày (sau entry "Chạy `setup_env.sh` lần đầu..." ở trên — môi trường đã dựng xong trong entry đó).

- **Tải dataset**: `images.cocodataset.org` bị lỗi cert HTTPS phía server chính thức (SSL cert trả về là của `s3.amazonaws.com`, không match `images.cocodataset.org` — lỗi đã tồn tại lâu ở phía COCO, không phải do máy/mạng của mình). Workaround: tải qua `http://` thay vì `https://` (server vẫn phục vụ HTTP bình thường). Đã tải + giải nén `annotations_trainval2017.zip` (252MB) và `val2017.zip` (815MB, 5000 ảnh) vào `data/coco/` (không commit — đúng `.gitignore`). Cần cài thêm `unzip` (chưa có trong apt deps của `setup_env.sh`) — đã thêm vào script.
- **Viết `scripts/download_checkpoints.sh`** — tải 4 checkpoint Controlled Panel qua `mim download mmdet --config <identifier> --dest checkpoints/`. Phát hiện 2 lỗi, đã sửa:
  1. `mim download` báo lỗi `model-index.yml ... not found, please upgrade your mmdet` — nguyên nhân: mmdetection cài kiểu editable PEP 660 (meta path finder, không phải `.egg-link` như `setup.py develop` kiểu cũ) khiến `mim` (dùng `pkg_resources.get_distribution(...).location`) resolve sai đường dẫn cài đặt — trỏ vào `.venv/lib/.../site-packages/mmdet` (không tồn tại vật lý) thay vì `third_party/mmdetection/mmdet` (nơi code thật nằm, và nơi mình đã tự tạo symlink `.mim` → `configs/`, `tools/`, `model-index.yml` ở bước trước vì `add_mim_extension()` trong `setup.py` của mmdet chỉ chạy khi `'develop' in sys.argv`, không xảy ra với pip PEP 660). Fix: tạo thêm symlink `'.venv/lib/python3.10/site-packages/mmdet' -> 'third_party/mmdetection/mmdet'` — sau đó `mim` resolve đúng, `import mmdet` vẫn hoạt động bình thường (Python ưu tiên thư mục thật trong site-packages hơn finder do `.pth` đăng ký). **Chưa đưa fix này vào `setup_env.sh`** (làm thủ công trong phiên này) — cần thêm vào script cho lần thuê máy sau, xem mục "còn mở" cuối entry.
  2. Tên checkpoint identifier trong `protocol_lock.md`/`model_registry.md` ghi kiểu v2 cũ (`mask_rcnn_r50_fpn_mstrain-poly_3x_coco`, gạch dưới) — nhưng `mim` (model-index v3.3.0) chỉ nhận identifier **khớp đúng tên file config** (`mask-rcnn_r50_fpn_mstrain-poly_3x_coco`, gạch nối `mask-rcnn`). Đã sửa cả `scripts/download_checkpoints.sh` và 2 file docs cho khớp thực tế (xác nhận bằng cách đọc `configs/*/metafile.yml` field `Name`).
  - Checkpoint Swin-T lần tải đầu qua `mim` bị **đứt file** (94.9MB thay vì 191MB thật, load checkpoint báo `RuntimeError: unexpected EOF`) — không rõ nguyên nhân (mim không báo lỗi, exit code 0). Fix: tải lại trực tiếp từ URL trong `metafile.yml` (field `Weights`) bằng `curl`, ra đúng 191,464,891 bytes, load OK. Bài học: sau khi `mim download`, nên luôn `init_detector(...)` thử load thật trước khi tin checkpoint nguyên vẹn, không chỉ tin exit code hay tin kích thước file "trông hợp lý".
  - Cả 4 config `.py` cũng được `mim` copy vào `checkpoints/` nhưng bị thiếu 1 file (Swin) — cuối cùng quyết định **dùng thẳng config gốc trong `third_party/mmdetection/configs/.../*.py`** làm nguồn chính thức thay vì bản copy trong `checkpoints/`, để tránh 2 nguồn config lệch nhau.
- **Verify 4/4 checkpoint bằng eval clean AP thật trên full COCO val2017 (5000 ảnh)**, dùng `third_party/mmdetection/tools/test.py` — mỗi model mất ~5-7 phút trên RTX 3090. Kết quả khớp chính xác README/`model_registry.md`, không lệch:
  - Mask R-CNN R50 (surrogate): bbox AP **40.9**
  - Mask R-CNN R101 (same-family target): bbox AP **42.7**
  - Mask R-CNN ConvNeXt-T (cross-CNN target): bbox AP **46.2**
  - Mask R-CNN Swin-T (CNN→Transformer target): bbox AP **46.0**
  - Log + metrics đầy đủ (bbox + segm AP, AP50/75/S/M/L) nằm ở `results/eval_clean/{r50,r101,convnext-t,swin-t}/` — nhẹ, git-tracked được.
  - Đã cập nhật `docs/model_registry.md` (điền "Ngày verify" = 2026-09-22, thêm cột AP verify GPU thật) và `docs/protocol_lock.md` (sửa tên checkpoint identifier + ghi chú đã verify).
- **Quyết định chấp nhận, không phải lỗi chặn**: dùng `http://` thay `https://` cho `images.cocodataset.org` — vẫn xác thực nội dung đúng qua kích thước file + giải nén thành công + annotation JSON đọc được bình thường qua `pycocotools`, rủi ro MITM chấp nhận được vì đây là dataset public, không phải secret/credential.
- **Còn mở, cần làm ở phiên tiếp theo hoặc trước khi trả máy**:
  1. Đưa 2 fix vào `scripts/setup_env.sh`/`scripts/download_checkpoints.sh` cho tái lập được ở máy thuê tiếp theo: symlink `site-packages/mmdet` (mục 1 ở trên) hiện chưa nằm trong script, chỉ làm thủ công trong phiên này — **phải thêm vào `setup_env.sh` trước khi trả máy**, nếu không lần sau `mim download` sẽ lại lỗi.
  2. Chưa tạo `data/image_lists/` (danh sách cố định n=300/n=1000 image_id theo seed) — bắt buộc trước khi chạy Baseline-First Stage (idea.md §4, §8).
  3. Chưa viết code attack nào (MI-FGSM/DI-FGSM/OSFD/AugTrans port sang mmdet v3) — vẫn ở đúng giai đoạn "chưa có code" như CLAUDE.md mô tả, chỉ mới xong phần hạ tầng (env + checkpoint + dataset pool).
  4. Chưa commit + push các thay đổi trong phiên này (`scripts/setup_env.sh`, `scripts/download_checkpoints.sh`, `docs/*.md`, `results/eval_clean/`) — cần làm trước khi trả máy GPU, theo đúng quy ước trong CLAUDE.md.

---

## 2026-09-22 — Commit hạ tầng + viết bootstrap.sh gộp toàn bộ setup, chốt danh sách ảnh n=300/n=1000

Tiếp tục cùng phiên GPU (RTX 3090) — đã commit + push xong phần hạ tầng của 2 entry trước (env fix + checkpoint verify), giờ làm nốt phần còn thiếu #1 và #2 ở mục "còn mở" của entry trước.

- **Fix #1 (site-packages/mmdet symlink cho `mim download`) đã đưa vào `scripts/setup_env.sh`** (trước đó chỉ làm thủ công) — cùng với fix `mmdet/.mim` symlink, script giờ tự làm cả 2 fix khi chạy trên máy mới, không cần sửa tay nữa.
- **Viết `scripts/download_dataset.sh`** — tải + giải nén COCO val2017 (ảnh + annotations) vào `data/coco/`, idempotent (bỏ qua nếu đã có đủ 5000 ảnh / đã có annotations). Dùng lại workaround `http://` cho `images.cocodataset.org` (xem entry trước).
- **Viết `scripts/generate_image_lists.py`** — chốt danh sách ảnh n=300/n=1000 theo seed cố định. Quyết định (đã ghi vào `docs/protocol_lock.md`):
  - Pool: COCO val2017 lọc còn ảnh có ≥1 instance annotation (4952/5000 ảnh).
  - `SEED=42`, `n1000 = random.Random(42).sample(sorted(pool), 1000)`, `n300 = n1000[:300]` — **n=300 là tập con của n=1000** (quyết định có chủ đích: tránh Confirmation Stage và Final Stage dùng 2 nguồn ảnh không lồng nhau, dễ gây nhiễu khi so sánh kết quả giữa 2 phase).
  - Script có guard idempotent: nếu `data/image_lists/meta.json` đã tồn tại thì không chạy lại (danh sách coi như khóa vĩnh viễn sau lần đầu, dù script tất định 100% nên chạy lại cũng ra kết quả giống hệt — guard chỉ để tránh lẫn lộn nếu code bị sửa sau này).
  - Đã chạy thật lần đầu: `data/image_lists/n300.csv` (300 dòng), `n1000.csv` (1000 dòng), `meta.json` — đã verify n300 ⊂ n1000 bằng script kiểm tra riêng, đúng.
- **Viết `scripts/bootstrap.sh`** — gộp cả 4 bước (`setup_env.sh` → `download_checkpoints.sh` → `download_dataset.sh` → `generate_image_lists.py`) thành 1 lệnh duy nhất cho máy GPU thuê mới, đúng theo yêu cầu "thuê pod mới thì auto setup hết". Đã test từng script con trên máy hiện tại (idempotency check hoạt động đúng — bỏ qua phần đã có).
- Cập nhật `CLAUDE.md` và `docs/environment_setup.md`: entrypoint chính giờ là `bash scripts/bootstrap.sh` thay vì `scripts/setup_env.sh` đơn lẻ.
- **Việc còn lại trước khi bắt đầu code attack**: commit + push các file mới của entry này (`scripts/bootstrap.sh`, `scripts/download_dataset.sh`, `scripts/generate_image_lists.py`, `data/image_lists/{n300,n1000}.csv`, `data/image_lists/meta.json`, docs đã sửa). Sau đó mới bắt đầu port code attack (MI-FGSM/DI-FGSM/OSFD/AugTrans) từ `ref-repo/OSFD-main` sang mmdet v3 API — vẫn hoàn toàn chưa làm gì ở phần này.

---

## 2026-09-22 — Đọc kỹ ref-repo/OSFD-main/attack/, viết khung harness cho mmdet v3

Tiếp tục cùng phiên GPU (RTX 3090).

- **Đọc toàn bộ `ref-repo/OSFD-main/attack/`** để hiểu kiến trúc trước khi port. Phát hiện quan trọng:
  - Repo tách 2 loại module ghép qua registry: `base_attack` (`BASEATK`: IFGSM/MI/DI/RRB — cơ chế update noise/biến đổi input, không quan tâm loss từ đâu) và `transfer_attack` (`TSFATK`: chỉ có **OSFD** — định nghĩa loss nào dùng để lấy gradient, ở đây là feature-disruption `MSE(k·feat_clean, feat_adv)`, không cần GT).
  - `attack/comparing/__init__.py` **rỗng** — dù tên thư mục gợi ý "attack để so sánh", không có MI-FGSM/DI-FGSM task-loss chuẩn nào được đăng ký thật sự.
  - `ummdet/detectors/model_hook.py` có sẵn `ModelHook.forward_bottom()` (tính loss task thật qua GT, kiểu `forward_train` v2) nhưng **không được wire vào transfer_attack nào cả** — code chết.
  - Kết luận: repo gốc chỉ thực sự implement được OSFD. MI-FGSM/DI-FGSM baseline (task-loss chuẩn, theo idea.md §8) và AugTrans đều phải viết mới hoàn toàn, không có ref code dùng được trực tiếp.
- **Khảo sát API mmdet v3 thật** (đọc source `third_party/mmdetection/mmdet/apis/inference.py`, `mmdet/models/detectors/{base,two_stage}.py`, `mmengine/model/base_model/data_preprocessor.py`) để thiết kế harness:
  - Phát hiện quan trọng: normalize (mean/std) + pad + bgr2rgb trong v3 nằm trong `model.data_preprocessor` — một `nn.Module` thật, **không có `torch.no_grad()`/`.detach()` nào bên trong** (đã đọc source xác nhận). Nghĩa là có thể cộng noise thẳng vào tensor pixel [0,255] THÔ (trước data_preprocessor) và gradient lan truyền ngược tự nhiên — **đơn giản hơn nhiều so với v2** (OSFD phải tự denormalize/renormalize thủ công vì v2 normalize trong CPU pipeline, không khả vi).
  - Verify cả 4 config Controlled Panel dùng **chung hệt** `test_pipeline` (`Resize(scale=(1333,800), keep_ratio=True)`) và **chung hệt** `data_preprocessor` (mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], bgr_to_rgb=True, pad_size_divisor=32) — nghĩa là sinh 1 tensor pixel duy nhất cho mỗi ảnh (ở resolution đã resize theo surrogate) feed thẳng được vào cả 4 model mà không cần resize riêng noise cho từng target như v2 phải làm (`resizer = transforms.Resize(...)` trong `single_gpu_test` cũ). Đơn giản hóa đáng kể.
  - `model.loss(batch_inputs, batch_data_samples)` gọi thẳng được (định nghĩa trong `BaseDetector`/`TwoStageDetector`), không cần custom detector subclass như `MaskRCNNAdv`/`ModelHook` bên v2.
  - Test pipeline của tất cả config Controlled Panel **đã có sẵn `LoadAnnotations(with_bbox=True)`** — GT box tự động có trong `data_sample.gt_instances` khi build dataset qua registry chuẩn, không cần viết pipeline riêng.
- **Viết `attack/` package** (mmdet v3 harness, thay thế hoàn toàn `ref-repo/OSFD-main/attack/utils/mmdet.py`):
  - `attack/models.py` — `CONTROLLED_PANEL` registry (config+checkpoint path cho 4 model, khớp `protocol_lock.md`), `load_model()`/`load_surrogate()`/`load_all_targets()`.
  - `attack/data.py` — `load_image_ids()` (đọc `data/image_lists/*.csv`), `AttackDataset` (build qua `mmdet.registry.DATASETS` từ chính config surrogate, lọc+reorder theo danh sách n=300/n=1000 cố định, `serialize_data=False` để giữ `data_list` là list thường). Trả về `(img_id, inputs pixel-space chưa normalize, data_sample có gt_instances)`.
  - `attack/preprocess.py` — lớp glue khả vi: `to_batch()` (gọi `model.data_preprocessor` trực tiếp trên list tensor pixel), `compute_gt_loss()` (loss GT-assisted cho MI-FGSM/DI-FGSM sau này), `extract_features()` (nền cho OSFD), `predict()` (nền cho eval mAP).
- **Verify thật trên GPU** (`scripts/smoke_test_harness.py`, không phải attack method thật, chỉ verify harness):
  1. `AttackDataset('n300')` load đúng 300 ảnh, ảnh đầu tiên (img_id=776) có 4 GT box.
  2. Gradient lan truyền được: tạo noise=0 (`requires_grad=True`), forward qua `compute_gt_loss` → `loss.backward()` → `noise.grad` khác None, `grad_norm=0.0086` (khác 0) — xác nhận đồ thị tính toán không bị đứt qua `data_preprocessor`.
  3. `extract_features`: 5 stage FPN, shape đúng kỳ vọng cho ảnh 1196×800 (sau `Resize(1333,800,keep_ratio)` + pad `divisor=32`).
  4. `predict` trên ảnh sạch: 15 detection, top score 0.989 — model hoạt động đúng (không phải rác).
  5. Load cả 4 model qua `attack/models.py` (`load_all_targets()` cho 3 target + surrogate riêng) — không lỗi.
- Lưu ý kỹ thuật: script cần `sys.path.insert(0, REPO_ROOT)` để import được package `attack/` (không có `PYTHONPATH`/package install nào set up) — đã thêm vào `scripts/smoke_test_harness.py`, các script sau này dùng `attack/` cũng cần dòng này ở đầu.
- **Việc tiếp theo**: viết attack core thật (`attack/base/{ifgsm,mi,di}.py` port từ ref-repo — thuần tensor math, rủi ro thấp; `attack/losses/osfd.py` port OSFD; loss GT-based chuẩn cho MI-FGSM/DI-FGSM dùng `compute_gt_loss` đã có sẵn trong harness; AugTrans viết từ paper). Commit + push toàn bộ `attack/` + `scripts/smoke_test_harness.py` của entry này trước.

---

## 2026-09-22 — Viết attack core: IFGSM/MI/DI engine + OSFD loss port, verify thật trên GPU

Tiếp tục cùng phiên GPU (RTX 3090).

- **Viết `attack/methods/core.py`** — `run_iterative_attack()`: 1 hàm chung thay cho hệ registry+pipeline 4 giai đoạn của ref-repo (không cần abstraction đó vì chỉ có 3 biến thể cố định). `momentum=0` → IFGSM thuần, `momentum>0` → MI-FGSM (chuẩn hoá gradient bằng `mean(abs(.))` trước khi tích lũy, khớp `attack/base/MI.py` gốc), `diversity_fn` → DI-FGSM. Noise re-leaf mỗi step bằng `torch.autograd.grad` (không dùng `.grad`/`zero_grad` thủ công như ref-repo — an toàn hơn, tương đương về mặt toán học, đã verify bằng cách đọc kỹ luồng `@torch.no_grad()` trong `UpdateNoise.__call__` của ref-repo để xác nhận hành vi giống hệt).
- **Viết `attack/methods/diversity.py`** — port `DI.input_diversity`, **có sửa 1 lỗi trong ref-repo**: bản gốc chỉ dùng `imgs.shape[2]` (chiều cao) làm kích thước cho CẢ 2 chiều khi resize/pad → ép ảnh về hình vuông. Với ảnh classification (ImageNet crop vuông sẵn) không sao, nhưng ảnh detection sau `Resize(keep_ratio=True)` không vuông (vd COCO ảnh mẫu 1196×800) — ép vuông sẽ méo tỉ lệ nghiêm trọng, không đúng tinh thần "input diversity" (biến đổi scale nhẹ, không phá cấu trúc ảnh) của paper gốc DI-FGSM (Xie et al., CVPR'19). Đã sửa xử lý H/W độc lập, giống cách `attack/base/RRB.py`'s `adaptive_random_resizing` làm đúng trong cùng ref-repo.
- **Viết `attack/losses/osfd.py`** — port `attack/ours/OSFD.py`: cache feature sạch 1 lần (`torch.no_grad()`), loss = tổng `MSE(k·feat_clean, feat_adv)` trên từng stage. Ghi rõ trong code vì sao ASCENT (tăng loss này) vẫn đúng hướng "đẩy feature ra xa clean" dù nhìn qua tưởng cần descent: tại bước đầu adv≈clean nên `(feat_adv - k·feat_clean) ≈ (1-k)·feat_clean`, với k=3 hướng ascent xấp xỉ đẩy feat_adv theo `-2·feat_clean` — tức ra xa feat_clean thật. Nhờ vậy giữ được quy ước "toàn bộ ascent" thống nhất giữa 3 method, không cần loss riêng kiểu descent.
- **Viết `attack/methods/baselines.py`** — 3 hàm sẵn dùng khớp tên trong idea.md §8: `mi_fgsm_attack`, `di_fgsm_attack`, `osfd_attack`. Hyperparameter mặc định: epsilon=5.0 (=5/255 quy về pixel-space [0,255]), alpha=1.0, momentum=1.0 (MI, theo Dong et al. CVPR'18), prob=0.7/scale=1.1 (DI, theo `attack/base/DI.py` gốc), k=3.0 (OSFD, theo `protocol_lock.md`/`config/default.py` gốc).
  - **Quyết định CHƯA chốt, ghi rõ trong docstring của file**: `osfd_attack()` hiện chỉ ghép loss OSFD với update rule IFGSM thuần (momentum=0, không diversity) — CHƯA gồm RRB (`attack/base/RRB.py`, chưa port) hay MI, trong khi ref-repo's `base.yaml` định nghĩa "OSFD" đầy đủ = MI+RRB+OSFD-loss. Cần quyết định trước khi chạy baseline table chính thức: dùng OSFD-loss đơn lẻ (đơn giản hơn, tách bạch rõ "đóng góp của loss") hay full recipe của paper gốc (đúng fidelity nhưng cần port thêm RRB).
- **Verify thật trên GPU** (`scripts/smoke_test_attacks.py`, ảnh đầu tiên của n300, steps=20, epsilon=5.0):
  - Cả 3 method: `torch.isnan` sạch, noise cuối luôn nằm đúng trong `[-5,5]` (constraint epsilon đúng).
  - **Bài học quan trọng về cách đánh giá**: heuristic ban đầu "đếm tổng số detection giảm" SAI — cả 3 attack đều làm tổng số detection TĂNG mạnh (15→57-100), ban đầu tưởng là bug. Kiểm tra kỹ hơn bằng cách match GT-box (IoU>0.5 + đúng label) thì phát hiện: detection thật KHỚP GT vẫn còn đó nhưng confidence giảm rõ, đồng thời attack sinh rất nhiều false-positive tràn lan ở chỗ khác — đây là hiện tượng THẬT đã biết trong literature tấn công detector (maximize loss RPN/ROI khiến model "ảo giác" object khắp nơi, sập Precision/AP dù recall/count tăng), khớp đúng threat model "untargeted, random-output" của idea.md §3. Đã sửa lại `scripts/smoke_test_attacks.py` dùng đúng tín hiệu (GT-matched confidence) thay vì đếm box.
  - Kết quả (mean confidence drop trên 4 GT box, ảnh mẫu img_id=776): **MI-FGSM: -0.058, DI-FGSM: -0.188, OSFD: -0.563** (suppress hoàn toàn cả 4 GT box về confidence 0). OSFD mạnh nhất — khớp kỳ vọng literature (feature-disruption thường hiệu quả hơn task-loss trực tiếp, đặc biệt ở near white-box).
- **Việc tiếp theo**: (1) quyết định RRB có cần port cho OSFD baseline hay không (xem "chưa chốt" ở trên), (2) viết AugTrans (không có ref code, đọc từ `papers/AugTrans.pdf`), (3) viết harness đánh giá đầy đủ (loop qua n=300, sinh ảnh adv, eval AP thật qua cả 4 model Controlled Panel bằng pycocotools, ghi `results/runs/*/metrics.json` theo idea.md §8-9) — hiện mới test được 1 ảnh, chưa có baseline table thật nào. Commit + push `attack/methods/`, `attack/losses/`, `scripts/smoke_test_attacks.py` của entry này trước.

---

## 2026-09-22 — Port RRB, osfd_attack() lên full recipe (MI+RRB+OSFD-loss)

Tiếp tục cùng phiên GPU (RTX 3090) — trả lời câu hỏi "chưa chốt" ở entry trước.

- Quyết định: **"OSFD" baseline = full recipe của paper gốc** (base_attack=['MI','RRB'] + transfer_attack=OSFD, đúng theo `ref-repo/OSFD-main/config/base.yaml`), không phải OSFD-loss đơn lẻ — vì đó mới là method paper thực sự định nghĩa và report kết quả.
- **Viết `attack/methods/rrb.py`** — port `attack/base/RRB.py`: `random_axis_rotation` (xoay quanh tâm 1 GT box ngẫu nhiên có jitter, hoặc tâm ảnh), `adaptive_random_resizing` (resize+pad theo kích thước 1 GT box ngẫu nhiên, giữ H/W độc lập — không mắc lỗi ép-vuông như DI), `gaussian_blur`, và `rrb_views()` sinh đúng 2 view theo CHAIN của bản gốc (view1=rotate(adv), view2=resize(view1), cả 2 cùng blur). Sửa 1 lỗi nhỏ: điểm xoay fallback bản gốc dùng `[H//2, W//2]` như thể là (x,y) trong khi đúng ra là (cx,cy) — với ảnh không vuông (luôn đúng ở đây) làm lệch điểm xoay; đã sửa đúng thứ tự `(W//2, H//2)`.
- **Tổng quát hoá `attack/methods/core.py`**: đổi tham số `diversity_fn` (1 ảnh -> 1 ảnh) thành `views_fn` (1 ảnh -> list ảnh) để hỗ trợ RRB batch-doubling. Cách tính: gọi `loss_fn` riêng cho từng view rồi CỘNG loss lại trước 1 lần `torch.autograd.grad()` duy nhất — về mặt toán học tương đương ghép batch thật rồi backward 1 lần (autograd cộng gradient qua mọi đường), đơn giản hơn nhiều so với phải batch qua `model.data_preprocessor`, và vẫn khớp đúng định nghĩa B trong `protocol_lock.md` ("concat 2 view rồi backward 1 lần tính B=1").
- **Cập nhật `attack/methods/baselines.py`**: `osfd_attack()` giờ mặc định `use_rrb=True, momentum=1.0` (full recipe), hyperparameter RRB khớp `base.yaml` (theta=7, l_s=10, rho=0.8, s_max=1.1, sigma=6.0) chứ không phải default khiêm tốn hơn của từng class. Vẫn giữ đường tắt `use_rrb=False, momentum=0.0` cho ablation OSFD-loss đơn lẻ sau này nếu cần.
- **Bug phát hiện + sửa khi test thật**: `gt_boxes` lấy từ `data_sample.gt_instances` không tự động cùng device với ảnh — `AttackDataset` trả về `data_sample` nguyên bản từ pipeline (luôn ở CPU), chỉ tensor ảnh mới được code gọi `.to(device)` thủ công. RRB truy cập trực tiếp `gt_boxes` (khác với `attack/preprocess.py` vốn tự đồng bộ device qua `model.data_preprocessor`) nên bị lỗi `RuntimeError: ... cpu and cuda:0`. Đã sửa: `rrb_views()` tự `.to(adv_pixels.device)` cho gt_boxes.
- **Verify lại `scripts/smoke_test_attacks.py`** sau khi sửa: OSFD full recipe vẫn suppress hoàn toàn 4 GT box (confidence → 0.000 cả 4, giống kết quả OSFD-loss-đơn-lẻ trước đó). DI-FGSM vẫn giảm rõ (-0.170). MI-FGSM lần chạy này cho kết quả khác lần trước (delta≈-0.04 thay vì -0.058, tức gần như không đổi) — **không phải bug**: MI-FGSM không có randomness trong code, nhưng ROIAlign backward trên GPU (dùng trong roi_head của Mask R-CNN) vốn KHÔNG deterministic (atomicAdd trong CUDA kernel), sai số nhỏ này bị khuếch đại qua 20 bước cập nhật kiểu `sign()` (hàm bậc thang, nhạy với sai số nhỏ gần 0) → kết quả cuối khác nhau giữa các lần chạy dù cùng input/hyperparameter. Ghi chú quan trọng cho sau này: **đánh giá từng ảnh đơn lẻ có thể nhiễu**, phải luôn nhìn số liệu tổng hợp trên n=300 (khớp đúng lý do idea.md §10 dùng paired bootstrap 95% CI thay vì so sánh 1 ảnh).
- **Việc tiếp theo**: viết AugTrans (không có ref code, đọc từ `papers/AugTrans.pdf`), rồi viết harness đánh giá đầy đủ (loop n=300, sinh + lưu ảnh adv, eval AP thật qua cả 4 model bằng pycocotools, ghi `results/runs/*/metrics.json`). Commit + push `attack/methods/rrb.py`, `attack/methods/core.py`, `attack/methods/baselines.py` đã sửa của entry này trước.

---

## 2026-09-22 — Port AugTrans, phát hiện + sửa 2 bug thật, ghi nhận giới hạn còn mở

Tiếp tục cùng phiên GPU (RTX 3090). Đọc `papers/AugTrans.pdf` (Pandey et al., "AugTrans: Boosting Adversarial Transferability in Object Detection with a Dynamic, Object-Aware Augmentation Pipeline") — **không có ref code trong repo**, port trực tiếp từ Algorithm 1/2, Eq (2)-(6), bảng hyperparameter Section 4.1.

- **Viết `attack/methods/augtrans.py`** — 4 thành phần biến đổi tuần tự mỗi EOT sample (dynamic object-centric rotation với curriculum θ_max(k), multi-box aware resizing dựa trên K_obj=3 GT box lớn nhất, contextual crop/reflective-pad, composite Gaussian+salt-pepper noise) + loss đa thành phần Eq (6) (weighted, có nonlinear scaling γ=0.8 cho cls/objectness). N_EOT=10, K_max=B/10 theo đúng quy ước đã khoá trong `protocol_lock.md`.
- Ghi chú 1 câu mơ hồ trong paper (Section 3.3 mô tả pipeline cố định 4 bước, nhưng phần "Hyperparameter Configuration" lại viết "apply 2 sequential transformations, randomly sampling" — mâu thuẫn) — chọn theo Section 3.3 + Algorithm 2 (có pseudocode cụ thể) vì nhất quán hơn.
- **Bug #1 phát hiện qua verify thật (fairness alpha)**: `alpha=η=0.0004` (giá trị paper) chỉ cho `K_max*alpha` tối đa 0.51-2.04 ở `B∈{50,200}` đã khoá (paper dùng K_max=160, mình chỉ có K_max=5-20 do N_EOT=10) — noise KHÔNG BAO GIỜ chạm epsilon=5, khiến AugTrans thua thiệt giả tạo do quy đổi đơn vị B, không phải do method yếu. **Đã hỏi ý kiến, quyết định**: `alpha` tự tính = `2*epsilon/K_max` (margin 2x) thay vì dùng thẳng giá trị paper — đảm bảo AugTrans cũng tận dụng hết budget perturbation như 3 baseline kia.
- **Bug #2 phát hiện qua verify thật (box không co-transform)**: sau khi sửa alpha, `realized L_inf` đã đúng bằng epsilon nhưng confidence trên GT-matched box KHÔNG giảm (thậm chí tăng) — nghi ngờ do rotation/resize làm object DỊCH CHUYỂN trong ảnh nhưng `data_sample.gt_instances.bboxes` vẫn giữ toạ độ GỐC khi tính `model.loss()`, khiến loss tính sai vị trí hoàn toàn. Xác nhận đúng bằng thực nghiệm (đo điểm mốc qua `torchvision.transforms.functional.rotate` để suy ra đúng công thức xoay góc — quy ước: điểm lệch (dx,dy) quanh tâm chuyển thành `(dx·cosθ+dy·sinθ, -dx·sinθ+dy·cosθ)`, khác cả 2 chiều "chuẩn toán học y-lên" lẫn "trực giác visual y-xuống" nếu suy đoán không kiểm chứng). Đã sửa:
  - Viết `attack/methods/box_transforms.py` (`rotate_boxes`, `scale_shift_boxes`) dùng chung.
  - `attack/methods/augtrans.py`: `_object_centric_rotate`/`_resize_crop_pad` giờ trả về cả box đã biến đổi; `augtrans_transform` build `DetDataSample` MỚI (deepcopy + gán box mới) cho mỗi EOT view.
  - Tổng quát hoá `attack/methods/core.py`: `views_fn` đổi từ trả `List[Tensor]` sang `List[Tuple[Tensor, DetDataSample]]` — mỗi view mang data_sample riêng.
  - **Áp dụng luôn fix này cho DI-FGSM** (`attack/methods/diversity.py`, hàm mới `input_diversity_with_boxes`) — DI-FGSM ghép `input_diversity` (resize+pad) với `compute_gt_loss` (GT-based) nên mắc CÙNG LOẠI bug, dù không tự nhận ra khi viết ban đầu (ref-repo gốc không có bug này vì DI trong ref-repo chỉ từng ghép với loss OSFD, không phụ thuộc GT). Sau fix, DI-FGSM cải thiện rõ rệt (confidence drop trên ảnh mẫu: -0.046→-0.325), xác nhận bug này có thật và có ảnh hưởng đáng kể. RRB (dùng cho OSFD) KHÔNG cần fix vì loss OSFD (feature-disruption) không phụ thuộc GT.
- **Bug #3 phát hiện qua verify thật (thiếu loss_mask)**: sau khi sửa bug #2, AugTrans vẫn yếu. Debug bằng cách in từng thành phần loss trên ảnh sạch: `loss_cls=0.52, loss_bbox=0.43, loss_rpn_*≈0.001-0.03, loss_mask=1.68` — `loss_mask` chiếm ~62% tổng loss nhưng KHÔNG có trong Eq (6) của paper. Lý do: paper AugTrans dùng **Faster R-CNN** làm source model (Section 3.5.1), không có mask head, nên Eq (6) 4 số hạng đã là toàn bộ task loss của họ — nhưng surrogate của dự án này bị khoá là **Mask R-CNN** (`protocol_lock.md`, để giữ kiến trúc nhất quán Controlled Panel), có thêm mask loss lớn nhất mà Eq (6) port nguyên văn bỏ sót hoàn toàn. **Đã hỏi ý kiến, quyết định**: thêm số hạng thứ 5 `alpha_mask * loss_mask` (alpha_mask=1.0, không có exponent — cùng kiểu linear như loss_bbox/loss_rpn_bbox) vào `augtrans_loss()`, ghi rõ đây là ĐIỀU CHỈNH so với paper gốc (không phải Eq 6 nguyên văn), lý do và bằng chứng đầy đủ trong docstring.
- **Giới hạn còn mở, CHƯA giải quyết được**: sau khi sửa cả 3 vấn đề trên, AugTrans vẫn cho kết quả suppress YẾU HƠN rõ rệt so với MI-FGSM/DI-FGSM/OSFD trên ảnh mẫu (img_id=776) — kể cả khi test ở đúng K_max=160 (khớp scale paper gốc dùng, budget_B=1600) thay vì K_max=5-20 của protocol B∈{50,200}. Thử cô lập từng thành phần biến đổi (chỉ rotate / chỉ resize / chỉ noise / tổ hợp) cho kết quả KHÔNG NHẤT QUÁN (vd "chỉ noise" suppress khá tốt nhưng "full pipeline" lại yếu hơn cả — ngược trực giác) — dấu hiệu nhiễu thống kê cao trên phép đo 1-ảnh-1-lần-chạy hơn là 1 bug cụ thể còn sót (đã biết trước: ROIAlign backward không deterministic trên GPU, xem entry "Port RRB" — càng rõ hơn với AugTrans vì EOT lấy trung bình qua nhiều transform NGẪU NHIÊN mỗi step, biến thiên giữa các lần chạy vốn đã cao hơn 3 method kia).
  - **Quyết định (đã hỏi ý kiến)**: KHÔNG debug sâu thêm dựa trên 1 ảnh — khớp đúng lý do idea.md dùng paired bootstrap 95% CI trên n=300 thay vì tin số liệu từng ảnh đơn lẻ (idea.md §10). Code AugTrans coi như ĐÃ XONG về mặt kỹ thuật (không crash, gradient chảy đúng, epsilon đúng, 3 bug thật đã tìm và sửa có bằng chứng rõ ràng) nhưng **hiệu quả tương đối so với 3 baseline kia CHƯA kết luận được cho tới khi chạy thật trên n=300** và nhìn số liệu tổng hợp.
  - Nếu khi chạy Baseline-First Stage thật (n=300) mà AugTrans vẫn yếu hơn hẳn 1 cách nhất quán (không phải nhiễu ảnh đơn lẻ nữa), cần quay lại điều tra thêm — nghi vấn hàng đầu tiếp theo (chưa kiểm chứng): công thức content-adaptive resize (Eq 3-4) có thể tạo scale factor khá lớn (lên tới ~1.5-1.65x) với ảnh có object to, kết hợp crop ngẫu nhiên có thể đẩy phần lớn object ra ngoài vùng crop — cần trực quan hoá vài EOT view thật (lưu ảnh ra xem) để kiểm tra thay vì chỉ suy luận từ số liệu.

---

## 2026-09-23 — Sửa bug GT lệch khung ảnh (LoadAnnotations sau Resize) + co-transform GT mask

Phiên GPU mới (RTX 3090). Phát hiện khi đọc lại code, verify thật trên GPU.

- **Bug nghiêm trọng (ảnh hưởng MI-FGSM, DI-FGSM, AugTrans — mọi loss dựa trên GT):** test pipeline của config mmdet đặt `LoadAnnotations` **sau** `Resize` (vì lúc eval GT chỉ đọc từ file annotation, ở tọa độ gốc). `AttackDataset` dùng nguyên pipeline này nên `inputs` đã resize (vd 1196×800) nhưng `gt_instances.bboxes/masks` vẫn ở tọa độ ảnh **gốc** (vd 640×428, scale ~1.87) — box chỉ phủ ~góc trên-trái ảnh. Mọi `model.loss()` từ trước tới nay tính trên vị trí sai. OSFD không bị trực tiếp (loss feature), RRB chỉ lệch nhẹ (tâm xoay/scale theo box sai tỉ lệ).
  - Fix: `attack/data.py::_gt_before_resize()` đưa `LoadAnnotations` lên trước `Resize` (Resize tự co-transform box+mask) và bật `poly2mask=True` (mask dạng BitmapMasks, xem dưới). Eval không đổi: `predict(rescale=True)` vẫn dùng `scale_factor` về tọa độ gốc.
  - Hệ quả: các kết luận 1-ảnh trong entry trước ("MI-FGSM gần như không giảm confidence", "AugTrans yếu nghi do nhiễu") được đo với GT lệch — **không còn giá trị**.
- **GT mask chưa co-transform trong DI/AugTrans** (entry trước chỉ co-transform box): `loss_mask` dùng `gt_masks`, sau xoay/resize view thì mask vẫn ở vị trí cũ. Fix: `box_transforms.get_gt()/with_gt()` — mask lấy dạng tensor [N,H,W], đi qua CHÍNH phép biến đổi tensor của ảnh (rotate/interpolate/crop), riêng pad dùng 0 thay vì reflect (vùng reflect không có GT box tương ứng); threshold 0.5 rồi gói lại `BitmapMasks`.
- **Verify thật** (`scripts/smoke_test_attacks.py`, viết lại: 3 ảnh đầu n300, steps=20, eps=5, AugTrans B=50; so khớp trong khung ảnh đã resize, `predict(rescale=False)`):
  - GT frame: box max khớp kích thước ảnh resize, IoU(bbox của mask, box) 0.93–1.00, detection sạch khớp GT 75–100%.
  - Co-transform: IoU(bbox mask, box) sau biến đổi DI 0.93–1.00, AugTrans 0.84–0.92 (thấp hơn do box xoay lấy AABB — lỏng hơn mask, đúng kỳ vọng).
  - Mean GT-confidence drop (white-box surrogate): MI-FGSM **+0.716**, DI-FGSM **+0.716**, OSFD **+0.716** (cả 3 suppress 100% GT cả 3 ảnh), AugTrans **+0.503** (+0.110/+0.455/+0.943 — yếu hơn ở white-box với K_max=5 là dự kiến, EOT làm mịn gradient; quan trọng là transfer, chưa đo).
  - A/B tái tạo bug cũ (GT về tọa độ gốc): MI-FGSM **−0.099** (confidence còn TĂNG), DI-FGSM +0.163 — xác nhận bug có thật và ảnh hưởng rất lớn.
  - GT loss ảnh sạch (img 776): 0.79 (trước ~2.7 khi GT lệch). Loss breakdown trung bình 20 ảnh sạch: cls 0.20, bbox 0.24, rpn_cls 0.03, rpn_bbox 0.06, mask 0.29 (~36%, không phải ~62% như đo khi GT lệch) — lý do thêm `loss_mask` vào AugTrans vẫn đứng (thành phần lớn nhất), đã sửa số liệu trong docstring.
- **Còn mở:**
  1. Định nghĩa B chưa nhất quán: RRB 2 view/step tính B=1, AugTrans 10 view/step (cộng loss, 1 lần `autograd.grad`) lại tính B=10. `protocol_lock.md` ("B = số lần `.backward()`") và code AugTrans đang theo 2 cách hiểu khác nhau — cần chốt đơn vị (số backward vs số view forward-backward qua surrogate) trước khi chạy baseline table.
  2. Chưa có harness eval AP n=300 × 4 model.

---

## 2026-09-23 — Quick transfer check sơ bộ n=30 (không phải baseline table chính thức)

`scripts/quick_transfer_check.py 30 50`: 30 ảnh đầu n300, surrogate R50, eps=5, B=50 (MI/DI/OSFD 50 step; AugTrans K_max=5×N_EOT=10 — định nghĩa B vẫn chưa chốt), bbox AP pycocotools giới hạn 30 ảnh. Kết quả: `results/quick_transfer/n30_B50_20260923_055238.json`.

| Relative AP drop | R50 (white-box) | R101 | ConvNeXt-T | Swin-T |
|---|---|---|---|---|
| MI-FGSM | 100.0% | 81.2% | 56.4% | 38.6% |
| DI-FGSM | 100.0% | 92.2% | 73.6% | 55.2% |
| OSFD | 99.9% | 98.1% | 84.5% | 78.9% |
| AugTrans (tự cài lại) | 89.0% | 65.8% | 47.5% | 44.2% |

- Pattern same-family > cross-CNN > CNN→Transformer xuất hiện ở cả 4 method (AugTrans: ConvNeXt≈Swin). OSFD KHÔNG bão hòa ở cross-family (~15-20 điểm gap) — còn chỗ cho RQ1/RQ3.
- AugTrans (bản tự cài lại) yếu nhất, dưới cả DI-FGSM ở mọi target — trái với claim của paper gốc; cùng với việc code gốc không tồn tại (repo công bố trong paper trả 404 ngày 2026-09-23) → rủi ro cài đặt sai cao.
- Chỉ là tín hiệu sơ bộ: n=30, chưa có CI, không dùng để tune bất kỳ hyperparameter nào.

---

## 2026-09-23 — Bỏ AugTrans khỏi plan, rà MI/DI, phát hiện vấn đề đánh giá "tensor vs ảnh thật"

- **AugTrans tạm bỏ khỏi plan** (quyết định của user): không có code chính thức (repo công bố trong paper `github.com/sudhirpandey243/LLM-Model` trả 404), mô tả mâu thuẫn, bản tự cài lại lệch paper nhiều chỗ và yếu hơn DI-FGSM ở quick check n=30. Đã sửa idea.md §8/§12, protocol_lock.md (mục B), smoke test; `attack/methods/augtrans.py` giữ lại, có ghi chú. OSFD là strong baseline chính — benchmark 2602.16494 chỉ chọn OSFD/EBAD/CAA/PhantomSponges, trong đó chỉ OSFD khớp threat model (1 surrogate, image-specific, không query). **Còn mở**: tiêu chí idea.md §10.4 ("≥2 strong baselines") — đã đánh dấu, chưa sửa.
- **Rà MI-FGSM/DI-FGSM — không thấy bug thuật toán**: update MI khớp Dong et al. + `ref-repo/.../MI.py` (grad chuẩn hoá L1-mean, μ=1, sign step, clip ε); DI khớp `DI.py` (resize+pad, sửa ép vuông, co-transform box+mask đã verify IoU 0.93–1.00); loss GT = tổng 5 loss (rpn_cls, rpn_bbox, cls, bbox, mask) với GT đúng khung; gradient qua data_preprocessor OK, preprocessor không sửa data_sample gốc (cast_data tạo bản sao). Loss có ngẫu nhiên do RandomSampler (RPN 256 anchor, RCNN 512 RoI) của train_cfg — chuẩn với attack dùng train loss, cần seed để tái lập.
  - **Lựa chọn chưa chốt (không phải bug)**: (a) DI prob=0.7 (default class trong ref) vs 1.0 (`ref-repo/config/default.py`); (b) DI hiện KHÔNG có momentum (DI²-FGSM thuần) trong khi OSFD có MI — cân nhắc M-DI²-FGSM (MI+DI) cho so sánh công bằng hơn; (c) α=1 pixel cho mọi method (ref default), khác α=ε/T của MI-FGSM gốc — nhất quán giữa các method nên chấp nhận được.
- **Vấn đề đánh giá lớn (ảnh hưởng MỌI method)**: harness đang đưa adv dạng tensor float ở khung ảnh ĐÃ resize (vd 1196×800, upsample ~1.87× từ 640×428) thẳng vào target. Ảnh adversarial thực tế phải là ảnh ở kích thước gốc, uint8, rồi mỗi target tự resize (đúng lý do benchmark loại AFOG). Thử nghiệm: `to_image_space()` trong `scripts/quick_transfer_check.py` (thu delta về ori_shape bằng INTER_AREA, cộng vào ảnh gốc uint8, làm tròn, rồi Resize lại như pipeline; round-trip ảnh sạch khớp tuyệt đối). Nhiễu ngẫu nhiên ±5 ở khung upsample chỉ còn |δ| trung bình ~1.2–1.7 sau khi về ảnh thật.
  - Kết quả n=5, B=50, eps=5 (`results/quick_transfer/n5_B50_20260923_060441.json`), relative AP drop R50/R101/ConvNeXt-T/Swin-T:
    - tensor: MI 100/74.8/47.9/45.3 · DI 99.9/90.5/73.6/57.1 · OSFD 100/100/89.1/88.3
    - **ảnh thật**: MI 78.5/42.1/26.7/21.2 · DI 85.6/43.3/40.0/33.3 · OSFD 100/78.3/55.7/56.6
  - → Đánh giá dạng tensor thổi phồng transfer rất nhiều (OSFD cross-family ~89% → ~56%). Số n=30 ở entry trước (đánh giá dạng tensor) phải coi là KHÔNG hợp lệ cho kết luận. Gap cùng họ > khác họ vẫn thấy ở cả 2 cách (n=5, rất nhiễu).
- **Việc tiếp theo (đề xuất, chưa làm)**: sửa harness để tối ưu perturbation ngay ở không gian ảnh gốc (δ ở ori_shape, resize khả vi bên trong vòng lặp lên kích thước model) + đánh giá luôn qua ảnh uint8 — như vậy perturbation không mất khi lưu ảnh; rồi chạy lại quick check n=30. Sau đó chốt: đơn vị B (OSFD/RRB 2 view/step), DI có momentum không, §10.4.

---

## 2026-09-23 — Sửa harness: tối ưu δ ở không gian ảnh gốc + OSFD dùng feature backbone; quick check n=30 lại

Phiên GPU (RTX 3090). Rà lại toàn bộ harness, so từng dòng với `ref-repo/OSFD-main`, verify bằng số trên GPU.

- **Bug 1 — δ tối ưu ở khung đã resize (ảnh hưởng mọi method):** trước đây δ nằm trên tensor đã upsample (~1.87×, vd 640×428 → 1196×800); ảnh adversarial thật (cỡ gốc, uint8) mất phần lớn δ — đo: nhiễu sign ±5 chỉ còn mean|δ| **1.79** sau INTER_AREA về cỡ gốc (mất ~64%). Số `@img` n=5 ở entry trước là CẬN DƯỚI (attack không tối ưu cho ảnh thật), số tensor là thổi phồng — cả hai không dùng được.
  - Fix: `AttackDataset` trả thêm `orig_inputs` (ảnh gốc BGR, `mmcv.imread`); `core.run_iterative_attack` nhận ảnh gốc, δ cùng shape ảnh gốc, mỗi step `resize_to_model(clamp(orig+δ))` (F.interpolate bilinear, align_corners=False, khả vi) rồi mới vào surrogate. Đánh giá: `to_adv_image` = round/clamp uint8 cỡ gốc → `pipeline_resize` (mmcv.imresize cv2 — ĐÚNG Resize của test pipeline) → target.
  - Đã đo: resize torch vs cv2 pipeline lệch tối đa 1 mức xám, trung bình ~0.12 (fixed-point cv2) trên 5 ảnh — gradient đúng không gian. `pipeline_resize(orig) == inputs` tuyệt đối (assert trong smoke test + quick check). L_inf thực tế trên ảnh uint8 = 5.000.
- **Bug 2 — OSFD lấy feature sai tầng:** bản gốc dùng `model.backbone(...)` (ResNet C2–C5, 256/512/1024/2048 kênh — `pipelines.py:50`, `TransferAttack.py:18`), bản port dùng `model.extract_feat` (backbone + FPN, 5 mức 256 kênh). Fix: `preprocess.extract_features` trả `model.backbone(...)`.
- Ghi chú đính chính: `['MI','RRB']` của OSFD nằm trong `config/attack_*.yaml` (config tác giả chạy paper), không phải `base.yaml` (`base.yaml` là `['IFGSM','MI','DI','RRB']`) — code chọn đúng, chỉ entry cũ ghi sai nguồn.
- Lệch nhỏ còn lại so với ref (chưa sửa, ghi nhận): noise init = 0 (ref: randint [−2,2]); DI prob 0.7 (ref default.py: 1.0).
- Smoke test (3 ảnh, 20 step, đánh giá qua ảnh uint8): GT-conf drop surrogate MI +0.695, DI +0.683, OSFD +0.713; A/B GT lệch cũ vẫn ≈0 (đúng kỳ vọng).
- **Quick check n=30, B=50, eps=5, đánh giá ảnh thật** (`results/quick_transfer/n30_B50_20260923_122751.json`), relative AP drop R50 / R101 / ConvNeXt-T / Swin-T:
  - MI-FGSM: 100.0 / 76.6 / 57.2 / 42.8
  - DI-FGSM: 100.0 / 88.4 / 70.2 / 56.2
  - OSFD: 99.4 / 95.3 / 79.7 / 77.7
  - Pattern cùng họ > khác họ giữ nguyên ở cả 3 method; gap OSFD R101→cross-family ~16–18 điểm. Vẫn chỉ là tín hiệu sơ bộ (n=30, chưa CI), không tune gì dựa trên số này.
  - Runtime ~3–3.5 s/ảnh/method ở B=50.
- Cài thêm `tmux` vào apt deps của `scripts/setup_env.sh` (chạy n=300/1000 không bị ngắt khi mất kết nối).
- **Việc tiếp theo:** chốt đơn vị B (RRB 2 view/step), DI có momentum không + prob, §10.4; viết harness eval n=300 × 4 model ghi `results/runs/*/metrics.json` (có paired bootstrap CI).

---

## 2026-09-23 — Đường bão hòa OSFD theo B (n=30)

`scripts/budget_sweep.py 30 200`: chạy OSFD 1 lần B=200, chụp noise tại các mốc B (callback `on_step` mới trong `core.run_iterative_attack`), đánh giá ảnh thật trên 4 model. Kết quả: `results/budget_sweep/osfd_n30_Bmax200_20260923_124034.json`. Relative AP drop (%) R50 / R101 / ConvNeXt-T / Swin-T:

| B | R50 | R101 | ConvNeXt-T | Swin-T | cross-avg |
|---|---|---|---|---|---|
| 10 | 92.2 | 74.6 | 52.7 | 52.0 | 52.3 |
| 20 | 97.9 | 90.2 | 72.8 | 67.8 | 70.3 |
| 30 | 99.2 | 92.6 | 78.7 | 69.0 | 73.9 |
| 50 | 99.4 | 95.3 | 79.6 | 78.1 | 78.8 |
| 100 | 99.6 | 97.0 | 83.5 | 79.3 | 81.4 |
| 200 | 99.8 | 97.5 | 85.2 | 80.1 | 82.7 |

- OSFD gần bão hòa từ B≈50: B50→B200 chỉ thêm ~4 điểm cross-avg (~2 điểm R101), mức chênh cỡ nhiễu n=30. Mốc B=50 khớp quick check trước (79.7/77.7) — sanity OK.
- Gap R101 − cross-family ổn định ~15–17 điểm ở mọi B ≥ 20 → thêm budget KHÔNG thu hẹp gap (tốt cho RQ1).
- Runtime OSFD B=200: 12.1 s/ảnh → n=300 ≈ 1 h/method.
- Chưa đổi bộ B đã khóa ({50, 200}) — chỉ là số liệu lập kế hoạch; nếu đổi phải ghi quyết định vào protocol_lock.md.

---

## 2026-09-23 — Chốt các điểm mở trước baseline table

Quyết định của user:
- **Budget:** B=50 là budget chính cho Confirmation Stage @300; B=200 chạy 1 lần cho bảng cuối (dựa trên sweep bão hòa ở entry trước).
- **Đơn vị B = số backward** (OSFD/RRB 2 view/step vẫn tính B=1, đúng recipe paper); bảng kết quả báo cáo thêm runtime + số view forward.
- **DI-FGSM = M-DI²-FGSM** (MI μ=1.0 + DI), p=1.0 theo ref `config/default.py` — `di_fgsm_attack` đã sửa default (momentum=1.0, prob=1.0).
- **idea.md §10.4** định nghĩa lại: gap phải xuất hiện ở cả OSFD và M-DI²-FGSM; §10.5 giữ nguyên.
- Đã cập nhật idea.md §8/§10, protocol_lock.md (mục B + bảng hyperparameter baseline), CLAUDE.md (giai đoạn hiện tại).
- Việc tiếp theo: harness eval n=300 × 4 model (`results/runs/*/metrics.json`, paired bootstrap 95% CI), chạy baseline table B=50.

---

## 2026-09-23 — Baseline table chính thức n=300, B=50, ε=5 (Controlled Panel)

Harness mới: `attack/evaluation.py` (COCO AP có trọng số ảnh cho paired bootstrap — tự kiểm khớp pycocotools |Δ|<1e-9 mỗi lần chạy) + `scripts/run_baselines.py` (attack → PNG uint8 cỡ gốc ở `artifacts/runs/<run>/adv/`, 527 MB, gitignored, dùng lại được cho Generalization Panel; eval đọc PNG từ đĩa; resume được). Kết quả: `results/runs/n300_B50_eps5/metrics.json` (+ `attack_stats.jsonl`). Bootstrap 1000 mẫu, resample ảnh, paired giữa mọi điều kiện/model.

Relative AP drop % [95% CI], R50 (white-box) / R101 / ConvNeXt-T / Swin-T:
- MI-FGSM: 99.9 / 77.3 [71.6, 79.7] / 54.0 [47.2, 56.0] / 45.1 [39.3, 47.9]
- M-DI²-FGSM: 99.7 / 93.0 [89.7, 94.7] / 78.0 [72.7, 80.3] / 67.1 [61.5, 69.7]
- OSFD: 98.9 / 92.4 [88.4, 95.4] / 80.4 [75.7, 83.6] / 78.7 [73.9, 81.2]

TransferGap (drop R101 − drop target), điểm [95% CI], p(gap≤0) = 0 ở mọi ô:
- MI-FGSM: →ConvNeXt 23.3 [20.6, 28.1], →Swin 32.1 [28.3, 36.4], →avg 27.7
- M-DI²-FGSM: →ConvNeXt 15.0 [12.5, 18.8], →Swin 25.8 [22.4, 30.8], →avg 20.4
- OSFD: →ConvNeXt 12.0 [9.4, 15.9], →Swin 13.8 [11.0, 18.1], →avg 12.9 [10.2, 16.7]

CrossAvg: MI 49.6, M-DI² 72.5, OSFD 79.5. OSFD − M-DI² = +7.0 [3.9, 11.3]. Clean AP trên n300: 45.0 / 46.2 / 51.0 / 49.8. L_inf = 5 (uint8) mọi ảnh; runtime ~3.1–3.6 s/ảnh.

Đối chiếu Hypothesis Pass (idea.md §10): (1) same-family > cross-family ở cả 3 method, cả 2 target ✓; (3) CI của gap > 0 ở mọi ô ✓; (4) có ở OSFD và M-DI²-FGSM ✓; (5) không chỉ MI ✓; (2) "effect size đủ meaningful" chưa định lượng trong idea.md — gap nhỏ nhất 12.0 điểm (OSFD→ConvNeXt). **Chưa tuyên bố PASS chính thức** — chờ user xác nhận tiêu chí (2).

Quan sát (cho Mechanism Stage, chưa kết luận):
- Gap co lại khi attack mạnh hơn (MI 27.7 → M-DI² 20.4 → OSFD 12.9) — một phần có thể do trần: R101 đã ~92–93% với M-DI²/OSFD nên gap (hiệu số drop) bị nén; cân nhắc thêm chỉ số ít bị trần (vd tỉ số AP còn lại adv/clean, hoặc ε=3) nếu cần.
- Với MI/M-DI², Swin khó hơn ConvNeXt rõ (~9–11 điểm); với OSFD hai target khác họ gần bằng nhau (80.4 vs 78.7).
- Lưu ý thống kê: bootstrap mean của AP lệch lên ~1.2 điểm so với điểm gốc (category hiếm rơi khỏi mẫu → trung bình category đổi), CI percentile hơi lệch phải; với gap (hiệu số, paired) lệch chỉ +0.2–0.8 — không đổi kết luận. Nếu cần chặt hơn: BCa hoặc cố định tập category.
- Chưa tính: ASR, APloc, CSR, LPIPS (cần định nghĩa/cài `lpips`).

---

## 2026-09-23 — Chốt tiêu chí 2 của Hypothesis Pass → PASS @300

- **Quyết định (user):** tiêu chí 2 idea.md §10 ("effect size đủ meaningful") = **cận dưới 95% CI paired bootstrap của gap ≥ 5 điểm relative AP drop**, cho từng target khác họ (ConvNeXt-T, Swin-T), ở cả OSFD và M-DI²-FGSM.
- **Minh bạch:** ngưỡng chốt SAU khi đã thấy bảng n=300 (idea.md trước đó không định lượng). Đã chọn phương án chặt hơn "CI > 0" (tiêu chí 3) và chặt hơn "điểm ước lượng ≥ 5"; phương án chặt nhất đưa ra (cận dưới ≥ 10) sẽ fail đúng 1 ô (OSFD→ConvNeXt 9.4). Phải nêu điều này nếu báo cáo tiêu chí trong paper.
- Đối chiếu `results/runs/n300_B50_eps5/metrics.json`: cận dưới CI gap — M-DI²-FGSM →ConvNeXt 12.5, →Swin 22.4; OSFD →ConvNeXt 9.4, →Swin 11.0 — đều ≥ 5.
- **Hypothesis Pass @300: PASS** (tiêu chí 1–5 đều đạt, xem entry baseline table). Được phép chuyển sang Mechanism Stage (idea.md §11).

---

## 2026-09-23 — Duyệt plan Mechanism Stage

- `docs/mechanism_plan.md` APPROVED sau chỉnh của user: thứ tự chạy **A2 → A1 → A3 → A4 → A5** (A2 trước vì ảnh adv B=50 chỉ còn trên máy); A3 (can thiệp) là thí nghiệm quyết định cho stage-aware, nặng ký hơn A2 (quan sát); tiêu chí A2 không khóa dấu tuyệt đối D_k(R101) > D_k(cross) mà khóa "tách nhau theo stage, nhất quán với thứ tự transfer, reproduce trên OSFD + M-DI²"; ceiling: chưa chạy ε=3, báo cáo song song chỉ số ít bão hòa (retained confidence, adv AP/clean AP), chỉ chạy ε=3 nếu kết luận đổi theo chỉ số.

---

## 2026-09-23 — Mechanism A2: feature similarity + lan truyền distortion (n=300)

`scripts/mech_a2_features.py` (+ `attack/mechanism.py`: linear CKA — tự kiểm self=1, bất biến trực giao/scale, độc lập≈0.04; chỉ số transfer per-ảnh). Kết quả: `results/mechanism/a2_features.json`, `a2_per_image.json` (per-ảnh transfer dùng lại cho A1/A3/A4). Ảnh adv = đúng PNG của run `n300_B50_eps5`.

**(a) CKA(R50, target), ảnh sạch, stage 1→4:** R101 0.88 / 0.85 / 0.81 / 0.86; ConvNeXt 0.46 / 0.58 / 0.57 / 0.75; Swin 0.66 / 0.68 / 0.63 / 0.60 (CI rộng ~±0.005). Cùng họ giống R50 hơn hẳn ở MỌI stage. Thứ tự ConvNeXt vs Swin đảo theo độ sâu: stage 1 Swin > ConvNeXt, stage 4 ConvNeXt > Swin — khớp thứ tự transfer (ConvNeXt dễ hơn Swin với MI/M-DI²).

**(b) Distortion D_k (mean), stage 1→4:**
- R101 khuếch đại distortion theo độ sâu (OSFD 0.22→0.70→1.18→1.31; M-DI² 0.22→0.60→0.85→0.94).
- ConvNeXt bị distortion LỚN HƠN R101 ở stage 1 (0.46–0.48) rồi giảm ở stage 2 (0.31–0.41), tăng lại chậm; Swin tăng tới stage 3 rồi giảm ở stage 4.
- Separation D_k(R101) − D_k(cross): **âm ở stage 1** (−0.12 đến −0.26), **dương từ stage 2 và lớn dần** (stage 4: 0.37–0.59); separation_k − separation_1 có CI > 0 ở mọi ô, cả 3 method → **k* = 2, reproduce trên OSFD và M-DI²**.
- Tương quan per-ảnh D_k với transfer: Spearman(D_k, retained) âm, CI loại trừ 0 ở mọi (method, target, stage) (|ρ| 0.12–0.58, mạnh nhất stage 4 ở target khác họ). Suppression cho cùng kết luận (dấu ngược, CI loại 0 mọi ô) → **không kích hoạt ceiling check ε=3**.

**Đối chiếu tiêu chí A2 (khóa trong mechanism_plan.md):**
- ✓ Tách theo stage: k* = 2, trước đó (stage 1) không tách theo chiều R101 (thậm chí ngược), CI loại 0, reproduce OSFD + M-DI².
- ✓ Nhất quán thứ tự theo TARGET ở stage 4 (separation Swin > ConvNeXt, khớp Swin khó hơn); ✗ ở stage 2–3 (ConvNeXt > Swin, ngược thứ tự transfer).
- ✗ Nhất quán theo METHOD với hiệu số tuyệt đối (chỉ số đã khóa): gap MI 27.7 > M-DI² 20.4 > OSFD 12.9 nhưng separation stage 4 OSFD LỚN NHẤT (0.48/0.59) > M-DI² (0.37/0.49) ≈ MI (0.37/0.46) — hiệu số tuyệt đối lớn lên theo độ mạnh attack.
- **Kết luận A2 theo tiêu chí khóa: ĐẠT MỘT PHẦN — chưa đủ điều kiện "có bằng chứng".** Có bằng chứng quan sát rõ rằng divergence nảy sinh theo độ sâu (từ stage 2: R101 khuếch đại distortion, target khác họ suy giảm/không khuếch đại), nhưng điều kiện nhất quán độ lớn theo method không đạt với chỉ số đã khóa.
- **Exploratory (post-hoc, KHÔNG dùng để kết luận):** tỉ số D_4(cross)/D_4(R101) — ConvNeXt MI 0.54 / M-DI² 0.60 / OSFD 0.63; Swin 0.44 / 0.48 / 0.55 — thứ tự khớp gap theo method. Chỉ số chuẩn hóa scale này chỉ được nghĩ ra sau khi thấy dữ liệu → ghi nhận làm giả thuyết cho A3, không đổi tiêu chí A2.
- Theo quy tắc, A3 (can thiệp) là thí nghiệm quyết định cho stage-aware; A2 không chặn.

---

## 2026-09-23 — Mechanism A1: input-gradient alignment (n=300)

`scripts/mech_a1_gradients.py` (tiêu chí vận hành chốt + commit TRƯỚC khi chạy, `27ccccd`: phải đạt với CẢ cosine VÀ sign agreement). Kết quả: `results/mechanism/a1_gradients.json` (+ `_per_image.json`). Gradient ∇ₓ L_task ở không gian ảnh gốc, ảnh sạch.

- **Alignment với R50** (mean [CI]): cosine R101 0.201 / ConvNeXt 0.142 / Swin 0.109 (trần nhiễu R50 khác seed 0.973); sign agreement 0.5205 / 0.5097 / 0.5066 (trần 0.868, ngẫu nhiên 0.5).
- **R101 − cross, CI > 0 ở cả 2 chỉ số, cả 2 target** (cosine +0.059 / +0.092; sign +0.011 / +0.014). Thứ tự R101 > ConvNeXt > Swin khớp đúng thứ tự transfer của MI/M-DI².
- **Tương quan per-ảnh alignment ↔ suppression (target khác họ):**
  - sign agreement: dương, CI > 0 ở cả MI, M-DI², OSFD (ρ 0.14–0.25) ✓.
  - cosine: M-DI² chỉ Swin (0.136) ✓; **OSFD ngược chiều**: ConvNeXt −0.212 [−0.320, −0.091], Swin −0.083 (CI qua 0) ✗.
- Retained confidence cho cùng kết luận với suppression (dấu ngược, cùng ô đạt/trượt ở mức tiêu chí) → không kích hoạt ceiling check.
- **Kết luận theo tiêu chí khóa: A1 KHÔNG ĐẠT** (cosine trượt điều kiện tương quan per-ảnh với OSFD). Đạt một phần: khác biệt alignment cùng họ vs khác họ rõ và đúng thứ tự ở cả 2 chỉ số; liên hệ per-ảnh giữ với sign agreement ở cả 3 method.
- Ghi chú diễn giải (không đổi kết luận): OSFD tối ưu loss feature chứ không phải task loss, nên alignment của task-gradient không phải cơ chế trực tiếp của nó; cosine bị chi phối bởi vài pixel gradient lớn, sign agreement đếm đều mọi pixel. Mức sign agreement chỉ trên ngẫu nhiên 1–2 điểm % dù transfer chênh lớn → gợi ý khác biệt không nằm ở hướng gradient pixel mà ở cách mạng khuếch đại/triệt tiêu nhiễu theo độ sâu (khớp A2).
- Đang chạy mở rộng quỹ đạo (`--trajectory`, alignment tại step 10/25/50 của M-DI² và OSFD, mô tả, không thuộc tiêu chí).

---

## 2026-09-23 — Chuẩn bị Generalization Panel + chốt quy tắc khả thi (trước khi có kết quả)

- User quyết định: chạy Generalization Panel ngay để đánh giá chỗ trống của OSFD trước khi đầu tư method mới (có thể không phải hướng stage-aware).
- **Quy tắc quyết định (chốt TRƯỚC khi có kết quả):** headroom = drop OSFD @R101 − drop OSFD @model; ĐI TIẾP nếu cận dưới CI ≥ 10 ở ≥2/4 model gồm DINO-Swin-L; DỪNG/cân nhắc nếu điểm < 5 ở ≥3/4 model; giữa = vùng xám. Chi tiết: protocol_lock.md.
- YOLOX: S chính, L phụ (không tính vào quy tắc).
- Đã tải checkpoint (`scripts/download_checkpoints.sh --gen`; YOLOX mim id là `yolox_{s,l}_8x8_300e_coco`, khác tên file config). Chưa verify clean AP full val2017 cho 5 model này (model_registry.md vẫn trống ngày verify).
- `scripts/eval_generalization.py`: dùng lại PNG adv của `n300_B50_eps5`, `inference_detector` trên file (pipeline riêng từng model — YOLOX Resize 640+Pad 114, FCOS caffe normalize); đã thử YOLOX-S trên CPU: ảnh sạch/adv chạy đúng.
- **Sự cố GPU (~14:55):** tiến trình mới báo `No CUDA GPUs are available` / `nvidia-smi: Failed to initialize NVML: Unknown Error` dù `/dev/nvidia*` còn; tiến trình đã mở GPU trước đó (A1 trajectory) vẫn chạy. Dấu hiệu container mất quyền device cgroup phía host → cần restart container/pod từ nhà cung cấp. **Ảnh adv (`artifacts/`, 527 MB) chưa lưu ra ngoài** — nếu restart không giữ ổ /workspace thì mất.

---

## 2026-09-23 — A1 trajectory + lưu artifacts lên HF

- **A1 mở rộng quỹ đạo** (`results/mechanism/a1_gradients_traj.json`, mô tả, không thuộc tiêu chí): sign agreement R50 vs target tại step 10/25/50 — M-DI²: R101 0.515→0.512, ConvNeXt 0.508→0.506, Swin 0.505→0.504; OSFD: R101 0.525→0.522, ConvNeXt 0.512→0.510, Swin 0.508→0.506 (ảnh sạch: 0.5205 / 0.5097 / 0.5066). Thứ tự R101 > ConvNeXt > Swin giữ ở mọi step; alignment giảm nhẹ dần theo quỹ đạo, khoảng cách cùng họ − khác họ gần như không đổi → không thấy divergence về hướng gradient tăng dần khi attack "chui sâu" vào surrogate. Phần chính chạy lại trong cùng lần → A1_pass = False như trước.
- **Artifacts** `n300_B50_eps5` (900 PNG + dets.json, tar 552 MB, sha256 9eadf774…f8d590) đã upload lên HF dataset `congdanh99/transfer-attack`, repo chuyển **private** trước khi upload (theo quyết định user). Đã verify tải lại khớp sha256, đủ 900 PNG. Script `scripts/sync_artifacts.py` (upload/download, token qua env `HF_TOKEN`, từ chối nếu repo public); `huggingface_hub<1.0` thêm vào `setup_env.sh` (dry-run: không đổi gói đã pin).

---

## 2026-09-23 — Trước khi restart pod (sự cố GPU/NVML) — VIỆC CẦN LÀM NGAY SAU RESTART

Restart pod xóa ổ container (`/`), giữ volume `/workspace` (xfs riêng) → repo, `.venv`, `third_party/`, checkpoint (cả Generalization Panel), `data/coco`, `artifacts/` còn nguyên. Mất: gói apt, `/root` (memory Claude, session), `/tmp`. `.venv` trỏ `/usr/bin/python3.10` của image gốc (Ubuntu 22.04, python3.10-minimal) → dự kiến vẫn dùng được.

**KHÔNG chạy lại `scripts/setup_env.sh`** (nó xóa + dựng lại venv, mất ~20 phút). Chỉ cần:
1. `apt-get update && apt-get install -y tmux unzip libgl1 libglib2.0-0` (libgl1 cần cho `import cv2`).
2. Verify: `nvidia-smi` chạy được; `source .venv/bin/activate && python -c "import torch, cv2, mmdet; print(torch.cuda.is_available())"` → True.
3. Nếu `artifacts/runs/n300_B50_eps5/` mất vì lý do nào đó: `HF_TOKEN=... python scripts/sync_artifacts.py download n300_B50_eps5`.
4. Chạy Generalization Panel (tmux, ~20 phút): `python scripts/eval_generalization.py` → áp quy tắc khả thi đã khóa trong `protocol_lock.md` (headroom OSFD).
5. Verify clean AP full val2017 cho 5 model Generalization Panel (`third_party/mmdetection/tools/test.py`), điền `model_registry.md`.
6. Sau đó mới tới A3 (docs/mechanism_plan.md) — tùy kết quả bước 4.

---

## 2026-09-23 — Generalization Panel @300 (B=50, ε=5) + quyết định khả thi theo quy tắc khóa

- GPU dùng lại được sau khi reload cửa sổ VSCode — **không cần restart pod** (các bước "sau restart" ở entry trước không cần làm; tmux/apt còn nguyên).
- Thứ tự (user duyệt, pivot thứ tự chứ không đổi RQ): A2 → Generalization baselines → xác định vùng transfer khó → Mechanism → Method.
- `scripts/eval_generalization.py`, dùng lại PNG adv của `n300_B50_eps5` (surrogate Mask R-CNN R50), mỗi model đọc file qua pipeline riêng. Kết quả: `results/runs/n300_B50_eps5/generalization_metrics.json` (+ `gen_run.log`); detection thô `artifacts/runs/n300_B50_eps5/gen_dets/`.
- Clean AP trên n300: FCOS 41.9, DETR 44.3, YOLOX-S 43.9, YOLOX-L 53.5, DINO-Swin-L 62.4 — đều cao hơn README ~3–4 điểm, cùng mức lệch của R50 (45.0 vs 40.9) → nhất quán do tập con, checkpoint load đúng. Chưa verify full val2017.

Relative AP drop % [95% CI] (R101 = mốc cùng họ Controlled Panel):

| Method | R101 | FCOS-R50 | DETR-R50 | YOLOX-S | YOLOX-L (phụ) | DINO-Swin-L |
|---|---|---|---|---|---|---|
| MI-FGSM | 77.3 | 78.8 | 93.6 | 34.9 | 33.8 | 16.9 |
| M-DI²-FGSM | 93.0 | 93.1 | 96.1 | 58.0 | 60.5 | 26.1 |
| OSFD | 92.4 | 92.5 [89.7, 95.3] | 96.3 [93.9, 98.2] | 77.4 [71.8, 80.1] | 71.7 | 24.6 [20.4, 27.4] |

Headroom OSFD = drop(R101) − drop(model): FCOS −0.1 [−3.8, 3.3]; DETR −3.9 [−7.2, −1.5]; YOLOX-S 15.0 [12.1, 19.7]; DINO-Swin-L 67.8 [63.6, 72.7] (YOLOX-L 20.7, phụ).

**Áp quy tắc khóa (protocol_lock.md):** cận dưới CI headroom ≥ 10 ở YOLOX-S (12.1) và DINO-Swin-L (63.6) = 2/4 model, có DINO-Swin-L → **ĐI TIẾP** phát triển method. (Điều kiện DỪNG: headroom < 5 chỉ ở 2/4 — FCOS, DETR — không đạt ≥ 3/4.)

Quan sát (diễn giải, chưa kết luận):
- **Cùng backbone ResNet-50, khác detector (FCOS, DETR): transfer ≥ mức R101** (OSFD 92.5 / 96.3) → đổi detector head không làm yếu transfer khi backbone cùng họ; ủng hộ backbone-family effect. Caveat: FCOS/DETR khác surrogate cả về cách train (caffe 1x / 150e); và nhiều khả năng cùng khởi tạo từ ResNet-50 ImageNet-pretrained như surrogate → có thể chia sẻ feature cấp thấp — chưa kiểm.
- **YOLOX-S (CNN khác họ, CSPDarknet): OSFD 77.4 ≈ ConvNeXt 80.4** — nhất quán với cross-CNN ở Controlled Panel. OSFD vượt M-DI² rõ ở YOLOX (+19.4 S, +11.2 L).
- **DINO-Swin-L gần như miễn nhiễm**: OSFD 24.6, M-DI² 26.1, MI 16.9 (OSFD ≈ M-DI², hiệu −1.5 CI qua 0). Trong khi Mask R-CNN Swin-T (Controlled) bị OSFD 78.7 → **backbone Transformer đơn thuần KHÔNG giải thích được**; DINO-Swin-L lẫn 3 yếu tố: kiểu detector (DETR-family deformable, 5-scale), capacity/pretrain (Swin-L, ImageNet-22k, 384), và họ backbone. Panel hiện tại không tách được.
- Hệ quả cho RQ: vùng khó nhất (DINO-Swin-L) có thể KHÔNG phải do backbone-family → method nhắm vào đó có thể lệch khỏi RQ1–3. Đề xuất chẩn đoán (CHƯA chạy, cần user duyệt vì ngoài panel định trước): DINO-R50 (`dino-4scale_r50_8xb2-12e_coco`, có trong mmdet v3) để tách detector-paradigm khỏi backbone.

---

## 2026-09-23 — Chẩn đoán DINO-R50: quy tắc diễn giải khóa TRƯỚC khi chạy

> "DINO-R50 added post hoc as a diagnostic control to disentangle detector-paradigm effects from backbone-family effects observed for DINO-Swin-L; it is not used for target selection or method tuning." (user duyệt)

- Model: `dino-4scale_r50_8xb2-12e_coco` (AP README 49.0) — bản DINO-R50 DUY NHẤT có checkpoint chính thức trong mmdet v3 (24e/36e không có weights). Caveat: khác DINO-Swin-L cả ở lịch train (12e vs 36e) và số scale (4 vs 5), không chỉ backbone.
- Cùng ảnh adv `n300_B50_eps5`, cùng cách eval; kết quả ghi file riêng `results/runs/n300_B50_eps5/generalization_diag_dino.json` (không ghi đè bảng Generalization Panel).
- **Quy tắc diễn giải (user đề xuất; ngưỡng số do Claude chuẩn hóa từ ví dụ của user, chốt trước khi chạy)** — dựa trên relative AP drop của OSFD trên DINO-R50 (điểm ước lượng), kiểm chéo M-DI²:
  - **Case 1** — drop ≥ 80: kiểu detector DINO tự nó không giải thích failure → Swin-L / họ backbone / capacity-pretrain là nghi phạm chính; DINO-Swin-L phù hợp hướng cross-backbone.
  - **Case 2** — drop ≤ 40 (gần DINO-Swin-L 24.6): failure chủ yếu do kiểu detector DINO → KHÔNG dùng DINO-Swin-L làm bằng chứng backbone-gap; Controlled Panel vẫn là bằng chứng backbone sạch.
  - **Case 3** — 40 < drop < 80: cả detector lẫn backbone/capacity đều đóng góp → DINO-Swin-L chỉ là "generalization hard case", không quy nhân quả cho backbone.
  - **Contrast DETR vs DINO (cùng R50):** nếu cận dưới CI của drop(DETR-R50) − drop(DINO-R50) ≥ 10 → bằng chứng khác biệt nằm ở thiết kế detector giữa DETR và DINO.
- Sau run này mới quyết định có nhắm method vào DINO-Swin-L hay không.

---

## 2026-09-23 — Kết quả chẩn đoán DINO-R50 → Case 1

`results/runs/n300_B50_eps5/generalization_diag_dino.json` (+ `diag_dino_run.log`). Clean AP n300: DINO-R50 53.1 (README 49.0, cùng mức lệch tập con).

Relative AP drop % [95% CI] — DETR-R50 / DINO-Swin-L / **DINO-R50**:
- MI-FGSM: 93.6 / 16.9 / **92.8** [89.3, 94.7]
- M-DI²-FGSM: 96.1 / 26.1 / **95.4** [92.7, 96.5]
- OSFD: 96.3 / 24.6 / **97.3** [95.5, 98.5]

Contrast OSFD: DETR-R50 − DINO-R50 = −1.0 [−2.9, 0.9]; DINO-R50 − DINO-Swin-L = 72.7 [69.6, 76.6].

**Áp quy tắc khóa:** OSFD drop DINO-R50 = 97.3 ≥ 80 → **Case 1**: kiểu detector DINO tự nó KHÔNG giải thích failure; nghi phạm chính là backbone Swin-L (họ backbone và/hoặc capacity/pretrain). Contrast DETR−DINO (cận dưới −2.9) < 10 → không có bằng chứng khác biệt do thiết kế detector DETR vs DINO. DINO-Swin-L giữ được làm case cho hướng cross-backbone.

Quan sát (chưa kết luận):
- Cùng detector DINO, chỉ đổi R50 → Swin-L: drop 97.3 → 24.6 — hiệu ứng backbone cực lớn (hơn hẳn Mask R-CNN R101→Swin-T: 92.4 → 78.7).
- Trong Swin còn lẫn **capacity/pretrain**: Swin-T (Mask R-CNN, IN-1k, window 7) bị OSFD 78.7, Swin-L (DINO, IN-22k, 384, window 12) chỉ 24.6. Chưa tách được "họ Transformer" khỏi "Swin lớn + pretrain 22k". Chẩn đoán khả dĩ (CHƯA chạy, cần duyệt): Mask R-CNN Swin-S (`mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco`, cùng detector với Swin-T, lớn hơn) — đo độ dốc theo capacity trong cùng họ.
- Mọi target backbone ResNet-50 (FCOS, DETR, DINO) bị lừa ≥ R101 (MI-FGSM: 78.8 / 93.6 / 92.8 so với R101 77.3) → dấu hiệu "cùng kiến trúc R50 (có thể cùng ImageNet init)" transfer tốt hơn cả "cùng họ ResNet khác độ sâu". Chưa kiểm cùng init.

---

## 2026-09-23 — Chẩn đoán Mask R-CNN Swin-S: quy tắc khóa TRƯỚC khi chạy (chẩn đoán CUỐI)

> "Mask R-CNN Swin-S added as a diagnostic control to test whether model capacity within the Swin family explains the large transfer drop observed on DINO-Swin-L; not used for target selection or method tuning." (user duyệt)

- Model `mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco`: config giống hệt Swin-T (Controlled Panel), chỉ khác depths [2,2,18,2] vs [2,2,6,2]; cùng pretrain IN-1k 224, cùng lịch 3x → chỉ đổi capacity. KHÔNG tách được pretrain IN-22k/384 của Swin-L (mmdet không có Mask R-CNN Swin-L).
- Đo: d = drop_OSFD(Swin-T) − drop_OSFD(Swin-S), paired bootstrap trên cùng ảnh adv `n300_B50_eps5`; khoảng cần giải thích Swin-T → DINO-Swin-L = 78.7 − 24.6 = 54.1. Kiểm chéo M-DI².
- **Quy tắc (3 case của user; ngưỡng số do Claude chuẩn hóa, chốt trước khi chạy):**
  - **Case A — Swin-S ≈ Swin-T:** CI của d chứa 0, hoặc d < 5 (kể cả Swin-S bị lừa NHIỀU hơn) → capacity riêng không giải thích collapse ở Swin-L; nghi phạm còn lại: scale Swin-L + pretrain IN-22k / representation shift.
  - **Case B — capacity đóng góp lớn:** cận dưới CI d ≥ 5 VÀ d ≥ 27 (≥ nửa khoảng 54.1, tức drop Swin-S ≤ 51.7) → không được claim toàn bộ effect là do họ backbone.
  - **Case C — cộng dồn:** cận dưới CI d ≥ 5 VÀ d < 27 → family + capacity + pretraining cùng đóng góp; framing thận trọng.
- **Nguyên tắc (user): đây là chẩn đoán CUỐI.** Dù kết quả thế nào: đóng băng diễn giải → quay lại Mechanism Stage; không thêm chẩn đoán model nữa (tránh trôi sang "model robustness taxonomy").

---

## 2026-09-23 — Kết quả chẩn đoán Swin-S (cuối) — rơi vào khe hở của quy tắc

`results/runs/n300_B50_eps5/generalization_diag_swin_s.json` (+ `diag_swin_s_run.log`). Clean AP n300: Swin-S 52.2 (Swin-T 49.8).

Relative AP drop % — Swin-T / **Swin-S** / DINO-Swin-L: MI 45.1 / **38.1** / 16.9; M-DI² 67.1 / **58.1** / 26.1; OSFD 78.7 / **72.4** [67.3, 75.2] / 24.6.
OSFD: d = Swin-T − Swin-S = **6.3 [3.8, 9.1]**; Swin-S − DINO-Swin-L = 47.8 [43.1, 51.6]. d chiếm ~12% khoảng 54.1.

**Áp quy tắc khóa:** KHÔNG case nào khớp chặt — Case A cần CI chứa 0 hoặc d < 5 (CI [3.8, 9.1] không chứa 0, d = 6.3 ≥ 5); Case B/C cần cận dưới CI ≥ 5 (3.8 < 5). Khe hở do Claude chuẩn hóa ngưỡng: A dùng điểm ước lượng, B/C dùng cận dưới CI, không phủ vùng "d ≥ 5 nhưng cận dưới < 5". Không đổi ngưỡng sau khi thấy dữ liệu; ghi nhận nguyên trạng.

Mô tả trung tính: capacity trong họ Swin (T→S, cùng pretrain IN-1k) có hiệu ứng NHỎ nhưng khác 0 (CI loại 0; M-DI² 9.0, MI 7.0 cùng chiều), chỉ ~12% khoảng Swin-T → DINO-Swin-L; phần lớn (47.8) nằm giữa Swin-S và DINO-Swin-L — gắn với Swin-L scale + pretrain IN-22k/384 (+ tổ hợp với DINO), panel hiện tại không tách thêm được. **Diễn giải đóng băng: chờ user xác nhận câu chữ.** Theo nguyên tắc đã chốt: không thêm chẩn đoán model; quay lại Mechanism Stage.

**Chốt diễn giải Swin-S (user xác nhận, đóng băng):** capacity trong họ Swin có hiệu ứng nhỏ nhưng khác 0 (~12% mức tụt), không giải thích collapse ở DINO-Swin-L; phần lớn gắn với Swin-L scale + pretrain IN-22k/384, panel không tách thêm được → DINO-Swin-L là ca khó minh họa generalization, KHÔNG dùng làm bằng chứng nhân quả cho họ backbone. Bằng chứng backbone-family sạch: Controlled Panel + nhóm cùng R50 (FCOS/DETR/DINO-R50 ≥ 92%). **Dừng chẩn đoán model**, quay lại Mechanism Stage (A3). Khi đọc A3: lõi cơ chế = R101 vs ConvNeXt-T vs Swin-T; Swin-S/DINO-Swin-L chỉ là bằng chứng generalization (mô tả).

---

## 2026-09-23 — Mechanism A3: OSFD theo từng stage (n=100) → KHÔNG ĐẠT → bỏ hướng stage-aware

`scripts/mech_a3_stage_attack.py` (tiêu chí chốt + commit trước khi chạy, `18fe556`). Kết quả: `results/mechanism/a3_stage_attack.json` (+ `a3_run.log`); PNG `artifacts/runs/a3_stage_n100_B50_eps5/` (chưa upload HF).

Relative AP drop % (100 ảnh đầu n300), R101 / ConvNeXt-T / Swin-T | mô tả: Swin-S / DINO-Swin-L:
- all (OSFD gốc): 92.2 / 80.6 / 77.6 | 73.5 / 27.2
- stage1: 15.9 / 8.8 / 11.3 | 7.8 / 2.4
- stage2: 42.6 / 29.8 / 29.4 | 22.8 / 6.3
- stage3: 86.7 / 71.9 / 72.1 | 65.8 / 22.3
- stage4: 93.8 / 80.5 / 73.2 | 69.2 / 18.7

**Áp tiêu chí khóa:** không có cặp stage nào ĐẢO thứ hạng giữa R101 và target khác họ (reversals = [], A3_pass = False). Thứ tự stage1 < stage2 < stage3 ≤ stage4 giữ ở mọi target lõi.
- Có interaction khác 0 về ĐỘ LỚN (không đảo chiều): lợi ích của stage sâu nhỏ hơn trên Swin-T — vd stage3 vs stage4: R101 −7.1, Swin-T −1.1, interaction 6.0 [1.6, 11.0]; stage1 vs stage4 Swin 16.0 [9.5, 24.8]. Không đủ theo tiêu chí (cần đảo thứ hạng).
- Stage đơn so với OSFD gốc: không stage nào hơn "all" ở target khác họ (stage4 − all: ConvNeXt −0.1 [−3.4, 3.8], Swin-T −4.4 [−7.3, −2.1]; stage3 − all: −8.7 / −5.5) → chọn 1 stage KHÔNG tạo chỗ trống; OSFD gốc đã gần bằng stage tốt nhất.
- Mô tả: với Transformer (Swin-T, DINO-Swin-L) stage3 ≥ stage4 (DINO-Swin-L 22.3 vs 18.7), còn R101 stage4 tốt nhất — khớp quan sát A2 (distortion Swin giảm ở stage 4). Chỉ là quan sát.
- Alignment sign ∇(OSFD stage k) vs ∇ task target ≈ 0.500 ở mọi stage/target → OSFD không hoạt động qua căn hướng task-gradient (khớp A1: cosine OSFD ngược chiều).

**Quyết định theo quy tắc khóa (mechanism_plan.md):** giữ stage-aware chỉ khi A2(b) hoặc A3 có bằng chứng — A2 đạt một phần (không phải "có bằng chứng"), A3 không đạt → **BỎ hướng "stage-aware backward regularization"**. A1 cũng không đạt. Còn A4 (phổ tần), A5 (iterative stability) chưa chạy. Theo quy tắc: nếu chỉ A1/A4 có bằng chứng → đề xuất hướng method khác phù hợp cơ chế.

---

## 2026-09-23 — Mechanism A4: phổ tần + tập trung không gian (n=300) → KHÔNG ĐẠT

`scripts/mech_a4_spectral.py` (tiêu chí chốt + commit trước khi chạy, `14869e9`). Kết quả: `results/mechanism/a4_spectral.json` (+ `a4_run.log`). 3 dải cố định (cycles/pixel): thấp [0,1/6), trung [1/6,1/3), cao [1/3,∞).

- Tỉ lệ năng lượng thấp/trung/cao — gradient: R50 .236/.415/.349, R101 .270/.396/.334, ConvNeXt .290/.437/.274, Swin .253/.400/.347; δ: MI .228/.330/.442, M-DI² .279/.365/.356, **OSFD .390/.335/.275** (OSFD tần thấp hơn hẳn).
- (i) L1 phổ g_R50 vs g_t: R101 .095, ConvNeXt .172, Swin .094. ConvNeXt − R101 = +.077 [.069, .086] ✓; **Swin − R101 = −.001 [−.009, .007] ✗** → không đạt "cả 2 target khác họ".
- (ii) Spearman(match δ↔g_t, suppression) ở target khác họ **ÂM**, CI < 0 ở nhiều ô (OSFD: ConvNeXt −.177, Swin −.216; M-DI²: ConvNeXt −.186) — NGƯỢC chiều giả thuyết; retained cho cùng kết luận (dấu ngược) → không kích hoạt ceiling check. ✗
- **A4_pass = False.** Diễn giải (không đổi kết luận): tương quan âm per-ảnh nhiều khả năng do nội dung ảnh gây nhiễu (texture ảnh ảnh hưởng cả phổ δ lẫn độ dễ bị lừa), không phải cơ chế.
- Mô tả: gradient task tập trung mạnh theo không gian (top-5% pixel ≈ 87–89% năng lượng; năng lượng trong GT box gấp 6.9 lần tỉ lệ diện tích với R50/R101, 8.5 lần với ConvNeXt/Swin), còn δ (sign-step) gần như đều (top-5% ≈ 6%, trong box ≈ 1.0×). Giữa 3 method, method có δ tần thấp hơn (OSFD) transfer tốt hơn — chỉ 3 điểm dữ liệu, không kết luận.

**Tổng Mechanism Stage (A1–A4) theo tiêu chí khóa:** A1 ✗, A2 đạt một phần, A3 ✗, A4 ✗; A5 chưa chạy. Chưa có cơ chế nào "có bằng chứng"; hướng stage-aware đã bỏ (entry A3).

---

## 2026-09-23 — Tổng hợp cuối ngày

Viết `docs/synthesis_2026-09-23.md`: kết quả chắc chắn (gap PASS, backbone > detector, headroom Generalization → ĐI TIẾP, DINO-Swin-L chỉ là ca khó, stage-aware bị loại), trạng thái A1–A4, manh mối chưa đạt tiêu chí, chỗ trống theo target, 3 hướng method giả thuyết (M1 phân bổ budget theo vật thể, M2 ưu tiên tần thấp, M3 giảm lệ thuộc khuếch đại stage sâu) cần pilot với quy tắc chốt trước. CLAUDE.md trỏ tới file này. Dừng phiên, mai tiếp. Chưa push. Ảnh adv A3 đã upload HF (`runs/a3_stage_n100_B50_eps5.tar`, 282 MB, sha256 cba15cb5…b366c964, verify tải lại khớp, 400 PNG).
