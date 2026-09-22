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
