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

---

## 2026-09-24 — Đóng Mechanism Stage; chốt quy tắc pilot method A′1 (TRƯỚC khi code/chạy)

Máy mới (RTX 3090), `bootstrap.sh` xong; sanity: torch 2.1.2+cu118 `cuda_available=True`, mmcv 2.1.0 (ops OK), mmdet 3.3.0, 4 checkpoint Controlled Panel, COCO val2017 5000 ảnh.

**Đóng Mechanism Stage (user duyệt):** A1 ✗, A2 một phần, A3 ✗, A4 ✗, A5 để sau (prior yếu). Claim an toàn: transferability gắn với biểu diễn backbone, không chỉ kiến trúc detector; KHÔNG claim cơ chế cụ thể (gradient alignment / tần số / stage).

**Vì sao không dùng "W⊙g trước sign()" (đề xuất ban đầu của hướng M1):** với W > 0 mọi pixel, sign(W⊙g) = sign(g); kể cả với MI (chuẩn hóa g/mean|g| chỉ là vô hướng) nếu W cố định qua step thì sign(W⊙m) = sign(m) → δ giống hệt OSFD. Ngoài ra δ đã gần bão hòa L∞: RMS|δ| run `n300_B50_eps5` = 4.65 / 4.66 / 4.77 trên ε = 5 (MI / M-DI² / OSFD) → không thể "dồn thêm biên độ" vào vật thể; thứ còn thay đổi được là HƯỚNG (dấu) của δ. Vì vậy chọn A′1: trọng số không gian đặt BÊN TRONG loss OSFD (đổi chính gradient, không chỉ thang).

Động cơ (giả thuyết, KHÔNG phải cơ chế đã chứng minh): A4 thấy gradient task tập trung mạnh quanh vật thể trong khi δ sign-step gần đều. Wording cho phép: "motivated by the observed mismatch between spatial gradient concentration and nearly uniform sign-based perturbations".

### Method A′1 — OSFD với loss feature có trọng số không gian (khóa)

L = Σ_l [ Σ_{h,w} W_l(h,w) · ‖k·F_l^clean(h,w) − F_l^adv(h,w)‖²₂ ] / [ C_l · Σ_{h,w} W_l(h,w) ]

- F_l: feature BACKBONE stage l (như OSFD, không qua FPN), k = 3. Chia C_l để W ≡ 1 trùng TUYỆT ĐỐI `F.mse_loss` của OSFD gốc — kiểm bằng số trước khi chạy (sanity, không phải cấu hình).
- W_l: dựng map M ở khung input của model (canvas đã pad của data_preprocessor, GT box của `data_sample.gt_instances` — đã ở khung resize, không gồm crowd), rồi downsample kiểu area về lưới của stage l. Không co-transform W theo view RRB (feature sạch của OSFD gốc cũng không co-transform; xoay ≤ 7°, scale ≤ 1.1).
- Ảnh không có GT box: W ≡ 1 (= OSFD gốc).
- Mọi thứ khác GIỮ NGUYÊN OSFD full recipe (protocol_lock.md): MI μ = 1, RRB θ=7, l_s=10, ρ=0.8, s_max=1.1, σ=6, ε = 5, α = 1, B = 50 (1 backward / 2 view mỗi step), noise init 0, seed ảnh = SEED + i như `run_baselines.py`.
- Threat model: GT-assisted (primary). Biến thể box dự đoán của surrogate chỉ để validation sau, không thuộc Go/No-Go.

**Đúng 2 cấu hình W (khai báo trước, không thêm cấu hình cứu vãn):**
- **W1 "box":** M = 1 trong hợp các GT box, M = 0.1 ở mọi chỗ khác (kể cả vùng pad).
- **W2 "box+ring":** mỗi box nở mỗi phía 0.25 × (rộng box) theo chiều ngang, 0.25 × (cao box) theo chiều dọc, tối thiểu 16 px (khung model), clip vào ảnh; M = 1 trong hợp các box đã nở, 0.1 chỗ khác.

### Tập dev100 (tune/sàng lọc CHỈ ở đây)

- Pool = ảnh val2017 có ≥ 1 instance annotation (cùng tiêu chí n1000, 4952 ảnh) − n1000 = 3952 ảnh; `dev100 = random.Random(20260924).sample(sorted(pool), 100)`; assert Dev100 ∩ N1000 = ∅. Sinh 1 lần, commit `data/image_lists/dev100.csv` + `dev_meta.json`, không sinh lại.
- n300/n1000 KHÔNG được dùng để chọn W hay bất cứ tham số nào.

### Sàng lọc @ dev100 (điểm ước lượng)

Chạy OSFD, OSFD+W1, OSFD+W2 trên dev100 (cùng seed ảnh). Đánh giá: ConvNeXt-T, Swin-T, YOLOX-S (quyết định); R50, R101, DINO-Swin-L (chỉ báo cáo). Δ = relative AP drop(candidate) − drop(OSFD).

Một cấu hình QUA sàng lọc nếu đồng thời: ΔCrossAvg(ConvNeXt-T, Swin-T) ≥ 2; Δ_ConvNeXt-T ≥ 0; Δ_Swin-T ≥ 0; Δ_YOLOX-S ≥ 0.
- Cả 2 qua → chọn ΔCrossAvg lớn hơn (chênh < 0.5 → chọn W1, đơn giản hơn). Chỉ 1 cấu hình được sang n300.
- Không cấu hình nào qua → DỪNG A′1, ghi log, không thêm biến thể.

### GO @ n300 (chạy 1 lần, cấu hình đóng băng, không tune thêm)

So với OSFD của run `n300_B50_eps5` (PNG trên HF, cùng seed ảnh → paired bootstrap 1000 mẫu, BOOT_SEED 2026). GO nếu đồng thời:
- ΔCrossAvg(ConvNeXt-T, Swin-T) ≥ 3 VÀ cận dưới CI 95% > 0;
- Δ_ConvNeXt-T > 0 VÀ Δ_Swin-T > 0 (điểm ước lượng);
- Δ_YOLOX-S ≥ 5 VÀ cận dưới CI 95% > 0 (bootstrap per-model của `eval_generalization.py` hỗ trợ được).
- DINO-Swin-L: chỉ báo cáo, không thuộc Go/No-Go. Báo cáo kèm R101 (Δ cùng họ), runtime, số view forward (không đổi: 2×B).
- Không đạt → NO-GO A′1, ghi log.

---

## 2026-09-24 — Pilot A′1 sàng lọc @ dev100 → W1 QUA (sát ngưỡng), W2 KHÔNG QUA

Code `6feb7ff` (commit trước khi chạy). Sanity trước khi chạy: W ≡ 1 trùng `F.mse_loss` tuyệt đối; W_l khớp lưới 4 stage. **Kiểm cơ chế (không phải kết quả transfer), 3 ảnh dev đầu, step 0:** gradient W1/W2 so với OSFD có cosine 0.62–0.91 nhưng sign agreement 0.935–0.985 — hạ trọng số nền trong loss chủ yếu làm gradient nền NHỎ đi chứ ít đổi dấu → tác động lên δ sign-step nhỏ theo thiết kế.

Kết quả `results/runs/dev100_B50_eps5/` (`metrics.json`, `generalization_metrics.json`, `pilot_a1_screen.json`, log). Clean AP dev100: R101 46.6, YOLOX-S 46.5, DINO-Swin-L 64.0. Relative AP drop % (điểm), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L:
- OSFD: 97.0 / 91.1 / 76.0 / 74.6 / 71.5 / 23.3
- OSFD-W1: 96.8 / 92.8 / 78.7 / 76.3 / 71.9 / 23.7
- OSFD-W2: 96.8 / 92.3 / 78.7 / 75.8 / 71.5 / 22.3

Áp quy tắc khóa (`scripts/pilot_a1_screen.py`):
- **W1 − OSFD:** ΔCrossAvg +2.13 ✓ (≥ 2), ConvNeXt +2.6 ✓, Swin +1.6 ✓, YOLOX-S +0.4 ✓ → **QUA**.
- **W2 − OSFD:** ΔCrossAvg +1.96 ✗, ConvNeXt +2.7, Swin +1.2, YOLOX-S 0.0 → **KHÔNG QUA** (sát ngưỡng; không làm tròn, không nới).
- → **Sang n300 với OSFD-W1** (cấu hình đóng băng).

Mô tả (không thuộc quy tắc): ΔCrossAvg paired bootstrap trên dev100 — W1 +2.1 [0.5, 4.1], W2 +2.0 [0.9, 3.4]; W1 ≈ W2 (−0.2 [−1.6, 1.3]). R101 cũng tăng (+1.7 / +1.2) → chưa phải gain riêng cho khác họ. YOLOX-S +0.4 [−2.6, 3.1], DINO-Swin-L +0.4 [−1.6, 2.6] — xa ngưỡng GO YOLOX-S ≥ 5 → kỳ vọng tiên nghiệm cho GO @ n300 thấp; vẫn chạy 1 lần theo quy tắc. Runtime không đổi (~5.0 s/ảnh, cả 3).

---

## 2026-09-24 — Pilot A′1 GO @ n300 (OSFD-W1, chạy 1 lần, đóng băng) → **NO-GO**

`results/runs/n300_B50_eps5/{metrics_pilot_a1.json, generalization_pilot_a1.json, pilot_a1_go.json, pilot_a1_go_run.log}` (file riêng, không đè `metrics.json` / `generalization_metrics.json` của baseline). OSFD dùng lại PNG của run baseline (không attack lại); **số OSFD tái lập đúng bảng baseline đã commit** (98.9 / 92.4 / 80.4 / 78.7; YOLOX-S 77.4; DINO-Swin-L 24.6) → pipeline nhất quán.

Relative AP drop % [95% CI], R50 / R101 / ConvNeXt-T / Swin-T | YOLOX-S / DINO-Swin-L:
- OSFD: 98.9 / 92.4 / 80.4 / 78.7 | 77.4 / 24.6
- OSFD-W1: 99.0 / 92.1 [88.3, 95.2] / 82.0 [77.1, 84.8] / 78.2 [73.5, 81.4] | 78.2 [72.9, 80.9] / 26.2 [21.6, 28.8]

W1 − OSFD (`scripts/pilot_a1_go.py`): ConvNeXt +1.6, Swin −0.5, R101 −0.3; **ΔCrossAvg +0.5 [−0.5, 2.1]**; **YOLOX-S +0.8 [−0.3, 2.3]**; DINO-Swin-L (báo cáo) +1.6 [0.1, 2.5].

Áp quy tắc khóa: ΔCrossAvg ≥ 3 ✗, CI > 0 ✗, ConvNeXt > 0 ✓, Swin > 0 ✗, YOLOX-S ≥ 5 ✗, CI > 0 ✗ → **NO-GO A′1**. Không thêm biến thể (theo quy tắc).

Ghi chú:
- Mức cải thiện ở dev100 (+2.1) co về +0.5 ở n300 — khớp kiểm cơ chế trước khi chạy (W chỉ đổi dấu gradient ở ~2–6% pixel) và cho thấy sàng lọc bằng điểm ước lượng trên 100 ảnh nhiễu cỡ ±2.
- Runtime ghi 3.1 (OSFD, lấy từ stats phiên 2026-09-23) vs 4.9 s/ảnh (W1, máy hôm nay) — khác máy, không phải chi phí method (ở dev100 cùng máy: 5.0 vs 5.1).
- Hệ quả cho hướng M1: dưới L∞ với sign-step, phân bổ không gian qua trọng số loss không tạo chỗ trống đáng kể. Hướng M1 dạng này coi như đóng.
- Artifact mới (PNG OSFD-W1 n300, `dets_pilot_a1.json`, `gen_dets/`; run `dev100_B50_eps5`) CHƯA upload HF.

---

## 2026-09-24 — Chốt quy tắc pilot P2 (gộp lõm theo channel) + dev300 (TRƯỚC khi code/chạy)

**Bỏ P1 (drop channel ngẫu nhiên trong loss, p = 0.1/0.2) trước khi chạy** — kiểm cơ chế 5 ảnh dev100: mask chỉ chọn số hạng của loss (không đổi forward) nên g_M = Σ_c M_c g_c tuyến tính theo mask; cos(g_M, g_OSFD) 0.993–0.996 / 0.984–0.991, sign agreement 0.96–0.97 / 0.94–0.95; trung bình 16 mask cos 0.999+ với g_OSFD (momentum lấy đúng trung bình đó) → P1 ≈ OSFD. P1′ (dropout trong forward, kiểu Ghost Networks, Li et al. AAAI 2020) để dự phòng xa (rủi ro trùng lặp cao).

**Manh mối cho P2 (đính chính số đã nêu trong chat):** "top-10% channel chiếm 57–66% loss stage 4" đo trên d ≈ 4·F_clean² ở δ≈0 — chủ yếu là tập trung năng lượng feature SẠCH. Đo đúng trên distortion thật e_c = mean_hw (F_adv − F_clean)² của R50, 100 PNG OSFD dev100: top-10% share s1–s4 = 0.29 / 0.28 / 0.29 / **0.48 ± 0.03** (đều = 0.10); năng lượng sạch 0.32 / 0.27 / 0.37 / 0.59; loss OSFD 0.30 / 0.26 / 0.33 / 0.55. OSFD-W1 giống hệt. → distortion OSFD tập trung rõ ở stage 4, một phần kế thừa từ feature sạch.

Giả thuyết (chưa là cơ chế): để vài channel sâu thống trị distortion có thể là chuyên biệt hóa theo representation của R50; giảm lợi ích biên của channel đã bị phá mạnh có thể giúp transfer khác họ. Ngược hướng FIA/NAA (tăng trọng feature quan trọng). Chưa claim novelty trước khi rà literature đầy đủ.

### Method P2 (khóa)

d_{l,c} = mean_{h,w} (F^adv_{l,c,h,w} − k·F^clean_{l,c,h,w})², k = 3 (OSFD gốc = Σ_l mean_c d_{l,c}, đúng `F.mse_loss`).
**L_P2 = Σ_l mean_c φ_l(d_{l,c}),  φ_l(d) = τ_l · log(1 + d/τ_l)**  (φ′ = 1/(1 + d/τ_l)).
- τ_l = s · median_c d⁰_{l,c}, d⁰ = d tại δ = 0 KHÔNG qua RRB (= (1−k)²·mean_hw F_clean²), tính 1 lần mỗi ảnh, detach, cố định suốt 50 step; chặn dưới 1e-12.
- Loss cộng trên 2 view RRB như OSFD. Mọi thứ khác giữ OSFD full recipe (MI μ=1, RRB, ε=5, α=1, B=50, noise init 0, seed ảnh SEED+i).
- Sanity trước khi chạy: φ(d) = d trùng tuyệt đối `F.mse_loss` OSFD.
- **Đúng 2 cấu hình, chỉ đổi s (giữ log1p):** P2a s = 1 (median, lõm mạnh); P2b s = 2 (nhẹ hơn). Không thử φ khác (sqrt/Huber/clip...).

### dev300 (tập development DUY NHẤT cho RQ3 từ đây)

dev300 = dev100 (giữ nguyên) ∪ `random.Random(20260925).sample(sorted(pool − dev100), 200)`, pool = ảnh val2017 có ≥ 1 instance − n1000; assert dev300 ∩ n1000 = ∅. Sinh 1 lần, commit `data/image_lists/dev300.csv`. **Không mở thêm dev500/dev700** dù P2 fail.

### Quy trình (một đường duy nhất)

1. **Mechanism gate** — 20 ảnh đầu của `dev100.csv`, mỗi cấu hình, cả hai điều kiện:
   - (G1) sign agreement(∇ₓL_P2, ∇ₓL_OSFD) tại δ = 0, không RRB, trung bình 20 ảnh **< 0.95**;
   - (G2) sau 10 step full recipe (MI + RRB, cùng seed ảnh), top-10% share của distortion thật e_{4,c} ở stage 4 (R50) trung bình 20 ảnh: **C10₄(P2) ≤ C10₄(OSFD) − 0.05**.
   Chỉ cấu hình qua gate mới vào dev300; không cấu hình nào qua → DỪNG P2.
2. **Sàng lọc @ dev300** (điểm ước lượng): ΔCrossAvg(ConvNeXt-T, Swin-T) ≥ 3; Δ_ConvNeXt-T > 0; Δ_Swin-T > 0; Δ_YOLOX-S ≥ 2. R50, R101, DINO-Swin-L chỉ báo cáo. Cả 2 qua → chọn ΔCrossAvg lớn hơn; chênh < 0.5 → chọn P2b (gần OSFD hơn).
3. **GO @ n300** (1 lần, đóng băng; so OSFD của `n300_B50_eps5`): ΔCrossAvg ≥ 3 VÀ cận dưới CI 95% > 0; Δ_ConvNeXt-T > 0 VÀ Δ_Swin-T > 0; Δ_YOLOX-S ≥ 5 VÀ cận dưới CI > 0. DINO-Swin-L chỉ báo cáo.
4. **Điều kiện cho diễn giải cơ chế (không phải điều kiện GO):** nếu GO, phải kèm C10₄(P2) < C10₄(OSFD) trên ảnh adv n300 (paired CI < 0) và báo cáo Spearman per-ảnh(ΔC10₄, Δ suppression) ở target khác họ. Không đạt → method có thể hữu ích nhưng KHÔNG viết theo câu chuyện nhân quả "giảm tập trung channel".
5. **Nếu P2 NO-GO:** không làm tiếp P3/P4 kiểu feature engineering; RQ3 pivot sang nguyên lý khác hẳn hoặc chấp nhận kết luận "các can thiệp đã thử không thu hẹp được phần gap còn lại một cách hệ thống".

---

## 2026-09-24 — Pilot P2: mechanism gate → KHÔNG cấu hình nào qua → DỪNG P2 (không chạy dev300)

Code `f78d785` (commit trước khi chạy). `scripts/pilot_p2_gate.py`, kết quả `results/runs/dev300_B50_eps5/p2_gate.json`. 20 ảnh đầu dev100.

- Sanity: φ(d) = d trùng `F.mse_loss` (sai số tương đối tối đa 2.2e-7) ✓.
- **G1 (đổi hướng gradient):** sign agreement với OSFD tại δ = 0 — P2a 0.837, P2b 0.859 (< 0.95 ✓; khoảng 0.72–0.93 theo ảnh). P2 đổi gradient thật, mạnh hơn nhiều so với W1 (0.935–0.985) và P1 (0.94–0.97).
- **G2 (giảm tập trung distortion stage 4 sau 10 step):** C10₄ OSFD 0.483; P2a 0.476 (Δ −0.008), P2b 0.477 (Δ −0.006) — cần ≤ −0.05 ✗. Theo ảnh, Δ nằm trong khoảng −0.03 đến +0.03; 3/20 ảnh P2 còn tập trung HƠN OSFD.
- **Áp quy tắc khóa:** cả 2 cấu hình trượt G2 → **DỪNG P2**, không chạy dev300, không thử φ/s khác.

Mô tả (không đổi kết luận): dù objective đổi hướng gradient đáng kể (~15% pixel đổi dấu), mức tập trung theo channel của distortion thật ở stage 4 gần như bất biến (0.48 ± 0.01). Gợi ý: sự tập trung này là thuộc tính của representation stage 4 của R50 (channel nào phản ứng với nhiễu cỡ ε = 5) chứ không phải do hàm gộp tuyến tính của OSFD — can thiệp ở mức loss khó phân bổ lại. Đây là quan sát, không phải kết luận nhân quả.

**Theo nguyên tắc đã chốt (entry quy tắc P2, mục 5):** không làm tiếp P3/P4 kiểu feature engineering. Chuỗi can thiệp mức loss trên OSFD đã thử: A′1 (không gian, NO-GO @ n300), P1 (bỏ trước khi chạy — ≈ OSFD), P2 (dừng ở gate). RQ3 phải pivot sang nguyên lý khác hẳn, hoặc chấp nhận kết luận "các can thiệp đã thử không thu hẹp được phần gap còn lại một cách hệ thống". Chờ user quyết.

---

## 2026-09-24 — Tổng kết cuối ngày (dừng phiên, mai tiếp)

Phiên hôm nay (máy mới RTX 3090, `bootstrap.sh` + `download_checkpoints.sh --gen`): đóng Mechanism Stage; tạo dev100 rồi dev300 (tập development DUY NHẤT cho RQ3); pilot A′1 → NO-GO @ n300; P1 bỏ trước khi chạy; P2 dừng ở mechanism gate. Chi tiết + số liệu: các entry 2026-09-24 ở trên.

**Trạng thái:** RQ3 chờ user quyết — (1) pivot sang nguyên lý khác hẳn (không phải đổi hàm loss trên feature R50; ví dụ can thiệp đường tính forward kiểu P1′/Ghost Networks), quy tắc chốt trước, chỉ dùng dev300; hoặc (2) chấp nhận kết luận âm, chuyển trọng tâm sang analysis paper (còn thiếu cho bảng cuối: B=200 + A5, ε=8, clean AP full val2017 cho Generalization Panel).

**Việc mở cho phiên sau:**
1. Push các commit hôm nay (máy không có credential GitHub).
2. Artifact CHƯA upload HF (mất nếu trả máy): `artifacts/runs/dev100_B50_eps5/` (211 MB: PNG OSFD/W1/W2, dets, gen_dets) và phần mới trong `artifacts/runs/n300_B50_eps5/` (PNG OSFD-W1, `dets_pilot_a1.json`, `gen_dets/`). Upload `n300_B50_eps5` sẽ GHI ĐÈ tar cũ trên HF (chỉ thêm nội dung) — cân nhắc upload phần mới dưới tên riêng.
3. Token HF đã dùng hôm nay bị lộ trong chat — revoke, tạo token mới.
4. Nếu máy mới: `bash scripts/bootstrap.sh` + `bash scripts/download_checkpoints.sh --gen`; ảnh adv baseline: `python scripts/sync_artifacts.py download n300_B50_eps5`.

---

## 2026-10-03 — Chốt quy tắc pilot KG (chọn cặp object theo Knowledge Graph + OSFD) — TRƯỚC khi code/chạy

> **Trạng thái: LOCKED (user duyệt 2026-10-03).** Mọi giá trị dưới đây (kể cả các giá trị Claude đề xuất ban đầu: PPMI, margin 15%, khớp diện tích ≤ 2×, S2 ≥ 0.5, S3 ≤ 1.5×, bảng đồng nghĩa) đã được user chấp nhận; user thêm min support = 20 và ghi rõ tập loại trừ rò rỉ. Không đổi sau khi thấy kết quả.

**Pivot RQ3 (user quyết):** sau P2, chuyển sang nguyên lý khác hẳn — can thiệp ở **view đầu vào** (thay 1 view của RRB bằng crop vùng hợp của 1 cặp object), không đổi hàm loss feature.

Giả thuyết (chưa là cơ chế): view tập trung vào vùng hợp của 2 object có quan hệ ngữ nghĩa (theo Visual Genome) cho δ transfer tốt hơn sang target khác họ, **và phần lợi ích đó đến từ tri thức ngữ nghĩa**, không chỉ do có thêm augmentation. Prior của Claude: E1 > E0 có thể xảy ra (augmentation scale kiểu DI), còn E2 > E1 là phần rủi ro — với surrogate, KG chỉ quyết định **vùng không gian nào** được crop; chưa có cầu nối cơ chế tới gap họ backbone (RQ1/RQ2). Novelty: cần rà Cai et al. (AAAI 2022 "Context-aware transfer attacks for object detection"; CVPR 2022 "Zero-query transfer attacks on context-aware object detectors") — dùng đồ thị đồng xuất hiện nhưng cho mục đích khác (targeted, né context-consistency check).

### Knowledge Graph (khóa, sinh 1 lần, commit trước mọi lần chạy attack)

- Nguồn: Visual Genome v1.4 `relationships.json` + `image_data.json` (để lấy `coco_id`).
- **Chống rò rỉ:** bỏ mọi ảnh VG có `coco_id` ∈ dev300 ∪ n300 ∪ n1000 (n300 ⊂ n1000, ghi tường minh để dễ kiểm tra).
- Map tên object VG → COCO-80: khớp tên chính xác (sau lowercase, bỏ khoảng trắng thừa, dùng `synsets` nếu có) + **bảng từ đồng nghĩa viết tay** (vd man/woman/boy/girl/child/people/lady/guy → person; motorbike → motorcycle; plane → airplane; tv/television → tv; phone/cellphone → cell phone; sofa → couch; table → dining table). Bảng này commit cùng ma trận, viết xong TRƯỚC khi nhìn bất kỳ kết quả attack nào; không sửa sau. Tên không map được thì bỏ.
- Đếm C_ij = số relationship instance có subject ∈ class i, object ∈ class j; đối xứng hóa C̃ = C + Cᵀ (tính cả i = j, vd person–person).
- **Chuẩn hóa (chọn 1, không thử cái khác): PPMI với min support** — S_ij = max(0, log[ C̃_ij · Σ C̃ / (C̃_i· · C̃_·j) ]), tổng biên tính trên toàn bộ C̃; sau đó **C̃_ij < 20 → S_ij = 0** (min support = 20 relationship instance, không tune). Mục đích: PPMI cho điểm rất cao với cặp cực hiếm; ngưỡng buộc KG chỉ dùng quan hệ có đủ bằng chứng. Lý do: đếm thô bị person/car chi phối → "chọn cặp KG" gần như thành "chọn cặp có person" (nhiễu do tần suất class lẫn với kích thước box), còn PPMI đo ái lực quan hệ vượt trên tần suất.
- Output: `data/kg/coco80_vg_relation_matrix.npy` (S, 80×80), `data/kg/coco80_vg_relation_metadata.json` (phiên bản VG, bảng đồng nghĩa, min support, số ảnh VG bị loại, số relation map được, số ô bị min support loại, tỉ lệ ô S > 0).

### Cặp hợp lệ, ảnh có thông tin

- Object hợp lệ = GT instance không phải crowd (`data_sample.gt_instances`, khung model). Cặp hợp lệ = 2 instance khác nhau (cho phép cùng class).
- Phân loại ảnh (tính sẵn trên dev300/n300 từ annotation, báo cáo trước khi chạy):
  - **0 cặp** (1 object): view 2 = view RRB gốc → ảnh này giống hệt E0.
  - **1 cặp:** E1 = E2 = cặp đó (có pair-augmentation nhưng không có thông tin KG).
  - **≥ 2 cặp = "ảnh có thông tin"** (KG có thể chọn khác random).
  - Nếu mọi cặp của ảnh có S = 0: E2 sample đều (= E1 về prior), ảnh tính vào nhóm "KG suy biến", báo cáo riêng.

### Chọn cặp mỗi step (E1 và E2 dùng chung luồng RNG)

Mỗi step t, RNG riêng cho việc chọn cặp, seed theo ảnh (SEED + i), giống nhau giữa E1 và E2:
1. Rút cặp tham chiếu q ~ p(i,j) ∝ S_{c_i c_j} trên các cặp hợp lệ của ảnh.
2. **E2** dùng q.
3. **E1 (geometry-matched random control)** rút đều trong tập ứng viên {p hợp lệ : |log(A_p / A_q)| ≤ log 2}, A = diện tích union box đã nở margin. Tập luôn chứa q nên không bao giờ rỗng.

→ E1 và E2 có cùng mức ngẫu nhiên mỗi step và cùng phân bố kích thước crop (xấp xỉ), chỉ khác ở prior ngữ nghĩa. Báo cáo: tỉ lệ step mà cặp E1 ≠ cặp E2, phân bố |log(A_E1/A_E2)|.

### View cặp (giống hệt nhau giữa E1 và E2)

- T_pair: union box của 2 GT box, nở mỗi phía **15%** cạnh tương ứng của union, clip vào ảnh; crop từ adv ở khung model (KHÔNG xoay); resize bilinear khả vi theo đúng luật Resize của test pipeline (keep_ratio, cạnh dài ≤ 1333, cạnh ngắn ≤ 800), không chặn hệ số phóng; sau đó cộng nhiễu Gaussian σ = 6 như các view RRB (`rrb.gaussian_blur`).
- **Thay view 2 của RRB:** mỗi step = [view1 = rotate(adv) + nhiễu (giữ nguyên RRB), view2 = T_pair(adv) + nhiễu]. 2 view forward, 1 backward → B = 50, số view forward = 2×B, bằng E0.
- **Feature sạch đúng theo view:** loss OSFD của view cặp so với k·F_clean(T_pair(x_clean)) (cùng crop, không nhiễu, k = 3); view1 vẫn so với F_clean(x_clean) như OSFD gốc. F_clean theo cặp tính no_grad, cache theo cặp (forward thêm không tính vào B; báo cáo số forward no_grad + runtime).
- Loss = L_OSFD(view1) + L_OSFD(view2), cộng như OSFD hiện tại (hệ số ½ trong spec không đổi δ vì MI chuẩn hóa gradient).
- Mọi thứ khác giữ OSFD full recipe (protocol_lock.md): MI μ = 1, ε = 5, α = 1, B = 50, noise init 0, RRB θ = 7, l_s = 10, σ = 6, seed ảnh SEED + i. Threat model GT-assisted.

### Điều kiện

- **E0** — OSFD gốc (chạy mới trên dev300; trên n300 dùng lại PNG `n300_B50_eps5`).
- **E1** — geometry-matched random control (cặp ngẫu nhiên, điều kiện theo hình học của cặp tham chiếu do KG rút) + OSFD. Trong log/paper gọi đúng tên này, KHÔNG gọi là "Random-pair" (vì E1 không ngẫu nhiên hoàn toàn).
- **E2** — KG + OSFD.

### Chỉ số

- **CrossAvg₃ = mean(drop ConvNeXt-T, Swin-T, YOLOX-S)** — ĐỊNH NGHĨA MỚI, khác CrossAvg 2 target của A′1/P2 (baseline n300: CrossAvg₃(OSFD) = 78.8, CrossAvg₂ = 79.5). R101 = cùng họ; R50 white-box, DINO-Swin-L chỉ báo cáo.
- Paired bootstrap 1000 mẫu trên ảnh, 95% CI, BOOT_SEED 2026. Báo cáo E2−E0, E2−E1, E1−E0 cho CrossAvg₃, từng target, R101 — trên toàn bộ ảnh (chính) và trên tập ảnh có thông tin (phụ, không thuộc quy tắc).
- Output `results/kg_pilot/<set>/<condition>/{metrics.json, per_image_metrics.csv, config.json, runtime.json}`: clean/adv AP, relative AP drop, AP50, suppression, retained confidence, CrossAvg₃, TransferGap, L∞, L2, runtime, số backward, số view forward (+ forward no_grad).

### Quy trình (1 đường duy nhất)

**Bước 0 — Sanity gate kỹ thuật** (20 ảnh có thông tin đầu tiên theo thứ tự `dev100.csv`; không đo transfer). Phải đạt hết, nếu không thì sửa bug hoặc dừng:
- (S1) Crop chứa trọn 2 GT box; F_clean(T_pair) đúng shape; với 1 cặp cố định và tắt nhiễu, loss tại δ = 0 của view cặp = (1−k)²·mean F_clean² (đúng như OSFD).
- (S2) Tỉ lệ step có cặp E1 ≠ cặp E2 **≥ 0.5** — thấp hơn thì KG gần như trùng random sau khi khớp hình học → phép thử vô nghĩa → **DỪNG** hướng KG.
- (S3) Trung vị |log(A_E1 / A_E2)| ≤ log 1.5 (khớp hình học thực sự hoạt động).
- (S4) Sign agreement(∇ E2, ∇ E0) tại step 0, trung bình 20 ảnh < 0.95 (view cặp thật sự đổi hướng gradient — cùng ngưỡng G1 của P2).

**Bước 1 — Sàng lọc @ dev300** (E0, E1, E2; 1 lần). QUA nếu đồng thời:
- ΔCrossAvg₃(E2 − E0) ≥ 3 (điểm ước lượng);
- ΔCrossAvg₃(E2 − E1) > 0 VÀ cận dưới CI 95% > 0;
- E2 − E0 > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S (điểm ước lượng);
- drop R101(E2) ≥ drop R101(E0) − 3.

Không qua → NO-GO hướng KG, ghi log, không thêm biến thể (không đổi chuẩn hóa S, margin, cách khớp). **Trường hợp E1 > E0 nhưng E2 ≈ E1** (CI E2−E1 chứa 0): kết luận "augmentation cặp có thể hữu ích, KG không đóng góp rõ" → NO-GO cho claim KG; pair-augmentation KHÔNG tự động thành method — muốn đi tiếp phải chốt quy tắc mới riêng.

**Bước 2 — Đóng băng cấu hình → GO @ n300** (chạy 1 lần E1 + E2; E0 = PNG baseline). GO nếu đồng thời:
- (G1) ΔCrossAvg₃(E2 − E0) ≥ 3 VÀ cận dưới CI > 0;
- (G2) ΔCrossAvg₃(E2 − E1) ≥ 1.5 VÀ cận dưới CI > 0;
- (G3) E2 − E0 > 0 ở cả 3 target khác họ VÀ ≥ 2/3 target có E2 − E0 ≥ 3;
- (G4) drop R101(E2) ≥ drop R101(E0) − 3.

Không đạt → NO-GO, ghi log. Chỉ khi GO mới cân nhắc n1000 (idea.md §14).

Sanity gate (bước 0) chỉ kiểm tính đúng kỹ thuật và tính khả thi của phép thử, KHÔNG dùng để quyết định về hiệu năng.

Ghi chú độ nhạy: CI hiệu của 2 method gần nhau rộng khoảng ±1.5–1.8 ở 100 ảnh (W1−W2 @dev100: [−1.6, 1.3]), ước tính khoảng ±0.9 ở 300 ảnh → vì vậy sàng lọc dùng đủ dev300, không dùng dev100.

---

## 2026-10-03 — Pilot KG: dựng ma trận KG + thống kê cặp trên dev300/n300 (chưa chạy attack)

Máy RTX 3090; bootstrap xong (venv có sẵn dùng được, 4 checkpoint Controlled + Generalization Panel `--gen`, COCO val2017 5000 ảnh).

**KG** (`scripts/build_kg.py` → `data/kg/coco80_vg_relation_matrix.npy` + `_metadata.json`; VG gốc ở `data/vg/`, gitignored, sha256 zip ghi trong metadata):
- **Lựa chọn thực thi (không đổi quy tắc khóa):** map tên VG → COCO-80 bằng KHỚP TÊN CHÍNH XÁC sau chuẩn hóa + bảng viết tay (`NAME_TABLE` trong script, có số nhiều; loại có chủ đích tên mơ hồ: bag, glass, glove, plant, monitor, desk, van, calf, pitcher, female/male, crowd), **không dùng synset VG** — kiểm trước khi viết bảng: synset nhiễu nặng (`dog.n.01` gồm "hot dog" 3.5k lần, `child.n.01` gồm "fries"/"french fries" ~1.5k, `accountant.n.01` = "controller"), ~17% object không có synset. Bảng viết dựa trên tần suất tên, trước khi tính S.
- Loại 577/108077 ảnh VG có coco_id ∈ dev300 ∪ n300 ∪ n1000 (1300 coco_id). Relation map được 166,201/2,303,408 (phần lớn relation VG là bộ phận cơ thể/quần áo/cảnh vật); bỏ 2,459 quan hệ object–chính nó.
- 3240 ô tam giác trên: 1294 có đếm > 0, min support (C̃ < 20) loại 871, **308 ô S > 0 (9.5%)**. Lớp không có S > 0: potted plant, toaster.
- Quan sát: PPMI cao nhất thường ở cặp CÙNG class (parking meter, traffic light, zebra, sheep, cow… 4.5–6.2) và cặp đồ vật cùng ngữ cảnh (microwave–oven 4.8, mouse–keyboard 4.0, toilet–sink 3.7); person–X chỉ ~0.5–1.0. Đo bên dưới: thiên lệch cùng class trên ảnh thật nhỏ.

**Thống kê cặp** (`scripts/kg_pair_stats.py` → `results/kg_pilot/pair_stats.json`; giải tích từ annotation, không đo transfer):

| | dev300 | n300 |
|---|---|---|
| 0 cặp (→ = E0) | 43 (14.3%) | 45 (15.0%) |
| 1 cặp (E1 = E2) | 38 (12.7%) | 50 (16.7%) |
| ≥ 2 cặp (có thông tin) | 219 (73.0%) | 205 (68.3%) |
| … KG suy biến (mọi S = 0) | 1 | 7 |
| … S bằng nhau mọi cặp (E2 = đều) | 13 | 9 |
| Kỳ vọng P(cặp E1 ≠ E2), TB [ảnh có thông tin] | 0.826 | 0.819 |
| Kỳ vọng \|log(A_E1/A_E2)\|, TB | 0.248 | 0.236 |
| TV(phân bố cặp E2, E1), TB / trung vị | 0.413 / 0.414 | 0.410 / 0.446 |
| P(cặp cùng class): E2 vs chọn đều | 0.366 vs 0.333 | 0.324 vs 0.332 |

- Kỳ vọng S2 (≥ 0.5) và S3 (≤ log 1.5 = 0.405) đạt rộng rãi; sanity gate thật vẫn đo trên 20 ảnh dev100 theo quy tắc.
- TV ≈ 0.41: KG làm phân bố chọn cặp khác rõ so với random khớp hình học (không phải phép thử rỗng). Khoảng 27% ảnh dev300 không có thông tin KG → hiệu ứng E2−E1 trên toàn bộ ảnh bị pha loãng tương ứng.
- Việc tiếp theo: code view cặp + chọn cặp (E1/E2) trong engine OSFD, chạy sanity gate (bước 0).

---

## 2026-10-03 — Pilot KG: code view cặp + sanity gate (bước 0) → ĐẠT S1–S4

**Code:**
- `attack/methods/core.py`: view có thể mang loss riêng (phần tử thứ 3 của tuple; tương thích ngược).
- `attack/methods/kg_pair.py`: `PairSampler` (mỗi step rút q ~ p ∝ S rồi e1 đều trong M(q); LUÔN rút cả hai từ numpy RNG seed theo ảnh → chuỗi (q, e1) giống hệt giữa E1/E2), `crop_resize` (= luật `mmcv.rescale_size` keep_ratio 1333×800), `kg_osfd_attack(mode='kg'|'random_matched')`. View1 tiêu thụ random/torch toàn cục đúng thứ tự như `rrb_views` (step 0 view1 trùng E0); nhiễu σ của view cặp dùng `torch.Generator` riêng (crop E1/E2 khác kích thước sẽ làm lệch RNG toàn cục nếu dùng chung). Feature sạch theo cặp cache, đếm forward no_grad.
- Unit check: ảnh 0 cặp → E2 trùng tuyệt đối E0 (E0 tất định khi lặp lại); RNG toàn cục sau 3 step giống nhau giữa E1/E2; L∞ = 5.
- **Seed:** seed ảnh = SEED + chỉ số trong **dev300** (dev300 sắp theo image_id, dev100 không nằm ở đầu) — gate chọn ảnh theo thứ tự `dev100.csv` nhưng dùng seed dev300 → chuỗi cặp kiểm ở gate = chuỗi thật của lần sàng lọc.

**Sanity gate** (`scripts/pilot_kg_gate.py` → `results/kg_pilot/dev300/sanity_gate.json`; 20 ảnh có thông tin đầu theo dev100, 50 step thật E1 và E2):
- S1 ✓: crop chứa trọn 2 box (mọi cặp), lưới feature đúng stride, loss tại δ = 0 khớp (k−1)²·mean F² (sai số tương đối 1.1e-7).
- S2 ✓: tỉ lệ step cặp E1 ≠ E2 = **0.857** (≥ 0.5).
- S3 ✓: trung vị |log(A_E1/A_E2)| = **0.248** (≤ 0.405).
- S4 ✓: sign agreement(∇E2, ∇E0) step 0 = **0.759** (< 0.95).
- → **ĐẠT, sang sàng lọc dev300.**

Chẩn đoán (không thuộc quy tắc):
- Chuỗi (q, e1) của lần chạy E1 và E2 trùng nhau ở cả 20 ảnh.
- Tỉ lệ cặp cùng class: **E2 0.302 vs E1 0.405** — trên các ảnh này KG chọn cặp KHÁC class nhiều hơn random khớp hình học (ngược với lo ngại PPMI ưu tiên cặp cùng class).
- **S4 gần như không có sức phân biệt:** sàn sign agreement E0 vs E0 khác seed = **0.516** (nhiễu σ = 6 của RRB chi phối dấu gradient step 0); E2 vs E0 cùng seed = 0.759 (view1 trùng), E2 vs E1 = 0.886. Ngưỡng 0.95 (mượn từ G1 của P2, nơi không có RRB) quá lỏng cho thiết lập có RRB — ghi nhận, không đổi quy tắc; S4 chỉ xác nhận view cặp có đổi gradient.
- 1/20 ảnh (105455, 3 cặp) có P(E1≠E2) = 0: tập khớp hình học suy biến/KG một cặp chiếm ưu thế — trường hợp tự nhiên, nằm trong tỉ lệ pha loãng.
- Runtime gate: 142 s cho 20 ảnh (mỗi ảnh 2 lần chạy 50 step + 4 lần 1 step) → ước ~3.5 s/ảnh/điều kiện ở B = 50.

---

## 2026-10-03 — Pilot KG sàng lọc @ dev300 → **KHÔNG QUA (NO-GO hướng KG)**

Run `dev300_B50_eps5` (E0 = OSFD, E1 = `OSFD-KG-E1`, E2 = `OSFD-KG-E2`; `run_baselines.py --out metrics_kg.json`, `eval_generalization.py --out generalization_kg.json`), phân tích `scripts/pilot_kg_screen.py` → `results/kg_pilot/dev300/{screen.json, E0,E1,E2/{metrics.json, per_image_metrics.csv, config.json, runtime.json}}`. Paired bootstrap 1000 mẫu, BOOT_SEED 2026. Sự cố nhỏ: `eval_generalization.py` lỗi ghi JSON ở bước cuối (locale tmux ASCII, `ensure_ascii=False`) — dets đã lưu đủ; chạy lại với `PYTHONUTF8=1`.

Relative AP drop % (điểm), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- E0 OSFD: 98.1 / 93.0 / 77.4 / 74.6 / 68.7 / 24.9 | **73.6**
- E1 random khớp hình học: 95.3 / 87.0 / 76.3 / 73.8 / 62.7 / 23.2 | **70.9**
- E2 KG: 95.3 / 86.6 / 74.1 / 71.2 / 61.3 / 21.9 | **68.9**

Δ [95% CI] (toàn bộ 300 ảnh):
- **E2 − E0:** CrossAvg₃ **−4.7 [−6.2, −3.1]**; ConvNeXt −3.3, Swin −3.5, YOLOX-S −7.3, R101 −6.4 [−8.4, −3.9], DINO-Swin-L −3.0.
- **E2 − E1:** CrossAvg₃ **−2.1 [−3.0, −0.8]**; ConvNeXt −2.2 [−3.5, −0.6], Swin −2.6 [−4.0, −0.9], YOLOX-S −1.3 [−3.2, 0.4], R101 −0.4.
- **E1 − E0:** CrossAvg₃ −2.6 [−4.3, −1.2]; ConvNeXt −1.1, Swin −0.8 (CI qua 0), YOLOX-S −6.0, R101 −6.0.
- Ảnh có thông tin (n = 219, chỉ báo cáo): cùng chiều, lớn hơn chút (E2−E0 −5.2, E2−E1 −2.6 [−3.6, −1.0]).

**Áp quy tắc khóa:** cả 7 điều kiện trượt (ΔE2−E0 ≥ 3 ✗, E2−E1 > 0 ✗, CI ✗, 3 target > 0 ✗, R101 ≥ E0 − 3 ✗) → **NO-GO hướng KG**. Theo quy tắc: không thêm biến thể (không đổi chuẩn hóa S, margin, cách khớp, không chạy n300).

Diễn giải (mô tả, không đổi kết luận):
- **Thay view resize của RRB bằng crop cặp làm GIẢM transfer** (E1 < E0), mạnh nhất ở R101 (−6) và YOLOX-S (−6), white-box R50 cũng giảm (98.1 → 95.3). Crop phóng to một vùng nhỏ làm loss của view 2 chỉ phủ một phần ảnh → δ ngoài vùng crop nhận ít tín hiệu hơn; view resize toàn ảnh của RRB có vẻ quan trọng hơn ta nghĩ.
- **KG còn tệ hơn random khớp hình học** (CI loại 0, hai target Mask R-CNN khác họ). Có thể do KG tập trung vào ít cặp hơn (forward no_grad / ảnh: E2 12.9 vs E1 17.4 cặp khác nhau được dùng) → ít đa dạng vùng crop hơn. Không kiểm thêm (ngoài quy tắc).
- Chẩn đoán cặp trên dev300: tỉ lệ step cặp E1 ≠ E2 = 0.705 (gồm cả ảnh 1 cặp), tỉ lệ cặp cùng class E2 0.346 vs E1 0.349; 10 ảnh có mọi cặp S = 0 (gồm cả ảnh 1 cặp). Runtime ~3.3 s/ảnh cả 3 điều kiện; số view forward 2×B như nhau.
- Mức OSFD E0 trên dev300 nhất quán với n300 (R101 93.0 vs 92.4; ConvNeXt 77.4 vs 80.4; Swin 74.6 vs 78.7; YOLOX-S 68.7 vs 77.4 — dev300 khó hơn ở YOLOX-S).

**Trạng thái RQ3:** chuỗi đã thử — A′1 (loss không gian, NO-GO @ n300), P1 (bỏ), P2 (dừng ở gate), KG (view cặp, NO-GO @ dev300, còn làm giảm transfer). Chờ user quyết hướng tiếp.

Artifact CHƯA upload HF: `artifacts/runs/dev300_B50_eps5/` (PNG OSFD/E1/E2 dev300, `dets_kg.json`, `gen_dets/`).

---

## 2026-10-03 — Chốt quy tắc pilot R1 (OSFD-adapted SGM backward + OSFD) — TRƯỚC khi code/chạy

> **Trạng thái: LOCKED (user duyệt 2026-10-03).** Gồm cả bản SGM chỉ-nhánh-residual (mục dưới). **Tên gọi: "OSFD-adapted SGM" / "stage-balanced SGM" — KHÔNG gọi là port SGM nguyên bản.** Wording cho paper (user): *"We adapt SGM to OSFD's multi-stage feature objective by applying residual-gradient decay only within residual branches, avoiding depth-dependent rescaling of stage losses."* Không claim đây là exact SGM implementation.

**Cam kết trước (user):** đây là pilot method CUỐI của RQ3. **R1 NO-GO (ở sàng lọc dev300 hoặc GO @ n300) → dừng RQ3, chuyển sang analysis paper.**

Động cơ (giả thuyết): hướng M3 trong `docs/synthesis_2026-09-23.md`; A2 thấy từ stage 2 R101 khuếch đại distortion còn ConvNeXt/Swin thì không → gradient chuẩn của R50 có thể quá đặc thù đường residual của ResNet. SGM (Wu et al., ICLR 2020, "Skip Connections Matter") giảm gradient qua nhánh residual, ưu tiên skip. Khác A3 (chọn stage) và A′1/P2 (đổi loss) và KG (đổi view): giữ forward, loss, RRB, MI; chỉ đổi đường backward. Novelty: SGM/LinBP/PAS đã có ở classification — nếu GO, đóng góp phải nằm ở bằng chứng can thiệp cho RQ2/detector, không phải "gắn SGM vào OSFD". Nếu E_best > E0 rõ, đây là thí nghiệm can thiệp gắn với quan sát khuếch đại residual của A2 (có giá trị cho analysis paper dù method không mới).

### Ngữ nghĩa SGM (port từ code chính thức `csdongxian/skip-connections-matter/utils_sgm.py`)

Code gốc với ResNet-50: hook backward trên MODULE `relu` của mọi Bottleneck trừ block đầu mỗi stage (`'0.relu'` — block có downsample), nhân gradient √γ. Module này dùng 3 lần mỗi block (sau norm1, sau norm2, sau phép cộng residual) → residual × γ^1.5, skip × √γ → **tỉ lệ residual/skip = γ** (phần cốt lõi). Thêm: relu stem cũng bị nhân; hook chuẩn hóa `grad / std` ở mỗi block. Với classification (loss vào DUY NHẤT ở đầu ra) mấy phần thêm chỉ là hằng số vô hướng → vô tác dụng sau `sign()`.

**Vì sao không port nguyên văn — loss nhiều stage:** OSFD đặt loss ở 4 stage, gradient vào backward ở 4 độ sâu. Nhân √γ sau phép cộng (cả skip) làm số hạng stage 4 bị nhân ~(√γ)^12 ≈ 6·10⁻⁵ (γ = 0.2; R50 có 2+3+5+2 = 12 block không phải block đầu) trước khi tới input, còn stage 1 chỉ ~γ → gradient gần như CHỈ từ stage 1 (A3: tấn công riêng stage 1 rất yếu, R101 15.9%). Port nguyên văn sẽ đo "OSFD-stage-1", không đo SGM. Chuẩn hóa std theo block cũng đổi trọng số tương đối giữa các số hạng stage.
→ **Khóa (OSFD-adapted SGM):** chỉ giữ phần cốt lõi — trong mọi Bottleneck trừ block đầu mỗi stage, nhân gradient √γ ở 2 lần dùng ReLU TRONG nhánh residual (sau norm1, sau norm2); KHÔNG nhân ở ReLU sau phép cộng, KHÔNG nhân relu stem, KHÔNG chuẩn hóa std. Forward giữ nguyên tuyệt đối.
- **Unit test hướng SGM:** với loss chỉ đặt ở đầu ra stage 4 (1 điểm vào), gradient theo input của bản này phải song song với bản port nguyên văn (3 lần √γ + stem + chuẩn hóa std): cosine ≥ 0.999 trên 5 ảnh, γ ∈ {0.2, 0.5}. Diễn giải: xác nhận bản adapted giữ CÙNG hiệu ứng hướng của SGM trong trường hợp loss một điểm vào — KHÔNG phải "tương đương nói chung"; với OSFD nhiều stage hai bản cố ý không tương đương.

### Điều kiện (dev300; seed ảnh SEED + i như run_baselines)

| | backward |
|---|---|
| E0 | OSFD full recipe chuẩn — dùng lại PNG OSFD của run `dev300_B50_eps5` (cùng seed, không attack lại) |
| E1a | SGM γ = 0.2 (giá trị code gốc minh họa cho ResNet) |
| E1b | SGM γ = 0.5 (nhẹ hơn; cũng là default argparse của code gốc) |
| E2 | mỗi step chọn đều γ ∈ {0.2, 0.5} (KHÔNG gồm γ = 1); numpy RNG riêng seed SEED + i (không đụng RNG của RRB) |

Mọi thứ khác giữ OSFD full recipe (MI μ = 1, RRB θ = 7, l_s = 10, ρ = 0.8, s_max = 1.1, σ = 6, ε = 5, α = 1, B = 50, k = 3, noise init 0). Compute không đổi: 1 backward + 2 view forward / step. Không thử γ khác sau khi thấy kết quả.

### Quy trình (1 đường duy nhất)

**Bước 0 — Sanity kỹ thuật** (5 ảnh đầu dev100; không đo transfer, không có ngưỡng hiệu năng):
- Forward (feature 4 stage + loss OSFD) giữ nguyên tuyệt đối khi bật/tắt SGM.
- γ = 1 tái tạo gradient chuẩn (trùng tuyệt đối hoặc sai số tương đối < 1e-6).
- Unit test tương đương với code gốc (mục trên), cosine ≥ 0.999.
- γ = 0.2 / 0.5 đổi gradient có ý nghĩa: sign agreement(∇SGM, ∇chuẩn) tại δ = 0, KHÔNG RRB/nhiễu, trung bình 5 ảnh < 0.95.
- Báo cáo thêm (giữ, user nhấn mạnh): tỉ trọng đóng góp gradient từng số hạng stage (chuẩn vs SGM) — nếu bản adapted vẫn để 1 stage chiếm gần toàn bộ gradient input thì diễn giải sẽ lẫn với A3 → phải nêu rõ.
Hỏng → sửa bug; không chạy dev300 khi chưa đạt.

**Chẩn đoán cơ chế (KHÔNG phải tiêu chí):** trên 20 ảnh đầu dev100, sau 10 step (E0, E1a, E1b), đo D_k (như A2) trên R101 / ConvNeXt-T / Swin-T và tỉ số D_4/D_1; báo cáo cạnh kết quả transfer.

**Bước 1 — Sàng lọc @ dev300** (E1a, E1b, E2; 1 lần; đánh giá R50, R101, ConvNeXt-T, Swin-T, YOLOX-S, DINO-Swin-L như pilot KG). CrossAvg₃ = mean(ConvNeXt-T, Swin-T, YOLOX-S). E_best = điều kiện có CrossAvg₃ cao nhất (chênh < 0.5 → ưu tiên E1 hơn E2, và γ = 0.2 hơn 0.5). E_best QUA nếu đồng thời:
- ΔCrossAvg₃(E_best − E0) ≥ 3 (điểm ước lượng);
- E_best − E0 > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S (điểm);
- drop R101(E_best) ≥ drop R101(E0) − 3.
Báo cáo (không thuộc quy tắc): E2 − E1a, E2 − E1b, E1a − E1b với CI.
Không qua → **NO-GO R1 → dừng RQ3.**

**Bước 2 — Đóng băng E_best → GO @ n300** (1 lần; E0 = PNG `n300_B50_eps5` từ HF). GO nếu đồng thời:
- (G1) ΔCrossAvg₃(E_best − E0) ≥ 3 VÀ cận dưới CI 95% > 0;
- (G3) E_best − E0 > 0 ở cả 3 target khác họ VÀ ≥ 2/3 target ≥ 3;
- (G4) drop R101(E_best) ≥ drop R101(E0) − 3.
Không đạt → NO-GO → dừng RQ3. GO → cân nhắc n1000 (idea.md §14); diễn giải cơ chế chỉ khi chẩn đoán D_k nhất quán (không thì method có thể hữu ích nhưng không viết câu chuyện nhân quả).

---

## 2026-10-03 — Pilot R1 (OSFD-adapted SGM): sanity + sàng lọc dev300 → **NO-GO** → theo cam kết: **dừng RQ3**

**Code** (`29c763e`, `68ca341`): `attack/methods/sgm.py` — `StageBalancedSGM` (forward hook trên norm1/norm2 của Bottleneck không phải block đầu mỗi stage, chèn identity có backward × √γ trước ReLU inplace; γ đổi được mỗi step), `LiteralSGM` (port nguyên văn, chỉ cho unit test), `sgm_osfd_attack`. Thêm `OSFD-SGM-0.2/0.5/mix` vào `run_baselines.py`.

**Sanity kỹ thuật** (`scripts/pilot_r1_gate.py` → `results/r1_pilot/dev300/sanity_gate.json`, 5 ảnh đầu dev100, δ = 0, không RRB): đạt hết — forward (feature + loss) giữ nguyên tuyệt đối; γ = 1 tái tạo gradient chuẩn (sai số tương đối 1.5e-7); hướng SGM với loss chỉ ở stage 4: cos(adapted, nguyên văn) = 0.9999999 (γ 0.2 và 0.5); sign agreement với gradient chuẩn 0.612 (γ 0.2) / 0.729 (γ 0.5). Tỉ trọng ‖∇‖ theo số hạng stage 1→4: chuẩn .003/.159/.361/.478; γ 0.2 .009/.302/.400/.290; γ 0.5 .007/.280/.401/.313 — không stage nào bị triệt, nhưng SGM dời trọng tâm từ stage 4 sang stage 2–3.

**Chẩn đoán D_k** (`scripts/mech_r1_diag.py` → `mech_diag.json`; 20 ảnh dev100, 10 step; chạy TRƯỚC khi có số transfer): SGM làm distortion stage sâu GIẢM ở mọi model, kể cả target khác họ — D_4: R101 1.01 → 0.71 (γ 0.2) / 0.90 (γ 0.5); ConvNeXt 0.58 → 0.39 / 0.52; Swin 0.50 → 0.36 / 0.47; stage 1 gần như không đổi. D_4/D_1: ConvNeXt 1.30 → 0.86 / 1.12; Swin 1.53 → 1.03 / 1.38.

**Sàng lọc dev300** (`scripts/pilot_r1_screen.py` → `results/r1_pilot/dev300/{screen.json, E0,E1a,E1b,E2/...}`; E0 = PNG OSFD của run KG cùng seed; run `metrics_sgm.json` / `generalization_sgm.json`). Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- E0 OSFD: 98.1 / 93.0 / 77.4 / 74.6 / 68.7 / 24.9 | **73.6**
- E1a SGM γ 0.2: 79.4 / 58.2 / 37.8 / 42.4 / 39.2 / 11.0 | **39.8**
- E1b SGM γ 0.5: 97.1 / 87.9 / 68.3 / 67.8 / 63.1 / 20.9 | **66.4**
- E2 mix {0.2, 0.5}: 93.5 / 79.0 / 56.6 / 59.1 / 54.0 / 16.5 | **56.6**

Δ CrossAvg₃ [95% CI]: E1a − E0 **−33.8 [−36.8, −30.8]**; E1b − E0 **−7.2 [−8.6, −5.5]**; E2 − E0 −17.0 [−19.1, −14.8]; E2 − E1b −9.8; E1a − E1b −26.6. Mọi target, mọi điều kiện SGM đều GIẢM, CI loại 0.

**Áp quy tắc khóa:** E_best = E1b (γ 0.5); ΔCrossAvg₃ −7.2 (cần ≥ 3) ✗, ConvNeXt −9.1 ✗, Swin −6.8 ✗, YOLOX-S −5.6 ✗, R101 −5.1 (cần ≥ −3) ✗ → **NO-GO R1**. Theo cam kết đã khóa: **dừng RQ3, chuyển trọng tâm sang analysis paper.** Không thử γ/biến thể khác (LinBP, PAS…).

Diễn giải (mô tả, không đổi kết luận):
- **Hiệu ứng liều–đáp ứng đơn điệu:** γ càng nhỏ (càng giảm gradient nhánh residual) transfer càng giảm, ở mọi target kể cả cùng họ và cả white-box R50 (98.1 → 97.1 → 79.4). Ngược với SGM ở classification (Wu et al. báo tăng transfer). Với OSFD trên detector, gradient qua nhánh residual của R50 là cần thiết — không phải "overfit đặc thù ResNet" cần triệt.
- Nhất quán với chẩn đoán D_k (chạy trước): SGM làm δ phá feature sâu kém đi ở MỌI model, không riêng R101; và với dịch chuyển tỉ trọng gradient về stage 2–3 (A3: tấn công stage nông yếu hơn). Không tách được phần nào do "giảm residual" và phần nào do "dời trọng tâm stage" — ghi nhận.
- Đây là thêm 1 kết quả can thiệp âm cho RQ2: giảm đường gradient residual KHÔNG thu hẹp gap họ backbone (gap R101 − CrossAvg₃: E0 19.4, E1b 21.5, E1a 18.4 — gap gần như không đổi trong khi mọi drop đều giảm).
- Compute như nhau (B = 50, 2 view/step; runtime 3.3–3.5 s/ảnh); L∞ = 5; E2 dùng γ 0.2 ở 49.9% step.

**Tổng kết RQ3** (theo cam kết, đóng): A′1 (trọng số không gian trong loss) NO-GO @ n300; P1 bỏ (≈ OSFD); P2 (gộp lõm theo channel) dừng ở gate; KG (view cặp object) NO-GO @ dev300, làm giảm transfer; R1 (SGM backward) NO-GO @ dev300, giảm mạnh theo liều. Không can thiệp nào thu hẹp phần gap còn lại của OSFD.

Artifact CHƯA upload HF: `artifacts/runs/dev300_B50_eps5/` (PNG OSFD, KG-E1/E2, SGM-0.2/0.5/mix; `dets_kg.json`, `dets_sgm.json`, `gen_dets/`).

---

## 2026-10-03 — Mở lại RQ3 theo plan mới: ablation OSFD → cơ chế → method (`docs/plan_sota.md`, LOCKED)

**Quyết định (user):** mở lại RQ3 dù entry R1 đã cam kết đóng. Lý do: 5 pilot trước (A′1, P1, P2, KG, R1) đều đoán cơ chế rồi can thiệp, đều thất bại. Hướng mới chỉ phân tích SOTA (OSFD): bỏ/thay đúng 1 thành phần mỗi ô (7 ô: task loss, bỏ RRB, RRB bỏ nhiễu / xoay / resize, bỏ MI, k = 5), đo tụt bao nhiêu → "nguyên liệu" → suy cơ chế (forward trên PNG) → method khuếch đại nguyên liệu → GO @ n300 → n1000. Ablation có giá trị cho analysis paper kể cả khi không ra method.

Chạy 2 bậc: dev100 trước (seed theo chỉ số dev300), ô mơ hồ mới nối đủ dev300. Câu hỏi phụ DINO-Swin-L: chỉ mô tả (eval mọi ô + D_k trên backbone Swin-L), không thêm model. Chi tiết cổng và tiêu chí: `docs/plan_sota.md`.

---

## 2026-10-03 — Ablation OSFD bước 1: code + unit test (trước khi chạy)

- `attack/methods/rrb.py`: tách rút tham số (`_sample_rotation`, `_sample_resize`) khỏi áp biến đổi; `rrb_views` thêm `use_rotate/use_resize/use_noise` (tắt resize: view2 = lần xoay MỚI của adv) và `with_gt` (co-transform GT box + mask cho ô task loss).
- `attack/methods/baselines.py`: `osfd_attack(loss="task", rrb_rotate/resize/noise)`.
- `scripts/run_baselines.py`: 7 ô `ABL-*`; `--subset dev100` chạy trong run dev300 với seed theo chỉ số dev300. `scripts/eval_generalization.py`: `--subset`, resume theo (điều kiện, ảnh) qua khóa `_done` (điều kiện cũ coi như đủ cả split) — bậc 2 chỉ predict ảnh còn thiếu.
- Unit test: (1) `rrb_views` cờ mặc định trùng TUYỆT ĐỐI bản cũ (lấy từ git, 3 ảnh × 3 seed); ảnh ở chế độ `with_gt` trùng ảnh mặc định. (2) Co-transform: IoU(bbox của mask, box) — không biến đổi 0.958, chỉ resize 0.955 (khớp), chỉ xoay 0.834 (AABB của box xoay lỏng hơn mask, như AugTrans trước đây); kiểm chiều xoay: tỉ lệ pixel mask sau xoay nằm trong box đã xoay trung vị 1.000, p5 1.000, min 0.917 (n = 628). (3) Cả 7 ô chạy được, L∞ đúng.

---

## 2026-10-03 — Ablation OSFD bậc 1 (dev100) → RRB là nguyên liệu chính, loss feature KHÔNG phải

Run `dev300_B50_eps5` subset dev100 (`metrics_abl_dev100.json`, `generalization_abl_dev100.json`), phân loại `scripts/ablation_screen.py` → `results/ablation/dev100/screen.json`. n = 100, paired bootstrap 1000 mẫu.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 96.4 / 91.8 / 78.8 / 74.3 / 70.2 / 25.3 | 74.5 | 0.83
- task loss (+ MI + RRB co-transform GT): 98.6 / 92.5 / 80.1 / 73.7 / 67.6 / **33.8** | 73.8 | 0.83
- bỏ RRB: 99.6 / 84.8 / 52.8 / 41.1 / 33.2 / 13.3 | **42.4** | 0.55
- RRB bỏ nhiễu: 96.4 / 94.2 / 80.5 / 75.8 / **61.0** / 22.2 | 72.4 | 0.83
- RRB bỏ xoay: 97.0 / 92.4 / 73.8 / 67.0 / 65.9 / 19.2 | **68.9** | 0.76
- RRB bỏ resize: 96.7 / 92.3 / 79.0 / 75.7 / 67.8 / 24.3 | 74.2 | 0.84
- bỏ MI: 96.8 / 90.4 / 77.2 / 71.5 / 68.1 / 22.4 | 72.3 | 0.82
- k = 5: 96.6 / 90.9 / 78.0 / 74.4 / 69.6 / 23.9 | 74.0 | 0.84

Δ = OSFD − ô trên CrossAvg₃ [95% CI] → phân loại (quy tắc khóa):
- bỏ RRB **+32.1 [26.3, 37.8] → nguyên liệu**
- RRB bỏ xoay **+5.5 [3.3, 8.7] → nguyên liệu**
- RRB bỏ resize +0.3 [−1.4, 1.8] → không phải
- k = 5 +0.4 [−1.1, 2.0] → không phải
- task loss +0.7 [−2.8, 5.6] → mơ hồ
- RRB bỏ nhiễu +2.0 [−0.2, 4.4] → mơ hồ
- bỏ MI +2.2 [1.1, 4.1] → mơ hồ
→ Bậc 2 (đủ dev300) cho 3 ô mơ hồ: đang chạy (`results/ablation/run_dev300.log`).

Quan sát (mô tả, chưa kết luận cuối — chờ bậc 2):
- **Ưu thế của OSFD đến từ RRB, không từ loss feature:** task loss + RRB ≈ OSFD (73.8 vs 74.5). Hệ quả: OSFD thắng M-DI² (+7 CrossAvg₂ @ n300) chủ yếu vì RRB > DI, không vì loss. Bỏ RRB (1 view, không biến đổi) làm η tụt 0.83 → 0.55.
- Trong RRB: **xoay** là phần đơn lẻ quan trọng nhất; resize không cần (2 view xoay độc lập ≈ OSFD). Bỏ nhiễu làm YOLOX-S tụt mạnh (−9.2) nhưng R101 tăng (+2.4) — nhiễu có thể giúp riêng target khác pipeline (YOLOX resize 640).
- Lưu ý compute: ô bỏ RRB chỉ 1 view forward/step (OSFD 2). Nhưng 2 view giống hệt nhau cho cùng gradient với 1 view, nên phần chênh đến từ ĐA DẠNG biến đổi, không từ số forward thuần.
- **DINO-Swin-L (câu hỏi phụ):** task loss + RRB = 33.8 vs OSFD 25.3, Δ −8.5 [−11.7, −1.6] (CI loại 0) — với ca khó, loss task tốt hơn loss feature; bỏ xoay làm DINO-Swin-L tụt 6.1. Mô tả, không thuộc cổng.

---

## 2026-10-03 — Ablation OSFD bậc 2 (dev300, 3 ô mơ hồ) → kết luận Cổng 1

Run `dev300_B50_eps5` đủ 300 ảnh (`metrics_abl_dev300.json`, `generalization_abl_dev300.json`; bậc 2 chỉ attack/predict 200 ảnh mới), `scripts/ablation_screen.py --tag dev300` → `results/ablation/dev300/screen.json`.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 98.1 / 93.0 / 77.4 / 74.6 / 68.7 / 24.9 | 73.6 | 0.82
- task loss + MI + RRB: 99.3 / 95.6 / **85.0** / **78.8** / 69.4 / **34.9** | **77.7** | 0.86
- RRB bỏ nhiễu: 98.7 / 95.6 / 81.5 / 76.5 / **59.7** / 21.5 | 72.5 | 0.83
- bỏ MI: 98.6 / 91.8 / 74.9 / 70.7 / 64.1 / 21.1 | 69.9 | 0.79

Δ = OSFD − ô trên CrossAvg₃ [95% CI]: task loss **−4.1 [−5.4, +0.3]**; bỏ nhiễu +1.0 [−0.4, 2.3]; bỏ MI **+3.7 [2.5, 4.7]**.

**Cổng 1 (quyết 1 lần ở 300 ảnh, nhị phân: nguyên liệu ⇔ cận dưới ≥ 2):**
- **Nguyên liệu: RRB (bậc 1, +32.1), xoay trong RRB (bậc 1, +5.5), MI (bậc 2, +3.7).**
- Không phải: resize trong RRB, k = 5, nhiễu σ (bậc 2, cận dưới −0.4), **loss feature** (thay bằng task loss không làm yếu — ngược lại còn cao hơn 4.1 điểm, CI vừa chạm 0).

Quan sát (mô tả):
- **Task loss + MI + RRB ≥ OSFD**: CrossAvg₃ +4.1 (CI [−0.3, 5.4] theo chiều ô − OSFD), ConvNeXt +7.6, Swin +4.2, DINO-Swin-L **+10.1 [5.8, 12.1]**, R101 +2.6. Feature loss của OSFD không những không phải nguồn ưu thế mà có thể kém task loss khi đã có RRB.
- Nhiễu σ: không đóng góp CrossAvg₃ nhưng đổi hướng theo target — bỏ nhiễu YOLOX-S −9.0, R101 +2.6, ConvNeXt +4.1, DINO-Swin-L −3.4. Có thể nhiễu giúp target khác pipeline (YOLOX 640) và ca khó, hại ConvNeXt/R101.
- Bức tranh: ưu thế OSFD ≈ đa dạng biến đổi hình học (xoay) + momentum; loại loss không quan trọng (task loss thậm chí tốt hơn).

Tiếp theo theo plan: bước 2 (cơ chế, chỉ forward). Cần user quyết trước bước 3: ô task loss không phải "khuếch đại nguyên liệu" nên theo quy tắc khóa KHÔNG tự động là ứng viên.

---

## 2026-10-03 — Bước 2 plan SOTA: cơ chế từ ablation OSFD (chỉ forward, dev100)

`scripts/mech_ablation.py` → `results/ablation/mech/mech.json` (+ `mech_run.log`). 100 ảnh dev100; 8 điều kiện chính (OSFD + 7 ô) + 5 mô tả thêm (KG-E1/E2, SGM ×3). D_k trên backbone R50 / R101 / ConvNeXt-T / Swin-T / DINO-Swin-L (cùng Resize 1333×800 + normalize); phổ δ 3 dải; năng lượng δ trong GT box.

**1. Nguyên liệu đổi gì ở δ** (OSFD − ô, CI 95% bootstrap ảnh):
- **RRB:** năng lượng δ dải thấp **+0.124** (0.26 → 0.39), dải cao −0.143; D_4 trên CHÍNH R50 **−0.37** (1.86 → 1.49) nhưng D_4 trên target khác họ **tăng**: ConvNeXt +0.17, Swin-T +0.17, DINO-Swin-L +0.19; D_4/D_1 target +0.43 / +0.56.
- **Xoay:** cùng chiều, nhỏ hơn — dải thấp +0.016, D_4 R50 −0.03, D_4 target +0.035 / +0.035 / +0.059.
- **MI:** KHÁC cơ chế — dải thấp −0.026, D_4 target gần như không đổi (ConvNeXt −0.015, Swin −0.003, Swin-L +0.017). Mô tả hiện có chưa giải thích được đóng góp +3.7 của MI.

**2. Tỉ số chuyển distortion sâu D_4(target) / D_4(R50):** OSFD ConvNeXt 0.55 / Swin 0.47 / Swin-L 0.61; bỏ RRB 0.35 / 0.28 / 0.39; bỏ xoay 0.52 / 0.44 / 0.56. → RRB/xoay làm δ BỚT ĐẶC THÙ surrogate: ít phá feature sâu của R50 hơn nhưng phá feature sâu của target khác họ nhiều hơn.

**3. Theo ảnh (trong OSFD):** Spearman(D_4 trên target, retained confidence) −0.54 đến −0.69 ở ConvNeXt-T, Swin-T, YOLOX-S (D_4 R50 chỉ −0.03 đến −0.19; phổ thấp −0.15 đến −0.32) → distortion sâu TRÊN TARGET quyết định suppression; distortion trên surrogate thì không.

**4. Giữa điều kiện** (13 điều kiện, Spearman với CrossAvg₃): D_1 (mọi model) −0.80 [−0.87, −0.69]; D_4 DINO-Swin-L +0.79; phổ thấp +0.68 [0.53, 0.80]. Với 8 điều kiện chính tương quan yếu hơn (ρ D_4 Swin-L +0.86 nhưng CI rộng [0.31, 0.98]).

**Ngoại lệ — task loss:** D_k thấp hơn OSFD ở MỌI model (D_4 ConvNeXt 0.57 vs 0.82, Swin-L 0.73 vs 0.91) nhưng CrossAvg₃ ngang và DINO-Swin-L cao hơn → task loss phá feature ÍT hơn mà vẫn lừa được bằng/hơn; độ lớn distortion feature không giải thích được khác biệt GIỮA loại loss.

**Câu cơ chế (cho bước 3):** *Đa dạng biến đổi (RRB, phần lớn qua xoay) đẩy δ về tần thấp và làm nó bớt đặc thù surrogate — giảm distortion sâu trên R50, tăng distortion sâu trên target khác họ; distortion sâu trên target là thứ quyết định suppression theo từng ảnh. MI và loại loss đóng góp theo cơ chế khác, chưa đo được bằng D_k/phổ.*

---

## 2026-10-03 — Vì sao DINO-Swin-L bền: distortion qua backbone → neck → encoder (DINO-Swin-L vs DINO-R50)

`scripts/mech_swinl.py` → `results/ablation/mech/swinl.json` (+ `swinl_run.log`); DINO-R50 eval dev100 → `results/runs/dev300_B50_eps5/generalization_dino_r50_dev100.json`. 100 ảnh dev100; OSFD / task loss / bỏ RRB. Chỉ số: rel = ‖a − c‖/‖c‖ và **cosd = 1 − cos(a, c)** (không phụ thuộc thang đo kiến trúc — rel so giữa kiến trúc dễ gây hiểu nhầm).

OSFD, cosd theo tầng (cùng detector DINO, chỉ khác backbone):

| | backbone tầng cuối | neck (lớn nhất) | **encoder (memory)** | retained conf. | drop AP |
|---|---|---|---|---|---|
| DINO-R50 | **0.873** | 0.840 | **0.324** | 0.06 | 96.0 |
| DINO-Swin-L | **0.424** | 0.356 | **0.066** | 0.76 | 25.3 |
| Mask R-CNN Swin-T (tham chiếu) | 0.244 | **0.515** (FPN khuếch đại) | — | 0.32 | 74.3 |

- **Đính chính bước 2:** "D_4 Swin-L 0.91 > Swin-T 0.70" là theo rel; theo cosd, feature sâu của Swin-L vẫn còn khá thẳng hàng với bản sạch (cos ≈ 0.58), trong khi R50 gần như trực giao (cos ≈ 0.13).
- **Encoder DINO dập distortion:** tỉ lệ cosd encoder / backbone cuối: Swin-L 0.16, R50 0.37. Stage nông của Swin-L gần như không bị chạm (cosd 0.07 / 0.11).
- **Neck đối lập:** FPN của Mask R-CNN Swin-T KHUẾCH ĐẠI distortion (0.24 → 0.49–0.52); ChannelMapper + encoder của DINO-Swin-L thì giảm.
- Theo ảnh (DINO-Swin-L): distortion ở encoder dự đoán retained mạnh nhất (ρ −0.44; backbone −0.30 đến −0.36).
- **Task loss trên DINO-Swin-L:** retained như OSFD (0.76) nhưng drop AP cao hơn (33.8 vs 25.3 @ dev100) → phần lợi của task loss ở ca khó KHÔNG đến từ làm mất GT mà từ false positive / sai định vị.

**Diễn giải (giả thuyết, mô tả):** Swin-L bền do 2 yếu tố CỘNG LẠI: (1) attack từ R50 chỉ làm lệch vừa phải feature sâu của Swin-L và gần như không chạm stage nông; (2) encoder deformable của DINO (đa scale, dùng cả stage nông còn sạch) "sửa" được distortion vừa phải. Với DINO-R50, distortion gần như toàn phần nên vượt qua được encoder. → Kết luận cũ "detector không giải thích failure" (dựa trên DINO-R50 97%) cần thu hẹp: detector không gây failure KHI distortion bão hòa (cùng họ), nhưng có thể góp phần khi distortion chỉ vừa phải (khác họ). Không kiểm được trực tiếp nếu không có DINO-Swin-T (không thêm model mới — chỉ ghi nhận).

---

## 2026-10-03 — Literature + probe hướng "distortion bất biến" → KHÔNG ủng hộ

**Rà literature (web, 2026-10-03):** input-transformation cho transfer đã đông ở classification (SIA ICCV'23, BSR, SID ICCV'25, Block Stretch&Shrink 11/2025, Local Invariance 2025, Perspective-Invariant Attack 8/2026) → "tăng xoay / thêm biến đổi" không có novelty; ở detection gần như chỉ có RRB (OSFD) và AugTrans. Tấn công encoder DETR ("Corrupting Attention", arXiv 2608.06674, có vẻ white-box trên DETR) khớp phát hiện encoder là chỗ "khóa" của DINO nhưng không áp dụng được với surrogate Mask R-CNN R50. Chưa thấy method detection 1-surrogate công khai vượt OSFD (rà chưa đầy đủ). Hướng được chọn để probe (user đồng ý): "distortion bất biến" tường minh.

**Probe** (`scripts/probe_invariance.py` → `results/ablation/mech/invariance_probe.json`; 100 ảnh dev100, 13 điều kiện; 8 góc xoay ±7° áp cho cả sạch/adv, feature R50 xoay ngược về cùng khung, vùng hợp lệ chung): E = mean_t ‖Δ_t‖², S = ‖mean_t Δ_t‖², r = S/E.
- r CAO ở MỌI điều kiện (stage 4: 0.77–0.82), kể cả bỏ RRB (0.769 vs OSFD 0.815) — chênh 0.046 trong khi CrossAvg₃ chênh 32 điểm.
- Giữa điều kiện: Spearman(r stage 4, CrossAvg₃) = 0.74 (13 điểm), nhưng biên độ nhỏ; r stage 1 tương quan âm (−0.59).
- Theo ảnh (OSFD): phần chung S dự đoán retained KHÔNG tốt hơn tổng E (vd stage 2 ConvNeXt −0.35 vs −0.35; YOLOX −0.40 vs −0.36); tỉ lệ r còn yếu hơn.
→ Distortion trên surrogate vốn đã ~80% "chung giữa các góc xoay nhỏ" dù không có RRB; phần chung không phải thứ phân biệt δ transfer tốt với δ transfer kém. **Không có cơ sở cơ chế cho hướng "distortion bất biến" tường minh** (trong phạm vi probe: xoay ±7°, đo trên surrogate). Không khóa ứng viên theo hướng này.

---

## 2026-10-03 — Probe loại lỗi (mất GT / false positive / định vị) → hallucination KHÔNG qua được DINO; lợi của task loss chủ yếu là phá ĐỊNH VỊ

`scripts/probe_error_types.py` → `results/ablation/mech/error_types.json` (+ `error_types_run.log`); dev300, OSFD vs task loss + MI + RRB, từ dets có sẵn (không GPU). Score ≥ 0.3; FP tách nền (IoU < 0.1) / định vị (cùng lớp, 0.1 ≤ IoU < 0.5) / nhầm lớp; AP_noFP = AP khi bỏ mọi FP.

- **Hallucination đã là kiểu phá chính của OSFD trên target kiểu Mask R-CNN/YOLOX** — FP nền/ảnh: R50 2.9 → 41.0, R101 → 28.5, ConvNeXt → 23.4, Swin-T → 18.2, YOLOX-S 0.9 → 9.5 — **nhưng DINO-Swin-L gần như miễn nhiễm**: 1.1 → 1.8 (task loss 2.7). → hướng "hallucination chuyển giao được" KHÔNG nhắm được ca khó; bỏ.
- DINO-Swin-L, OSFD: recall@0.3 0.76 → 0.57, FP gần như không tăng — thiệt hại đến từ mất GT.
- **Task loss phá định vị mạnh hơn ở MỌI target khác họ:** AP/AP50 — ConvNeXt 0.62 (OSFD) vs **0.49**; Swin-T 0.56 vs **0.50**; YOLOX-S 0.63 vs **0.55**; DINO-Swin-L 0.74 vs **0.72**; FP định vị/ảnh cũng tăng (ConvNeXt 2.0 → 3.2, Swin-T 1.9 → 3.1, YOLOX 0.8 → 1.7, Swin-L 1.0 → 1.4). Recall@0.3 của task loss KHÔNG thấp hơn OSFD (ConvNeXt 0.32 vs 0.26; Swin-L 0.56 vs 0.57) → task loss làm mất GT ÍT hơn nhưng làm lệch box NHIỀU hơn.
- DINO-Swin-L: AP drop task loss 34.9 vs OSFD 24.9; AP50 drop 29.5 vs 20.7 — vừa định vị vừa thêm FP (nền + định vị).

**Diễn giải (giả thuyết):** "đồng tiền" chuyển giao sang target khác họ / ca khó không phải hallucination mà là **phá định vị (bbox regression)** — có thể vì tín hiệu định vị dựa trên cấu trúc biên/hình học object chung giữa các kiến trúc. Cần kiểm: thành phần nào của task loss (rpn_cls, rpn_bbox, cls, bbox, mask) gây ra lợi ích.

---

## 2026-10-03 — Ablation thành phần task loss (trước khi chạy)

User đồng ý (mở rộng phân tích bước 2, chưa phải ứng viên method): nền = task loss + MI + RRB (co-transform GT); 3 ô chỉ giữ 1 nhóm số hạng: `ABL-task-cls` (loss_rpn_cls + loss_cls), `ABL-task-box` (loss_rpn_bbox + loss_bbox), `ABL-task-mask` (loss_mask). dev100 (seed theo chỉ số dev300), đánh giá 6 model + AP/AP50. `attack/preprocess.make_gt_loss(terms)`; unit test: 5 key trùng tuyệt đối `compute_gt_loss`.

---

## 2026-10-03 — Ablation thành phần task loss (dev100) → lợi ở DINO-Swin-L đến từ số hạng PHÂN LOẠI, không từ định vị

Run `dev300_B50_eps5` subset dev100 (`metrics_task_dev100.json`, `generalization_task_dev100.json`), `scripts/ablation_screen.py --tag task_dev100` → `results/ablation/task_dev100/{screen.json, error_types.json}`.

Relative AP drop % (dev100), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- OSFD: 96.4 / 91.8 / 78.8 / 74.3 / 70.2 / 25.3 | 74.5
- task loss đủ 5 số hạng: 98.6 / 92.5 / 80.1 / 73.7 / 67.6 / **33.8** | 73.8
- chỉ cls (rpn_cls + cls): 98.7 / 90.8 / 78.7 / 75.7 / 59.1 / **32.5** | 71.2
- chỉ box (rpn_bbox + bbox): 95.4 / 83.6 / 65.8 / 61.5 / 55.8 / 25.1 | 61.0
- chỉ mask: 89.7 / 77.2 / 63.9 / 57.0 / 49.5 / 26.0 | 56.8

Loại lỗi (dev100): chỉ-box làm AP/AP50 thấp nhất (ConvNeXt 0.44, Swin-T 0.48, YOLOX 0.54) và tăng FP định vị, nhưng giữ recall cao (0.45–0.49) và gần như không tạo FP nền → **phá định vị có thật nhưng đóng góp ít vào AP drop tổng**. Chỉ-cls giữ phần lớn hiệu quả của task loss (CrossAvg₃ 71.2 vs 73.8), tạo FP nền trên target kiểu Mask R-CNN, và trên DINO-Swin-L cho AP 43.2 (≈ đủ 42.4; OSFD 47.8) với AP50 thấp nhất (57.1) mà recall như OSFD (0.56 vs 0.57) → hạ chất lượng xếp hạng score / thêm FP chứ không xóa GT.

**Đính chính giả thuyết entry trước:** "phá định vị là kênh chuyển giao" KHÔNG đứng — lợi của task loss ở ca khó đến từ **số hạng phân loại** (rpn_cls + cls). Phá định vị (box) chỉ đóng góp nhỏ.

**Bổ trợ theo target (mô tả):** loss feature (OSFD) tốt nhất ở YOLOX-S (70.2 vs cls 59.1); loss phân loại tốt nhất ở DINO-Swin-L (32.5–33.8 vs 25.3); ConvNeXt/Swin-T ngang nhau → hai loại objective có thể BỔ TRỢ nhau.

---

## 2026-10-03 — Probe bổ trợ "feature + cls" (trước khi chạy)

User đồng ý probe rẻ: `PROBE-osfd+cls` = MI + RRB (co-transform GT) + L_OSFD + λ·(loss_rpn_cls + loss_cls), 1 forward/view (feature backbone bắt bằng hook trong `model.loss`; kiểm: feature-only qua hook trùng OSFD gốc, sai số 1.5e-7). **λ = 4 khóa trước khi chạy**: median tỉ lệ mean|∇feat| / mean|∇cls| tại δ = 0 trên 5 ảnh đầu dev100 = 3.95 (3.7 / 4.57 / 3.95 / 7.7 / 2.15) → hai thành phần đóng góp gradient ngang nhau; không tune λ theo transfer. dev100, so OSFD / task loss / chỉ-cls. Câu hỏi mô tả: có giữ được YOLOX-S (feature) và tăng DINO-Swin-L (cls) cùng lúc không.

---

## 2026-10-03 — Probe "feature + cls" (dev100) → có dấu hiệu bổ trợ, chưa có ý nghĩa thống kê ở 100 ảnh

`results/ablation/probe_dev100/screen.json` (run `metrics_probe_dev100.json`, `generalization_probe_dev100.json`). Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- OSFD: 96.4 / 91.8 / 78.8 / 74.3 / **70.2** / 25.3 | 74.5
- task loss: 98.6 / 92.5 / 80.1 / 73.7 / 67.6 / **33.8** | 73.8
- chỉ cls: 98.7 / 90.8 / 78.7 / 75.7 / 59.1 / 32.5 | 71.2
- **OSFD + 4·cls: 99.3 / 94.5 / 81.1 / 80.2 / 69.9 / 30.5 | 77.0**

So OSFD: CrossAvg₃ **+2.6 [−1.7, +5.3]**, Swin-T +5.9, ConvNeXt +2.3, YOLOX-S −0.3 (giữ được), DINO-Swin-L +5.2 [−0.8, 8.4], R101 +2.7. → Bổ trợ đúng chiều dự đoán (giữ YOLOX của loss feature, tăng DINO-Swin-L/Swin-T nhờ cls) nhưng ở n = 100 CI còn chứa 0; DINO-Swin-L thấp hơn task loss / chỉ cls (30.5 vs 33.8 / 32.5).
Lưu ý: trên dev300, task loss + MI + RRB đã hơn OSFD +4.1 (dev100 chỉ −0.7) → dev100 nhiễu; ứng viên kết hợp phải vượt CẢ OSFD lẫn task loss + RRB.

---

## 2026-10-03 — Bước 3 plan SOTA: khai báo ứng viên (TRƯỚC khi chạy sàng lọc dev300)

User đồng ý. Đúng 2 ứng viên, cấu hình đóng băng, không thêm ứng viên cứu vãn:
- **C1** = task loss (5 loss Mask R-CNN) + MI μ = 1 + RRB (co-transform GT box + mask) — `ABL-task-loss` (đã có đủ dev300).
- **C2** = MI + RRB (co-transform GT) + L_OSFD + 4·(loss_rpn_cls + loss_cls), 1 forward/view — `PROBE-osfd+cls` (λ = 4 khóa trước theo tỉ lệ gradient; nối 200 ảnh còn lại của dev300, cùng seed).
Ghi chú sửa plan (user duyệt): plan yêu cầu method "khuếch đại nguyên liệu"; C1/C2 thay/bổ sung LOẠI LOSS, dựa trên phát hiện bước 2 mở rộng (feature loss mạnh ở target kiểu CNN/YOLOX, loss phân loại mạnh ở target có encoder tự sửa — DINO-Swin-L), giữ nguyên nguyên liệu RRB + MI.

**Sàng lọc @ dev300 (tiêu chí khóa plan bước 3)**, mỗi ứng viên so OSFD, paired bootstrap 1000 mẫu: QUA nếu ΔCrossAvg₃ ≥ 3 VÀ cận dưới CI > 0; Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S; drop R101 ≥ R101(OSFD) − 3. Cả 2 qua → chọn ΔCrossAvg₃ lớn hơn (chênh < 0.5 → C1, đơn giản hơn). Báo cáo kèm C2 − C1 (CI) và DINO-Swin-L. Không qua cả 2 → dừng method.

---

## 2026-10-03 — Bước 3 sàng lọc @ dev300 → **C2 (OSFD + 4·cls) QUA**, C1 không qua (CI)

Run `dev300_B50_eps5` (`metrics_step3_dev300.json`, `generalization_step3_dev300.json`); `scripts/ablation_screen.py --tag step3_dev300` + `scripts/step3_screen.py` → `results/ablation/step3_dev300/{screen.json, step3.json}`. n = 300, paired bootstrap 1000 mẫu.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 98.1 / 93.0 / 77.4 / 74.6 / 68.7 / 24.9 | 73.6 | 0.82
- C1 task loss + RRB: 99.3 / 95.6 / 85.0 / 78.8 / 69.4 / 34.9 | 77.7 | 0.86
- **C2 OSFD + 4·cls: 99.5 / 96.6 / 84.7 / 79.5 / 70.6 / 31.8 | 78.3 | 0.85**

Ứng viên − OSFD [95% CI]:
- C1: CrossAvg₃ +4.1 [−0.3, 5.4] ✗ (cận dưới ≤ 0); ConvNeXt +7.6, Swin +4.1, YOLOX +0.8, R101 +2.6, DINO-Swin-L +10.1 [5.8, 12.1] → **KHÔNG QUA**.
- **C2: CrossAvg₃ +4.7 [1.7, 6.0] ✓; ConvNeXt +7.2 [3.6, 9.1], Swin-T +4.9 [1.5, 6.5], YOLOX-S +1.9 [−2.2, 4.8] (điểm > 0 ✓), R101 +3.6 ✓, DINO-Swin-L +7.0 [2.8, 8.9] (báo cáo) → QUA.**
- C2 − C1 (điểm): CrossAvg₃ +0.6, YOLOX +1.2, DINO-Swin-L −3.1.

**Theo quy tắc khóa: chọn C2, ĐÓNG BĂNG** (MI μ = 1 + RRB θ = 7, l_s = 10, ρ = 0.8, s_max = 1.1, σ = 6 với co-transform GT + L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), 1 forward/view, B = 50, ε = 5). Bước tiếp: GO @ n300 (1 lần) so với PNG OSFD của run `n300_B50_eps5` (HF).
Ghi chú: CI bootstrap percentile lệch (điểm +4.7 gần cận trên 6.0) — cùng hiện tượng đã ghi ở baseline (bootstrap mean lệch do category hiếm); không đổi kết luận.

---

## 2026-10-03 — Bước 4: GO @ n300 cho C2 (bắt đầu chạy)

Tải `n300_B50_eps5` từ HF (token đọc từ `.env`, không in ra; `.env` thêm vào `.gitignore`), sha256 9eadf774…f8d590 khớp entry 2026-09-23; đủ 300 PNG OSFD. Chạy `PROBE-osfd+cls` (C2, đóng băng) trên n300 (seed SEED + i như baseline), OSFD dùng lại PNG baseline (không attack lại). Tiêu chí GO khóa (plan_sota.md bước 4): ΔCrossAvg₃ ≥ 3 VÀ cận dưới CI > 0; Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S VÀ ≥ 2/3 target ≥ 3; drop R101 ≥ R101(OSFD) − 3. Chạy 1 lần.

---

## 2026-10-03 — Tạm hoãn GO @ n300 (user); C2 ghi nhận tích cực; chuyển sang nhánh Swin-L

- **User quyết:** hủy GO đang chạy; **ghi nhận C2 là kết quả tích cực** (qua sàng lọc dev300: CrossAvg₃ +4.7 [1.7, 6.0]) — chạy lại GO khi cần. GO đã attack 110/300 ảnh n300 (PNG ở `artifacts/runs/n300_B50_eps5/adv/PROBE-osfd+cls/`, stats trong `results/runs/n300_B50_eps5/attack_stats.jsonl`); **CHƯA eval, chưa nhìn số nào** → GO vẫn là "chạy 1 lần" hợp lệ; chạy lại đúng lệnh trong CLAUDE.md sẽ nối tiếp.
- Việc mở cho C2 (để sau): đối chứng cân thời gian — C2 có cùng B và số view forward nhưng wall-clock ~2.3× OSFD (7.4 vs 3.3 s/ảnh, do `model.loss` chạy RPN/ROI/head); budget sweep n30 cho OSFD B50 → B100 +2.6 cross-avg → so OSFD cùng wall-clock (B ≈ 115) trên dev300, khóa trước khi chạy.
- **Nhánh mới — tấn công cả DINO-Swin-L:** chạy chẩn đoán miễn phí (chỉ forward): điểm phân loại các query top-k ở encoder DINO (cơ chế chọn query) thay đổi ra sao giữa sạch / OSFD / C1 / C2 / chỉ-cls / bỏ RRB — kiểm kênh "xếp hạng confidence" có phải đường lọt qua encoder không.
