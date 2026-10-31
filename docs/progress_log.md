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

---

## 2026-10-03 — Chẩn đoán chọn query encoder DINO → Swin-L giữ 95% GT trong top-900; ý tưởng "phá xếp hạng confidence" KHÔNG được ủng hộ

`scripts/diag_query_select.py` → `results/ablation/mech/query_select.json` (+ log); 100 ảnh dev100; tái tạo `DINO.pre_decoder` (điểm encoder s = max_c sigmoid, top-900). (Lần chạy đầu bị kill ở 20/100 khi phiên khởi động lại; chạy lại trong tmux.)

| DINO | điều kiện | top100 mean | top100 còn giữ | điểm proposal GT | GT còn trong top-900 |
|---|---|---|---|---|---|
| Swin-L | sạch | 0.130 | 1.00 | 0.521 | 0.988 |
| Swin-L | OSFD / C1 / C2 / chỉ-cls | 0.141–0.145 | 0.32–0.41 | 0.409–0.428 | **0.946–0.956** |
| Swin-L | bỏ RRB | 0.127 | 0.49 | 0.482 | 0.976 |
| R50 | sạch | 0.143 | 1.00 | 0.404 | 0.971 |
| R50 | OSFD / C1 / C2 / chỉ-cls | 0.21–0.26 | 0.04–0.18 | 0.09–0.15 | **0.52–0.66** |

- DINO-R50 thua vì GT bị đẩy KHỎI top-900 (chỉ còn ~57%) → decoder không có query để khôi phục.
- DINO-Swin-L: thứ tự query bị xáo mạnh nhưng ~95% GT vẫn vào decoder (điểm proposal GT chỉ giảm ~20%) → decoder khôi phục.
- C1 / C2 / chỉ-cls KHÔNG khác OSFD ở tầng encoder → lợi +7 đến +10 của loss phân loại trên Swin-L nằm ở decoder / điểm cuối, không ở chọn query. Ý tưởng I1 (phá xếp hạng confidence encoder) bỏ.
- Điểm nghẽn Swin-L: phải hạ điểm proposal vùng GT đủ mạnh để GT rơi khỏi top-900.

---

## 2026-10-03 — Nhánh Swin-L, probe S1: C2 + lọc thông thấp gradient (trước khi chạy)

User chọn S1 sau rà literature (ViT dựa vào thông tin tần thấp / pha tần thấp — WACV 2024 "Exploring Adversarial Robustness of ViTs in the Spectral Perspective"; COGO ICLR 2024 nhắm tần trung-thấp; TESSER làm trơn phổ) + dữ liệu của ta (RRB đẩy δ về tần thấp; phổ thấp tương quan +0.68 với CrossAvg₃ giữa điều kiện). Phần "nhắm pha" để sau nếu S1 có tín hiệu.
- `core.run_iterative_attack(grad_transform=)`; `baselines.gaussian_lowpass(g, f0)`: H(f) = exp(−f²/(2 f0²)), áp lên gradient mỗi step TRƯỚC chuẩn hóa MI.
- 2 mức liều khóa trước: `S1-lp6` (f0 = 1/6 cycles/pixel) và `S1-lp12` (f0 = 1/12), nền = C2 nguyên vẹn. Kiểm: nhiễu trắng sau lọc có 63% / 98% năng lượng ở dải thấp (trước 9%).
- Unit test: C2 tự nó KHÔNG tất định (2 lần cùng seed lệch 4.95% pixel sau 3 step — ROIAlign backward GPU; OSFD tất định); `grad_lowpass_f0=None` lệch C2 5.07% — cùng mức → tương đương.
- dev100, so OSFD / C2; báo cáo 6 model, ưu tiên DINO-Swin-L và việc giữ ConvNeXt/Swin-T/YOLOX.

---

## 2026-10-03 — Probe S1 (C2 + lọc thông thấp gradient) → ÂM, giảm đơn điệu theo liều, kể cả DINO-Swin-L

`results/ablation/s1_dev100/screen.json` (run `metrics_s1_dev100.json`, `generalization_s1_dev100.json`). Relative AP drop % (dev100), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- OSFD: 96.4 / 91.8 / 78.8 / 74.3 / 70.2 / 25.3 | 74.5
- C2: 99.3 / 94.5 / 81.1 / 80.2 / 69.9 / **30.5** | 77.0
- S1 f0 = 1/6: 98.3 / 90.5 / 74.6 / 70.0 / 64.5 / **22.7** | 69.7
- S1 f0 = 1/12: 95.9 / 78.2 / 55.6 / 56.5 / 56.7 / **15.5** | 56.2

→ Ép δ về tần thấp bằng lọc gradient làm attack YẾU đi đơn điệu theo liều ở MỌI target, kể cả DINO-Swin-L (30.5 → 22.7 → 15.5). Tương quan "phổ thấp ↔ transfer" giữa điều kiện (bước 2) KHÔNG phải nhân quả theo kiểu can thiệp này: tần thấp của RRB là hệ quả đi kèm, không phải nguồn lợi; bỏ tần trung/cao làm mất thông tin hữu ích. Nhánh S1 (kể cả biến thể "nhắm pha" chưa làm) KHÔNG đi tiếp.

---

## 2026-10-04 — Phân tích vì sao OSFD/C2 không thắng DINO-Swin-L + chốt quy tắc hướng "Repair-Robust Attack" (TRƯỚC khi code/chạy)

> **Trạng thái: LOCKED (user duyệt 2026-10-04).** Không đổi định nghĩa/ngưỡng sau khi thấy kết quả. Thứ tự ưu tiên: R-a chính, R-b dự phòng (khai báo ngay bây giờ, KHÔNG phải ứng viên cứu vãn).

### Phân tích (từ số đã commit: `results/ablation/mech/{swinl,query_select,error_types}.json`, dev100)

- **Điểm nghẽn ở sau backbone, không ở backbone:** cosd cuối backbone Swin-T (Mask R-CNN) **0.244 < Swin-L 0.424**, nhưng drop Swin-T 74% vs Swin-L 25%. FPN của Swin-T khuếch đại (→ 0.49–0.52); encoder DINO dập (Swin-L 0.424 → **0.066**, ×6.4; DINO-R50 0.873 → 0.324, ×2.7).
- **Stage nông Swin-L gần như sạch:** cosd stride 4 / 8 = 0.067 / 0.108 (OSFD; task loss, bỏ RRB tương tự) → encoder deformable đa scale có "mỏ neo sạch" để sửa distortion vừa phải ở stage sâu. Theo ảnh, distortion encoder dự đoán retained mạnh nhất (ρ −0.44).
- **Chọn query không bị phá:** GT còn trong top-900 0.95–0.96 ở mọi method (DINO-R50: 0.52–0.66); điểm proposal GT 0.52 → 0.41–0.43.
- **Budget hallucination lãng phí với DINO:** FP nền/ảnh trên Swin-L 1.1 → 1.8 (Mask R-CNN target 18–41). `loss_rpn_cls + loss_cls` của C2 (ascent) đẩy CẢ mẫu âm thành foreground.
- **Lợi của loss phân loại nằm sau encoder:** task loss có cosd encoder 0.033 (< OSFD 0.066) nhưng drop cao hơn (33.8 vs 25.3).
- Đã loại: trọng số không gian, gộp lõm channel, crop cặp KG, SGM, ép tần thấp (S1), distortion bất biến tường minh.
- Caveat giữ nguyên: sức bền Swin-L phần lớn gắn với scale + pretrain IN-22k/384 → "thắng Swin-L" = thắng ca generalization khó, không phải bằng chứng nhân quả họ backbone.

**Giả thuyết (chưa là cơ chế):** encoder DINO hoạt động như một bộ lọc nhiễu ngầm dùng các mức nông còn sạch. Mô phỏng phép "sửa" đó NGAY TRÊN surrogate R50 và tối ưu δ để tín hiệu phân loại sống sót qua phép sửa (EOT trên một "defense" là thiên kiến kiến trúc của target). Vẫn đúng threat model (1 surrogate, 0 query, không dùng weight target). Khác OSFD (đa dạng đầu vào) và C2 (trộn loss): đa dạng trong không gian FEATURE theo mô hình phép sửa. Rủi ro novelty: Clean Feature Mixup (CVPR'23), Ghost Networks — phải rà literature trước GO @ n300.

### Probe phụ V — loss phân loại "chỉ đẩy GT về nền" (vanishing-only)

- **Mục đích:** tách hiệu ứng "bỏ budget hallucination" khỏi hiệu ứng phép sửa; quyết định thành phần phân loại cho R-a. Không phải đóng góp novelty.
- **Định nghĩa:** L_V = −[ mean_{a∈P_rpn} BCE(o_a, 0) + mean_{r∈P_roi} CE(z_r, background) ]; P = mọi anchor/RoI dương do assigner train_cfg của surrogate gán (RPN IoU ≥ 0.7, RCNN IoU ≥ 0.5, kể cả GT-as-proposal), KHÔNG lấy mẫu âm, KHÔNG có số hạng đẩy mẫu âm. Ascent trên L_V = kéo objectness/lớp của vùng GT về nền.
- **Phương thức:** C2 thay `(loss_rpn_cls + loss_cls)` bằng L_V; mọi thứ khác giữ C2 (MI μ = 1, RRB đủ + co-transform GT, L_OSFD k = 3, 1 forward/view, B = 50, ε = 5, α = 1, noise init 0, seed ảnh SEED + chỉ số dev300).
- **λ_V theo ĐÚNG luật đã dùng cho C2** (khóa trước): median tỉ lệ mean|∇ₓL_OSFD| / mean|∇ₓL_V| tại δ = 0 trên 5 ảnh đầu dev100, làm tròn số nguyên gần nhất (≥ 1); ghi số vào log trước khi chạy attack.
- **Unit test:** P khớp tập dương của assigner chuẩn; L_V tại δ = 0 hữu hạn; ảnh không có mẫu dương → L_V = 0 (chỉ còn L_OSFD).
- **Chạy dev100**, so C2 (cùng seed). **Quy tắc chọn (điểm ước lượng):** dùng L_V làm thành phần phân loại cho R-a nếu Δ DINO-Swin-L(V − C2) ≥ +2 VÀ ΔCrossAvg₃(V − C2) ≥ −2; ngược lại R-a dùng `loss_rpn_cls + loss_cls` của C2. Báo cáo CI + loại lỗi (FP nền, recall@0.3) trên 6 model.

### R-a — thay mức nông bằng bản sạch trước FPN (ứng viên chính)

- **Phép sửa R_a:** sau backbone R50, thay C2, C3 (stride 4, 8 — ứng với stage Swin-L cosd 0.067 / 0.108) bằng feature SẠCH (detach) rồi mới qua FPN → RPN/ROI. C4, C5 giữ adv.
- **Objective:** L = L_OSFD(backbone adv, đủ 4 stage, k = 3, KHÔNG qua phép sửa) + λ·L_cls(head(FPN(R_a(C_adv)))), L_cls = thành phần chọn theo probe V; λ thừa kế (tại δ = 0 phép sửa không đổi gì → luật λ cho cùng giá trị).
- **Ngẫu nhiên hóa (liều):** mỗi step 1 lần Bernoulli(p) (numpy RNG riêng, seed SEED + i, không đụng RNG của RRB); trúng → thay CẢ C2 và C3 ở mọi view của step đó. **Đúng 2 liều: p = 0.5, p = 1.0** (p = 0 = nền đã có).
- **Feature sạch đúng hình học view:** C2/C3 sạch tính trên ảnh SẠCH đã áp CÙNG phép xoay/resize của view RRB (không cộng nhiễu σ), no_grad; forward thêm không tính vào B — báo cáo số forward no_grad + runtime.
- **Sanity:** p = 0 trùng nền (C2 hoặc V) trong sai số không tất định đã biết (~5% pixel sau 3 step); tại δ = 0 loss R_a = loss nền; L∞ = 5.

### Cổng độ trung thực (forward-only, TRƯỚC mọi attack R-a)

PNG OSFD + C2 trên dev100 (sinh lại nếu thiếu, cùng seed) + dets DINO-Swin-L.
- Q_rep(ảnh) = retained confidence (định nghĩa `attack/mechanism.py`: GT detect được ở ảnh sạch, score ≥ 0.3, IoU ≥ 0.5) của **R50 khi C2, C3 thay bằng sạch** (p = 1, không RRB); Q_plain = cùng chỉ số với R50 thường.
- **QUA nếu ở CẢ OSFD và C2:** Spearman(Q_rep, retained DINO-Swin-L) ≥ +0.30 VÀ ρ(Q_rep) − ρ(Q_plain) ≥ +0.10 với cận dưới CI 95% paired bootstrap (1000 mẫu, BOOT_SEED 2026) > 0.
- Mô tả kèm: mean Q_rep vs Q_plain, cosd FPN có/không sửa, tương quan với Swin-T / ConvNeXt-T / YOLOX-S.
- **Trượt → R-b** (dự phòng, cùng cổng, q = 0.3). R-b cũng trượt → dừng hướng Repair-Robust.

### R-b — trộn sạch theo ô không gian (DỰ PHÒNG, chỉ khi R-a trượt cổng độ trung thực)

Chia khung model thành ô 64×64 px (căn theo stride: C2 16×16, C3 8×8, C4 4×4, C5 2×2 cell); mỗi ô thay bằng sạch với xác suất q, độc lập, CÙNG mặt nạ (theo tọa độ ảnh) cho cả 4 stage, rồi qua FPN → head. Objective/λ/RNG như R-a. Liều q ∈ {0.3, 0.5}. Nếu R-a qua cổng nhưng trượt các bước sau → KHÔNG chạy R-b.

### Liều–đáp ứng @ dev100 (R-a, hoặc R-b nếu thay)

So nền (p = 0) cùng seed. Đi tiếp nếu: ≥ 1 liều có Δ DINO-Swin-L ≥ +3 (điểm) VÀ liều đó có ΔCrossAvg₃ ≥ −2 VÀ ΔR101 ≥ −3. Chọn liều có DINO-Swin-L cao hơn (chênh < 1 → p = 0.5). Không qua → dừng hướng.

### Sàng lọc @ dev300 (1 lần, cấu hình đóng băng)

QUA nếu đồng thời:
- Δ DINO-Swin-L(ứng viên − C2) ≥ 3 VÀ cận dưới CI 95% > 0;
- ΔCrossAvg₃(ứng viên − C2) ≥ −1.5 (không thua C2, điểm);
- ΔCrossAvg₃(ứng viên − OSFD) ≥ 3 VÀ cận dưới CI > 0; Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S so OSFD;
- drop R101 ≥ R101(OSFD) − 3.
Không qua → dừng, ghi log, không biến thể mới. Qua → rà literature (CFM, Ghost Networks, feature-level augmentation cho detection) rồi GO @ n300 (cần hoàn tất C2 @ n300 trước để có mốc C2; tiêu chí GO khóa riêng trước khi chạy).

---

## 2026-10-04 — Repair-Robust: code + unit test + hiệu chỉnh λ; đổi cách làm tròn λ_V (user duyệt, TRƯỚC khi chạy V)

- **Môi trường:** máy mới RTX 3090, bootstrap xong; `download_checkpoints.sh --gen` đủ 7 checkpoint Generalization Panel; COCO val2017 5000 ảnh. Không có `artifacts/` (PNG dev300 cũ chưa từng lên HF → mất) → sinh lại PNG OSFD + C2 trên dev100 (cùng seed theo chỉ số dev300).
- **Code:** `attack/methods/repair.py` (`vanish_loss` = L_V; `make_view_loss` hook backbone: L_OSFD trên C_adv, nhánh head qua R-a nếu có; `repair_robust_attack`); `rrb_views(companion=)` áp cùng phép xoay/resize lên ảnh sạch, không nhiễu, không tiêu thụ RNG.
- **Unit test** (`scripts/rr_unit_test.py`) ĐẠT: rrb_views mặc định trùng tuyệt đối HEAD (3 ảnh × 3 seed, cả with_gt); companion không đổi RNG và khớp hình học; `make_view_loss(c2)` trùng `make_feature_plus_gt_loss` (rtol 1e-6); L_V hữu hạn, ≤ 0, có anchor/RoI dương (vd 22/56, 14/29, 6/12), ảnh không GT → 0; R-a tại δ = 0, σ = 0 trùng không sửa (cả c2 và vanish); attack p = 0/0.5/1 chạy, đếm forward no_grad = 2 × số step sửa.
- **Hiệu chỉnh λ** (`scripts/rr_calibrate_lambda.py` → `results/repair_robust/lambda_calibration.json`): áp luật cho C2 tái tạo được — 3.88 / 3.70 / 3.55 / 7.89 / 2.19, median 3.70 → λ = 4 (log cũ 3.95 → 4; lệch nhỏ do không tất định + resize khả vi). L_V: 0.28 / 0.19 / 6.79 / 0.14 / 0.31, median **0.28**.
- **Đổi luật làm tròn (user duyệt, chưa có số transfer nào):** luật khóa "làm tròn số nguyên, ≥ 1" cho λ_V = 1 → gradient L_V ≈ 3.6× L_OSFD, trái ý định cân bằng 1:1 của luật C2 (không lường trước tỉ lệ < 1). Sửa: **làm tròn 1 chữ số có nghĩa → λ_V = 0.3** (C2 vẫn = 4, không đổi). Ảnh 3 (11615, 6.79): R50 vốn không phát hiện GT duy nhất (objectness 0.027) → L_V gần 0; median không bị ảnh hưởng.
- λ của R-a thừa kế từ thành phần phân loại được chọn (C2: 4, V: 0.3).

---

## 2026-10-04 — Repair-Robust: probe V KHÔNG chọn; cổng độ trung thực R-a TRƯỢT → chuyển R-b (theo quy tắc khóa)

Sự cố vận hành: pipeline chờ bằng `tmux has-session -t rr` khớp TIỀN TỐ với chính session `rr2` → treo ~45 phút; sửa bằng khớp chính xác (`-t =name`). Không ảnh hưởng kết quả.

**PNG dev100 sinh lại** (seed theo chỉ số dev300): OSFD tái tạo R50 / R101 trùng tuyệt đối số đã commit (96.4 / 91.8); ConvNeXt-T 78.4 (cũ 78.8), Swin-T 74.8 (74.3), YOLOX-S 69.8 (70.2), DINO-Swin-L 24.0 (25.3) — lệch ≤ 1.3, mức không tất định của suy luận target. Runtime: OSFD ~3.3 s/ảnh, C2 ~8.9 s/ảnh, V tương tự C2.

**Probe V** (`results/repair_robust/v_dev100/screen.json`; run `metrics_rr_dev100.json`, `generalization_rr_dev100.json`). Relative AP drop % dev100, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- C2: 99.3 / 94.0 / 81.2 / 79.4 / 69.1 / 31.4 | 76.6
- V (λ_V = 0.3): 97.2 / 94.3 / 80.6 / 79.4 / 70.9 / 30.6 | 77.0
- OSFD: 96.4 / 91.8 / 78.4 / 74.8 / 69.8 / 24.0 | 74.3

V − C2: CrossAvg₃ +0.4 [−1.9, 3.3], DINO-Swin-L −0.8 [−5.1, 3.6], YOLOX-S +1.8. **Quy tắc khóa (Δ Swin-L ≥ +2) → KHÔNG chọn; R-a/R-b dùng loss phân loại của C2 (λ = 4).** Mô tả: V làm đúng cơ chế dự định — FP nền/ảnh Swin-T 24.4 → 16.8, YOLOX 11.4 → 7.6, DINO-Swin-L 3.0 → 1.7; recall@0.3 thấp hơn (Swin-T 0.35 → 0.26; DINO-Swin-L 0.57 → 0.52) — nhưng DINO-Swin-L AP drop không tăng → budget hallucination KHÔNG phải yếu tố giới hạn trên Swin-L.

**Cổng độ trung thực R-a** (`scripts/rr_fidelity_gate.py` → `results/repair_robust/fidelity_gate_Ra.json`) — **TRƯỢT ở cả OSFD và C2, không sát ngưỡng:**
- OSFD: ρ(Q_rep, retained Swin-L) +0.15 [−0.04, 0.33], ρ(Q_plain) +0.15, chênh +0.00 [−0.02, 0.03].
- C2: ρ_rep +0.12, ρ_plain +0.08, chênh +0.04 [−0.01, 0.13].
- Mô tả: thay C2/C3 bằng sạch gần như KHÔNG đổi R50 — retained 0.029 → 0.031 (OSFD), 0.047 → 0.055 (C2); cosd FPN 0.772 → 0.756, 0.647 → 0.631. Q_plain tương quan tốt với target Mask R-CNN (R101 +0.60, ConvNeXt +0.47, Swin-T +0.37) và YOLOX (+0.37) nhưng yếu với DINO-Swin-L (+0.08 đến +0.15).
- Diễn giải: R50 Mask R-CNN không có cơ chế "sửa" dựa vào mức nông — thiệt hại nằm ở C4/C5 và đi thẳng qua FPN top-down; phép thay C2/C3 không mô phỏng được encoder DINO.

**Theo quy tắc khóa:** R-a trượt cổng → chạy **R-b** (ô 64×64 thay sạch, cùng mặt nạ cho 4 stage, q = 0.3) qua CÙNG cổng. Mặt nạ cổng: 1 lần rút mỗi ảnh, seed = SEED + chỉ số dev300 (numpy). R-b trượt → dừng hướng Repair-Robust.

---

## 2026-10-04 — Dừng nhánh Repair-Robust (user); pilot xu hướng "đa dạng hình học kiểu PIA" @ dev30 — quy tắc khóa TRƯỚC khi code/chạy

**Dừng Repair-Robust (user):** cổng R-b bị dừng giữa chừng theo yêu cầu user — KHÔNG có kết quả, không kết luận về R-b. Trọng tâm chuyển sang phân tích OSFD: chấp nhận (user) "RRB đóng góp phần lớn; loss feature không phải nguồn ưu thế, có target kém hơn task loss (DINO-Swin-L −10, ConvNeXt −7.6 @ dev300) nhưng bổ trợ loss phân loại".

**Bỏ BSR trước khi code (user):** bản nháp đầu của entry này khai báo pilot BSR (GEO-B1/B2, chưa code, chưa chạy, chưa commit) — thay bằng PIA. Lý do: ablation cho thấy nguyên liệu là XOAY (biến đổi toàn cục); phép chiếu phối cảnh ⊃ affine ⊃ similarity ⊃ xoay → PIA là mở rộng trực tiếp nguyên liệu đã có tác dụng ("khuếch đại nguyên liệu", plan_sota bước 3), còn shuffle khối của BSR phá cấu trúc object (khó co-transform GT cho detection).

**Câu hỏi:** thay RRB bằng biến đổi phối cảnh đa bậc tự do kiểu PIA (Liang, Cao, Xiao — "Perspective-Invariant Attack…", arXiv 2608.15115, IEEE TIFS 2026) có tăng transfer trên detection không; phần lợi (nếu có) đến từ số hạng phối cảnh hay chỉ từ affine mạnh; liều δ_max của paper (cho ảnh phân loại 299×299) có hợp với detection không. Port PIA sang detection = BASELINE, không phải đóng góp; phát hiện riêng cho detection chỉ có thể là đường liều / vai trò số hạng phối cảnh.

**dev30 (user chọn — CHỈ xem xu hướng, mô tả):** = 30 ảnh đầu của dev100 theo thứ tự dev300 (thứ tự eval), `data/image_lists/dev30.csv`, ⊂ dev100 ⊂ dev300. **Không ra quyết định GO/sàng lọc ở dev30** (CI hiệu 2 method ở n = 30 ước ±5 điểm); ứng viên muốn đi tiếp phải khóa tiêu chí riêng rồi chạy dev100/dev300.

**Nền chung (mọi ô):** loss feature OSFD (backbone R50, 4 stage, k = 3), MI μ = 1, ε = 5, α = 1, noise init 0, 50 step, seed ảnh SEED + chỉ số dev300; mỗi step 2 view, 1 backward (trừ ô N20), mỗi view cộng nhiễu Gaussian σ = 6 như RRB (giữ để chỉ HÌNH HỌC khác nhau). **Feature sạch co-transform:** mỗi view so với k·F_clean(T(x_clean)) — cùng tham số hình học, không nhiễu, no_grad (forward thêm không tính B, báo cáo). Lý do: phối cảnh δ_max = 0.6 dời góc ảnh tới 30% cạnh → so với F_clean chưa biến đổi là vô nghĩa.

| Ô | Biến đổi mỗi view | Câu hỏi |
|---|---|---|
| OSFD | RRB gốc, F_clean KHÔNG co-transform (PNG có sẵn) | mốc |
| GEO-C0 | RRB gốc (chain xoay → resize, nhiễu), F_clean co-transform | đối chứng: riêng việc co-transform F_clean đổi OSFD bao nhiêu |
| PIA | PIA đúng tham số paper: δ_max = 0.6, π = (π₄, π₃, π₂, π₁) = (0.4, 0.3, 0.2, 0.1) | PIA có thắng RRB cùng compute? |
| PIA-d0.3 | như PIA, δ_max = 0.3 | liều: ảnh detection nhiều object nhỏ / sát biên — 0.6 có quá mạnh? |
| PIA-aff | như PIA nhưng π = (0, 1, 0, 0) (chỉ mode 3 = affine 6-DOF), δ_max = 0.6 | lợi đến từ số hạng phối cảnh (h31, h32) hay chỉ affine? |
| PIA-N20 (mô tả) | PIA tham số paper, N = 20 view/step như paper, gradient cộng dồn 4 cụm × 5 view → 4 backward/step (B thực = 200, view forward = 1000) | trần khi bỏ ràng buộc compute; báo kèm runtime |

**Biến đổi PIA (port theo paper, không có code chính thức trong paper):** khung = ảnh ở khung model (img_shape, trước pad). 4 góc ảnh ra (TL, TR, BR, BL) ánh xạ về 4 điểm nguồn trong ảnh vào, điểm nguồn = góc + (int(ξ_x·δ_max·w/2), int(ξ_y·δ_max·h/2)), ξ ~ U(−1, 1), được phép ra ngoài ảnh. Mode rút theo π mỗi view: mode 4 — 4 góc độc lập (8-DOF); mode 3 — 3 góc độc lập (góc thiếu chọn đều ngẫu nhiên), góc còn lại hoàn thiện hình bình hành (6-DOF); mode 2 — TL, BR độc lập, TR/BL dựng hình chữ nhật trục (4-DOF); mode 1 — 1 vector dịch chung (2-DOF). Giải homography 8 ẩn (float64), warp bằng `grid_sample` bilinear khả vi, vùng ngoài ảnh nguồn lấp 0 (= "crop" về cỡ gốc). RNG: `random` của Python cho mode/góc, torch cho nhiễu σ (cùng nguồn như RRB). Tham số (δ_max, π, N) lấy từ paper, KHÔNG tune; δ_max = 0.3 và π affine là 2 ô liều/đối chứng khai báo trước. Lệch đã biết: paper không có nhiễu σ (ta giữ σ = 6 ở mọi ô GEO để chỉ hình học khác OSFD); paper N = 20 (ô chính dùng N = 2 để cùng compute với OSFD).

**Báo cáo (mô tả):** relative AP drop 6 model, CrossAvg₃, Δ so OSFD và so GEO-C0 (paired bootstrap, chỉ để thấy độ rộng CI), runtime, số view forward, số forward no_grad.

---

## 2026-10-04 — Pilot PIA: code + unit test (TRƯỚC khi chạy)

- `attack/methods/pia.py`: `sample_source_corners` (multi-DOF, π, δ_max), `homography` (8 ẩn, float64), `warp_grid`/`warp` (grid_sample bilinear, ngoài = 0), `pia_views` (companion sạch cùng phép warp, không nhiễu), `geo_osfd_attack` (geo = rrb → GEO-C0 | pia; feature sạch co-transform mỗi view, no_grad). `core.run_iterative_attack(backward_per_step=)` cộng gradient nhiều backward/step (chỉ PIA-N20); mặc định 1.
- `scripts/run_baselines.py`: ô `GEO-C0`, `PIA`, `PIA-d0.3`, `PIA-aff`, `PIA-N20`; stats ghi `forward_no_grad`, `mode_frac`, `backward` thực (N20 = 4 × B).
- **Unit test** (`scripts/pia_unit_test.py`) ĐẠT: core mặc định trùng tuyệt đối HEAD (OSFD 3 step, 2 ảnh); H·dst = src và ràng buộc DOF đúng từng mode (1: tịnh tiến; 2: không xoay/không phối cảnh; 3: h31 = h32 = 0; 4: có phối cảnh); warp đồng nhất lệch ≤ 6.6e-3, tịnh tiến đúng chiều, ngoài nguồn = 0; tần suất mode (20000 mẫu) 0.398 / 0.305 / 0.193 / 0.103 ≈ π; gradient 4 view × 1 backward = 2 × 2 view (lệch tương đối 8.8e-8); σ = 0 → view == companion (PIA, GEO-C0); 5 ô chạy được, L∞ đúng, forward no_grad = step × view. Ghi chú: lần chạy test ĐẦU TIÊN sau khi khởi động có 1 lệch "core == HEAD"; kiểm riêng HEAD/HEAD, mới/mới, HEAD/mới đều trùng tuyệt đối, 2 lần chạy lại toàn bộ đều đạt → không tất định lúc khởi động GPU, không do sửa code.

---

## 2026-10-04 — Pilot PIA @ dev30 (mô tả, KHÔNG quyết định) → PIA ≈ RRB trên CrossAvg₃; tín hiệu duy nhất ở DINO-Swin-L

User bỏ ô PIA-N20 giữa chừng (để có số sớm; 15/30 ảnh GEO-C0 đã có từ lần chạy đầu, giữ nguyên, cùng seed). Run `dev300_B50_eps5` subset dev30 (`metrics_pia_dev30.json`, `generalization_pia_dev30.json`), `scripts/ablation_screen.py --subset dev30 --tag pia_dev30` → `results/ablation/pia_dev30/screen.json` (cột "verdict" của script là quy tắc ablation bước 1 — KHÔNG áp dụng cho pilot này). Log `results/pia_dev30/run.log`. n = 30, paired bootstrap 1000.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8 | 0.86
- GEO-C0 (RRB + co-transform): 97.8 / 91.8 / 78.2 / 80.8 / 78.2 / 21.5 | 79.0 | 0.87
- PIA (δ 0.6, π paper): 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / **28.8** | 79.5 | 0.89
- PIA-d0.3: 97.4 / 91.0 / 82.7 / 79.8 / 77.0 / 24.7 | 79.8 | 0.89
- PIA-aff (chỉ 6-DOF): 95.8 / 88.5 / 73.1 / 78.8 / 76.5 / 27.9 | 76.1 | 0.86

Runtime ~5.5–6.0 s/ảnh mọi ô GEO/PIA (OSFD ~3.3; thêm 2 forward no_grad/step cho feature sạch co-transform). Tỉ lệ mode 4 thực tế 0.39 (≈ π₄).

Mô tả (n = 30, CI hiệu ước ±5):
- **Thay RRB bằng PIA cùng compute: CrossAvg₃ ngang** (PIA − GEO-C0 +0.5; − OSFD +0.7 [−1.8, 5.1]). Không có lợi trung bình.
- **DINO-Swin-L:** PIA − GEO-C0 **+7.2 [−1.0, 13.1]**, nhưng GEO-C0 − OSFD −5.2 [−9.8, 0.4] (co-transform có vẻ HẠI Swin-L) → PIA − OSFD chỉ +2.0 [−4.4, 8.3]. Chưa tách được "PIA giúp" với "co-transform hại".
- R101 giảm theo độ mạnh hình học (OSFD 93.8 → PIA 89.9 → PIA-aff 88.5) trong khi cross ngang → η tăng nhẹ (0.86 → 0.89): δ bớt đặc thù surrogate, cùng chiều cơ chế của RRB.
- δ 0.3 vs 0.6: CrossAvg₃ ngang; DINO-Swin-L thấp hơn ở 0.3 (24.7 vs 28.8).
- Chỉ affine: CrossAvg₃ thấp hơn PIA 3.4, chủ yếu ConvNeXt (73.1 vs 79.7) → số hạng phối cảnh có thể có ích cho ConvNeXt; DINO-Swin-L ngang.
- Đọc chung: giống kết luận về loss — có đa dạng hình học TOÀN CỤC là đủ, LOẠI biến đổi (xoay+resize vs phối cảnh) gần như không đổi trung bình. Muốn đi tiếp với PIA phải khóa tiêu chí (trọng tâm DINO-Swin-L, đối chứng cả OSFD lẫn GEO-C0) rồi chạy dev100.

---

## 2026-10-04 — Pilot PIA bổ sung @ dev30: C2-PIA + PIA-noCT (user) — quy tắc khóa TRƯỚC khi code/chạy

Vẫn dev30, CHỈ mô tả, không quyết định. Nền như entry PIA (MI μ = 1, ε = 5, α = 1, 50 step, 2 view/step, 1 backward, σ = 6, PIA δ_max = 0.6, π paper, seed SEED + chỉ số dev300).

| Ô | Định nghĩa | Câu hỏi |
|---|---|---|
| C2-PIA | C2 (L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), 1 forward/view, feature bắt bằng hook) với RRB thay bằng PIA; feature sạch co-transform như ô PIA; GT co-transform qua homography | lợi DINO-Swin-L của PIA và của loss cls có cộng được không? so C2 (PNG có sẵn) và PIA |
| PIA-noCT | ô PIA nhưng feature sạch KHÔNG co-transform (so với F_clean của ảnh sạch chưa biến đổi, như OSFD gốc) | DINO-Swin-L +7.2 (PIA − GEO-C0) là nhờ PIA hay do co-transform hạ GEO-C0? |

**Co-transform GT qua homography (C2-PIA):** H ánh xạ pixel ra → nguồn, nên box (khung vào) đi qua H⁻¹: 4 góc mỗi box → AABB → clip vào khung; mask warp bằng CÙNG lưới grid_sample với ảnh (threshold 0.5 như `with_gt`). Instance có box sau clip rộng hoặc cao < 1 px (object bị đẩy ra khỏi khung — δ_max = 0.6 dời góc tới 30%) hoặc mẫu số homography ≤ 0 ở góc box → BỎ instance đó (box, nhãn, mask) cho view đó; ảnh mất hết GT → chỉ còn loss feature + loss cls với toàn mẫu âm (như model.loss xử lý). Báo cáo tỉ lệ instance bị bỏ.

**Unit test trước khi chạy:** (1) ô GEO-C0/PIA/PIA-d0.3/PIA-aff trùng tuyệt đối code HEAD (cờ mặc định); (2) tịnh tiến nguyên: box dời đúng (dx, dy); (3) IoU(AABB của mask đã warp, box đã warp) — so với mức của xoay RRB (0.834, entry ablation); (4) PIA-noCT: 0 forward no_grad, loss tại view = loss OSFD; (5) C2-PIA tại cờ cls_weight = 0 trùng ô PIA về loss; không NaN trên 2 ảnh × 3 step; L∞ đúng.

**Báo cáo:** 6 model, CrossAvg₃, so OSFD / C2 / PIA / GEO-C0 (paired bootstrap, chỉ để thấy CI), runtime, tỉ lệ instance bị bỏ.

**Code + unit test (TRƯỚC khi chạy):** `pia.warp_boxes` (H⁻¹, AABB, clip, bỏ < 1 px / mẫu số ≤ 0), `pia.gt_with_keep`, `pia_views(data_sample=)`; `geo_osfd_attack(co_transform=, cls_weight=)`; ô `C2-PIA`, `PIA-noCT` trong `run_baselines.py` (stats thêm `gt_kept_frac`). `scripts/pia_c2_unit_test.py` ĐẠT: 4 ô cũ trùng tuyệt đối HEAD (3 step); tịnh tiến → box dời (−dx, −dy); đẩy ra ngoài → bỏ hết; IoU(AABB mask warp, box warp) view PIA mean **0.823** (trung vị 0.839, p5 0.580; không biến đổi 0.951; xoay RRB 0.834) → co-transform GT dùng được; 159/180 instance giữ (12% bị bỏ); PIA-noCT 0 forward no_grad, loss == loss OSFD; make_view_loss(c2, 0) == loss feature; GT rỗng → loss hữu hạn; C2-PIA 3 step × 2 ảnh hữu hạn, L∞ đúng.

**Kết quả @ dev30 (mô tả, KHÔNG quyết định)** — `metrics_pia2_dev30.json`, `generalization_pia2_dev30.json`, `results/ablation/pia2_dev30/screen.json`, log `results/pia_dev30/run2.log`. C2 = PNG `PROBE-osfd+cls` sinh lại trên dev100 (cùng seed).

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8 | 0.86
- C2: 99.9 / 94.7 / 84.6 / 82.6 / 67.8 / 26.1 | 78.3 | 0.88
- GEO-C0: 97.8 / 91.8 / 78.2 / 80.8 / 78.2 / 21.5 | 79.0 | 0.87
- PIA: 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / 28.8 | 79.5 | 0.89
- **C2-PIA: 99.5 / 94.8 / 86.8 / 85.9 / 78.2 / 34.7 | 83.6 | 0.91**
- PIA-noCT: 53.6 / 47.6 / 32.7 / 40.0 / 37.0 / 5.3 | 36.6 | 0.76

Runtime: C2-PIA 11.5 s/ảnh, PIA-noCT 3.7. C2-PIA giữ trung bình 90.7% instance GT mỗi view.

- **C2-PIA − OSFD: CrossAvg₃ +4.8 [1.5, 10.6], DINO-Swin-L +7.9 [1.6, 13.8]**, ConvNeXt +3.3, Swin-T +8.6, YOLOX +2.5, R101 +1.0. So C2: YOLOX +10.4 [2.8, 19.1], DINO-Swin-L +8.7 [0.3, 13.6]. So PIA: DINO-Swin-L +5.9 [1.6, 9.9], CrossAvg₃ +4.1. → Loss cls và PIA CỘNG được: C2-PIA cao nhất ở mọi target khác họ, và lấy lại YOLOX mà C2 (RRB) mất trên dev30.
- **PIA-noCT sụp** (CrossAvg₃ 36.6, kể cả white-box R50 53.6): với biến đổi mạnh, so feature adv đã warp với feature sạch CHƯA warp làm loss gần như vô nghĩa → co-transform là BẮT BUỘC cho PIA; không phải co-transform "hại" (câu hỏi gây nhiễu của entry trước được trả lời: +7.2 DINO-Swin-L của PIA − GEO-C0 không phải do so với một đối chứng bị hạ giả tạo; co-transform cần cho PIA, còn với RRB biên độ nhỏ thì không cần).
- Tương tác loss × biến đổi: với RRB, thêm cls gần như không đổi CrossAvg₃ trên dev30 (C2 − OSFD −0.5; dev300 +4.7); với PIA, thêm cls +4.1 CrossAvg₃ và +5.9 DINO-Swin-L.
- Lưu ý: C2-PIA khác C2 ở 2 chỗ (RRB → PIA VÀ co-transform feature sạch). n = 30, CI rộng. Muốn đi tiếp: khóa tiêu chí sàng lọc (so OSFD và C2) rồi chạy dev100 → dev300.

---

## 2026-10-04 — Pilot MORGD @ dev30 (user: xem xu hướng trước) — quy tắc khóa TRƯỚC khi code/chạy

**Bối cảnh:** đề xuất MORGD (Multi-Object Relative Geometry Diversification — biến dạng mượt toàn ảnh, điều kiện theo từng object, tạo đa dạng vị trí/tỉ lệ TƯƠNG ĐỐI giữa các object) đến từ trao đổi với một trợ lý khác, không có paper/code. Prior yếu (đã nêu với user): pilot KG (quan hệ object) và A′1 (trọng số theo box) đều NO-GO; loại biến đổi toàn cục ít ảnh hưởng (RRB ≈ PIA); paper PIA: warp phi cứng (Elastic 23.0, TPS 37.5) thua phối cảnh (49.3) ở classification. Câu hỏi ở dev30 (CHỈ mô tả): MORGD có tiến tới / vượt C2-PIA không, và phần "theo object" có hơn warp mượt generic cùng biên độ không.

**Khung:** như C2-PIA (MI μ = 1, ε = 5, α = 1, 50 step, 2 view/step, 1 backward, σ = 6, L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), 1 forward/view, feature sạch co-transform, GT co-transform), CHỈ thay phép biến đổi PIA bằng:

| Ô | Biến đổi | So với |
|---|---|---|
| C2-MORGD | trường dịch chuyển mượt, neo tại từng GT object | C2-PIA, C2, OSFD |
| C2-SMOOTH | cùng cơ chế, neo tại vị trí NGẪU NHIÊN đều trên ảnh, kích thước = kích thước GT hoán vị ngẫu nhiên (cùng số neo, cùng phân phối biên độ) | C2-MORGD: phần "theo object" |

**Định nghĩa biến đổi (khóa, tự chọn vì nguồn không cho số — KHÔNG tune):** mỗi neo i (box tâm c_i, cỡ w_i × h_i ở khung model): Δ_i = (λ·w_i·r_x, λ·h_i·r_y), r ~ U(−1, 1), **λ = 0.25**; s_i = exp(U(−ln 1.25, ln 1.25)). Ánh xạ ngược của neo i (pixel ra q → nguồn): u_i(q) = (q − c_i)(1/s_i − 1) − Δ_i/s_i (đồng dạng: tâm c_i dời tới c_i + Δ_i, co giãn s_i). Trọng số K_i(q) = exp(−½[((q_x − c'_x)/w_i)² + ((q_y − c'_y)/h_i)²]), c' = c_i + Δ_i. Trường u = Σ K_i u_i / (Σ K_i + **ε = 0.05**) (xa mọi neo → u → 0, nền giữ nguyên); ảnh ra(q) = x(q + u(q)) bằng grid_sample bilinear, ngoài = 0. Đa dạng tương đối r′_ij − r_ij đến từ việc rút (Δ_i, s_i) ĐỘC LẬP mỗi object (không có sampler cặp tường minh — mức tối thiểu của ý tưởng). RNG: `random` của Python. Ảnh không GT → không neo → đồng nhất.

**GT co-transform:** box = AABB của ảnh warp (cùng lưới) của mặt nạ chữ nhật box (≥ 0.5); mask warp cùng lưới; bỏ instance rỗng / < 1 px.

**Unit test trước khi chạy:** (1) ô cũ (GEO/PIA/C2-PIA) trùng HEAD; (2) 1 neo, ε = 0 → trùng phép đồng dạng tương ứng qua homography (warp PIA); (3) xa neo → ảnh gần như giữ nguyên; (4) box 1 neo, ε = 0 khớp c' ± s·(w, h)/2 (sai số rời rạc hóa); (5) SMOOTH dùng đúng multiset kích thước GT; (6) chạy được, hữu hạn, L∞ đúng. Báo cáo thêm biên độ dịch chuyển trung bình |u| (px) mỗi view của MORGD / SMOOTH / PIA để thấy chênh liều.

**Code + unit test (TRƯỚC khi chạy):** `attack/methods/morgd.py` (`sample_anchors`, `flow_grid`, `warp_boxes_by_grid`, `morgd_views`); `geo_osfd_attack(geo="morgd" | "smooth")`; ô `C2-MORGD`, `C2-SMOOTH` (stats thêm `mean_disp_px`). `scripts/morgd_unit_test.py` ĐẠT: GEO-C0, PIA trùng tuyệt đối HEAD; C2-PIA lệch HEAD/mới 0.032 ≈ HEAD/HEAD 0.036 (không tất định đã biết); 1 neo ε = 0 trùng phép đồng dạng (lưới lệch 1.8e-7); xa neo lệch ≤ 0.006; box 1 neo khớp đúng c' ± s·(w, h)/2; SMOOTH đúng multiset kích thước GT; C2-MORGD/C2-SMOOTH chạy được, L∞ đúng. **Biên độ |u| trung bình mỗi view (3 ảnh × 10 view): MORGD 43.0 px, SMOOTH 50.1, PIA 176.7** → MORGD/SMOOTH yếu hơn PIA ~4× về biên độ hình học (nền giữ nguyên theo thiết kế) — phải nhớ khi đọc kết quả.

**Kết quả @ dev30 (mô tả, KHÔNG quyết định)** — `metrics_morgd_dev30.json`, `generalization_morgd_dev30.json`, `results/ablation/morgd_dev30/screen.json`, log `results/pia_dev30/run3.log`.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8 | 0.86
- C2: 99.9 / 94.7 / 84.6 / 82.6 / 67.8 / 26.1 | 78.3 | 0.88
- C2-PIA: 99.5 / 94.8 / 86.8 / 85.9 / 78.2 / **34.7** | **83.6** | 0.91
- C2-MORGD: 99.9 / 96.0 / 87.7 / 83.5 / 72.8 / 27.3 | 81.3 | 0.89
- C2-SMOOTH: 99.5 / 95.0 / 86.0 / 84.4 / 75.3 / 28.1 | 81.9 | 0.90

Runtime ~11 s/ảnh cả hai; |u| trung bình 40.9 (MORGD) / 40.5 px (SMOOTH); giữ 99.5% / 99.2% instance GT.

- **Phần "theo object" KHÔNG có tác dụng:** C2-MORGD − C2-SMOOTH: CrossAvg₃ −0.6, YOLOX −2.5 [−6.5, 0.5], DINO-Swin-L −0.8 [−5.2, 3.5]. Neo theo GT object không hơn neo ngẫu nhiên cùng biên độ.
- **Cả hai thua C2-PIA**, rõ nhất ở DINO-Swin-L: MORGD − C2-PIA −7.5 [−10.2, −0.9], SMOOTH − C2-PIA −6.7 [−9.8, −0.0]; YOLOX −5.4 [−10.7, −2.0] (MORGD). CrossAvg₃ thấp hơn C2-PIA 1.7–2.3.
- Vẫn hơn C2 (RRB) và OSFD về CrossAvg₃ (+2.5 đến +3.6 so OSFD, CI chứa 0).
- Gây nhiễu chưa tách: biên độ hình học của MORGD/SMOOTH ~¼ PIA (41 vs 177 px) → thua C2-PIA có thể do liều, KHÔNG kết luận được "phi cứng kém hơn phối cảnh". Nhưng câu hỏi chính của MORGD (object-conditioned > generic cùng liều) trả lời ÂM ở dev30, khớp prior từ KG / A′1.

---

## 2026-10-04 — MORGD / SMOOTH thuần (không cls) @ dev30 — làm baseline so RRB (user), khóa TRƯỚC khi chạy

Ô `MORGD` / `SMOOTH` = đúng khung ô `PIA` (loss feature OSFD k = 3, MI μ = 1, 2 view/step, 1 backward, σ = 6, feature sạch co-transform, KHÔNG cls, KHÔNG GT), phép biến đổi như entry MORGD (λ = 0.25, s ∈ [0.8, 1.25], ε = 0.05). So OSFD, GEO-C0 (RRB + co-transform — đối chứng công bằng nhất), PIA. Chỉ mô tả. Không code mới (chỉ đăng ký ô trong `run_baselines.py`; `geo_osfd_attack(geo="morgd" | "smooth", cls_weight=0)` đã qua unit test entry MORGD).

**Kết quả @ dev30 (mô tả)** — `metrics_morgd0_dev30.json`, `generalization_morgd0_dev30.json`, `results/ablation/morgd0_dev30/screen.json`, log `results/pia_dev30/run4.log`. Runtime 5.2 s/ảnh cả hai.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8 | 0.86
- GEO-C0 (RRB): 97.8 / 91.8 / 78.2 / 80.8 / 78.2 / 21.5 | 79.0 | 0.87
- PIA: 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / 28.8 | 79.5 | 0.89
- MORGD: 96.9 / 91.5 / 83.3 / 80.2 / 77.9 / 27.3 | 80.5 | 0.89
- SMOOTH: 97.4 / 91.7 / 83.8 / 79.1 / 76.4 / 25.7 | 79.7 | 0.89

- **Bản thuần: mọi phép biến đổi toàn cục/mượt đều ngang nhau** — CrossAvg₃ 79.0 (RRB) / 79.5 (PIA) / 79.7 (SMOOTH) / 80.5 (MORGD), chênh ≤ 1.5, CI ±4–5. MORGD − GEO-C0: DINO-Swin-L +5.8 [−3.1, 11.3]; MORGD − PIA: DINO-Swin-L −1.5.
- MORGD − SMOOTH (thuần): CrossAvg₃ +0.8, YOLOX +1.6, DINO-Swin-L +1.6 — ngược dấu với khung C2 (−0.6) → phần "theo object" ≈ 0, trong nhiễu.
- **Tương tác với cls (thuần → C2):** PIA +4.1 (79.5 → 83.6), SMOOTH +2.2 (79.7 → 81.9), MORGD +0.8 (80.5 → 81.3); DINO-Swin-L: PIA +5.9, SMOOTH +2.4, MORGD 0.0. → lợi khi thêm cls lớn nhất ở PIA (biên độ hình học lớn nhất, ~4× MORGD/SMOOTH); gợi ý (chưa kiểm) loss cls cần biến đổi MẠNH mới phát huy.

---

## 2026-10-04 — Pilot MORGD-RaPA @ dev30 (user: chỉ áp RaPA lên MORGD) — quy tắc khóa TRƯỚC khi code/chạy

**Nguồn đã đọc:** paper RaPA (Su et al., CVPR 2026, arXiv 2504.18594, TeX) + code chính thức `github.com/molarsu/RaPA` (`core/attacker/DropConnect.py`, `config/__base__/attacker/RaPA.py`). Paper: DropConnect lên weight + bias của MỌI lớp Linear và Normalization (BN/LN), M ~ Bernoulli(1 − p), KHÔNG nhân bù 1/(1 − p), mask rút lại mỗi lần inference; ResNet-50 p = 0.05; setting chính targeted, 1000 iteration, S = 5. **Phụ lục: ở untargeted RaPA cho "similar ASRs" (không lợi rõ)** — tác giả giải thích do bão hòa (CFM 98.2%); detection khác họ của ta chưa bão hòa (CrossAvg₃ ~80, DINO-Swin-L 25–35).

**Phát hiện ở code chính thức:** pre-hook gán `weight.data = original·mask`, forward hook trả `weight.data = original` ngay sau forward → khi backward autograd đọc trọng số ĐÃ KHÔI PHỤC (đổi `.data` không tăng version counter) → gradient theo input là "lai" (forward masked, đạo hàm tại chính lớp đó dùng trọng số gốc). Paper không nhắc.

**Ô `MORGD-RaPA`** = đúng ô `MORGD` thuần (loss feature OSFD k = 3, MI μ = 1, 2 view/step, 1 backward, σ = 6, feature sạch co-transform, λ = 0.25, s ∈ [0.8, 1.25], ε = 0.05) + RaPA:
- Phạm vi: 53 lớp BatchNorm2d của backbone R50 (weight + bias affine, 53 120 tham số). Backbone KHÔNG có Linear; loss feature chỉ đi qua backbone → tương đương "mọi Linear + Norm" trên đường loss.
- p = 0.05 (giá trị paper cho ResNet-50), S = 1, mask Bernoulli(0.95) từng phần tử, không nhân bù, **rút mới mỗi view**; forward ảnh sạch co-transform và forward ảnh adv của CÙNG view dùng CÙNG mask (nếu khác mask, chênh feature do tham số bị tính như distortion adv).
- **Lệch có chủ đích so với code chính thức:** tham số masked dùng nhất quán cả forward lẫn backward (thay tạm tham số bằng tensor masked qua `torch.nn.utils.stateless._reparametrize_module`, không sửa `.data`) — gradient đúng toán học.
- RNG mask: `torch.Generator` riêng trên GPU, seed = `torch.initial_seed()` của ảnh (= SEED + chỉ số dev300) → không đụng RNG của biến đổi/nhiễu: p = 0 phải trùng tuyệt đối ô MORGD.
- Compute: không thêm forward/backward (forward sạch đã có do co-transform).

**Unit test trước khi chạy:** p = 0 trùng tuyệt đối MORGD; tỉ lệ phần tử bị mask ≈ 0.05; tham số model sau attack không đổi; cùng mask → tại adv = sạch, σ = 0 feature view == feature companion và khác feature không mask; gradient chảy qua tham số masked (kiểm bằng so sánh với forward thủ công có mask); chạy được, L∞ đúng.

**Báo cáo (mô tả):** so MORGD, SMOOTH, GEO-C0, PIA, OSFD; 6 model, CrossAvg₃.

**Code + unit test (TRƯỚC khi chạy):** `pia.rapa_sampler` (53 BN backbone, Bernoulli(0.95), `_reparametrize_module`, RNG riêng), `geo_osfd_attack(rapa_p=)`; ô `MORGD-RaPA`. `scripts/rapa_unit_test.py` ĐẠT: MORGD trùng tuyệt đối HEAD; rapa_p = 0 trùng tuyệt đối MORGD; tỉ lệ mask 0.0502; tham số model không đổi sau attack; cùng context → cùng mask, khác feature không mask; gradient theo input dưới context == khi gán tham số masked thật (lệch tương đối 7.1e-8, tức masked nhất quán cả forward lẫn backward); L∞ đúng.

**Kết quả @ dev30 (mô tả)** — `metrics_rapa_dev30.json`, `generalization_rapa_dev30.json`, `results/ablation/rapa_dev30/screen.json`, log `results/pia_dev30/run5.log`. Runtime 6.1 s/ảnh (MORGD 5.2). Tỉ lệ mask thực tế ≈ 0.050.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- MORGD: 96.9 / 91.5 / 83.3 / 80.2 / 77.9 / 27.3 | 80.5 | 0.89
- **MORGD-RaPA: 94.0 / 78.5 / 52.4 / 57.9 / 56.3 / 18.7 | 55.5 | 0.70**
(OSFD 78.8, GEO-C0 79.0, PIA 79.5, SMOOTH 79.7 — như entry trước.)

- **RaPA làm attack yếu đi mạnh ở mọi target:** MORGD-RaPA − MORGD: CrossAvg₃ −25.0, YOLOX −21.6 [−29.6, −12.8], DINO-Swin-L −8.6 [−16.9, −1.1], R101 −13.0; cả R50 white-box −2.9.
- **Chẩn đoán (forward-only, 5 ảnh dev30, ảnh sạch):** cos(feature backbone sạch, feature cùng ảnh dưới mask) theo stage 1→4: p = 0.05: 0.855 / 0.749 / 0.689 / 0.702; cos giữa 2 mask khác nhau: 0.827 / 0.723 / 0.676 / 0.700. p = 0.01: 0.962 / 0.934 / 0.915 / 0.925 (2 mask: 0.945 / 0.905 / 0.880 / 0.891). → Với p = 0.05, mỗi view là một biến thể surrogate lệch ~30% feature sâu; với loss feature (đo trực tiếp trên feature backbone) và chỉ 50 step × S = 1, hướng gradient giữa các biến thể gần như không cộng hưởng → mất "utility". p = 0.05 được paper chọn cho ResNet-50 CLASSIFICATION (loss trên logit, 1000 iteration, S = 5), không chuyển được sang setting này.
- Chưa tách: (a) liều p (0.01 ít phá feature hơn nhiều); (b) gradient "lai" của code chính thức (backward dùng trọng số gốc → hướng gradient gần surrogate gốc hơn) có thể chính là thứ giữ utility ở paper — bản của ta dùng gradient nhất quán. Không chạy thêm biến thể nếu user không yêu cầu.

---

## 2026-10-04 — Pilot GRGD-v0 @ dev30 (user, spec từ trợ lý khác) — quy tắc khóa TRƯỚC khi code/chạy

**Ý tưởng:** T_GRGD = T_relative ∘ T_PIA trên khung C2-PIA (không đổi gì khác: ε = 5, α = 1, B = 50, μ = 1, 2 view/step, 1 backward, σ = 6 chỉ cộng vào adv, L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), feature sạch + GT co-transform qua CẢ HAI phép). T_relative thay đổi quan hệ vị trí/tỉ lệ GIỮA các object (sau PIA), đã trừ phần giải thích được bằng một homography. Prior (đã nêu với user): MORGD neo theo object ≈ neo ngẫu nhiên (dev30, cả 2 khung) — câu hỏi lần này là sample QUAN HỆ cặp trực tiếp + trừ thành phần phối cảnh có khác không.

**T_relative (trên GT còn lại sau PIA, N object; N < 2 → đồng nhất, bỏ bước warp thứ hai):**
1. Đồ thị đầy đủ mọi cặp i < j. Mỗi cạnh: ξˣ, ξʸ, ξˢ ~ U(−1, 1); d_x = λ_r·(w_i + w_j)/2·ξˣ, d_y = λ_r·(h_i + h_j)/2·ξʸ, d_s = ln(1.25)·ξˢ; **λ_r = 0.25** (từ MORGD, không tune).
2. Ma trận liên thuộc A (E × N, A_ei = 1, A_ej = −1); giải riêng x, y, log-scale bằng KKT: [[AᵀA + 1e-6·I, ω], [ωᵀ, 0]]·[z; ν] = [Aᵀd; 0], ω_i = w_i h_i / Σ w h (trung bình có trọng số diện tích = 0) → t_i = (z_x, z_y), s_i = exp(z_s).
3. Trường thô u_raw = Σ K_i u_i / (Σ K_i + 0.05), u_i(q) = (q − c_i)(1/s_i − 1) − t_i/s_i, K_i Gaussian tâm c'_i = c_i + t_i, độ rộng (w_i, h_i) — đúng công thức MORGD.
4. **Trừ thành phần phối cảnh:** điểm điều khiển = 5 điểm/object (4 góc + tâm của box cỡ w_i × h_i đặt tại c'_i), p_m = q_m + u_raw(q_m); fit H_r: q → p bằng DLT bình phương tối thiểu (chuẩn hóa Hartley, float64, SVD); u_H(q) = H_r(q) − q; M(q) = Σ K_i / (Σ K_i + 0.05); **u_rel = u_raw − M·u_H**.
5. Ảnh ra = grid_sample(ảnh sau PIA, q + u_rel), bilinear, ngoài = 0. GT: box sau PIA → AABB của mặt nạ chữ nhật warp (≥ 0.5); mask warp; bỏ instance rỗng / < 1 px (như MORGD).

**Ô (dev30, chỉ mô tả):**
| Ô | Định nghĩa |
|---|---|
| C2-PIA | PNG có sẵn (mốc, 83.6) |
| C2-PIA-RESIDUAL | như C2-GRGD nhưng tâm neo c_i thay bằng tâm ngẫu nhiên đều trên khung; GIỮ NGUYÊN kích thước, ξ_ij, d_ij, nghiệm (t, s), kernel, bước trừ phối cảnh |
| C2-GRGD | như trên, tâm neo = tâm GT thật (sau PIA) |
Hiệu chính: Δ_extra = RESIDUAL − C2-PIA; **Δ_object = GRGD − RESIDUAL** (chỉ khi > 0 mới là bằng chứng cho đóng góp mới).

**RNG:** H của PIA và nhiễu σ dùng đúng RNG như C2-PIA (`random`, torch); ξ_ij từ `random.Random` riêng (seed = torch.initial_seed() + 1); tâm ngẫu nhiên của RESIDUAL từ `random.Random` riêng thứ hai (seed + 2) → mỗi view: H, ξ, σ trùng nhau giữa 2 ô, chỉ khác tâm neo.

**Log mỗi ảnh:** n_objects, n_edges, |u| trung bình phần PIA / u_raw / u_H·M / u_rel, R_proj = 1 − ‖u_rel‖₂ / ‖u_raw‖₂, D_pair_pos (mục tiêu, theo t) và D_pair_pos_eff (hiệu dụng, dời tâm ≈ −u_rel(c'_i)) chuẩn hóa bởi (√A_i + √A_j)/2, D_pair_scale = mean |ln((s_i/s_j))| (mục tiêu), gt_kept_frac.

**Unit test (sửa từ spec, user được báo):** C2 không tất định → "λ_r = 0 trùng C2-PIA" kiểm ở MỨC VIEW (ảnh, companion, GT trùng tuyệt đối pia_views), không ở mức attack. Danh sách: (1) λ_r = 0 → view trùng tuyệt đối PIA; (2) N < 2 → trùng; (3) ξ = 0 → |u_rel| ≈ 0; (4) 2 box: (t₁ − t₂) = d₁₂ đúng dấu và độ lớn; (5) Σ ω t = 0; (6) u_raw sinh từ MỘT homography → sau khi trừ, RMS ≈ 0; (7) 2 object chuyển động khác nhau → báo RMS dư; (8) xa object |u_rel| ≈ 0; (9) σ = 0, adv = sạch → view == companion; (10) IoU(mask, box) view GRGD; (11) ô cũ (PIA, MORGD) trùng HEAD; (12) C2-GRGD / C2-PIA-RESIDUAL chạy được, hữu hạn, L∞ đúng.

**Code + unit test (TRƯỚC khi chạy):** `attack/methods/grgd.py` (`complete_edges`, `sample_pairwise`, `solve_graph_motion` — KKT, `fit_homography_dlt` — Hartley + SVD, `RelativeField` — u_raw / M·u_H / u_rel, `relative_motion`, `grgd_views`); `geo_osfd_attack(geo="grgd" | "grgd_random")`; ô `C2-GRGD`, `C2-PIA-RESIDUAL` trong `run_baselines.py` (stats `grgd`: trung bình mỗi ảnh các chỉ số log ở trên + `frac_views_relative`). `morgd.flow_grid` tách thành `flow_field` / `pixel_coords` / `to_grid` (không đổi số). `scripts/grgd_unit_test.py` ĐẠT: λ_r = 0 và N = 0 / 1 → view (ảnh, companion, GT) trùng tuyệt đối `pia_views`; ξ = 0 → max |u_rel| 6.7e-13 px; 2 box: t₁ − t₂ = d₁₂ đúng (22.0, −14.375, log-scale 0.134); Σ ω t = Σ ω log s = 0 (≤ 1.3e-15); 1 homography → fit lại đúng (RMS dư 1.4e-13); 2 object khác chuyển động → RMS dư 3.72 px; xa object |u_rel| 1.3e-13; σ = 0, adv = sạch → view == companion; IoU(AABB mask warp, box warp) mean 0.754 (trung vị 0.753, p5 0.456; PIA 0.823 — thấp hơn do 2 lần nội suy mask); mẫu số H_r > 0 trên toàn khung; GEO-C0 / PIA / MORGD trùng tuyệt đối HEAD, `morgd_views` HEAD == mới, C2-PIA lệch HEAD/mới 0.031 ≈ HEAD/HEAD 0.034; **chẩn đoán RNG: C2-GRGD và C2-PIA-RESIDUAL có mode PIA + nhiễu σ trùng tuyệt đối C2-PIA mỗi view (4 step × 2 view)** — sampler mmdet dùng `torch.randperm` trên CPU, nhiễu σ dùng RNG CUDA nên không lệch theo GT; 2 ô chạy được, L∞ đúng, forward no_grad = 2 × step.

**Chẩn đoán liều (forward-only, dev30, 6 view/ảnh, 156 view có N ≥ 2; 26/30 ảnh có N ≥ 2, trung vị N = 10) — T_relative theo spec gần như KHÔNG đổi hình học tương đối:**
- D_pair_pos mục tiêu 0.126 → hiệu dụng (sau trừ phối cảnh) **0.025** (tỉ lệ 0.22); theo N: 2–4 object 0.180 → 0.023 (0.13), 5–10: 0.135 → 0.028, ≥ 11: 0.094 → 0.025.
- |u| trung bình raw / M·u_H / rel: 8.4 / 9.8 / 6.0 px (N 2–4), 3.7 / 3.2 / 2.8 px (N ≥ 11); PIA ~179 px, MORGD ~41 px. R_proj mean 0.21, có view âm (bước trừ thêm biến dạng lớn hơn phần nó bỏ).
- Ca 2 box (unit test 4): mục tiêu dời tương đối (22.0, −14.4) px; không trừ phối cảnh → (20.5, −13.5); có trừ → **(1.5, −0.5)**.
- Hai nguyên nhân: (1) nghiệm bình phương tối thiểu trên đồ thị đầy đủ là t_i = (1/N) Σ_j d_ij → biên độ co theo N; (2) homography 8-DOF giải thích gần hết chuyển động của vài object (affine đã đưa 3 tâm tới vị trí bất kỳ) → bước trừ phối cảnh xóa phần lớn chính quan hệ cặp mà nhánh này định tạo.
- Hệ quả: chạy đúng spec thì C2-GRGD và C2-PIA-RESIDUAL gần như là C2-PIA cộng thêm warp ~3–6 px → kết quả ngang nhau sẽ không phân biệt được "hình học tương đối không có ích" với "liều quá nhỏ". **Chưa chạy attack — chờ user quyết** (chạy đúng spec / sửa spec trước khi chạy / dừng nhánh).

---

## 2026-10-04 — GRGD-v0 sửa (B): bỏ phép trừ phối cảnh + cổng độ trung thực forward-only (user, khóa TRƯỚC khi chạy cổng; CHƯA có attack nào của GRGD)

**Quyết định (user):** *GRGD-v0 original projection is rejected before attack execution.* Chẩn đoán forward cho thấy fit rồi trừ trường phối cảnh tốt nhất xóa chính chuyển động tương đối giữa object, vì chuyển động phối cảnh và hình học tương đối giữa object KHÔNG phải hai không gian rời nhau (homography toàn cục cũng đổi khoảng cách, tỉ lệ, vị trí tương đối). GRGD được định nghĩa lại TRƯỚC khi thấy bất kỳ kết quả attack nào: thành phần toàn cục = H_PIA do chính ta rút; nhánh relative từ đồ thị cặp chỉ bị ràng buộc gauge Σ ω_i t_i = 0 và Σ ω_i a_i = 0 (a = log s). **T_GRGD(x) = W_rel(H_PIA(x)), u_rel = u_raw** (bước 4 của entry GRGD-v0 bỏ; fit H_r giữ lại CHỈ để chẩn đoán). Không tune hyperparameter attack nào; λ_r = 0.25 giữ nguyên. Mọi thứ khác của entry GRGD-v0 (đồ thị đầy đủ, KKT, kernel, RNG, 3 ô, Δ_object = GRGD − RESIDUAL) giữ nguyên.

**Code:** `RelativeField(remove_projective=False)` mặc định (True = spec v0 gốc, chỉ để đối chiếu); stats thêm `d_pair_pos_req` (theo d_ij đã rút, chuẩn hóa (√A_i + √A_j)/2), `rho_pos` = D_pair_pos_eff / D_pair_pos_req, `cos_pair_eff_req`, `d_pair_scale_req`, `rms_rel_px`, `d_pair_pos_box` / `d_pair_scale_box` (đo trên GT box trước/sau warp relative, cặp còn giữ cả hai). Unit test chạy lại ĐẠT toàn bộ + 2 mục mới: mặc định u_rel == u_raw; 2 box sửa B dời hiệu dụng / mục tiêu = 0.93 / 0.94 (spec v0: (1.5, −0.5) vs (22.0, −14.4)).

**Cổng độ trung thực (forward-only, `scripts/grgd_fidelity_gate.py` → `results/grgd/fidelity_gate.json`):** dev30, 10 view/ảnh, seed như `run_baselines` (SEED + chỉ số dev300; ξ, tâm ngẫu nhiên từ RNG riêng seed + 1 / + 2), GRGD và RESIDUAL cùng H, ξ, nhiễu; chỉ view N ≥ 2.
- **G1:** median ρ_pos (GRGD) ≥ 0.75. "Requested" = d_ij đã rút (KHÔNG phải t sau khi giải) — để cổng bắt được cả việc nghiệm bình phương tối thiểu co biên độ.
- **G2:** mean D_pair_pos_eff > 0 VÀ mean D_pair_scale_box > 0 (GRGD).
- **G3:** mean RMS |u_rel| GRGD / RESIDUAL ∈ [0.9, 1.1] (liều hình học khớp).
- **PASS cả 3 → chạy 2 ô attack** C2-GRGD, C2-PIA-RESIDUAL @ dev30 (mô tả, như entry GRGD-v0). **FAIL → KHÔNG chạy attack**, ghi log; nếu trượt G1 do co bình phương tối thiểu thì ứng viên sửa tiếp (user nêu) là đồ thị thưa liên thông (cây khung, |E| = N − 1, ràng buộc nhất quán theo cấu trúc) — cần entry khóa mới, KHÔNG tăng λ_r để bù.
- **Dự đoán ghi trước (giải tích):** với đồ thị đầy đủ và d_ij độc lập, t_i − t_j = (1/N)[2d_ij + Σ_{k≠i,j}(d_ik + d_kj)] → ‖t_i − t_j‖ / ‖d_ij‖ ≈ √(2/N) (≈ 0.45 ở N = 10); unit test (ảnh 4 object) đã cho ρ_pos mean 0.45 → **G1 nhiều khả năng TRƯỢT**.

**Kết quả cổng (`results/grgd/fidelity_gate.json`; 257 view N ≥ 2, N trung bình 12.5) — TRƯỢT (G1, G3) → KHÔNG chạy attack GRGD (theo quy tắc khóa):**
- **G1 TRƯỢT:** median ρ_pos = **0.19** (ngưỡng 0.75). Theo N: 2–4 object 0.25 (√(2/N) = 0.80), 5–10: 0.17 (0.53), ≥ 11: 0.18 (0.33). cos(dời cặp hiệu dụng, d_ij) mean 0.30.
- G2 đạt (tầm thường): D_pair_pos_eff 0.049, D_pair_scale_box 0.049.
- **G3 TRƯỢT:** RMS |u_rel| GRGD / RESIDUAL = **0.82** (7.5 vs 9.2 px) — neo ngẫu nhiên rải đều nên kernel ít chồng nhau hơn, triệt tiêu ít hơn.
- **Tách 2 tầng mất mát (mean):** d_ij yêu cầu 0.212 → nghiệm t 0.132 (×0.62, co bình phương tối thiểu — nguyên nhân 1) → hiệu dụng sau ráp trường 0.049 (×0.37). **Tầng thứ hai lớn hơn và KHÔNG được dự đoán:** kernel Gaussian rộng bằng cỡ box của các object gần / chồng nhau (COCO) bị trung bình hóa trong u = Σ K_i u_i / (Σ K_i + ε), nên tại c'_i trường không bằng t_i. Ca 2 box tách xa trong unit test (0.93) không đại diện. → Đồ thị cây khung chỉ sửa tầng ×0.62; trần lý thuyết vẫn ≈ 0.37 < 0.75 nếu giữ cách ráp trường của MORGD.
- **Quan trọng cho tính hợp lệ của phép thử E2 − E1:** thay đổi hình học tương đối mà GT box (thứ detector thấy) THỰC SỰ chịu gần như như nhau giữa hai ô — D_pair_pos_box GRGD **0.046** = RESIDUAL **0.046**; D_pair_scale_box 0.049 vs 0.039. Trường neo ngẫu nhiên cũng làm lệch quan hệ giữa các GT object ngang trường neo theo object → ở thiết kế này hai ô gần như không khác nhau về đại lượng mà giả thuyết nói tới.
- Không có attack nào của GRGD được chạy (v0 lẫn sửa B). Hướng tiếp (user quyết): dừng nhánh, hoặc khóa entry mới sửa CẢ hai tầng (cây khung + cách ráp trường giữ được t_i tại object, vd phân hoạch theo mask/box thay kernel Gaussian rộng) và đặt lại cổng (gồm điều kiện D_pair_pos_box GRGD > RESIDUAL).

---

## 2026-10-04 — Pivot (user): DINO-Swin-L làm failure case — PIA phá được Swin-L qua đâu? (forward-only @ dev30, quy tắc khóa TRƯỚC khi chạy)

**Bối cảnh:** dừng nhánh MORGD/GRGD (user). Giả thuyết mới (user): "Swin-L bền ≈ distortion hình học đa scale không tới được phân cấp nông→sâu của Swin-L + encoder dập". PIA là phép hình học duy nhất có tín hiệu rõ trên Swin-L (C2-PIA 34.7 vs OSFD 26.8 vs C2 26.1; PIA − GEO-C0 +7.2 @ dev30). Nếu lợi của PIA đi qua stage nông và/hoặc làm encoder bớt dập → cơ sở thiết kế "Multi-Scale Projective Diversity" (CHƯA code). Nếu không → bỏ tiền đề đa scale.

**Đo (chỉ forward, PNG có sẵn `artifacts/runs/dev300_B50_eps5/adv/`, dev30, `scripts/mech_swinl_geo.py` → `results/ablation/mech/swinl_geo.json`):** DINO-Swin-L, ảnh sạch vs adv qua cùng test pipeline (như `mech_swinl.py`): backbone stage 1–4 (stride 4/8/16/32), ChannelMapper 5 mức, memory encoder (toàn bộ + tách 5 mức theo `spatial_shapes`). Chỉ số cosd = 1 − cos (vector phẳng) và rel = ‖a − c‖/‖c‖, mỗi ảnh. A_enc(ảnh) = cosd_encoder(toàn bộ) / cosd_backbone stage 4. Kèm A_l = cosd_encoder mức l / cosd_neck mức l.

**Ô:** cặp CHÍNH (cùng loss, khác hình học): **P1 = C2-PIA vs C2** (`PROBE-osfd+cls`), **P2 = PIA vs GEO-C0** (cùng co-transform feature sạch — cặp sạch nhất). Mô tả: OSFD, PIA-aff, PIA-d0.3, MORGD, SMOOTH, C2-MORGD, C2-SMOOTH (tham chiếu âm).

**Quy tắc (paired bootstrap 1000 mẫu trên ảnh, BOOT_SEED 2026; Δ = mean(A) − mean(B); rel = mean(A) / mean(B)):**
- **SHALLOW-SPECIFIC** đúng trong một cặp nếu ∃ stage s ∈ {1, 2}: cận dưới CI Δcosd_s > 0 VÀ rel_s ≥ 1.25 VÀ rel_s ≥ rel_4 (tăng ở nông không kém tăng ở sâu — phân biệt "tới được scale nông" với "mọi thứ mạnh hơn đều").
- **ENCODER** đúng trong một cặp nếu cận dưới CI ΔA_enc > 0.
- **Kết luận:** SHALLOW-SPECIFIC ở CẢ P1 và P2 → ủng hộ tiền đề đa scale → được khóa entry thiết kế Multi-Scale Projective Diversity. Không SHALLOW nhưng ENCODER ở cả hai → một phần: hình học tạo distortion encoder khó sửa nhưng không qua stage nông; tiền đề đa scale KHÔNG được ủng hộ, user quyết. Không cái nào ở cả hai cặp → bác giả thuyết, KHÔNG xây hình học đa scale.
- Mô tả (không quyết định): cosd encoder theo 5 mức, A_l; Spearman giữa các ô (11 ô) của mean cosd từng tầng với AP drop DINO-Swin-L đã có; câu hỏi mode PIA (Q3) chỉ trả lời gián tiếp qua PIA-aff (chỉ 6-DOF) và PIA-d0.3 — tách thật theo mode cần attack mới, chưa làm.
- Caveat: n = 30 (CI rộng); PNG C2 là bản sinh lại trên dev100 (cùng seed).

**Kết quả (`results/ablation/mech/swinl_geo.json`, n = 30) — theo quy tắc khóa: BÁC giả thuyết (không SHALLOW-SPECIFIC ở cặp nào; ENCODER chỉ ở P1) → KHÔNG xây hình học đa scale.**

cosd trung bình, DINO-Swin-L (drop % | bb1 / bb2 / bb3 / bb4 | neck5 | enc | A_enc):
- OSFD 26.8 | 0.070 / 0.115 / 0.281 / 0.426 | 0.246 | 0.067 | 0.153
- GEO-C0 21.5 | 0.071 / 0.121 / 0.271 / 0.392 | 0.206 | 0.056 | 0.142
- PIA 28.8 | 0.071 / 0.121 / 0.285 / 0.420 | 0.232 | 0.063 | 0.147
- C2 26.1 | 0.070 / 0.106 / 0.227 / 0.337 | 0.171 | 0.042 | 0.123
- C2-PIA 34.7 | 0.070 / 0.109 / 0.246 / 0.369 | 0.195 | 0.053 | 0.140
- (PIA-aff, PIA-d0.3, MORGD, SMOOTH, C2-MORGD, C2-SMOOTH trong file.)

- **Stage nông KHÔNG đổi theo bất kỳ phương pháp nào:** bb1 = 0.070–0.071 ở cả 11 ô (P1 Δ +0.000, P2 Δ −0.000); bb2 0.106–0.122, rel P1 ×1.03, P2 ×1.00. Mức distortion stage 1 có vẻ là "sàn" của nhiễu ε = 5 ở độ phân giải cao, không phụ thuộc loss hay hình học → tiền đề "đưa distortion hình học tới scale nông" không có chỗ bám trong dữ liệu.
- **PIA tăng ở SÂU và ở encoder:** bb4 ×1.09 (P1) / ×1.07 (P2), neck5 ×1.14 / ×1.13, encoder ×1.27 / ×1.13 (mọi mức encoder trừ enc5 ở P1 có CI > 0). A_enc: P1 +0.016 [+0.008, +0.026] (ENCODER đúng), P2 +0.005 [−0.001, +0.012] (không).
- **Độ lớn distortion KHÔNG giải thích AP drop của Swin-L ở bất kỳ tầng nào:** Spearman giữa 11 ô của mean cosd từng tầng với drop Swin-L nằm trong [−0.45, +0.18] (bb2 −0.45, enc −0.15, A_enc −0.34). Họ C2 có distortion THẤP hơn họ loss-feature ở MỌI tầng (C2-PIA enc 0.053 < OSFD 0.067, PIA 0.063) nhưng C2-PIA có drop cao nhất (34.7). Khớp phát hiện cũ ("task loss có cosd encoder thấp hơn OSFD nhưng drop cao hơn"): lợi trên Swin-L đến từ HƯỚNG (tín hiệu phân loại sau encoder), không từ lượng distortion feature.
- Q3 (mode PIA, gián tiếp): PIA-aff ≈ PIA ở mọi tầng (bb4 0.410 vs 0.420, enc 0.056 vs 0.063); PIA-d0.3 ở giữa. Không có dấu hiệu số hạng phối cảnh tạo loại distortion riêng.
- Đọc chung: trong cặp cùng loss, PIA làm feature sâu + encoder lệch thêm ~10–25% tương đối, nhưng trục "lượng distortion" không theo được AP drop giữa các ô → muốn hiểu vì sao C2-PIA phá Swin-L tốt hơn cần đo ở đầu ra (điểm phân loại query decoder, xếp hạng, loại lỗi) chứ không ở độ lớn feature. Hướng tiếp: user quyết.

---

## 2026-10-04 — Phân tích đầu ra decoder DINO-Swin-L: C2-PIA phá Swin-L qua phân loại hay định vị? (forward-only @ dev30, quy tắc khóa TRƯỚC khi chạy)

**Câu hỏi (user):** lợi của C2-PIA trên DINO-Swin-L (34.7 vs C2 26.1) có nằm ở bằng chứng PHÂN LOẠI của query khớp GT (margin, xếp hạng) không, hay ở định vị / gán lại query. Chưa thiết kế method; chưa phân tích theo tầng decoder (chỉ làm nếu đầu ra cuối có hiệu ứng).

**Cặp:** P1 = C2-PIA vs C2 (`PROBE-osfd+cls`) — chính, cùng loss khác hình học; P2 = PIA vs GEO-C0. Mô tả: OSFD, PIA-aff. PNG có sẵn, dev30, test pipeline như `mech_swinl.py`.

**Định nghĩa "query cố định" cho DINO (điều chỉnh bắt buộc so với đề xuất, user được báo):** DINO two-stage — slot k của decoder = embedding nội dung học sẵn thứ k + box tham chiếu của token encoder có điểm xếp hạng thứ k (`DINO.pre_decoder`: top-900 theo max_c logit encoder). Slot k trên adv vì vậy KHÔNG cùng object với slot k trên ảnh sạch. → **Primary ("forced"):** trên adv, decoder dùng ĐÚNG tập + thứ tự token encoder đã chọn trên ảnh sạch (topk_indices sạch; box tham chiếu và memory lấy từ adv tại các token đó) → slot k cùng token, cùng embedding nội dung, chỉ khác nội dung ảnh. Loại bỏ hiệu ứng chọn/gán lại query (đo riêng ở secondary). **Secondary ("natural"):** forward bình thường trên adv.

**Chọn query trên ảnh sạch (giống nhau cho mọi điều kiện):** với mỗi GT i (lớp c_i), q*_i = argmax_q IoU(box cuối của q, GT), giữ GT nếu IoU ≥ 0.5. **Dự phòng khai báo trước:** nếu trung vị sigmoid(z_{q*,c}) trên ảnh sạch < 0.3 (argmax IoU rơi vào query trùng lặp điểm thấp) → q*_i = argmax z_{q,c_i} trong các query IoU ≥ 0.5.

**Chỉ số mỗi GT (lớp cuối decoder, logit thô — sigmoid focal):** s = z_{q*,c}; margin m = z_{q*,c} − max_{c'≠c} z_{q*,c'}; rank percentile r = (R − 1)/(Q − 1), R = hạng của z_{q*,c} trong 900 query theo lớp c (1 = cao nhất); I = IoU(box q*, GT). Δ = adv − sạch.

**Quy tắc (paired bootstrap theo ẢNH, 1000 mẫu, BOOT_SEED 2026; trung bình theo GT có trọng số ảnh):**
- **CLS** đúng ở cặp (A, B) nếu cận trên CI[Δm(A) − Δm(B)] < 0 VÀ cận dưới CI[Δr(A) − Δr(B)] > 0.
- **LOC** đúng ở cặp nếu cận trên CI[ΔI(A) − ΔI(B)] < 0 (chỉ xác định KIỂU cơ chế, không phải điều kiện của CLS).
- Diễn giải: P1 CLS + P2 CLS → hình học PIA tự tạo bất ổn phân loại query; **P1 CLS, P2 không → tương tác loss × hình học**; P1 không CLS nhưng LOC → lợi ở định vị / gán query; P1 và P2 đều không CLS, không LOC → giả thuyết mức query bị bác, chuyển sang xếp hạng toàn cục / FP.

**Secondary (mô tả):** (a) natural: tỉ lệ token sạch của q* bị loại khỏi top-900 adv; trên phần còn lại: Δm, Δr, ΔI của slot chứa token đó. (b) gán lại: s_best = max_q z_{q,c_i} (natural adv), IoU của query đạt max; Δs_best. (c) Phân rã lỗi (natural, 100 detection như `predict` của DINO, ngưỡng 0.3), trên GT được phát hiện ở ảnh sạch (đúng lớp, IoU ≥ 0.5, score ≥ 0.3): preserved (vẫn vậy trên adv) → nếu không: **conf/rank failure** (có query adv IoU ≥ 0.5 với GT, bất kể lớp; tách riêng phần nhầm lớp) → **loc failure** (có query adv sigmoid z_c ≥ 0.3 với 0.1 ≤ IoU < 0.5) → **lost**. (d) FP/ảnh (score ≥ 0.3): nền / định vị / nhầm lớp như `probe_error_types.py`, phân bố score FP.

**Sanity trước khi đọc kết quả:** forced trên ảnh sạch == natural trên ảnh sạch (logit trùng); 100 detection natural khớp `model.predict` (box, score); tỉ lệ GT có q* IoU ≥ 0.5; trung vị score sạch của q*; chọn q* độc lập điều kiện.

**Code + sanity (TRƯỚC khi chạy đủ dev30):** `scripts/diag_decoder_output.py`. Sanity trên 3 ảnh đầu: forced trên ảnh sạch == natural (logit lệch tối đa 0.0); detection tự dựng khớp tuyệt đối `model.predict` (score, box lệch 0.0) — sửa ghi chú: DINO `test_cfg.max_per_img` = **300** (không phải 100), script lấy từ model; phân rã lỗi / FP dùng ngưỡng 0.3 nên gần như không đổi. 9/9 GT có q* IoU ≥ 0.5. **Trung vị sigmoid score sạch của q* (argmax IoU) = 0.012 < 0.3 → kích hoạt dự phòng khai báo trước: chạy chính với `--select maxlogit`** (q* = argmax z_{q,c} trong các query IoU ≥ 0.5). Số của lần thử 3 ảnh KHÔNG dùng.

**Kết quả @ dev30 (`results/ablation/mech/decoder_output.json`, `--select maxlogit`) — theo quy tắc khóa: P1 không CLS, không LOC; P2 cũng không → GIẢ THUYẾT MỨC QUERY BỊ BÁC.**

Sanity đủ 30 ảnh: forced sạch == natural (0.0), khớp `predict` (0.0); 352/362 GT (97%) có q*; trung vị score sạch q* 0.51 (75.6% ≥ 0.3). q* sạch: logit 0.17, margin 4.14, rank pct 0.011, IoU 0.82.

Forced, Δ (adv − sạch) trung bình theo GT [CI 95%] — d_s / d_margin / d_rank_pct / d_iou:
- OSFD −1.21 / −1.16 / +0.067 / −0.110
- GEO-C0 −1.06 / −1.02 / +0.051 / −0.102
- PIA −1.20 / −1.15 / +0.060 / −0.109
- PIA-aff −1.16 / −1.09 / +0.058 / −0.099
- C2 −1.26 / −1.18 / +0.060 / −0.111
- C2-PIA −1.30 / −1.19 / +0.063 / −0.118

- **P1 (C2-PIA − C2):** d_margin −0.006 [−0.18, +0.18], d_rank +0.003 [−0.013, +0.022], d_iou −0.007 [−0.023, +0.009] → không khác. **P2 (PIA − GEO-C0):** d_margin −0.14 [−0.25, −0.03] (CI < 0) nhưng d_rank +0.009 [−0.006, +0.024] → CLS sai theo quy tắc (cần cả hai); d_iou −0.007.
- **Ở mức query, tác động lên DINO-Swin-L gần như BẤT BIẾN theo phương pháp:** cả 6 ô giảm margin ~1.0–1.2 logit, rank tụt 5–7 percentile, IoU −0.10 đến −0.12 — chênh giữa các ô nhỏ hơn nhiều so với CI.
- Natural (mô tả): token q* bị loại khỏi top-900 19–24% (C2-PIA 23.6% cao nhất, C2 21.3%); trên phần còn lại margin tụt 1.21–1.41. Gán lại: max_q z_c (adv) chỉ giảm 0.29–0.42 logit (forced giảm ~1.2) → decoder phần lớn "nhận lại" bằng query khác (IoU của query max chỉ 0.19–0.21 — bị lẫn bởi instance khác cùng lớp, chỉ để tham khảo).
- Phân rã lỗi (266 GT phát hiện được ở ảnh sạch): preserved C2-PIA **0.726** vs C2 0.771 / OSFD 0.763 / PIA 0.756 / GEO-C0 0.774; phần mất thêm của C2-PIA rơi vào conf/rank (0.241 vs 0.211). Loc / lost ≤ 3% ở mọi ô. FP/ảnh: họ C2 tăng FP nền (2.9 vs sạch 1.6), C2-PIA không hơn C2 (5.27 vs 5.70 tổng).
- Đọc chung: chênh AP +8.6 (C2-PIA − C2 trên Swin-L @ dev30, CI [0.3, 13.6]) KHÔNG thấy ở bằng chứng phân loại / xếp hạng / định vị trung bình theo GT trên query cố định; chỉ thấy ~4.5 điểm preserved@0.3 (12/266 GT) — cỡ nhiễu ở n = 30. Hai khả năng chưa tách: (a) lợi AP tập trung ở ít ảnh / ít GT (đuôi), (b) chênh AP dev30 phần lớn là nhiễu (cận dưới CI 0.3). Hướng tiếp: user quyết.

---

## 2026-10-04 — Xác nhận C2-PIA trên DINO-Swin-L @ dev100 (user: "khóa rồi chạy") — quy tắc khóa TRƯỚC khi chạy

**Lý do:** 3 vòng phân tích (đa scale, distortion, đầu ra decoder) không thấy dấu vết cơ chế cho chênh Swin-L +8.6 (C2-PIA − C2 @ dev30, CI [0.3, 13.6]) → phải xác nhận chính hiện tượng trước khi giải thích tiếp.

**Chạy:** `run_baselines.py --split dev300 --subset dev100 --methods OSFD PROBE-osfd+cls GEO-C0 PIA C2-PIA --out metrics_pia_dev100.json` (resume: OSFD, C2 có đủ 100 PNG; GEO-C0 / PIA / C2-PIA attack thêm 70 ảnh, cùng seed SEED + chỉ số dev300, code KHÔNG đổi so với lần chạy dev30) → `eval_generalization.py --subset dev100 --models yolox_s dino_swin_l` → `scripts/pair_screen.py --tag confirm_dev100 --pairs C2-PIA:PROBE-osfd+cls PIA:GEO-C0 C2-PIA:OSFD --exclude dev30`. Paired bootstrap 1000, BOOT_SEED 2026.

**Quy tắc (trên dev100, n = 100):**
- **P1 XÁC NHẬN** nếu Δ DINO-Swin-L (C2-PIA − C2) ≥ +3 VÀ cận dưới CI 95% > 0. Ngược lại **KHÔNG XÁC NHẬN**.
- P2 (PIA − GEO-C0) cùng quy tắc, báo riêng.
- XÁC NHẬN → hiện tượng có thật ở dev100: (i) phân tích cơ chế (decoder / theo ảnh) chạy lại trên dev100 trước khi kết luận; (ii) C2-PIA đủ điều kiện làm ứng viên sàng lọc dev300 theo plan_sota bước 3 — cần entry khóa riêng, user quyết. KHÔNG XÁC NHẬN → dừng giải thích lợi Swin-L của C2-PIA; chênh dev30 coi là nhiễu / không ổn định.
- Mô tả (không quyết định): CrossAvg₃, R101, từng target; Δ C2-PIA − OSFD; cùng các chỉ số trên 70 ảnh mới (dev100 \ dev30) — dev100 chứa dev30 nên không độc lập hoàn toàn.

**Kết quả @ dev100 (`results/ablation/confirm_dev100/screen.json`; `metrics_pia_dev100.json`, `generalization_pia_dev100.json`; log `results/confirm_dev100/run.log`) — theo quy tắc khóa: P1 XÁC NHẬN, P2 XÁC NHẬN.**

Relative AP drop % dev100 (n = 100), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- OSFD: 96.4 / 91.8 / 78.4 / 74.8 / 69.8 / 24.0 | 74.3
- GEO-C0: 96.3 / 87.7 / 73.4 / 72.6 / 70.5 / 22.4 | 72.2
- PIA: 94.5 / 88.2 / 76.4 / 74.2 / 70.4 / 28.5 | 73.7
- C2: 99.3 / 94.0 / 81.2 / 79.4 / 69.1 / 31.4 | 76.6
- **C2-PIA: 96.7 / 93.3 / 86.2 / 84.3 / 77.1 / 37.2 | 82.6**

- **P1 (C2-PIA − C2): DINO-Swin-L +5.8 [1.7, 9.1]** (p ≤ 0: 0.004) → XÁC NHẬN; CrossAvg₃ +6.0 [4.1, 8.8]; YOLOX +7.1 [4.5, 12.8], ConvNeXt +6.0, Swin-T +5.9 (cả hai CI > 0); R101 +0.4 [−2.1, 3.8].
- **P2 (PIA − GEO-C0): DINO-Swin-L +6.0 [2.3, 9.8]** → XÁC NHẬN; CrossAvg₃ +1.5 [0.3, 4.0].
- C2-PIA − OSFD (mô tả): DINO-Swin-L +13.2 [6.2, 15.5], CrossAvg₃ +8.2 [5.4, 11.3], Δ > 0 ở cả 3 target khác họ, R101 +1.5 (93.3 vs 91.8).
- 70 ảnh mới (dev100 \ dev30, mô tả): P1 Swin-L +3.6 [0.05, 7.3], CrossAvg₃ +6.3 [4.2, 9.8]; P2 Swin-L +5.7 [1.8, 8.7]; C2-PIA − OSFD Swin-L +14.2, CrossAvg₃ +6.4 → hiệu ứng tái lập trên ảnh không dùng ở dev30.
- Đọc chung: lợi của PIA trên DINO-Swin-L CÓ THẬT ở cả hai khung loss (≈ +6 điểm), và với loss cls còn kéo theo +6 CrossAvg₃ trên mọi target khác họ. Như vậy 3 phân tích cơ chế ở dev30 (đa scale, distortion, đầu ra decoder) đã thất bại trong việc giải thích một hiệu ứng có thật, không phải đang giải thích nhiễu. Theo quy tắc khóa: (i) phân tích cơ chế chạy lại trên dev100 trước khi kết luận; (ii) C2-PIA đủ điều kiện làm ứng viên sàng lọc dev300 (plan_sota bước 3) — cần entry khóa riêng, user quyết.

---

## 2026-10-04 — Chạy lại 2 phân tích cơ chế trên dev100 (user: không chạy dev300, phân tích tiếp trên dev100) — khóa TRƯỚC khi chạy

Theo quy tắc của entry xác nhận (i). Giữ NGUYÊN định nghĩa, cặp (P1 = C2-PIA vs C2, P2 = PIA vs GEO-C0), quy tắc và bootstrap (1000, BOOT_SEED 2026) của 2 entry gốc; chỉ đổi n = 30 → 100. Ô có PNG dev100: OSFD, GEO-C0, PIA, C2, C2-PIA (PIA-aff, PIA-d0.3, MORGD/SMOOTH chỉ có dev30 → bỏ).
1. **Distortion Swin-L** (entry "Pivot … failure case"): `mech_swinl_geo.py --subset dev100 --conds OSFD GEO-C0 PIA PROBE-osfd+cls C2-PIA --gen pia_dev100 --out swinl_geo_dev100.json`. SHALLOW-SPECIFIC / ENCODER như cũ. Spearman giữa ô chỉ còn 5 ô → chỉ mô tả.
2. **Đầu ra decoder** (entry "Phân tích đầu ra decoder"): `diag_decoder_output.py --subset dev100 --conds OSFD GEO-C0 PIA PROBE-osfd+cls C2-PIA --select maxlogit --out decoder_output_dev100.json`. Chọn q* bằng maxlogit (dự phòng đã kích hoạt ở dev30 — giữ cùng định nghĩa để so được giữa 2 cỡ mẫu); sanity in ra cả trung vị score q*. CLS / LOC như cũ.
Nếu kết luận dev100 khác dev30: dev100 là kết quả dùng (n lớn hơn, hiệu ứng AP đã xác nhận trên chính tập này); dev30 giữ trong log như bản thử.

**Kết quả @ dev100** (`results/ablation/mech/{swinl_geo_dev100,decoder_output_dev100}.json`, log `results/confirm_dev100/mech.log`).

**1. Distortion Swin-L — theo quy tắc: ENCODER ở cả P1 và P2, không SHALLOW → "một phần": PIA tạo distortion encoder khó sửa hơn nhưng KHÔNG qua stage nông; tiền đề đa scale vẫn không được ủng hộ.** (dev30: ENCODER chỉ P1 — dev30 thiếu lực.)
- bb1 0.067–0.068 ở cả 5 ô (lại bất biến); bb2 P1 ×1.04, P2 ×1.00.
- P1: bb4 ×1.13, neck5 ×1.21, encoder ×1.29 (enc3 ×1.37, enc4 ×1.44), A_enc +0.016 [+0.010, +0.022]. P2: bb4 ×1.08, encoder ×1.14 (enc4 ×1.27, enc5 ×1.36), A_enc +0.007 [+0.003, +0.012]. Tăng lớn nhất ở các mức encoder THÔ (stride 32–64).
- Giữa 5 ô, độ lớn distortion vẫn KHÔNG theo drop (ρ −0.2 đến −0.8): họ C2 distortion thấp hơn nhưng drop cao hơn. Trong cùng loss, PIA tăng cả distortion sâu/encoder lẫn drop.

**2. Đầu ra decoder — theo quy tắc: P1 không CLS (margin có, rank không) nhưng LOC → nhãn khóa trước "lợi ở định vị / gán query"; P2 giống hệt.**
- Sanity: forced sạch == natural (0.0), khớp `predict`; 955/982 GT có q* (97.3%); trung vị score sạch q* 0.56.
- P1 (C2-PIA − C2): d_margin **−0.19 [−0.30, −0.08]**, d_s −0.20 [−0.33, −0.07], d_rank_pct +0.007 [−0.003, +0.017] (không), d_iou **−0.020 [−0.030, −0.010]**. P2 (PIA − GEO-C0): d_margin −0.20 [−0.30, −0.10], d_rank +0.004 (không), d_iou −0.017 [−0.029, −0.006]. (dev30: P1 margin −0.006 — dev30 thiếu lực.)
- **Phân rã lỗi (736 GT phát hiện được ở ảnh sạch) KHÔNG khớp nhãn "định vị":** preserved C2-PIA **0.693** vs C2 0.746 (−5.3 điểm ≈ 39 GT), PIA 0.736 vs GEO-C0 0.769 (−3.3); phần mất thêm gần như toàn bộ là **conf/rank** (C2-PIA 0.265 vs C2 0.221; PIA 0.228 vs GEO-C0 0.198) — GT vẫn có query IoU ≥ 0.5 nhưng score < 0.3. Loc failure ≤ 0.7% và lost ≤ 2% ở mọi ô.
- Natural: token q* bị loại khỏi top-900 C2-PIA 23.4% vs C2 20.6%, PIA 21.3% vs GEO-C0 18.5% (+2.8 điểm cả hai cặp). Gán lại: max_q z_c giảm 0.36–0.53 (forced 1.2–1.5).
- FP/ảnh: C2-PIA 5.58 vs C2 5.20 (FP nền 3.59 vs 3.02); PIA 2.80 vs GEO-C0 2.98.
- **Đọc chung (không đổi nhãn khóa):** tiêu chí CLS trượt chỉ vì rank percentile không đổi; margin và score GT giảm có ý nghĩa, ΔIoU tuy CI < 0 nhưng rất nhỏ (−0.02, IoU sạch 0.84). Chỉ số gắn trực tiếp với AP (preserved@0.3) cho thấy lợi của PIA trên Swin-L là **hạ confidence lớp GT dưới ngưỡng trong khi vẫn định vị đúng**, cộng thêm phá chọn query nhẹ (+2.8 điểm deselection). Rank percentile trong 900 query có thể kém nhạy (q* sạch ở pct 0.009, adv ~0.07 ở mọi ô). Cơ chế phía feature: PIA làm lệch mạnh hơn ở feature sâu và đặc biệt mức encoder thô, encoder dập ít hơn (A_enc ↑) — hai phân tích nhất quán ở mức "PIA đẩy thêm một lượng nhỏ, đều" chứ không mở kênh mới. Hướng tiếp: user quyết.

---

## 2026-10-04 — Cải tiến MORGD (user): chẩn đoán vì sao MORGD âm → MORGD-v2 (phân vùng theo box) xếp chồng trên PIA @ dev30 — quy tắc khóa TRƯỚC khi chạy

**Giả thuyết:** MORGD chưa từng thực hiện được chuyển động riêng theo object. Cách ráp trường u = Σ K_i u_i / (Σ K_i + ε) với kernel Gaussian rộng bằng cỡ box trung bình hóa chuyển động của các object gần / chồng nhau (đã đo ở GRGD: tầng ráp trường làm mất ×0.37) → trường "neo theo object" ≈ warp mượt chung chung → object ≈ random. Prior thấp (3 lần âm: KG, A′1, MORGD); user chọn hướng này vì novelty.

**Bước 0 — chẩn đoán MORGD hiện tại (forward-only, `scripts/morgd_fidelity.py --version v1`):** dev30, 10 view/ảnh, seed như run_baselines (SEED + chỉ số dev300), `sample_anchors` + `flow_grid` gốc (λ = 0.25, s ∈ [0.8, 1.25], ε = 0.05). Mỗi object (MORGD): dời tâm thực tế = tâm AABB(mặt nạ chữ nhật warp) − tâm gốc; ratio_pos = ‖thực tế‖ / ‖Δ_i‖ (bỏ object |Δ_i| < 1 px), cos(thực tế, Δ_i). Cả MORGD và SMOOTH: D_pair_pos_box, D_pair_scale_box (như GRGD), RMS |u|.
- **Giả thuyết ĐƯỢC ỦNG HỘ** nếu median ratio_pos < 0.75 HOẶC D_pair_pos_box(MORGD) / D_pair_pos_box(SMOOTH) < 1.5 → sang bước 1. Ngược lại (MORGD đã thực hiện đúng mà vẫn ≈ random) → kết quả âm của MORGD là thật → **dừng nhánh**.

**Bước 1 — MORGD-v2 + cổng độ trung thực (forward-only):**
- Mỗi object i (box ở khung hiện tại; bản xếp chồng: box SAU PIA): t_i = (λ w_i r_x, λ h_i r_y), r ~ U(−1, 1), a_i = U(−ln 1.25, ln 1.25), λ = 0.25 (giữ của MORGD); gauge: trừ trung bình có trọng số diện tích ω (Σ ω t = 0, Σ ω a = 0); s_i = exp(a_i); c'_i = c_i + t_i.
- Ánh xạ ngược đồng dạng chính xác: u_i(q) = c_i + (q − c'_i)/s_i − q.
- Vùng của object i (khung ra) = AABB(box gốc ∪ box đã dời cỡ s_i·(w_i, h_i) tại c'_i) — phủ cả chỗ object rời đi để không để lại "bóng" object chưa gán nhãn. Trọng số mềm α_i(q) = σ((d_x)/τ_i)·σ((d_y)/τ_i), d = khoảng cách có dấu vào trong biên vùng, τ_i = 0.05·√(w_i h_i).
- Chồng lấn: ghép "over" theo diện tích giảm dần (object nhỏ ưu tiên, thường ở trước): u ← α_i u_i + (1 − α_i) u. Ngoài mọi vùng u = 0.
- SMOOTH-v2 (đối chứng): cùng cơ chế, CÙNG (t_i, s_i, kích thước theo chỉ số), tâm c_i đều ngẫu nhiên trên khung.
- RNG: (r, a) từ `random.Random(seed + 1)`, tâm ngẫu nhiên từ `random.Random(seed + 2)`; H của PIA và nhiễu σ như C2-PIA → mỗi view H, (t, s), σ trùng giữa các ô, chỉ khác tâm.
- GT co-transform: AABB mặt nạ chữ nhật warp, mask warp, bỏ instance rỗng / < 1 px (như MORGD).
- **Cổng (bản xếp chồng trên PIA, cả 3):** median ratio_pos (MORGD-v2) ≥ 0.75; D_pair_pos_box(MORGD-v2) / D_pair_pos_box(SMOOTH-v2) ≥ 1.5; RMS |u| MORGD-v2 / SMOOTH-v2 ∈ [0.9, 1.1]. Trượt → dừng nhánh, KHÔNG attack.
- Unit test trước cổng: λ = 0 hoặc không có GT → view trùng tuyệt đối pia_views; 1 object không chồng lấn → box dời đúng c' ± s·(w, h)/2 (≤ 2 px); xa mọi vùng u = 0; σ = 0, adv = sạch → view == companion; ô cũ trùng HEAD; chạy được, L∞ đúng.

**Bước 2 — attack @ dev30 (mô tả, khung C2: L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), MI μ = 1, ε = 5, α = 1, B = 50, 2 view/step, 1 backward, σ = 6, feature sạch + GT co-transform):** C2-PIA (PNG có sẵn, mốc) | C2-PIA-SMOOTH2 | **C2-PIA-MORGD2** | C2-MORGD2 (MORGD-v2 không PIA). Hiệu chính **Δ_object = C2-PIA-MORGD2 − C2-PIA-SMOOTH2**; kèm Δ_extra = C2-PIA-SMOOTH2 − C2-PIA.

**Bước 3 — quyết định (dev30, điểm ước lượng):** đi tiếp (khóa entry xác nhận dev100: Δ_object DINO-Swin-L hoặc CrossAvg₃ ≥ +3 VÀ cận dưới CI > 0) nếu Δ_object ≥ +2 trên DINO-Swin-L HOẶC CrossAvg₃ VÀ Δ_object > 0 ở ≥ 3/5 target khác surrogate (R101, ConvNeXt-T, Swin-T, YOLOX-S, DINO-Swin-L). **Δ_object ≤ 0 trên cả DINO-Swin-L và CrossAvg₃ → dừng hẳn nhánh MORGD.** Không thêm biến thể cứu vãn.

**Bước 0 — kết quả (`results/morgd2/fidelity_v1.json`, 3569 object-view): GIẢ THUYẾT ĐƯỢC ỦNG HỘ → sang bước 1.** median ratio_pos 0.91 (tứ phân vị 0.48 / 0.91 / 2.12) nhưng **cos(dời thực tế, Δ_i) trung bình chỉ 0.29** — object có di chuyển cỡ mục tiêu nhưng SAI HƯỚNG (bị object lân cận chi phối). D_pair_pos_box MORGD **0.215 < SMOOTH 0.278** (tỉ lệ 0.78 < 1.5): neo theo object còn đổi quan hệ tương đối giữa GT ÍT hơn neo ngẫu nhiên. D_pair_scale_box 0.126 vs 0.121; RMS |u| 45.5 vs 44.3 px. → MORGD v1 chưa từng thực hiện được chuyển động riêng theo object; kết quả âm của nó không trả lời được câu hỏi "theo object có ích không".

**Bước 1 — code + unit test (TRƯỚC cổng):** `attack/methods/morgd2.py` (`sample_motion`, `field`, `morgd2_views`); `geo_osfd_attack(geo="pia_morgd2" | "pia_smooth2" | "morgd2")`; ô `C2-PIA-MORGD2`, `C2-PIA-SMOOTH2`, `C2-MORGD2` (stats `morgd2`); `scripts/morgd_fidelity.py --version v2`. **Sửa spec phát hiện ở unit test (trước mọi cổng / attack):** vùng = AABB(gốc ∪ đã dời) **nới thêm 3τ_i mỗi phía** — không nới thì cạnh box nằm đúng chỗ α = 0.5 nên chỉ dời nửa (lệch 6.1 px); sau sửa lệch 0.64 px. N = 1 → gauge cho t = 0, s = 1 → bỏ warp (đồng nhất). `scripts/morgd2_unit_test.py` ĐẠT: λ = 0 và không GT → trùng tuyệt đối pia_views; 2 object tách xa → box dời đúng c' ± s·(w, h)/2 (≤ 0.64 px); xa mọi vùng |u| 2e-5 px; σ = 0, adv = sạch → view == companion (3 ô); GEO-C0 / PIA / MORGD trùng tuyệt đối HEAD, C2-PIA lệch HEAD/mới 0.036 = HEAD/HEAD; C2-PIA-MORGD2 / -SMOOTH2 có mode PIA + nhiễu σ trùng tuyệt đối C2-PIA mỗi view, C2-MORGD2 tất định; 3 ô chạy được, L∞ đúng, forward no_grad = 2 × step.

**Bước 1 — cổng v2 (`results/morgd2/fidelity_v2.json`, bản xếp chồng trên PIA, 3190 object-view): TRƯỢT → theo quy tắc khóa DỪNG nhánh MORGD, KHÔNG attack.**
- median ratio_pos **0.94** ≥ 0.75 ĐẠT (tứ phân vị 0.61 / 0.94 / 1.10; v1: 0.48 / 0.91 / 2.12); **cos trung bình 0.62** (v1 0.29).
- **D_pair_pos_box MORGD-v2 / SMOOTH-v2 = 1.22 < 1.5 TRƯỢT** (0.289 vs 0.236; v1 0.78). D_pair_scale_box 0.307 vs 0.217 (×1.41, không thuộc cổng).
- RMS |u| 45.1 vs 44.8 px (×1.01) ĐẠT.
- Đọc: v2 sửa được việc thực hiện chuyển động theo object (hướng đúng hơn gấp đôi, biên độ sát mục tiêu), và object đã vượt random (1.22 so với 0.78 ở v1). Nhưng đối chứng neo ngẫu nhiên vẫn đổi quan hệ giữa các GT tới ~82% mức của bản neo theo object: các vùng cỡ GT đặt ngẫu nhiên vẫn trùm và dời cứng một phần object thật. Về cấu trúc, mọi warp cục bộ cỡ object đều đổi hình học tương đối của GT, nên tương phản object − random chỉ còn ~20% đại lượng mà giả thuyết nói tới → so 2 ô attack ở dev30 (CI ±5) gần như chắc chắn không phân biệt được. Ngưỡng 1.5 đã khóa trước; không nới sau khi thấy số.

**User BỎ QUA cổng (quyết định SAU khi thấy số cổng v2) — chạy bước 2 để xem xu hướng.** Kết quả bước 2 chỉ MÔ TẢ, không dùng để biện minh cho nhánh; quy tắc bước 3 vẫn áp dụng như đã khóa (Δ_object ≤ 0 trên cả DINO-Swin-L và CrossAvg₃ → dừng hẳn), nhưng một kết quả dương cũng phải được đọc kèm việc cổng đã trượt (tương phản object − random chỉ ~20% về hình học). Chạy: `run_baselines.py --split dev300 --subset dev30 --methods OSFD PROBE-osfd+cls C2-PIA C2-MORGD C2-PIA-SMOOTH2 C2-PIA-MORGD2 C2-MORGD2 --out metrics_morgd2_dev30.json` → `eval_generalization.py --subset dev30 --models yolox_s dino_swin_l` → `pair_screen.py --subset dev30 --tag morgd2_dev30 --pairs C2-PIA-MORGD2:C2-PIA-SMOOTH2 C2-PIA-SMOOTH2:C2-PIA C2-PIA-MORGD2:C2-PIA C2-MORGD2:C2-MORGD C2-MORGD2:C2-PIA`. Log `results/morgd2/run.log`.

**Bước 2 — kết quả @ dev30 (mô tả; cổng đã trượt, user bỏ qua) — `results/ablation/morgd2_dev30/screen.json`, `metrics_morgd2_dev30.json`, `generalization_morgd2_dev30.json`, log `results/morgd2/run.log`.**

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ (runtime s/ảnh):
- C2-PIA: 99.5 / 94.8 / 86.8 / 85.9 / 78.2 / **34.7** | **83.6** (11.3)
- C2-PIA-SMOOTH2: 98.3 / 92.6 / 78.6 / 83.3 / 76.9 / 22.5 | 79.6 (12.6)
- C2-PIA-MORGD2: 96.8 / 89.3 / 78.3 / 76.9 / 71.7 / 23.3 | 75.7 (12.4)
- C2-MORGD (v1): 99.9 / 96.0 / 87.7 / 83.5 / 72.8 / 27.3 | 81.3
- C2-MORGD2: 99.7 / 93.6 / 80.2 / 81.9 / 70.8 / 21.5 | 77.6 (11.9)

- **Δ_object (C2-PIA-MORGD2 − C2-PIA-SMOOTH2): DINO-Swin-L +0.7 [−3.3, +3.4], CrossAvg₃ −4.0 [−7.4, −0.9]**; > 0 chỉ ở 1/5 target (Swin-L) → theo bước 3: KHÔNG đi tiếp (điều kiện "dừng hẳn" Δ ≤ 0 trên CẢ HAI không kích hoạt vì Swin-L +0.7, nhưng điều kiện đi tiếp trượt rõ) → **dừng nhánh MORGD**.
- **Thêm warp cục bộ lên PIA làm attack YẾU đi mạnh, bất kể neo:** SMOOTH2 − C2-PIA: Swin-L −12.2 [−15.9, −4.8], CrossAvg₃ −4.0; MORGD2 − C2-PIA: Swin-L −11.4, CrossAvg₃ −7.9 [−12.6, −5.0]; cả R50 white-box (99.5 → 96.8–98.3) và R101 (−2.2 / −5.6) giảm → tối ưu trên surrogate kém đi, không chỉ transfer.
- **MORGD-v2 đứng riêng kém v1:** C2-MORGD2 − C2-MORGD: Swin-L −5.8 [−9.1, −2.4], CrossAvg₃ −3.7 — trường cứng từng mảnh (seam) kém trường mượt.
- Đọc (giả thuyết, chưa kiểm): **có điểm tối ưu của đa dạng hình học ở B cố định.** PIA δ_max 0.6 có vẻ đã gần mức đó; thêm biến đổi (dù toàn cục RaPA hay cục bộ SMOOTH2/MORGD2) làm gradient giữa các view mất tương quan → với 50 step × 2 view thì δ không tích lũy được (cùng chiều với MORGD-RaPA −25). Hệ quả kiểm được: tăng số view mỗi backward (giảm phương sai gradient) sẽ cho phép biến đổi mạnh hơn mà không mất utility. Hướng tiếp: user quyết.

---

## 2026-10-04 — Giai đoạn A: PIA hoạt động nhờ YẾU TỐ hình học nào? Ablation leave-one-out các thành phần homography (khung loss feature thuần, KHÔNG C2) — quy tắc khóa TRƯỚC khi code/chạy

**Bối cảnh (user):** bỏ khung C2, giữ loss feature OSFD; mục tiêu là một dạng đa dạng hình học vượt SOTA và có novelty so với PIA (classification). Trong khung thuần @ dev100, PIA chưa vượt OSFD trên CrossAvg₃ (73.7 vs 74.3), chỉ hơn trên DINO-Swin-L (28.5 vs 24.0). Cách làm như plan_sota với OSFD: phân tích SOTA → tìm nguyên liệu → khuếch đại (Giai đoạn B, khóa sau). Giả thuyết nghiêng về (chưa kiểm): **co giãn** — đưa object qua ranh giới mức FPN (resize ≤ 1.1× của RRB vô dụng; PIA dời góc tới 30%).

**Tách homography (Hartley–Zisserman, H = H_S·H_A·H_P):** H của PIA (pixel ra → nguồn) đổi sang tọa độ TÂM ẢNH: H' = C⁻¹ H C, C = tịnh tiến gốc về tâm khung (để xoay / co giãn quanh tâm, không lẫn tịnh tiến), chuẩn hóa H'[2,2] = 1. Viết H' = [[A, t], [vᵀ, 1]]: H_P = [[I, 0], [vᵀ, 1]] (phối cảnh v, 2-DOF); M = A − t vᵀ = s·R(θ)·K, QR với đường chéo dương: R xoay (θ), s = √det (co giãn đẳng hướng), K = [[k₁, k₂], [0, 1/k₁]] (lệch tỉ lệ k₁ + trượt k₂); t tịnh tiến (2-DOF). Tổng 8-DOF.

**Ô (mỗi ô = PIA với ĐÚNG 1 thành phần đặt về đồng nhất, các thành phần còn lại giữ nguyên giá trị đã rút; ghép lại H'' = H_S·H_A·H_P → H = C H'' C⁻¹):** PIA-noT (t = 0), PIA-noR (θ = 0), PIA-noS (s = 1), PIA-noK (K = I), PIA-noP (v = 0). Khung = ô `PIA` (loss feature OSFD k = 3, MI μ = 1, ε = 5, α = 1, B = 50, 2 view/step, 1 backward, σ = 6, feature sạch co-transform, δ_max = 0.6, π paper). RNG như PIA (rút góc giống hệt, chỉ sửa H sau khi rút) → mỗi view của ô X = view PIA tương ứng bỏ 1 thành phần.

**Báo cáo kèm (gây nhiễu liều):** mỗi ô ghi |u| trung bình mỗi view (px) và phân bố từng thành phần trong PIA (|t|, |θ|, |ln s|, |ln k₁|, |k₂|, ‖v‖·cỡ ảnh). Bỏ một thành phần vừa đổi LOẠI vừa giảm BIÊN ĐỘ → phải đọc Δ kèm Δ|u|.

**Chạy & quy tắc:** dev30 trước (mô tả, xem xu hướng), rồi mở rộng ĐỦ dev100 cho cả 5 ô (không chọn lọc theo dev30; cùng seed, resume). **Quyết định chỉ ở dev100** (n = 100, paired bootstrap 1000, BOOT_SEED 2026), Δ = drop(PIA) − drop(ô), tính RIÊNG cho DINO-Swin-L và CrossAvg₃ (như ablation OSFD bước 1): cận dưới CI ≥ 2 → **nguyên liệu**; cận trên CI < 2 → **không phải**; còn lại → **mơ hồ**. PIA trên dev100 = PNG có sẵn (`metrics_pia_dev100.json`).

**Unit test trước khi chạy:** (1) ghép lại các thành phần == H (sai số tương đối < 1e-9, 10 000 mẫu); (2) mỗi ô: tách lại H đã bỏ → thành phần bị bỏ = đồng nhất, các thành phần khác không đổi; (3) mode 3/2/1 (v = 0 sẵn) → lưới PIA-noP ≈ PIA (≤ 1e-4); mode 1 → PIA-noT ≈ đồng nhất; (4) không truyền drop → PIA trùng tuyệt đối HEAD; (5) 5 ô chạy được, L∞ đúng, RNG (mode, nhiễu σ) trùng PIA.

**Code + unit test (TRƯỚC khi chạy):** `pia.decompose_h` / `compose_h` / `drop_factor` (tọa độ tâm, QR đường chéo dương), `pia_views(drop=, factor_log=)`, `geo_osfd_attack(pia_drop=)`; ô `PIA-noT/R/S/K/P` (stats `pia_factors`). `scripts/pia_factor_unit_test.py` ĐẠT: ghép lại == H (sai số tương đối ≤ 2.4e-14, 10 000 mẫu); bỏ 1 thành phần → thành phần đó = đồng nhất, khác không đổi (≤ 2.8e-13); mode 3/2/1: PIA-noP ≈ PIA (6e-8), mode 1: PIA-noT ≈ đồng nhất; PIA không drop trùng tuyệt đối HEAD; 5 ô chạy được, L∞ đúng, mode + nhiễu σ trùng PIA.
**Phân bố thành phần PIA (10 000 mẫu, khung 800 × 1199, trung vị):** |t| 157.5 px, |θ| 3.1°, |ln s| 0.104, |ln k₁| 0.091, |k₂| 0.140, v = 0 ở ~60% view (mode 3/2/1). **|u| trung bình:** PIA 194 px; noT 91, noR 192, noS 190, noK 181, noP 198 → biên độ PIA chủ yếu là TỊNH TIẾN (kèm vùng lấp 0 khi dịch); co giãn của PIA chỉ ~±10% (giả thuyết "vượt mức FPN" kém khả năng hơn dự kiến — ghi trước khi có kết quả).

**Đổi thứ tự (user, trước khi có số dev30):** chỉ chạy dev30 để xem xu hướng, TẠM HOÃN phần mở rộng dev100 (pipeline dừng sau khi xong dev30). Quy tắc "nguyên liệu / không phải / mơ hồ" vẫn chỉ áp dụng ở dev100 khi (nếu) chạy; số dev30 chỉ mô tả.

**Kết quả @ dev30 (mô tả; dev100 hoãn) — `results/ablation/piafac_dev30/screen.json`, `metrics_piafac_dev30.json`, `generalization_piafac_dev30.json`, log `results/pia_factor/run.log`.** (Pipeline dừng chủ động khi bắt đầu dev100; watcher chỉ dừng script cha, tiến trình con `run_baselines` dev100 chạy sót ~4 phút → đã attack thêm 11 ảnh dev100 \ dev30 cho PIA-noT (cùng seed, giữ lại để resume), rồi bị dừng tay. Không có eval dev100.)

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | |u| px:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8
- PIA: 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / 28.8 | 79.5 | ~194
- PIA-noT: 97.0 / 91.1 / 82.0 / 81.8 / 77.7 / 29.8 | 80.5 | 90.5
- PIA-noR: 97.0 / 90.2 / 73.8 / 79.7 / 77.4 / 31.1 | 77.0 | 186.8
- PIA-noS: 97.1 / 89.8 / 79.0 / 80.3 / 78.9 / 27.5 | 79.4 | 181.8
- PIA-noK: 97.2 / 91.3 / 85.2 / 82.3 / 76.1 / 27.8 | 81.2 | 177.3
- PIA-noP: 96.8 / 90.4 / 81.8 / 75.9 / 76.0 / 25.9 | 77.9 | 190.8

Δ = PIA − ô (dương = thành phần có ích), DINO-Swin-L | CrossAvg₃ [CI 95%]:
- **T (tịnh tiến): −1.0 [−4.5, +3.0] | −1.0 [−2.7, +0.6]** — bỏ T giảm |u| hơn một nửa (194 → 91 px) mà KHÔNG mất gì (hơi tốt hơn) → tịnh tiến (phần lớn biên độ của PIA) không phải nguyên liệu; giả thuyết "đa dạng vị trí/pha lưới qua tịnh tiến lớn" không được ủng hộ.
- R (xoay): −2.3 [−5.4, +0.6] | +2.5 [−0.8, +4.2] — giúp ConvNeXt (+5.9), hơi hại Swin-L.
- S (co giãn): +1.2 | +0.1 — ≈ 0 → giả thuyết "co giãn vượt mức FPN" không được ủng hộ (co giãn PIA chỉ ~±10%).
- K (lệch tỉ lệ + trượt): +1.0 | −1.7 (ConvNeXt −5.5 [−10.6, −1.4]: bỏ K TỐT hơn cho ConvNeXt).
- **P (phối cảnh): +2.8 [−1.1, +6.0] | +1.6 [−0.2, +3.5]** — thành phần duy nhất dương ở cả Swin-L và CrossAvg₃ (YOLOX +2.4 [+0.3, +7.5], Swin-T +4.5); bỏ P gần như không giảm |u| (191 px) → lợi của P là do LOẠI biến đổi, không do biên độ.
- Đọc chung (n = 30, phần lớn CI chứa 0): nguyên liệu khả dĩ nhất của PIA là **số hạng phối cảnh**; tịnh tiến chiếm phần lớn biên độ nhưng "lãng phí" (bỏ đi không mất, khớp quan sát trần đa dạng–hiệu dụng). (Lưu ý: PIA-aff @ dev30 — đổi phân phối mode sang toàn 6-DOF — có Swin-L 27.9 ≈ PIA; phép bỏ P ở đây giữ nguyên các thành phần khác nên sạch hơn.) Muốn kết luận phải chạy dev100 (quy tắc khóa).

---

## 2026-10-04 — Giai đoạn B (thử nhanh @ dev30, user): "đa dạng mô hình camera" — xoay camera ảo thuần phối cảnh + méo ống kính (ngoài họ homography) — quy tắc khóa TRƯỚC khi code/chạy

**Bối cảnh:** Giai đoạn A @ dev30 gợi ý phối cảnh là nguyên liệu của PIA, tịnh tiến (≥ nửa biên độ) vô ích. User chọn thử thẳng method mới ở dev30 (bỏ xác nhận dev100 của Giai đoạn A). Rủi ro novelty đã nêu: xoay camera ảo vẫn trong họ homography của PIA (chỉ là phân phối rút khác) → thêm **méo ống kính xuyên tâm** — biến đổi camera toàn cục, mượt, có nghĩa vật lý, KHÔNG biểu diễn được bằng homography.

**Khung:** như ô `PIA` (loss feature OSFD k = 3, MI μ = 1, ε = 5, α = 1, B = 50, 2 view/step, 1 backward, σ = 6 cộng vào adv, feature sạch co-transform cùng lưới, KHÔNG cls). Seed ảnh SEED + chỉ số dev300; tham số biến đổi rút từ `random` của Python mỗi view; vùng ngoài ảnh nguồn lấp 0.

**Biến đổi (ánh xạ ngược pixel ra q → nguồn, khung model h × w, tâm c = ((w−1)/2, (h−1)/2)):**
- **CAM-R(θ):** K = [[f, 0, c_x], [0, f, c_y], [0, 0, 1]], f = max(h, w); yaw, pitch ~ U(−θ, θ) độc lập, roll = 0; H = K·R_y(yaw)·R_x(pitch)·K⁻¹; nguồn = H·q (chia đồng nhất). Hai liều θ = 15°, 30° (khai báo trước, không tune thêm).
- **LENS(κ):** d = q − c, ρ = ‖d‖ / R, R = nửa đường chéo; nguồn = c + d·(1 + k·ρ²), k ~ U(−κ, κ), κ = 0.25 (k > 0 / < 0: pincushion / barrel theo chiều ánh xạ ngược).
- **CAM-R+LENS:** nguồn = H·(c + d·(1 + kρ²)) — méo ống kính ở khung ra, rồi xoay camera; θ = 15°, κ = 0.25.
- GT co-transform (khi cần cho khung cls về sau): AABB mặt nạ chữ nhật warp như MORGD; không dùng trong khung thuần.

**Ô @ dev30 (mô tả):** CAM-R15, CAM-R30, LENS, CAM-R15+LENS; mốc: PIA, PIA-noT, OSFD, GEO-C0 (PNG có sẵn). Báo |u| trung bình mỗi view.
**Đọc xu hướng (khai báo trước, không phải cổng):** ứng viên "triển vọng" nếu ≥ PIA + 2 trên DINO-Swin-L VÀ ≥ PIA trên CrossAvg₃ (điểm ước lượng, dev30) → khóa tiêu chí xác nhận dev100 (so PIA, ≥ +3 Swin-L, cận dưới CI > 0). LENS hoặc CAM-R15+LENS triển vọng mới hỗ trợ được claim "ngoài homography".
**Unit test trước khi chạy:** θ = 0, κ = 0 → lưới ≈ đồng nhất; yaw thuần → tâm ảnh ánh xạ tới c_x + f·tan(yaw) (đúng chiều); LENS k > 0 → góc ảnh ánh xạ ra ngoài đúng k·R; σ = 0, adv = sạch → view == companion; ô cũ trùng HEAD; chạy được, L∞ đúng.

**Code + unit test (TRƯỚC khi chạy):** `attack/methods/camgeo.py` (`rotation_homography`, `camera_grid`, `cam_views`); `geo_osfd_attack(geo="cam", cam_theta=, cam_kappa=)`; ô CAM-R15, CAM-R30, LENS, CAM-R15+LENS. **Sửa spec phát hiện ở unit test (trước khi chạy):** xoay camera bậc nhất = dịch f·tan θ (yaw 15° ≈ 320 px; CAM-R15 |u| 254 px — chính thành phần tịnh tiến Giai đoạn A cho là vô ích) → **căn lại tâm**: H ← T·H, T đưa ảnh của tâm về tâm; chỉ giữ méo phối cảnh. `scripts/camgeo_unit_test.py` ĐẠT: θ = κ = 0 → đồng nhất (8e-6 px); yaw 10° căn tâm → tâm giữ nguyên, mép trái / phải dời 31.4 / 78.3 px (keystone bất đối xứng); LENS k = 0.2 → góc ra ngoài đúng k·R; σ = 0, adv = sạch → view == companion; GEO-C0 / PIA / PIA-noT trùng tuyệt đối HEAD; 4 ô chạy được, L∞ đúng. **|u| trung bình (sau sửa):** CAM-R15 22.3, CAM-R30 84.9, LENS 21.4, CAM-R15+LENS 31.2 px (PIA ~194, PIA-noT ~91).

---

## 2026-10-04 — Thí nghiệm cơ chế: "đa dạng có ích = biến đổi NGOÀI nhóm đối xứng của surrogate"? (gradient surrogate, dev30) — quy tắc khóa TRƯỚC khi chạy

**Mục tiêu (user):** hiểu biến đổi thế nào cho lợi tối đa, rồi diễn đạt bằng phép biến đổi cơ bản. **Giả thuyết:** biến đổi mà surrogate (gần) đồng biến — tịnh tiến (conv), co giãn đẳng hướng (FPN + huấn luyện đa scale) — cho gradient view chỉ là gradient gốc dời/co lại → trung bình không thêm hướng mới (khớp: tịnh tiến PIA vô ích, resize RRB vô dụng, co giãn PIA ≈ 0). Biến đổi phá đối xứng — xoay, trượt/lệch tỉ lệ, phối cảnh, méo ống kính — cho gradient khác thật → triệt tiêu phần nhiễu đặc thù cách surrogate xử lý hướng/hình dạng cục bộ (khớp: xoay là nguyên liệu RRB, phối cảnh là thành phần dương của PIA). Điểm chưa khớp đã biết: bỏ xoay khỏi PIA làm Swin-L tốt hơn (−2.3 @ dev30).

**Đo (`scripts/mech_equivariance.py` → `results/mechanism/equivariance.json`):** surrogate R50, loss feature OSFD (backbone 4 stage, k = 3), đích feature co-transform (F(T(x_sạch))), KHÔNG nhiễu σ. Gradient theo ảnh khung model g_T = ∇ₓ L(T(x)) (backprop qua grid_sample → tự về khung gốc); g_I = gradient không biến đổi. **Đa dạng hiệu dụng d_T = 1 − cos(g_T, g_I)** trên các pixel mà view phủ (mặt nạ nguồn hợp lệ), kèm tỉ lệ trùng dấu. Hai điểm δ: δ = 0 và δ = PNG OSFD (adv cuối) của cùng ảnh.
- **Họ biến đổi cơ bản (tham số quanh tâm ảnh):** tịnh tiến (hướng ngẫu nhiên), co giãn đẳng hướng (s hoặc 1/s), xoay, trượt (k₂ ngang hoặc dọc), phối cảnh (xoay camera căn tâm, yaw hoặc pitch), méo ống kính (k).
- **Cùng biên độ:** mỗi họ hiệu chỉnh tham số để |u| trung bình = 10 / 30 / 60 px (chia đôi trên khung từng ảnh); 4 lần rút (dấu / hướng) mỗi (ảnh, họ, mức). dev30, n = 30.
- **Quy tắc:** NGUYÊN LÝ ĐƯỢC ỦNG HỘ nếu ở CẢ 3 mức biên độ và CẢ 2 điểm δ: min(d_xoay, d_phối cảnh) − max(d_tịnh tiến, d_co giãn) > 0 với cận dưới CI 95% paired bootstrap theo ảnh (1000, BOOT_SEED 2026) > 0. Ngược lại: không được ủng hộ. Trượt và méo ống kính: mô tả.
- Mô tả kèm: so d_T với các ô đã chạy (tương quan giữa họ biến đổi và lợi transfer ở Giai đoạn A/B) — không quyết định.

**Kết quả @ dev30 (mô tả) — `results/ablation/cam_dev30/screen.json`, `metrics_cam_dev30.json`, `generalization_cam_dev30.json`, log `results/camgeo/run.log`.** |u| thực: CAM-R15 22.6, CAM-R30 78.5, LENS 20.1, CAM-R15+LENS 31.2 px.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- PIA: 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / **28.8** | 79.5
- PIA-noT: 97.0 / 91.1 / 82.0 / 81.8 / 77.7 / **29.8** | 80.5
- CAM-R15: 98.2 / 92.1 / 81.8 / 81.0 / 74.5 / 25.1 | 79.1
- **CAM-R30: 96.8 / 92.3 / 84.0 / 82.7 / 81.5 / 21.3 | 82.7**
- LENS: 97.8 / 91.1 / 81.4 / 79.5 / 69.5 / 24.8 | 76.8
- CAM-R15+LENS: 98.7 / 91.5 / 81.9 / 81.2 / 77.4 / 25.3 | 80.2

- **Không ô nào "triển vọng"** theo tiêu chí khai báo (≥ PIA + 2 trên Swin-L VÀ ≥ PIA trên CrossAvg₃).
- **CAM-R30 (phối cảnh thuần, 78 px): CrossAvg₃ +3.2 [+0.3, +5.7] so PIA** (cao nhất trong mọi ô hình học khung thuần đến nay; ConvNeXt +4.3, YOLOX +3.1, Swin-T +2.3, R101 +2.4) **nhưng DINO-Swin-L −7.4 [−10.8, −0.4]**. Đánh đổi rõ: phối cảnh thuần mạnh giúp target CNN / Mask R-CNN / YOLOX, hại DINO-Swin-L.
- LENS: yếu (CrossAvg₃ −2.7, YOLOX −8.9); thêm LENS lên CAM-R15 gần như không đổi (+0.2 Swin-L, +1.1 CrossAvg₃) → claim "ngoài homography" KHÔNG được ủng hộ ở dev30.
- Đọc: lợi Swin-L của PIA KHÔNG đến từ phối cảnh thuần đứng riêng; nó trải trên nhiều thành phần (Giai đoạn A: bỏ P −2.8, S −1.2, K −1.0, bỏ R +2.3) — có thể cần TỔ HỢP. Hai nhóm target phản ứng khác nhau với cùng một loại biến đổi → một tiêu chí "đa dạng" duy nhất có thể không đủ; thí nghiệm cơ chế (đang chạy) sẽ cho biết đa dạng gradient trên surrogate có theo được nhóm nào.

**Kết quả thí nghiệm cơ chế (`results/mechanism/equivariance.json`, n = 30, 4 lần rút) — NGUYÊN LÝ KHÔNG ĐƯỢC ỦNG HỘ (theo quy tắc khóa; sai cả 3 mức × 2 điểm δ, và sai CHIỀU).**

d_T = 1 − cos (δ = 0), |u| = 10 / 30 / 60 px: tịnh tiến 0.628 / 0.634 / 0.634; co giãn 0.626 / 0.658 / 0.714; xoay 0.620 / 0.658 / 0.718; trượt 0.500 / 0.581 / 0.699; phối cảnh 0.574 / 0.651 / 0.684; méo ống kính 0.571 / 0.661 / 0.751. (δ = OSFD gần như y hệt.) min(xoay, phối cảnh) − max(tịnh tiến, co giãn): −0.054 [−0.065, −0.045] / −0.007 [−0.012, −0.002] / −0.030 [−0.037, −0.023] — ÂM, CI < 0.

- **Tịnh tiến chỉ 10 px đã làm gradient mất tương quan ~63% (cos ≈ 0.37), ngang mọi họ khác.** Gradient theo pixel của surrogate cực nhạy với BẤT KỲ phép lấy mẫu lại nào (lệch pha stride + nội suy bilinear) → đa dạng gradient mức pixel BÃO HÒA ngay ở biên độ nhỏ, không phân biệt được loại biến đổi, và không theo được transfer (tịnh tiến: d cao nhưng vô ích về transfer).
- Hệ quả: "đa dạng hiệu dụng" đo bằng cos gradient pixel KHÔNG phải trục giải thích; giả thuyết "đa dạng có ích = phá đối xứng" (theo cách đo này) bị bác. Không thêm biến thể đo hậu kiểm vào quy tắc này. Hướng tiếp theo đã khai báo trước: đo ĐỒNG HƯỚNG giữa gradient surrogate (trung bình qua biến đổi) và gradient target — dùng target CHỈ để phân tích, không sinh attack. User quyết.

---

## 2026-10-04 — TẠM DỪNG (user): tổng kết chuỗi "hiểu bản chất đa dạng hình học" + việc dở dang

**Mục tiêu đang theo (user):** hiểu biến đổi hình học NÀO, THẾ NÀO cho lợi tối đa (nhất là DINO-Swin-L), rồi diễn đạt bằng phép biến đổi cơ bản → method mới có novelty ở NGUYÊN LÝ (không phải thêm một họ biến đổi như PIA — PIA đã chiếm luận điểm "phủ phân cấp 2→8-DOF"). Khung ưu tiên: loss feature thuần (KHÔNG C2).

**Đã biết (dev30 trừ khi ghi khác):**
1. Lợi của PIA trên Swin-L là THẬT (dev100: PIA − GEO-C0 +6.0 [2.3, 9.8]); trong khung thuần PIA chưa vượt OSFD trên CrossAvg₃ (dev100 73.7 vs 74.3).
2. Giai đoạn A (bỏ từng thành phần homography): phối cảnh là thành phần duy nhất dương ở cả Swin-L (+2.8) và CrossAvg₃ (+1.6); tịnh tiến chiếm > nửa biên độ PIA nhưng VÔ ÍCH (bỏ đi hơi tốt hơn: PIA-noT Swin-L 29.8, CrossAvg₃ 80.5); co giãn ≈ 0; xoay giúp ConvNeXt, hơi hại Swin-L.
3. Giai đoạn B: phối cảnh thuần căn tâm CAM-R30 (78 px) cho CrossAvg₃ 82.7 (+3.2 [+0.3, +5.7] so PIA — cao nhất khung thuần) NHƯNG Swin-L −7.4; LENS yếu, không cộng thêm → hai nhóm target (CNN / Mask R-CNN / YOLOX vs DINO-Swin-L) phản ứng NGƯỢC nhau với cùng loại biến đổi; lợi Swin-L của PIA cần TỔ HỢP thành phần.
4. Xếp chồng thêm biến đổi lên PIA làm hại (warp cục bộ −11 Swin-L, RaPA −25) → có trần đa dạng–hiệu dụng ở B / số view cố định.
5. Trục "LƯỢNG" đều thất bại: lượng distortion feature không theo AP drop giữa các ô; đa dạng gradient pixel (1 − cos) bão hòa ngay từ tịnh tiến 10 px (~0.63, ngang mọi họ) → nguyên lý "phá đối xứng" (theo cách đo này) bị bác.
6. Nhánh đã đóng: MORGD v1/v2, GRGD, object-aware nói chung (4 lần âm), đa scale/stage nông, Repair-Robust, RaPA, LENS.

**Việc dở dang — ĐỀ XUẤT, CHƯA KHÓA, CHƯA CODE:** thí nghiệm "đồng hướng gradient surrogate ↔ target" (dùng target CHỈ để phân tích, không sinh attack): mỗi phép biến đổi cơ bản (tịnh tiến, co giãn, xoay, trượt, phối cảnh căn tâm, méo ống kính) ở cùng |u| (30, 60 px), gradient R50 (loss feature OSFD, đích co-transform) TRUNG BÌNH qua 8 view → cos với gradient target trên cùng ảnh sạch, cho DINO-Swin-L, ConvNeXt-T, YOLOX-S. Loss target đề xuất: loss PHÂN LOẠI có GT của chính target (decoder: lợi Swin-L nằm ở hạ confidence lớp GT) + loss feature backbone target để đối chiếu. Câu hỏi: phép / liều nào làm gradient trung bình R50 giống gradient Swin-L nhất, có khác nhóm CNN không, có khớp thứ hạng transfer đã đo không. Ước tính ~10 phút code + ~15 phút chạy dev30. Phải khóa quy tắc (định nghĩa loss target, tiêu chí "khớp thứ hạng") TRƯỚC khi chạy.

**Khác:** GO @ n300 cho C2 vẫn tạm hoãn (110/300); C2-PIA đã xác nhận @ dev100 (Swin-L +5.8 so C2, CrossAvg₃ +8.2 so OSFD) — chưa sàng lọc dev300 (user chọn không chạy dev300 lúc này). 11 ảnh dev100 \ dev30 của PIA-noT có sẵn (resume được). Nhiều commit CHƯA push.

---

## 2026-10-04 — Quy luật "đa dạng có ích = biến đổi NGOÀI augmentation lúc huấn luyện của detector" + POOL @ dev30 (user, quy tắc khóa TRƯỚC khi chạy)

**Giả thuyết:** 3 phép vô ích (tịnh tiến / crop, co giãn, resize) đều nằm trong augmentation lúc train của mọi checkpoint mmdet dùng ở đây (multi-scale, crop, flip); 3 phép có ích (xoay, phối cảnh, trượt) KHÔNG có. **Dự đoán kiểm được:** lật ngang (có trong augmentation train) vô ích như tịnh tiến.

**Khung (mọi ô):** loss feature OSFD k = 3, MI μ = 1, ε = 5, α = 1, B = 50, 2 view/step, 1 backward, nhiễu σ = 6 cộng vào adv, feature sạch co-transform; seed SEED + chỉ số dev300; RNG biến đổi = `random` của Python.
- **NOISE:** 2 view chỉ có nhiễu σ (không hình học) — đối chứng.
- **FLIP:** mỗi view lật ngang với xác suất 0.5 (torch.flip, chính xác), + nhiễu σ.
- **POOL:** mỗi view rút một trong hai loại với xác suất 0.5: (a) PIA bỏ tịnh tiến (như ô PIA-noT: rút góc PIA δ_max 0.6, π paper, rồi drop T); (b) phối cảnh thuần căn tâm như CAM-R30 (yaw, pitch ~ U(±30°)). Không chồng hai phép lên một view.
**Đọc (dev30, mô tả, khai báo trước):** quy luật được ỦNG HỘ nếu |FLIP − NOISE| < 2 trên CẢ DINO-Swin-L VÀ CrossAvg₃ (điểm ước lượng) trong khi PIA-noT − NOISE ≥ +5 trên CrossAvg₃ (có biến đổi ngoài augmentation thì có lợi rõ). POOL "triển vọng" nếu ≥ max(PIA, PIA-noT) trên Swin-L VÀ ≥ CAM-R30 − 1 trên CrossAvg₃.
**Unit test trước khi chạy:** FLIP p = 1, σ = 0 → view == torch.flip(ảnh); NOISE σ = 0 → view == ảnh; POOL σ = 0, adv = sạch → view == companion; nhánh (a) của POOL trùng lưới PIA-noT khi cùng tham số; ô cũ trùng HEAD; chạy được, L∞ đúng.

**Code + unit test (TRƯỚC khi chạy):** `attack/methods/geopool.py` (`flip_views`, `pool_grid`, `pool_views`), `geo_osfd_attack(geo="noise" | "flip" | "pool")`, ô NOISE / FLIP / POOL. `scripts/geopool_unit_test.py` ĐẠT: FLIP p = 1, σ = 0 → view == lật ngang (companion cùng phép); NOISE σ = 0 → view == ảnh; POOL σ = 0, adv = sạch → view == companion; nhánh (a) POOL == lưới PIA-noT cùng tham số; PIA / PIA-noT / CAM-R30 trùng tuyệt đối HEAD; 3 ô chạy được, L∞ đúng.

**Kết quả @ dev30 (mô tả) — `results/ablation/pool_dev30/screen.json`, `metrics_pool_dev30.json`, `generalization_pool_dev30.json`, log `results/geopool/{run,eval2}.log`.** (Pipeline lỗi ở dòng IN cuối của `eval_generalization.py` khi run không có OSFD — json đã lưu; thêm điều kiện bỏ qua phần in, chạy lại eval + screen, số không đổi.)

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- NOISE: 98.7 / 89.8 / 67.2 / 69.0 / 57.3 / 15.0 | 64.5
- FLIP: 98.9 / 92.1 / 76.7 / 73.6 / 71.5 / 18.1 | 73.9
- PIA: 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / 28.8 | 79.5
- PIA-noT: 97.0 / 91.1 / 82.0 / 81.8 / 77.7 / 29.8 | 80.5
- CAM-R30: 96.8 / 92.3 / 84.0 / 82.7 / 81.5 / 21.3 | 82.7
- POOL: 96.5 / 91.5 / 80.9 / 82.4 / 81.9 / 27.4 | 81.7

- **Quy luật "ngoài augmentation lúc train" KHÔNG được ủng hộ (theo tiêu chí khai báo):** FLIP − NOISE: CrossAvg₃ **+9.4 [+7.5, +12.8]**, Swin-L +3.1 [−0.5, +6.1] (≥ 2). Lật ngang — có trong augmentation train — vẫn có ích rõ. Nhưng yếu hơn nhiều so với biến đổi ngoài augmentation: PIA-noT − NOISE CrossAvg₃ +16.0, Swin-L **+14.8 [+9.0, +18.9]** → dạng yếu hơn ("trong augmentation thì ít lợi hơn") còn đứng, dạng mạnh ("vô ích") sai.
- Mốc NOISE xác nhận lại: không có hình học thì Swin-L chỉ 15.0, CrossAvg₃ 64.5 — hình học kiểu PIA-noT cộng +15 / +16.
- **POOL không "triển vọng":** Swin-L 27.4 < max(PIA 28.8, PIA-noT 29.8); CrossAvg₃ 81.7 = CAM-R30 − 1.0 (vừa chạm ngưỡng). POOL ≈ trung bình hai thành phần: kéo CrossAvg₃ lên so PIA (+2.2 [+0.2, +5.1], YOLOX +4.2 so PIA-noT) nhưng mất một phần Swin-L của PIA-noT (−2.4). Trộn theo view không lấy được "cả hai".

---

## 2026-10-05 — Phép biến đổi cơ bản: quét liều từng phép đứng riêng @ dev30 (bước 1 của chuỗi "công thức hình học tối giản hơn RRB") — quy tắc khóa TRƯỚC khi chạy

**Mục tiêu (user):** xác định phép biến đổi hình học CƠ BẢN nào, ở liều nào, cho transfer tốt nhất; ghép vài phép đơn giản thành công thức cuối vượt RRB (OSFD gốc). Lỗ hổng dữ liệu: xoay là nguyên liệu của RRB nhưng chưa từng tăng liều (RRB ±7°, PIA trung vị |θ| 3.1°); trượt và lệch tỉ lệ chưa từng thử riêng (chỉ lẫn trong PIA); chỉ phối cảnh đã có đường liều (CAM-R15 / R30).

**Máy:** RTX 3090, venv có sẵn; tải lại 10 checkpoint (`--gen`) + COCO val2017. Không có `artifacts/` (PNG dev300 chưa từng lên HF) → **sinh lại các ô mốc trên dev30** cùng seed: OSFD, NOISE, FLIP, PIA, PIA-noT, CAM-R15, CAM-R30. Sanity: số sinh lại phải ≈ số đã commit (`results/ablation/{pool,cam,piafac}_dev30/screen.json`); lệch > 3 điểm CrossAvg₃ ở ô tất định → dừng, điều tra trước khi đọc ô mới.

**Khung (mọi ô mới):** = ô `PIA` — loss feature OSFD k = 3, MI μ = 1, ε = 5, α = 1, B = 50, 2 view/step, 1 backward, nhiễu σ = 6 cộng vào adv, feature sạch co-transform cùng lưới, KHÔNG cls; seed SEED + chỉ số dev300; ngoài ảnh nguồn lấp 0. Phép biến đổi (`attack/methods/elem.py`): tuyến tính A quanh TÂM ảnh, không tịnh tiến, không phối cảnh; tham số rút đều mỗi view (`random` của Python):

| Ô | Phép | Liều |
|---|---|---|
| ROT7 / ROT15 / ROT30 | xoay A = R(θ) | θ ~ U(±7°), U(±15°), U(±30°) |
| SHR15 / SHR30 | trượt [[1, k], [0, 1]] hoặc [[1, 0], [k, 1]] (xác suất 0.5) | k ~ U(±0.15), U(±0.30) |
| ANI10 / ANI25 | lệch tỉ lệ diag(k, 1/k) | ln k ~ U(±0.10), U(±0.25) |
| SCL25 | co giãn đều s·I (đối chứng — dự đoán vô ích) | ln s ~ U(±0.25) |

Mốc so sánh: **OSFD (= RRB gốc, KHÔNG co-transform — mốc "hơn RRB")**, NOISE (không hình học), FLIP, phối cảnh CAM-R15 / R30, PIA, PIA-noT. Không so với GEO-C0 (yếu hơn OSFD @ dev100: 72.2 vs 74.3).

**Tiêu chí xếp hạng (user chọn phương án a):** CrossAvg₃ là chính; ràng buộc DINO-Swin-L không thấp hơn OSFD. Ở dev30 (CI hiệu ±5) ràng buộc nới thành ≥ OSFD − 2 (điểm ước lượng); xác nhận dev100 dùng ràng buộc chặt.

**Quy tắc chọn họ phép sang bước 2 (dev30, điểm ước lượng, khai báo trước):** họ = {xoay, trượt, lệch tỉ lệ, co giãn đều, phối cảnh (CAM-R15/R30), lật (FLIP)}. Liều tốt nhất của họ = liều có CrossAvg₃ cao nhất trong các liều thỏa Swin-L ≥ OSFD − 2. Họ ĐƯỢC CHỌN nếu ở liều tốt nhất: CrossAvg₃ ≥ NOISE + 5 VÀ CrossAvg₃ ≥ OSFD − 2. Tối đa 3 họ (CrossAvg₃ cao nhất). Không họ nào được chọn → dừng chuỗi, ghi log.
**Bước 2 (khóa entry riêng sau bước 1):** ghép các họ được chọn TRONG CÙNG MỘT VIEW (mỗi phép rút độc lập mỗi view, liều tốt nhất), so với từng phép đứng riêng; POOL đã cho thấy trộn GIỮA view chỉ ra trung bình. **Bước 3:** công thức tốt nhất xác nhận @ dev100 so OSFD (+ PIA), tiêu chí khóa trước.

**Báo cáo kèm:** |u| trung bình mỗi view (tách hiệu ứng loại phép khỏi biên độ), 6 model, R101, η, runtime. Paired bootstrap 1000 (BOOT_SEED 2026) chỉ để thấy độ rộng CI.

**Unit test trước khi chạy (`scripts/elem_unit_test.py`):** liều 0 → lưới đồng nhất; tâm ảnh cố định, xoay đúng góc / chiều; det A = 1 (xoay, trượt, lệch tỉ lệ); |u| mỗi ô; [GPU] σ = 0, adv = sạch → view == companion; 8 ô chạy 3 step, L∞ ≤ 5, hữu hạn, forward no_grad = 2 × step. Code ô cũ không đổi (chỉ thêm nhánh `geo="elem"`); ô mốc sinh lại khớp số cũ là kiểm tra hồi quy.

**Tạm dừng (user, 2026-10-05 — tối chạy tiếp). CHƯA có kết quả nào của bước 1; chưa nhìn số nào.**
- Pipeline `results/elem/run.sh` (tmux `elem`, log `results/elem/run.log`) đang chạy lúc ghi: xong attack OSFD, NOISE, đang FLIP (3/15 ô). Ước tính đủ pipeline ~1 h: attack ~3.5–5 s/ảnh/ô × 15 ô × 30 ảnh ≈ 40 phút + eval 16 điều kiện × 6 model ≈ 15 phút.
- **Nối tiếp (máy mới hoặc máy cũ):** `bash scripts/bootstrap.sh` rồi `tmux new -s elem "bash results/elem/run.sh 2>&1 | tee -a results/elem/run.log"` — resume theo (ô, ảnh): ô / ảnh đã có PNG + dòng stats thì bỏ qua. Máy mới mất `artifacts/` → chạy lại từ đầu (~1 h). `attack_stats.jsonl` có thể có dòng của lần chạy dở (máy cũ) — không ảnh hưởng (resume cần CẢ PNG lẫn dòng stats; dòng trùng lấy bản cuối).
- Khi xong: đối chiếu sanity ô mốc với số đã commit (lệch > 3 CrossAvg₃ ở ô tất định → dừng), rồi áp quy tắc chọn họ ở entry trên (`results/ablation/elem_dev30/screen.json`).
- **Hạ tầng (commit cùng entry này):** `scripts/bootstrap.sh` chạy 1 lần là đủ — bỏ qua `setup_env.sh` khi `.venv` đã dùng được (setup_env luôn xóa venv); tải checkpoint `--gen` (Controlled + Generalization + chẩn đoán); `download_checkpoints.sh --gen` thêm Mask R-CNN Swin-S (trước đây thiếu, `eval_generalization` mrcnn_swin_s cần); nếu `.env` có `HF_TOKEN` thì tự kéo `n300_B50_eps5` từ HF. Đã chạy thử trên máy này: đạt. Máy mới chưa có git identity → `git config user.name/user.email` theo tác giả commit cũ.
- Đề xuất để lần sau nhanh hơn: upload PNG dev30 của các ô mốc lên HF sau khi chạy xong (cần token) → không phải sinh lại.

---

## 2026-10-06 — Vì sao đa dạng hình học hiệu quả (và vì sao trên DINO-Swin-L)? 3 thí nghiệm rẻ E1–E3 @ dev30 / n300 (user, quy tắc đọc khóa TRƯỚC khi chạy)

**Bối cảnh (user):** có / không có đa dạng hình học chênh rất lớn (bỏ RRB −32 CrossAvg₃ @ dev100; NOISE 64.5 vs hình học ~80 @ dev30; Swin-L 15 → ~30), còn LOẠI phép chỉ chênh bậc hai (RRB / PIA / SMOOTH / MORGD 79.0–80.5; cả dải FLIP 73.9 … CAM-R30 82.7) nhưng phụ thuộc target (CAM-R30 Swin-L 21.3 vs PIA-noT 29.8). Mục tiêu: hiểu VÌ SAO đa dạng hình học giúp transfer, đặc biệt trên DINO-Swin-L (gấp đôi nhưng vẫn ~30 so với ~80), rồi dùng hiểu biết đó tối ưu attack. **Cách làm (user):** dev30 xem xu hướng, ưu tiên thí nghiệm rẻ (forward / gradient, PNG có sẵn); thí nghiệm tốn kém phải phân tích vì sao đáng chạy, user confirm rồi mới chạy. Số dev30 chỉ MÔ TẢ; claim đưa vào paper phải xác nhận ≥ dev100.

**Dữ liệu HF (đã tải 2026-10-06):** `runs/n300_B50_eps5.tar` = PNG MI-FGSM / M-DI²-FGSM / OSFD × 300 ảnh n300 + `dets.json` Controlled Panel (sha256 9eadf774…f8d590, khớp). Không có gen_dets (DINO-Swin-L phải predict lại). `runs/a3_stage_n100_B50_eps5.tar` (OSFD từng stage, chưa dùng). PNG dev300/dev30 của các ô PIA/MORGD/… chưa từng upload → dev30 dùng PNG của pipeline `results/elem/run.sh` (đang chạy lại trên máy này).

### E1 — Gradient KỲ VỌNG qua đa dạng có tiến gần gradient target không? (`scripts/e1_expected_grad.py`, chỉ gradient, dev30, δ = 0)
- **Surrogate:** mỗi phép T (sampler y hệt ô attack: `elem_views` / `pia_views` / `cam_views` / `flip_views` / `rrb_views`, kèm nhiễu σ = 6, đích feature co-transform, loss OSFD k = 3), gradient theo ẢNH GỐC (qua resize khả vi) của từng view; ḡ_T^K = trung bình K view đầu, K ∈ {1, 4, 16}. Mốc NONE = không biến đổi, không nhiễu.
- **Target** (chỉ phân tích, không sinh attack): g_t = ∇ₓ loss GT của chính target theo ảnh gốc (qua resize khả vi về khung target). t ∈ {R101, ConvNeXt-T, Swin-T, YOLOX-S, DINO-Swin-L}. **Loss chính = phân loại** (Mask R-CNN: loss_rpn_cls + loss_cls; YOLOX: loss_cls + loss_obj; DINO: mọi số hạng `*loss_cls` không-dn, gồm enc), phụ = đủ task loss (không-dn). DINO tính loss ở train mode với MỌI Dropout / DropPath đặt eval (dn_mask chặn query khớp nhìn dn query → loss khớp không phụ thuộc nhiễu dn).
- **Chỉ số:** A(T, K, t) = cos(ḡ_T^K, g_t) trên toàn ảnh gốc, kèm tỉ lệ trùng dấu.
- **Câu hỏi + quy tắc đọc (dev30, mô tả):**
  - Q1: đa dạng có làm gradient kỳ vọng gần target hơn không — A(T, 16, t) − A(NOISE, 16, t) theo cặp ảnh (bootstrap 1000, BOOT_SEED 2026).
  - Q2 (chính): giữa các ô (mọi ô có drop dev30 đã đo: OSFD-RRB, GEO-C0, NOISE, FLIP, PIA, PIA-noT, PIA-aff, PIA-d0.3, CAM-R15, CAM-R30, LENS + 8 ô elem khi có số), Spearman(A(T, 16, t), drop dev30 của ô trên t), từng t. **"Alignment giải thích transfer" được ỦNG HỘ** nếu ρ ≥ 0.5 với cận dưới CI (bootstrap ảnh trên A, drop cố định) > 0 ở ≥ 3/4 target khác họ, TRONG ĐÓ có DINO-Swin-L. Kèm: ρ theo K (nếu ρ tăng theo K → lợi nằm ở trung bình, không ở từng view).
  - Q3 (Swin-L): CAM-R30 có A cao với target CNN/YOLOX nhưng thấp với Swin-L không (giải thích trade-off).
- Chi phí ước tính: ~20 min GPU.

### E2 — Hiệu ứng của δ có "sống sót" qua biến đổi không, và có dự đoán Swin-L không? (`scripts/e2_survival.py`, chỉ forward)
- T' cố định (không ngẫu nhiên), áp CÙNG LÚC lên ảnh sạch và adv ở khung R50: xoay +10°, trượt ngang 0.15, phối cảnh yaw 15° (căn tâm), tịnh tiến (+7, +5) px; kèm đồng nhất.
- R50 predict trên T'(sạch) → detection tham chiếu (score ≥ 0.3); trên T'(adv) → score khớp (cùng lớp, IoU ≥ 0.5) → retained_T'. Không cần GT warp. **E_id = 1 − retained_id; E_surv = 1 − mean_T' retained_T'.**
- **Đối chứng KHÔNG hình học (thêm trước khi chạy):** E_noise8 = 1 − retained khi cộng nhiễu Gaussian σ = 8 cố định (cùng nhiễu cho sạch / adv). Lý do: với OSFD, E_id trên R50 gần bão hòa (≈ 1) → E_surv có thể "dự đoán tốt hơn" chỉ vì có phương sai; so với E_noise8 (cũng làm yếu attack nhưng không hình học) mới tách được "độ bền HÌNH HỌC".
- (a) **n300, trong từng method (MI, M-DI², OSFD):** Spearman theo ảnh giữa E_surv và retained DINO-Swin-L (`eval_generalization --models dino_swin_l --run n300_B50_eps5 --out generalization_e2_swinl.json`), so với Spearman(E_id, …) và Spearman(E_noise8, …). **ỦNG HỘ "chỉ phần δ bền với biến đổi HÌNH HỌC chuyển được sang Swin-L"** nếu ρ(E_surv) âm hơn CẢ ρ(E_id) VÀ ρ(E_noise8) với CI hiệu (paired bootstrap ảnh) loại 0 ở CẢ OSFD và M-DI², VÀ hiệu so E_noise8 ở Swin-L lớn hơn ở ConvNeXt-T / Swin-T (dets.json). Mô tả: mean E_surv theo method (MI < M-DI² < OSFD dự đoán).
- (b) **dev30, giữa các ô** (PNG pipeline elem): Spearman(mean E_surv của ô, drop Swin-L), mô tả.
- Chi phí: ~15 min (gồm predict Swin-L n300).

### E3 — Target nào "cứng" (đồng biến) với biến đổi hình học? (`scripts/e3_equivariance.py`, chỉ forward, dev30 ảnh sạch, mô tả)
- Mỗi backbone (R50, R101, ConvNeXt-T, Swin-T, CSPDarknet YOLOX-S, Swin-L DINO), cùng T' của E2: cosd(F_s(T'x), T'(F_s(x))) theo stage s (T' áp lên feature bằng cùng lưới, chỉ vùng hợp lệ). Đo độ đồng biến (sai lệch nhỏ = đồng biến). Câu hỏi: Swin-L có khác hẳn các backbone khác không, và sắp thứ tự có khớp thứ tự "khó transfer" không. Chỉ mô tả, không quyết định.

**Không tune attack nào theo kết quả E1–E3.** Hướng method (nếu có) sẽ khóa entry riêng, user confirm.

**Sự cố vận hành (không ảnh hưởng số):** (1) E1 OOM lần đầu ở DINO-Swin-L (tắt gradient checkpointing → > 21 GB) → giữ checkpointing, dùng `.backward()` + đóng băng tham số (peak 9.8 GB; gradient trùng bản cũ, tất định giữa 2 seed: cos 1.0000; YOLOX tất định; Mask R-CNN 0.98 do sampler). (2) Bước phân tích E1 lỗi ghi JSON (numpy bool) → sửa, `--analyze-only`. (3) Nguồn drop dev30 đổi từ `screen.json` (2 định dạng, sót 3 ô) sang `metrics/generalization_*_dev30.json` gốc — số khớp log (vd NOISE 64.5, Swin-L 15.0). Predict DINO-Swin-L n300 tái lập đúng số đã commit (OSFD 24.6, M-DI² 26.1, MI 16.9). Log: `results/e123/run{,2}.log` (`run_oom.log` = lần OOM).

### Kết quả E1 (`results/mechanism/e1_expected_grad.json`, dev30, 20 ô × 16 view; Q2 trên 11 ô có drop) — theo quy tắc khóa: **KHÔNG ỦNG HỘ** (cả cls lẫn task, 0/4 target đạt).
- Gradient kỳ vọng R50 gần như TRỰC GIAO với gradient target ở mọi ô: cos ×100 (cls, K = 16) R101 0.13–0.53, ConvNeXt 0.18–0.48, Swin-T 0.07–0.38, **DINO-Swin-L −0.18 … +0.02**, **YOLOX âm nhất quán −0.98 … −1.84** (dù transfer sang YOLOX 70–80%).
- Q1: đa dạng KHÔNG kéo gradient kỳ vọng gần target hơn NOISE (hiệu cỡ ±0.003, phần lớn CI chứa 0; YOLOX đa số âm có ý nghĩa; Swin-L chỉ OSFD +0.0019 [+0.0005, +0.0035]).
- Q2 (Spearman giữa 11 ô, cls K = 16): ConvNeXt +0.35 [−0.18, 0.55], Swin-T +0.08, YOLOX **−0.41 [−0.65, −0.03]**, DINO-Swin-L +0.67 [−0.28, 0.85]. Swin-L dương nhất quán qua K và 2 objective (+0.61 … +0.73) nhưng CI chạm 0 — chỉ mô tả.
- Ước lượng kỳ vọng nhiễu: cos giữa 2 nửa 8 view chỉ 0.35–0.65.
- Đọc: mở rộng A1 — kể cả trung bình qua đa dạng, HƯỚNG gradient pixel tại ảnh sạch không phải kênh giải thích transfer. Giới hạn: chỉ δ = 0, n = 30, K ≤ 16.

### Kết quả E2 (`results/mechanism/e2_survival_n300.json`, n300) — theo quy tắc khóa: **KHÔNG ỦNG HỘ** (ở mọi method).
- **Mô tả — đa dạng làm δ BỀN hơn:** E (1 − retained R50) ID → trung bình 4 phép hình học / nhiễu σ8: MI 0.694 → **0.519** / 0.346; M-DI² 0.657 → 0.634 / 0.469; OSFD 0.965 → 0.959 / 0.948. δ của MI mất ~¼ hiệu ứng qua biến đổi hình học nhỏ, M-DI² và OSFD gần như không mất. Thứ tự giữa method (MI < M-DI² ≈ OSFD) khớp thứ tự drop Swin-L (16.9 < 26.1 ≈ 24.6) — chỉ 3 điểm.
- **Theo ảnh, độ bền KHÔNG dự đoán Swin-L:** ρ(E_surv, retained Swin-L) = −0.02 / −0.08 / −0.05 (MI / M-DI² / OSFD); E_id, E_noise8 cũng ≈ 0 (−0.11 … +0.09). surv − noise8 ≈ +0.02, CI chứa 0 ở cả 3 method.
- Target khác: ρ(E_surv) có ý nghĩa (OSFD: R101 −0.59, ConvNeXt −0.42, Swin-T −0.41) nhưng NGANG ρ(E_noise8) (surv − noise ≈ 0) → phần dự đoán được là "độ mạnh chung của δ trên R50", không riêng hình học. Với MI, surv − id âm có ý nghĩa ở R101/ConvNeXt/Swin-T (−0.20 … −0.25) nhưng cũng không hơn noise8.
- Đọc: mức tổn thương theo từng ảnh của DINO-Swin-L KHÔNG đo được bằng bất kỳ đại lượng nào phía R50 đã thử (nhất quán với cổng Repair-Robust: Q_plain ↔ Swin-L +0.08 … +0.15).

### Kết quả E3 (`results/mechanism/e3_equivariance.json`, dev30, mô tả)
cosd(F(T'x), T'F(x)) (thấp = đồng biến), stage nông → sâu, xoay 10° | phối cảnh 15°:
- R50 0.04 / 0.16 / 0.21 / 0.19 | 0.03 / 0.12 / 0.14 / 0.11; R101 tương tự.
- **ConvNeXt-T 0.09 / 0.04 / 0.05 / 0.05 | 0.06 / 0.03 / 0.03 / 0.03** — đồng biến nhất ở stage sâu.
- Swin-T 0.08 / 0.16 / 0.23 / **0.08** | 0.07 / 0.12 / 0.13 / **0.04**.
- **DINO-Swin-L 0.07 / 0.10 / 0.16 / 0.26 | 0.06 / 0.07 / 0.12 / 0.18** — tăng đơn điệu theo độ sâu, stage cuối lệch NHIỀU nhất trong họ Transformer (ngược Swin-T).
- YOLOX-S 0.17 / 0.21 / 0.24 | 0.14 / 0.17 / 0.15; DINO-R50 0.14 / 0.22 / 0.19 | 0.11 / 0.15 / 0.12.
- Caveat: cosd không trừ trung bình kênh → so GIỮA kiến trúc bị nhiễu bởi thống kê feature (vd ConvNeXt có thể có thành phần DC lớn). Không thấy thứ tự khớp độ khó transfer (YOLOX ít đồng biến nhưng transfer tốt). Chỉ ghi nhận: Swin-L là backbone duy nhất có stage cuối nhạy hình học hơn stage giữa.

---

## 2026-10-06 — E4: mức tổn thương của DINO-Swin-L do PHÍA ẢNH / TARGET quyết định? (n300, không GPU, quy tắc khóa TRƯỚC khi chạy)

**Lý do:** E1, E2 (và cổng Repair-Robust trước đó) cho thấy không đại lượng nào phía R50 / δ dự đoán được theo ảnh việc Swin-L bị lừa (ρ ≈ 0). Giả thuyết: tổn thương do thuộc tính của object / độ tự tin của chính target.

**Dữ liệu (có sẵn, không attack mới):** dets n300 của MI / M-DI² / OSFD — Swin-L từ `artifacts/runs/n300_B50_eps5/gen_dets/dino_swin_l.json` (E2 vừa predict lại), R50 / R101 / ConvNeXt-T / Swin-T từ `dets.json`.

**Đơn vị = GT (không crowd) mà target phát hiện ở ảnh sạch** (cùng lớp, IoU ≥ 0.5, score ≥ 0.3, định nghĩa `attack/mechanism.py`). Biến kết quả: retained = s_adv / s_clean (0 nếu mất), suppressed = s_adv < 0.3.
**Thuộc tính (đo trước khi nhìn kết quả):** s_clean của chính target; s_clean của R50 cho cùng GT (phía surrogate, đối chứng); log diện tích GT (ảnh gốc); diện tích tương đối; số GT trong ảnh; độ chồng lấn (IoU lớn nhất với GT khác); lớp COCO (mô tả theo nhóm).
**Thống kê:** Spearman theo GT, CI 95% bootstrap theo ẢNH (cụm, 1000, BOOT_SEED 2026); bảng theo bin (COCO small / medium / large; s_clean tứ phân vị).
**Quy tắc đọc:** "phía target giải thích" được ỦNG HỘ nếu có thuộc tính với |ρ| ≥ 0.3 (CI loại 0) trên DINO-Swin-L ở CẢ OSFD và M-DI², VÀ |ρ| đó lớn hơn rõ |ρ| của s_clean R50 (đối chứng phía surrogate). Mô tả kèm: cùng phân tích trên R101 / ConvNeXt-T / Swin-T để xem Swin-L có khác không.

### Kết quả E4 (`results/mechanism/e4_target_side.json`, n300, ~1540–1690 GT mỗi target) — theo quy tắc khóa: **KHÔNG ỦNG HỘ** (OSFD trên Swin-L: |ρ| < 0.3 ở mọi thuộc tính; M-DI²: s_clean Swin-L +0.36 nhưng KHÔNG hơn s_clean R50 +0.47).
- **Tổn thương Swin-L là sức bền DIỆN RỘNG, không phải "một nhóm object miễn nhiễm":** OSFD retained trung bình 0.73 (Swin-T 0.30, ConvNeXt 0.27, R101 0.11); bin yếu nhất (object nhỏ) vẫn giữ 0.69.
- **"Độ dễ" của object là yếu tố CHUNG, không riêng target:** s_clean của R50 cho cùng GT dự đoán retained trên Swin-L (+0.30 OSFD, +0.47 M-DI²) TỐT HƠN s_clean của chính Swin-L (+0.18 / +0.36) → object dễ (mọi model tự tin) thì sống sót; hiệu ứng mạnh ở attack yếu, yếu dần khi attack mạnh (OSFD ở mọi target chỉ +0.15 … +0.30).
- Bin s_clean của Swin-L: suppression Q1 [0.30, 0.47] 0.58 → Q4 [0.87, 0.98] 0.08 nhưng retained gần phẳng (0.70 / 0.67 / 0.70 / 0.83) → chênh suppression phần lớn là do GẦN NGƯỠNG 0.3 (thang score DINO thấp hơn Mask R-CNN ~1.0), không phải tổn thương khác.
- **Phát hiện riêng Swin-L — kích thước object ĐẢO CHIỀU so với target khác dưới OSFD:** suppression nhỏ / vừa / lớn: Swin-L **0.38 / 0.30 / 0.19**; Swin-T 0.64 / 0.65 / **0.72**; ConvNeXt 0.66 / 0.68 / **0.79**. Khoảng cách Swin-L so Swin-T: nhỏ 0.26, lớn **0.53** → sức bền Swin-L tập trung ở object VỪA / LỚN.
- **Theo thang MI → M-DI² → OSFD:** object nhỏ trên Swin-L bị hạ ngay từ MI / M-DI² (sup 0.29 / 0.43 / 0.38 — OSFD không hơn M-DI²); object lớn chỉ tăng khi tới OSFD (0.04 / 0.08 / 0.19), và trên ConvNeXt / Swin-T bước nhảy lớn nhất cũng ở object lớn (ConvNeXt 0.24 / 0.46 / 0.79). Hai "kênh": object nhỏ (attack nào cũng phá) và object lớn (cần OSFD = RRB + loss feature; chưa tách được phần nào). Swin-L khác target khác chủ yếu ở kênh object lớn.
- Giả thuyết (CHƯA kiểm): Swin-L nhận object lớn bằng hình dạng / ngữ cảnh toàn cục (cửa sổ 12, IN-22k, encoder DINO đa scale), δ chủ yếu phá cấu trúc cục bộ. Kiểm rẻ được khi có PNG dev30 của pipeline elem: đa dạng hình học (NOISE → PIA / CAM / ROT) tăng suppression Swin-L ở object nhỏ hay lớn.

---

## 2026-10-06 — E5: đa dạng hình học giúp Swin-L qua kênh object NHỎ hay LỚN? (dev30, forward-only trên PNG pipeline elem; khóa TRƯỚC khi có PNG)

**User đồng ý chạy pipeline elem** (`results/elem/run.sh`, tmux `elem`, bắt đầu 2026-10-06 ~14:50; quy tắc bước 1 vẫn là entry 2026-10-05).

**Phân tích (sau khi pipeline xong):** `e4_target_side.py --run dev300_B50_eps5 --subset dev30 --dets dets_elem_dev30.json --methods <15 ô> --base NOISE --tag _elem_dev30`. Cùng tập GT (target phát hiện ở ảnh sạch, không phụ thuộc ô), Δ suppression (ô − NOISE) theo kích thước COCO small / medium / large + (large − small), bootstrap cụm theo ảnh (1000, BOOT_SEED 2026). Trên DINO-Swin-L (chính), Swin-T, ConvNeXt-T (đối chiếu).
**Kiểm tra code trên n300 (base = MI, không thuộc E5):** Swin-L — M-DI² − MI: nhỏ +0.14, lớn +0.04, large − small −0.10 [−0.15, −0.05]; OSFD − MI: nhỏ +0.09, lớn +0.15, large − small +0.07 [0.00, 0.13]. ConvNeXt — OSFD − MI large − small +0.33 [0.26, 0.41].
**Đọc (mô tả, n = 30, khai báo trước):** "đa dạng hình học giúp Swin-L qua kênh object LỚN" được ủng hộ nếu, trên Swin-L, (large − small) > 0 (điểm) ở CẢ PIA và PIA-noT (2 ô Swin-L cao nhất đã biết), VÀ (large − small) của CAM-R30 (ô Swin-L thấp dù CrossAvg₃ cao) nhỏ hơn của cả hai. Ngược lại: không ủng hộ. Muốn đưa vào paper phải xác nhận ≥ dev100.

---

## 2026-10-06 — E6 (khóa trước, CHẠY SAU pipeline elem nếu cần): Swin-L nhận object lớn bằng HÌNH DẠNG hay CẤU TRÚC CỤC BỘ? (ảnh sạch n300, chỉ forward)

**Giả thuyết cần kiểm (từ E4):** Swin-L nhận object vừa / lớn chủ yếu bằng hình dạng / bố cục toàn cục, nên δ (phá cấu trúc cục bộ) ít chạm tới; target khác dựa nhiều vào cấu trúc cục bộ hơn.

**Can thiệp trên ảnh SẠCH (ảnh gốc uint8, trong từng GT box không-crowd, box lớn trước → nhỏ sau), mỗi model đọc ảnh qua test pipeline riêng (`inference_detector`):**
- **BLUR (phá cục bộ, giữ hình dạng):** Gaussian blur σ = f·√(w·h) trong box, f ∈ {0.02, 0.05}.
- **SHUF (phá hình dạng, giữ cục bộ):** chia box thành g × g ô, hoán vị ngẫu nhiên các ô (RNG theo ảnh: seed = 2026 + img_id), g ∈ {3, 6}.
- Target: R50, R101, ConvNeXt-T, Swin-T, YOLOX-S, DINO-Swin-L. Đơn vị = GT target phát hiện ở ảnh sạch (score ≥ 0.3); retained = s_corrupt / s_clean (0 nếu mất); tách small / medium / large (COCO).

**Chỉ số tương tác (khử "Swin-L bền với mọi thứ"):** với nhóm kích thước z, R_t(c, z) = mean retained; others = trung bình ConvNeXt-T, Swin-T, YOLOX-S. I(z) = [R_SwinL(BLUR) − R_others(BLUR)] − [R_SwinL(SHUF) − R_others(SHUF)].
**Quy tắc:** giả thuyết "hình dạng" được ỦNG HỘ nếu I(large) > 0 với cận dưới CI 95% (bootstrap cụm theo ảnh, 1000, BOOT_SEED 2026) > 0 ở CẢ 2 cặp liều (f 0.02 ↔ g 3, f 0.05 ↔ g 6), VÀ I(large) > I(small) (điểm) ở cả 2 cặp. Ngược lại: không ủng hộ. Mô tả kèm: R_t(c, z) đầy đủ, R50 / R101.
Caveat đã biết: BLUR và SHUF không cùng "liều" tuyệt đối → chỉ so tương tác (Swin-L so với others), không so BLUR với SHUF trực tiếp; box chồng nhau bị làm hỏng chéo.

### Kết quả bước 1 quét liều phép cơ bản @ dev30 (pipeline elem, 2026-10-06; `metrics/generalization_elem_dev30.json`, `results/ablation/elem_dev30/screen.json`)
**Sanity ô mốc sinh lại (máy mới):** lệch CrossAvg₃ so với số đã commit: OSFD +0.0, NOISE −0.2, FLIP −0.4, PIA +0.1, PIA-noT −0.0, CAM-R15 −0.1, CAM-R30 −0.6 → ĐẠT (ngưỡng dừng 3).

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | |u| px:
- OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9; NOISE … / 14.9 | 64.3; FLIP … / 17.8 | 73.6
- ROT7 97.9 / 91.0 / 79.9 / 80.5 / 75.1 / 23.4 | 78.5 | 23; ROT15 97.0 / 91.2 / 79.6 / 81.7 / 76.9 / 23.7 | 79.4 | 50; ROT30 97.1 / 90.5 / 78.7 / 77.8 / 70.7 / 24.9 | 75.7 | 100
- SHR15 98.7 / 92.1 / 81.2 / 78.9 / 72.0 / 17.2 | 77.4 | 19; SHR30 98.6 / 94.0 / 80.6 / 78.0 / 76.5 / 25.2 | 78.3 | 37
- ANI10 98.6 / 91.5 / 81.4 / 81.0 / 78.6 / 23.3 | 80.3 | 19; ANI25 97.0 / 91.2 / 78.9 / 82.6 / 77.5 / 25.1 | 79.7 | 48
- SCL25 96.3 / 91.0 / 82.1 / 82.6 / 78.2 / 27.4 | 81.0 | 48
- (PIA 28.6 | 79.6; PIA-noT 29.8 | 80.4; CAM-R15 25.6 | 79.1; CAM-R30 20.9 | 82.1)

**Quy tắc chọn họ (khóa 2026-10-05):** ngưỡng Swin-L ≥ OSFD − 2 = 25.22; CrossAvg₃ ≥ max(NOISE + 5, OSFD − 2) = 76.87. Xoay: không liều nào đạt Swin-L (23.4 / 23.7 / 24.9) → KHÔNG chọn; lệch tỉ lệ: ANI25 25.107 < 25.22 → KHÔNG; lật: Swin-L 17.8 → KHÔNG. **ĐƯỢC CHỌN (3): co giãn đều SCL25 (81.0), phối cảnh CAM-R15 (79.1), trượt SHR30 (78.3, Swin-L 25.248 — vừa sát ngưỡng).**
Đọc (mô tả): (1) **co giãn đều — đối chứng được dự đoán "vô ích" — lại có CrossAvg₃ cao nhất và Swin-L cao nhất trong các phép cơ bản** ở liều ln s ±0.25 (|u| 48 px); kết luận "co giãn ≈ 0" trước đây (PIA-noS) là ở liều PIA ~±10% → giả thuyết "phép mà surrogate đồng biến thì vô ích" sai thêm một lần. (2) Mọi phép cơ bản cho CrossAvg₃ 75.7–81.0 (trừ FLIP), chênh nhau ≤ 5 — đúng bức tranh "có đa dạng >> loại". (3) Không phép cơ bản đứng riêng nào vượt PIA-noT trên Swin-L (29.8); tăng liều xoay KHÔNG giúp CrossAvg₃ (ROT30 75.7 < ROT15). n = 30 — mọi chênh ≤ 3 là trong nhiễu.

### Kết quả E5 (`results/mechanism/e4_target_side_elem_dev30.json`) — theo quy tắc khóa: ỦNG HỘ (yếu, mô tả)
- Swin-L, Δ suppression so NOISE, (large − small): PIA +0.01 [−0.10, 0.14], PIA-noT +0.07 [−0.05, 0.19] (> 0 cả hai), CAM-R30 −0.02 (< cả hai) → đạt tiêu chí khai báo; mọi CI chứa 0, PIA chỉ +0.01.
- **Mô tả mạnh hơn — giữa 14 ô, drop Swin-L đi theo phần tăng ở object LỚN:** Spearman(Δsup, drop Swin-L) theo kích thước: nhỏ +0.22, vừa +0.49, **lớn +0.64**. Ngược lại Swin-T: nhỏ +0.42, vừa +0.82, lớn +0.24; ConvNeXt: +0.47 / +0.32 / +0.25. → Trên Swin-L, mọi ô hạ object nhỏ gần như ngang nhau (+0.06 … +0.10 so NOISE); khác biệt giữa các ô nằm ở object lớn: OSFD +0.18, PIA-noT +0.16, PIA +0.11 so với phép cơ bản +0.02 … +0.09 (SCL25 chỉ +0.02 dù Swin-L 27.4 — lợi của SCL25 nằm ở object vừa).
- Khớp E4 (n300): sức bền Swin-L tập trung ở object lớn, và đó cũng là chỗ các ô khác nhau. Đáng chú ý: OSFD gốc (RRB, feature sạch KHÔNG co-transform) có Δ object lớn trên Swin-L cao nhất.

### Kết quả E6 (`results/mechanism/e6_shape_texture.json`, n300 ảnh sạch, log `results/e123/e6.log`) — theo quy tắc khóa: **ỦNG HỘ giả thuyết "hình dạng"** (cả 2 cặp liều, CI loại 0, I(large) ≫ I(small)).
Mean retained nhỏ / vừa / lớn:
- BLUR05 (phá cục bộ, giữ hình dạng): R50 0.37 / 0.25 / 0.21; ConvNeXt 0.44 / 0.31 / 0.24; Swin-T 0.40 / 0.27 / 0.19; YOLOX 0.36 / 0.20 / 0.12; **DINO-Swin-L 0.45 / 0.34 / 0.37**.
- SHUF3 (phá hình dạng, giữ cục bộ): R50 0.15 / 0.13 / 0.27; ConvNeXt 0.15 / 0.15 / 0.23; Swin-T 0.16 / 0.16 / 0.30; YOLOX 0.18 / 0.12 / 0.17; **DINO-Swin-L 0.21 / 0.17 / 0.16**.
- Tương tác I = [Swin-L − others](BLUR) − [Swin-L − others](SHUF): **f 0.02 | g 3: lớn +0.195 [0.159, 0.233]**, vừa +0.018, nhỏ −0.017; **f 0.05 | g 6: lớn +0.185 [0.150, 0.222]**, vừa +0.023, nhỏ −0.005.
- Đọc: với object LỚN, Swin-L giữ được khi mất chi tiết cục bộ (BLUR05: 0.37 vs others 0.18) nhưng sụp như / hơn others khi mất hình dạng (SHUF3: 0.16 vs 0.23) — target khác (và chính R50) đi ngược: mất hình dạng còn giữ được nhiều hơn mất chi tiết ở object lớn. Với object nhỏ không khác (I ≈ 0). → **DINO-Swin-L nhận object lớn chủ yếu bằng HÌNH DẠNG / bố cục; R50 (surrogate) và các target khác dựa nhiều vào cấu trúc cục bộ.**
- **Chuỗi giải thích (E4 + E5 + E6):** surrogate R50 thiên cấu trúc cục bộ → δ phá cấu trúc cục bộ → chuyển tốt sang target cũng thiên cục bộ, và sang mọi target ở object nhỏ (hình dạng ít thông tin); với object lớn trên Swin-L (dựa vào hình dạng) δ ít chạm tới → sức bền tập trung ở object lớn (E4); các method khác nhau trên Swin-L chủ yếu ở object lớn (E5). Khớp literature shape/texture bias (Geirhos et al. ICLR'19: CNN ImageNet thiên texture; mô hình lớn / pretrain lớn thiên hình dạng hơn).
- Caveat: Swin-L lẫn kiến trúc + scale + IN-22k + encoder DINO — không tách được nguồn của shape bias (như caveat cũ); BLUR / SHUF không cùng liều tuyệt đối (chỉ so tương tác); box chồng nhau bị hỏng chéo; E6 là quan sát trên ảnh sạch — quan hệ nhân quả với attack cần thí nghiệm can thiệp (method nhắm hình dạng object lớn → Swin-L tăng, đặc biệt ở object lớn).

---

## 2026-10-06 — E7: phần δ có tác dụng lên object LỚN của Swin-L là cấu trúc QUY MÔ LỚN hay CỤC BỘ? (n300 PNG có sẵn, chỉ forward, khóa TRƯỚC khi chạy)

**Lý do:** E6 cho thấy Swin-L nhận object lớn bằng hình dạng (trên ảnh sạch). E7 hỏi phía attack: trong δ hiện có, phần nào gây tác dụng lên từng target. Nếu phần chạm được Swin-L (object lớn) là phần cấu trúc quy mô lớn của δ, còn target khác ăn phần cục bộ → cơ sở cho method "đẩy δ lên cấu trúc quy mô object".

**Can thiệp lên δ (không lên ảnh):** δ = PNG adv − ảnh gốc (n300, OSFD và M-DI²). Trong từng GT box (lớn trước → nhỏ sau), biến đổi δ bằng ĐÚNG phép của E6: BLUR_δ (σ = f·√(wh), f ∈ {0.02, 0.05}; giữ cấu trúc lớn của δ, bỏ cục bộ) | SHUF_δ (hoán vị g × g ô, g ∈ {3, 6}; giữ cục bộ, bỏ bố cục); ngoài box giữ δ nguyên. Ảnh = round(clip(x + T(δ))) uint8, mỗi model đọc qua test pipeline riêng.
**Chỉ số:** đơn vị GT target phát hiện ở ảnh sạch (score ≥ 0.3); sức attack A(c, z) = 1 − mean retained; F(c, z) = A(c, z) / A(δ nguyên, z) (phần tác dụng còn lại). I_δ(z) = [F_SwinL − F_others](BLUR) − [F_SwinL − F_others](SHUF), others = ConvNeXt-T, Swin-T, YOLOX-S; bootstrap cụm theo ảnh (1000, BOOT_SEED 2026; tỉ số tính trên từng mẫu bootstrap).
**Quy tắc:** "phần δ chạm Swin-L ở object lớn là cấu trúc quy mô lớn" được ỦNG HỘ nếu với OSFD: I_δ(large) > 0, cận dưới CI > 0, ở CẢ 2 cặp liều (f 0.02 ↔ g 3, f 0.05 ↔ g 6), VÀ I_δ(large) > I_δ(small) (điểm). M-DI²: mô tả (cùng chiều = củng cố). Ngược lại: không ủng hộ.
Chi phí ước tính ~15 min GPU (6 model × 300 ảnh × 8 điều kiện; ảnh sạch / δ nguyên dùng dets có sẵn của E6 và run n300).

### E7 — DỪNG ở bản thử, KHÔNG chạy đủ n300 (lỗi thiết kế: hiệu ứng sàn) — quy tắc khóa KHÔNG áp dụng được
Thử 3 ảnh rồi 30 ảnh đầu n300 (OSFD; `results/mechanism/e7_delta_scale_limit30_pilot.json`, dets `artifacts/e7/dets_limit30.json`): CẢ BLUR_δ LẪN SHUF_δ đều xóa gần hết tác dụng của δ ở mọi target → F chạm sàn, tương tác vô nghĩa (CI rộng ±0.3; tỉ số nổ khi A(δ nguyên) nhỏ, vd M-DI² Swin-L). Không chạy đủ n300.
Phần còn lại F (object lớn, n = 30), BLUR02 / BLUR05 / SHUF3 / SHUF6: R50 0.12 / 0.10 / 0.22 / 0.18; R101 0.07 / 0.05 / 0.16 / 0.14; ConvNeXt 0.07 / 0.04 / 0.16 / 0.13; Swin-T 0.10 / 0.08 / 0.22 / 0.16; YOLOX 0.14 / 0.13 / 0.30 / 0.22; **DINO-Swin-L −0.03 / −0.06 / 0.03 / 0.09**.
Đọc (mô tả, n = 30): **δ của OSFD là mẫu tần cao KHỚP CHÍNH XÁC với nội dung ảnh** — làm mờ δ trong box (giữ cấu trúc quy mô lớn của δ) còn ≤ 14% tác dụng; xáo ô δ (giữ mẫu cục bộ, lệch khỏi nội dung) còn 13–30% ở target thiên cục bộ; trên Swin-L cả hai ≈ 0. → δ gần như KHÔNG có thành phần quy mô lớn / hình dạng; phần nhỏ "texture lệch vị trí" còn tác dụng ở target CNN-like nhưng không ở Swin-L. Khớp chuỗi E4–E6. Hệ quả cho method: thành phần hình dạng phải được TẠO RA có chủ đích; câu hỏi mở là trong ngân sách L∞ 5/255 có làm được không.
**Bước 2 chuỗi elem (ghép SCL25 + CAM-R15 + SHR30): đề xuất tạm gác** (chờ user) — các phép cơ bản chỉ tăng phá object lớn trên Swin-L +0.02 … +0.09 (OSFD / PIA-noT +0.16 … +0.18), không nhắm nút thắt; CrossAvg₃ hơn OSFD ~2 @ dev30 (trong nhiễu).

---

## 2026-10-06 — E8 (user chọn CHỈ ô WB-OSFD): OSFD white-box trên CHÍNH DINO-Swin-L @ dev30 (đo trần, không phải method) — khóa TRƯỚC khi chạy

**Câu hỏi:** phá feature backbone của chính Swin-L (white-box) có hạ được object lớn không? Tách "δ từ R50 không chuyển được" với "DINO-Swin-L sửa được distortion backbone kể cả white-box".
**Cấu hình:** `osfd_attack` đúng công thức OSFD (loss MSE(k·F_sạch, F_adv) trên 4 stage backbone, k = 3, MI μ = 1, RRB θ 7 / l_s 10 / ρ 0.8 / s_max 1.1 / σ 6, ε = 5, α = 1, B = 50), surrogate = DINO-Swin-L (khung + GT theo test pipeline của Swin-L), gradient checkpointing của Swin tắt (chỉ backbone). Seed SEED + chỉ số dev300. `scripts/e8_whitebox_lowfreq.py --osfd`; PNG `artifacts/e8/WB-OSFD/`. Các ô tần thấp WB-s1/s16/s32 KHÔNG chạy (user).
**Đo:** Swin-L (`inference_detector`) suppression theo kích thước + AP drop; mốc transfer OSFD / PIA-noT / NOISE (PNG dev30).
**Đọc (khai báo trước, mô tả n = 30):** (a) WB-OSFD suppression object lớn ≥ 0.6 → Swin-L KHÔNG bền cơ bản với phá feature; nút thắt là tính chuyển giao từ R50. (b) ≤ suppression lớn của OSFD transfer + 0.15 → kể cả white-box, phá feature backbone không hạ được object lớn → DINO (encoder / decoder) sửa được distortion backbone; hướng "phá feature / hình dạng từ surrogate" có trần thấp. Ở giữa → xám, thảo luận.

### Kết quả E8 (`results/mechanism/e8_whitebox_lowfreq.json`, dev30, log `results/e123/e8.log`) — theo quy tắc: VÙNG XÁM, nghiêng mạnh về (a)
DINO-Swin-L, AP drop | suppression nhỏ / vừa / lớn [CI]:
- **WB-OSFD (white-box trên Swin-L): 84.6 | 0.71 [0.56, 0.86] / 0.68 [0.55, 0.80] / 0.57 [0.38, 0.75]**
- OSFD transfer (R50): 27.2 | 0.24 / 0.21 / 0.25; PIA-noT: 29.8 | 0.26 / 0.23 / 0.23; NOISE: 14.9 | 0.17 / 0.15 / 0.07
- Object lớn 0.57: dưới ngưỡng (a) 0.6 một chút, trên ngưỡng (b) (0.25 + 0.15 = 0.40) → xám theo quy tắc. Đọc: **Swin-L KHÔNG bền cơ bản ở ε = 5** — phá feature backbone của chính nó (cùng công thức OSFD, B = 50) đưa AP drop 27 → 85; encoder / decoder DINO KHÔNG "sửa" được distortion white-box → giả thuyết cũ "encoder DINO dập distortion" chỉ đúng với distortion VỪA (transfer), không phải giới hạn cơ bản. Object lớn vẫn khó hơn nhỏ (0.57 vs 0.71) — một phần sức bền hình dạng còn lại kể cả white-box. Peak bộ nhớ 19.7 GB.

---

## 2026-10-06 — TỔNG HỢP E1–E8: vì sao đa dạng hình học giúp, vì sao Swin-L khó (mô tả; claim cho paper cần ≥ dev100)

1. **Đa dạng hình học hoạt động qua TÍNH BỀN của nghiệm, không qua hướng gradient từng bước.** E1: gradient kỳ vọng R50 trực giao gradient target ở mọi ô, đa dạng không kéo gần (Q2 không đạt). E2 (n300): δ của MI mất ~¼ tác dụng dưới phép hình học nhỏ, M-DI² / OSFD gần như không mất.
2. **Sức bền DINO-Swin-L tập trung ở object VỪA / LỚN** (E4 n300: suppression OSFD nhỏ / vừa / lớn 0.38 / 0.30 / 0.19, ngược Swin-T / ConvNeXt), và **khác biệt giữa các method trên Swin-L nằm ở object lớn** (E5: ρ(Δsup lớn, drop Swin-L) +0.64 giữa 14 ô; nhỏ +0.22).
3. **Swin-L nhận object lớn bằng HÌNH DẠNG; R50 + target khác dựa vào cấu trúc cục bộ** (E6 n300, ỦNG HỘ, tương tác +0.19 [0.16, 0.23] chỉ ở object lớn).
4. **δ từ R50 là mẫu tần cao khớp chính xác nội dung, gần như không có thành phần quy mô lớn** (E7 bản thử n = 30: làm mờ / xáo δ còn ≤ 14% / 13–30% tác dụng, Swin-L ≈ 0).
5. **Swin-L KHÔNG bền cơ bản** (E8: white-box OSFD 84.6% AP drop so transfer 27.2%) → khoảng trống 27 → 85 là vấn đề CHUYỂN GIAO.
6. Yếu tố chung phía object: object "dễ" (mọi model tự tin) sống sót hơn ở mọi target (E4).
**Câu cơ chế:** surrogate R50 thiên cấu trúc cục bộ → δ là mẫu tần cao khớp nội dung → chuyển được sang target / object mà nhận dạng dựa vào cấu trúc cục bộ (CNN, Swin-T, object nhỏ); DINO-Swin-L nhận object lớn bằng hình dạng nên δ ít chạm tới — dù chính Swin-L không bền khi bị tấn công trực tiếp. Đa dạng hình học làm δ bền hơn và cải thiện phần object lớn một chút (OSFD / PIA-noT +0.16 … +0.18 so NOISE), chưa đủ đóng khoảng trống.
**Hệ quả cho method (chưa khóa):** cần δ tác động lên biểu diễn ở MỨC OBJECT / hình dạng thay vì từng vị trí. Manh mối hỗ trợ: loss phân loại (C2, C2-PIA — loss ở mức RoI, gộp cả object) là thứ duy nhất đã tăng Swin-L rõ (+7 … +13). Kiểm rẻ đề xuất: phân rã theo kích thước C2 / C2-PIA vs OSFD / PIA trên Swin-L (sinh lại dev30) — lợi của loss mức object có tập trung ở object lớn không.

---

## 2026-10-06 — Bước 2 chuỗi elem: TẠM GÁC (user); E9: lợi của loss mức object (cls) trên Swin-L có tập trung ở object lớn? — khóa TRƯỚC khi chạy

**Bước 2 chuỗi elem (ghép SCL25 + CAM-R15 + SHR30) tạm gác (user đồng ý):** phép cơ bản không nhắm nút thắt object lớn của Swin-L (E5); lợi CrossAvg₃ so OSFD ~2 @ dev30 (trong nhiễu). Quay lại nếu cần recipe đơn giản làm baseline phụ.

**E9:** sinh lại C2 (`PROBE-osfd+cls`) và C2-PIA trên dev30 (cùng seed SEED + chỉ số dev300, code không đổi; PNG cũ mất khi trả máy), eval 6 model; `e4_target_side.py --base OSFD` (C2 vs OSFD: cùng RRB, khác loss) và so C2-PIA với PIA (cùng PIA, khác loss) theo kích thước trên DINO-Swin-L.
**Đọc (mô tả, n = 30, khai báo trước):** "lợi của loss phân loại (mức object) trên Swin-L tập trung ở object lớn" được ỦNG HỘ nếu trên Swin-L Δsup(large) − Δsup(small) > 0 (điểm) ở CẢ C2 − OSFD VÀ C2-PIA − PIA. Ngược lại: không ủng hộ. Sanity: drop sinh lại so số đã commit (C2 dev30: Swin-L 26.1, CrossAvg₃ 78.3; C2-PIA 34.7 / 83.6) — lệch > 3 CrossAvg₃ → ghi chú (C2 không tất định đã biết, ~5% pixel).

### Kết quả E9 (`results/mechanism/e4_target_side_e9_vs{OSFD,PIA}.json`; `metrics/generalization_e9_dev30.json`; log `results/e123/e9.log`) — theo quy tắc: **KHÔNG ỦNG HỘ**
**Sanity sinh lại:** C2 Swin-L 25.6 (cũ 26.1), CrossAvg₃ 78.0 (78.3) — khớp. **C2-PIA Swin-L 29.3 (cũ 34.7, −5.4)**, CrossAvg₃ 83.8 (83.6) — CrossAvg₃ khớp, Swin-L lệch 5.4 (C2 không tất định đã biết; CI dev30 ±5). dev100 trước đây đã xác nhận C2-PIA − C2 Swin-L +5.8 [1.7, 9.1].
Relative AP drop % (dev30), R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CrossAvg₃: OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9; PIA 96.8 / 89.6 / 79.5 / 80.5 / 78.7 / 28.6 | 79.6; C2 99.9 / 94.8 / 82.9 / 84.9 / 66.2 / 25.6 | 78.0; C2-PIA 99.4 / 94.8 / 83.1 / 88.0 / 80.4 / 29.3 | 83.8.
DINO-Swin-L, Δ suppression nhỏ / vừa / lớn | (lớn − nhỏ):
- C2 − OSFD: +0.09 / −0.04 / **−0.16** | **−0.25 [−0.44, −0.08]**
- C2-PIA − PIA: +0.09 / −0.05 / 0.00 | −0.09 [−0.18, +0.01]
- (C2-PIA − OSFD: +0.12 / −0.04 / −0.07 | −0.19 [−0.34, −0.04])
- Đọc: phần lợi (nếu có) của loss phân loại trên Swin-L nằm ở object NHỎ (+0.09 … +0.12), KHÔNG ở object lớn — C2 còn phá object lớn của Swin-L KÉM hơn OSFD. Giả thuyết "loss mức RoI (cls) mở kênh object lớn" bị bác ở dev30. Caveat lớn: ở lần sinh lại này lợi AP của loss cls trên Swin-L gần như không còn (C2 − OSFD −1.6; C2-PIA − PIA +0.7) → phân rã một lợi gần 0, lực yếu; muốn kết luận chắc cần dev100 (PNG dev100 đã mất). Swin-T: C2 − OSFD lớn − nhỏ −0.15, cùng chiều.
- Hệ quả: loss phân loại của surrogate tuy tính trên RoI nhưng δ sinh ra vẫn là nhiễu cục bộ (khớp E7) → "mức object" theo nghĩa loss RoI chưa đủ; cần cách tạo thành phần quy mô lớn / hình dạng trực tiếp hơn.

**HF (2026-10-06):** upload `runs/dev300_B50_eps5.tar` (363 MB, sha256 08f5e429…dcad93ea — PNG dev30 của 15 ô elem + C2 + C2-PIA, dets / gen_dets dev30; GHI ĐÈ bản cũ nếu có) và `runs/e8_wbosfd_dev30.tar` (PNG WB-OSFD dev30). Push GitHub CHƯA được (máy không có credential; `.env` chỉ có HF_TOKEN) — user push tay.

---

## 2026-10-06 — TẠM DỪNG (user, mai tiếp): trạng thái + việc mở

**Xong trong phiên:** E1–E9 (kết quả + quy tắc ở các entry 2026-10-06 phía trên; tổng hợp: entry "TỔNG HỢP E1–E8"); bước 1 chuỗi elem (chọn SCL25, CAM-R15, SHR30), bước 2 tạm gác. Không có tiến trình nào đang chạy.
**Artifact:** HF có `runs/n300_B50_eps5.tar` (MI / M-DI² / OSFD n300), `runs/dev300_B50_eps5.tar` (PNG dev30: 15 ô elem + C2 + C2-PIA + dets), `runs/e8_wbosfd_dev30.tar`. Chưa upload: `artifacts/e6/dets.json`, `artifacts/e7/` (bản thử), gen_dets Swin-L n300 mới (sinh lại được ~10 phút bằng `eval_generalization --models dino_swin_l --out generalization_e2_swinl.json`).
**Git:** commit cục bộ `45792ba`, `1f60a25` + entry này — CHƯA PUSH (máy không có credential GitHub). Máy mới: kiểm `git log origin/main` có các commit này chưa; nếu chưa, máy cũ phải push trước khi trả.

**Việc mở cho phiên sau (user chọn):**
1. **Xác nhận E9 @ dev100** (~30 phút: sinh lại OSFD / PIA / C2 / C2-PIA dev100, tách theo kích thước) — vì lần sinh lại dev30 lợi Swin-L của loss cls gần như mất (C2-PIA 29.3 vs 34.7 cũ), phân rã thiếu lực.
2. **Thiết kế thí nghiệm "tạo thành phần hình dạng trực tiếp"** — vd attack R50 trên view đã xóa texture (blur / lọc giữ biên / stylize) để buộc δ có tác dụng quy mô object; đo theo kích thước trên Swin-L (kênh object lớn). Lưu ý: lọc thông thấp GRADIENT (S1) đã âm — khác can thiệp lên VIEW. Phải khóa quy tắc + ước chi phí, user confirm trước khi chạy.
3. Các nợ cũ: GO @ n300 cho C2 (110 / 300), C2-PIA chưa sàng lọc dev300, B = 200 / ε = 8, verify clean AP Generalization Panel.

---

## 2026-10-07 — Định hướng lại (user) + E10, E11: khóa quy tắc TRƯỚC khi chạy (dev30, chỉ mô tả / sinh giả thuyết)

**3 câu hỏi user chốt (thay danh sách việc mở của entry trước):**
- **Q1:** đa dạng hình học giúp transfer qua ĐẶC TÍNH nào của δ — phải định danh được để tối ưu trực tiếp, không tăng augmentation mò mẫm.
- **Q2:** vì sao DINO-Swin-L chỉ ~30% (NOISE 14.9 → OSFD / PIA-noT 27–30, gấp đôi) trong khi white-box 84.6% (E8) — δ đánh được Swin-L khác δ từ R50 ở đâu.
- **Q3:** từ Q1 + Q2 xây method SOTA + novelty. Chỉ thiết kế SAU khi Q1 / Q2 có đặc tính cụ thể; rà literature (shape-bias / low-frequency / object-scale attack) trước khi khóa.
**Bỏ:** việc mở 1 (E9 @ dev100 — kết quả âm, không trả lời Q1 / Q2). **Hoãn:** việc mở 2 (tổng hợp δ hình dạng từ R50 — nếu âm không tách được "hình dạng không phải yếu tố" với "chưa tạo được δ hình dạng"; chờ E10).
**Đính chính:** "độ bền ↔ transfer theo ẢNH" đã kiểm ở E2(a) — KHÔNG ủng hộ. E2(b) (giữa ô) chưa chạy; đó mới là mức đúng của Q1 (vì sao METHOD có đa dạng transfer hơn, không phải vì sao ẢNH này dễ hơn).

### E10 (Q2) — mổ xẻ δ ĐÃ đánh được Swin-L (white-box) so với δ từ R50
**Lý do:** E7 cho thấy δ OSFD-R50 là mẫu tần cao khớp nội dung, ≈ 0 tác dụng còn lại trên Swin-L khi làm mờ / xáo. Chưa biết δ white-box (84.6%) có khác về quy mô không. Mổ xẻ δ đã hiệu quả cho câu trả lời trực tiếp hơn là cố tổng hợp δ hình dạng từ R50.
**E10a (forward, PNG có sẵn):** δ = PNG − ảnh gốc, dev30, 2 nguồn: **WB-OSFD** (`artifacts/e8/WB-OSFD/`, HF `runs/e8_wbosfd_dev30.tar`) và **OSFD** (R50, `artifacts/runs/dev300_B50_eps5/adv/OSFD/`). Điều kiện trên δ:
- ID; 4 phép E7 trong GT box (BLUR02, BLUR05, SHUF3, SHUF6 — đúng `corrupt` của E6);
- **mới — LP toàn ảnh, liều nhẹ (tránh hiệu ứng sàn của bản thử E7):** δ ← G_σ * δ (Gaussian từng kênh, cả ảnh), σ ∈ {1, 2, 4} px.
Ảnh = round(clip(x + T(δ))) uint8; 6 model (R50, R101, ConvNeXt-T, Swin-T, YOLOX-S, DINO-Swin-L) qua `inference_detector`; dets ảnh sạch dev30 sinh mới (dets E6 là n300, không chứa dev30). Chỉ số như E7: A(c, z) = 1 − mean retained (GT phát hiện ở ảnh sạch, score ≥ 0.3), F(c, z) = A(c, z) / A(ID, z); bootstrap cụm theo ảnh (1000, BOOT_SEED 2026). Mô tả kèm: phân bố năng lượng 3 dải của δ (`band_fracs` của A4) và tỉ lệ năng lượng trong GT box / diện tích box, WB-OSFD vs OSFD.
**Quy tắc E10a (khai báo trước; chính: Swin-L, object LỚN, điều kiện LP σ = 2):**
- (i) **"δ đánh được Swin-L mang thành phần QUY MÔ LỚN"** — ỦNG HỘ nếu F_WB(LP2, large) ≥ 0.5 VÀ F_WB − F_OSFD (LP2, large, Swin-L) > 0 với cận dưới CI > 0. Hệ quả: đặc tính đích cho method = thành phần quy mô lớn; nút thắt = R50 không sinh ra thành phần đó.
- (ii) **"Swin-L cũng bị đánh bằng mẫu tần cao khớp nội dung"** — nếu F_WB(LP2, large) ≤ 0.2. Hệ quả: câu chuyện "hình dạng / quy mô" SAI cho phía δ; khác biệt nằm ở nội dung mẫu (hướng), không ở quy mô → Q2 phải tìm hướng khác.
- Ở giữa → xám, thảo luận. Mô tả: đường F theo σ ∈ {1, 2, 4}, phép E7 (BLUR / SHUF) cùng chiều = củng cố.
- Sanity: A(ID) WB-OSFD trên Swin-L khớp E8 (AP drop 84.6, sup large 0.57) trong ±0.05 — lệch → kiểm đường đọc trước khi đọc.
**E10b (white-box, GPU nặng):** chạy các ô quy mô thô đã thiết kế trong E8 nhưng chưa chạy: `e8_whitebox_lowfreq.py --scales 1 16 32 --budget 50` (δ = upsample(z), lưới thô s, loss cls DINO, MI, ε = 5; B = 50 để ngang WB-OSFD). Ước ~15 min / ô (WB-OSFD 913 s / 30 ảnh), tổng ~45 min, peak ≲ 20 GB.
**Quy tắc E10b (chính: sup large Swin-L):** "trong ε = 5, RIÊNG nhiễu quy mô lớn hạ được object lớn Swin-L" — ỦNG HỘ nếu WB-s16 sup(large) ≥ 0.40 (= OSFD transfer 0.25 + 0.15, cùng ngưỡng E8). WB-s1 là mốc cùng loss. s32 mô tả. Không ủng hộ → trần của hướng "δ quy mô lớn" thấp, method không nên đặt cược vào đó.
**Đọc chung E10:** (i) + E10b ủng hộ → đặc tính đích cho method rõ; (ii) → bỏ hướng quy mô, Q2 chuyển sang phân tích hướng / nội dung của δ WB (vd tương đồng δ WB vs δ R50 theo vùng object).

### E11 (Q1) — đặc tính δ nào đi theo transfer GIỮA các ô (E2(b) mở rộng)
**Ô:** 17 ô dev30 có PNG (HF `runs/dev300_B50_eps5.tar`): OSFD, NOISE, FLIP, PIA, PIA-noT, CAM-R15, CAM-R30, ROT7/15/30, SHR15/30, ANI10/25, SCL25, C2 (`PROBE-osfd+cls`), C2-PIA. Drop từng target lấy từ `metrics/generalization_{elem,e9}_dev30.json` (đã có, không đo lại).
**Đặc tính (chỉ phía R50 / δ, không chạm target):**
- P1 **độ bền hình học thuần:** E_surv − E_noise8 trên R50 (`e2_survival.py --run dev300_B50_eps5 --subset dev30 --methods <17 ô> --tag dev30`).
- P2 **tần thấp:** tỉ lệ năng lượng dải thấp [0, 1/6) của δ (`band_fracs`, A4).
- P3 **phân bổ không gian:** năng lượng δ trong GT box / tỉ lệ diện tích box (`spatial`, A4).
- P4 **quy mô hiệu dụng trên R50:** F_R50(LP2) — phần tác dụng còn lại trên R50 sau LP σ = 2 (như E10a, chỉ R50).
**Phân tích:** Spearman giữa 17 ô (đặc tính trung bình ô ↔ drop dev30), CI bootstrap ảnh trên đặc tính (drop cố định), từng target: ConvNeXt-T, Swin-T, YOLOX-S, DINO-Swin-L; kèm R101 (đối chứng cùng họ).
**Quy tắc (mô tả, khai báo trước):** đặc tính X "đi theo transfer khác họ" nếu ρ ≥ 0.5, cận dưới CI > 0, ở ≥ 2/3 target CrossAvg₃ VÀ ρ(X, khác họ) > ρ(X, R101) (điểm). Riêng Swin-L báo cáo tách. Đặc tính đạt → ứng viên can thiệp (thí nghiệm nhân quả riêng, khóa entry mới). Không đặc tính nào đạt → Q1 chưa có đặc tính đo được phía δ; ghi nhận.
**Caveat khai báo trước:** 17 ô phần lớn cùng loss OSFD, chênh drop giữa ô ~5 điểm ≈ nhiễu dev30 (±5) → lực thấp; KHÔNG dùng làm claim paper, chỉ chọn đặc tính để can thiệp. P1 có thể bão hòa (OSFD E ≈ 0.96 ở n300).
**Chi phí:** E11 ~10 min GPU (R50 forward) + CPU; E10a ~10–15 min (6 model × 30 ảnh × 2 nguồn × 8 điều kiện + ảnh sạch). Cần tải HF: `runs/dev300_B50_eps5.tar`, `runs/e8_wbosfd_dev30.tar`.
**Thứ tự:** E10a → E11 (rẻ, forward) → E10b (nặng, chỉ chạy sau khi user xem E10a). Không tune attack nào theo E10 / E11.

### Kết quả E10a (`results/mechanism/e10a_wb_dissect.json`, dev30, log `results/e10/e10a.log`, dets `artifacts/e10/dets.json`; script `scripts/e10a_wb_dissect.py`) — theo quy tắc: **(ii) — Swin-L cũng bị đánh bằng mẫu TẦN CAO khớp nội dung**
**Sanity:** AP drop Swin-L WB-OSFD 84.6 (E8 84.6), OSFD 27.2 (27.2) — khớp tuyệt đối. sup large WB 0.55 (E8 0.57, trong ±0.05). Lưu ý: sup ở script này = s_adv < 0.3·s_clean (tương đối), E8 dùng s_adv < 0.3 tuyệt đối → OSFD 0.18 vs 0.25 là do định nghĩa, không dùng cho quy tắc.
**Quy tắc:** F_WB(LP2, lớn, Swin-L) = **0.10 [0.01, 0.18]** ≤ 0.2 → (ii). F_WB − F_OSFD = +0.08 [−0.08, +0.16] (CI chứa 0).
Sức attack A(ID) nhỏ / vừa / lớn trên Swin-L: WB 0.68 / 0.66 / 0.58; OSFD 0.19 / 0.20 / 0.24.
F (phần còn lại) trên Swin-L, nhỏ / vừa / lớn: WB LP1 0.40 / 0.38 / 0.49, LP2 0.12 / 0.10 / 0.10, LP4 0.04 / 0.03 / 0.03; OSFD LP1 0.26 / 0.36 / 0.17, LP2 0.08 / 0.09 / 0.02. Phép E7 trong box (object lớn): WB BLUR02 0.05, SHUF3 0.09 — sàn như OSFD.
**Mô tả thêm:**
- **δ WB còn TẦN CAO HƠN δ R50:** năng lượng thấp / trung / cao WB 0.338 / 0.344 / 0.319 vs OSFD 0.398 / 0.329 / 0.272 (CI dải thấp WB [0.329, 0.347] vs OSFD [0.382, 0.418], tách rời). Không tập trung vào object (in-box ≈ 1.0 cả hai).
- Ở MỌI model, LP σ = 1 px đã mất ~½ tác dụng, σ = 2 px còn ≤ 0.20 — cả δ WB lẫn δ R50 đều là mẫu mịn cỡ 1–2 px khớp nội dung; đường suy giảm của δ WB trên Swin-L giống δ R50 trên R50 (LP1 0.49 vs 0.57, LP2 0.10 vs 0.17).
- **Chuyển giao ngược yếu tương tự:** δ WB-Swin-L đánh R50 / R101 / ConvNeXt / Swin-T / YOLOX ở A 0.29–0.54 (vs 0.58–0.68 trên chính Swin-L); tức Swin-L → CNN cũng "rơi" như R50 → Swin-L.
**Đọc:** Giả thuyết "δ cần thành phần quy mô lớn / hình dạng để đánh Swin-L" bị BÁC ở phía δ (dev30). E6 (Swin-L nhận object lớn sạch bằng hình dạng) vẫn đứng, nhưng attack hiệu quả KHÔNG cần phá hình dạng — một mẫu tần cao đúng "nội dung" đủ hạ cả object lớn (0.58). Khác biệt WB vs R50 không ở quy mô / phổ / phân bổ không gian mà ở **HƯỚNG / NỘI DUNG mẫu** (khớp E1: gradient R50 ⟂ gradient target). Hệ quả: Q2 chuyển sang phân tích hướng / nội dung (vd tương quan δ WB vs δ R50 theo vùng, thành phần chung của δ WB-Swin-L và δ đa dạng hình học). Method không nên đặt cược vào "δ tần thấp / quy mô object". E10b (ô quy mô thô): kỳ vọng âm vì δ WB tối ưu tự do vẫn chọn tần cao; nhưng E10a chỉ lọc δ có sẵn, không loại trừ δ tối ưu trực tiếp ở lưới thô → chạy hay bỏ: user quyết.

### Kết quả E11 (`results/mechanism/e11_cell_props.json`, `e2_per_image_dev30.json` + `e2_survival_dev30.json`; log `results/e10/e11.log`, `results/e10/e11_e2.log`; script `scripts/e11_cell_props.py`) — theo quy tắc: **P4 ĐẠT (hình thức), P1 / P2 / P3 không**
Spearman giữa 17 ô [CI bootstrap ảnh, drop cố định], ConvNeXt / Swin-T / YOLOX / Swin-L / R101:
- P1 (bền hình học thuần, E_surv − E_noise8 trên R50): +0.21 / +0.69 / +0.32 / **+0.60 [0.42, 0.75]** / −0.00 → chỉ Swin-T đạt.
- P2 (tỉ lệ năng lượng dải thấp của δ): +0.40 / +0.25 / +0.53 / +0.43 / **−0.29** → chỉ YOLOX đạt.
- P3 (năng lượng trong box / diện tích): không có phương sai (1.004–1.007 ở mọi ô) → vô thông tin.
- P4 (F_R50(LP2)): **+0.80 / +0.67** / +0.41 / +0.50 / +0.66 → ĐẠT (ConvNeXt, Swin-T), nhưng Swin-T chỉ hơn R101 0.01.
**Mô tả thêm (KHÔNG thuộc quy tắc, khai báo là phân tích sau):** bỏ 2 ô C2 (khác loss) → 15 ô cùng loss OSFD: P2 +0.65 / +0.61 / +0.61 / **+0.67** / **−0.10**; P4 +0.74 / +0.52 / +0.58 / +0.41 / +0.50; P1 +0.00 / +0.55 / +0.41 / +0.54 / −0.46. Bỏ thêm NOISE / FLIP (13 ô có hình học): P2 +0.47 / +0.40 / +0.43 / +0.50 / −0.26; P4 +0.62 / +0.29 / +0.36 / +0.17 / +0.53.
**Đọc:** (1) P4 đạt nhưng KHÔNG đặc thù khác họ (đi theo cả R101) và bị kéo bởi 2 ô C2 (P4 0.25 / 0.21 vs 0.10–0.16) → P4 ≈ "độ mạnh chung của δ trên R50". (2) Tín hiệu ĐẶC THÙ khác họ duy nhất là P2 trong các ô cùng loss: tần thấp hơn ↔ transfer khác họ tốt hơn ở cả 4 target, KHÔNG ở R101 — khớp ablation cũ (RRB làm δ tần thấp hơn, D₄ R50 giảm). Kết hợp E10a (δ WB-Swin-L có ÍT tần thấp hơn mà vẫn đánh mạnh): tần thấp KHÔNG phải thứ target cần, mà là DẤU HIỆU δ R50 bớt phần tần cao đặc thù surrogate. (3) Caveat: CI chỉ tính nhiễu ảnh trên đặc tính, không tính nhiễu drop dev30 (±5) → quá hẹp; chênh P2 giữa ô nhỏ (0.34–0.42). Không phải claim.

---

## 2026-10-07 — E12 (Q2, hướng mới sau E10a): NỘI DUNG δ R50 có chung với δ WB-Swin-L không, và đa dạng hình học có kéo gần không? — khóa TRƯỚC khi chạy (CPU, PNG có sẵn, dev30)
**Lý do:** E10a — khác biệt δ WB vs δ R50 không ở quy mô / phổ / không gian → nằm ở hướng / nội dung. E12 đo trực tiếp: tương đồng giữa δ của 17 ô (R50) và δ WB-OSFD (Swin-L) trên cùng ảnh.
**Đo (theo ảnh, rồi trung bình ô):** cos(δ_ô, δ_WB) trên toàn ảnh (3 kênh) cho 2 phần: **HP** = δ − G₂ * δ (phần mịn, mang tác dụng theo E10a — CHÍNH) và LP = G₂ * δ (mô tả). Mốc 0: cos(δ_ô(ảnh i), δ_WB(ảnh j ≠ i)) sau resize về cùng kích thước (ước nhiễu nền), mô tả.
**Quy tắc (khai báo trước):**
- (a) **"Đa dạng hình học kéo nội dung δ R50 về phía δ WB-Swin-L"** — ỦNG HỘ nếu cos_HP(OSFD) − cos_HP(NOISE) > 0 với CI (bootstrap ảnh, paired) > 0, VÀ cos_HP(OSFD) > cos_HP mốc 0 + 3 lần sd bootstrap của mốc.
- (b) **"Mức chung nội dung với δ WB giải thích transfer Swin-L giữa ô"** — ỦNG HỘ nếu Spearman(cos_HP ô, drop Swin-L) ≥ 0.5 với cận dưới CI > 0 trên 17 ô, VÀ ρ(Swin-L) > trung bình ρ(ConvNeXt, Swin-T, YOLOX) (điểm) — tức đặc thù Swin-L, không chỉ "δ mạnh chung".
- (a) ∧ (b) → cơ chế Q2 = đa dạng tăng phần nội dung chung với hướng tấn công của Swin-L; nút thắt = phần chung còn nhỏ. (a) ∧ ¬(b) → đa dạng có kéo gần nhưng không phải thứ quyết định giữa ô. ¬(a) → nội dung δ R50 và δ WB gần trực giao bất kể đa dạng (khớp E1) → Q2 phải tìm ở phía target (cách Swin-L / DINO xử lý δ), không ở phía δ.
Chi phí: CPU vài phút. Không tune attack nào theo E12.

### Kết quả E12 (`results/mechanism/e12_content_align.json`, log `results/e10/e12.log`, script `scripts/e12_content_align.py`) — theo quy tắc: **(a) ∧ (b)** (b sát ngưỡng)
cos_HP(δ_ô, δ_WB) trung bình ảnh: OSFD **+0.0256**, PIA-noT +0.0234, PIA +0.0230, các phép cơ bản +0.0206 … +0.0221, C2 +0.0192, C2-PIA +0.0204, FLIP +0.0193, **NOISE +0.0172**; mốc 0 (ảnh lệch) −0.0001 … −0.0007 (sd bootstrap 0.0003). LP: OSFD +0.0399 … NOISE +0.0280.
- (a) cos_HP(OSFD) − cos_HP(NOISE) = **+0.0084 [0.0066, 0.0101]**; OSFD ≫ mốc 0 + 3 sd → ĐẠT.
- (b) Spearman(cos_HP ô, drop) ConvNeXt / Swin-T / YOLOX / **Swin-L** / R101: +0.36 / +0.16 / +0.53 / **+0.50 [0.44, 0.70]** / −0.28; Swin-L > TB ba target khác (0.35) → ĐẠT (đúng ngưỡng 0.50). LP (mô tả): +0.66 / +0.46 / +0.51 / **+0.77** / +0.29.
**Kiểm confound (phân tích sau, mô tả):** cả δ R50 lẫn δ WB đều tương quan ÂM với high-pass của chính ảnh (cos(δ_HP, x_HP): WB −0.026; ô R50 −0.047 … −0.094 — attack làm giảm tương phản cục bộ). Chiếu bỏ thành phần theo ảnh khỏi cả hai δ rồi tính lại: phần chung còn **+0.015 (NOISE) … +0.024 (OSFD)**, Spearman với drop Swin-L **+0.59**, R101 −0.21 → phần chung KHÔNG chỉ là cấu trúc ảnh.
**Đọc (dev30, mô tả):** δ R50 và δ WB-Swin-L có một thành phần nội dung CHUNG nhỏ nhưng thật (cos ~0.02, ≈ 0.05% năng lượng); đa dạng hình học tăng nó ~50% (NOISE → OSFD) đúng lúc drop Swin-L 14.9 → 27.2; giữa ô, phần chung đi theo transfer khác họ (đặc biệt Swin-L, LP mạnh hơn HP) và KHÔNG theo R101. Câu cơ chế ứng viên (Q1 + Q2): **đa dạng hình học hoạt động bằng cách triệt phần δ đặc thù surrogate (tần cao, không bền qua view) và giữ phần "chung giữa các model"; Swin-L kẹt ~30% vì phần chung này chỉ chiếm phần rất nhỏ của δ** — chứ không phải do Swin-L cần quy mô lớn (E10a). Caveat: n = 30, CI không tính nhiễu drop, (b) đúng ngưỡng; claim cần ≥ dev100 và một can thiệp nhân quả (tăng phần chung → tăng transfer).

---

## 2026-10-07 — User: hoãn E12 @ dev100, làm method luôn (method = phép can thiệp nhân quả cho E12); rà literature + pilot VC @ dev30 — khóa TRƯỚC khi chạy
**Vì sao PIA / MORGD ≈ OSFD (mô tả, suy từ E12, chưa kiểm riêng):** mọi ô có hình học có phần chung với δ WB trong cụm hẹp cos_HP 0.021–0.026 (NOISE 0.017) → loại biến đổi nào cũng triệt phần đặc thù surrogate gần ngang nhau → đổi loại không tăng phần chung; khớp kết luận cũ "có đa dạng >> loại". MORGD không có PNG → chưa đo.
**Literature (rà 2026-10-07):** GRA (ICCV 2023) — phạt dao động dấu gradient giữa 2 bước liền kề (≈ mask "ngược momentum" → KHÔNG dùng, trùng); VT (CVPR 2021) — bù phương sai gradient lân cận; PGN / GNP — vùng phẳng; Centralized Perturbation (AAAI 2024) — tối ưu trên hệ số tần số trội; FRA-Attack (2026, MLLM) — lọc thông thấp gradient để bỏ thành phần đặc thù surrogate (≈ S1 của ta, đã ÂM trên detection); LPM (mask patch học được); COGO (ensemble); detection: ICICS 2025 (ensemble feature + giảm phương sai gradient, ensemble + patch — khác threat model). Chưa thấy: đồng thuận dấu giữa các view hình học ĐỘC LẬP tại CÙNG iterate cho detection single-surrogate. Novelty của toán tử đơn lẻ mỏng → đóng góp phải nằm ở cơ chế (E10–E12) + bằng chứng can thiệp.
**Code:** `core.run_iterative_attack(combine="consensus")` — gộp backward_per_step gradient, chỉ giữ pixel mọi gradient cùng dấu (mặc định "sum" = như cũ); `osfd_attack(vc_k=K)` — K backward / bước, mỗi backward 1 lần rút RRB độc lập, steps = B // K. Ô: `OSFD-K2` / `VC-OSFD` (K = 2, 25 bước, B = 50), `OSFD-K4` / `VC4-OSFD` (K = 4, 12 bước, B = 48). **Unit test (2 ảnh dev30):** OSFD mặc định tái tạo PNG cũ (0% / 1.2% pixel khác — GPU không tất định); backward = 50; tỉ lệ pixel đồng thuận dấu giữa 2 view RRB độc lập **≈ 0.54** (mức ngẫu nhiên 0.5) → gradient từng view gần như toàn đặc thù view (khớp E1); với K = 2 mask giữ nhiều đồng thuận ngẫu nhiên → thêm K = 4 (mức ngẫu nhiên 0.125).
**Pilot dev30 (mô tả; chỉ quyết định mở rộng dev100):** 4 ô mới + OSFD (PNG có sẵn), eval 6 model, đo cos_HP(δ, δ_WB) như E12.
**Quy tắc (cho từng K):** GO dev100 nếu ĐỒNG THỜI (1) kiểm tra thao tác: cos_HP(VC_K) > cos_HP(OSFD-K_K) (điểm); (2) VC_K − OSFD-K_K ≥ +2 CrossAvg₃ VÀ ≥ +2 DINO-Swin-L (điểm); (3) CrossAvg₃(VC_K) ≥ OSFD − 1. (1) đạt mà (2) không → bằng chứng CHỐNG "tăng phần chung → tăng transfer" (ghi nhận). Không K nào đạt → NO-GO đồng thuận cứng; không thêm biến thể ngưỡng / K khác mà không khóa entry mới.

### Kết quả pilot VC @ dev30 (`results/mechanism/vc_pilot_dev30.json`, `metrics_vc_dev30.json`, `generalization_vc_dev30.json`, log `results/e10/vc_dev30.log`) — theo quy tắc: **NO-GO cả K = 2 và K = 4** (trượt cả 3 điều kiện)
Relative AP drop %, R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CrossAvg₃ | cos_HP:
- OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9 | +0.0256
- OSFD-K2 97.3 / 92.7 / 79.8 / 79.1 / 71.1 / 29.6 | 76.6 | +0.0277
- VC-OSFD 97.1 / 90.3 / 75.8 / 77.3 / 68.7 / 23.0 | 73.9 | +0.0272
- OSFD-K4 95.9 / 88.2 / 71.9 / 74.9 / 63.6 / 25.7 | 70.2 | +0.0311
- VC4-OSFD 93.1 / 72.6 / 64.0 / 62.1 / 46.1 / 20.4 | 57.4 | +0.0284
K = 2: cos 0.0272 < 0.0277 (thao tác trượt); ΔCA3 −2.7, ΔSwin-L −6.5. K = 4: cos 0.0284 < 0.0311; ΔCA3 −12.7, ΔSwin-L −5.3.
**Đọc:** (1) Đồng thuận dấu cứng KHÔNG tăng phần chung (kiểm tra thao tác trượt) → pilot KHÔNG kiểm được giả thuyết E12, chỉ bác toán tử này. Lý do khả dĩ: gradient 2 view chỉ cùng dấu ~54% pixel → mask xóa ~½ (K = 2) đến phần lớn (K = 4) cập nhật; với bước sign, pixel bị xóa không được cập nhật → δ yếu đi (VC4 white-box R50 chỉ 93.1, R101 tụt 88 → 73) — toán tử phá ngân sách hiệu dụng, không lọc "phần chung". (2) Ít bước hơn (OSFD-K2 / K4: 25 / 12 bước, nhiều view độc lập hơn mỗi bước) làm cos_HP TĂNG (0.0256 → 0.0277 → 0.0311) trong khi CrossAvg₃ GIẢM (78.9 → 76.6 → 70.2) → cos_HP và transfer TÁCH nhau khi đổi số bước (lẫn độ mạnh δ: R50 97.8 → 95.9) — caveat cho E12: cos_HP có thể đo một phần "δ chưa bão hòa", không chỉ "phần chung". Swin-L OSFD-K2 29.6 > OSFD 27.2 (trong nhiễu ±5).
**Hệ quả:** phần chung KHÔNG có sẵn ở gradient từng bước (khớp E1, đồng thuận ~0.54), nó hình thành qua TÍCH LŨY (momentum × nhiều view × nhiều bước). Không thêm biến thể đồng thuận (mềm / ngưỡng) mà không khóa entry mới; hướng tiếp: user quyết.

---

## 2026-10-07 — E12c (user chọn A): cos_HP của E12 đo "phần chung" hay chỉ độ mạnh / bão hòa của δ? — khóa TRƯỚC khi chạy (dev30)
**Lý do:** pilot VC — ít bước hơn (OSFD-K2 / K4) làm cos_HP tăng mà transfer giảm. Nếu cos_HP lẫn với mức bão hòa / số bước, kết luận E12 (a)(b) không đứng.
**(i) Giữa ô (CPU, PNG có sẵn, 21 ô = 17 ô E12 + OSFD-K2 / VC-OSFD / OSFD-K4 / VC4-OSFD):** sat = tỉ lệ pixel |δ| = ε (= 5 trên PNG uint8); cos_HP như E12; **cos_sgn** = cos(HP(sign δ_ô), HP(δ_WB)) — bỏ thông tin biên độ của δ_ô. Báo cáo Spearman(sat, cos_HP) trên 21 ô. **Tương quan riêng** trên 17 ô E12: phần dư của cos_HP sau hồi quy tuyến tính theo sat (giữa ô) ↔ drop Swin-L và CrossAvg₃ (Spearman).
**(ii) Quỹ đạo OSFD (GPU ~10 min):** 1 lần chạy OSFD B = 50 trên dev30 (seed như run_baselines), lưu snapshot δ sau b ∈ {10, 20, 30, 40, 50} backward (`on_step`, noise sau k step = kết quả chạy steps = k); mỗi snapshot: cos_HP, sat, AP drop 6 model (inference_detector).
**Quy tắc:** cos_HP được GIỮ làm thước đo phần chung nếu ĐỒNG THỜI (a) tương quan riêng (| sat) với drop Swin-L ≥ 0.4 trên 17 ô, VÀ (b) trên quỹ đạo, cos_HP không giảm khi CrossAvg₃ tăng (Spearman(cos_HP, CA3) theo 5 snapshot ≥ 0). cos_HP bị LOẠI nếu (a) < 0.2 VÀ (b) < 0 → kết luận E12 (a)(b) vô hiệu, ghi lại. Còn lại → xám. Mô tả: cos_sgn cùng các phân tích.

### Kết quả E12c (`results/mechanism/e12c_measure_check.json`, log `results/e10/e12c.log`, snapshot `artifacts/e12c/b{10..50}/`, script `scripts/e12c_measure_check.py`) — theo quy tắc: **GIỮ cos_HP** ((b) đúng biên 0)
**(i) Giữa ô:** ρ(sat, cos_HP) 21 ô = −0.43 (ô ít bão hòa — K2 / K4 / VC — có cos cao hơn: có lẫn). Tương quan riêng cos_HP | sat trên 17 ô E12: Swin-L **+0.53**, CA3 +0.54, R101 −0.27 → giữ nguyên sau khi trừ bão hòa. cos_sgn (bỏ biên độ δ_ô) cho cùng kết quả (Swin-L +0.51, CA3 +0.54, R101 −0.26) và gần trùng cos_HP ở mọi ô (0.0169–0.0289).
**(ii) Quỹ đạo OSFD (dev30), b | sat | cos_HP | drop R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3:**
10 | 0.68 | 0.0247 | 88.5 / 73.2 / 53.3 / 49.0 / 35.1 / 18.5 | 45.8; 20 | 0.81 | 0.0260 | 95.0 / 88.9 / 71.2 / 72.3 / 62.1 / **27.6** | 68.5; 30 | 0.86 | 0.0260 | … / 27.8 | 75.7; 40 | 0.89 | 0.0258 | … / 24.1 | 77.2; 50 | 0.90 | 0.0255 | 97.8 / 93.6 / 83.1 / 77.6 / 75.7 / 26.2 | 78.8. ρ(cos_HP, CA3) = 0.00.
**Đọc:** (1) cos_HP KHÔNG chỉ là bão hòa: giữa method, phần chung (đã trừ bão hòa) vẫn đi theo transfer khác họ, không theo R101. (2) TRONG một method, phần chung được xác lập SỚM (b = 10 đã 0.0247) rồi phẳng; transfer CNN tăng tiếp theo b nhờ biên độ / bão hòa (sat 0.68 → 0.90). (3) **Swin-L bão hòa sớm ~b = 20 (27.6) rồi đứng / dao động (24–28)** trong khi CNN target còn tăng → mô hình làm việc: transfer ≈ (độ mạnh δ) × (tỉ lệ phần chung); CNN khác họ còn hưởng thêm độ mạnh, Swin-L bị chặn bởi tỉ lệ phần chung (~0.026) → giải thích "kẹt ~30%". (4) Hệ quả cho VC: K2 / K4 có tỉ lệ chung cao hơn nhưng yếu (sat 0.74–0.86) → transfer thấp; method phải tăng tỉ lệ phần chung MÀ KHÔNG giảm độ mạnh (giữ đủ bước / bão hòa). Caveat: dev30, (b) đúng biên, CI chưa tính.

---

## 2026-10-07 — User: hướng 1 (method mới, mục tiêu SOTA). Pilot V @ dev30: tăng số view độc lập / backward, GIỮ 50 bước — khóa TRƯỚC khi chạy
**Lý do (từ E12c + VC):** tỉ lệ phần chung tăng khi gộp nhiều view độc lập mỗi bước (K2 / K4) nhưng K2 / K4 mất bước → δ yếu. Protocol: B = số backward, gộp view trong 1 backward = 1 (protocol_lock.md) → tăng view / backward giữ B = 50 và giữ 50 bước. Kỳ vọng (hồi quy giữa ô E12c, ngoại suy): +1.5 Swin-L / +1.4 CA3 mỗi +0.001 cos_HP. **Đây là phép thử giả thuyết + mức trần, CHƯA phải method có novelty** (nhiều bản sao augmentation ≈ SIA / BSR / PIA-N); nếu đạt → thiết kế cách đạt cùng hiệu ứng có novelty / rẻ hơn (khóa entry mới). Báo kèm view forward / runtime.
**Ô:** `OSFD-V4` (2 lần rút RRB độc lập = 4 view / backward), `OSFD-V8` (4 lần = 8 view), 50 bước, còn lại = OSFD. Mặc định rrb_draws = 1 trùng OSFD.
**Quy tắc (từng ô, dev30, mô tả; chỉ quyết định mở rộng dev100):** GO dev100 nếu ĐỒNG THỜI (1) thao tác: cos_HP > OSFD (0.0256) VÀ sat ≥ 0.88; (2) CA3 ≥ OSFD + 2 VÀ Swin-L ≥ OSFD + 2 (điểm). (1) đạt mà (2) không → bằng chứng CHỐNG "tăng phần chung khi giữ độ mạnh → tăng transfer". (1) không đạt → thao tác thất bại, không kết luận cơ chế.

### Kết quả pilot V @ dev30 (`results/mechanism/v_pilot_dev30.json`, `metrics_v_dev30.json`, `generalization_v_dev30.json`, log `results/e10/v_dev30.log`) — theo quy tắc: **không GO** (cả hai: (1) đạt, (2) trượt ở Swin-L)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | cos_HP | sat | runtime: OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9 | 0.0256 | 0.901 | ~3.1 s; **V4 98.6 / 92.9 / 85.9 / 81.3 / 78.6 / 28.9 | 81.9** | 0.0260 | 0.911 | ~6 s; **V8 98.2 / 94.6 / 85.2 / 81.6 / 79.4 / 24.2 | 82.1** | 0.0262 | 0.917 | 12.6 s.
**Đọc:** (1) Giữ 50 bước, gộp thêm view: phần chung tăng RẤT ÍT (+0.0004 / +0.0006, hồi quy dự đoán +0.6 / +0.9 Swin-L) → phần cos cao của K2 / K4 chủ yếu đến từ ÍT BƯỚC (δ chưa bão hòa), không từ nhiều view — caveat E12c xác nhận lại; momentum qua 50 bước đã trung bình view gần đủ. (2) CNN khác họ tăng ~+3 CA3 (ConvNeXt +2.6, Swin-T +3.5, YOLOX +3.0 … +3.8; trong nhiễu dev30 ±5), Swin-L không đổi (+1.7 / −3.1). (3) Hệ quả: với surrogate R50, tỉ lệ phần chung với Swin-L dường như có TRẦN ~0.026 mà đa dạng / số view không vượt được → Swin-L có thể là giới hạn của single-surrogate R50 ở ε = 5. V4 là ứng viên plug-in rẻ cho CA3 (novelty yếu: nhiều bản sao augmentation).

---

## 2026-10-07 — Kiểm "PHA LƯỚI" (user: mục 1 — vì sao RRB / PIA / MORGD ≈ nhau): chỉ tịnh tiến, bội stride vs lẻ — khóa TRƯỚC khi chạy (dev30)
**User:** tạm dừng mục 2 (dùng cos_HP / E12 để dự đoán transfer Swin-L — thấy không hợp lý).
**Giả thuyết:** phần δ đặc thù surrogate gắn với LƯỚI lấy mẫu của R50 (stride / aliasing / vị trí pixel); view hình học làm lệch pha lưới khác nhau mỗi view → gradient (warp ngược về khung gốc) của phần gắn lưới triệt tiêu qua momentum, phần gắn nội dung cộng dồn. Mọi phép làm lệch pha lưới đều triệt ngang nhau → loại / biên độ ít quan trọng (khớp: ablation "có >> loại", PIA-noT, ROT30 ≯ ROT15, FLIP — giữ lưới — nằm giữa NOISE và hình học, E10a mẫu 1–2 px).
**Ô (khung `geo_osfd_attack(geo="elem")`, như NOISE / ROT: loss feature OSFD k = 3, MI, 2 view / backward, σ = 6, feature sạch co-transform, B = 50, ε = 5):** chỉ tịnh tiến (tx, ty) rút độc lập mỗi trục, khung model, lấp 0:
- `SH32`: tx, ty ∈ {−64, −32, 0, 32, 64} — GIỮ pha lưới mọi stride ≤ 32 (|t| TB 38.4 px);
- `SHODD`: số nguyên trong [−64, 64] KHÔNG chia hết 32 — lệch pha (|t| TB ~33 px);
- `SHSUB`: U(−1, 1) px, bilinear — lệch pha dưới 1 px (mô tả; lẫn: nội suy bilinear làm mờ nhẹ).
Mốc: NOISE (không hình học), OSFD, ROT15 (PNG dev30 có sẵn, eval lại cùng lần). Unit test: warp dịch nguyên == dịch chính xác (sai số ≤ 0.002).
**Quy tắc (dev30, CA3 = TB ConvNeXt / Swin-T / YOLOX):** giả thuyết pha lưới ỦNG HỘ nếu ĐỒNG THỜI (1) CA3(SHODD) − CA3(SH32) ≥ 5; (2) SH32 gần NOISE hơn: CA3(SH32) − CA3(NOISE) ≤ ½·[CA3(SHODD) − CA3(NOISE)]; (3) CA3(SHODD) ≥ CA3(ROT15) − 3. BÁC nếu CA3(SH32) ≥ CA3(SHODD) − 2 (dịch bội stride hiệu quả ngang dịch lẻ → pha lưới không phải cơ chế). Còn lại → xám. Swin-L, R101, SHSUB: mô tả.

### Kết quả kiểm "pha lưới" @ dev30 (`metrics_grid_dev30.json`, `generalization_grid_dev30.json`, log `results/e10/grid_dev30.log`) — theo quy tắc: **ỦNG HỘ** (cả 3 điều kiện)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: NOISE 98.7 / 89.8 / 67.2 / 68.4 / 57.4 / 14.9 | 64.3; **SH32 98.7 / 92.3 / 69.2 / 71.8 / 65.7 / 16.4 | 68.9**; **SHODD 97.5 / 91.8 / 80.0 / 81.5 / 76.0 / 22.3 | 79.2**; SHSUB 98.7 / 91.2 / 74.5 / 72.7 / 64.1 / 17.3 | 70.4; ROT15 97.0 / 91.2 / 79.6 / 81.7 / 76.9 / 23.7 | 79.4; OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9.
(1) SHODD − SH32 = **+10.3** ≥ 5; (2) SH32 − NOISE = +4.6 ≤ ½ · 14.9; (3) SHODD 79.2 ≥ ROT15 − 3. Swin-L cùng chiều (SHODD 22.3 vs SH32 16.4 vs NOISE 14.9).
**Đọc (dev30, mô tả):** (1) CHỈ tịnh tiến nguyên pixel lệch pha lưới (SHODD) đạt TRỌN lợi ích đa dạng hình học (CA3 ≈ ROT15 ≈ OSFD) — không cần xoay / co giãn / phối cảnh. (2) Cùng biên độ nhưng giữ pha lưới (bội 32) gần như mất hết (68.9 ≈ NOISE 64.3) → lợi ích KHÔNG đến từ "nội dung dời chỗ" mà từ LỆCH PHA LƯỚI lấy mẫu của surrogate. Phần dư +4.6 của SH32 khả dĩ do stride 64 (P6 của RPN) vẫn bị lệch pha khi dịch 32. (3) Lệch dưới 1 px (SHSUB) chỉ giúp ít (70.4) → lưới quyết định là lưới THÔ (stride vài px trở lên), không phải lưới pixel. (4) R101 (cùng lưới stride với R50) ~không đổi giữa các ô (89.8–92.3) → phần δ gắn lưới vẫn chuyển sang model CÙNG họ / cùng lưới; khớp gap cùng họ > khác họ (RQ1). **Giải thích RRB ≈ PIA ≈ MORGD ≈ phép cơ bản:** mọi phép hình học liên tục đều làm lệch pha lưới gần như ngẫu nhiên ở mọi stride → triệt phần đặc thù lưới ngang nhau; loại và biên độ (sau ngưỡng) không quan trọng.
**Việc mở:** (a) tìm stride quyết định: dịch bội 4 / 8 / 16 (giữ pha cho stride ≤ k); (b) xác nhận dev100; (c) hệ quả method: lấy mẫu pha có chủ đích (vd phủ đều các pha mod 32) thay cho augmentation ngẫu nhiên — khóa entry riêng.

---

## 2026-10-07 — TẠM DỪNG (user, mai tiếp): tổng kết phiên + việc mở

**Câu hỏi user chốt (entry "Định hướng lại" 2026-10-07):** Q1 đa dạng hình học giúp transfer nhờ đặc tính nào; Q2 vì sao DINO-Swin-L kẹt ~30% dù white-box 84.6%; Q3 method SOTA + novelty.
**Kết quả phiên (chi tiết ở các entry phía trên, đều dev30, mô tả):**
1. **E10a:** δ white-box Swin-L cũng là mẫu TẦN CAO khớp nội dung (còn tần cao hơn δ R50) → giả thuyết "Swin-L cần δ quy mô lớn / hình dạng" BỊ BÁC. E10b (ô lưới thô) bỏ.
2. **E11:** đặc tính phía δ: P4 đạt hình thức nhưng không đặc thù khác họ; P2 (tần thấp) chỉ là dấu hiệu, không phải thứ target cần.
3. **E12 / E12c:** δ R50 và δ WB-Swin-L có phần chung nhỏ (cos ~0.02); đa dạng tăng nó NOISE → OSFD; phần chung xác lập sớm (b = 10), Swin-L bão hòa ~b = 20. **User tạm dừng hướng dùng cos_HP để nhận định / dự đoán transfer Swin-L (không hợp lý)** — chỉ giữ như mô tả.
4. **Pilot VC (đồng thuận dấu giữa view):** NO-GO — làm δ yếu; gradient 2 view RRB độc lập chỉ cùng dấu ~54% pixel.
5. **Pilot V (V4 / V8: nhiều lần rút RRB / backward, 50 bước):** CA3 +3.1 / +3.2, Swin-L không đổi → không GO; = nhiều bản sao augmentation, KHÔNG novelty, tăng compute thật (2× / 4×) → nếu dùng phải có đối chứng cùng compute.
6. **Kiểm "PHA LƯỚI" — ỦNG HỘ (trả lời Q1):** chỉ tịnh tiến lẻ pixel (SHODD) đạt trọn lợi ích đa dạng (CA3 79.2 ≈ ROT15 79.4 ≈ OSFD 78.9); tịnh tiến bội 32 cùng biên độ (SH32) ≈ NOISE (68.9 vs 64.3); dưới 1 px chỉ 70.4; R101 không đổi. → Đa dạng hình học hoạt động bằng LỆCH PHA LƯỚI STRIDE của surrogate, triệt phần δ gắn lưới R50; mọi phép hình học liên tục đều lệch pha ngẫu nhiên → RRB ≈ PIA ≈ MORGD ≈ phép cơ bản.

**Việc mở cho phiên sau (user chọn thứ tự):**
- (a) **Stride quyết định:** ô dịch bội 4 / 8 / 16 px (giữ pha cho stride ≤ k), cùng khung `geo="elem"` (`shift_offset` trong `attack/methods/elem.py` — thêm mode). ~15 phút dev30. Khóa quy tắc trước.
- (b) **Xác nhận SH32 vs SHODD @ dev100** (+ NOISE, OSFD, ROT15). Sinh 70 ảnh / ô dev100 \ dev30 (~4.8 s/ảnh/ô) + eval. OSFD dev100 đã có ~27 ảnh từ lần đo bộ nhớ (resume được).
- (c) **Method từ cơ chế pha lưới** (hướng novelty cho Q3): lấy mẫu pha có chủ đích (vd phủ đều residue mod 32 / stratified) thay augmentation ngẫu nhiên; ghép với C2 (loss cls). Rà literature "grid / aliasing / stride phase" trong transfer attack TRƯỚC khi khóa.
- Nợ cũ: GO @ n300 cho C2 (110 / 300); Q2 (Swin-L) chưa có cơ chế đứng được sau khi dừng hướng cos_HP.

**Code mới phiên này (mặc định giữ nguyên hành vi cũ — OSFD tái tạo PNG cũ, đã kiểm):** `core.run_iterative_attack(combine=)`; `osfd_attack(vc_k=, combine=, rrb_draws=)`; `elem.shift_offset` + spec `{"shift": s32|odd|sub}`; ô `OSFD-K2/K4`, `VC-OSFD`, `VC4-OSFD`, `OSFD-V4/V8`, `SH32/SHODD/SHSUB` trong `scripts/run_baselines.py`. Script: `scripts/e10a_wb_dissect.py`, `e11_cell_props.py`, `e12_content_align.py`, `e12c_measure_check.py`, `vc_pilot_check.py` (`--pilot vc|v`).
**Kết quả:** `results/mechanism/{e10a_wb_dissect, e11_cell_props, e2_*_dev30, e12_content_align, e12c_measure_check, vc_pilot_dev30, v_pilot_dev30}.json`; `results/runs/dev300_B50_eps5/{metrics,generalization}_{vc,v,grid}_dev30.json`; log `results/e10/`.
**Artifact:** HF `runs/dev300_B50_eps5.tar` ĐÃ upload 2026-10-07 (576 MB, sha256 cada93d5…18ba49; GHI ĐÈ bản 08f5e429… — tập cha: PNG dev30 của mọi ô cũ + 10 ô mới OSFD-K2/K4, VC-OSFD, VC4-OSFD, OSFD-V4/V8, SH32/SHODD/SHSUB + ~27 ảnh OSFD dev100 + dets_{vc,v,grid}_dev30). KHÔNG upload (sinh lại được vài phút bằng script): `artifacts/e10/` (dets E10a), `artifacts/e11/`, `artifacts/e12c/` (snapshot quỹ đạo). Máy mới: `sync_artifacts.py download dev300_B50_eps5` + `download e8_wbosfd_dev30`.
**Git:** repo ở `transfer-attack-final/` (lúc đầu phiên kiểm nhầm `/workspace` → tưởng không có git); commit + push cuối phiên.
**Không có tiến trình nào đang chạy.**

---

## 2026-10-08 — (a) Stride quyết định của "pha lưới": tịnh tiến bội 4 / 8 / 16 px — khóa TRƯỚC khi chạy (dev30)
**Lý do:** kiểm pha lưới (2026-10-07) ỦNG HỘ: SHODD (79.2) ≫ SH32 (68.9) ≈ NOISE (64.3). Chưa biết phần δ gắn lưới thuộc stride nào của R50 (C2–C5: 4 / 8 / 16 / 32; P6 của RPN: 64) → quyết định thiết kế method lấy mẫu pha (việc mở c).
**Ô (khung y hệt SH32 / SHODD: `geo="elem"`, loss feature OSFD k = 3, MI, 2 view / backward, σ = 6, feature sạch co-transform, B = 50, ε = 5):** `SHk`, k ∈ {4, 8, 16}: tx, ty rút độc lập, đều trên các bội của k trong [−64, 64] (gồm 0) — GIỮ pha mọi stride là ước của k, LỆCH pha stride > k. |t| TB (đo 200k mẫu): SH4 32.9, SH8 33.8, SH16 35.5 (SH32 38.3, SHODD 32.1) — biên độ ngang nhau. Code: `elem.shift_offset` mode `s4 / s8 / s16` (tổng quát `s32`; `s32` trùng tuyệt đối bản cũ kể cả RNG — đã kiểm 1000 mẫu).
**Mốc:** SINH LẠI SH32 và SHODD cùng lần (PNG cũ không có trên máy này, rẻ ~2.5 min / ô) → so cặp cùng lần. Sanity: CA3 sinh lại lệch số đã commit (SH32 68.9, SHODD 79.2) > 3 → ghi chú (GPU không tất định); quy tắc vẫn dùng bản sinh lại.
**Chỉ số:** CA3 = TB drop ConvNeXt-T / Swin-T / YOLOX-S. f(k) = [CA3(SHk) − CA3(SH32)] / [CA3(SHODD) − CA3(SH32)] = phần lợi ích còn giữ khi CHỈ lệch pha các stride > k.
**Quy tắc (mô tả, dev30; chỉ để chọn thiết kế cho (b) / (c), không claim):**
- Hợp lệ: CA3(SHODD) − CA3(SH32) ≥ 5 (tái lập kết quả 2026-10-07). Không đạt → không đọc stride, ghi nhận "không tái lập".
- **Stride quyết định k\*** = k nhỏ nhất ∈ {4, 8, 16} có f(k) < 0.5; không có → k\* = 32 (riêng lệch pha stride 32 / 64 gánh phần lớn lợi ích — C5 / P6). Diễn giải: giữ pha tới stride k\* làm mất phần lớn lợi ích → lệch pha ở stride k\* là cần. k\* = 4 → gắn lưới nông (stem / C2).
- **Xám** nếu không đơn điệu rõ: có k < k′ với f(k′) − f(k) > 0.3 → báo "không đơn điệu, trong nhiễu", không chốt k\*.
- Mô tả kèm: Swin-L, R101, CI bootstrap cặp (SHk − SH32) trên CA3 thành phần.
**Caveat khai báo trước:** khoảng SHODD − SH32 chỉ ~10 điểm CA3, nhiễu dev30 → f có sai số lớn (cỡ ±0.3).
**Chi phí:** 5 ô × 30 ảnh × ~4.9 s ≈ 12 min + eval Controlled + YOLOX-S / DINO-Swin-L ~10 min. Không cần HF (mốc sinh lại).
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods SH32 SHODD SH4 SH8 SH16 --out metrics_stride_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods SH32 SHODD SH4 SH8 SH16 --dets dets_stride_dev30.json --out generalization_stride_dev30.json
```

### Kết quả (a) stride quyết định @ dev30 (`metrics_stride_dev30.json`, `generalization_stride_dev30.json`, log `results/stride/stride_dev30.log`) — theo quy tắc: **k\* = 4** (đơn điệu, không xám)
**Sanity:** sinh lại SH32 CA3 68.9 (cũ 68.9), SHODD 79.5 (cũ 79.2) — tái lập. Hợp lệ: SHODD − SH32 = 10.6 ≥ 5.
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | f:
- SH32 98.7 / 92.5 / 69.2 / 72.0 / 65.5 / 16.0 | 68.9 | 0
- **SH4** 98.6 / 91.9 / 75.6 / 77.0 / 66.8 / 17.5 | 73.1 | **0.40**
- SH8 98.6 / 92.8 / 71.1 / 73.6 / 64.3 / 18.2 | 69.7 | 0.07
- SH16 98.5 / 91.7 / 67.6 / 71.0 / 61.1 / 13.8 | 66.6 | −0.22
- SHODD 97.5 / 92.0 / 81.0 / 81.6 / 75.9 / 21.9 | 79.5 | 1
CI cặp so SH32 [(ConvNeXt + Swin-T)/2 | YOLOX | Swin-L]: SH4 +5.7 [3.1, 9.7] | +1.3 [−4.8, 7.0] | +1.4 [−1.1, 5.5]; SH8 +1.8 [−1.8, 5.7] | −1.2 | +2.1; SH16 −1.3 | −4.4 [−12.1, −0.3] | −2.3; SHODD +10.7 [7.3, 16.6] | +10.4 [5.4, 16.7] | +5.8 [0.9, 11.1].
**Đọc (dev30, mô tả):** (1) f(4) = 0.40 < 0.5 → k\* = 4: chỉ cần GIỮ pha stride 2 / 4 đã mất ~60% lợi ích; lệch pha stride 8 góp phần còn lại nhỏ (SH4 > SH8, CI cặp CNN > 0), stride 16 / 32 / 64 ≈ 0 (SH8 ≈ SH16 ≈ SH32). → Phần δ đặc thù surrogate gắn với lưới NÔNG của R50 (conv1 stride 2 + maxpool → C2 stride 4, phần nào C3 stride 8), KHÔNG phải lưới sâu C5 / P6 như giả định khi chọn SH32. (2) Phần dư +4.6 của SH32 so NOISE (2026-10-07) không do P6 mà khả dĩ do biên / lấp 0 — chưa kiểm. (3) R101 không đổi (91.7–92.8) — khớp: cùng lưới nông. (4) YOLOX / Swin-L cùng chiều nhưng CI chứa 0 ở SH4.
**Hệ quả cho (b) / (c):** lấy mẫu pha có chủ đích chỉ cần phủ residue mod 4 (có thể mod 8), không cần mod 32; câu hỏi tách tiếp: stride 2 hay 4 (ô SH2 = dịch chẵn ±64). SHSUB (< 1 px, 70.4) yếu → lệch pha phải là nguyên pixel (hoặc nội suy làm mờ triệt tác dụng) — caveat.
**Ghi chú kỹ thuật:** máy mới locale ASCII → `eval_generalization.py` lỗi khi ghi JSON tiếng Việt; sửa `open(..., encoding="utf-8")`, chạy lại phần ghi (dets đã lưu, resume). Artifact chưa upload HF: PNG 5 ô + `dets_stride_dev30.json` + `gen_dets/` trong `artifacts/runs/dev300_B50_eps5/` — KHÔNG upload đè `runs/dev300_B50_eps5.tar` trước khi tải bản HF về và gộp (bản HF là tập cha).

---

## 2026-10-08 — Tách stride 2 vs 4: ô SH2 (dịch chẵn) — khóa TRƯỚC khi chạy (dev30)
**Lý do:** (a) cho k\* = 4 (f(4) = 0.40) nhưng chưa tách được pha stride 2 (conv1) với stride 4 (maxpool → C2).
**Ô:** `SH2`: tx, ty rút độc lập, đều trên số CHẴN trong [−64, 64] (65 giá trị, |t| TB 32.5) — giữ pha stride 2, lệch pha stride ≥ 4. Khung y hệt SH4 / SHODD. Code: mode `s2` của `elem.shift_offset` (`s32` vẫn trùng bản cũ — đã kiểm).
**Mốc:** PNG SH32 / SHODD / SH4 của lần (a) hôm nay (cùng máy), eval lại CÙNG lần với SH2 (`--out metrics_stride2_dev30.json`); f(k) như (a), span = CA3(SHODD) − CA3(SH32) của lần eval này.
**Quy tắc (mô tả, dev30):**
- f(2) ≥ 0.75 → **stride 4 quyết định**: giữ pha stride 2 vẫn được gần trọn lợi ích; phần gắn lưới nằm ở maxpool / C2.
- f(2) ≤ f(4) + 0.2 → **stride 2 quyết định**: lệch pha stride 4 không thêm gì; cần lệch pha lẻ pixel (conv1 stride 2).
- Ở giữa → **cả hai góp phần**, không chốt một stride.
Caveat: sai số f cỡ ±0.3 (dev30). Mô tả: Swin-L, R101, CI cặp SH2 − SH32 / SHODD − SH2.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods SH32 SHODD SH4 SH2 --out metrics_stride2_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods SH32 SHODD SH4 SH2 --dets dets_stride2_dev30.json --out generalization_stride2_dev30.json
```

### Kết quả SH2 @ dev30 (`metrics_stride2_dev30.json`, `generalization_stride2_dev30.json`, log `results/stride/stride2_dev30.log`) — theo quy tắc: **stride 4 quyết định** (f(2) = 0.86 ≥ 0.75)
Mốc eval lại cùng lần trùng tuyệt đối (a) (SH32 68.9, SH4 73.1, SHODD 79.5; span 10.6).
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: SH32 98.7 / 92.5 / 69.2 / 72.0 / 65.5 / 16.0 | 68.9; SH4 98.6 / 91.9 / 75.6 / 77.0 / 66.8 / 17.5 | 73.1; **SH2 98.3 / 91.4 / 81.3 / 78.9 / 74.0 / 24.0 | 78.0**; SHODD 97.5 / 92.0 / 81.0 / 81.6 / 75.9 / 21.9 | 79.5.
CI cặp [(ConvNeXt + Swin-T)/2 | YOLOX | Swin-L]: SH2 − SH32 +9.5 [6.1, 13.8] | +8.5 [−0.1, 15.8] | +8.0 [2.6, 13.2]; SHODD − SH2 +1.3 [−0.5, 4.1] | +2.0 [−2.5, 9.0] | −2.2 [−5.0, 1.9]; SH2 − SH4 +3.8 [0.9, 6.2] | +7.2 [0.3, 12.6] | +6.5 [0.8, 10.6].
**Đọc (dev30, mô tả):** (1) Giữ pha stride 2 (chỉ dịch chẵn) vẫn được ~86% lợi ích, SHODD − SH2 trong nhiễu → pha stride 2 (conv1) KHÔNG cần; phần δ gắn lưới nằm ở **stride 4 (maxpool → C2)**, cộng phần nhỏ stride 8 (từ (a)). (2) SH2 − SH4 dương ở cả 3 nhóm target, CI > 0 — chỉ khác nhau ở việc có residue 2 mod 4 hay không → bằng chứng trực tiếp nhất cho pha stride 4. (3) **Swin-L: SH2 24.0** — cao nhất trong các ô chỉ-tịnh-tiến, gần OSFD (27.2) — lệch pha stride 4 cũng là phần lợi ích hình học trên Swin-L (dev30, CI rộng). (4) R101 không đổi (91.4–92.5).
**Hệ quả method (c):** đa dạng cần thiết tối thiểu = phủ residue mod 4 của tịnh tiến nguyên (4 × 4 = 16 pha) — ứng viên "lấy mẫu pha stride 4 có chủ đích" thay augmentation ngẫu nhiên. Phải xác nhận dev100 (SH32 / SH4 / SH2 / SHODD) trước khi khóa method; rà literature "aliasing / downsampling phase / shift-invariance" trong transfer attack.

---

## 2026-10-08 — E13: δ có thành phần KHÓA LƯỚI chu kỳ 4 không — R50 (theo ô) vs WB-OSFD Swin-L? — khóa TRƯỚC khi chạy (CPU, PNG có sẵn, dev30)
**Lý do:** pha lưới stride 4 quyết định transfer (SH2 / SH4, 2026-10-08) — suy ra từ transfer, chưa thấy trực tiếp trên δ. Đồng thời user hỏi điểm chung δ WB-Swin-L vs δ transfer theo góc nhìn mới. **Sự thật đã kiểm:** DINO-Swin-L và Mask R-CNN dùng cùng `Resize(1333, 800, keep_ratio)`, pad phải / dưới → lưới stride 4 của R50 và lưới patch 4 của Swin-L TRÙNG pha trong khung model (loại giả thuyết "lệch pha do resize khác nhau").
**Đo (từng ảnh, khung model):** δ_m = imrescale(PNG adv) − imrescale(ảnh sạch) (cv2 bilinear như test pipeline, float); HP = δ_m − G₂ * δ_m (σ = 2, giữ 99% biên độ ở tần 1/4, bỏ rò tần thấp). Mẫu khóa lưới chu kỳ p: T_p[c, a, b] = mean HP[c, y ≡ a, x ≡ b (mod p)] − mean HP[c]. Z_p = (‖T_p‖² / p²) / (mean HP² / N_px) (≈ 1 nếu không có cấu trúc tuần hoàn chu kỳ p; chuẩn hóa theo số ô dư). **Đỉnh chu kỳ 4:** P4 = Z4 / mean(Z3, Z5, Z7).
Nguồn: WB-OSFD (HF `e8_wbosfd_dev30`); ô R50 từ HF `dev300_B50_eps5` (NOISE, OSFD, ROT15, PIA, C2 = `PROBE-osfd+cls`, MI chưa có ở dev30 → bỏ) và lần chạy hôm nay (`artifacts/local_dev300_20261008`: SH32, SH4, SH2, SHODD).
**Quy tắc (mô tả, dev30; trung bình ảnh, CI bootstrap ảnh 1000, BOOT_SEED 2026):**
- (i) **"δ R50 có thành phần khóa lưới 4"** — ỦNG HỘ nếu P4(SH32) ≥ 3 VÀ P4(NOISE) ≥ 3.
- (ii) **"đa dạng làm lệch pha stride 4 triệt thành phần khóa lưới" (kiểm trực tiếp cơ chế Q1)** — ỦNG HỘ nếu (i) VÀ Z4(SH4) ≥ 0.7·Z4(SH32) VÀ Z4(SH2) ≤ 0.5·Z4(SH32) VÀ Z4(SHODD) ≤ 0.5·Z4(SH32). Mô tả: Z4 của OSFD / ROT15 / PIA / C2 (dự đoán thấp), Spearman(Z4 ô, CA3) trên các ô.
- (iii) **"δ WB-Swin-L có khóa lưới 4 của Swin-L"** — ỦNG HỘ nếu P4(WB) ≥ 3.
- (iv) **Điểm chung mẫu khóa lưới** (chỉ đọc nếu (i) và (iii) ủng hộ): r = corr(T4_ô, T4_WB) theo ảnh (48 giá trị), trung bình ảnh; "mẫu khóa lưới R50 và Swin-L GIỐNG nhau" nếu r(SH32) ≥ 0.3 với cận dưới CI > 0; ≤ 0.1 → khác nhau (phần khóa lưới R50 không trúng lưới Swin-L dù cùng pha → giải thích một phần Swin-L thấp ở ô giữ pha). Ở giữa → xám.
Caveat: PNG uint8 (lượng tử 1 mức) + resize bilinear 1.6–1.9× làm mờ; nếu (i) không đạt có thể do đo không đủ nhạy, không bác cơ chế.
Script: `scripts/e13_grid_template.py`. Không tune attack nào theo E13.

### Kết quả E13 (`results/mechanism/e13_grid_template.json`, log `results/stride/e13.log`) — nguyên văn quy tắc: (i) / (ii) / (iii) KHÔNG đạt; **có lỗi chuẩn hóa trong entry khóa (phát hiện SAU khi thấy số)**
**Đính chính đo lường (khai báo rõ là sau kết quả):** entry khóa ghi Z_p "≈ 1 nếu không có cấu trúc"; thực tế với nhiễu trắng Z_p ≈ p² (mỗi ô dư có phương sai σ²p²/N, p² ô, chia p²) — khớp số đo (Z3 ≈ 9 ở mọi nguồn). Vì Z7 ≈ 49 × (tương quan) bị thổi phồng, P4 nguyên văn có mốc sai. Bản đúng ý định: Zn_p = Z_p / p², P4n = Zn4 / mean(Zn3, Zn5, Zn7). Báo cả hai; điều kiện (ii) dùng tỉ số Z4 giữa ô nên KHÔNG phụ thuộc chuẩn hóa. Quyết định dùng bản nào: user.
| nguồn | Z4 | P4 (nguyên văn) | **P4n** | r(T4, WB) | CA3 / Swin-L |
|---|---|---|---|---|---|
| WB-OSFD (Swin-L) | 23.9 | 0.32 | **0.69** [0.58, 0.80] | — | — |
| NOISE | 183.5 | 2.10 | **4.44** [3.12, 6.08] | 0.04 | 64.3 / 14.9 |
| SH32 | 175.0 | 1.71 | **3.60** [2.63, 4.74] | 0.00 | 68.9 / 16.0 |
| SH4 | 165.9 | 1.66 | **3.31** [2.50, 4.31] | 0.03 | 73.1 / 17.5 |
| SH2 | 32.6 | 0.40 | 0.82 | 0.07 | 78.0 / 24.0 |
| SHODD | 32.6 | 0.38 | 0.79 | 0.09 [0.02, 0.16] | 79.5 / 21.9 |
| OSFD | 26.9 | 0.36 | 0.77 | 0.01 | 78.9 / 27.2 |
| ROT15 | 25.9 | 0.31 | 0.66 | 0.10 [0.01, 0.18] | 79.4 / 23.7 |
| PIA | 25.1 | 0.30 | 0.63 | 0.02 | 79.6 / 28.6 |
| C2 | 26.0 | 0.41 | 0.86 | 0.06 | 78.0 / 25.6 |
Theo P4n: (i) ĐẠT (NOISE 4.4, SH32 3.6 ≥ 3); (ii) ĐẠT (Z4: SH4 = 0.95·SH32, SH2 = SHODD = 0.19·SH32); (iii) KHÔNG (WB 0.69) → (iv) không đọc. Spearman(Z4 ô, CA3) −0.78, (Z4, Swin-L) −0.88 trên 9 ô.
**Đọc (dev30, mô tả):** (1) **Thấy TRỰC TIẾP thành phần khóa lưới chu kỳ 4 trong δ R50** ở các ô giữ pha stride 4 (NOISE / SH32 / SH4: Z4 ~170, gấp ~6–7 lần), và nó biến mất ở MỌI ô lệch pha (SH2, SHODD, OSFD, ROT15, PIA, C2: Z4 25–33, ngang δ WB). Z3 / Z5 / Z7 gần như không đổi giữa hai nhóm → hiệu ứng đặc thù chu kỳ 4. Khớp hoàn toàn suy luận từ transfer (SH4 ≈ SH32, SH2 ≈ SHODD) → cơ chế Q1 có bằng chứng phía δ, độc lập với phía target. (2) **δ WB-Swin-L KHÔNG có khóa lưới 4** dù Swin-L có patch 4×4 cùng pha → "phần khóa lưới" là đặc thù R50 (maxpool / C2), không phải tính chất chung của model stride 4; điểm chung kiểu "cùng mẫu lưới" giữa R50 và Swin-L không tồn tại. (3) Sau khi triệt khóa lưới, mọi ô có Z4 ngang δ WB nhưng Swin-L vẫn chỉ 22–29 → khóa lưới giải thích phần NOISE → OSFD, KHÔNG giải thích khoảng 27 → 85 của Swin-L (Q2 vẫn mở). Caveat: dev30; chuẩn hóa đính chính sau kết quả.

---

## 2026-10-08 — Pilot method "pha lưới" @ dev30: PH4 (dịch 0–3 px) + PH4A (đối pha) — khóa TRƯỚC khi chạy
**Lý do:** cơ chế Q1 (SH2 / SH4 + E13): lợi ích đa dạng hình học = triệt thành phần δ khóa lưới stride 4 của R50. Mọi ô có hình học đều dịch xa (|t| ~33 px) → chưa tách "chỉ pha" với "pha + dời nội dung". Ứng viên method tối giản = đa dạng CHỈ ở pha mod 4.
**Literature (rà 2026-10-08):** chưa thấy công trình nối aliasing / pha lưới downsampling của surrogate với transferability (gần nhất: ASAP — aliasing ↔ độ bền; TIPS — shift-invariance kiến trúc). **Trùng ý một phần: TI-FGSM (Dong et al., CVPR 2019)** — tối ưu trên ảnh dịch ≤ 10 px, cài bằng làm mượt gradient (≈ S1, đã ÂM trên detection). → Novelty không nằm ở "dịch ảnh" mà ở cơ chế (pha stride 4, thấy trực tiếp E13) + tập pha tối thiểu / đối pha.
**Ô (khung `geo="elem"` y hệt SH*: loss feature OSFD k = 3, MI, 2 view / backward, σ = 6, feature sạch co-transform, B = 50, ε = 5):**
- `PH4`: tx, ty rút độc lập đều trong {0, 1, 2, 3} mỗi view (phủ 16 pha mod 4, dời ≤ 3 px).
- `PH4A` (đối pha): view 1 (a, b) đều trong {0..3}², view 2 = ((a + 2) mod 4, (b + 2) mod 4) — nửa chu kỳ lưới 4 → thành phần tần 1/4 khóa lưới của 2 view ngược dấu trong CÙNG 1 backward.
Unit test: warp dịch nguyên == dịch chính xác (≤ 2e-3), đối pha đúng (+2, +2), `s32` không đổi.
**Mốc (PNG HF `dev300_B50_eps5`, eval lại cùng lần):** OSFD, NOISE, SHODD (bản 2026-10-07), ROT15. `--out metrics_ph_dev30.json`.
**Quy tắc (dev30, mô tả; CA3 = TB ConvNeXt / Swin-T / YOLOX):**
- Kiểm thao tác (E13, `--extra PH4 PH4A`): Z4(ô) ≤ 0.5·Z4(NOISE). Trượt → ô đó không kiểm được cơ chế.
- **"Chỉ pha là đủ"** — ỦNG HỘ nếu CA3(PH4) ≥ CA3(SHODD) − 3; BÁC (cần cả dời nội dung) nếu CA3(PH4) ≤ CA3(NOISE) + ½·[CA3(SHODD) − CA3(NOISE)]; giữa → xám.
- **"Đối pha có chủ đích giúp"** — nếu CA3(PH4A) − CA3(PH4) ≥ 2.
- **GO sàng lọc dev300 (ứng viên method)** — ô tốt hơn trong PH4 / PH4A đạt ĐỒNG THỜI CA3 ≥ CA3(OSFD) + 2, Swin-L ≥ Swin-L(OSFD) − 3, R101 ≥ R101(OSFD) − 3 (điểm). Không đạt → không GO đứng riêng; nếu "chỉ pha là đủ" ủng hộ thì ghi nhận là RÚT GỌN tương đương (đơn giản hơn RRB), bước tiếp (vd ghép loss cls C2) khóa entry mới, user quyết.
Kỳ vọng khai báo trước: mọi đa dạng hình học hội tụ ~79 CA3 → khả năng cao PH ≈ OSFD, không vượt; giá trị chính là kiểm "chỉ pha".
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods OSFD NOISE SHODD ROT15 PH4 PH4A --out metrics_ph_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods OSFD NOISE SHODD ROT15 PH4 PH4A --dets dets_ph_dev30.json --out generalization_ph_dev30.json
python scripts/e13_grid_template.py --extra PH4 PH4A --out results/mechanism/e13_ph_dev30.json
```

### Kết quả pilot pha @ dev30 (`metrics_ph_dev30.json`, `generalization_ph_dev30.json`, `results/mechanism/e13_ph_dev30.json`, log `results/stride/ph_dev30.log`) — theo quy tắc: thao tác ĐẠT; "chỉ pha là đủ" **XÁM** (sát ngưỡng bác); đối pha KHÔNG giúp; **KHÔNG GO**
Mốc eval lại trùng tuyệt đối số cũ (OSFD 78.9, NOISE 64.3, SHODD 79.2, ROT15 79.4).
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | Z4 | s/ảnh: NOISE 98.7 / 89.8 / 67.2 / 68.4 / 57.4 / 14.9 | 64.3 | 183.5 | 4.3; OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9 | 26.9 | 3.0; SHODD 97.5 / 91.8 / 80.0 / 81.5 / 76.0 / 22.3 | 79.2 | 32.6; **PH4 98.3 / 92.1 / 76.7 / 73.2 / 67.3 / 20.2 | 72.4 | 33.0**; **PH4A 97.6 / 91.1 / 74.8 / 76.8 / 67.9 / 19.3 | 73.1 | 24.5**.
- Thao tác: Z4 PH4 33.0, PH4A 24.5 ≤ 91.8 → ĐẠT (khóa lưới bị triệt ngang SHODD / OSFD).
- Chỉ pha: CA3(PH4) 72.4 < SHODD − 3 (76.2) và > ngưỡng bác 71.8 → XÁM (sát bác). PH4 − SHODD: (ConvNeXt + Swin-T)/2 −5.8 [−8.4, −3.1], YOLOX −8.7 [−15.8, −1.4]; PH4 − NOISE +7.1 / +9.9 / Swin-L +5.2 (CI > 0).
- Đối pha: PH4A − PH4 +0.7 < 2 → không giúp.
- GO: KHÔNG (PH4A − OSFD: CNN −4.7 [−7.7, −0.4], Swin-L −7.9 [−10.8, −1.7]).
**Đọc (dev30):** (1) **Triệt khóa lưới stride 4 là CẦN nhưng KHÔNG ĐỦ.** PH4 / PH4A có Z4 thấp ngang SHODD / OSFD mà chỉ lấy ~54% lợi ích (NOISE → SHODD). Phân rã thô: dời xa giữ pha (SH32) +4.6, chỉ pha (PH4) +8.1, cả hai (SHODD) +14.9 → hai phần gần cộng tính, có chút cộng hưởng. Câu cơ chế "chỉ pha lưới stride 4" ở các entry trước bị NÓI QUÁ: phần còn lại đến từ dịch chuyển lớn. (2) Ứng viên giải thích phần còn lại (chưa kiểm): PH4 (0–3 px) chỉ phủ một nửa residue mod 8 và 1/8 residue mod 32 → pha các stride THÔ hơn (8 — đã thấy góp phần ở (a) — 16, 32) bị phủ thiếu; hoặc dời nội dung / biên thật sự có vai trò. Kiểm rẻ: PH8 (0–7) và PH32 (0–31) — nếu CA3 tăng dần về SHODD → "độ phủ pha nhiều stride" (vẫn là câu chuyện lưới); nếu PH32 vẫn thấp → dời nội dung. (3) Z4 không còn là chỉ báo đủ cho transfer (PH4 vs SHODD cùng Z4, khác 6.8 CA3) — đính chính cách đọc Spearman E13. (4) Method "chỉ dịch 0–3 px" thua OSFD → loại.

---

## 2026-10-08 — Phần còn thiếu của PH4: phủ pha stride THÔ hay DỜI NỘI DUNG? Ô PH8 / PH32 / PH4L — khóa TRƯỚC khi chạy (dev30)
**Ô (khung y hệt PH4):** `PH8` t ∈ {0..7}; `PH32` t ∈ {0..31}; `PH4L` t = 32·k + r, k ∈ {−2..2}, r ∈ {0..3} (mỗi trục độc lập). Độ phủ (200k mẫu) | |t| TB: PH4 4 pha mod 8, 4 pha mod 32 | 1.5; PH8 8 | 8 | 3.5; PH32 8 | 32 | 15.5; **PH4L 4 | 4 | 38.7** (= độ phủ PH4, dời ngang SH32); SHODD 8 | 31 | 32.1. `s32` không đổi (đã kiểm).
**Mốc (eval lại cùng lần):** NOISE, OSFD, SHODD, PH4 (PNG có sẵn trong run HF + pilot pha).
**Quy tắc (dev30; CA3 như trước):**
- **"Phần còn thiếu = DỜI NỘI DUNG"** — ỦNG HỘ nếu CA3(PH4L) ≥ CA3(SHODD) − 3 VÀ CA3(PH4L) − CA3(PH4) ≥ 3.
- **"Phần còn thiếu = PHỦ PHA stride thô"** — ỦNG HỘ nếu CA3(PH4L) ≤ CA3(PH4) + 2 VÀ CA3(PH32) ≥ CA3(SHODD) − 3.
- Còn lại → xám (cả hai góp phần / không cái nào). PH8: mô tả (đường liều theo độ phủ). Z4 (E13 `--extra`): mô tả.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods NOISE OSFD SHODD PH4 PH8 PH32 PH4L --out metrics_ph2_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods NOISE OSFD SHODD PH4 PH8 PH32 PH4L --dets dets_ph2_dev30.json --out generalization_ph2_dev30.json
python scripts/e13_grid_template.py --extra PH4 PH8 PH32 PH4L --out results/mechanism/e13_ph2_dev30.json
```

### Kết quả PH8 / PH32 / PH4L @ dev30 (`metrics_ph2_dev30.json`, `generalization_ph2_dev30.json`, `results/mechanism/e13_ph2_dev30.json`, log `results/stride/ph2_dev30.log`) — theo quy tắc: **XÁM** (cả hai nhánh không đạt hình thức; nhánh "phủ pha thô" trượt 0.01)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 (mốc eval lại trùng số cũ): NOISE 98.7 / 89.8 / 67.2 / 68.4 / 57.4 / 14.9 | 64.3; PH4 98.3 / 92.1 / 76.7 / 73.2 / 67.3 / 20.2 | 72.4; **PH8 98.3 / 92.8 / 81.5 / 80.7 / 74.2 / 21.2 | 78.8**; PH32 98.9 / 92.6 / 79.9 / 78.0 / 70.6 / 23.3 | 76.2; **PH4L 97.7 / 90.6 / 76.3 / 77.3 / 67.1 / 20.8 | 73.6**; SHODD 97.5 / 91.8 / 80.0 / 81.5 / 76.0 / 22.3 | 79.2; OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9.
- "Dời nội dung": PH4L − PH4 = +1.2 < 3 (CI cặp CNN +1.9 [−2.7, 3.9]); PH4L − SHODD −5.6 → KHÔNG đạt.
- "Phủ pha thô": PH4L ≤ PH4 + 2 ĐẠT (73.6 ≤ 74.4); PH32 ≥ SHODD − 3: 76.15 vs 76.16 → trượt 0.01 → không đạt hình thức → XÁM.
**Mô tả (KHÔNG thuộc quy tắc):** (1) **PH8 (dời TB 3.5 px, phủ đủ 8 pha mod 8) = 78.8 ≈ SHODD 79.2 ≈ OSFD 78.9** (PH8 − PH4: CNN +6.2 [2.6, 7.9], YOLOX +7.0 [1.1, 11.7]). (2) Dời xa thêm vào PH4 (PH4L) KHÔNG bù được: dời nội dung không phải phần thiếu. (3) Chu kỳ 8 trong δ (Zn8 = Z8 / 64, mô tả sau kết quả): PH4 5.8, PH4L 6.3 (còn khóa lưới 8) vs PH8 3.6, PH32 3.5, SHODD 3.5, OSFD 3.1, WB-Swin-L 3.5 — thành phần khóa lưới 8 còn đúng ở 2 ô transfer kém, khớp (a) (stride 8 góp ~40%: f(4) = 0.40).
**Đọc (dev30):** đính chính entry pilot pha — "cần dịch xa" SAI; PH4 thiếu vì chỉ phủ nửa pha stride 8. Câu cơ chế sửa: **lợi ích đa dạng hình học ≈ triệt thành phần δ khóa lưới stride 4 VÀ 8 của R50 (C2, C3); dịch nguyên 0–7 px phủ đủ 8 × 8 pha là đủ đạt mức OSFD**, không cần xoay / co giãn / dời xa. PH32 (76.2) hơi thấp hơn PH8 — trong nhiễu. Swin-L: mọi ô dịch 20–23 < OSFD 27 (RRB / xoay còn thêm gì đó cho Swin-L, hoặc nhiễu ±5).
**Hệ quả method:** PH8 = recipe đa dạng tối giản ngang OSFD trên CNN (không vượt). Ứng viên vượt OSFD phải thêm thành phần trực giao (vd loss cls C2: C2-PH8) — khóa entry mới, user quyết.

---

## 2026-10-08 — Đòn bẩy Swin-L của PIA có phải PHỐI CẢNH? PIA vs PIA-noP @ dev100 — khóa TRƯỚC khi chạy
**Bối cảnh (user, mục tiêu duy nhất: SOTA + novelty):** đa dạng hình học bão hòa trên CNN (~79 CA3); riêng Swin-L loại biến đổi còn tạo khác biệt (dev100: PIA − GEO-C0 +6.0 [2.3, 9.8], C2-PIA − C2 +5.8 [1.7, 9.1]). Giả thuyết "lệch pha lưới window Swin-L (12 patch = 48 px)" đề xuất trước đó bị BỎ trước khi chạy: lưới window của target cố định trên ảnh, augmentation phía surrogate không thể lệch pha nó. Thay bằng xác nhận thành phần: Giai đoạn A (dev30) cho phối cảnh là thành phần duy nhất dương trên Swin-L (+2.8, trong nhiễu).
**Ô:** `PIA` (sinh lại 70 ảnh dev100 \ dev30 cùng seed; 30 ảnh dev30 có PNG HF) và `PIA-noP` (PIA bỏ thành phần phối cảnh, như Giai đoạn A), dev100, B = 50, ε = 5. Sanity: Swin-L PIA sinh lại vs số commit dev100 28.5 — lệch > 3 → ghi chú.
**Quy tắc (chính: DINO-Swin-L, paired bootstrap dev100):** "phối cảnh là đòn bẩy Swin-L" — ỦNG HỘ nếu PIA − PIA-noP ≥ 3 VÀ cận dưới CI > 0; BÁC nếu điểm ≤ 0; còn lại xám. Mô tả: CA3, R101.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --methods PIA PIA-noP --out metrics_piap_dev100.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev100 --models yolox_s dino_swin_l --methods PIA PIA-noP --dets dets_piap_dev100.json --out generalization_piap_dev100.json
```

### Kết quả PIA vs PIA-noP @ dev100 (`metrics_piap_dev100.json`, `generalization_piap_dev100.json`, log `results/stride/piap_dev100.log`) — theo quy tắc: **XÁM**
Sanity: PIA sinh lại Swin-L 28.9 (commit 28.5) — khớp. R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: PIA 94.9 / 88.4 / 76.7 / 73.8 / 70.4 / 28.9 | 73.7; PIA-noP 94.4 / 87.2 / 74.6 / 73.8 / 70.1 / 26.1 | 72.8.
PIA − PIA-noP: Swin-L **+2.8 [−0.3, 4.6]** (< 3, CI chứa 0); YOLOX +0.3; (ConvNeXt + Swin-T)/2 +1.1 [−0.5, 3.1].
**Đọc:** phối cảnh đóng góp nhỏ cho Swin-L (cùng chiều dev30 +2.8, lặp lại đúng độ lớn) nhưng KHÔNG giải thích được PIA − RRB +6.0 trên Swin-L; bỏ phối cảnh PIA vẫn hơn GEO-C0 (22.4 @ dev100, số commit, khác lần sinh) ~+3.7. → Đòn bẩy Swin-L của PIA phân tán, không quy về một thành phần.

---

## 2026-10-08 — Sàng lọc C2-PIA @ dev300 (bước 3 plan_sota) — khóa TRƯỚC khi chạy
**Lý do (user: mục tiêu SOTA + novelty):** đa dạng hình học bão hòa trên CNN; ứng viên mạnh nhất đang có là C2-PIA (dev100: CA3 − OSFD +8.2 [5.4, 11.3]; Swin-L 37.2 vs 24.0), chưa từng sàng lọc dev300. Mục đích: QUYẾT ĐỊNH (có vượt OSFD chắc không), không giải thích; PNG / dets dev300 để phân tích sau nếu qua.
**Ô:** OSFD, C2 (`PROBE-osfd+cls`), C2-PIA — dev300 (PNG dev300 cũ đã mất → sinh lại cùng seed; dev30 / dev100 có sẵn được giữ). B = 50, ε = 5. Sanity: số sinh lại vs commit `metrics_step3_dev300` (OSFD R101 93.0 / CA3 73.6 / Swin-L 24.9; C2 96.6 / 78.3 / 31.8) — lệch CA3 > 3 → ghi chú.
**Quy tắc (giữ nguyên bước 3 plan_sota, CA3 = TB ConvNeXt-T / Swin-T / YOLOX-S):** C2-PIA QUA nếu ĐỒNG THỜI ΔCA3 (C2-PIA − OSFD) ≥ 3 VÀ cận dưới CI > 0; Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S; R101 ≥ R101(OSFD) − 3. Mô tả: C2-PIA − C2, Swin-L, runtime. QUA → GO @ n300 (khóa riêng, 1 lần). Không qua → ghi log, user quyết.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --methods OSFD PROBE-osfd+cls C2-PIA --out metrics_c2pia_dev300.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --models yolox_s dino_swin_l --methods OSFD PROBE-osfd+cls C2-PIA --dets dets_c2pia_dev300.json --out generalization_c2pia_dev300.json
```

### Kết quả sàng lọc C2-PIA @ dev300 (`metrics_c2pia_dev300.json`, `generalization_c2pia_dev300.json`, log `results/stride/c2pia_dev300.log`) — theo quy tắc: **QUA**
Sanity: OSFD CA3 73.5 (commit 73.6), C2 78.4 (78.3) — khớp.
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD 98.1 / 93.1 / 77.5 / 74.6 / 68.4 / 24.9 | 73.5 | 3.2; C2 99.5 / 96.6 / 85.2 / 79.6 / 70.3 / 31.4 | 78.4 | 8.0; **C2-PIA 99.1 / 96.3 / 88.6 / 86.8 / 78.9 / 37.6 | 84.8 | 10.6**.
- **ΔCA3 (C2-PIA − OSFD) = +11.2 [8.1, 12.7]** (paired bootstrap 1000, BOOT_SEED 2026, tính lại từ dets vì file chỉ lưu CI thành phần) ≥ 3, CI > 0 ✓; ConvNeXt +11.1, Swin-T +12.1, YOLOX +10.5 > 0 ✓; R101 +3.2 ≥ −3 ✓.
- Mô tả: C2-PIA − C2: (ConvNeXt + Swin-T)/2 +5.3 [3.3, 7.3], YOLOX +8.6 [6.0, 11.3], Swin-L +6.2 [3.8, 8.4]; Swin-L − OSFD +12.8 [8.8, 14.8]. Runtime 3.3× OSFD (B = 50 như nhau; PIA thêm forward no_grad cho feature sạch co-transform).
**Bước tiếp theo theo plan_sota:** GO @ n300 (khóa riêng, chạy 1 lần). Caveat novelty: C2-PIA = OSFD + loss cls + PIA (PIA là công trình 2026) — chưa có thành phần mới; user quyết hướng novelty.

---

## 2026-10-08 — C2-PH8 @ dev30: loss cls cần biến đổi MẠNH hay chỉ cần triệt khóa lưới? — khóa TRƯỚC khi chạy
**Lý do (user: hướng novelty từ C2-PIA):** số dev100 có sẵn (`metrics_pia_dev100`) cho thấy TƯƠNG TÁC loss × đa dạng: với loss feature PIA − RRB = −0.6 CA3 (74.3 → 73.7), với loss feature + cls C2-PIA − C2 = +6.0 (76.6 → 82.6). Câu hỏi: lợi của PIA khi có loss cls là do triệt khóa lưới (như loss feature) hay do biến đổi mạnh triệt một thành phần đặc thù surrogate KHÁC của loss cls.
**Ô:** `C2-PH8` = khung C2-PIA (feature OSFD co-transform + 4·(loss_rpn_cls + loss_cls), GT co-transform, 1 forward / view, MI, σ = 6, B = 50) nhưng phép biến đổi = PH8 (dịch nguyên 0–7 px). Code: `elem_views(data_sample=)` co-transform GT qua `warp_boxes` / `gt_with_keep` như `pia_views`. Unit test: box / mask / ảnh dịch đúng t; 3 bước L∞ = 3, hữu hạn.
**Mốc (PNG dev30 có sẵn, eval lại cùng lần):** C2 (`PROBE-osfd+cls`, RRB), C2-PIA, PH8, OSFD.
**Quy tắc (dev30, CA3):** r = [CA3(C2-PH8) − CA3(C2)] / [CA3(C2-PIA) − CA3(C2)]. Hợp lệ nếu mẫu số ≥ 3 (dev30). r ≥ 0.75 → **"chỉ cần triệt khóa lưới"** (method gọn: C2 + PH8); r ≤ 0.25 → **"loss cls cần biến đổi mạnh"** (thành phần đặc thù surrogate thứ hai — hướng đào tiếp); giữa → xám. Mô tả: Swin-L (tỉ số tương tự), R101, runtime.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods OSFD PROBE-osfd+cls C2-PIA PH8 C2-PH8 --out metrics_c2ph8_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods OSFD PROBE-osfd+cls C2-PIA PH8 C2-PH8 --dets dets_c2ph8_dev30.json --out generalization_c2ph8_dev30.json
```

### Kết quả C2-PH8 @ dev30 (`metrics_c2ph8_dev30.json`, `generalization_c2ph8_dev30.json`, log `results/stride/c2ph8_dev30.log`) — theo quy tắc: **"loss cls cần biến đổi MẠNH"** (r = −0.32 ≤ 0.25; mẫu số 5.8 ≥ 3, hợp lệ)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD 97.8 / 93.7 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9 | 3.0; PH8 98.3 / 92.8 / 81.6 / 80.7 / 74.2 / 21.2 | 78.8; C2 99.9 / 94.8 / 82.9 / 84.9 / 66.2 / 25.6 | 78.0 | 7.4; **C2-PH8 99.9 / 94.4 / 82.4 / 81.7 / 64.2 / 26.6 | 76.1** | 12.2; C2-PIA 99.4 / 94.7 / 83.1 / 88.0 / 80.4 / 29.3 | 83.8 | 9.5.
C2-PIA − C2-PH8: (ConvNeXt + Swin-T)/2 +3.5 [1.5, 10.1], **YOLOX +16.2 [6.4, 23.4]**, Swin-L +2.7 [−1.9, 7.3]. C2-PH8 − C2: −1.8 / −2.0 / +1.0 (trong nhiễu). Swin-L r = 0.26.
**Đọc (dev30):** (1) Triệt khóa lưới (PH8) đủ cho loss feature (PH8 ≈ OSFD) nhưng KHÔNG đủ khi có loss cls: C2-PH8 ≈ C2 (RRB). → Loss cls tạo một thành phần đặc thù surrogate THỨ HAI, khác khóa lưới, chỉ biến đổi mạnh kiểu PIA mới triệt. (2) Dấu vết rõ nhất ở YOLOX: thêm loss cls với đa dạng yếu (RRB / PH8) làm YOLOX TỤT dưới OSFD (64–66 vs 75.6); PIA kéo lên 80.4. Tức thành phần thứ hai "overfit" head / phân loại của Mask R-CNN (RPN + RoI) — không chuyển sang head một tầng YOLOX. (3) Hướng novelty: định danh thành phần đặc thù của loss mức object (cls) và cách triệt có chủ đích thay PIA. Caveat: dev30.

---

## 2026-10-08 — Thành phần thứ hai của loss cls = KHÓA ANCHOR / RoI? C2-SCL25 / C2-ANI25 @ dev30 — khóa TRƯỚC khi chạy
**Giả thuyết:** loss cls (RPN + RoI) tính trên anchor / RoI có kích thước, tỉ lệ cố định → δ khớp đặc thù với kích thước / tỉ lệ box so với anchor của Mask R-CNN; PIA đổi kích thước + tỉ lệ box mỗi view nên triệt; PH8 chỉ dịch nên không.
**Ô (khung C2-PH8, GT co-transform qua `warp_boxes`):** `C2-SCL25` (co giãn đẳng hướng quanh tâm, ln s ~ U(±0.25)), `C2-ANI25` (lệch tỉ lệ, ln k ~ U(±0.25)) — cùng liều ô elem SCL25 / ANI25. Smoke test: box biến đổi theo view, 3 bước L∞ = 3, hữu hạn.
**Mốc (PNG dev30 có sẵn, eval lại cùng lần):** C2, C2-PH8, C2-PIA.
**Quy tắc (dev30, CA3):** r_X = [CA3(C2-X) − CA3(C2)] / [CA3(C2-PIA) − CA3(C2)]. **ĐỊNH DANH (khóa anchor / RoI ủng hộ)** nếu max(r_SCL, r_ANI) ≥ 0.75 — thành phần = kích thước (SCL) hoặc tỉ lệ (ANI) tùy ô đạt. **BÁC** nếu cả hai r ≤ 0.25 (đổi kích thước / tỉ lệ đơn lẻ không đủ). Còn lại xám. Mô tả: YOLOX (dấu vết chính ở C2-PH8), Swin-L.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods PROBE-osfd+cls C2-PH8 C2-PIA C2-SCL25 C2-ANI25 --out metrics_c2anchor_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods PROBE-osfd+cls C2-PH8 C2-PIA C2-SCL25 C2-ANI25 --dets dets_c2anchor_dev30.json --out generalization_c2anchor_dev30.json
```

### Kết quả C2-SCL25 / C2-ANI25 @ dev30 (`metrics_c2anchor_dev30.json`, `generalization_c2anchor_dev30.json`, log `results/stride/c2anchor_dev30.log`) — theo quy tắc: **ĐỊNH DANH — KÍCH THƯỚC** (r_SCL = 0.81 ≥ 0.75)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: C2 99.9 / 94.8 / 82.9 / 84.9 / 66.2 / 25.6 | 78.0; C2-PH8 … / 64.2 / 26.6 | 76.1; **C2-SCL25 99.9 / 95.5 / 89.0 / 84.1 / 75.1 / 28.0 | 82.7**; C2-ANI25 99.9 / 95.8 / 82.7 / 81.9 / 73.3 / 26.8 | 79.3; C2-PIA 99.4 / 94.8 / 83.1 / 88.0 / 80.4 / 29.3 | 83.8. Mẫu số 5.8.
r: SCL 0.81, ANI 0.22, PH8 −0.32; riêng YOLOX: SCL 0.63, ANI 0.50, PH8 −0.14.
**Đọc (dev30):** đổi KÍCH THƯỚC box (co giãn đẳng hướng) lấy lại ~80% lợi ích PIA khi có loss cls; đổi tỉ lệ đơn lẻ ít hơn nhiều; dịch không có tác dụng. Ủng hộ: thành phần đặc thù thứ hai của loss cls khóa theo THANG KÍCH THƯỚC (anchor scale / mức FPN / RoI) của Mask R-CNN. Lưu ý: không có tác dụng với loss feature (SCL25 ô elem cũ ≈ các ô khác) → đặc thù cho loss mức object. Caveat: dev30, chưa CI; co giãn quanh tâm còn kèm cắt biên / lấp 0.
**Hướng method (chưa khóa):** đa dạng KÍCH THƯỚC có chủ đích theo object (vd lấy mẫu scale phủ đều qua ranh giới mức FPN / anchor) thay PIA; kiểm liều–đáp ứng + xác nhận dev100 trước.

---

## 2026-10-08 — Method "đa dạng TÁCH CƠ CHẾ" (kích thước octave + pha PH8) @ dev30 — khóa TRƯỚC khi chạy
**Cơ sở:** hai thành phần δ đặc thù surrogate đã định danh (dev30): (1) khóa lưới stride 4 / 8 (loss feature) — triệt bằng dịch phủ pha 0–7 px; (2) khóa thang KÍCH THƯỚC (loss cls) — triệt bằng co giãn (C2-SCL25 r = 0.81). Ranh giới mức FPN / anchor Mask R-CNN cách nhau hệ số 2 → liều có cơ sở: phủ đúng 1 octave, ln s ∈ [−ln2/2, ln2/2].
**Ứng viên (≤ 3, khung C2: feature OSFD co-transform + 4·cls, GT co-transform, MI, σ = 6, 2 view / backward, B = 50):**
- `C2-SCL50`: chỉ co giãn, ln s ~ U(±0.5) (liều–đáp ứng so C2-SCL25).
- `C2-OCT`: co giãn ln s ~ U(±ln2/2) ∘ dịch pha PH8 (mỗi view độc lập).
- `C2-OCTS`: như OCT nhưng phân tầng: view 2 lệch nửa khoảng (ln2/2) so với view 1 trong cùng backward.
Unit test: hợp thành H = Lin · Shift đúng; phân tầng |u₁ − u₂| = a, trong khoảng; `s32` không đổi; 3 bước L∞ = 3, hữu hạn.
**Mốc (PNG dev30 có sẵn, eval lại):** C2, C2-SCL25, C2-PIA.
**Quy tắc (dev30, CA3):**
- **GO dev100** nếu ô tốt hơn trong {C2-OCT, C2-OCTS} đạt CA3 ≥ CA3(C2-PIA) − 1 VÀ Swin-L ≥ Swin-L(C2-PIA) − 3. Tín hiệu "vượt C2-PIA" nếu CA3 ≥ C2-PIA + 2.
- Liều: C2-SCL50 − C2-SCL25 (mô tả). Phân tầng có chủ đích giúp nếu OCTS − OCT ≥ 2.
- Không GO → ghi log, user quyết. Báo runtime.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods PROBE-osfd+cls C2-SCL25 C2-PIA C2-SCL50 C2-OCT C2-OCTS --out metrics_oct_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods PROBE-osfd+cls C2-SCL25 C2-PIA C2-SCL50 C2-OCT C2-OCTS --dets dets_oct_dev30.json --out generalization_oct_dev30.json
```

### Kết quả method tách cơ chế @ dev30 (`metrics_oct_dev30.json`, `generalization_oct_dev30.json`, log `results/stride/oct_dev30.log`) — theo quy tắc: **GO dev100 (C2-OCT)**; KHÔNG có tín hiệu vượt C2-PIA; phân tầng không giúp
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2 99.9 / 94.8 / 82.9 / 84.9 / 66.2 / 25.6 | 78.0 | 7.4; C2-SCL25 … | 82.7; C2-SCL50 99.0 / 93.8 / 85.9 / 84.6 / 78.2 / 29.2 | 82.9 | 11.5; **C2-OCT 99.9 / 95.7 / 85.3 / 86.6 / 78.9 / 31.9 | 83.6** | 11.6; C2-OCTS 99.6 / 95.3 / 88.9 / 85.3 / 76.3 / 27.9 | 83.5 | 11.5; C2-PIA 99.4 / 94.8 / 83.1 / 88.0 / 80.4 / 29.3 | 83.8 | 9.5.
- GO: C2-OCT CA3 83.6 ≥ 82.8 ✓, Swin-L 31.9 ≥ 26.3 ✓. Vượt (≥ +2): không (−0.2). C2-OCT − C2-PIA: CNN +0.4 [−2.4, 3.1], YOLOX −1.5 [−5.6, 4.1], Swin-L +2.6 [−2.1, 7.4].
- Liều SCL50 − SCL25 +0.2 (bão hòa từ ±0.25). Phân tầng OCTS − OCT −0.1 → không giúp.
**Đọc (dev30):** đa dạng tách cơ chế (chỉ co giãn 1 octave + dịch pha) TÁI TẠO trọn lợi ích C2-PIA (không cần phối cảnh / homography), nhưng không vượt. Runtime ~ C2-PIA (đều bị chi phối bởi loss cls + forward feature sạch co-transform).

---

## 2026-10-08 — Tối ưu tốc độ C2-OCT: skip_mask (tương đương) + AMP fp16 → ô `C2-OCT-fast` @ dev30 — khóa TRƯỚC khi chạy
**Đo (3 ảnh dev30, 50 bước, RTX 3090):** C2-OCT ~0.23 s / bước; forward: backbone 26%, RPN 18%, mask head 6%, FPN 4%, bbox head 3%, chép GT + mask 2%; còn lại phần lớn là backward. Thời gian chủ yếu do loss cls (`model.loss` đầy đủ: RPN proposals + RoI + mask) và forward feature sạch co-transform, không do phép biến đổi.
**skip_mask** (`geo_osfd_attack(skip_mask=True)`): loss C2 chỉ dùng loss_rpn_cls + loss_cls → tắt mask head + bỏ co-transform mask GT; RoI sampling / RNG không đổi. Kiểm tương đương (3 ảnh): % pixel δ khác gốc-vs-skip 18.3 / 24.8 / 17.5 ≈ gốc-vs-gốc (GPU không tất định) 17.7 / 24.9 / 16.3 → TƯƠNG ĐƯƠNG. Nhanh hơn chỉ 0–10%.
**AMP fp16** (`amp=True`, autocast forward / backward; δ, momentum, bước sign fp32; bf16 không dùng được — RoIAlign mmcv không hỗ trợ): nhanh thêm ×1.30–1.36 trên skip_mask, hữu hạn, độ bão hòa δ trùng fp32 (0.92–0.93). Gộp: ~7.5 s / ảnh vs ~11 s (×~1.45). ĐỔI số học → phải kiểm bằng eval.
**Quy tắc (dev30; mốc: PNG C2-OCT có sẵn, eval lại cùng lần):** `C2-OCT-fast` THAY THẾ ĐƯỢC C2-OCT nếu ĐỒNG THỜI |CA3(fast) − CA3(C2-OCT)| ≤ 2, YOLOX ≥ C2-OCT − 3, Swin-L ≥ C2-OCT − 3 (điểm), và runtime ≤ 0.8 × C2-OCT. Không đạt → giữ fp32 (skip_mask vẫn dùng được vì tương đương).
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods C2-OCT C2-OCT-fast --out metrics_octfast_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT C2-OCT-fast --dets dets_octfast_dev30.json --out generalization_octfast_dev30.json
```

### Kết quả C2-OCT-fast @ dev30 (`metrics_octfast_dev30.json`, `generalization_octfast_dev30.json`, log `results/stride/octfast_dev30.log`) — theo quy tắc: **THAY THẾ ĐƯỢC**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT 99.9 / 95.7 / 85.3 / 86.6 / 78.9 / 31.9 | 83.6 | 11.6; **C2-OCT-fast 99.5 / 94.7 / 86.4 / 86.1 / 77.4 / 29.2 | 83.3 | 8.0**. ΔCA3 −0.3, YOLOX −1.5, Swin-L −2.7 (trong ngưỡng), runtime ×0.69 (2.5× OSFD thay vì 3.7×). Từ đây dùng C2-OCT-fast cho các lần chạy C2-OCT tiếp (ghi rõ trong bảng: skip_mask + AMP fp16).

---

## 2026-10-08 — CLS-OCT @ dev30: bỏ loss feature, chỉ loss cls + đa dạng OCT — khóa TRƯỚC khi chạy
**Lý do:** dev100 (RRB): chỉ cls YOLOX 59.1 / Swin-L 32.5 / CA3 71.2 vs OSFD 70.2 / 25.3 / 74.5 — nhưng RRB chưa triệt thành phần khóa kích thước của loss cls (C2-PH8 / C2-SCL25). Với OCT, loss cls đứng riêng có thể đủ (bỏ loss feature → đơn giản + nhanh hơn).
**Ô:** `CLS-OCT-fast` = C2-OCT-fast bỏ loss feature (`feature=False`: loss = 4·(loss_rpn_cls + loss_cls), không forward feature sạch co-transform). Smoke: 3 bước L∞ = 3, forward_no_grad = 0, mask head khôi phục.
**Mốc (PNG dev30 có sẵn, eval lại):** C2-OCT-fast, OSFD.
**Quy tắc (dev30):** "loss feature THỪA (bỏ được)" nếu CA3(CLS-OCT) ≥ CA3(C2-OCT-fast) − 1 VÀ YOLOX ≥ YOLOX(C2-OCT-fast) − 3; "loss feature CẦN" nếu CA3(CLS-OCT) ≤ CA3(C2-OCT-fast) − 3; còn lại xám. Mô tả: Swin-L, runtime.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods OSFD C2-OCT-fast CLS-OCT-fast --out metrics_clsoct_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods OSFD C2-OCT-fast CLS-OCT-fast --dets dets_clsoct_dev30.json --out generalization_clsoct_dev30.json
```

### Kết quả CLS-OCT @ dev30 (`metrics_clsoct_dev30.json`, `generalization_clsoct_dev30.json`, log `results/stride/clsoct_dev30.log`) — theo quy tắc: **loss feature CẦN**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9 | 3.0; C2-OCT-fast 99.5 / 94.7 / 86.4 / 86.1 / 77.4 / 29.2 | 83.3 | 8.0; **CLS-OCT-fast 98.4 / 93.5 / 85.4 / 82.9 / 70.0 / 30.8 | 79.4 | 6.4**.
CA3 −3.9 so C2-OCT-fast (≤ −3 → cần); YOLOX −7.4 (giống dev100 RRB: chỉ cls yếu ở YOLOX); Swin-L +1.6 (trong nhiễu). OCT có kéo cls-only lên (YOLOX 70 vs 59 ở RRB dev100, khác tập ảnh) nhưng không thay được loss feature. Bổ trợ feature + cls được giữ.

---

## 2026-10-08 — RPN-OCT @ dev30: số hạng cls chỉ là loss_rpn_cls (objectness) — khóa TRƯỚC khi chạy
**Nguyên tắc (user, từ đây áp dụng):** mục tiêu = transfer sang target CHƯA THẤY; không tune theo 1 target / mẹo riêng surrogate. Quyết định chỉ dựa chỉ số GỘP (CA3 + Swin-L + YOLOX cùng lúc, không tối ưu riêng model nào). Đề xuất (chờ user chốt): giữ YOLOX-L + Mask R-CNN Swin-S làm target HELD-OUT, không nhìn khi ra quyết định, chỉ dùng bảng cuối.
**Lý do ô:** (a) hiểu cơ chế — lợi của loss cls đến từ objectness (RPN) hay phân loại 80 lớp (RoI); objectness là khái niệm chung mọi detector → ít gắn surrogate hơn; (b) giảm thời gian. Đo: `loss_rpn_cls` rpn-only = model.loss (0.1299 = 0.1299); thời gian 50 bước chỉ ×0.85–0.94 so C2-OCT-fast (gán anchor RPN vẫn tốn).
**Ô:** `RPN-OCT-fast` = C2-OCT-fast với số hạng cls = 4·loss_rpn_cls (`cls_mode="rpn"`, `repair.cls_term`).
**Mốc (PNG dev30 có sẵn):** C2-OCT-fast, OSFD.
**Quy tắc (dev30):** "objectness đủ (thay thế được)" nếu ĐỒNG THỜI CA3 ≥ C2-OCT-fast − 1, YOLOX ≥ −3, Swin-L ≥ −3 (điểm, so C2-OCT-fast). "RoI cls cần" nếu CA3 ≤ C2-OCT-fast − 3 HOẶC Swin-L ≤ C2-OCT-fast − 5. Còn lại xám.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods OSFD C2-OCT-fast RPN-OCT-fast --out metrics_rpnoct_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods OSFD C2-OCT-fast RPN-OCT-fast --dets dets_rpnoct_dev30.json --out generalization_rpnoct_dev30.json
```

### Kết quả RPN-OCT @ dev30 (`metrics_rpnoct_dev30.json`, `generalization_rpnoct_dev30.json`, log `results/stride/rpnoct_dev30.log`) — theo quy tắc: **RoI cls CẦN**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9 | 3.0; C2-OCT-fast 99.5 / 94.7 / 86.4 / 86.1 / 77.4 / 29.2 | 83.3 | 8.0; **RPN-OCT-fast 97.8 / 93.7 / 78.3 / 78.2 / 75.6 / 28.0 | 77.4** | 7.1.
CA3 −5.9 so C2-OCT-fast, ≈ OSFD (−1.5): objectness RPN gần như không thêm gì so loss feature; lợi của loss cls nằm ở phân loại RoI (80 lớp). Tốc độ chỉ ×0.89. → Giữ C2 (rpn_cls + RoI cls).

**Chốt (user, 2026-10-08):** YOLOX-L + Mask R-CNN Swin-S là target HELD-OUT — không eval / không nhìn khi ra quyết định method; chỉ dùng ở bảng cuối (n300 / n1000).

---

## 2026-10-08 — C2-LVX @ dev30: đa dạng VƯỢT MỨC FPN theo object (ứng viên novelty) — khóa TRƯỚC khi chạy
**Giả thuyết:** thành phần khóa kích thước của loss cls = khóa theo MỨC FPN mà RoI được gán (SingleRoIExtractor: mức l khi 56·2^l ≤ √(wh) < 56·2^(l+1), ranh giới √(wh) = 112 / 224 / 448). Đa dạng có chủ đích: mỗi view chọn 1 object GT ngẫu nhiên, chọn scale đẳng hướng để object SANG phía kia ranh giới mức gần nhất, cách ranh giới U(0, 0.5) octave, |Δ| ≤ 1 octave (`elem.lvx_log_scale`), ∘ dịch pha PH8.
**Thao tác (dev30, đo trước, 200 view / ảnh):** object chọn đổi mức FPN 72% view (OCT: 13%); |Δ| TB 0.71 octave (OCT ~0.25). Lẫn biên độ đã biết: liều SCL50 ≈ SCL25 (bão hòa) → biên độ ít khả năng là yếu tố. Smoke: L∞ = 3 sau 3 bước, hữu hạn.
**Ô:** `C2-LVX-fast` (khung C2-OCT-fast: skip_mask + AMP). **Mốc (PNG dev30 có sẵn):** OSFD, C2-OCT-fast, C2-PIA.
**Quy tắc (dev30, chỉ số GỘP — nguyên tắc target chưa thấy):** GO dev100 (method) nếu CA3(LVX) ≥ CA3(C2-OCT-fast) + 2 VÀ YOLOX ≥ C2-OCT-fast − 2 VÀ Swin-L ≥ C2-OCT-fast − 2. "Vượt mức FPN KHÔNG phải yếu tố thêm" nếu CA3(LVX) ≤ CA3(C2-OCT-fast) + 0.5. Giữa → xám. Mô tả: so C2-PIA, runtime.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods OSFD C2-OCT-fast C2-PIA C2-LVX-fast --out metrics_lvx_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods OSFD C2-OCT-fast C2-PIA C2-LVX-fast --dets dets_lvx_dev30.json --out generalization_lvx_dev30.json
```

### Kết quả C2-LVX @ dev30 (`metrics_lvx_dev30.json`, `generalization_lvx_dev30.json`, log `results/stride/lvx_dev30.log`) — theo quy tắc: **KHÔNG GO; vượt mức FPN KHÔNG phải yếu tố thêm**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9; C2-OCT-fast 99.5 / 94.7 / 86.4 / 86.1 / 77.4 / 29.2 | 83.3; C2-PIA 99.4 / 94.8 / 83.1 / 88.0 / 80.4 / 29.3 | 83.8; **C2-LVX-fast 92.1 / 88.0 / 80.8 / 79.3 / 81.6 / 24.5 | 80.5**.
ΔCA3 −2.7 so C2-OCT-fast. White-box R50 tụt 99.5 → 92.1, R101 94.7 → 88.0 → δ YẾU đi (co giãn lớn TB 0.71 octave làm loãng gradient); riêng YOLOX +4.2.
**Đọc:** đẩy object qua ranh giới mức FPN có chủ đích không giúp; OCT chỉ đổi mức 13% view mà đã đủ → thành phần khóa kích thước KHÔNG phải "khóa theo mức FPN của RoI" (hoặc lợi ích bị lấn bởi δ yếu đi). Bỏ hướng LVX.

---

## 2026-10-08 — Dịch + XOAY (PH8R, C2-OCTR-fast) và OCT thuần loss feature vs OSFD @ dev30 — khóa TRƯỚC khi chạy
**Câu hỏi (user):** (1) xoay có thêm gì khi đã có dịch phủ pha? (2) OCT (co giãn 1 octave + dịch 0–7 px) với loss feature thuần có ngang OSFD?
**Ô (khung `geo="elem"`):** `PH8R` = loss feature, xoay U(±15°) quanh tâm ∘ dịch PH8; `OCT` = loss feature, co giãn ln s ~ U(±ln2/2) ∘ PH8; `C2-OCTR-fast` = C2-OCT-fast + xoay U(±15°). Hợp thành H = Lin(rot · scale) · Shift (unit test đạt); smoke 3 bước L∞ = 3. Kèm tùy chọn `--surrogate` của run_baselines (chưa dùng; mặc định r50, không đổi hành vi).
**Mốc (PNG dev30 có sẵn, eval lại):** OSFD, PH8, ROT15, C2-OCT-fast.
**Quy tắc (dev30, chỉ số gộp CA3; YOLOX / Swin-L không tụt quá 2):**
- Xoay thêm vào dịch (loss feature): PH8R − PH8 ≥ 2 → "xoay có thêm"; ≤ 0.5 → "không thêm"; giữa xám. Tương tự C2-OCTR-fast − C2-OCT-fast.
- OCT vs OSFD: |ΔCA3| ≤ 2 → "ngang"; OCT − OSFD ≥ 2 → "hơn"; ≤ −2 → "kém".
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods OSFD PH8 ROT15 C2-OCT-fast PH8R OCT C2-OCTR-fast --out metrics_rot_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods OSFD PH8 ROT15 C2-OCT-fast PH8R OCT C2-OCTR-fast --dets dets_rot_dev30.json --out generalization_rot_dev30.json
```

### Kết quả dịch + xoay / OCT thuần feature @ dev30 (`metrics_rot_dev30.json`, `generalization_rot_dev30.json`, log `results/stride/rot_dev30.log`) — theo quy tắc: PH8R − PH8 **xám** (+1.2); C2-OCTR − C2-OCT **không thêm** (−0.9); **OCT − OSFD = +4.8 → HƠN**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9 | 3.0; ROT15 … | 79.4; PH8 98.3 / 92.8 / 81.5 / 80.7 / 74.2 / 21.2 | 78.8; PH8R 97.3 / 90.7 / 82.3 / 80.6 / 77.3 / 26.7 | 80.0 | 5.8; **OCT 96.6 / 91.9 / 85.0 / 83.9 / 82.1 / 27.7 | 83.6 | 5.5**; C2-OCT-fast 99.5 / 94.7 / 86.4 / 86.1 / 77.4 / 29.2 | 83.3 | 8.0; C2-OCTR-fast 99.0 / 93.8 / 86.3 / 87.9 / 72.8 / 32.1 | 82.3 | 8.3.
**Đọc (dev30):** (1) Xoay thêm vào dịch: không / ít (trong nhiễu). (2) **Đính chính:** entry C2-SCL25 ghi "co giãn không có tác dụng với loss feature" là SAI — ô elem SCL25 (feature thuần, chỉ co giãn) đã là ô cao nhất (81.0, +2.1 vs OSFD), và co giãn 1 octave + dịch pha với loss feature thuần (OCT) đạt 83.6 = C2-OCT-fast, YOLOX 82.1 (cao nhất mọi ô). → Đa dạng KÍCH THƯỚC + PHA giúp cả loss feature; với đa dạng này, loss cls gần như không thêm CA3 (−0.3), chỉ Swin-L +1.5 (nhiễu). Chi phí OCT 5.5 s vs C2-OCT-fast 8.0 s. Hệ quả: câu "thành phần khóa kích thước đặc thù loss cls" cần xem lại — co giãn có ích chung; C2 chỉ hơn OSFD vì RRB thiếu đa dạng kích thước (resize ≤ 1.1). Cần xác nhận dev100 (OCT vs C2-OCT-fast vs OSFD) trước khi chốt.

---

## 2026-10-08 — RaPA trên ĐƯỜNG PHÂN LOẠI + loss cls thuần + OCT @ dev30 (ý user) — khóa TRƯỚC khi chạy
**Lý do (user):** lần trước RaPA ghép với loss FEATURE (MORGD-RaPA: CA3 −25) — sai setting gốc của RaPA (loss phân loại trên logit). Lần này coi bài toán là PHÂN LOẠI: loss chỉ cls (rpn_cls + RoI cls), RaPA chỉ trên đường đi của nó, + đa dạng OCT. Hợp nguyên tắc "target chưa thấy" (RaPA chống dựa vào ít tham số surrogate).
**Ô:** `CLS-OCT-RaPA05` / `CLS-OCT-RaPA01` = CLS-OCT-fast (loss 4·(rpn_cls + cls), OCT, skip_mask, AMP, không forward sạch) + RaPA p = 0.05 / 0.01 trên BN backbone (53 lớp) + Linear RoI head nhánh phân loại (shared_fcs.0/1, fc_cls) — 14.0 M tham số; DropConnect weight + bias, không nhân bù, mask MỚI mỗi view, nhất quán forward + backward (`pia.rapa_cls_sampler`), S = 1.
**Unit test:** p = 0 → loss trùng tuyệt đối (0.966161); tỉ lệ mask 0.0500 / 0.0100; p = 0.05 loss cls 0.97 → 1.72; gradient hữu hạn; tham số model không đổi; 3 bước L∞ = 3; mask head khôi phục.
**Mốc (PNG dev30 có sẵn):** CLS-OCT-fast (cùng loss, không RaPA), C2-OCT-fast, OCT, OSFD.
**Quy tắc (dev30, chỉ số gộp):**
- "RaPA giúp (đường phân loại)" nếu ô tốt hơn đạt CA3 ≥ CA3(CLS-OCT-fast) + 2 VÀ YOLOX, Swin-L ≥ CLS-OCT-fast − 2. CA3 ≤ CLS-OCT-fast − 2 → "RaPA hại".
- Tín hiệu SOTA (GO dev100) nếu CA3 ≥ max(OCT, C2-OCT-fast) + 2 VÀ Swin-L ≥ max(...)  − 2.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods OSFD OCT C2-OCT-fast CLS-OCT-fast CLS-OCT-RaPA05 CLS-OCT-RaPA01 --out metrics_rapacls_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods OSFD OCT C2-OCT-fast CLS-OCT-fast CLS-OCT-RaPA05 CLS-OCT-RaPA01 --dets dets_rapacls_dev30.json --out generalization_rapacls_dev30.json
```

### Kết quả RaPA đường phân loại @ dev30 (`metrics_rapacls_dev30.json`, `generalization_rapacls_dev30.json`, log `results/stride/rapacls_dev30.log`) — theo quy tắc: **RaPA giúp (p = 0.01, sát ngưỡng)**; KHÔNG có tín hiệu SOTA
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9 | 3.0; OCT 96.6 / 91.9 / 85.0 / 83.9 / 82.1 / 27.7 | 83.6 | 5.5; C2-OCT-fast 99.5 / 94.7 / 86.4 / 86.1 / 77.4 / 29.2 | 83.3 | 8.0; CLS-OCT-fast 98.4 / 93.5 / 85.4 / 82.9 / 70.0 / 30.8 | 79.4 | 6.4; CLS-OCT-RaPA05 98.5 / 95.0 / 76.8 / 73.2 / 76.5 / 27.4 | 75.5 | 7.2; **CLS-OCT-RaPA01 98.9 / 95.1 / 87.5 / 84.7 / 72.7 / 32.5 | 81.6** | 7.0.
- RaPA01 − CLS-OCT: CA3 +2.2 (≥ 2), YOLOX +2.7, Swin-L +1.7 → "RaPA giúp" (sát ngưỡng, dev30 nhiễu ±5). RaPA05: CA3 −3.9 (ConvNeXt / Swin-T −9) → p = 0.05 quá mạnh, như lần trước.
- SOTA: RaPA01 81.6 < max(OCT, C2-OCT-fast) + 2 = 85.6 → không GO. Swin-L 32.5 cao nhất các ô OCT (trong nhiễu).
**Đọc:** đặt RaPA đúng setting (loss phân loại) đổi dấu hiệu quả so với lần ghép loss feature (−25 → +2.2 ở p nhỏ); nhưng loss cls thuần vẫn thiếu YOLOX nên tổng thể chưa vượt OCT / C2-OCT. Ứng viên tiếp: C2-OCT (feature + cls) + RaPA01 chỉ trên đường phân loại — khóa entry mới nếu user muốn.

---

## 2026-10-08 — Định vị lợi ích RaPA: chỉ backbone vs chỉ RoI head (p = 0.01, loss cls + OCT) @ dev30 — khóa TRƯỚC khi chạy
**Lý do (user, hướng novelty CLS-OCT-RaPA):** nếu lợi nằm ở RoI head → câu chuyện "loss cls làm δ overfit head phân loại surrogate" (khớp YOLOX tụt khi thêm cls) → ngẫu nhiên hóa có chủ đích đúng chỗ overfit, thay RaPA đồng đều.
**Ô:** `CLS-OCT-RaPA01-BB` (chỉ 53 BN backbone, 53 120 tham số) | `CLS-OCT-RaPA01-HEAD` (chỉ shared_fcs + fc_cls, 13.98 M) — còn lại = CLS-OCT-RaPA01 (`rapa_scope`). Smoke: L∞ = 3, mask 0.0099 / 0.0100.
**Mốc (PNG dev30 có sẵn):** CLS-OCT-fast (không RaPA, 79.4), CLS-OCT-RaPA01 (cả hai, 81.6), OCT, C2-OCT-fast.
**Quy tắc (dev30, CA3; lực thấp — hiệu ứng gốc chỉ +2.2):** chỗ X "mang lợi" nếu CA3(X) ≥ CA3(RaPA01 cả hai) − 1; "không mang" nếu CA3(X) ≤ CA3(CLS-OCT-fast) + 1. **HEAD** nếu HEAD mang VÀ BB không; **BACKBONE** nếu ngược lại; còn lại xám (cả hai / không rõ). Mô tả YOLOX, Swin-L.
**Lệnh:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods CLS-OCT-fast CLS-OCT-RaPA01 CLS-OCT-RaPA01-BB CLS-OCT-RaPA01-HEAD --out metrics_rapaloc_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods CLS-OCT-fast CLS-OCT-RaPA01 CLS-OCT-RaPA01-BB CLS-OCT-RaPA01-HEAD --dets dets_rapaloc_dev30.json --out generalization_rapaloc_dev30.json
```

### Kết quả định vị RaPA @ dev30 (`metrics_rapaloc_dev30.json`, `generalization_rapaloc_dev30.json`, log `results/stride/rapaloc_dev30.log`) — theo quy tắc: **BACKBONE**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: CLS-OCT-fast 98.4 / 93.5 / 85.4 / 82.9 / 70.0 / 30.8 | 79.4; CLS-OCT-RaPA01 (cả hai) 98.9 / 95.1 / 87.5 / 84.7 / 72.7 / 32.5 | 81.6; **RaPA01-BB 99.2 / 95.8 / 86.8 / 84.7 / 72.7 / 29.9 | 81.4**; RaPA01-HEAD 98.2 / 94.1 / 84.7 / 82.0 / 68.2 / 30.9 | 78.3.
**Đọc (dev30, lực thấp):** toàn bộ lợi CA3 của RaPA đến từ DropConnect 53 k tham số BN backbone; ngẫu nhiên hóa 14 M tham số RoI head không giúp (−1.1). → Giả thuyết "δ overfit head phân loại surrogate" KHÔNG được ủng hộ; nếu có overfit thì ở biểu diễn backbone (BN affine). Swin-L: lợi +1.7 của RaPA cả hai không nằm ở BB (29.9) — trong nhiễu.

---

## 2026-10-08 — CLS-OCT-RaPA01 (B = 50) vs OSFD B = 200 trên 5 ảnh (user) — CHỈ MÔ TẢ, không quyết định
5 ảnh đầu của dev30 theo thứ tự dev300 (`data/image_lists/dev5.csv`). AP trên 5 ảnh rất nhiễu → chỉ xem xu hướng + runtime; KHÔNG dùng ra quyết định. OSFD B = 200 run riêng `dev300_B200_eps5` (seed như mọi run).
**Kết quả (5 ảnh, mô tả; log `results/stride/vs200_dev5.log`):** R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD B50 100 / 94.9 / 87.4 / 58.3 / 46.0 / 35.8 | 63.9 | 3.0; **OSFD B200 100 / 100 / 90.1 / 69.4 / 84.4 / 38.0 | 81.3 | 12.4**; **CLS-OCT-RaPA01 B50 100 / 99.3 / 85.8 / 76.3 / 80.8 / 30.8 | 81.0 | 6.8**. Trên 5 ảnh: CLS-OCT-RaPA01 ≈ OSFD B200 về CA3 với ~½ thời gian; Swin-L thấp hơn. Lưu ý OSFD B50 ở 5 ảnh này (63.9) thấp bất thường so dev30 (78.9) → nhiễu 5 ảnh rất lớn.
**Bổ sung (user):** CLS-OCT-RaPA01 B = 200 trên cùng 5 ảnh (run `dev300_B200_eps5`) — so cùng B với OSFD B200; mô tả.
**Kết quả (5 ảnh, mô tả; log `results/stride/b200_dev5.log`):** CLS-OCT-RaPA01 B200 100 / 100 / 84.1 / 76.3 / 82.5 / 31.0 | CA3 81.0 | 27.4 s — y hệt B50 (81.0, 6.8 s) → RaPA01 bão hòa từ B50; OSFD B200 81.3 / Swin-L 38.0 (12.4 s). Cùng B = 200 OSFD ≥ RaPA01 trên 5 ảnh này (nhiễu rất lớn).
**Bổ sung (user):** CLS-OCT-fast (không RaPA) B50 vs B200 trên cùng 5 ảnh — mô tả.
**Kết quả (5 ảnh, mô tả; log `results/stride/clsoct_b200_dev5.log`):** CLS-OCT-fast B50 99.5 / 83.6 / 85.6 / 74.8 / 74.9 / 27.2 | CA3 78.4 | 6.8 s; B200 100 / 95.8 / 88.5 / 76.9 / 80.5 / 27.5 | 82.0 | 24.5 s. → không RaPA: B50 → B200 +3.6 CA3; RaPA01: B50 = B200 = 81.0 (RaPA đạt sớm mức mà không-RaPA cần B lớn). Swin-L ~27–31 mọi ô loss cls; OSFD B200 38.0.

---

## 2026-10-08 — OCT-cache: cache feature sạch trên 64 phép biến đổi cố định (đo tốc độ 1 ảnh, user)
**Code:** `geo_osfd_attack(cache_views=64)` (loss feature thuần, OCT): tập cố định 64 phép = đủ 8×8 pha dịch, mỗi pha 1 mức co giãn phân tầng trong ±ln2/2 (RNG = initial_seed + 3); feature sạch 64 view tính 1 lần / ảnh (fp16), mỗi view rút chỉ số đều. Ô `OCT-cache`. ĐỔI method (rời rạc vs liên tục) → cần eval.
**Tốc độ (ảnh 7088, RTX 3090):** B50: OCT 6.8 s → OCT-cache 5.0 s (−26%); B200: 20.1 → 14.1 s (−30%). Bão hòa δ như nhau (0.95 / 0.98). Hình: `results/figs/oct_cache_7088.png` (ảnh sạch / adv OCT / adv OCT-cache + box Swin-T score ≥ 0.3: 7 → 29 / 27 box; δ ×25). Chưa eval transfer.

---

## 2026-10-08 — TẠM DỪNG (user, mai tiếp): tổng kết findings + việc mở
**Nguyên tắc đã chốt hôm nay:** mục tiêu = transfer sang target CHƯA THẤY (quyết định chỉ dựa chỉ số gộp, không tune theo 1 target / mẹo riêng surrogate); held-out = YOLOX-L + Mask R-CNN Swin-S. Mục tiêu cuối: SOTA + novelty.
**Findings (dev30 trừ khi ghi khác):**
1. **Đa dạng hình học = triệt thành phần δ khóa lưới downsample stride 4 / 8 của R50** (SH4 / SH2 / PH4 / PH8 / PH4L; thấy trực tiếp trên δ — E13 Z4). Dịch nguyên 0–7 px (PH8) ≈ OSFD; loại biến đổi (xoay, phối cảnh, warp) và dời xa không quan trọng; xoay không thêm khi đã có dịch.
2. **Đa dạng KÍCH THƯỚC (co giãn 1 octave) giúp thêm** — với cả loss feature: OCT (co giãn + dịch, loss feature thuần) CA3 83.6 vs OSFD 78.9 (+4.8), YOLOX 82.1. (Đính chính: không đặc thù loss cls.) Vượt mức FPN có chủ đích (LVX) KHÔNG giúp (δ yếu đi).
3. **Loss:** feature + RoI cls bổ trợ (feature giữ YOLOX, cls giúp Swin-L); objectness RPN không thay được RoI cls; với đa dạng OCT, cls gần như không thêm CA3.
4. **C2-PIA QUA sàng lọc dev300:** CA3 − OSFD +11.2 [8.1, 12.7], Swin-L 37.6 vs 24.9. C2-OCT ≈ C2-PIA (dev30).
5. **RaPA:** ghép loss feature hại (−25); trên ĐƯỜNG PHÂN LOẠI (loss cls thuần, p = 0.01) +2.2 CA3, lợi nằm ở BN backbone (không ở RoI head); bão hòa từ B50 (5 ảnh).
6. **Tốc độ:** skip_mask (tương đương) + AMP fp16 → C2-OCT −31%; OCT-cache (64 phép cố định) −26–30% (chưa eval transfer). OCT 1.8× OSFD; C2-OCT-fast 2.7×.
7. **Novelty:** OCT như thao tác KHÔNG mới (≈ DI resize + TI + OSFD loss; AugTrans 2026 có resize theo box). Novelty khả dĩ = cơ chế (2 thành phần đặc thù surrogate + chọn tham số có cơ sở).
8. **Phân rã lỗi (dev30):** CNN / Swin-T: ~60% object bị xóa + box sai ×3; **DINO-Swin-L: chỉ ~23% object bị xóa, không thêm box sai** → biểu diễn Swin-L vẫn không bị phá — nút thắt chính chưa giải.
**Việc mở cho phiên sau:**
- (a) Xác nhận dev100: OCT vs C2-OCT-fast vs OSFD (+ đối chứng cùng thời gian OSFD ~B90).
- (b) Đường bão hòa theo B (50 / 100 / 150 / 200) cho OCT vs OSFD; eval OCT-cache.
- (c) Swin-L: vì sao object không bị xóa — hướng phân tích chính cho novelty.
- Code mới: `elem` (s2..s32, ph4 / ph4a / ph8 / ph32 / ph4l, scale_strat, lvx, GT co-transform), `geo_osfd_attack(skip_mask, amp, feature, cls_mode, rapa_cls_p, rapa_scope, cache_views)`, `run_baselines --surrogate` (chưa dùng), `scripts/e13_grid_template.py`. Kết quả: `results/runs/dev300_B50_eps5/*_{stride,ph,ph2,c2pia,c2ph8,c2anchor,oct,octfast,clsoct,rpnoct,lvx,rot,rapacls,rapaloc,...}*`, `results/runs/dev300_B200_eps5/`, log `results/stride/`.

---

## 2026-10-10 — E14 (hướng c, user): DINO-Swin-L trung hòa δ transfer Ở ĐÂU — ghép feature backbone (activation patching), độ lớn hay hướng? — khóa TRƯỚC khi chạy (dev30, forward-only, chỉ xem xu hướng)
**Lý do:** đã biết Swin-L không bền cơ bản (E8 WB-OSFD 84.6 vs transfer 27.2); không phải quy mô / phổ (E10a); độ lớn distortion KHÔNG theo drop giữa ô (swinl_geo); lỗi = hạ confidence, định vị đúng, chọn query không vỡ. Mọi phân tích cũ là TƯƠNG QUAN với độ lớn distortion, chưa can thiệp nhân quả, chưa cho δ WB đi qua cùng phép đo. E14 hỏi: thiệt hại nằm ở stage backbone nào, và δ R50 thua δ WB vì đẩy feature Swin-L QUÁ ÍT (độ lớn) hay SAI HƯỚNG.
**Nguồn δ (PNG dev30 có sẵn):** WB-OSFD (`artifacts/runs/e8_wbosfd_dev30/WB-OSFD`, mốc trần); transfer **OSFD (chính)**, OCT, C2-PIA (mô tả / kiểm nhất quán).
**Thao tác (DINO-Swin-L, test pipeline như `mech_swinl_geo.py`):** tính backbone 4 stage (stride 4/8/16/32) cho ảnh sạch c_s và adv a_s; feature ghép → neck → transformer → `bbox_head.predict` (rescale). Điều kiện mỗi nguồn: `full` (= adv thường); `noise_s` (sạch, chỉ stage s ← adv), s = 1..4; `denoise_s` (adv, chỉ stage s ← sạch); `noise_12`, `noise_34`. Riêng nguồn transfer T: `up` = c_s + α_s (a_s^T − c_s), α_s = ‖a^WB_s − c_s‖ / ‖a^T_s − c_s‖ theo ảnh, MỌI stage (khớp độ lớn WB, giữ hướng T); `x2`, `x4` (α cố định, mô tả). Riêng WB: `down_T` = c_s + (‖a^T_s − c_s‖ / ‖a^WB_s − c_s‖)(a^WB_s − c_s) (hướng WB, độ lớn T), cho từng T. Kèm cosd / rel từng stage mỗi nguồn.
**Chỉ số:** GT (COCO, không crowd) được phát hiện ở ảnh sạch (cùng lớp, IoU ≥ 0.5, score ≥ 0.3, `e4_target_side.matched`); **D = 1 − tỉ lệ GT còn score ≥ 0.3** (gộp GT, bootstrap theo ảnh 1000, BOOT_SEED 2026). AP drop: mô tả. r_X = D(X) / D(WB full).
**Sanity (trước khi đọc):** AP drop `full` WB-OSFD / OSFD khớp E8 (84.6 / 27.2) trong ±3; `noise_s` với a = c ⇒ trùng sạch (ghép không làm lệch).
**Quy tắc (nguồn OSFD; dev30 ⇒ quyết định theo điểm, chỉ là xu hướng; OCT / C2-PIA cùng nhãn = củng cố):**
- **(R1) stage gánh thiệt hại WB:** s nếu D(WB, noise_s) ≥ 0.5 · D(WB, full). Báo cả transfer (mô tả).
- **(R2) "nút thắt ở stage NÔNG"** ỦNG HỘ nếu D(WB, noise_12) ≥ 0.5 · D(WB, full) VÀ rel trung bình bb1, bb2 của WB ≥ 2 × của OSFD.
- **(R3) độ lớn vs hướng** (r_T = D(OSFD full) / D(WB full)): **MAG** nếu r_up ≥ 0.7 VÀ r_down ≤ r_T + 0.2 (δ R50 đúng hướng, chỉ đẩy quá ít); **DIR** nếu r_up ≤ r_T + 0.2 VÀ r_down ≥ 0.6 (khuếch đại vô ích, hướng WB ở độ lớn T vẫn mạnh → Swin-L/DINO bất nhạy với HƯỚNG của δ R50); còn lại → **XÁM (cả hai)**, mô tả.
**Caveat khai báo trước:** n = 30; `up` / `x2` / `x4` đẩy feature ra khỏi phân phối tự nhiên (chẩn đoán, không phải attack); ghép ở backbone giả định neck + DINO là nơi trung hòa — không tách neck / encoder / decoder (làm sau nếu R3 rõ). Không tune attack nào theo E14.
**Chi phí:** ~60 lượt head / ảnh × 30 ảnh, ước 10–15 phút RTX 3090. Script `scripts/e14_patching.py` → `results/mechanism/e14_patching.json`, log `results/e14/run.log`.

### Kết quả E14 @ dev30 (`results/mechanism/e14_patching.json`, log `results/e14/run.log`, dets `artifacts/e14/dets.json` — sinh lại ~12 phút) — theo quy tắc: **R1 = stage 3; R2 KHÔNG; R3 = DIR ở cả OSFD / OCT / C2-PIA**
**Sanity:** đường ghép trên sạch trùng `model.predict` (lệch 0.0); AP drop `full` WB-OSFD 84.6 / OSFD 27.2 — khớp tuyệt đối E8. 266 GT phát hiện ở sạch. Kiểm thêm (scratch, 3 ảnh): load lại JPG sạch qua `pipeline_resize` trùng tuyệt đối dataset → không có sàn do pipeline; **nhiễu đều ±5 ngẫu nhiên cho cosd bb1 ≈ OSFD** (0.046 vs 0.047; 0.090 vs 0.092) → bb1 bất biến giữa ô (swinl_geo) là SÀN của mọi nhiễu ε = 5, không mang thông tin δ.
D (1 − tỉ lệ GT còn ≥ 0.3), WB / OSFD / OCT / C2-PIA:
- full 0.677 / 0.233 / 0.237 / 0.263; noise_1 0.02 / 0.01 / 0.01 / 0.02; noise_2 0.02 / 0.01 / 0.01 / 0.02; **noise_3 0.440 / 0.165 / 0.135 / 0.158**; noise_4 0.263 / 0.045 / 0.056 / 0.045; noise_12 0.03 / 0.04 / 0.05 / 0.06; noise_34 0.658 / 0.199 / 0.184 / 0.218.
- denoise_3 0.282 / **0.079** / 0.102 / 0.102; denoise_4 0.470 / 0.199 / 0.195 / 0.218; denoise_1, _2 ≈ full.
- Độ lớn vs hướng: OSFD up 0.286 (α trung vị bb1–4 1.00 / 0.91 / 1.17 / 1.39), x2 0.500, x4 0.744; WB down_OSFD 0.560, down_OCT 0.583, down_C2-PIA 0.545.
- Distortion rel bb1–4: WB 0.37 / 0.43 / 0.87 / 1.24; OSFD 0.37 / 0.48 / 0.74 / 0.91.
- R3: OSFD r_T 0.34, r_up 0.42, r_down 0.83 → DIR; OCT 0.35 / 0.43 / 0.86 → DIR; C2-PIA 0.39 / 0.51 / 0.81 → DIR.
**Đọc (dev30, xu hướng):**
1. **Thiệt hại trên Swin-L đi qua stage 3 (stride 16)** — cho cả WB lẫn transfer: riêng bb3 adv mang 65% thiệt hại WB và 71% thiệt hại OSFD; trả bb3 về sạch xóa 2/3 thiệt hại OSFD (0.233 → 0.079). Stage 4 phụ (WB 0.26, transfer ≈ 0.05). Stage 1–2 = 0 (kể cả WB) → tiền đề "stage nông / độ phân giải cao" bị bác dứt khoát; DINO không cần "mỏ neo sạch" ở mức nông để chống lại.
2. **Nút thắt là HƯỚNG, không phải độ lớn:** khuếch đại δ R50 tới đúng độ lớn WB từng stage chỉ 0.23 → 0.29; hướng WB thu về độ lớn của R50 vẫn 0.56 (83% WB). Cùng độ lớn, hướng WB hiệu quả ~2.4× hướng R50. Ở bb3, distortion WB chỉ lớn hơn OSFD 17% (rel 0.87 vs 0.74) mà thiệt hại riêng bb3 gấp 2.7×.
3. **Hướng R50 không vô dụng, chỉ kém hiệu quả:** ×2 → 0.50, ×4 → 0.74 (≈ WB) — nhưng ×4 là rel bb4 ~3.6, xa phân phối; trong ε = 5 không đạt được bằng độ lớn.
4. OSFD / OCT / C2-PIA gần như trùng mọi điều kiện → đa dạng hình học / loss cls không đổi KÊNH (vẫn bb3) mà chỉ nhích nhẹ; khớp "Swin-L kẹt ~30%".
**Hệ quả (chưa khóa):** mục tiêu đúng = làm distortion bb3 (stride 16) của target transformer lớn ĐÚNG HƯỚNG hơn, không phải mạnh hơn. Câu hỏi kế: hướng WB ở bb3 khác hướng R50 thế nào (theo kênh / không gian / trong-ngoài object; phần chung cos(Δ_T, Δ_WB) ở bb3) và có thuộc tính nào đo được phía surrogate (không dùng Swin-L) dự đoán được nó. Claim cần ≥ dev100 (phải sinh 70 ảnh WB-OSFD, ~35 phút).

---

## 2026-10-10 — E14b (user): hướng distortion stage 3 của WB khác R50 Ở ĐÂU — không gian (object / nền) hay không gian con kênh (trội / phụ)? — khóa TRƯỚC khi chạy (dev30, forward-only)
**Lý do:** E14 — thiệt hại Swin-L đi qua bb3 (stride 16), nút thắt là HƯỚNG (cùng độ lớn, hướng WB ~2.4× hướng R50). Cần biết phần nào của hướng WB mang thiệt hại, rồi xem δ R50 đặt năng lượng ở đó ít hơn không.
**Thao tác (khung `noise_3` của E14: sạch mọi stage, riêng bb3 = c₃ + T(Δ₃), Δ₃ = a₃ − c₃):**
- **Không gian:** `in` = Δ₃ chỉ giữ ở ô lưới stride 16 có tâm nằm trong ≥ 1 GT box (khung model, không ignore); `out` = phần còn lại.
- **Không gian con kênh:** PCA theo ảnh trên token sạch c₃ (HW × 768, trừ trung bình); `P32` = Δ₃ chiếu lên 32 thành phần chính đầu (theo token), `Q32` = Δ₃ − P32.
- `n3` = Δ₃ nguyên (= `noise_3` E14, kiểm tái lập).
Nguồn: WB-OSFD, OSFD (chính), OCT, C2-PIA; NOISE chỉ mô tả.
**Mô tả mỗi nguồn:** cos(Δ₃ˣ, Δ₃^WB) toàn ảnh / trong box / ngoài box; tỉ lệ năng lượng ‖Δ₃‖² trong box (kèm tỉ lệ token trong box); tỉ lệ năng lượng trong P32.
**Chỉ số:** D như E14 (266 GT, bootstrap ảnh 1000, BOOT_SEED 2026).
**Quy tắc (WB định danh phần nhạy; OSFD so sánh; điểm, xu hướng):**
- **(H-S) không gian:** phần nhạy = `in` nếu D(WB, in) ≥ 0.7 · D(WB, n3) VÀ D(WB, out) ≤ 0.3 · D(WB, n3); = `out` nếu ngược lại; khác → không định danh được bằng không gian. Nếu định danh được: **"R50 đặt ít năng lượng vào phần nhạy"** ỦNG HỘ nếu tỉ lệ năng lượng của OSFD ở phần đó ≤ tỉ lệ của WB − 0.10.
- **(H-P) không gian con:** tương tự với `P32` / `Q32`; ỦNG HỘ nếu tỉ lệ năng lượng OSFD ở phần nhạy ≤ 0.6 × tỉ lệ của WB.
- Cả hai không định danh / không ủng hộ → khác biệt hướng KHÔNG nằm ở phân bổ không gian thô hay không gian con trội — ghi nhận, bước sau là hướng trong-token (cos theo vị trí).
- Mô tả: D(OSFD, in / out / P32 / Q32) và cos.
**Caveat:** n = 30; mask box dùng GT (chỉ phân tích); PCA theo ảnh trên ~4–5k token. Script `scripts/e14b_direction.py` → `results/mechanism/e14b_direction.json`, log `results/e14/e14b.log`.

### Kết quả E14b @ dev30 (`results/mechanism/e14b_direction.json`, log `results/e14/e14b.log`, dets `artifacts/e14/dets_b.json`) — theo quy tắc: **H-S định danh phần nhạy = TRONG object nhưng KHÔNG ủng hộ "R50 đặt ít năng lượng ở đó"; H-P không định danh** → khác biệt hướng nằm TRONG-TOKEN ở vùng object
**Tái lập:** D(n3) WB 0.440 / OSFD 0.165 / OCT 0.135 / C2-PIA 0.158 — trùng `noise_3` E14 tuyệt đối.
D, n3 / in / out / P32 / Q32: WB 0.440 / **0.383** / 0.034 / 0.162 / 0.158; OSFD 0.165 / 0.147 / 0.015 / 0.086 / 0.060; OCT 0.135 / 0.117 / 0.019 / 0.083 / 0.060; C2-PIA 0.158 / 0.143 / 0.019 / 0.071 / 0.068; NOISE 0.086 / 0.079 / 0.004 / 0.045 / 0.034.
Mô tả, e_in (tỉ lệ năng lượng Δ₃ trong box; tỉ lệ token trong box 0.336) / e_P32 / cos(Δ, Δ_WB) toàn / trong / ngoài: WB 0.327 / 0.338 / 1 / 1 / 1; OSFD 0.296 / 0.344 / 0.438 / 0.422 / 0.441; OCT 0.301 / 0.352 / 0.418 / 0.413 / 0.418; C2-PIA 0.295 / 0.354 / 0.420 / 0.416 / 0.423; **NOISE 0.287 / 0.361 / 0.397 / 0.373 / 0.404**.
**Đọc (dev30, xu hướng):**
1. **Phần nhạy = distortion bb3 TẠI object:** riêng phần trong box mang 87% thiệt hại bb3 của WB (0.383 / 0.440), ngoài box ≈ 0 — cùng mẫu ở mọi nguồn. Nhưng KHÔNG nguồn nào tập trung năng lượng vào object (e_in ≈ tỉ lệ token 0.33, cả WB) → WB thắng KHÔNG nhờ dồn năng lượng vào object; R50 đặt năng lượng trong box gần bằng WB (0.296 vs 0.327).
2. **Không gian con trội không tách được:** P32 và Q32 mỗi phần chỉ ~36% thiệt hại WB, tổng (0.32) < nguyên (0.44) → thiệt hại cần hướng ĐẦY ĐỦ (cộng hưởng, không low-rank); không phải "R50 đẩy lệch không gian con".
3. **Phần chung cos ~0.42 với WB chủ yếu là phản ứng CHUNG của Swin-L với mọi nhiễu:** NOISE đã 0.40; OSFD chỉ hơn +0.04, đa dạng hình học (OCT) / loss cls (C2-PIA) không hơn OSFD. Cái thừa riêng của attack R50 so nhiễu ngẫu nhiên rất mỏng — khớp: D(in) OSFD 0.147 chỉ ~2× NOISE 0.079, WB ~5×.
**Hệ quả:** khác biệt WB vs R50 nằm ở HƯỚNG TRONG TỪNG TOKEN object ở stride 16 (768 chiều), không ở vị trí, không ở không gian con thô. Theo quy tắc khóa: bước sau là hướng trong-token. Câu cơ chế ứng viên (chưa claim): "δ R50 tạo ở Swin-L chủ yếu phản ứng chung kiểu nhiễu; phần đặc thù đúng hướng tại token object stride 16 rất nhỏ và đa dạng hình học / loss cls không tăng được nó".

---

## 2026-10-10 — E14c (user duyệt): hướng hiệu quả của WB ở token object bb3 = ĐẨY VỀ PHÍA NỀN ("xóa ngữ nghĩa object")? — khóa TRƯỚC khi chạy (dev30, forward-only)
**Lý do:** E14b — phần nhạy là distortion bb3 tại token object; khác biệt WB vs R50 nằm ở hướng trong-token. Giả thuyết cụ thể, kiểm được, có hệ quả method: WB đẩy token object về thống kê token nền; R50 thì không (hoặc có trên R50 nhưng không chuyển sang Swin-L).
**Hướng nền theo object:** token bb3 (stride 16) gán cho GT box nhỏ nhất chứa tâm ô; với mỗi GT g: u_g = μ_nền − μ_g (trung bình token SẠCH ngoài mọi box / trong box g). Ảnh không có token nền → bỏ khỏi thống kê hướng, thao tác = 0.
**Mô tả (mỗi nguồn WB / OSFD / OCT / C2-PIA / NOISE):** s = trung bình theo token object của (Δ_t · u_g) / ‖u_g‖² (tỉ lệ "khoảng cách object → nền" đã đi); e_u = Σ(Δ_t · û_g)² / Σ‖Δ_t‖² (tỉ lệ năng lượng dọc hướng nền; mức ngẫu nhiên ~1/768).
**Thao tác nhân quả (khung `noise_3`, chỉ token object, ngoài box = 0):** `U` = chỉ thành phần dọc û_g của Δ; `Uperp` = Δ − thành phần đó; `synthWB` / `synthT` = c₃ + β·u_g tại token object, β theo ảnh sao cho ‖·‖ = ‖Δ_in‖ của WB / của OSFD (đẩy thuần về nền ở độ lớn WB / R50). Mốc: `in` (= E14b).
**Mô tả phía surrogate:** cùng s, e_u trên C4 của R50 (stride 16) cho OSFD, OCT, C2-PIA, NOISE — δ R50 có đẩy object về nền TRÊN CHÍNH R50 không.
**Quy tắc (điểm, xu hướng):**
- **(B1) khác biệt hướng:** ỦNG HỘ nếu s_WB > 0 VÀ s_WB ≥ 2·s_NOISE VÀ s_WB ≥ 2·s_OSFD (trên Swin-L).
- **(B2) hướng nền mang thiệt hại WB:** ỦNG HỘ nếu D(WB, U) ≥ 0.5 · D(WB, in).
- **(B3) hướng nền là hướng tấn công HIỆU QUẢ:** ỦNG HỘ nếu D(synthT) ≥ 0.7 · D(WB, in) (đẩy thuần về nền ở độ lớn R50 đạt ≥ 70% WB).
- B1 ∧ B2 → cơ chế "WB xóa ngữ nghĩa object về phía nền, R50 không". B3 → mục tiêu method khả dĩ (loss đẩy token object về nền trên surrogate — kiểm riêng, khóa entry mới). ¬B2 ∧ ¬B3 → hướng nền hạng 1 không phải cơ chế; ghi nhận.
**Caveat:** n = 30; u_g dùng GT (phân tích); hạng 1 là kiểm chặt (nền đa dạng). Script `scripts/e14c_bgshift.py` → `results/mechanism/e14c_bgshift.json`, log `results/e14/e14c.log`.

### Kết quả E14c @ dev30 (`results/mechanism/e14c_bgshift.json`, log `results/e14/e14c.log`, dets `artifacts/e14/dets_c.json`) — theo quy tắc: **B1 ỦNG HỘ, B2 KHÔNG, B3 ỦNG HỘ (mạnh)**
Tái lập: D(in) WB 0.383 / OSFD 0.147 / OCT 0.117 / C2-PIA 0.143 / NOISE 0.079 — trùng E14b.
- **Swin-L, s (tỉ lệ khoảng cách object → nền đã đi) | e_u (bội so ngẫu nhiên):** WB **+0.353** | ×58; OSFD +0.155 | ×44; OCT +0.154 | ×44; C2-PIA +0.130 | ×41; NOISE +0.100 | ×42. → B1: s_WB = 2.3× OSFD, 3.5× NOISE.
- **Nhân quả (khung noise_3, chỉ token object):** WB U (riêng thành phần hạng 1 dọc hướng nền) 0.071 (19% của `in`), Uperp 0.259 → B2 KHÔNG (thành phần dọc nền không mang phần lớn thiệt hại của chính WB; tổng không cộng tính). **synthT (đẩy THUẦN về nền ở độ lớn của OSFD) 0.575 = 1.5 × D(WB, in), 3.9 × D(OSFD, in); synthWB 0.808** → B3 ỦNG HỘ.
- **Phía surrogate (R50 C4):** s OSFD +0.317, OCT +0.361, C2-PIA +0.295, **NOISE +0.396** → attack R50 hiện tại KHÔNG đẩy object về nền trên chính R50 hơn nhiễu ngẫu nhiên (phần s > 0 là hiệu ứng chung của nhiễu — giảm tương phản object / nền).
**Đọc (dev30, xu hướng):**
1. WB có đi về phía nền nhiều hơn R50 (B1), nhưng đó KHÔNG phải cách chính WB gây thiệt hại (B2): WB dùng một hướng giàu hơn, hạng 1 chỉ là một phần.
2. **Phát hiện chính (B3): trên Swin-L, dịch token object bb3 đồng loạt về trung bình nền là hướng tấn công CỰC hiệu quả** — ở đúng độ lớn distortion mà δ R50 đang tạo, nó gây thiệt hại gấp ~4× δ R50, vượt cả WB. Tức nút thắt KHÔNG phải ngân sách distortion mà là hướng; và tồn tại một hướng đơn giản, định nghĩa không cần target (μ_nền − μ_object), rất hiệu quả.
3. Loss OSFD (MSE khuếch đại feature) / loss cls trên R50 không nhắm hướng này (s trên R50 ≤ NOISE).
**Hệ quả method (chưa khóa):** loss "object → nền" trên surrogate: kéo feature token object (stride 16, có thể thêm 8 / 32) của R50 về μ_nền của chính R50 (GT box đã có trong threat model GT-assisted). Câu hỏi then chốt: dịch về nền trên R50 có kéo theo dịch về nền trên Swin-L (s_Swin-L tăng) không. Caveat: synth là dịch nhất quán theo object trong không gian feature — chưa biết đạt được qua pixel với ε = 5; rà literature (feature-level object vanishing / background-feature attack) trước khi khóa method.

---

## 2026-10-10 — Pilot loss "object → nền" (BG-OCT, OCT+BG) @ dev30 (user duyệt) — khóa TRƯỚC khi chạy
**Lý do:** E14c B3 — trên Swin-L, kéo token object bb3 về μ_nền là hướng cực hiệu quả (ở độ lớn δ R50 gây D 0.575 vs 0.147). Thử tối ưu hướng tương tự TRÊN SURROGATE.
**Literature (rà nhanh 2026-10-10):** OSFD gốc (Ding et al., AAAI'24) nén feature trội của object + khuếch đại vùng lân cận; Lu et al. 2026 (Springer, ensemble) nén feature trội + tăng feature nền; Lian et al. 2026 / "Attack Anything" — attack CHỈ trên nền. Chưa thấy: kéo token object về TRUNG BÌNH NỀN SẠCH ở stride 16, single-surrogate, có bằng chứng cơ chế cross-backbone (E14–E14c). Novelty của toán tử mỏng → nếu dùng, đóng góp phải là cơ chế.
**Loss:** L_bg = −mean_obj ‖f_t − μ_nền‖² / mean_obj ‖f^sạch_t − μ_nền‖², f = C4 R50 (stride 16) của view, μ_nền = trung bình token SẠCH (co-transform) ngoài mọi GT box (box co-transform theo view), bỏ token pad; ascent. `attack/methods/pia.py` (`_obj_mask`, `_bg_loss`, `geo_osfd_attack(bg_weight, bg_only)`).
**Ô (khung OCT: MI, 2 view / backward, co giãn ±ln2/2 + dịch PH8, σ = 6, B = 50, ε = 5):** `BG-OCT` = chỉ L_bg; `OCT+BG` = L_OSFD + λ·L_bg, **λ = 6** (luật C2: median mean|∇L_OSFD| / mean|∇L_bg| tại δ = 0, 5 ảnh đầu dev100 = 5.98, làm tròn 1 chữ số có nghĩa như λ_V; `results/bg/lambda_calibration.json`). Mốc: OSFD, OCT (PNG có sẵn, eval lại cùng lần).
**Unit test (`scripts/bg_unit_test.py`) ĐẠT:** L_bg(δ = 0) = −1 cả 5 ảnh, gradient hữu hạn ≠ 0; thiếu nền / object → 0. **Ghi nhận trước khi chạy:** sau 10 bước trên ảnh 7088, s_R50 (như E14c) BG-OCT 0.44, **OCT 0.48**, nhiễu đều ±5 0.01 → loss feature OCT ĐÃ kéo object về nền trên R50 ngang (hơn) BG-OCT; tức trên surrogate hướng này không thiếu — câu hỏi thực là có CHUYỂN sang Swin-L không.
**Chạy:**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --methods OSFD OCT BG-OCT OCT+BG --out metrics_bg_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods OSFD OCT BG-OCT OCT+BG --dets dets_bg_dev30.json --out generalization_bg_dev30.json
python scripts/bg_pilot_check.py   # s trên Swin-L bb3 + R50 C4 (hàm E14c)
```
**Quy tắc (dev30, điểm; chỉ quyết định mở rộng dev100):**
- Thao tác: (M1) s_R50(ô) ≥ s_R50(OCT) + 0.10; (M2) s_Swin-L(ô) ≥ s_Swin-L(OCT) + 0.05.
- **GO dev100:** OCT+BG nếu Swin-L ≥ OCT + 3 VÀ CA3 ≥ OCT − 1; BG-OCT nếu Swin-L ≥ OCT + 3 VÀ CA3 ≥ OSFD.
- M1 ∧ ¬M2 → kéo về nền trên surrogate KHÔNG chuyển sang Swin-L (bằng chứng chống họ method này; khớp E14 "hướng"). M2 mà Swin-L không tăng → dịch về nền tới được Swin-L nhưng chưa đủ — mô tả. ¬M1 → thao tác thất bại trên surrogate, không kết luận.

### Kết quả pilot "object → nền" @ dev30 (`metrics_bg_dev30.json`, `generalization_bg_dev30.json`, `results/bg/pilot_check_dev30.json`, log `results/bg/pilot_dev30.log`) — theo quy tắc: **KHÔNG GO cả hai ô; thao tác M1 trượt sát (¬M1 → không kết luận nhân quả chính thức)**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD 97.8 / 93.8 / 83.3 / 77.8 / 75.6 / 27.2 | 78.9 | 3.0; OCT 96.6 / 91.9 / 85.0 / 83.9 / 82.1 / 27.7 | 83.7 | 5.5; **BG-OCT 86.0 / 58.3 / 37.4 / 42.8 / 34.2 / 10.1 | 38.1** | 7.8; **OCT+BG 96.0 / 86.7 / 67.5 / 75.3 / 63.1 / 15.7 | 68.6** | 7.9. (OSFD / OCT tái lập đúng số đã commit.)
Thao tác, s: R50 C4 — OSFD 0.317, OCT 0.361, BG-OCT 0.439 (+0.079), OCT+BG 0.460 (+0.099) → M1 trượt cả hai (ngưỡng +0.10; OCT+BG sát biên). **Swin-L bb3 — OSFD 0.155, OCT 0.154, BG-OCT 0.056, OCT+BG 0.083 → M2 trượt, NGƯỢC chiều.**
**Đọc (mô tả, dev30):**
1. Thêm loss kéo về nền trên R50 làm attack YẾU ĐI ở MỌI model (kể cả R50 white-box 86.0, R101 58.3) và Swin-L tụt 27.7 → 10.1 / 15.7 — âm rất mạnh, không phải nhiễu.
2. Kéo được object về nền trên R50 (s 0.44–0.46, tăng nhẹ so OCT) nhưng trên Swin-L object lại BỚT về nền (0.15 → 0.06–0.08): "về nền" trên R50 đạt bằng một đường đặc thù R50 không chuyển giao; hướng nền ở Swin-L không thể với tới bằng cách tối ưu hướng nền của R50.
3. Củng cố E14 (nút thắt là HƯỚNG): hướng nhạy của Swin-L định nghĩa trong không gian feature của Swin-L; "μ_nền − μ_object" tuy định nghĩa không cần target nhưng KHÔNG đồng nghĩa giữa R50 và Swin-L ở mức δ. B3 (E14c) vẫn đúng về Swin-L, nhưng chưa có đường từ surrogate tới nó.
**Hệ quả:** dừng họ "kéo object về nền trên surrogate" (không thêm biến thể stage / λ mà không khóa entry mới). Câu hỏi còn mở cho (c): thành phần nào của δ R50 có ánh xạ nhất quán sang hướng Swin-L — gợi ý: thứ chuyển được là distortion năng lượng cao kiểu OSFD, không phải mục tiêu ngữ nghĩa trên surrogate.

---

## 2026-10-10 — Cải tiến OCT @ dev30 (user duyệt): OCT-fast, OCT-fast-V4, OCT-C4, OCT-C34 — khóa TRƯỚC khi chạy
**Lý do:** OCT chậm (5.5 s vs OSFD 3.0) và chưa SOTA rõ (CA3 +4.8 vs OSFD @ dev30, Swin-L không tăng, chưa dev100 / held-out). Hai trục: tốc độ, và độ mạnh có cơ sở từ E14 (thiệt hại Swin-L đi qua stride 16). Đo trước (scratch, 5 ảnh dev100, δ ~ U(±3)): tỉ lệ mean|∇ₓ| của L_OSFD theo stage C2 / C3 / C4 / C5 = 0.003 / 0.19 / 0.39 / 0.41 → C5 (stride 32) đang ngang C4.
**Ô (khung OCT: MI, co giãn ±ln2/2 + PH8, σ = 6, k = 3, B = 50, ε = 5):**
- `OCT-fast` = OCT + `cache_views=64` + AMP fp16 (đổi method: 64 phép cố định rời rạc; đổi số học).
- `OCT-fast-V4` = OCT-fast với 4 view / backward (≈ thời gian OCT; so cùng thời gian chạy).
- `OCT-C4` = L_OSFD chỉ trên C4 (stride 16); `OCT-C34` = C3 + C4. Code: `geo_osfd_attack(feat_stages=)`; kiểm: `feat_stages=(0,1,2,3)` trùng tuyệt đối OCT mặc định (0% pixel khác sau 5 bước), C4 khác 41%.
**Chạy:** `run_baselines.py --split dev300 --subset dev30 --methods OSFD OCT OCT-fast OCT-fast-V4 OCT-C4 OCT-C34 --out metrics_octimp_dev30.json` → `eval_generalization.py ... --models yolox_s dino_swin_l --methods <cùng> --dets dets_octimp_dev30.json --out generalization_octimp_dev30.json`. Log `results/octimp/run.log`.
**Quy tắc (dev30, điểm; mốc OCT tái lập cùng lần):**
- `OCT-fast` THAY OCT nếu ΔCA3 (− OCT) ≥ −1 VÀ runtime ≤ 0.7 × OCT.
- `OCT-fast-V4`, `OCT-C4`, `OCT-C34`: GO dev100 nếu CA3 ≥ OCT + 2, HOẶC (Swin-L ≥ OCT + 3 VÀ CA3 ≥ OCT − 1). Báo runtime; V4 phải ≤ 1.1 × runtime OCT mới tính là "cùng thời gian".
- Nhiều ô GO → dev100 tất cả (cùng mốc), không ghép trước dev100. Không ô nào → ghi nhận, không thêm biến thể mà không khóa entry mới. Held-out (YOLOX-L, Swin-S) KHÔNG dùng ở bước này.

**TRẠNG THÁI (dừng giữa chừng theo user, 2026-10-10):** pilot cải tiến OCT CHƯA có kết quả. Đã sinh PNG: OCT-fast 30 / 30, OCT-fast-V4 16 / 30, OCT-C4 0, OCT-C34 0 (PNG nằm trên máy cũ, KHÔNG upload — sinh lại được). **Runtime đo được: OCT-fast 4.4 s/ảnh (= 0.80 × OCT 5.5) → nhiều khả năng TRƯỢT điều kiện tốc độ ≤ 0.7 × OCT** (quyết định chính thức vẫn chờ eval); OCT-fast-V4 6.3 s/ảnh (1.15 × OCT → vượt ngưỡng "cùng thời gian" ≤ 1.1). Lý do ước sai (~2.8 s): cache 64 phép ở B = 50 chỉ bớt ròng ~36 forward sạch (bỏ 100, trả trước 64); lợi AMP ×1.45 đo trên C2-OCT (có RPN + RoI head), loss feature thuần chỉ qua backbone nên lợi nhỏ hơn. Ý tưởng tăng tốc thật (CHƯA khóa): (1) gộp 2 view vào 1 batch forward / backward (toán y hệt — BN eval; B không đổi; kiểm δ trùng trong sai số GPU); (2) cache lười (chỉ tính phép được rút) + 16–32 phép; (3) 1 view / bước (đổi method). Công bằng: nếu so tốc độ, OSFD cũng bật được AMP.

---

## 2026-10-10 — TẠM DỪNG (user trả máy, tối chạy tiếp): tổng kết phiên + việc mở
**Xong trong phiên (dev30, mô tả — claim cần ≥ dev100):**
1. **E14 (ghép feature backbone DINO-Swin-L):** thiệt hại trên Swin-L đi qua **stage 3 (stride 16)** cho cả WB lẫn transfer; stage 1–2 = 0 (bb1 bất biến = sàn mọi nhiễu ε = 5, kiểm bằng nhiễu đều). **Nút thắt là HƯỚNG (DIR)**: khuếch đại δ R50 tới độ lớn WB 0.23 → 0.29; hướng WB ở độ lớn R50 vẫn 0.56 (83% WB).
2. **E14b:** phần nhạy = distortion bb3 TẠI token object (87% thiệt hại WB); R50 KHÔNG thiếu năng lượng ở đó (e_in 0.30 vs WB 0.33); không tách được bằng không gian con 32 PC; cos với WB ~0.42 phần lớn là phản ứng chung với nhiễu (NOISE 0.40).
3. **E14c:** WB kéo object về phía nền nhiều hơn (s 0.35 vs OSFD 0.155) nhưng đó không phải đường chính của WB (B2 ✗); **kéo thuần token object về μ_nền ở độ lớn R50 → D 0.575 (≈ 4× δ R50, > WB)** (B3 ✓).
4. **Pilot BG-OCT / OCT+BG (loss "object → nền" trên R50): KHÔNG GO, âm mạnh** (CA3 38.1 / 68.6 vs OCT 83.7; Swin-L 10.1 / 15.7); s trên Swin-L NGƯỢC chiều (0.15 → 0.06–0.08) → kéo về nền trên surrogate không chuyển giao. Dừng họ method này.
5. Literature (rà nhanh): OSFD (AAAI'24), Lu et al. 2026 (ensemble, nén feature trội + tăng nền), Lian et al. 2026 / Attack Anything (attack nền).
6. Đo gradient L_OSFD theo stage R50: C2 / C3 / C4 / C5 = 0.003 / 0.19 / 0.39 / 0.41.
**Việc mở (theo thứ tự đề xuất):**
- (1) **Chạy tiếp pilot cải tiến OCT** (quy tắc đã khóa ở entry trên — không đổi): lệnh resume y hệt (`run_baselines.py ... --methods OSFD OCT OCT-fast OCT-fast-V4 OCT-C4 OCT-C34 --out metrics_octimp_dev30.json` rồi `eval_generalization.py ... --dets dets_octimp_dev30.json --out generalization_octimp_dev30.json`), ~25 phút trên máy mới (OCT-fast / V4 phải sinh lại vì PNG không upload; OSFD / OCT dev30 có trong HF `runs/dev300_B50_eps5.tar`).
- (2) Nếu cần tốc độ: khóa entry "gộp view vào batch" (+ cache lười), kiểm tương đương trước.
- (3) Xác nhận E14 / E14b / E14c @ dev100 (cần sinh 70 ảnh WB-OSFD ~35 phút; `scripts/e8_whitebox_lowfreq.py --osfd`).
- Nợ cũ: (a) dev100 OCT vs C2-OCT-fast vs OSFD; (b) đường bão hòa theo B; GO @ n300 cho C2 (110 / 300).
**Code mới phiên này:** `scripts/e14_patching.py`, `e14b_direction.py`, `e14c_bgshift.py`, `bg_unit_test.py`, `bg_pilot_check.py`; `attack/methods/pia.py`: `_obj_mask`, `_bg_loss`, `_view_loss_bg`, `geo_osfd_attack(bg_weight, bg_only, feat_stages)`, `_view_loss(stages)`; ô `BG-OCT`, `OCT+BG`, `OCT-fast`, `OCT-fast-V4`, `OCT-C4`, `OCT-C34` trong `run_baselines.py`. Mặc định giữ nguyên hành vi cũ (kiểm: `feat_stages=(0..3)` trùng OCT 0% pixel).
**Artifact:** KHÔNG upload HF phiên này. Không cần cho việc mở (1). E14* dets (`artifacts/e14/`) sinh lại ~12 / 5 / 8 phút; WB-OSFD dev30 có trong HF `runs/e8_wbosfd_dev30.tar`.
**Không có tiến trình nào đang chạy.**

---

## 2026-10-17 — Máy mới V100 16GB; OCT 5 / 10 ảnh (mô tả) + OCT-1V (1 view / backward) @ dev30 — khóa TRƯỚC khi chạy dev30
**Máy:** Tesla V100-SXM2-16GB (KHÔNG phải 3090 / 4000 Ada), bootstrap đủ checkpoint + COCO; KHÔNG có `.env` → không kéo PNG HF. **Tái lập giữa GPU KHÔNG đạt:** OSFD dev5 cùng seed, V100 vs số commit (3090): R101 89.8 vs 94.9, Swin-T 71.3 vs 58.3, YOLOX 64.5 vs 46.0, Swin-L 19.8 vs 35.8 (khả dĩ: TF32 trên Ampere + 50 bước sign khuếch đại sai số; cộng nhiễu n = 5). → Từ nay mọi so sánh sinh lại MỐC trên cùng máy; không trộn PNG HF với PNG V100. Run tách riêng bằng `--limit` (`dev300_B50_eps5_limit{5,10,30}`) để không ghi đè stats đã commit.
**Mô tả (KHÔNG quyết định):**
- dev5 (`results/runs/dev300_B50_eps5_limit5/`, log `results/octdev5/run.log`): R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD 100 / 89.8 / 85.7 / 71.3 / 64.5 / 19.8 | 73.8 | 3.9; OCT 100 / 94.6 / 79.6 / 72.6 / 67.8 / 38.0 | 73.3 | 6.5.
- Hình box (`scripts/viz_dets.py` → `results/figs/viz_oct_dev5/`, so box adv với box SẠCH của chính model): DINO-Swin-L giữ 23 / 34 (OSFD), 21 / 34 (OCT) box sạch, thêm chỉ 5 / 2 box; CNN / Swin-T mất gần hết box sạch + thêm 56–192 box. OCT ≈ OSFD về số box sạch mất (ConvNeXt 36 vs 37, Swin-T 37 vs 37) nhưng THÊM nhiều box giả hơn (98 vs 76, 78 vs 56), box giả nhỏ / dày hơn → lợi CNN của OCT nghiêng về hallucination (kênh không qua DINO). n = 5, chỉ sinh giả thuyết.
- dev10 (`dev10.csv` = 10 ảnh đầu dev30; `results/runs/dev300_B50_eps5_limit10/`, log `results/octdev5/oct1v_dev10.log`): OSFD 99.1 / 84.3 / 79.4 / 69.9 / 62.8 / 29.8 | 70.7 | 3.9; OCT 100 / 84.1 / 75.1 / 70.9 / 72.1 / 35.5 | 72.7 | 6.3; **OCT-1V** 100 / 82.2 / 69.7 / 65.3 / 71.9 / 34.4 | 69.0 | **3.0** (0.77× OSFD).
**Ô mới `OCT-1V`** = OCT với `n_views=1` (1 view / backward, 50 view forward thay vì 100; feature sạch co-transform 1 view / bước). Câu hỏi (user): bỏ trung bình trong-bước, giữ trung bình theo thời gian qua MI, có giữ được lợi OCT với thời gian ≤ OSFD không.
**Chạy (dev30, V100, ε = 5; mốc sinh lại cùng máy):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 30 --methods OSFD OCT OCT-1V --out metrics_oct1v_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5_limit30 --subset dev30 --models yolox_s dino_swin_l --methods OSFD OCT OCT-1V --dets dets_oct1v_dev30.json --out generalization_oct1v_dev30.json
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 30 --budget 65 --methods OCT-1V --out metrics_oct1v_dev30.json
python scripts/eval_generalization.py --run dev300_B65_eps5_limit30 --subset dev30 --models yolox_s dino_swin_l --methods OCT-1V --dets dets_oct1v_dev30.json --out generalization_oct1v_dev30.json
```
B = 65 cho OCT-1V = cùng thời gian OSFD B50 (3.9 / 3.0 × 50 ≈ 65, đo dev10). Log `results/octdev5/oct1v_dev30.log`.
**Quy tắc (dev30, điểm — chỉ quyết định mở rộng dev100; CA3 = TB ConvNeXt-T / Swin-T / YOLOX-S):**
- **Chính — OCT-1V B65 (cùng thời gian OSFD):** GO dev100 nếu CA3 ≥ CA3(OSFD) + 2 VÀ Swin-L ≥ Swin-L(OSFD) − 2 VÀ runtime ≤ 1.1 × OSFD. **NO-GO** (1 view không khả thi đứng riêng) nếu CA3 ≤ CA3(OSFD). Giữa → xám, mô tả.
- **Phụ — OCT-1V B50:** "thay được OCT" nếu CA3 ≥ CA3(OCT) − 1 VÀ Swin-L ≥ Swin-L(OCT) − 3; khác → mô tả.
- Mô tả: OCT − OSFD (tái lập xu hướng dev30 cũ +4.8 trên V100?), R101, ConvNeXt / Swin-T riêng (dev10 thấy 1 view tụt ~5 ở 2 target này).
**Caveat:** n = 30; B65 khác B50 nên OCT-1V B65 ở run riêng — so điểm giữa 2 run (cùng ảnh, cùng seed theo chỉ số dev300). Held-out không dùng.

### Kết quả OCT-1V @ dev30 (V100; `results/runs/dev300_B{50,65}_eps5_limit30/{metrics,generalization}_oct1v_dev30.json`, log `results/octdev5/oct1v_dev30.log`) — theo quy tắc: **CHÍNH: GO dev100 (OCT-1V B65); PHỤ: OCT-1V B50 thay được OCT**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh:
- OSFD B50: 98.3 / 91.8 / 81.0 / 80.4 / 74.5 / 25.2 | 78.6 | 3.96
- OCT B50: 96.8 / 91.8 / 83.9 / 82.8 / 77.1 / 29.4 | 81.3 | 6.62
- OCT-1V B50: 96.6 / 86.3 / 82.5 / 81.2 / 78.6 / 28.0 | 80.8 | 3.12
- **OCT-1V B65: 97.0 / 91.8 / 82.8 / 81.9 / 82.0 / 27.2 | 82.2 | 4.13**
- Chính: CA3 82.2 ≥ 78.6 + 2 ✓; Swin-L 27.2 ≥ 23.2 ✓; runtime 4.13 ≤ 1.1 × 3.96 = 4.36 ✓ → **GO dev100**. Δ so OSFD: ConvNeXt +1.8, Swin-T +1.5, YOLOX +7.5, Swin-L +2.0, R101 0.0 (so điểm giữa 2 run, chưa có CI cặp).
- Phụ: OCT-1V B50 CA3 80.8 ≥ 81.3 − 1 ✓, Swin-L 28.0 ≥ 26.4 ✓ → thay được OCT (cùng B, 0.47× thời gian OCT, 0.79× OSFD).
**Đọc (mô tả):** (1) OCT − OSFD trên V100 chỉ +2.6 CA3 (dev30 cũ 3090: +4.8) — lợi OCT nhỏ hơn đã thấy; YOLOX gánh phần lớn. (2) Bỏ trung bình trong-bước gần như không mất gì (OCT-1V B50 − OCT −0.5 CA3); tụt ~5 ở ConvNeXt / Swin-T của dev10 KHÔNG lặp lại ở dev30 (−1.4 / −1.6) → nhiễu n = 10. R101 tụt ở B50 (86.3) nhưng hồi đủ ở B65. (3) Lợi của OCT-1V B65 so OSFD tập trung ở YOLOX (+7.5); ConvNeXt / Swin-T / Swin-L chỉ +1.5–2 — trong nhiễu dev30.
**Bước tiếp (đề xuất, chưa khóa):** dev100 OCT-1V B65 vs OSFD B50 (sinh lại cùng máy), paired bootstrap; quy tắc khóa entry mới.

---

## 2026-10-17 — Xác nhận OCT-1V B65 vs OSFD @ dev100 + OCT-1V-S (phủ có hệ thống thay rút ngẫu nhiên) — khóa TRƯỚC khi chạy
**Lý do:** OCT-1V B65 GO dev100 (entry trên). User hỏi: thay rút độc lập (co giãn U(±ln2/2), pha 0–7 px mỗi trục) bằng cách phủ tối ưu hơn. Với 1 view × 65 bước, rút độc lập không phủ đều 64 tổ hợp pha (trùng / sót).
**Ô mới `OCT-1V-S`** = OCT-1V + `strat=True` (`elem.strat_schedule`): pha (dx, dy) đi theo nối các hoán vị ngẫu nhiên của đủ 64 tổ hợp (B65 → mỗi pha ≥ 1 lần, tối đa 2); ln s = Latin hypercube 65 ô đều trên [−ln2/2, ln2/2], thứ tự xáo; seed theo ảnh. Kiểm: 64 / 64 pha, 65 / 65 ô co giãn; Linf 5.0; `strat=False` đi đúng nhánh cũ.
**Không tất định (ghi nhận):** OCT-1V cùng code, cùng seed, chạy 2 lần liên tiếp trên V100 lệch 20.8% pixel (ảnh 7088) — khả dĩ backward `grid_sample` CUDA (atomicAdd). → chênh nhỏ giữa ô ở n nhỏ càng phải đọc theo CI.
**Công cụ:** `pair_screen.py` nhận điều kiện `TÊN@RUN` (so cặp giữa 2 run khác B, cùng ảnh). Kiểm dev30: OCT-1V B65 − OSFD CA3 +3.6 [−0.2, 7.4], YOLOX +7.5 [1.7, 13.0], Swin-L +2.0 [−4.7, 7.6] (khớp điểm đã ghi).
**Chạy (V100; dev30 của OSFD B50 / OCT-1V B65 tái dùng từ run `_limit30` — chép PNG + stats, cùng seed, cùng máy):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --methods OSFD --out metrics_oct1v_dev100.json
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods OSFD --dets dets_oct1v_dev100.json --out generalization_oct1v_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 65 --methods OCT-1V OCT-1V-S --out metrics_oct1v_dev100.json
python scripts/eval_generalization.py --run dev300_B65_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods OCT-1V OCT-1V-S --dets dets_oct1v_dev100.json --out generalization_oct1v_dev100.json
python scripts/pair_screen.py --run dev300_B50_eps5_limit100 --subset dev100 --dets dets_oct1v_dev100.json --tag oct1v_dev100 --exclude dev30 \
  --pairs OCT-1V@dev300_B65_eps5_limit100:OSFD OCT-1V-S@dev300_B65_eps5_limit100:OSFD OCT-1V-S@dev300_B65_eps5_limit100:OCT-1V@dev300_B65_eps5_limit100
```
Log `results/octdev5/oct1v_dev100.log`.
**Quy tắc (dev100, paired bootstrap 1000, BOOT_SEED 2026; Δ = drop(A) − drop(B)):**
- **(Q1) OCT-1V B65 vs OSFD B50 — XÁC NHẬN** nếu ΔCA3 ≥ 2 VÀ cận dưới CI > 0 VÀ ΔSwin-L ≥ −2 (điểm) VÀ ΔR101 ≥ −3 VÀ runtime ≤ 1.1 × OSFD (đo cùng lần). **BÁC** nếu ΔCA3 ≤ 0 (điểm). Còn lại → XÁM.
- **(Q2) OCT-1V-S vs OCT-1V (cùng B65) — "phủ có hệ thống GIÚP"** nếu ΔCA3 ≥ 2 VÀ cận dưới CI > 0; "HẠI" nếu ΔCA3 ≤ −2; còn lại "ngang". Chọn ô đại diện: OCT-1V-S chỉ khi GIÚP; ngược lại giữ OCT-1V.
- Mô tả: Q1 cho OCT-1V-S; tách dev100 \ dev30 (70 ảnh mới — kiểm dev30 có thiên lệch chọn không); YOLOX / ConvNeXt / Swin-T riêng.
- Q1 XÁC NHẬN → bước sau (khóa entry mới): sàng lọc dev300 theo plan_sota bước 3, cùng thời gian OSFD. Held-out không dùng.

### Kết quả dev100 (V100; `results/ablation/oct1v_dev100/screen.json`, `results/runs/dev300_B{50,65}_eps5_limit100/`, log `results/octdev5/oct1v_dev100.log`) — theo quy tắc: **Q1 XÁM (≈ 0); Q2 "ngang" → giữ OCT-1V**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD B50 96.6 / 91.5 / 78.9 / 75.5 / 70.8 / 24.9 | 75.0 | 3.89; OCT-1V B65 95.7 / 90.8 / 79.0 / 73.4 / 73.8 / 24.4 | 75.4 | 4.04; OCT-1V-S B65 96.4 / 91.5 / 78.4 / 74.2 / 70.5 / 23.2 | 74.4 | 4.04.
- **Q1 OCT-1V − OSFD: ΔCA3 +0.4 [−1.6, 2.9]** (< 2, điểm > 0 → XÁM); YOLOX +3.0 [0.3, 6.0], ConvNeXt +0.1, Swin-T −2.1, Swin-L −0.6, R101 −0.7. Runtime 1.04× ✓.
- **Q2 OCT-1V-S − OCT-1V: ΔCA3 −1.0 [−2.6, 1.2]** → ngang (YOLOX −3.3 [−5.9, −0.8]). Phủ có hệ thống KHÔNG giúp.
- 70 ảnh mới (dev100 \ dev30): OCT-1V − OSFD CA3 −0.6 [−2.1, 2.6] → lợi +3.6 ở dev30 KHÔNG lặp lại ngoài dev30 (nhiễu / thiên lệch tập nhỏ).
**Đọc:** cùng thời gian, trên V100, OCT-1V ≈ OSFD; phần lợi duy nhất ổn định là YOLOX (+2–3). Hệ quả rộng hơn: lợi "+4.8 CA3" của OCT (dev30, 3090) chưa từng được xác nhận ở dev100 — có thể phần lớn cũng là nhiễu dev30. Chưa có số OCT 2 view @ dev100 trên V100.
**User (2026-10-17):** nếu phát triển tiếp, đi theo hướng 1 view.

---

## 2026-10-17 — Rà literature (web) + OCT-1V-SEA (OCT-1V + cập nhật kiểu Season) @ dev100 — khóa TRƯỚC khi chạy
**Literature (web, 2026-10-17):**
- **AugTrans** (CMC v87n3, bản HTML): N_EOT = 10 view / bước × K_max = 160 (≈ 1600 lượt gradient), ε = 5/255, η = 0.0004, Faster R-CNN R50; 16.7 s / ảnh (3090). **KHÔNG so OSFD**; so DAG / RAP / CAP / TOG / Shi et al. / UAP; tự nhận **BFDA thắng trên DETR / DINO**; không so cùng compute; 1 seed. Target transformer của họ đều backbone R50 (DETR-R50, DINO-R50 — với ta OSFD đã 96–97%).
- **BFDA** (Ding et al. 2024, "Bidirectional Feature Distortion", cùng nhóm OSFD): nén phần tử feature quan trọng + tăng phần tử gắn box ứng viên. Không tìm thấy code. → rủi ro baseline cho claim SOTA, rà lại ở bước 4.
- **Corrupting Attention** (arXiv 2608.06674): white-box nhắm logit attention (trước softmax) của encoder tại token object → DINO-Swin-L 56.8 → 1.4 mAP; luận điểm "phá memory encoder là đủ" khớp E14 (thiệt hại qua token object stride 16 = đầu vào encoder). Không đo transfer từ CNN; surrogate R50 không có encoder → không dùng trực tiếp.
- **AFOG** (ICCV 2025): white-box attention, không transfer. JAD / NAMEA / OmniPatch / IKD / TGR / ATT: cần surrogate ViT hoặc ensemble → ngoài threat model.
- **Season** (arXiv 2608.04441, CNN→ViT classification): tách gradient g_L = Gauss_σ(g), g_H = g − g_L; momentum + δ riêng từng dòng; δ_H chiếu trực giao δ_L; tổng clip + co theo pixel. R50 → ViT-B/16: MI +5.5, TI +11; ablation: phần lợi ViT chủ yếu từ tách 2 dòng (+4) + trực giao; mask saliency hơi hại ViT. Không thêm gradient. KHÁC S1 (S1 bỏ tần cao — đã âm): Season GIỮ tần cao.
**Ô mới `OCT-1V-SEA`** = OCT-1V, cập nhật MI thay bằng Season (`core.run_iterative_attack(season_sigma=3.0)`, bỏ mask): σ = 3 px ở không gian ảnh gốc (δ tối ưu ở ảnh gốc ~640 px; quy đổi σ = 1 @ 224 px theo tỉ lệ ≈ 2.9; bán kính 2σ như Season), α = 1 mỗi dòng, B = 65 (cùng OCT-1V). Giả thuyết: sign của 1 dòng MI để tần cao (độ lớn áp đảo) quyết định δ → δ là mẫu tần cao khớp nội dung (E7), ít chạm kênh hình dạng của Swin-L (E6); dòng tần thấp có bước sign riêng → luôn được phần ngân sách.
**Kiểm trước khi chạy (3 ảnh đầu dev100, B65):** L∞ 5.00; tỉ lệ năng lượng δ qua Gauss σ = 3 (thao tác): OCT-1V 0.146 / 0.097 / 0.108 → **SEA 0.450 / 0.406 / 0.417**; runtime ≈ như nhau (3.7–4.1 s). `season_sigma=None` = đường code cũ (nhánh riêng, `continue`).
**Chạy (V100; OSFD B50 + OCT-1V / OCT-1V-S B65 dev100 có sẵn):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 65 --methods OCT-1V OCT-1V-S OCT-1V-SEA --out metrics_oct1v_dev100.json
python scripts/eval_generalization.py --run dev300_B65_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods OCT-1V OCT-1V-S OCT-1V-SEA --dets dets_oct1v_dev100.json --out generalization_oct1v_dev100.json
python scripts/pair_screen.py --run dev300_B50_eps5_limit100 --subset dev100 --dets dets_oct1v_dev100.json --tag sea_dev100 \
  --pairs OCT-1V-SEA@dev300_B65_eps5_limit100:OCT-1V@dev300_B65_eps5_limit100 OCT-1V-SEA@dev300_B65_eps5_limit100:OSFD
```
Log `results/octdev5/sea_dev100.log`.
**Quy tắc (dev100, paired bootstrap 1000, BOOT_SEED 2026; Δ = drop(SEA) − drop(OCT-1V), cùng mọi thứ trừ quy tắc cập nhật):**
- Thao tác (M): tỉ lệ năng lượng tần thấp của δ (σ = 3), trung bình 100 ảnh, SEA ≥ OCT-1V + 0.10. ¬M → không kết luận.
- **GIÚP** nếu (ΔSwin-L ≥ 3 VÀ cận dưới CI > 0 VÀ ΔCA3 ≥ −1) HOẶC (ΔCA3 ≥ 2 VÀ cận dưới CI > 0 VÀ ΔSwin-L ≥ −2). **HẠI** nếu ΔCA3 ≤ −3 HOẶC ΔSwin-L ≤ −3 (điểm). Còn lại → NGANG. Runtime phải ≤ 1.1 × OCT-1V.
- Mô tả: SEA − OSFD (cùng thời gian); ConvNeXt / Swin-T / YOLOX / R101 riêng.
- GIÚP → ứng viên (khóa entry mới: sàng lọc dev300 so OSFD cùng thời gian + đối chứng OSFD-1V). NGANG / HẠI → ghi nhận, không thêm biến thể σ mà không khóa entry mới.

### Kết quả OCT-1V-SEA @ dev100 (V100; `results/ablation/sea_dev100/screen.json`, log `results/octdev5/sea_dev100.log`) — theo quy tắc: **M ✓ (thao tác đạt), HẠI (âm rất mạnh)**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: **SEA 66.7 / 43.6 / 28.2 / 28.7 / 38.5 / 7.3 | 31.8 | 4.01**; OCT-1V 95.7 / 90.8 / 79.0 / 73.4 / 73.8 / 24.4 | 75.4 | 4.04; OSFD B50 96.6 / 91.5 / 78.9 / 75.5 / 70.8 / 24.9 | 75.0.
- SEA − OCT-1V: CA3 **−43.6 [−48.7, −38.2]**, Swin-L **−17.1 [−20.3, −13.0]**, R101 −47.2; mọi model âm, kể cả white-box R50 (66.7).
- Chẩn đoán (ảnh 7088, loss OSFD white-box trên ảnh adv không biến đổi; δ = 0: 11.41): OCT-1V 15.5 / 18.2 / 19.1 / 19.6 tại bước 5 / 20 / 40 / 65; SEA 15.5 / 16.3 / 16.5 / 16.7 → SEA CÓ tối ưu (loss tăng đơn điệu) nhưng bão hòa sớm, thấp hơn → yếu do thiết kế, không phải hỏng đường gradient (chưa loại trừ hoàn toàn lỗi tinh vi).
**Đọc:** ở ε = 5 (Season dùng ε = 16, 10 bước), dòng tần thấp có momentum mượt → bước sign +α nhất quán mỗi bước → δ_L bão hòa ±ε thành khối lớn rất sớm; phép co theo pixel min(1, ε / |δ_L + δ_H|) nén δ_H ở mọi chỗ cùng dấu → ngân sách bị khóa vào khối tần thấp, phần tần cao (thứ phá được R50 / CNN) gần như không còn chỗ. Củng cố S1: ÉP thêm tần thấp vào δ (lọc gradient S1 hay dòng riêng SEA) làm yếu MỌI target, KỂ CẢ Swin-L — giả thuyết "Swin-L cần thành phần quy mô lớn trong δ" bị bác thêm một lần ở dạng này. Không thêm biến thể σ / chia ngân sách mà không khóa entry mới.

---

## 2026-10-17 — E15: kênh "DỜI / TÁCH object" trên DINO-Swin-L — ghép feature bb3 (forward-only, dev100) — khóa TRƯỚC khi chạy
**Lý do:** mọi hướng "đổi tính chất δ trên surrogate" ≈ OSFD hoặc hại ở dev100 (OCT-1V, OCT-1V-S, OCT-1V-SEA, trước đó S1, BG). DINO chặn box giả (FP nền 1.1 → 1.8) và chỉ để lọt kênh hạ confidence. Giả thuyết kênh khác: nếu nội dung token object bị DỜI, matching một-một của DINO đặt box ở chỗ mới → 1 lần vừa mất GT vừa có box lệch, không cần "đúng hướng" feature Swin-L theo nghĩa E14. Kiểm khả thi trên Swin-L trước khi viết loss cho R50.
**Thao tác (khung `noise_3`: mọi stage sạch, riêng bb3 = c₃ sửa; `scripts/e15_shift_object.py`):** object = chữ nhật token của GT box; dời ρ = 0.5 × bề rộng (token, ≥ 1, về phía còn chỗ; IoU box dời / gốc ≈ 1/3).
- `MOVE`: token object (gán như E14c) ← μ_nền sạch; dán nội dung chữ nhật sạch của từng GT ở vị trí dời (lớn → nhỏ).
- `DUP`: giữ nguyên + dán thêm bản dời. `SPLIT`: token object ← μ_nền; nửa trái dời trái ρ/2·w, nửa phải dời phải ρ/2·w.
- `X_T` = c₃ + Δ_X · ‖Δ_in^OSFD‖ / ‖Δ_X‖ (cùng độ lớn phần trong-object của δ OSFD thật). Mốc: `OSFD_in` = c₃ + Δ_in^OSFD (PNG OSFD B50 dev100, V100), `bg_T` = kéo object về μ_nền ở độ lớn OSFD (synthT E14c).
**Chỉ số:** D như E14 (GT phát hiện ở sạch, score ≥ 0.3; bootstrap ảnh 1000, BOOT_SEED 2026); F = tỉ lệ GT đó có detection cùng lớp, score ≥ 0.3, IoU ≥ 0.5 với box ĐÃ DỜI (SPLIT: 1 trong 2 nửa); n_det / ảnh.
**Sanity:** thao tác trên tensor giả đúng (MOVE dời 1 token, DUP 2 chỗ, SPLIT 2 mảnh tách; sát biên → dời ngược; chiếm hết bề rộng → 0); chạy 3 ảnh không lỗi. Đường `head` trên c₃ sạch = `model.predict` (kiểm ở E14, lệch 0.0).
**Quy tắc (dev100, điểm + CI):**
- **(K1) kênh tồn tại** với X ∈ {MOVE, DUP, SPLIT}: D(X) ≥ 0.5 VÀ F(X) ≥ 0.3 (DINO đi theo nội dung token, đặt box ở chỗ mới).
- **(K2) với tới được ở ngân sách R50:** với X* = argmax D(X_T): D(X*_T) ≥ 2 × D(OSFD_in) VÀ cận dưới CI của D(X*_T) − D(OSFD_in) > 0.
- K1 ∧ K2 → khóa entry mới: loss "dời object" trên R50 (bản feature: đích = feature sạch R50 với token object dời — chi phí ≈ OCT-1V) + kiểm thao tác phía Swin-L (như M2 của pilot BG). K1 ∧ ¬K2 → kênh có nhưng cần độ lớn ngoài tầm R50 — mô tả. ¬K1 → DINO không đi theo nội dung token dời ở bb3 — bỏ hướng.
- Mô tả: so `bg_T` (E14c dev30 synthT 0.575 vs OSFD 0.147 — kiểm lại ở dev100); n_det (DUP có tạo box thừa không).
**Caveat khai báo trước:** cũng như E14c, hiệu quả trong không gian feature Swin-L KHÔNG đảm bảo với tới được từ R50 (BG-OCT là phản ví dụ); chỉ bb3 (stage 4 sạch); dời ngang. Log `results/e15/run.log`.

### Kết quả E15 @ dev100 (`results/mechanism/e15_shift_object.json`, log `results/e15/run.log`, dets `artifacts/e15/dets.json` — sinh lại ~3 phút) — theo quy tắc: **¬K1 (sát ngưỡng), ¬K2 → BỎ hướng "dời / tách object"**
734 GT phát hiện ở sạch (100 ảnh), n_det sạch 10.6 / ảnh. D [CI] | D − OSFD_in | F (box ở chỗ dời) | n_det:
- MOVE 0.499 [0.451, 0.547] | +0.375 | 0.48 | 9.4; DUP 0.298 | +0.174 | 0.43 | 10.5; SPLIT 0.305 | +0.181 | 0.16 | 9.2.
- MOVE_T 0.223 [0.176, 0.282] | +0.099 [0.057, 0.145] | 0.25; DUP_T 0.193 | +0.069; SPLIT_T 0.155 | +0.031.
- OSFD_in 0.124 [0.093, 0.157]; **bg_T 0.586 [0.518, 0.648]** | +0.462 (n_det 5.1).
- K1: MOVE D 0.499 < 0.5 (trượt 0.001, không làm tròn) dù F 0.48 ≥ 0.3; DUP / SPLIT D ~0.30 → ¬K1. K2: MOVE_T 0.223 < 2 × 0.124 = 0.248 → ¬K2.
**Đọc (mô tả):** (1) DINO CÓ đi theo nội dung token: dời đầy đủ thì ~½ GT mất box đúng và 48% GT có box ở chỗ mới (object được "dời", không biến mất) — nhưng đó là thao tác feature cỡ lớn. (2) Ở độ lớn mà δ R50 thật tạo ra, dời chỉ 1.8× OSFD; kéo về nền (bg_T) ở CÙNG độ lớn 4.7× OSFD — tái lập E14c (dev30 0.575) trên dev100. → Trên Swin-L, hướng "xóa ngữ nghĩa object về nền" hiệu quả hơn hẳn "dời object" ở cùng ngân sách distortion; vấn đề vẫn là KHÔNG với tới hướng đó từ R50 (pilot BG âm). (3) DUP: DINO khử trùng lặp một phần (D 0.30, n_det không tăng) — không mở kênh box thừa.
**Hệ quả:** họ "đổi kênh đầu ra bằng dời token" không đủ mạnh ở ngân sách R50 → không viết loss dời object. Tổng kết chuỗi phiên 10-17: mọi can thiệp mới (OCT-1V / -S / -SEA, E15) đều không vượt OSFD hoặc không đạt tiêu chí ở dev100; đòn bẩy Swin-L duy nhất đã xác nhận vẫn là loss cls + PIA (C2-PIA, dev300).

---

## 2026-10-17 — C2-OCT-1V / C2-PIA-1V trên 5 ảnh (mô tả, đo thời gian) + TẠM DỪNG (user, mai chạy tiếp)
**Ô mới (`run_baselines.py`):** `C2-OCT-1V` = loss C2 (L_OSFD + 4·(loss_rpn_cls + loss_cls), GT co-transform) + OCT, 1 view / bước, skip_mask + AMP; `C2-PIA-1V` = như trên với PIA (tham số paper). Chi phí view forward = B.
**dev5 (V100, B50; `results/runs/dev300_B50_eps5_limit5/{metrics,generalization}_c2_1v_dev5.json`, log `results/octdev5/c2_1v_dev5.log`) — mô tả, KHÔNG quyết định:** R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: OSFD 89.8 / 85.7 / 71.3 / 64.5 / 19.8 | 73.8 | 3.9; C2-OCT-1V 96.1 / 83.9 / 72.6 / 68.7 / 37.1 | 75.1 | **4.7 (1.2×)**; C2-PIA-1V 97.1 / 83.8 / 72.8 / 67.7 / 23.5 | 74.8 | **4.3 (1.1×)**. n = 5 → chỉ dùng thời gian (OCT không cls cũng từng 38.0 Swin-L trên dev5 rồi ≈ OSFD ở dev100).
**Việc chạy tiếp (ĐỀ XUẤT — user xác nhận / khóa entry trước khi chạy):** dev100, cùng thời gian OSFD B50: `C2-OCT-1V` B41 (= 50 × 3.9 / 4.7), `C2-PIA-1V` B45 (= 50 × 3.9 / 4.3); OSFD B50 dev100 có trong `results/runs/dev300_B50_eps5_limit100/` (PNG ở `artifacts/` — mất nếu đổi máy → sinh lại ~7 phút). Lệnh dự kiến:
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --methods C2-OCT-1V --out metrics_c2_1v_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 45 --methods C2-PIA-1V --out metrics_c2_1v_dev100.json
# eval_generalization cho 2 run B41 / B45 (--subset dev100 --models yolox_s dino_swin_l), rồi
python scripts/pair_screen.py --run dev300_B50_eps5_limit100 --subset dev100 --dets <dets OSFD> --tag c2_1v_dev100 --pairs C2-OCT-1V@dev300_B41_eps5_limit100:OSFD C2-PIA-1V@dev300_B45_eps5_limit100:OSFD
```
Lưu ý `pair_screen.py` đọc CÙNG tên `--dets` ở mọi run → đặt cùng `--out` / `--dets` cho cả 3 run, hoặc sửa script nhận dets theo run. Quy tắc đề xuất: GO dev300 nếu ΔCA3 (− OSFD) ≥ 2 VÀ cận dưới CI > 0, HOẶC ΔSwin-L ≥ 3 VÀ cận dưới CI > 0 VÀ ΔCA3 ≥ 0; ΔR101 ≥ −3; runtime ≤ 1.1 × OSFD.

### TẠM DỪNG 2026-10-17 — tổng kết phiên (V100 16GB)
**Xong (theo quy tắc khóa, dev100 trừ khi ghi khác):**
1. Máy V100: tái lập giữa GPU KHÔNG đạt (OSFD dev5 lệch tới 16 điểm Swin-L so số 3090); GPU không tất định cả trên cùng máy (OCT-1V 2 lần cùng seed lệch 20.8% pixel). → luôn sinh lại mốc cùng máy, đọc theo CI, không ra quyết định ở dev30.
2. **OCT-1V B65 (cùng thời gian OSFD) ≈ OSFD:** CA3 +0.4 [−1.6, 2.9]; chỉ YOLOX +3.0 [0.3, 6.0]. Lợi +3.6 ở dev30 không lặp lại trên 70 ảnh mới. OCT 2 view chưa chạy dev100 → lợi "+4.8" của OCT (dev30, 3090) chưa từng xác nhận.
3. OCT-1V-S (phủ có hệ thống): không giúp (−1.0). OCT-1V-SEA (Season): HẠI (CA3 −43.6, Swin-L −17.1) — ép tần thấp vào δ yếu mọi target (lần 2 sau S1).
4. E15 (dời / tách token object bb3 Swin-L): ¬K1 (D 0.499 sát ngưỡng), ¬K2 → bỏ. Mô tả: ở cùng độ lớn distortion R50, kéo về nền 0.586 vs dời 0.223 vs δ OSFD 0.124 (tái lập E14c ở dev100).
5. Hình box (dev5, `scripts/viz_dets.py`): DINO-Swin-L giữ ~2/3 box sạch, gần như không thêm box; OCT tăng box giả (hallucination) trên CNN so OSFD, không tăng xóa object.
6. Literature (web): AugTrans (10 view × 160 bước, không so OSFD, thua BFDA trên DETR / DINO), BFDA (baseline chưa xét, không code), Season, Corrupting Attention (encoder DINO là nút thắt, white-box). OCT như toán tử: không novelty.
**Code mới:** ô `OCT-1V`, `OCT-1V-S` (`elem.strat_schedule`, `geo_osfd_attack(strat)`), `OCT-1V-SEA` (`core.run_iterative_attack(season_sigma)`), `C2-OCT-1V`, `C2-PIA-1V`; `pair_screen.py` nhận `TÊN@RUN`; `scripts/viz_dets.py`, `scripts/e15_shift_object.py`; `data/image_lists/dev10.csv`. Run riêng V100: `results/runs/dev300_B{50,65}_eps5_limit{5,10,30,100}/`.
**Việc mở:** (1) C2-OCT-1V / C2-PIA-1V @ dev100 cùng thời gian (trên). (2) Đối chứng OSFD-1V cùng thời gian nếu một ô 1V vượt OSFD. (3) Cân nhắc định vị analysis paper (cơ chế: khóa lưới / thang kích thước; Swin-L: stride 16, hướng, kéo về nền vs dời). (4) Rà BFDA. (5) Nợ cũ: GO @ n300 cho C2 (110 / 300).
**Artifact:** KHÔNG upload HF. PNG V100 ở `artifacts/runs/dev300_B*_limit*` (mất khi trả máy; sinh lại được). **Commit local, CHƯA push.** Không có tiến trình nào đang chạy.

---

## 2026-10-18 — C2-OCT-1V B41 / C2-PIA-1V B45 vs OSFD B50 @ dev100 (cùng thời gian; user duyệt "oke run") — khóa TRƯỚC khi chạy
**Máy:** V100 16GB (cùng loại phiên 10-17, máy mới) — PNG OSFD B50 dev100 cũ đã mất → **sinh lại mốc OSFD cùng máy** (cùng seed). Bootstrap lại checkpoint + COCO.
**Ô (đã có trong `run_baselines.py`, entry 2026-10-17):** `C2-OCT-1V` B41 (= 50 × 3.9 / 4.7), `C2-PIA-1V` B45 (= 50 × 3.9 / 4.3) — B chọn để cùng thời gian OSFD B50 theo số đo dev5. ε = 5.
**Lệnh** (log `results/octdev5/c2_1v_dev100.log`; cùng `--out` / `--dets` ở cả 3 run để `pair_screen.py` đọc được):
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --methods OSFD --out metrics_c2_1v_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --methods C2-OCT-1V --out metrics_c2_1v_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 45 --methods C2-PIA-1V --out metrics_c2_1v_dev100.json
# eval_generalization (--subset dev100 --models yolox_s dino_swin_l --dets dets_c2_1v_dev100.json --out generalization_c2_1v_dev100.json) cho 3 run B50 / B41 / B45
python scripts/pair_screen.py --run dev300_B50_eps5_limit100 --subset dev100 --dets dets_c2_1v_dev100.json --tag c2_1v_dev100 --exclude dev30 \
  --pairs C2-OCT-1V@dev300_B41_eps5_limit100:OSFD C2-PIA-1V@dev300_B45_eps5_limit100:OSFD
```
**Quy tắc (dev100, paired bootstrap 1000, BOOT_SEED 2026; Δ = drop(ô) − drop(OSFD B50); CA3 = TB ConvNeXt-T / Swin-T / YOLOX-S), áp riêng cho từng ô:**
- **GO dev300** nếu [(ΔCA3 ≥ 2 VÀ cận dưới CI > 0) HOẶC (ΔSwin-L ≥ 3 VÀ cận dưới CI > 0 VÀ ΔCA3 ≥ 0)] VÀ ΔR101 ≥ −3 VÀ runtime ≤ 1.1 × OSFD (đo cùng lần).
- **NO-GO** nếu ΔCA3 ≤ 0 VÀ ΔSwin-L < 3 (điểm). Còn lại → XÁM, mô tả.
- Runtime > 1.1 × OSFD nhưng Δ đạt → ghi "đạt nhưng vượt thời gian", không GO (user quyết đổi B).
- Cả hai GO → chọn ΔCA3 lớn hơn (chênh < 0.5 → C2-OCT-1V, đơn giản hơn / không phụ thuộc PIA). GO → khóa entry mới: sàng lọc dev300 + đối chứng OSFD-1V cùng thời gian.
- Mô tả: dev100 \ dev30 (70 ảnh mới); R101 / ConvNeXt / Swin-T / YOLOX riêng; so số OSFD B50 sinh lại với số 10-17 (CA3 75.0, Swin-L 24.9).
**Caveat:** GPU không tất định (đọc theo CI); held-out (YOLOX-L, Swin-S) không dùng.

### Kết quả @ dev100 (V100; `results/ablation/c2_1v_dev100/screen.json`, `results/runs/dev300_B{50,41,45}_eps5_limit100/{metrics,generalization}_c2_1v_dev100.json`, log `results/octdev5/c2_1v_dev100.log`) — theo quy tắc: **cả hai GO dev300 (qua nhánh Swin-L); chọn C2-OCT-1V**
Mốc OSFD B50 sinh lại: 96.7 / 91.3 / 79.9 / 74.3 / 69.5 / 23.3 | CA3 74.6 | 3.9 s/ảnh (10-17: CA3 75.0, Swin-L 24.9 — khớp trong nhiễu).
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh:
- **C2-OCT-1V B41:** 96.9 / 93.2 / 82.4 / 78.4 / 70.6 / 32.1 | 77.2 | 3.8
- **C2-PIA-1V B45:** 95.4 / 91.3 / 79.7 / 80.4 / 69.6 / 31.4 | 76.6 | 4.2
Δ − OSFD [CI]:
- C2-OCT-1V: **Swin-L +8.8 [2.0, 11.3]**, CA3 +2.6 [−2.1, 6.8], R101 +1.8, ConvNeXt +2.6, Swin-T +4.1, YOLOX +1.1; runtime 0.97× ✓ → nhánh CA3 ✗ (CI chạm 0), nhánh Swin-L ✓ → **GO**.
- C2-PIA-1V: **Swin-L +8.2 [2.2, 10.9]**, CA3 +2.0 [−2.1, 5.0], R101 0.0, ConvNeXt −0.2, Swin-T +6.1 [0.6, 9.4], YOLOX +0.1; runtime 1.08× ✓ (sát 1.1×) → **GO**.
- Cả hai GO → ΔCA3 2.60 vs 2.01 (chênh 0.59 ≥ 0.5) → **chọn C2-OCT-1V**.
- 70 ảnh mới (dev100 \ dev30): C2-OCT-1V Swin-L +8.4 [2.7, 10.8], CA3 +2.4 [−2.6, 6.9]; C2-PIA-1V Swin-L +10.2 [3.3, 13.2], CA3 −0.0 [−3.9, 3.6] → lợi Swin-L lặp lại ngoài dev30 ở cả hai; lợi CA3 của C2-PIA-1V thì không.
**Đọc:** cùng thời gian OSFD, thêm loss cls + 1 view / bước cho lợi chắc trên DINO-Swin-L (~+8–9, CI > 0), lợi CNN / YOLOX nhỏ, chưa chắc (CA3 CI chứa 0). YOLOX gần như không đổi (OCT-1V B65 từng +3.0) — chưa tách được do B thấp hơn (41 vs 65) hay do loss cls.
**Bước tiếp (theo quy tắc: khóa entry mới trước khi chạy):** sàng lọc dev300 C2-OCT-1V B41 vs OSFD B50 (cùng máy) + đối chứng OSFD-1V cùng thời gian. User quyết.

---

## 2026-10-18 — Tinh chỉnh C2-OCT-1V trên dev10 (user: "thử nhanh xem xu hướng") — CHỈ MÔ TẢ, KHÔNG quyết định
**Ô mới (`run_baselines.py`), mọi thứ như C2-OCT-1V trừ 1 chỗ:** `C2-OCT-1V-w2` / `-w8` (trọng số cls 2 / 8 thay 4); `C2-OCT-1V-RaPA` (RaPA p = 0.01 trên 53 BN backbone, mask mới mỗi view, CÙNG mask cho feature sạch co-transform và forward adv → tác động cả L_feature lẫn L_cls; không thêm forward). Code: `geo_osfd_attack` cho phép `rapa_cls_p` khi `feature=True` chỉ với `rapa_scope="backbone"`; đường cũ không đổi.
**Kiểm:** 1 ảnh × 4 bước — base vs base cùng seed lệch 2.3% pixel (sàn không tất định); RaPA p = 0 vs base 4.5% (cùng cỡ sàn); p = 0.01: mask 1.0%, L∞ ≤ ε.
**Chạy (V100, dev10 = 10 ảnh đầu dev30, mốc sinh lại cùng máy; B41 = cùng thời gian OSFD B50):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev10 --limit 10 --methods OSFD --out metrics_tune_dev10.json
python scripts/run_baselines.py --split dev300 --subset dev10 --limit 10 --budget 41 --methods C2-OCT-1V C2-OCT-1V-w2 C2-OCT-1V-w8 C2-OCT-1V-RaPA --out metrics_tune_dev10.json
# + eval_generalization (yolox_s dino_swin_l) cho 2 run
```
Log `results/octdev5/tune_dev10.log`. **n = 10 (CI ±10 điểm) → chỉ xem xu hướng**; ô nào đáng đi tiếp phải khóa quy tắc riêng @ dev100 (so C2-OCT-1V B41 đã có).

### Kết quả dev10 (V100; `results/runs/dev300_B{50,41}_eps5_limit10/{metrics,generalization}_tune_dev10.json`, log `results/octdev5/tune_dev10.log`) — MÔ TẢ
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh:
- OSFD B50: 99.5 / 84.3 / 77.3 / 70.6 / 62.5 / 29.9 | 70.1 | 4.0
- C2-OCT-1V B41: 99.7 / 89.0 / 81.5 / 77.0 / 72.9 / 30.6 | 77.1 | 4.1
- C2-OCT-1V-w2: 99.8 / 88.3 / 79.6 / 74.3 / 76.3 / 39.0 | 76.7 | 3.8
- C2-OCT-1V-w8: 98.4 / 90.5 / 81.3 / 74.3 / 72.4 / 38.2 | 76.0 | 4.0
- C2-OCT-1V-RaPA: 99.7 / 92.0 / 82.7 / 77.5 / 75.3 / 36.7 | 78.5 | **4.6 (1.15× OSFD)**
**Đọc:** CA3 mọi biến thể trong ±1.5 của C2-OCT-1V (nhiễu n = 10 ~±10). Swin-L: cả w2 LẪN w8 đều cao hơn w4 (+8.4 / +7.6) → không đơn điệu theo trọng số → nhiều khả năng ô gốc thấp do nhiễu trên 10 ảnh này (dev100: C2-OCT-1V − OSFD Swin-L +8.8, ở đây chỉ +0.7), không phải w = 4 lệch. RaPA cao nhất CA3 (+1.4) và R101 nhưng chậm 1.15× → so công bằng cần B ≈ 36. Không biến thể nào cho xu hướng rõ.
**Sửa tốc độ RaPA (sau dev10):** chậm 1.15× do `int((m == 0).sum())` mỗi tensor (106 lần đồng bộ CPU–GPU / view) chỉ để ghi thống kê `rapa_masked_frac`. Bộ đếm chuyển lên GPU, đọc 1 lần cuối ảnh (`rapa_sampler` + `rapa_cls_sampler`). Kiểm: mask trùng tuyệt đối bản cũ (cùng generator, 3 lượt × 106 tensor), masked 1.00%; end-to-end 3 ảnh B41: C2-OCT-1V 4.10 vs C2-OCT-1V-RaPA 4.11 s/ảnh → RaPA không còn tốn thời gian, so được ở cùng B41. Số dev10 ở trên giữ nguyên giá trị (mask không đổi; chỉ runtime 4.6 là của bản cũ).

---

## 2026-10-18 — Quét p của RaPA trên C2-OCT-1V @ dev30 (user: thử mẫu nhỏ) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Loss C2-OCT-1V(-RaPA) mỗi bước (1 view):** v = nhiễu N(0, 6²) ∘ warp(adv; co giãn s = e^u, u ~ U(±ln2/2) quanh tâm ∘ dịch (dx, dy) ∈ {0..7}²); L = Σ_{4 stage} MSE(3·F_sạch(warp(sạch)), F(v)) + 4·(loss_rpn_cls + loss_cls) (GT co-transform); RaPA: mask Bernoulli(1 − p) trên weight + bias 53 BN backbone, mới mỗi view, chung cho sạch + adv. Cập nhật MI μ = 1, α = 1, ε = 5, B41, AMP, bỏ mask head.
**Đo nhanh tại δ = 0 (3 ảnh đầu dev300):** L_feat 12–18 vs 4·L_cls 0.3–2.3; tỉ lệ ‖∇L_feat‖₁ / ‖∇4·L_cls‖₁ = 2.66 / 0.68 / 0.49 (đổi theo ảnh); cos(∇L_feat, ∇L_cls) ≈ −0.08 (gần trực giao) → hai số hạng cạnh tranh từng pixel qua sign, trọng số 4 cho cls áp đảo ở ảnh nhiều object.
**Ô:** `C2-OCT-1V-RaPA005` (p = 0.005), `C2-OCT-1V-RaPA` (0.01), `C2-OCT-1V-RaPA02` (0.02); trước đó CLS-OCT p = 0.05 hại, 0.01 giúp (dev30, 3090). Chạy trong run `dev300_B41_eps5_limit100` với `--subset dev30` → so CẶP với C2-OCT-1V B41 + OSFD B50 sẵn có (cùng máy, cùng seed).
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 41 --methods C2-OCT-1V C2-OCT-1V-RaPA005 C2-OCT-1V-RaPA C2-OCT-1V-RaPA02 --out metrics_rapa_dev30.json
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --stage eval --methods OSFD --out metrics_rapa_dev30.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V C2-OCT-1V-RaPA005 C2-OCT-1V-RaPA C2-OCT-1V-RaPA02 --dets dets_rapa_dev30.json --out generalization_rapa_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods OSFD --dets dets_rapa_dev30.json --out generalization_rapa_dev30.json
python scripts/pair_screen.py --run dev300_B50_eps5_limit100 --subset dev30 --dets dets_rapa_dev30.json --tag rapa_dev30 --pairs \
  C2-OCT-1V-RaPA005@dev300_B41_eps5_limit100:C2-OCT-1V@dev300_B41_eps5_limit100 C2-OCT-1V-RaPA@dev300_B41_eps5_limit100:C2-OCT-1V@dev300_B41_eps5_limit100 \
  C2-OCT-1V-RaPA02@dev300_B41_eps5_limit100:C2-OCT-1V@dev300_B41_eps5_limit100 C2-OCT-1V@dev300_B41_eps5_limit100:OSFD
```
Log `results/octdev5/rapa_dev30.log`. **n = 30 → mô tả**; dấu hiệu "đáng xác nhận" (không phải quyết định): ΔCA3 hoặc ΔSwin-L ≥ 3 so C2-OCT-1V VÀ xu hướng theo p hợp lý (không đổi dấu ngẫu nhiên). Ô đáng xác nhận → khóa quy tắc @ dev100.

### Kết quả quét p @ dev30 (`results/ablation/rapa_dev30/screen.json`, log `results/octdev5/rapa_dev30.log`) — MÔ TẢ: **RaPA không giúp → bỏ**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: C2-OCT-1V 99.4 / 93.6 / 82.7 / 81.4 / 69.9 / 32.6 | 78.0; p = 0.005: 78.2, Swin-L 31.0; p = 0.01: 76.3, 30.0; p = 0.02: 78.1, 28.6; OSFD 98.2 / 91.4 / 82.5 / 80.8 / 73.7 / 22.6 | 79.0.
- Δ so C2-OCT-1V: CA3 +0.2 / −1.7 / +0.1 (mọi CI chứa 0); **Swin-L −1.6 / −2.6 / −4.0 — giảm đơn điệu theo p** (p = 0.02 CI [−7.1, +0.5]). Runtime 4.2–4.4 s/ảnh (bản đã sửa đếm GPU).
- Mô tả: trên 30 ảnh này C2-OCT-1V − OSFD CA3 −1.0 [−5.5, 7.7], Swin-L +10.0 [1.2, 16.8] (dev100: +2.6 / +8.8) → lợi Swin-L ổn định, CA3 dao động theo tập ảnh.
**Đọc:** RaPA trên BN backbone, khi mask dùng chung cho cả L_feature, không thêm gì cho CNN và hạ dần đúng thứ mà C2-OCT-1V mạnh (Swin-L). Không xác nhận dev100.

---

## 2026-10-18 — λ theo ảnh cho C2-OCT-1V @ dev30 (user: "xong quét p thì quét λ") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lý do:** λ = 4 chọn 1 lần bằng quy tắc "cân bằng gradient" (median mean|∇L_feat| / mean|∇L_cls| tại δ = 0 trên 5 ảnh dev100 = 3.95), nhưng tỉ lệ đổi mạnh theo ảnh (2.2–7.7; hôm nay 2.0–10.6) → λ cố định chỉ cân bằng ảnh trung vị. dev10: λ = 2 / 8 không khác rõ λ = 4.
**Ô `C2-OCT-1V-LA{05,1,2}`:** như C2-OCT-1V, trừ cls_weight = c · λ_ảnh, λ_ảnh = clip(mean|∇ₓL_feat| / mean|∇ₓL_cls|, 1, 16) đo tại δ = 0 trên ảnh KHÔNG biến đổi (`pia.calib_cls_weight`, AMP như attack), c ∈ {0.5, 1, 2}. Chi phí +2 backward / ảnh (B ghi 43). RNG torch khôi phục sau đo → chuỗi view trùng ô gốc. Stats ghi `cls_lambda`, `cls_ratio_raw`. Clip [1, 16] khai báo trước (bảo vệ ảnh không GT / gradient cls ≈ 0). **Kiểm (3 ảnh đầu dev300, 3 bước):** λ_ảnh 10.66 / 3.15 / 2.15 (khớp đo tay 10.6 / 2.7 / 2.0); chuỗi view (độ dời) trùng ô gốc ở cả 3 ảnh; L∞ đúng.
**Chạy** (cùng khung quét p: run `dev300_B41_eps5_limit100`, `--subset dev30`, so CẶP với C2-OCT-1V B41 + OSFD B50 sẵn có):
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 41 --methods C2-OCT-1V C2-OCT-1V-LA05 C2-OCT-1V-LA1 C2-OCT-1V-LA2 --out metrics_la_dev30.json
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --stage eval --methods OSFD --out metrics_la_dev30.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V C2-OCT-1V-LA05 C2-OCT-1V-LA1 C2-OCT-1V-LA2 --dets dets_la_dev30.json --out generalization_la_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods OSFD --dets dets_la_dev30.json --out generalization_la_dev30.json
python scripts/pair_screen.py --run dev300_B50_eps5_limit100 --subset dev30 --dets dets_la_dev30.json --tag la_dev30 --pairs <LA05 / LA1 / LA2 : C2-OCT-1V, C2-OCT-1V : OSFD>
```
Log `results/octdev5/la_dev30.log`. Dấu hiệu "đáng xác nhận" như quét p: ΔCA3 hoặc ΔSwin-L ≥ 3 so C2-OCT-1V VÀ xu hướng theo c hợp lý; mô tả thêm phân bố λ_ảnh.

### Kết quả λ theo ảnh @ dev30 (`results/ablation/la_dev30/screen.json`, log `results/octdev5/la_dev30.log`) — MÔ TẢ: **c là núm ĐÁNH ĐỔI CNN ↔ Swin-L, không phải đòn bẩy; λ theo ảnh (c = 1) ≈ λ = 4**
λ_ảnh thô (30 ảnh): trung vị 3.49, p10 1.90, p90 12.34, min 0.94 / max 20.06 (1 ảnh chạm mỗi biên clip). Runtime 3.85–3.96 s/ảnh (≈ ô gốc 3.94; +2 backward không thấy được).
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: C2-OCT-1V 99.4 / 93.6 / 82.7 / 81.4 / 69.9 / 32.6 | 78.0; **LA05** 99.3 / 93.2 / 83.4 / 84.9 / 73.2 / 30.4 | 80.5; **LA1** 99.1 / 93.0 / 81.2 / 79.2 / 74.6 / 32.6 | 78.4; **LA2** 99.2 / 93.8 / 80.2 / 79.8 / 64.5 / 34.3 | 74.8; OSFD | 79.0, Swin-L 22.6.
Δ so C2-OCT-1V: LA05 CA3 +2.5 [−1.6, 5.3], Swin-L −2.2; LA1 CA3 +0.4, Swin-L 0.0; LA2 CA3 **−3.2 [−6.8, −0.6]**, Swin-L +1.7.
**Đọc:** (1) Đơn điệu theo c: c nhỏ (nghiêng feature) → CA3 ↑ (YOLOX, Swin-T), Swin-L ↓; c lớn (nghiêng cls) → ngược lại. Khớp cơ chế "2 kênh cạnh tranh ngân sách ε qua sign" (cos ∇ ≈ −0.08). (2) Cân bằng theo TỪNG ảnh (c = 1) không thêm gì so λ = 4 cố định (trung vị λ_ảnh 3.5 ≈ 4) → tính thích nghi theo ảnh không phải đòn bẩy. (3) Không ô nào cải thiện cả hai phía (không dời được đường đánh đổi) → giữ λ = 4. Muốn vượt cần làm hai kênh BỔ TRỢ thay vì chia ngân sách (vd feature loss trọng số theo độ quan trọng của head cls).

---

## 2026-10-18 — ILPD trên đường cls (stage 3 / stride 16) cho C2-OCT-1V @ dev30 (user: "thử đi") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Nguồn:** ILPD (Li et al., NeurIPS 2023): trong forward, feature trung gian = F_sạch + γ·(F_adv − F_sạch), γ < 1 → buộc nhiễu ở tầng đó đủ lớn / bền để đầu ra vẫn hỏng. Rà literature 10-18: không thấy ILPD cho detector.
**Lý do chọn chỗ đặt:** E14 — thiệt hại DINO-Swin-L đi qua stage 3 (stride 16); R-a (thay C2 / C3 bằng sạch, γ = 0 ở tầng nông) trượt vì thiệt hại R50 nằm ở C4 / C5. Ở đây: chỉ C4 (stage index 2), chỉ đường head (L_cls); L_OSFD vẫn trên F_adv gốc; F_sạch = feature sạch co-transform của view (đã tính sẵn → 0 forward thêm).
**Ô `C2-OCT-1V-ILPD{05,075}`:** như C2-OCT-1V, γ ∈ {0.5, 0.75}. Code `repair.make_view_loss(ilpd_gamma, ilpd_stage=2)`, `geo_osfd_attack(ilpd_gamma)`.
**Kiểm (ảnh đầu dev300, view cố định):** γ = 1 trùng tuyệt đối loss cũ (13.452575); L_feat không đổi theo γ (13.03902); 4·L_cls 0.4136 → 0.4236 (γ = 0.5); attack 3 bước hữu hạn, L∞ đúng.
**Chạy:** cùng khung quét p / λ (run `dev300_B41_eps5_limit100`, `--subset dev30`, so CẶP với C2-OCT-1V B41 + OSFD B50), `--out metrics_ilpd_dev30.json`, `--dets dets_ilpd_dev30.json`, tag `ilpd_dev30`, log `results/octdev5/ilpd_dev30.log`. Dấu hiệu "đáng xác nhận": ΔCA3 hoặc ΔSwin-L ≥ 3 so C2-OCT-1V VÀ không đổi dấu vô lý giữa 2 mức γ; mô tả runtime.

### Kết quả ILPD @ dev30 (`results/ablation/ilpd_dev30/screen.json`, log `results/octdev5/ilpd_dev30.log`) — MÔ TẢ: **ILPD không giúp, γ nhỏ hại Swin-L → bỏ**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: C2-OCT-1V 99.4 / 93.6 / 82.7 / 81.4 / 69.9 / 32.6 | 78.0; **γ = 0.5** 99.6 / 93.8 / 83.6 / 81.0 / 65.2 / 24.5 | 76.6; **γ = 0.75** 99.4 / 93.1 / 81.6 / 79.9 / 70.9 / 30.8 | 77.5.
Δ so C2-OCT-1V: γ = 0.5 CA3 −1.4 [−3.1, 1.2], **Swin-L −8.1 [−10.5, +0.1]**, YOLOX −4.7; γ = 0.75 CA3 −0.5, Swin-L −1.8 (trong nhiễu).
**Đọc:** đơn điệu theo γ (γ càng nhỏ → Swin-L càng mất), khớp 2 lần quét trước: mọi thao tác làm nhỏ / làm nhòe tín hiệu cls trên R50 (RaPA, ILPD, λ nhỏ) đều lấy đi đúng phần lợi Swin-L. Giảm nhiễu ở C4 trên đường head buộc phần cls dựa vào thành phần C4 lớn của R50 — không phải thứ lọt qua DINO-Swin-L. Không xác nhận dev100.

---

## 2026-10-18 — Ablation co-transform feature sạch: C2-OCT-1V-noCT @ dev30 (user) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Câu hỏi:** OCT (co giãn 0.71–1.41×, dịch 0–7 px) có cần so feature adv với feature sạch ĐÃ warp cùng phép không? Tiền lệ: PIA-noCT sụp (dev30: CA3 36.6 vs PIA 79.5, R50 white-box 53.6); RRB biên độ nhỏ thì không cần (OSFD ≈ GEO-C0).
**Ô `C2-OCT-1V-noCT`:** như C2-OCT-1V, trừ L_feature so với F_sạch của ảnh sạch CHƯA biến đổi (tính 1 lần / ảnh, như OSFD gốc). Box GT cho L_cls VẪN co-transform → chỉ tách co-transform phía feature. Kiểm (ảnh đầu, 3 bước): chuỗi view + box giữ trùng ô gốc, forward_no_grad 0 (ô gốc 3), L∞ đúng. Rẻ hơn ô gốc (bỏ 1 forward sạch / bước) — cùng B41, KHÔNG cùng thời gian (đây là ablation, không phải ứng viên).
**Chạy:** khung như ILPD (run `dev300_B41_eps5_limit100`, `--subset dev30`, so CẶP với C2-OCT-1V B41 + OSFD B50), out `metrics_noct_dev30.json`, dets `dets_noct_dev30.json`, tag `noct_dev30`, log `results/octdev5/noct_dev30.log`. Mô tả: Δ so C2-OCT-1V trên 6 model + runtime.

### Kết quả C2-OCT-1V-noCT @ dev30 (`results/ablation/noct_dev30/screen.json`, log `results/octdev5/noct_dev30.log`) — MÔ TẢ: **KHÔNG sụp (khác PIA-noCT); ≈ ô gốc, nghiêng CNN ↔ Swin-L**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT-1V 99.4 / 93.6 / 82.7 / 81.4 / 69.9 / 32.6 | 78.0 | 3.63; **noCT 99.1 / 93.2 / 82.9 / 83.0 / 73.6 / 28.9 | 79.8 | 3.40**; OSFD 98.2 / 91.4 / 82.5 / 80.8 / 73.7 / 22.6 | 79.0.
Δ noCT − C2-OCT-1V: CA3 +1.8 [−3.3, 4.0], Swin-L −3.7 [−5.4, +0.9], YOLOX +3.7, R101 −0.5. noCT − OSFD: Swin-L +6.3 [−1.1, 13.8], CA3 +0.8.
**Đọc:** (1) Trái dự đoán (suy từ PIA-noCT), bỏ co-transform feature KHÔNG phá attack: white-box R50 99.1, CA3 ngang. Khác PIA-noCT ở 2 chỗ không tách được: biến đổi OCT nhẹ hơn PIA (không phối cảnh / xoay) VÀ ở đây có L_cls (box GT co-transform) giữ hướng đúng — PIA-noCT là loss feature thuần. (2) Mẫu giống quét λ: CNN / YOLOX nhích, Swin-L giảm → noCT chỉ đổi cân bằng 2 kênh, không phải đòn bẩy. (3) noCT rẻ hơn 6% (bỏ 1 forward sạch / bước). Giữ co-transform (Swin-L là lợi duy nhất đã xác nhận); claim "co-transform cần thiết" cho OCT KHÔNG được ủng hộ ở dev30 — paper phải nói đúng như vậy.

---

## 2026-10-18 — "Đổi hướng đánh": C2-OCT-1V-FIA / -POS / -POS-CAP @ dev30 B41 (user) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lý do:** mọi núm cường độ (λ, RaPA, ILPD, co-transform) chỉ trượt dọc đường đánh đổi CNN ↔ Swin-L. Thử đổi HƯỚNG: (1) cls chỉ đường cho feature; (2) dồn L_cls vào object thật / object còn đúng (kiểu DAG, Xie 2017).
**Ô (mọi thứ như C2-OCT-1V B41, trừ):**
- `C2-OCT-1V-FIA`: L_OSFD từng stage = mean(w_c · (3·F_sạch − F_adv)²), w = 0.5 + 0.5 · g_c / mean g, g_c = mean_hw |∂L_cls / ∂F_l| trung bình 4 view OCT của ảnh SẠCH (nhiễu σ = 6, GT co-transform). Theo KÊNH (bất biến warp). Dạng mềm vì NAA / RPA (họ FIA, thay hẳn loss) thua OSFD rõ trong paper OSFD. +4 backward / ảnh (B ghi 45) — KHÔNG cùng thời gian. RNG khôi phục → chuỗi view trùng ô gốc.
- `C2-OCT-1V-POS`: loss_cls RoI chỉ trên RoI dương (bỏ RoI nền), avg_factor = số RoI giữ (giữ độ lớn TB / RoI). loss_rpn_cls không đổi.
- `C2-OCT-1V-POS-CAP`: như POS + bỏ RoI dương có p(lớp đúng) < 0.1 (không đánh tiếp object đã hỏng trên R50).
**Kiểm:** w ≡ 1 trùng tuyệt đối loss cũ (13.495076); filter tắt sau dùng → loss trở lại đúng; chuỗi view trùng ô gốc (2 ảnh × 3 bước); w_max 1.5–2.9; **RoI dương chỉ 14–25% số RoI** (→ ~80% loss_cls hiện tại nằm trên RoI nền = đẩy nền thành object); CAP giữ 79–94% RoI dương sau 3 bước.
**Chạy:** khung như noCT (run `dev300_B41_eps5_limit100`, `--subset dev30`, so CẶP với C2-OCT-1V B41 + OSFD B50), out `metrics_dir_dev30.json`, dets `dets_dir_dev30.json`, tag `dir_dev30`, log `results/octdev5/dir_dev30.log`. Dấu hiệu "đáng xác nhận": ΔCA3 hoặc ΔSwin-L ≥ 3 so C2-OCT-1V VÀ phía kia không mất > 2 (tức RỜI khỏi đường đánh đổi); FIA nếu đạt phải xác nhận lại cùng thời gian (B ≈ 37).

### Kết quả FIA / POS / POS-CAP @ dev30 B41 (`results/ablation/dir_dev30/screen.json`, log `results/octdev5/dir_dev30.log`) — MÔ TẢ: **không ô nào rời đường đánh đổi; POS-CAP hại mạnh → manh mối cơ chế ngược với giả thuyết DAG**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT-1V 99.4 / 93.6 / 82.7 / 81.4 / 69.9 / 32.6 | 78.0 | 3.9; **FIA** 99.4 / 94.5 / 82.0 / 81.6 / 72.4 / 30.7 | 78.7 | 4.4; **POS** 99.9 / 91.6 / 79.8 / 79.1 / 71.9 / 30.6 | 76.9 | 3.9; **POS-CAP** 99.9 / 91.5 / 77.4 / 73.5 / 62.3 / 20.9 | 71.0 | 3.8.
Δ so C2-OCT-1V: FIA CA3 +0.7, Swin-L −1.9 (nhiễu); POS CA3 −1.1, Swin-L −2.1 (nhiễu); **POS-CAP CA3 −7.0 [−9.9, −2.3], Swin-L −11.7 [−14.9, −6.0]**, ConvNeXt −5.3, Swin-T −7.9.
Thống kê RoI (30 ảnh × 41 bước): RoI dương chỉ 11% số RoI lấy mẫu; POS-CAP giữ trung bình 32% RoI dương (68% đã p(đúng) < 0.1 trên R50).
**Đọc:** (1) FIA (kênh, mềm) ≈ ô gốc — đánh trọng số feature theo head R50 không đổi hướng đánh. (2) POS ≈ ô gốc dù bỏ 89% RoI (nền) → phần loss_cls "đẩy nền thành box giả" KHÔNG phải nguồn lợi cũng không phải lãng phí đáng kể; tín hiệu hữu ích nằm ở RoI dương. (3) **POS-CAP hại mạnh mọi target, nhất là Swin-L**: ngừng đánh object khi R50 đã sai (p < 0.1) làm mất transfer → lợi của L_cls đến từ việc TIẾP TỤC đẩy object đã sai trên R50 đi XA hơn biên quyết định của surrogate (vượt biên / biên độ lớn). Ngược với giả thuyết "dồn sức cho object còn đúng" (DAG). Hệ quả cho hướng sau: thứ đáng thử là ĐẨY XA HƠN (loss không bão hòa / biên lớn), không phải phân bổ lại. Không xác nhận dev100 cho 3 ô này.

---

## 2026-10-18 — C2-OCT-1V vs OSFD @ dev100: phân rã lỗi (mất / thêm / lệch / đổi lớp) + hình 10 ảnh (user) — MÔ TẢ, không GPU
**Công cụ:** `scripts/viz_dets.py` nhận method `TÊN@RUN` (OSFD từ B50, C2-OCT-1V từ B41), `--draw N` (vẽ N ảnh đầu, thống kê mọi ảnh), thêm score_adv / score_sạch TB của box "giữ". Hình: `results/figs/viz_c2oct1v_vs_osfd_dev100/` (10 ảnh = dev10). Phân rã AP: `scripts/probe_error_types.analyse` → `results/mechanism/error_types_c2oct1v_vs_osfd_dev100.json`.
**AP + phân rã (dev100; AP_noFP = AP khi bỏ mọi FP → phần còn lại phản ánh mất GT / thứ hạng TP; "mất do FP" = AP_noFP − AP):**
| model | cond | AP | AP_noFP | mất do FP | recall@0.3 | FP / ảnh (nền / lệch / nhầm lớp) |
|---|---|---|---|---|---|---|
| DINO-Swin-L | sạch | 64.0 | 73.1 | 9.0 | 0.75 | 3.0 (1.3 / 1.3 / 0.1) |
| DINO-Swin-L | OSFD | 49.1 | 59.6 | 10.5 | 0.58 | 3.6 (1.9 / 1.2 / 0.2) |
| DINO-Swin-L | C2-OCT-1V | 43.5 | 56.9 | 13.4 | 0.55 | 4.7 (2.6 / 1.5 / 0.2) |
| Swin-T | OSFD / C2-OCT-1V | 12.9 / 10.8 | 21.9 / 19.7 | 9.1 / 8.9 | 0.30 / 0.31 | 23.7 / 27.4 |
| ConvNeXt-T | OSFD / C2-OCT-1V | 10.1 / 8.8 | 16.3 / 15.8 | 6.2 / 7.0 | 0.27 / 0.28 | 29.4 / 32.6 |
| YOLOX-S | OSFD / C2-OCT-1V | 14.2 / 13.7 | 23.1 / 23.4 | 8.9 / 9.8 | 0.23 / 0.27 | 11.2 / 13.3 |
So với box SẠCH của chính model (score ≥ 0.3, 100 ảnh; giữ / mất / thêm / lệch / đổi lớp): DINO-Swin-L OSFD 698 / 362 / 158 / 78 / 15 vs C2-OCT-1V 710 / 350 / 221 / 92 / 15; Swin-T 431 / 1044 / 1961 vs 616 / 859 / 2083; ConvNeXt 405 / 1125 / 2523 vs 601 / 929 / 2534.
**Đọc:**
1. **DINO-Swin-L (AP 49.1 → 43.5, −5.6):** ~½ từ FP (mất do FP 10.5 → 13.4, +2.9; FP / ảnh 3.6 → 4.7, chủ yếu nền + lệch), ~½ từ TP yếu đi (AP_noFP 59.6 → 56.9, −2.7; recall@0.3 0.58 → 0.55). Số box sạch bị XÓA gần như như nhau (350 vs 362). → Lợi của C2-OCT-1V trên Swin-L KHÔNG phải xóa thêm object, mà là THÊM box sai (nhỏ, thường sát object thật — hình 31118, 42563, 19109) + hạ thứ hạng / confidence box đúng. Đính chính entry 10-17 ("DINO chặn box giả, chỉ kênh hạ confidence lọt"): box giả CÓ lọt một phần và gánh ~½ lợi.
2. **CNN / Swin-T:** C2-OCT-1V GIỮ nhiều box sạch hơn OSFD (Swin-T 616 vs 431) nhưng AP vẫn thấp hơn chút nhờ thêm FP (Swin-T 27.4 vs 23.7 / ảnh); recall ≈ nhau. Hình: FP của OSFD là box LỚN phủ nền / bầu trời; FP của C2-OCT-1V nhỏ hơn, dày, mang nhãn object ("chair", "person", "bottle") — dấu của loss cls (đẩy RoI sang lớp object).
3. YOLOX: C2-OCT-1V recall cao hơn OSFD (0.27 vs 0.23) → xóa object KÉM hơn, bù bằng FP.
**Hệ quả:** C2-OCT-1V đổi kiểu thiệt hại từ "xóa object" (OSFD) sang "chen box giả cỡ object + làm yếu box đúng". Với DINO (matching 1–1, ít NMS) kênh box giả cỡ object lọt được một phần. n = 100, 1 lần chạy.

---

## 2026-10-18 — Hồ sơ lỗi theo model: Mask R-CNN Swin-T (dễ phá) vs DINO-Swin-L (kháng), OSFD vs C2-OCT-1V @ dev100 (user) — MÔ TẢ, không GPU
**Công cụ:** `scripts/err_profile.py` → `results/mechanism/err_profile_dev100.json`. GT 982 (nhỏ 446 / vừa 362 / lớn 174). Recall@0.3 theo cỡ GT; conf = TB score_adv / score_sạch của GT phát hiện ở cả hai; FP (score ≥ 0.3) theo vị trí / cỡ (√diện tích tương đối ảnh) / score.
| model | cond | recall nhỏ / vừa / lớn | conf | FP / ảnh | FP sát object / nền | FP score ≥ 0.5 |
|---|---|---|---|---|---|---|
| Swin-T | sạch | 0.60 / 0.86 / 0.95 | – | 7.0 | 3.8 / 3.3 | 3.4 |
| Swin-T | OSFD | 0.25 / 0.35 / 0.31 | 0.83 | 23.7 | 6.3 / 17.3 | 12.6 |
| Swin-T | C2-OCT-1V | 0.20 / 0.40 / 0.45 | 0.84 | 27.4 | 6.4 / 21.0 | 18.2 |
| DINO-Swin-L | sạch | 0.60 / 0.84 / 0.92 | – | 3.0 | 1.6 / 1.3 | 0.3 |
| DINO-Swin-L | OSFD | 0.44 / 0.63 / 0.82 | 0.91 | 3.6 | 1.7 / 1.9 | 0.4 |
| DINO-Swin-L | C2-OCT-1V | 0.37 / 0.64 / 0.86 | 0.89 | 4.7 | 2.1 / 2.6 | **1.1** |
(R101 / ConvNeXt / YOLOX cùng mẫu Swin-T: FP nền 9–31 / ảnh, phần lớn score ≥ 0.5.)
**Đọc:**
1. **Hai lớp kháng của DINO-Swin-L:** (a) object LỚN gần như miễn nhiễm (recall 0.92 → 0.82–0.86; Swin-T 0.95 → 0.31–0.45); vừa mất ~¼; nhỏ mất nhiều nhất. Khớp E4 / E6 (Swin-L nhận object lớn bằng hình dạng). (b) **FP gần như không phát sinh**: Mask R-CNN Swin-T bị ngập 24–27 FP / ảnh (đa số nền, score cao — RPN + RoI head nhận nhiễu thành object), DINO chỉ 3.6–4.7, đa số score thấp (0.3–0.5). (c) Box đúng còn sống giữ confidence (0.89–0.91 vs CNN 0.80–0.85).
2. **Lợi của C2-OCT-1V trên Swin-L đến từ:** object NHỎ (0.44 → 0.37) + FP score cao (0.4 → 1.1 / ảnh, ×2.75). Object lớn còn bị phá KÉM hơn OSFD (0.86 vs 0.82) — tái lập E9 (dev30, C2 − OSFD lớn −0.16) ở dev100. Trên Swin-T / CNN cùng mẫu: C2-OCT-1V phá nhỏ mạnh hơn, lớn yếu hơn OSFD.
3. Khoảng trống còn lại trên Swin-L chủ yếu là object VỪA / LỚN: trong 543 GT DINO-Swin-L VẪN phát hiện được sau C2-OCT-1V, 70% là vừa / lớn (nhỏ 164 / vừa 230 / lớn 149) — đúng kênh mà mọi thử "tần thấp / quy mô lớn" (S1, SEA, LVX, BG-OCT) đã thất bại.

---

## 2026-10-18 — Hướng A: box MỒI cỡ object (chế tạo có chủ đích) C2-OCT-1V-FAB4 / FAB8 @ dev30 B41 (user) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lý do:** hồ sơ lỗi dev100 — lợi C2-OCT-1V trên DINO-Swin-L ~½ từ FP score cao (0.4 → 1.1 / ảnh), FP nhỏ, cỡ object; DINO chặn FP nền lớn của OSFD. L_cls hiện tại đã có "chế tạo" vô hướng (ascent trên RoI nền) nhưng POS cho thấy phần đó không quan trọng → thử chế tạo CÓ CHỦ ĐÍCH: vị trí + lớp cố định suốt attack.
**Ô `C2-OCT-1V-FAB{4,8}`:** như C2-OCT-1V, thêm − μ·L_fab (descent), μ ∈ {4, 8}. Mỗi ảnh 3 box mồi cố định (khung model): cỡ = cỡ 1 GT thật ngẫu nhiên, vị trí đều, IoU < 0.1 với GT + mồi khác, nhãn đều 80 lớp; RNG riêng (seed + 5) → chuỗi view trùng ô gốc; co-transform cùng GT (gộp gt_instances, nhãn + 1000). L_fab = loss_rpn_cls (objectness tại mồi) + loss_cls RoI head trên 32 proposal rung quanh mỗi mồi (σ = 0.1 cạnh), CHỈ mẫu dương (`roi_cls_filter(head=)`), dùng lại feature FPN của forward hiện có. Code: `repair.split_fab / fab_term / make_view_loss(fab_weight)`, `pia.geo_osfd_attack(fab_k, fab_weight)`; stats ghi box mồi (ảnh gốc) + nhãn.
**Kiểm thao tác (white-box R50, 4 ảnh × B41):** bản đầu (mọi mẫu) ép mọi chỗ khác thành nền → R50 det 78 → 32, mồi 1 / 6 → sửa chỉ mẫu dương. Bản cuối: mồi phát hiện đúng lớp (score ≥ 0.3, IoU ≥ 0.5) 4 / 12 (cả μ = 4 và 8), bất kỳ lớp 6–7 / 12; L∞ 5; runtime 4.5–5.7 s (~1.15× ô gốc → nếu đạt phải so lại cùng thời gian).
**Chạy:** khung như các ô dev30 trước (run `dev300_B41_eps5_limit100`, `--subset dev30`, so CẶP với C2-OCT-1V B41 + OSFD B50), out `metrics_fab_dev30.json`, dets `dets_fab_dev30.json`, tag `fab_dev30`, log `results/octdev5/fab_dev30.log`. Mô tả thêm: tỉ lệ mồi được phát hiện trên TỪNG target (chuyển giao của chế tạo). Dấu hiệu "đáng xác nhận": ΔSwin-L ≥ 3 so C2-OCT-1V VÀ ΔCA3 ≥ −2.

### Kết quả FAB @ dev30 (`results/ablation/fab_dev30/screen.json`, `results/mechanism/fab_check_dev30.json`, log `results/octdev5/fab_dev30.log`) — MÔ TẢ: **HẠI mọi target, đơn điệu theo μ; mồi KHÔNG chuyển giao → bỏ hướng A**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT-1V 99.4 / 93.6 / 82.7 / 81.4 / 69.9 / 32.6 | 78.0 | 3.9; **FAB4** 97.8 / 83.8 / 76.8 / 73.6 / 66.0 / 24.7 | 72.1 | 4.7; **FAB8** 96.1 / 83.0 / 72.7 / 72.9 / 59.1 / 22.5 | 68.2 | 4.8.
Δ so C2-OCT-1V: FAB4 CA3 −5.9 [−11.4, −3.1], Swin-L −7.9 [−13.4, −0.1], R101 −9.8; FAB8 CA3 −9.8 [−16.3, −6.9], Swin-L −10.1 [−15.6, −2.2].
Mồi được phát hiện đúng lớp (score ≥ 0.3, IoU ≥ 0.5; mốc sạch / C2-OCT-1V đều 0.00): R50 0.31–0.32 (white-box), R101 0.10, ConvNeXt 0.04–0.07, Swin-T 0.02–0.03, YOLOX 0.01, **DINO-Swin-L 0.00**.
**Đọc:** (1) Chế tạo object CÓ CHỦ ĐÍCH chỉ thành công một phần trên chính R50 (~⅓ mồi) và gần như KHÔNG chuyển giao — giảm dần theo khoảng cách họ (R101 > ConvNeXt > Swin-T > YOLOX > DINO = 0). (2) Ngân sách δ dành cho mồi lấy từ phần phá vô hướng → mọi target yếu đi, đơn điệu theo μ. (3) Hệ quả cho hồ sơ lỗi: FP score cao của C2-OCT-1V trên Swin-L KHÔNG phải "object giả có hình dạng / lớp ổn định" mà là sản phẩm phụ của việc làm hỏng object thật (sát object, lệch) — muốn tăng kênh đó phải đi qua phá object thật, không qua chế tạo riêng. Không xác nhận dev100.

---

## 2026-10-18 — WB-probe: white-box DINO-Swin-L vs surrogate R50 @ dev10 (user: "tìm thêm manh mối") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Câu hỏi mới so với E8 (E8 chỉ có AP drop + suppression theo cỡ):** (1) attack phá ĐƯỢC DINO-Swin-L (white-box) phá theo kiểu nào — recall theo cỡ, box giả (FP / ảnh, FP score cao), confidence box còn sống — so với C2-OCT-1V / OSFD từ R50; (2) chiều NGƯỢC: δ sinh trên DINO-Swin-L hại R50 / CNN / Swin-T bao nhiêu (bất đối xứng chuyển giao?).
**Ô (surrogate = DINO-Swin-L, ngoài threat model, chỉ phân tích):** `WB-OSFD` (công thức OSFD k = 3, MI, RRB, B50, **AMP fp16** — E8 fp32 trên 3090 không vừa V100: đo 0.33 s / bước, 11.2 GB); `WB-CLS` (số hạng phân loại DINO như E8 WB-s1, B50, checkpointing: 1.05 s / bước, 9.8 GB). Mốc R50: OSFD B50, C2-OCT-1V B41 (PNG có sẵn, cùng 10 ảnh). Seed theo chỉ số dev300.
**Script:** `scripts/wb_probe.py` → PNG `artifacts/wb_probe/`, kết quả `results/mechanism/wb_probe_dev10.json`, log `results/e15/wb_probe.log`. n = 10 → chỉ sinh giả thuyết.

### Kết quả WB-probe @ dev10 (`results/mechanism/wb_probe_dev10.json`, log `results/e15/wb_probe.log`, PNG `artifacts/wb_probe/`) — MÔ TẢ (n = 10; GT nhỏ / vừa / lớn = 88 / 44 / 10)
AP drop %, R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L:
- OSFD (R50): 99.4 / 84.3 / 77.3 / 69.5 / 61.1 / 27.9; C2-OCT-1V (R50): 99.6 / 89.8 / 83.3 / 75.5 / 72.1 / 34.8
- **WB-OSFD (DINO-Swin-L): 31.4 / 24.0 / 42.1 / 53.2 / 32.7 / 77.1**; **WB-CLS (DINO-Swin-L): 7.6 / −0.6 / 17.3 / 24.1 / 6.5 / 91.2**
Hồ sơ lỗi DINO-Swin-L (recall nhỏ / vừa / lớn | conf | FP / ảnh | FP score ≥ 0.5): sạch 0.57 / 0.82 / 1.00 | – | 5.1 | 0.7; OSFD 0.40 / 0.73 / 1.00 | 0.90 | 4.7 | 0.4; C2-OCT-1V 0.36 / 0.73 / 0.90 | 0.90 | 5.8 | 1.2; **WB-OSFD 0.15 / 0.30 / 0.50 | 0.83 | 7.2 | 1.1**; **WB-CLS 0.06 / 0.09 / 0.50 | 1.04 | 37.2 | 27.3**.
**Đọc (sinh giả thuyết):**
1. **Bất đối xứng KHÔNG có — khoảng cách họ backbone là HAI CHIỀU:** δ sinh trên DINO-Swin-L chuyển sang R50 / R101 / YOLOX chỉ 24–33% (WB-OSFD) hay ~0–8% (WB-CLS) — ngang hoặc tệ hơn chiều R50 → Swin-L (28–35%). Chuyển tốt nhất sang Swin-T (53.2), rồi ConvNeXt (42.1) → thứ tự theo khoảng cách họ, cả hai chiều. Kết quả trực tiếp cho RQ1 / RQ2: gap không phải vì Swin-L "bền", mà vì hướng nhạy của hai họ gần như rời nhau.
2. **DINO KHÔNG miễn nhiễm box giả:** WB-CLS tạo 37 FP / ảnh (27 score ≥ 0.5) — ngang mức Mask R-CNN bị R50 phá. DINO chặn FP từ δ của R50 vì δ đó không tạo được feature "giống object" trong không gian Swin-L, không phải vì kiến trúc (khớp FAB: mồi 0% trên DINO).
3. **Object lớn KHÔNG phải bức tường cơ bản:** WB-OSFD hạ recall lớn 1.00 → 0.50, vừa 0.82 → 0.30 (C2-OCT-1V: 0.90 / 0.73) — nhưng n lớn = 10, chỉ là dấu hiệu.
4. Hai kênh white-box khác nhau: WB-OSFD xóa object (recall ↓ mọi cỡ, FP ít) — đúng kiểu OSFD làm với CNN; WB-CLS làm ngập FP (box sống vẫn tự tin, conf 1.04). Loss feature thuần chuyển ngược sang CNN tốt hơn hẳn loss cls (31 vs 8 trên R50) — cùng chiều với phía R50 (feature giữ CNN / YOLOX, cls giúp riêng Swin-L).

---

## 2026-10-18 — Embedding mức object: C2-OCT-1V-OBJ15 / OBJ45 @ dev30 B41 (user) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lý do (từ WB-probe + POS-CAP):** chỉ loss mức object (L_cls qua RoI-pool) chạm được Swin-L; Swin-L nhận object bằng hình dạng toàn cục; đẩy xa hơn biên giúp. L_cls bị trói vào nhãn → thử khoảng cách thuần trên embedding object, không nhãn.
**Ô:** C2-OCT-1V + w · L_obj, L_obj = MSE(e_sạch, e_adv) với e = embedding RoI 1024-d (RoIAlign 7×7 FPN → 2 fc + ReLU của Shared2FCBBoxHead, `repair.roi_embed`) tại box GT của view; e_sạch qua neck(feature backbone sạch co-transform), no_grad. w ∈ {15, 45}.
**Thiết kế đổi trước khi chạy:** bản k = 3 (kiểu OSFD) cho ∇L_obj NGƯỢC ∇(4·L_cls) (cos −0.56 / −0.13 / −0.53 / +0.12 / −0.22, 5 ảnh đầu dev100, δ = 0) → dùng k = 1 (khoảng cách thuần; cos −0.40 … +0.55, không xung đột hệ thống). w = 15 = trung vị |∇4·L_cls| / |∇L_obj| (k = 1, tại δ ngẫu nhiên ±5; 8.8 / 14.5 / 15.0 / 56.8 / 86.2); 45 = 3×.
**Kiểm:** chuỗi view trùng ô gốc; L∞ 5; 10 bước: δ lệch ô gốc 0.266 (w = 15) / 0.334 (w = 45) pixel vs sàn không tất định 0.100; thời gian ≈ ô gốc.
**Chạy:** khung như các ô dev30 trước, out `metrics_obj_dev30.json`, dets `dets_obj_dev30.json`, tag `obj_dev30`, log `results/octdev5/obj_dev30.log`. Dấu hiệu "đáng xác nhận": ΔSwin-L ≥ 3 so C2-OCT-1V VÀ ΔCA3 ≥ −2 (rời đường đánh đổi).

### Kết quả OBJ @ dev30 (`results/ablation/obj_dev30/screen.json`, log `results/octdev5/obj_dev30.log`) — MÔ TẢ: **không rời đường đánh đổi; Swin-L không tăng → bỏ**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT-1V 99.4 / 93.6 / 82.7 / 81.4 / 69.9 / 32.6 | 78.0 | 3.9; **OBJ15** 99.9 / 94.6 / 84.6 / 82.9 / 72.1 / 30.1 | 79.9 | 4.1; **OBJ45** 99.7 / 93.8 / 82.7 / 76.1 / 64.5 / 29.8 | 74.4 | 3.9.
Δ so C2-OCT-1V: OBJ15 CA3 +1.9 [−4.7, 3.9], Swin-L −2.5 [−5.7, 0.9]; OBJ45 CA3 −3.6 [−9.0, 0.3], Swin-L −2.8 [−8.2, 0.3].
**Đọc:** w = 15 cùng mẫu quét λ / noCT (CNN nhích, Swin-L lùi, trong nhiễu); w = 45 hại CNN, Swin-L vẫn lùi. Làm lệch embedding RoI của R50 (không nhãn) KHÔNG chạm Swin-L như L_cls — phần lợi Swin-L của L_cls gắn với việc đẩy KHỎI LỚP ĐÚNG (tín hiệu ngữ nghĩa lớp), không phải với mức object nói chung. Khớp WB-probe: hướng nhạy của Swin-L nằm ngoài không gian R50; đổi thước đo trong R50 không mở được hướng mới. Dấu hiệu "đáng xác nhận" không đạt.

---

## 2026-10-18 — Sàng lọc dev300: C2-OCT-1V B41 vs OSFD B50 (cùng thời gian, V100) — chốt C2-OCT-1V (user); quy tắc khóa TRƯỚC khi chạy
**Ô:** OSFD B50; C2-OCT-1V B41 (= L_OSFD + 4·(loss_rpn_cls + loss_cls), GT co-transform, OCT 1 view / bước: co giãn ±½ octave + dịch 0–7 px + nhiễu σ = 6, MI μ = 1, α = 1, ε = 5, skip_mask + AMP). User bỏ đối chứng OSFD-1V (lần này) → KHÔNG tách được lợi do "1 view / nhiều bước hơn" khỏi lợi do loss cls + OCT; ghi rõ khi báo cáo.
**Dữ liệu:** dev300 (300 ảnh), V100, mốc OSFD sinh lại cùng máy. 100 ảnh dev100 tái dùng (PNG + stats chép từ run `_limit100`, cùng seed theo chỉ số dev300, cùng máy) → sinh thêm 200 ảnh mỗi ô. Run `dev300_B50_eps5_limit300`, `dev300_B41_eps5_limit300`; out `metrics_screen_dev300.json`, dets `dets_screen_dev300.json`; `pair_screen --tag screen_c2oct1v_dev300 --exclude dev100`. Chạy trong tmux session `screen300` (`results/screen_dev300/run.sh`, log `results/screen_dev300/run.log`).
**Quy tắc (paired bootstrap 1000, BOOT_SEED 2026; Δ = drop(C2-OCT-1V) − drop(OSFD); CA3 = TB ConvNeXt-T / Swin-T / YOLOX-S):**
- **QUA-CA3** (bước 3 plan_sota): ΔCA3 ≥ 3 VÀ cận dưới CI > 0 VÀ Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S VÀ ΔR101 ≥ −3.
- **QUA-SwinL** (nhánh khai báo ở dev100): ΔSwin-L ≥ 3 VÀ cận dưới CI > 0 VÀ ΔCA3 ≥ 0 (điểm) VÀ ΔR101 ≥ −3.
- Điều kiện chung: runtime ≤ 1.1 × OSFD (TB 300 ảnh, cùng máy).
- QUA-CA3 → GO @ n300 theo plan_sota bước 4 (khóa entry riêng). Chỉ QUA-SwinL → "GO có điều kiện": claim chỉ về target Transformer khó (DINO-Swin-L), user quyết có đi n300 không. Không qua nhánh nào → NO-GO cho C2-OCT-1V; ghi log, user quyết.
- Mô tả: dev300 \ dev100 (200 ảnh mới — kiểm dev100 có lạc quan không); R101, R50; hồ sơ lỗi nếu cần.
**Kỳ vọng ghi trước (dev100):** ΔCA3 +2.6 [−2.1, 6.8], ΔSwin-L +8.8 [2.0, 11.3] → dự đoán trượt QUA-CA3, có khả năng QUA-SwinL.

### Kết quả sàng lọc dev300 (V100; `results/ablation/screen_c2oct1v_dev300/screen.json`, `results/runs/dev300_B{50,41}_eps5_limit300/`, log `results/screen_dev300/run.log`) — theo quy tắc: **QUA-CA3 (và QUA-SwinL) → GO @ n300 (khóa entry riêng)**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh (TB 300 ảnh): OSFD B50 98.2 / 93.2 / 78.9 / 74.8 / 68.0 / 23.5 | 73.9 | 3.87; **C2-OCT-1V B41 98.7 / 95.7 / 86.9 / 80.6 / 75.2 / 32.4 | 80.9 | 3.80**.
Δ (C2-OCT-1V − OSFD) [CI 95%]: **CA3 +7.0 [3.4, 8.1]**; ConvNeXt +8.0 [4.2, 9.8], Swin-T +5.8 [2.5, 7.4], YOLOX +7.2 [1.7, 9.2]; **Swin-L +9.0 [4.9, 11.3]**; R101 +2.4 [0.2, 4.0].
- QUA-CA3: ΔCA3 7.0 ≥ 3 ✓, cận dưới 3.4 > 0 ✓, Δ > 0 cả 3 target ✓, ΔR101 +2.4 ≥ −3 ✓. QUA-SwinL: +9.0 ≥ 3, cận dưới 4.9 > 0 ✓. Runtime 3.80 ≤ 1.1 × 3.87 = 4.26 ✓.
- 200 ảnh mới (dev300 \ dev100): CA3 +5.9 [2.3, 7.8], Swin-L +7.8 [3.1, 10.4] → lợi lặp lại ngoài dev100; dev100 KHÔNG lạc quan (ngược lại, CA3 dev100 +2.6 thấp hơn).
- Sanity mốc: OSFD dev300 V100 CA3 73.9 / Swin-L 23.5 vs commit 3090 (`metrics_c2pia_dev300`) 73.5 / 24.9 — khớp.
**Đọc:** ở cùng thời gian OSFD, C2-OCT-1V vượt OSFD trên MỌI target khác họ với CI > 0, kể cả DINO-Swin-L (+9). Kỳ vọng ghi trước (trượt CA3) SAI — CA3 ở dev100 thấp do nhiễu tập nhỏ. Caveat khai báo: không có đối chứng OSFD-1V → chưa tách lợi "1 view / bước" khỏi "loss cls + OCT"; novelty thấp (OSFD + DI/TI-style + loss cls); GT-assisted.
**Bước tiếp (plan_sota bước 4):** GO @ n300 — khóa entry riêng (1 lần, tiêu chí GO khóa sẵn: ΔCA3 ≥ 3 VÀ CI > 0; Δ > 0 cả 3 target VÀ ≥ 2/3 target ≥ 3; ΔR101 ≥ −3). Cần mốc OSFD n300 cùng máy (PNG HF là 3090 → sinh lại). n300 ∩ dev300 = ∅ (kiểm trước khi chạy).

---

## 2026-10-18 — TẠM DỪNG (user, mai chạy tiếp): tổng kết phiên + việc mở
**Máy:** V100 16GB (máy mới hôm nay, bootstrap lại). Mốc OSFD sinh lại cùng máy ở mọi so sánh. Không có tiến trình nào đang chạy (tmux `screen300` đã xong).
**Kết quả chính (theo quy tắc khóa):**
1. **C2-OCT-1V B41 (cùng thời gian OSFD B50) — QUA sàng lọc dev300:** CA3 +7.0 [3.4, 8.1], ConvNeXt +8.0, Swin-T +5.8, YOLOX +7.2, **DINO-Swin-L +9.0 [4.9, 11.3]**, R101 +2.4, runtime 0.98×; lặp lại trên 200 ảnh ngoài dev100 (CA3 +5.9, Swin-L +7.8). dev100 (CA3 +2.6 CI chứa 0) là bi quan do nhiễu tập nhỏ. C2-PIA-1V B45 cũng GO dev100 nhưng thua C2-OCT-1V (ΔCA3 2.0 vs 2.6, sát 1.1× thời gian) → không đi tiếp.
2. **Mọi tinh chỉnh quanh C2-OCT-1V (dev10 / dev30, mô tả) đều không rời đường đánh đổi CNN ↔ Swin-L:** λ = 2 / 8; λ theo ảnh c ∈ {0.5, 1, 2} (đơn điệu: c nhỏ → CA3 ↑ Swin-L ↓); RaPA p ∈ {0.005, 0.01, 0.02} (Swin-L ↓ theo p); ILPD stage 3 γ ∈ {0.5, 0.75} (γ nhỏ hại Swin-L); noCT (không sụp, ≈ ô gốc); FIA kênh (≈); POS (≈, bỏ 89% RoI nền không mất gì); POS-CAP (hại mạnh — lợi cls đến từ việc đẩy tiếp object ĐÃ sai đi xa biên); box mồi FAB (hại, mồi 0% trên DINO); embedding object OBJ (≈ / hại).
3. **Cơ chế (mô tả):** gap họ backbone là HAI CHIỀU (WB-probe dev10: δ sinh trên DINO-Swin-L sang R50 chỉ 31% / 8%); lợi C2-OCT-1V trên Swin-L ~½ từ FP score cao sát object + ½ từ TP yếu đi, chủ yếu object nhỏ; lỗ hổng còn lại = object vừa / lớn. Loss cls và loss feature gần trực giao (cos ∇ ≈ −0.08), tỉ lệ độ lớn đổi 5× theo ảnh.
4. Literature (web): không có attack OD mới khớp threat model vượt OSFD; **BFDA (ICASSP'24, cùng nhóm OSFD) là rủi ro baseline lớn nhất** — không code, không có PDF trong repo. ε: OSFD / AugTrans đều 5/255 (OSFD gốc 200 bước, L∞ thực 7 theo benchmark; AugTrans 160 × 10 EOT); setup ta chặt hơn (init 0, assert L∞ ≤ 5).
5. Sửa code: RaPA đếm mask trên GPU (bỏ 106 lần đồng bộ / view, mask trùng tuyệt đối).
**Việc mở (thứ tự đề xuất cho phiên sau):**
- (a) **Đường bão hòa theo B của C2-OCT-1V** @ dev100: 1 lần B100, chụp δ ở B 20 / 41 / 65 / 100 (`on_step`), ~20 phút. Còn dốc ở B41 → cải thiện bằng tăng tốc (bỏ forward sạch / cache) để thêm bước cùng thời gian; bão hòa → biết trần. Khóa quy tắc trước khi chạy.
- (b) **L_cls dạng margin (kiểu CW) trên RoI dương** (z_đúng − max z_khác, không chặn, 1 lớp sai mạnh nhất) @ dev30 B41, ~10 phút — cơ sở: POS-CAP (đẩy xa biên có lợi), POS (RoI nền không quan trọng). Rủi ro: lại chỉ là núm cường độ.
- (c) Nếu (a), (b) không ra gì → coi C2-OCT-1V gần trần của nhóm "1 surrogate R50"; **GO @ n300** (khóa entry riêng, chạy 1 lần; sinh lại OSFD n300 trên V100 — PNG HF là 3090; đề xuất kèm eval held-out YOLOX-L + Mask R-CNN Swin-S và đối chứng OSFD-1V). dev300 ∩ n300 = ∅ (đã kiểm).
- (d) Rà BFDA: cần PDF ICASSP'24 (IEEE Xplore / email tác giả) để quyết cài lại.
**Code mới phiên này (mặc định giữ hành vi cũ):** ô `C2-OCT-1V-{w2,w8,RaPA,RaPA005,RaPA02,LA05,LA1,LA2,ILPD05,ILPD075,noCT,FIA,POS,POS-CAP}` (+ FAB / OBJ của phiên song song); `pia.calib_cls_weight`, `pia.fia_channel_weights`, `geo_osfd_attack(cls_auto, ilpd_gamma, fia_views, roi_pos, roi_cap)`, `rapa_cls_p` cho phép với loss feature (scope backbone); `repair.roi_cls_filter`, `make_view_loss(ilpd_gamma, feat_chw, roi_ctx)`.
**Artifact (V100) ĐÃ upload HF** (`scripts/sync_artifacts.py upload`, dataset private `congdanh99/transfer-attack`): `runs/dev300_B50_eps5_limit{10,100,300}.tar`, `runs/dev300_B41_eps5_limit{10,100,300}.tar`, `runs/dev300_B45_eps5_limit100.tar` (sha256 trong commit message HF). Tải lại: `python scripts/sync_artifacts.py download dev300_B41_eps5_limit300` (cần HF_TOKEN). KHÔNG trộn với tar 3090 cũ (`dev300_B50_eps5.tar`, …). Chưa upload: `artifacts/wb_probe/` (17 MB), `artifacts/e15/`. `bootstrap.sh` sửa: đọc được `HF_TOKEN = ...` có dấu cách.

---

## 2026-10-20 — Máy mới RTX A4500 20GB; (a) đường bão hòa theo B của C2-OCT-1V + (d) gỡ cạnh tranh qua sign (DM / DMA) @ dev100 (user chọn) — khóa TRƯỚC khi chạy
**Máy:** NVIDIA RTX A4500 20GB (KHÔNG phải V100 / 3090), bootstrap lại (`results/bootstrap/a4500.log`), đủ 11 checkpoint + COCO; không có `.env`. → sinh lại mọi mốc trên máy này, không trộn PNG V100 / 3090.
**Đo thời gian (`scripts/dm_unit_test.py`, ảnh 1–2 dev300, ảnh 0 = khởi động), s / bước:** OSFD 0.0767; C2-OCT-1V 0.0671; DM 0.0773; DMA 0.0473 → **B cùng thời gian OSFD B50 trên A4500: C2-OCT-1V ≈ 57 (V100: 41), DM ≈ 50, DMA ≈ 81.** Định nghĩa đã đóng băng của C2-OCT-1V (sàng lọc dev300) là B41 = cùng thời gian trên V100; B nào dùng cho GO @ n300 trên A4500 — user quyết riêng, KHÔNG quyết trong entry này.
**Code mới (mặc định giữ hành vi cũ):**
- `core.run_iterative_attack(dual)`: loss view trả cặp (L_feat, L_cls); mỗi số hạng gradient + momentum riêng (chuẩn hoá mean|·|), cập nhật δ += α·sign(m_feat + m_cls) → mỗi kênh "1 phiếu" ở từng bước bất kể độ lớn gradient (bản gốc: sign(MI(g_feat + 4·g_cls)), kênh lớn hơn thắng từng pixel; cos ∇ ≈ −0.08, tỉ lệ độ lớn đổi 5× theo ảnh — entry 10-18). Trọng số 4 không còn tác dụng (bất biến thang đo).
- `C2-OCT-1V-DM` (`dual="exact"`): cả 2 số hạng mỗi bước (2 lần autograd.grad cùng graph). `C2-OCT-1V-DMA` (`dual="alt"`): bước chẵn chỉ L_feat (chỉ backbone), bước lẻ chỉ L_cls (bỏ forward sạch co-transform) → mỗi momentum cập nhật mỗi 2 bước, rẻ hơn / bước.
- `repair.make_view_loss(split)`; `run_baselines.py --snapshots`: 1 lần chạy B lớn ghi δ sau từng mốc B vào run `..._B<k>_...` (PNG + stats, runtime = thời gian tới bước k).
**Kiểm:** split: tổng trùng tuyệt đối (14.609420); DM / DMA L∞ 5.0; DMA forward sạch 21 / 41 bước; snapshot k = 10 của lần chạy B20 vs chạy B10 lệch 0.299 pixel = sàn không tất định (B10 vs B10: 0.294).
**Chạy (A4500, dev100, seed theo chỉ số dev300; log `results/sat_dm/run.log`, script `results/sat_dm/run.sh`; cùng `--out metrics_sat_dev100.json` / dets `dets_sat_dev100.json` mọi run):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --methods OSFD C2-OCT-1V-DM --out metrics_sat_dev100.json          # B50
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 100 --snapshots 20 41 57 65 --methods C2-OCT-1V --out metrics_sat_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 81 --methods C2-OCT-1V-DMA --out metrics_sat_dev100.json
# --stage eval cho các run B20 / B41 / B57 / B65 (methods C2-OCT-1V); eval_generalization (yolox_s dino_swin_l) cho mọi run; rồi
python scripts/pair_screen.py --run dev300_B50_eps5_limit100 --subset dev100 --dets dets_sat_dev100.json --tag sat_dm_dev100 --pairs \
  C2-OCT-1V@dev300_B57_eps5_limit100:OSFD C2-OCT-1V@dev300_B100_eps5_limit100:C2-OCT-1V@dev300_B57_eps5_limit100 \
  C2-OCT-1V@dev300_B57_eps5_limit100:C2-OCT-1V@dev300_B41_eps5_limit100 \
  C2-OCT-1V-DM:C2-OCT-1V@dev300_B57_eps5_limit100 C2-OCT-1V-DMA@dev300_B81_eps5_limit100:C2-OCT-1V@dev300_B57_eps5_limit100
```
**Quy tắc (dev100, paired bootstrap 1000, BOOT_SEED 2026; CA3 = TB ConvNeXt-T / Swin-T / YOLOX-S; Δ = drop(A) − drop(B)):**
- **(a) Bão hòa** — Δ_hi = B100 − B57. **CÒN DỐC** nếu ΔCA3_hi ≥ 2 VÀ cận dưới CI > 0, HOẶC ΔSwin-L_hi ≥ 2 VÀ cận dưới CI > 0 → hướng tăng tốc (bỏ / cache forward sạch để thêm bước cùng thời gian) đáng làm, khóa entry riêng. **BÃO HÒA** nếu ΔCA3_hi < 1 VÀ ΔSwin-L_hi < 1 (điểm) → bỏ hướng tăng tốc. Còn lại → XÁM. Mô tả: đường CA3 / Swin-L / R101 theo B ∈ {20, 41, 57, 65, 100}, runtime theo B; B57 − B41; **C2-OCT-1V B57 − OSFD B50 (cùng thời gian A4500)** — tái lập dev100 V100 (B41: Swin-L +8.8, CA3 +2.6) không.
- **(d) DM / DMA vs C2-OCT-1V B57 (cùng thời gian)**, áp riêng từng ô: **GIÚP (rời đường đánh đổi)** nếu [(ΔCA3 ≥ 2 VÀ cận dưới CI > 0 VÀ ΔSwin-L ≥ −1) HOẶC (ΔSwin-L ≥ 3 VÀ cận dưới CI > 0 VÀ ΔCA3 ≥ −1)] VÀ ΔR101 ≥ −3 VÀ runtime ≤ 1.1 × OSFD B50 (đo cùng lần). **HẠI** nếu ΔCA3 ≤ −3 HOẶC ΔSwin-L ≤ −3 (điểm). **TRÊN ĐƯỜNG** nếu một phía ≥ 2 và phía kia ≤ −1 (điểm). Còn lại → NGANG. Cả hai GIÚP → chọn ΔCA3 + ΔSwin-L lớn hơn (chênh < 1 → DMA, rẻ hơn). GIÚP → bước sau: sàng lọc dev300 so C2-OCT-1V cùng thời gian (entry riêng).
- Mô tả: DM / DMA − OSFD B50; dev100 \ dev30; ConvNeXt / Swin-T / YOLOX / R101 riêng.
**Caveat khai báo trước:** quét λ theo ảnh (LA1 ≈ λ = 4) gợi ý cân bằng 1 phiếu có thể chỉ là một điểm khác trên cùng đường đánh đổi; DMA đổi đồng thời 2 thứ (xen kẽ + nhiều bước hơn ở cùng thời gian) — không tách được nếu GIÚP. GPU không tất định → đọc theo CI. Held-out (YOLOX-L, Swin-S) không dùng.

### Kết quả @ dev100 (A4500; `results/ablation/sat_dm_dev100/screen.json`, `results/runs/dev300_B{20,41,50,57,65,81,100}_eps5_limit100/{metrics,generalization}_sat_dev100.json`, log `results/sat_dm/run.log`) — theo quy tắc: **(a) XÁM; (d) DM NGANG, DMA NGANG**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh:
- OSFD B50: 96.6 / 90.0 / 78.1 / 72.8 / 67.0 / 24.1 | 72.6 | 3.85
- C2-OCT-1V B20 / B41 / **B57** / B65 / B100: CA3 69.0 / 77.8 / **80.1** / 80.6 / 81.3; Swin-L 29.1 / 30.5 / **30.7** / 31.6 / 34.1; R101 88.6 / 93.3 / 94.3 / 94.7 / 95.5; s/ảnh 1.37 / 2.78 / **3.87** / 4.41 / 6.77
- C2-OCT-1V-DM B50: 96.5 / 93.8 / 85.5 / 81.7 / 76.5 / 31.4 | 81.3 | 3.92
- C2-OCT-1V-DMA B81: 97.3 / 94.3 / 85.4 / 81.3 / 74.9 / 31.4 | 80.5 | 3.95
**(a)** B100 − B57: CA3 **+1.2 [0.1, 3.1]**, Swin-L **+3.4 [−0.004, 4.9]** → không "CÒN DỐC" (CA3 < 2; Swin-L ≥ 2 nhưng cận dưới −0.004 ≤ 0), không "BÃO HÒA" (CA3 1.2 ≥ 1) → **XÁM**. B57 − B41: CA3 +2.3 [1.2, 3.8], Swin-L +0.2. Mô tả: CNN gần bão hòa từ B57 (+1.2 cho gần gấp đôi thời gian); Swin-L tăng chủ yếu B65 → B100 (+2.5), trên 70 ảnh mới chỉ +1.7 [−1.1, 3.8].
**C2-OCT-1V B57 − OSFD B50 (cùng thời gian A4500, 3.87 vs 3.85 s):** CA3 **+7.5 [2.5, 11.8]**, Swin-L **+6.6 [1.5, 9.7]**, ConvNeXt +7.7, Swin-T +8.1, YOLOX +6.6, R101 +4.3 → tái lập (dev100 V100 B41: +2.6 / +8.8; dev300 V100: +7.0 / +9.0). B41 (= 0.72× thời gian OSFD trên A4500): CA3 +5.1 [0.0, 9.1], Swin-L +6.4.
**(d)** so C2-OCT-1V B57: DM CA3 +1.1 [−0.3, 3.1], Swin-L +0.7, YOLOX +2.9 [0.4, 5.0], R101 −0.5; DMA CA3 +0.4, Swin-L +0.7 → cả hai **NGANG** (không GIÚP, không HẠI, không "trên đường"). Runtime 1.02× / 1.03× OSFD ✓. 70 ảnh mới: DM +1.4 / −0.4, DMA −0.1 / −0.2.
**Đọc:** cân bằng 1 phiếu / bước giữa 2 kênh (DM) hay xen kẽ (DMA) không đổi được gì so với sign(g_feat + 4·g_cls) ở cùng thời gian → cạnh tranh độ lớn gradient qua sign KHÔNG phải nút thắt (khớp LA1 ≈ λ = 4). Trên A4500, công thức C2-OCT-1V gần bão hòa cho CNN ở B ≈ 57; tăng tốc để thêm bước chỉ hứa hẹn ≤ ~1 điểm CA3, Swin-L có thể +2–3 (chưa chắc). Không ô nào thành ứng viên.

---

## 2026-10-20 — (b) L_cls dạng margin CW trên RoI dương: C2-OCT-1V-MRG / -MRGF @ dev100 B57 (user duyệt "approve b") — khóa TRƯỚC khi chạy
**Cơ sở:** POS (CE chỉ RoI dương ≈ ô gốc → RoI nền không quan trọng); POS-CAP (ngừng đánh object đã sai p < 0.1 → hại mạnh, Swin-L −11.7) → lợi của L_cls đến từ việc tiếp tục đẩy object ĐÃ sai ra xa biên của surrogate.
**Ô (như C2-OCT-1V, trừ loss_cls của RoI head; loss_rpn_cls giữ nguyên, trọng số 4 giữ nguyên):** L_mrg = s · mean_{RoI dương} (max_{c ≠ đúng} z_c − z_đúng), không chặn (κ = ∞).
- `C2-OCT-1V-MRG`: đối thủ = mọi logit khác lớp đúng, KỂ CẢ nền; s = 0.0594.
- `C2-OCT-1V-MRGF`: chỉ 80 lớp object (đẩy sang lớp sai, không về nền); s = 0.0484.
- s = trung vị mean|∇ₓ CE_RoI| / mean|∇ₓ L_mrg| tại δ = 0, ảnh không biến đổi, 5 ảnh đầu dev100 (`scripts/mrg_calib.py`; all 0.061 / 0.059 / 0.041 / 0.026 / 0.156; fg 0.048 / 0.035 / 0.059 / 0.024 / 0.082) — cùng cách chọn λ = 4; margin chỉ trung bình trên RoI dương (~11%) nên gradient gấp ~17–20× CE. Code `repair.roi_margin`, `geo_osfd_attack(margin, margin_scale)`.
**Kiểm:** 6 bước ảnh đầu dev300: L∞ 5.0; mask head + loss_cls.forward khôi phục; giá trị trong context là margin (vd ảnh 7088: CE 0.114 vs margin all −1.65 / fg −6.86).
**Đổi so với đề xuất 10-18 (dev30 B41):** chạy dev100 B57 — B57 = cùng thời gian OSFD B50 trên A4500 (entry trên), mốc C2-OCT-1V B57 + OSFD B50 dev100 đã có cùng máy → so cặp trực tiếp, ít nhiễu hơn dev30. Runtime kỳ vọng = ô gốc (chỉ đổi hàm loss trên logit).
**Chạy (log `results/mrg/run.log`, script `results/mrg/run.sh`):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 57 --methods C2-OCT-1V C2-OCT-1V-MRG C2-OCT-1V-MRGF --out metrics_sat_dev100.json
python scripts/eval_generalization.py --run dev300_B57_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods C2-OCT-1V C2-OCT-1V-MRG C2-OCT-1V-MRGF --dets dets_sat_dev100.json --out generalization_sat_dev100.json
python scripts/pair_screen.py --run dev300_B57_eps5_limit100 --subset dev100 --dets dets_sat_dev100.json --tag mrg_dev100 --exclude dev30 --pairs \
  C2-OCT-1V-MRG:C2-OCT-1V C2-OCT-1V-MRGF:C2-OCT-1V C2-OCT-1V-MRG:OSFD@dev300_B50_eps5_limit100 C2-OCT-1V-MRGF:OSFD@dev300_B50_eps5_limit100
```
**Quy tắc (dev100, paired bootstrap 1000, BOOT_SEED 2026; Δ = drop(ô) − drop(C2-OCT-1V B57)), áp riêng từng ô — cùng khung (d):** **GIÚP** nếu [(ΔCA3 ≥ 2 VÀ cận dưới CI > 0 VÀ ΔSwin-L ≥ −1) HOẶC (ΔSwin-L ≥ 3 VÀ cận dưới CI > 0 VÀ ΔCA3 ≥ −1)] VÀ ΔR101 ≥ −3 VÀ runtime ≤ 1.1 × OSFD B50 (3.85 s). **HẠI** nếu ΔCA3 ≤ −3 HOẶC ΔSwin-L ≤ −3. **TRÊN ĐƯỜNG** nếu một phía ≥ 2 và phía kia ≤ −1. Còn lại → NGANG. Cả hai GIÚP → chọn ΔCA3 + ΔSwin-L lớn hơn (chênh < 1 → MRG). GIÚP → sàng lọc dev300 so C2-OCT-1V cùng thời gian (entry riêng). Không ô nào GIÚP → theo đề xuất 10-18 mục (c): coi C2-OCT-1V gần trần nhóm 1 surrogate R50, chuẩn bị GO @ n300 (user quyết B).
- Mô tả: Δ − OSFD; 70 ảnh mới; hồ sơ lỗi DINO-Swin-L nếu GIÚP.
**Caveat khai báo trước:** CE ascent vốn KHÔNG bão hòa theo z_đúng (∂CE/∂z_đúng → −1 khi p_đúng → 0) → khác biệt thực của MRG so với CE chủ yếu là: chỉ RoI dương (POS ≈ ô gốc), chỉ 1 lớp đối thủ mạnh nhất, (MRGF) không về nền — kỳ vọng tiên nghiệm: dễ NGANG.

### Kết quả (b) @ dev100 B57 (A4500; `results/ablation/mrg_dev100/screen.json`, `results/runs/dev300_B57_eps5_limit100/{metrics,generalization}_sat_dev100.json`, log `results/mrg/run.log`) — theo quy tắc: **MRG HẠI, MRGF HẠI**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT-1V 98.3 / 94.3 / 85.7 / 81.0 / 73.7 / 30.7 | 80.1 | 3.87; **MRG** 97.0 / 93.1 / 83.3 / 76.2 / 70.1 / 28.6 | 76.6 | 3.82; **MRGF** 97.2 / 91.2 / 81.1 / 75.3 / 66.8 / 27.2 | 74.4 | 3.82; OSFD B50 | 72.6, Swin-L 24.1.
Δ so C2-OCT-1V: MRG CA3 **−3.6 [−6.0, +0.1]** (≤ −3 → HẠI), Swin-L −2.1 [−5.3, −0.0]; MRGF CA3 **−5.7 [−7.9, −2.6]**, Swin-L **−3.5 [−7.3, −1.8]**, YOLOX −6.9, R101 −3.1. 70 ảnh mới cùng chiều (MRG −3.1 / −3.2; MRGF −4.4 / −4.1).
**Đọc:** margin "1 đối thủ mạnh nhất" yếu hơn CE ở MỌI target; bỏ nền (MRGF) còn tệ hơn → (1) đẩy về NỀN là phần có ích của L_cls (kể cả cho Swin-L), không phải đổi sang lớp object khác — ngược giả thuyết từ hồ sơ lỗi (FP mang nhãn object là sản phẩm phụ, không phải kênh); (2) CE đẩy đồng thời mọi lớp sai theo softmax tốt hơn dồn vào 1 lớp. Cùng với POS-CAP: lợi của L_cls = CE đầy đủ trên RoI dương, đẩy tiếp object đã sai — dạng CE hiện tại đã là điểm tốt trong họ này. Theo quy tắc khóa: không ô nào GIÚP → coi C2-OCT-1V gần trần nhóm 1 surrogate R50; bước tiếp GO @ n300 (user quyết B: 41 đóng băng hay 57 = cùng thời gian A4500).

---

## 2026-10-20 — TẠM DỪNG (user trả máy, mai chạy tiếp): tổng kết phiên A4500 + việc mở
**Máy:** RTX A4500 20GB (bootstrap lại; đủ 11 checkpoint + COCO; KHÔNG có `.env`). Không có tiến trình nào đang chạy.
**Kết quả (dev100, theo quy tắc khóa, so cùng thời gian OSFD B50 = 3.85 s / ảnh trên A4500):**
1. **B cùng thời gian trên A4500 = 57** cho C2-OCT-1V (V100: 41). C2-OCT-1V B57 − OSFD B50: CA3 +7.5 [2.5, 11.8], Swin-L +6.6 [1.5, 9.7] → lợi tái lập trên máy thứ ba.
2. (a) Bão hòa theo B: XÁM — CNN gần bão hòa từ B57 (B100 − B57 CA3 +1.2), Swin-L +3.4 (CI chạm 0) → tăng tốc để thêm bước hứa hẹn ít.
3. (d) DM / DMA (momentum riêng 2 kênh, 1 phiếu / bước): NGANG → cạnh tranh độ lớn gradient qua sign không phải nút thắt.
4. (b) Margin CW: MRG HẠI (CA3 −3.6), MRGF HẠI (CA3 −5.7, Swin-L −3.5) → đẩy về NỀN là phần có ích của L_cls; CE đầy đủ là điểm tốt nhất trong họ này.
**Vấn đề user nêu (chưa giải):** transfer vẫn KHÔNG qua được DINO-Swin-L — trên hình box, C2-OCT-1V không xóa thêm box sạch (giữ 710 / 1060 vs OSFD 698) và gần như không thêm box nhiễu (FP 3.6 → 4.7 / ảnh, so Swin-T Mask R-CNN 24–27); drop Swin-L ~31% vs CNN ≥ 80%. Mọi núm trên R50 (loss, đa dạng, cập nhật, trọng số) đều tìm hướng trong không gian R50 → trượt dọc 1 đường đánh đổi (WB-probe: gap hai chiều; E14: nút thắt = hướng ở stage 3).
**Việc mở (thứ tự đề xuất cho phiên sau):**
- (1) **Chẩn đoán đổi surrogate (CHƯA khóa — khóa quy tắc entry riêng trước khi chạy):** C2-OCT-1V với `run_baselines.py --surrogate {r101, convnext_t, swin_t}` (cờ có từ 10-08, chưa từng dùng; run riêng `*_sur-<key>`), dev100, đo trên DINO-Swin-L (+ panel). Chỉ phân tích, threat model vẫn khóa surrogate R50 (idea.md §5). Đọc đề xuất: surrogate Swin-T phá Swin-L rõ (vd ≥ 50%) → bức tường là khác họ CNN ↔ Transformer → method làm R50 "giống Transformer" (hoặc user cân nhắc đổi threat model); Swin-T cũng ~30% → bức tường là chính DINO-Swin-L (encoder sửa + Swin-L / IN-22k) → ngừng đuổi Swin-L bằng method, báo cáo như kết quả độ bền có cơ chế (RQ2). Cần kiểm trước: `skip_mask` / AMP / hook backbone chạy được với surrogate Swin-T / ConvNeXt; B cùng thời gian đo lại cho từng surrogate.
- (2) GO @ n300 cho C2-OCT-1V — user quyết B (41 đóng băng = 0.72× thời gian OSFD trên A4500, hay 57 = cùng thời gian); sinh lại OSFD n300 cùng máy.
- (3) Rà BFDA (cần PDF ICASSP'24).
**Code mới phiên này (mặc định giữ hành vi cũ):** `core.run_iterative_attack(dual)`, `repair.make_view_loss(split)`, `repair.roi_margin`, `geo_osfd_attack(dual, margin, margin_scale)`; ô `C2-OCT-1V-{DM, DMA, MRG, MRGF}`; `run_baselines.py --snapshots` (+ `run_name`); `scripts/dm_unit_test.py`, `scripts/mrg_calib.py`.
**Artifact A4500 ĐÃ upload HF** (user thêm `.env` cuối phiên) với hậu tố `_a4500` để KHÔNG ghi đè tar V100 cùng tên (`dev300_B{41,50}_eps5_limit100.tar`): `runs/dev300_B{20,41,50,57,65,81,100}_eps5_limit100_a4500.tar` (sha256 trong commit message HF). Tải lại: `python scripts/sync_artifacts.py download dev300_B57_eps5_limit100 --tag _a4500` (giải nén ra `artifacts/runs/dev300_B57_eps5_limit100/` — không trộn với PNG V100 cùng tên run). `sync_artifacts.py` thêm `--tag` (hậu tố tên file HF).

---

## 2026-10-21 — Máy mới V100 16GB; chẩn đoán đổi surrogate: C2-OCT-1V với surrogate Swin-T @ dev100 B41 (user: "chỉ cần câu trả lời chính") — khóa TRƯỚC khi chạy
**Máy:** Tesla V100-SXM2 16GB (cùng loại máy sàng lọc dev300 10-18 → B41 = cùng thời gian OSFD B50 với surrogate R50), bootstrap lại; không có `.env` → không tải được dets V100 cũ từ HF → sinh lại mốc R50 cùng máy.
**Câu hỏi (việc mở (1) entry 10-20, chỉ nhánh Swin-T):** transfer không qua DINO-Swin-L (drop ~31%) là do bức tường **khác họ CNN ↔ Transformer** hay do **chính DINO-Swin-L** (encoder DINO "sửa" + Swin-L / IN-22k)? Chỉ PHÂN TÍCH — threat model vẫn khóa surrogate R50 (idea.md §5); r101 / convnext_t KHÔNG chạy (user chỉ cần câu trả lời chính).
**Ô:** `C2-OCT-1V` y nguyên (λ = 4, OCT 1 view, MI, skip_mask + AMP, ε = 5, B41), chỉ đổi `--surrogate swin_t` (Mask R-CNN Swin-T, cùng head RPN + RoI → loss C2 định nghĩa y hệt; L_OSFD trên 4 stage backbone Swin-T). Cùng số bước B41 (không cùng thời gian — runtime báo mô tả).
**Chạy (log `results/sur/run.log`, script `results/sur/run.sh`):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --methods C2-OCT-1V --out metrics_sur_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --surrogate swin_t --methods C2-OCT-1V --out metrics_sur_dev100.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods C2-OCT-1V --dets dets_sur_dev100.json --out generalization_sur_dev100.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100_sur-swin_t --subset dev100 --models yolox_s dino_swin_l --methods C2-OCT-1V --dets dets_sur_dev100.json --out generalization_sur_dev100.json
python scripts/pair_screen.py --run dev300_B41_eps5_limit100 --subset dev100 --dets dets_sur_dev100.json --tag sur_swint_dev100 --exclude dev30 \
  --pairs C2-OCT-1V@dev300_B41_eps5_limit100_sur-swin_t:C2-OCT-1V
```
**Quy tắc (dev100, paired bootstrap 1000, BOOT_SEED 2026; Δ = drop_Swin-L(sur Swin-T) − drop_Swin-L(sur R50), cùng ảnh):**
- **Hợp lệ** chỉ khi white-box drop của Swin-T (surrogate) ≥ 80% (attack thật sự khớp surrogate mới). Không đạt → KHÔNG HỢP LỆ, không kết luận, ghi log.
- **TƯỜNG HỌ (CNN ↔ Transformer)** nếu drop_Swin-L(sur Swin-T) ≥ 50 VÀ cận dưới CI của Δ > 0 → hướng method: làm δ từ R50 "giống Transformer" (hoặc user cân nhắc threat model).
- **TƯỜNG DINO-Swin-L** nếu drop_Swin-L(sur Swin-T) < 40 → bức tường nằm ở chính DINO-Swin-L; ngừng đuổi Swin-L bằng method, báo cáo như kết quả độ bền có cơ chế (RQ2).
- Còn lại → XÁM (mô tả, user quyết).
- Mô tả: chiều ngược Swin-T → R50 / R101 / ConvNeXt / YOLOX (so WB-probe: gap hai chiều); runtime s/ảnh của 2 surrogate; 70 ảnh mới (dev100 \ dev30).
**Caveat khai báo trước:** λ = 4 và mọi liều OCT hiệu chỉnh trên R50, không tune lại cho Swin-T (thiên về bất lợi cho surrogate Swin-T → nếu vẫn TƯỜNG HỌ thì kết luận càng chắc; nếu TƯỜNG DINO-Swin-L có thể một phần do chưa tune); Swin-T là surrogate "cùng họ" với Swin-L nhưng khác detector (Mask R-CNN vs DINO) và khác cỡ / dữ liệu pretrain (IN-1k vs IN-22k) — không tách được 2 yếu tố này trong 1 thí nghiệm. GPU không tất định → đọc theo CI. Held-out (YOLOX-L, Swin-S) không dùng.

### Kết quả @ dev100 B41 (V100; `results/ablation/sur_swint_dev100/screen.json`, `results/runs/dev300_B41_eps5_limit100{,_sur-swin_t}/{metrics,generalization}_sur_dev100.json`, log `results/sur/run.log`) — theo quy tắc: **HỢP LỆ, TƯỜNG HỌ (CNN ↔ Transformer)**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3 | s/ảnh:
- sur R50: 97.7 / 92.7 / 82.4 / 77.7 / 69.1 / **32.0** | 76.4 | 4.1
- sur Swin-T: 78.7 / 75.7 / 85.2 / **95.3** (white-box) / 70.8 / **58.0** | 83.8 | 5.3 (1.3×)
**Quy tắc:** white-box Swin-T 95.3 ≥ 80 → HỢP LỆ. drop_Swin-L(sur Swin-T) 58.0 ≥ 50 VÀ Δ = **+26.0 [+19.6, +29.6]** (cận dưới > 0) → **TƯỜNG HỌ**. 70 ảnh mới cùng chiều: 53.2 vs 31.1, Δ +22.1 [+17.5, +26.9].
**Mô tả:** chiều ngược gap y hệt — Swin-T → R50 78.7 / R101 75.7 (Δ R101 −17.0) ≈ R50 → Swin-T 77.7 (khớp WB-probe: gap hai chiều). ConvNeXt +2.8 [−0.2, 7.1] (CNN "thiết kế kiểu Transformer" hưởng nhẹ), YOLOX +1.7 (CI chứa 0). DINO-Swin-L vẫn kháng hơn Swin-T (58 vs 95 white-box) → encoder DINO / cỡ Swin-L có góp phần nhưng KHÔNG phải bức tường chính.
**Đọc:** DINO-Swin-L KHÔNG miễn nhiễm transfer — đổi surrogate sang cùng họ Transformer (detector khác, Mask R-CNN vs DINO; pretrain khác IN-1k vs IN-22k; λ / liều OCT chưa tune cho Swin-T) đã gần gấp đôi drop (32 → 58). Bức tường chính là khác họ backbone CNN ↔ Transformer. Theo đề xuất 10-20: hướng method = làm δ sinh từ R50 "giống Transformer" (hoặc user cân nhắc threat model). Bước tiếp — user quyết.

---

## 2026-10-21 — Hướng method (user chọn): "làm δ từ R50 giống Transformer". Bước 0 — E16: phần nào của δ_SwinT mang transfer sang DINO-Swin-L? (lọc tần / màu δ có sẵn, KHÔNG attack lại) @ dev100 — MÔ TẢ, khai báo TRƯỚC khi chạy
**Lý do:** 5 pilot đoán-cơ-chế trước đều NO-GO → không đoán "giống Transformer" là gì; đo trực tiếp trên 2 bộ δ cùng 100 ảnh, cùng ε / B / loss / OCT, chỉ khác surrogate (entry trên: DINO-Swin-L 32.0 vs 58.0, G = +26.0).
**Thao tác (`scripts/e16_delta_band.py`, PNG `artifacts/e16/`, kết quả `results/mechanism/e16_delta_band.json`, log `results/e16/run.log`):** δ_X = PNG adv − ảnh gốc (không gian ảnh gốc, int, |δ| ≤ 5), X ∈ {R50, SwinT}. Biến thể f(δ), mỗi kênh riêng: LP_σ = Gauss σ ∈ {1, 2, 4} px; HP_2 = δ − LP_2; GRAY = trung bình 3 kênh (bỏ màu). Mọi biến thể **co giãn lại L∞ = 5** (cùng ngân sách), làm tròn int, kẹp [0, 255]. FULL = PNG gốc. → 2 × 6 = 12 điều kiện.
**Đo:** relative AP drop trên **DINO-Swin-L** (chính), Mask R-CNN R50 và Swin-T (đối chiếu: biến thể còn phá surrogate của chính nó?); mọi model qua `inference_detector` trên file (cùng cách cho clean + mọi điều kiện → so trong thí nghiệm này, KHÔNG so trực tiếp số Controlled Panel của run_baselines). Paired bootstrap 1000, BOOT_SEED 2026. Vân tay δ (CPU, theo ảnh): tỉ lệ năng lượng 3 dải tần A4 (thấp < 1/6, trung < 1/3, cao, cycles/px), tỉ lệ năng lượng màu (δ − trung bình kênh), năng lượng trong GT box / tỉ lệ diện tích, cos(δ_R50, δ_SwinT).
**Đọc (khai báo trước; G' = drop_SwinL(FULL δ_SwinT) − drop_SwinL(FULL δ_R50) đo trong thí nghiệm này):**
- **(A) "δ R50 chỉ quá cao tần"** nếu max_σ [drop(LP_σ δ_R50) − drop(FULL δ_R50)] ≥ 0.5·G' VÀ cận dưới CI > 0 → hướng method: ràng buộc / tham số hoá δ tần thấp trong attack R50.
- Không (A), và **(B) "thành phần Transformer là tần thấp, R50 không sinh ra được"** nếu [drop(LP_2 δ_SwinT) − drop(FULL δ_R50)] ≥ 0.7·G' → làm mượt thôi không đủ; hướng: đổi HƯỚNG (feature / gradient) chứ không phải phổ.
- Không (A), và **(C) "cấu trúc mịn (mức patch / token)"** nếu [drop(HP_2 δ_SwinT) − drop(FULL δ_R50)] ≥ 0.7·G' → hướng: biến đổi đầu vào / ràng buộc theo cấu trúc patch.
- Cả (B) và (C) → cả hai dải đều mang (thành phần "đúng hướng" có mặt mọi tần) → như (B). Còn lại → XÁM (mô tả). GRAY: mô tả (vai trò màu).
**Caveat:** co giãn L∞ = 5 làm L2 của biến thể khác FULL (báo L2); inference_detector ≠ pipeline Controlled của run_baselines; dev100 duy nhất, không chọn tham số nào cho method từ đây (chỉ chọn hướng).

### Kết quả E16 @ dev100 (`results/mechanism/e16_delta_band.json`, log `results/e16/run.log`) — theo quy tắc: **XÁM** (không A, không B, không C)
(Sự cố: bản đầu khởi tạo lại CocoImageEval cho mỗi mẫu bootstrap → treo > 1 h ở bước AP; sửa, chạy lại, dets DINO giữ nguyên — số không đổi.)
Relative AP drop % DINO-Swin-L / R50 / Swin-T (inference_detector):
- R50: FULL 32.0 / 97.7 / 77.7; LP1 10.6; LP2 3.4; LP4 1.7; HP2 8.1; GRAY 15.9 (DINO)
- SwinT: FULL 58.0 / 78.7 / 95.3; LP1 22.4; LP2 10.1; LP4 5.3; HP2 16.4; GRAY 24.4 (DINO)
G' = +26.0 [19.6, 29.6] (= số Controlled, cùng PNG). (A) LP tốt nhất của δ_R50 (LP1) −21.4 → KHÔNG; (B) LP2 δ_SwinT giữ −0.84·G → KHÔNG; (C) HP2 giữ −0.60·G → KHÔNG.
**Mô tả (không thuộc quy tắc; so CÙNG phép lọc giữa 2 nguồn → cùng L2):** δ_SwinT − δ_R50 trên DINO: LP1 +11.8, LP2 +6.7, LP4 +3.6, HP2 +8.3, GRAY +8.5 → lợi thế của δ_SwinT **có mặt ở mọi dải và cả khi bỏ màu** (tỉ lệ ~1.5–3×), không nằm riêng ở dải nào.
**Vân tay (TB 100 ảnh):** phổ 3 dải R50 0.34 / 0.36 / 0.30 vs SwinT 0.35 / 0.35 / 0.31 (gần như trùng); mật độ năng lượng trong GT box 1.01 vs 1.01 (không dồn vào object); màu 0.49 vs 0.61 (SwinT màu hơn — nhưng GRAY vẫn giữ lợi thế); cos(δ_R50, δ_SwinT) = 0.06 (gần trực giao).
**Đọc:** mọi phép lọc đều phá phần lớn tác dụng của CẢ HAI δ (δ băng rộng, dạng sign ±5 khắp ảnh) → (A) bị loại rõ: làm mượt δ R50 chỉ HẠI. Hai δ giống nhau về phổ, phân bố không gian, gần giống về màu, nhưng gần trực giao và δ_SwinT phá DINO gấp ~2× ở mọi dải → khác biệt là **HƯỚNG / mẫu không gian cụ thể** (δ đánh vào feature nào), không phải thống kê bậc thấp (phổ, màu, vị trí). Đáng chú ý: δ_SwinT vẫn phá R50 78.7% → tồn tại vùng δ adversarial cho CẢ R50 lẫn Transformer mà gradient ascent trên R50 không tìm tới. Hướng method: thay đổi HƯỚNG gradient R50 (đường lan ngược / mục tiêu feature / biến đổi kiểu token), không phải ràng buộc phổ. User quyết ô cụ thể.

---

## 2026-10-21 — Hướng (1) "đổi hướng gradient R50": C2-OCT-1V-BPA / -BPA34 @ dev30 B41 (user: test nhanh 5–10 phút để thấy xu hướng) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Ô (`attack/methods/bpa.py`):** C2-OCT-1V y nguyên; forward ReLU giữ nguyên, backward thay 1[x > 0] bằng đạo hàm x·σ(βx), β = 1.702 (≈ GELU của Swin). `BPA`: mọi ReLU backbone R50 (17 module, stem + 16 Bottleneck); `BPA34`: chỉ res_layers 3–4 (stage 3 = nút thắt E14). Kiểm: loss forward lệch 0.0; cos(∇ mới, ∇ gốc) trên loss feature, ảnh ngẫu nhiên: BPA 0.66, BPA34 0.96; AMP hữu hạn; khôi phục ReLU đúng. Runtime kỳ vọng ≈ ô gốc.
**Chạy (V100, dev30 ⊂ dev100, PNG C2-OCT-1V B41 dev100 tái dùng; log `results/bpa/run.log`):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 41 --methods C2-OCT-1V C2-OCT-1V-BPA C2-OCT-1V-BPA34 --out metrics_bpa_dev30.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V C2-OCT-1V-BPA C2-OCT-1V-BPA34 --dets dets_bpa_dev30.json --out generalization_bpa_dev30.json
python scripts/pair_screen.py --run dev300_B41_eps5_limit100 --subset dev30 --dets dets_bpa_dev30.json --tag bpa_dev30 --pairs C2-OCT-1V-BPA:C2-OCT-1V C2-OCT-1V-BPA34:C2-OCT-1V
```
**Đọc xu hướng (dev30, điểm ước lượng, KHÔNG quyết định method):** "đáng lên dev100 (khóa entry riêng)" nếu ΔSwin-L ≥ +3 VÀ ΔCA3 ≥ −2; "hại" nếu ΔSwin-L ≤ −3 HOẶC ΔCA3 ≤ −3; còn lại "ngang". β, stage KHÔNG tune thêm trên dev30.

### Kết quả BPA @ dev30 B41 (V100; `results/ablation/bpa_dev30/screen.json`, `results/runs/dev300_B41_eps5_limit100/{metrics,generalization}_bpa_dev30.json`, log `results/bpa/run.log`) — xu hướng: **BPA NGANG, BPA34 NGANG**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT-1V 99.3 / 94.4 / 81.9 / 80.3 / 64.8 / 31.9 | 75.7 | 4.1; **BPA** 99.0 / 92.8 / 82.9 / 79.6 / 71.1 / 29.8 | 77.9 | 5.2; **BPA34** 99.3 / 93.4 / 81.8 / 79.1 / 68.0 / 30.0 | 76.3 | 4.4.
Δ so C2-OCT-1V: BPA Swin-L −2.0 [−5.0, 1.6], CA3 +2.2 [−1.9, 3.6], YOLOX +6.3, Swin-T −0.8; BPA34 Swin-L −1.8, CA3 +0.6, YOLOX +3.2, Swin-T −1.2.
**Đọc:** backward ReLU ≈ GELU KHÔNG đưa δ R50 về phía Transformer (Swin-T / Swin-L không tăng, hơi giảm); lợi chỉ ở YOLOX (CNN) → lại là một điểm khác trên đường đánh đổi CNN ↔ Transformer. "Giống Transformer" theo nghĩa activation cục bộ (ReLU vs GELU) không phải chỗ khác biệt về hướng (E16). Không đưa lên dev100. BPA còn chậm 1.27× (sigmoid fp32 trong backward).

---

## 2026-10-21 — Hướng (3) "mục tiêu feature toàn cục ở stage 3": C2-OCT-1V-GLB7 / -GLBG @ dev30 B41 (user: test nhanh) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Hướng (2) (biến đổi đầu vào kiểu patch / token) chưa từng chạy** — BSR khai báo rồi bỏ trước khi code (entry PIA); chỉ có dịch pha lưới (SH*) và xáo δ phân tích (E7). Vẫn còn mở.
**Ô (`repair._pooled_mse`, `make_view_loss(stage_pool)`, `geo_osfd_attack(stage_pool)`):** C2-OCT-1V y nguyên trừ số hạng L_OSFD của **stage 3** (res_layers[2], stride 16): MSE(k·P(F_sạch), P(F_adv)) với P = avg-pool 7×7 stride 7 (`GLB7`, cửa sổ kiểu Swin) | trung bình toàn cục (`GLBG`), nhân hệ số detach để GIÁ TRỊ = MSE từng điểm gốc → chỉ đổi HƯỚNG gradient của stage 3 (mô phỏng trộn token: δ phải dịch chuyển feature gộp vùng thay vì từng vị trí). Stage 1, 2, 4 và L_cls không đổi. Kiểm: giá trị lệch 0.0; cos(∇ gộp, ∇ từng điểm) trên tensor ngẫu nhiên 0.10 (7×7) / 0.02 (GAP); chạy thử 1 ảnh 3 bước OK.
**Chạy (V100, dev30, log `results/glb/run.log`):** như BPA, `--methods C2-OCT-1V C2-OCT-1V-GLB7 C2-OCT-1V-GLBG --out metrics_glb_dev30.json`, eval_generalization `--dets dets_glb_dev30.json --out generalization_glb_dev30.json`, pair_screen `--tag glb_dev30 --pairs C2-OCT-1V-GLB7:C2-OCT-1V C2-OCT-1V-GLBG:C2-OCT-1V`.
**Đọc xu hướng (dev30, điểm ước lượng, như BPA):** "đáng lên dev100" nếu ΔSwin-L ≥ +3 VÀ ΔCA3 ≥ −2; "hại" nếu ΔSwin-L ≤ −3 HOẶC ΔCA3 ≤ −3; còn lại "ngang". Mô tả thêm Δ Swin-T (target Transformer thứ 2). Kernel / stage KHÔNG tune trên dev30.

### Kết quả GLB @ dev30 B41 (V100; `results/ablation/glb_dev30/screen.json`, `results/runs/dev300_B41_eps5_limit100/{metrics,generalization}_glb_dev30.json`, log `results/glb/run.log`) — xu hướng: **GLB7 HẠI, GLBG HẠI** (ΔSwin-L ≤ −3)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT-1V 99.3 / 94.4 / 81.9 / 80.3 / 64.8 / 31.9 | 75.7 | 4.1; **GLB7** 99.4 / 93.5 / 81.2 / 78.8 / 68.4 / 28.1 | 76.1 | 4.1; **GLBG** 99.1 / 92.8 / 80.6 / 81.8 / 69.5 / 27.1 | 77.3 | 3.9.
Δ so C2-OCT-1V: GLB7 Swin-L −3.8 [−7.3, 1.4], Swin-T −1.5, YOLOX +3.6, CA3 +0.4; GLBG Swin-L −4.8 [−8.8, −0.1], Swin-T +1.5, YOLOX +4.7, CA3 +1.6.
**Đọc:** gộp feature stage 3 (mô phỏng trộn token) KHÔNG đưa δ về phía Transformer — DINO-Swin-L giảm, lợi lại dồn sang YOLOX. Cùng mẫu với BPA (10-21), DM / DMA, margin, λ, RaPA, ILPD (10-18 / 10-20): mọi thay đổi phía R50 đều trượt dọc đường đánh đổi YOLOX ↑ / Swin-L ↓. Không đưa lên dev100.

---

## 2026-10-21 — Hướng (2) "biến đổi đầu vào kiểu patch": C2-OCT-1V-PD16 / -PD32 @ dev30 B41 (user: quick test) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Ô (`geo_osfd_attack(patch_drop=(ps, p))`):** C2-OCT-1V y nguyên; mỗi view (sau OCT + nhiễu), che ngẫu nhiên từng patch ps × ps (khung model) với xác suất p = 0.3 bằng trung bình kênh của view sạch — CÙNG mask cho view adv và view sạch co-transform (L_OSFD) ; box GT giữ nguyên (không dịch chuyển hình học; L_cls tính trên ảnh đã che). RNG riêng (seed + 7) → chuỗi phép OCT trùng ô gốc (kiểm: disp 3 bước trùng). `PD16`: ps = 16 (= stride 16 của stage 3 R50); `PD32`: ps = 32. Cơ sở: Transformer tổng hợp ngữ cảnh toàn cục và chịu được thiếu patch — δ phải còn tác dụng khi một phần ngữ cảnh cục bộ biến mất → bớt phụ thuộc mẫu cục bộ đặc thù R50. Khác BSR (bỏ trước đây): không xáo vị trí → không cần co-transform GT. Kiểm: tỉ lệ che / view 0.30–0.31; δ khác ô gốc (|Δ| TB 1.7 sau 3 bước).
**Chạy (V100, dev30, log `results/pd/run.log`):** như BPA / GLB, `--methods C2-OCT-1V C2-OCT-1V-PD16 C2-OCT-1V-PD32 --out metrics_pd_dev30.json`, dets `dets_pd_dev30.json`, pair_screen `--tag pd_dev30`.
**Đọc xu hướng (dev30, như BPA / GLB):** "đáng lên dev100" nếu ΔSwin-L ≥ +3 VÀ ΔCA3 ≥ −2; "hại" nếu ΔSwin-L ≤ −3 HOẶC ΔCA3 ≤ −3; còn lại "ngang". Mô tả Δ Swin-T. ps / p KHÔNG tune trên dev30. **Nếu cả hai không "đáng lên dev100" → đề xuất dừng đuổi Swin-L bằng method R50, chuyển GO @ n300 (user quyết).**

### Kết quả PD @ dev30 B41 (V100; `results/ablation/pd_dev30/screen.json`, `results/runs/dev300_B41_eps5_limit100/{metrics,generalization}_pd_dev30.json`, log `results/pd/run.log`) — xu hướng: **PD16 HẠI, PD32 HẠI** (mạnh, mọi target)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT-1V 99.3 / 94.4 / 81.9 / 80.3 / 64.8 / 31.9 | 75.7 | 4.1; **PD16** 80.4 / 45.9 / 36.9 / 45.6 / 34.8 / 13.7 | 39.1 | 3.8; **PD32** 95.9 / 80.3 / 65.8 / 71.3 / 54.7 / 20.6 | 63.9 | 3.7.
Δ so C2-OCT-1V: PD16 Swin-L −18.2 [−23.3, −12.8], CA3 −36.6; PD32 Swin-L −11.3 [−15.1, −5.8], CA3 −11.8, Swin-T −9.0.
**Đọc:** che 30% patch làm yếu attack ngay trên chính surrogate (R50 white-box 99.3 → 80.4 / 95.9) → liều này phá tối ưu (biên patch nhân tạo + L_cls trên ảnh bị che), không phải "δ bền ngữ cảnh". Patch nhỏ (16) hại hơn patch lớn (32). Không cho thấy dấu hiệu tốt hơn trên Transformer (Swin-L giảm tỉ lệ ngang CNN). Caveat: chỉ 1 liều p = 0.3 (không tune theo khai báo) — p nhỏ hơn chưa thử.
**Tổng kết 3 hướng "làm δ R50 giống Transformer" (dev30, mô tả):** (1) BPA NGANG, (3) GLB HẠI, (2) PD HẠI → theo khai báo: đề xuất dừng đuổi DINO-Swin-L bằng method trên R50, chuyển GO @ n300 cho C2-OCT-1V; bức tường CNN ↔ Transformer báo cáo như kết quả có cơ chế (đổi surrogate + E16 + WB-probe). User quyết.

---

## 2026-10-21 — Hướng MỚI (user duyệt, sau phản biện "dừng = như method cũ"): sửa HÌNH HỌC GRADIENT của surrogate — nhiễu trọng số R50 trong lúc attack (họ AWT / SAM phía model): C2-OCT-1V-WP2 / -WP5 @ dev30 B41 — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lập luận:** mọi núm phía attack trên θ R50 CỐ ĐỊNH (loss, lan ngược, biến đổi đầu vào kể cả biến dạng mượt ≈ DeCoWA / EDA) đều trượt 1 đường đánh đổi; δ_SwinT phá R50 78.7% → tồn tại vùng δ chung cho 2 họ mà trường gradient của θ R50 không chỉ tới → can thiệp vào θ (surrogate refinement), loại chưa từng thử trong dự án. Literature: AWT (data-free weight tuning, lợi Transformer ~10% vs CNN ~5%, classification), SASD-WS, flat-minima over surrogates; chưa thấy cho object detection.
**Threat model (user đồng ý):** vẫn single surrogate R50 — nhiễu trọng số TẠM trong lúc attack, không dữ liệu ngoài, không truy cập target; θ khôi phục sau mỗi ảnh.
**Ô (`core.run_iterative_attack(weight_perturb)`, `geo_osfd_attack(weight_rho)`):** C2-OCT-1V; trong CÙNG lần autograd.grad lấy thêm ∇θ (42 tensor trọng số conv backbone có requires_grad; stage đóng băng của mmdet giữ nguyên); bước sau dùng θ' = θ − ρ·‖θ_t‖·∇θ_t/‖∇θ_t‖ (từng tensor) — surrogate "chống lại" δ hiện tại (trễ 1 bước, không forward thêm); feature sạch co-transform cũng trên θ'. autocast `cache_enabled=False` (bản fp16 cache từ forward no_grad mất liên kết θ).
**Kiểm:** θ khôi phục y nguyên (torch.equal mọi tham số); thời gian / bước ≈ ô gốc (0.091 vs 0.089 s, ảnh 2) → B41 = cùng thời gian. **Hiệu chỉnh ρ trước khi chạy, chỉ theo sức khỏe tối ưu (KHÔNG nhìn transfer), ảnh 1–4 dev300:** ρ ∈ {0.001, 0.003, 0.005}: L∞ 5, ~90% pixel chạm ±5 (= ô gốc); ρ = 0.01: dao động (L∞ 4, 0–31% chạm); ρ = 0.03: L∞ 2 → chọn **WP2 (ρ = 0.002)**, **WP5 (ρ = 0.005, lớn nhất còn khỏe)**.
**Chạy (V100, dev30, log `results/wp/run.log`):** như BPA / GLB / PD, `--methods C2-OCT-1V C2-OCT-1V-WP2 C2-OCT-1V-WP5 --out metrics_wp_dev30.json`, dets `dets_wp_dev30.json`, pair_screen `--tag wp_dev30`.
**Đọc xu hướng (dev30, như BPA):** "đáng lên dev100" nếu ΔSwin-L ≥ +3 VÀ ΔCA3 ≥ −2; "hại" nếu ΔSwin-L ≤ −3 HOẶC ΔCA3 ≤ −3; còn lại "ngang". Mô tả Δ Swin-T, runtime. ρ không tune thêm trên dev30.

### Kết quả WP @ dev30 B41 (V100; `results/ablation/wp_dev30/screen.json`, `results/runs/dev300_B41_eps5_limit100/{metrics,generalization}_wp_dev30.json`, log `results/wp/run.log`) — xu hướng: **WP2 ĐÁNG LÊN dev100**, WP5 NGANG
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: C2-OCT-1V 99.3 / 94.4 / 81.9 / 80.3 / 64.8 / 31.9 | 75.7 | 4.1; **WP2** 99.0 / 92.4 / 83.8 / 81.4 / 74.7 / 38.9 | 80.0 | 4.7; **WP5** 97.3 / 87.5 / 79.9 / 75.1 / 72.8 / 32.2 | 75.9 | 4.6.
Δ so C2-OCT-1V: **WP2 Swin-L +7.0 [+1.3, +8.3]**, CA3 +4.3 [−2.1, 7.6], YOLOX +9.9, ConvNeXt +1.9, Swin-T +1.0, R101 −1.9; WP5 Swin-L +0.4, CA3 +0.2, R101 −6.9, Swin-T −5.3.
**Đọc:** WP2 là ô ĐẦU TIÊN trong mọi thử nghiệm phía R50 tăng ĐỒNG THỜI DINO-Swin-L và CA3 (không trượt đường đánh đổi YOLOX ↑ / Swin-L ↓) — ủng hộ giả thuyết nút thắt nằm ở hình học gradient của θ surrogate. Liều nhạy: ρ = 0.005 mất lợi Swin-L và hại R101 / Swin-T (gần ngưỡng dao động 0.01). **Caveat:** dev30 nhiễu (CA3 CI chứa 0); runtime 4.7 vs 4.1 = **1.15×** (đo 2 ảnh trước đó ≈ 1.0× — thiếu đại diện) → so CÙNG THỜI GIAN ở dev100 cần WP2 ở B ≈ 36 (= 41 / 1.15) hoặc giảm chi phí.

---

## 2026-10-21 — TẠM DỪNG (user đi ngủ, mai chạy tiếp): tổng kết phiên V100 + việc mở
**Máy:** Tesla V100-SXM2 16GB (bootstrap lại; đủ 11 checkpoint + COCO; KHÔNG có `.env` → PNG phiên này CHƯA upload HF). Không có tiến trình nào đang chạy.
**Kết quả chính:**
1. **Chẩn đoán đổi surrogate (dev100 B41): TƯỜNG HỌ** — surrogate Swin-T đưa DINO-Swin-L 32.0 → 58.0 (Δ +26.0 [19.6, 29.6]); gap hai chiều đối xứng (Swin-T → R50 78.7 ≈ R50 → Swin-T 77.7). **δ_SwinT vẫn phá R50 78.7% → tồn tại vùng δ chung cho 2 họ mà gradient θ R50 không chỉ tới.**
2. **E16 (lọc δ, dev100): XÁM** — khác biệt δ_SwinT vs δ_R50 là HƯỚNG (cos 0.06), không phải phổ / màu / vị trí; làm mượt δ R50 chỉ hại.
3. **User xác định mục tiêu paper = vượt gap CNN → Transformer (KHÔNG dừng ở kết luận cơ chế).** Ba hướng "δ R50 giống Transformer" phía attack (dev30): BPA NGANG, GLB HẠI, PD HẠI — lại trượt đường đánh đổi.
4. **Hướng mới — sửa hình học gradient của surrogate (nhiễu trọng số R50 trong lúc attack, họ AWT / SAM phía model; user đồng ý vẫn là single surrogate R50):** **C2-OCT-1V-WP2 (ρ = 0.002) @ dev30: Swin-L +7.0 [1.3, 8.3], CA3 +4.3, YOLOX +9.9** — ô ĐẦU TIÊN tăng cả hai phía; WP5 (ρ = 0.005) NGANG. Runtime 1.15× → chưa cùng thời gian.
**Việc tiếp (phiên sau, theo thứ tự):**
- (1) **Xác nhận WP2 @ dev100, CÙNG THỜI GIAN** — khóa quy tắc entry riêng TRƯỚC khi chạy (đề xuất khung GIÚP / HẠI như (d) 10-20): WP2 B36 (= 41 / 1.15; đo lại s/bước trên máy mới, đặt B theo đó) vs C2-OCT-1V B41 cùng máy + OSFD B50 làm mốc SOTA; ~25–30 phút. Máy mới ≠ V100 → sinh lại mọi mốc cùng máy (B cùng thời gian theo máy, như A4500 = B57).
- (2) Qua → sàng lọc dev300 → GO @ n300 kèm held-out YOLOX-L + Mask R-CNN Swin-S. Không qua → các biến thể cùng họ (AWT đầy đủ / SAM phía model không trễ / chỉ stage 3–4), khóa trước.
- (3) Rà literature surrogate-refinement cho detection (AWT, SASD-WS, flat-minima-over-surrogates) — khẳng định novelty; BFDA vẫn mở.
**Code mới phiên này (mặc định giữ hành vi cũ):** `core.run_iterative_attack(weight_perturb)`, `geo_osfd_attack(stage_pool, patch_drop, weight_rho)` (autocast `cache_enabled=False` khi weight_rho), `repair._pooled_mse`, `make_view_loss(stage_pool)`, `attack/methods/bpa.py`; ô `C2-OCT-1V-{BPA, BPA34, GLB7, GLBG, PD16, PD32, WP2, WP5}`; `scripts/e16_delta_band.py`.
**Artifact CHƯA upload (mất khi trả máy; sinh lại được):** `artifacts/runs/dev300_B41_eps5_limit100{,_sur-swin_t}/` (PNG dev100 C2-OCT-1V sur R50 / Swin-T + PNG dev30 các ô BPA / GLB / PD / WP), `artifacts/e16/`. Số liệu + dets tổng hợp đã nằm trong `results/` (git).

---

## 2026-10-22 — Máy mới V100 16GB; xác nhận C2-OCT-1V-WP2 @ dev100 CÙNG THỜI GIAN (việc mở (1) entry 10-21; user: "đợi xong rồi run WP2") — khóa TRƯỚC khi chạy
**Máy:** Tesla V100-SXM2 16GB (cùng loại máy sàng lọc dev300 10-18 / phiên 10-21 → C2-OCT-1V B41 = cùng thời gian OSFD B50), bootstrap lại; đủ 11 checkpoint + COCO. Không có PNG cũ (phiên 10-21 chưa upload) → sinh lại mọi mốc cùng máy.
**Đo thời gian (10 ảnh đầu dev300, chỉ attack, KHÔNG eval; log `results/wp/timing_v100.log`), s / ảnh:** C2-OCT-1V B41 3.9; WP2 B41 4.2 (1.08×); OSFD B50 4.0. dev30 (10-21) cho 1.15×. → chọn **WP2 B36** (= 41 / 1.15, tỉ lệ lớn hơn → bất lợi cho WP2; nếu vẫn GIÚP thì kết luận càng chắc). Runtime thực đo lại trong lần chạy chính.
**Ô:** `C2-OCT-1V-WP2` (ρ = 0.002, y nguyên 10-21; ρ KHÔNG tune thêm) B36 vs `C2-OCT-1V` B41 (cùng thời gian); OSFD B50 làm mốc SOTA; WP2 B41 (cùng số bước) chỉ mô tả.
**Chạy (V100, dev100, seed theo chỉ số dev300; script `results/wp/run_dev100.sh`, log `results/wp/run_dev100.log`; cùng `--out metrics_wp_dev100.json` / dets `dets_wp_dev100.json` mọi run):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --methods C2-OCT-1V C2-OCT-1V-WP2 --out metrics_wp_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 36 --methods C2-OCT-1V-WP2 --out metrics_wp_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 50 --methods OSFD --out metrics_wp_dev100.json
# eval_generalization --models yolox_s dino_swin_l cho 3 run (B41: C2-OCT-1V C2-OCT-1V-WP2; B36: C2-OCT-1V-WP2; B50: OSFD), rồi
python scripts/pair_screen.py --run dev300_B41_eps5_limit100 --subset dev100 --dets dets_wp_dev100.json --tag wp_dev100 --exclude dev30 --pairs \
  C2-OCT-1V-WP2@dev300_B36_eps5_limit100:C2-OCT-1V C2-OCT-1V-WP2:C2-OCT-1V \
  C2-OCT-1V-WP2@dev300_B36_eps5_limit100:OSFD@dev300_B50_eps5_limit100 C2-OCT-1V:OSFD@dev300_B50_eps5_limit100
```
**Quy tắc (dev100, paired bootstrap 1000, BOOT_SEED 2026; CA3 = TB ConvNeXt-T / Swin-T / YOLOX-S; Δ = drop(WP2 B36) − drop(C2-OCT-1V B41), cùng ảnh) — như (d) 10-20:**
- **GIÚP** nếu [(ΔSwin-L ≥ 3 VÀ cận dưới CI > 0 VÀ ΔCA3 ≥ −1) HOẶC (ΔCA3 ≥ 2 VÀ cận dưới CI > 0 VÀ ΔSwin-L ≥ −1)] VÀ ΔR101 ≥ −3 VÀ runtime WP2 B36 ≤ 1.1 × C2-OCT-1V B41 (đo cùng lần).
- **HẠI** nếu ΔCA3 ≤ −3 HOẶC ΔSwin-L ≤ −3 (điểm). **TRÊN ĐƯỜNG** nếu một phía ≥ 2 và phía kia ≤ −1 (điểm). Còn lại → **NGANG**.
- Mô tả (không thuộc quy tắc): **70 ảnh mới (dev100 \ dev30)** — dev30 là nơi phát hiện WP2, nên đọc riêng phần chưa thấy; WP2 B41 − C2-OCT-1V B41 (cùng số bước); WP2 B36 − OSFD B50 và C2-OCT-1V − OSFD (cùng thời gian, tái lập mốc); ConvNeXt / Swin-T / YOLOX / R101 riêng.
- **GIÚP** → bước sau (entry riêng, khóa trước): sàng lọc dev300 so C2-OCT-1V + OSFD cùng thời gian, **thêm đối chứng AWT port sang detection** (code chính thức xaddwell/AWT, cùng thời gian) — điều kiện cho claim novelty. **NGANG / TRÊN ĐƯỜNG / HẠI** → ghi nhận; biến thể cùng họ (SAM phía model không trễ / chỉ stage 3–4 / AWT đầy đủ) chỉ chạy sau khi khóa entry mới.
**Caveat khai báo trước:** 30 / 100 ảnh là dev30 (đã thấy kết quả) → thiên lạc quan cho WP2, vì vậy báo 70 ảnh mới; ρ chọn theo sức khỏe tối ưu trên ảnh 1–4 dev300 (10-21), không nhìn transfer; GPU không tất định → đọc theo CI. Held-out (YOLOX-L, Swin-S) không dùng.

### Kết quả @ dev100 (V100; `results/ablation/wp_dev100/screen.json`, `results/runs/dev300_B{36,41,50}_eps5_limit100/{metrics,generalization}_wp_dev100.json`, log `results/wp/run_dev100.log`) — theo quy tắc: **WP2 B36 GIÚP**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh:
- C2-OCT-1V B41: 96.4 / 92.8 / 82.1 / 78.9 / 70.0 / 30.3 | 77.0 | 3.93
- **WP2 B36 (cùng thời gian):** 95.9 / 91.3 / 84.1 / 78.9 / 72.7 / **36.4** | 78.6 | 3.77
- WP2 B41 (cùng số bước): 96.2 / 92.4 / 83.7 / 80.8 / 74.9 / 37.7 | 79.8 | 4.70
- OSFD B50: 96.6 / 91.6 / 78.6 / 74.3 / 70.7 / 25.0 | 74.6
**Quy tắc** (WP2 B36 − C2-OCT-1V B41): Swin-L **+6.2 [+3.1, +8.2]** (≥ 3, cận dưới > 0) VÀ CA3 +1.6 [−0.1, 3.4] (≥ −1) ✓; R101 −1.5 ≥ −3 ✓; runtime 3.77 ≤ 1.1 × 3.93 ✓ → **GIÚP**. Từng target: ConvNeXt +2.0, YOLOX +2.7, Swin-T −0.0.
**Mô tả:**
- **70 ảnh mới (dev100 \ dev30):** Swin-L **+5.1 [+1.8, +8.1]** — lợi Swin-L giữ ở ảnh chưa thấy; CA3 +0.2 [−0.9, 2.0], Swin-T −1.7 → phần lợi CA3 ở dev30 (+4.3) phần lớn KHÔNG tái lập; WP2 ở cùng thời gian = **tăng Swin-L, giữ nguyên CNN** (không trượt đường đánh đổi, nhưng cũng không tăng CA3).
- Cùng số bước (WP2 B41 − C2 B41): Swin-L +7.4 [4.1, 9.2], CA3 +2.8 [0.4, 4.9] → thêm 5 bước (≈ 1.18× thời gian) cho thêm lợi CA3.
- So SOTA cùng thời gian (WP2 B36 − OSFD B50): Swin-L **+11.5 [+5.7, +14.3]**, CA3 +4.0 [−0.3, 8.1]; C2-OCT-1V − OSFD: Swin-L +5.3, CA3 +2.4 (tái lập mốc, dev100 nhiễu hơn dev300).
**Đọc:** WP2 qua quy tắc khóa ở cùng thời gian; lợi DINO-Swin-L bền trên ảnh mới (~+5–6), lợi CA3 nhỏ / không chắc. **Lưu ý cho claim "CNN → Transformer":** Mask R-CNN Swin-T KHÔNG tăng (−0.0 / −1.7 ảnh mới) → lợi nằm riêng ở DINO-Swin-L, chưa phải ở họ Transformer nói chung; cần xem ở dev300 + held-out Swin-S. Bước tiếp theo quy tắc: sàng lọc dev300 so C2-OCT-1V + OSFD cùng thời gian + đối chứng AWT-OD (entry riêng, khóa trước).

---

## 2026-10-22 — Quick test ε = 8/255 trên 10 ảnh (dev10): WP2 B36 / C2-OCT-1V B41 / OSFD B50 (user: "quick test 10 ảnh ở epsilon = 8") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Mục đích:** ε = 8 là điều kiện phụ của plan (idea.md §7); xem nhanh mức DINO-Swin-L đạt được (câu hỏi user: lên ~50% được không). Cấu hình y hệt dev100 hôm nay (cùng thời gian V100), chỉ đổi `--epsilon 8`; α = 1, ρ = 0.002 KHÔNG đổi (ρ hiệu chỉnh ở ε = 5). Run `dev300_B{41,36,50}_eps8_limit100`, subset dev10 (⊂ dev30); script `results/eps8/run.sh`, log `results/eps8/run.log`.
**Đọc:** n = 10 → chỉ điểm ước lượng, KHÔNG quyết định method, KHÔNG chọn tham số. Mô tả: drop 6 model, Δ WP2 − C2-OCT-1V, Δ WP2 − OSFD; so mốc ε = 5 trên cùng 10 ảnh (từ run dev100 hôm nay).

### Kết quả quick test ε = 8 @ dev10 (`results/ablation/eps8_dev10/screen.json`, log `results/eps8/run.log`; mốc ε = 5 cùng 10 ảnh: `results/ablation/eps5_ref_dev10/screen.json`) — MÔ TẢ, n = 10
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3, ε = 8 (ε = 5 cùng 10 ảnh):
- C2-OCT-1V B41: 100 / 98.2 / 96.0 / 88.9 / 84.8 / **49.4** | 89.9 (Swin-L 32.3 | CA3 76.7)
- WP2 B36: 100 / 95.3 / 94.4 / 91.0 / 84.2 / **50.6** | 89.9 (36.2 | 75.7)
- OSFD B50: 100 / 99.4 / 96.1 / 92.1 / 86.9 / **46.1** | 91.7 (27.9 | 69.6)
Δ WP2 − C2-OCT-1V: ε = 8 Swin-L +1.2 [−0.6, 3.3], CA3 −0.0, R101 −2.9; (ε = 5: Swin-L +4.0). Δ WP2 − OSFD ε = 8: Swin-L +4.5, CA3 −1.8.
**Tiêu chí "khả quan" (user chốt trong hội thoại TRƯỚC khi thấy số: WP2 ε8 Swin-L ≥ 50 VÀ WP2 − C2-OCT-1V ≥ +3):** 50.6 ✓ nhưng +1.2 ✗ → **KHÔNG khả quan** → chuyển hướng "sửa surrogate mạnh hơn".
**Đọc:** tăng ε 5 → 8 cho +17 – 18 điểm Swin-L ở MỌI method (kể cả OSFD 28 → 46) — ε là núm lớn nhất nhưng không phải đóng góp; ở ε = 8 CNN gần bão hòa và các method dồn lại (OSFD ≈ C2 ≈ WP2 trên CA3), lợi riêng của WP2 co lại (+4.0 → +1.2). n = 10 rất nhiễu (CI Swin-T ±10).

---

## 2026-10-22 — Phân tích theo object (dets dev100 có sẵn, CPU) + quick test "attack theo cỡ object": C2-OCT-1V-WP2-SZ / -R / -SZR @ dev30 B36 (user duyệt "quick test 10–30 ảnh") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Phân tích theo object (dev100, dets `artifacts/runs/dev300_B{36,41,50}_eps5_limit100/`; object "bị xóa" = không còn box cùng lớp IoU ≥ 0.5 score ≥ 0.3; chỉ object model phát hiện được khi sạch):**
- DINO-Swin-L (734 object), tỉ lệ bị xóa OSFD / C2-OCT-1V / WP2: tất cả 25 / 27 / 30; **nhỏ 29 / 39 / 42; vừa 28 / 25 / 30; lớn 14 / 8 / 10**. Swin-T (744): nhỏ 62 / 68 / 67; vừa 59 / 59 / 58; **lớn 68 / 52 / 53**.
- Bù trừ (cận trên lý tưởng, object bị ÍT NHẤT 1 trong OSFD / WP2 xóa): Swin-L 30 → 36%, Swin-T 60 → 70%.
- Lồng nhau: P(xóa Swin-L | xóa Swin-T) = 0.43 (WP2) / 0.38 (C2) vs P(xóa Swin-L | Swin-T còn) = 0.05 → kênh phá Swin-L ⊂ kênh phá Swin-T; WP2 xóa SÂU hơn trên Swin-L (50 object chỉ WP2 xóa vs 24 chỉ C2) nhưng KHÔNG mở rộng tập xóa trên Swin-T.
**Giả thuyết:** method hiện tại (L_cls + OCT) mạnh ở object nhỏ, OSFD (loss feature + RRB xoay) mạnh ở object lớn → dồn L_OSFD vào object lớn (+ xoay) lấy lại phần object lớn mà không mất phần nhỏ.
**Ô (`repair._size_weighted_mse`, `make_view_loss(size_w)`, `geo_osfd_attack(size_w)`):** gốc = WP2 (C2-OCT-1V, ρ = 0.002).
- `WP2-SZ`: L_OSFD từng stage = mean(w·(k·F_sạch − F_adv)²) / mean(w), w = 1 + 3 trong box GT của view có diện tích quy về ảnh gốc ≥ 96² (quy theo scale_factor; box sau co giãn OCT — object sát ngưỡng có thể đổi phía giữa các view), 1 ngoài; trung bình kênh. L_cls, λ = 4 không đổi.
- `WP2-R`: OCT thêm xoay θ ~ U(−7°, 7°) (liều RRB của OSFD) quanh tâm, GT co-transform (elem "rot") — tách riêng vai trò xoay.
- `WP2-SZR`: cả hai.
Liều a = 3, ngưỡng 96², 7° chọn trước theo định nghĩa COCO / OSFD, KHÔNG tune trên dev30.
**Kiểm:** a = 0 trùng tuyệt đối F.mse_loss; chỉ box nhỏ → trùng MSE; sai khác dồn trong box lớn → loss tăng (4.23 → 7.08); chạy thử 3 ảnh: L∞ 5.0, runtime ≈ WP2 (3.8–3.9 s / ảnh) → B36 cho mọi ô = cùng thời gian.
**Chạy (V100, dev30 ⊂ dev100; PNG WP2 B36 dev100 tái dùng; script `results/sz/run.sh`, log `results/sz/run.log`).**
**Đọc xu hướng (dev30, điểm ước lượng, như BPA / WP 10-21; KHÔNG quyết định method):** so WP2 B36. "đáng lên dev100" nếu ΔSwin-T ≥ +3 VÀ ΔSwin-L ≥ −1 VÀ ΔCA3 ≥ −1 (mục tiêu chính: mở rộng tập xóa trên Transformer, cửa ngõ của Swin-L); "hại" nếu ΔSwin-L ≤ −3 HOẶC ΔCA3 ≤ −3; còn lại "ngang". Mô tả: tỉ lệ xóa object lớn trên Swin-T / Swin-L (cùng định nghĩa trên), Δ Swin-L, R101.

### Kết quả SZ @ dev30 B36 (V100; `results/ablation/sz_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/{metrics,generalization}_sz_dev30.json`, log `results/sz/run.log`) — xu hướng: **SZ NGANG, R NGANG, SZR NGANG**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: WP2 98.7 / 91.0 / 81.4 / 83.0 / 76.8 / 36.2 | 80.4; **SZ** 99.0 / 92.3 / 81.9 / 84.2 / 76.8 / 37.0 | 81.0; **R** 97.6 / 90.2 / 82.0 / 81.9 / 75.1 / 36.3 | 79.7; **SZR** 98.3 / 90.4 / 84.1 / 85.1 / 76.3 / 35.0 | 81.8.
Δ so WP2: SZ Swin-T +1.2, Swin-L +0.7, CA3 +0.6; R Swin-T −1.1, Swin-L 0.0, CA3 −0.7; SZR Swin-T +2.0 [−2.6, 7.6], Swin-L −1.2, CA3 +1.4 → không ô nào đạt ΔSwin-T ≥ +3.
**Mô tả — tỉ lệ object bị xóa theo cỡ (dev30; Swin-L 265 object, lớn 44; Swin-T 270, lớn 45), WP2 / SZ / R / SZR:** Swin-L lớn 11 / 18 / 14 / 14%, tất cả 30 / 31 / 32 / 29%; Swin-T lớn 60 / 62 / 62 / 64%, tất cả 58 / 59 / 58 / 59%.
**Đọc:** dồn L_OSFD vào object lớn có nhích object lớn trên Swin-L (11 → 18%, ≈ +3 object / 44 — n quá nhỏ) nhưng KHÔNG lấy lại được khoảng cách OSFD ở object lớn trên Swin-T (dev100: OSFD 68 vs WP2 53) và không đổi tổng. Xoay RRB không giúp. → Phần OSFD thắng ở object lớn KHÔNG nằm ở "trọng số không gian của L_OSFD" hay "xoay" trong khung WP2; có thể do L_cls (λ = 4) lấn át hướng ở object lớn, hoặc do 2 view / bước của RRB. Không đưa lên dev100; không thêm liều a / ngưỡng (theo khai báo).

---

## 2026-10-22 — Quick test WP2-SZC (bỏ L_cls RoI ở object lớn) @ dev30 B36 (user chọn "a") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Giả thuyết (từ SZ NGANG ở trên):** ở object lớn, L_cls (λ = 4) lấn át hướng gradient → dồn thêm L_OSFD không đủ; phải GIẢM L_cls ở object lớn để object lớn được đánh gần như OSFD (OSFD thắng ở object lớn: Swin-T 68 vs 53, Swin-L 14 vs 10, dev100), object nhỏ giữ L_cls (điểm mạnh hiện tại).
**Ô `C2-OCT-1V-WP2-SZC` (`repair.roi_size_weight`, `geo_osfd_attack(roi_size_w)`):** WP2 y nguyên; loss_cls RoI head nhân **0** ở RoI DƯƠNG gán cho GT có diện tích (khung view) ≥ 96² · sx · sy (quy về ảnh gốc theo scale_factor; box sau co giãn OCT — object sát ngưỡng có thể đổi phía), avg_factor giữ nguyên (không dồn sang RoI khác). RoI nền và loss_rpn_cls KHÔNG đổi (giới hạn khai báo: anchor RPN của object lớn vẫn bị đẩy). L_OSFD không đổi. Liều 0 chọn trước (kiểm thẳng giả thuyết), KHÔNG tune.
**Kiểm (`scripts/szc_unit_test.py`, 5 ảnh đầu dev30, δ = 0, cùng seed RoI):** w = 1 trùng loss_cls gốc (5 / 5); w = 0 giảm đúng ở ảnh có GT lớn (vd 7088: 0.1135 → 0.1016; 17959: 0.4651 → 0.3709), ảnh không GT lớn không đổi (11615); phương thức gốc khôi phục. Chạy thử 3 ảnh: L∞ 5.0, ~3.8 s / ảnh (= WP2) → B36.
**Chạy (V100, dev30; PNG WP2 B36 tái dùng; script `results/sz/run_szc.sh`, log `results/sz/run_szc.log`).**
**Đọc xu hướng (như SZ):** so WP2 B36. "đáng lên dev100" nếu ΔSwin-T ≥ +3 VÀ ΔSwin-L ≥ −1 VÀ ΔCA3 ≥ −1; "hại" nếu ΔSwin-L ≤ −3 HOẶC ΔCA3 ≤ −3; còn lại "ngang". Mô tả: tỉ lệ xóa object theo cỡ trên Swin-T / Swin-L (object nhỏ phải KHÔNG giảm — nếu giảm thì L_cls ở object lớn cũng góp cho object nhỏ).

### Kết quả SZC @ dev30 B36 (V100; `results/ablation/szc_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/{metrics,generalization}_szc_dev30.json`, log `results/sz/run_szc.log`) — xu hướng: **NGANG theo quy tắc (sát ngưỡng HẠI), hại phía CNN**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: WP2 98.7 / 91.0 / 81.4 / 83.0 / 76.8 / 36.2 | 80.4; **SZC** 98.2 / 86.6 / 80.4 / 81.6 / 70.3 / 37.6 | 77.5.
Δ so WP2 (số chính xác): Swin-L +1.40 [−2.9, 3.5], CA3 **−2.95** [−4.3, 0.6] (> −3 → không chạm "hại"), Swin-T −1.43, YOLOX **−6.46**, R101 −4.31, ConvNeXt −0.97 → không đạt ΔSwin-T ≥ +3 → NGANG.
**Mô tả — tỉ lệ xóa theo cỡ (dev30), WP2 / SZC:** Swin-L nhỏ 40 / 41, vừa 27 / 26, lớn 11 / 14%; Swin-T nhỏ 67 / 67, vừa 47 / 51, **lớn 60 / 58%**.
**Đọc:** bỏ L_cls ở object lớn KHÔNG lấy lại lợi thế object lớn của OSFD trên Swin-T (60 → 58%) và làm mất phía CNN (YOLOX −6.5, R101 −4.3) → giả thuyết "L_cls lấn át ở object lớn" BỊ BÁC; L_cls ở object lớn đang có ích cho CNN. Cùng SZ / R / SZR: phần OSFD thắng ở object lớn không tái tạo được bằng cách trộn thành phần OSFD vào khung WP2 theo cỡ (trọng số không gian, xoay, bỏ cls) — còn lại giả thuyết 2 view / bước của RRB (chưa kiểm). Không đưa lên dev100.

---

## 2026-10-22 — "RRB cho texture": C2-OCT-1V-WP2-TXC2 / -TXC4 (làm mờ riêng nội dung sạch của view) @ dev30 B36 (user duyệt "Approve") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lập luận:** RRB (+32) không "tune" — nó xóa một LỐI TẮT đặc thù surrogate (hình học chính xác). Lối tắt tương ứng cho CNN → Transformer theo dữ liệu dự án: **texture / cấu trúc cục bộ** — E6: DINO-Swin-L nhận object (nhất là lớn) bằng HÌNH DẠNG, R50 bằng cấu trúc cục bộ; phân tích theo object hôm nay: method mạnh ở object nhỏ, yếu ở object lớn. → mỗi bước cho R50 nhìn ảnh có texture nội dung bị xóa ngẫu nhiên (giữ hình dạng, như BLUR của E6) → gradient theo δ buộc nhắm kênh hình dạng. KHÁC S1 / SEA / E16 (làm mượt δ / gradient) và PD (che, phá tối ưu): chỉ biến đổi NỘI DUNG x, δ giữ băng rộng.
**Ô (`geo_osfd_attack(tex_blur)`, `pia._blur`):** WP2 y nguyên; mỗi view sau OCT: v' = v + (Blur_s(c) − c) = Blur_s(warp x) + warp δ + nhiễu σ = 6 (view adv elem có nhiễu, view sạch không), s ~ U(0, σ_max) px khung model (Gaussian tách được, pad reflect, s < 0.3 → không mờ); feature sạch L_OSFD tính trên Blur_s(c); box GT không đổi; RNG riêng seed + 9 → chuỗi OCT trùng ô gốc.
**Hiệu chỉnh liều — CHỈ sức khỏe white-box (`scripts/tx_calib.py`, ảnh 1–4 dev300, chỉ R50, không đo target), B36:** |δ| = 5 / R50 xóa: WP2 79% / 92% (34 / 37); σ_max 1, 2: 79% / 92%; 3: 78% / 89%; 4: 78% / 86%; **6, 8: 77% / 54% (sụp)**. → **TXC2** (lớn nhất chưa mất white-box), **TXC4** (lớn nhất trước khi sụp). Runtime 3.7–3.8 s / ảnh ≈ WP2 → B36 = cùng thời gian.
**Chạy (V100, dev30; PNG WP2 B36 tái dùng; `results/sz/run_tx.sh`, log `results/sz/run_tx.log`).**
**Đọc (khai báo trước; so WP2 B36 cùng 30 ảnh; ngưỡng cao theo yêu cầu user "khác biệt rõ rệt như RRB"):**
- **BƯỚC NHẢY** nếu ΔSwin-L ≥ +10 → lên dev100 ngay (khóa entry).
- **CÓ TÍN HIỆU** nếu +3 ≤ ΔSwin-L < +10 VÀ ΔCA3 ≥ −3 → xem liều / kết hợp (khóa entry).
- **KHÔNG** nếu ΔSwin-L < +3 → lối tắt texture (ở dạng blur nội dung) không phải nút thắt; chuyển hướng.
- Mô tả: Swin-T, tỉ lệ xóa object lớn Swin-T / Swin-L (giả thuyết dự đoán tăng), white-box R50, R101.

### Kết quả TX @ dev30 B36 (V100; `results/ablation/tx_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/{metrics,generalization}_tx_dev30.json`, log `results/sz/run_tx.log`) — theo khai báo: **KHÔNG** (cả hai liều HẠI mạnh, theo liều)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: WP2 98.7 / 91.0 / 81.4 / 83.0 / 76.8 / 36.2 | 80.4; **TXC2** 95.4 / 87.6 / 73.9 / 75.1 / 66.6 / 31.2 | 71.9; **TXC4** 90.7 / 79.8 / 63.5 / 65.3 / 54.6 / 18.8 | 61.1.
Δ so WP2: TXC2 Swin-L −5.0 [−9.6, 0.0], CA3 −8.5, Swin-T −7.9; TXC4 Swin-L −17.4, CA3 −19.3, Swin-T −17.7.
**Mô tả — tỉ lệ xóa theo cỡ (dev30), WP2 / TXC2 / TXC4:** Swin-L lớn 11 / 11 / 11%, vừa 27 / 18 / 12, nhỏ 40 / 33 / 27; Swin-T lớn 60 / 58 / 40, nhỏ 67 / 59 / 49.
**Đọc:** giả thuyết "lối tắt texture" ở dạng blur nội dung BỊ BÁC: kể cả liều nhẹ (σ_max 2, không mất white-box trên ảnh hiệu chỉnh) đã hạ MỌI target, Swin-L / Swin-T giảm ngang CNN, object lớn trên Swin-L không nhích (11%). → tác dụng transfer của δ GẮN VỚI texture của chính nội dung ảnh (khớp E7: δ là mẫu khớp nội dung): tối ưu δ trên nội dung đã mờ làm δ lệch khỏi nội dung thật → yếu khắp nơi, không chuyển sang kênh hình dạng. Không thêm liều / dạng xóa texture khác (stylize, bilateral) mà không khóa entry mới.

---

## 2026-10-22 — Đo trần họ WP: WP2 B100 (+ đối chứng C2-OCT-1V B100) @ dev30 (user duyệt "approve") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Câu hỏi:** cùng công thức WP2 (ρ = 0.002), nhiều bước hơn (B100 ≈ 2.8× thời gian WP2 B36) thì DINO-Swin-L lên tới đâu — họ "can thiệp θ surrogate" còn dư địa để tiến về 50% ở ε = 5 không? Đối chứng C2-OCT-1V B100: WP2 có tăng theo B tốt hơn method nền (giả thuyết "mỗi bước thấy θ' khác → ensemble theo thời gian")?
**Chạy (V100, dev30; run `dev300_B100_eps5_limit100`; WP2 B36 eval lại với dets `dets_b100_dev30.json` để ghép cặp; `results/b100/run.sh`, log `results/b100/run.log`).**
**Đọc (khai báo trước, điểm ước lượng dev30):**
- **CÒN DƯ ĐỊA** nếu drop Swin-L(WP2 B100) ≥ 45 → hướng tiếp: đạt hiệu quả nhiều bước trong cùng thời gian (WPA tích lũy / nhiễu theo stage liều cao), khóa entry riêng.
- **BÃO HÒA** nếu < 42 → với 1 surrogate R50 ở ε = 5, 50% Swin-L gần như ngoài tầm họ method hiện tại → user quyết định vị paper.
- 42–45 → XÁM. Mô tả: Δ WP2 B100 − C2-OCT-1V B100 (WP có tăng theo B tốt hơn không, so với Δ ở B36/B41); CA3, Swin-T, R101; runtime.

### Kết quả đo trần @ dev30 (V100; `results/ablation/b100_dev30/screen.json`, `results/runs/dev300_B{100,36}_eps5_limit100/{metrics,generalization}_b100_dev30.json`, log `results/b100/run.log`) — theo khai báo: **BÃO HÒA** (Swin-L WP2 B100 = 38.0 < 42)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3 | s/ảnh: WP2 B36 98.7 / 91.0 / 81.4 / 83.0 / 76.8 / 36.2 | 80.4 | 3.8; **WP2 B100** 99.8 / 95.7 / 86.6 / 89.0 / 80.4 / **38.0** | 85.3 | 10.9; C2-OCT-1V B100 99.9 / 95.4 / 87.9 / 84.9 / 76.3 / 34.6 | 83.0 | 9.6.
Δ WP2 B100 − B36: Swin-L **+1.8** [−3.6, 5.1], CA3 +4.9 [3.6, 10.1], Swin-T +6.0, R101 +4.7. Δ WP2 − C2-OCT-1V (cùng B100): Swin-L +3.4 [−2.6, 7.1], Swin-T +4.1 [1.3, 7.1], CA3 +2.3, YOLOX +4.1.
**Đọc:** gần 3× thời gian chỉ thêm ~2 điểm Swin-L → họ WP (ρ = 0.002) bão hòa trên DINO-Swin-L quanh 36–38% ở ε = 5; thêm bước chủ yếu lợi phía CNN / Swin-T. Lợi riêng của WP so method nền vẫn giữ ở B100 (Swin-L +3.4, Swin-T +4.1 — Swin-T lần đầu dương rõ). Theo khai báo: 50% Swin-L với 1 surrogate R50 ở ε = 5 gần như ngoài tầm họ method hiện tại (loss / đa dạng đầu vào / nhiễu trọng số 1 bước trễ) → user quyết định vị paper.

---

## 2026-10-22 — Lối tắt "biên độ": C2-OCT-1V-WP2-LN (L_OSFD trong không gian LayerNorm) @ dev30 B36; WP2-IN loại ở cổng sức khỏe (user duyệt "Approve") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lập luận:** L_OSFD đẩy F_adv về k·F_sạch (k = 3) = khuếch đại BIÊN ĐỘ. CNN (BN thống kê cố định lúc suy luận) truyền nguyên thay đổi biên độ; Transformer có LayerNorm trước mọi khối → xóa thay đổi biên độ / lệch theo token, chỉ còn HƯỚNG vector feature mỗi vị trí. Khớp: E16 (khác biệt = hướng), loss cls giúp riêng Swin-L, loss feature giúp CNN. Cùng cấu trúc RRB (xóa lối tắt surrogate có mà target không có). Chưa từng thử loss bất biến biên độ trong dự án.
**Ô `C2-OCT-1V-WP2-LN` (`repair._ln_mse`, `make_view_loss(ln_feat)`, `geo_osfd_attack(ln_feat)`):** WP2 y nguyên trừ L_OSFD mỗi stage = MSE(LN_c(k·F_sạch), LN_c(F_adv)) (LN theo kênh tại từng vị trí, không affine; LN(k·F) = LN(F) → k vô tác dụng), nhân hệ số detach để GIÁ TRỊ = MSE gốc (giữ cân bằng với 4·L_cls như GLB, không tune). Kiểm: giá trị = MSE gốc; F_adv = 2.5·F + 0.7 → loss LN = 0.
**`WP2-IN` (BN backbone theo thống kê ảnh, "all" / "s34") — LOẠI trước khi chạy (cổng sức khỏe, ảnh 1–4 dev300):** ở chế độ IN, R50 chỉ phát hiện 5 / 37 ("all") và 6 / 37 ("s34") GT khi sạch (thường: 37) → surrogate hỏng, gradient vô nghĩa (δ của nó "xóa 100%" chỉ vì còn quá ít object). Không chạy dev30.
**Sức khỏe WP2-LN (ảnh 1–4 dev300, chỉ R50):** |δ| = 5 ở 82% pixel (WP2 79%), R50 white-box xóa 86% (32 / 37; WP2 92%) — đạt; runtime 4.0 s / ảnh ≈ WP2 → B36.
**Chạy (V100, dev30; `results/sz/run_ln.sh`, log `results/sz/run_ln.log`).** **Đọc (như TX):** so WP2 B36 cùng 30 ảnh — **BƯỚC NHẢY** ΔSwin-L ≥ +10; **CÓ TÍN HIỆU** +3 ≤ ΔSwin-L < +10 VÀ ΔCA3 ≥ −3; **KHÔNG** ΔSwin-L < +3. Mô tả: Swin-T (cũng có LN — cơ chế đúng thì phải tăng cùng chiều), CNN (dự đoán giảm vì mất kênh biên độ), tỉ lệ xóa theo cỡ.

### Kết quả LN @ dev30 B36 (V100; `results/ablation/ln_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/{metrics,generalization}_ln_dev30.json`, log `results/sz/run_ln.log`) — theo khai báo: **KHÔNG** (ΔSwin-L +1.6 < 3)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: WP2 98.7 / 91.0 / 81.4 / 83.0 / 76.8 / 36.2 | 80.4; **LN** 98.4 / 89.6 / 83.9 / 83.8 / 75.5 / 37.9 | 81.1. Δ: Swin-L +1.6 [−2.9, 5.8], Swin-T +0.8, CA3 +0.7, ConvNeXt +2.5, YOLOX −1.3, R101 −1.3.
Tỉ lệ xóa theo cỡ (WP2 / LN): Swin-L lớn 11 / 16%, tất cả 30 / 29; Swin-T vừa 47 / 53, tất cả 58 / 60.
**Đọc:** bỏ thành phần biên độ khỏi L_OSFD gần như KHÔNG đổi gì (mọi Δ trong nhiễu, CNN cũng không giảm như dự đoán) → kênh biên độ không phải thứ làm δ R50 thắng CNN, và cũng không phải thứ chặn Transformer; lối tắt "biên độ" BỊ BÁC. Cùng TX (texture), SZ (cỡ object), BPA / GLB / PD (10-21): giả thuyết "δ R50 khai thác 1 thuộc tính đặc thù CNN mà Transformer bỏ qua" chưa tìm ra thuộc tính nào tách được.

---

## 2026-10-22 — E17: hướng NGỮ NGHĨA "xóa object" trong không gian pixel có trung lập họ backbone không? (bước 0 của hướng "đích = ảnh đã xóa object"; user duyệt "Approve") — MÔ TẢ, forward-only, khai báo TRƯỚC khi chạy
**Lý do:** WB thắng nhờ HƯỚNG — trên Swin-L, kéo token object stage 3 về phía nền rất hiệu quả (E14c / E15), nhưng "nền theo feature R50" ≠ "nền theo Swin-L" (BG-OCT âm). Thứ chung giữa 2 họ là NGỮ NGHĨA ảnh → định nghĩa hướng qua ảnh x_bg = x đã xóa object (inpaint OpenCV Telea r = 5 trong GT box không-crowd; không model học → không thêm surrogate; GT-assisted). Trước khi viết attack có đích F_R50(x_bg), kiểm tiền đề rẻ: chính hướng pixel "về x_bg" ở ±5 có tác dụng ngang nhau lên CNN và Transformer không.
**Điều kiện (`scripts/e17_inpaint_probe.py`, dev100, PNG `artifacts/e17/`, kết quả `results/mechanism/e17_inpaint_probe.json`, log `results/e17/run.log`):** INP: δ = 5·sign(x_bg − x) trong box, 0 ngoài; RNDBOX: δ = ±5 ngẫu nhiên trong box (cùng vùng, cùng L∞; seed 2026 + img_id); WP2: PNG C2-OCT-1V-WP2 B36 dev100 (mốc transfer). Đo relative AP drop 6 model qua inference_detector (cùng cách cho clean; KHÔNG so trực tiếp số run_baselines), paired bootstrap 1000, BOOT_SEED 2026.
**Đọc (khai báo trước):** r(c) = drop_SwinL(c) / TB drop(R101, ConvNeXt, YOLOX).
- **TIỀN ĐỀ ĐÚNG** nếu r(INP) ≥ 0.8 VÀ drop_SwinL(INP) − drop_SwinL(RNDBOX) ≥ 5 với cận dưới CI > 0 → hướng ngữ nghĩa pixel trung lập họ; bước 1 = WP2 + loss đích F_R50(x_bg) @ dev30 (khóa entry riêng).
- **QUÁ YẾU** nếu drop INP < 5 ở mọi model → ±5 theo hướng này không đủ để đọc; XÁM (user quyết bước 1).
- **SAI** nếu r(INP) ≤ r(WP2) (Swin-L vẫn kháng như với δ R50) → dừng hướng.
- Còn lại → XÁM. Mô tả: Swin-T, R50.

### Kết quả E17 @ dev100 (`results/mechanism/e17_inpaint_probe.json`, log `results/e17/run.log`, dets `artifacts/e17/`) — theo khai báo: **QUÁ YẾU → XÁM** (user quyết bước 1)
Relative AP drop %, R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | r = Swin-L / TB CNN3:
- INP: 2.0 / 0.4 / 0.6 / 2.1 / 2.0 / 2.0 | r 1.93
- RNDBOX: 0.5 / 2.5 / 0.2 / 1.6 / 1.6 / 0.7 | 0.48
- WP2 (mốc): 95.9 / 91.3 / 84.1 / 78.9 / 72.7 / 36.4 | 0.44
Swin-L INP − RNDBOX +1.3 [−1.5, 2.3]; CNN3 −0.4. Box chiếm 44% pixel; INP δ ≠ 0 ở 98% pixel trong box.
**Đọc:** nhích pixel object về phía ảnh đã xóa object ở ±5 gần như KHÔNG tác dụng lên MỌI model (~2%, ngang nhiễu ngẫu nhiên cùng vùng) → hướng ngữ nghĩa "thô" ở mức pixel quá yếu ở ε = 5; r(INP) 1.93 vô nghĩa ở mức nhiễu. KHÔNG bác cũng KHÔNG xác nhận ý "đích = F_R50(x_bg)" (bản attack tối ưu theo đích, khác hẳn δ cố định) — chỉ cho biết lợi nếu có phải đến từ tối ưu, không phải từ bản thân hướng pixel.

---

## 2026-10-22 — TẠM DỪNG (user, 22:35 GMT+7): tổng kết phiên V100 + việc mở
**Máy:** Tesla V100-SXM2 16GB (bootstrap lại; đủ 11 checkpoint + COCO; CÓ `.env` nhưng bootstrap chạy trước khi có → PNG n300 chưa tải). Không có tiến trình nào đang chạy. Commit phiên này CHƯA push, PNG CHƯA upload HF (chờ user đồng ý).
**Kết quả chính (ε = 5, surrogate R50, cùng thời gian OSFD trừ khi ghi khác):**
1. **WP2 @ dev100: GIÚP** — WP2 B36 − C2-OCT-1V B41: Swin-L **+6.2 [3.1, 8.2]** (70 ảnh mới +5.1), CA3 +1.6, Swin-T 0.0; WP2 − OSFD B50: Swin-L **+11.5 [5.7, 14.3]**, CA3 +4.0. Lợi WP2 nằm ở DINO-Swin-L, KHÔNG ở Mask R-CNN Swin-T (cùng thời gian).
2. **Đo trần họ WP @ dev30: BÃO HÒA** — WP2 B100 Swin-L 38.0 (+1.8 so B36 cho ~3× thời gian); WP2 − C2-OCT-1V ở B100: Swin-L +3.4, Swin-T +4.1 [1.3, 7.1].
3. ε = 8 @ dev10 (mô tả): mọi method Swin-L ~46–51; lợi WP2 co còn +1.2; CNN gần trần, OSFD ≈ C2 ≈ WP2 → user gác ε = 8, tập trung ε = 5.
4. **Phân tích theo object (dev100):** method mạnh ở object nhỏ, OSFD thắng ở object lớn (Swin-T 68 vs 53); xóa trên Swin-L ⊂ xóa trên Swin-T (P = 0.43 vs 0.05).
5. **Chuỗi "bước nhảy kiểu RRB" (dev30, so WP2) — đều không đạt:** SZ / R / SZR / SZC (cỡ object) NGANG; TX (xóa texture nội dung) HẠI theo liều; LN (L_OSFD bất biến biên độ) KHÔNG (+1.6); IN (BN theo ảnh) loại ở cổng sức khỏe; E17 (hướng pixel về ảnh xóa object) QUÁ YẾU (~2% mọi model).
6. Tổng hợp cơ chế WB (E8 / E14 / E14b / E14c / E15, đọc lại): WB thắng nhờ HƯỚNG ở token object stage 3 (≈ xóa ngữ nghĩa về nền), không nhờ độ lớn; δ R50 chỉ ~2× nhiễu ngẫu nhiên theo hướng đó (WB ~5×).
**Trạng thái method:** tốt nhất = **C2-OCT-1V-WP2** (ρ = 0.002, B36 trên V100 = cùng thời gian OSFD B50). Trần Swin-L thực tế ~36–40% ở ε = 5 với 1 surrogate R50 (mọi họ đã thử). **Mục tiêu user: method mới vượt "bước nhảy kiểu RRB", Swin-L > 50% ở ε = 5** — user KHÔNG duyệt định vị "SOTA + cơ chế" (phương án A), muốn tiếp tục tìm.
**Việc mở (phiên sau):**
- (1) Ý chưa chạy: **WP2 + loss đích F_R50(x_bg)** (x_bg = inpaint xóa object; E17 không ủng hộ cũng không bác) @ dev30; **thành phần universal** (u học trên ~500 ảnh COCO val ngoài mọi danh sách, R50 + WP + OCT; UNI / UNI+WP2) @ dev30 — đa dạng GIỮA ảnh, trục chưa từng thử. Khóa entry trước khi chạy, quy tắc "bước nhảy" ΔSwin-L ≥ +10.
- (2) Biến thể họ WP (nếu quay lại): WPA (θ tích lũy kiểu AWT), WP-s34 liều cao, WP-cls; đối chứng AWT-OD bắt buộc cho claim novelty.
- (3) Khi chốt method: sàng lọc dev300 → GO @ n300 + held-out YOLOX-L / Swin-S; rà BFDA.
**Code mới phiên này (mặc định giữ hành vi cũ):** `repair.{_size_weighted_mse, _ln_mse, roi_size_weight}`, `make_view_loss(size_w, ln_feat)`, `geo_osfd_attack(size_w, roi_size_w, tex_blur, ln_feat, bn_inst)`, `pia._blur`; ô `C2-OCT-1V-WP2-{SZ, R, SZR, SZC, TXC1..8, LN, INall, INs34}`; `scripts/{szc_unit_test, tx_calib, e17_inpaint_probe}.py`.
**Artifact CHƯA upload (mất khi trả máy; sinh lại được):** `artifacts/runs/dev300_B{36,41,50,100}_eps5_limit100/`, `dev300_B{36,41,50}_eps8_limit100/`, `artifacts/e17/`. Số liệu + dets tổng hợp đã ở `results/` (git); dets thô ở `artifacts/` (không git).

---

## 2026-10-23 — Máy mới V100 16GB; cổng 2 "đích F_R50(x_bg)": C2-OCT-1V-WP2-XBG / -XBGS @ dev30 (user: "làm cổng 2 đi", việc mở (1) entry 10-22) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Máy:** Tesla V100-SXM2 16GB, bootstrap lại; đủ 11 checkpoint + COCO; KHÔNG có `.env` → không có PNG cũ, mốc WP2 B36 dev30 sinh lại cùng máy.
**Lập luận:** WB trên Swin-L thắng nhờ HƯỚNG "token object → nền" (E14c / E15); BG-OCT (kéo về μ nền theo feature R50) âm vì "nền theo R50" ≠ "nền theo Swin-L"; E17: hướng pixel cố định về x_bg ở ±5 quá yếu (~2% mọi model) nhưng không bác bản TỐI ƯU theo đích. Ở đây đích là feature R50 của CHÍNH ảnh đã xóa object (ngữ nghĩa chung giữa 2 họ), δ được tối ưu để F_adv trong box tiến về đó.
**Ô (`repair._xbg_mse`, `make_view_loss(xbg_feats, xbg_mode)`, `geo_osfd_attack(xbg)`):** gốc = WP2 (C2-OCT-1V, ρ = 0.002, L_cls λ = 4 không đổi). x_bg = ảnh gốc inpaint OpenCV Telea r = 5 trong mọi box GT (như E17; không model học → không thêm surrogate; GT-assisted), resize về khung model, đi kèm ảnh sạch qua CÙNG phép biến đổi OCT (companion 6 kênh); F_bg = backbone R50 (θ' của WP, no_grad) trên view x_bg → +1 forward / view. Mỗi stage backbone (cả 4, như L_OSFD; KHÔNG chọn stage), mặt nạ M = ô giao box GT của view, sq = mean_c (k·F_sạch − F_adv)², T = mean_{M} mean_c (F_bg − F_adv)², c = detach(frac_M · mean_M sq / T) (giá trị c·T = phần MSE gốc trong box, quy ước như LN / GLB, không tune):
- `WP2-XBG` ("add"): L_feat = Σ_s [MSE gốc − c·T] — giữ đẩy OSFD khắp nơi, thêm kéo về F_bg trong box.
- `WP2-XBGS` ("swap"): L_feat = Σ_s [phần MSE gốc NGOÀI box − c·T] — trong box chỉ kéo về F_bg (tránh 2 số hạng giằng nhau trong box).
**Kiểm:** `scripts/xbg_unit_test.py` (không box = F.mse_loss; giá trị add / swap đúng công thức; gradient số hạng kéo chỉ trong box, cos(∇T, F_adv − F_bg) = 1.000). Ảnh x_bg nhìn bằng mắt (dev300 #4: object bị xóa thành vùng nhòe). Sức khỏe white-box (`scripts/tx_calib.py`, ảnh 1–4 dev300, B36, chỉ R50, KHÔNG đo target): |δ| = 5 / R50 xóa — WP2 79% / 92%; XBG 79% / 92%; XBGS 78% / 86% → đạt. **Thời gian (cùng lần, ảnh 1–4, `results/xbg/calib.log`):** WP2 4.14, XBG 5.01, XBGS 5.02 s / ảnh (1.21×) → **B30 = cùng thời gian WP2 B36** (36 / 1.21 = 29.7; làm tròn lên, lệch < 1% thời gian).
**Chạy (V100, dev30; `results/xbg/run.sh`, log `results/xbg/run.log`; metrics `metrics_xbg_dev30.json`, dets `dets_xbg_dev30.json`):** WP2 B36 (mốc, sinh lại), XBG / XBGS B30 (quy tắc) và B36 (cùng số bước, mô tả).
**Đọc (khai báo trước; so WP2 B36 cùng 30 ảnh; như TX / LN):**
- **BƯỚC NHẢY** nếu ΔSwin-L ≥ +10 (ô B30) → lên dev100 ngay (khóa entry).
- **CÓ TÍN HIỆU** nếu +3 ≤ ΔSwin-L < +10 VÀ ΔCA3 ≥ −3 → xem biến thể (stage 3 riêng, inpaint tốt hơn) / dev100 (khóa entry).
- **KHÔNG** nếu ΔSwin-L < +3 → đích "ảnh xóa object theo feature R50" không mở hướng mới; đóng hướng (1a).
- Mô tả: Swin-T, R101, CNN; B36 − WP2 B36 (cùng số bước); tỉ lệ object bị xóa theo cỡ trên Swin-L / Swin-T.
**Caveat khai báo trước:** Telea r = 5 thô — box lớn bị nhòe gần toàn bộ (vd dev300 #2: box chiếm 86% ảnh), đích ở đó gần "ảnh nhòe" hơn "nền thật"; dev30 đã dùng cho nhiều ô trước (không chọn tham số ở đây); GPU không tất định → đọc theo CI; n = 30.

### Kết quả cổng 2 @ dev30 (V100; `results/ablation/xbg_dev30/screen.json`, `results/runs/dev300_B{30,36}_eps5_limit100/{metrics,generalization}_xbg_dev30.json`, log `results/xbg/run.log`) — theo khai báo: **KHÔNG** (cả 2 ô HẠI mọi target)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: WP2 B36 (sinh lại) 98.5 / 92.1 / 83.2 / 83.2 / 75.2 / 37.5 | 80.5 (10-22: 36.2 | 80.4 — tái lập); **XBG B30** 98.3 / 91.0 / 76.8 / 75.7 / 72.1 / 28.5 | 74.9; **XBGS B30** 96.5 / 87.2 / 73.9 / 75.7 / 56.7 / 27.8 | 68.8; XBG B36 98.9 / 91.7 / 80.1 / 76.2 / 63.8 / 26.4 | 73.4; XBGS B36 97.4 / 89.6 / 77.4 / 77.0 / 54.6 / 28.0 | 69.7.
Δ so WP2 B36 (quy tắc, B30): XBG Swin-L **−9.0 [−12.6, −4.3]**, CA3 −5.6, Swin-T −7.5; XBGS Swin-L **−9.6 [−13.8, −2.2]**, CA3 −11.7, YOLOX −18.4. Cùng số bước (B36): XBG Swin-L −11.1, XBGS −9.4. Runtime thực: XBG / XBGS B36 4.9 / 5.1 s / ảnh, B30 4.2 = WP2 B36 4.2 → cùng thời gian đúng như hiệu chỉnh.
**Đọc:** kéo feature R50 trong box về F_R50(x_bg) làm δ YẾU ĐI trên mọi target (Swin-L −9 với CI xa 0, kể cả Swin-T / ConvNeXt), XBGS (bỏ L_OSFD trong box) hại nặng nhất ở CNN → đẩy OSFD trong box là phần có ích; đích "ảnh xóa object theo R50" ngược lại kéo δ về một mẫu feature cụ thể của R50 (đặc thù surrogate, cùng kiểu thất bại BG-OCT 10-10) và về nội dung nhòe (cùng chiều TX: δ lệch khỏi nội dung thật → yếu khắp nơi). Đóng hướng (1a) "đích F_R50(x_bg)"; không chạy biến thể stage 3 / inpaint tốt hơn (quy tắc: chỉ khi CÓ TÍN HIỆU). Tỉ lệ xóa theo cỡ (mô tả đã khai báo) KHÔNG tính — kết luận không phụ thuộc. Còn mở: (1b) thành phần universal (cổng UNI-alone), họ WP (2), user quyết.

---

## 2026-10-23 — E18 / Gate A của RSE ("xóa quan hệ object–nền", đề xuất RIME-WP của user): cùng một δ, hiệu ứng trên R50 có ĐỒNG BIẾN với trên Swin-L không? (user duyệt "Approve") — MÔ TẢ, forward-only, khai báo TRƯỚC khi chạy
**Lý do (user chốt sau phản biện):** bất biến channel của MMD không tạo transfer — loss RSE vẫn tính và lấy gradient trên R50. Điều kiện cần để tối ưu RSE phía R50 có ích: mức "object giống nền sạch hơn" do một δ gây ra trên R50 phải đi cùng mức đó trên Swin-L, và mức trên Swin-L phải đi cùng việc DINO mất object. Gate quyết định có viết attack RSE (Phase 1) hay không.
**Đo (`scripts/e18_rse_gate.py`, dev100, PNG WP2 B36 dev100 — 70 ảnh mới sinh hôm nay, `results/rse/attack_wp2_dev100.log`; XBG / XBGS B30 chỉ dev30; NOISE ±5 dấu ngẫu nhiên toàn ảnh, seed 2026 + img_id, mốc null):** R50 C4 và backbone Swin-L stage 3 của DINO (cả hai stride 16, cùng ảnh vào). Mỗi object GT không-crowd: O = token có tâm trong box, B = token có tâm trong box nở 1.5× quanh tâm và KHÔNG thuộc box GT nào; cần ≥ 4 token mỗi tập (object quá nhỏ ở stride 16 bị loại — giới hạn khai báo), ≤ 64 token mỗi tập (seed 2026 + ann_id, cùng chỉ số mọi điều kiện). z = LN_kênh(F) / ‖·‖; k = exp((cos − 1)/τ), τ = trung vị (1 − cos) cặp token sạch O ∪ B của object (cố định, dùng chung mọi điều kiện); MMD² dạng V. r = MMD²(z_O^adv, z_B^sạch) / MMD²(z_O^sạch, z_B^sạch), **Δr = 1 − r**. DINO xóa = score khớp sạch ≥ 0.3 và adv < 0.3 (như E14). Bootstrap theo ảnh 1000, BOOT_SEED 2026; Spearman trên object gộp.
**Quy tắc (trên δ WP2, dev100):**
- **QUA** nếu S1 = Spearman(Δr_R50, Δr_SwinL) > 0 với cận dưới CI > 0 **VÀ** S2 = Spearman(Δr_SwinL, DINO xóa) > 0 với cận dưới CI > 0 (S2 chỉ trên object DINO thấy khi sạch) → Phase 1 RSE @ dev30 (khóa entry riêng; quy tắc +10 / +3 như TX / LN).
- **KHÔNG QUA** nếu một trong hai không đạt → dừng RSE, không viết attack.
- Mô tả: S3 = Spearman(Δr_R50, DINO xóa); Δr trung bình mỗi model / điều kiện (NOISE là mốc null); S1, S2 theo cỡ (nhỏ / vừa / lớn COCO) — kiểm nhiễu do cỡ object; 70 ảnh mới; S1–S3 trên XBG / XBGS / NOISE.
**Caveat khai báo trước:** tương quan giữa object có thể do thuộc tính object chung (cỡ, độ tương phản nền) chứ không do δ → báo theo cỡ; đạt gate là điều kiện cần, không đủ (tối ưu Δr_R50 có thể đi theo hướng khác hướng mà δ WP2 tình cờ có). Smoke test 3 ảnh (chỉ kiểm code, không đọc số).

### Kết quả E18 / Gate A @ dev100 (`results/mechanism/e18_rse_gate.json`, log `results/rse/e18.log`, raw `artifacts/e18/raw.json`; 137 s forward) — theo quy tắc: **KHÔNG QUA** (S1 đạt, S2 không) → dừng RSE, không viết attack
639 object có đủ token O / B ở cả 2 model (537 DINO thấy khi sạch). Δr trung bình R50 / Swin-L | DINO xóa:
- NOISE: −0.010 / −0.027 | 0.011; **WP2: −0.067 / −0.074 | 0.244**; XBG (dev30): −0.071 / −0.052 | 0.215; XBGS: −0.056 / −0.045 | 0.205.
- **S1 (Δr R50 ~ Δr Swin-L), WP2: +0.335 [+0.242, +0.425]** ✓ (NOISE +0.118 [0.03, 0.21]; giữ ở mọi cỡ: nhỏ +0.34, vừa +0.29, lớn +0.38; 70 ảnh mới +0.33).
- **S2 (Δr Swin-L ~ DINO xóa), WP2: −0.050 [−0.156, +0.058]** ✗ (nhỏ −0.13, vừa −0.09, lớn +0.00; 70 ảnh mới −0.02; XBG / XBGS cũng ≈ 0).
- S3 (Δr R50 ~ DINO xóa), WP2: −0.101 [−0.194, −0.001].
**Đọc:** (1) hiệu ứng của cùng δ lên quan hệ object–nền có đồng biến giữa R50 và Swin-L (S1 ≈ 3× mốc nhiễu, không do cỡ) — phần "coupling" mà đề xuất cần là có thật; (2) nhưng đại lượng đó KHÔNG liên quan tới việc DINO mất object (S2 ≈ 0 ở mọi cỡ, cả trên ảnh mới) → "object giống nền cục bộ hơn ở stage 3" không phải kênh mà δ transfer dùng để xóa object trên DINO-Swin-L; (3) Δr trung bình ÂM ở mọi δ đối kháng (object xa nền sạch HƠN, không gần hơn) dù R50 white-box xóa 92% → δ hiện tại không phá theo kiểu "hòa vào nền" ở cả 2 model; khớp E14: transfer chủ yếu hạ confidence phân loại. Theo quy tắc: dừng RSE. Giới hạn: tương quan trên δ có sẵn — không loại trừ tuyệt đối việc tối ưu thẳng Δr tạo kênh mới (WB E14c kéo token về μ nền CÓ xóa object), nhưng không có bằng chứng ủng hộ để tốn Phase 1.

---

## 2026-10-23 — WPA: nhiễu trọng số R50 TÍCH LŨY, chiếu về bán kính ρ = 0.002: C2-OCT-1V-WPA @ dev30 B36 (user chọn "WPA trước, rồi mới dev300; không UNI") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lập luận (user):** họ WP là họ DUY NHẤT tới giờ tăng DINO-Swin-L với CI dương @ dev100; WP2 đặt lại θ' = θ0 − ρ·‖θ0‖·ĝ mỗi bước (nhớ 1 bước). Trục chưa thử: tích lũy / đa dạng hóa đường đi trong lân cận trọng số. Giữ threat model (1 R50, không dữ liệu phụ, không đổi protocol).
**Ô `C2-OCT-1V-WPA` (`core.wp_update`, `geo_osfd_attack(weight_eta)`):** WP2 y nguyên trừ cập nhật θ' mỗi tensor conv backbone: Δ ← Δ − η·‖θ0‖·ĝ, rồi chiếu ‖Δ‖ ≤ ρ·‖θ0‖ (Δ = θ' − θ0; ĝ = ∇θ loss attack tại θ' hiện tại, cùng lần autograd như WP2). **ρ = 0.002 (= WP2, bán kính tổng KHÔNG tăng theo số bước — không thành WP5), η = ρ / 4 chọn trước** (trí nhớ ~4 bước: bước mới chỉ đóng góp ¼ bán kính). Một biến thể duy nhất, KHÔNG quét η / ρ trên DINO.
**Kiểm:** `scripts/wpa_unit_test.py` — η = None trùng tuyệt đối WP2; 200 bước gradient ngẫu nhiên ‖Δ‖ ≤ bán kính; hướng cố định: ‖Δ‖ / bán kính 0.25 → 0.5 → 0.75 → 1 rồi trùng điểm WP2; sau khi gradient đổi hướng cos(−Δ, g_cũ) WPA 0.97 vs WP2 0.00. Sức khỏe white-box (ảnh 1–4 dev300, B36, chỉ R50): WP2 79% / 92%, **WPA 80% / 92%**. Thời gian: WP2 4.14, WPA 4.22 s / ảnh (1.02×) → **B36 = cùng thời gian**.
**Chạy (V100, dev30; PNG WP2 B36 dev30 có sẵn từ hôm nay; `results/wpa/run.sh`, log `results/wpa/run.log`).**
**Đọc (khai báo trước; so WP2 B36 cùng 30 ảnh; như TX / LN / XBG):**
- **BƯỚC NHẢY** nếu ΔSwin-L ≥ +10 → lên dev100; mới cân nhắc UNI như thành phần bổ sung sau đó.
- **CÓ TÍN HIỆU** nếu +3 ≤ ΔSwin-L < +10 VÀ ΔCA3 ≥ −3 → chỉ lên dev100 (khóa entry), CHƯA gọi là method mới.
- **KHÔNG** nếu ΔSwin-L < +3 → đóng họ WP động, bỏ UNI, đưa WP2 lên sàng lọc dev300 (khóa entry riêng).
- Mô tả: Swin-T, R101, CNN; ‖Δ‖ / bán kính không log (unit test cho thấy bám biên khi gradient ngẫu nhiên).
**Caveat:** dev30 đã dùng nhiều lần (không chọn tham số ở đây); GPU không tất định → đọc theo CI; n = 30.

### Kết quả WPA @ dev30 B36 (V100; `results/ablation/wpa_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/{metrics,generalization}_wpa_dev30.json`, log `results/wpa/run.log`) — theo khai báo: **CÓ TÍN HIỆU** (+3 ≤ ΔSwin-L < +10, ΔCA3 ≥ −3)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: WP2 98.5 / 92.1 / 83.2 / 83.2 / 75.2 / 37.5 | 80.5; **WPA** 98.6 / 92.0 / 85.0 / 84.1 / 78.6 / **41.8** | 82.6.
Δ WPA − WP2: Swin-L **+4.3 [−1.5, +8.8]**, CA3 +2.0 [+0.5, +4.2], YOLOX +3.5, ConvNeXt +1.8, Swin-T +0.9, R101 −0.1. Runtime attack thực 4.7 s / ảnh (WP2 tái dùng PNG → không đo cùng lần; hiệu chỉnh 1.02×) → đo lại cùng lần ở dev100.
**Đọc:** tích lũy θ' trong cùng bán kính ρ cho thêm lợi cùng chiều cả Swin-L lẫn CNN, CI Swin-L còn chứa 0 (n = 30). Theo khai báo: lên dev100, CHƯA gọi là method mới.

---

## 2026-10-23 — WPA @ dev100 B36 so WP2 B36 (theo khai báo "CÓ TÍN HIỆU" ở trên) — khóa TRƯỚC khi chạy
**Chạy (V100, dev100; PNG WP2 B36 dev100 có sẵn từ E18 hôm nay; WPA sinh 70 ảnh mới (30 ảnh dev30 tái dùng); `results/wpa/run_dev100.sh`, log `results/wpa/run_dev100.log`; metrics `metrics_wpa_dev100.json`, dets `dets_wpa_dev100.json`).** **Runtime đo cùng lần:** attack lại WP2 + WPA nối tiếp trên dev10 vào run riêng `dev300_B36_eps5_limit12` (chỉ để đo thời gian, không eval), so s / ảnh trung bình ảnh 2–10 (bỏ ảnh khởi động).
**Quy tắc (dev100, paired bootstrap 1000, BOOT_SEED 2026; CA3 = TB ConvNeXt-T / Swin-T / YOLOX-S; Δ = drop(WPA B36) − drop(WP2 B36), cùng ảnh) — như WP2 @ dev100 (10-22):**
- **GIÚP** nếu [(ΔSwin-L ≥ 3 VÀ cận dưới CI > 0 VÀ ΔCA3 ≥ −1) HOẶC (ΔCA3 ≥ 2 VÀ cận dưới CI > 0 VÀ ΔSwin-L ≥ −1)] VÀ ΔR101 ≥ −3 VÀ runtime WPA ≤ 1.1 × WP2 (đo cùng lần).
- **HẠI** nếu ΔCA3 ≤ −3 HOẶC ΔSwin-L ≤ −3. **TRÊN ĐƯỜNG** nếu một phía ≥ 2 và phía kia ≤ −1. Còn lại → **NGANG**.
- Mô tả: **70 ảnh mới (dev100 \ dev30)**; từng target.
- **GIÚP** → WPA thay WP2 làm ứng viên, sàng lọc dev300 so WP2 + C2-OCT-1V + OSFD cùng thời gian (+ đối chứng AWT-OD cho claim novelty), entry riêng. **Khác** → WP2 lên sàng lọc dev300 (như khai báo WPA dev30), đóng họ WP động.
**Caveat:** 30 / 100 ảnh là dev30 (đã thấy) → đọc 70 ảnh mới; η không tune.

### Kết quả WPA @ dev100 B36 (V100; `results/ablation/wpa_dev100/screen.json`, `results/runs/dev300_B36_eps5_limit100/{metrics,generalization}_wpa_dev100.json`, log `results/wpa/run_dev100.log`) — theo quy tắc: **NGANG**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: WP2 96.0 / 92.2 / 83.9 / 80.4 / 73.9 / 36.8 | 79.4; **WPA** 96.1 / 91.4 / 84.7 / 80.1 / 73.4 / **39.6** | 79.4.
Δ WPA − WP2: Swin-L **+2.8 [−0.1, +4.7]** (< 3, cận dưới < 0), CA3 +0.0 [−1.6, 1.3], R101 −0.8, YOLOX −0.4, ConvNeXt +0.8, Swin-T −0.3 → không nhánh GIÚP nào; không HẠI / TRÊN ĐƯỜNG → NGANG. Runtime cùng lần (dev10, ảnh 2–10, `results/runs/dev300_B36_eps5_limit12/attack_stats.jsonl`): WP2 3.67, WPA 3.70 s / ảnh (1.01×).
**70 ảnh mới:** Swin-L +1.0 [−1.4, 3.2], CA3 −0.2 → lợi dev30 (+4.3) phần lớn KHÔNG tái lập. WP2 dev100 tái lập 10-22 (Swin-L 36.8 vs 36.4).
**Đọc:** tích lũy θ' trong cùng bán kính không thêm gì đáng kể so với WP2; họ WP bão hòa ở ~37–40% Swin-L. Theo khai báo: đóng họ WP động, bỏ UNI, đưa **WP2** lên sàng lọc dev300 (entry riêng, khóa trước).

---

## 2026-10-23 — E19 "oracle-δ dissection": trên CHÍNH R50, cách ước lượng hướng nào (chỉ dùng R50) trỏ về δ_SwinT? (bước 0, rà literature 10-2026; user duyệt "approve") — MÔ TẢ, chỉ đo trên R50, khai báo TRƯỚC khi đo
**Vì sao:** E16 — δ_SwinT (C2-OCT-1V, surrogate Swin-T, B41) phá R50 78.7% VÀ DINO-Swin-L 58.0%, gần trực giao δ_R50 (cos 0.06) → tồn tại δ trong quả cầu ε = 5 mà R50 "thấy", nhưng gradient ascent trên R50 không đi tới. δ_SwinT chỉ là **oracle để phân tích**; method cuối vẫn chỉ dùng R50 (như chẩn đoán đổi surrogate 10-21). Literature (10-2026): CNN → ViT ít người làm, detection / DINO gần như trống; ước lượng hướng kiểu IG (MIG, ICCV'23) và độ phẳng phía ĐẦU VÀO (PGN / CWA) repo chưa thử (đổi hướng gradient theo kiểu BPA / GLB / PD đã thử 10-21: NGANG / HẠI).
**Đã chạy trước entry này (chỉ dựng lại oracle, không đo):** δ_SwinT dev100 sinh lại (PNG E16 mất khi đổi máy): `run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --surrogate swin_t --stage attack --methods C2-OCT-1V` (V100, 5.4 s / ảnh, log `results/e19/attack_swint.log`). **Sức khỏe H0 QUA:** L2 từng ảnh so lần chạy cũ (`attack_stats.jsonl` cùng run) lệch trung vị 0.1%, tối đa 0.5%.
**Đo (script `scripts/e19_oracle_dissect.py`, dev100, R50 cố định fp32, không AMP; δ tính ở không gian ảnh GỐC như biến `noise` của attack):**
- δ_S = PNG δ_SwinT − ảnh gốc; δ_W = PNG **C2-OCT-1V-WP2 B36** dev100 (method tốt nhất, có sẵn) − ảnh gốc (δ_R50 C2-OCT-1V B41 không còn trên máy → δ_W thay làm mốc R50).
- Loss L = loss C2: L_OSFD (k = 3, feature sạch co-transform) + 4·(loss_rpn_cls + loss_cls), skip_mask, như C2-OCT-1V. **8 view OCT cố định mỗi ảnh** (elem {scale ln2/2, shift ph8}, nhiễu σ = 6, GT co-transform; RNG seed = SEED + seed_index như attack), dùng CHUNG cho mọi ước lượng.
- **Độ căn hướng** A_S(h) = cos(sign h, δ_S), A_W(h) = cos(sign h, δ_W) (phẳng 3×H×W, theo ảnh; cos(h, ·) thô ghi mô tả). Điểm đo **P0 = ảnh sạch (δ = 0)** — điểm xuất phát chung của mọi attack. Ước lượng h, **cùng ngân sách 8 backward**:
  - **G8 (đối chứng chính):** Σ_v g_v / mean|g_v| trên 8 view (gộp như MI). G1 = view 1 (mô tả).
  - **WP4:** 4 view; mỗi view: ∇θ → θ' = cập nhật WP2 (ρ = 0.002, `core.wp_update`) → ∇x dưới θ' (feature sạch cũng trên θ').
  - **IG-0 / IG-blur:** h = (x − b) ⊙ Σ_{k=1..8} ĝ(b + (k/8)(x − b)), điểm k dùng view k; b = ảnh đen | Gauss σ = 8 px của ảnh gốc.
  - **PGN4 (kiểu PGN, không phải bản gốc):** 4 mẫu x' = x + U(−3ε, 3ε) (kẹp [0, 255]), g' = ∇ tại x' (view i), x* = x' − ε·sign(g'), g* = ∇ tại x* → h = Σ (0.5 ĝ' + 0.5 ĝ*).
  - **Thành phần (phân rã CÙNG 8 view của G8, không phải ngân sách riêng):** chỉ L_OSFD; chỉ 4·loss_rpn_cls; chỉ 4·loss_cls RoI; L_OSFD riêng stage C2 / C3 / C4 / C5.
  - **Mốc null:** δ_W (A_S(δ_W) = cos(δ_W, δ_S)); RAND = Rademacher (seed 2026 + img_id); CONTENT = x − Gauss_σ2(x) (tương quan với cấu trúc ảnh — chặn IG thắng chỉ nhờ nhân với x).
- **Độ phẳng:** L̄(δ) = TB 8 view của L tại x + δ (không grad); Γ(δ) = L̄(δ) − L̄(0). Mốc R50 **cùng mức gain:** t*·δ_W, t* ∈ {0.1, …, 1.0} theo ảnh sao cho Γ gần Γ(δ_S) nhất. Giữ lại r(δ) = [TB_j L̄(δ + ξ_j) − L̄(0)] / Γ(δ), ξ_j Rademacher ±2 px, j = 1..4, **không kẹp về ε** (đo địa hình, không phải attack; kẹp sẽ thiên lệch giữa δ dạng sign và δ co giãn). Bỏ ảnh có Γ(δ_S) ≤ 0.05·Γ(δ_W) (ghi số ảnh). Mô tả: độ phân tán gain giữa 8 view (std / mean).
- **Hồ sơ loss:** p_c = ΔL_c(δ_S) / ΔL_c(δ_W) (tỉ số của TB qua ảnh, c ∈ {rpn, roi, OSFD C2..C5}; δ_W đầy đủ), p_tot = Γ(δ_S) / Γ(δ_W). Mô tả: ‖F(x + δ) − F(x)‖ / ‖F(x)‖ từng stage, không view.
- **Mô tả thêm (không thuộc quy tắc):** cùng các ước lượng tại **P1 = x + δ_W** trên dev30; **H1:** relative AP drop R50 / DINO-Swin-L của δ_S mới (inference_detector trên PNG như E16), kỳ vọng ≈ 78.7 / 58.0 (± 3; lệch hơn → ghi, vẫn đọc).
**Quy tắc (dev100, paired bootstrap 1000 theo ảnh, BOOT_SEED 2026; Δ = A_S(h) − A_S(G8) cùng ảnh):**
- **Sức khỏe:** |A_S(RAND)| < 0.005 VÀ A_S(G8) − A_S(RAND) cận dưới CI > 0; kiểm 2 ảnh: G8 khớp tổ hợp tuyến tính của gradient thành phần (cos ≥ 0.999), A_S(δ_S) = 1. Không qua → **KHÔNG ĐO ĐƯỢC**, sửa / user quyết, không đọc các dòng dưới.
- **TÁCH ĐƯỢC** (cho h là ước lượng hoặc thành phần): A_S(h) ≥ 1.5·A_S(G8) VÀ Δ ≥ +0.01 VÀ cận dưới CI Δ > 0 VÀ cận dưới CI [A_S(h) − A_S(CONTENT)] > 0.
- **PHẲNG HƠN:** r(δ_S) − r(t*·δ_W) ≥ +0.10 VÀ cận dưới CI > 0.
- **KHÁC HỒ SƠ:** ∃ c: p_c ≥ 0.8 (cận dưới CI ≥ 0.6) VÀ p_tot ≤ 0.5.
- **Ánh xạ sang method (theo thứ tự ưu tiên, khóa):**
  1. Một ƯỚC LƯỢNG (WP4 / IG-0 / IG-blur / PGN4) TÁCH ĐƯỢC → thay gradient mỗi bước của C2-OCT-1V-WP2 bằng ước lượng có Δ lớn nhất, quick test dev30 **cùng thời gian WP2 B36** (B giảm theo chi phí đo), entry riêng khóa trước.
  2. Không (1), PHẲNG HƠN → "dual-flat": WP2 + phẳng phía đầu vào kiểu PGN4, dev30 cùng thời gian, entry riêng.
  3. Không (1)–(2), một THÀNH PHẦN TÁCH ĐƯỢC hoặc KHÁC HỒ SƠ → đặt lại trọng số loss theo thành phần đó (novelty thấp), dev30.
  4. Không gì → **DỪNG**: không ước lượng nào chỉ dùng R50 trỏ được về vùng δ_SwinT; ghi kết luận âm (cho paper), user quyết.
**Caveat (đọc kết quả với):** (a) Swin-T → DINO-Swin-L là transfer **cùng họ Swin**; δ_S phá R50 78.7% là chiều ViT → CNN (chiều dễ theo literature) → "R50 thấy δ_S" KHÔNG hàm ý gradient R50 tìm tới được; kỳ vọng trước: dòng (4) khá có thể. (b) P0 là gradient 1 điểm so với điểm cuối 41 bước MI + OCT của δ_S → A_S tuyệt đối sẽ nhỏ; quyết định dựa trên so tương đối với G8 cùng ngân sách. (c) 30 / 100 ảnh là dev30 (đã thấy). (d) Tham số ước lượng (T = 8, σ_blur = 8, ζ = 3, ±2 px) đặt trước, không tune.
**Chi phí ước tính (V100):** ~88 backward + ~130 forward / ảnh ≈ 20 s → dev100 ~35 phút; P1 dev30 ~10 phút; H1 ~5 phút. Output `results/mechanism/e19_oracle_dissect.json`, thô `artifacts/e19/`, log `results/e19/run.log`.

### E19 — sửa khai báo (user, 22:22 GMT+7, TRƯỚC khi xem bất kỳ kết quả nào; đo đang chạy 9/100 ảnh): đọc dev30 trước
- **Đọc sơ bộ trên dev30** (= 30 ảnh đầu theo thứ tự đo, `--analyze-only --limit 30`, bootstrap trên 30 ảnh): áp NGUYÊN quy tắc sức khỏe / TÁCH ĐƯỢC / PHẲNG HƠN / KHÁC HỒ SƠ / ánh xạ ở trên, không đổi ngưỡng. Báo user; user quyết bước tiếp theo từ kết quả dev30.
- Tiến trình đo vẫn chạy tiếp tới 100 ảnh (không tốn thêm công; dừng nếu user muốn). Nếu dùng dev100: quy tắc như khai báo gốc; dev100 có thể lật kết luận dev30, khi đó **dev100 thắng**.
- Caveat thêm: CI rộng hơn ~1.8× so dev100 → dễ ra "không tách được" dù có tín hiệu nhỏ; 30 ảnh đều là dev30 (đã thấy trong mọi pilot trước).

### Kết quả E19 @ dev30 (sơ bộ, 22:30 GMT+7; `results/mechanism/e19_oracle_dissect_dev30.json`, thô `artifacts/e19/raw_snapshot30.json`, log `results/e19/run.log`) — theo quy tắc: **dòng (3)** (KHÁC HỒ SƠ ở stage C2), không ước lượng nào TÁCH ĐƯỢC
- **Sức khỏe QUA:** A_S(RAND) −0.0003; G8 − RAND +0.044 [0.040, 0.048]; kiểm 2 ảnh QUA.
- **Độ căn hướng tại P0 (A_S | A_W):** G8 0.044 | 0.200; G1 0.022 | 0.109; δ_W 0.061. Δ so G8: **IG-0 +0.0007 [−0.0008, 0.0021]** (×1.01), PGN4 −0.010, WP4 −0.012, IG-blur −0.047 (≈ 0); mọi thành phần riêng < G8 (L_OSFD ×0.83, rpn ×0.21, roi ×0.45, stage C2..C5 ×0.70 / 0.76 / 0.73 / 0.47) → **không ước lượng / thành phần nào TÁCH ĐƯỢC**. CONTENT −0.058 (cả δ_S lẫn δ_W ngược hướng high-pass của ảnh).
- **Độ phẳng:** r(δ_S) 0.993 vs r(t*·δ_W) 0.977 (t* trung vị 0.5), Δ +0.016 [0.010, 0.023] < 0.10 → KHÔNG PHẲNG HƠN.
- **Hồ sơ loss p_c = ΔL_c(δ_S) / ΔL_c(δ_W):** OSFD-C2 **0.83**, C3 0.61, C4 0.44, C5 0.41, rpn 0.29, roi 0.47, tổng 0.43 → KHÁC HỒ SƠ (C2) theo quy tắc.
- **P1 = x + δ_W (mô tả):** G8 0.030 | 0.243; L_OSFD-C2 / C3 0.048 / 0.049 (≈ 1.6× G8) nhưng A_W cũng tăng (0.38 / 0.43).
- **Đọc / caveat (chưa khóa, cần đối chứng trước khi theo dòng 3):** p_C2 cao có thể chỉ vì feature stage nông phản ứng gần như với MỌI nhiễu ±5 (gần tuyến tính) — hồ sơ chưa có δ ngẫu nhiên làm mốc. Gradient của L_OSFD-C2 nhỏ hơn ~10³× các thành phần khác (kiểm 2 ảnh) → C2 gần như không tham gia hướng attack hiện tại. Kết luận chính: tại ảnh sạch, gradient R50 căn với điểm cuối của chính nó ~4.5× hơn với δ_SwinT, và không ước lượng chỉ-R50 nào đã thử (IG, PGN, WP, tách thành phần) đổi được tỉ lệ đó.

### E19 — đối chứng hồ sơ loss bằng nhiễu ngẫu nhiên @ dev30 (user duyệt "Duyệt", 22:33 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** dòng (3) dev30 dựa trên p_C2 = 0.83; stage nông có thể phản ứng như nhau với mọi nhiễu ±5. **Đo (forward, dev30, cùng 8 view / seed như E19):** p_c(X) = ΔL_c(X) / ΔL_c(δ_W) cho X ∈ {δ_S, RAND} (RAND = Rademacher ±5, seed 2026 + img_id như mốc null E19), tính lại L(0), L(δ_W) cùng lần; paired bootstrap 1000 (BOOT_SEED 2026) trên d = p_C2(δ_S) − p_C2(RAND).
**Quy tắc:** **C2 THẬT** nếu d ≥ 0.3 VÀ cận dưới CI > 0 → giữ dòng (3) (ứng viên: tăng trọng số L_OSFD-C2), entry riêng. Ngược lại **C2 ẢO** → E19 dev30 đọc là dòng (4) (DỪNG). Mô tả: p_c(RAND) mọi thành phần.

### Kết quả đối chứng nhiễu ngẫu nhiên @ dev30 (`results/mechanism/e19_rand_profile_dev30.json`, log `results/e19/rand_profile.log`) — theo quy tắc: **C2 THẬT (sát ngưỡng)** → giữ dòng (3)
p_c δ_S | RAND: C2 0.83 | 0.50, C3 0.61 | 0.23, C4 0.44 | 0.07, C5 0.41 | 0.04, rpn 0.29 | 0.01, roi 0.47 | 0.02, tổng 0.43 | 0.05 (p_C2(δ_S) khớp lần đo chính). d = p_C2(δ_S) − p_C2(RAND) **+0.326 [0.292, 0.361]** ≥ 0.3, cận dưới > 0.
**Đọc (mô tả):** ~60% của p_C2 là phản ứng chung với mọi nhiễu ±5; phần VƯỢT nhiễu ngẫu nhiên của δ_S gần ĐỀU ở mọi thành phần (+0.29 … +0.46: C3 +0.38, C4 +0.36, C5 +0.37, roi +0.46) → δ_S KHÔNG dồn riêng vào C2; C2 chỉ là thành phần duy nhất vượt mốc tuyệt đối 0.8. Bằng chứng cho "tăng trọng số L_OSFD-C2" yếu.

### Kết quả E19 @ dev100 (`results/mechanism/e19_oracle_dissect.json`, thô `artifacts/e19/raw.json`, log `results/e19/run.log`; V100 ~18 s / ảnh) — theo quy tắc: **dòng (3)** (KHÁC HỒ SƠ ở C2), TÁI LẬP dev30
- Sức khỏe QUA (RAND 0.0000; G8 − RAND +0.047 [0.045, 0.050]).
- A_S | A_W: G8 0.047 | 0.216; δ_W 0.065. Δ so G8: **IG-0 +0.0017 [+0.0007, +0.0026] (×1.04)** — cận dưới > 0 nhưng < +0.01 và < 1.5× → không TÁCH; PGN4 −0.011, WP4 −0.013, IG-blur −0.050; thành phần L_OSFD ×0.84, rpn ×0.20, roi ×0.41, C2..C5 ×0.69 / 0.75 / 0.74 / 0.48.
- Phẳng: Δr +0.017 [0.013, 0.020] < 0.10 → KHÔNG. Hồ sơ: C2 0.82, C3 0.60, C4 0.44, C5 0.42, rpn 0.30, roi 0.44, tổng 0.43 → KHÁC HỒ SƠ (C2).
- **Đọc:** dev100 khớp dev30 ở mọi con số (lệch ≤ 0.005 ở A_S, ≤ 0.04 ở p_c). Đối chứng nhiễu ngẫu nhiên (dev30) cho C2 THẬT sát ngưỡng nhưng phần vượt nhiễu đều ở mọi thành phần; gradient riêng L_OSFD-C2 căn với δ_S KÉM G8 (×0.69). Kết luận chính: không ước lượng hướng chỉ-R50 nào đã thử (IG / MIG, PGN, WP, tách thành phần) trỏ về δ_SwinT tốt hơn gradient thường. Dòng (3) theo quy tắc → user quyết có quick test "WP2 + tăng trọng số L_OSFD-C2" hay không (kỳ vọng Swin-L 0 … +3).
- **H1 (mô tả, `results/mechanism/e19_h1.json`):** δ_S sinh lại: drop R50 **77.8** [71.4, 82.0], DINO-Swin-L **59.1** [50.6, 62.0] (E16: 78.7 / 58.0) → oracle tái lập trong ± 1.1.

---

## 2026-10-23 — TẠM DỪNG (user, 22:55 GMT+7): mai chạy tiếp
**Máy:** V100 16GB, trả máy. Không tiến trình nào đang chạy. Commit đã push (user, 23:00); PNG đã upload HF (user duyệt).
**Phiên này:** E19 oracle-δ dissection xong (dev30 sơ bộ + dev100 + đối chứng nhiễu ngẫu nhiên + H1). Kết quả theo quy tắc: **dòng (3)** (KHÁC HỒ SƠ ở L_OSFD-C2, sát ngưỡng), thực chất gần **dòng (4)**: không ước lượng hướng chỉ-R50 nào (IG / MIG ×1.04, PGN4, WP4, tách thành phần) trỏ về δ_SwinT tốt hơn G8; δ_SwinT không phẳng hơn trên R50. Số liệu thô dev100 đã chép vào `results/e19/raw_dev100.json` (git).
**Việc mở — user quyết khi bắt đầu phiên mai:**
- (A) Quick test dev30 "C2-OCT-1V-WP2 + tăng trọng số L_OSFD-C2" để đóng dòng (3) (kỳ vọng Swin-L 0 … +3; gradient C2 căn δ_S KÉM G8 ×0.69). Khóa quy tắc trước khi chạy.
- (B) (Claude đề xuất) Bỏ dòng (3), ghi E19 như kết luận âm cho paper; quay lại kế hoạch: **WP2 lên sàng lọc dev300** (entry riêng, khóa trước).
**Artifact đã upload HF (23:05 GMT+7, tag `_v100`):** `dev300_B36_eps5_limit100` (WP2 / WPA / XBG / XBGS B36 PNG dev100), `dev300_B41_eps5_limit100_sur-swin_t` (δ_SwinT dev100), `dev300_B30_eps5_limit100` (XBG / XBGS B30), `dev300_B36_eps5_limit{12,5}` (đo thời gian). Máy mới: `python scripts/sync_artifacts.py download <run> --tag _v100` (HF_TOKEN trong `.env`). Chưa upload: `artifacts/e18/` (17 MB, sinh lại được; số liệu ở `results/`).

---

## 2026-10-24 — Máy mới V100 16GB; việc mở (A) entry 10-23: C2-OCT-1V-WP2 + tăng trọng số L_OSFD-C2 (C2W03 / C2W1) @ dev30 B36 (user chọn "A") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Máy:** Tesla V100-SXM2 16GB, bootstrap lại; KHÔNG có `.env` → không có PNG cũ, mốc WP2 B36 dev30 sinh lại cùng máy.
**Lập luận:** E19 dòng (3): p_C2 = ΔL_OSFD-C2(δ_SwinT) / ΔL_OSFD-C2(δ_W) ≈ 0.82 (C2 THẬT sát ngưỡng sau đối chứng nhiễu ngẫu nhiên), trong khi gradient của L_OSFD-C2 nhỏ hơn ~10³× các thành phần khác → C2 gần như không tham gia hướng attack hiện tại. Quick test để đóng dòng (3). Kỳ vọng trước: Swin-L 0 … +3 (gradient riêng C2 căn δ_S KÉM G8, ×0.69).
**Ô (`make_view_loss(stage_w)`, `geo_osfd_attack(stage_w)`):** gốc = WP2 (C2-OCT-1V, ρ = 0.002, L_cls λ = 4), L_OSFD = w_C2·MSE_C2 + MSE_C3 + MSE_C4 + MSE_C5. **w_C2 = a·R**, R = trung vị mean|∇ₓ L_rest| / mean|∇ₓ L_OSFD-C2| tại δ = 0, ảnh không biến đổi, θ0, 5 ảnh đầu dev100 (`scripts/c2w_calib.py`, cùng cách chọn λ = 4 / thang MRG; L_rest = Σ_{C3..C5} MSE + 4·cls). **a = 0.3 (`C2W03`) | 1 (`C2W1`, gradient C2 ngang phần còn lại)**, chọn trước, không quét thêm. Chi phí tính không đổi (cùng forward / backward) → **B36 = cùng thời gian WP2 B36** (ghi runtime thực để kiểm).
**Kiểm:** stage_w = (1, 1, 1, 1) trùng loss gốc (sai số số học); sức khỏe white-box (`scripts/tx_calib.py`, ảnh 1–4 dev300, B36, chỉ R50): |δ| = 5 VÀ R50 xóa ≥ 80% mức WP2, không đạt → ghi lại, ô đó không chạy.
**Chạy (V100, dev30; `results/c2w/run.sh`, log `results/c2w/run.log`; metrics `metrics_c2w_dev30.json`, dets `dets_c2w_dev30.json`):** WP2 B36 (mốc, sinh lại), C2W03 / C2W1 B36.
**Đọc (khai báo trước; so WP2 B36 cùng 30 ảnh; như XBG / WPA), mỗi ô:**
- **BƯỚC NHẢY** nếu ΔSwin-L ≥ +10 → lên dev100 ngay (khóa entry).
- **CÓ TÍN HIỆU** nếu +3 ≤ ΔSwin-L < +10 VÀ ΔCA3 ≥ −3 → chỉ lên dev100 (khóa entry), CHƯA gọi là method mới.
- **KHÔNG** nếu cả 2 ô ΔSwin-L < +3 (hoặc ΔCA3 < −3) → đóng dòng (3) của E19; E19 ghi là kết luận âm cho paper; WP2 lên sàng lọc dev300 (entry riêng, khóa trước).
- Mô tả: Swin-T, R101, ConvNeXt, YOLOX; cos(∇C2, ∇rest) trong hiệu chỉnh.
**Caveat:** dev30 đã dùng nhiều lần (không chọn tham số ở đây); GPU không tất định → đọc theo CI; n = 30.

### Kết quả C2W @ dev30 B36 (V100; `results/ablation/c2w_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/{metrics,generalization}_c2w_dev30.json`, log `results/c2w/run.log`) — theo khai báo: **KHÔNG** (cả 2 ô ΔSwin-L < +3)
**Hiệu chỉnh:** R = 476.3 (5 ảnh: 578.6 / 519.6 / 476.3 / 431.9 / 472.8; cos(∇C2, ∇rest) +0.01 … +0.07 — gần trực giao) → w_C2 = 142.9 (C2W03) | 476.3 (C2W1). **Sức khỏe QUA:** |δ| = 5 / R50 xóa — WP2 80% / 92%, C2W03 82% / 92%, C2W1 85% / 89%. Runtime 4.0 / 3.9 / 4.0 s / ảnh → cùng thời gian.
R50 / R101 / ConvNeXt / Swin-T / YOLOX / Swin-L | CA3: WP2 B36 (sinh lại) 98.8 / 91.4 / 82.7 / 82.6 / 77.1 / 35.5 | 80.8 (10-23: 37.5 | 80.5); **C2W03** 98.5 / 92.2 / 84.0 / 82.7 / 76.9 / 35.6 | 81.2; **C2W1** 97.0 / 91.0 / 83.3 / 74.9 / 69.1 / 35.9 | 75.8.
Δ so WP2: C2W03 Swin-L **+0.1 [−2.2, +4.5]**, CA3 +0.4; C2W1 Swin-L **+0.4 [−3.0, +3.0]**, CA3 **−5.0 [−6.6, −0.3]** (YOLOX −8.0, Swin-T −7.7).
**Đọc:** dồn gradient vào L_OSFD-C2 không tăng DINO-Swin-L ở cả 2 liều; liều ngang (C2W1) hại Swin-T / YOLOX. Khớp kỳ vọng trước (0 … +3) và E19 (gradient C2 căn δ_SwinT kém G8). Theo khai báo: **đóng dòng (3) của E19; E19 ghi là kết luận âm** (không ước lượng / trọng số thành phần chỉ-R50 nào trỏ về δ_SwinT); bước tiếp theo khai báo: **WP2 lên sàng lọc dev300** (entry riêng, khóa trước) — chờ user.

---

## 2026-10-24 — Định hướng (user duyệt "approve", 15:00 GMT+7): E20 basin test + method PFA (chèn attention KHÔNG tham số vào R50) @ dev30 — khóa TRƯỚC khi code / chạy
**Lý do (tổng hợp dữ liệu, chi tiết ở trả lời Claude 10-24):** mọi núm phía đầu vào / loss trượt 1 đường đánh đổi; chỉ phía model (WP2) tách được Swin-L (+6) nhưng bão hòa ~37–40; E19: không ước lượng hướng chỉ-R50 nào trỏ về δ_SwinT; δ_SwinT không phẳng hơn trên R50 (loại RAP / PGN). Mục tiêu user: Swin-L ≥ 50 ở ε = 5, chỉ R50 cố định, novelty. Literature 10-2026: RaPA (CVPR'26, arXiv 2504.18594, tỉa tham số ngẫu nhiên, training-free) = rủi ro novelty cho họ WP → bắt buộc làm baseline ở dev300; chưa thấy công trình chèn attention không tham số vào surrogate CNN để transfer sang ViT (gần nhất: FPR CVPR'25, PNA — đều surrogate ViT).

### E20 — basin test: gradient R50 GIỮ hay PHÁ vùng chung δ_SwinT? (MÔ TẢ / cổng định hướng, không phải method)
**Ô:** δ_S = C2-OCT-1V, `--surrogate swin_t`, B41, dev30 (sinh lại; run `dev300_B41_eps5_limit100_sur-swin_t`). `C2-OCT-1V-WP2-IS` = WP2 y nguyên (ρ = 0.002, B36) nhưng δ khởi tạo = PNG δ_S − ảnh gốc (thay cho 0; momentum MI khởi tạo 0) — chỉ dùng để phân tích (dùng surrogate Swin-T, NGOÀI threat model). Snapshot B12; B0 = chính δ_S.
**Đo (dev30, so mốc WP2 B36 sinh sáng nay, cùng máy):** R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L ở B0 / B12 / B36; cos(δ_IS, δ_S), cos(δ_IS, δ_W) (mô tả).
**Đọc (khai báo trước):**
- **GIỮ ĐƯỢC** nếu IS B36: Swin-L ≥ 52 VÀ R50 ≥ 95 → vùng chung là nơi R50 "ở lại được"; vấn đề là ĐƯỜNG VÀO (khởi tạo / tìm kiếm toàn cục) → đáng đầu tư method; PFA vẫn chạy.
- **RỜI VÙNG** nếu IS B36: Swin-L ≤ 42 → chính gradient R50 kéo δ khỏi vùng chung → mọi method chỉ sửa đầu vào / loss bị chặn trần; chỉ còn đường đổi trường gradient của surrogate (PFA). PFA cũng KHÔNG → đề xuất user dừng đuổi Swin-L ≥ 50 ở ε = 5 với chỉ R50.
- Còn lại → XÁM (mô tả, PFA vẫn chạy).
**Caveat:** δ_S tối ưu trên Swin-T (cùng họ Swin-L, khác detector / pretrain); n = 30, dev30 đã dùng nhiều lần; GPU không tất định.

### PFA — Parameter-Free Attention injection vào R50 lúc attack (ứng viên method)
**Cơ sở:** E14 / E14b — nút thắt Swin-L = HƯỚNG distortion ở token object stage 3 (stride 16); 10-03: encoder DINO dập distortion vừa; R-a (10-04) trượt vì R50 không có đường "sửa" nào; E7: δ_R50 là mẫu tần cao khớp nội dung cục bộ. Token Swin = trung bình có trọng số các token tương tự trong cửa sổ → δ chỉ tạo phản ứng cục bộ rời rạc bị trung bình hóa. PFA cho R50 một đường trộn kiểu attention (không tham số, không huấn luyện) để gradient phải tạo distortion NHẤT QUÁN trên vùng object. Đổi trường gradient của surrogate (loại duy nhất đã tách Swin-L: WP).
**Phép toán (`pia.pfa_hook`, `geo_osfd_attack(pfa=(α, τ, w))`):** tại đầu ra `backbone.layer3` (C4, stride 16, 1024 kênh) — đầu ra này đi cả vào L_OSFD (stage C4) lẫn layer4 → C5 → FPN → head: F' = (1 − α)·F + α·A(F); A trong cửa sổ không chồng w × w = 7 × 7 token, lưới cửa sổ lệch (oy, ox) ∈ {0..6}² rút đều MỖI VIEW (RNG riêng seed + 11 → chuỗi view OCT trùng WP2), CÙNG lệch cho feature sạch co-transform và adv của view; token pad ngoài ảnh không làm key; trọng số softmax_j(cos(f_i, f_j) / τ), value = F (không chiếu), **τ = 0.1 cố định** (không tune). Gốc = WP2 (C2-OCT-1V, ρ = 0.002, λ = 4). θ R50 không đổi; hook gỡ sau mỗi ảnh.
**Kiểm:** α = 0 trùng WP2 (sai số số học); A với τ → 0 ≈ đồng nhất; θ khôi phục; unit test cửa sổ / pad / lệch.
**Chọn α CHỈ theo sức khỏe (KHÔNG nhìn target), α ∈ {0.25, 0.5, 0.75}:** (H1) AP sạch R50 có PFA (lệch 0) trên dev30 ≥ 85% AP sạch R50 thường; (H2) white-box (`scripts/tx_calib.py`, ảnh 1–4 dev300, B36, đo trên R50 THƯỜNG): |δ| = 5 ≥ 80% pixel VÀ R50 xóa ≥ 0.8 × mức WP2. Chạy **2 α lớn nhất qua cả H1, H2** (`C2-OCT-1V-WP2-PFA<α×100>`); chỉ 1 qua → 1 ô; không α nào → ghi lại, PFA dừng ở cổng sức khỏe.
**B:** đo s / ảnh (ảnh 1–4) → B = ⌈36 / tỉ lệ thời gian so WP2⌉ = cùng thời gian WP2 B36; ghi runtime thực.
**Đọc (dev30, so WP2 B36 cùng 30 ảnh, như TX / XBG / C2W), mỗi ô:**
- **BƯỚC NHẢY** nếu ΔSwin-L ≥ +10 → dev100 ngay (khóa entry).
- **CÓ TÍN HIỆU** nếu +3 ≤ ΔSwin-L < +10 VÀ ΔCA3 ≥ −3 → dev100 (khóa entry), chưa gọi method mới.
- **KHÔNG** nếu mọi ô ΔSwin-L < +3 (hoặc ΔCA3 < −3).
- Mô tả bắt buộc: ΔSwin-T (claim "CNN → Transformer" cần Swin-T cũng tăng, ≥ +2), R101, ConvNeXt, YOLOX.
**Caveat:** PFA đổi hàm của surrogate (AP sạch giảm) → gradient có thể của một model "hỏng"; τ / w / vị trí C4 đặt trước theo cơ sở E14, không quét; dev30 nhiễu.
**Chạy:** `results/pfa/run.sh`, log `results/pfa/run.log`; metrics `metrics_pfa_dev30.json`, dets `dets_pfa_dev30.json`, pair_screen tag `pfa_dev30` / `e20_dev30`.

### PFA — đính chính cổng H2 (~15:25 GMT+7, TRƯỚC khi có bất kỳ số transfer nào của PFA / E20)
Lỗi chép quy tắc: H2 ghi "|δ| = 5 ≥ 80% pixel", nhưng ý định (như C2W / XBG: "|δ| = 5 VÀ R50 xóa ≥ 80% mức WP2") là sức khỏe SO VỚI WP2; chính mốc WP2 chỉ có 79.51% pixel chạm ±5 (ảnh 1–4) → đọc nguyên văn thì mọi ô kể cả mốc đều trượt. Áp: L∞ = 5, tỉ lệ bão hòa ≥ 0.95 × WP2, R50 xóa ≥ 0.8 × WP2. Quyết định KHÔNG phụ thuộc cách đọc: mọi ô trùng WP2.
**Cổng sức khỏe (`results/pfa/health_h1.json`, `scripts/tx_calib.py`, log `results/pfa/h2_attack.log`):** H1 AP sạch R50 dev30: thường 40.8; α 0.25 / 0.5 / 0.75 → 41.0 / 41.3 / 42.5 (100–104%) QUA. H2 (B36, ảnh 1–4): bão hòa WP2 79.51%, PFA25 79.64%, PFA50 79.52%, PFA75 79.48%; R50 xóa 92% (34/37) ở cả 4 → QUA. Unit test `scripts/pfa_unit_test.py` ĐẠT (khớp vòng lặp ngây thơ ≤ 5e-7; α = 0 trùng R50). Thời gian ảnh 1–4 (s / ảnh): WP2 3.83, PFA25 3.97, PFA50 3.79, PFA75 3.82 → **B36 = cùng thời gian**. Theo quy tắc: chạy 2 α lớn nhất qua cổng: **PFA75, PFA50**.

### Kết quả E20 + PFA @ dev30 (V100; `results/ablation/{e20_dev30,pfa_dev30}/screen.json`, `results/runs/dev300_B{12,36}_eps5_limit100/*_e20pfa_dev30.json`, `results/runs/dev300_B41_eps5_limit100_sur-swin_t/*_e20pfa_dev30.json`, log `results/pfa/run.log`)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3:
- WP2 B36 (mốc): 98.8 / 91.4 / 82.7 / 82.6 / 77.1 / **35.5** | 80.8
- δ_S = B0 (sur Swin-T B41): 83.3 / 79.6 / 88.2 / 96.4 / 71.5 / **59.3** | 85.4
- WP2-IS B12: 96.5 / 86.5 / 84.9 / 84.9 / 74.4 / **35.5** | 81.4
- WP2-IS B36: 99.3 / 94.1 / 91.4 / 90.9 / 84.1 / **39.8** | 88.8 (Δ so WP2: Swin-L +4.3 [−1.9, +9.6], CA3 +8.0 [+5.5, +13.6])
- PFA50: Swin-L 36.0 (Δ +0.5 [−2.0, +2.6]); PFA75: Swin-L 33.9 (Δ −1.6 [−4.6, +1.8]); YOLOX −0.5 / +1.2; CNN ≈ WP2.
**Đọc theo quy tắc:** E20 = **RỜI VÙNG** (IS B36 Swin-L 39.8 ≤ 42): 12 bước R50 đã kéo Swin-L 59.3 → 35.5 (= mốc từ 0), sau đó R50 đạt 99 mà Swin-L chỉ hồi 39.8. PFA = **KHÔNG** (cả 2 ô ΔSwin-L < +3). Theo khai báo: kết luận DỪNG (gradient ascent trên R50 cố định chặn trần Swin-L ~37–40 ở ε = 5). Mô tả: khởi tạo từ δ_S cho CNN +8 (ngoài threat model — dùng surrogate Swin-T).

---

## 2026-10-24 — E21: OTSV (object-token surrogate virtualization) — cổng ORACLE-RETENTION trước mọi attack từ 0 (user duyệt đề xuất kèm chỉnh sửa, 15:40 GMT+7) — khóa TRƯỚC khi chạy
**Lý do:** đề xuất ngoài (OTSV) + phản biện của Claude: PFA (cùng họ "tái tổ hợp feature R50") vừa KHÔNG, E20 cho thấy trường gradient R50 chủ động đẩy δ khỏi vùng chung → OTSV chỉ đáng attack từ 0 nếu landscape của nó GIỮ được δ_S. Cổng căn hướng tại P0 (E19) bị bỏ: phản ví dụ WP4 < G8 nhưng WP2 thắng. Kỳ vọng trước (user + Claude): qua cổng ~10–20%.
**Phép toán (`pia.otsv_mix` / `otsv_hook`, `geo_osfd_attack(otsv=β)`):** tại đầu ra `backbone.layer3` (C4, stride 16): Z = LayerNorm kênh (không affine) của F; key t_j = trung bình RoIAlign 7×7 (aligned, sampling 2) của Z tại box GT j (co-transform của view); value v_j = cùng phép trên F (khác đề xuất gốc dùng T từ Z — chọn value trong không gian F để residual cùng thang, quyết định TRƯỚC khi đo); A_ij = softmax_j(z_i·t_j / √C); F' = F + β·A·V. Lịch 1 : 1 — bước lẻ forward ảo, bước chẵn R50 gốc; feature sạch co-transform qua CÙNG chế độ. Gốc = WP2 (ρ = 0.002, λ = 4, OCT 1V). Không box → đồng nhất. Lưu ý: ảnh 1 box → mọi token cộng cùng β·v (suy biến). Unit test `scripts/otsv_unit_test.py` ĐẠT.
**Chọn β (target-free, `scripts/otsv_health.py` + `scripts/tx_calib.py`), β ∈ {0.25, 0.5, 1.0}: β LỚN NHẤT đạt đồng thời** H1 AP sạch R50 (forward ảo, dev30) ≥ 90% R50 thường; H3a ‖F' − F‖ / ‖F‖ tại C4 ≥ 0.05; H3b cos(∇ₓ loss attack ảo, ∇ₓ gốc) tại δ = 0 ≤ 0.95 (ảnh 1–4 dev300); H2 white-box OTSV-WP2 từ 0 (B36, ảnh 1–4): bão hòa ≥ 0.95 × WP2, R50 xóa ≥ 85%. Không β nào → OTSV dừng ở cổng sức khỏe. Đúng 1 β đi tiếp.
**Cổng oracle-retention (dev30, cùng ảnh / seed / RNG OCT với WP2-IS của E20):** `C2-OCT-1V-WP2-OTSV<β>-IS` = δ khởi tạo δ_S, B36, snapshot B1 / B4 / B12; đối chứng WP2-IS bổ sung B1 / B4 (chạy lại B4, snapshot B1).
- **QUA** nếu B36: Swin-L ≥ 52 VÀ ≥ WP2-IS B36 + 10 (≥ 49.8) VÀ R50 drop(B36) > R50 drop(B0 = 83.3) VÀ δ thực sự di chuyển: cos(δ_B36, δ_B0) ≤ 0.95 (trung bình ảnh; báo kèm L2 tương đối và giá trị của WP2-IS) → attack từ 0 dev30 cùng thời gian WP2 (entry riêng, ngưỡng +10 / +3).
- **XÁM** nếu 48 ≤ Swin-L B36 < 52 (các điều kiện khác đạt): chỉ attack từ 0 nếu quỹ đạo B12 → B36 không giảm (Swin-L B36 ≥ B12 − 1); ngược lại dừng.
- **DỪNG** nếu Swin-L B36 < 48, hoặc tụt kiểu WP2 về ~40.
- Mô tả quỹ đạo: rơi ở B1–B4 = trường gradient sai cơ bản; giữ tới B12 rồi rơi = forward ảo có tác dụng nhưng bị forward gốc / momentum lấn.
**Không làm bây giờ:** LPAA / H-LPA, AES (snapshot soup), RaPA (bản đúng paper CVPR'26, phân biệt với RaPA hiện có trong repo), BSR (chỉ loss feature) → để bảng dev300.
**Chạy:** `results/otsv/run.sh`, log `results/otsv/run.log`.

### Kết quả E21 @ dev30 (V100; `results/otsv/health.json`, `results/ablation/otsv_gate_dev30/screen.json`, `results/runs/dev300_B{1,4,12,36}_eps5_limit100/*_otsv_dev30.json`, `results/otsv/delta_move.json`, log `results/otsv/{run,h2_attack}.log`) — theo quy tắc: **DỪNG**
**Chọn β:** H1 / H3a / H3b — β 0.25: AP 101%, ΔC4 0.127, cos 0.914 QUA; **β 0.5: 97%, 0.254, 0.847 QUA**; β 1.0: 84%, 0.508, 0.618 TRƯỢT (H1). H2 (B36, ảnh 1–4): OTSV50 bão hòa 80% (= WP2), R50 xóa 92% QUA; 3.9 s / ảnh ≈ WP2 → **β = 0.5**.
**Quỹ đạo DINO-Swin-L (B0 = δ_S 59.3), WP2-IS | OTSV50-IS:** B1 57.2 | 57.2; B4 48.8 | 48.6; B12 35.5 | 35.1; **B36 39.8 | 44.0**. Δ (OTSV − WP2-IS) B36: Swin-L **+4.2 [−0.1, +5.8]**, CA3 **−2.3 [−4.1, −0.3]** (ConvNeXt −2.4, Swin-T −3.2, YOLOX −1.3), R101 +0.8; B12 −0.4, B4 −0.2, B1 0.0.
**δ:** cos(δ, δ_S) B1 0.992 / B4 0.885 / B12 0.316 / B36 0.218 (WP2-IS 0.992 / 0.884 / 0.314 / 0.216); cos(δ_OTSV, δ_WP2-IS) B4 0.973, B12 0.809, B36 0.646.
**Đọc:** B36 44.0 < 48 → DỪNG. OTSV KHÔNG giữ vùng δ_S (sụp cùng tốc độ WP2 tới B12, cùng mức "nhớ" δ_S ở B36); khác biệt chỉ ở pha hồi B12 → B36, và đổi lấy CNN (CA3 −2.3, CI < 0) → cùng đường đánh đổi CNN ↔ Swin-L. Phân rã 44.0 = 35.5 (WP2 từ 0) + 4.3 (oracle init, ngoài threat model) + 4.2 (OTSV, CI chạm 0). Mô tả: Swin-L hình chữ U (sụp tới B12 rồi hồi) ở cả 2 ô.

---

## 2026-10-24 — E22: WP2 thắng nhờ CÁI GÌ? Chẩn đoán hướng + vị trí nhiễu trọng số @ dev30 B36 (user: "phân tích thật kỹ vì sao C2-OCT-1V-WP2 thành công, rồi improve") — MÔ TẢ / sinh cơ chế, khai báo TRƯỚC khi chạy
**Bối cảnh:** literature — AWT (arXiv 2408.09469, classification): surrogate GIẢM loss trên AE (min-max kiểu SAM, tích lũy), CNN → ViT +8–10 → WP2 ≈ bản rẻ của AWT cho detection → novelty phải đến từ hiểu biết cơ chế đặc thù detection. WP2 đã biết: lợi tập trung ở DINO-Swin-L (+6.2 dev100), KHÔNG ở Mask R-CNN Swin-T; ρ = 0.005 mất lợi; tích lũy (WPA) không thêm; 42 tensor conv layer2 (13) / layer3 (19) / layer4 (10).
**Giả thuyết H_def ("surrogate phòng thủ ≈ target biết SỬA"):** DINO-Swin-L "sửa" distortion vừa phải (encoder dập cosd 0.42 → 0.07, 10-03) còn FPN của Mask R-CNN Swin-T khuếch đại → δ tối ưu chống một R50 đang tự chỉnh để giảm loss chính là δ bền trước phép sửa → lợi riêng ở DINO. Dự đoán: (i) hướng PHÒNG THỦ quan trọng (ngẫu nhiên / đồng lõa không thay được); (ii) nếu sửa diễn ra ở tầng tổng hợp ngữ nghĩa, phòng thủ ở HEAD phân loại (đường L_cls — đòn bẩy Swin-L) có thể thêm lợi.
**Ô (gốc C2-OCT-1V-WP2, ρ = 0.002, B36 = cùng thời gian; `core.run_iterative_attack(weight_perturb mode)`, `geo_osfd_attack(weight_scope, weight_mode)`):**
- `WP2-RAND`: hướng ngẫu nhiên Gauss chuẩn hóa từng tensor, cùng bán kính ρ‖θ0‖, rút mới mỗi bước (RNG CUDA riêng seed + 13) — đối chứng "đa dạng mô hình" (họ RaPA / Gauss).
- `WP2-ASC`: đổi dấu, θ' = θ0 + ρ‖θ0‖ĝ (surrogate ĐỒNG LÕA với attacker).
- `WP2-L3`: phòng thủ chỉ trên layer3 (19 tensor, C4 — nút thắt E14); `WP2-noL3`: chỉ layer2 + layer4 (23 tensor).
- `WPH`: phòng thủ chỉ trên head đường phân loại (rpn_conv, rpn_cls, shared_fcs.0 / .1, fc_cls; chỉ weight), KHÔNG backbone; `WP2+H`: backbone + head. ρ = 0.002 cho head (không tune).
- Mốc: WP2 B36 (PNG sáng nay), C2-OCT-1V B41 (sinh lại hôm nay, cùng thời gian).
**Sức khỏe (ảnh 1–4 dev300, `tx_calib`):** R50 xóa ≥ 0.8 × WP2 và bão hòa ≥ 0.95 × WP2; ô trượt → ghi, không chạy dev30.
**Đọc (khai báo trước, dev30, mô tả — claim cần dev100):**
- H_def (i) ỦNG HỘ nếu WP2 − RAND ≥ +3 VÀ WP2 − ASC ≥ +3 (Swin-L); BÁC nếu RAND ≥ WP2 − 1 (→ WP2 = đa dạng mô hình, không phải min-max).
- Vị trí: L3 giữ ≥ 75% lợi Swin-L của WP2 so C2-OCT-1V (cùng ngày) → "phòng thủ ở C4 đủ"; noL3 giữ ≥ 75% → không đặc thù C4; cả hai / không → phân tán.
- Head (ứng viên cải tiến): WP2+H hoặc WPH có ΔSwin-L ≥ +3 so WP2 VÀ ΔCA3 ≥ −3 → dev100 (khóa entry riêng).
**Caveat:** 6 ô trên 30 ảnh (so sánh bội) → chỉ sinh giả thuyết; dev30 đã dùng nhiều lần; GPU không tất định.
**Chạy:** `results/e22/run.sh`, log `results/e22/run.log`; metrics `metrics_e22_dev30.json`, dets `dets_e22_dev30.json`, pair_screen tag `e22_dev30`.

### Kết quả E22 @ dev30 B36 (V100; `results/ablation/e22_dev30/screen.json`, `results/runs/dev300_B{36,41}_eps5_limit100/*_e22_dev30.json`, log `results/e22/run.log`)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3: C2-OCT-1V B41 99.3 / 93.5 / 82.8 / 80.6 / 62.0 / 30.2 | 75.1; **WP2** 98.8 / 91.4 / 82.7 / 82.6 / 77.1 / 35.5 | 80.8; RAND 99.4 / 94.4 / 82.9 / 78.2 / 62.7 / 31.6 | 74.6; ASC 98.7 / 90.7 / 76.1 / 74.6 / 59.2 / 23.7 | 70.0; L3 99.1 / 92.5 / 83.7 / 80.6 / 74.6 / 33.3 | 79.6; noL3 99.1 / 91.7 / 81.6 / 84.4 / 75.8 / 36.0 | 80.6; WPH 99.3 / 93.7 / 80.5 / 80.9 / 62.1 / 29.8 | 74.5; WP2+H 98.5 / 92.8 / 81.0 / 81.8 / 77.5 / 37.5 | 80.1.
Δ so WP2 (Swin-L | CA3 | YOLOX): RAND −3.9 [−7.0, +1.2] | −6.2 | −14.4; ASC −11.8 [−16.4, −6.1] | −10.8 | −17.9; L3 −2.2 | −1.2 | −2.5; noL3 +0.5 | −0.2 | −1.2; WPH −5.7 [−7.8, −0.8] | −6.3 | −15.0; WP2+H +2.0 [−1.2, +5.5] | −0.7 | +0.4. WP2 − C2-OCT-1V (hôm nay): Swin-L +5.3 [−0.5, +7.9], YOLOX +15.1.
**Đọc theo khai báo:**
- **H_def (i) ỦNG HỘ:** WP2 − RAND = +3.9 ≥ 3 VÀ WP2 − ASC = +11.8 ≥ 3 → lợi của WP2 đến từ HƯỚNG PHÒNG THỦ, không phải đa dạng mô hình; RAND ≈ C2-OCT-1V (không thêm gì), ASC (đồng lõa) hại mạnh mọi target. Mô tả: YOLOX là nơi hướng phòng thủ tạo chênh lớn nhất (+15 so C2 / RAND / WPH).
- **Vị trí:** L3 giữ 3.1 / 5.3 = 58% (< 75%); noL3 giữ 5.7 / 5.3 ≈ 108% (≥ 75%) → **không đặc thù C4**; layer2 + layer4 đủ.
- **Head:** WPH HẠI (≈ C2-OCT-1V: phòng thủ chỉ ở head không thay được backbone); WP2+H +2.0 < +3 → không phải ứng viên cải tiến.
**Hệ quả:** WP2 = min-max PHÒNG THỦ ở backbone (cùng nguyên lý AWT), không phải augmentation; phòng thủ ở head phân loại không thêm lợi đáng kể. Giả thuyết "phòng thủ ở head mô phỏng phép sửa của DINO" KHÔNG được ủng hộ.

---

## 2026-10-24 — E23: min-max TÁCH MỤC TIÊU — surrogate chỉ phòng thủ chống MỘT thành phần loss (DEF-CLS / DEF-FEAT) @ dev30 B36 (user duyệt, ~17:35 GMT+7) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Cơ sở:** E22 — lợi WP2 = hướng PHÒNG THỦ (RAND −3.9, ASC −11.8); WP2 phòng thủ chống CHÍNH loss attack L = L_OSFD + 4·L_cls. Hai thành phần gần trực giao (cos ∇ ≈ −0.08, 10-18) và phục vụ target khác nhau (L_cls → DINO-Swin-L; L_OSFD → CNN / YOLOX). Câu hỏi: surrogate chống thành phần nào thì δ "tôi luyện" ở kênh đó?
**Literature (search 10-24):** AWT (2408.09469): defender cùng loss với attacker (+ loss sạch), classification. BMAT (2608.11815): bilevel-minimax trên trọng số, classification + segmentation, abstract không nói tách loss → baseline / trích dẫn bắt buộc. Chưa thấy defender dùng thành phần loss khác attacker, chưa thấy cho detection.
**Ô (gốc WP2: ρ = 0.002, backbone, B36; `make_view_loss(split=True)`, `core.run_iterative_attack` thêm thành phần phòng thủ):** attacker luôn tăng L_OSFD + 4·L_cls (y nguyên); θ' = θ0 − ρ‖θ0‖·ĝ_c với g_c = ∇θ của:
- `WP2-DEFCLS`: chỉ 4·L_cls; `WP2-DEFFEAT`: chỉ L_OSFD.
**Kiểm:** split trùng tổng (sai số số học); DEF với cả 2 thành phần ≡ WP2. Sức khỏe (ảnh 1–4 dev300, `tx_calib`): R50 xóa ≥ 0.8 × WP2, bão hòa ≥ 0.95 × WP2. Thời gian đo trên ảnh 1–4 → B = ⌈36 / tỉ lệ⌉ (2 lần autograd.grad trên cùng graph).
**Đọc (so WP2 B36 cùng 30 ảnh, như các ô trước):** BƯỚC NHẢY ΔSwin-L ≥ +10; CÓ TÍN HIỆU +3 ≤ ΔSwin-L < +10 VÀ ΔCA3 ≥ −3 → dev100 (khóa entry); KHÔNG nếu cả 2 ô ΔSwin-L < +3 → chạy ý phụ (phòng thủ trong không gian BN affine, entry riêng). Mô tả: DEF-FEAT có tăng CNN / YOLOX không (dự đoán của giả thuyết tách kênh).
**Caveat:** dev30, n = 30, GPU không tất định; máy đang chậm bất thường (xung GPU thấp từ ~16:30) → thời gian đo lại trước khi chạy.
**E23 sức khỏe + thời gian (ảnh 1–4 dev300, `results/e23/health_attack.log`):** bão hòa / R50 xóa — WP2 80% / 92%, DEFCLS 79% / 95%, DEFFEAT 80% / 92% → QUA. s / ảnh: WP2 3.83, DEFCLS 4.27 (1.11×), DEFFEAT 4.05 (1.06×) → **DEFCLS B33, DEFFEAT B34** = cùng thời gian WP2 B36 (máy đang chậm bất thường — tỉ lệ có thể lệch, ghi lại).

### Kết quả E23 @ dev30 (V100; `results/ablation/e23_dev30/screen.json`, `results/runs/dev300_B{33,34,36}_eps5_limit100/*_e23_dev30.json`, log `results/e23/run.log`) — theo khai báo: **KHÔNG**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3: WP2 B36 98.8 / 91.4 / 82.7 / 82.6 / 77.1 / 35.5 | 80.8; DEFCLS B33 98.4 / 91.8 / 83.4 / 84.3 / 69.7 / 36.8 | 79.1; DEFFEAT B34 98.8 / 92.2 / 85.3 / 82.2 / 73.5 / 30.5 | 80.3.
Δ so WP2: DEFCLS Swin-L +1.3 [−3.5, +5.5], CA3 −1.7, YOLOX −7.4 [−11.4, −0.7]; DEFFEAT Swin-L −5.0 [−7.6, −0.2], CA3 −0.5, ConvNeXt +2.6.
**Đọc:** cả 2 ô ΔSwin-L < +3 → KHÔNG. Phòng thủ chống TOÀN BỘ loss (WP2) tốt nhất; chống riêng L_OSFD làm mất lợi Swin-L, chống riêng L_cls làm mất lợi YOLOX → mỗi thành phần phòng thủ "giữ" lợi ở kênh của thành phần KIA (dự đoán tách kênh bị đảo / không đứng). Theo khai báo: bước kế = ý phụ (phòng thủ trong không gian BN affine) — chờ user (Claude đề xuất cân nhắc dừng vòng cải tiến: WP5 / WPA / head / tách mục tiêu đều không vượt WP2).

---

## 2026-10-24 — E24: có THƯỚC ĐO chỉ-dùng-R50 nào xếp δ theo mức phá DINO-Swin-L không? (điều kiện cần cho mọi method 1-R50 đạt Swin-L ≥ 50; user duyệt "Approve E24") — MÔ TẢ / sinh cơ chế, forward-only trên PNG có sẵn, khóa TRƯỚC khi chạy
**Lập luận:** E20/E21 — bộ tối ưu không phải vấn đề (đặt δ trong vùng tốt, gradient R50 vẫn kéo ra); vấn đề là MỤC TIÊU. Mọi method 1-R50 (bất kể bộ tối ưu / khởi tạo / biến đổi đầu vào) chỉ đạt Swin-L cao nếu TỒN TẠI đại lượng φ(x+δ) tính chỉ từ R50 mà ĐỒNG BIẾN với mức phá DINO-Swin-L qua nhiều δ. E19 đã bác các φ trên R50 GỐC θ0 (hồ sơ loss δ_S/δ_W mọi thành phần < 1; độ phẳng đầu vào không phân biệt) — E24 mở rộng sang R50 BIẾN ĐỔI (phòng thủ / nhiễu trọng số lớn / biến đổi đầu vào mạnh / đo theo HƯỚNG) + kiểm trên ~20 δ thay vì 2 điểm (sửa điểm yếu E19). Khác E18 (cổng RSE, đo "object↔nền"): E24 đo trực tiếp giá trị dùng làm LOSS.
**Dữ liệu (đã có PNG + dets DINO-Swin-L dev30, KHÔNG attack mới):** 23 δ / ảnh — C2-OCT-1V (B41), δ_SwinT (sur Swin-T B41), WP2 / RAND / ASC / L3 / noL3 / WPH / WP2+H / C2W03 / C2W1 / PFA50 / PFA75 (B36), DEFCLS (B33), DEFFEAT (B34), WP2-IS & OTSV50-IS (B1/B4/B12/B36 — quỹ đạo chữ U: Swin-L 57→35→44). y = suppression DINO-Swin-L theo ảnh (`attack.mechanism.per_image_transfer`, trên GT DINO thấy khi sạch).
**Thước đo φ (mỗi δ, mỗi ảnh; chỉ R50; trung bình n_mc mẫu khi ngẫu nhiên; forward trên PNG → feature R50):**
- `L_plain`: L_OSFD(k=3) + 4·L_cls trên θ0 (đối chứng âm — E19 kỳ vọng ρ ≤ 0).
- `L_def1`: trên θ' = θ0 − ρ‖θ0‖ĝ 1 bước (WP). `L_def3`: 3 bước tích lũy.
- `L_rand`: trung bình trên 8 mẫu cắt tham số Gauss lớn (ρ = 0.01, 0.02).
- `L_octbig`: trung bình 8 view co giãn ±1 octave + làm mờ σ ∈ {0,2,4} (bền trước biến đổi đầu vào mạnh).
- `cosd_C45`: 1 − cos(F_adv, F_sạch) trung bình C4, C5 (ĐO HƯỚNG thay vì MSE — E14 nút thắt là hướng).
- `L_cls_only`: chỉ 4·L_cls trên θ0.
**Quy tắc (dev30, 23 δ × 30 ảnh; Spearman theo ảnh giữa φ và suppression Swin-L, gộp mọi δ; paired bootstrap 1000 ảnh, BOOT_SEED 2026):**
- **ĐỐI CHỨNG:** báo ρ(L_plain). Thước đo "hơn hẳn" nếu ρ(φ) − ρ(L_plain) > 0 cận dưới CI > 0.
- **ỨNG VIÊN MỤC TIÊU** nếu ρ(φ) > 0 CI loại trừ 0 VÀ hơn hẳn L_plain VÀ trên RIÊNG quỹ đạo IS (B1→B12, Swin-L sụp 57→35) φ GIẢM theo (Spearman(φ, B) theo mốc < 0) — tức φ bắt được pha sụp.
- **ÂM** nếu không φ nào đạt → bằng chứng định lượng: không mục tiêu chỉ-R50 nào (đã thử) trỏ về vùng phá Swin-L → method 1-R50 đạt Swin-L ≥ 50 không khả thi theo nghĩa này; kết luận cho paper + user quyết (surrogate thứ 2 / định vị SOTA + cơ chế).
- Mô tả: ρ cho ConvNeXt / Swin-T / YOLOX (φ có đồng biến target khác không); thứ hạng δ theo φ tốt nhất (δ_S đứng đâu).
**Caveat:** tương quan quan sát trên δ CÓ SẴN (không chứng minh tối ưu thẳng φ sẽ đạt — đó là bước sau, điều kiện đủ); 23 δ không độc lập (nhiều biến thể WP2); suppression bão hòa; GPU không tất định; dev30. Tham số (ρ, octave, σ, n_mc = 8) đặt trước, không tune.
**Chạy:** `scripts/e24_target_probe.py` → `results/mechanism/e24_target_probe.json`, log `results/e24/run.log`.

### Kết quả E24 @ dev30 (V100; `results/mechanism/e24_target_probe.json`, log `results/e24/run.log`) — theo quy tắc: **ÂM** (không φ nào đạt)
Spearman(φ, suppression DINO-Swin-L) qua 23 δ × 30 ảnh (bootstrap ảnh, paired): L_plain **0.172** [−0.12, 0.43]; L_def1 0.148; L_def3 0.158; L_rand 0.174; L_octbig 0.028; cosd_C45 0.055; **L_cls_only 0.340 [0.06, 0.55]** (duy nhất CI loại trừ 0).
**Cổng quỹ đạo IS (quyết định):** Swin-L hình chữ U — suppression B1/B4/B12/B36 = 0.352 / 0.328 / **0.24** / 0.353 (sụp tới B12 rồi hồi). MỌI φ TĂNG ĐƠN ĐIỆU theo B (Spearman(φ, B) = +1.00 tất cả): L_cls_only 3.84 → 4.31 → 5.2 → 7.17; L_plain 21.9 → 29.4; cosd_C45 0.32 → 0.62 → φ tăng TRONG KHI Swin-L sụp (B1→B12) → không φ nào bắt pha sụp.
**Đọc:** L_cls_only tương quan dương QUA các δ nhưng NGƯỢC hướng trên chính đường tối ưu (bệnh E19/E20 tái hiện định lượng: thước đo xếp đúng δ không-đồng-nhất vẫn bị gradient ascent kéo ra khỏi vùng Swin-L). Không φ nào vừa dự đoán qua δ vừa giảm theo pha sụp IS → **ÂM**. Bằng chứng định lượng cho paper: trong 7 thước đo chỉ-R50 đã thử (loss gốc, phòng thủ 1/3 bước, nhiễu trọng số lớn, biến đổi đầu vào mạnh, đo hướng cosd, chỉ phân loại), KHÔNG cái nào là mục tiêu trỏ về vùng phá DINO-Swin-L theo cách gradient ascent khai thác được → method 1-R50 đạt Swin-L ≥ 50 ở ε = 5 không khả thi theo nghĩa này. Chưa tính ρ cho ConvNeXt/Swin-T/YOLOX (mô tả — bỏ, kết luận không phụ thuộc). Giới hạn: 7 φ không vét cạn; 23 δ không độc lập; suppression bão hòa; dev30.
**Trạng thái chuỗi "đạt Swin-L ≥ 50 với 1 R50":** E19 (hướng ước lượng), E20 (basin), E21 (OTSV), E22 (phòng thủ), E23 (tách mục tiêu), E24 (mục tiêu) — TẤT CẢ ÂM. Ứng viên tốt nhất vẫn là WP2 (Swin-L ~36–37 dev100). Hai đường còn lại (user đã loại ε = 8): surrogate thứ 2 (đổi câu hỏi) HOẶC định vị SOTA + cơ chế + trần có bằng chứng (dev300, baseline AWT/BMAT/RaPA/LPAA). User quyết.

---

## 2026-10-24 — E25: WP-Ensemble (min-max phòng thủ NHIỀU bản R50 song song) — cổng oracle-retention trước, từ-0 sau (user duyệt "Duyệt", ~18:30 GMT+7) — khóa TRƯỚC khi code / chạy
**Lý do:** E22 — phòng thủ là lý do WP2 tách Swin-L (RAND vô dụng, ASC hại). WP2 = min-max 1 ĐIỂM trọng số, trễ 1 bước; WPA (tích lũy 1 hướng) không thêm. Chưa thử: phòng thủ trên cả LÂN CẬN trọng số = K bản θ' tỏa nhiều hướng phòng thủ, δ phải thắng đồng thời cả K (đúng SAM/AWT min-max lân cận, WP2 chỉ xấp xỉ 1 điểm). Kết hợp ý user: δ_SwinT init (E20/E21) → cổng retention kiểm WP-Ensemble có GIỮ vùng tốt mà WP2-IS (39.8) / OTSV-IS (44) làm tuột không. CAVEAT threat model: δ_SwinT cần surrogate Swin-T (NGOÀI threat model) + E24 (không dựng lại được từ R50) → -IS CHỈ là cổng chẩn đoán, KHÔNG phải method; method hợp lệ = WP-Ensemble TỪ 0 (chỉ R50), chỉ chạy nếu cổng qua.
**Phép toán (`core.run_iterative_attack(weight_perturb` mode `"ens"`)`, `geo_osfd_attack(weight_ens=(K, spread))`):** mỗi bước, 1 forward lấy ∇ₓL + ∇θL (ĝ_θ chuẩn hóa từng tensor như WP). Tạo K bản: θ'_k = θ0 − ρ‖θ0‖·normalize(ĝ_θ + spread·ξ_k), ξ_k Gauss riêng từng tensor (RNG CUDA seed + 17, mới mỗi bước), k = 1..K; **ĝ_θ chung (từ bước hiện tại), K hướng tỏa quanh nó**. Tính ∇ₓL qua từng θ'_k (feature sạch co-transform cũng trên θ'_k), **trung bình K gradient** → cập nhật δ. θ khôi phục θ0 sau mỗi ảnh. K = 3, spread = 0.5 (cùng cỡ ĝ sau chuẩn hóa) đặt trước.
**Chi phí:** K lần forward+backward / bước (feature sạch cache 1 lần/bản) → cùng thời gian WP2 B36 cần **B ≈ 36/K = 12** (đo thực trên ảnh 1–4, đặt B theo đó). Rủi ro: ít bước hơn.
**Kiểm:** K = 1, spread = 0 trùng WP2 (sai số số học); θ khôi phục (torch.equal); K bản khác nhau (‖θ'_1 − θ'_2‖ > 0). Sức khỏe (ảnh 1–4, `tx_calib`): R50 xóa ≥ 0.8 × WP2, bão hòa ≥ 0.95 × WP2.
**(1) CỔNG oracle-retention (dev30, cùng ảnh/seed E20/E21):** `C2-OCT-1V-WP2-ENS-IS` = WP-Ensemble, δ init = δ_SwinT, B = B_ens (cùng thời gian), snapshot B1/B4/B12. So WP2-IS (39.8) / OTSV-IS (44) đã có.
- **QUA** nếu Swin-L B_end ≥ 52 VÀ ≥ WP2-IS + 10 → (2) chạy từ 0.
- **XÁM** 48–52 VÀ quỹ đạo B cuối ≥ giữa → cân nhắc (2).
- **DỪNG** nếu tuột về ~40 như WP2/OTSV.
**(2) TỪ 0 (chỉ chạy nếu cổng (1) QUA; khóa entry riêng):** `C2-OCT-1V-WP2-ENS` từ 0, dev30, so WP2 B36 cùng thời gian. BƯỚC NHẢY ΔSwin-L ≥ +10 & ΔCA3 ≥ −3; CÓ TÍN HIỆU +3..+10 → quét K ∈ {2,5}, spread.
**Caveat:** dev30; K/spread/ρ đặt trước; GPU không tất định; cổng (1) dùng δ ngoài threat model (chỉ chẩn đoán).
**Chạy:** `results/e25/run.sh`, log `results/e25/run.log`.

### E25 đính chính B + sức khỏe (TRƯỚC khi có số transfer): ENS ~4× / bước (1 fwd ∇θ + K=3 bản), B12 = 5.28 s/ảnh → cùng thời gian WP2 B36 (3.83) là **B9**. Bão hòa B12 70% < 0.95×WP2 (76%), R50 xóa 89% ≥ 0.8×WP2 — trung bình K gradient làm δ LOÃNG (bất lợi cho từ-0; cổng retention init δ_SwinT đã bão hòa nên vẫn chạy). Cổng retention chụp CẢ B9 (cùng thời gian) và B36 (cùng số bước, 4× thời gian, mô tả) trong 1 lần (snapshot B9).

### Kết quả E25 cổng retention @ dev30 (V100; `results/runs/dev300_B{9,36}_eps5_limit100/*_e25_dev30.json`, log `results/e25/run.log`) — theo quy tắc: **DỪNG** (không chạy từ 0)
ENS-IS (init δ_SwinT, Swin-L gốc 59): **B9 (cùng thời gian WP2) Swin-L 46.0** (R101 91.1, YOLOX 79.1); **B36 (cùng số bước, 4× thời gian) 44.8** (R101 96.5, YOLOX 85.5). So cùng init: WP2-IS 39.8, OTSV-IS 44.0.
**Đọc:** ENS-IS 45–46 < 52 → DỪNG. Phòng thủ K bản giữ được thêm ~+5 so WP2-IS nhưng vẫn tuột 59 → 45; B9 (46.0) > B36 (44.8) → càng nhiều bước R50 càng xói mòn vùng chung (khớp E20/E24). WP-Ensemble KHÔNG định hình lại landscape đủ để giữ vùng tốt → không chạy từ 0 (kỳ vọng từ 0 còn thấp hơn: δ loãng, bão hòa 70%). Lưu ý: pair_screen lỗi (dets WP2 không cùng file) — không cần, quyết định theo tuyệt đối. Literature (search 10-24): khởi tạo δ từ surrogate khác KHÔNG có tiền lệ như method (vô nghĩa nếu đã có surrogate kia) — ở đây chỉ là PROBE chẩn đoán (planted-solution / basin), không phải method.
**Trạng thái:** E19–E25 đều ÂM. WP-Ensemble là biến thể cuối của họ phòng thủ. Mọi hướng 1-R50 đã cạn. Ứng viên SOTA vẫn là WP2 (Swin-L ~36–37 dev100, +11 so OSFD). User quyết: (1) surrogate thứ 2 (đổi câu hỏi) | (2) định vị SOTA + cơ chế + trần có bằng chứng (dev300 + baseline AWT/BMAT/RaPA/LPAA).

---

## 2026-10-24 — E26: λ-schedule cho L_cls (curriculum theo bước) @ dev30 B36 (user duyệt, ~19:30 GMT+7) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lý do:** L_cls là đòn bẩy DUY NHẤT từng tăng Swin-L (cls-only 32.5 vs OSFD 25.3; C2 thêm cls = bước nhảy đầu). Mọi DẠNG của L_cls đã thử (λ 2/4/8, margin CW, roi_pos, vanish, rpn) — chưa thử bậc tự do THEO BƯỚC. E25: thành phần chung hai họ xói mòn dần theo bước; literature (search 10-24): scheduled step-size thấy I-FGSM transfer giảm khi bước tăng → nửa sau có thể là lúc dồn đòn bẩy Swin-L. Novelty: lịch TRỌNG SỐ LOSS theo bước chưa có tiền lệ trong transfer attack (search: chỉ có scheduled step-size, ILA tĩnh, λ cố định).
**Thiết kế (tách hình dạng lịch khỏi mức λ TB — cả 3 ô λ_TB = 4 = WP2):** attacker + defender dùng cùng L = L_OSFD + λ(t)·L_cls, λ(t) theo step_idx t ∈ [0, B−1]:
- `WP2-LUP`: λ tuyến tính 2 → 6 (TB 4) — dồn cls về cuối.
- `WP2-LSTEP`: λ = 0 nửa đầu, 8 nửa sau (TB 4) — tách pha rõ.
- `WP2-LDOWN`: λ tuyến tính 6 → 2 (TB 4) — ĐỐI CHỨNG chiều (dồn cls về đầu).
Chi phí = WP2 (chỉ đổi hệ số vô hướng, không thêm forward) → B36 cùng thời gian, không đo lại.
**Ô (`geo_osfd_attack(cls_sched=)`, `run_iterative_attack` truyền step_idx → make_view_loss(cls_weight=cls_weight·sched(t))):** gốc WP2 (ρ = 0.002, OCT 1V, k = 3). `cls_sched` ∈ {"up", "step", "down"}.
**Kiểm:** sched ≡ 1 (λ hằng 4) trùng WP2 (sai số số học); λ(t) đúng công thức mỗi bước (in 3 mốc); λ_TB = 4 cả 3.
**Đọc (so WP2 B36 cùng 30 ảnh):** BƯỚC NHẢY ΔSwin-L ≥ +10 & ΔCA3 ≥ −3; CÓ TÍN HIỆU +3 ≤ ΔSwin-L < +10 & ΔCA3 ≥ −3 → dev100 (khóa entry). Mô tả: LUP vs LDOWN (chiều có quan trọng không — nếu LUP > LDOWN rõ thì "dồn cls về cuối" là cơ chế thật); R101, YOLOX.
**Caveat:** dev30; 3 ô so sánh bội; λ_TB giữ 4 nên nếu λ cao hơn mới giúp thì E26 không bắt được (đó là câu hỏi khác — mức, đã thử); GPU không tất định.
**Chạy:** `results/e26/run.sh`, log `results/e26/run.log`.

### Kết quả E26 @ dev30 (V100; `results/ablation/e26_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/*_e26_dev30.json`, log `results/e26/run.log`) — theo quy tắc: **KHÔNG**
R50/R101/ConvNeXt/Swin-T/YOLOX/Swin-L | CA3: WP2 98.8/91.4/82.7/82.6/77.1/35.5 | 80.8; LUP 98.9/91.2/81.1/81.8/76.3/36.8 | 79.7; LSTEP 98.4/91.2/82.7/81.2/72.8/29.9 | 78.9; LDOWN 98.6/91.0/85.0/83.1/77.3/36.6 | 81.8.
Δ so WP2 (Swin-L | CA3): LUP +1.3 [−2.7, +3.9] | −1.1; LDOWN +1.1 [−1.7, +4.5] | +1.0; LSTEP −5.6 [−11.2, −0.4] | −1.9. **LUP − LDOWN: Swin-L −0.1 [−2.9, +4.5]** (CHIỀU KHÔNG QUAN TRỌNG).
**Đọc:** không lịch nào ΔSwin-L ≥ +3 → KHÔNG. LUP ≈ LDOWN → giả thuyết "dồn cls về cuối theo xói mòn E25" BÁC: hình dạng lịch không quan trọng, chỉ λ_TB ≈ 4. LSTEP hại (λ = 0 nửa đầu thiếu đòn bẩy Swin-L, không bù kịp). → λ phân bổ ĐỀU (WP2) là tốt nhất; bậc tự do theo-bước của L_cls đã đóng.
**Trạng thái:** E19–E26 đều ÂM/KHÔNG. L_cls (đòn bẩy Swin-L) cạn mọi bậc tự do (mức λ, dạng, lịch theo bước). WP2 vẫn là tốt nhất (Swin-L ~36–37 dev100). Mọi hướng 1-R50 ở ε = 5 đã thử. User quyết định vị: (1) SOTA + cơ chế + trần (dev300 + AWT-OD/RaPA/BSR) | (2) surrogate thứ 2.

---

## 2026-10-24 — E27: SPIKE-WP2 — quy tắc cập nhật δ kiểu phát-xung (LIF event-driven) @ dev30 B36 (user: "Thiết kế và run thử đi", ~20:30 GMT+7) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Ý tưởng (user):** áp nguyên lý SNN vào ĐỘNG LỰC cập nhật δ (không phải forward surrogate): cập nhật rời rạc, có ngưỡng, thưa; thời điểm phát xung mang thông tin. Khác momentum (dày, mọi pixel mỗi bước), khác consensus/VC (đồng thuận giữa view — đã thử NO-GO), khác λ-schedule (E26).
**Cơ chế (`core.run_iterative_attack(spike=θ_fire)`):** thay cập nhật MI thường bằng LIF trên δ:
- điện thế màng V (per-pixel, per-channel) tích lũy gradient chuẩn hóa: V ← μ·V + g/mean|g| (như momentum, μ = 1).
- phát xung tại pixel |V| ≥ θ_fire: δ ← clamp(δ + α·sign(V), ±ε); reset V ← 0 tại pixel đã phát. Pixel dưới ngưỡng GIỮ V (tích lũy tiếp), δ không đổi.
- θ_fire = 0 trùng MI thường (mọi pixel phát mỗi bước, V luôn reset) — nhưng reset làm mất momentum → KHÔNG trùng WP2; nên WP2 là mốc riêng, không phải giới hạn θ→0.
- WP (nhiễu trọng số ρ = 0.002) giữ nguyên, dùng cùng gradient.
**Chọn θ_fire CHỈ theo sức khỏe white-box (ảnh 1–4 dev300, `tx_calib`, KHÔNG nhìn target):** θ_fire ∈ {1.0, 2.0} (so độ lớn V ~ O(1) sau chuẩn hóa mean|g|); chọn θ lớn nhất còn: R50 xóa ≥ 0.8 × WP2 VÀ bão hòa ≥ 0.7 × WP2 (thưa hơn nên nới ngưỡng bão hòa; mô tả tỉ lệ pixel |δ|=5). Không θ nào đạt → ghi, chọn θ nhỏ nhất đã thử, đánh dấu caveat.
**Chi phí:** = WP2 (chỉ đổi phép cập nhật vô hướng, không thêm forward) → B36 cùng thời gian; đo lại s/ảnh để chắc.
**Kiểm:** `scripts/spike_unit_test.py` — θ_fire = 0 cho mọi pixel phát mỗi bước (δ đổi mọi bước); θ cao → thưa hơn (đếm % pixel phát/bước giảm); V reset đúng; L∞ ≤ ε; θ=0 không WP so MI thường (chỉ khác reset momentum — ghi).
**Đọc (so WP2 B36 cùng 30 ảnh):** BƯỚC NHẢY ΔSwin-L ≥ +10 & ΔCA3 ≥ −3 → dev100; CÓ TÍN HIỆU +3 ≤ ΔSwin-L < +10 & ΔCA3 ≥ −3 → dev100 (khóa entry); KHÔNG nếu mọi θ ΔSwin-L < +3. Mô tả: tỉ lệ pixel phát/bước, R101/YOLOX, δ thưa đến đâu.
**Caveat (khai báo trước):** đổi quy tắc cập nhật, không đổi hướng gradient (E22: RAND vô dụng, WP giúp; E24: nút thắt là hướng) → kỳ vọng thấp; δ thưa có tiền lệ âm (PD hại); dev30, n=30, GPU không tất định. Giá trị chính = thành phần novelty / ablation.
**Chạy:** `results/e27/run.sh`, log `results/e27/run.log`.

### Kết quả E27 @ dev30 (V100; `results/ablation/e27_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/*_e27_dev30.json`, log `results/e27/run.log`) — theo quy tắc: **KHÔNG**
Sức khỏe (ảnh 1–4): SPK1/SPK2 R50 xóa 92% (= WP2) nhưng bão hòa 30%/21% (δ thưa — < 0.7×WP2; caveat khai báo). Runtime SPK1 4.23 / SPK2 3.98 (~WP2 3.83).
R50/R101/ConvNeXt/Swin-T/YOLOX/Swin-L | CA3: WP2 98.8/91.4/82.7/82.6/77.1/35.5 | 80.8; SPK1 98.8/92.7/84.9/84.7/76.1/31.9 | 81.9; SPK2 98.1/90.8/85.2/81.6/65.7/37.9 | 77.5.
Δ so WP2: SPK1 Swin-L −3.6 [−9.8, +3.0], CA3 +1.1; SPK2 Swin-L +2.4 [−3.8, +8.4], CA3 −3.3, **YOLOX −11.4** [−16.7, 0.0].
**Đọc:** không ô nào (ΔSwin-L ≥ +3 VÀ ΔCA3 ≥ −3) → KHÔNG. SPK1 ≠ SPK2 ngược chiều Swin-L (không cơ chế ổn định); SPK2 thưa hơn đổi chút Swin-L lấy YOLOX −11.4 (δ thưa hại CNN, khớp tiền lệ PD). Khớp caveat: đổi quy tắc cập nhật không chạm nút thắt hướng (E22/E24). Event-driven LIF đóng như thành phần ablation/novelty âm.
**Trạng thái:** E19–E27 đều ÂM/KHÔNG. WP2 vẫn tốt nhất. Quy tắc cập nhật (SPIKE), lịch (E26), mục tiêu (E24), phòng thủ-ensemble (E25), tách mục tiêu (E23), vị trí WP (E22), OCT, feature loss — tất cả không vượt WP2. User quyết định vị (dev300 + baseline) hoặc nới threat model.

---

## 2026-10-24 — E28: SPIKE-WP — cơ chế phát-xung áp vào NHIỄU TRỌNG SỐ (phía gradient) @ dev30 B36 (user chọn "1", ~21:00 GMT+7) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy; ĐÒN CUỐI của tối ưu WP2
**Lý do:** E27 đặt spiking ở OUTPUT (cập nhật δ) = sai phía (nhóm budget — E22 RAND vô dụng, E24 nút thắt là hướng). Đặt spiking vào WP (nhóm TRƯỜNG GRADIENT — chỗ DUY NHẤT từng đẩy Swin-L). Chưa thử trong họ WP (WPA/scope/ensemble đều plateau ~36–45). Kỳ vọng thấp (cùng họ đã chạm trần); chạy vì khác cơ chế + đặt đúng phía. Dù kết quả gì → DỪNG tối ưu WP2 sau E28.
**Cơ chế (`core.run_iterative_attack(weight_perturb` mode `("wspike", θ_fire, [])`)`):** mỗi tensor conv backbone có điện thế M (per-element). Mỗi bước: M ← M + ∇θ/mean|∇θ| (chuẩn hóa per-tensor); phát xung tại |M| ≥ θ_fire; θ'_i = θ0_i − ρ‖θ0_i‖·d_i/‖d_i‖ với d = sign(M)·fire (giữ ‖θ'−θ0‖ = ρ‖θ0‖ như WP, bất kể bao nhiêu element phát); reset M = 0 tại element phát; element dưới ngưỡng giữ M (không nhiễu). ρ = 0.002. Phòng thủ THƯA + theo thời điểm: chỉ filter "chịu áp lực nhất" phản ứng.
**Chọn θ_fire (sức khỏe white-box, ảnh 1–4 dev300): θ_fire ∈ {1, 2};** chọn θ lớn nhất còn R50 xóa ≥ 0.8 × WP2 VÀ bão hòa ≥ 0.95 × WP2 (WP làm δ dày như WP2, không thưa như E27 — ngưỡng chặt lại). Không đạt → θ nhỏ nhất, caveat.
**Chi phí:** = WP2 (chỉ đổi quy tắc nhiễu θ, không thêm forward) → B36; đo lại s/ảnh.
**Kiểm:** `scripts/wspike_unit_test.py` — θ khôi phục; θ_fire cao → ít element phát hơn; ‖θ'−θ0‖ ≈ ρ‖θ0‖ khi có phát; θ_fire = 0 (mọi element phát mỗi bước) ≈ WP2 (sign thay unit-direction — ghi khác).
**Đọc (so WP2 B36 cùng 30 ảnh):** BƯỚC NHẢY ΔSwin-L ≥ +10 & ΔCA3 ≥ −3 → dev100; CÓ TÍN HIỆU +3..+10 & ΔCA3 ≥ −3 → dev100 (khóa entry); KHÔNG nếu mọi θ ΔSwin-L < +3. Mô tả: % element phát, R101/YOLOX.
**Caveat:** cùng họ WP đã plateau; dev30; GPU không tất định. ĐÒN CUỐI — sau E28 dừng tối ưu bất kể kết quả.
**Chạy:** `results/e28/run.sh`, log `results/e28/run.log`.

### Kết quả E28 @ dev30 (V100; `results/ablation/e28_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/*_e28_dev30.json`, log `results/e28/run.log`) — theo quy tắc: **KHÔNG** (đòn cuối tối ưu WP2)
Sức khỏe (ảnh 1–4): WSPK1/2 R50 xóa 92%, bão hòa 80% (= WP2, δ dày — khác E27). Runtime 4.6 (1.2×, chạy B36 = ưu ái).
R50/R101/ConvNeXt/Swin-T/YOLOX/Swin-L | CA3: WP2 98.8/91.4/82.7/82.6/77.1/35.5 | 80.8; WSPK1 98.5/92.5/81.5/80.1/76.3/35.9 | 79.3; WSPK2 98.8/92.4/81.3/81.8/77.7/35.8 | 80.3.
Δ so WP2: WSPK1 Swin-L +0.4 [−2.4, +3.0], CA3 −1.5; WSPK2 Swin-L +0.3 [−2.0, +2.7], CA3 −0.5.
**Đọc:** cả 2 ≈ WP2 (ΔSwin-L ≈ 0) → KHÔNG. Phát xung thưa trên trọng số hội tụ về WP2 → trong họ WP, "element nào / khi nào nhiễu" KHÔNG quan trọng, chỉ hướng phòng thủ (củng cố E22). Theo cam kết: **DỪNG tối ưu WP2**.
**KẾT CHUỖI E19–E28 (10 thí nghiệm, tất cả ÂM/KHÔNG):** mục tiêu (E24), khởi tạo (E20), forward (OTSV/PFA), phòng thủ vị trí (E22) / ensemble (E25) / tách mục tiêu (E23) / spiking trọng số (E28), lịch λ (E26), spiking δ (E27), OCT, feature loss — không gì vượt WP2 (Swin-L ~36–37 dev100). Trần transfer 1-R50 ở ε = 5 có bằng chứng. **WP2 = method chốt.** Bước tiếp: định vị SOTA + cơ chế (dev300 + baseline AWT-OD/RaPA/BSR + held-out) HOẶC nới threat model 2-surrogate. Chưa push E20–E28.

---

## 2026-10-24 — E29: FPA-OCT-WP2 — hoán vị vị trí feature (Feature Permutation Attack, port từ classification) @ dev30 B36 (user: "Duyệt luôn", ~21:45 GMT+7) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lý do (user: lấy ý SOTA classification CNN→ViT sang OD):** FPA (arXiv 2503.20310, 2025): chèn hoán vị vị trí không gian feature vào forward CNN → nơ-ron dùng thông tin NGOÀI receptive field → R50 hành xử "toàn cục" giống ViT; training-free, 1 surrogate; R50→ViT/Swin MI 6.8/7.9 → +FPA-N 28.0/28.7 (+~20, ε=8). Khớp nút thắt đã xác nhận: E7 (δ_R50 cục bộ tần cao) + E6 (Swin đọc object toàn cục/shape). KHÁC OTSV/PFA (đã âm): OTSV/PFA = attention TĨNH (head hấp thụ); FPA = hoán vị NGẪU NHIÊN mỗi bước = ensemble ngầm R50 "đổi dây" — đa dạng feature-space (OCT là đa dạng input-space). Trục đa dạng MỚI, chưa thử trong repo.
**Cơ chế (`repair.make_view_loss(fpa=)`, `pia._fpa_permute`):** FPA-N (giữ cấu trúc cục bộ): tại đầu ra backbone stage (chọn), với mỗi kênh được chọn (tỉ lệ γ), xác suất p mỗi bước: mỗi vị trí (h,w) đổi với 1 trong 4 láng giềng (chọn ngẫu nhiên, RNG riêng seed + 19). Áp lên **F_adv của nhánh L_OSFD** (loss feature không cần GT — hoán vị tự do); feature sạch k·F_cln hoán vị CÙNG mask (để MSE so đúng cặp). Nhánh **L_cls KHÔNG hoán vị** (RoIAlign tại box GT cần tương ứng không gian — FPA-N jitter 1 px vẫn có thể lệch; khóa an toàn: cls nguyên). Stage áp: thử **s2 (block 2, như paper FPA-N) và s34** → 2 ô.
**Tham số (theo paper, KHÔNG tune):** FPA-N γ = 0.6, p = 0.5. Hoán vị differentiable (memory op) → gradient chảy qua.
**Kiểm:** `scripts/fpa_unit_test.py` — p = 0 trùng WP2 (không hoán vị); hoán vị giữ tổng năng lượng (chỉ đổi vị trí); mask hoán vị F_adv và F_cln TRÙNG nhau (MSE so đúng cặp); γ, p đúng tỉ lệ; L∞ = 5; θ backbone khôi phục (WP vẫn chạy). Sức khỏe (ảnh 1–4): R50 xóa ≥ 0.8 × WP2, bão hòa ≥ 0.95 × WP2.
**Ô:** `C2-OCT-1V-WP2-FPA2` (stage 2), `C2-OCT-1V-WP2-FPA34` (stage 3+4). Gốc WP2 (ρ=0.002, OCT 1V, L_cls λ=4). Chi phí thêm = memory op → đo s/ảnh, B36 cùng thời gian (hoặc giảm B theo đo).
**Đọc (so WP2 B36 cùng 30 ảnh):** BƯỚC NHẢY ΔSwin-L ≥ +10 & ΔCA3 ≥ −3 → dev100; CÓ TÍN HIỆU +3 ≤ ΔSwin-L < +10 & ΔCA3 ≥ −3 → dev100 (khóa entry); KHÔNG nếu mọi ô ΔSwin-L < +3. Mô tả: ΔSwin-T, R101, YOLOX.
**Caveat:** cùng tinh thần "làm R50 giống ViT" (OTSV/PFA âm) nhưng khác cơ chế (permutation-diversity); lợi classification +20 có thể không giữ sang detection (như Season mất lợi); γ/p/stage theo paper không tune; dev30.
**Chạy:** `results/e29/run.sh`, log `results/e29/run.log`.

### Kết quả E29 @ dev30 (V100; `results/ablation/e29_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/*_e29_dev30.json`, log `results/e29/run.log`) — theo quy tắc: **KHÔNG**
Sức khỏe (ảnh 1–4): FPA2/34 R50 xóa 92–95%, bão hòa 79% (= WP2). Runtime 4.0/3.9 (~WP2 3.83).
R50/R101/ConvNeXt/Swin-T/YOLOX/Swin-L | CA3: WP2 98.8/91.4/82.7/82.6/77.1/35.5 | 80.8; FPA2 .../92.0/.../.../77.8/34.0; FPA34 .../92.6/.../.../73.4/37.8.
Δ so WP2: FPA2 Swin-L +0.7 [−1.7, 3.9] (YOLOX −1.5); **FPA34 Swin-L +2.3 [−3.3, 6.4]** (YOLOX **−3.7** [−7.4, −1.0]).
**Đọc:** không ô nào ΔSwin-L ≥ +3 → KHÔNG. FPA2 (đúng stage paper) không giúp; FPA34 nhích Swin-L +2.3 nhưng đổi YOLOX −3.7 (CI < 0) → lại trượt đường đánh đổi CNN↔Swin-L. Lợi classification FPA (+20 CNN→ViT) KHÔNG giữ sang detection — giống Season (frequency), OTSV/PFA (attention): kỹ thuật "làm R50 giống ViT" ở classification mất lợi ở OD vì gap OD là HƯỚNG trực giao (E14/E24), không phải locality/phổ đơn thuần. Củng cố kết luận cơ chế.

---

## 2026-10-24 — E30: IS-WP2 — Individuality Suppression (port COGO, gradient-space) @ dev30 B36 (user: "Duyệt luôn", ~22:45 GMT+7) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Lý do (user: tìm ý gradient-space):** COGO (arXiv 2506.06992, 2025) IS — single-surrogate, backward, training-free: tại tầng backbone, xét cặp kênh gradient (i,j), tính MI + Pearson corr; cặp tương quan cao (vượt ngưỡng thích nghi) = dư thừa = **đặc thù surrogate (overfit)** → giảm trọng số kênh đó; kênh độc lập = tổng quát → giữ. Nhắm thẳng cơ chế E20 (hướng đặc thù R50 xói mòn vùng chung). ĐÚNG không gian gradient (forward-space PFA/OTSV/FPA đã đóng); KHÁC WP (trọng số), KHÁC FIA (FIA khuếch đại kênh quan trọng; IS TRIỆT kênh dư thừa — ngược). Bỏ CE của COGO (= frequency DCT, ta đã thử Season → hại).
**Cơ chế (`repair.is_backward_hook`, `make_view_loss(is_supp=)`):** hook backward trên đầu ra backbone layer (chọn). Khi có grad_output g [1,C,H,W]: rút N=10 cặp kênh (i,j) ngẫu nhiên (RNG seed+21); tính |Pearson corr| giữa g_i, g_j (phẳng hóa spatial) [bỏ MI — ước lượng MI tốn/nhiễu ở C,H,W nhỏ, dùng |corr| như proxy chính, khai báo lệch paper]; τ = β·mean(|corr| các cặp), β = 1.0; cặp |corr| ≥ τ → cả i, j +1 chỉ số dư thừa; w_c = clamp(1 − α·(số lần c dư thừa), min=0), α = 0.3; nhân grad_output kênh c với w_c. Áp trên gradient (không đổi forward). θ/WP giữ nguyên.
**Stage áp:** thử **s3 (stride 16, nút thắt E14)** và **s34** → 2 ô. Gốc WP2 (ρ=0.002, OCT 1V, cls λ=4).
**Kiểm:** `scripts/is_unit_test.py` — α=0 (hoặc N=0) ≡ WP2 (không reweight); w_c ∈ [0,1]; kênh tương quan cao bị giảm (dựng cặp giả corr=1 → w giảm); hook gỡ sau mỗi lần; L∞=5; θ khôi phục. Sức khỏe (ảnh 1–4): R50 xóa ≥ 0.8×WP2, bão hòa ≥ 0.95×WP2.
**Ô:** `C2-OCT-1V-WP2-IS3` (layer3), `C2-OCT-1V-WP2-IS34` (layer3+4). Chi phí thêm = N corr/bước (rẻ) → B36, đo s/ảnh.
**Đọc (so WP2 B36 cùng 30 ảnh):** BƯỚC NHẢY ΔSwin-L ≥ +10 & ΔCA3 ≥ −3 → dev100; CÓ TÍN HIỆU +3 ≤ ΔSwin-L < +10 & ΔCA3 ≥ −3 → dev100 (khóa entry); KHÔNG nếu mọi ô ΔSwin-L < +3. Mô tả: ΔSwin-T, YOLOX, R101; tỉ lệ kênh bị triệt TB.
**Caveat:** COGO验证 ViT→ViT (giả định "corr cao = đặc thù" cho CNN→Transformer OD chưa kiểm); gần họ FIA (ngang) nhưng logic ngược; bỏ MI (chỉ corr); β/α/N/stage theo paper + chọn trước; dev30.
**Chạy:** `results/e30/run.sh`, log `results/e30/run.log`.

### Kết quả E30 @ dev30 (V100; `results/ablation/e30_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/*_e30_dev30.json`, log `results/e30/run.log`) — theo quy tắc: **KHÔNG**
Sức khỏe (ảnh 1–4): IS3/34 R50 xóa 92%, bão hòa 79–80% (= WP2). Runtime 3.86/3.75 (= WP2).
R50/R101/ConvNeXt/Swin-T/YOLOX/Swin-L | CA3: WP2 98.8/91.4/82.7/82.6/77.1/35.5 | 80.8; IS3 98.8/92.1/84.4/83.5/78.1/35.4 | 82.0; IS34 98.7/92.5/84.6/82.4/77.9/35.8 | 81.6.
Δ so WP2: IS3 Swin-L −0.0 [−2.5, 2.1], CA3 +1.2 [−0.3, 2.4] (ConvNeXt +1.7, YOLOX +1.0); IS34 Swin-L +0.3, CA3 +0.9.
**Đọc:** Swin-L không đổi → KHÔNG. KHÁC các đòn trước: IS cho cải thiện CNN đồng đều RẤT NHỎ (CA3 +1, mọi target dương nhẹ), KHÔNG trượt đánh đổi — triệt kênh gradient dư thừa làm δ bớt đặc thù R50 một chút, lợi đều trên CNN nhưng không chạm Swin-L. Giả thuyết "kênh gradient corr cao = đặc thù R50, triệt đi lộ hướng chung" KHÔNG đúng cho Swin-L (khớp E24: hướng phá Swin-L không nằm trong không gian KÊNH gradient R50 mà thao tác kênh chạm tới). Lợi COGO classification không giữ sang OD.
**Trạng thái:** gradient-space (E30 IS, cùng WP-family E22/E25/E28, SGM/BPA/ILPD/Season) đều không chạm Swin-L; forward-space (OTSV/PFA/FPA) đóng; toolbox classification port hết (chỉ weight-tuning=WP2 giữ lợi). E19–E30 (12 thí nghiệm) đều âm cho Swin-L ≥ 50. WP2 chốt.

---

## 2026-10-24 — TẠM DỪNG (user, ~23:30 GMT+7, trả máy ngủ): tổng kết phiên + khóa việc mở AWT-OD
**Máy:** V100 16GB. Không tiến trình chạy. Commit E20–E30 user ĐÃ push tay. Commit cuối này (TẠM DỪNG + AWT-OD + CLAUDE.md) **cần push** trước khi trả máy.
**Phiên này (E20–E30, 11 thí nghiệm, TẤT CẢ ÂM/KHÔNG cho Swin-L ≥ 50):** E20 basin (RỜI VÙNG), E21 OTSV (retention 44 < 52), E22 WP hướng/vị trí (phòng thủ là lý do, không đặc thù C4, head không thêm), E23 tách mục tiêu DEF-CLS/FEAT (KHÔNG), E24 dò thước đo chỉ-R50 (ÂM — không φ nào bắt pha xói mòn), E25 WP-ensemble (retention 45), E26 λ-schedule (KHÔNG, chiều không quan trọng), E27 spike δ (KHÔNG), E28 spike trọng số (≈ WP2), E29 FPA port (KHÔNG, trượt đánh đổi), E30 IS/COGO port (KHÔNG, lợi CNN +1 đều nhưng Swin-L ~0). Port 8+ họ classification: chỉ weight-tuning (= WP2) giữ lợi sang OD.
**Cơ chế đã khóa (viết cho paper, chi tiết ở các entry + câu chốt):** gap CNN→Swin-L KHÔNG do Swin-L bền (WB 85%), mà do **hướng đối kháng hai họ gần trực giao** (E16 cos 0.06); gap hai chiều; nút thắt = HƯỚNG distortion token object stage 3 (E14); tồn tại δ chung (δ_SwinT phá R50 78.7 / Swin-L 59) nhưng **trường gradient R50 xói mòn nó** (E20: 59→35 sau 12 bước) và không thước đo chỉ-R50 nào trỏ tới (E24). WP2 (nhiễu trọng số phòng thủ) = cách thu hẹp tốt nhất không cần target (+11 so OSFD, Swin-L ~36–37 dev100). Trần single-R50 ε=5: ~37.
**Không gian còn lại (user nhận định): WEIGHT-SPACE là duy nhất từng đẩy Swin-L.** Nhánh "phòng thủ 1 bước" (WP2/WPA/ensemble/scope/spike) đã cạn ~37. Thứ chưa vắt = **AWT đầy đủ** (SAM 2 bước, tích lũy) + là baseline novelty bắt buộc.

### VIỆC MỞ — E31: AWT-OD (thay khối WP bằng AWT đầy đủ), chạy phiên sau — THIẾT KẾ KHÓA Ở ĐÂY
**Ô `C2-OCT-1V-AWT`:** giữ OCT + C2 (L_OSFD + 4·L_cls) + OCT 1V y nguyên; CHỈ thay khối WP (`core.run_iterative_attack` weight_perturb) bằng AWT (arXiv 2408.09469):
- Mỗi bước, 2 gradient θ: (1) θ̂ = θ₀ + β·ĝθ (ascend, β = 0.005·‖θ₀‖ per-tensor như WP scale); (2) θ ← θ_prev − lr·∇θL(θ̂) (descend tại θ̂, lr = 0.002·‖θ₀‖). θ **tích lũy** qua T bước (KHÁC WP2 reset mỗi bước), khôi phục θ₀ cuối ảnh (torch.equal kiểm).
- Dùng cùng ĝθ lần backward input (ascend không thêm forward); descend cần +1 forward+backward tại θ̂ → ~2× ∇θ/bước.
**Code:** thêm mode `("awt", beta, lr, [mem_theta])` vào nhánh weight_perturb của core (gần wspike/ens); `geo_osfd_attack(weight_awt=(β,lr))`. Unit test: β=0,lr=0 ≈ θ không đổi; θ khôi phục; β/lr đúng thang ‖θ₀‖.
**Sức khỏe (ảnh 1–4):** R50 xóa ≥ 0.8×WP2, bão hòa ≥ 0.95×WP2. Thời gian: ~2× ∇θ → đo s/ảnh, cùng thời gian WP2 B36 cần B thấp hơn (ước B≈24–28); chạy CẢ B cùng-thời-gian VÀ B36 (cùng bước, mô tả) như E25.
**Đọc (so WP2 B36 cùng 30 ảnh, dev30):** CẢI TIẾN nếu ΔSwin-L ≥ +3 & ΔCA3 ≥ −3 → dev100 (khóa entry). NOVELTY: nếu WP2 ≥ AWT-OD cùng thời gian → WP2 vững là "AWT rẻ native cho OD". Mô tả: ΔSwin-T, YOLOX, runtime.
**Caveat:** gần WPA (tích lũy, âm) + bước descend SAM mới; đắt 2× → B thấp bất lợi (cũng là điểm bán WP2 rẻ); β/lr theo paper (scale theo ‖θ₀‖ như WP), không tune. **Baseline novelty chặt:** sau component-swap, cân nhắc thêm bản AWT gốc (input-sampling PGN riêng) — để sau.
**Sau E31 (bất kể kết quả): DỪNG tối ưu, sang dev300.** Bộ baseline dev300: OSFD, MI, M-DI², C2-OCT-1V, WP2, **AWT-OD**, LPAA, BSR (feature-only), OPS, RaPA — port dần. Verify clean AP Generalization Panel (model_registry còn trống). Định vị: SOTA + cơ chế + trần có bằng chứng (chuỗi E19–E30).

---

## 2026-10-25 — Máy mới V100 16GB; E31 AWT-OD: code + unit test + cổng sức khỏe (theo thiết kế khóa entry "TẠM DỪNG 2026-10-24") — **TRƯỢT cổng sức khỏe, KHÔNG chạy dev30**
**Code (commit `fbe5b23`):** `core.run_iterative_attack` mode `("awt", β, lr)`; `geo_osfd_attack(weight_awt=(β, lr))`; ô `C2-OCT-1V-AWT` (β = 0.005, lr = 0.002, × ‖θ0‖ per-tensor, như thiết kế). Chi tiết thực thi (khai báo): bước descend tại θ̂ dùng lại CÙNG view của bước (ảnh detach), feature sạch L_OSFD giữ từ θ_t (không forward sạch lại tại θ̂). Unit test `scripts/awt_unit_test.py` QUA: β = lr = 0 trùng tuyệt đối không-WP; 1 bước ‖θ1 − θ0‖ = 0.002‖θ0‖; 8 bước 0.016 (tích lũy gần tuyến tính); R50 θ khôi phục `torch.equal`, L∞ = 5.
**Thời gian (5 ảnh đầu dev300, B36, log `results/e31/calib.log`):** WP2 4.8 s/ảnh, AWT 6.5 s/ảnh (1.35×) → B cùng thời gian ≈ 26–27.
**Sức khỏe (ảnh 1–4, `tx_calib --run dev300_B36_eps5_limit5`):** WP2 bão hòa 80%, xóa 92% (34/37); **AWT bão hòa 67% (< 0.95 × 80 = 76), xóa 68% (25/37; < 0.8 × 92 = 74)** → **TRƯỢT cả hai tiêu chí** → theo quy tắc khóa: ghi nhận, không chạy dev30. β / lr KHÔNG tune (caveat khóa).
**Đọc (mô tả):** θ tích lũy gần tuyến tính (unit test) → sau 36 bước ‖θ − θ0‖ ước ~0.07‖θ0‖, gấp ~14× WP5 (ρ = 0.005, đã sát ngưỡng) và vượt vùng dao động ρ = 0.01 (10-21) → surrogate trôi khỏi R50, δ tối ưu cho model khác. Phù hợp WPA (tích lũy, âm, 10-23). Thang lr = 0.002‖θ0‖/bước là cách dịch của thiết kế, không phải đúng nguyên văn paper (paper dùng lr tuyệt đối trên trọng số classification).
**Việc mở — user quyết:** (a) kết thúc E31, sang dev300 như đã khóa (AWT-OD làm baseline novelty cần cấu hình khỏe); (b) entry mới khóa trước: AWT với bán kính tích lũy bị chặn (chiếu ‖θ − θ0‖ ≤ ρ‖θ0‖, ρ = 0.002 như WP2) hoặc lr chia T — đây là biến thể, không còn là AWT nguyên bản.

---

## 2026-10-25 — E32: WP2 thắng nhờ TÍNH CHẤT gì của δ? So δ_WP2 B36 vs δ_C2 B41 trên dev100, forward-only (user duyệt "duyệt đo thử", ~11:20 GMT+7) — MÔ TẢ / sinh cơ chế, khóa TRƯỚC khi chạy
**Vì sao:** E22/E23 chỉ là can thiệp (hướng phòng thủ là lý do, chống toàn bộ loss, không đặc thù stage); CHƯA đo δ_WP2 khác δ_C2 ở tính chất nào, vì sao lợi rơi vào DINO-Swin-L chứ không Swin-T.
**Dữ liệu (HF, không attack mới):** δ_W = `dev300_B36_eps5_limit100` tag `_v100` / `C2-OCT-1V-WP2` (10-23); δ_C = `dev300_B41_eps5_limit100` (không tag) / `C2-OCT-1V` (10-18, V100); δ_S = `dev300_B41_eps5_limit100_sur-swin_t` tag `_v100`. Dets 6 model có sẵn (`dets_wpa_dev100` / `dets_c2_1v_dev100`, `gen_dets/`). **Caveat mốc:** δ_C là bản 10-18, không phải bản 10-22 của quy tắc GIÚP (chưa từng upload): DINO-Swin-L drop C 32.1 / W 36.4 → Δ **+4.3** (10-22: +6.2); suppression Swin-L 0.259 → 0.274 (+1.5 điểm).
**Đo (script `scripts/e32_wp2_dissect.py`, dev100, fp32, mọi δ = PNG − ảnh gốc; mỗi ảnh, X ∈ {C, W}):**
- **(a) Độ phẳng trong weight-space trên R50:** G(θ; X) = L(x + δ_X; θ) − L(x; θ), L = L_OSFD (k = 3, feature sạch tính tại CÙNG θ) + 4·(loss_rpn_cls + loss_cls) GT, 1 view không biến đổi. Giữ lại r_ρ(X) = TB_ξ G(θ0 + ρ‖θ0‖ξ̂) / G(θ0), ξ Gauss chuẩn hóa từng tensor conv backbone, ρ ∈ {0.002, 0.005, 0.01, 0.02}, 4 mẫu / ρ (CÙNG ξ cho C và W, RNG seed 2026 + img_id). r_def(X): G tại θ' = WP2 phòng thủ 1 bước (ρ = 0.002, ∇θ của L tại δ_X). Mô tả: G trên R101 cùng công thức (tỉ số W / C).
- **(b) D_k = ‖f_k(adv) − f_k(sạch)‖ / ‖f_k(sạch)‖** backbone stage 1–4 trên R50, R101, ConvNeXt-T, Swin-T, DINO-Swin-L (test pipeline, như `mech_a2_features` / `mech_swinl`). Chỉ số chuyển giao T_k(m) = D_k(m) / D_k(R50).
- **(c) DINO-Swin-L "sửa":** cosd = 1 − cos ở backbone 1–4, neck 5 mức, memory encoder; A_enc = cosd_enc / cosd_bb4 (như `mech_swinl_geo`).
- **(d) Tính chất δ + loại lỗi:** năng lượng 3 dải tần (thấp / trung / cao như A4); cos(δ_W, δ_C), cos(δ_X, δ_S); năng lượng trong box GT / tỉ lệ diện tích. Loại lỗi (CPU, dets có sẵn, như `err_profile`): recall@0.3 theo cỡ, conf_ratio, FP/ảnh trên Swin-T, YOLOX-S, DINO-Swin-L.
**Quy tắc đọc (paired bootstrap 1000 theo ảnh, BOOT_SEED 2026; Δ = W − C cùng ảnh):**
- **H_flat (WP2 = δ phẳng hơn trong weight-space) ỦNG HỘ** nếu ∃ ρ ∈ {0.005, 0.01, 0.02}: Δr_ρ ≥ 0.05 VÀ cận dưới CI > 0. **DỰ ĐOÁN TRANSFER** (ứng viên mục tiêu, entry riêng) nếu thêm Spearman theo ảnh (Δr_ρ*, Δsuppression Swin-L) > 0 CI loại 0 (ρ* = ρ có Δr lớn nhất). Ngược lại BÁC.
- **H_feat (WP2 đẩy distortion sang Swin-L nhiều hơn sang R50):** ∃ k: ΔT_k(Swin-L) ≥ 0.05 VÀ cận dưới CI > 0 → ỦNG HỘ; mô tả cùng chỉ số cho Swin-T (dự đoán: không tăng, khớp lợi chỉ ở DINO).
- **H_repair (δ_W sống sót qua encoder DINO tốt hơn):** ΔA_enc > 0 VÀ cận dưới CI > 0 VÀ Δcosd_enc ≥ 0.01 → ỦNG HỘ.
- (d) mô tả, không quy tắc. Đây là E mô tả: không đổi method, không quyết định GO; kết quả chỉ chọn hướng (entry riêng khóa trước).
**Caveat:** 30 / 100 ảnh là dev30 (đã thấy); 1 cặp δ (n = 100), mốc C khác ngày; tham số ρ / số mẫu đặt trước, không tune; GPU không tất định.
**Chạy:** `python scripts/e32_wp2_dissect.py` → `results/mechanism/e32_wp2_dissect.json`, log `results/e32/run.log`.

### Kết quả E32 @ dev100 (V100, chạy 11:07–11:25 GMT+7; `results/mechanism/e32_wp2_dissect.json`, log `results/e32/run.log`) — theo quy tắc: **H_flat BÁC, H_feat BÁC, H_repair BÁC**
- **(a)** r_ρ (nhiễu θ NGẪU NHIÊN) = 1.000 ở mọi ρ ≤ 0.02 cho CẢ C và W (Δ ≤ +0.001) → hướng ngẫu nhiên không chạm loss → **H_flat BÁC**; Spearman(Δr, Δsup) +0.005 [−0.17, 0.19]. Mô tả: **r_def (phòng thủ CÓ HƯỚNG 1 bước) C 0.765 → W 0.832, Δ +0.067 [+0.057, +0.077]**; G0 (gain white-box R50) W thấp hơn 13% (13.6 vs 15.6), G trên R101 −0.71 [−0.84, −0.57].
- **(b)** ΔT_k (W − C) stage 1–4: DINO-Swin-L −0.002 / −0.002 / +0.015 / **+0.031 [0.027, 0.036]** (< 0.05 → **H_feat BÁC**); cùng chiều nhỏ hơn: Swin-T +0.009 / +0.014, ConvNeXt +0.001 / +0.011, R101 +0.002 / +0.011 (stage 3 / 4). Mô tả: WP2 dời distortion về stage sâu, mạnh nhất ở Swin-L.
- **(c)** cosd DINO-Swin-L bb4 0.326 → 0.346, neck 2–4 tăng ~0.015, encoder 0.044 → 0.049 (Δ +0.005 < 0.01); A_enc Δ +0.006 [0.002, 0.010] → **H_repair BÁC** (encoder vẫn dập ~7×, như C).
- **(d)** cos(δ_W, δ_C) **0.50** [0.48, 0.52] (WP2 đổi hướng δ đáng kể dù cùng loss / view); cos(δ, δ_S) 0.064 → 0.068 (+0.003); năng lượng dải thấp +1.2 điểm, cao −0.7; box_ratio ~1.0 cả hai. Loại lỗi (recall@0.3 nhỏ / vừa / lớn; FP nền / ảnh): **DINO-Swin-L** C .368/.635/.856, W .343/.597/.839, conf_ratio .889 = .890, FP nền 2.65 → 3.12 → lợi Swin-L đến từ **MẤT object** (mọi cỡ), không từ hạ score object còn sống. YOLOX / Swin-T: W ít FP nền hơn (10.3 → 7.0 / 21.0 → 16.9) + recall thấp hơn → WP2 dời lỗi từ "tạo FP" sang "xóa object".
**Đọc:** không giả thuyết khóa nào đứng. Mô tả nhất quán: δ_WP2 (i) giữ gain trước R50 phòng thủ có hướng tốt hơn (+0.067, gần như theo định nghĩa của WP2), (ii) gain white-box THẤP hơn 13% nhưng transfer cao hơn (ít overfit R50), (iii) chỉ nửa trùng hướng δ_C2, dời distortion về stage 4 (mạnh nhất ở Swin-L), (iv) chuyển kiểu lỗi sang xóa object. KHÔNG thấy tiến về δ_SwinT (+0.003) hay vượt phép sửa encoder → khớp trần ~37 (E20/E24). Caveat: mốc C 10-18; r_def gần như vòng tròn (WP2 tối ưu đúng đại lượng đó) — chưa kiểm r_def có dự đoán transfer theo ảnh.

---

## 2026-10-25 — Chẩn đoán đổi surrogate (2): C2-OCT-1V với surrogate **DINO-R50** @ dev100 B41 (user duyệt "approve a", ~11:40 GMT+7) — khóa TRƯỚC khi chạy
**Câu hỏi:** gap R50 → DINO-Swin-L (~32%) có một phần do **khác kiểu detector** giữa surrogate (Mask R-CNN) và target (DINO) không? Bổ trợ 2 chẩn đoán cũ: target DINO-R50 bị R50 phá 97.3% (09-23, phía target); surrogate Swin-T Mask R-CNN → Swin-L 58.0 (10-21, cùng họ backbone, khác detector). Ở đây: **cùng detector DINO, backbone CNN (R50)**. Chỉ PHÂN TÍCH — threat model vẫn khóa surrogate Mask R-CNN R50.
**Ô `C2-OCT-1V-DETR` (surrogate `dino_r50` = `dino-4scale_r50_8xb2-12e_coco`):** C2-OCT-1V y nguyên (OCT 1 view, MI, AMP, ε = 5, B41), trừ: L_OSFD trên 3 stage backbone R50 của DINO (out_indices 1–3); L_cls = tổng 7 số hạng `*loss_cls` không-dn (enc + 6 decoder; `repair.cls_term("detr")`, train mode với BN / Dropout eval — dropout = 0, BN đóng băng); skip_mask tắt (không có mask head). **λ = 0.1 theo ĐÚNG quy tắc chọn λ = 4 cũ** (median 5 ảnh đầu dev300 của mean|∇ₓL_feat| / mean|∇ₓL_cls| tại δ = 0): R50 tái lập 3.78 ≈ 4; DINO-R50 0.10 (0.25 / 0.11 / 0.10 / 0.08 / 0.06). Khung ảnh DINO-R50 trùng R50 (kiểm dev10).
**Sức khỏe (ảnh 1–4 dev300, B41, đã đo):** L∞ 5, bão hòa 76%; DINO-R50 white-box xóa 33 / 36 = 92%. 12.1 s / ảnh (~3× R50; cùng số bước, KHÔNG cùng thời gian — như chẩn đoán Swin-T).
**Mốc:** sur R50 = `dev300_B41_eps5_limit100` / `C2-OCT-1V` (PNG + dets 10-18, HF); sur Swin-T = số 10-21 (58.0, không chạy lại).
**Chạy (log `results/sur_dino/run.log`):**
```bash
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --surrogate dino_r50 --methods C2-OCT-1V-DETR --out metrics_c2_1v_dev100.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100_sur-dino_r50 --subset dev100 --models yolox_s dino_swin_l dino_r50 --methods C2-OCT-1V-DETR --dets dets_c2_1v_dev100.json --out generalization_surdino_dev100.json
python scripts/pair_screen.py --run dev300_B41_eps5_limit100 --subset dev100 --dets dets_c2_1v_dev100.json --tag sur_dino_dev100 --exclude dev30 \
  --pairs C2-OCT-1V-DETR@dev300_B41_eps5_limit100_sur-dino_r50:C2-OCT-1V
```
**Quy tắc (dev100, paired bootstrap 1000, BOOT_SEED 2026; Δ = drop_Swin-L(sur DINO-R50) − drop_Swin-L(sur R50 Mask R-CNN), cùng ảnh):**
- **Hợp lệ** chỉ khi white-box drop DINO-R50 (surrogate) ≥ 80%. Không đạt → KHÔNG HỢP LỆ.
- **DETECTOR GÓP PHẦN** nếu Δ ≥ +10 VÀ cận dưới CI > 0 → một phần gap do khác kiểu detector; hướng method mới (vd surrogate R50 + "đầu" kiểu DETR / loss query-based) — entry riêng, user quyết threat model.
- **CHỈ BACKBONE** nếu Δ < +5 (điểm) → gap gần như hoàn toàn do họ backbone (củng cố 09-23 + 10-21; câu chắc cho paper).
- Còn lại → XÁM (mô tả, user quyết).
- Mô tả: chiều DINO-R50 → Mask R-CNN R50 (cùng backbone, khác detector) / R101 / ConvNeXt / Swin-T / YOLOX; tỉ lệ Δ so với +26 của sur Swin-T; 70 ảnh mới (dev100 \ dev30); runtime.
**Caveat khai báo trước:** DINO-R50 khác DINO-Swin-L ở lịch train (12e vs 36e), số scale (4 vs 5) ngoài backbone; R50 của DINO-R50 khác trọng số R50 của Mask R-CNN (cùng kiến trúc + pretrain IN-1k, fine-tune khác); λ theo quy tắc cũ, liều OCT giữ nguyên (không tune); mốc R50 là PNG 10-18 (GPU không tất định).

### Kết quả @ dev100 B41 (V100, xong 12:06 GMT+7; `results/ablation/sur_dino_dev100/screen.json`, `results/runs/dev300_B41_eps5_limit100_sur-dino_r50/{metrics_c2_1v,generalization_surdino}_dev100.json`, log `results/sur_dino/run.log`) — theo quy tắc: **HỢP LỆ, CHỈ BACKBONE**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3 | s/ảnh:
- sur R50 (10-18): 96.9 / 93.2 / 82.4 / 78.4 / 70.6 / **32.1** | 77.2 | ~4
- sur DINO-R50: 86.8 / 72.4 / 65.5 / 58.1 / 53.7 / **29.2** | 59.1 | 11.9 (~3×); DINO-R50 white-box **99.0**
**Quy tắc:** white-box 99.0 ≥ 80 → HỢP LỆ. Δ_Swin-L = **−2.9 [−8.8, −0.5]** < +5 → **CHỈ BACKBONE**. 70 ảnh mới: −3.8 [−9.3, −0.3].
**Mô tả:** surrogate DINO-R50 transfer KÉM HƠN ở mọi target: CA3 −18.1 [−22.5, −14.6], R101 −20.7, Swin-T −20.3, YOLOX −17.0, ConvNeXt −16.9; kể cả Mask R-CNN R50 (cùng kiến trúc backbone) chỉ 86.8. Bất đối xứng: R50 Mask R-CNN → DINO-R50 97.3% (09-23, n300) nhưng DINO-R50 → R50 Mask R-CNN 86.8%. DINO-Swin-L ~30 bất kể surrogate là Mask R-CNN hay DINO (cùng backbone CNN); chỉ đổi họ backbone (sur Swin-T, 10-21) mới đẩy lên 58.
**Đọc:** ba chẩn đoán độc lập (target DINO-R50 09-23; sur Swin-T 10-21; sur DINO-R50 hôm nay) cùng chỉ: **bức tường R50 → DINO-Swin-L là họ backbone, không phải kiểu detector**; khớp cả encoder "sửa" lẫn query-based loss đều KHÔNG giúp khi backbone surrogate vẫn là CNN. Hướng "đổi / khớp kiến trúc detector của surrogate" đóng. Caveat: λ theo quy tắc cũ, liều OCT không tune cho DINO; DINO-R50 khác DINO-Swin-L cả lịch train / số scale.

---

## 2026-10-25 — E33: dồn phòng thủ WP về layer4 + lịch ρ @ dev30 B36 (user: "ưu tiên rẻ, xác định lại dựa trên suy luận về data hiện có") — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Chọn lọc từ data (bỏ trước khi code):** WP2-AS (ASAM từng phần tử) — E28 cho thấy chọn phần tử / thời điểm không quan trọng; WP2-BN (phòng thủ γ/β BN) — họ "bất biến biên độ / kênh" âm 3 lần (LN, IN, C2W); WP2-RAP (phẳng phía input, DRAP arXiv 2504.16474) — E19: δ_SwinT không phẳng hơn phía input, và tốn thêm forward. LTA (latent VAE, arXiv 2603.06311) cần thêm mạng VAE + ε = 16 mềm → không làm.
**Cơ sở (data có sẵn):** E22 noL3 (layer2 + 4) giữ 108% lợi, L3 chỉ 58%; E32 WP2 dời distortion về stage 4, ΔT₄ Swin-L +0.031 (gấp đôi Swin-T); WP5 (ρ 0.005 toàn backbone) mất lợi + hại R101 / Swin-T → giả thuyết: liều cao chỉ hại ở layer nông, tăng liều CỤC BỘ ở layer4 giữ lợi. Literature hỗ trợ diễn giải WP2 = robust hóa surrogate lúc attack ("Defense That Attacks", arXiv 2512.02830: surrogate AT transfer CNN→ViT tốt hơn hẳn; không có bản không-train).
**Ô (gốc C2-OCT-1V-WP2, B36, cùng chi phí — không thêm forward; code `pia.geo_osfd_attack(weight_scope="layer4", weight_rho_l4, weight_rho_sched)`, `core` ρ list / callable):**
- `C2-OCT-1V-WP-L4`: phòng thủ CHỈ layer4 (10 tensor), ρ = 0.002 — layer4 có tự gánh lợi không.
- `C2-OCT-1V-WP2-L4x`: toàn backbone ρ = 0.002, riêng layer4 ρ = 0.005 — ứng viên cải tiến.
- `C2-OCT-1V-WP2-RS`: ρ tuyến tính 0.004 → 0 theo bước (TB 0.002) — lợi đến sớm hay muộn (kỳ vọng thấp: pha sụp E20 là với init δ_SwinT).
**Kiểm:** `scripts/e33_unit_test.py` QUA (ρ float = list hằng = callable hằng trùng tuyệt đối; list → ‖Δθ‖ = ρ_i‖θ0‖; lịch đúng; scope layer4 chỉ chạm layer4; θ khôi phục torch.equal; L∞ 5). **Sức khỏe (ảnh 1–4, B36):** bão hòa / R50 xóa — WP2 80% / 92%; WP-L4 79% / 95%; L4x 80% / 92%; RS 80% / 92% → QUA cả 3; 3.6 s / ảnh (= WP2).
**Mốc:** WP2 B36 dev30 = PNG 10-23 (HF `_v100`, cùng run dir) — eval lại cùng lần.
**Đọc (so WP2 B36 cùng 30 ảnh, như E27 / E28):** BƯỚC NHẢY ΔSwin-L ≥ +10 & ΔCA3 ≥ −3 → dev100; CÓ TÍN HIỆU +3 ≤ ΔSwin-L < +10 & ΔCA3 ≥ −3 → dev100 (khóa entry); KHÔNG nếu ΔSwin-L < +3. Mô tả: WP-L4 giữ bao nhiêu % lợi của WP2 so C2-OCT-1V (mốc C2 dev30 = PNG 10-18 cùng dets dev100); R101 / YOLOX / Swin-T.
**Caveat:** dev30 đã dùng nhiều lần; 3 ô so sánh bội; mốc WP2 khác ngày (GPU không tất định); ρ = 0.005 / lịch đặt trước, không tune.
**Chạy:** `results/e33/run.sh`, log `results/e33/run.log`; metrics `metrics_e33_dev30.json`, dets `dets_e33_dev30.json`, pair_screen tag `e33_dev30`.

### Kết quả E33 @ dev30 B36 (V100, xong 13:12 GMT+7; `results/ablation/e33_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/*_e33_dev30.json`, log `results/e33/run.log`) — theo quy tắc: **KHÔNG** (cả 3 ô)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3: WP2 98.5 / 92.1 / 83.2 / 83.2 / 75.2 / 37.5 | 80.5; WP-L4 99.1 / 92.6 / 83.4 / 80.2 / 76.2 / 34.7 | 80.0; WP2-L4x 98.7 / 90.2 / 80.9 / 82.6 / 75.4 / 34.7 | 79.6; WP2-RS 98.9 / 91.5 / 80.2 / 77.7 / 76.7 / 32.8 | 78.2.
Δ so WP2 (Swin-L | CA3): WP-L4 −2.8 [−6.0, +1.9] | −0.6; L4x −2.8 [−6.5, +1.3] | −0.9; **RS −4.7 [−6.5, −2.2]** | −2.3 (Swin-T −5.5).
**Đọc:** không ô nào ΔSwin-L ≥ +3 → KHÔNG. (i) Tăng liều cục bộ layer4 KHÔNG thêm lợi (L4x ≈ WP-L4 < WP2) → giả thuyết "liều cao chỉ hại ở layer nông" BÁC. (ii) Chỉ layer4 giữ ~60% lợi Swin-L (34.7 vs WP2 37.5, mốc C2 dev30 ~30.2 ngày 10-24 — khác ngày, chỉ ước lượng); cùng E22 (L3 58%, noL3 108%) → lợi PHÂN TÁN trên các stage, không stage nào đủ một mình. (iii) Lịch ρ giảm dần HẠI (CI loại 0) → phòng thủ cần ở CẢ các bước cuối; ρ hằng là tốt nhất trong các dạng đã thử. Họ WP (liều, cộng dồn, AWT, ensemble, spike, scope, lịch, liều theo stage) đều ≤ WP2.

---

## 2026-10-25 — E34: RJ — jitter THANG RoI (tách rời rescale ảnh) @ dev30 B41 (user: "duyệt, run dev30", đào lại C2) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Cơ sở (data + search):** khóa thang kích thước của L_cls mới gỡ ~80% bằng co giãn ảnh (C2-SCL25 r = 0.81), SCL50 bão hòa; LVX (đẩy object qua ranh giới mức FPN) ÂM vì rescale ảnh làm δ loãng (R50 WB 99→92). Search (10-25): bất biến scale ở mức ảnh = SIM/SI-NI-FGSM (classification) = thứ SCL/OCT đã làm; KHÔNG thấy prior art jitter thang RoI tách rời rescale ảnh cho transfer detection → RJ là núm chưa động + có novelty. Kỳ vọng thấp–vừa (SCL bão hòa → khóa RoI có thể gần cạn).
**Ô (gốc C2-OCT-1V, B41 = cùng thời gian OSFD; `pia.geo_osfd_attack(roi_jit=r)`, `repair.roi_jitter`):** trước RoIAlign, co giãn ĐẲNG HƯỚNG mỗi RoI (pos + neg) quanh tâm theo s = exp(U(−r, r)); nhãn GT (gán từ proposal gốc) KHÔNG đổi; ảnh / δ / L_OSFD / loss_rpn_cls KHÔNG đổi (RPN trước extractor) → classifier RoI head bị ép bền trước nhiều thang pooling (mức FPN + lưới 7×7), KHÁC LVX (không rescale ảnh → không loãng δ). `C2-OCT-1V-RJ30` (r = 0.3), `C2-OCT-1V-RJ60` (r = 0.6). Gen riêng seed + 23.
**Kiểm:** `scripts/rj_unit_test.py` QUA (trên tensor rois cố định vì R50+amp không tất định): r = 0 no-op; r = 0.3 batch_idx + tâm giữ, |log s| ≤ 0.3 có phương sai, đẳng hướng; attack L∞ = 5 hữu hạn. **Sức khỏe (ảnh 1–4, B41):** bão hòa / R50 WB — RJ30 80% / 95%, RJ60 80% / 89% (≥ 0.8 × C2-OCT-1V 92%) QUA; 4.2 / 3.9 s/ảnh ≈ C2-OCT-1V B41 (cùng thời gian).
**Mốc:** C2-OCT-1V B41 dev30 = PNG `dev300_B41_eps5_limit100` (HF, 10-18) — eval cùng lần.
**Đọc (so C2-OCT-1V B41 cùng 30 ảnh, paired bootstrap 1000, BOOT_SEED 2026):**
- **GO dev100** nếu [(ΔCA3 ≥ 2 VÀ cận dưới CI > 0) HOẶC (ΔSwin-L ≥ 3 VÀ cận dưới CI > 0 VÀ ΔCA3 ≥ −1)] VÀ ΔR101 ≥ −3. **KHÔNG** nếu cả hai nhánh trượt. Liều: RJ60 − RJ30 (mô tả). Chọn ô Δ lớn hơn nếu GO.
- Mô tả: YOLOX / Swin-T / ConvNeXt riêng; so C2-SCL25 cũ (cùng ý "gỡ khóa kích thước").
**Caveat:** dev30 đã dùng nhiều; mốc C2 PNG 10-18 khác ngày (GPU không tất định); r đặt trước, không tune; RJ chỉ chạm loss_cls RoI (loss_rpn_cls không đổi).
**Chạy:** `results/e34/run.sh`, log `results/e34/run.log`; metrics `metrics_e34_dev30.json`, dets `dets_e34_dev30.json`, tag `e34_dev30`.

### Kết quả E34 @ dev30 B41 (V100, xong 13:53 GMT+7; `results/ablation/e34_dev30/screen.json`, `results/runs/dev300_B41_eps5_limit100/*_e34_dev30.json`, log `results/e34/run.log`) — theo quy tắc: **KHÔNG** (cả 2 ô)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3: C2-OCT-1V 99.4 / 93.6 / 82.7 / 81.4 / 69.9 / 32.6 | 78.0; RJ30 98.7 / 92.2 / 82.5 / 80.9 / 67.1 / 30.3 | 76.9; RJ60 96.9 / 90.1 / 79.0 / 79.6 / 68.4 / 33.6 | 75.7.
Δ so C2-OCT-1V (Swin-L | CA3): RJ30 −2.3 [−4.6, +1.2] | −1.1; **RJ60 +1.0 [−2.4, +2.7] | −2.3** (ConvNeXt −3.6, R101 −3.5). Liều RJ60 − RJ30: Swin-L +3.3, CA3 −1.2.
**Đọc:** không ô nào đạt (ΔCA3 ≥ 2 hoặc ΔSwin-L ≥ 3 với CI > 0) → KHÔNG. RJ30 ≈ C2-OCT-1V (hơi âm); RJ60 liều cao bắt đầu loãng δ (R50 WB 96.9, R101 −3.5, CA3 −2.3) đổi lấy Swin-L +1.0 (CI chứa 0) — cùng mẫu trượt đường đánh đổi như LVX nhưng nhẹ hơn (RJ không rescale ảnh nên loãng ít hơn). Khớp dự đoán: khóa thang kích thước gần cạn sau SCL25 (bão hòa). **Jitter thang RoI KHÔNG thêm lợi.**
**KẾT: đào lại C2 (RJ) + WP2 (E33) hôm nay đều KHÔNG. C2-OCT-1V + WP2 là cặp method chốt. Chuyển sang dev300 (định vị SOTA + cơ chế).**

---

## 2026-10-25 — Sàng lọc C2-OCT-1V-WP2 @ dev300 (user: "Approve, run luôn") — khóa TRƯỚC khi chạy
**Mục đích:** xác nhận lợi WP2 ở dev100 (Swin-L +6.2 so C2-OCT-1V, +11.5 so OSFD) có giữ ở 300 ảnh không — bước định vị cho paper. KHÔNG phải GO/NO-GO method mới (WP2 đã chốt); đây là mở rộng tập.
**Dùng lại (KHÔNG chạy lại):** OSFD B50 (`dev300_B50_eps5_limit300`) + C2-OCT-1V B41 (`dev300_B41_eps5_limit300`) — PNG + dets đã có trên HF (sàng lọc 10-18, cùng V100). Chỉ kéo dets + gen_dets về.
**Chạy mới:** `C2-OCT-1V-WP2` **B36** (= cùng thời gian OSFD B50 trên V100, nhất quán dev100), run `dev300_B36_eps5_limit300`, attack + eval 6 model (Controlled Panel qua run_baselines, yolox_s + dino_swin_l qua eval_generalization), dets cùng tên `dets_screen_dev300.json` để pair_screen gộp 3 run.
**Quy tắc đọc (dev300, paired bootstrap 1000, BOOT_SEED 2026; như dev100):**
- **XÁC NHẬN WP2** (vào bảng cuối làm method Swin-L) nếu ΔSwin-L(WP2 − C2-OCT-1V) ≥ 3 VÀ cận dưới CI > 0 VÀ ΔR101 ≥ −3 VÀ ΔCA3 ≥ −1.
- **KHÔNG tái lập** nếu ΔSwin-L < 3 hoặc CI chứa 0 → ghi nhận lợi dev100 không giữ ở 300 ảnh; WP2 vẫn báo cáo nhưng không claim; user quyết.
- Báo cả WP2 − OSFD (so SOTA) + C2-OCT-1V − OSFD (tái lập mốc 10-18: CA3 +7.0, Swin-L +9.0). 200 ảnh mới (dev300 \ dev100) mô tả riêng.
**Caveat:** mốc OSFD / C2-OCT-1V là PNG 10-18 (cùng V100, cùng seed); WP2 B36 sinh hôm nay (GPU không tất định → đọc CI); ρ = 0.002 không tune.
**Chạy:** `results/screen_wp2_dev300/run.sh`, log `.../run.log`; metrics/dets/gen `*_screen_dev300.json`, tag `screen_wp2_dev300`.

### Kết quả sàng lọc WP2 @ dev300 (V100, xong 14:52 GMT+7; `results/ablation/screen_wp2_dev300/screen.json`, `results/runs/dev300_B36_eps5_limit300/*_screen_dev300.json`, log `results/screen_wp2_dev300/run.log`) — theo quy tắc: **XÁC NHẬN WP2**
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3: OSFD B50 98.2 / 93.2 / 78.9 / 74.8 / 68.0 / 23.5 | 73.9; C2-OCT-1V B41 98.7 / 95.7 / 86.9 / 80.6 / 75.2 / 32.4 | 80.9; **WP2 B36 98.4 / 94.3 / 86.0 / 82.6 / 75.6 / 37.3 | 81.4**.
- **WP2 − C2-OCT-1V: Swin-L +4.8 [+2.7, +6.7]** (≥ 3, CI > 0 ✓) VÀ R101 −1.3 [−2.3, 0.0] (≥ −3 ✓) VÀ CA3 +0.5 [−0.4, +1.8] (≥ −1 ✓) → **XÁC NHẬN**. Swin-T +2.0 [+0.1, +3.5] (DƯƠNG, CI > 0 — khác dev100 nơi Swin-T ≈ 0); ConvNeXt −0.8, YOLOX +0.3 (CI chứa 0).
- WP2 − OSFD: **Swin-L +13.8 [+9.5, +15.7]**, CA3 +7.5 [+4.2, +8.5], mọi target khác họ > 0 CI loại 0 (ConvNeXt +7.2, Swin-T +7.8, YOLOX +7.6), R101 +1.1.
- C2-OCT-1V − OSFD: tái lập mốc 10-18 (CA3 +7.0, Swin-L +9.0) — khớp.
- **200 ảnh mới (dev300 \ dev100):** WP2 − C2 Swin-L +4.7 [+1.5, +6.4], Swin-T +2.1 [+0.1, +4.3] → lợi WP2 LẶP LẠI ngoài dev100 (dev100 Swin-L +6.2, Swin-T −0.0 → ở 300 ảnh Swin-T cũng nhích).
**Đọc:** lợi WP2 trên DINO-Swin-L giữ vững ở 300 ảnh (+4.8 so C2-OCT-1V, +13.8 so OSFD; tuyệt đối 37.3). Khác dev100: Swin-T cũng dương nhỏ (+2.0) — WP2 giúp họ Transformer rộng hơn chút, không chỉ DINO. Bộ ba method chốt cho bảng cuối: **OSFD (baseline) < C2-OCT-1V < C2-OCT-1V-WP2**, tăng đơn điệu trên mọi target khác họ. Trần Swin-L ~37 (1-R50, ε=5) khớp E19–E34.
**Bước tiếp:** bảng baseline đầy đủ (MI, M-DI²) @ dev300; verify clean AP Generalization Panel (model_registry trống); rồi cân nhắc n300/n1000 + ε=8. Push + upload HF.

---

## 2026-10-25 — CHỐT panel đánh giá A/B/C (user confirm qua thảo luận + đối chiếu papers/2602.16494 benchmark) — khóa định nghĩa panel
**Mục tiêu (user):** attack được CẢ nhóm A/B/C với 1 surrogate CNN duy nhất. Đây là **panel generalization/đánh giá cuối**, nhóm theo QUAN HỆ họ backbone với surrogate (khác panel Controlled của RQ1 — giữ nguyên, không bỏ).
**Đối chiếu benchmark 2602.16494 (papers/):** benchmark dùng YOLOv3(DarkNet-53), Faster R-CNN(R50), FCOS(R50), DETR(R50), YOLOX(CSPNet), Mask R-CNN(Swin), DINO(Swin); chốt "cross-architecture CNN→ViT, đặc biệt DINO" là frontier (mọi attack fail transfer sang DINO). Panel dưới phủ FCOS/DINO/YOLOX/Swin/DarkNet của benchmark; xoáy DINO-Swin-L = đúng frontier.
**Surrogate (khóa):** **Mask R-CNN R50** (checkpoint `mask-rcnn_r50_fpn_ms-poly-3x`, đã verify 40.9). Trong attack `skip_mask=True` → mask head KHÔNG dùng → tương đương Faster R-CNN two-stage về gradient. Mô tả paper: "Mask R-CNN R50, mask head unused in attack". KHÔNG relabel thành Faster R-CNN R50 (trọng số khác checkpoint FRC độc lập; đổi = phải chạy lại tất cả).
**Panel A/B/C (6 target) — user confirm 2026-10-25:**
| Nhóm | Định nghĩa | Model | Checkpoint / verify |
|---|---|---|---|
| A | backbone R50 (= surrogate), KHÁC detector head | FCOS-R50 | có, verified-pending-fill |
| A | | **DINO-R50** (thay deformable_detr — user chọn; ghép cặp DINO-Swin-L cô lập backbone) | có (dino-4scale_r50), đã verify clean 53.1 |
| B | backbone CNN KHÁC R50 | **YOLOv3-DarkNet53** (user thêm — phủ DarkNet của benchmark) | **THIẾU — cần tải + verify** |
| B | | **YOLOX-S** (thay yolox_l — clean AP 40.5 ≈ surrogate, công bằng) | có, cần verify |
| C | backbone Transformer | Mask R-CNN Swin-T | có, verified 46.0 |
| C | | DINO-Swin-L (frontier) | có |
**Giữ riêng — Controlled Panel (RQ1, không đổi):** Mask R-CNN R50 → R101 (same-family) / ConvNeXt-T (cross-CNN) / Swin-T (Transformer) — giữ detector cố định để cô lập backbone-family (benchmark không làm được điều này → điểm mạnh phương pháp).
**Caveat khóa:** (1) DINO-R50 khác DINO-Swin-L ở lịch train (12e vs 36e) + scale (4 vs 5) ngoài backbone; (2) YOLOv3 clean AP (~33.7 nếu bản 608-ms) thấp hơn surrogate → relative drop đọc kèm clean AP; (3) DINO-Swin-L lệch capacity/pretrain (đã ghi protocol_lock).
**Việc cần làm để dùng panel này:** (a) tải checkpoint YOLOv3 (mmdet v3, đề xuất `yolov3_d53_8xb8-ms-608-273e_coco`) + thêm vào `GEN_PANEL`/`targets.GEN`; (b) verify clean AP full val2017 cho YOLOv3 + YOLOX-S (+ FCOS-R50, DINO-R50 điền vào model_registry — hiện trống); (c) bỏ DINO-R50 khỏi `DIAG_ONLY` khi vào panel chính thức; (d) eval mọi method (OSFD, C2-OCT-1V, WP2, MI, M-DI²) trên 6 target A/B/C tại dev300 (forward trên PNG có sẵn, không attack lại). ~30–40 phút tải+verify + eval.

---

## 2026-10-25 — Quick run panel A/B/C @ dev30 (user: "Approve", xem nhanh xu hướng trên panel mới) — MÔ TẢ, không quyết định
**Mục đích:** nhìn nhanh 5 method (MI, M-DI², OSFD, C2-OCT-1V, WP2) trên 5 target generalization MỚI gộp [A: fcos_r50, dino_r50 | B: yolov3_d53, yolox_s | C: dino_swin_l] @ dev30. CHỈ mô tả (n=30 nhiễu); số paper cần dev300.
**Dùng lại:** OSFD B50 (`dev300_B50_eps5_limit300`), C2-OCT-1V B41 (`dev300_B41_eps5_limit100`, dets_c2_1v_dev100), WP2 B36 (`dev300_B36_eps5_limit100`, dets_wpa_dev100). **Attack mới:** MI-FGSM, M-DI²-FGSM B50 trên dev30. **YOLOv3** mới tải (`yolov3_d53_mstrain-608`, DarkNet-53, clean AP ~33.7) + wire GEN_PANEL/targets; smoke infer OK.
**Budget:** OSFD/MI/M-DI² B50, C2-OCT-1V B41, WP2 B36 (cùng thời gian V100). ε=5.
**Đọc:** mô tả relative AP drop 5 target × 5 method; chú ý nhóm A (fcos/dino_r50 — backbone R50 giống surrogate, kỳ vọng drop cao), B (yolov3/yolox — CNN khác, vừa), C (dino_swin_l — thấp). KHÔNG quy tắc quyết định; chỉ xem WP2/C2 có giữ thứ hạng trên panel mới không.
**Caveat:** n=30 nhiễu lớn; clean AP target tính trên 30 ảnh (không phải verify full val2017 — để sau); YOLOv3/DINO-R50 chưa điền model_registry.
**Chạy:** `results/abc_dev30/run.sh` (tmux), log `.../run.log`.

### Kết quả quick panel A/B/C @ dev30 (V100, xong 16:31 GMT+7; `results/runs/*/generalization_abc_dev30.json`, log `results/abc_dev30/run.log`) — MÔ TẢ
clean AP (n=30): fcos_r50 37.9 / dino_r50 49.5 / yolov3_d53 36.0 / yolox_s 40.0 / dino_swin_l 59.7.
relative AP drop % — | A: fcos_r50 / dino_r50 | B: yolov3 / yolox_s | C: dino_swin_l |:
- OSFD B50:        92.3 / 95.8 | 74.6 / 73.7 | 22.6
- MI-FGSM B50:     77.7 / 92.2 | 36.7 / 35.4 | 16.2
- M-DI²-FGSM B50:  92.8 / 94.0 | 62.0 / 56.7 | 24.8
- C2-OCT-1V B41:   94.1 / 95.2 | 76.8 / 69.9 | 32.6
- **C2-OCT-1V-WP2 B36: 90.6 / 94.2 | 77.2 / 75.2 | 37.5**
**Đọc (mô tả, n=30 nhiễu):** thứ hạng giữ trên panel mới — nhóm **A** (backbone R50 = surrogate) mọi method cao 90s (dino_r50 94–96 tất cả; fcos: MI thấp 77.7, còn lại ~92–94). Nhóm **B** (CNN khác): **WP2 ≥ C2 > OSFD > M-DI² > MI** (yolov3 77 / yolox 75 cho WP2; MI sụp 37/35). Nhóm **C** DINO-Swin-L: **WP2 37.5 > C2 32.6 > M-DI² 24.8 ≈ OSFD 22.6 > MI 16.2**. → Panel mới tái lập đúng câu chuyện: our method dẫn ở mọi nhóm khác họ, WP2 tốt nhất ca khó. YOLOv3 (DarkNet) hành xử như yolox (CNN khác). Số paper vẫn cần dev300.

---

## 2026-10-25 — Quick test ε=10 @ dev30: WP2 / M-DI² / OSFD (user: "Dev30, epsilon 10 thử") — CHỈ MÔ TẢ, thăm dò ngoài idea.md (primary ε=5, secondary ε=8)
**Mục đích:** xem ε=10 có đẩy DINO-Swin-L lên không và WP2 còn dẫn không (so quick test ε=8 dev10 cũ: WP2 Swin-L 50.6 nhưng WP2−C2 chỉ +1.2). PNG ε=5 KHÔNG dùng lại được → attack mới cả 3.
**Budget:** OSFD B50, M-DI² B50, WP2 B36 (cùng thời gian V100; runtime độc lập ε). **Eval:** cross-family + DINO-Swin-L (r101, ConvNeXt-T, Swin-T controlled; yolox_s, dino_swin_l gen).
**Đọc (mô tả, n=30):** bảng drop ε=10 cho 3 method; WP2−OSFD và WP2−M-DI² (pair_screen CI); so tuyệt đối Swin-L với ε=5 (WP2 dev30 37.5). KHÔNG quy tắc quyết định — ε=10 ngoài plan, chỉ thăm dò; nếu hứa hẹn thì user quyết có đưa vào không.
**Chạy:** `results/eps10_dev30/run.sh` (tmux), log `.../run.log`.

### Kết quả quick test ε=10 @ dev30 (V100, xong 17:00 GMT+7; `results/ablation/eps10_dev30/screen.json`, `results/runs/dev300_B{50,36}_eps10_limit100/*_eps10_dev30.json`, log `results/eps10_dev30/run.log`) — MÔ TẢ
R50/R101/ConvNeXt/Swin-T/YOLOX/DINO-Swin-L | Cross3: OSFD B50 99.4/98.2/96.4/97.0/93.8/**57.4**|95.8; M-DI² B50 99.3/98.0/90.8/88.5/76.1/55.2|85.1; **WP2 B36 100.0/98.5/97.0/95.5/91.9/57.9**|94.8.
Δ: **WP2 − OSFD: Swin-L +0.4 [−8.1,+7.1], Cross3 −1.0** (NGANG — ưu thế biến mất); WP2 − M-DI²: Cross3 +9.7 [5.1,13.5], YOLOX +15.8, Swin-L +2.7; M-DI² − OSFD: Cross3 −10.7.
**Đọc:** (1) ε=10 đẩy DINO-Swin-L ~57–58 cho CẢ WP2 lẫn OSFD → vượt 50 là do EPSILON, không do method. (2) **Ưu thế WP2 vs OSFD (ε=5: +13.8 Swin-L dev300) BIẾN MẤT ở ε=10** — ε lớn thì OSFD tự transfer tốt, mọi attack bão hòa (CNN ~96–100). Khớp quan sát ε=8 cũ (WP2−C2 +1.2). (3) WP2 vẫn hơn M-DI² rõ. **Hệ quả paper:** giá trị method nằm ở ε NHỎ (budget siết → overfit surrogate là nút thắt); ủng hộ chọn ε=5 primary; KHÔNG claim "Swin-L>50 ở ε=10" là đóng góp method (OSFD cũng 57).

### ε=10 @ dev30 trên panel A/B/C đầy đủ (bổ sung fcos/dino_r50/yolov3 từ PNG ε=10 có sẵn; `results/eps10_dev30/eval_abc.log`) — MÔ TẢ
relative AP drop % | A:fcos / dino_r50 | B:yolov3 / yolox | C:dino_sl | Avg5 | AvgB+C:
- OSFD B50:   98.3 / 99.2 | 95.5 / 93.8 | 57.4 | 88.8 | **82.2**
- M-DI² B50:  97.4 / 97.6 | 80.9 / 76.1 | 55.2 | 81.4 | 70.7
- WP2 B36:    99.3 / 98.6 | 93.8 / 91.9 | 57.9 | 88.3 | **81.2**
**So ε=5 vs ε=10 (AvgB+C, dev30):** WP2−OSFD +6.3 (ε=5) → **−1.0 (ε=10)**; WP2−M-DI² +15.5 → +10.5. DINO-Swin-L ε=10 hòa (OSFD 57.4 / WP2 57.9). → Trên panel A/B/C cũng xác nhận: ưu thế method chỉ ở ε NHỎ (ε=5, native OSFD), ε=10 OSFD bão hòa bắt kịp (ε=10 ngoài native OSFD ε=5 — xem check paper 2026-10-25). WP2 luôn hơn M-DI². Mốc chuẩn báo cáo = ε=5.

---

## 2026-10-25 — Quick test k của L_OSFD (k=1/2 vs WP2 k=3) @ dev30 panel A/B/C (user: "approve") — CHỈ MÔ TẢ
**Giả thuyết:** ablation cũ k=5≈k=3 (không phải nguyên liệu cho OSFD thuần); k=1 CHƯA test. OBJ note (10-18): k=3 tạo gradient feature NGƯỢC L_cls (cos tới −0.56) → k nhỏ có thể giảm xung đột, giúp kênh cls (DINO-Swin-L) dù yếu hơn cho feature (CNN/YOLOX). λ=4 GIỮ NGUYÊN (không tune lại cho k mới) để tách riêng hiệu ứng k.
**Ô (gốc C2-OCT-1V-WP2, B36, chỉ đổi k):** K1 (k=1), K2 (k=2); mốc WP2 (k=3, PNG abc_dev30 có sẵn). **Sức khỏe (ảnh 1–4):** bão hòa/R50 xóa — WP2 80/92, K1 77/89, K2 79/92 (≥0.8×WP2) QUA; 4.0 s/ảnh ≈ WP2.
**Đọc (so WP2 k=3 cùng 30 ảnh, panel A/B/C):** nếu DINO-Swin-L TĂNG khi k↓ VÀ CA3/B+C không sụp → xác nhận giả thuyết xung đột, cân nhắc dev100; CNN/YOLOX dự kiến giảm (feature yếu hơn). Mô tả, không quy tắc GO.
**Chạy:** `results/ksweep_dev30/run.sh` (tmux).

### Kết quả k-sweep @ dev30 A/B/C (V100, xong 18:45 GMT+7; `results/runs/dev300_B36_eps5_limit100/generalization_k_dev30.json`, log `results/ksweep_dev30/run.log`) — MÔ TẢ: **giả thuyết BÁC, giữ k=3**
relative AP drop % | A:fcos/dino_r50 | B:yolov3/yolox | C:dino_sl | Avg5 | AvgB+C:
- WP2 k=3: 90.6/94.2 | 77.2/75.2 | **37.5** | 74.9 | 63.3
- K2 k=2:  88.0/91.8 | 77.2/77.9 | 34.8 | 73.9 | 63.3
- K1 k=1:  93.0/92.4 | 74.9/71.1 | 34.7 | 73.2 | 60.2
**Đọc:** DINO-Swin-L k=3 (37.5) > k=2 (34.8) ≈ k=1 (34.7) → k nhỏ KHÔNG giúp kênh cls (giả thuyết xung đột gradient OBJ-note BÁC); AvgB+C k=3=k=2>k=1 (k=1 hại feature, khớp OSFD gốc k=3>k=1). Chênh nhỏ (dev30 nhiễu) nhưng xu hướng rõ: **k=3 tối ưu, giữ nguyên.** Đóng câu hỏi k (cùng ablation cũ k=5≈k=3).

---

## 2026-10-25 — E35: CAP-WP (Clean-Anchored Projected WP) — CỔNG oracle-retention trước (user duyệt đề xuất paste) — khóa TRƯỚC khi code
**Ý tưởng (đề xuất user):** thay WP2 (1 bước phòng thủ trễ) bằng inner-optimization bilevel: mỗi outer step giải θ* = argmin_θ [L_adv(x+δ;θ) + β·D_clean(x;θ,θ0)] trong ρ-ball (chiếu từng tensor ‖θ_l−θ0_l‖≤ρ‖θ0_l‖), rồi g_x tại θ*. **Mới so họ WP cũ:** (a) inner-descent nhiều bước có chiếu (sửa drift của AWT-OD E31); (b) **clean-anchor** D_clean = Σ_l[1−cos(F_l(T(x);θ), sg F_l(T(x);θ0))] — ép surrogate thành "repairer giữ hành vi sạch", nối E32 (H_repair BÁC: δ_WP2 KHÔNG qua được phép sửa encoder DINO) + nhánh Repair-Robust (10-04).
**Vì sao cổng trước:** đắt (1.7–2.2×, B16–21 cùng thời gian); họ WP (WPA/ENS/spike/scope/AWT/schedule) tới giờ đều ≤ WP2; chỉ clean-anchor là đòn bẩy mới. E25 ENS fail cổng này (retention 45). Test landscape "defended+clean-anchored" có GIỮ được vùng δ chung mà gradient ascent xói mòn (E20/E24) không.
**Cấu hình (khóa, KHÔNG quét):** K=2 inner; ρ=0.002 (kế thừa WP2); η=0.001/inner (2 bước → bán kính ≤ 0.002 trước chiếu); clean-anchor backbone **toàn bộ** (vòng đầu kiểm giả thuyết; tối ưu layer2+4 để sau); β = balancing per-ảnh tại θ1 (sau inner step 1, vì tại θ0 ∇D_clean≈0), clip [0.1, 10], **KHÔNG chọn theo Swin-L**; first-order bilevel (detach θ sau mỗi inner, không second-order). Reuse CÙNG OCT view suốt inner loop (không EOT). Khôi phục θ0 cuối mỗi outer step + cuối ảnh.
**Cổng oracle-retention (dev30, init δ = δ_SwinT, Swin-L B0 ~59; snapshot B4/B12/B24; so WP2-IS B12 39.8 (E20), ENS-IS 45–46 (E25)):**
- **QUA** (đáng chạy từ-zero): B12 Swin-L **≥ 52 VÀ R50 ≥ 95**.
- **XÁM** (thử from-zero dev10): B12 Swin-L 47–52.
- **DỪNG**: B12 Swin-L ≤ 45 → clean-anchor cũng không giữ vùng chung; tăng K vô ích.
**Kiểm trước khi chạy cổng:** unit test (θ khôi phục torch.equal; ‖θ−θ0‖≤ρ‖θ0‖ từng tensor; β=0 & η=0 ≈ không đổi θ; L∞=ε); sức khỏe white-box ảnh 1–4 (R50 xóa ≥ 0.8×WP2). Ràng buộc: thao tác weight lúc attack từng ảnh, khôi phục θ0 — KHÔNG fine-tune bằng dữ liệu (như WP2/AWT).
**Chạy:** code `core` mode ("capwp",K,η,β_clip,stages) + `pia.geo_osfd_attack(weight_capwp=)`; ô `C2-OCT-1V-CAPWP` / `-IS`. Cổng: `results/e35/run.sh`.

### Kết quả E35 cổng CAP-WP @ dev30 (V100, xong 19:16 GMT+7; `results/runs/dev300_B{4,12,24}_eps5_limit100/*_capwp_is.json`, log `results/e35/run.log`) — theo quy tắc: **DỪNG** (cổng không qua)
CAP-WP-IS (init δ_SwinT, Swin-L B0 ~58) — R50 drop / Swin-L drop: B4 88.1 / 51.4; **B12 96.4 / 36.7**; B24 97.6 / 39.8.
**B12 Swin-L = 36.7 ≤ 45 → DỪNG.** CAP-WP KHÔNG giữ δ_SwinT đã cắm — xói mòn như WP2-IS (B12 ~35–40) / ENS-IS (45–46). Clean-anchor + inner-optimization projected KHÔNG ngăn được trường gradient R50 kéo δ khỏi vùng chung (khớp E20/E24/E25). B4 51.4 (xói chậm hơn chút lúc đầu) nhưng B12 sụp = đúng pha sụp E20. Giả thuyết "surrogate = clean-preserving repairer → δ repair-robust" BÁC ở cổng; không chạy from-zero. Timing 7.8–8.5 s/ảnh (~3.2×/bước, clean-anchor forward đắt).
**KẾT: họ weight-space (WP2 1-bước / WPA tích lũy / ENS / spike / AWT đầy đủ / scope / CAP-WP bilevel+anchor) — CHỈ WP2 1-bước đạt ~37 Swin-L, mọi biến thể "giải defender kỹ hơn" đều KHÔNG giữ vùng chung. Trần single-R50 ε=5 ~37 vững qua E19–E35.** WP2 = method chốt; bước tiếp = dev300 panel A/B/C đầy đủ + baseline (AWT-OD/BSR) để chắc SOTA.

---

## 2026-10-25 — E36: REL (Relational Geometry) + Tangent Steering — 3 CỔNG, Gate 1 forward-only trước (user duyệt, "chạy Claude session không tmux") — khóa TRƯỚC khi code
**Ý tưởng (đề xuất user, sau CAP-WP DỪNG):** vấn đề KHÔNG ở cách perturb weight mà ở việc tiếp tục ascent R50 sau khi R50 đã vỡ (B4 Swin-L 51 → B12 37 trên quỹ đạo IS). Hai phần: (1) **REL** — thay feature-MSE pointwise bằng divergence ma trận affinity token–token A_l = softmax(F̂_l F̂_lᵀ/τ) trên C4/C5 trong GT-box (đại lượng quan hệ, kiến-trúc-trung-lập, gần thứ attention tiêu thụ); (2) **tangent steering** — chiếu ∇REL trực giao ∇C2, đi dọc level-set R50 thay vì ascent tiếp. Giả thuyết CHƯA bị E19–E34 bác: δ chung nằm trên level-set R50, gần trực giao hướng ascent tức thời → tối ưu scalar R50 không tự tìm ra.
**3 CỔNG (chỉ qua hết mới code attack + from-zero):**
- **Gate 1 (forward-only, CÁI NÀY):** tính REL(δ) cho oracle δ_SwinT, C2, WP2, WPA, các biến thể + quỹ đạo CAP-WP-IS (B4/B12/B24, Swin-L 51/37/40), trên dev30. **QUA nếu:** (a) Spearman(REL, Swin-L drop) qua δ > 0.3, hơn hẳn L_plain/cosd, oracle REL cao hơn WP2 rõ; **VÀ** (b) REL GIẢM dọc quỹ đạo Swin-L sụp (chốt chặn kiểu L_cls_only E24 — qua cross-δ nhưng trượt trục quỹ đạo). Thiếu một trong hai → DỪNG REL, ghi negative.
- Gate 2 (gradient diag): alignment q=∇REL⊥∇C2 với δ_SwinT hơn G8/WP2 ≥1.5× hoặc +0.01, nhất quán qua ảnh.
- Gate 3 (oracle-retention): init δ_SwinT, REL-tangent B4→B12, Swin-L ≥50–52 VÀ R50 ≥85–90.
**REL cụ thể (Gate 1):** C4 (stride 16) + C5 (stride 32); token = vị trí không gian trong GT-box (khung model / stride), normalize theo channel (F̂ = token/‖token‖); A = softmax(F̂ F̂ᵀ / τ), τ=0.1; D = symmetric-KL(A_adv ‖ A_clean) tổng theo hàng; tổng qua box + 2 stage. Box < 2 token bỏ. Swin-L drop = suppression per-image (DINO inference trên PNG, dev30). R50 θ0 cố định, forward-only.
**Caveat (khóa):** base-rate E24 — thước đo qua-δ dễ TRƯỢT trục quỹ đạo (L_cls_only). Tangent có thể chỉ giúp oracle-init, không from-zero (Gate 3 cần nhưng chưa đủ). Xác suất >50: ~5–10%. Nếu REL cũng không tách oracle/WP2 → kết luận mạnh: 1-R50 ε=5, >50 không khả thi bằng biến thể gradient.
**Chạy:** `scripts/e36_rel_probe.py` → `results/mechanism/e36_rel_probe.json`, log `results/e36/run.log`. Background Claude (không tmux).

### Kết quả E36 Gate 1 (forward-only, V100, xong 19:36 GMT+7; `results/mechanism/e36_rel_probe.json`, log `results/e36/run.log`) — theo quy tắc: **DỪNG** (cả 1a + 1b trượt); negative MẠNH
REL (phá affinity token–token R50, C4/C5 GT-box) | Swin-L drop %: **oracle δ_SwinT 0.077 (THẤP NHẤT) / 37.6 (CAO NHẤT)**; C2-OCT-1V 0.165/28.2; WP2 0.158/27.9; WPA 0.159/32.7; FIA 0.166/26.7; RaPA 0.155/25.4; LA1 0.158/24.3; RJ30 0.162/24.6; noCT 0.151/24.5; WP2-XBG 0.149/25.9. Quỹ đạo CAP-WP-IS: B4 0.098/34.6, B12 0.129/29.6, B24 0.150/30.2.
- **Gate 1a: ρ(REL, Swin-L drop) = −0.006** (p 0.99); oracle REL 0.077 < WP2 0.158 → FAIL (oracle REL THẤP NHẤT).
- **Gate 1b: ρ quỹ đạo = −0.500** (sai chiều: REL tăng khi Swin-L sụp) → FAIL.
**Đọc:** REL trên R50 **NGHỊCH BIẾN** transfer Swin-L — oracle phá quan hệ R50 ít nhất nhưng transfer tốt nhất; mọi δ chỉ-R50 phá quan hệ R50 mạnh (overfit) nhưng transfer kém. Tangent-steering theo REL cao sẽ đi XA oracle. Xác nhận E16/E19/E24: **hướng δ chung VÔ HÌNH với mọi objective chỉ-R50 (kể cả quan hệ kiến-trúc-trung-lập).** Theo stopping rule đã khóa (đề xuất user): kết luận MẠNH — với 1 R50 cố định, không thêm cross-family signal, **>50 không khả thi bằng các biến thể gradient**. Không làm Gate 2/3, không WP3/schedule/projection khác.
**CHỐT RQ3 (E19–E36, ~18 thí nghiệm):** method = **C2-OCT-1V-WP2** (Swin-L ~37 dev300, +13.8 so OSFD); trần single-R50 ε=5 ~37 có bằng chứng toàn diện (ascent xói vùng chung E20/E24; weight-space mọi biến thể ≤ WP2 E31/E35; không thước đo chỉ-R50 nào — loss/phẳng/quan hệ — trỏ về δ chung E19/E24/E36). Bước tiếp = HOÀN THIỆN paper: dev300 panel A/B/C đầy đủ + baseline AWT-OD/BSR + verify clean AP + mAP metric; định vị SOTA (ε=5 native OSFD) + cơ chế + trần.

---

## 2026-10-25 — E37: HardBank/CVaR-OCT — worst-view optimization; Gate A oracle-retention trước (user duyệt) — khóa TRƯỚC khi code
**Ý tưởng (đề xuất user, sau E36):** OCT hiện max E[L_t] (mean view) → sau vài bước một số scale/phase đã vỡ, gradient tiếp tục tối ưu view DỄ (R50 88→96) trong khi Swin-L tụt 51→37. Thay bằng **max worst-view** (CVaR): mỗi bước ưu tiên transform mà δ hiện tại phá R50 KÉM NHẤT. Khác mọi OCT ablation cũ (chúng thay *tập* transform; cái này thay *cách tối ưu trên tập*). Nhắm trực tiếp pha sụp B4→B12. **Phân biệt với E36:** E36 bác thước đo TĨNH; cái này đổi ĐỘNG LỰC tối ưu — E36 chưa test.
**Cơ chế (Gate dùng CVaR trực tiếp, không cold-start EMA):** bank 8 transform cố định (4 scale strata trong [−ln2/2, ln2/2] × 2 phase {(0,0),(2,2)}); mỗi bước forward no-grad cả 8 (fc cached), chọn view có L_adv THẤP NHẤT (hardest = phá kém nhất), backward CHỈ view đó → 1 backward/bước + 8 forward. C2 loss giữ nguyên (L_OSFD + 4·L_cls, GT co-transform).
**Caveat (khóa):** vẫn là tiêu chí chỉ-R50 (chọn view theo L_adv R50) → theo E36 có thể không trỏ Swin-L; pha sụp là hiện tượng oracle-init, from-zero có thể hội tụ ~37. Prior >50: ~5–8%.
**Gate A (oracle-retention, CÁI NÀY): init δ_SwinT, HB chạy, snapshot B4/B12.** QUA nếu: Swin-L B12 **≥ 50** VÀ R50 B12 ≥ 85 VÀ Swin-L B12 ≥ CAP-WP-IS B12 (36.7) **+ 10** (≥ 46.7). Vẫn ~37–42 → DỪNG nhánh, chốt trần (measure E36 + dynamics E37 đều không cứu).
**Gate B (chỉ nếu A qua): from-zero dev30** C2-OCT-HB / -HB-WP2 so WP2: BƯỚC NHẢY Swin-L ≥ WP2+10 & CA3 ≥ WP2−3; TÍN HIỆU ≥ WP2+4 R50/CA3 không sụp; <+4 không mở CVaR-4.
**Cam kết:** Gate A fail → DỪNG method search, hoàn thiện paper (dev300 A/B/C + baseline + mAP).
**Chạy:** code `pia.geo_osfd_attack(hardview=(n_scale,n_phase))`; ô `C2-OCT-1V-HB` / `-HB-IS`. Gate A: `results/e37/run.sh`, background Claude.

### Kết quả E37 Gate A HardBank @ dev30 (V100, xong 19:56 GMT+7; `results/runs/dev300_B{4,12}_eps5_limit100/*_hb_is.json`, log `results/e37/run.log`) — theo quy tắc: **DỪNG** (xói NHANH HƠN CAP)
HardBank-IS (init δ_SwinT B0 ~58) — R50 / Swin-L drop: B4 87.8 / 49.3; **B12 94.2 / 29.0**.
**B12 Swin-L = 29.0** < 50 VÀ < 46.7 ngưỡng → FAIL; thấp hơn CAP-WP-IS (36.7) và WP2-IS (~37) → worst-view XÓI NHANH HƠN, không chậm hơn. Cơ chế: tập trung view "hard" (R50 chưa vỡ) đẩy gradient R50 mạnh nhất đúng chỗ δ chưa khớp R50 → kéo δ về hướng đặc-thù-R50 nhanh nhất → xói thành phần Swin-chung nhanh nhất. Không chạy Gate B.
**CHỐT CUỐI RQ3 (E19–E37):** hai trục đều đóng — **E36 (measure):** mọi thước đo chỉ-R50 (loss/phẳng/quan hệ) nghịch/không biến transfer Swin-L; **E37 (dynamics):** đổi động lực tối ưu (worst-view) xói vùng chung nhanh hơn. Trần single-R50 ε=5 ~37 có bằng chứng TOÀN DIỆN. **DỪNG method search (cam kết đã khóa). Method chốt = C2-OCT-1V-WP2.** Chuyển hẳn sang hoàn thiện paper: (1) dev300 panel A/B/C đầy đủ (eval 3 target mới trên PNG có sẵn); (2) baseline 2024–26 (AWT-OD/BSR); (3) verify clean AP Generalization Panel + báo mAP (so benchmark/OSFD); (4) định vị SOTA (ε=5 native OSFD) + cơ chế hai khóa + trần có bằng chứng (chuỗi E19–E37) + 3 chẩn đoán "tường = backbone-family".

---

## 2026-10-28 — E38: quét liều ρ của WP — WP2 có nằm ở đỉnh đánh đổi "mượt ↔ tương đồng gradient" không? @ dev30 B36 (user: "approve, run dev30", ~19:30 GMT+7) — khóa TRƯỚC khi chạy
**Máy:** V100 16GB (đủ 11 checkpoint + COCO; KHÔNG có `.env` → không kéo PNG HF, mọi ô sinh lại hôm nay).
**Vì sao (literature 10-28 + data):** "Why Does Little Robustness Help?" (S&P'24, arXiv 2307.07873): robust hóa surrogate tăng ĐỘ MƯỢT nhưng (quá liều) giảm TƯƠNG ĐỒNG GRADIENT với target → transfer đạt đỉnh ở robust NHỎ. Bound 2504.16474: rủi ro target ≤ surrogate + sharpness + độ khác biệt model (độ phẳng không chạm số hạng khác biệt). Data mình: liều ρ = 0.002 tốt, 0.005 mất lợi + hại R101 / Swin-T, cộng dồn (WPA / AWT) không thêm / trôi. Giả thuyết **H_tradeoff:** WP2 = robust hóa cục bộ không cần train, nằm ở đỉnh đánh đổi; trần ~37 = số hạng khác biệt model mà 1 R50 không chạm. Mục đích: cơ chế cho paper (KHÔNG tìm method mới).
**Attack (dev30, B36 mọi ô = cùng số bước, ε = 5, seed như cũ; ô mới `C2-OCT-1V-WP1` / `-WP10` trong `run_baselines.py`):** ρ ∈ {0 (= `C2-OCT-1V`), 0.001 (WP1), 0.002 (WP2), 0.005 (WP5), 0.01 (WP10)}; eval 6 model (Controlled 4 + yolox_s + dino_swin_l). Run `dev300_B36_eps5_limit100`, tag `e38_dev30`.
**Đo (script `scripts/e38_rho_tradeoff.py`, fp32, dev30):** điểm chung P1 = PNG của ô ρ = 0 (cùng mọi ρ → tách hình học surrogate khỏi δ); mô tả thêm P0 = ảnh sạch. Mỗi ảnh, mỗi ρ: θ'_ρ = θ0 − ρ‖θ0‖·∇θL/‖∇θL‖ (từng tensor conv backbone, ∇θ của L tại P, đúng `core.wp_update`; ρ = 0 → θ0); L = L_OSFD (k = 3, feature sạch tại CÙNG θ') + 4·C2 cls, view đồng nhất (không OCT), RoI seed cố định. g_ρ = ∇ theo ẢNH GỐC (resize khả vi).
- **Sm(ρ)** (độ mượt) = TB 2 mẫu cos(g_ρ(P + ξ), g_ρ(P)), ξ Rademacher ±2 mức xám ảnh gốc (seed 2026 + img_id, CÙNG ξ mọi ρ), θ'_ρ giữ cố định.
- **Sim(ρ)** = cos(g_ρ(P), g_T(P)), g_T = ∇ theo ảnh gốc của loss cls GT của target (`targets.target_grads`, như E1): **DINO-Swin-L (quy tắc)**; Swin-T, R101 (mô tả). Mô tả thêm tỉ lệ trùng dấu.
**Quy tắc (paired bootstrap 1000 theo ảnh, BOOT_SEED 2026; tại P1):**
- (i) MƯỢT TĂNG: Sm(0.01) − Sm(0) > 0 với cận dưới CI > 0 VÀ Spearman(ρ, TB Sm) qua 5 liều ≥ 0.9.
- (ii) TƯƠNG ĐỒNG GIẢM Ở LIỀU CAO: Sim_DINO(0.01) − Sim_DINO(0.002) < 0 với cận trên CI < 0.
- (iii) ĐỈNH TRANSFER: argmax_ρ drop DINO-Swin-L (điểm) ∈ {0.001, 0.002, 0.005}.
- **H_tradeoff ỦNG HỘ** nếu (i) ∧ (ii) ∧ (iii). **BÁC** nếu ¬(i) HOẶC Sim_DINO(0.01) ≥ Sim_DINO(0.002) (điểm). Còn lại **XÁM**.
- **H_sim (mô tả, cơ chế phụ):** WP2 tăng TRỰC TIẾP tương đồng với Swin-L nếu Sim_DINO(0.002) − Sim_DINO(0) > 0, cận dưới CI > 0 VÀ ≥ 10% Sim_DINO(0).
- Mô tả: bảng drop 6 model × 5 liều (pair_screen so WP2), Sm / Sim theo ρ ở P0, Swin-T, R101.
**Caveat (khai báo trước):** dev30 đã dùng nhiều; phòng thủ 1 bước tại điểm cố định ≠ phòng thủ trễ 1 bước dọc quỹ đạo; view đồng nhất ≠ OCT; Sim dùng gradient target CHỈ để phân tích (không vào attack); 5 liều ít điểm; ρ = 0.01 có thể làm δ yếu (đọc kèm R50 white-box); GPU không tất định. E19 đã thấy WP4 KHÔNG tăng căn hướng với δ_SwinT → kỳ vọng H_sim BÁC.
**Chạy:** `results/e38/run.sh`, log `results/e38/run.log`; kết quả đo `results/mechanism/e38_rho_tradeoff.json`.

### Kết quả E38 @ dev30 B36 (V100, xong 20:10 GMT+7; `results/mechanism/e38_rho_tradeoff.json`, `results/ablation/e38_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/*_e38_dev30.json`, log `results/e38/{run,measure}.log`) — theo quy tắc: **H_tradeoff BÁC** (ii trượt), **H_sim BÁC**
(Sự cố: lần đầu script đo crash ở bước đọc `generalization_e38_dev30.json` — sai thứ tự khóa {method: {model}}; sửa 1 dòng, chạy lại riêng bước đo, số đo trùng log lần đầu.)
**Drop % theo ρ (0 / 0.001 / 0.002 / 0.005 / 0.01)** — R50: 99.3 / 98.9 / 98.6 / 96.1 / **48.8**; R101: 92.9 / 92.4 / 92.3 / 85.5 / 39.7; ConvNeXt: 82.3 / 81.7 / 84.2 / 75.5 / 32.1; Swin-T: 79.7 / 80.7 / 82.6 / 73.6 / 40.0; YOLOX: 64.1 / 74.4 / 77.6 / 64.6 / 28.4; **DINO-Swin-L: 31.2 / 34.1 / 36.2 / 29.8 / 13.5**; CA3: 75.4 / 78.9 / 81.5 / 71.2 / 33.5. So WP2: WP1 Swin-L −2.1 [−5.1, +2.7]; WP5 −6.4 [−10.0, −0.7]; WP10 −22.7; ρ = 0 −5.0 [−7.0, +0.2].
**Đo tại P1 (TB 30 ảnh, ρ như trên):** Sm 0.668 / 0.671 / 0.679 / 0.713 / 0.748 (Spearman 1.0; 0.01 − 0 +0.079 [+0.054, +0.104]); **Sim_DINO-Swin-L 0.0120 / 0.0115 / 0.0112 / 0.0137 / 0.0140** (0.01 − 0.002 **+0.0028 [+0.0004, +0.0053]**; 0.002 − 0 −0.0008 [−0.0050, +0.0031]; trùng dấu 0.501 ≈ ngẫu nhiên ở mọi ρ); Sim_Swin-T 0.047 / 0.057 / 0.070 / 0.077 / 0.066 (0.002 − 0 +0.022 [+0.010, +0.033]); Sim_R101 0.105 / 0.121 / 0.127 / 0.105 / 0.072 (0.002 − 0 +0.022 [+0.004, +0.039]; 0.01 − 0.002 −0.056 [−0.072, −0.039]). P0 (ảnh sạch, mô tả): Sm và mọi Sim GIẢM đơn điệu theo ρ (phòng thủ tại điểm sạch ≈ nhiễu θ vô hướng).
**Quy tắc:** (i) ĐẠT; (ii) TRƯỢT — Sim_DINO(0.01) ≥ Sim_DINO(0.002), CI loại 0 theo chiều NGƯỢC → **BÁC**; (iii) ĐẠT (đỉnh 0.002). H_sim: Sim_DINO(0.002) − Sim_DINO(0) CI chứa 0 → **BÁC** (khớp kỳ vọng khai báo, E19).
**Đọc:**
1. Đường liều transfer là hình chữ U ngược, đỉnh ρ = 0.002 ở MỌI target khác họ — ρ = 0.002 đúng là liều tối ưu (cùng số bước).
2. Đánh đổi "mượt ↔ tương đồng gradient" kiểu S&P'24 CÓ XUẤT HIỆN ở target mà gradient R50 mang thông tin: R101 (Sim ~0.1) và Swin-T (~0.05–0.08) — Sim tăng tới ρ 0.002–0.005 rồi giảm, mượt tăng đơn điệu. WP2 tăng tương đồng gradient với R101 / Swin-T (+0.022, CI > 0).
3. **Với DINO-Swin-L thì KHÔNG:** Sim ≈ 0.012 (gần trực giao, trùng dấu = ngẫu nhiên) và PHẲNG theo ρ. Lợi DINO của WP2 (31 → 36) và sụp ở liều cao (→ 13.5) KHÔNG đi qua tương đồng gradient bậc 1. Sụp ở ρ = 0.01 là do δ mất sức trên chính R50 (white-box 48.8) chứ không do lệch hướng khỏi DINO.
4. Hệ quả cho paper: (a) câu "WP2 ở đỉnh đánh đổi mượt ↔ tương đồng" ĐÚNG cho CNN / Swin-T, KHÔNG dùng cho DINO-Swin-L; (b) thiệt hại trên DINO-Swin-L từ δ R50 gần như KHÔNG giải thích được bằng căn hướng gradient tuyến tính (cos ~0.01 ở mọi liều) — khớp E14b (phản ứng chung kiểu nhiễu) và E32 (lợi DINO đến từ đổi kiểu lỗi sang xóa object). Trần ~37 nhất quán với việc 1 R50 không có kênh bậc 1 nào tới DINO-Swin-L.
**Caveat:** dev30; Sim dùng loss cls GT của target tại 1 điểm cố định (P1 = δ của ρ = 0), view đồng nhất; cos tuyệt đối nhỏ nên Δ ~0.003 tuy CI loại 0 nhưng rất nhỏ về độ lớn.

---

## 2026-10-28 — E39: WP2 RÚT GỌN — bỏ loss_rpn_cls + bỏ stage C2 của L_OSFD @ dev30 B36 (user: "chạy thử cái rút gọn", ~20:30 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** data cho thấy 2 số hạng gần như không đóng góp: (a) loss_rpn_cls — RPN-OCT ≈ OSFD (lợi của L_cls nằm ở RoI cls 80 lớp, 10-08); (b) stage C2 của L_OSFD — gradient nhỏ ~10³× phần còn lại (E19), tăng trọng số chỉ hại CNN (C2W, 10-24). Mục đích: method GỌN hơn cho paper (bỏ số hạng thừa), KHÔNG kỳ vọng tăng.
**Ô (gốc C2-OCT-1V-WP2, B36, ρ = 0.002, mọi thứ khác y nguyên):** `C2-OCT-1V-WP2-RD` = L_cls chỉ `loss_cls` RoI (`repair.cls_term("roi")`) + L_OSFD stage C3–C5 (`stage_w = (0, 1, 1, 1)`), **λ = 6 theo ĐÚNG quy tắc chọn λ = 4 cũ** (trung vị 5 ảnh đầu dev300 mean|∇ₓL_feat| / mean|∇ₓL_cls| tại δ = 0, `scripts/rd_calib.py`, `results/e39/calib.log`: c2 tái lập 3.78 → 4; roi 10.75 / 3.34 / 2.17 / 7.23 / 6.06 → **6.06 → 6**). Mô tả: `C2-OCT-1V-WP2-RD4` (cùng rút gọn, giữ λ = 4 — bỏ rpn mà không bù độ lớn).
**Mốc:** WP2 B36 dev30 = PNG E38 (cùng ngày, cùng máy), eval lại cùng lần.
**Quy tắc (dev30, điểm ước lượng, so WP2 cùng 30 ảnh; CI paired bootstrap mô tả):** **GỌN ĐƯỢC** nếu RD có |ΔSwin-L| < 2 VÀ |ΔCA3| < 2 VÀ ΔR101 ≥ −2 VÀ R50 white-box ≥ 0.8 × WP2 → đề xuất RD làm bản chính thức (xác nhận dev300 ở entry riêng khi chạy panel). **TỐT HƠN** (ghi nhận, không đổi quy trình) nếu ΔSwin-L ≥ +2 VÀ ΔCA3 ≥ −1. Còn lại → **KHÔNG GỌN** (giữ WP2 đầy đủ; số hạng "thừa" thực ra có vai trò → ghi nhận cho paper).
**Caveat:** dev30 đã dùng nhiều; λ = 6 theo quy tắc, không tune; GPU không tất định (Δ ~1–2 điểm nằm trong nhiễu dev30).
**Chạy:** `results/e39/run.sh`, log `results/e39/run.log`; tag `e39_dev30`.

### Kết quả E39 @ dev30 B36 (V100, xong 20:33 GMT+7; `results/ablation/e39_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/*_e39_dev30.json`, log `results/e39/run.log`) — theo quy tắc: **GỌN ĐƯỢC** (RD)
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3 | s/ảnh: WP2 98.6 / 92.3 / 84.2 / 82.6 / 77.6 / 36.2 | 81.5 | 4.5 (E38); **RD (λ = 6) 99.0 / 92.1 / 84.7 / 85.2 / 76.6 / 38.0 | 82.2 | 4.0**; RD4 (λ = 4) 99.0 / 93.1 / 83.5 / 80.7 / 77.8 / 39.1 | 80.6 | 4.0.
Δ RD − WP2: Swin-L +1.8 [−0.8, +4.5], CA3 +0.7 [−1.4, +2.3], R101 −0.2, Swin-T +2.6 [−0.0, +5.6], YOLOX −1.1, ConvNeXt +0.5. RD4 − WP2 (mô tả): Swin-L +2.9 [−3.1, +6.2], CA3 −0.8, Swin-T −2.0.
**Quy tắc:** |ΔSwin-L| 1.8 < 2 ✓, |ΔCA3| 0.7 < 2 ✓, ΔR101 −0.2 ≥ −2 ✓, R50 99.0 ≥ 0.8 × 98.6 ✓ → **GỌN ĐƯỢC**; TỐT HƠN không đạt (ΔSwin-L 1.8 < 2).
**Đọc:** bỏ loss_rpn_cls + stage C2 của L_OSFD KHÔNG làm tụt (mọi Δ trong nhiễu, hơi dương) và nhanh hơn ~11% (4.0 vs 4.5 s/ảnh, đo khác lần chạy — cần đo cùng lần trước khi claim). Hai số hạng là thừa → bản rút gọn RD (λ = 6 theo quy tắc) đề xuất làm bản chính thức; xác nhận ở dev300 khi chạy panel (entry riêng). RD4 (không bù λ) nghiêng Swin-L / lùi Swin-T — trượt đường đánh đổi như quét λ. Nhanh hơn → ở CÙNG THỜI GIAN OSFD B50 có thể chạy B nhiều hơn 36 (đo lại tỉ lệ cùng lần).

---

## 2026-10-28 — E40: B bao nhiêu thì WP2-RD bão hòa? Quét B ∈ {36, 50, 75, 100} @ dev30 + OSFD B50 đo thời gian cùng lần (user: "thử nhanh B bao nhiêu là bão hòa", ~20:40 GMT+7) — CHỈ MÔ TẢ, khai báo TRƯỚC khi chạy
**Mốc cũ:** WP2 B36 → B100 dev30 (10-22): Swin-L +1.8, CA3 +4.9 (BÃO HÒA Swin-L ~38); C2-OCT-1V dev100 A4500: CNN gần bão hòa ~B57, Swin-L tăng chậm tới B100.
**Ô:** `C2-OCT-1V-WP2-RD` (λ = 6, E39) B36 (PNG E39) / B50 / B75 / B100; OSFD B50 cùng lần (thời gian + mốc). Run `dev300_B{36,50,75,100}_eps5_limit100`, dets `dets_e40_dev30.json`, tag `e40_dev30`.
**Đọc (mô tả, điểm + CI paired bootstrap):** **bão hòa** ở bước B_k → B_{k+1} nếu ΔSwin-L < 1 VÀ ΔCA3 < 1 (điểm); B bão hòa = B nhỏ nhất thỏa. Báo s/ảnh RD và OSFD B50 CÙNG lần → B_cùng-thời-gian(RD) = 50 × t_OSFD / t_RD-mỗi-bước. Không quyết định method; chọn B cho panel dev300 ở entry riêng.
**Caveat:** dev30 nhiễu (Δ ~1–2 trong nhiễu); B36 là PNG chạy ở lần E39 (khác lần); GPU không tất định.
**Chạy:** `results/e40/run.sh`, log `results/e40/run.log`.

### Kết quả E40 @ dev30 (V100, xong 21:03 GMT+7; `results/ablation/e40_dev30/screen.json`, `results/runs/dev300_B{36,50,75,100}_eps5_limit100/*_e40_dev30.json`, log `results/e40/run.log`) — MÔ TẢ
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3 | s/ảnh (cùng lần):
- OSFD B50: 98.3 / 91.2 / 81.7 / 80.8 / 74.5 / 22.6 | 79.0 | 4.1
- RD B36 (E39): 99.0 / 92.1 / 84.7 / 85.2 / 76.6 / **38.0** | 82.2 | 4.0 (lần E39)
- RD B50: 99.1 / 94.5 / 87.0 / 85.5 / 78.8 / **37.2** | 83.8 | 5.6
- RD B75: 99.5 / 95.3 / 90.9 / 89.0 / 82.6 / **37.8** | 87.5 | 8.1
- RD B100: 99.8 / 95.1 / 92.1 / 89.3 / 83.9 / **39.2** | 88.4 | 11.1
Δ liên tiếp (Swin-L | CA3): 36→50 −0.8 [−2.4, +2.3] | +1.6; 50→75 +0.6 [−3.6, +3.3] | **+3.8 [+2.4, +8.1]**; 75→100 +1.4 [−2.5, +4.0] | +0.9 [−0.6, +2.6].
**Đọc:** theo định nghĩa khai báo (ΔSwin-L < 1 VÀ ΔCA3 < 1) KHÔNG bước nào đạt "bão hòa" trên điểm ước lượng (75→100: Swin-L +1.4 nằm trong nhiễu). Mô tả: **DINO-Swin-L bão hòa NGAY từ B36** (37–39 ở mọi B, mọi Δ trong nhiễu; khớp WP2 B100 10-22 = 38.0); **CNN / Swin-T bão hòa quanh B75** (50→75 +3.8 CI > 0, 75→100 +0.9). **Thời gian cùng lần:** RD ~0.11 s/bước (5.6 / 50, 8.1 / 75, 11.1 / 100), OSFD B50 4.1 s → **B cùng thời gian OSFD B50 ≈ 37** (≈ B36 hiện dùng; tiết kiệm "~11%" ở E39 là do khác lần đo, KHÔNG tái lập). RD B36 − OSFD B50 (cùng thời gian): Swin-L **+15.4 [+5.5, +21.5]**, CA3 +3.1.
**Hệ quả:** giữ B36 (cùng thời gian OSFD) cho bảng chính; báo thêm 1 dòng B75 (~2× thời gian) cho CNN (CA3 +5.3 so B36) — Swin-L không đổi theo B → trần ~37–39 không phải do thiếu bước.

---

## 2026-10-28 — E41: giả thuyết "bậc 2 / năng lượng" cho DINO-Swin-L — bão hòa δ_RD về ±ε (RD-SAT) + sàn nhiễu ngẫu nhiên ±5 (RAND5) @ dev30, hậu xử lý PNG, chỉ forward (user: "Approve", ~21:15 GMT+7) — khóa TRƯỚC khi chạy
**Lập luận:** ΔL_T(δ) ≈ g_Tᵀδ + ½δᵀH_Tδ. E38: cos(g_R50, g_DINO) ≈ 0.012, trùng dấu 0.501 → bậc 1 ≈ 0 với DINO-Swin-L; nếu hướng δ "ngẫu nhiên" so với H_T thì E[δᵀH_Tδ] ≈ tr(H_T)/d·‖δ‖₂² → thiệt hại theo NĂNG LƯỢNG. Khớp: Swin-L nhạy mạnh theo ε (5 → 8: +17 mọi method), bão hòa theo B (E40), E14b (phản ứng chung kiểu nhiễu). Hệ quả kiểm được: trong hộp L∞, ‖δ‖₂ max khi mọi pixel ±ε → giữ dấu R50 (cho CNN), đẩy biên độ lên ε. Đồng thời là ứng viên giải thích trần ~37 (năng lượng bị chặn theo ε).
**Ghi nhận trước khi chạy (từ attack_stats E39):** RMS δ_RD ≈ 4.73 / 5 → năng lượng đã ~89% mức tối đa → SAT chỉ thêm ~+12% năng lượng → kỳ vọng nhỏ (~+0–2).
**Ô (CHỈ eval, `scripts/e41_saturate.py`, run `dev300_B36_eps5_limit100`):** `C2-OCT-1V-WP2-RD-SAT` = clip(x + 5·sign(δ_RD)), pixel δ = 0 → dấu Rademacher (seed 2026 + img_id); `RAND5` = clip(x + 5·r), r Rademacher (seed 2026 + img_id + 1). Mốc RD B36 (PNG E39), eval lại cùng lần. 6 model.
**Quy tắc (dev30, điểm; CI paired bootstrap mô tả):** RD-SAT − RD: **CÓ TÍN HIỆU** nếu ΔSwin-L ≥ +2 VÀ ΔCA3 ≥ −1 → entry riêng đưa bước bão hòa vào attack. Còn lại **KHÔNG**. Mô tả: Swin-L của RAND5 (sàn năng lượng thuần) → tách "năng lượng" vs "hướng" của RD: phần hướng = drop(RD) − drop(RAND5) ở cùng ε; tỉ lệ năng lượng SAT / RD.
**Caveat:** dev30; PNG uint8 + clip [0, 255] (pixel sát biên không bão hòa được); RAND5 1 mẫu nhiễu / ảnh.
**Chạy:** `results/e41/run.sh`, log `results/e41/run.log`, tag `e41_dev30`.

### Kết quả E41 @ dev30 (V100, xong 21:20 GMT+7; `results/ablation/e41_dev30/screen.json`, `results/runs/dev300_B36_eps5_limit100/*_e41_dev30.json`, log `results/e41/run.log`) — theo quy tắc: **KHÔNG**; giả thuyết "năng lượng" BÁC MẠNH
Năng lượng TB δ²: RD 22.45 → SAT 24.54 (×1.093; chỉ 2.2% pixel δ_RD = 0).
R50 / R101 / ConvNeXt / Swin-T / YOLOX / DINO-Swin-L | CA3: RD 99.0 / 92.1 / 84.7 / 85.2 / 76.6 / **38.0** | 82.2; **RD-SAT 98.7 / 91.1 / 84.2 / 83.4 / 73.3 / 36.9 | 80.3**; **RAND5 (±5 ngẫu nhiên, năng lượng TỐI ĐA) 4.5 / 3.2 / 2.8 / 8.4 / 1.8 / −2.3 | 4.3**.
Δ RD-SAT − RD: Swin-L −1.1 [−2.2, +0.5], CA3 **−1.9 [−3.2, −0.8]**, YOLOX −3.3 → KHÔNG (ΔSwin-L < +2).
**Đọc:**
1. **Nhiễu ±5 ngẫu nhiên với năng lượng tối đa KHÔNG phá DINO-Swin-L (−2.3) và gần như không phá model nào** → thiệt hại KHÔNG đến từ năng lượng / bậc 2 kiểu "hướng ngẫu nhiên". Giả thuyết E41 (và lập luận "trần do năng lượng bị chặn theo ε" Claude nêu trong hội thoại trước khi chạy) **BÁC**.
2. Toàn bộ 38 điểm Swin-L của RD đến từ HƯỚNG / CẤU TRÚC của δ, dù cos(g_R50, g_DINO) tại 1 điểm chỉ ~0.012 (E38) → thước đo căn hướng gradient bậc 1 cục bộ KHÔNG nắm được phần hướng hữu ích; hiệu ứng là phi tuyến / tích lũy dọc δ. Nhất quán E16 (cùng phổ, cùng năng lượng, δ_SwinT phá DINO gấp ~2× δ_R50 → khác biệt là mẫu cụ thể).
3. Lấp đầy biên độ ở pixel attack chọn |δ| nhỏ làm HẠI nhẹ (CA3 −1.9, CI < 0) → biên độ nhỏ ở các pixel đó là thông tin, không phải lãng phí.
4. Hệ quả cho câu chuyện trần: KHÔNG được nói "trần do năng lượng"; câu đúng: Swin-L cần HƯỚNG cụ thể (ngẫu nhiên = 0, R50 = 38, Swin-T = 58 cùng ε), và các ước lượng hướng chỉ-R50 đã thử không tiến gần hướng Swin-T (E19 / E24 / E36 / E38).

---

## 2026-10-28 — E42: đặt cạnh AugTrans (CMC 2026, Table 2: FR-R50 → DETR-R50 / DINO-R50) — eval DETR-R50 + DINO-R50 trên PNG dev30 có sẵn (user: "thử nhanh cái a", ~21:40 GMT+7) — CHỈ MÔ TẢ
**Đối chiếu (papers/AugTrans.pdf §4.1):** surrogate của họ Faster R-CNN R50 torchvision (clean AP 36.9) ≠ Mask R-CNN R50 mmdet của mình; DETR-R50 chính thức 500e (42.5) vs mmdet `detr_r50_8xb2-150e`; DINO-R50 "repo chính thức" clean AP **63.2 (bất thường — DINO-R50 chính thức ~49–51)** vs mmdet `dino-4scale_r50_12e`; ε = 5 khớp; ngân sách họ 160 iter × 10 EOT ≈ 1600 gradient vs B36 / B50 của mình (~44×); metric họ AP tuyệt đối, COCO val 5000 ảnh; code AugTrans không có (repo 404, 09-23). "Cross-paradigm" của họ = backbone VẪN R50.
**Chạy:** eval_generalization `detr_r50`, `dino_r50` trên dev30: run B36 (C2-OCT-1V, WP2, RD — PNG E38 / E39) + run B50 (OSFD — PNG E40). Báo relative AP drop + AP tuyệt đối. Chỉ tham khảo, KHÔNG phải so sánh công bằng (khác surrogate, checkpoint target, ngân sách, tập ảnh).
**Chạy:** `results/e42/run.sh`, log `results/e42/run.log`.
(Sửa run.sh sau lần chạy đầu lỗi: eval_generalization cần dets Controlled CÙNG run chứa đủ method → dùng dets E38 / E39 / E40; out `generalization_e42{a,b,}_dev30.json`.)

### Kết quả E42 @ dev30 (V100, ~21:50 GMT+7; `results/runs/dev300_B36_eps5_limit100/generalization_e42{a,b}_dev30.json`, `results/runs/dev300_B50_eps5_limit100/generalization_e42_dev30.json`, log `results/e42/run.log`) — MÔ TẢ
Clean AP (n = 30): DETR-R50 36.4, DINO-R50 49.5. Relative drop % (AP adv) — DETR-R50 | DINO-R50:
- OSFD B50: 95.7 (1.6) | 96.0 (2.0)
- C2-OCT-1V B36: 96.5 (1.3) | 94.3 (2.8)
- WP2 B36: 95.9 (1.5) | 94.4 (2.8)
- **RD B36: 96.0 (1.5) | 94.6 (2.7)**
- AugTrans (Table 2, FR-R50 nguồn, COCO 5000 ảnh, ~1600 gradient): DETR 42.5 → 2.8 (93.4%) | DINO 63.2 → 5.4 (91.5%).
**Đọc:** mọi method của mình (kể cả OSFD) đều phá DETR-R50 / DINO-R50 ~94–96% ở ε = 5 với ~1/44 ngân sách AugTrans → setting "CNN → detector Transformer cùng backbone R50" BÃO HÒA, KHÔNG phân biệt method (khớp DINO-R50 94–97% từ 09-23 / panel A/B/C). Số mình ngang hoặc mạnh hơn AugTrans về drop, nhưng KHÔNG so công bằng (khác surrogate, checkpoint target, ngân sách, tập ảnh; DINO clean 63.2 của họ bất thường). Định vị paper: ca khó thật là ĐỔI HỌ BACKBONE (DINO-Swin-L: OSFD 22.6 → RD 38.0 cùng dev30), nơi AugTrans không báo cáo.

---

## 2026-10-28 — CHỐT panel A/B/C cuối + surrogate (user approve, ~22:20 GMT+7) — thay entry "CHỐT panel đánh giá A/B/C" 10-25 ở nhóm B
**Đối chiếu báo cáo gửi thầy** (surrogate Faster R-CNN R50; A: fcos_r50, deformable_detr; B: yolov3_d53, yolox_l; C: mask_rcnn_swin_t, dino_swin_l; metric ASR + % mAP giảm) với setting repo. **Quyết định (user approve đề xuất Claude):**
- **Surrogate: GIỮ Mask R-CNN R50** (`mask-rcnn_r50_fpn_ms-poly-3x`, mask head không dùng khi attack → tương đương Faster R-CNN về gradient). Lý do: Controlled Panel (RQ1) cần CÙNG detector cho R50 / R101 / ConvNeXt-T / Swin-T — mmdet chỉ có checkpoint Mask R-CNN chính thức cho đủ 4 backbone; mọi bằng chứng E1–E42 trên surrogate này. Bù: 1 lần kiểm surrogate FRC-R50 cho method chốt ở bước cuối (entry riêng). → Báo thầy thay đổi này.
- **A: FCOS-R50 + DINO-R50** (giữ setting repo, KHÔNG Deformable DETR): DINO-R50 ghép cặp DINO-Swin-L (cùng detector, chỉ khác backbone) → cô lập họ backbone ở nhóm C; nhóm A bão hòa với mọi method (E42) nên đổi không mất khả năng phân biệt. Deformable DETR: tùy chọn phụ. → Báo thầy.
- **B: YOLOv3-D53 + YOLOX-L** (QUAY VỀ báo cáo, thay YOLOX-S): YOLOX-L có trong OSFD (AAAI'24) + benchmark 2602.16494 → so được với literature; lý do "cân capacity" của YOLOX-S yếu (panel không kiểm soát capacity). YOLOX-L trước là held-out, CHƯA từng eval → đưa vào panel chính không rò rỉ quyết định method. YOLOX-S thành model phụ (giữ số đã có).
- **C: Mask R-CNN Swin-T + DINO-Swin-L** (khớp).
- **Controlled (RQ1): R101 / ConvNeXt-T / Swin-T** giữ. **Phụ:** YOLOX-S, DETR-R50. **Held-out:** Mask R-CNN Swin-S (duy nhất).
- **Metric:** relative AP drop (AP 0.5:0.95) + **ASR** = trong các GT target detect được ở ảnh sạch (đúng lớp, IoU ≥ 0.5, score ≥ 0.3), tỉ lệ không còn detect sau attack (= `mechanism.per_image_transfer` "suppression").
**Thực thi:** tải YOLOv3 (`mim` id `yolov3_d53_mstrain-608_273e_coco`, thêm vào `scripts/download_checkpoints.sh --gen`); smoke load + infer 1 ảnh QUA cả 5 target ngoài Controlled (fcos_r50, dino_r50, yolov3_d53, yolox_l, dino_swin_l); `eval_generalization.py` mặc định `--models` = `PANEL_GEN` (5 model trên); cập nhật `docs/model_registry.md` + `CLAUDE.md`. Clean AP full val2017 của panel A/B/C CHƯA verify (việc tiếp).

---

## 2026-10-28 — E43: nhóm B panel chốt — eval YOLOv3-D53 + YOLOX-L trên PNG dev30 có sẵn (user: "1 thôi", ~22:15 GMT+7) — CHỈ MÔ TẢ
PNG: run B36 (C2-OCT-1V, WP2 — E38; RD — E39), run B50 (OSFD — E40). Báo relative AP drop + ASR (suppression, `per_image_transfer`). YOLOX-L lần đầu được eval (trước là held-out). Không quyết định method.
**Chạy:** `results/e43/run.sh`, log `results/e43/run.log`.

### Kết quả E43 @ dev30 (V100, xong 22:27 GMT+7; `results/runs/dev300_B36_eps5_limit100/generalization_e43{a,b}_dev30.json`, `results/runs/dev300_B50_eps5_limit100/generalization_e43_dev30.json`, `results/e43/{summary,pair_rd_osfd}.json`, log `results/e43/run.log`) — MÔ TẢ
Clean AP (n = 30): YOLOv3 36.0, YOLOX-L 49.8. Relative AP drop % / ASR % — YOLOv3 | YOLOX-L:
- OSFD B50: 74.3 / 71.3 | 72.8 / 61.6
- C2-OCT-1V B36: 76.4 / 62.6 | 61.3 / 50.0
- WP2 B36: 76.0 / 65.5 | 71.9 / 56.7
- **RD B36: 79.7 / 64.7 | 72.2 / 51.6**
RD − OSFD (paired bootstrap 1000, BOOT_SEED 2026): YOLOv3 drop +5.4 [−3.3, +15.0], ASR −6.6 [−15.6, +2.0]; **YOLOX-L drop −0.6 [−7.1, +7.6], ASR −10.0 [−20.4, −1.3]**.
**Đọc:** (1) Nhóm B mới KHÔNG lặp lại lợi thế đã thấy ở YOLOX-S (dev300 WP2 − OSFD +7.6): trên **YOLOX-L RD ≈ OSFD về AP drop**, YOLOv3 nhỉnh hơn nhưng CI chứa 0. (2) **ASR nghiêng về OSFD** ở cả hai (YOLOX-L CI loại 0): OSFD làm MẤT object nhiều hơn; RD hạ AP qua đường khác (lệch định vị / sai lớp / box giả — cần err_profile để tách). (3) WP giúp rõ trên YOLOX-L so C2-OCT-1V (61.3 → 71.9) nhưng chỉ đưa về ngang OSFD. (4) Hệ quả cho claim: "hơn OSFD ở mọi target khác họ" ĐÚNG với YOLOX-S / ConvNeXt / Swin-T / DINO-Swin-L, CHƯA đúng với YOLOX-L (dev30, n nhỏ) — phải kiểm ở dev300 trước khi viết; báo cả 2 metric. Caveat: dev30, bootstrap CI rộng; YOLOX-L lần đầu được eval nên không có thiên lệch chọn method.
(Ghi chú thực thi: lần bootstrap đầu khởi tạo lại CocoImageEval mỗi mẫu → chậm, dừng; chạy lại khởi tạo 1 lần.)

### Bổ sung E43 — bảng panel CHỐT đầy đủ @ dev30 (thêm FCOS-R50, `generalization_e44{a,b,}_dev30.json`, log `results/e43/fcos.log`; bảng `results/e43/panel_dev30.json`) — MÔ TẢ
Relative AP drop % / ASR % (AP điểm, không bootstrap; ASR = suppression GT detect ở ảnh sạch):
| method | R50 (WB) | R101 | ConvNeXt-T | FCOS-R50 (A) | DINO-R50 (A) | YOLOv3 (B) | YOLOX-L (B) | Swin-T (C) | DINO-Swin-L (C) |
|---|---|---|---|---|---|---|---|---|---|
| OSFD B50 | 98.3 / 97.6 | 91.2 / 88.8 | 81.7 / 71.1 | 92.3 / 90.2 | 96.0 / 91.7 | 74.3 / 71.3 | 72.8 / 61.6 | 80.8 / 71.4 | 22.6 / 22.0 |
| C2-OCT-1V B36 | 99.3 / 91.6 | 92.9 / 83.3 | 82.3 / 65.0 | 92.7 / 84.3 | 94.3 / 86.2 | 76.4 / 62.6 | 61.3 / 50.0 | 79.7 / 66.1 | 31.2 / 24.3 |
| WP2 B36 | 98.6 / 90.5 | 92.3 / 78.2 | 84.2 / 64.5 | 91.0 / 84.2 | 94.4 / 86.4 | 76.0 / 65.5 | 71.9 / 56.7 | 82.6 / 64.1 | 36.2 / 28.1 |
| **RD B36** | 99.0 / 89.9 | 92.1 / 78.1 | 84.7 / 65.3 | 93.1 / 82.7 | 94.6 / 81.1 | 79.7 / 64.7 | 72.2 / 51.6 | 85.2 / 62.6 | **38.0 / 29.0** |
**Đọc:** (1) AP drop: RD ≥ OSFD ở mọi target trừ DINO-R50 (−1.4, nhóm A bão hòa) và YOLOX-L (−0.6); lợi lớn nhất DINO-Swin-L (+15.4), Swin-T (+4.4), YOLOv3 (+5.4). (2) **ASR: OSFD CAO HƠN mọi method của mình ở MỌI target trừ DINO-Swin-L** (RD − OSFD: R101 −10.7, ConvNeXt −5.8, FCOS −7.5, DINO-R50 −10.6, YOLOv3 −6.6, YOLOX-L −10.0, Swin-T −8.8; DINO-Swin-L +7.0) → OSFD "xóa object" nhiều hơn; method của mình hạ AP qua đường khác (giữ object đúng lớp với score ≥ 0.3 nhưng thêm box giả / lệch định vị / hạ thứ hạng — cần err_profile). Loss cls (C2) là bước làm ASR tụt (OSFD → C2-OCT-1V: ASR giảm ở mọi target trừ DINO-Swin-L). (3) Hệ quả: báo cáo gửi thầy dùng ASR làm 1 trong 2 metric → với ASR, claim "vượt OSFD" chỉ đúng ở DINO-Swin-L. Phải chọn rõ metric chính (AP drop) + giải thích ASR, hoặc xem lại thành phần loss cls. dev30, chưa CI.

---

## 2026-10-28 — E45: RD-V — thay loss cls RoI của RD bằng loss "vanishing" để kéo ASR lên mà giữ AP drop @ dev30 B36 (user chọn hướng (2), ~22:45 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** bảng panel chốt dev30 (E43 bổ sung): ASR của OSFD cao hơn RD ở 7/8 target hộp đen (chỉ DINO-Swin-L RD hơn); ASR tụt đúng ở bước thêm loss cls (OSFD → C2-OCT-1V). Báo cáo gửi thầy dùng ASR làm metric → cần method không thua OSFD về ASR. Tiền lệ: Probe V (10-04, nền C2/RRB, dev100): AP drop ngang C2 (CA3 +0.4), recall thấp hơn + ít FP nền hơn ở mọi target → đúng kiểu lỗi làm ASR tăng.
**Ô `C2-OCT-1V-WP2-RDV`:** RD y nguyên (OCT 1V, MI, L_OSFD stage C3–C5, WP ρ = 0.002, B36) chỉ đổi số hạng cls: `repair.vanish_loss` (ascent: BCE objectness anchor RPN dương → 0, CE RoI dương → lớp nền; không đụng mẫu âm), **λ = 0.3 theo ĐÚNG quy tắc** (`scripts/rd_calib.py vanish`, `results/e45/calib.log`: 0.26 / 0.38 / 0.26 / 0.28 / 0.19 → trung vị 0.26 → 1 chữ số có nghĩa = 0.3, trùng λ_V Probe V; c2 tái lập 3.78). Smoke dev5 B3 OK.
**Mốc:** RD B36 (PNG E39, eval lại cùng lần vào dets_e45), OSFD B50 (PNG + dets E40). Bảng: `scripts/panel_table.py` (mới) — drop + ASR trên Controlled + PANEL_GEN, T8 = TB 8 target hộp đen, paired bootstrap 1000 so RD.
**Quy tắc (dev30, điểm; CI mô tả):** **ĐẠT** (RD-V thay RD làm method chốt, xác nhận dev300 ở entry riêng) nếu ΔASR_T8(RDV − RD) ≥ +3 VÀ Δdrop_T8 ≥ −1 VÀ Δdrop DINO-Swin-L ≥ −2. **KHÔNG** nếu ΔASR_T8 < +3. Còn lại (ASR lên nhưng drop / Swin-L tụt quá ngưỡng) → **ĐÁNH ĐỔI** (ghi nhận, user quyết). Mô tả: số target hộp đen mà RDV ≥ OSFD về ASR / về drop.
**Caveat:** dev30; λ theo quy tắc, không tune; GPU không tất định.
**Chạy:** `results/e45/run.sh`, log `results/e45/run.log`.

### Kết quả E45 @ dev30 B36 (V100, xong 22:50 GMT+7; `results/e45/panel_dev30.json`, `results/runs/dev300_B36_eps5_limit100/*_e45_dev30.json`, log `results/e45/run.log`) — theo quy tắc: **ĐÁNH ĐỔI**
Drop / ASR % (T8 = TB 8 target hộp đen):
| | R101 | ConvNeXt | Swin-T | FCOS | DINO-R50 | YOLOv3 | YOLOX-L | DINO-Swin-L | T8 |
|---|---|---|---|---|---|---|---|---|---|
| OSFD B50 | 91.2 / 88.8 | 81.7 / 71.1 | 80.8 / 71.4 | 92.3 / 90.2 | 96.0 / 91.7 | 74.3 / 71.3 | 72.8 / 61.6 | 22.6 / 22.0 | 76.5 / 71.0 |
| RD B36 | 92.1 / 78.0 | 84.7 / 65.3 | 85.2 / 62.6 | 93.1 / 82.7 | 94.6 / 81.1 | 79.7 / 64.7 | 72.2 / 51.6 | **38.0** / 29.0 | **80.0** / 64.4 |
| **RD-V B36** | 93.2 / 87.0 | 78.3 / 70.5 | 81.2 / 69.3 | 95.3 / 89.1 | 93.2 / 91.6 | 76.4 / 75.8 | 69.5 / 59.5 | 29.9 / 26.4 | 77.1 / **71.1** |
RD-V − RD: **ASR_T8 +6.7 [+2.2, +11.7]** ✓; drop_T8 **−2.8** [−4.3, +1.5] (< −1 ✗); **DINO-Swin-L drop −8.1 [−11.3, −3.1]** (< −2 ✗) → **ĐÁNH ĐỔI**. OSFD − RD: ASR_T8 +6.6 [+2.5, +11.8], drop_T8 −3.5, Swin-L −15.4.
RD-V so OSFD: ASR ≥ OSFD 2 / 8 target (YOLOv3, DINO-Swin-L; còn lại thua 0.1–2.1); drop ≥ OSFD 5 / 8; T8 drop 77.1 vs 76.5, ASR 71.1 vs 71.0 (≈ ngang); DINO-Swin-L 29.9 vs 22.6 (+7.3).
**Đọc:** (1) Loss cls CE RoI (đẩy KHỎI lớp đúng) là nguồn của CẢ lợi DINO-Swin-L LẪN ASR thấp; thay bằng vanishing kéo ASR về ngang OSFD (+6.7) nhưng mất 8 điểm DINO-Swin-L (38 → 30) và ~3 điểm drop T8 → **đánh đổi ASR ↔ AP drop / Swin-L thật sự**, cùng kiểu đường biên hai kênh đã thấy (λ, noCT, OBJ). Khớp ghi nhận POS-CAP (10-18): lợi Swin-L của L_cls đến từ đẩy object ĐÃ sai trên R50 đi xa hơn khỏi lớp đúng. (2) RD-V ≈ OSFD trên T8 (cả drop lẫn ASR) + hơn OSFD ở DINO-Swin-L +7.3 → "OSFD-cấp ASR, hơn ở Transformer". (3) Không có ô nào hơn OSFD đồng thời cả ASR lẫn drop ở mọi target.

---

## 2026-10-28 — TẠM DỪNG (user, ~22:55 GMT+7, trả máy): tổng kết phiên + việc mở
**Máy:** V100 16GB. Không tiến trình chạy. **PNG + dets + gen_dets phiên này ĐÃ lên HF (user thêm `.env` cuối phiên, ~23:00 GMT+7), tag `_1028v100`:** `dev300_B{36,50,75,100}_eps5_limit100` (tar 233 / 51 / 24 / 24 MB). Máy mới: `python scripts/sync_artifacts.py download <run> --tag _1028v100` (⚠️ giải nén ra cùng thư mục `artifacts/runs/<run>/` với tar không tag / `_v100` của ngày khác — tải tag nào thì dùng tag đó). Số tổng hợp (metrics / generalization / screen / panel_dev30 json) ĐÃ commit trong `results/`. Sinh lại được bằng `results/e3{8,9}/run.sh`, `results/e4{0..5}/run.sh` (GPU không tất định → lệch nhỏ).
**Phiên này (E38–E45 + chốt panel):**
- E38 quét ρ: đỉnh transfer ρ = 0.002 (chữ U ngược mọi target); đánh đổi "mượt ↔ tương đồng gradient" (S&P'24) chỉ thấy ở R101 / Swin-T, KHÔNG ở DINO-Swin-L (Sim ~0.012 phẳng theo ρ).
- E39 **RD** (bỏ loss_rpn_cls + stage C2, λ = 6 theo quy tắc): GỌN ĐƯỢC (≈ WP2, Swin-L 38.0 dev30).
- E40 quét B: DINO-Swin-L bão hòa từ B36 (37–39 đến B100), CNN ~B75; B cùng thời gian OSFD B50 ≈ 37.
- E41: nhiễu ±5 ngẫu nhiên phá Swin-L −2.3 → thiệt hại do HƯỚNG, không do năng lượng (bác lập luận "trần do năng lượng" của Claude); bão hòa ±ε không giúp.
- E42 (đặt cạnh AugTrans): DETR-R50 / DINO-R50 bão hòa ~94–96% mọi method kể cả OSFD → setting "CNN → detector Transformer cùng R50" không phân biệt method; ca khó là đổi họ backbone.
- **CHỐT panel** (entry "CHỐT panel A/B/C cuối"): surrogate Mask R-CNN R50; A FCOS-R50 / DINO-R50; B YOLOv3-D53 / **YOLOX-L**; C Mask R-CNN Swin-T / DINO-Swin-L; Controlled R101 / ConvNeXt-T / Swin-T; phụ YOLOX-S, DETR-R50; held-out Swin-S; metric AP drop + ASR. YOLOv3 đã tải + vào download_checkpoints.sh.
- E43 + bảng panel dev30: **AP drop RD ≥ OSFD trừ DINO-R50 / YOLOX-L; ASR: OSFD cao hơn RD ở 7/8 target hộp đen** (chỉ DINO-Swin-L RD hơn).
- E45 **RD-V** (loss cls vanishing, λ = 0.3 theo quy tắc): ĐÁNH ĐỔI — ASR_T8 +6.7 (≈ OSFD 71.1 vs 71.0), drop_T8 77.1 ≈ OSFD 76.5, nhưng DINO-Swin-L 38.0 → 29.9 (vẫn +7.3 so OSFD).
**Code mới:** `scripts/e38_rho_tradeoff.py`, `scripts/rd_calib.py` (argv cls_mode), `scripts/e41_saturate.py`, **`scripts/panel_table.py`** (bảng drop + ASR panel chốt, paired bootstrap, T8); `repair.cls_term("roi")`; ô `C2-OCT-1V-WP1 / -WP10 / -WP2-RD / -RD4 / -RDV / -RD-SAT / RAND5`; `eval_generalization.PANEL_GEN` (mặc định panel chốt).
**VIỆC MỞ — user quyết khi bắt đầu phiên mai:**
1. **Định vị method:** (a) **RD** — ưu tiên AP drop + Transformer (DINO-Swin-L +15.4 so OSFD), thua ASR 7/8; (b) **RD-V** — ≈ OSFD trên T8 cả drop lẫn ASR, DINO-Swin-L +7.3; (c) quick test ô kết hợp CE + vanishing (kỳ vọng nằm giữa a và b, theo các đường biên hai kênh trước).
2. **Định nghĩa ASR:** hiện = TB theo ảnh (macro, `per_image_transfer`); báo cáo gửi thầy ghi "% object không detect được" → cân nhắc gộp theo object (micro) hoặc báo cả hai (sửa `panel_table.py`, chạy CPU trên dets — tải từ HF tag `_1028v100`).
3. Sau khi chốt (1)–(2): verify clean AP full val2017 panel A/B/C (FCOS, DINO-R50, YOLOv3, YOLOX-L, DINO-Swin-L); method chốt + OSFD + MI + M-DI² (+ RaPA) @ dev300 panel đầy đủ; kiểm surrogate FRC-R50; n300 GO; n1000; ε = 8.
4. Cần báo thầy: surrogate Mask R-CNN R50 (thay Faster R-CNN, vì Controlled Panel); DINO-R50 thay Deformable DETR (ghép cặp DINO-Swin-L); YOLOX-L vào lại nhóm B; ASR nghiêng OSFD ngoài nhóm C.
5. Máy mới: tạo `.env` (HF_TOKEN) ngay đầu phiên; dets phiên 10-28 tải lại từ HF tag `_1028v100` (không cần sinh lại để tính ASR micro).

---

## 2026-10-28 — Bổ sung metric theo benchmark 2602.16494 (APloc, CSR) + ASR micro cho bảng E45 @ dev30, CHỈ CPU trên dets có sẵn (user: "làm đi luôn", xong ~23:05 GMT+7) — MÔ TẢ
**Định nghĩa (benchmark §3.1.1, IoU 0.5; `scripts/panel_table.py`):** APloc = AP50 với mọi lớp gộp thành 1 lớp; CSR = recall micro đúng lớp (mỗi GT lấy detection score cao nhất IoU ≥ 0.5 mọi lớp, TP nếu đúng lớp) — `csr0` không ngưỡng score (đúng chữ paper, paper không ghi ngưỡng), `csr3` score ≥ 0.3; ASR micro = Σ object mất / Σ object detect được ở ảnh sạch. APloc / CSR báo "% giảm so ảnh sạch". Kiểm: drop + ASR macro trùng bảng E45. Kết quả `results/e45/panel_metrics_dev30.json`, log `results/e45/panel_metrics.log`.
**OSFD − RD (điểm, T8 = TB 8 target hộp đen; CI ở json):** ASR macro T8 +6.7 | **ASR micro T8 +4.4 [+1, +8]** (ConvNeXt +0.4, Swin-T +3.7, YOLOX-L +3.3 — CI chứa 0; FCOS +12.9, DINO-R50 +11.8, R101 +7.4, YOLOv3 +5.8 CI > 0; DINO-Swin-L −10.2) | **APloc drop T8 −1.4** (RD ≥ OSFD ở ConvNeXt, Swin-T, YOLOv3 −4.8, YOLOX-L −4.1, DINO-Swin-L −12.1; OSFD hơn ở R101 +3.7, FCOS +4.2, DINO-R50 +4.1) | CSR0 drop T8 +2.5 (OSFD hơn rõ ở nhóm A: FCOS +9.4, DINO-R50 +17.3) | CSR3 drop T8 +3.7.
RD-V − RD: ASR micro T8 +5.1, APloc −2.1, CSR3 +4.7, DINO-Swin-L APloc −7.7.
**Đọc:** (1) Gộp theo object (micro) thu hẹp chênh ASR OSFD − RD còn +4.4; phần thua ASR của RD tập trung ở target CÙNG backbone R50 / cùng họ (FCOS, DINO-R50, R101) + YOLOv3; ở target khác họ (ConvNeXt, Swin-T, YOLOX-L) chênh nhỏ, CI chứa 0; DINO-Swin-L RD hơn rõ (+10.2). (2) APloc: RD ≥ OSFD ở mọi target khác họ (trừ nhóm A / R101) → RD KHÔNG thua về "làm mất / lệch object" ở target khác họ; OSFD mạnh hơn ở target cùng backbone R50 (feature R50 chuyển thẳng). (3) Câu định vị chính xác hơn: RD ≥ OSFD ở target khác họ backbone (mục tiêu RQ) trên drop / APloc, ASR ngang (CI chứa 0) trừ DINO-Swin-L RD hơn rõ; OSFD giữ ưu thế ở target cùng backbone R50 (nhóm A + R101), nơi mọi method đã ~90%+.

---

## 2026-10-30 — E46: RD-CV — KẾT HỢP loss cls CE RoI (RD) + vanishing (RD-V) @ dev30 B36 (việc mở 1(c) entry "TẠM DỪNG 2026-10-28"; user: "Approve", ~10:40 GMT+7) — khóa TRƯỚC khi chạy
**Máy:** RTX 3090 24GB, mới dựng; KHÔNG có `.env` → không kéo PNG HF, mốc RD / RD-V (B36) + OSFD (B50) sinh lại cùng lần.
**Vì sao:** E45: CE RoI (đẩy khỏi lớp đúng) cho AP drop + DINO-Swin-L, vanishing (đẩy về nền) cho ASR → đánh đổi. Thử cả hai số hạng cùng lúc; kỳ vọng (khai báo trước) nằm GIỮA RD và RD-V theo các đường biên hai kênh trước (λ, noCT, OBJ).
**Ô `C2-OCT-1V-WP2-RDCV`:** RD y nguyên (OCT 1V, MI, L_OSFD C3–C5, WP ρ = 0.002, B36, ε = 5), số hạng cls = **6·loss_cls RoI + 0.3·L_V** (`repair.cls_term("roi+vanish")`, chung 1 forward backbone → cùng chi phí mỗi bước). **Giữ NGUYÊN 2 λ đã calib** (6 từ E39, 0.3 từ E45) — không calib / tune lại. Smoke dev5 B3 OK (0.4 s/ảnh).
**Quy tắc (dev30, điểm; CI paired bootstrap mô tả; giống E45, so RD):** **ĐẠT** (RD-CV thay RD làm method chốt, xác nhận dev300 ở entry riêng) nếu ΔASR_T8(RDCV − RD) ≥ +3 VÀ Δdrop_T8 ≥ −1 VÀ Δdrop DINO-Swin-L ≥ −2. **KHÔNG** nếu ΔASR_T8 < +3. Còn lại → **ĐÁNH ĐỔI** (user quyết). Mô tả: so OSFD và RD-V (số target ≥ OSFD về ASR / drop), ASR micro, APloc, CSR.
**Caveat:** dev30 đã dùng nhiều; mốc sinh lại trên máy khác (3090 vs V100) → RD / RD-V / OSFD lệch nhỏ so E45, mọi so sánh dùng số CÙNG lần này; GPU không tất định.
**Chạy:** `results/e46/run.sh`, log `results/e46/run.log`, tag `e46_dev30`.

### Kết quả E46 @ dev30 B36 (RTX 3090, xong 10:55 GMT+7; `results/e46/panel_dev30.json`, `results/runs/dev300_B{36,50}_eps5_limit100/*_e46_dev30.json`, log `results/e46/run.log`) — theo quy tắc: **KHÔNG** (ΔASR_T8 +2.0 < +3)
Drop / ASR macro % (cùng lần, mốc sinh lại trên 3090):
| | R101 | ConvNeXt | Swin-T | FCOS | DINO-R50 | YOLOv3 | YOLOX-L | DINO-Swin-L | T8 |
|---|---|---|---|---|---|---|---|---|---|
| OSFD B50 | 93.7 / 87.7 | 83.4 / 71.3 | 77.5 / 66.1 | 94.0 / 89.0 | 95.7 / 93.6 | 75.9 / 71.8 | 74.2 / 61.1 | 26.5 / 24.1 | 77.6 / 70.6 |
| RD B36 | 93.0 / 78.7 | 85.8 / 64.5 | 85.1 / 65.0 | 94.5 / 82.5 | 91.8 / 86.6 | 80.7 / 65.0 | 70.5 / 59.3 | 30.6 / 26.2 | 79.0 / 66.0 |
| RD-V B36 | 92.9 / 86.6 | 78.8 / 69.7 | 78.3 / 72.2 | 95.0 / 89.4 | 93.7 / 92.4 | 82.0 / 73.7 | 73.2 / 60.8 | 29.3 / 26.5 | 77.9 / 71.4 |
| **RD-CV B36** | 95.2 / 83.2 | 85.8 / 69.5 | 82.5 / 66.4 | 94.7 / 85.0 | 92.9 / 90.2 | 82.5 / 69.7 | 71.8 / 54.5 | **32.8** / 25.4 | **79.8** / 68.0 |
RD-CV − RD: ASR_T8 **+2.0 [−1, +6]** (< +3 ✗), drop_T8 +0.8 [−1, +3] ✓, DINO-Swin-L drop +2.2 [−2, +5] ✓ → **KHÔNG**. Mô tả: ASR micro T8 +2.9 [+1, +5], **APloc drop T8 +2.6 [+1, +4]** (cao nhất 4 ô: 79.3 vs OSFD 75.3), CSR3 +3.2. RD-CV so OSFD: drop T8 +2.2, ASR macro T8 −2.6, ASR micro −2.3.
s/ảnh (3090, cùng lần): RD 2.7, RD-V 2.4, RD-CV 3.1 (+15%: forward RPN / RoI thêm của L_V), OSFD B50 3.1.
**⚠️ Không tái lập E45 (V100):** clean AP 6 model trùng E45 (lệch ≤ 0.2) → eval đúng; lệch ở bước SINH attack: DINO-Swin-L RD 38.0 → **30.6**, OSFD 22.6 → 26.5, RD-V 29.9 → 29.3; RD − OSFD Swin-L +15.4 → **+4.1**; đánh đổi RD-V (Swin-L −8.1 ở E45) còn −1.3. Lớn hơn nhiều mức "GPU không tất định ~1–2" từng thấy. Nghi vấn: TF32 trên Ampere (`torch.backends.cudnn.allow_tf32 = True` mặc định; V100 không có TF32) → gradient conv kém chính xác, có thể ảnh hưởng riêng nhánh WP (∇θ chuẩn hóa) / Swin-L. CHƯA kiểm. Hệ quả: mọi so sánh trên Swin-L giữa máy V100 và Ampere KHÔNG được ghép; trước khi chốt method cần xác định nguồn lệch.
**Đọc:** (1) Kết hợp CE + vanishing nằm GIỮA RD và RD-V về ASR (như khai báo) nhưng KHÔNG mất drop / Swin-L — trên lần chạy này RD-CV có drop T8, APloc T8 và DINO-Swin-L cao nhất 4 ô. (2) Theo quy tắc khóa → KHÔNG (ASR chưa đủ). (3) Kết luận E45 "RD-V mất 8 điểm Swin-L" không tái lập trên 3090 → câu chuyện đánh đổi CE ↔ vanishing trên Swin-L chưa chắc chắn.

---

## 2026-10-30 — E47: TF32 có phải nguồn lệch DINO-Swin-L giữa V100 (E45) và 3090 (E46)? RD B36 + OSFD B50 @ dev30 FP32 thuần (user: "approve", ~10:59 GMT+7) — CHẨN ĐOÁN, khai báo TRƯỚC khi chạy
**Vì sao:** E46 (3090) RD Swin-L 30.6 vs E45 (V100) 38.0; clean AP trùng → lệch ở bước sinh attack. 3090 mặc định `cudnn.allow_tf32 = True` (conv cắt mantissa 10 bit); V100 không có TF32.
**Thao tác:** cờ mới `run_baselines.py --no-tf32` (tắt TF32 conv + matmul, run riêng `*_fp32`); RD B36 + OSFD B50 dev30, mọi thứ khác y nguyên; eval Controlled (dets) + DINO-Swin-L (inference để mặc định — clean AP không đổi theo TF32).
**Đọc (khai báo trước, điểm):** **TF32 LÀ NGUỒN** nếu RD_fp32 Swin-L ≥ 35 (gần E45 38.0 hơn E46 30.6) → protocol: mọi run trên Ampere+ dùng `--no-tf32`; số Swin-L E46 không dùng. **KHÔNG PHẢI** nếu ≤ 33 → lệch là nhiễu giữa lần chạy (dev30 quá nhỏ cho Swin-L) → đọc Swin-L từ dev100 trở lên. Giữa → XÁM. OSFD_fp32 mô tả (E45 22.6 / E46 26.5).
**Chạy:** `results/e47/run.sh`, log `results/e47/run.log`.

### Kết quả E47 @ dev30 (3090, xong 11:05 GMT+7; `results/runs/dev300_B{36,50}_eps5_limit100_fp32/*_e47_dev30.json`, log `results/e47/run.log`) — theo quy tắc: **KHÔNG PHẢI TF32**
DINO-Swin-L drop: **RD_fp32 31.5** [24.9, 42.6] (E46 TF32 30.6, E45 V100 38.0); OSFD_fp32 26.2 (E46 26.5, E45 22.6). Controlled RD_fp32 98.2 / 92.6 / 84.2 / 85.2 (≈ E46). s/ảnh RD 2.7 (không đổi), OSFD 3.7.
**Đọc:** tắt TF32 không đổi gì → TF32 không phải nguồn. Lưu ý (phát hiện sau khi khóa): ô C2-OCT* chạy **AMP fp16** (`_C2OCT amp=True`) → conv trong attack là fp16, TF32 vốn không áp dụng cho phần lớn tính toán; khai báo E47 bỏ sót điểm này. Lệch KHÔNG giống nhiễu ngẫu nhiên giữa lần chạy: 2 lần trên 3090 cho RD Swin-L 30.6 / 31.5, các lần trên V100 37–39 (E39 B36 38.0, E40 B50 / B75 / B100 37.2 / 37.8 / 39.2) → lệch có HỆ THỐNG theo phần cứng (nghi: kernel fp16 / thuật toán cuDNN khác giữa Volta và Ampere). CHƯA kiểm (ứng viên: RD `amp=False` trên 3090). Hệ quả: số Swin-L không ghép giữa V100 và Ampere; claim "+15 Swin-L so OSFD" (V100) chưa vững — trên 3090 chỉ +4–5.

---

## 2026-10-30 — E48: lệch DINO-Swin-L V100 vs 3090 đến từ WP và/hoặc AMP fp16? @ dev30 B36 trên 3090 (user: "Kiểm tìm nguyên nhân thử", ~11:14 GMT+7) — CHẨN ĐOÁN, khai báo TRƯỚC khi chạy
**Manh mối (log cũ):** C2-OCT-1V (KHÔNG WP) tái lập giữa V100 và A4500 (Ampere): Swin-L dev300 V100 32.4, dev100 A4500 B41 30.5; lợi so OSFD +8.8 (V100 dev100) vs +6.4 (A4500). Mọi run có WP đều trên V100. WP tính ∇θ trong autocast fp16, không GradScaler → underflow có thể khác kernel Volta / Ampere.
**Ô (3090, dev30, B36, seed như cũ; mốc V100 = E38 / E39 cùng dev30 B36):** `C2-OCT-1V` (V100 31.2), `C2-OCT-1V-WP2` (V100 36.2), `C2-OCT-1V-WP2-RD-NOAMP` (RD, `amp=False`; RD V100 38.0, 3090 AMP 30.6 / 31.5). RD AMP (PNG E46) eval lại cùng lần.
**Đọc (khai báo trước, Swin-L điểm):**
- **AMP LÀ NGUỒN (trên Ampere):** RD-NOAMP ≥ 35 → protocol: ô WP chạy không AMP trên Ampere; đo lại thời gian.
- **WP NHẠY PHẦN CỨNG:** C2-OCT-1V(3090) trong ±2.5 của 31.2 VÀ WP2(3090) − C2-OCT-1V(3090) ≤ +2 (V100: +5.0) → lợi Swin-L của WP2 không tái lập trên Ampere.
- Cả C2-OCT-1V(3090) cũng lệch > 2.5 → lệch chung của máy / nhiễu dev30, không riêng WP → cần dev100.
- Còn lại → XÁM, mô tả.
**Caveat:** dev30; 1 lần chạy mỗi ô; GPU không tất định.
**Chạy:** `results/e48/run.sh`, log `results/e48/run.log`.

### Kết quả E48 @ dev30 B36 (3090, xong 11:21 GMT+7; `results/runs/dev300_B36_eps5_limit100/*_e48_dev30.json`, log `results/e48/run.log`) — theo quy tắc: **lệch CHUNG của máy / nhiễu dev30, KHÔNG riêng WP, KHÔNG do AMP** → cần dev100
(Ghi chú: giờ trong các entry E46–E48 ban đầu ước đoán sai, đã sửa theo timestamp file log.)
DINO-Swin-L drop, 3090 | V100 (E38 / E39): C2-OCT-1V **25.5** | 31.2 (−5.7); WP2 **31.8** | 36.2 (−4.4); RD (AMP, PNG E46) 30.6 | 38.0; **RD-NOAMP 32.5** | — . OSFD B50: 26.5 (E46) | 22.6 (E40). Controlled: R50 / R101 / ConvNeXt / Swin-T — C2-OCT-1V 99.3 / 93.5 / 81.7 / 83.4; WP2 98.5 / 92.1 / 86.4 / 81.2; RD-NOAMP 98.5 / 93.2 / 85.5 / 86.2 (≈ V100 trong ±2–3). s/ảnh: C2-OCT-1V 2.5, WP2 2.7, RD 2.7, RD-NOAMP 3.5 (AMP nhanh hơn 1.3×).
**Theo quy tắc:** RD-NOAMP 32.5 < 35 → không phải AMP; C2-OCT-1V lệch −5.7 > 2.5 → không riêng WP. **Mô tả: lợi TƯƠNG ĐỐI của WP2 TÁI LẬP** (WP2 − C2-OCT-1V: 3090 +6.3, V100 +5.0); RD ≈ WP2 trên cả hai máy.
**Đọc:** (1) Mức TUYỆT ĐỐI Swin-L của họ C2 thấp hơn ~5–7 trên 3090, OSFD lại cao hơn ~4 → không phải dịch đều theo máy. 2 lần cùng máy (E46 / E47) gần trùng (30.6 / 31.5) → trên cùng phần cứng gần tất định; đổi phần cứng = quỹ đạo sign khác (như đổi seed) → với n = 30 và CI Swin-L rộng ~±8, mỗi máy là 1 "lần rút". Giả thuyết: biến thiên theo SEED của Swin-L ở dev30 ~ cỡ 5 điểm → mọi chênh Swin-L < ~6 ở dev30 không đọc được; số V100 không phải "tốt hơn" mà là 1 lần rút. CHƯA kiểm (cần đổi seed trên cùng máy). (2) Hệ quả cho claim: "+15 Swin-L so OSFD" (V100 dev30) và "+4" (3090 dev30) đều là 1 lần rút; số đáng tin là dev300 (V100: WP2 sàng lọc 10-25, C2-OCT-1V − OSFD +9.0) → cần dev100 / dev300 trên máy chạy final, đo CẢ OSFD cùng máy.

---

## 2026-10-30 — E49: gộp 3 quỹ đạo RD (E46 / E47 / E48) — phần phá Swin-L có CHUNG giữa các quỹ đạo không? @ dev30, hậu xử lý PNG, chỉ forward (user: "approve", ~11:28 GMT+7) — PHÉP DÒ, khai báo TRƯỚC khi chạy
**Vì sao:** E47 / E48: đổi số học (TF32 / AMP / phần cứng) = đổi quỹ đạo sign, Swin-L dev30 dao động 30.6–38.0 → nhiều nghiệm δ tốt ngang trên R50, mức phá Swin-L khác nhau. Nếu phần "phá Swin-L" chung giữa quỹ đạo còn phần đặc thù R50 ngẫu nhiên, bỏ phiếu dấu lọc được phần chung.
**Ô (`scripts/e49_traj_vote.py`, run `dev300_B36_eps5_limit100`):** `RD-VOTE3` = clip(x + 5·sign(δ_E46 + δ_E47 + δ_E48)), hòa → Rademacher (seed 2026 + img_id); `RD-MEAN3` = clip(x + round(TB δ)) (mô tả). Eval Controlled + DINO-Swin-L + YOLOX-L. **Dùng 3× ngân sách → CHỈ là phép dò, KHÔNG phải method.**
**Đọc (khai báo trước, Swin-L điểm; 3 nguồn lẻ 30.6 / 31.5 / 32.5):** **TÍN HIỆU** nếu VOTE3 − max(lẻ) ≥ +4 VÀ CA3 (ConvNeXt / Swin-T / YOLOX-L) không giảm > 2 so TB lẻ → entry riêng thiết kế bản CÙNG ngân sách (chia B thành K nhánh). **KHÔNG** nếu VOTE3 − max(lẻ) ≤ +1 → đóng ý gộp quỹ đạo. Còn lại XÁM. Mô tả: đồng thuận dấu giữa cặp δ; tương quan Pearson / Spearman suppression Swin-L theo ảnh giữa 3 nguồn; SD theo ảnh giữa nguồn.
**Caveat:** dev30; 3 quỹ đạo khác nhau do số học (không phải seed khởi tạo) → độ đa dạng có thể nhỏ; E41: đẩy biên độ lên ±ε hại nhẹ (CA3 −1.9).
**Chạy:** `results/e49/run.sh`, log `results/e49/run.log`.

### Kết quả E49 @ dev30 (3090, xong 11:32 GMT+7; `results/mechanism/e49_traj_vote.json`, `results/runs/dev300_B36_eps5_limit100*/*_e49*_dev30.json`, log `results/e49/run.log`) — theo quy tắc: **XÁM** (VOTE3 − max lẻ = +1.7)
DINO-Swin-L drop: lẻ E46 / E47 / E48 = 30.6 / 31.5 / 32.5; **VOTE3 34.2** [28.2, 42.2]; MEAN3 32.6. ConvNeXt / Swin-T / YOLOX-L: VOTE3 84.7 / 87.2 / 71.5 vs TB lẻ 85.2 / 85.5 / 71.7 (không giảm > 2 ✓). R101 92.5 vs 93.0.
**Mô tả:** đồng thuận dấu giữa cặp δ 0.77–0.82 (hòa 1.8%); suppression Swin-L theo ảnh giữa 3 nguồn **Pearson 0.92–0.93, Spearman 0.93–0.95**, SD theo ảnh giữa nguồn 2.6 điểm; ASR macro Swin-L lẻ 26.2–26.5 → VOTE3 28.1.
**Đọc:** (1) 3 quỹ đạo trên CÙNG máy 3090 rất giống nhau (≈ 80% dấu trùng, ảnh nào bị phá gần như cố định) → đa dạng quá nhỏ để bỏ phiếu lọc được nhiều; VOTE3 nhích +1.7 (trong nhiễu). (2) Vì thế E49 KHÔNG trả lời được câu "V100 vs 3090": 3 nguồn ở đây đều là 3090; quỹ đạo V100 có thể khác nhiều hơn. Phép kiểm trực tiếp: tải PNG V100 (HF tag `_1028v100`, `.env` đã có) và eval trên 3090 (eval không phụ thuộc máy — clean AP trùng) + đo đồng thuận dấu δ_V100 vs δ_3090.

---

## 2026-10-30 — E50: vai trò của pixel δ Ở NỀN — che δ_RD theo GT box (IN / OUT / RING) @ dev30, hậu xử lý PNG E46, chỉ forward (user: "approve", ~11:37 GMT+7) — PHÉP DÒ, khai báo TRƯỚC khi chạy
**Vì sao (thảo luận background với user):** mọi lần khai thác nền trước đây đều là "nền làm ĐÍCH cho feature object trên R50" (BG-OCT, XBG, E17, E18) — đều hại / yếu. Chưa từng đo ở mức pixel: δ ngoài box đóng góp bao nhiêu, cho target nào. Dữ kiện: E14b — thiệt hại feature bb3 Swin-L 87% từ token trong box, ngoài ≈ 0, δ trải năng lượng đều (~2/3 token là nền); box giả trên nền là kiểu phá chính của OSFD trên CNN (FP nền / ảnh tới 18–41) nhưng DINO chặn (1.1 → 1.8). Dưới L∞ không dồn ngân sách giữa pixel được, nhưng ĐỔI MỤC TIÊU cho pixel nền được.
**Ô (`scripts/e50_bg_mask.py`, run `dev300_B36_eps5_limit100`, nguồn δ = RD B36 PNG E46):** vùng theo hộp COCO (mọi annotation, khung ảnh gốc): `RD-IN` = δ chỉ trong hợp box; `RD-OUT` = chỉ ngoài; `RD-RING` = vành (box nới mỗi cạnh 25%) trừ IN. Eval panel đủ 9 model (`panel_table.py`, so FULL = RD) + FP / ảnh nền | sát object (định nghĩa `err_profile`).
**Đọc (khai báo trước; share_X(m) = drop(X) / drop(FULL), điểm):**
- **"NỀN CHỈ PHỤC VỤ CNN"** nếu share_OUT(DINO-Swin-L) ≤ 0.2 VÀ TB share_OUT(ConvNeXt, FCOS, YOLOv3, YOLOX-L) ≥ 0.4 → pixel nền là ngân sách đổi mục tiêu được cho Swin-L (entry riêng thiết kế loss vùng nền).
- **"VÀNH CÓ VAI TRÒ Ở SWIN-L"** nếu share_RING(DINO-Swin-L) ≥ 0.25 → ngữ cảnh sát object là kênh; ứng viên loss riêng cho vành.
- Mô tả: share_IN + share_OUT (tính cộng tính?), ASR, FP nền theo ô, tỉ lệ diện tích từng vùng.
**Caveat:** dev30; che δ cắt cạnh đột ngột ở biên box (biên nhân tạo); δ được tối ưu cho cả ảnh nên share ≠ "đóng góp nếu tối ưu riêng"; GT box chỉ dùng được vì threat model GT-assisted.
**Chạy:** `results/e50/run.sh`, log `results/e50/run.log`.

### Kết quả E50 @ dev30 (3090, xong 11:45 GMT+7; `results/e50/panel_dev30.json`, `results/mechanism/e50_bg_mask.json`, log `results/e50/run.log`) — theo quy tắc: **không nhánh nào** ("nền chỉ phục vụ CNN" trượt vì TB share_OUT CNN 0.24 < 0.4; vành share Swin-L 0.01 < 0.25)
Diện tích TB: IN 0.36 / OUT 0.64 / RING 0.17. Drop % (FULL | IN | OUT | RING): R101 93.0 | 87.2 | 25.5 | 14.8; ConvNeXt 85.8 | 73.5 | 20.2 | 7.5; Swin-T 85.1 | 70.6 | 18.1 | 5.4; FCOS 94.5 | 81.0 | 29.2 | 19.1; DINO-R50 91.8 | 81.9 | 26.0 | 9.1; YOLOv3 80.7 | 59.2 | 20.4 | 9.2; YOLOX-L 70.5 | 58.0 | 11.6 | 4.0; **DINO-Swin-L 30.6 | 26.0 | 0.5 | 0.4**; T8 79.0 | 67.2 | 18.9 | 8.7. ASR T8 66.0 | 57.7 | 10.7 | 5.6.
share_OUT: Swin-L **0.02**; ConvNeXt 0.24, FCOS 0.31, YOLOv3 0.25, YOLOX-L 0.16. share_IN: Swin-L 0.85, CNN 0.73–0.87. Bỏ δ nền (IN − FULL): T8 −11.8, YOLOv3 −21.5, Swin-T −14.5, ConvNeXt −12.3, **Swin-L −4.6**.
FP nền / ảnh (sạch → FULL | IN | OUT): R50 3.8 → 32.8 | 11.4 | 25.0; ConvNeXt 3.8 → 20.3 | 7.8 | 14.6; Swin-T 3.5 → 17.4 | 6.4 | 13.5; YOLOX-L 1.1 → 9.1 | 3.0 | 6.1; **DINO-Swin-L 1.6 → 3.3 | 2.2 | 2.3**.
**Đọc:** (1) Với δ hiện tại, pixel nền (64% ảnh) gần như KHÔNG tác động DINO-Swin-L (OUT 0.5; vành 0.4) — kể cả ngữ cảnh sát object. (2) Với CNN, pixel nền đóng góp thật (bỏ đi mất ~12 điểm T8) chủ yếu qua box giả trên nền (~70–75% FP nền đến từ δ ngoài box). (3) Hệ quả: nếu ĐỔI MỤC TIÊU pixel nền sang phục vụ Swin-L thì trả giá ~10–20 điểm ở CNN — và E50 KHÔNG cho thấy pixel nền CÓ THỂ tác động Swin-L (chỉ cho thấy δ hiện tại thì không). Câu hỏi khả thi cần cổng oracle: white-box DINO-Swin-L chỉ trên vùng OUT (chỉ phân tích, như E8).

---

## 2026-10-30 — E51: cổng ORACLE — pixel nền CÓ THỂ tác động DINO-Swin-L không? White-box trên CHÍNH DINO-Swin-L theo vùng GT box @ dev30 B20 (user: "approve", ~11:56 GMT+7) — CHỈ PHÂN TÍCH (như E8), khai báo TRƯỚC khi chạy
**Vì sao:** E50 — δ_RD ở nền gần như không tác động Swin-L (share 0.02), nhưng đó là δ tối ưu cho R50; chưa biết pixel nền CÓ THỂ tác động Swin-L (qua attention / ngữ cảnh) hay không. Dùng gradient target CHỈ để đo khả năng (không vào method, không vi phạm threat model của attack transfer).
**Ô (`scripts/e51_wb_bgmask.py`, PNG `artifacts/e51/`):** loss = số hạng phân loại không-dn của DINO (như E8 WB-s1), ascent, MI μ = 1, α = 1, B20, ε = 5; δ ← δ·mask mỗi bước, mask = vùng E50: WB-FULL / WB-IN / WB-OUT / WB-RING. Eval DINO-Swin-L: AP drop + ASR (suppression). Smoke 1 ảnh OK (61 s / ảnh cho 4 ô, peak 9.8 GB; ước ~35 phút dev30).
**Đọc (khai báo trước, điểm):** share = drop(ô) / drop(WB-FULL).
- **CÓ KÊNH NỀN** nếu drop(WB-OUT) ≥ 15 VÀ share_OUT ≥ 0.3 → pixel nền tác động được Swin-L khi nhắm đúng → entry riêng: tìm loss chỉ-R50 cho vùng nền (khóa trước).
- **KHÔNG KÊNH** nếu drop(WB-OUT) ≤ 5 → pixel nền vô giá trị với Swin-L ở ε = 5 → đóng hướng background cho method (giữ cho phần giải thích paper).
- Còn lại → XÁM. Mô tả: WB-RING (ngữ cảnh sát object) vs phần nền xa; ASR từng ô (mất object thật hay chỉ box giả).
**Caveat:** dev30; B20 (ít hơn E8 WB-OSFD B50); loss cls tạo nhiều FP (WB-probe 10-18) → drop của WB-OUT có thể đến từ box giả trên nền, đọc kèm ASR.
**Chạy:** `results/e51/run.sh`, log `results/e51/run.log`.

### Kết quả E51 @ dev30 B20 (3090, xong 12:30 GMT+7; `results/mechanism/e51_wb_bgmask.json`, PNG `artifacts/e51/`, log `results/e51/run.log`) — theo quy tắc: **CÓ KÊNH NỀN** (WB-OUT drop 66.3 ≥ 15, share 0.69 ≥ 0.3)
White-box DINO-Swin-L, AP drop / ASR (share so FULL): **FULL 96.0 / 69.9**; IN 90.6 / 62.8 (0.94); **OUT 66.3 / 32.9 (0.69)**; **RING 48.8 / 26.1 (0.51)** — diện tích IN 0.36 / OUT 0.64 / RING 0.17 (E50).
So transfer RD (E50, cùng vùng): FULL 30.6 / 26.2; IN 26.0; **OUT 0.5 / 3.0; RING 0.4 / 2.2**.
**Đọc:** (1) DINO-Swin-L RẤT nhạy với pixel nền khi nhắm đúng: chỉ nhiễu NGOÀI box (không chạm pixel object) white-box đã hạ AP 66% và làm MẤT 33% object (ASR — không chỉ box giả); riêng vành sát object (17% ảnh) 49 / 26 — mạnh hơn TOÀN BỘ attack transfer RD hiện tại (30.6 / 26.2). Kênh ngữ cảnh (attention đọc quanh object) là thật. (2) δ_RD ở cùng vùng gần 0 → pixel nền của attack R50 hiện tại dùng sai mục tiêu cho Swin-L: chúng bị L_OSFD trên feature NỀN của R50 kéo đi tạo box giả cho CNN (E50), không đánh vào object qua ngữ cảnh. (3) Đây là trục CHƯA khai thác duy nhất có bằng chứng oracle mạnh. Câu hỏi tiếp: R50 có kênh ngữ cảnh CHUYỂN ĐƯỢC không — cổng rẻ: attack R50 với δ chỉ ở OUT / RING, loss CHỈ trên object (L_OSFD giới hạn token trong box + CE RoI dương) → pixel nền chỉ nhận gradient qua ảnh hưởng ngữ cảnh lên object; so RD-OUT 0.5 trên Swin-L. Caveat: WB B20, loss cls (tạo FP) → drop có phần box giả; dev30.

---

## 2026-10-30 — E52: kênh NGỮ CẢNH có chuyển từ R50 sang DINO-Swin-L không? δ chỉ ở nền (OUT) / vành (RING), loss RD gốc vs loss CHỈ object @ dev30 B36 (user: "approve", ~12:36 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** E51 — white-box chỉ ngoài box hạ Swin-L 66 (ASR 33), vành 49 / 26; δ_RD cùng vùng ≈ 0.5 (E50) vì L_OSFD trên feature NỀN của R50 kéo pixel nền đi tạo box giả cho CNN. Giả thuyết: nếu pixel nền chỉ nhận gradient qua ảnh hưởng NGỮ CẢNH lên object của R50, hướng đó có thể chuyển sang Swin-L.
**Ô (RD: OCT 1V, MI, WP ρ 0.002, AMP, B36, ε 5, λ = 6 giữ nguyên; vùng theo box GT của data_sample quy về ảnh gốc, nới 25% như E50 — khác E50 nhỏ: E50 dùng mọi annotation COCO kể cả crowd):** `RD-OUTF` / `RD-RINGF` = δ chỉ ở OUT / RING (gradient × mask trước chuẩn hóa), loss RD gốc; `RD-OUTO` / `RD-RINGO` = cùng vùng, loss CHỈ object: L_OSFD C3–C5 chỉ token giao box (chia tổng token → mỗi token trong box như gốc, `repair.make_view_loss(box_feat)`) + CE RoI chỉ RoI dương (`roi_pos`). Smoke dev5 B3: δ ngoài vùng = 0 tuyệt đối; OUTO phủ 43–96% vùng (pixel xa object không nhận gradient).
**Mốc:** RD-OUT / RD-RING hậu xử lý (E50): Swin-L 0.5 / 0.4.
**Quy tắc (dev30, điểm, DINO-Swin-L drop):** **CHUYỂN ĐƯỢC** nếu max(OUTO, RINGO) ≥ 5 VÀ ô đó − ô -F cùng vùng ≥ +2 → loss chỉ-object là đòn bẩy; entry riêng thiết kế method chia vai trò theo vùng (trong box RD, vành loss ngữ cảnh). **KHÔNG** nếu cả 4 ô ≤ 3 → kênh ngữ cảnh đặc thù Swin-L, R50 không với tới; đóng hướng background cho method. Còn lại → XÁM (vd -F cũng ≥ 5: chỉ việc tối ưu CÓ mask đã đủ). Mô tả: panel 9 model, ASR, FP nền.
**Caveat:** dev30 (nhiễu Swin-L ~±5 giữa máy, ~±1 cùng máy); WB E51 dùng loss DINO, B20.
**Chạy:** `results/e52/run.sh`, log `results/e52/run.log`.

### Kết quả E52 @ dev30 B36 (3090, xong 12:50 GMT+7; `results/e52/panel_dev30.json`, `results/runs/dev300_B36_eps5_limit100/*_e52_dev30.json`, log `results/e52/run.log`) — theo quy tắc: **KHÔNG** (cả 4 ô ≤ 3 trên DINO-Swin-L)
DINO-Swin-L drop: RD-OUTF **1.9**, RD-OUTO **1.3**, RD-RINGF **1.0**, RD-RINGO **1.1** (mốc hậu xử lý RD-OUT 0.5 / RD-RING 0.4; oracle WB E51 OUT 66.3 / RING 48.8).
Drop CNN (R101 / ConvNeXt / Swin-T / FCOS / DINO-R50 / YOLOv3 / YOLOX-L): OUTF 31.3 / 26.6 / 20.3 / 35.8 / 33.9 / 18.7 / 19.0; OUTO 24.1 / 23.4 / 15.6 / 32.3 / 25.9 / 19.2 / 15.5; RINGF 22.1 / 20.2 / 11.0 / 31.7 / 19.7 / 12.0 / 10.2; RINGO 19.0 / 16.7 / 12.7 / 26.0 / 17.8 / 15.0 / 7.9 (hậu xử lý RD-OUT: 25.5 / 20.2 / 18.1 / 29.2 / 26.0 / 20.4 / 11.6).
**Đọc:** (1) Kênh ngữ cảnh ĐẶC THÙ Swin-L / DINO: oracle chỉ-nền 66% nhưng mọi tối ưu R50 trên cùng vùng ~1–2% — R50 không với tới, kể cả khi pixel nền chỉ nhận gradient qua ảnh hưởng ngữ cảnh lên object (-O). (2) Tối ưu CÓ mask giúp CNN (+5–7 so hậu xử lý), loss chỉ-object làm CNN yếu đi (mất kênh box giả nền). (3) Khớp E19 / E24 / E36: hướng nhạy của Swin-L vô hình với R50 — cả trực tiếp lên object lẫn qua ngữ cảnh. **Đóng hướng background cho method.** Giữ cho paper (RQ2): DINO-Swin-L dựa mạnh vào ngữ cảnh (oracle), ngữ cảnh đó không chuyển giao từ CNN; pixel nền của attack R50 chỉ phục vụ CNN qua box giả (E50).

---

## 2026-10-30 — E53: giả thuyết "SAI THỨ TỰ ĐỘ DỐC" (RQ2 → method) — tại ảnh sạch, hướng chung δ_S thoải trên R50 nhưng mạnh trên DINO-Swin-L, còn δ_RD ngược lại? @ dev30 (user: "Thử nghiệm hướng 1 trước", ~13:01 GMT+7) — MÔ TẢ / sinh cơ chế, khai báo TRƯỚC khi đọc số
**Vì sao (thảo luận với user):** "CNN không biểu diễn được" là sai với data — δ_S (sinh trên Mask R-CNN Swin-T) phá R50 83% (E20 B0), nhưng 12 bước gradient R50 kéo δ rời vùng chung (59 → 35 Swin-L). Giả thuyết H_steep: gap = R50 xếp hướng ĐẶC THÙ CNN dốc hơn hướng chung (do inductive bias: khóa lưới E13, texture cục bộ E6 / E7, …), gradient ascent tham lam chọn hướng dốc nhất → cực đại riêng R50. Nếu đúng: method = làm phẳng hướng dốc do bias CNN để hướng chung thành dốc nhất, và E53 cho biết CẦN làm phẳng bao nhiêu.
**Đo (`scripts/e53_steepness.py`, dev30, tại ảnh sạch x):** δ_X = PNG − x: S = C2-OCT-1V sur Swin-T B41 (sinh lại trên 3090; ngoài threat model, chỉ phân tích); RD = RD B36 (E46); C = C2-OCT-1V B36 (E48); W = WB-FULL DINO-Swin-L (E51); RAND = ±5 Rademacher. Mỗi model M ∈ {R50 (L = L_OSFD k = 3 + 4·C2 cls, view đồng nhất, như E32 / E38), DINO-Swin-L (loss cls không-dn, chỉ phân tích)}: **G_M(X) = L_M(x + δ_X) − L_M(x)** (lợi thật); r_M(X) = g_M·δ_X / (ε‖g_M‖₁) (bậc 1, mô tả). Thêm AP drop Swin-L của δ_S trên máy này (mô tả).
**Quy tắc (paired bootstrap 1000 theo ảnh, tỉ số trung bình):** **H_steep ỦNG HỘ** nếu G_R50(RD) / G_R50(S) ≥ 1.5 (cận dưới CI > 1) VÀ G_Swin-L(S) / G_Swin-L(RD) ≥ 1.5 (cận dưới CI > 1). **BÁC** nếu G_R50(S) ≥ G_R50(RD). Còn lại XÁM. Mô tả: "hệ số làm phẳng cần thiết" = G_R50(RD) / G_R50(S); r theo model; W, C, RAND.
**⚠️ Minh bạch:** bản khóa đầu (trong script, chưa commit) dùng r (bậc 1) làm quy tắc; smoke 2 ảnh (`--limit 2`) cho thấy r trên DINO ~0.016 với cả S và RD — đã biết từ E38 (cos(g_R50, g_DINO) ~0.012) nên r không phân biệt được trên Swin-L → đổi quy tắc sang G TRƯỚC khi chạy đủ 30 ảnh. Đã nhìn số smoke 2 ảnh (G Swin-L S/RD ~2.8, R50 RD/S ~2.0) — n = 2, không đọc được, nhưng ghi nhận là thông tin có trước khi khóa.
**Caveat:** dev30; δ_S tối ưu trên Swin-T (cùng họ, khác detector); loss đo ≠ loss sinh từng δ (RD tối ưu L_OSFD C3–C5 + 6·roi, đo bằng C2 4·); đo tại ảnh sạch, 1 điểm.
**Chạy:** `results/e53/run.sh`, log `results/e53/run.log`.

### Kết quả E53 @ dev30 (3090, xong 13:04 GMT+7; `results/mechanism/e53_steepness.json`, `results/runs/dev300_B41_eps5_limit100_sur-swin_t/*_e53_dev30.json`, log `results/e53/run.log`) — theo quy tắc: **H_steep ỦNG HỘ**
δ_S trên máy này: R50 / R101 / ConvNeXt / Swin-T 84.7 / 79.5 / 87.9 / 95.6, **DINO-Swin-L 54.2** (V100 E20: 59.3).
Lợi thật G (TB 30 ảnh) — S / RD / C / W / RAND: **R50** 6.78 / 12.71 / 15.44 / 2.50 / 1.32; **DINO-Swin-L** 2.85 / 1.35 / 1.11 / 36.66 / −0.03. Tỉ số: **G_R50(RD)/G_R50(S) 1.88 [1.76, 1.99]**; **G_Swin-L(S)/G_Swin-L(RD) 2.11 [1.47, 2.98]**. Bậc 1 r (mô tả): R50 S 0.048 / RD 0.146 / C 0.139 / W 0.012; DINO S 0.014 / RD 0.008 / C 0.006 / W 0.465.
**Đọc:** (1) Thứ tự độ dốc ĐẢO giữa hai họ: hướng R50 tạo lợi loss R50 gấp ~1.9× hướng chung, hướng chung tạo lợi Swin-L gấp ~2.1× hướng R50. R50 vẫn "thấy" hướng chung (G_R50(S) 6.8 ≫ RAND 1.3; AP drop R50 84.7%) — không phải không biểu diễn được. (2) **WP là một bước làm phẳng một phần đúng theo H_steep:** C → RD: G_R50 15.4 → 12.7 (−18%), G_Swin-L 1.11 → 1.35 (+22%); δ_S ở xa hơn trên cùng trục (G_R50 −56% so C, G_Swin-L +158%). (3) Hệ số làm phẳng cần thêm ≈ 1.9× trên lợi R50 so RD. (4) R50 bão hòa AP: δ_S với nửa lợi loss đã cho 85% AP drop R50 → phần lợi loss vượt của RD gần như là "overkill" đặc thù R50. (5) Hệ quả method: cần một đại lượng CHỈ-R50 tách S khỏi RD (hướng chung có tính chất gì trên R50?) để làm regularizer — E24 trước chỉ so các δ sinh bởi R50, chưa có δ_S làm mẫu dương.
**Caveat:** dev30; 1 điểm (ảnh sạch); loss đo C2 4· khác loss sinh RD; δ_S ngoài threat model (chỉ phân tích).

---

## 2026-10-30 — E54: tính chất CHỈ-R50 nào tách hướng chung δ_S khỏi hướng R50 δ_RD? (RQ2 -> regularizer method) @ dev30, forward/backward R50 (user: "Approve viết tiếp", ~13:14 GMT+7) — MÔ TẢ / sinh giả thuyết method, khai báo TRƯỚC khi đọc đủ
**Vì sao:** E53 H_steep ỦNG HỘ — gap = sai thứ tự độ dốc, WP là bước làm phẳng một phần trên trục C -> RD -> S. Cần 1 đại lượng CHỈ-R50 phân biệt S (hướng chung, transfer Swin-L cao) với RD (hướng R50) để làm regularizer; E24 trước thất bại vì chưa có δ_S làm mẫu dương. Giờ có δ_S + trục C -> RD kiểm hướng.
**Đo (`scripts/e54_property_probe.py`, dev30, R50, δ_X = PNG − ảnh gốc, X in {C = C2-OCT-1V B36 E48, RD = RD B36 E46, S = C2-OCT-1V sur Swin-T B41}):** P1 deep = D_C5/ΣD_s (D_s = distortion feature stage s); P2 r_def = G(θ' WP)/G(θ0); P3 view_rob = TB_v Dfeat(view) / Dfeat(id), v in {co 0.8, giãn 1.25, dịch 4px}; P4 scramble = Dfeat(δ xáo ô 16px)/Dfeat(δ); P5 boxconc = năng lượng distortion C4 trong box / tỉ lệ token; P6 clsfrac = G_cls/G. (S ngoài threat model, chỉ phân tích.)
**Quy tắc:** tính chất là **ỨNG VIÊN regularizer** nếu ĐƠN ĐIỆU C -> RD -> S VÀ |bước S−RD| > |bước RD−C| VÀ CI(S−RD) loại 0. Mô tả các tính chất khác.
**⚠️ Minh bạch:** đã xem smoke 3 ảnh (`--limit 3`, KHÔNG đọc kết luận): scramble C 0.28 / RD 0.29 / S 0.47 nổi bật nhất (khớp E7/E13: δ_R50 là mẫu tần cao khớp CHÍNH XÁC nội dung cục bộ + khóa lưới); deep, view_rob cũng đơn điệu. Ghi nhận là thông tin có trước khi khóa; quy tắc trên khóa trước khi chạy đủ 30.
**Caveat:** dev30; δ_S ngoài threat model; 1 điểm (ảnh sạch); scramble giữ nội dung trong ô nhưng xáo vị trí ô (thô, không phải phép bất biến chuẩn).
**Chạy:** `results/e54/run.sh`, log `results/e54/run.log`.

### Kết quả E54 @ dev30 (3090, xong 13:15 GMT+7; `results/mechanism/e54_property_probe.json`, log `results/e54/run.log`) — theo quy tắc: **3 ỨNG VIÊN** (scramble mạnh nhất), 3 không
Tính chất — C / RD / S | bước RD−C | bước S−RD [CI]:
- **scramble (Dfeat sau xáo ô 16px / Dfeat gốc): 0.255 / 0.264 / 0.411 | +0.009 | +0.147 [+0.134, +0.159]** -> ỨNG VIÊN (bước S−RD gấp ~16× RD−C). δ_S giữ 41% tác dụng feature khi xáo ô, δ_RD chỉ 26% -> δ_R50 phụ thuộc VỊ TRÍ cục bộ chính xác, δ_S KHÔNG.
- **view_rob (Dfeat view / Dfeat id): 0.959 / 0.961 / 0.977 | +0.003 | +0.016 [+0.011, +0.020]** -> ỨNG VIÊN (bền view hơn; khớp OCT).
- **deep (D_C5/ΣD_s): 0.452 / 0.441 / 0.407 | −0.011 | −0.034 [−0.047, −0.021]** -> ỨNG VIÊN (S distortion NÔNG hơn; lưu ý ngược chiều E32 "WP dời về sâu" — ở đây trục tới S là nông hơn, WP chỉ nhích).
- r_def: 0.763 / 0.864 / 0.771 -> KHÔNG đơn điệu (RD đỉnh; r_def là thứ WP tối ưu trực tiếp, không phải trục tới S).
- boxconc: 1.067 / 1.053 / 1.032 -> đơn điệu nhẹ, CI chứa 0.
- clsfrac: 0.412 / 0.314 / 0.293 -> đơn điệu (S ít phần cls hơn) nhưng bước S−RD CI chứa 0.
**Đọc:** (1) **scramble là ứng viên trội**: chênh S−RD +0.147 (CI chặt), gấp ~16× bước RD−C, khớp cơ chế E7 (δ_R50 mẫu tần cao khớp nội dung) + E13 (khóa lưới stride 4). δ chuyển giao Swin-L bền với xáo vị trí cục bộ; δ R50 thì không. (2) view_rob, deep cùng hướng nhưng biên độ nhỏ. (3) **Hướng method (regularizer suy từ cơ chế):** phạt phần δ phụ thuộc vị trí cục bộ trên R50 = ép G_feat bền với xáo ô. Ứng viên loss: tại mỗi bước, cực đại hóa Dfeat của CẢ δ gốc LẪN δ xáo ô (hoặc min khoảng cách giữa hai), kéo δ về vùng bền-xáo như δ_S. KHÁC OCT (xáo PHÁ cấu trúc cục bộ thay vì biến đổi hình học toàn cục) -> novelty. (4) Caveat: xáo ô là phép thô; cần thiết kế loss khả vi (xáo ô không khả vi theo δ — dùng 2 forward: δ và detach(xáo δ), hoặc regularizer trên tự tương quan không gian của gradient). Khóa entry method trước khi code.

---

## 2026-10-30 — E55: METHOD mới RD-SCR — regularizer "bền xáo ô" suy từ cơ chế (E53 H_steep + E54 scramble) @ dev30 B36 (user: "đồng ý", ~13:20 GMT+7) — khóa TRƯỚC khi chạy
**Chuỗi suy luận (novelty):** E53 — gap = SAI THỨ TỰ ĐỘ DỐC (hướng R50 dốc hơn hướng chung ~1.9× trên loss R50, hướng chung mạnh hơn ~2.1× trên Swin-L); WP là bước làm phẳng một phần trên trục C→RD→S. E54 — đại lượng CHỈ-R50 tách δ_S khỏi δ_RD là **scramble** (δ_S bền xáo ô 0.41 vs δ_RD 0.26; khớp E7/E13: δ_R50 là mẫu tần cao khớp VỊ TRÍ cục bộ + khóa lưới). → regularizer ép δ bền với xáo ô vị trí ⇒ kéo δ về vùng ít phụ thuộc vị trí cục bộ như δ_S.
**Số hạng (khả vi, hoán vị ô là reindex cố định; `pia._scramble`, `_with_scramble`):** mỗi view, mỗi bước: + λ·L_OSFD(c + scramble(v − c)) = δ của view bị xáo ô tile×tile (hoán vị MỚI mỗi bước, gen seed + 13) phải vẫn tạo distortion feature như δ gốc trên CÙNG nền sạch c. Khác OCT (biến đổi hình học TOÀN CỤC) — đây là phá cấu trúc VỊ TRÍ cục bộ của δ.
**Ô (nền RD: OCT 1V, MI, WP ρ 0.002, AMP, B36, λ_cls = 6; tile = 16 px khung model):** `C2-OCT-1V-WP2-RD-SCR` λ = 1 (= trọng số L_OSFD chính, không tune); `-SCR05` λ = 0.5; `-SCR2` λ = 2. Smoke dev5 B3 OK.
**Mốc:** RD B36 + OSFD B50 (PNG E46), eval lại cùng lần.
**Quy tắc (dev30, điểm; cùng máy 3090 ~tất định nên đọc được ~±3; CI mô tả):** **CÓ TÍN HIỆU** (đi tiếp dev100 cùng-thời-gian) nếu tồn tại λ: ΔSwin-L(SCR − RD) ≥ +3 VÀ ΔCA3(ConvNeXt/Swin-T/YOLOX-L) ≥ −2 VÀ ΔR101 ≥ −3. **KHÔNG** nếu mọi λ có ΔSwin-L ≤ +1. Còn lại XÁM. Mô tả: thời gian/ảnh (số hạng thêm 1 forward+backward backbone → kỳ vọng ~1.5×; nếu có tín hiệu PHẢI so cùng-thời-gian ở entry sau), panel đủ, ASR.
**Caveat:** dev30; λ không tune (3 mức khai báo trước); scramble whole-δ (cả nền + object) — biến thể chỉ-trong-box để sau; chưa so cùng-thời-gian (B36 = thời gian OSFD B50 cho RD, SCR chậm hơn).
**Chạy:** `results/e55/run.sh`, log `results/e55/run.log`.

### Kết quả E55 @ dev30 B36 (3090, xong 13:33 GMT+7; `results/e55/panel_dev30.json`, `results/runs/dev300_B36_eps5_limit100/*_e55_dev30.json`, log `results/e55/run.log`) — theo quy tắc: **KHÔNG** (mọi λ ΔSwin-L ≤ +1)
DINO-Swin-L drop / ASR | T8 drop / ASR (mốc RD 30.6 / 26.2 | 79.0 / 66.0):
- RD-SCR λ=1: **31.6** / 28.3 | 78.7 / 64.7 — ΔSwin-L **+1.0 [−4.0, +3.5]**
- RD-SCR λ=0.5: **31.6** / 25.2 | 78.2 / 65.4 — ΔSwin-L **+1.0 [−2.9, +2.9]**
- RD-SCR λ=2: **30.0** / 24.2 | 77.5 / 61.7 — ΔSwin-L **−0.5 [−3.2, +2.0]**
Thời gian: RD-SCR 3.4 s/ảnh vs RD 2.7 (×1.26; thêm 1 extract_features/bước). CNN (T8 drop) tụt nhẹ theo λ (78.7 → 77.5).
**Đọc:** (1) Regularizer "bền xáo ô" KHÔNG dịch được Swin-L (ΔSwin-L ≤ +1 mọi λ, CI chứa 0) dù E54 cho thấy scramble là đại lượng tách δ_S/δ_RD rõ nhất. → **việc δ_RD BỀN HƠN với xáo ô KHÔNG kéo nó về vùng transfer của δ_S.** Scramble là CHỈ BÁO (phân biệt δ đã có) chứ không phải ĐÒN BẨY nhân quả — tối ưu để bền xáo ô chỉ làm δ "trải đều" hơn (giảm nhẹ CNN), không tạo được hướng ngữ cảnh/hình-dạng mà Swin-L cần (E51: kênh đó đặc thù Swin-L, R50 không với tới — E52). (2) Khớp bài học xuyên suốt (E19/E24/E36/E52): mọi đại lượng CHỈ-R50 "giống δ_S" đều là tương quan, không phải kênh nhân quả tới Swin-L; ép R50 khớp 1 thống kê của δ_S không sinh ra hướng Swin-L. (3) **Giá trị cho paper (RQ2):** E53+E54 là bằng chứng ĐỊNH LƯỢNG rằng gap = sai thứ tự độ dốc + δ_R50 phụ thuộc vị trí cục bộ; E55 đóng giả thuyết "sửa bằng regularizer bền-xáo trên R50". Novelty còn lại là PHÂN TÍCH, không phải method vượt trần.
**Hệ quả:** đóng hướng regularizer-từ-scramble. Trần Swin-L ~30–38 (tùy máy) với 1 surrogate R50 đứng vững qua mọi hướng đã thử. Method chốt vẫn là RD / RD-CV (E46). Việc tiếp (user quyết): chốt method + chạy dev300 cùng máy (RD / RD-CV / OSFD / MI / M-DI²), hoặc định vị lại paper quanh RQ1 + cơ chế (E41/E50/E51/E53/E54).

---

## 2026-10-30 — E56: RD-CWT — view co-phóng theo KHỐI (CWT, arXiv 2501.11901) trên nền RD @ dev30 B36 (user: "code đi", ~13:51 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao (literature + cơ chế):** CWT (arXiv 2501.11901, CNN surrogate, single-surrogate, chỉ đụng input): co từng khối rồi phóng lại -> phá high-freq CỤC BỘ; hơn BSR ~6–7% trên ViT. Khớp E54 (δ_S bớt phụ thuộc lưới pixel) nhưng qua ĐƯỜNG ĐẦU VÀO (họ OCT đã thắng +9 Swin-L) thay vì regularizer δ (E55 — chết). Giả thuyết: buộc δ sống sót qua lấy mẫu lại theo khối -> đẩy δ ra khỏi mẫu khóa-vị-trí đặc thù R50, về phía δ_S, mà không vấp đường nhân quả đã chết của E55.
**Ô (nền RD: OCT 1V, MI, WP ρ 0.002, AMP, B36, λ_cls = 6; `pia._cwt_spec/_cwt_apply`, áp CÙNG spec cho ảnh sạch + adv của view, hệ số co mới mỗi bước seed + 15):** `C2-OCT-1V-WP2-RD-CWT` n = 2 (lưới 2×2, s_max = 1.3 như paper); `-CWTN3` n = 3. Bỏ xoay khối (paper có) để giữ vị trí khối -> box GT còn đúng cho L_cls (ghi nhận: biến thể có xoay để sau nếu có tín hiệu). Smoke dev5 B3 OK, δ đúng L∞.
**Mốc:** RD B36 + OSFD B50 (PNG E46), eval lại cùng lần.
**Quy tắc (dev30, điểm, cùng máy 3090 ~tất định nên đọc ~±3; CI mô tả):** **CÓ TÍN HIỆU** (đi tiếp dev100 cùng-thời-gian) nếu tồn tại n: ΔSwin-L(CWT − RD) ≥ +3 VÀ ΔCA3(ConvNeXt/Swin-T/YOLOX-L) ≥ −2 VÀ ΔR101 ≥ −3. **KHÔNG** nếu mọi n có ΔSwin-L ≤ +1. Còn lại XÁM. Mô tả: thời gian/ảnh (CWT thêm interpolate rẻ), panel đủ, ASR.
**Caveat:** dev30; s_max không tune (1.3 theo paper); bỏ xoay; chưa so cùng-thời-gian.
**Chạy:** `results/e56/run.sh`, log `results/e56/run.log`.

### Kết quả E56 @ dev30 B36 (3090, xong 14:00 GMT+7; `results/e56/panel_dev30.json`, `results/runs/dev300_B36_eps5_limit100/*_e56_dev30.json`, log `results/e56/run.log`) — theo quy tắc: **CÓ TÍN HIỆU** (n=3: ΔSwin-L +4.0 điểm, CNN không tụt) — tín hiệu DƯƠNG sạch đầu tiên
DINO-Swin-L drop/ASR | Conv / Swin-T / YOLOX-L / R101 drop | T8 drop/ASR (mốc RD 30.6):
- RD: 30.6/26.2 | 85.8 / 85.1 / 70.5 / 93.0 | 79.0/66.0
- RD-CWT n=2: 31.8/25.6 | 85.6 / 86.7 / 72.0 / 94.0 | 80.0/66.9 — ΔSwin-L **+1.2 [−3.2, +4.3]**
- **RD-CWT n=3: 34.6/26.4 | 87.0 / 87.8 / 71.7 / 93.5 | 80.8/66.3 — ΔSwin-L +4.0 [−0.2, +5.8]**
Thời gian 2.7 s/ảnh = RD (CWT chỉ thêm interpolate, gần như miễn phí) -> KHÁC các hướng trước, KHÔNG tốn thêm budget.
**Theo quy tắc:** n=3 ΔSwin-L +4.0 ≥ +3 (điểm) ✓; CA3 (Conv +1.2, Swin-T +2.7, YOLOX-L +1.2) ≥ −2 ✓; R101 +0.5 ≥ −3 ✓ -> **CÓ TÍN HIỆU**. CI cận dưới −0.2 (chạm 0, dev30 nhiễu) nên chưa xác nhận CI — đúng lý do quy tắc đòi dev100.
**Đọc:** (1) Lần đầu một method single-surrogate SẠCH nhích Swin-L +4 mà KHÔNG hại CNN (CNN còn tăng nhẹ: Conv +1.2, Swin-T +2.7) và KHÔNG tốn thêm thời gian. (2) **Phụ thuộc liều rõ: n=2 +1.2 < n=3 +4.0** -> khối mịn hơn = hiệu ứng mạnh hơn -> dấu hiệu CƠ CHẾ THẬT (không phải nhiễu). (3) Khớp giả thuyết khai báo: đẩy δ bớt phụ thuộc lưới pixel qua ĐƯỜNG ĐẦU VÀO (OCT-family) làm được cái regularizer-δ E55 không làm được -> xác nhận "input-diversity là đòn bẩy, regularizer-δ là chỉ báo". (4) ASR ~ngang RD (26.4 vs 26.2). **VIỆC TIẾP:** (a) quét liều n ∈ {4, 6} + biến thể có xoay khối @ dev30 (rẻ, xác nhận đơn điệu); (b) nếu còn tăng -> dev100 cùng-thời-gian (CWT = RD về thời gian nên B36 so trực tiếp) + OSFD cùng máy. Caveat: dev30, CI chạm 0, 1 máy.

---

## 2026-10-30 — E57: quét liều CWT n ∈ {4, 6} + biến thể XOAY khối (n=3, ±15°) — xác nhận đơn điệu tín hiệu E56 @ dev30 B36 (user: "Đồng ý luôn", ~14:07 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** E56 — RD-CWT có tín hiệu, phụ thuộc liều n=2 (+1.2) < n=3 (+4.0). Quét n cao hơn để xác nhận đơn điệu (cơ chế thật, CI tách 0 ở liều cao). Thêm biến thể xoay khối (CWT gốc) đã bỏ ở E56.
**Ô (nền RD):** `-CWTN4` (n=4), `-CWTN6` (n=6), `-CWTN3R` (n=3 + xoay khối ±15°). Mốc RD + CWTN3 (PNG E56), eval lại cùng lần. Smoke dev5 B3 OK.
**Quy tắc (dev30, điểm):** **ĐƠN ĐIỆU XÁC NHẬN** nếu ΔSwin-L tăng tiếp n=3 -> n=4 (hoặc n=6) với ΔSwin-L(tốt nhất − RD) ≥ +5 VÀ CA3 ≥ −2 VÀ R101 ≥ −3 -> chọn n tốt nhất, lên dev100 cùng-thời-gian + OSFD cùng máy. **BÃO HÒA** nếu n ≥ 4 không hơn n=3 (vẫn giữ tín hiệu n=3). **MẤT** nếu n cao làm Swin-L tụt về ≤ RD+1 (tín hiệu n=3 là nhiễu). Xoay (CWTN3R): mô tả, đọc kèm caveat box GT lệch.
**Caveat:** dev30, CI chạm 0; n rất cao -> khối nhỏ, co-phóng mất nhiều nội dung (có thể hại white-box R50 -> đọc kèm R50); xoay làm box GT lệch nhẹ (nhiễu cho L_cls).
**Chạy:** `results/e56/run_sweep.sh`, log `results/e56/run_sweep.log`.

### Kết quả E57 @ dev30 B36 (3090, xong 14:19 GMT+7; `results/e56/panel_sweep_dev30.json`, `results/runs/dev300_B36_eps5_limit100/*_e57_dev30.json`, log `results/e56/run_sweep.log`) — theo quy tắc: **BÃO HÒA** (đỉnh n=3; giữ tín hiệu n=3)
ΔSwin-L vs RD: n=2 +1.2, **n=3 +4.0 [−0.2, +5.8]**, n=4 +1.5 [−1.8, +3.7], n=6 −0.8 [−4.4, +3.4]. R50 WB ≈ 98 mọi n (nội dung không mất trên R50). Xoay (n=3, ±15°): R50 WB **93.1** (δ yếu đi trên surrogate), Swin-L 24.1, T8 71.1 — HẠI mọi target.
**Đọc:** (1) Đường liều **chữ U ngược, đỉnh n=3** (giống quét ρ WP E38) -> độ hạt khối tối ưu tồn tại -> **cơ chế thật**, không nhiễu. (2) **Xoay khối = đường chết** (box GT lệch + dời nội dung phá cả white-box) -> bỏ. (3) Method chốt: **RD + CWT(n=3)**, Swin-L 30.6 -> 34.6 (3090), 0 thêm thời gian, không hại CNN. KHÔNG phải nhảy tới 50 nhưng là cải tiến thật, grounded, miễn phí. (4) +4 với CI chạm 0 -> dev30 không phân giải -> **BẮT BUỘC dev100** để co CI. VIỆC TIẾP: dev100 cùng máy RD-CWTN3 + RD + OSFD (CWT = RD thời gian -> so B36 trực tiếp); nếu ΔSwin-L CI loại 0 -> method novelty thật.

---

## 2026-10-30 — E58: XÁC NHẬN RD-CWT(n=3) @ dev100 cùng máy 3090 (bản gọn) — RD-CWTN3 vs RD vs OSFD (user: "approve", ~14:57 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** E56/E57 — RD-CWT(n=3) ΔSwin-L +4.0 nhưng CI dev30 chạm 0 [−0.2, +5.8]; chữ U ngược theo n (cơ chế thật). dev100 co CI ~±2.5 -> phân giải được +4. Đây là method single-surrogate SẠCH duy nhất có tín hiệu dương; xác nhận trước khi đầu tư full panel / dev300.
**Ô (dev100, 100 ảnh):** RD-CWTN3 (B36), RD (B36), OSFD (B50) — sinh lại cùng máy cùng lần (không trộn PNG V100). Eval gọn: Controlled (r50/r101/convnext_t/swin_t) + DINO-Swin-L + YOLOX-L. CWT = RD về thời gian (E56: 2.7 s/ảnh) -> so B36 trực tiếp.
**Quy tắc (dev100, paired bootstrap 1000 so RD):** **XÁC NHẬN** nếu ΔSwin-L(CWTN3 − RD) > 0 với cận dưới CI95 > 0 VÀ CA3 (ConvNeXt/Swin-T/YOLOX-L) ≥ −2 (điểm) VÀ R101 ≥ −3 -> RD-CWT(n=3) là method chốt; chạy full panel + dev300. **KHÔNG** nếu cận dưới CI ≤ 0 -> tín hiệu dev30 không vững; RD / RD-CV là method chốt. Mô tả: drop + ASR + APloc so OSFD.
**Caveat:** dev100 lần đầu dùng phiên này (chưa "đốt" như dev30); 1 máy; bản gọn (full panel sau nếu xác nhận).
**Chạy:** `results/e58/run.sh`, log `results/e58/run.log`.

### Kết quả E58 @ dev100 B36 (3090, xong 15:17 GMT+7; `results/e58/panel_dev100.json`, `results/runs/dev300_B{36,50}_eps5_limit100/*_e58_dev100.json`, log `results/e58/run.log`) — theo quy tắc: **CWT KHÔNG xác nhận**; nhưng RD − OSFD XÁC NHẬN mạnh
DINO-Swin-L drop/ASR | R101/Conv/SwinT/YOLOX-L drop (dev100, cùng máy cùng lần):
- OSFD B50: 24.0/24.8 | 91.8/78.4/75.0/69.7
- RD B36: 33.8/27.5 | 92.2/85.0/80.4/71.1
- RD-CWTN3 B36: 34.4/27.4 | 92.4/86.0/81.0/69.6
**ΔSwin-L(CWTN3 − RD) = +0.6 [−2.47, +2.43]** → cận dưới ≤ 0 → **KHÔNG**: tín hiệu +4 ở dev30 (E56/E57) KHÔNG sống sót dev100 (nhiễu dev30). CWT trung tính (Conv +1.0, Swin-T +0.6, YOLOX −1.5, đều CI chứa 0). **Đóng CWT.**
**Phát hiện chính (ngoài CWT): RD − OSFD trên Swin-L = +9.8 [+5.1, +13.1]** (OSFD − RD = −9.81 [−13.07, −5.06]) — CI loại 0 CHẮC; cũng hơn Conv +6.6 [+2.2, +11.3], Swin-T +5.4 [−0.4, +8.6]. R50 white-box ≈ nhau (96.0–96.4). **Lần đầu xác nhận dev100 cùng máy CI chặt: RD vượt OSFD rõ ở target khác họ backbone.**
**Đọc:** (1) Method chốt = **RD** (C2-OCT-1V-WP2-RD): đã validate hơn OSFD +9.8 Swin-L dev100. Con số Swin-L trung thực của RD = **33.8** (dev100 3090) — không phải 38 (V100 dev30) hay 30.6 (dev30 3090). (2) CWT: tín hiệu dev30 không tái lập → quy tắc "chỉ đọc Swin-L từ dev100" được củng cố (bài học cho mọi screen dev30 sau). (3) Khoảng cách tới 50 còn ~16 điểm = khoảng cách họ backbone (δ_S Swin-T 54–59); AWP đa bước là cú đặt cược còn lại có cơ sở nhưng tốn ngân sách B (khó thắng cùng-B). (4) **Việc tiếp (user quyết):** (a) chốt RD, chạy full panel + MI/M-DI² @ dev100/dev300 cùng máy → bảng paper; hoặc (b) thử AWP đa bước @ dev30 trước khi chốt.

---

## 2026-10-30 — E59: chênh V100 vs 3090 là PHƯƠNG SAI QUỸ ĐẠO hay hiệu ứng hệ thống? Quét seed attack trên cùng 3090 @ dev30 (user: "phân tích vì sao V100 khác 3090, có thể có idea improve cao hơn", ~15:33 GMT+7) — MÔ TẢ / chẩn đoán, khai báo TRƯỚC khi chạy
**Lập luận:** E47/E48 — chênh V100 (RD Swin-L 38) vs 3090 (31–34) KHÔNG do TF32/AMP; là kernel fp16/cuDNN khác -> lật dấu vài pixel trong bước sign -> quỹ đạo khác (như đổi seed). KHÔNG hệ thống: OSFD đi NGƯỢC (V100 22.6 < 3090 26.5). Giả thuyết: chênh máy = phương sai quỹ đạo. Hệ quả nếu đúng: không có "numerics V100 tốt" để tái tạo; nhưng nếu spread lớn -> không gian δ nhiều lòng chảo transfer khác nhau -> mở lại ensemble quỹ đạo (E49 chết vì rerun cùng máy trùng 80%; seed khác nhau = đa dạng thật, VẪN 1 R50 nên KHÔNG phải pseudo-ensemble model).
**Đo (`scripts/e59_seed_spread.py`):** RD B36 dev30 ở seed ∈ {0 (E56), 100, 200, 300} trên cùng 3090 (đổi `--seed-base` -> view/noise/WP khác). (a) spread DINO-Swin-L drop theo seed; (b) đồng thuận dấu δ giữa seed (so E49 rerun 0.80).
**Đọc (khai báo trước):** (1) **PHƯƠNG SAI** nếu range(Swin-L theo seed) ≥ 4 (cỡ chênh V100/3090) -> chênh máy chỉ là seed-variance, đóng câu hỏi "numerics V100"; đồng thời spread này = sàn nhiễu dev30 (củng cố "chỉ đọc Swin-L từ dev100"). (2) **CÓ CẤU TRÚC** nếu range < 2 -> seed không giải thích được chênh máy -> chênh V100/3090 là hệ thống theo kiến trúc GPU (hiếm, cần probe tiếp). (3) Đồng thuận dấu seed < 0.80 -> seed đa dạng hơn rerun -> nếu kèm spread lớn thì vote-nhiều-seed là lever đáng thử (entry riêng).
**Caveat:** dev30 n=30; 4 seed ít điểm; không chọn seed theo target (E24: không proxy chỉ-R50 xếp được δ theo transfer) -> lever khả dĩ là VOTE, không CHỌN.
**Chạy:** `results/e59/run.sh`, log `results/e59/run.log`.

### Kết quả E59 @ dev30 (3090, xong ~15:52 GMT+7; `results/mechanism/e59_seed_spread.json`, `results/runs/dev300_B36_eps5_limit100_seed{100,200,300}/*`, log `results/e59/run.log`) — theo quy tắc: XÁM theo ngưỡng cứng, nhưng NGHIÊNG MẠNH về **HỆ THỐNG** (chênh máy ≠ phương sai seed)
Swin-L drop RD B36 theo seed: 0→30.6, 100→27.6, 200→30.7, 300→30.8. **range 3.2, std 1.3**. Đồng thuận dấu giữa seed **0.60** (E49 rerun cùng-numerics 0.80).
**Đọc (user phản biện đúng: mỗi máy tái lập cao):** (1) 3/4 seed tụm 30.6–30.8 (lệch 0.26), seed100 outlier 27.6 → 3090 bám ~30.7; V100 bám ~38. **Chênh máy ~8 > 2× spread seed (3.2), KHÔNG seed nào chạm 38** → nếu là nhiễu quỹ đạo thì phải có seed lên ~38, không có → chênh V100/3090 là **HỆ THỐNG theo kiến trúc GPU**, không phải seed-variance. (2) Dấu δ giữa seed khác 40% mà Swin-L bất biến ~30.7 → Swin-L transfer bị quy định bởi thứ hệ thống (numerics máy), không bởi quỹ đạo cụ thể. (3) Giả thuyết SỐNG: **fp16/cuDNN V100 bơm hiệu ứng hệ thống (làm phẳng ngầm / nhiễu gradient low-precision) → +8 Swin-L, tái lập**; khớp câu chuyện WP/flatness. Nếu tái tạo được trên 3090 → improve thật +8, có lý thuyết. (4) spread seed 3.2 cũng = sàn nhiễu dev30 (củng cố "đọc Swin-L từ dev100"). **VIỆC TIẾP:** probe tái tạo — thêm nhiễu gradient / giả lập stochastic rounding fp16 khi attack RD trên 3090, quét mức, xem Swin-L có tiến về 38 không (dev30 sàng, dev100 xác nhận).

---

## 2026-10-30 — E60: TÁI TẠO cơ chế V100 — flush gradient fp16-kiểu-V100 (RD-GQ) @ dev30 B36 (user: "approve", ~16:24 GMT+7) — khóa TRƯỚC khi chạy
**Root cause đã chốt (E59 + probe δ V100):** chênh V100(38)/3090(32) KHÔNG do seed/code/eval — là **kernel fp16 Volta lúc SINH** (code cũ trên 3090 = 32.4; δ V100 eval trên 3090 = 37.7). Mổ δ: V100 vs 3090 **giống hệt mọi thống kê bậc thấp** (RMS 4.74, phổ [.36,.35,.29], mượt 5.81, bão hòa 86%) nhưng **cos chỉ 0.30, dấu khác 37%** → khác biệt là MẪU DẤU cụ thể. Giả thuyết: fp16 V100 flush thành phần gradient NHỎ (overfit R50, E53) → dấu do thành phần lớn/bền quyết định → transfer tốt hơn.
**Ô (`pia.gflush`, grad_transform trước bước sign; bỏ |g| < phân vị p theo ảnh):** `RD-GQ30/50/70` (p = 0.3/0.5/0.7), nền RD y nguyên. Mốc RD code hiện tại sinh lại CÙNG lap. Smoke dev5 OK.
**Quy tắc (dev30, điểm; mốc V100 trên Swin-L = 37.7 eval-3090):** **TÁI TẠO** nếu tồn tại p: ΔSwin-L(GQ − RD) ≥ +3 VÀ CA3 (Conv/Swin-T/YOLOX-L) ≥ −2 VÀ R50 white-box ≥ 0.8×RD → cơ chế V100 tái tạo được + method mới; BẮT BUỘC xác nhận dev100 (bài học E56→E58). **KHÔNG** nếu mọi p ΔSwin-L ≤ +1 → flush thành phần nhỏ không phải cơ chế; V100 là artefact kernel không emulate được bằng thao tác gradient hậu kỳ. Còn lại XÁM.
**Caveat:** flush phân vị trên ∇x cuối ≠ flush lúc tích lũy fp16 (xấp xỉ cơ chế, không bit-exact); dev30 (CI chạm 0 → dev100 nếu có tín hiệu); p cao có thể làm yếu δ (đọc R50 WB).
**Chạy:** `results/e60/run.sh`, log `results/e60/run.log`.

### Kết quả E60 @ dev30 B36 (3090, xong 16:34 GMT+7; `results/e60/panel_dev30.json`, `results/runs/dev300_B36_eps5_limit100/*_e60_dev30.json`, log `results/e60/run.log`) — theo quy tắc: XÁM (GQ70 +2.4 < +3) nhưng ĐƠN ĐIỆU đúng hướng cơ chế
DINO-Swin-L drop (R50 WB): RD 30.3 (98.0) | GQ30 30.8 (98.0) | GQ50 31.6 (98.8) | **GQ70 32.7 (98.2)**. ΔSwin-L: +0.5 / +1.3 / **+2.4 [−1.5,+4.2]**. CNN không tụt (YOLOX-L 71.9→75.1). Mốc tái tạo V100 = 37.7.
**Đọc:** (1) **Đơn điệu p=0.3→0.7: Swin-L 30.8→31.6→32.7** → flush thành phần gradient nhỏ CÀNG MẠNH càng tăng transfer → ỦNG HỘ giả thuyết "thành phần nhỏ = overfit R50" (E53). Chưa đỉnh. (2) R50 WB ~98 dù chỉ giữ 30% thành phần lớn (p=0.7) → phần lớn gradient thừa cho R50 (khớp E53 overkill). (3) Chưa đạt +3 và chưa tới V100 37.7 → đẩy p cao hơn (E61). (4) Khác CWT (E56, đỉnh rồi chết dev100): đây đơn điệu + có cơ chế numerics đo được → đáng đẩy tiếp. **VIỆC TIẾP (E61):** p ∈ {0.8, 0.9} xem tới 37.7 + R50 vỡ ở đâu; nếu đạt +3 → dev100 xác nhận.

---

## 2026-10-30 — E61: đẩy liều flush gradient p ∈ {0.8, 0.9} — RD-GQ tới mốc V100 37.7? @ dev30 B36 (user: "approve"/tiếp E60, ~16:36 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** E60 đơn điệu p=0.3→0.7 (Swin-L 30.8→31.6→32.7), chưa đỉnh, R50 WB ~98. Đẩy tiếp xem có tới V100 37.7 + R50 vỡ ở đâu.
**Ô:** `RD-GQ80` (p=0.8, giữ top 20% thành phần gradient), `RD-GQ90` (p=0.9, giữ top 10%). Mốc RD + GQ70 (PNG E60).
**Quy tắc (dev30, điểm):** **ĐẠT (tái tạo V100)** nếu tồn tại p: ΔSwin-L(GQ − RD) ≥ +3 VÀ R50 WB ≥ 0.8×RD VÀ CA3 ≥ −2 → dev100 xác nhận. **R50 VỠ** nếu R50 WB < 0.8×RD (flush quá mạnh phá δ trên chính surrogate) → p đó loại; đọc p lớn nhất còn giữ R50. Mô tả đường liều đầy đủ 0.3→0.9.
**Caveat:** dev30; p rất cao → gần như chỉ giữ vài pixel gradient → có thể vỡ.
**Chạy:** `results/e60/run_ext.sh`, log `results/e60/run_ext.log`.

### Kết quả E61 @ dev30 B36 (3090, xong 16:43 GMT+7; `results/e60/panel_ext_dev30.json`, `results/runs/dev300_B36_eps5_limit100/*_e61_dev30.json`, log `results/e60/run_ext.log`) — theo quy tắc: XÁM; đỉnh p=0.7, KHÔNG tái tạo hết V100
Đường liều Swin-L drop (R50 WB): RD 30.3(98) | GQ30 30.8(98) | GQ50 31.6(98.8) | **GQ70 32.7(98.2, đỉnh)** | GQ80 31.9(98.2) | GQ90 31.0(97.2). Mốc V100 = 37.7.
**Đọc:** (1) Chữ U ngược, đỉnh p=0.7 +2.4 — flush thành phần nhỏ là lever THẬT nhưng chỉ tái tạo **~1/3** lợi thế V100 (2.4/7.7). (2) R50 WB giữ ~98 tới p=0.8, p=0.9 bắt đầu nhẹ (97.2) → flush quá mạnh mất cả thành phần hữu ích. (3) Phần ~5 điểm còn lại của V100 KHÔNG bắt được bằng flush gradient hậu kỳ → là hiệu ứng tích lũy fp16 sâu hơn (đổi cả thành phần lớn qua 36 bước), chỉ tái tạo được trên chính Volta. (4) +2.4 CI chạm 0, đỉnh kiểu CWT → khả năng không sống sót dev100.
**KẾT LUẬN truy tìm V100 (E47→E61):** root cause = kernel fp16 Volta lúc SINH δ (loại seed E59, code E60-checkout, eval probe). Mẫu dấu khác (cos 0.30) cùng thống kê bậc thấp. Khai thác hậu kỳ (RD-GQ) tái tạo ~1/3 (+2.4), không đủ. **Trần Swin-L một-R50-chuẩn ~33–38** (3090 dev100 33.8; V100 37.7; cùng-họ Swin-T 54–59). 50 cần đổi surrogate (robust/cùng-họ) — ngoài ràng buộc. **DỪNG hướng numerics/flush.** Method chốt = RD (E58: +9.8 Swin-L so OSFD dev100, CI loại 0). Việc tiếp user quyết: dựng bảng paper quanh RQ1 + cơ chế (E41/E51/E53/E54/E59) + RD; hoặc nới ràng buộc surrogate.

---

## 2026-10-31 — E62: kiểm máy mới RTX A4500 — RD B36 + OSFD B50 @ dev30 full panel có khớp mốc 3090 không? (user: "run 1 trước", ~12:05 GMT+7) — CHẨN ĐOÁN, khóa TRƯỚC khi chạy
**Vì sao:** máy phiên này là RTX A4500 20GB (Ampere, compute capability 8.6 như 3090; torch 2.1.2+cu118, driver 570.158.01), chưa có trong docs. E59–E61: kernel GPU lúc SINH δ đổi được Swin-L ~8 điểm (V100 vs 3090) → phải đo trước khi diễn giải số máy này. PNG 3090 đã mất, nên chỉ so được số tổng hợp, KHÔNG đo được đồng thuận dấu δ.
**Ô:** RD B36, OSFD B50, seed 0, dev30, full panel (cùng lệnh E46). Mốc 3090: Swin-L RD 30.6 (E46; E47 31.5, E60 30.3), OSFD 26.5; T8 drop RD 79.0, OSFD 77.6.
**Quy tắc (điểm):** **KHỚP AMPERE** nếu |RD Swin-L − 30.6| ≤ 2 VÀ |OSFD Swin-L − 26.5| ≤ 2 VÀ |T8 drop − mốc| ≤ 2 ở cả hai method → dùng 3090 làm mốc diễn giải. **LỆCH** nếu một trong ba độ lệch > 4 (trên spread seed 3.2, E59) hoặc RD Swin-L ≥ 35 (phía V100) → coi A4500 là nền riêng, không đối chiếu số với 3090. Còn lại XÁM (xử như LỆCH khi so số, nhưng không chặn việc tiếp).
**Dù kết quả nào:** bảng paper sinh TOÀN BỘ trên một máy, không ghép số giữa các máy; KHỚP không cho phép ghép.
**Caveat:** dev30 chỉ phát hiện lệch lớn cỡ máy; n = 30, không đọc hiệu method từ đây.
**Chạy:** `results/e62/run.sh`, log `results/e62/run.log`.

### Kết quả E62 @ dev30 (RTX A4500, xong ~12:13 GMT+7; `results/e62/panel_dev30.json`, `results/runs/dev300_B{36,50}_eps5_limit100/*_e62_dev30.json`, log `results/e62/run.log`) — theo quy tắc: **XÁM** (RD Swin-L lệch +3.2, trong khoảng 2–4)
Drop A4500 (lệch so E46 3090): RD Swin-L 33.8 (+3.2), T8 80.3 (+1.3) | OSFD Swin-L 26.4 (−0.1), T8 77.3 (−0.3). ASR T8: RD 65.0 (−0.9), OSFD 70.0 (−0.6). Ngoài quy tắc: RD YOLOX-L 76.9 (+6.4); các target còn lại của RD và OSFD lệch ≤ 2.7 về drop.
**Đọc:** (1) Theo quy tắc khóa, không đối chiếu số A4500 với 3090; A4500 là nền riêng, không chặn việc tiếp. (2) OSFD khớp sát 3090 (Swin-L, T8); phần lệch nằm ở RD trên Swin-L và YOLOX-L. Với n = 30, chưa tách được khác máy hay nhiễu dev30. RD − OSFD trên Swin-L ở đây +7.4 [+0.5, +14.7]. (3) Không xa phía V100 (RD Swin-L < 35). (4) Thời gian trên máy này: RD B36 3.03 s/ảnh, OSFD B50 3.96 s/ảnh, nên "B36 = cùng thời gian OSFD B50" không còn đúng ở đây; B36 giữ nguyên, thiệt về phía RD.
**Việc tiếp:** khóa quy tắc GO cho RD (CA3 theo YOLOX-L) trước khi chạy full panel dev100 → dev300, toàn bộ trên A4500.

---

## 2026-10-31 — E63: surrogate ConvNeXt-T — CNN "kiểu Transformer" có làm cầu nối tới DINO-Swin-L không? RD port @ dev30 B36 full panel, A4500 (user: "1 thử", ~12:25 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** muốn Swin-L > 50 mà giữ MỘT surrogate CNN. Chưa từng chạy surrogate ConvNeXt (mới có Swin-T 58.0, DINO-R50 29.2). Cơ sở: δ_SwinT → ConvNeXt 85.2; R50 → ConvNeXt 84–86. **Đổi surrogate = rời mốc "surrogate R50 cố định"**; đây là thăm dò, chưa đổi threat model chính.
**Port (đo trước khi khóa, không xem số transfer):** RD nguyên trạng không dùng được: WP chỉ phủ `Conv2d` = 6.7% trọng số backbone ConvNeXt (R50 99.8%); λ theo quy tắc `rd_calib` (trung vị 5 ảnh) = 0.21 (R50 5.32 -> λ 6) → λ = 6 làm L_cls lấn L_feat ~28×. Sửa: `weight_linear` (WP phủ cả nn.Linear; R50 không có Linear trong backbone nên ô cũ không đổi), λ = 0.2 theo quy tắc, không tune. `panel_table` thêm nhãn `NHÃN=TÊN@RUN:FILE`.
**Ô:** **RD-CNX** (chính: λ 0.2, WP cả Linear, ρ 0.002); CNX-NOWP (λ 0.2, không WP) và CNX-NGUYEN (RD nguyên λ 6, WP chỉ Conv2d) chỉ MÔ TẢ. Mốc: RD-R50 của E62 cùng máy (Swin-L 33.8; T7 = r101, fcos, dino_r50, yolov3, yolox_l, swin_t, dino_swin_l: 79.6). T7 thay T8 vì ConvNeXt thành white-box.
**Quy tắc (RD-CNX, dev30):** **HỢP LỆ** nếu white-box ConvNeXt ≥ 80, không thì đọc là port hỏng. **ĐẠT** nếu Swin-L ≥ 50 → dev100. **HỨA HẸN** nếu 38 ≤ Swin-L < 50 (≥ mốc + 4, sàn nhiễu dev30) → dev100. **KHÔNG** nếu Swin-L < 38 → đóng surrogate ConvNeXt. Mô tả cái giá: R50, R101, T7.
**Caveat:** dev30, Swin-L phải xác nhận dev100 (E56→E58); ρ, OCT, k giữ theo R50, chưa hiệu chỉnh cho ConvNeXt nên có thể chưa hết tiềm năng; smoke 5 ảnh chỉ attack, chưa đọc số.
**Chạy:** `results/e63/run.sh`, log `results/e63/run.log`.

### Kết quả E63 @ dev30 B36 (A4500, xong 12:38 GMT+7; `results/e63/panel_dev30.json`, `results/runs/dev300_B36_eps5_limit100_sur-convnext_t/*_e63_dev30.json`, log `results/e63/run.log`) — theo quy tắc: **HỢP LỆ, ĐẠT** (white-box 95.4; Swin-L 62.8 ≥ 50) → cần dev100
R50 / R101 / ConvNeXt / Swin-T / FCOS / DINO-R50 / YOLOv3 / YOLOX-L / DINO-Swin-L | T7 drop:
- RD-R50 (E62): 98.7 / 92.5 / 85.8 / 84.3 / 95.0 / 92.7 / 81.7 / 76.9 / **33.8** | 79.6
- **RD-CNX: 83.7 / 82.2 / 95.4 / 87.1 / 85.0 / 83.6 / 71.7 / 75.1 / 62.8 | 78.2**
- CNX-NOWP (mô tả): 86.7 / 83.5 / 98.6 / 88.5 / 83.4 / 86.3 / 76.1 / 70.2 / 52.9 | 77.3
- CNX-NGUYEN (mô tả): 81.7 / 79.7 / 98.3 / 88.8 / 81.5 / 82.5 / 77.8 / 64.5 / 55.6 | 75.8
RD-CNX − RD-R50: **Swin-L +29.0 [+17.9, +31.9]**, Swin-T +2.8 [−2.7, +8.5], YOLOX-L −1.8 (CI chứa 0); cái giá: R50 −15.0, R101 −10.4, FCOS −10.0, DINO-R50 −9.2, YOLOv3 −10.0 (đều CI < 0). Swin-L ASR 45.2 (RD-R50 27.0), APloc 50.8 (34.2). ASR T7 60.8 (65.4). Thời gian 4.17 s/ảnh.
**Kiểm toàn vẹn (sau khi thấy số):** 30/30 PNG, L∞ đúng 5, RMS 4.76; clean dets DINO-Swin-L và YOLOX-L trùng tuyệt đối với run R50; PNG khác RD-R50 thật (mean |diff| 4.5).
**Đọc:** (1) Lần đầu vượt 50 ở ε = 5 với MỘT surrogate CNN; trần 33–38 là trần của R50, không phải của CNN → bức tường không đơn thuần "CNN ↔ Transformer": cần sửa câu RQ2 sau khi dev100 xác nhận. (2) Đổi lại mất ~10 điểm ở họ ResNet: gap chỉ dịch chỗ một phần, T7 gần như giữ (−1.4). (3) WP nhiễu cả Linear có vẻ góp phần (+9.9 so NOWP) nhưng là so sánh mô tả n = 30. (4) Số dev30, 1 máy; mốc Swin-T surrogate 58.0 là V100 dev100 với C2-OCT-1V, chưa so trực tiếp được.
**Việc tiếp (user quyết):** E64 xác nhận dev100 cùng máy: RD-CNX, RD-R50, OSFD full panel. Đổi surrogate là rời "surrogate R50 cố định" của idea.md §5: nếu dùng làm method chính cần sửa plan có chủ đích.

---

## 2026-10-31 — E64: vì sao δ từ surrogate ConvNeXt-T yếu trên họ ResNet? @ dev30, PNG E62 / E63, chỉ forward (user: "Phân tích vì sao tuột trước", ~12:45 GMT+7) — MÔ TẢ, khai báo TRƯỚC khi đọc số
**Đo (`scripts/e64_cnx_drop.py`):** δ_R (RD-R50), δ_C (RD-CNX), δ_O (OSFD). (1) phổ 3 dải, cos + đồng thuận dấu δ_R/δ_C; (2) D_k theo stage trên R50 / R101 / ConvNeXt / Swin-T / DINO-Swin-L; (3) AP drop LP2 / HP2 (như E16) + MIX = LP2(δ_C) + HP2(δ_R), kẹp ±5; (4) recall theo cỡ GT + FP / ảnh trên R50.
**Đọc (khai báo trước, trên R50):** **H_TẦN** nếu band_hi δ_C < δ_R ≥ 0.03 VÀ drop HP2(δ_R) − HP2(δ_C) ≥ 10 VÀ |drop LP2(δ_R) − LP2(δ_C)| < 10: δ_C thiếu đúng dải R50 nhạy. **H_HƯỚNG** nếu phổ gần nhau (chênh band_hi < 0.03) mà drop vẫn lệch: khác ở hướng chứ không ở dải tần. **H_CỠ** (cộng thêm được) nếu recall còn lại chênh tập trung ở một nhóm cỡ. D_k cho biết δ_C tắt ở stage nào của R50. MIX chỉ là phép dò hậu kỳ, dùng δ từ 2 surrogate nên không phải method.
**Caveat:** n = 30; drop qua `inference_detector` trên file, không so trực tiếp số panel; không chọn gì từ đây vào method nếu chưa khóa entry riêng.
**Chạy:** `python scripts/e64_cnx_drop.py`, log `results/e64/run.log`.

### Kết quả E64 @ dev30 (A4500, xong ~12:48 GMT+7; `results/mechanism/e64_cnx_drop.json`, log `results/e64/run.log`) — theo khai báo: **KHÔNG khớp đủ H_TẦN lẫn H_HƯỚNG**, gần H_HƯỚNG; H_CỠ có
Phổ δ thấp/trung/cao: δ_R .360/.351/.289, δ_C .419/.327/.254 (chênh band_hi 0.035, vừa qua ngưỡng 0.03 nên H_HƯỚNG trượt sát). cos(δ_R, δ_C) 0.09, đồng thuận dấu 0.546.
Drop trên R50: FULL 98.7 vs 83.7; HP2 79.4 vs 44.1 (Δ 35.3); LP2 48.5 vs 22.9 (Δ 25.6 ≥ 10 nên H_TẦN trượt). δ_C yếu ở CẢ HAI dải.
D_k trên R50 (stage 1→4): δ_R .340/.731/1.067/1.192, δ_C .326/.690/.899/.950, δ_O .342/.808/1.311/1.519. Trên DINO-Swin-L: δ_R .370/.457/.674/.821, δ_C .371/.478/.732/.912, δ_O .371/.475/.742/.909.
Recall R50 nhỏ/vừa/lớn: sạch 55.6/83.1/91.7, δ_R 6.6/11.0/4.2, δ_C 21.4/35.6/43.8. FP / ảnh: sạch 8.7, δ_R 42.9, δ_C 20.9, δ_O 49.8.
MIX = LP2(δ_C) + HP2(δ_R): R50 96.3, R101 86.2, ConvNeXt 80.1, Swin-T 72.5, DINO-Swin-L 29.3.
**Đọc:** (1) Nguyên nhân chính là HƯỚNG, không phải thiếu tần cao: δ_C gần như không tương quan δ_R, yếu ở cả hai dải. (2) δ_C VÀO được R50 (stage 1–2 ngang δ_R) nhưng không được khuếch đại ở stage sâu (C4 −16%, C5 −20%). (3) Hệ quả: object lớn sống (43.8 vs 4.2), box giả giảm một nửa. (4) Độ lớn distortion KHÔNG giải thích transfer: trên Swin-L δ_O và δ_C có D_k gần bằng nhau mà drop 26.4 vs 62.8 → cái quyết định là hướng distortion. (5) MIX: lấy lại R50 nhưng Swin-L tụt 62.8 → 29.3: thành phần cao tần là chỗ hai họ TRANH NHAU, không tách theo dải được; chia dải tần cho từng họ là hướng đóng. Caveat: LP / HP co lại L∞ = 5 và phi tuyến, n = 30.
**Điểm phải kiểm trước khi đổi surrogate:** so RD-CNX với OSFD-R50 là so lẫn hiệu ứng surrogate; CNX-NOWP 52.9 và CNX-NGUYEN 55.6 gợi ý phần lớn lợi đến từ surrogate. Cần OSFD chạy trên ConvNeXt mới biết phần của method.

---

## 2026-10-31 — E65: lợi của RD-CNX do SURROGATE hay do METHOD? OSFD / MI / M-DI² trên surrogate ConvNeXt-T @ dev30 B50 full panel, A4500 (user: "approve nhưng dev30 thôi", ~12:51 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** E63 so RD-CNX với mốc surrogate R50, lẫn hiệu ứng surrogate; E63 CNX-NOWP 52.9, CNX-NGUYEN 55.6 gợi ý phần lớn lợi đến từ ConvNeXt. Phải so CÙNG surrogate trước khi đổi.
**Ô:** OSFD, MI-FGSM, M-DI²-FGSM trên ConvNeXt-T, B50, cấu hình protocol_lock, không chỉnh gì. Mốc: RD-CNX B36 (E63: Swin-L 62.8, T7 78.2), OSFD-R50 B50 (E62: Swin-L 26.4, T7 77.2).
**Quy tắc (Δ = RD-CNX − OSFD-CNX, dev30, điểm; sàn nhiễu Swin-L ~4):** **HỢP LỆ** mỗi baseline nếu white-box ConvNeXt ≥ 80. **METHOD GÓP** nếu ΔSwin-L ≥ +4 VÀ ΔT7 ≥ −2. **CHỈ SURROGATE** nếu |ΔSwin-L| < 4: lợi chủ yếu do ConvNeXt, RD không đóng góp rõ trên Swin-L. **OSFD-CNX THẮNG** nếu ΔSwin-L ≤ −4. Còn lại XÁM. Mô tả: phần surrogate = (OSFD-CNX − OSFD-R50) / (RD-CNX − OSFD-R50) trên Swin-L; từng target.
**Caveat:** dev30 chỉ xem xu hướng, không kết luận; B lệch (36 vs 50) thiệt cho RD; dev100 xác nhận sau, user quyết.
**Chạy:** `results/e65/run.sh`, log `results/e65/run.log`.

### Kết quả E65 @ dev30 (A4500, xong 13:08 GMT+7; `results/e65/panel_dev30.json`, `results/runs/dev300_B50_eps5_limit100_sur-convnext_t/*_e65_dev30.json`, log `results/e65/run.log`) — theo quy tắc: **HỢP LỆ, METHOD GÓP**
R50 / R101 / ConvNeXt / Swin-T / FCOS / DINO-R50 / YOLOv3 / YOLOX-L / DINO-Swin-L | T7 drop | ASR T7:
- RD-CNX B36: 83.7 / 82.2 / 95.4 / 87.1 / 85.0 / 83.6 / 71.7 / 75.1 / **62.8** | 78.2 | 60.8
- OSFD-CNX B50: 74.9 / 65.1 / 95.6 / 73.0 / 79.1 / 75.1 / 61.4 / 54.3 / **26.1** | 62.0 | 55.5
- MI-CNX B50: 47.5 / 40.8 / 99.0 / 52.4 / 47.0 / 45.1 / 33.0 / 31.9 / 21.9 | 38.9 | 26.7
- M-DI²-CNX B50: 74.9 / 69.2 / 99.9 / 75.5 / 70.1 / 66.3 / 51.8 / 52.4 / 37.9 | 60.5 | 41.1
- OSFD-R50 B50 (E62): 98.0 / 92.4 / 86.0 / 78.3 / 93.2 / 95.4 / 74.0 / 72.4 / 26.4 | 76.0 | 69.7
RD-CNX − OSFD-CNX: **Swin-L +36.7 [+26.4, +41.6]**, T7 +16.2, mọi target hộp đen > 0 với CI loại 0. RD-CNX − M-DI²-CNX: Swin-L +24.9 [+13.7, +28.1]. Phần surrogate theo công thức khai báo ≈ 0 ((26.1 − 26.4) / (62.8 − 26.4)). Thời gian: RD-CNX 4.17 s/ảnh, OSFD-CNX 6.62 s/ảnh.
**Đính chính:** T7 của OSFD-R50 là **76.0**; số 77.2 ở khai báo E65 và trong tin nhắn báo cáo E63 là tính sai. Quy tắc không dùng số này nên phán quyết không đổi. Giờ của E64 sửa lại theo thực tế (~12:45, xong ~12:48).
**Đọc:** (1) Giả thuyết "lợi chủ yếu do surrogate" (E64) BỊ BÁC: chỉ đổi surrogate cho OSFD không tăng gì. (2) Bảng 2×2 Swin-L: OSFD-R50 26.4, RD-R50 33.8, OSFD-CNX 26.1, RD-CNX 62.8 → tương tác surrogate × method ≈ +29.3: bước nhảy chỉ có khi có CẢ HAI. T7 2×2: 76.0 / 79.6 / 62.0 / 78.2. (3) Nguyên liệu khả dĩ: phần chung của 3 ô CNX Swin-L > 50 là OCT + loss cls (NOWP 52.9, nguyên 55.6) so OSFD-CNX 26.1 (RRB + feature), WP thêm khoảng +10; chưa ablation riêng, chỉ là suy đoán. (4) So đối thủ thực tế OSFD-R50: Swin-L +36.4 [+22.6, +41.1], T7 +2.2, nhưng R50 / R101 / FCOS / DINO-R50 giảm 8–14 (CI < 0), ASR T7 −8.9. RD-CNX vs RD-R50 vẫn là đánh đổi (T7 −1.4). (5) λ và WP đã port cho RD, OSFD chạy nguyên cấu hình protocol; OSFD-CNX có thể chưa phát huy hết. dev30 chỉ xem xu hướng.
**Việc tiếp (user quyết):** dev100 bảng 2×2 (RD-CNX, RD-R50, OSFD-CNX, OSFD-R50) để xác nhận tương tác.

---

## 2026-10-31 — E66: nguyên liệu nào bắc cầu CNN↔Transformer trên surrogate ConvNeXt? Tách OCT-feat / cls / WP @ dev30 B36 (user: "Thực hiện 1", ~13:21 GMT+7) — khóa TRƯỚC khi chạy
**Vì sao:** E65 cho thấy METHOD GÓP (RD-CNX ≫ OSFD-CNX) nhưng chưa biết thành phần nào tạo cầu, và đâu là nút đánh đổi CNN↔Transformer. Cần tách để (a) claim cơ chế = novelty, (b) tìm xem có cấu hình giữ ResNet mà vẫn được Swin-L không.
**Ô (ConvNeXt-T, B36, dev30; RD-CNX = OCT + feat C3-C5 + cls RoI + WP):**
- RD-CNX = cả bốn (E63: Swin-L 62.8, R50 83.7, R101 82.2, T7 78.2)
- NOWP = OCT + feat + cls, bỏ WP (E63: Swin-L 52.9, R50 86.7, R101 83.5)
- FEAT = OCT + feat, bỏ cls + WP (mới)
- CLS = OCT + cls, bỏ feat + WP (mới; feature=False)
**Quy tắc đọc (dev30, điểm; sàn nhiễu Swin-L ~4):** mỗi thành phần X: đóng góp Swin-L = drop(ô có X) − drop(ô bỏ X); giá CNN = drop R50/R101 tương ứng.
- **BẮC CẦU = cls** nếu drop(NOWP) − drop(FEAT) ≥ +15 trên Swin-L (cls là nguyên liệu chính), **= feat** nếu drop(NOWP) − drop(CLS) ≥ +15, **= cả hai** nếu cả hai ≤ +15 (bổ trợ).
- **WP là NÚT TILT** nếu RD-CNX − NOWP: ΔSwin-L ≥ +4 VÀ ΔR101 ≤ −2 (được Transformer, mất ResNet) → ρ-sweep là lever giữ-cả-hai (entry sau). **WP TRUNG TÍNH** nếu |ΔR101| < 2.
- **GIỮ ĐƯỢC CẢ HAI** (trả lời câu user) nếu tồn tại ô: Swin-L ≥ 50 VÀ R50 ≥ 90 VÀ R101 ≥ 88 (gần mốc OSFD-R50 92.4). Không ô nào đạt → ghi: đánh đổi là nội tại ở cấu hình hiện có; lever giữ-cả-hai (nếu có) phải tới từ ρ-sweep / ensemble (threat model), user quyết.
**Caveat:** dev30 xem xu hướng; CLS feature=False nên scale λ không ảnh hưởng (bước sign 1 số hạng); FEAT/CLS chưa từng chạy trên ConvNeXt.
**Chạy:** `results/e66/run.sh`, log `results/e66/run.log`.

### Kết quả E66 @ dev30 B36 (A4500, xong 13:30 GMT+7; `results/e66/panel_dev30.json`, `results/runs/dev300_B36_eps5_limit100_sur-convnext_t/*_e66_dev30.json`, log `results/e66/run.log`) — theo quy tắc: **BẮC CẦU = cls (chính) + feat (phụ); WP TRUNG TÍNH trên CNN nhưng +9.9 Swin-L; KHÔNG ô nào giữ được cả hai**
Drop (R50/R101/ConvNeXt/Swin-T/FCOS/DINO-R50/YOLOv3/YOLOX-L/Swin-L | T7 | TCNN5 = R101+FCOS+DINO-R50+YOLOv3+YOLOX-L | Swin-L):
- RD-CNX (OCT+feat+cls+WP): 83.7/82.2/95.4/87.1/85.0/83.6/71.7/75.1/62.8 | 78.2 | 79.5
- NOWP (OCT+feat+cls):      86.7/83.5/98.6/88.5/83.4/86.3/76.1/70.2/52.9 | 77.3 | 79.9
- FEAT (OCT+feat):          75.0/68.0/91.7/73.9/76.4/72.8/65.4/65.2/34.7 | 65.2 | 69.6
- CLS  (OCT+cls):           76.6/73.3/97.7/84.0/75.1/79.0/61.7/56.6/43.1 | 67.5 | 69.1
**Phân rã Swin-L:** cls = drop(NOWP) − drop(FEAT) = **+18.2** (≥15 → nguyên liệu chính); feat = drop(NOWP) − drop(CLS) = **+9.8** (phụ); WP = RD-CNX − NOWP = **+9.9 [+3.8, +14.5]** (CI loại 0). Ba thành phần đều dương, feat+cls cộng hưởng (riêng 34.7 / 43.1, gộp 52.9).
**WP KHÔNG phải nút tilt:** RD-CNX − NOWP trên CNN: R101 −1.4, TCNN5 −0.4 (|ΔR101| < 2 → TRUNG TÍNH), R50 target −3.0; nhưng Swin-L +9.9 → **WP cho Swin-L gần như MIỄN PHÍ, không đổi bằng CNN**. Sửa giả thuyết trước đó ("WP là nút đánh đổi"): sai.
**Trả lời câu "giữ cả hai":** KHÔNG ô nào đạt (Swin-L ≥ 50 VÀ R50 ≥ 90 VÀ R101 ≥ 88). Hụt ~10 điểm ở target ResNet (R101/FCOS/DINO-R50 ~82–85 vs R50-sur ~92–95) **có trong MỌI ô** (feat, cls, cả hai) → không phải do một nút gạt nào trong cấu hình hiện tại mà là **mất mát transfer nội tại ConvNeXt→ResNet** (khớp E64: δ_CNX không khuếch đại ở C4/C5 của R50). → lever lấy lại CNN phải tới từ NGOÀI 3 thành phần: (a) tăng B (E40: target CNN còn tăng tới B75, Swin-L bão hòa B36 — chưa kiểm trên ConvNeXt), (b) thành phần mới nhắm deep-stage xuyên họ, (c) ensemble (đổi threat model).
**Novelty sạch cho paper:** cls = cầu chính CNN→Transformer (OSFD feature-only không có → E65), feat bổ trợ, WP nâng Transformer gần như miễn phí. Cơ chế: object-level cls signal (RoI) transfer xuyên họ tốt hơn feature-matching thuần.
**Việc tiếp (user quyết):** E67 quét B ∈ {36, 50, 75} cho RD-CNX @ dev30 — kiểm "B tách Swin-L (bão hòa) khỏi CNN (còn tăng)"; nếu B75 kéo TCNN5 lên ~88 mà Swin-L giữ → đúng hướng "giữ cả hai", xác nhận dev100.

---

## 2026-10-31 — E67: cơ chế — vì sao surrogate ConvNeXt bắc cầu Transformer mà tụt ResNet? CKA biểu diễn backbone @ dev30, forward-only (user: "Phân tích cơ chế vì sao ConvNeXt improve ViT/Swin-L và giảm target còn lại", ~13:43 GMT+7) — MÔ TẢ, khai báo TRƯỚC khi đọc số
**Giả thuyết:** ConvNeXt (CNN macro-design kiểu Swin) có BIỂU DIỄN gần Swin hơn gần ResNet -> δ tối ưu trên ConvNeXt rơi vào subspace Transformer (transfer mạnh Swin-L), lệch subspace ResNet (E64: cos δ_CNX/δ_R50 = 0.09, không khuếch đại C4/C5 R50).
**Đo ():** linear CKA từng cặp trong {r50, r101, convnext_t, swin_t, dino_swin_l}, ảnh sạch dev30, cùng pixel đầu vào (khung ConvNeXt), 4 stage. Trục chính (stage sâu 3-4): d = TB CKA(ConvNeXt, {swin_t, dino_swin_l}) − TB CKA(ConvNeXt, {r50, r101}), paired bootstrap theo ảnh.
**Quy tắc (khai báo trước):** **CONVNEXT ≈ TRANSFORMER** nếu d > 0 với CI loại 0 → cơ chế xác nhận: bắc cầu là do ConvNeXt biểu diễn kiểu Transformer. **KHÔNG** nếu d ≤ 0 hoặc CI chứa 0 → cơ chế không nằm ở tương đồng biểu diễn, phải tìm chỗ khác (phổ tần / receptive field). Kèm mô tả: liên hệ CKA(ConvNeXt, target) với drop δ_CNX (E63) vs δ_R50 (E62) trên 4 model backbone trong bộ.
**Hệ quả nếu xác nhận:** để GIỮ Swin-L + NÂNG ResNet cùng budget, δ phải nằm đồng thời 2 subspace gần trực giao (cos 0.09) tranh nhau ε (E64 MIX: thêm cao tần δ_R50 giết Swin-L) → đánh đổi có thể nội tại; lever cùng-budget chỉ còn tín hiệu BẤT BIẾN kiến trúc (cls object-level, E66) hoặc ensemble. Đây là khung cho câu trả lời "giữ cả hai".
**Caveat:** CKA trên ảnh sạch = tính chất kiến trúc, không phải δ; n=30; cùng pixel đầu vào (so biểu diễn, không phải pipeline riêng từng model).
**Chạy:** `python scripts/e67_convnext_cka.py`, log `results/e67/run.log`.
