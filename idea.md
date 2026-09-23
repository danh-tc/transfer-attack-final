Đây là **plan final** mình nghĩ nên khóa để triển khai từ giờ, theo hướng **baseline-first** và chỉ phát triển method sau khi có evidence đủ mạnh.

## 1. Research Question

**RQ1:** Cross-backbone-family có làm giảm transferability trong object detection không?

$$
Transfer_{same-family}
>
Transfer_{cross-family}
$$

**RQ2:** Nếu có gap, mechanism nào giải thích tốt nhất?

**RQ3:** Có thể thiết kế method mới để giảm gap đó và vẫn cạnh tranh với strong baselines không?

---

## 2. Problem Setup

Input:

$$
(x,y,f_s)
$$

Output:

$$
x_{adv}=x+\delta,\qquad \|\delta\|_\infty\le\epsilon
$$

Mục tiêu: perturbation được optimize trên **một surrogate duy nhất** nhưng vẫn transfer sang unseen target detectors.

---

## 3. Threat Model

Theo taxonomy của benchmark:

* digital
* entire-image
* image-specific
* untargeted
* random-output
* transfer-based black-box

Constraint riêng:

* exactly one surrogate
* no target weights
* no target gradients
* zero target queries during generation

Primary setting: GT-assisted.

Secondary setting: surrogate-prediction-only.

---

## 4. Dataset & Experimental Phases

Dataset chính:

$$
\boxed{\text{MS COCO}}
$$

Hai phase:

| Phase        |                Số ảnh | Mục tiêu                                            |
| ------------ | --------------------: | --------------------------------------------------- |
| Confirmation |  **300 fixed images** | baseline, hypothesis, mechanism, ablation, Go/No-Go |
| Final        | **1000 fixed images** | final paper results                                 |

Không tune trên 1000 ảnh.

---

## 5. Controlled Model Panel

Dùng để test backbone-family effect.

Surrogate:

$$
\text{Mask R-CNN + ResNet-50}
$$

Targets:

* Mask R-CNN + ResNet-101 → same-family
* Mask R-CNN + ConvNeXt-T → cross-CNN
* Mask R-CNN + Swin-T → CNN→Transformer

Mục tiêu:

$$
\text{hold detector architecture approximately fixed}
$$

và chỉ thay backbone family.

---

## 6. Generalization Panel

Chỉ chạy sau khi controlled experiment xác nhận gap.

Có thể thêm:

* FCOS-R50
* DETR-R50
* YOLOX-CSP
* DINO-Swin

Mục đích: kiểm tra phenomenon và method có generalize khi detector architecture cũng thay đổi.

---

## 7. Attack Conditions

Primary:

$$
\epsilon=5/255
$$

Secondary:

$$
8/255
$$

Fair comparison phải cùng:

* images
* surrogate
* targets
* preprocessing
* perturbation budget
* evaluation metrics
* gradient-evaluation budget

Không chỉ so same number of iterations.

Định nghĩa:

$$
B=\text{number of surrogate backward evaluations}
$$

Có thể khảo sát:

$$
B\in\{10,50,200\}
$$

---

## 8. Baseline-First Stage

Đây là việc cần làm **ngay trước khi phát triển method tiếp**.

Run trên cùng `n=300`:

* MI-FGSM
* DI-FGSM — dùng biến thể M-DI²-FGSM (MI + DI, chốt 2026-09-23, xem docs/protocol_lock.md)
* OSFD

~~AugTrans~~ — **tạm bỏ khỏi plan (2026-09-23)**: không có code chính thức (repo công bố trong paper trả 404), paper có mô tả mâu thuẫn, bản tự cài lại phải lệch khỏi paper ở nhiều chỗ (step size, loss_mask, K_max) và cho kết quả yếu hơn cả DI-FGSM ở quick check n=30 — không đủ tin cậy để làm strong baseline. Code giữ lại ở `attack/methods/augtrans.py` nhưng không chạy. Xem `docs/progress_log.md`. OSFD là strong baseline chính (được benchmark 2602.16494 đánh giá là attack transfer mạnh nhất trong nhóm có code). Có thể bổ sung baseline mạnh khác sau, nếu có code chính thức và khớp threat model — phải ghi quyết định vào progress_log trước khi chạy.

Mục tiêu là tạo baseline table chuẩn:

$$
\text{same-family}
\quad
\text{cross-CNN}
\quad
\text{CNN}\rightarrow\text{Transformer}
$$

Sau đó mới biết baseline ceiling thật sự là gì.

---

## 9. Metrics

Primary:

* Clean AP
* Adversarial AP
* Relative AP Drop

Secondary:

* AP50
* ASR
* CrossAvg
* TransferGap

Additional:

* APloc
* CSR
* realized \(L_\infty\)
* \(L_2\)
* LPIPS
* runtime
* backward evaluations

---

## 10. Hypothesis Pass @ n=300

Hypothesis pass nếu:

1. same-family transfer consistently > cross-family transfer
2. effect size đủ meaningful
3. paired bootstrap 95% CI của gap > 0
4. ~~pattern xuất hiện trên ít nhất 2 strong baselines~~ → **pattern xuất hiện ở cả OSFD và M-DI²-FGSM** (chốt 2026-09-23)
5. không chỉ xuất hiện với MI-FGSM

> Chốt (2026-09-23): sau khi bỏ AugTrans chỉ còn OSFD là strong baseline khớp threat model (benchmark 2602.16494) — tiêu chí 4 định nghĩa lại thành OSFD + M-DI²-FGSM; tiêu chí 5 giữ nguyên.

Nếu không đạt:

$$
\boxed{\text{NO-GO for cross-backbone premise}}
$$

---

## 11. Mechanism Stage

Chỉ sau khi hypothesis pass mới phân tích sâu:

* input-gradient alignment
* forward feature similarity
* stage-wise backward sensitivity
* gradient concentration
* iterative stability

Mục tiêu là tìm:

$$
\text{where and why cross-family divergence emerges}
$$

---

## 12. Method Development

Không khóa method trước.

Candidate direction hiện tại:

$$
\text{stage-aware backward regularization}
$$

nhưng chỉ giữ nếu mechanism evidence support.

Phải test như standalone và plug-in:

$$
Ours
$$

$$
DI+Ours
$$

$$
OSFD+Ours
$$

(~~AugTrans+Ours~~ — AugTrans tạm bỏ khỏi plan, xem §8.)

---

## 13. Go / No-Go for Method @ n=300

**GO standalone** nếu Ours vượt strong baseline trên CrossAvg và gain trên ít nhất 2 cross-family targets.

**GO plug-in** nếu:

$$
StrongBaseline+Ours > StrongBaseline
$$

một cách ổn định.

**NO-GO** nếu:

* chỉ hơn MI-FGSM
* chỉ tốt white-box
* không cải thiện Transformer targets
* thua strong baselines mà không có complementary gain

---

## 14. Final Evaluation @ n=1000

Chỉ chạy khi pass Go/No-Go ở 300.

Final run:

* strongest baselines
* final Ours
* controlled panel
* generalization panel
* fixed epsilon
* fixed compute budget
* fixed hyperparameters

Không tuning nữa.

---

## 15. Final Contribution nếu thành công

Paper sẽ có 3 contribution chính:

1. **Controlled evidence** về cross-backbone transfer gap.
2. **Mechanistic explanation** cho gap đó.
3. **Method / regularizer** giúp giảm gap và bổ sung cho strong attacks.

Workflow cuối cùng:

$$
\boxed{
\text{Lock Protocol}
\rightarrow
\text{Run Baselines @300}
\rightarrow
\text{Confirm Gap}
\rightarrow
\text{Analyze Mechanism}
\rightarrow
\text{Develop Method}
\rightarrow
\text{Go/No-Go @300}
\rightarrow
\text{Final @1000}
}
$$

Và việc nên làm tiếp ngay bây giờ là **khóa exact config của 4 baseline + model checkpoints + gradient budget cho run 300 ảnh**.
