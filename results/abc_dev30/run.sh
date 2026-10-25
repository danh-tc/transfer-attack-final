#!/usr/bin/env bash
# Quick panel A/B/C @ dev30 (progress_log 2026-10-25). Resume được.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 300"
GENM="fcos_r50 dino_r50 yolov3_d53 yolox_s dino_swin_l"
R50=dev300_B50_eps5_limit300; R41=dev300_B41_eps5_limit100; R36=dev300_B36_eps5_limit100
# 1) attack MI/M-DI2 (OSFD reuse PNG) + eval controlled panel, B50
python scripts/run_baselines.py $A --budget 50 --methods OSFD MI-FGSM M-DI2-FGSM --out metrics_abc_dev30.json
# 2) eval gen panel (5 target) cho từng method từ run dir của nó
python scripts/eval_generalization.py --run $R50 --subset dev30 --models $GENM --methods OSFD MI-FGSM M-DI2-FGSM --dets dets_abc_dev30.json --out generalization_abc_dev30.json
python scripts/eval_generalization.py --run $R41 --subset dev30 --models $GENM --methods C2-OCT-1V --dets dets_c2_1v_dev100.json --out generalization_abc_dev30.json
python scripts/eval_generalization.py --run $R36 --subset dev30 --models $GENM --methods C2-OCT-1V-WP2 --dets dets_wpa_dev100.json --out generalization_abc_dev30.json
# 3) gộp bảng
python - <<'PY'
import json, os
runs = {"OSFD":"dev300_B50_eps5_limit300","MI-FGSM":"dev300_B50_eps5_limit300","M-DI2-FGSM":"dev300_B50_eps5_limit300",
        "C2-OCT-1V":"dev300_B41_eps5_limit100","C2-OCT-1V-WP2":"dev300_B36_eps5_limit100"}
tgt = ["fcos_r50","dino_r50","yolov3_d53","yolox_s","dino_swin_l"]
print(f"\n=== PANEL A/B/C @ dev30 — relative AP drop % ===")
print(f"{'method':<16}" + "".join(f"{t:>13}" for t in tgt))
for m,r in runs.items():
    f=f"results/runs/{r}/generalization_abc_dev30.json"
    d=json.load(open(f))["relative_ap_drop"].get(m,{})
    row=[]
    for t in tgt:
        v=d.get(t); row.append(f"{v['point']:.1f}" if isinstance(v,dict) else "-")
    print(f"{m:<16}" + "".join(f"{x:>13}" for x in row))
PY
echo "[run.sh] XONG $(date '+%F %T')"
