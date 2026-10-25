#!/usr/bin/env bash
# Hoàn thiện ε=10 dev30 theo panel A/B/C: eval thêm fcos_r50/dino_r50/yolov3 trên PNG ε=10 có sẵn (forward-only).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
GENM="fcos_r50 dino_r50 yolov3_d53 yolox_s dino_swin_l"; D=dets_eps10_dev30.json
R50=dev300_B50_eps10_limit100; R36=dev300_B36_eps10_limit100
python scripts/eval_generalization.py --run $R50 --subset dev30 --models $GENM --methods OSFD M-DI2-FGSM --dets $D --out generalization_eps10_dev30.json
python scripts/eval_generalization.py --run $R36 --subset dev30 --models $GENM --methods C2-OCT-1V-WP2 --dets $D --out generalization_eps10_dev30.json
python - <<'PY'
import json
runs={"OSFD":"dev300_B50_eps10_limit100","M-DI2-FGSM":"dev300_B50_eps10_limit100","C2-OCT-1V-WP2":"dev300_B36_eps10_limit100"}
tgt=["fcos_r50","dino_r50","yolov3_d53","yolox_s","dino_swin_l"]
d0=json.load(open("results/runs/dev300_B50_eps10_limit100/generalization_eps10_dev30.json"))
print("\n=== PANEL A/B/C @ dev30, ε=10 — relative AP drop % ===")
print("clean AP:", {t:round(d0["ap"][t]["clean"]["point"],1) for t in tgt})
print(f"{'method':<15}|{'A:fcos':>8}{'A:dino_r50':>11}|{'B:yolov3':>9}{'B:yolox':>8}|{'C:dino_sl':>10}|{'Avg5':>7}{'AvgB+C':>8}")
for m,r in runs.items():
    rad=json.load(open(f"results/runs/{r}/generalization_eps10_dev30.json"))["relative_ap_drop"][m]
    v={t:rad[t]['point'] for t in tgt if t in rad and isinstance(rad[t],dict)}
    a5=sum(v.values())/len(v); bc=(v['yolov3_d53']+v['yolox_s']+v['dino_swin_l'])/3
    g=lambda t:f"{v[t]:.1f}"
    print(f"{m:<15}|{g('fcos_r50'):>8}{g('dino_r50'):>11}|{g('yolov3_d53'):>9}{g('yolox_s'):>8}|{g('dino_swin_l'):>10}|{a5:>7.1f}{bc:>8.1f}")
PY
echo "[eval_abc] XONG $(date '+%F %T')"
