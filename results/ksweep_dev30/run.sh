#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; R=dev300_B36_eps5_limit100; GENM="fcos_r50 dino_r50 yolov3_d53 yolox_s dino_swin_l"
python scripts/run_baselines.py $A --budget 36 --methods C2-OCT-1V-WP2-K1 C2-OCT-1V-WP2-K2 --out metrics_k_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --models $GENM --methods C2-OCT-1V-WP2-K1 C2-OCT-1V-WP2-K2 --dets dets_k_dev30.json --out generalization_k_dev30.json
python - <<'PY'
import json
tgt=["fcos_r50","dino_r50","yolov3_d53","yolox_s","dino_swin_l"]
src={"WP2(k=3)":("dev300_B36_eps5_limit100","generalization_abc_dev30.json","C2-OCT-1V-WP2"),
     "K2(k=2)":("dev300_B36_eps5_limit100","generalization_k_dev30.json","C2-OCT-1V-WP2-K2"),
     "K1(k=1)":("dev300_B36_eps5_limit100","generalization_k_dev30.json","C2-OCT-1V-WP2-K1")}
print("\n=== k-sweep @ dev30 panel A/B/C — relative AP drop % ===")
print(f"{'variant':<10}|{'A:fcos':>8}{'A:dino_r50':>11}|{'B:yolov3':>9}{'B:yolox':>8}|{'C:dino_sl':>10}|{'Avg5':>7}{'AvgB+C':>8}")
for lab,(r,f,m) in src.items():
    rad=json.load(open(f"results/runs/{r}/{f}"))["relative_ap_drop"][m]
    v={t:rad[t]['point'] for t in tgt}
    a5=sum(v.values())/5; bc=(v['yolov3_d53']+v['yolox_s']+v['dino_swin_l'])/3
    g=lambda t:f"{v[t]:.1f}"
    print(f"{lab:<10}|{g('fcos_r50'):>8}{g('dino_r50'):>11}|{g('yolov3_d53'):>9}{g('yolox_s'):>8}|{g('dino_swin_l'):>10}|{a5:>7.1f}{bc:>8.1f}")
PY
echo "[ksweep] XONG $(date '+%F %T')"
