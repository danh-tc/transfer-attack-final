#!/usr/bin/env bash
# E35 cổng oracle-retention CAP-WP (progress_log 2026-10-25). Init δ_SwinT, snapshot B4/12/24. Resume được.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; INIT=dev300_B41_eps5_limit100_sur-swin_t
O=metrics_capwp_is.json; D=dets_capwp_is.json
python scripts/run_baselines.py $A --budget 24 --stage attack --methods C2-OCT-1V-CAPWP-IS --init-from $INIT --snapshots 4 12
for B in 4 12 24; do
  python scripts/run_baselines.py $A --budget $B --stage eval --methods C2-OCT-1V-CAPWP-IS --out $O
  python scripts/eval_generalization.py --run dev300_B${B}_eps5_limit100 --subset dev30 --models dino_swin_l --methods C2-OCT-1V-CAPWP-IS --dets $D --out generalization_capwp_is.json
done
python - <<'PY'
import json
print("\n=== E35 CỔNG CAP-WP-IS (init δ_SwinT) @ dev30 ===")
print(f"{'B':>4} {'R50 drop':>9} {'Swin-L drop':>12}")
for B in (4,12,24):
    r=f"dev300_B{B}_eps5_limit100"
    r50=json.load(open(f"results/runs/{r}/metrics_capwp_is.json"))["relative_ap_drop"]["C2-OCT-1V-CAPWP-IS"]["r50"]["point"]
    sl=json.load(open(f"results/runs/{r}/generalization_capwp_is.json"))["relative_ap_drop"]["C2-OCT-1V-CAPWP-IS"]["dino_swin_l"]["point"]
    print(f"{B:>4} {r50:>9.1f} {sl:>12.1f}")
print("Mốc: δ_SwinT B0 ~59; WP2-IS B12 39.8 (E20); ENS-IS 45-46 (E25). Cổng B12: ≥52 QUA / 47-52 xám / ≤45 DỪNG")
PY
echo "[e35] XONG $(date '+%F %T')"
