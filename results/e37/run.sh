#!/usr/bin/env bash
# E37 Gate A oracle-retention HardBank (progress_log 2026-10-25). Init δ_SwinT, snapshot B4/B12.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; INIT=dev300_B41_eps5_limit100_sur-swin_t; O=metrics_hb_is.json; D=dets_hb_is.json
python scripts/run_baselines.py $A --budget 12 --stage attack --methods C2-OCT-1V-HB-IS --init-from $INIT --snapshots 4
for B in 4 12; do
  python scripts/run_baselines.py $A --budget $B --stage eval --methods C2-OCT-1V-HB-IS --out $O
  python scripts/eval_generalization.py --run dev300_B${B}_eps5_limit100 --subset dev30 --models dino_swin_l --methods C2-OCT-1V-HB-IS --dets $D --out generalization_hb_is.json
done
python - <<'PY'
import json
print("\n=== E37 Gate A HardBank-IS (init δ_SwinT) @ dev30 ===")
print(f"{'B':>4} {'R50 drop':>9} {'Swin-L drop':>12}")
for B in (4,12):
    r=f"dev300_B{B}_eps5_limit100"
    r50=json.load(open(f"results/runs/{r}/metrics_hb_is.json"))["relative_ap_drop"]["C2-OCT-1V-HB-IS"]["r50"]["point"]
    sl=json.load(open(f"results/runs/{r}/generalization_hb_is.json"))["relative_ap_drop"]["C2-OCT-1V-HB-IS"]["dino_swin_l"]["point"]
    print(f"{B:>4} {r50:>9.1f} {sl:>12.1f}")
print("Cổng B12: Swin-L ≥50 & R50 ≥85 & ≥ CAP(36.7)+10=46.7 → QUA; vẫn ~37-42 → DỪNG. (δ_SwinT B0 ~58; CAP-WP-IS B12 36.7)")
PY
echo "[e37] XONG $(date '+%F %T')"
