#!/usr/bin/env bash
# E84: factorial surrogate × aug × stage trên dev300 \ dev100 (200 ảnh). Chạy trong tmux để ngắt SSH vẫn chạy:
#   tmux new -s e84 'bash results/e84/run.sh'      # tách: Ctrl-b d | xem lại: tmux attach -t e84
# 2 worker sinh song song (mỗi surrogate 1 process), xong mới eval + phán quyết. Resume an toàn (bỏ PNG đã có).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
# Nghẽn CPU (GPU ~30%): 2 process × ~176 thread tranh nhau -> giới hạn thread intra-op mỗi process.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
L=results/e84
python scripts/e84_recipe_factorial.py --gen-only \
  --cells R50-RRB-C25,R50-RRB-C35,R50-OCT-C25,R50-OCT-C35,R50-OCT2V-C35,R50-OSFD > $L/gen_r50.log 2>&1 &
P1=$!
python scripts/e84_recipe_factorial.py --gen-only \
  --cells CNX-RRB-C25,CNX-RRB-C35,CNX-OCT-C25,CNX-OCT-C35,CNX-OCT2V-C35,CNX-OSFD > $L/gen_cnx.log 2>&1 &
P2=$!
wait $P1; wait $P2
echo "[e84] sinh xong $(date '+%F %T')" | tee -a $L/run.log
python scripts/e84_recipe_factorial.py 2>&1 | grep -v -i warn | tee -a $L/run.log
