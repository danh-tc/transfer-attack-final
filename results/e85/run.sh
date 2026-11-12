#!/usr/bin/env bash
# E85 (bước 3b, trục mới): tầng feature × họ surrogate, 4 surrogate, dev100. Chạy trong tmux:
#   tmux new -s e85 'bash results/e85/run.sh'      # tách: Ctrl-b d | xem lại: tmux attach -t e85
# 2 worker sinh song song (cân tải: R50+SWT | CNX+R101), xong mới eval. Resume an toàn (bỏ PNG đã có).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
L=results/e85
F=RRB-C25,RRB-C35,OCT-C25,OCT-C35
pre() { echo "$F" | sed "s/[^,]*/$1-&/g"; }
python scripts/e85_surrogate_stage.py --gen-only --cells $(pre R50),$(pre SWT),SWT-CLS > $L/gen_a.log 2>&1 &
P1=$!
python scripts/e85_surrogate_stage.py --gen-only --cells $(pre CNX),$(pre R101),R101-CLS > $L/gen_b.log 2>&1 &
P2=$!
wait $P1; wait $P2
echo "[e85] sinh xong $(date '+%F %T')" | tee -a $L/run.log
python scripts/e85_surrogate_stage.py 2>&1 | grep -v -i warn | tee -a $L/run.log
