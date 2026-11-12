#!/usr/bin/env bash
# E83 (bước 3a): factorial dev100. Chạy trong tmux để ngắt SSH vẫn chạy:
#   tmux new -s e83 'bash results/e83/run.sh'      # tách: Ctrl-b d | xem lại: tmux attach -t e83
# 2 worker sinh song song (mỗi surrogate 1 process), xong mới eval + phán quyết. Resume an toàn (bỏ PNG đã có).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
# Nghẽn CPU (GPU ~30%): 2 process × ~176 thread tranh nhau -> giới hạn thread intra-op mỗi process.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
L=results/e83
python scripts/e83_factorial.py --gen-only \
  --cells R50-FEAT,R50-CLS,R50-FC,R50-FC-L05,R50-FC-L2,OSFD-R50,RD-R50 > $L/gen_r50.log 2>&1 &
P1=$!
python scripts/e83_factorial.py --gen-only \
  --cells CNX-FEAT,CNX-CLS,CNX-FC,CNX-FC-L05,CNX-FC-L2,OSFD-CNX,RD-CNX > $L/gen_cnx.log 2>&1 &
P2=$!
wait $P1; wait $P2
echo "[e83] sinh xong $(date '+%F %T')" | tee -a $L/run.log
python scripts/e83_factorial.py 2>&1 | grep -v -i warn | tee -a $L/run.log
