#!/bin/bash
set -e
cd /workspace/transfer-attack-final && source .venv/bin/activate
C="OSFD GEO-C0 PIA PROBE-osfd+cls C2-PIA"
python scripts/mech_swinl_geo.py --subset dev100 --conds $C --gen pia_dev100 --out swinl_geo_dev100.json
python scripts/diag_decoder_output.py --subset dev100 --conds $C --select maxlogit --out decoder_output_dev100.json
echo MECH_DONE
