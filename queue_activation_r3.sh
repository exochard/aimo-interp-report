#!/usr/bin/env bash
set -euo pipefail
if [[ "${AIMO_SCOPED:-0}" != 1 ]]; then
    exec env XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
        systemd-run --user --scope -q -p CollectMode=inactive-or-failed -p MemoryMax=14G \
        -p MemorySwapMax=0 -p CPUQuota=200% nice -n 19 env AIMO_SCOPED=1 bash "$0"
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY=/home/peppe/AI/var/venvs/k12-vl/bin/python
RUN=contests/aimo-interp/runs/activation-r3
REPORT=coordination/2026-10-14-r3/report-codex.md
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"  # Immutable attempt; a second invocation cannot reuse old evidence.
exec >> "$RUN/queue.log" 2>&1
echo "QUEUE $(date -Iseconds)"
release=$(date -d '2026-10-14 13:30:00 +0200' +%s)
last_start=$(date -d '2026-10-14 15:30:00 +0200' +%s)
ready=0
while (( $(date +%s) < last_start )); do
    if (( $(date +%s) >= release )) && python3 contests/aimo-interp/gpu_available.py; then
        ready=1
        break
    fi
    sleep 30
done
extraction_exit=75
fit_exit=75
reason='GPU slot unavailable before the bounded queue expired'
if (( ready )); then
    echo "START $(date -Iseconds)"
    free -g
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    set +e
    python3 contests/aimo-interp/gpu_available.py --run -- \
        timeout --signal=TERM --kill-after=10s 3600 python3 contests/geoai-arctic/src/exclusive_gpu_run.py \
        --lock /tmp/athlon-gpu.lock -- "$PY" contests/aimo-interp/run_activation_probe.py \
        --out "$RUN/features" --minutes 58
    extraction_exit=$?
    set -e
    reason='extraction failed or timed out; inspect queue.log'
    if (( extraction_exit == 0 )); then
        # A 34-by-34 analytic solve has no GPU path; this stays in the same bounded scope.
        set +e
        CUDA_VISIBLE_DEVICES= "$PY" contests/aimo-interp/fit_activation_probe.py \
            --features "$RUN/features" --out "$RUN/probe.json"
        fit_exit=$?
        set -e
        if (( fit_exit == 0 )); then
            reason='complete nested validation'
        else
            reason='feature coverage or nested evaluation failed; inspect queue.log'
        fi
    fi
fi
python3 - "$RUN" "$extraction_exit" "$fit_exit" "$reason" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
out,code,fit_code,reason=sys.argv[1:]
probe=Path(out,'probe.json')
probe_hash=hashlib.sha256(probe.read_bytes()).hexdigest() if int(code)==0 and int(fit_code)==0 else None
Path(out,'terminal.json').write_text(json.dumps(dict(extraction_exit=int(code),fit_exit=int(fit_code),
    probe_sha256=probe_hash,reason=reason,
    finished=datetime.datetime.now().astimezone().isoformat()),indent=2))
PY
python3 contests/aimo-interp/record_activation_result.py "$RUN" "$REPORT"
# Restrict the local commit to the report; other staged lane work stays untouched.
git commit --only -m 'docs(aimo): record activation-probe gate outcome' -- "$REPORT"
echo "TERMINAL $(date -Iseconds) $reason"
cat "$RUN/terminal.json"
