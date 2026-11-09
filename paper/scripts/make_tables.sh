#!/usr/bin/env bash
# Regenerate every table in ../tables/ from its script.
# Usage:  ./make_tables.sh            (all tables)
#         ./make_tables.sh qcorr      (only scripts whose name contains "qcorr")
# Python: $PYTHON if set, else "python" on PATH.
# table_n126_gs.py reads its cache ../data/n126_gs.txt; without the cache it needs the raw pnFAM outputs ($PNFAM_RAW).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

SCRIPTS=(table_calibsets.py table_posteriors.py table_pool_nuclei.py table_qcorr.py table_n126_gs.py)
FILTER="${1:-}"

fail=0
for s in "${SCRIPTS[@]}"; do
    [[ -n "$FILTER" && "$s" != *"$FILTER"* ]] && continue
    echo "=== $s"
    out="$("$PYTHON" "$HERE/$s" 2>&1)"; rc=$?
    # drop the ArviZ refactor notice, keep everything else
    printf '%s\n' "$out" | grep -v -e "FutureWarning" -e "^  warn(" -e "ArviZ is undergoing" \
        -e "migration guidance" -e "backward incompatible" -e "^\s*$" || true
    if [[ $rc -ne 0 ]]; then
        echo "!!! $s FAILED (exit $rc)"; fail=1
    fi
done
[[ $fail -eq 0 ]] && echo "all tables written to $(cd "$HERE/../tables" && pwd)" || { echo "some tables failed (see above)"; exit 1; }
