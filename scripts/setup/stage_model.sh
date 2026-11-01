# stage_model.sh -- optional: copy the FP8 checkpoint to node-local disk before serving.
# Sourced by scripts/{speed,quality}.sbatch when STAGE_DIR is set, with SRC (checkpoint)
# and DST ($STAGE_DIR/<name>). Sets MODEL=$DST, or exits 3 if the disk lacks room.
# Loading 640-700 GB of weights from a network filesystem works but cold start is slow.
_NS=$(ls "$SRC"/*.safetensors 2>/dev/null | wc -l)
_ND=$(ls "$DST"/*.safetensors 2>/dev/null | wc -l)
if [ "$_NS" -gt 0 ] && [ "$_ND" -ge "$_NS" ]; then
    echo "### stage: using already-staged $DST ($_ND/$_NS shards) ###"
    MODEL="$DST"
else
    _FREE=$(df -BG --output=avail "$(dirname "$DST")" 2>/dev/null | tail -1 | tr -dc '0-9')
    _NEED=$(du -sBG "$SRC" 2>/dev/null | cut -f1 | tr -dc '0-9'); _NEED=$(( ${_NEED:-0} + ${_NEED:-0} / 20 ))
    if [ -z "$_FREE" ] || [ "$_FREE" -lt "$_NEED" ]; then
        echo "FATAL: ${_FREE:-?}G free under $(dirname "$DST"), need ${_NEED}G to stage $SRC on $(hostname)." >&2
        exit 3
    fi
    mkdir -p "$DST"; echo "### stage: $SRC -> $DST (free=${_FREE}G) ###"
    for _try in 1 2 3; do   # rsync --partial is idempotent; retry transient failures
        rsync -a --partial "$SRC"/ "$DST"/ 2>&1 | tail -1
        _ND=$(ls "$DST"/*.safetensors 2>/dev/null | wc -l)
        [ "$_ND" -ge "$_NS" ] && break
        echo "### stage: attempt $_try incomplete ($_ND/$_NS shards), retrying ###" >&2; sleep 5
    done
    [ "$_ND" -ge "$_NS" ] || { echo "FATAL: staging incomplete ($_ND/$_NS shards)." >&2; exit 3; }
    MODEL="$DST"
fi
