#!/bin/bash
# Fetch the LongBench summarization tasks (gov_report, multi_news, qmsum) that
# eval/eval_longbench_served.py reads offline from $WORK/lb_extra/<task>.jsonl
# (mounted in the container at /hh/lb_extra). Source: the official THUDM/LongBench data.zip.
#   WORK=/path/to/workspace scripts/setup/fetch_longbench_summ.sh
set -euo pipefail
DEST="${WORK:?set WORK}/lb_extra"; mkdir -p "$DEST"
curl -fL -o "$DEST/data.zip" https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip -o -j "$DEST/data.zip" data/gov_report.jsonl data/multi_news.jsonl data/qmsum.jsonl -d "$DEST"
ls -la "$DEST"/*.jsonl
