#!/bin/bash
# Indexer-only distillation runs used in the paper (Sec. 3.3, App. "Distillation recipe").
# One Slurm job per (model, retention 1/F, span rule, seed), each training only the
# retained layers' lightning indexers for 300 steps on GovReport (2 nodes x 8 GPUs).
# Outputs land in $OUT_DIR (default $WORK/distill, which scripts/quality.sbatch mounts):
#
#   F=1                  ab_<model>_hybrid.pt                    (MLI, every layer retained)
#   F>1, SPAN_MODE=lag   ab_<model>_hybrid_spanaware_f<F>.pt     (IndexCache + MLI)
#   F>1, SPAN_MODE=ic    ab_<model>_hybrid_spanaware_ic_f<F>.pt  (IndexCache, its own spans)
#   SEED!=0 appends _s<SEED>.
#
# Usage (from the repo root; dry run unless --submit):
#   DS_CKPT=/path/DeepSeek-V3.2 GLM_CKPT=/path/GLM-5-FP8 WORK=/path/ws \
#     scripts/distill/launch_distill.sh [--submit]
# Env: MODELS ("deepseek glm5"), FREQS ("1 2 4 8 16 32"), SPAN_MODES ("lag ic"),
#      SEEDS ("0"), STEPS (300), NODES (2), SBATCH_ARGS (e.g. "-p <partition>").
set -euo pipefail
MODELS="${MODELS:-deepseek glm5}"
FREQS="${FREQS:-1 2 4 8 16 32}"
SPAN_MODES="${SPAN_MODES:-lag ic}"
SEEDS="${SEEDS:-0}"
STEPS="${STEPS:-300}"
NODES="${NODES:-2}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
OUT_DIR="${OUT_DIR:-${WORK:?set WORK or OUT_DIR}/distill}"
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1
mkdir -p "$OUT_DIR" logs

declare -A LAYERS=([deepseek]=61 [glm5]=78)
declare -A SEQ_FOR=([deepseek]=32768 [glm5]=16384)
ckpt_for () { case "$1" in deepseek) echo "${DS_CKPT:?set DS_CKPT}";; glm5) echo "${GLM_CKPT:?set GLM_CKPT}";; esac; }

# Full (indexer-running) layers under IndexCache retention 1/F, matching the overlay's
# rule (model_config.dsa_layer_skips_topk): layer l is Full iff max(l-1, 0) % F == 0.
refresh_layers () {
  local n="$1" f="$2" out="" l m
  for ((l = 0; l < n; l++)); do m=$(( l > 0 ? l - 1 : 0 )); (( m % f == 0 )) && out+="${out:+,}$l"; done
  echo "$out"
}

for model in $MODELS; do
  ckpt="$(ckpt_for "$model")"
  for f in $FREQS; do
    for mode in $SPAN_MODES; do
      [[ "$f" == 1 && "$mode" == ic ]] && continue   # IndexCache at F=1 is the unmodified model
      for seed in $SEEDS; do
        if [[ "$f" == 1 ]]; then stem="ab_${model}_hybrid"; layers=""; sis=""
        else
          sfx=""; sis=""; [[ "$mode" == ic ]] && { sfx="_ic"; sis=1; }
          stem="ab_${model}_hybrid_spanaware${sfx}_f${f}"; layers="$(refresh_layers "${LAYERS[$model]}" "$f")"
        fi
        [[ "$seed" != 0 ]] && stem="${stem}_s${seed}"
        out="$OUT_DIR/$stem.pt"
        [[ -f "$out" ]] && { echo "SKIP $out (exists)" >&2; continue; }
        echo "MODEL=$model F=$f span=$mode seed=$seed -> $out"
        if [[ "$SUBMIT" == 1 ]]; then
          MODEL="$model" CKPT_PATH="$ckpt" DATASET=govreport SEQ="${SEQ_FOR[$model]}" STEPS="$STEPS" \
          OBJECTIVE=hybrid SEED="$seed" REFRESH_LAYERS="$layers" SPAN_INCLUDE_SELF="$sis" OUTPUT="$out" \
            sbatch $SBATCH_ARGS --nodes="$NODES" --job-name="distill-$stem" scripts/distill/distill_indexer.sbatch
        fi
      done
    done
  done
done
