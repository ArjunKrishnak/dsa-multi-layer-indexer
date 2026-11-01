#!/bin/bash
# Phase-1 cross-layer KL distillation of the lagged lightning indexer.
# Produces a {"layers": {...}} payload served via the index_distilled_weights override
# via --distilled-indexer.
#
# Required env vars:
#   MODEL          deepseek | glm5 | glm5.1
#   CKPT_PATH      absolute path to checkpoint directory
#
# Optional env vars:
#   DATASET        govreport | arxiv (default: govreport)
#   SEQ            training sequence length (default: 16384)
#   STEPS          training steps (default: 300)
#   NUM_STREAMS    packed training streams (default: 8)
#   PACK_DOCS      docs to pack for the corpus (default: 256)
#   LR             learning rate (default: 1e-3)
#   SEED           rng seed (default: 0)
#   OUTPUT         output payload path (default: results/distill/indexer_lag_<model>.pt)
#
# Example:
#   MODEL=glm5 CKPT_PATH=/path/to/GLM-5-FP8 SEQ=16384 \
#       sbatch --nodes=2 scripts/distill/distill_indexer.sbatch
#SBATCH --job-name=distill-idx
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --signal=B:SIGTERM@120
# Relative to the submit directory (run from the repo root); override with
# `sbatch --output=...` for a different location.
#SBATCH --output=logs/distill_%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"

MODEL="${MODEL:?MODEL must be set (deepseek | glm5 | glm5.1)}"
CKPT_PATH="${CKPT_PATH:?CKPT_PATH must be set}"
DATASET="${DATASET:-govreport}"
SEQ="${SEQ:-16384}"
STEPS="${STEPS:-300}"
NUM_STREAMS="${NUM_STREAMS:-8}"
PACK_DOCS="${PACK_DOCS:-256}"
LR="${LR:-1e-3}"
SEED="${SEED:-0}"
OBJECTIVE="${OBJECTIVE:-kl}"
RANK_PAIRS="${RANK_PAIRS:-1024}"
RANK_MARGIN="${RANK_MARGIN:-0.0}"
LIST_MLE_CANDIDATES="${LIST_MLE_CANDIDATES:-2048}"
KL_WEIGHT="${KL_WEIGHT:-0.5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
REFRESH_LAYERS="${REFRESH_LAYERS:-}"
SPAN_INCLUDE_SELF="${SPAN_INCLUDE_SELF:-}"   # 1 = IndexCache-style spans (retained layer is its own consumer)

MODEL_TAG="${MODEL//./}"
OUTPUT="${OUTPUT:-results/distill/indexer_lag_${MODEL_TAG}.pt}"

NNODES="${SLURM_NNODES:-1}"
if [[ -n "${NODES:-}" && "$NODES" != "$NNODES" ]]; then
    echo "ERROR: NODES=$NODES was set but Slurm allocated $NNODES node(s). Re-submit with sbatch --nodes=$NODES." >&2
    exit 2
fi
NPROC=8
# Use raw IP so env:// TCPStore is reachable cross-node (hostname not always routable).
_RAW=$(scontrol show hostname "$SLURM_JOB_NODELIST" | sed -n '1p')
export MASTER_ADDR
if [[ "$_RAW" =~ ^ip-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+)$ ]]; then
    MASTER_ADDR="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}.${BASH_REMATCH[4]}"
else
    MASTER_ADDR="$(getent hosts "$_RAW" | awk '{print $1; exit}' || true)"
    MASTER_ADDR="${MASTER_ADDR:-$_RAW}"
fi
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))

# Build CUDA extension on compute node if not already available.
if ! uv run python -c "import fast_hadamard_transform" 2>/dev/null; then
    echo "Building fast-hadamard-transform from git (CUDA extension required by indexer)..."
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
    export PATH="$CUDA_HOME/bin:$PATH"
    uv pip install --no-build-isolation \
        'fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@e7706faf8d1c3b9f241e36860640ad1dac644ede' \
        2>&1 | tail -5
fi

echo "=== distill-indexer ==="
echo "Job: $SLURM_JOB_ID  Nodes: $NNODES  GPUs/node: $NPROC"
echo "Model: $MODEL  CkptPath: $CKPT_PATH  Dataset: $DATASET"
echo "Seq: $SEQ  Steps: $STEPS  Streams: $NUM_STREAMS  LR: $LR  Seed: $SEED"
echo "Objective: $OBJECTIVE  RankPairs: $RANK_PAIRS  RankMargin: $RANK_MARGIN"
echo "ListMLECandidates: $LIST_MLE_CANDIDATES  KLWeight: $KL_WEIGHT  GradClip: $GRAD_CLIP"
[[ -n "$REFRESH_LAYERS" ]] && echo "RefreshLayers: $REFRESH_LAYERS (span-aware)"
echo "Output: $OUTPUT"
nvidia-smi -L

trap 'echo "SIGTERM received — exiting for Slurm/preemption handling"; exit 143' SIGTERM

WORLD_SIZE=$((NNODES * NPROC))
export WORLD_SIZE

srun \
  --ntasks="$WORLD_SIZE" \
  --ntasks-per-node="$NPROC" \
  bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    # Reduce allocator fragmentation so the freed dense-prefill attention
    # buffers can be reused across layers without reserving extra segments.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    cd "$1" || exit 1
    exec uv run python -m lagged_indexer_dsa.distill.train \
      --model "$2" \
      --ckpt-path "$3" \
      --dataset "$4" \
      --seq "$5" \
      --steps "$6" \
      --num-streams "$7" \
      --pack-docs "$8" \
      --lr "$9" \
      --seed "${10}" \
      --output "${11}" \
      --objective "${12}" \
      --rank-pairs "${13}" \
      --rank-margin "${14}" \
      --list-mle-candidates "${15}" \
      --kl-weight "${16}" \
      --grad-clip "${17}" \
      ${18:+--refresh-layers "${18}"} \
      $( [ "${19:-}" = "1" ] && printf %s --span-include-self )
  ' _ "${PROJECT_ROOT}" "${MODEL}" "${CKPT_PATH}" "${DATASET}" "${SEQ}" "${STEPS}" "${NUM_STREAMS}" "${PACK_DOCS}" "${LR}" "${SEED}" "${OUTPUT}" "${OBJECTIVE}" "${RANK_PAIRS}" "${RANK_MARGIN}" "${LIST_MLE_CANDIDATES}" "${KL_WEIGHT}" "${GRAD_CLIP}" "${REFRESH_LAYERS}" "${SPAN_INCLUDE_SELF:-}"

echo "=== distill-indexer complete ==="
