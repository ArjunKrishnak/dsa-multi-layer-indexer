#!/bin/bash
# Import the stock SGLang v0.5.16 container as an enroot squashfs ($WORK/sglang-v0.5.16-cu130.sqsh).
# Run on a compute node (the squashfs pack is CPU/IO heavy). Set LOCAL_TMP to fast node-local disk.
#   WORK=/path/to/workspace sbatch -p <partition> scripts/setup/import_image.sbatch
#SBATCH --job-name=mli-import-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/import_%j.out
set -euo pipefail
WORK="${WORK:?set WORK}"
LOCAL="${LOCAL_TMP:-/tmp}/enroot-import-${SLURM_JOB_ID}"
mkdir -p "$WORK" "$LOCAL"/{run,data,tmp}
trap 'rm -rf "$LOCAL"' EXIT
export ENROOT_CACHE_PATH="$WORK/.enroot-cache" ENROOT_RUNTIME_PATH="$LOCAL/run"
export ENROOT_DATA_PATH="$LOCAL/data" ENROOT_TEMP_PATH="$LOCAL/tmp" TMPDIR="$LOCAL/tmp"
enroot import -o "$LOCAL/sglang.sqsh" docker://lmsysorg/sglang:v0.5.16-cu130
mv "$LOCAL/sglang.sqsh" "$WORK/sglang-v0.5.16-cu130.sqsh"
ls -la "$WORK/sglang-v0.5.16-cu130.sqsh"
