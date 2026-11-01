#!/bin/bash
# Quality runner: boots each arm once and runs every served evaluation against it
# (LongBench 11 tasks x100, needle retrieval, decode perplexity x5 seeds, RULER 4K-128K,
# LongBench v2). Selections are bit-identical between the two MLI realizations, so the
# MLI arms here use the deferred stream; the arm names keep the historical "lag_fused"
# stem so result files match the paper's table generators.
#
# Arms (name -> override):
#   base                     unmodified DSA
#   lag_fused                MLI (every indexer retained)
#   lag_fused_d              + distilled indexers ($DISTILL_DIR/ab_<model>_hybrid.pt)
#   ic_f{F}                  IndexCache at retention 1/F
#   ic_lag_fused_f{F}        IndexCache + MLI
#   ic_lag_fused_d_f{F}      + distilled indexers (ab_<model>_hybrid_spanaware_f{F}.pt)
#   ic_d_f{F}                IndexCache + its own distilled indexers (ab_<model>_hybrid_spanaware_ic_f{F}.pt)
#   ic_s2 / ic_s4 / ic_lag_fused_s2 / ic_lag_fused_s4   GLM-5 IndexCache searched patterns (+ MLI)
#   *_s{SEED}                seed replicates of a distilled arm (..._f{F}_s{SEED}.pt)
#
# Required env: WORK, SGLANG_SRC, MODEL_PATH, MODEL_TAG (ds | glm5), ARMS.
# Optional: SQSH, STAGE_DIR, DISTILL_DIR ($WORK/distill), EVALS ("lb1 niah ppl ruler lbv2"),
#   LBV2_MAX_SAMPLES (150; 0 = all 503, as in the paper), RULER_LENGTHS, RULER_DEPTHS, RULER_N.
# Usage (from the repo root):
#   MODEL_TAG=ds ARMS="base lag_fused ic_f4 ic_lag_fused_f4" RULER_N=3 LBV2_MAX_SAMPLES=0 \
#     sbatch -p <partition> scripts/quality.sbatch
#SBATCH --job-name=mli-quality
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/quality_%j.out
set -uo pipefail
MODEL_TAG="${MODEL_TAG:?MODEL_TAG must be set (ds | glm5)}"
ARMS="${ARMS:?ARMS must be set (space-separated arm names)}"
EVALS="${EVALS:-lb1 niah ppl ruler lbv2}"
LBV2_MAX_SAMPLES="${LBV2_MAX_SAMPLES:-150}"
LBV2_THINK="${LBV2_THINK:-0}"          # 1 = run LongBench v2 in thinking mode (reasoning-model-faithful)
LBV2_MAXNEW="${LBV2_MAXNEW:-128}"      # answer/reasoning token budget (raise for --think)
RULER_LENGTHS="${RULER_LENGTHS:-4096,8192,16384,32768,65536,131072}"
RULER_DEPTHS="${RULER_DEPTHS:-10,50,90}"
RULER_N="${RULER_N:-1}"   # samples per (task, length, depth) cell; >1 writes <arm>_ruler_n<N>.json
RESBASE="${RESBASE:-lcfm_quality}"   # override to isolate probe runs from the paper tables (e.g. collapse_probe)
# ---- configuration (see README "Reproducing the paper") ----------------------
WORK="${WORK:?set WORK to a writable workspace (caches, wheel, results)}"
SQSH="${SQSH:-$WORK/sglang-v0.5.16-cu130.sqsh}"
SGLANG_SRC="${SGLANG_SRC:?set SGLANG_SRC to an sglang v0.5.16 clone with sglang_overlay/mli-sglang-v0.5.16.patch applied}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO/sglang_overlay/mli-sglang-v0.5.16.patch" ] || { echo "REPO=$REPO is not this repository (submit from the repo root or set REPO)" >&2; exit 2; }
HH="$WORK"; HH_SRC="$SGLANG_SRC"; NLP="$REPO"
case "$MODEL_TAG" in
  ds)   MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V3.2 FP8 checkpoint}"; SERVER_EXTRA=""; EVAL_EXTRA="--deepseek-chat --model-path /model"
        DW_EVERY="ab_deepseek_hybrid.pt"; DW_SPAN_PREFIX="ab_deepseek_hybrid_spanaware_f"; LBV2_CT="" ; LBV2_RAW=1 ;;
  glm5) MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the GLM-5 FP8 checkpoint}"; SERVER_EXTRA="--reasoning-parser glm45"
        EVAL_EXTRA="--glm-raw --model-path /model"; DW_EVERY="ab_glm5_hybrid.pt"; DW_SPAN_PREFIX="ab_glm5_hybrid_spanaware_f"
        LBV2_CT='{"enable_thinking":false}'; LBV2_RAW=0 ;;
  *) echo "MODEL_TAG must be ds or glm5" >&2; exit 1 ;;
esac
# Thinking-mode LongBench v2: GLM flips its chat template's enable_thinking; DS (raw
# prompt) is handled by the eval's --think flag (omits the forced </think>).
if [ "$LBV2_THINK" = 1 ] && [ "$LBV2_RAW" = 0 ]; then LBV2_CT='{"enable_thinking":true}'; fi
# Optional: stage the weights to node-local disk first (STAGE_DIR); otherwise load MODEL_PATH directly.
if [ -n "${STAGE_DIR:-}" ]; then
  SRC="$MODEL_PATH"; DST="$STAGE_DIR/$(basename "$MODEL_PATH")"
  source "$REPO/scripts/setup/stage_model.sh"
else
  MODEL="$MODEL_PATH"
fi
JOB=$SLURM_JOB_ID
echo "### QUALITY job=$JOB model=$MODEL_TAG node=$(hostname) arms=[$ARMS] evals=[$EVALS] ###"
DISTILL_DIR="${DISTILL_DIR:-$WORK/distill}"; mkdir -p "$DISTILL_DIR"
MOUNTS="${MODEL}:/model:ro,${HH}:/hh:rw,${HH_SRC}/python/sglang:/usr/local/lib/python3.12/dist-packages/sglang:ro,${NLP}:/research:ro,${DISTILL_DIR}:/distill:ro,${HH_SRC}:/hh/src"
srun --container-image="$SQSH" --container-mounts="$MOUNTS" --container-workdir=/hh bash -lc '
  set -uo pipefail
  export HOME=/hh/.container_home HF_HOME=/hh/.hf_cache
  export TORCHINDUCTOR_CACHE_DIR=/hh/.inductor_cache TRITON_CACHE_DIR=/hh/.triton_cache
  export TORCH_EXTENSIONS_DIR=/hh/.torch_ext_hide
  export PYTHONPATH=/research/src:/research/kernels:/research/eval:${PYTHONPATH:-}
  [ -n "'"${HH_HIDE_NSPLIT:-}"'" ] && export HH_HIDE_NSPLIT='"${HH_HIDE_NSPLIT:-16}"'
  [ -n "'"${HH_HIDE_NSPLIT_ATTN:-}"'" ] && export HH_HIDE_NSPLIT_ATTN='"${HH_HIDE_NSPLIT_ATTN:-16}"'
  mkdir -p "$HOME" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
# sgl_kernel = stock sgl-kernel 0.4.5 (public) + our patched flash_mla.py wrapper (pure Python).
  # Fail LOUD and never delete the container sgl_kernel unless a replacement built OK (repro fix).
  SGLK=/tmp/sglk_$$; rm -rf "$SGLK"
  WHEELOUT="${WHEELOUT:-/hh/wheelout}"
  if ls "$WHEELOUT"/*.whl >/dev/null 2>&1; then
    pip install --target="$SGLK" --no-deps --force-reinstall "$WHEELOUT"/*.whl 2>&1 | tail -1
  else
    echo "[setup] no wheel in $WHEELOUT; trying stock sgl-kernel 0.4.5 from PyPI (see README.md)"
    pip install --target="$SGLK" --no-deps sgl-kernel==0.4.5 2>&1 | tail -2 || true
  fi
  if [ ! -d "$SGLK/sgl_kernel" ]; then
    echo "FATAL setup: sgl_kernel not installed. Provide $WHEELOUT/sglang_kernel-0.4.5-cp310-abi3-linux_x86_64.whl (sha256 7e6ff5684eb84f41a65ea66ed7517765b64c3b97410703c458d6fd2e68f1a171) or a reachable PyPI. See README.md. Container sgl_kernel left intact. Aborting."
    exit 3
  fi
  [ -f /hh/src/sgl-kernel/python/sgl_kernel/flash_mla.py ] && cp /hh/src/sgl-kernel/python/sgl_kernel/flash_mla.py "$SGLK/sgl_kernel/flash_mla.py"
  SP_SGLK=/usr/local/lib/python3.12/dist-packages/sgl_kernel
  rm -rf "$SP_SGLK"; cp -r "$SGLK/sgl_kernel" "$SP_SGLK"
  python3 -c "import sgl_kernel" 2>/dev/null || { echo "FATAL setup: sgl_kernel import failed after install. Aborting."; exit 3; }
  export PYTHONPATH="$SGLK:${PYTHONPATH:-}"
  JOB='"$JOB"'; MODEL_TAG='"$MODEL_TAG"'; ARMS="'"$ARMS"'"; EVALS="'"$EVALS"'"
  SERVER_EXTRA="'"$SERVER_EXTRA"'"; EVAL_EXTRA="'"$EVAL_EXTRA"'"; LBV2_CT_B64='"$(printf %s "$LBV2_CT" | base64 -w0)"'
  DW_EVERY='"$DW_EVERY"'; DW_SPAN_PREFIX='"$DW_SPAN_PREFIX"'; LBV2_RAW='"$LBV2_RAW"'
  LBV2_MAX='"$LBV2_MAX_SAMPLES"'; RULER_LENGTHS='"$RULER_LENGTHS"'; RULER_DEPTHS='"$RULER_DEPTHS"'; RULER_N='"$RULER_N"'
  RULER_TAG="ruler"; [ "$RULER_N" != 1 ] && RULER_TAG="ruler_n${RULER_N}"
  LBV2_THINK='"$LBV2_THINK"'; LBV2_MAXNEW='"$LBV2_MAXNEW"'
  RESBASE='"${RESBASE:-lcfm_quality}"'; EXTRA_OVERRIDE='"${EXTRA_OVERRIDE:-}"'
  # Thinking runs write to a distinct file (lbv2_think) so they neither collide with nor
  # get resume-skipped by an existing non-thinking lbv2 result for the same arm.
  LBV2_TAG="lbv2"; [ "$LBV2_THINK" = 1 ] && LBV2_TAG="lbv2_think"
  # The full 503-question set writes its own file (lbv2full) so it neither overwrites nor
  # gets resume-skipped by the 150-question subsample; the table generator prefers it.
  [ "$LBV2_MAX" = 0 ] && LBV2_TAG="${LBV2_TAG}full"
  BK="--dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv --kv-cache-dtype fp8_e4m3"
  # IndexCache published greedy-searched GLM-5 patterns (78 layers; 1/2: 40 F, 1/4: 22 F)
  PAT_S2="FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"; PAT_S4="FFSFSSSFSSFSFSSSSSSSFSSSSSSFSSSFSFSSSSFFFFFSSSFFSSSFSSSSSSSSFSSSFSSSSSSFSFSSSS"
  # DeepSeek-V3.2 ships no tokenizer chat template (the LongBench v1 eval sends raw text
  # too), so LBv2 uses the raw prompt there; GLM-5 uses its template with thinking off.
  LBV2_ARGS=(); if [ "$LBV2_RAW" = 1 ]; then LBV2_ARGS=(--no-chat-template --prompt-format deepseek_chat); elif [ -n "$LBV2_CT_B64" ]; then LBV2_ARGS=(--chat-template-kwargs "$(echo "$LBV2_CT_B64" | base64 -d)"); fi
  LBV2_ARGS+=(--max-new-tokens "$LBV2_MAXNEW")
  [ "$LBV2_THINK" = 1 ] && [ "$LBV2_RAW" = 1 ] && LBV2_ARGS+=(--think)
  # Resume: an (arm, eval) whose complete output already exists in ANY earlier
  # lcfm_quality_<model>_* dir (a preempted/requeued predecessor) is skipped. The
  # table generators merge across dirs, so nothing is copied.
  done_eval () {  # NAME EV -> 0 if complete output exists
    local n="$1" ev="$2" d f
    for d in /hh/results/${RESBASE}_${MODEL_TAG}_*; do
      case "$ev" in
        lb1)  f="$d/${n}_lb1.json"; [ -f "$f" ] && python3 -c "import json,sys; j=json.load(open(\"$f\")); sys.exit(0 if (not j.get(\"partial\") and j.get(\"completed\")==j.get(\"total\")) else 1)" && return 0 ;;
        niah) local k=0; for C in 32000 130000 260000 520000; do for M in single aggregate; do [ -f "$d/${n}_niah_c${C}_${M}.json" ] && k=$((k+1)); done; done; [ "$k" = 8 ] && return 0 ;;
        ppl)  local k=0; for IL in 4096 16384 32768 65536; do for S in 1234 2345 3456 4567 5678; do [ -f "$d/${n}_ppl_in${IL}_s${S}.json" ] && k=$((k+1)); done; done; [ "$k" = 20 ] && return 0 ;;
        ruler) [ -f "$d/${n}_${RULER_TAG}.json" ] && return 0 ;;
        lbv2)  [ -f "$d/${n}_${LBV2_TAG}.json" ] && return 0 ;;
      esac
    done
    return 1
  }
  TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,passage_count,passage_retrieval_en"
  RESDIR=/hh/results/${RESBASE}_${MODEL_TAG}_${JOB}; mkdir -p "$RESDIR"
  override_for () {  # arm name -> json override (empty = unknown arm)
    local a="$1" F
    case "$a" in
      base) echo "{\"index_topk_freq\": 1}";;
      lag_fused) echo "{\"index_topk_freq\": 1, \"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true}";;
      lag_fused_d) echo "{\"index_topk_freq\": 1, \"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true, \"index_distilled_weights\": \"/distill/$DW_EVERY\"}";;
      ic_f*) F=${a#ic_f}; echo "{\"index_topk_freq\": $F}";;
      # Seed replicates of a distillation: ic_d_f8_s1 / ic_lag_fused_d_f8_s1 load ..._f8_s1.pt
      # (same recipe, SEED=1) and write their own result files.
      ic_d_f*_s*) F=${a#ic_d_f}; S=${F#*_s}; F=${F%_s*}; echo "{\"index_topk_freq\": $F, \"index_distilled_weights\": \"/distill/${DW_SPAN_PREFIX/spanaware_f/spanaware_ic_f}${F}_s${S}.pt\"}";;
      ic_lag_fused_d_f*_s*) F=${a#ic_lag_fused_d_f}; S=${F#*_s}; F=${F%_s*}; echo "{\"index_topk_freq\": $F, \"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true, \"index_distilled_weights\": \"/distill/${DW_SPAN_PREFIX}${F}_s${S}.pt\"}";;
      ic_d_f*) F=${a#ic_d_f}; echo "{\"index_topk_freq\": $F, \"index_distilled_weights\": \"/distill/${DW_SPAN_PREFIX/spanaware_f/spanaware_ic_f}${F}.pt\"}";;
      ic_s2) echo "{\"index_topk_pattern\": \"$PAT_S2\"}";;
      ic_s4) echo "{\"index_topk_pattern\": \"$PAT_S4\"}";;
      ic_lag_fused_s2) echo "{\"index_topk_pattern\": \"$PAT_S2\", \"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true}";;
      ic_lag_fused_s4) echo "{\"index_topk_pattern\": \"$PAT_S4\", \"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true}";;
      ic_lag_fused_d_f*) F=${a#ic_lag_fused_d_f}; echo "{\"index_topk_freq\": $F, \"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true, \"index_distilled_weights\": \"/distill/${DW_SPAN_PREFIX}${F}.pt\"}";;
      ic_lag_fused_f*) F=${a#ic_lag_fused_f}; echo "{\"index_topk_freq\": $F, \"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true}";;
      *) echo "";;
    esac
  }
  for NAME in $ARMS; do
    OV=$(override_for "$NAME")
    [ -z "$OV" ] && { echo "UNKNOWN ARM $NAME -- skipped"; continue; }
    # Merge EXTRA_OVERRIDE (e.g. {\"index_prefill_full_indexer\": true}) into every arm JSON by string surgery (no nested quotes).
    if [ -n "$EXTRA_OVERRIDE" ]; then OV="${OV%\}}, ${EXTRA_OVERRIDE#\{}"; fi
    case "$OV" in *index_distilled_weights*) DWP=$(echo "$OV" | sed -E "s/.*\"index_distilled_weights\": \"([^\"]+)\".*/\1/"); [ -f "$DWP" ] || { echo "SKIP $NAME: $DWP not found"; continue; };; esac
    LOG=$RESDIR/server_${NAME}.log
    TODO=""; for EV in $EVALS; do done_eval "$NAME" "$EV" || TODO="$TODO $EV"; done
    if [ -z "$TODO" ]; then echo "###### QUALITY $MODEL_TAG arm=$NAME: all evals already complete (resume) -- skipped ######"; continue; fi
    echo "###### QUALITY $MODEL_TAG arm=$NAME ov=$OV evals_todo=[$TODO] ######"
    python3 -m sglang.launch_server --model-path /model --trust-remote-code --tp-size 8 \
      --mem-fraction-static 0.82 --context-length 140000 --max-total-tokens 560000 \
      --cuda-graph-max-bs-decode 4 --cuda-graph-bs-decode 1 2 4 \
      $BK $SERVER_EXTRA --host 127.0.0.1 --port 30000 \
      --json-model-override-args "$OV" > "$LOG" 2>&1 &
    SP=$!; ok=0
    for i in $(seq 1 600); do
      curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1 && { echo "  $NAME healthy after $((i*10))s"; ok=1; break; }
      kill -0 $SP 2>/dev/null || { echo "  $NAME DIED"; tail -30 "$LOG"; break; }
      sleep 10
    done
    if [ "$ok" = 1 ]; then
      for EV in $TODO; do
        echo "  --- $NAME :: $EV ---"; T0=$(date +%s)
        case "$EV" in
          lb1)  python3 /research/eval/eval_longbench_served.py --base-url http://127.0.0.1:30000 --n-per-task 100 --hf-home /hh/.hf_cache \
                  --tasks "$TASKS" $EVAL_EXTRA --output "$RESDIR/${NAME}_lb1.json" 2>&1 | tail -4 ;;
          niah) for CHARS in 32000 130000 260000 520000; do for MODE in single aggregate; do
                  python3 /research/eval/eval_needle.py --base-url http://127.0.0.1:30000 --target-chars $CHARS --num-needles 10 --mode $MODE \
                    --output "$RESDIR/${NAME}_niah_c${CHARS}_${MODE}.json" 2>&1 | tail -1
                done; done ;;
          ppl)  for IL in 4096 16384 32768 65536; do for SEED in 1234 2345 3456 4567 5678; do
                  python3 /research/eval/eval_decode_ppl.py --base-url http://127.0.0.1:30000 --input-len $IL --output-len 512 --seed $SEED \
                    --tag "${MODEL_TAG}_${NAME}_s${SEED}" --output "$RESDIR/${NAME}_ppl_in${IL}_s${SEED}.json" 2>&1 | tail -1
                done; done ;;
          ruler) python3 /research/eval/eval_ruler_served.py --base-url http://127.0.0.1:30000 --model-path /model \
                  --lengths "$RULER_LENGTHS" --depths "$RULER_DEPTHS" --n-per-cell "$RULER_N" --output "$RESDIR/${NAME}_${RULER_TAG}.json" 2>&1 | tail -2 ;;
          lbv2) python3 /research/eval/eval_longbench_v2_served.py --base-url http://127.0.0.1:30000 --model-path /model \
                  --max-samples $LBV2_MAX --max-ctx-tokens 120000 --hf-home /hh/.hf_cache "${LBV2_ARGS[@]}" --output "$RESDIR/${NAME}_${LBV2_TAG}.json" 2>&1 | tail -2 ;;
          *) echo "unknown eval $EV" ;;
        esac
        echo "  --- $NAME :: $EV done in $(( $(date +%s) - T0 ))s ---"
        curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1 || { echo "  $NAME DIED during $EV"; tail -20 "$LOG"; break; }
      done
    fi
    grep -aE "first .lag_refresh_deferred.|first .hide_fused.|HIDE VOID|kernel unavailable|using normal path|Traceback" "$LOG" | head -4
    # Engagement gate (serving-anti-false-claim): MLI/lag arms MUST show the deferred marker, else the number is not attributable to MLI.
    case "$NAME" in *lag_fused*) grep -q "first .lag_refresh_deferred." "$LOG" || echo "!!!!! ENGAGEMENT MISSING: $NAME lag_refresh_deferred marker NOT fired -- RESULT SUSPECT !!!!!";; esac
    kill $SP 2>/dev/null || true
    for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 2; done
    echo "  $NAME done"
  done
  echo "QUALITY DONE $MODEL_TAG -> $RESDIR"
'
RC=$?
echo "=== job done (rc=$RC) ==="
exit $RC
