#!/bin/bash
# Decode-throughput cell: one (model, regime, context) with every arm booted in the
# same job on the same node (paper Sec. 4 "Protocol and fairness").
#
#   arms: base | lag_fused | lag_def | ic_f{F} | ic_lag_fused_f{F} | ic_lag_def_f{F} (F in FREQS) | base_again
#     base / base_again   unmodified DSA (drift sentinel: booted first and last)
#     ic_f{F}             IndexCache at retention 1/F
#     lag_def, ic_lag_def MLI, deferred-stream realization (no kernel change)
#     lag_fused, ic_lag_fused  MLI, fused decode kernel (DeepSeek-V3.2 realization)
#   SEARCHED=1 adds IndexCache's searched GLM-5 patterns (ic_s2/ic_s4 + MLI).
#
# Regimes: per_request = one request, decode tok/s + TTFT (Table 1);
#          saturation  = KV-capacity-sized request set pre-warmed into the radix cache.
# Each cell writes $WORK/results/grid_<model>_<regime>_<ctx>_<job>/ with per-arm .jsonl,
# server logs, meta.txt and validation.json; the job fails if a gate fails.
#
# Required env: WORK, SGLANG_SRC, MODEL_PATH, MODEL_TAG (ds | glm5). Optional: SQSH,
#   STAGE_DIR, REGIME (per_request), CTX (122880), FREQS ("2 4 8 16 32"), OUTLEN (128),
#   INCLUDE_FUSED (1; set 0 on GLM-5), SAT_NUM_PROMPTS (64), PAR_FLAGS
#   ("--enable-dp-attention --dp-size 8" for the DP-attention replication).
# Usage (from the repo root):
#   MODEL_TAG=ds   CTX=131072 sbatch -p <partition> scripts/speed.sbatch
#   MODEL_TAG=glm5 CTX=199680 INCLUDE_FUSED=0 sbatch -p <partition> scripts/speed.sbatch
#SBATCH --job-name=mli-speed
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/speed_%j.out
set -uo pipefail
MODEL_TAG="${MODEL_TAG:?MODEL_TAG must be set (ds | glm5)}"
REGIME="${REGIME:-per_request}"
CTX="${CTX:-122880}"
FREQS="${FREQS:-2 4 8 16 32}"
INCLUDE_FUSED="${INCLUDE_FUSED:-1}"
SEARCHED="${SEARCHED:-0}"   # 1 = add IC searched-pattern arms (GLM-5 only); 2 = ONLY base + searched arms   # 0 = deferred + IC arms only (fused kernel loses in serving; keep for per-request tables)
OUTLEN="${OUTLEN:-128}"
SAT_NUM_PROMPTS="${SAT_NUM_PROMPTS:-64}"
PAR_FLAGS="${PAR_FLAGS:-}"
# ---- configuration (see README "Reproducing the paper") ----------------------
WORK="${WORK:?set WORK to a writable workspace (caches, wheel, results)}"
SQSH="${SQSH:-$WORK/sglang-v0.5.16-cu130.sqsh}"
SGLANG_SRC="${SGLANG_SRC:?set SGLANG_SRC to an sglang v0.5.16 clone with sglang_overlay/mli-sglang-v0.5.16.patch applied}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO/sglang_overlay/mli-sglang-v0.5.16.patch" ] || { echo "REPO=$REPO is not this repository (submit from the repo root or set REPO)" >&2; exit 2; }
HH="$WORK"; HH_SRC="$SGLANG_SRC"; NLP="$REPO"
case "$MODEL_TAG" in
  ds)   MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V3.2 FP8 checkpoint}"; SERVER_EXTRA=""; CTX_LIMIT=163840 ;;
  glm5) MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the GLM-5 FP8 checkpoint}"; SERVER_EXTRA="--reasoning-parser glm45"; CTX_LIMIT=202752 ;;
  *) echo "MODEL_TAG must be ds or glm5" >&2; exit 1 ;;
esac
# Optional: stage the weights to node-local disk first (STAGE_DIR); otherwise load MODEL_PATH directly.
if [ -n "${STAGE_DIR:-}" ]; then
  SRC="$MODEL_PATH"; DST="$STAGE_DIR/$(basename "$MODEL_PATH")"
  source "$REPO/scripts/setup/stage_model.sh"
else
  MODEL="$MODEL_PATH"
fi
JOB=$SLURM_JOB_ID
CELL="${MODEL_TAG}_${REGIME}_${CTX}"
echo "### GRID CELL $CELL job=$JOB node=$(hostname) MODEL=$MODEL freqs=[$FREQS] par=[$PAR_FLAGS] nsplit=${HH_HIDE_NSPLIT:-default} ###"
MOUNTS="${MODEL}:/model:ro,${HH}:/hh:rw,${HH_SRC}/python/sglang:/usr/local/lib/python3.12/dist-packages/sglang:ro,${NLP}:/research:ro,${HH_SRC}:/hh/src"
srun --container-image="$SQSH" --container-mounts="$MOUNTS" --container-workdir=/hh bash -lc '
  set -uo pipefail
  export HOME=/hh/.container_home HF_HOME=/hh/.hf_cache
  export TORCHINDUCTOR_CACHE_DIR=/hh/.inductor_cache TRITON_CACHE_DIR=/hh/.triton_cache
  export TORCH_EXTENSIONS_DIR=/hh/.torch_ext_hide
  export PYTHONPATH=/research/src:/research/kernels:/research/eval:${PYTHONPATH:-}
  [ -n "'"${HH_HIDE_NSPLIT:-}"'" ] && export HH_HIDE_NSPLIT='"${HH_HIDE_NSPLIT:-16}"'
  mkdir -p "$HOME" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
  SGLK=/tmp/sglk_$$; rm -rf "$SGLK"
  # Fail LOUD and never delete the container sgl_kernel unless a replacement built OK (repro fix).
  WHEELOUT="${WHEELOUT:-/hh/wheelout}"
  if ls "$WHEELOUT"/*.whl >/dev/null 2>&1; then
    pip install --target="$SGLK" --no-deps --force-reinstall "$WHEELOUT"/*.whl 2>&1 | tail -1
  else
    echo "[setup] no wheel in $WHEELOUT; trying stock sgl-kernel 0.4.5 from PyPI (see README.md)"
    pip install --target="$SGLK" --no-deps sgl-kernel==0.4.5 2>&1 | tail -2 || true
  fi
  if [ ! -d "$SGLK/sgl_kernel" ]; then
    echo "FATAL setup: sgl_kernel not installed. Provide a wheel in $WHEELOUT or a reachable PyPI (sgl-kernel==0.4.5). See README.md. Container sgl_kernel left intact. Aborting."
    exit 3
  fi
  [ -f /hh/src/sgl-kernel/python/sgl_kernel/flash_mla.py ] && cp /hh/src/sgl-kernel/python/sgl_kernel/flash_mla.py "$SGLK/sgl_kernel/flash_mla.py"
  SP_SGLK=/usr/local/lib/python3.12/dist-packages/sgl_kernel
  rm -rf "$SP_SGLK"; cp -r "$SGLK/sgl_kernel" "$SP_SGLK"
  python3 -c "import sgl_kernel" 2>/dev/null || { echo "FATAL setup: sgl_kernel import failed after install. Aborting."; exit 3; }
  export PYTHONPATH="$SGLK:${PYTHONPATH:-}"
  JOB='"$JOB"'; CELL='"$CELL"'; MODEL_TAG='"$MODEL_TAG"'; REGIME='"$REGIME"'; CTX='"$CTX"'; OUTLEN='"$OUTLEN"'
  INCLUDE_FUSED='"$INCLUDE_FUSED"'; SEARCHED='"$SEARCHED"'; SATNP='"$SAT_NUM_PROMPTS"'; OUTLEN_SAT='"${OUTLEN_SAT:-1024}"'; FREQS="'"$FREQS"'"; SERVER_EXTRA="'"$SERVER_EXTRA"'"; PAR_FLAGS="'"$PAR_FLAGS"'"; CTX_LIMIT='"$CTX_LIMIT"'
  BK="--dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv --kv-cache-dtype fp8_e4m3"
  # Decode CUDA graphs must cover the largest batch a regime decodes: per-request = 1;
  # saturation = up to SATNP requests together (cell 29922 ran batch 6 eager under a cap of 4).
  if [ "$REGIME" = saturation ]; then
    GRAPH_BS="--cuda-graph-max-bs-decode $SATNP --cuda-graph-bs-decode 1 2 4 8 16 32 64"
  else
    GRAPH_BS="--cuda-graph-max-bs-decode 4 --cuda-graph-bs-decode 1 2 4"
  fi
  RESDIR=/hh/results/grid_${CELL}_${JOB}; mkdir -p "$RESDIR"
  { echo "job=$JOB cell=$CELL node=$(hostname) date=$(date -u +%FT%TZ) regime=$REGIME ctx=$CTX outlen=$OUTLEN satnp=$SATNP par=[$PAR_FLAGS] nsplit=${HH_HIDE_NSPLIT:-default}"
    echo "overlay_src=$(cat /hh/src/FROZEN_STATE.txt 2>/dev/null || echo live) overlay=$(git -C /hh/src rev-parse --short HEAD 2>/dev/null) dirty_sha=$(git -C /hh/src diff 2>/dev/null | sha256sum | cut -c1-12) wheel=$(sha256sum /hh/wheelout/*.whl | cut -c1-12) kernel_src=$(sha256sum /research/kernels/cuda_fused_sparse_splitk.py | cut -c1-12)"
    echo "server_flags: --tp-size 8 $PAR_FLAGS --mem-fraction-static 0.83 --context-length $CTX_LIMIT $GRAPH_BS $BK $SERVER_EXTRA"
  } | tee "$RESDIR/meta.txt"
  cell () {  # NAME OVERRIDE
    local NAME="$1" OV="$2" ENVV="${3:-}" ok=0
    local LOG=$RESDIR/server_${NAME}.log
    echo "###### $CELL arm=$NAME ov=$OV ######"
    env $ENVV python3 -m sglang.launch_server --model-path /model --trust-remote-code --tp-size 8 $PAR_FLAGS \
      --mem-fraction-static 0.83 --context-length $CTX_LIMIT \
      $GRAPH_BS \
      $BK $SERVER_EXTRA --host 127.0.0.1 --port 30000 \
      --json-model-override-args "$OV" > "$LOG" 2>&1 &
    local SP=$!
    for i in $(seq 1 600); do
      curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1 && { echo "  $NAME healthy after $((i*10))s"; ok=1; break; }
      kill -0 $SP 2>/dev/null || { echo "  $NAME DIED"; tail -30 "$LOG"; break; }
      sleep 10
    done
    if [ "$ok" = 1 ]; then
      if [ "$REGIME" = per_request ]; then
        python3 -m sglang.benchmark.one_batch_server --model None --base-url http://127.0.0.1:30000 \
          --batch-size 1 --input-len $CTX --output-len $OUTLEN \
          --result-filename $RESDIR/${NAME}.jsonl --show-report 2>&1 | tail -1
      else
        # FULL-KV-CACHE regime (IndexCache full-KV-cache throughput): size the
        # request set to the server KV capacity, warm every prompt into the radix
        # cache (output_len=1), then re-send the SAME prompts (same seed) with a long
        # output so every request decodes from a cache hit -> a pure saturated decode
        # phase; the metric is that pass output tokens/s. mean_ttft of pass 2 must
        # be small (cache hit), which the summary echoes.
        MAXTOK=$(grep -a -m1 -oE "max_total_num_tokens=[0-9]+" "$LOG" | grep -oE "[0-9]+$")
        NP=$(( ${MAXTOK:-0} / (CTX + OUTLEN_SAT) )); [ "$NP" -lt 1 ] && NP=1; [ "$NP" -gt "$SATNP" ] && NP=$SATNP
        echo "  saturation: max_total_num_tokens=$MAXTOK -> num_prompts=$NP (ctx=$CTX, out=$OUTLEN_SAT)"
        python3 -m sglang.bench_serving --backend sglang --base-url http://127.0.0.1:30000 \
          --dataset-name random --random-input-len $CTX --random-output-len 1 \
          --random-range-ratio 1.0 --num-prompts $NP --max-concurrency $NP --request-rate inf --seed 1 \
          --output-file $RESDIR/${NAME}_warm.jsonl 2>&1 | tail -1
        # Two decode passes; the FIRST is a warm-up of the decode path (a fresh server
        # first ~20 s pass is cold: GLM-5 200K cell 29926 measured base 103 vs 111 tok/s on
        # the closing repeat) and only the SECOND is recorded as the result of the arm.
        python3 -m sglang.bench_serving --backend sglang --base-url http://127.0.0.1:30000 \
          --dataset-name random --random-input-len $CTX --random-output-len $OUTLEN_SAT \
          --random-range-ratio 1.0 --num-prompts $NP --max-concurrency $NP --request-rate inf --seed 1 \
          --output-file $RESDIR/${NAME}_pass1.jsonl 2>&1 | tail -1
        python3 -m sglang.bench_serving --backend sglang --base-url http://127.0.0.1:30000 \
          --dataset-name random --random-input-len $CTX --random-output-len $OUTLEN_SAT \
          --random-range-ratio 1.0 --num-prompts $NP --max-concurrency $NP --request-rate inf --seed 1 \
          --output-file $RESDIR/${NAME}.jsonl 2>&1 | tail -2
        python3 - "$RESDIR/${NAME}.jsonl" "$NP" <<PY
import json, sys
p, np_ = sys.argv[1], int(sys.argv[2])
L = [json.loads(l) for l in open(p) if l.strip()]
r = dict(L[-1]); r["last_gen_throughput"] = r.get("output_throughput"); r["sat_num_prompts"] = np_
r["sat_note"] = "decode-phase aggregate tok/s: prompts pre-warmed into the radix cache, so TTFT is a cache hit"
open(p, "a").write(json.dumps(r) + "\n")
print("  saturation decode tok/s=%.2f concurrency=%.2f mean_ttft_ms=%.0f mean_tpot_ms=%.2f" % (
    r["output_throughput"], r.get("concurrency", 0), r.get("mean_ttft_ms", 0), r.get("mean_tpot_ms", 0)))
PY
      fi
    fi
    grep -aE "first .hide_fused.|first .lag_refresh_deferred.|HIDE VOID|kernel unavailable|using normal path|Traceback" "$LOG" | head -4
    kill $SP 2>/dev/null || true
    for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 2; done
    echo "  $NAME done"
  }
  PAT_S2="FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"; PAT_S4="FFSFSSSFSSFSFSSSSSSSFSSSSSSFSSSFSFSSSSFFFFFSSSFFSSSFSSSSSSSSFSSSFSSSSSSFSFSSSS"
  cell base      "{\"index_topk_freq\": 1}"
  if [ "$SEARCHED" != 0 ]; then
    DEFJ="\"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true"
    cell ic_s2          "{\"index_topk_pattern\": \"$PAT_S2\"}"
    cell ic_lag_def_s2  "{\"index_topk_pattern\": \"$PAT_S2\", $DEFJ}" "HH_LAG_NO_ALT=1"
    cell ic_s4          "{\"index_topk_pattern\": \"$PAT_S4\"}"
    cell ic_lag_def_s4  "{\"index_topk_pattern\": \"$PAT_S4\", $DEFJ}" "HH_LAG_NO_ALT=1"
  fi
  if [ "$SEARCHED" = 2 ]; then FREQS=""; INCLUDE_FUSED=0; fi
  [ "$INCLUDE_FUSED" = 1 ] && cell lag_fused "{\"index_topk_freq\": 1, \"index_lag_fused\": true}" "HH_HIDE_DEFER_SCAN=1 HH_HIDE_DEFER_NOJOIN=1 HH_HIDE_EARLY_PREP=1 HH_HIDE_NSPLIT=64"
  [ "$SEARCHED" != 2 ] && cell lag_def   "{\"index_topk_freq\": 1, \"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true}" "HH_LAG_NO_ALT=1"
  for F in $FREQS; do
    cell ic_f${F}           "{\"index_topk_freq\": $F}"
    [ "$INCLUDE_FUSED" = 1 ] && cell ic_lag_fused_f${F} "{\"index_topk_freq\": $F, \"index_lag_fused\": true}" "HH_HIDE_DEFER_SCAN=1 HH_HIDE_DEFER_NOJOIN=1 HH_HIDE_EARLY_PREP=1 HH_HIDE_NSPLIT=64"
    cell ic_lag_def_f${F}   "{\"index_topk_freq\": $F, \"index_reuse_lag_refresh\": true, \"index_reuse_lag_refresh_deferred\": true}" "HH_LAG_NO_ALT=1"
  done
  cell base_again "{\"index_topk_freq\": 1}"
  echo "=== GRID CELL SUMMARY $CELL ($REGIME; output tok/s vs base) ==="
  python3 - <<PY
import json, os, glob
resdir = "$RESDIR"; freqs = "$FREQS".split()
names = ["base", "lag_fused", "lag_def", "ic_s2", "ic_lag_def_s2", "ic_s4", "ic_lag_def_s4"] + [n for F in freqs for n in ("ic_f%s" % F, "ic_lag_fused_f%s" % F, "ic_lag_def_f%s" % F)] + ["base_again"]
def read(n):
    f = os.path.join(resdir, n + ".jsonl")
    if not os.path.exists(f): return None
    L = [json.loads(l) for l in open(f) if l.strip()]
    if not L: return None
    r = L[-1]
    return dict(tps=r.get("last_gen_throughput") or r.get("output_throughput"), ttft=r.get("ttft") or r.get("median_ttft_ms"))
base = read("base")
for n in names:
    r = read(n)
    if r is None or r["tps"] is None: print("  %-20s -" % n); continue
    d = "" if not base or not base["tps"] else " (%+.1f%%)" % (100 * (r["tps"] / base["tps"] - 1))
    print("  %-20s %9.2f tok/s%s   ttft=%s" % (n, r["tps"], d, r["ttft"]))
PY
  echo "--- validation ---"
  ARGS="--cell base:base:$RESDIR/base.jsonl:$RESDIR/server_base.log"
  [ -f $RESDIR/lag_def.jsonl ] && ARGS="$ARGS --cell lag_def:reuse_fused_s2:$RESDIR/lag_def.jsonl:$RESDIR/server_lag_def.log"
  [ "$INCLUDE_FUSED" = 1 ] && ARGS="$ARGS --cell lag_fused:hide:$RESDIR/lag_fused.jsonl:$RESDIR/server_lag_fused.log"
  for F in $FREQS; do ARGS="$ARGS --cell ic_f${F}:copy:$RESDIR/ic_f${F}.jsonl:$RESDIR/server_ic_f${F}.log --cell ic_lag_def_f${F}:reuse_fused_s2:$RESDIR/ic_lag_def_f${F}.jsonl:$RESDIR/server_ic_lag_def_f${F}.log"; [ "$INCLUDE_FUSED" = 1 ] && ARGS="$ARGS --cell ic_lag_fused_f${F}:hide:$RESDIR/ic_lag_fused_f${F}.jsonl:$RESDIR/server_ic_lag_fused_f${F}.log"; done
  for n in ic_s2 ic_s4; do [ -f $RESDIR/$n.jsonl ] && ARGS="$ARGS --cell $n:copy:$RESDIR/$n.jsonl:$RESDIR/server_$n.log"; done
  for n in ic_lag_def_s2 ic_lag_def_s4; do [ -f $RESDIR/$n.jsonl ] && ARGS="$ARGS --cell $n:reuse_fused_s2:$RESDIR/$n.jsonl:$RESDIR/server_$n.log"; done
  ARGS="$ARGS --cell base_again:base:$RESDIR/base_again.jsonl:$RESDIR/server_base_again.log"
  python3 /research/eval/validate_serving_run.py --tol 0.03 $ARGS --json-out $RESDIR/validation.json 2>&1 | tail -20
  VRC=${PIPESTATUS[0]}
  echo "GRID CELL DONE $CELL (validator rc=$VRC)"
  exit $VRC
'
RC=$?
echo "=== job done (rc=$RC) ==="
exit $RC
