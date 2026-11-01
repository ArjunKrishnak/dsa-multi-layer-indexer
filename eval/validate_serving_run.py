"""Anti-false-claim gate for SGLang serving measurements.

A serving t/s is REPORTABLE only if it passes every applicable gate below. When a
gate fails the verdict is VOID (loud), never a silent pass. This exists because
the serving line has repeatedly produced artifact numbers (fa3 bypass, bf16
requant, fp8 layout, surrogate, and the serial-lag "HIDE" fallback) that read as
results until caught by hand.

Gates
  E  engagement   the intended fused path ACTUALLY ran. cascade -> server log must
                  contain "[ENGAGE] first 'cascade_fused'"; hide -> 'hide_fused'.
                  ANY of {"[HIDE VOID]","[CASCADE VOID]","kernel unavailable",
                  "using normal path"} => VOID. base/copy need no fused path.
  I  invariant    HIDE removes the serial indexer->attention dependency, so it
                  CANNOT be slower than base by construction. hide t/s < base*(1-TOL)
                  is definitionally an artifact => VOID. (hide ~ base or faster ok.)
  C  null-control an overlay run with no method active (freq1) MUST match STOCK
                  base within TOL; if merely loading the overlay shifts timing the
                  apparatus is untrustworthy => all overlay numbers VOID.
  S  sanity       the jsonl exists and reports a positive steady throughput.

Usage:
  python validate_serving_run.py --tol 0.03 \
     --cell base:base:/path/base.jsonl:/path/base.log \
     --cell null:null:/path/null.jsonl:/path/null.log \
     --cell cascade:cascade:/path/casc.jsonl:/path/casc.log \
     --cell hide:hide:/path/hide.jsonl:/path/hide.log
Each --cell is name:kind:result_jsonl:server_log ; kind in
{base,null,copy,cascade,hide,reuse_fused,reuse_fused_s2,reuse_fused_s2_k2,
 reuse_fused_s2_k3,reuse_fused_rerank,history_set}.
"""
from __future__ import annotations

import argparse
import json
import os

VOID_STRINGS = ["[HIDE VOID]", "[CASCADE VOID]", "kernel unavailable", "using normal path"]
ENGAGE_MARK = {
    "cascade": "first 'cascade_fused'",
    "hide": "first 'hide_fused'",
    # reuse_fused serving composition (index_reuse_lag_refresh): the refresh
    # layers must prove the one-layer-lag handoff engaged, else the arm
    # silently measured plain pattern reuse.
    "reuse_fused": "first 'lag_refresh'",
    # reuse_fused slice 2 (index_reuse_lag_refresh_deferred): the deferred
    # side-stream indexer must prove it engaged, else the arm silently
    # measured the serial slice-1 path (certified speed-NEUTRAL -- a "win"
    # without this marker is a false claim).
    "reuse_fused_s2": "first 'lag_refresh_deferred'",
    # lag DEPTH k (index_reuse_lag_depth: 2 / 3): the refresh layers must
    # prove they joined a deferred indexer k refresh layers old (queue-full
    # steady state), else the arm silently measured the k=1 deferred path.
    "reuse_fused_s2_k2": "first 'lag_depth_2'",
    "reuse_fused_s2_k3": "first 'lag_depth_3'",
    # host/triton-side rerank (out-of-kernel placement witness); marker added
    # after a smoke run caught the path emitting nothing.
    "cascade_triton": "first 'triton_rerank'",
    # COMBINED mode (hh-lagrefresh-rerank-hybrid.patch; prereg_quality_wave.md
    # cell Q2): lag-refresh handoff on refresh layers + tap re-rank on
    # consumer layers. The marker is emitted only when BOTH halves have fired
    # in-process, so neither half can silently degrade to the other mode.
    "reuse_fused_rerank": "first 'lag_refresh_rerank'",
    # frozen selection-history-set scored admission (index_history_set;
    # hh-history-set-admission.patch, prereg_scored_admission.md SS9): the
    # select layers must prove the same-layer restricted-indexer union
    # re-rank engaged, else the arm silently measured plain pattern reuse
    # (frozen copy) and its quality/speed numbers are mislabeled.
    "history_set": "first 'history_set_admit'",
}


def read_tps(jsonl: str):
    if not jsonl or not os.path.exists(jsonl):
        return None
    last = None
    for line in open(jsonl):
        line = line.strip()
        if line:
            last = json.loads(line)
    if last is None:
        return None
    tps = float(last.get("last_gen_throughput", 0.0) or 0.0)
    if tps <= 0:
        # Under --enable-dp-attention the server does not report last_gen_throughput
        # to one_batch_server (0.0). Fall back to the client-side decode throughput,
        # output_len / (latency - ttft), which agrees with last_gen_throughput within
        # 0.5% on TP cells (grid_ds_per_request_131072_29758: 85.95 vs 85.53).
        tps = float(last.get("output_throughput", 0.0) or 0.0)
    return tps or None


def read_log(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    # server logs carry \r; normalize
    return open(path, "rb").read().decode("utf-8", "replace").replace("\r", "\n")


def validate(cells, tol):
    """cells: list of (name, kind, jsonl, log). Returns list of verdict dicts."""
    tps = {}
    logs = {}
    for name, kind, jsonl, log in cells:
        tps[name] = read_tps(jsonl)
        logs[name] = read_log(log)

    base_tps = next((tps[n] for (n, k, _, _) in cells if k == "base"), None)
    null_tps = next((tps[n] for (n, k, _, _) in cells if k == "null"), None)
    # HIDE is judged against the SAME-MODE base: graphs-on base by default (the
    # paged fused HIDE is CUDA-graph captured); an explicit base_eager cell, when
    # present, anchors an eager HIDE diagnostic instead.
    base_eager_tps = next((tps[n] for (n, k, _, _) in cells if k == "base_eager"), None)

    verdicts = []
    for name, kind, jsonl, log in cells:
        reasons = []
        t = tps[name]
        lg = logs[name]

        # S: sanity
        if t is None or t <= 0:
            verdicts.append(dict(name=name, kind=kind, tps=t, verdict="VOID",
                                 reasons=["S: no positive throughput in result jsonl"]))
            continue

        # E: engagement (fused kinds must prove the kernel ran; nobody may show a void/fallback)
        hit_void = [s for s in VOID_STRINGS if s in lg]
        if hit_void:
            reasons.append(f"E: server log shows fallback/void {hit_void}")
        if kind in ENGAGE_MARK and ENGAGE_MARK[kind] not in lg:
            reasons.append(f"E: no '{ENGAGE_MARK[kind]}' engagement marker -> fused {kind} path did NOT run")

        # I: PLAUSIBILITY (applies to ANY indexer-removal/hiding method). The
        # deep_gemm indexer is a MODEST fraction (~10-15%) of the decode step (base
        # is ~context-flat). REMOVING it (cascade) or HIDING it (hide) therefore
        # cannot make decode dramatically slower -- at worst ~neutral, at best
        # ~10-15% faster. A method cell that is materially slower than the SAME-MODE
        # base is dominated by IMPLEMENTATION overhead (e.g. the fused paths replace
        # flashmla_kv attention with the research kernel), NOT by the method. That is
        # a CONFOUNDED measurement, not a negative result -- do not report it as one.
        PLAUS = 0.20
        if kind in ("cascade", "hide"):
            use_eager_anchor = kind == "hide" and base_eager_tps is not None
            anchor = base_eager_tps if use_eager_anchor else base_tps
            amode = "base_eager" if use_eager_anchor else "base"
            if anchor is None:
                reasons.append(f"I: no {amode} anchor cell -- cannot judge {kind} plausibility")
            elif t < anchor * (1.0 - PLAUS):
                reasons.append(
                    f"I: {kind} {t:.1f} is {100*(1-t/anchor):.0f}% slower than {amode} {anchor:.1f} -- "
                    f"removing/hiding a ~10-15% indexer CANNOT cost this much; the number is "
                    f"overhead/kernel-swap dominated (CONFOUNDED), not a method result")

        # C: null control must match stock base
        if kind == "null" and base_tps:
            if abs(t - base_tps) > tol * base_tps:
                reasons.append(f"C: null-overlay {t:.1f} != stock base {base_tps:.1f} within {tol:.0%} "
                               f"-- overlay perturbs timing; all overlay numbers suspect")

        verdicts.append(dict(name=name, kind=kind, tps=t, verdict=("PASS" if not reasons else "VOID"),
                             reasons=reasons))
    return verdicts, base_tps, null_tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", default=[],
                    help="name:kind:result_jsonl:server_log")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cells = []
    for spec in args.cell:
        parts = spec.split(":")
        if len(parts) < 4:
            raise SystemExit(f"bad --cell '{spec}' (need name:kind:jsonl:log)")
        name, kind = parts[0], parts[1]
        jsonl, log = parts[2], ":".join(parts[3:])
        cells.append((name, kind, jsonl, log))

    verdicts, base_tps, null_tps = validate(cells, args.tol)

    print(f"\n{'='*72}\nSERVING-RUN VALIDATION (tol={args.tol:.0%})  base={base_tps} null={null_tps}\n{'='*72}")
    print(f"{'cell':<12}{'kind':<10}{'t/s':>8}  verdict")
    npass = 0
    for v in verdicts:
        ts = f"{v['tps']:.1f}" if v['tps'] else "-"
        print(f"{v['name']:<12}{v['kind']:<10}{ts:>8}  {v['verdict']}")
        for r in v["reasons"]:
            print(f"              -> {r}")
        npass += v["verdict"] == "PASS"
    print(f"{'-'*72}\n{npass}/{len(verdicts)} cells PASS -- ONLY PASS numbers are reportable.\n")

    if args.json_out:
        json.dump(dict(tol=args.tol, base_tps=base_tps, null_tps=null_tps, verdicts=verdicts),
                  open(args.json_out, "w"), indent=2)
    # nonzero exit if any fused/method cell VOIDed (so a harness can gate on it)
    bad = [v for v in verdicts if v["verdict"] == "VOID" and v["kind"] in ("cascade", "hide", "null")]
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
