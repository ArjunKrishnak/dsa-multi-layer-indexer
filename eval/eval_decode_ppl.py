"""Continuous DECODE-path quality probe against a served SGLang instance.

Unlike the needle exact-match probe (0/1, saturates at 1.00), this measures a
*continuous* signal: the per-token logprob of a long greedy generation, read
from the autoregressive DECODE path.

Why the decode path: the selection-reuse quality gap comes from a frozen
top-2048 index set going stale across decode steps. The only tokens that carry
decode-step logprobs in SGLang are GENERATED tokens -- `meta_info[
"output_token_logprobs"]` holds one `(logprob, token_id, text)` per generated
token, each computed by a decode forward that reuses the frozen selection. A
single prefill over prompt+continuation would recompute the indexer fresh for
every position and show no difference; this harness deliberately does the
opposite by forcing a full-length greedy generation.

Metric (see QUALITY_HARNESS_NOTES.md for the feasibility analysis):
  - mean_logprob and ppl = exp(-mean_logprob) over the whole generation
  - early-window vs late-window mean logprob, and drift = late - early
  - least-squares slope of logprob vs decode position
Staleness signature = negative drift/slope within a run; cross-variant headline
number = Delta_ppl (variant ppl minus base ppl). True teacher-forced continuation
Delta_ppl through decode is not supported by the SGLang HTTP API (no forced-output-
token param; the score/select paths are prefill-only), so this self-logprob
metric is the best available discriminating decode-path signal.

Fail-loud, structured logging, greedy/deterministic. Does NOT launch a server.
"""

import argparse
import json
import logging
import time
import urllib.error
import urllib.request

# eval_needle lives in the same directory; when this script is run its directory
# is on the module search path, so this is a normal import (no sys.path hacking).
from eval_needle import build_haystack
import random

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("decode_ppl")


def _post_generate(base_url: str, payload: dict, timeout_s: int) -> dict:
    """POST /generate and return the parsed JSON, raising loudly on any failure."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"/generate returned HTTP {exc.code}: {detail}"
        ) from exc
    return json.loads(body)


def build_prompt(rng: random.Random, target_chars: int) -> str:
    """Long-context prompt: a haystack document plus a continue instruction.

    Reuses eval_needle's deterministic haystack builder. Needles are inserted so
    the document has concrete, position-dependent facts distributed through the
    context -- the kind of distant content a stale selection would drop.
    """
    needle_keys = ["cherry", "falcon", "granite", "lantern", "meadow",
                   "quartz", "saffron", "trellis", "velvet", "walnut"]
    needles = {k: str(rng.randint(100000, 999999)) for k in needle_keys}
    haystack = build_haystack(rng, target_chars, needles)
    prompt = (
        haystack
        + "\n\nContinue the report above in the same style, elaborating on the"
        + " topics and details discussed, in a single flowing narrative:\n"
    )
    return prompt


def extract_decode_logprobs(resp: dict) -> list:
    """Pull the per-decode-step logprobs out of the /generate response.

    Each entry of meta_info['output_token_logprobs'] is
    [logprob, token_id, token_text_or_null]; the logprob is index 0.
    Fails loudly if the field is missing (e.g. return_logprob not honored).
    """
    meta = resp["meta_info"]
    if "output_token_logprobs" not in meta:
        raise RuntimeError(
            "response meta_info has no 'output_token_logprobs'; the server did "
            "not return decode logprobs (check return_logprob support). "
            f"meta_info keys: {sorted(meta.keys())}"
        )
    entries = meta["output_token_logprobs"]
    if not entries:
        raise RuntimeError(
            "'output_token_logprobs' is empty; no decode steps were scored "
            "(did the generation produce zero tokens?)."
        )
    logprobs = []
    token_ids = []
    for entry in entries:
        # entry is (logprob, token_id, token_text). Be strict about shape.
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise RuntimeError(f"unexpected output_token_logprobs entry: {entry!r}")
        logprobs.append(float(entry[0]))
        token_ids.append(int(entry[1]))
    return logprobs, token_ids


def compute_degeneration(token_ids: list) -> dict:
    """Repetition/degeneration guard for the self-logprob metric.

    Self-logprob measures the model's confidence in its OWN greedy output, not
    correctness: a stale variant that collapses into a confident repetition loop
    scores LOW ppl (looks 'good'). This flags that failure mode so a low ppl born
    of collapse is not misread as a quality win. Trust `drift` + this guard over
    absolute ppl.
    """
    n = len(token_ids)
    if n == 0:
        raise RuntimeError("no decode token_ids to score for degeneration.")
    distinct_ratio = len(set(token_ids)) / n
    k = 4  # fraction of positions that begin an immediately repeated 4-gram
    reps = 0
    if n >= 2 * k:
        for i in range(n - 2 * k + 1):
            if token_ids[i:i + k] == token_ids[i + k:i + 2 * k]:
                reps += 1
        repeat_4gram_frac = reps / (n - 2 * k + 1)
    else:
        repeat_4gram_frac = 0.0
    degenerate = distinct_ratio < 0.2 or repeat_4gram_frac > 0.3
    return {
        "distinct_token_ratio": distinct_ratio,
        "repeat_4gram_frac": repeat_4gram_frac,
        "degenerate": degenerate,
    }


def compute_metrics(logprobs: list, window_frac: float) -> dict:
    """Continuous staleness metrics from per-decode-step logprobs."""
    n = len(logprobs)
    mean_lp = sum(logprobs) / n
    ppl = _exp(-mean_lp)

    win = max(1, int(n * window_frac))
    early = logprobs[:win]
    late = logprobs[-win:]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    drift = late_mean - early_mean  # negative == degrading (staleness signature)

    # Least-squares slope of logprob vs decode position (index 0..n-1).
    xs = list(range(n))
    mean_x = sum(xs) / n
    cov = sum((x - mean_x) * (lp - mean_lp) for x, lp in zip(xs, logprobs))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = cov / var_x if var_x > 0 else 0.0

    return {
        "num_decode_steps": n,
        "mean_logprob": mean_lp,
        "ppl": ppl,
        "window_frac": window_frac,
        "window_tokens": win,
        "early_window_mean_logprob": early_mean,
        "late_window_mean_logprob": late_mean,
        "drift_logprob": drift,
        "late_over_early_ppl_ratio": _exp(-(late_mean - early_mean)),
        "slope_logprob_per_token": slope,
    }


def _exp(x: float) -> float:
    import math
    return math.exp(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--input-len", type=int, default=32000,
        help="target prompt length in TOKENS (approximate; actual is measured "
        "and reported from meta_info.prompt_tokens).",
    )
    parser.add_argument(
        "--output-len", type=int, default=512,
        help="number of decode steps to force (min==max, ignore_eos).",
    )
    parser.add_argument(
        "--chars-per-token", type=float, default=4.0,
        help="approx chars/token used to size the haystack to --input-len.",
    )
    parser.add_argument(
        "--window-frac", type=float, default=0.2,
        help="fraction of the decode horizon for early/late window means.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--timeout-s", type=int, default=3600,
        help="HTTP timeout for the (long) generation request.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="optional label for this run (e.g. 'base' or 'frozen_reuse'), "
        "recorded in the output JSON for cross-variant Delta_ppl comparison.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not 0.0 < args.window_frac <= 0.5:
        raise ValueError("--window-frac must be in (0, 0.5].")

    rng = random.Random(args.seed)
    target_chars = int(args.input_len * args.chars_per_token)
    prompt = build_prompt(rng, target_chars)
    log.info(
        "prompt built: target_input_len=%d tokens (~%d chars), output_len=%d, "
        "seed=%d, tag=%s",
        args.input_len, target_chars, args.output_len, args.seed, args.tag,
    )

    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,               # greedy / deterministic
            "max_new_tokens": args.output_len,
            "min_new_tokens": args.output_len,  # force full decode horizon
            "ignore_eos": True,                 # do not stop early on EOS
        },
        "return_logprob": True,               # top-level, per GenerateReqInput
        "return_text_in_logprobs": False,
        "logprob_start_len": -1,
    }

    log.info("POST %s/generate (timeout=%ds) ...", args.base_url, args.timeout_s)
    t0 = time.time()
    resp = _post_generate(args.base_url, payload, args.timeout_s)
    elapsed = time.time() - t0

    meta = resp["meta_info"]
    actual_prompt_tokens = meta.get("prompt_tokens")
    completion_tokens = meta.get("completion_tokens")
    logprobs, token_ids = extract_decode_logprobs(resp)
    log.info(
        "decode done: prompt_tokens=%s completion_tokens=%s decode_logprobs=%d "
        "elapsed=%.1fs",
        actual_prompt_tokens, completion_tokens, len(logprobs), elapsed,
    )

    metrics = compute_metrics(logprobs, args.window_frac)
    degen = compute_degeneration(token_ids)
    if degen["degenerate"]:
        log.warning(
            "DEGENERATE generation (distinct_ratio=%.3f repeat_4gram_frac=%.3f): "
            "ppl=%.4f is likely a confident-collapse artifact, NOT a quality signal. "
            "Trust drift over absolute ppl for this run.",
            degen["distinct_token_ratio"], degen["repeat_4gram_frac"], metrics["ppl"],
        )

    summary = {
        "tag": args.tag,
        "base_url": args.base_url,
        "target_input_len": args.input_len,
        "actual_prompt_tokens": actual_prompt_tokens,
        "requested_output_len": args.output_len,
        "completion_tokens": completion_tokens,
        "seed": args.seed,
        "chars_per_token": args.chars_per_token,
        "elapsed_s": round(elapsed, 1),
        "metric": "self_logprob_greedy_decode",
        **metrics,
        **degen,
        # Full per-step trace so the position-resolved drift can be re-plotted
        # or re-analyzed offline without re-running the (expensive) generation.
        "per_step_logprobs": logprobs,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "RESULT tag=%s ppl=%.4f mean_logprob=%.4f drift=%.4f slope=%.3e -> %s",
        args.tag, metrics["ppl"], metrics["mean_logprob"],
        metrics["drift_logprob"], metrics["slope_logprob_per_token"], args.output,
    )


if __name__ == "__main__":
    main()
