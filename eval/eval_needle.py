"""Multi-needle long-context retrieval probe against a served SGLang instance.

Self-contained (stdlib only, deterministic): builds a synthetic haystack with
N needle facts at controlled depths, then asks for each needle's payload with
greedy decoding via the native /generate endpoint (no chat template needed —
the prompt is completion-style). Score = exact-match fraction.

This probes exactly the mechanism the reuse-op ladder varies: whether the
sparse index set still covers needle KV at answer time (copy goes stale;
admission/cascade can re-admit).
"""

import argparse
import json
import logging
import random
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("needle")

FILLER_TOPICS = [
    "the migration patterns of arctic terns",
    "the history of the printing press",
    "techniques for fermenting vegetables",
    "the geology of basalt columns",
    "early developments in radio astronomy",
    "the economics of container shipping",
    "traditional methods of indigo dyeing",
    "the lifecycle of deciduous forests",
]

NEEDLE_KEYS = [
    "cherry", "falcon", "granite", "lantern", "meadow",
    "quartz", "saffron", "trellis", "velvet", "walnut",
    "amber", "birch", "copper", "dahlia", "ember",
    "fjord", "harbor", "ivory", "juniper", "kestrel",
]


def build_haystack(rng: random.Random, target_chars: int, needles: dict) -> str:
    """Deterministic filler prose with needles inserted at spread depths."""
    sentences = []
    while sum(len(s) for s in sentences) < target_chars:
        topic = rng.choice(FILLER_TOPICS)
        n = rng.randint(12, 30)
        words = " ".join(
            rng.choice(
                ["notably", "however", "records", "suggest", "patterns",
                 "observed", "during", "seasonal", "shifts", "gradual",
                 "documented", "analysis", "reveals", "distinct", "regional",
                 "variations", "throughout", "periods", "of", "study"]
            )
            for _ in range(n)
        )
        sentences.append(f"Regarding {topic}, {words}.")
    keys = list(needles)
    positions = [int(len(sentences) * (i + 1) / (len(keys) + 1)) for i in range(len(keys))]
    for key, pos in zip(keys, positions):
        sentences.insert(pos, f"The secret code for {key} is {needles[key]}.")
    return " ".join(sentences)


def generate(base_url: str, prompt: str, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--target-chars", type=int, default=260_000,
                        help="haystack size in chars (~4 chars/token)")
    parser.add_argument("--num-needles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--mode",
        choices=["single", "aggregate"],
        default="single",
        help="single: one short-answer query per needle (prefill-dominated). "
        "aggregate: one query must list ALL codes in one long decode — "
        "per-step index coverage is load-bearing, which is what the "
        "reuse-op ladder varies at decode time.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    keys = NEEDLE_KEYS[: args.num_needles]
    needles = {k: str(rng.randint(100000, 999999)) for k in keys}
    haystack = build_haystack(rng, args.target_chars, needles)
    log.info("haystack chars=%d needles=%d mode=%s",
             len(haystack), len(needles), args.mode)

    results = {}
    t0 = time.time()
    if args.mode == "aggregate":
        first = keys[0]
        prompt = (
            haystack
            + "\n\nBelow is the complete list of every secret code from the"
            + " document, in order of appearance:\n"
            + f"- The secret code for {first} is {needles[first]}.\n-"
        )
        out = generate(args.base_url, prompt, max_tokens=40 * len(keys))
        for key in keys[1:]:
            hit = needles[key] in out
            results[key] = {"expected": needles[key], "hit": hit}
            log.info("aggregate needle %-8s expected=%s hit=%s",
                     key, needles[key], hit)
        results["_generation"] = {"got": out[:2000]}
        score = sum(
            r["hit"] for k, r in results.items() if not k.startswith("_")
        ) / (len(keys) - 1)
    else:
        for key in keys:
            prompt = (
                haystack
                + f"\n\nQuestion: What is the secret code for {key}?"
                + f"\nAnswer: The secret code for {key} is"
            )
            out = generate(args.base_url, prompt, max_tokens=12)
            hit = needles[key] in out
            results[key] = {"expected": needles[key], "got": out.strip()[:40], "hit": hit}
            log.info("needle %-8s expected=%s hit=%s got=%r",
                     key, needles[key], hit, out.strip()[:40])
        score = sum(r["hit"] for r in results.values()) / len(results)
    summary = {
        "score": score,
        "mode": args.mode,
        "num_needles": len(keys),
        "haystack_chars": len(haystack),
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("SCORE=%.2f -> %s", score, args.output)


if __name__ == "__main__":
    main()
