"""RULER served-eval client (IndexCache's RULER setting, 4K-128K).

Generates RULER-style synthetic tasks in-process with the model's tokenizer
(``lagged_indexer_dsa.ruler_gen``: niah_single, niah_multikey, niah_multivalue,
vt, cwe -- deterministic given seed/task/length/depth/index) and scores greedy
completions from a served SGLang endpoint. Prompts are sent as token ids so
every arm sees byte-identical inputs. Each arm's score is comparable only to
arms evaluated with the same (tasks, lengths, depths, n_per_cell, seed).

Deviation from the official RULER suite, disclosed in the paper: a five-task
subset regenerated from a seed (the official 13-task suite is not a library),
with the same metric definitions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.request
from collections import defaultdict

from lagged_indexer_dsa.ruler_gen import (
    DEFAULT_DEPTHS,
    MAX_NEW_BY_TASK,
    METRIC_BY_TASK,
    RULER_TASKS,
    cells_for_task,
    generate_example,
    score_example,
)
from lagged_indexer_dsa.served_eval import aggregate_ruler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ruler")


def generate(base_url: str, input_ids: list[int], max_tokens: int, timeout: int = 3600) -> str:
    payload = json.dumps(
        {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model-path", default="/model", help="tokenizer source")
    ap.add_argument("--tasks", default=",".join(RULER_TASKS))
    ap.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072")
    ap.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS))
    ap.add_argument("--n-per-cell", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        ap.error("--tasks must name at least one task")
    if len(tasks) != len(set(tasks)):
        ap.error("--tasks must not contain duplicate task names (each would be double-counted)")
    unknown = [t for t in tasks if t not in RULER_TASKS]
    if unknown:
        ap.error(f"--tasks has unknown task(s) {unknown}; valid tasks are {sorted(RULER_TASKS)}")
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [int(x) for x in args.depths.split(",")]

    per_example = []
    scored: list[tuple[str, int, float]] = []
    t0 = time.time()
    n_total = sum(len(cells_for_task(t, lengths, depths, args.n_per_cell)) for t in tasks)
    done = 0
    for task in tasks:
        for length, depth, index in cells_for_task(task, lengths, depths, args.n_per_cell):
            ex = generate_example(
                task, tok, context_length=length, depth_percent=depth, seed=args.seed, index=index
            )
            out = generate(args.base_url, ex["ids"], MAX_NEW_BY_TASK[task])
            score = score_example(task, out, ex["answers"])
            scored.append((task, length, score))
            per_example.append(
                dict(task=task, length=length, depth=depth, index=index, n_ids=len(ex["ids"]),
                     score=score, answers=ex["answers"], got=out.strip()[:200])
            )
            done += 1
            log.info("[%d/%d] %s L=%d d=%s i=%d score=%.3f (%.0fs)",
                     done, n_total, task, length, depth, index, score, time.time() - t0)

    overall, by_length, table = aggregate_ruler(scored, tasks, lengths)
    summary = dict(
        overall=overall, by_length=by_length, by_task_length=table,
        metrics={t: METRIC_BY_TASK[t] for t in tasks},
        config=dict(tasks=tasks, lengths=lengths, depths=depths, n_per_cell=args.n_per_cell,
                    seed=args.seed, model_path=args.model_path),
        elapsed_s=time.time() - t0,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dict(summary=summary, examples=per_example), f, indent=1)
    log.info("RULER overall=%.4f by_length=%s -> %s", overall, by_length, args.output)


if __name__ == "__main__":
    main()
