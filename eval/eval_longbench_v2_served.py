"""LongBench v2 served-eval client (IndexCache's LongBench v2 setting).

Runs the official LongBench v2 zero-shot multiple-choice protocol (THUDM/LongBench-v2,
503 questions, 4 choices, accuracy) against a served SGLang endpoint with greedy
decoding. The official prompt is reproduced verbatim; the answer letter is parsed
from "The correct answer is (X)". Contexts longer than ``--max-ctx-tokens`` are
middle-truncated by TOKENS with the model tokenizer (the official pipeline's
rule), so every arm sees byte-identical inputs. The chat template is applied
client-side by default (``--chat-template-kwargs`` for e.g. thinking toggles);
``--no-chat-template`` sends raw text.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.request
from collections import defaultdict

from lagged_indexer_dsa.served_eval import extract_mcq_answer, middle_truncate_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("lbv2")

PROMPT = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


def generate(base_url: str, text: str, max_tokens: int, timeout: int = 3600) -> str:
    payload = json.dumps(
        {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens}}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["text"]


def wrap_deepseek_chat(user_content: str, think: bool = False) -> str:
    """DeepSeek-V3.2 chat format (encoding/encoding_dsv32.py: user_msg_template
    '<|User|>{content}<|Assistant|>'). The server prepends BOS.

    Non-thinking mode appends the thinking end token so the model answers directly;
    thinking mode appends the thinking start token (the encoding module's thinking_mode
    branch), so the model reasons and closes </think> before the answer. Leaving the tail
    empty is NOT thinking mode: the model then answers directly (verified empirically)."""
    tail = "<think>" if think else "</think>"
    return f"<｜User｜>{user_content}<｜Assistant｜>{tail}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model-path", default="/model", help="tokenizer (+ chat template) source")
    ap.add_argument("--max-ctx-tokens", type=int, default=120_000,
                    help="middle-truncate contexts longer than this many tokens")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all 503")
    ap.add_argument("--length-filter", default="", help="comma list of short,medium,long")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--prompt-format", default="none", choices=["none", "deepseek_chat"],
                    help="explicit prompt wrapper when the tokenizer has no chat template: "
                         "deepseek_chat = DeepSeek-V3.2 non-thinking format from the checkpoint's encoding/ module "
                         "(<|User|>...<|Assistant|></think>; BOS is added by the server)")
    ap.add_argument("--chat-template-kwargs", default=None, help='JSON, e.g. {"enable_thinking": false}')
    ap.add_argument("--think", action="store_true",
                    help="thinking mode: for --prompt-format deepseek_chat, omit the forced </think> so the "
                         "model reasons then answers (GLM uses --chat-template-kwargs enable_thinking instead). "
                         "The answer is extracted from the text after the final </think>.")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ct_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {}
    # --think only produces a </think> block via the deepseek_chat wrapper or an
    # enable_thinking chat template. Without one of those, thinking_on stays True but no
    # </think> is ever emitted, so every answer scores as unparsed -- fail loud instead.
    if args.think and args.prompt_format != "deepseek_chat" and not ct_kwargs.get("enable_thinking"):
        ap.error("--think requires --prompt-format deepseek_chat or "
                 "--chat-template-kwargs '{\"enable_thinking\": true}'; otherwise no <think> "
                 "block is produced and every answer is scored unparsed")
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    keep = {x.strip() for x in args.length_filter.split(",") if x.strip()}
    invalid_lengths = keep - {"short", "medium", "long"}
    if invalid_lengths:
        ap.error(f"unknown --length-filter value(s): {sorted(invalid_lengths)} (expected short/medium/long)")
    rows = [r for r in ds if not keep or r["length"] in keep]
    if args.max_samples:
        rows = rows[: args.max_samples]
    log.info("LongBench v2: %d questions (filter=%s)", len(rows), sorted(keep) or "none")

    per = []
    by = {k: defaultdict(list) for k in ("length", "domain", "difficulty")}
    t0 = time.time()
    for i, r in enumerate(rows):
        ctx, n_ids, truncated = middle_truncate_tokens(tok, r["context"], args.max_ctx_tokens)
        prompt = PROMPT.format(context=ctx.strip(), question=r["question"].strip(),
                               choice_A=r["choice_A"].strip(), choice_B=r["choice_B"].strip(),
                               choice_C=r["choice_C"].strip(), choice_D=r["choice_D"].strip())
        if not args.no_chat_template:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, **ct_kwargs
            )
        elif args.prompt_format == "deepseek_chat":
            prompt = wrap_deepseek_chat(prompt, think=args.think)
        out = generate(args.base_url, prompt, args.max_new_tokens)
        # In thinking mode the committed answer follows the final </think>; extracting from
        # the whole string would catch a letter uttered mid-reasoning. Split it off first
        # (a no-op when the output has no </think>, i.e. non-thinking runs).
        answer_region = out.rsplit("</think>", 1)[-1]
        thinking_on = args.think or bool(ct_kwargs.get("enable_thinking"))
        if thinking_on and "</think>" not in out:
            # Reasoning ran past the budget: the whole output is an unterminated think block.
            # Scoring it would credit a letter uttered mid-reasoning, so it is unparsed.
            pred = None
        else:
            pred = extract_mcq_answer(answer_region)
        correct = float(pred == r["answer"])
        for k in by:
            by[k][r[k]].append(correct)
        # Save the answer region (not the first 200 chars of a long reasoning trace) so
        # offline re-scoring sees the committed answer; flag whether a </think> was present
        # and how long the raw output was, to diagnose budget-truncated reasoning.
        per.append(dict(id=r["_id"], length=r["length"], domain=r["domain"], difficulty=r["difficulty"],
                        ctx_tokens=n_ids, truncated=truncated, gold=r["answer"], pred=pred, correct=correct,
                        had_think=("</think>" in out), raw_len=len(out), got=answer_region.strip()[:200]))
        log.info("[%d/%d] %s/%s gold=%s pred=%s acc_so_far=%.3f (%.0fs)", i + 1, len(rows), r["length"],
                 r["difficulty"], r["answer"], pred, sum(p["correct"] for p in per) / len(per), time.time() - t0)

    acc = sum(p["correct"] for p in per) / max(1, len(per))
    summary = dict(
        accuracy=acc, n=len(per),
        by_length={k: sum(v) / len(v) for k, v in by["length"].items()},
        by_domain={k: sum(v) / len(v) for k, v in by["domain"].items()},
        by_difficulty={k: sum(v) / len(v) for k, v in by["difficulty"].items()},
        unparsed=sum(1 for p in per if p["pred"] is None),
        truncated=sum(1 for p in per if p["truncated"]),
        # thinking diagnostics: how many outputs actually closed </think>, and how many
        # unparsed items look like the reasoning ran out of budget (no </think> and the raw
        # output hit the token cap). High budget_truncated => raise --max-new-tokens.
        had_think=sum(1 for p in per if p.get("had_think")),
        budget_truncated=sum(1 for p in per if p["pred"] is None and not p.get("had_think", False)),
        config=dict(max_ctx_tokens=args.max_ctx_tokens, max_new_tokens=args.max_new_tokens,
                    chat_template=not args.no_chat_template, chat_template_kwargs=ct_kwargs, prompt_format=args.prompt_format,
                    think=args.think, max_samples=args.max_samples, length_filter=sorted(keep)),
        elapsed_s=time.time() - t0,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dict(summary=summary, examples=per), f, indent=1)
    log.info("LongBench v2 accuracy=%.4f n=%d unparsed=%d -> %s", acc, len(per), summary["unparsed"], args.output)


if __name__ == "__main__":
    main()
