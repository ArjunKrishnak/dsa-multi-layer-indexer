"""LongBench subsets served-eval client for the reuse-op ladder.

Runs official LongBench v1 tasks against a served SGLang instance. Templates,
generation budgets, and metrics are copied VERBATIM from the research repo's
real_lag_longbench.py (itself verified 2026-07-21 against THUDM/LongBench).

Disclosed protocol deviation: the official pipeline middle-truncates by
TOKENS with the model tokenizer; here we middle-truncate by CHARS (default
250,000 ≈ 62K tokens) because the client is tokenizer-free. The same
truncation applies to every arm, so cross-arm comparisons remain valid; the
absolute scores are not comparable to published LongBench numbers.

Subsets chosen to discriminate reuse ops (multi-hop QA dips under sparse
reuse in our Phase-C data; counting/aggregation is where IndexCache's 1/8
collapse concentrates): hotpotqa, 2wikimqa, musique (qa_f1), passage_count
(count_score).
"""

import argparse
import json
import logging
import os
import re
import signal
import string
import time
import urllib.request
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("lb")

TASK_PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\nAnswer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Given {context}\n\nThe following is a summary of one of the paragraphs above:\n{input}\n\nPlease determine which paragraph the summary is from and output only the paragraph number.\n\nThe paragraph number is:",
}
TASK_MAX_NEW = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "passage_count": 32, "passage_retrieval_en": 32,
}


def _normalise(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, gold: str) -> float:
    p_toks = _normalise(pred).split()
    g_toks = _normalise(gold).split()
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_toks)
    rec = num_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def count_score(pred: str, gold: str) -> float:
    numbers = re.findall(r"\d+", pred)
    if not numbers:
        return 0.0
    return sum(1 for n in numbers if str(n) == str(gold)) / len(numbers)


def retrieval_score(pred: str, gold: str) -> float:
    pred_nums = re.findall(r"\d+", pred.strip())
    gold_nums = re.findall(r"\d+", gold.strip())
    return 1.0 if gold_nums and gold_nums[0] in pred_nums else 0.0


METRIC = {
    "narrativeqa": qa_f1,
    "qasper": qa_f1,
    "multifieldqa_en": qa_f1,
    "hotpotqa": qa_f1,
    "2wikimqa": qa_f1,
    "musique": qa_f1,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
}

# --- Decode-heavy summarization tasks (LongBench, loaded from local JSONL in
#     --local-jsonl-dir). Long references (100-520 words) force long generation,
#     so the DECODE-ONLY re-rank/admission/hub actually execute for hundreds of
#     steps -- the regime the short-answer QA eval could not exercise. Metric =
#     ROUGE-L F1 (LCS); an approximation of LongBench's rouge_score (no stemming)
#     but identical across arms, so cross-arm comparison is valid. ---
SUMM_TASKS = {"gov_report", "multi_news", "qmsum"}
TASK_PROMPT.update({
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news.\n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
})
TASK_MAX_NEW.update({"gov_report": 512, "multi_news": 512, "qmsum": 512})


def _lcs_len(a: list, b: list) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == b[j - 1] else (dp[j] if dp[j] >= dp[j - 1] else dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(pred: str, gold: str) -> float:
    p, g = _normalise(pred).split(), _normalise(gold).split()
    if not p or not g:
        return 0.0
    l = _lcs_len(p, g)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(g)
    return 2 * prec * rec / (prec + rec)


METRIC.update({t: rouge_l for t in SUMM_TASKS})


def middle_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + text[-half:]


# GLM-raw mode: for GLM-5, the CHAT path (/v1/chat/completions, or the chat-template
# rendered TEXT) degenerates into garbage at LONG context (~62K), even with
# --reasoning-parser glm45 (in our probes, chat-template text answers a SHORT prompt
# but the [gMASK]<sop>...<|assistant|> markers degenerate at 62K). The RAW-completion
# path via /generate — no chat template, exactly how the needle probe scores GLM 1.00
# at 128K — stays coherent (probe: RAW plain /generate -> " Paris, France."). So for
# GLM we send the LongBench prompt AS-IS to /generate.
_GLM_RAW = False  # set True in main() when --glm-raw
_DS_CHAT = False  # set True in main() when --deepseek-chat (DeepSeek-V3.2 instruct format via /generate)


def _strip_think(text: str):
    """Return (answer, degenerate). Answer = text after a CLOSED </think>. If a <think> block
    was OPENED but never closed (reasoning ran past the token budget), the whole text is a think
    fragment, NOT an answer -> return ("", True). FAIL-LOUD fix (methodology audit 2026-08-07):
    the old handler only split on </think>, so an unterminated block was silently SCORED as the
    answer -> a future arm entering extended thinking would tank and read as a quality regression."""
    if "</think>" in text:
        return text.split("</think>")[-1], False
    if "<think>" in text:
        return "", True
    return text, False


def generate(base_url: str, prompt: str, max_tokens: int,
             chat_template_kwargs: dict = None) -> str:
    if _GLM_RAW:
        body = {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens},
        }
        req = urllib.request.Request(
            f"{base_url}/generate", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3600) as resp:
            text = json.loads(resp.read())["text"] or ""
        answer, degen = _strip_think(text)
        if degen:
            log.warning("unterminated <think> block (glm-raw, max_new=%d) -> no answer scored "
                        "(raise max_new_tokens or check thinking mode)", max_tokens)
        return answer
    if _DS_CHAT:
        # DeepSeek-V3.2 instruct format, identical to LongBench v2's wrap_deepseek_chat.
        # The role markers use the FULLWIDTH pipe U+FF5C ('｜'), NOT ASCII '|', matching the
        # checkpoint's chat template; the </think> tail forces a direct (non-thinking) answer,
        # avoiding the instant-EOS empties of naive raw /generate on this instruct-tuned model.
        wrapped = "<｜User｜>" + prompt + "<｜Assistant｜></think>"
        body = {"text": wrapped, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens}}
        req = urllib.request.Request(
            f"{base_url}/generate", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3600) as resp:
            text = json.loads(resp.read())["text"] or ""
        answer, degen = _strip_think(text)
        if degen:
            log.warning("unterminated <think> block (deepseek-chat) -> no answer scored")
        return answer
    # Chat endpoint, NOT raw /generate: DS-V3.2 is instruct-tuned and emits
    # an instant EOS on many raw-completion LongBench prompts (verified live
    # 2026-07-28: raw gave '' on 2/3 hotpotqa examples, chat 3/3 exact).
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    # GLM-5 has a default-ON reasoning ("thinking") mode; pass
    # {"enable_thinking": false} so a 32-token budget yields the answer, not a
    # truncated <think> block. DS-V3.2 ignores unknown template kwargs.
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        msg = json.loads(resp.read())["choices"][0]["message"]
    text = msg.get("content") or ""
    answer, degen = _strip_think(text)
    if degen:
        log.warning("unterminated <think> block (chat) -> no answer scored")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--tasks", default="hotpotqa,2wikimqa,musique,passage_count")
    parser.add_argument("--n-per-task", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0,
                        help="skip the first N examples/task -> HELD-OUT confirmatory eval "
                        "on a slice DISJOINT from the n=50 selection set (idx>=offset). "
                        "Required for a valid escalation per rigorous-comparison-protocol.")
    parser.add_argument("--max-chars", type=int, default=250_000)
    parser.add_argument("--hf-home", default=None,
                        help="HF cache with pre-downloaded THUDM/LongBench")
    parser.add_argument("--chat-template-kwargs", default=None,
                        help='JSON, e.g. \'{"enable_thinking": false}\' for GLM-5')
    parser.add_argument("--local-jsonl-dir", default="/hh/lb_extra",
                        help="dir with {task}.jsonl for SUMM_TASKS (offline)")
    # --deepseek-chat and --glm-raw pick different generation wrappers; generate()
    # checks _GLM_RAW first, so enabling both would silently ignore --deepseek-chat.
    # Make them mutually exclusive so the conflict is rejected at parse time.
    gen_mode = parser.add_mutually_exclusive_group()
    gen_mode.add_argument("--deepseek-chat", action="store_true",
                        help="DeepSeek-V3.2: wrap prompts in the <|User|>...<|Assistant|></think> instruct format and hit /generate (matches LongBench v2; avoids the string-format fallback when the checkpoint ships no chat template).")
    gen_mode.add_argument("--glm-raw", action="store_true",
                        help="render prompts with the model tokenizer's chat "
                        "template (thinking off) and hit /generate — for GLM-5, "
                        "whose /v1/chat/completions rendering is broken in sglang.")
    parser.add_argument("--model-path", default="/model",
                        help="tokenizer source for --glm-raw.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    chat_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    if args.deepseek_chat:
        global _DS_CHAT
        _DS_CHAT = True
        log.info("deepseek-chat mode: <|User|>...<|Assistant|></think> instruct wrap via /generate")
    if args.glm_raw:
        global _GLM_RAW
        _GLM_RAW = True
        log.info("glm-raw mode: RAW-completion prompts via /generate (no chat "
                 "template — GLM's chat markers degenerate at long context)")

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    from datasets import load_dataset  # after HF_HOME is set

    tasks = args.tasks.split(",")
    # Build ONE global work list sorted by DESCENDING context length (the serving
    # order used for the paper's runs). Scoring stays per-task; serving order does
    # not affect the per-example F1 or the macro.
    work = []            # (task, orig_idx, row) — orig_idx is stable across runs
    n_target = {}
    for task in tasks:
        if task in SUMM_TASKS:
            path = os.path.join(args.local_jsonl_dir, task + ".jsonl")
            rows = [json.loads(line) for line in open(path)][args.offset : args.offset + args.n_per_task]
        else:
            ds = load_dataset("THUDM/LongBench", task, split="test",
                              trust_remote_code=True)
            rows = list(ds)[args.offset : args.offset + args.n_per_task]
        n_target[task] = len(rows)
        for orig_idx, row in enumerate(rows):
            work.append((task, orig_idx, row))
    work.sort(key=lambda w: len(w[2]["context"]), reverse=True)

    # RESUME (preemption / requeue safety): reload any prior partial output and
    # skip examples already scored. Requeued jobs keep the same SLURM_JOB_ID, so
    # the output path is stable and this continues instead of restarting.
    scores = {task: {} for task in tasks}     # task -> {orig_idx: score}
    sample_preds = {}
    if os.path.exists(args.output):
        try:
            prev = json.load(open(args.output))
            for task in tasks:
                sbi = prev.get("tasks", {}).get(task, {}).get("scores_by_idx", {})
                scores[task] = {int(k): v for k, v in sbi.items()}
            sample_preds = prev.get("sample_preds", {})
            done = sum(len(s) for s in scores.values())
            if done:
                log.info("RESUME: %d scores already done in %s", done, args.output)
        except Exception as e:  # corrupt/old-format file -> start clean
            log.warning("could not resume from %s (%s); starting fresh",
                        args.output, e)
            scores = {task: {} for task in tasks}

    t0 = time.time()

    def dump(partial: bool) -> float:
        results = {}
        for task in tasks:
            s = list(scores[task].values())
            results[task] = {
                "mean": (sum(s) / len(s) if s else 0.0),
                "n": len(s),
                "n_target": n_target[task],
                "scores_by_idx": {str(k): v for k, v in scores[task].items()},
            }
        graded = [r["mean"] for r in results.values() if r["n"] > 0]
        macro = sum(graded) / len(graded) if graded else 0.0
        summary = {"macro_mean": macro, "max_chars": args.max_chars,
                   "partial": partial,
                   "completed": sum(len(scores[t]) for t in tasks),
                   "total": len(work),
                   "elapsed_s": round(time.time() - t0, 1),
                   "tasks": results, "sample_preds": sample_preds}
        # Atomic write: a preemption mid-write must never corrupt the file.
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, args.output)
        return macro

    # SLURM sends SIGTERM before SIGKILL on preemption — flush progress first.
    def _on_sigterm(signum, frame):
        log.warning("SIGTERM: flushing %d completed before exit",
                    sum(len(scores[t]) for t in tasks))
        dump(partial=True)
        os._exit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)

    remaining = [(t, i, r) for (t, i, r) in work if i not in scores[t]]
    log.info("serving %d requests (%d already done) across %d tasks, "
             "descending-length order", len(remaining),
             len(work) - len(remaining), len(tasks))

    for n, (task, orig_idx, row) in enumerate(remaining):
        context = middle_truncate(row["context"], args.max_chars)
        prompt = TASK_PROMPT[task].format(context=context,
                                          input=row.get("input", ""))
        pred = generate(args.base_url, prompt, TASK_MAX_NEW[task], chat_kwargs)
        # QA/count: answer is the first line. Summarization: the whole output.
        pred_out = pred.strip() if task in SUMM_TASKS else pred.strip().split("\n")[0]
        best = max(
            (METRIC[task](pred_out, str(g)) for g in row["answers"]),
            default=0.0,
        )
        scores[task][orig_idx] = best
        if len(sample_preds.get(task, [])) < 5:
            sample_preds.setdefault(task, []).append(
                {"pred": pred_out[:100],
                 "gold": [str(g)[:60] for g in row["answers"]],
                 "f1": round(best, 3)}
            )
        # Frequent atomic checkpoint (every 10) — a preemption/wall loses at
        # most the current 10, and the longest docs run first (descending).
        if (n + 1) % 10 == 0:
            macro = dump(partial=True)
            log.info("progress %d/%d (total %d/%d) running macro=%.4f",
                     n + 1, len(remaining),
                     sum(len(scores[t]) for t in tasks), len(work), macro)

    macro = dump(partial=False)
    for task in tasks:
        s = list(scores[task].values())
        log.info("TASK %s mean=%.4f n=%d", task,
                 sum(s) / len(s) if s else 0.0, len(s))
    log.info("MACRO=%.4f -> %s", macro, args.output)


if __name__ == "__main__":
    main()
