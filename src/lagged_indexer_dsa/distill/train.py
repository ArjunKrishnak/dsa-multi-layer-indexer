"""Phase 1: cross-layer KL distillation of the lagged lightning indexer.

Trains each producing (layer ``L-1``) indexer so its score distribution matches
the consuming layer ``L``'s own indexer distribution -- the same-layer selection
the one-layer lag would ideally reuse. The backbone stays frozen; only the tiny
indexer projections (re-expressed as a differentiable
:class:`~lagged_indexer_dsa.distill.indexer_ref.IndexerRef`) are updated.

Each step:
  1. one frozen same-layer forward (``inference_mode``) harvests per-layer inputs
     ``(x, qr, freqs_cis)`` and teacher scores via forward hooks;
  2. for every adjacent (L-1, L) pair, the student ``IndexerRef`` for L-1 scores
     the sampled query rows and is pushed toward layer L's teacher distribution
     (per-pair backward to bound memory);
  3. AdamW step.

Tensor-parallel note: indexer params are replicated and the residual-stream
inputs / broadcast teacher scores are identical across ranks, so with rank-0
broadcast query sampling every rank computes identical grads and the students
stay in sync without gradient all-reduce.

Launch (single or multi node)::

    torchrun --nnodes=N --nproc-per-node=8 -m lagged_indexer_dsa.distill.train \\
        --model glm5 --ckpt-path /path/to/glm-5 --dataset govreport \\
        --seq 16384 --steps 300 --output results/distill/indexer_lag_glm5.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.optim import AdamW

from lagged_indexer_dsa.distill.collect import IndexerActivationCollector
from lagged_indexer_dsa.distill.indexer_ref import indexer_ref_from_module
from lagged_indexer_dsa.distill.losses import (
    causal_mask,
    cross_layer_kl,
    cross_layer_list_mle,
    cross_layer_rank,
    topk_overlap,
)
from lagged_indexer_dsa.inference import load_model

logger = logging.getLogger(__name__)

DATASET_REGISTRY: dict[str, str] = {
    "govreport": "ccdv/govreport-summarization",
    "arxiv": "ccdv/arxiv-summarization",
}


# ---------------------------------------------------------------------------
# Small infra helpers (no external project dependency)
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> SimpleNamespace:
    """Init the NCCL process group from env vars; return rank/world_size.

    Falls back to a single-process context when ``RANK`` is unset.
    """
    if dist.is_available() and "RANK" in os.environ:
        # Bind the device before NCCL init so each rank sets up its communicator
        # on the intended GPU (avoids wrong-device allocations / hangs).
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return SimpleNamespace(rank=dist.get_rank(), world_size=dist.get_world_size())
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return SimpleNamespace(rank=0, world_size=1)


def configure_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank{rank}] %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_documents_as_dicts(dataset: str, max_samples: int) -> list[dict]:
    """Return up to ``max_samples`` documents as ``[{"text": ...}]``."""
    from datasets import load_dataset as hf_load  # type: ignore[import]

    hf_name = DATASET_REGISTRY[dataset]
    try:
        ds = hf_load(hf_name, split="test", trust_remote_code=True)
    except ValueError:
        # Raised by HF datasets when the "test" split is absent; fall back to
        # "train". Any other failure (network, auth, ...) should propagate.
        ds = hf_load(hf_name, split="train", trust_remote_code=True)

    docs: list[dict] = []
    for i in range(min(max_samples, len(ds))):
        row = ds[i]
        text = row.get("article") or row.get("report") or row.get("document") or row.get("text") or ""
        docs.append({"text": text})
    return docs


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="deepseek")
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--dataset", default="govreport", choices=sorted(DATASET_REGISTRY))
    p.add_argument("--seq", type=int, default=8192, help="single-stream sequence length (non-chunked forward)")
    p.add_argument("--num-streams", type=int, default=8, help="packed training streams to sample from")
    p.add_argument("--pack-docs", type=int, default=256)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--query-samples", type=int, default=512, help="query rows scored per step")
    p.add_argument("--query-chunk", type=int, default=128)
    p.add_argument("--min-query-pos", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-query-samples", type=int, default=256)
    p.add_argument("--topk", type=int, default=None, help="k for the overlap proxy (default: model index_topk)")
    p.add_argument("--objective", default="kl", choices=["kl", "rank", "list_mle", "hybrid"],
                   help="distillation loss: cross-layer KL (soft distribution), "
                        "pairwise MarginRanking hinge (teacher key order), listwise "
                        "Plackett-Luce MLE on teacher top-C, or KL+ListMLE hybrid")
    p.add_argument("--rank-pairs", type=int, default=1024,
                   help="key pairs sampled per query row (rank objective)")
    p.add_argument("--rank-margin", type=float, default=0.0,
                   help="margin for the ranking hinge (rank objective; 0 = pure order)")
    p.add_argument("--list-mle-candidates", type=int, default=2048,
                   help="teacher top-C pool size (list_mle / hybrid objectives)")
    p.add_argument("--kl-weight", type=float, default=0.5,
                   help="KL mixing weight in [0,1] for the hybrid objective")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="max grad norm (<= 0 disables clipping)")
    p.add_argument("--span-include-self", action="store_true",
                   help="span-aware: also train each refresh layer against ITSELF as a consumer "
                        "(IndexCache's training-aware spans r..next-1, where the retained layer "
                        "serves its own attention); default = the lag's spans r+1..next-1")
    p.add_argument("--refresh-layers", default=None,
                   help="comma-separated refresh layer indices for span-aware distillation; "
                        "each refresh layer is trained against ALL consumers in its span "
                        "(up to the next refresh layer) instead of only the adjacent layer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    if args.seq <= 0:
        p.error("--seq must be positive")
    if args.rank_pairs <= 0:
        p.error("--rank-pairs must be positive")
    if args.rank_margin < 0:
        p.error("--rank-margin must be non-negative")
    if args.list_mle_candidates <= 0:
        p.error("--list-mle-candidates must be positive")
    if not 0.0 <= args.kl_weight <= 1.0:
        p.error("--kl-weight must be in [0, 1]")
    if args.num_streams < 2:
        p.error("--num-streams must be >= 2 (the last stream is reserved for eval)")
    if args.steps <= 0:
        p.error("--steps must be positive")
    if args.query_samples <= 0:
        p.error("--query-samples must be positive")
    if args.eval_query_samples <= 0:
        p.error("--eval-query-samples must be positive")
    if args.query_chunk <= 0:
        p.error("--query-chunk must be positive")
    if args.eval_every <= 0:
        p.error("--eval-every must be positive")
    if args.pack_docs <= 0:
        p.error("--pack-docs must be positive")
    if args.min_query_pos < 0:
        p.error("--min-query-pos must be non-negative")
    return args


def _broadcast_query_idx(num_samples: int, lo: int, seq: int, generator: torch.Generator) -> torch.Tensor:
    """Sample query rows in [lo, seq) identically across ranks (rank-0 broadcast)."""
    pool = torch.arange(lo, seq)
    if num_samples > pool.numel():
        raise ValueError(
            f"requested {num_samples} query samples but only {pool.numel()} "
            f"rows available in [{lo}, {seq})"
        )
    perm = torch.randperm(pool.numel(), generator=generator)[:num_samples]
    idx = pool[perm].sort().values.to("cuda")
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(idx, src=0)
    return idx


def _pack_streams(tokenizer, docs, seq: int, num_streams: int) -> list[torch.Tensor]:
    all_tokens: list[int] = []
    for doc in docs:
        all_tokens.extend(tokenizer(doc["text"], return_tensors="pt").input_ids[0].tolist())
    need = seq * num_streams
    if len(all_tokens) < need:
        raise RuntimeError(
            f"packed corpus {len(all_tokens)} < {need} required ({num_streams} x {seq}); raise --pack-docs"
        )
    return [
        torch.tensor(all_tokens[i * seq : (i + 1) * seq], dtype=torch.long)
        for i in range(num_streams)
    ]


def _frozen_collect(model, collector, tokens, query_idx):
    """One frozen same-layer forward, capturing inputs + teacher scores."""
    model.set_indexer_frozen(False)
    model.set_lag_mode(False)
    with collector:
        collector.set_query_idx(query_idx)
        with torch.inference_mode():
            model(tokens, start_pos=0, full_logits=False)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ctx = setup_distributed()
    configure_logging(ctx.rank)
    rank, world_size = ctx.rank, ctx.world_size
    is_main = rank == 0

    docs = load_documents_as_dicts(args.dataset, max_samples=args.pack_docs)
    t0 = time.perf_counter()
    model, tokenizer, ckpt_path = load_model(
        args.model, ckpt_path=args.ckpt_path,
        max_seq_len=args.seq + 2, rank=rank, world_size=world_size,
    )
    # This repo lazily allocates KV caches (seqlen=1 at init); size them to the
    # eval length before the frozen forward.
    model.reset_kv_cache(bsz=1, seqlen=args.seq + 2)
    first_indexer = next(
        (
            getattr(getattr(layer, "attn", None), "indexer", None)
            for layer in model.layers
            if getattr(getattr(layer, "attn", None), "indexer", None) is not None
        ),
        None,
    )
    if first_indexer is None:
        raise RuntimeError("no indexer modules found")
    index_topk = first_indexer.index_topk
    if args.topk is not None and args.topk <= 0:
        raise ValueError(f"--topk must be positive, got {args.topk}")
    topk = args.topk or index_topk
    if is_main:
        logger.info("Model loaded in %.1fs; index_topk=%d", time.perf_counter() - t0, index_topk)

    streams = _pack_streams(tokenizer, docs, args.seq, args.num_streams)
    lo = min(args.min_query_pos, args.seq - 1)

    # Discover which layers' indexers fire (same-layer forward), then form the
    # adjacent producer->consumer pairs the lag relies on.
    probe = IndexerActivationCollector(model)
    probe_idx = _broadcast_query_idx(8, lo, args.seq, torch.Generator().manual_seed(args.seed))
    _frozen_collect(model, probe, streams[0][: args.seq].unsqueeze(0).to("cuda"), probe_idx)
    fired = sorted(probe.inputs)
    probe.clear()
    if args.refresh_layers is not None:
        refresh = sorted(int(x) for x in args.refresh_layers.split(","))
        fired_set = set(fired)
        invalid = [r for r in refresh if r not in fired_set]
        if invalid:
            raise ValueError(f"refresh layers {invalid} not in fired set {fired}")
        pairs = []
        for i, r in enumerate(refresh):
            span_end = refresh[i + 1] if i + 1 < len(refresh) else max(fired) + 1
            c_start = r if args.span_include_self else r + 1
            for c in range(c_start, span_end):
                if c in fired_set:
                    pairs.append((r, c))
        producers = sorted({prod for prod, _ in pairs})
    else:
        pairs = [(L - 1, L) for L in fired if (L - 1) in fired]
        producers = sorted({prod for prod, _ in pairs})
    if not pairs:
        raise RuntimeError("no producer->consumer pairs found")
    if is_main:
        logger.info("Fired layers: %d; producer->consumer pairs: %d (producers %d..%d)%s",
                    len(fired), len(pairs), producers[0], producers[-1],
                    f" [span-aware: {len(producers)} refresh layers]" if args.refresh_layers else "")

    students = {
        prod: indexer_ref_from_module(model.layers[prod].attn.indexer).to("cuda")
        for prod in producers
    }
    params = [p for s in students.values() for p in s.parameters()]
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    collector = IndexerActivationCollector(model)
    rng = torch.Generator().manual_seed(args.seed + 1)
    # Dedicated on-device RNG for the ranking objective's pair sampling. Seeded
    # identically on every rank and drawn the same amount each step, so grads
    # stay in sync without all-reduce (same TP assumption as query sampling).
    rank_gen: torch.Generator | None = None
    if args.objective == "rank":
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        rank_gen = torch.Generator(device=gen_device).manual_seed(args.seed + 7)
    loss_name = args.objective
    history: list[dict] = []

    for step in range(1, args.steps + 1):
        # Reserve the last stream (streams[-1]) exclusively for _evaluate so the
        # held-out eval signal is never trained on directly.
        stream_i = int(torch.randint(0, args.num_streams - 1, (1,), generator=rng).item())
        if dist.is_initialized() and dist.get_world_size() > 1:
            stream_t = torch.tensor([stream_i], device="cuda")
            dist.broadcast(stream_t, src=0)
            stream_i = int(stream_t.item())
        tokens = streams[stream_i][: args.seq].unsqueeze(0).to("cuda")
        query_idx = _broadcast_query_idx(args.query_samples, lo, args.seq, rng)

        _frozen_collect(model, collector, tokens, query_idx)
        causal = causal_mask(query_idx, args.seq)

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        active_pairs = [
            (prod, cons)
            for prod, cons in pairs
            if cons in collector.teacher_scores and prod in collector.inputs
        ]
        if not active_pairs:
            raise RuntimeError("no producer->consumer indexer pairs fired this step")
        n_pairs = len(active_pairs)
        for prod, cons in active_pairs:
            x, qr, freqs_cis = (t.to("cuda") for t in collector.inputs[prod])
            student_score, _ = students[prod](
                x, qr, freqs_cis, query_idx=query_idx, query_chunk=args.query_chunk
            )
            teacher = collector.teacher_scores[cons]
            if args.objective == "kl":
                loss = cross_layer_kl(teacher, student_score, causal)
            elif args.objective == "rank":
                loss = cross_layer_rank(
                    teacher, student_score, causal,
                    args.rank_pairs, args.rank_margin, rank_gen,
                )
            elif args.objective == "list_mle":
                loss = cross_layer_list_mle(
                    teacher, student_score, causal, args.list_mle_candidates
                )
            else:  # hybrid
                loss = args.kl_weight * cross_layer_kl(teacher, student_score, causal) + (
                    1.0 - args.kl_weight
                ) * cross_layer_list_mle(
                    teacher, student_score, causal, args.list_mle_candidates
                )
            (loss / n_pairs).backward()
            step_loss += loss.item() / n_pairs
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        collector.clear()

        if is_main:
            logger.info("step %4d/%d  stream=%d  %s=%.5f",
                        step, args.steps, stream_i, loss_name, step_loss)

        if step % args.eval_every == 0 or step == args.steps:
            _evaluate(model, collector, students, pairs, streams, args, lo, topk, step, history, is_main)

    if is_main:
        out_path = Path(args.output or f"results/distill/indexer_lag_{args.model.replace('.', '')}.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "model": args.model, "ckpt_path": str(ckpt_path), "dataset": args.dataset,
                "seq": args.seq, "steps": args.steps, "lr": args.lr,
                "index_topk": index_topk, "topk": topk,
                "objective": args.objective,
                "rank_pairs": args.rank_pairs, "rank_margin": args.rank_margin,
                "list_mle_candidates": args.list_mle_candidates, "kl_weight": args.kl_weight,
                "grad_clip": args.grad_clip,
                "producers": producers, "pairs": pairs, "history": history,
            },
            "layers": {str(prod): students[prod].state_dict() for prod in producers},
        }
        torch.save(payload, out_path)
        logger.info("Saved distilled indexer weights for %d layers to %s", len(producers), out_path)
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(payload["meta"], indent=2))

    cleanup_distributed()


@torch.no_grad()
def _evaluate(model, collector, students, pairs, streams, args, lo, topk, step, history, is_main) -> None:
    """G1 proxy: does cross-layer top-k overlap rise vs the undistilled baseline?"""
    eval_idx = _broadcast_query_idx(
        args.eval_query_samples, lo, args.seq, torch.Generator().manual_seed(args.seed + 9999)
    )
    tokens = streams[-1][: args.seq].unsqueeze(0).to("cuda")
    _frozen_collect(model, collector, tokens, eval_idx)
    causal = causal_mask(eval_idx, args.seq)

    base_ov, dist_ov = [], []
    for prod, cons in pairs:
        if (
            cons not in collector.teacher_scores
            or prod not in collector.teacher_scores
            or prod not in collector.inputs
        ):
            continue
        teacher_cons = collector.teacher_scores[cons]
        teacher_prod = collector.teacher_scores[prod]
        base_ov.append(topk_overlap(teacher_prod, teacher_cons, topk, causal))
        x, qr, freqs_cis = (t.to("cuda") for t in collector.inputs[prod])
        student_score, _ = students[prod](x, qr, freqs_cis, query_idx=eval_idx, query_chunk=args.query_chunk)
        dist_ov.append(topk_overlap(student_score, teacher_cons, topk, causal))
    collector.clear()

    if is_main and base_ov:
        mb = sum(base_ov) / len(base_ov)
        md = sum(dist_ov) / len(dist_ov)
        logger.info("=== EVAL step %d: cross-layer top-k Jaccard  baseline=%.4f  distilled=%.4f  (delta=%+.4f) ===",
                    step, mb, md, md - mb)
        history.append({"step": step, "baseline_overlap": mb, "distilled_overlap": md})


if __name__ == "__main__":
    main()
