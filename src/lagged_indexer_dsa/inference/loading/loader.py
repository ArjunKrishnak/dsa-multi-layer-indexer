"""Top-level :func:`load_model` entry point.

Wires together :mod:`config` (config + dtype resolution),
:mod:`checkpoint` (format detection + tensor loading + tokenizer
loading), and :class:`Transformer` instantiation.

Two public entry points:

* :func:`load_model` — high-level, name-driven. Resolves the checkpoint
  path from ``.env`` via :func:`lagged_indexer_dsa.inference.loading.paths.ckpt_path_for` and
  builds the :class:`ModelConfig` internally. Production scripts use this.
* :func:`load_model_from_config` — low-level, takes a pre-built
  :class:`ModelConfig`. Tests that exercise the loader's edge cases use
  this directly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from safetensors.torch import load_model as load_safetensors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lagged_indexer_dsa.inference.modeling.transformer import Transformer

from .checkpoint import (
    _detect_ckpt_format,
    _load_from_hf_shards,
    _load_tokenizer,
)
from .config import DEFAULT_MAX_SEQ_LEN, ModelConfig, load_model_args
from .paths import ckpt_path_for


logger = logging.getLogger(__name__)


def load_model(
    name: str,
    *,
    ckpt_path: str | Path | None = None,
    config_path: str | Path | None = None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    dtype: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    trust_remote_code: bool = False,
) -> tuple[Transformer, PreTrainedTokenizerFast | AutoTokenizer, Path]:
    """Load a model + tokenizer by canonical name.

    Resolves the on-disk checkpoint path from ``.env`` via
    :func:`lagged_indexer_dsa.inference.loading.paths.ckpt_path_for`. Pass ``ckpt_path``
    to bypass the env-driven lookup (e.g. tests, ad-hoc invocations).

    Args:
        name: Canonical model name; one of
            :data:`lagged_indexer_dsa.inference.loading.paths.KNOWN_MODELS`.
        ckpt_path: Optional explicit checkpoint path. If ``None``, the
            path is resolved from ``.env``.
        config_path: Optional model-config JSON override.
        max_seq_len: Prompt-truncation length passed through to
            :class:`ModelConfig`.
        dtype: Compute dtype override (``"bf16"`` / ``"fp8"`` / ...).
            ``None`` defers to the checkpoint's native dtype.
        rank: Process rank for distributed loading.
        world_size: World size for distributed loading.
        trust_remote_code: Opt-in to executing checkpoint-provided
            tokenizer code (``AutoTokenizer`` path). Off by default;
            enable only for checkpoints you trust.

    Returns:
        ``(model, tokenizer, resolved_ckpt_path)``. The third element is
        the absolute path actually loaded; record it in run provenance so
        the metadata reflects what was on disk, not what the caller passed.
    """
    resolved_ckpt = (
        Path(ckpt_path) if ckpt_path else ckpt_path_for(name)
    ).expanduser().resolve()
    config = ModelConfig(
        ckpt_path=str(resolved_ckpt),
        config_path=str(config_path) if config_path else None,
        max_seq_len=max_seq_len,
        dtype=dtype,
    )
    model, tokenizer = load_model_from_config(
        config,
        rank=rank,
        world_size=world_size,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer, resolved_ckpt


def load_tokenizer(
    name: str,
    *,
    ckpt_path: str | Path | None = None,
    trust_remote_code: bool = False,
) -> tuple[PreTrainedTokenizerFast | AutoTokenizer, Path]:
    """Load *just* the tokenizer for a canonical model name.

    Mirrors :func:`load_model`'s ``name`` / ``ckpt_path`` resolution but
    skips weight loading entirely. Use from CPU-only entry points such
    as data-generation scripts that need the tokenizer family but must
    not pull weights onto a CPU node.

    ``trust_remote_code`` is off by default; enable it only for
    checkpoints you trust that ship custom tokenizer code.

    Returns:
        ``(tokenizer, resolved_ckpt_path)``. Record ``resolved_ckpt_path``
        in run provenance so the metadata reflects the on-disk tokenizer
        source.
    """
    resolved_ckpt = (
        Path(ckpt_path) if ckpt_path else ckpt_path_for(name)
    ).expanduser().resolve()
    tokenizer = _load_tokenizer(
        str(resolved_ckpt), trust_remote_code=trust_remote_code
    )
    return tokenizer, resolved_ckpt


def load_model_from_config(
    config: ModelConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    trust_remote_code: bool = False,
) -> tuple[Transformer, PreTrainedTokenizerFast | AutoTokenizer]:
    """Load a DeepSeek model and tokenizer from a checkpoint.

    Supports two checkpoint formats:

    - **Reference format**: a single
      ``model{rank}-mp{world_size}.safetensors``
    - **HuggingFace format**: sharded safetensors with
      ``model.safetensors.index.json``

    The format is auto-detected. Reference format is preferred when
    both exist.

    For multi-GPU inference, the caller must initialise
    ``torch.distributed`` and call ``torch.cuda.set_device(local_rank)``
    **before** calling this function. The :class:`Transformer`
    constructor reads ``dist.get_world_size()`` / ``dist.get_rank()``
    internally.

    Args:
        config: Model configuration specifying checkpoint path and
            overrides.
        rank: Process rank for distributed loading (determines weight
            shard).
        world_size: Total number of processes for distributed loading.
        trust_remote_code: Opt-in to executing checkpoint-provided
            tokenizer code; off by default.

    Returns:
        A (model, tokenizer) tuple.
    """
    if not config.ckpt_path:
        raise ValueError("ckpt_path must be set to a checkpoint directory")

    # ``Transformer`` derives its own rank/world from ``torch.distributed``.
    # If a caller passes values that disagree with the initialised process
    # group, shard selection here would silently diverge from the model's
    # internal sharding, so fail fast instead.
    dist_active = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if dist_active:
        dist_rank = torch.distributed.get_rank()
        dist_world = torch.distributed.get_world_size()
        if (rank, world_size) != (dist_rank, dist_world):
            raise ValueError(
                f"rank/world_size ({rank}/{world_size}) disagree with the "
                f"initialised torch.distributed state ({dist_rank}/{dist_world}); "
                "pass matching values so loader sharding aligns with the "
                "Transformer's internal rank/world."
            )
    elif world_size > 1:
        raise ValueError(
            f"world_size={world_size} requires an initialised torch.distributed "
            "process group, but none is active; Transformer would fall back to "
            "rank 0 / world 1 and load mismatched shards. Initialise the process "
            "group (and torch.cuda.set_device) before calling."
        )

    ckpt_path = Path(config.ckpt_path)
    fmt = _detect_ckpt_format(ckpt_path, rank, world_size)
    logger.info(
        "Detected checkpoint format: %s (rank=%d, world_size=%d)",
        fmt, rank, world_size,
    )

    args = load_model_args(config)
    logger.info(
        "Instantiating model: layers=%d, dim=%d, dtype=%s, max_seq_len=%d",
        args.n_layers,
        args.dim,
        args.dtype,
        args.max_seq_len,
    )

    t_init = time.perf_counter()
    # FP8 inference requires bf16 to remain the default dtype because the
    # upstream ``fp8_gemm`` kernel allocates its output via
    # ``a.new_empty(..., dtype=torch.get_default_dtype())`` and hard fails
    # on fp32. We therefore set bf16 as the default for the whole
    # construction-and-load block and only *keep* it on success and only
    # for FP8. Any failure (including in weight or tokenizer loading after
    # ``Transformer(args)`` has already returned) restores the previous
    # default so process-global state is not leaked into subsequent
    # tests / callers.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    keep_bf16_default = False
    try:
        with torch.device("cuda"):
            model = Transformer(args)
        init_elapsed = time.perf_counter() - t_init
        logger.info("Model instantiated in %.1fs", init_elapsed)

        t_load = time.perf_counter()

        if fmt == "reference":
            weights_path = ckpt_path / f"model{rank}-mp{world_size}.safetensors"
            logger.info("Loading weights from %s (reference format)", weights_path)
            load_safetensors(model, str(weights_path))
            load_elapsed = time.perf_counter() - t_load
            logger.info(
                "Weights loaded in %.1fs (reference, single file)", load_elapsed
            )

        elif fmt == "hf":
            logger.info(
                "Loading weights from %s (HuggingFace sharded format)",
                ckpt_path,
            )
            stats = _load_from_hf_shards(
                model,
                ckpt_path,
                rank=rank,
                world_size=world_size,
                n_experts=args.n_routed_experts,
                n_layers=args.n_layers,
            )
            logger.info(
                "Weights loaded in %.1fs (%d shards, %d tensors, %.1f GB)",
                stats["elapsed_s"],
                stats["n_shards"],
                stats["n_tensors"],
                stats["gb_loaded"],
            )

        else:
            logger.error(
                "Unknown checkpoint format %r returned by _detect_ckpt_format "
                "for ckpt_path=%s; weights cannot be loaded into model. "
                "Update _load_from_hf_shards / load_safetensors dispatch in "
                "loader.py to handle the new format.",
                fmt,
                ckpt_path,
            )
            raise RuntimeError(
                f"unhandled checkpoint format {fmt!r} for ckpt_path={ckpt_path}; "
                "expected 'reference' or 'hf'"
            )
        total_elapsed = time.perf_counter() - t_init
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "Model ready: %.2fB parameters, total startup %.1fs "
            "(init %.1fs + load %.1fs)",
            param_count / 1e9,
            total_elapsed,
            init_elapsed,
            total_elapsed - init_elapsed,
        )

        tokenizer = _load_tokenizer(
            config.resolve_tokenizer_path(),
            trust_remote_code=trust_remote_code,
        )
        logger.info("Tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)
        keep_bf16_default = args.dtype == "fp8"
    finally:
        if not keep_bf16_default:
            torch.set_default_dtype(prev_dtype)

    if keep_bf16_default:
        logger.info(
            "FP8 model: keeping torch default dtype set to bfloat16 "
            "globally (required by fp8_gemm output-buffer allocation)"
        )

    return model, tokenizer
