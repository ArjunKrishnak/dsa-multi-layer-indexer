"""Checkpoint format detection + HF-shard loader + tokenizer loader.

Three concerns kept together because they form a single linear pipeline
inside :func:`load_model`:

1. :func:`_detect_ckpt_format` — pick between the reference single-
   shard format and the HF sharded format.
2. :func:`_load_from_hf_shards` — copy tensors out of the HF shards
   into a freshly-instantiated :class:`Transformer`, applying
   :func:`_remap_hf_key` and :func:`_shard_tensor` along the way.
3. :func:`_assert_fp8_scales_loaded` — defense-in-depth check that
   every FP8 ``Linear.scale`` parameter received a checkpoint value
   (the GLM-5/5.1 zero-tensor regression that motivated the assertion).
4. :func:`_load_tokenizer` — pick between :class:`AutoTokenizer` and
   the direct :class:`PreTrainedTokenizerFast` workaround for
   DeepSeek's mis-declared ``LlamaTokenizerFast`` config.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lagged_indexer_dsa.inference.modeling.transformer import Transformer

from .config import _load_safetensors_weight_map
from .keys import _remap_hf_key
from .sharding import _is_foreign_expert, _shard_tensor


logger = logging.getLogger(__name__)


def _detect_ckpt_format(ckpt_path: Path, rank: int, world_size: int) -> str:
    """Detect checkpoint format: 'hf' (sharded HF) or 'reference' (single shard).

    Raises FileNotFoundError if neither format is found.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    hf_index = ckpt_path / "model.safetensors.index.json"
    ref_weights = [
        ckpt_path / f"model{i}-mp{world_size}.safetensors"
        for i in range(world_size)
    ]
    have_all_ref = all(p.exists() for p in ref_weights)
    have_any_ref = any(p.exists() for p in ref_weights)

    if have_all_ref:
        return "reference"
    if have_any_ref:
        missing = [p.name for p in ref_weights if not p.exists()]
        raise FileNotFoundError(
            f"Incomplete reference checkpoint set at {ckpt_path}; missing "
            f"{missing}. Refusing to mix reference and HuggingFace formats."
        )
    if hf_index.exists():
        return "hf"

    raise FileNotFoundError(
        f"No weights found at {ckpt_path}. Expected either "
        f"'{ref_weights[rank].name}' (reference format) or "
        f"'model.safetensors.index.json' (HuggingFace format)."
    )


def _assert_fp8_scales_loaded(
    model_state: dict[str, torch.Tensor],
    matched_params: set[str],
) -> None:
    """Assert that every FP8 ``Linear.scale`` parameter received a checkpoint value.

    The ``Linear`` module only allocates a ``.scale`` parameter when its
    weight is stored in FP8 (1-byte element size). If a ``.scale``
    parameter exists in the model but is missing from the loaded
    checkpoint, it stays at the (uninitialised) value produced by
    ``torch.empty`` — empirically zero — and ``fp8_gemm`` silently
    multiplies activations by zero, yielding all-zero outputs. The
    canonical trigger is forcing ``dtype="fp8"`` on a checkpoint whose
    indexer weights ship in bf16 (no ``weight_scale_inv`` keys), which
    leaves every ``Linear.scale`` uninitialised; the recommended
    workflow is to let :func:`load_model` auto-detect the native dtype
    via :func:`_detect_native_checkpoint_dtype` rather than overriding.
    """
    expected_scales = {k for k in model_state if k.endswith(".scale")}
    missing_scales = expected_scales - matched_params
    if not missing_scales:
        return

    n_total = len(expected_scales)
    n_missing = len(missing_scales)
    sample = sorted(missing_scales)[:5]
    raise RuntimeError(
        f"FP8 scale parameters present in the model but missing from the "
        f"checkpoint ({n_missing}/{n_total} unmatched, e.g. {sample}). "
        f"This typically indicates a dtype mismatch: a bf16-native "
        f"checkpoint was loaded as fp8, leaving Linear.scale uninitialised "
        f"(effectively zero), which makes fp8_gemm produce all-zero "
        f"outputs. Set the model dtype to match the checkpoint's native "
        f"precision (look for 'weight_scale_inv' keys in "
        f"model.safetensors.index.json: present => fp8, absent => bf16) or "
        f"call ``load_model`` without an explicit ``dtype`` so "
        f"``_detect_native_checkpoint_dtype`` picks the right precision "
        f"from the index file."
    )


def _load_from_hf_shards(
    model: Transformer,
    ckpt_path: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    n_experts: int = 256,
    n_layers: int = 61,
) -> dict[str, int | float]:
    """Load weights from HuggingFace sharded safetensors into the model.

    When ``world_size > 1``, tensors are sharded on-the-fly following
    the same tensor-parallel scheme as DeepSeek's reference
    ``convert.py``: column-parallel params split on dim 0, row-parallel
    on dim 1, and routed experts are partitioned by index.

    Returns a dict with loading statistics.
    """
    index_path = ckpt_path / "model.safetensors.index.json"
    weight_map: dict[str, str] = _load_safetensors_weight_map(index_path)

    shards_to_keys: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_file in weight_map.items():
        shards_to_keys[shard_file].append(tensor_name)

    model_state = model.state_dict()
    matched_params: set[str] = set()
    skipped_keys: list[str] = []
    n_tensors_loaded = 0
    total_bytes = 0

    sorted_shards = sorted(shards_to_keys.keys())
    n_shards = len(sorted_shards)

    t_start = time.perf_counter()

    for shard_idx, shard_file in enumerate(sorted_shards, 1):
        shard_rel = Path(shard_file)
        if shard_rel.is_absolute() or ".." in shard_rel.parts:
            raise ValueError(
                f"Invalid shard path {shard_file!r} in {index_path}; "
                "expected a relative path inside the checkpoint directory"
            )
        shard_path = str(ckpt_path / shard_rel)
        t_shard = time.perf_counter()

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_key in shards_to_keys[shard_file]:
                ref_key = _remap_hf_key(hf_key, skip_layer=n_layers)
                if ref_key is None:
                    skipped_keys.append(hf_key)
                    continue

                if world_size > 1 and _is_foreign_expert(
                    hf_key, rank, world_size, n_experts
                ):
                    skipped_keys.append(hf_key)
                    continue

                if ref_key not in model_state:
                    raise KeyError(
                        f"Remapped key {hf_key!r} -> {ref_key!r} is not "
                        f"present in the model state_dict; the HF-key "
                        f"remapping table is out of sync with the model "
                        f"architecture."
                    )

                tensor = f.get_tensor(hf_key)
                total_bytes += tensor.numel() * tensor.element_size()

                tensor = _shard_tensor(tensor, hf_key, rank, world_size, n_experts)
                if tensor is None:
                    skipped_keys.append(hf_key)
                    continue

                param = model_state[ref_key]
                if tensor.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for {ref_key}: "
                        f"sharded checkpoint tensor has {tensor.shape}, "
                        f"model expects {param.shape} "
                        f"(rank={rank}, world_size={world_size})"
                    )

                with torch.no_grad():
                    param.copy_(tensor)

                matched_params.add(ref_key)
                n_tensors_loaded += 1

        shard_elapsed = time.perf_counter() - t_shard
        if shard_idx % 20 == 0 or shard_idx == n_shards:
            logger.info(
                "Shard %d/%d loaded (%.1fs, cumulative %.1f GB)",
                shard_idx,
                n_shards,
                shard_elapsed,
                total_bytes / 1e9,
            )

    elapsed = time.perf_counter() - t_start
    gb_loaded = total_bytes / 1e9

    unmatched = set(model_state.keys()) - matched_params
    unmatched_non_buffer = {
        k for k in unmatched
        if not any(k.endswith(s) for s in (".freqs_cis",))
    }
    if unmatched_non_buffer:
        sample = sorted(unmatched_non_buffer)[:10]
        raise RuntimeError(
            f"{len(unmatched_non_buffer)} model parameters received no "
            f"checkpoint values (e.g. {sample}). Loading would proceed "
            f"with these parameters at their random init values; refuse "
            f"to do so silently. Investigate the HF-key remap or the "
            f"shard manifest before retrying."
        )

    _assert_fp8_scales_loaded(model_state, matched_params)

    if skipped_keys:
        logger.debug("Skipped %d HF keys (NTP layer or unmapped)", len(skipped_keys))

    return {
        "n_shards": n_shards,
        "n_tensors": n_tensors_loaded,
        "elapsed_s": elapsed,
        "gb_loaded": gb_loaded,
    }


def _needs_deepseek_tokenizer_workaround(ckpt_path: Path) -> bool:
    """Check if the checkpoint's tokenizer_config declares LlamaTokenizerFast.

    DeepSeek's tokenizer_config.json incorrectly declares
    LlamaTokenizerFast, which overrides the ByteLevel pre-tokenizer
    with a Metaspace-based one and drops spaces. When this is the
    case, we bypass AutoTokenizer and load tokenizer.json directly.
    """
    config_path = ckpt_path / "tokenizer_config.json"
    if not config_path.exists():
        return False
    with config_path.open() as f:
        cfg = json.load(f)
    tokenizer_class = cfg.get("tokenizer_class", "")
    return "Llama" in tokenizer_class


def _load_tokenizer(
    ckpt_path: str,
    *,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerFast | AutoTokenizer:
    """Load the tokenizer from a checkpoint directory.

    Uses AutoTokenizer by default. Falls back to loading tokenizer.json
    directly for DeepSeek checkpoints whose tokenizer_config.json
    declares LlamaTokenizerFast (which drops spaces due to a
    pre-tokenizer mismatch).

    ``trust_remote_code`` is off by default; callers loading the trusted
    released checkpoints pass ``True`` explicitly.
    """
    ckpt = Path(ckpt_path)
    if _needs_deepseek_tokenizer_workaround(ckpt):
        tokenizer_file = ckpt / "tokenizer.json"
        logger.info(
            "Using direct tokenizer.json load (DeepSeek LlamaTokenizerFast workaround)"
        )
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            eos_token="<｜end▁of▁sentence｜>",  # noqa: RUF001 — fullwidth bar is the DeepSeek special-token glyph
            bos_token="<｜begin▁of▁sentence｜>",  # noqa: RUF001
        )
    return AutoTokenizer.from_pretrained(
        ckpt_path,
        trust_remote_code=trust_remote_code,
    )
