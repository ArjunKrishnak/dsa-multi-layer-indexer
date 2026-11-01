"""End-to-end GLM / DeepSeek inference recipe (checkpoint loading + forward).

Loads released DeepSeek-V3.2-Exp, GLM-5, and GLM-5.1 checkpoints (HF-sharded
or reference safetensors, FP8/bf16 auto-detected) into the reference
:class:`~lagged_indexer_dsa.inference.modeling.transformer.Transformer` and
runs the sparse-attention forward. Adapted from the internal ``sparse_routing``
project; the modeling code is GPU-only (the FP8 Indexer requires TileLang/CUDA),
while the loader/config layer imports and resolves on CPU.

Re-exports the production loader (:func:`load_model`,
:func:`load_model_from_config`, :class:`ModelConfig`) from
:mod:`lagged_indexer_dsa.inference.loading` plus the random-weight builder
(:func:`create_mini_model`) from :mod:`lagged_indexer_dsa.inference.modeling.mini`. The
internal helpers (``_assert_fp8_scales_loaded``, ``_HF_KEY_REPLACEMENTS``,
``hf_config_to_model_args``, …) are also re-exported because the
loading-pipeline regression tests pin them by their package-level
import path.
"""

from __future__ import annotations

from lagged_indexer_dsa.inference.modeling.mini import create_mini_model
from lagged_indexer_dsa.inference.loading.checkpoint import (
    _assert_fp8_scales_loaded,
    _detect_ckpt_format,
    _load_from_hf_shards,
    _load_tokenizer,
    _needs_deepseek_tokenizer_workaround,
)
from lagged_indexer_dsa.inference.loading.config import (
    _DEFAULT_CONFIG,
    _HF_DTYPE_MAP,
    _HF_FIELD_MAP,
    _MINI_CONFIG,
    _MODEL_ARGS_FIELDS,
    ModelConfig,
    _bundled_config_path,
    _detect_native_checkpoint_dtype,
    hf_config_to_model_args,
    load_model_args,
)
from lagged_indexer_dsa.inference.loading.keys import (
    _HF_KEY_REPLACEMENTS,
    _HF_SHARD_DIM,
    _TOPLEVEL_RENAMES,
    _get_shard_dim,
    _remap_hf_key,
)
from lagged_indexer_dsa.inference.loading.loader import (
    load_model,
    load_model_from_config,
    load_tokenizer,
)
from lagged_indexer_dsa.inference.loading.sharding import _is_foreign_expert, _shard_tensor


__all__ = [
    "_DEFAULT_CONFIG",
    "_HF_DTYPE_MAP",
    "_HF_FIELD_MAP",
    "_HF_KEY_REPLACEMENTS",
    "_HF_SHARD_DIM",
    "_MINI_CONFIG",
    "_MODEL_ARGS_FIELDS",
    "_TOPLEVEL_RENAMES",
    "ModelConfig",
    "_assert_fp8_scales_loaded",
    "_bundled_config_path",
    "_detect_ckpt_format",
    "_detect_native_checkpoint_dtype",
    "_get_shard_dim",
    "_is_foreign_expert",
    "_load_from_hf_shards",
    "_load_tokenizer",
    "_needs_deepseek_tokenizer_workaround",
    "_remap_hf_key",
    "_shard_tensor",
    "create_mini_model",
    "hf_config_to_model_args",
    "load_model",
    "load_model_args",
    "load_model_from_config",
    "load_tokenizer",
]
