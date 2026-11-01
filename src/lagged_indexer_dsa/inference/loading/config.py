"""ModelConfig + HF→reference config translation + native-dtype detection.

Three concerns kept together because each one feeds the next:

1. :class:`ModelConfig` — what the *user* asks for (checkpoint path,
   tokenizer override, max sequence length, optional dtype).
2. :func:`hf_config_to_model_args` — translation of an HF ``config.json``
   into the kwargs that :class:`ModelArgs` expects.
3. :func:`_detect_native_checkpoint_dtype` — inspect the safetensors
   index for FP8 ``weight_scale_inv`` keys; the result overrides the
   config-declared dtype because some HF configs (DeepSeek-V3.2)
   declare ``bfloat16`` even though the weights ship in FP8.
4. :func:`load_model_args` — final assembly into a :class:`ModelArgs`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from importlib import resources
try:
    from importlib.resources.abc import Traversable
except ModuleNotFoundError:  # Python < 3.11
    from importlib.abc import Traversable
from pathlib import Path

from safetensors import safe_open

from lagged_indexer_dsa.inference.modeling.transformer import ModelArgs


logger = logging.getLogger(__name__)


_DEFAULT_CONFIG = "config_671B_v3.2.json"
_MINI_CONFIG = "config_mini.json"


#: Default truncation length for documents fed into the indexer-aware
#: forward passes. :class:`ModelConfig` and :func:`load_model` default to
#: this value; pass ``max_seq_len`` explicitly to override it.
DEFAULT_MAX_SEQ_LEN: int = 16384


_HF_FIELD_MAP: dict[str, str] = {
    "hidden_size": "dim",
    "intermediate_size": "inter_dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "first_k_dense_replace": "n_dense_layers",
    "num_attention_heads": "n_heads",
    "num_experts_per_tok": "n_activated_experts",
    "n_group": "n_expert_groups",
    "topk_group": "n_limited_groups",
    "scoring_func": "score_func",
    "routed_scaling_factor": "route_scale",
}

_HF_DTYPE_MAP: dict[str, str] = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "fp8": "fp8",
}

_MODEL_ARGS_FIELDS: set[str] = {f.name for f in fields(ModelArgs)}


def _bundled_config_path() -> Traversable:
    return resources.files("lagged_indexer_dsa.inference") / "configs" / _DEFAULT_CONFIG


def hf_config_to_model_args(hf_config: dict) -> dict:
    """Translate an HF-style ``config.json`` dict into ``ModelArgs`` kwargs.

    Fields that already have the same name in both formats (e.g.
    ``q_lora_rank``, ``kv_lora_rank``, ``vocab_size``) are passed
    through unchanged. Unknown HF fields are silently ignored.
    """
    result: dict = {}

    for hf_name, ref_name in _HF_FIELD_MAP.items():
        if hf_name in hf_config:
            result[ref_name] = hf_config[hf_name]

    for passthrough in (
        "n_routed_experts", "n_shared_experts", "q_lora_rank",
        "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "index_n_heads", "index_head_dim", "index_topk", "vocab_size",
        "indexer_rope_interleave", "gate_bias", "scale_fmt",
    ):
        if passthrough in hf_config:
            result[passthrough] = hf_config[passthrough]

    rope_params = hf_config.get("rope_parameters", {})
    if "rope_theta" in rope_params:
        result["rope_theta"] = rope_params["rope_theta"]
    if "rope_type" in rope_params:
        result["rope_type"] = rope_params["rope_type"]

    # HuggingFace configs vary on the field name; older models ship
    # ``torch_dtype`` and newer models ship ``dtype``. Reading both with
    # ``dtype`` taking precedence keeps the translator robust to the
    # exact era of the upstream config without silently defaulting to
    # whatever ModelArgs decides.
    raw_dtype = hf_config.get("dtype") or hf_config.get("torch_dtype")
    if raw_dtype in _HF_DTYPE_MAP:
        result["dtype"] = _HF_DTYPE_MAP[raw_dtype]
    elif raw_dtype:
        result["dtype"] = raw_dtype

    return {k: v for k, v in result.items() if k in _MODEL_ARGS_FIELDS}


@dataclass
class ModelConfig:
    """Configuration for loading a model with sparse attention."""

    ckpt_path: str = ""
    config_path: str | None = None
    tokenizer_path: str | None = None
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    dtype: str | None = None

    def resolve_config_path(self) -> Path | Traversable | None:
        """Return explicit config path, or ``None`` to trigger auto-detection."""
        if self.config_path is not None:
            return Path(self.config_path)
        return None

    def resolve_tokenizer_path(self) -> str:
        """Return the tokenizer path, falling back to ckpt_path."""
        if self.tokenizer_path is not None:
            return self.tokenizer_path
        return self.ckpt_path


def _load_safetensors_weight_map(index_path: Path) -> dict[str, str]:
    """Read and structurally validate a safetensors ``model.safetensors.index.json``.

    Used by both :func:`_detect_native_checkpoint_dtype` (to decide
    bf16-vs-fp8 from key suffixes) and the HF-shard loader (to drive
    per-shard tensor enumeration). Centralising the validation here
    means corrupt indices surface the same self-describing error
    regardless of which entry point first opens the file.

    Raises:
        ValueError: If the file does not parse as a JSON object or
            lacks the required ``"weight_map"`` key.
        TypeError: If ``"weight_map"`` is present but not a dict.
    """
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict) or "weight_map" not in index:
        raise ValueError(
            f"safetensors index at {index_path} is missing the required "
            "'weight_map' key (or is not a JSON object); the index file "
            "is corrupt or non-standard"
        )
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict):
        raise TypeError(
            f"safetensors index at {index_path} has a non-dict "
            f"'weight_map' (got {type(weight_map).__name__}); the index "
            "file is corrupt or non-standard"
        )
    return weight_map


def _detect_native_checkpoint_dtype(ckpt_path: str) -> str | None:
    """Inspect the safetensors index for FP8 ``weight_scale_inv`` keys.

    Returns:
        ``"fp8"`` if the checkpoint stores any FP8-quantised weights,
        ``"bf16"`` if a safetensors index (or single-shard) exists but
        contains no scale keys, ``None`` if neither file is present
        (caller falls back to whatever dtype the config specifies).

    Raises:
        Underlying ``json.JSONDecodeError``, ``OSError``, or any
        safetensors error when an index or shard exists but cannot be
        parsed. Surrendering silently to the config-declared dtype on
        parse failures masks corrupt checkpoints behind a quiet bf16
        fallback.
    """
    if not ckpt_path:
        return None
    ckpt_dir = Path(ckpt_path)
    index_path = ckpt_dir / "model.safetensors.index.json"

    if index_path.exists():
        weight_map = _load_safetensors_weight_map(index_path)
        keys: list[str] = list(weight_map.keys())
    else:
        single_shard = ckpt_dir / "model.safetensors"
        if not single_shard.exists():
            return None
        with safe_open(str(single_shard), framework="pt") as sf:
            keys = list(sf.keys())

    has_scale = any(k.endswith("weight_scale_inv") for k in keys)
    return "fp8" if has_scale else "bf16"


def load_model_args(config: ModelConfig) -> ModelArgs:
    """Load ModelArgs from a config JSON, applying overrides from ModelConfig.

    Config resolution order:

    1. Explicit ``config.config_path`` if set (highest precedence;
       use this to point at a non-default config layout).
    2. HF ``config.json`` in ``config.ckpt_path`` (auto-translated).
    3. Bundled default config (``config_671B_v3.2.json``) -- *only*
       when ``config.ckpt_path`` is empty (the test-fixture path).
       A non-empty ``ckpt_path`` whose directory carries no
       ``config.json`` raises rather than silently loading
       DeepSeek's args against a different family's weights.

    Dtype resolution:

    1. Explicit ``config.dtype`` if set (highest precedence).
    2. Native checkpoint dtype detected from ``weight_scale_inv`` keys
       in the safetensors index. This catches the DeepSeek-V3.2 case
       where the HF ``config.json`` declares ``torch_dtype: "bfloat16"``
       even though the weight files are FP8-quantised; without this
       override, the model loads at bf16 (doubling parameter memory)
       and OOMs on 8x H200.
    3. Whatever dtype the resolved config JSON specifies.

    Raises:
        FileNotFoundError: When ``config.ckpt_path`` is set but the
            directory contains no ``config.json``. Pass
            ``config.config_path`` to point at an alternative layout
            or stage the HF config alongside the safetensors shards.
    """
    explicit_path = config.resolve_config_path()

    if explicit_path is not None:
        with explicit_path.open() as f:
            raw_json = json.load(f)
        # ``config.config_path`` is documented as a config-path override
        # but does not promise either a ``ModelArgs``-shaped or
        # HF-shaped JSON. Sniff for HF marker keys (``hidden_size`` /
        # ``num_hidden_layers``) and route through the HF translator;
        # otherwise filter unknown keys so we don't trip the dataclass
        # constructor with leftover fields. Skipping this branch made
        # an HF-style ``--config-path`` silently produce wrong
        # ``ModelArgs`` (hidden_size never reached ``dim``).
        if "hidden_size" in raw_json or "num_hidden_layers" in raw_json:
            raw = hf_config_to_model_args(raw_json)
        else:
            raw = {k: v for k, v in raw_json.items() if k in _MODEL_ARGS_FIELDS}
    elif config.ckpt_path:
        hf_config_path = Path(config.ckpt_path) / "config.json"
        if not hf_config_path.exists():
            raise FileNotFoundError(
                f"No HF config.json found at {hf_config_path}; refusing "
                "to silently fall back to the bundled DeepSeek-V3.2 "
                "config. Stage the model's HF config alongside the "
                "weights, pass --config-path explicitly, or call "
                "load_model_args with an empty ModelConfig() to load "
                "the bundled config (test fixtures only).",
            )
        logger.info("Auto-detected HF config at %s", hf_config_path)
        with hf_config_path.open() as f:
            hf_raw = json.load(f)
        raw = hf_config_to_model_args(hf_raw)
    else:
        with _bundled_config_path().open() as f:
            raw = json.load(f)
        # Bundled config already uses ``ModelArgs`` field names; we
        # only filter to drop any unknown keys without translating.
        raw = {k: v for k, v in raw.items() if k in _MODEL_ARGS_FIELDS}

    raw["max_seq_len"] = config.max_seq_len
    if config.dtype is not None:
        raw["dtype"] = config.dtype
    else:
        native_dtype = _detect_native_checkpoint_dtype(config.ckpt_path)
        if native_dtype is not None:
            previous = raw.get("dtype")
            if previous != native_dtype:
                logger.info(
                    "Overriding config dtype %r with checkpoint-native dtype "
                    "%r (detected via weight_scale_inv keys)",
                    previous,
                    native_dtype,
                )
            raw["dtype"] = native_dtype
    return ModelArgs(**raw)
