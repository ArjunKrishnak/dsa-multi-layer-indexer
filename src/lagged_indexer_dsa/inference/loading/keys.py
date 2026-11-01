"""HuggingFace ↔ reference tensor-name remapping and shard-dim lookup.

Every key that crosses the boundary between an HF-format checkpoint
(``model.safetensors`` shards laid out following the HF naming
convention) and our reference Transformer goes through
:func:`_remap_hf_key`. Tensor-parallel sharding then dispatches via
:func:`_get_shard_dim` based on the HF component name.

These tables are intentionally separate from the rest of the loader:
they are pure metadata that any future checkpoint family
(GLM-5, GLM-5.1, DeepSeek-V4-Pro) can extend without touching the
loader logic in :mod:`checkpoint`.
"""

from __future__ import annotations

import re


_HF_KEY_REPLACEMENTS: list[tuple[str, str]] = [
    ("self_attn", "attn"),
    ("mlp", "ffn"),
    ("q_a_proj", "wq_a"),
    ("q_b_proj", "wq_b"),
    ("kv_a_proj_with_mqa", "wkv_a"),
    ("kv_b_proj", "wkv_b"),
    ("o_proj", "wo"),
    ("q_a_layernorm", "q_norm"),
    ("kv_a_layernorm", "kv_norm"),
    ("input_layernorm", "attn_norm"),
    ("post_attention_layernorm", "ffn_norm"),
    ("gate_proj", "w1"),
    ("up_proj", "w3"),
    ("down_proj", "w2"),
    ("weight_scale_inv", "scale"),
    ("e_score_correction_bias", "bias"),
    ("embed_tokens", "embed"),
]

_TOPLEVEL_RENAMES: dict[str, str] = {
    "lm_head": "head",
}

# Tensor-parallel shard dimension for each HF component name.
# dim=0 → column-parallel (split output), dim=1 → row-parallel (split input).
# None → replicated (every rank gets the full tensor).
_HF_SHARD_DIM: dict[str, int | None] = {
    "embed_tokens": 0,
    "q_proj": 0,
    "q_a_proj": None,
    "q_a_layernorm": None,
    "q_b_proj": 0,
    "kv_a_proj_with_mqa": None,
    "kv_a_layernorm": None,
    "kv_b_proj": 0,
    "o_proj": 1,
    "gate": None,
    "gate_proj": 0,
    "down_proj": 1,
    "up_proj": 0,
    "input_layernorm": None,
    "post_attention_layernorm": None,
    "norm": None,
    "lm_head": 0,
    "scale": None,
    "weight_scale_inv": None,
    # Indexer sub-modules (replicated)
    "wq_b": None,
    "wk": None,
    "k_norm": None,
    "weights_proj": None,
}


def _remap_hf_key(
    hf_key: str, *, skip_layer: int | list[int] | None,
) -> str | None:
    """Map an HF-format tensor name to the reference-format name.

    Returns ``None`` for keys whose ``layers.<idx>`` index appears in
    *skip_layer*. The current production checkpoints (DeepSeek-V3.2,
    GLM-5, GLM-5.1) each carry exactly one auxiliary head layer (the
    NTP / MTP block) at ``idx == n_layers`` and pass an ``int``; the
    broader signature accommodates future checkpoints that strip
    multiple auxiliary blocks (pass a list) or none at all (pass
    ``None``).
    """
    if skip_layer is None:
        skip_set: frozenset[int] = frozenset()
    elif isinstance(skip_layer, int):
        skip_set = frozenset({skip_layer})
    else:
        skip_set = frozenset(skip_layer)

    key = hf_key
    if key.startswith("model."):
        key = key[len("model."):]

    match = re.match(r"layers\.(\d+)\.", key)
    if match and int(match.group(1)) in skip_set:
        return None

    for hf_part, ref_part in _HF_KEY_REPLACEMENTS:
        key = key.replace(hf_part, ref_part)

    for hf_name, ref_name in _TOPLEVEL_RENAMES.items():
        if key.startswith(hf_name):
            key = ref_name + key[len(hf_name):]
            break

    return key


def _get_shard_dim(hf_key: str) -> int | None:
    """Return the tensor-parallel shard dimension for an HF key.

    Scale tensors (``weight_scale_inv``) inherit the shard dimension
    from their parent weight's component. For expert parameters, the
    tensor itself is *not* sharded — experts are partitioned by index
    instead. Returns ``None`` when the tensor should be replicated
    across all ranks.
    """
    if "experts." in hf_key and "shared_experts" not in hf_key:
        return None

    stripped = hf_key.replace("model.", "")

    # Scale tensors inherit shard dim from their parent component.
    is_scale = stripped.endswith(".weight_scale_inv")
    if is_scale:
        stripped = stripped.rsplit(".weight_scale_inv", 1)[0]

    parts = stripped.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in _HF_SHARD_DIM and parts[i] not in ("scale", "weight_scale_inv"):
            return _HF_SHARD_DIM[parts[i]]
    return None
