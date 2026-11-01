"""Production model-loading machinery.

The package is split by *concern*, not by file size:

* :mod:`keys` — HF ↔ reference tensor-name remapping and per-component
  shard-dim lookup. Pure metadata; the dispatcher in :mod:`checkpoint`
  consumes it without ever modifying it.
* :mod:`sharding` — tensor-parallel sharding utilities driven by the
  :mod:`keys` metadata.
* :mod:`config` — :class:`ModelConfig`, HF-config translation, native-
  dtype detection, and final :class:`ModelArgs` assembly.
* :mod:`checkpoint` — checkpoint format detection, HF-shard loader,
  FP8 ``Linear.scale`` defense-in-depth assertion, and tokenizer
  loading.
* :mod:`loader` — :func:`load_model` end-to-end entry point.

The public API is:

* :class:`ModelConfig` and :func:`load_model` — the high-level,
  name-driven entry points used by production scripts.
* :func:`load_model_from_config` — the low-level entry point that
  takes a pre-built :class:`ModelConfig`. Tests that exercise the
  loader's edge cases (config resolution, dtype detection, format
  dispatch) call into it directly to avoid round-tripping through
  ``.env`` lookup.
* :func:`load_tokenizer` — tokenizer-only entry point with the
  same ``name`` / ``ckpt_path`` resolution as :func:`load_model`.
  Use from CPU-only entry points (data generation, smoke tests)
  that need the tokenizer family but must not pull weights onto
  a CPU node.
* :func:`hf_config_to_model_args` and :func:`load_model_args` —
  helpers re-exported for callers that need to inspect or override
  config translation.

Mini-model creation for pre-flight / unit-test use lives in
:mod:`lagged_indexer_dsa.inference.modeling.mini` — it is not part of the production
loading path.
"""

from __future__ import annotations

from .config import (
    ModelConfig,
    hf_config_to_model_args,
    load_model_args,
)
from .loader import load_model, load_model_from_config, load_tokenizer

__all__ = [
    "ModelConfig",
    "hf_config_to_model_args",
    "load_model",
    "load_model_args",
    "load_model_from_config",
    "load_tokenizer",
]
