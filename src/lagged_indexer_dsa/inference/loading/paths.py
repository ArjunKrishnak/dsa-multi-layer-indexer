"""Single source of truth for model checkpoint paths.

Per-machine paths live in ``<repo_root>/.env`` (gitignored, copy from
``.env.example``). The first call to :func:`deepseek_ckpt_path` /
:func:`glm_ckpt_path` / :func:`glm51_ckpt_path` loads ``.env`` into
``os.environ``; an explicit ``export`` in the calling shell still wins.

Resolution policy (fail-fast, no fallback):
    1. Read the corresponding key from ``os.environ`` (populated from
       ``.env`` on first call).
    2. If unset or pointing at a non-existent directory, raise
       ``RuntimeError`` naming the missing key and the ``.env``
       location.

GLM-5 vs GLM-5.1 each carry their own env var
(``GLM_CKPT_PATH`` / ``GLM5_1_CKPT_PATH``) so a session that
evaluates both cannot accidentally load one weight set under the
other model name. There is no cross-fallback: ``--model glm5.1``
without a populated ``GLM5_1_CKPT_PATH`` raises rather than silently
loading the GLM-5 weights and writing a ``metadata.json`` that
labels them as GLM-5.1.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from lagged_indexer_dsa.inference.loading.dotenv import load_dotenv

DEEPSEEK_CKPT_ENV: str = "DEEPSEEK_CKPT_PATH"
GLM_CKPT_ENV: str = "GLM_CKPT_PATH"
GLM5_1_CKPT_ENV: str = "GLM5_1_CKPT_PATH"

_DOTENV_LOADED = False


def _ensure_dotenv() -> None:
    global _DOTENV_LOADED
    if not _DOTENV_LOADED:
        load_dotenv()
        _DOTENV_LOADED = True


def _resolve(env_name: str, *, must_exist: bool) -> Path:
    _ensure_dotenv()
    raw = os.environ.get(env_name)
    if not raw:
        raise RuntimeError(
            f"{env_name} is unset. Define it in <repo_root>/.env "
            f"(copy .env.example to .env and fill in the absolute path) "
            f"or export it in the calling shell."
        )
    path = Path(raw).expanduser().resolve()
    if must_exist and not path.is_dir():
        raise RuntimeError(
            f"{env_name}={raw!r} does not point to an existing directory. "
            f"Update <repo_root>/.env, or pre-stage the checkpoint at this path."
        )
    return path


def deepseek_ckpt_path(*, must_exist: bool = True) -> Path:
    """Return the absolute path to the DeepSeek-V3.2 checkpoint.

    Args:
        must_exist: If True (default), the resolved path must already be
            a directory. Use ``must_exist=False`` only in unit tests
            that mock the loader.

    Raises:
        RuntimeError: If the env var is unset or points to a missing
            directory and ``must_exist=True``.
    """
    return _resolve(DEEPSEEK_CKPT_ENV, must_exist=must_exist)


def glm_ckpt_path(*, must_exist: bool = True) -> Path:
    """Return the absolute path to the GLM-5 checkpoint."""
    return _resolve(GLM_CKPT_ENV, must_exist=must_exist)


def glm51_ckpt_path(*, must_exist: bool = True) -> Path:
    """Return the absolute path to the GLM-5.1 checkpoint.

    Reads ``GLM5_1_CKPT_PATH``. There is no fallback to
    ``GLM_CKPT_PATH``: a missing env var raises rather than silently
    loading GLM-5 weights under the GLM-5.1 model name.

    Raises:
        RuntimeError: If ``GLM5_1_CKPT_PATH`` is unset, or it points
            at a non-existent directory and ``must_exist=True``.
    """
    return _resolve(GLM5_1_CKPT_ENV, must_exist=must_exist)


#: Canonical model names accepted by :func:`ckpt_path_for` and
#: :func:`lagged_indexer_dsa.inference.loading.loader.load_model`. Each name maps
#: to a dedicated checkpoint env var with no cross-fallback (see the
#: per-resolver functions).
KNOWN_MODELS: tuple[str, ...] = ("deepseek", "glm5", "glm5.1")

#: Tokenizer-calibrated minimum ``--prefill-chunk-size`` for each model family.
#:
#: These values encode **corpus + tokenizer vocabulary knowledge**, not
#: hardware budget.  Hub scores are only produced by the *chunked*-prefill
#: path, which activates when ``seq_len > chunk_size``.  Each tokenizer maps
#: the same raw text to a different token count, so the safe floor varies per
#: family.  GLM-5/5.1 use a multilingual BPE vocabulary that is roughly 2x
#: larger than DeepSeek's, making the same document tokenize to ~40 % fewer
#: tokens.  The values below are calibrated so that the shortest document in
#: each evaluation corpus (GovReport, arXiv) is comfortably longer than the
#: chunk size, guaranteeing ≥2 chunks per document.
#:
#: SLURM scripts may **override upward** for throughput on faster hardware
#: (e.g. pass ``--prefill-chunk-size 4096`` on H200s to reduce the number
#: of forward passes per document).  Overriding *downward* would risk hitting
#: documents whose tokenized length is ≤ ``chunk_size``, which forces the
#: dense path and produces no hub scores — a hard ``RuntimeError``.
#:
#: ``prefill_chunk_size`` is an *execution parameter* with floating-point-
#: negligible effect on hub scores (within the FP8/BF16 matmul noise floor)
#: and is therefore **not** included in the resume-safety preflight key for
#: any evaluator script.  It is still stamped in ``metadata.json`` as
#: informational provenance.
#:
#: **When adding a new model family:**
#:
#: 1. Tokenize a representative sample of every evaluation dataset.
#: 2. Find the minimum tokenized length across that sample.
#: 3. Set ``chunk_size`` to at most half that minimum (ensures ≥2 chunks,
#:    giving a meaningful hub-score estimate even for short documents).
#: 4. Add the entry here *and* register the family in :data:`KNOWN_MODELS`.
PREFILL_CHUNK_SIZE_BY_MODEL: dict[str, int] = {
    # Calibrated against the shortest tokenised document across *all*
    # evaluation corpora (GovReport for token-NTP; LongBench single-doc
    # and multi-doc tasks for KV-cache eviction).  LongBench 2WikiMQA
    # has examples as short as ~967 DeepSeek tokens, so 512 is the safe
    # floor — well below the multi-doc minimum while still guaranteeing
    # ≥2 chunks for the vast majority of documents.
    "deepseek": 512,
    "glm5":      512,  # shortest govreport doc ≈ 1 600 tokens (GLM tokenizer)
    "glm5.1":    512,  # same tokenizer family as glm5
}


def ckpt_path_for(name: str, *, must_exist: bool = True) -> Path:
    """Return the checkpoint path for a canonical model name.

    Args:
        name: One of :data:`KNOWN_MODELS`.
        must_exist: Forwarded to the per-family resolver.

    Raises:
        ValueError: If ``name`` is not in :data:`KNOWN_MODELS`.
    """
    if name == "deepseek":
        return deepseek_ckpt_path(must_exist=must_exist)
    if name == "glm5":
        return glm_ckpt_path(must_exist=must_exist)
    if name == "glm5.1":
        return glm51_ckpt_path(must_exist=must_exist)
    raise ValueError(
        f"unknown model name {name!r}; expected one of {KNOWN_MODELS}"
    )


def model_path_tag(name: str) -> str:
    """Filesystem-safe tag for an artefact directory (``"glm5.1" -> "glm51"``).

    Mirrored in :file:`scripts/slurm/common.sh` via ``${MODEL//./}``.
    """
    if name not in KNOWN_MODELS:
        raise ValueError(
            f"unknown model name {name!r}; expected one of {KNOWN_MODELS}"
        )
    return name.replace(".", "")


def ckpt_path_suffix(ckpt_path: str | None) -> str:
    """Path-safe tag derived from ``--ckpt-path`` for default output dirs.

    Returns ``""`` when ``ckpt_path`` is ``None`` / empty (the
    eval-time path resolves later from ``.env``); otherwise returns
    ``"_<hash>"`` where ``<hash>`` is a short digest of the *full*
    resolved checkpoint path. Two runs with the same model but
    different explicit checkpoints (e.g. a fresh fine-tune vs the
    base release) would otherwise share an output directory and
    ``skip_if_complete`` could reuse stale per-example JSONL from
    the previous weights. Hashing the full path (rather than just the
    basename) keeps distinct directories that share a leaf name apart.

    """
    if not ckpt_path:
        return ""
    resolved = str(Path(ckpt_path).expanduser().resolve())
    tag = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return f"_{tag}" if tag else ""
