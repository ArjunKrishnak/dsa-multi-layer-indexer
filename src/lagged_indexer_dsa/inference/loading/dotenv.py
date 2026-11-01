"""Minimal POSIX-style ``.env`` loader.

Reads ``KEY=VALUE`` lines from ``.env`` at the repository root and
populates ``os.environ`` for keys that are not already set. Comments
(``#`` to end-of-line, but only outside quoted runs) and blank lines
are ignored. Single- and double-quoted values have their outer
quotes stripped; no other escapes are interpreted (no shell-style
command substitution, no ``\\n``, no variable interpolation, no
``export `` prefix).

The loader is opt-in: callers that need ``.env`` values invoke
``load_dotenv()`` explicitly. The function is idempotent and safe to
call from every entry point.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    # src/lagged_indexer_dsa/inference/loading/dotenv.py -> repo root is 4 up.
    return Path(__file__).resolve().parents[4]


def _strip_inline_comment(raw: str) -> str:
    """Drop ``#``-and-after, but only when ``#`` is outside a quoted run.

    A naive ``raw.split("#", 1)[0]`` corrupts values like
    ``TOKEN="abc#123"`` -- which is a realistic shape for API keys
    and password-style secrets. This walks the line character by
    character, tracking the active quote (if any), and returns
    everything before the first unquoted ``#``. Quote handling does
    not interpret escapes; the surrounding parser only strips outer
    quotes anyway.
    """
    quote: str | None = None
    for idx, ch in enumerate(raw):
        if quote is None:
            if ch in ("'", '"'):
                quote = ch
            elif ch == "#":
                return raw[:idx]
        elif ch == quote:
            quote = None
    return raw


def _parse_value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    return raw


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``.env`` into ``os.environ``.

    Args:
        path: Explicit path to the dotenv file. Defaults to
            ``<repo_root>/.env``.
        override: When ``True``, .env values overwrite an existing
            ``os.environ`` entry; otherwise existing entries win.

    Returns:
        Mapping of keys that the call assigned to ``os.environ``
        (does not include keys that were skipped because they were
        already set and ``override=False``).
    """
    target = path if path is not None else _repo_root() / ".env"
    if not target.is_file():
        return {}

    # Parse the whole file first and only mutate ``os.environ`` once it
    # has validated cleanly, so a malformed line late in the file cannot
    # leave a half-applied environment behind.
    pending: dict[str, str] = {}
    with target.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = _strip_inline_comment(raw_line).strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(
                    f"{target}:{lineno}: expected KEY=VALUE"
                )
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"{target}:{lineno}: empty key")
            parsed = _parse_value(value)
            if not override and (key in os.environ or key in pending):
                continue
            pending[key] = parsed

    assigned: dict[str, str] = {}
    for key, parsed in pending.items():
        os.environ[key] = parsed
        assigned[key] = parsed
    return assigned


__all__ = ["load_dotenv"]
