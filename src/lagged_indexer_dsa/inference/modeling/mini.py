"""Mini Transformer builder for pre-flight / unit-test use.

Constructs a :class:`Transformer` that uses the same code paths as the
full 671B model (including the FP8 Indexer) but with drastically
reduced dimensions so it fits on a single GPU and initialises in
seconds. Random weight initialisation; intended for plumbing checks
rather than any quality-bearing eval.
"""

from __future__ import annotations

import json
import logging
from importlib import resources

import torch

from lagged_indexer_dsa.inference.modeling.transformer import ModelArgs, Transformer

from ..loading.config import _MINI_CONFIG


logger = logging.getLogger(__name__)


def create_mini_model(
    *,
    device: str = "cuda",
    seed: int = 42,
    overrides: dict | None = None,
) -> tuple[Transformer, ModelArgs]:
    """Create a small Transformer with random weights for pre-flight checks.

    The mini model uses the same code paths as the full 671B model
    (including the FP8 Indexer) but with drastically reduced dimensions
    so it fits on a single GPU and initialises in seconds.

    Requires CUDA and the ``cuda`` extras (tilelang,
    fast-hadamard-transform) because the Indexer always uses FP8
    kernels.

    Args:
        device: Device to place the model on.
        seed: Random seed for reproducible weight initialisation.
        overrides: Optional config-field overrides applied on top of the
            mini JSON (e.g. ``{"kv_lora_rank": 512}`` for tests of the
            fused decode kernel, which is compiled for the full model's
            576-wide joint KV geometry).

    Returns:
        A (model, args) tuple.
    """
    config_path = resources.files("lagged_indexer_dsa.inference") / "configs" / _MINI_CONFIG
    with config_path.open() as f:
        raw = json.load(f)
    if overrides:
        unknown = set(overrides) - set(raw)
        if unknown:
            raise ValueError(f"unknown mini-config overrides: {sorted(unknown)}")
        raw.update(overrides)
    args = ModelArgs(**raw)

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(
            f"create_mini_model requires a CUDA device (got {device!r}); "
            "the FP8 Indexer always uses TileLang/CUDA kernels."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "create_mini_model requires CUDA, but torch.cuda.is_available() is False."
        )

    # Scope RNG to this helper so seeding does not shift the caller's
    # random stream for the rest of the interpreter session. ``cuda``
    # without an explicit index resolves to the *current* device, not 0.
    cuda_index = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_index]):
        torch.manual_seed(seed)
        # ``torch.set_default_dtype`` is a global; restoring the prior value
        # after model construction prevents the bfloat16 default from
        # leaking into other tests / callers that share the interpreter.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(device):
                model = Transformer(args)
        finally:
            torch.set_default_dtype(prev_dtype)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dtype == torch.float8_e4m3fn:
                    init_data = torch.randn(
                        param.shape, dtype=torch.bfloat16, device=param.device
                    ) * 0.02
                    param.copy_(init_data.to(torch.float8_e4m3fn))
                elif "scale" in name:
                    param.fill_(1.0)
                else:
                    param.normal_(std=0.02)

    logger.info(
        "Mini model created: layers=%d, dim=%d, params=%.2fM",
        args.n_layers,
        args.dim,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model, args
