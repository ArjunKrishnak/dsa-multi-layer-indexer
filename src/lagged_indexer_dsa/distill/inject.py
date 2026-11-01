"""Load distilled indexer weights back into a model for evaluation.

Distillation trains a reference indexer per producing layer and saves the
result as a ``.pt`` payload of the form::

    {"layers": {layer_idx: {"wq_b_weight": Tensor,
                            "wk_weight": Tensor,
                            "weights_proj_weight": Tensor,
                            "k_norm_weight": Tensor,
                            "k_norm_bias": Tensor}}}

To measure the quality recovery we load those weights into the shipped
:class:`Indexer` modules and run the existing lag eval through the genuine
``fp8_index`` kernel.

The shipped indexer projections (``wq_b``/``wk``) are stored in **bf16** even on
FP8 checkpoints, while ``weights_proj`` is **fp32**; the ``linear`` helper routes
any non-FP8 weight through plain ``F.linear``. Distilled weights are installed at
each target module's own dtype (bf16 for ``wq_b``/``wk``, fp32 for
``weights_proj``) -- exact and deployment-faithful, with no FP8 re-quantization.
The index *score* still runs through the real FP8 kernel (it quantizes
activations regardless), so kernel-level fidelity is preserved.

Only weight *values* change; arithmetic, kernel, and parameter count are
unchanged, so the zero-added-latency property is unaffected.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_distilled_payload(path: str) -> dict:
    """Load a distilled-indexer payload from ``path``."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "layers" not in payload:
        raise ValueError(f"{path} is not a distilled-indexer payload (missing 'layers')")
    return payload


def _install_projection(linear: nn.Module, weight: torch.Tensor) -> None:
    """Replace a ``Linear``'s weight with the distilled tensor, matching dtype.

    The distilled payload is fp32. Installing it at the target module's own
    dtype keeps the shipped precision contract: ``wq_b``/``wk`` are bf16 (even on
    FP8 checkpoints), while ``weights_proj`` is fp32, so blindly casting to bf16
    would break ``weights_proj(x.float())``. An actual FP8 weight is installed as
    bf16 (and its block-scale dropped) so the ``linear`` helper takes the plain
    ``F.linear`` path rather than re-quantizing distilled values.
    """
    if tuple(linear.weight.shape) != tuple(weight.shape):
        raise ValueError(
            f"distilled weight shape {tuple(weight.shape)} != target {tuple(linear.weight.shape)}"
        )
    target_dtype = linear.weight.dtype
    if target_dtype == torch.float8_e4m3fn:
        target_dtype = torch.bfloat16
    linear.weight = nn.Parameter(
        weight.to(target_dtype).to(linear.weight.device), requires_grad=False
    )
    # If the original weight was FP8 it carried a block-scale; drop it so the
    # ``linear`` helper takes the plain F.linear path.
    if getattr(linear, "scale", None) is not None:
        linear.register_parameter("scale", None)


def _copy_exact(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy ``source`` into ``target`` requiring an exact shape match.

    ``Tensor.copy_`` broadcasts, so a malformed payload tensor could be silently
    expanded; require identical shapes to fail fast instead.
    """
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"distilled tensor shape {tuple(source.shape)} != target {tuple(target.shape)}"
        )
    target.copy_(source.to(torch.float32).to(target.device))


@torch.no_grad()
def inject_distilled_indexers(model: nn.Module, payload: dict) -> int:
    """Overwrite producer-layer indexers in ``model`` with distilled weights.

    The payload stores an :class:`IndexerRef` ``state_dict`` per layer whose
    parameter names (``wq_b_weight``, ``wk_weight``, ``weights_proj_weight``,
    ``k_norm_weight``, ``k_norm_bias``) map onto the shipped ``Indexer``'s
    ``wq_b`` / ``wk`` / ``weights_proj`` / ``k_norm`` modules.

    Returns the number of layers injected. Raises if a referenced layer has no
    indexer or a shape mismatch occurs (fail-fast; no silent skips).
    """
    layers = payload["layers"]

    # Phase 1: validate every layer/state up front so a malformed entry cannot
    # leave the model partially injected. Nothing is mutated until all pass.
    plan: list[tuple[int, nn.Module, dict]] = []
    for layer_key, state in layers.items():
        layer_idx = int(layer_key)
        if layer_idx < 0 or layer_idx >= len(model.layers):
            raise ValueError(f"layer {layer_idx} is outside model.layers")
        attn = getattr(model.layers[layer_idx], "attn", None)
        indexer = getattr(attn, "indexer", None) if attn is not None else None
        if indexer is None:
            raise ValueError(f"layer {layer_idx} has no indexer to inject into")

        for key, target in (
            ("wq_b_weight", indexer.wq_b.weight),
            ("wk_weight", indexer.wk.weight),
            ("weights_proj_weight", indexer.weights_proj.weight),
            ("k_norm_weight", indexer.k_norm.weight),
            ("k_norm_bias", indexer.k_norm.bias),
        ):
            if key not in state:
                raise ValueError(f"layer {layer_idx} payload is missing '{key}'")
            if tuple(state[key].shape) != tuple(target.shape):
                raise ValueError(
                    f"layer {layer_idx} '{key}' shape {tuple(state[key].shape)} "
                    f"!= target {tuple(target.shape)}"
                )
        plan.append((layer_idx, indexer, state))

    # Phase 2: apply now that the whole payload is known-good.
    for _layer_idx, indexer, state in plan:
        _install_projection(indexer.wq_b, state["wq_b_weight"])
        _install_projection(indexer.wk, state["wk_weight"])
        _install_projection(indexer.weights_proj, state["weights_proj_weight"])

        _copy_exact(indexer.k_norm.weight, state["k_norm_weight"])
        _copy_exact(indexer.k_norm.bias, state["k_norm_bias"])

    n = len(plan)
    logger.info("Injected distilled indexer weights into %d layers", n)
    return n
