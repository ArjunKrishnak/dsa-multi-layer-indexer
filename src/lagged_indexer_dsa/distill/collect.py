"""Forward hooks to harvest per-layer indexer inputs and teacher scores.

During a single frozen forward pass (run under ``torch.inference_mode``), these
hooks capture, for every layer whose lightning indexer fires:

* the indexer *inputs* ``x`` (layer hidden states), ``qr`` (compressed query) and
  the rotary table ``freqs_cis`` -- everything the differentiable
  :class:`~lagged_indexer_dsa.distill.indexer_ref.IndexerRef` needs to recompute
  the score with gradients, and
* the *teacher* indexer score (the frozen FP8 kernel output), optionally
  subsampled to a set of query rows to bound memory.

The captured tensors are detached, so the frozen backbone never participates in
the student's backward pass. The student forward + loss run *after* the frozen
forward, outside ``inference_mode``.
"""

from __future__ import annotations

import logging
from functools import partial

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IndexerActivationCollector:
    """Capture indexer inputs and teacher scores via forward hooks.

    Usage::

        collector = IndexerActivationCollector(model)
        with collector:
            collector.set_query_idx(query_idx)   # optional subsample of query rows
            with torch.inference_mode():
                model(tokens, start_pos=0, full_logits=False)
        inputs = collector.inputs          # {layer_idx: (x, qr, freqs_cis)} on CPU
        teacher = collector.teacher_scores  # {layer_idx: (b, Q, Sk) fp32} on GPU

    ``inputs`` tensors are offloaded to host memory to bound GPU usage; move the
    tuple for the producer layer you are training back to the device before
    feeding it to :class:`~lagged_indexer_dsa.distill.indexer_ref.IndexerRef`.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.query_idx: torch.Tensor | None = None
        self.inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.teacher_scores: dict[int, torch.Tensor] = {}

    def set_query_idx(self, query_idx: torch.Tensor | None) -> None:
        """Restrict captured teacher scores to these query rows (None = all)."""
        self.query_idx = (
            None if query_idx is None else query_idx.to(dtype=torch.long).clone()
        )

    def clear(self) -> None:
        self.inputs.clear()
        self.teacher_scores.clear()

    def _pre_hook(self, layer_idx: int, _module: nn.Module, args: tuple) -> None:
        # Indexer.forward(self, x, qr, start_pos, freqs_cis, mask)
        x, qr, _start_pos, freqs_cis = args[0], args[1], args[2], args[3]
        # inference_mode(False) + clone(): the teacher forward runs under
        # torch.inference_mode(), and tensors cloned while it is active stay
        # inference tensors that autograd refuses to save for backward when they
        # feed IndexerRef / cross_layer_kl. Re-enable grad tracking so the stored
        # copies are ordinary tensors.
        #
        # The full-sequence x (b, Sk, dim) is ~200 MB per layer; keeping it on
        # GPU for all layers would add ~20 GB that accumulates across the frozen
        # forward and OOMs alongside the model weights and the dense-prefill
        # attention spike. Offload the inputs to host memory and move each
        # producer's copy back to GPU one at a time in the student loop.
        with torch.inference_mode(False):
            self.inputs[layer_idx] = (
                x.detach().clone().to("cpu"),
                qr.detach().clone().to("cpu"),
                freqs_cis.detach().clone().to("cpu"),
            )

    def _post_hook(self, layer_idx: int, _module: nn.Module, _args: tuple, output: tuple) -> None:
        # Indexer.forward returns (topk_indices, index_score)
        index_score = output[1].detach()
        if self.query_idx is not None:
            index_score = index_score.index_select(
                1, self.query_idx.to(index_score.device)
            )
        with torch.inference_mode(False):
            self.teacher_scores[layer_idx] = index_score.float().clone()

    def __enter__(self) -> "IndexerActivationCollector":
        registered = 0
        for layer_idx, layer in enumerate(self._model.layers):
            attn = getattr(layer, "attn", None)
            indexer = getattr(attn, "indexer", None) if attn is not None else None
            if indexer is None:
                continue
            self._handles.append(
                indexer.register_forward_pre_hook(partial(self._pre_hook, layer_idx))
            )
            self._handles.append(
                indexer.register_forward_hook(partial(self._post_hook, layer_idx))
            )
            registered += 1
        logger.info("Registered indexer hooks on %d layers", registered)
        if registered == 0:
            raise RuntimeError(
                "No indexer modules found to hook; expected layer.attn.indexer on "
                "sparse-attention layers."
            )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
