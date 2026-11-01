"""Lagged-indexer distillation (offline training; unused at inference).

Distils the producing (layer ``L-1``) indexer toward the consuming (layer
``L``) indexer's selection, the latency-free mitigation for the one-layer
lag's long-context quality cost. Components:

* :mod:`indexer_ref` -- differentiable reference of the lightning-indexer score.
* :mod:`collect` -- forward hooks that harvest per-layer indexer inputs and
  teacher scores from a frozen forward.
* :mod:`losses` -- the cross-layer KL, pairwise-ranking, and listwise ListMLE
  objectives plus the top-k overlap metric.
* :mod:`inject` -- load a trained payload back into the shipped indexers.
* :mod:`train` -- the offline training driver.

Only :mod:`inject` is imported eagerly here (pure torch); the training-only
modules import the GPU transformer and are imported on demand.
"""

from lagged_indexer_dsa.distill.inject import (
    inject_distilled_indexers,
    load_distilled_payload,
)

__all__ = ["inject_distilled_indexers", "load_distilled_payload"]
