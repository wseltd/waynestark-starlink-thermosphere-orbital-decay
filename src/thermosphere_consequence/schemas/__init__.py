"""Declared modes, the runtime claim gate and the run manifest.

This package holds the machinery that decides what a run is *allowed to say*
about itself. It is deliberately free of physics: nothing here computes a
number, so nothing here can be tempted to soften a gate to make a result
presentable.

Three pieces:

* :mod:`.modes` — the controlled vocabulary. Each member documents what it
  licenses a caller to claim, so "what mode is this run in?" and "what may this
  run assert?" are the same question.
* :mod:`.claim_gate` — the enforcement. Prohibited claims are predicates over
  the declared modes, evaluated at emission time, rather than a list of
  sentences grepped out of finished prose.
* :mod:`.run_manifest` — the record. A run states its modes, its inputs and its
  calibration status up front, and the gate reads its context from that
  statement.
"""

from __future__ import annotations

__all__ = ["claim_gate", "modes", "run_manifest"]
