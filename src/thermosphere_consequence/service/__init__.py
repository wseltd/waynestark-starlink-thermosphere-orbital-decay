"""The orchestration layer between the library and its adapters.

This package sits BELOW the HTTP adapter and the CLI, and ABOVE the physics. It
exists because the API and the CLI were each doing the same three things
separately — choosing providers, checking that those providers may be used in
the declared mode, and attaching provenance to the answer — and two
implementations of that sequence will eventually disagree about one of them.

What it does:

  * composes a configured provider set from an explicit
    :class:`~thermosphere_consequence.service.composition_root.ServiceConfig`;
  * enforces the licence gate and the analysis-mode licence before answering;
  * returns every answer paired with its provenance and its epistemic status.

What it does NOT do, and what a structural test asserts it does not do: any
scientific logic. There is no arithmetic anywhere in this package. Densities,
drag coefficients and areas are computed by the library functions that already
implement them, and the same functions back the API, the CLI, the analysis
scripts and the notebooks — so an answer cannot differ between entry points.

It is also not a second HTTP layer. :mod:`thermosphere_consequence.api.app`
remains the only adapter that knows about requests, status codes and wire
models; this package knows nothing about any of them.
"""

from __future__ import annotations

__all__: list[str] = []
