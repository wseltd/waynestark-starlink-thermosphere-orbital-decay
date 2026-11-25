"""Species-resolved thermosphere, free-molecular drag and orbital-consequence engine.

WHAT "SPECIES-RESOLVED" QUALIFIES in that title, since it governs one noun and
not all three. The THERMOSPHERE is resolved by constituent, and so is every
diagnostic the :mod:`.molecular` layer builds on it. The DRAG is an effective
scalar: ``F = 0.5 * C_d * rho_total * A * v_rel^2``, one coefficient against the
total mass density. Composition is what decides that coefficient — the
per-species Sentman coefficients are weighted by species mass flux to form it —
so the mixture is not absent from the force; it simply enters through C_d and
nowhere else. The honest one-line description of the whole arrangement is a
species-resolved molecular diagnostic and interpretation layer accompanying an
effective scalar free-molecular drag model. It is not a force closure in which
constituents act independently, and ``FUTURE_WORK.md`` records the closure that
would be.

Importing this package has no side effects: no file I/O, no network access, no
logging configuration and no registry construction. Everything expensive is
built lazily by the module that owns it.

Layering (enforced by tests/test_package_layering.py):

    domain       units, constants, species, provenance, temporal
    molecular    pure gas-kinetic and composition kernels
    aerodynamics pure drag kernels
    cli          thin adapter; never contains scientific logic

A kernel may depend on domain and on nothing above it. Layers reserved for
later stages are not created until they have contents, so this list stays a
description of the package rather than a plan for it: protocols (structural
contracts for atmosphere providers and propagators) arrives with Phase 3,
events (case-study manifests) with the Starlink scenario.

Canonical ticket: P00_02_T002.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
