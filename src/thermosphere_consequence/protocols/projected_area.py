"""Structural contract for stating the area the free stream sees.

Projected area is the quantity every drag number is silently referenced to, and
it is the one most easily computed twice in two slightly different ways. This
contract exists to pin the general form — a body's facets, plus the direction
the gas travels, yielding one area in square metres — so that a caller can be
handed a different area rule without being rewritten.

WHAT THE CONTRACT BUYS. A caller that accepts a ``ProjectedAreaModel`` can be
given the plain geometric projection
(:func:`~thermosphere_consequence.aerodynamics.geometry.projected_area`), or a
rule that first removes self-shadowed facets, or a rule that returns a fixed
reference area regardless of attitude — and nothing at the call site changes.
That matters because the three answers differ: for a convex body the geometric
projection and the shadowed area are identical, for a concave one the
projection is an upper bound, and a fixed reference area moves the attitude
dependence out of the area and into C_d. Which of the three a result used is a
statement about the result, and the swap has to be possible in order for the
statement to be worth making.

A CALLABLE PROTOCOL RATHER THAN A CLASS, for the same reason as
:class:`~thermosphere_consequence.protocols.atmosphere.DriverLookup`: a module
function, a bound method or a closure over a precomputed attitude table all
satisfy it without inheriting anything, and no area rule in this package needs
state that would justify forcing it to become an object.

WHAT IT DOES NOT PROMISE. Satisfying this contract says nothing about
self-shadowing. ``geometry.projected_area`` is documented as the geometric
projection only, and a caller that needs the shadowed area must ask for it
explicitly through
:func:`~thermosphere_consequence.aerodynamics.geometry.self_shadowing_visibility`
or read ``visible_area_m2`` off a faceted result. A contract cannot make that
distinction disappear and this one does not try.

Canonical tickets: P11_12_T001 through P11_12_T003.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..aerodynamics.geometry import Facet

__all__ = ["ProjectedAreaModel"]


@runtime_checkable
class ProjectedAreaModel(Protocol):
    """Anything that can state the area a faceted body presents to the flow."""

    def __call__(
        self, facets: Sequence[Facet], flow_unit_vector: Sequence[float]
    ) -> float:
        """Return the presented area in m^2.

        ``flow_unit_vector`` is the direction the FREE-STREAM GAS TRAVELS, in
        the same frame as the facet normals — the negative of the spacecraft's
        atmosphere-relative velocity direction. Taking the sign the other way
        round returns the lee-side area, which is a plausible-looking number
        and a completely wrong one, so the direction is named in the contract
        rather than left to each implementation's docstring.
        """
        ...
