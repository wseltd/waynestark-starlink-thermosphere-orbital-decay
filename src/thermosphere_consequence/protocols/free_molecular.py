"""Structural contract for a free-molecular force model.

A free-molecular model answers one question: given the neutral gas at a point
and a body moving through it, what force does the body feel and what drag-area
does that imply? Nothing that consumes the answer may know WHICH model produced
it, so that replacing the gas-surface treatment changes the number and never
the call site.

WHAT THE CONTRACT BUYS. The gas-surface kernel is the least certain part of the
whole chain — Sentman diffuse, DRIA, specular and Maxwell mixtures disagree by
more than the storm signal being measured — and it is selected by a string on
the concrete model rather than by a different type. A caller that accepts a
``FreeMolecularModel`` can be handed a body evaluated under any of those
kernels, at any attitude, with any accommodation provider, and does not change.
That is what lets the sensitivity sweep vary the surface physics without the
propagation, the reporting or the uncertainty machinery knowing it happened.

TWO METHODS, NOT ONE, AND ON PURPOSE. :meth:`FreeMolecularModel.evaluate`
returns the full vector result — force, torque, lift, per-facet contributions —
because attitude and off-centre force are the things a scalar model cannot
represent. :meth:`FreeMolecularModel.effective_inverse_ballistic_coefficient_m2_kg`
returns the single number orbital decay actually constrains. A contract with
only the first would make every caller re-derive the second and eventually two
of them would derive it differently; a contract with only the second would
throw away the torque that decides whether a tumbling body settles into a
high-drag or a low-drag attitude.

TERMINOLOGY. The scalar this contract returns is C_d*A/m in m^2/kg, the INVERSE
ballistic coefficient. It is not the ballistic coefficient, which is its
reciprocal in kg/m^2 and moves the other way; see
:mod:`thermosphere_consequence.domain.drag_parameters`.

Canonical tickets: P11_12_T009 through P11_12_T014.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..aerodynamics.faceted import FacetedAerodynamicResult

__all__ = ["FreeMolecularModel"]


@runtime_checkable
class FreeMolecularModel(Protocol):
    """Anything that can state the free-molecular force on a body."""

    #: The area every reported coefficient is referenced to. Part of the
    #: contract because C_d alone is meaningless without it: two models that
    #: referenced different areas have not disagreed about anything until their
    #: C_d * A_ref products are compared.
    reference_area_m2: float

    def evaluate(
        self,
        number_densities: Mapping[str, float],
        temperature_k: float,
        relative_velocity_m_s: Sequence[float],
        *,
        attitude_quaternion: Sequence[float] | None = None,
        centre_of_mass_body: Sequence[float] | None = None,
    ) -> FacetedAerodynamicResult:
        """Force, torque and coefficients for one atmospheric state and attitude.

        ``relative_velocity_m_s`` is the SPACECRAFT's velocity relative to the
        atmosphere. The free stream travels along its negative, and reversing
        the vector reverses the drag, which is why the result reports the flow
        direction it actually used.
        """
        ...

    def effective_inverse_ballistic_coefficient_m2_kg(
        self,
        mass_kg: float,
        number_densities: Mapping[str, float],
        temperature_k: float,
        relative_velocity_m_s: Sequence[float],
        *,
        attitude_quaternion: Sequence[float] | None = None,
    ) -> float:
        """C_d * A / m [m^2/kg] for this body, attitude and atmosphere.

        NOT a ballistic coefficient. A value in m^2/kg may never be reported
        under that name.
        """
        ...
