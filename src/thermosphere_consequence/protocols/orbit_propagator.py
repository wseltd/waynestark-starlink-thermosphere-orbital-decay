"""Structural contract for anything that can fly a state forward under drag.

The counterfactual this project exists to answer — "what would this orbit have
done under a different atmosphere?" — is only meaningful if the propagator is
replaceable without the question changing. This contract is what makes that
replacement checkable rather than hoped for.

WHAT THE CONTRACT BUYS. A caller that accepts an ``OrbitPropagator`` can be
handed:

  * :class:`~thermosphere_consequence.propagation.propagator.DragPropagator`,
    the Cowell integrator with J2 and free-molecular drag;
  * :class:`~thermosphere_consequence.scenarios.attribution.CounterfactualPropagator`,
    which adds neutral wind and thrust as corrections to the same drag;

and cannot tell them apart. The attribution study swaps the second for the
first at one construction site and nothing downstream — no lifetime reader, no
envelope corner, no report — is touched.

WHY ``density_function`` IS PART OF THE CONTRACT AND NOT AN IMPLEMENTATION
DETAIL. It is the seam the counterfactual runs through: the SAME trajectory is
integrated against a quiet atmosphere and a storm atmosphere with nothing else
changed, so the difference is attributable to the atmosphere and to nothing
else. A propagator that resolved its own density internally could not be used
for that comparison, so a contract that omitted this member would admit
implementations that cannot do the job the contract is named for.

WHAT IT DELIBERATELY DOES NOT ADMIT.
:class:`~thermosphere_consequence.validation.independent_propagator.GaussPropagator`
does not satisfy this protocol, and that is correct rather than an oversight.
It integrates Gauss's variational equations from classical elements, carries an
inverse ballistic coefficient instead of a spacecraft, and its ``propagate``
takes a separate epoch. It exists to be an INDEPENDENT check on the Cowell
propagator, and a shared interface that let the two be substituted for each
other would quietly permit the comparison to be run against itself.

Canonical tickets: P03_05_T040 through T052, P06_08_T001 through T012.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..propagation.propagator import (
        DensityFunction,
        OrbitState,
        PropagationResult,
        SpacecraftProperties,
    )

__all__ = ["OrbitPropagator"]


@runtime_checkable
class OrbitPropagator(Protocol):
    """Anything that can integrate an orbital state forward under drag."""

    #: The injected atmosphere. Part of the contract because it is the seam the
    #: counterfactual is run through; see the module docstring.
    density_function: DensityFunction

    #: Mass, area and drag coefficient. Carried whole rather than reduced to
    #: C_d*A/m because a result must be able to report the inverse ballistic
    #: coefficient AND the three non-identifiable factors it was formed from.
    spacecraft: SpacecraftProperties

    def propagate(
        self,
        initial: OrbitState,
        duration_days: float,
        *,
        sample_interval_s: float = 60.0,
        rtol: float = 1e-9,
        atol: float = 1e-6,
        max_step_s: float = 30.0,
    ) -> PropagationResult:
        """Integrate until reentry or until the duration elapses.

        The tolerances are part of the contract rather than left to the
        implementation: a decay time is a statement about an integration, and
        two implementations that agree only because one of them was run loose
        have not been compared.
        """
        ...
