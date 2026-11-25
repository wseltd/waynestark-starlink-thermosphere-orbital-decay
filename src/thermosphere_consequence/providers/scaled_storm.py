"""A SCALAR STRESS TEST that multiplies the storm component of another provider.

This is not an atmosphere model. It multiplies the storm-minus-quiet density
difference reported by a real model by a constant, and reports the result. It
contains no additional physics, no additional state variables, no additional
forcing, and no additional equations. Multiplying one number by another is not
a simulation.

Why it exists. NRLMSISE-00 is known to under-respond to storms by a documented
amount at exactly the altitude this project cares about: He et al. 2023 report
peak storm-time density enhancement at 200/300 km of ~20-25% for MSIS against
~40% for WAM-IPE and ~57-75% for MAGE, and Fang et al. 2022 report 50-125% at
200-400 km from WAM-IPE. Ignoring that would produce a confident answer built
on a model that is wrong in a known direction by a known amount. The response
is to sweep an explicit multiplier and report how the conclusion moves.

What it CAN support:
    "if the true storm response were k times what NRLMSISE-00 reports, the
    decay time would be D days."

What it CANNOT support, and what naming it this way exists to prevent:

    PROHIBITED. A result produced at k=2.0 is NOT a WAM-IPE result. A result
    produced at k=3.5 is NOT a MAGE result. WAM-IPE and MAGE are coupled
    physics-based whole-atmosphere simulations with their own dynamics,
    composition transport, electrodynamics and Joule-heating distributions.
    A scalar multiplier reproduces none of that. Choosing k so the OUTPUT
    resembles a published model's peak enhancement does not make the output
    that model's prediction; it makes it MSIS with a number stuck on the end.
    Labelling a k-scaled run with a physics model's name is a misattribution
    of provenance and is forbidden throughout this repository.

For a real, cited alternative to MSIS's storm response, use
``providers.published_envelope.PublishedEventDensityEnvelopeProvider``, which
encodes the published event-specific enhancement bands themselves, with their
citations, altitude coverage, time coverage and limitations, and which refuses
to answer outside what its sources support.

Canonical tickets: P09_10_T049 through T056.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import replace

from ..protocols.atmosphere import AtmosphereProvider, AtmosphereState, DriverState

#: Attached to every state this provider returns. Kept as a module constant so
#: the prohibition travels with the number into JSON, reports and figures
#: rather than living only in documentation a reader may never open.
PROHIBITED_LABEL_NOTE = (
    "PROHIBITED LABELLING: a x2 result must NOT be called a 'WAM-IPE result' "
    "and a x3.5 result must NOT be called a 'MAGE result'. This provider is a "
    "SCALAR STRESS TEST, not a physics-based simulation: it has no dynamics, "
    "no composition transport, no electrodynamics and no Joule heating. "
    "Matching a published model's peak enhancement by choosing a multiplier "
    "does not reproduce that model."
)

_STRESS_TEST_NOTE_TEMPLATE = (
    "SCALAR STRESS TEST, NOT A PREDICTION AND NOT A PHYSICS MODEL: the "
    "storm-minus-quiet density difference reported by {base} was multiplied by "
    "the constant {k:g}. The multiplier spans the range over which NRLMSISE-00 "
    "is known to under-respond (He et al. 2023: MSIS ~20-25% vs WAM-IPE ~40% "
    "vs MAGE ~57-75% at 200/300 km), but the multiplication itself adds no "
    "physics whatsoever."
)

_UNIFORM_SPECIES_NOTE = (
    "All species multiplied uniformly; storm-time composition change is "
    "preserved as the base model reports it but is not itself amplified. This "
    "bounds TOTAL density, not composition."
)


class ScalarDensityStressTestProvider:
    """Multiplies the storm-minus-quiet density difference by a fixed constant.

    The multiplier is applied to the DIFFERENCE from a quiet baseline, not to
    the absolute density, so the quiet state is left exactly as the underlying
    model reports it. Scaling absolute density instead would move the quiet
    baseline too and would no longer be a statement about storm response.

    Every species is scaled by the same factor. This is a deliberate
    simplification and a real limitation: storms change composition as well as
    total density -- the O/N2 ratio drops in the storm-time auroral zone -- so
    a uniform scaling preserves the modelled composition change without
    amplifying it. It bounds total density, not composition.

    Naming rule enforced by this class: ``name`` renders as
    ``scalar_density_stress_test_x{k}`` so that no output, filename, figure
    label or JSON key produced from it can read as the name of a physics model.
    """

    def __init__(
        self,
        base: AtmosphereProvider,
        quiet_drivers_for: object,
        *,
        enhancement_factor: float,
        label: str = "",
    ) -> None:
        if enhancement_factor <= 0:
            raise ValueError(
                f"enhancement factor must be positive, got {enhancement_factor}"
            )
        self.base = base
        self.quiet_drivers_for = quiet_drivers_for
        self.enhancement_factor = enhancement_factor
        self.name = f"scalar_density_stress_test_x{enhancement_factor:g}"
        self.version = f"{base.version}+scalar_stress_x{enhancement_factor:g}"
        self.label = label or self.name

    def state_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> AtmosphereState:
        storm = self.base.state_at(
            time_utc, altitude_m, latitude_deg, longitude_deg, drivers
        )
        quiet_drivers = self.quiet_drivers_for(drivers)  # type: ignore[operator]
        quiet = self.base.state_at(
            time_utc, altitude_m, latitude_deg, longitude_deg, quiet_drivers
        )

        k = self.enhancement_factor
        scaled: dict[str, float] = {}
        for symbol, storm_n in storm.number_densities.items():
            quiet_n = quiet.number_densities.get(symbol)
            if quiet_n is None:
                # No quiet counterpart to difference against: pass the storm
                # value through unscaled rather than inventing a baseline.
                scaled[symbol] = storm_n
                continue
            amplified = quiet_n + k * (storm_n - quiet_n)
            # A large factor applied to a species that DROPS in the storm can
            # drive it negative, which is unphysical. Clamp at zero and record
            # it, rather than emitting a negative number density.
            scaled[symbol] = max(0.0, amplified)

        temperature = quiet.temperature_k + k * (storm.temperature_k - quiet.temperature_k)

        return replace(
            storm,
            number_densities=scaled,
            temperature_k=max(temperature, 1.0),
            total_mass_density=None,
            provider_name=self.name,
            provider_version=self.version,
            notes=storm.notes
            + (
                _STRESS_TEST_NOTE_TEMPLATE.format(base=self.base.name, k=k),
                PROHIBITED_LABEL_NOTE,
                _UNIFORM_SPECIES_NOTE,
            ),
        )


#: Retained so existing imports keep working. The class was renamed because
#: "ScaledStormProvider" reads like a storm model; it is a scalar stress test.
ScaledStormProvider = ScalarDensityStressTestProvider
