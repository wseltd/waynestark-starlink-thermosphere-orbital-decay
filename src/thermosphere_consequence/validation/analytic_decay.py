"""Closed-form decay estimates, as an independent check on the integrator.

A numerical propagator can be wrong in ways its own tests do not reveal: a
mis-signed perturbation, a unit slip in the drag term, a bad frame. Energy
conservation catches some of that, but not an error that is smoothly wrong.

These closed forms share NO code with the propagator. They come from the
standard circular-orbit energy-balance result

    da/dt = -B * rho * sqrt(mu * a)

which follows from equating the work done by drag to the change in orbital
energy. If the numerical answer and the analytic bracket disagree by an order
of magnitude, one of them is broken, and finding out which is cheap.

These are BOUNDS, not predictions. The fixed-density bound holds density at its
initial value and therefore understates the decay rate, giving an UPPER bound
on decay time. The scale-height estimate applies perigee density around the
whole orbit and therefore overstates drag for an eccentric orbit, giving a
LOWER bound. A numerical result outside that bracket is a finding.

Canonical tickets: P06_08_T044, P11_12_T049.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain.constants import MU_EARTH, R_EARTH_EQUATORIAL

# Neutral-density scale height near 200-300 km. A representative value, not a
# fitted one: it is used only to bracket, and the bracket is wide.
NOMINAL_SCALE_HEIGHT_M = 35_000.0

REENTRY_ALTITUDE_M = 100_000.0


@dataclass(frozen=True, slots=True)
class AnalyticDecayBracket:
    """Upper and lower analytic bounds on decay time, with the rate that set them."""

    da_dt_m_per_s: float
    da_dt_km_per_day: float
    upper_bound_days: float
    lower_bound_days: float
    brackets: bool
    numerical_days: float | None
    note: str


def circular_decay_rate(
    semi_major_axis_m: float, inverse_ballistic_coefficient: float, density_kg_m3: float
) -> float:
    """da/dt in m/s for a circular orbit under drag. Negative by construction."""
    if semi_major_axis_m <= 0:
        raise ValueError(f"semi-major axis must be positive, got {semi_major_axis_m}")
    if inverse_ballistic_coefficient <= 0:
        raise ValueError(
            f"inverse ballistic coefficient must be positive, got {inverse_ballistic_coefficient}"
        )
    if density_kg_m3 < 0:
        raise ValueError(f"density must be non-negative, got {density_kg_m3}")
    return -inverse_ballistic_coefficient * density_kg_m3 * math.sqrt(
        MU_EARTH.value * semi_major_axis_m
    )


def bracket_decay_time(
    *,
    initial_altitude_m: float,
    inverse_ballistic_coefficient: float,
    density_kg_m3: float,
    numerical_days: float | None = None,
    scale_height_m: float = NOMINAL_SCALE_HEIGHT_M,
) -> AnalyticDecayBracket:
    """Bound the decay time analytically and compare against a numerical result."""
    if initial_altitude_m <= REENTRY_ALTITUDE_M:
        raise ValueError(
            f"initial altitude {initial_altitude_m} m is at or below the "
            f"{REENTRY_ALTITUDE_M} m reentry floor; nothing to decay"
        )
    if scale_height_m <= 0:
        raise ValueError(f"scale height must be positive, got {scale_height_m}")

    a = R_EARTH_EQUATORIAL.value + initial_altitude_m
    rate = circular_decay_rate(a, inverse_ballistic_coefficient, density_kg_m3)
    if rate == 0.0:
        raise ValueError("decay rate is zero; the orbit does not decay under this density")
    rate_per_day = rate * 86400.0
    drop = initial_altitude_m - REENTRY_ALTITUDE_M

    upper = drop / abs(rate_per_day)
    lower = (
        scale_height_m / abs(rate_per_day) * math.log1p(drop / scale_height_m)
    )

    brackets = (
        numerical_days is not None and lower <= numerical_days <= upper
    )
    if numerical_days is None:
        note = "no numerical result supplied; bounds reported alone"
    elif brackets:
        note = (
            f"numerical {numerical_days:.2f} d lies inside the analytic bracket "
            f"{lower:.2f}-{upper:.2f} d, computed from an independent closed form"
        )
    else:
        note = (
            f"NUMERICAL RESULT OUTSIDE THE ANALYTIC BRACKET: {numerical_days:.2f} d "
            f"against {lower:.2f}-{upper:.2f} d. One of the two is wrong; the "
            "closed form shares no code with the integrator, so this is worth "
            "resolving before the number is used."
        )

    return AnalyticDecayBracket(
        da_dt_m_per_s=rate,
        da_dt_km_per_day=rate_per_day / 1000.0,
        upper_bound_days=upper,
        lower_bound_days=lower,
        brackets=brackets,
        numerical_days=numerical_days,
        note=note,
    )
