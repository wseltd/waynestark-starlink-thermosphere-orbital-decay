"""Invert catalogued decay dates for the inverse ballistic coefficient.

The forward problem — assume an inverse ballistic coefficient, predict a decay time —
requires assuming the one number nobody published for these vehicles in this
configuration. The external reference C_B = 0.085 m^2/kg (Shambaugh 2024,
arXiv:2406.08342, p. 6) describes a Capella Space SAR (Whitney) satellite, not
SpaceX Starlink hardware. The Group 4-7 satellites were in safe mode, which
SpaceX described as flying edge-on to minimise drag, so adopting an external
reference value would bake in an error of unknown size.

The inverse problem is better posed and produces a real result: the catalogue
recorded when each object reentered, so ask what inverse ballistic coefficient the
atmosphere model requires to reproduce each observed decay. The answer is a
measurement of the vehicles' effective inverse ballistic coefficient during the event,
conditional on the density model — and stating that conditionality is the
whole point, because an error in density maps directly into the inferred B.

What the inversion CAN establish:
  * the effective B implied by each catalogued decay, given this atmosphere;
  * whether that B is consistent with an edge-on attitude;
  * how many days of orbital life the storm cost, at the SAME B, by re-running
    the counterfactual quiet atmosphere with everything else held fixed.

What it CANNOT establish:
  * the true density, because B and rho are degenerate in the product that
    drives decay. Halving the density and doubling B gives the same curve.

Canonical tickets: P09_10_T031 through T048.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

from ..events import starlink_g4_7 as event
from ..propagation.propagator import DragPropagator, OrbitState, SpacecraftProperties


@dataclass(frozen=True, slots=True)
class DecayPoint:
    """One (inverse ballistic coefficient, decay time) pair from a forward run."""

    inverse_ballistic_coefficient: float
    decay_days: float | None
    final_perigee_km: float
    reentered: bool


@dataclass(frozen=True, slots=True)
class InversionResult:
    """The inferred inverse ballistic coefficient for one observed decay time."""

    observed_decay_days: float
    inferred_inverse_ballistic_coefficient: float | None
    bracket: tuple[float, float] | None
    residual_days: float | None
    note: str

    @property
    def ratio_to_operational(self) -> float | None:
        if self.inferred_inverse_ballistic_coefficient is None:
            return None
        return (
            self.inferred_inverse_ballistic_coefficient
            / event.OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG
        )

    def implied_area_m2(self, drag_coefficient: float, mass_kg: float) -> float | None:
        """Effective area implied by the inferred B, at a given C_d and mass."""
        if self.inferred_inverse_ballistic_coefficient is None:
            return None
        return self.inferred_inverse_ballistic_coefficient * mass_kg / drag_coefficient


def decay_for_inverse_ballistic_coefficient(
    inverse_ballistic_coefficient: float,
    density_function: object,
    *,
    epoch: _dt.datetime | None = None,
    duration_days: float = 45.0,
    sample_interval_s: float = 1800.0,
    max_step_s: float = 60.0,
    mass_kg: float = event.MASS_KG_NOMINAL,
) -> DecayPoint:
    """Forward run: propagate the insertion orbit at one inverse ballistic coefficient.

    Mass is held fixed and area is solved from B, because only the combination
    matters to the trajectory. Reporting it this way avoids implying that the
    run constrains mass and area separately.
    """
    if inverse_ballistic_coefficient <= 0:
        raise ValueError(
            f"inverse ballistic coefficient must be positive, got {inverse_ballistic_coefficient}"
        )
    drag_coefficient = 2.4
    area = inverse_ballistic_coefficient * mass_kg / drag_coefficient

    spacecraft = SpacecraftProperties(
        mass_kg=mass_kg,
        reference_area_m2=area,
        drag_coefficient=drag_coefficient,
        name=f"B={inverse_ballistic_coefficient:.4f}",
    )
    propagator = (
        DragPropagator(density_function, spacecraft, include_j2=True)  # type: ignore[arg-type]
    )
    position, velocity = event.insertion_state_vectors()
    initial = OrbitState(epoch or event.DEPLOYMENT_UTC, position, velocity)

    result = propagator.propagate(
        initial,
        duration_days,
        sample_interval_s=sample_interval_s,
        max_step_s=max_step_s,
    )
    return DecayPoint(
        inverse_ballistic_coefficient=inverse_ballistic_coefficient,
        decay_days=result.decay_days,
        final_perigee_km=result.perigee_altitude_m[-1] / 1000.0,
        reentered=result.reentered,
    )


def invert_decay(
    observed_decay_days: float,
    curve: tuple[DecayPoint, ...],
    *,
    tolerance_days: float = 0.25,
) -> InversionResult:
    """Interpolate a decay-vs-B curve to find the B matching an observed decay.

    Decay time is monotonically decreasing in B, so a simple bracket-and-
    interpolate is sound. Interpolation is done in log-B against log-decay,
    where the relationship is close to linear because decay time scales
    roughly as 1/B.
    """
    usable = sorted(
        (p for p in curve if p.reentered and p.decay_days is not None),
        key=lambda p: p.inverse_ballistic_coefficient,
    )
    if len(usable) < 2:
        return InversionResult(
            observed_decay_days=observed_decay_days,
            inferred_inverse_ballistic_coefficient=None,
            bracket=None,
            residual_days=None,
            note="fewer than two reentering points on the curve; cannot invert",
        )

    # Decay decreases as B increases, so search for the bracketing pair.
    for lo, hi in zip(usable[:-1], usable[1:], strict=True):
        d_lo, d_hi = lo.decay_days, hi.decay_days
        if d_lo is None or d_hi is None:
            # Cannot happen: `usable` filters on decay_days is not None. Kept as
            # an explicit raise rather than an assert because assert is stripped
            # under python -O, which would turn this guard into a silent None
            # arithmetic error in exactly the optimised builds least watched.
            raise RuntimeError("decay curve contains a non-reentering point after filtering")
        if (d_lo - observed_decay_days) * (d_hi - observed_decay_days) <= 0:
            if d_lo == d_hi:
                inferred = lo.inverse_ballistic_coefficient
            else:
                t = (math.log(observed_decay_days) - math.log(d_lo)) / (
                    math.log(d_hi) - math.log(d_lo)
                )
                inferred = math.exp(
                    math.log(lo.inverse_ballistic_coefficient)
                    + t
                    * (
                        math.log(hi.inverse_ballistic_coefficient)
                        - math.log(lo.inverse_ballistic_coefficient)
                    )
                )
            return InversionResult(
                observed_decay_days=observed_decay_days,
                inferred_inverse_ballistic_coefficient=inferred,
                bracket=(lo.inverse_ballistic_coefficient, hi.inverse_ballistic_coefficient),
                residual_days=None,
                note="log-log interpolation between bracketing forward runs",
            )

    fastest = min(p.decay_days for p in usable if p.decay_days is not None)
    slowest = max(p.decay_days for p in usable if p.decay_days is not None)
    return InversionResult(
        observed_decay_days=observed_decay_days,
        inferred_inverse_ballistic_coefficient=None,
        bracket=None,
        residual_days=None,
        note=(
            f"observed {observed_decay_days:.1f} d lies outside the computed "
            f"decay range {fastest:.2f}-{slowest:.2f} d; widen the B grid"
        ),
    )


def decay_days_at(
    inverse_ballistic_coefficient: float,
    curve: tuple[DecayPoint, ...],
) -> float | None:
    """Decay time at one drag parameter, read off a computed curve.

    Interpolation is linear in log(B_inv) against log(decay days) because decay
    time goes roughly as the inverse of the drag parameter over this range, and
    a linear-linear reading of the same points is visibly biased between grid
    nodes.

    Returns None when the parameter falls outside the span of REENTERING grid
    points, rather than extrapolating. Outside that span there is no propagation
    behind the answer, and a number produced by continuing the last two nodes
    would be indistinguishable in the output from one that was computed.

    This was extracted from :func:`storm_cost_days`, where it was a closure. It
    is public because callers legitimately need the absolute lifetime and not
    only the storm-minus-quiet difference, and a second copy of a log-log
    interpolation is exactly the kind of duplicate that drifts.
    """
    usable = sorted(
        (p for p in curve if p.reentered and p.decay_days is not None),
        key=lambda p: p.inverse_ballistic_coefficient,
    )
    if len(usable) < 2:
        return None
    for lo, hi in zip(usable[:-1], usable[1:], strict=True):
        lo_b = lo.inverse_ballistic_coefficient
        hi_b = hi.inverse_ballistic_coefficient
        if lo_b <= inverse_ballistic_coefficient <= hi_b:
            d_lo, d_hi = lo.decay_days, hi.decay_days
            if d_lo is None or d_hi is None:
                raise RuntimeError(
                    "decay curve contains a non-reentering point after filtering"
                )
            span = math.log(hi_b) - math.log(lo_b)
            if span == 0:
                return d_lo
            t = (math.log(inverse_ballistic_coefficient) - math.log(lo_b)) / span
            return math.exp(math.log(d_lo) + t * (math.log(d_hi) - math.log(d_lo)))
    return None


def storm_cost_days(
    inverse_ballistic_coefficient: float,
    storm_curve: tuple[DecayPoint, ...],
    quiet_curve: tuple[DecayPoint, ...],
) -> float | None:
    """Extra orbital life the quiet atmosphere would have given, at the same B.

    Both curves are computed with identical spacecraft parameters and identical
    initial state, so the difference is attributable to the atmosphere alone.
    Returns None when either case fails to reenter within the simulated window,
    because "did not decay in 45 days" is not a number and must not be turned
    into one.
    """
    storm = decay_days_at(inverse_ballistic_coefficient, storm_curve)
    quiet = decay_days_at(inverse_ballistic_coefficient, quiet_curve)
    if storm is None or quiet is None:
        return None
    return quiet - storm
