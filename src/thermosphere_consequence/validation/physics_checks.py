"""Physics invariants that decide whether a propagation may be believed.

A propagation result is a long list of numbers, and a long list of numbers is
persuasive whether or not it is right. These checks exist to make specific,
falsifiable statements about a run BEFORE any conclusion is drawn from it, so
that a sign error or a failed integration is caught here rather than by a reader
who happens to look at the right plot.

Three decisions shape the module.

First, the tolerances are DERIVED, not chosen. Every "must not increase" check
has to tolerate the J2 short-period oscillation, because the propagator reports
OSCULATING elements and those genuinely oscillate: at first order the osculating
semi-major axis swings by roughly ``3 * J2 * (Re/a)**2 * a``, which is about
20 km in this orbit regime and is real physics, not error. Each tolerance is
therefore computed from that amplitude at the run's own mean semi-major axis, so
it scales with the orbit instead of being a constant that happens to suit one
case.

Second, a failing invariant is a RESULT, not an exception. A violated check
returns a failing ``CheckOutcome`` so a batch of runs can be screened without
exception handling. An input that cannot support the check at all — no samples,
an inverted altitude band, a negative tolerance — raises ``ValueError``, because
answering "passed" or "failed" to a question that was never asked is worse than
stopping.

Third, ``check_species_closure`` is deliberately a warning. A species sum that
disagrees with a model's own total is information about the data, and the one
thing that must never happen to it is silent reconciliation.

Related canonical ticket: P03_05_T055 (energy diagnostics).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..domain.constants import J2_EARTH, MU_EARTH, R_EARTH_EQUATORIAL
from ..molecular.composition import (
    NumberDensities,
    density_closure_fraction,
    total_species_mass_density,
)
from ..propagation.propagator import (
    MAX_ALTITUDE_M,
    REENTRY_ALTITUDE_M,
    PropagationResult,
)

__all__ = [
    "CheckOutcome",
    "Severity",
    "check_altitude_bounds",
    "check_density_positive",
    "check_energy_monotonic",
    "check_perigee_monotonic",
    "check_species_closure",
    "run_all_checks",
    "summarise",
]

Severity = Literal["error", "warning"]

VALID_SEVERITIES: tuple[str, ...] = ("error", "warning")

# Safety factor on the analytic J2 short-period amplitude for the energy check.
# A single sampling interval can alias the whole peak-to-peak swing, so the
# theoretical worst case is one full amplitude; 2x leaves headroom for the ~20%
# by which the measured swing exceeds the first-order term in eccentric orbits.
ENERGY_J2_SAFETY_FACTOR = 2.0

# Safety factor for the perigee trend check. Larger than the energy factor
# because perigee altitude carries the oscillation in BOTH a and e, and the
# measured peak-to-peak in perigee runs about 1.2x the amplitude in a alone.
PERIGEE_J2_SAFETY_FACTOR = 3.0

_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """The verdict of one invariant, with the numbers that produced it.

    ``measured_value`` and ``threshold`` are carried so a report can show WHY a
    check passed or failed rather than only that it did. They are ``None`` for
    checks whose violation is categorical (a non-finite sample, say) and has no
    meaningful scalar to compare.

    ``severity`` separates "this run is not trustworthy" (``error``) from "this
    is a finding you must look at, and the run is still usable" (``warning``).
    """

    name: str
    passed: bool
    detail: str
    severity: Severity = "error"
    measured_value: float | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CheckOutcome.name must be a non-empty identifier")
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"CheckOutcome.severity must be one of {VALID_SEVERITIES}, "
                f"got {self.severity!r}"
            )


def _j2_short_period_relative_amplitude(semi_major_axis_m: float) -> float:
    """Peak-to-peak J2 short-period swing in osculating ``a``, as a fraction of ``a``.

    First-order Brouwer short-period term, retaining only the magnitude:
    ``delta_a / a ~ 3 * J2 * (Re / a)**2``. The inclination- and
    anomaly-dependent factors are of order unity and are absorbed into the
    caller's safety factor rather than carried here, because the purpose is a
    tolerance and not a prediction.
    """
    if not math.isfinite(semi_major_axis_m) or semi_major_axis_m <= 0.0:
        raise ValueError(
            f"semi-major axis must be positive and finite, got {semi_major_axis_m}"
        )
    return 3.0 * J2_EARTH.value * (R_EARTH_EQUATORIAL.value / semi_major_axis_m) ** 2


def _specific_energy(semi_major_axis_m: float) -> float:
    """Specific orbital energy ``-mu / (2a)`` in J/kg."""
    return -MU_EARTH.value / (2.0 * semi_major_axis_m)


def _orbital_period_days(semi_major_axis_m: float) -> float:
    """Keplerian period in days, used to size the short-period oscillation."""
    return (
        2.0
        * math.pi
        * math.sqrt(semi_major_axis_m**3 / MU_EARTH.value)
        / _SECONDS_PER_DAY
    )


def _least_squares_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Ordinary least-squares slope dy/dx, summed with ``math.fsum``.

    Compensated summation matters here: perigee altitudes are ~1e5 m while the
    trend being fitted is ~1e3 m/day, so naive accumulation over thousands of
    samples loses the signal in the mean.
    """
    n = len(x)
    x_mean = math.fsum(x) / n
    y_mean = math.fsum(y) / n
    sxy = math.fsum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y, strict=True))
    sxx = math.fsum((xi - x_mean) ** 2 for xi in x)
    if sxx <= 0.0:
        raise ValueError("cannot fit a trend: every sample shares the same time")
    return sxy / sxx


def _count_unusable(values: Sequence[float]) -> int:
    """Number of samples that are not finite."""
    return sum(1 for v in values if not math.isfinite(v))


def check_energy_monotonic(
    result: PropagationResult,
    *,
    j2_safety_factor: float = ENERGY_J2_SAFETY_FACTOR,
) -> CheckOutcome:
    """Specific orbital energy must never increase under drag alone.

    Drag is dissipative: it can only remove energy from the orbit. An energy
    RISE therefore cannot be physics in this model — the propagator carries only
    two-body gravity, J2 and drag, and none of those adds energy secularly. A
    rise means a sign error in the drag term or an integration that has lost
    control of the step size, and either invalidates every number downstream.

    Energy is taken as ``-mu / (2a)`` from the osculating semi-major axis, which
    is the same quantity the propagator derived ``a`` from, so no independent
    approximation is introduced.

    The tolerance is the J2 short-period peak-to-peak swing at the run's mean
    semi-major axis, times ``j2_safety_factor``. A single sampling interval can
    land anywhere in that oscillation, so both the sample-to-sample rise and the
    first-to-last change can be contaminated by the full swing. The reported
    ``measured_value`` is the largest energy GAIN found either way — whichever of
    the net change over the run and the largest single-sample rise is greater —
    because a slow spurious gain shows up in the first and a broken step in the
    second, and both must be caught.
    """
    sma = result.semi_major_axis_m
    if len(sma) < 2:
        raise ValueError(
            "check_energy_monotonic needs at least two samples to compare; "
            f"the result carries {len(sma)}"
        )

    unusable = sum(1 for a in sma if not math.isfinite(a) or a <= 0.0)
    if unusable:
        return CheckOutcome(
            name="energy_monotonic",
            passed=False,
            detail=(
                f"{unusable} of {len(sma)} semi-major-axis samples are not finite "
                "and positive, so specific orbital energy is undefined there. The "
                "state left the bound-orbit regime, which is an integration "
                "failure and not a result."
            ),
            severity="error",
            measured_value=float(unusable),
            threshold=0.0,
        )

    energies = [_specific_energy(a) for a in sma]
    mean_sma = math.fsum(sma) / len(sma)
    tolerance = (
        j2_safety_factor
        * _j2_short_period_relative_amplitude(mean_sma)
        * abs(_specific_energy(mean_sma))
    )

    net_change = energies[-1] - energies[0]
    largest_step_rise = max(
        later - earlier for earlier, later in zip(energies[:-1], energies[1:], strict=True)
    )
    gain = max(net_change, largest_step_rise)
    passed = gain <= tolerance

    verdict = "no unexplained energy gain" if passed else "ENERGY GAINED"
    detail = (
        f"{verdict}: net change {net_change:+.6g} J/kg over the run, largest "
        f"single-sample rise {largest_step_rise:+.6g} J/kg, against a J2 "
        f"short-period tolerance of {tolerance:.6g} J/kg "
        f"({j2_safety_factor:g}x the peak-to-peak swing at a mean semi-major "
        f"axis of {mean_sma / 1000.0:.1f} km)."
    )
    if not passed:
        detail += (
            " Drag cannot add energy, so this is a sign error in the drag term "
            "or a failed integration, not a physical result."
        )
    return CheckOutcome(
        name="energy_monotonic",
        passed=passed,
        detail=detail,
        severity="error",
        measured_value=gain,
        threshold=tolerance,
    )


def check_perigee_monotonic(
    result: PropagationResult,
    *,
    j2_safety_factor: float = PERIGEE_J2_SAFETY_FACTOR,
) -> CheckOutcome:
    """Perigee altitude must not systematically rise.

    Stated as a TREND rather than a per-sample rule on purpose. Osculating
    perigee altitude swings by tens of kilometres over every orbit under J2, so
    "perigee never rises between two samples" is false for correct runs and
    would be a useless check. What cannot happen under gravity, J2 and drag is a
    secular rise: nothing in the model puts energy or angular momentum back.

    The trend is an ordinary least-squares slope in metres per day. Its
    tolerance is derived rather than picked: for a sinusoid of amplitude ``A``
    and period ``T`` sampled over a window ``D``, the least-squares slope is
    bounded by ``(12 / 2*pi) * A * T / D**2`` — integrate ``(t - D/2) * sin`` by
    parts for the numerator and use ``D**3 / 12`` for the denominator. Taking
    ``A`` as the J2 short-period amplitude gives the largest slope a correct run
    can fake, and it falls off as ``1 / D**2``, so a long run is judged much more
    strictly than a short one. That is the right behaviour: half an orbit
    genuinely cannot establish a trend.

    The reference semi-major axis is reconstructed from the perigee and apogee
    altitude series themselves — their mean is ``a - Re`` — so this check depends
    only on the two series it is about.
    """
    perigee = result.perigee_altitude_m
    apogee = result.apogee_altitude_m
    times = result.times_utc
    if len(perigee) < 3:
        raise ValueError(
            "check_perigee_monotonic needs at least three samples to fit a trend; "
            f"the result carries {len(perigee)}"
        )
    if len(times) != len(perigee) or len(apogee) != len(perigee):
        raise ValueError(
            "times, perigee and apogee series must be the same length; got "
            f"{len(times)}, {len(perigee)}, {len(apogee)}"
        )

    unusable = _count_unusable(perigee) + _count_unusable(apogee)
    if unusable:
        return CheckOutcome(
            name="perigee_monotonic",
            passed=False,
            detail=(
                f"{unusable} perigee/apogee samples are not finite, so no trend "
                "can be fitted. The element series broke down during the run, "
                "which is an integration failure and not a result."
            ),
            severity="error",
            measured_value=float(unusable),
            threshold=0.0,
        )

    days = [(t - times[0]).total_seconds() / _SECONDS_PER_DAY for t in times]
    duration_days = days[-1] - days[0]
    if duration_days <= 0.0:
        raise ValueError(
            "check_perigee_monotonic needs a positive time span; the samples "
            f"span {duration_days} days"
        )

    mean_perigee = math.fsum(perigee) / len(perigee)
    mean_apogee = math.fsum(apogee) / len(apogee)
    mean_sma = 0.5 * (mean_perigee + mean_apogee) + R_EARTH_EQUATORIAL.value
    amplitude_m = 0.5 * _j2_short_period_relative_amplitude(mean_sma) * mean_sma
    period_days = _orbital_period_days(mean_sma)
    tolerance = (
        j2_safety_factor
        * (12.0 / (2.0 * math.pi))
        * amplitude_m
        * period_days
        / (duration_days * duration_days)
    )

    slope = _least_squares_slope(days, perigee)
    passed = slope <= tolerance

    verdict = "no systematic perigee rise" if passed else "PERIGEE RISING"
    detail = (
        f"{verdict}: least-squares trend {slope:+.6g} m/day over "
        f"{duration_days:.3f} days, against a tolerance of {tolerance:+.6g} m/day "
        f"({j2_safety_factor:g}x the largest slope a J2 short-period oscillation "
        f"of {amplitude_m:.0f} m amplitude can fake over this window)."
    )
    if not passed:
        detail += (
            " Gravity, J2 and drag cannot raise perigee secularly; no thrust, "
            "third body or radiation pressure is modelled."
        )
    return CheckOutcome(
        name="perigee_monotonic",
        passed=passed,
        detail=detail,
        severity="error",
        measured_value=slope,
        threshold=tolerance,
    )


def check_density_positive(result: PropagationResult) -> CheckOutcome:
    """Every finite density sample must be strictly positive.

    Zero or negative mass density is not a small numerical blemish: drag scales
    linearly with it, so a negative sample means the vehicle was accelerated by
    the atmosphere over that interval.

    Non-finite samples are SKIPPED rather than failed, because the propagator
    writes NaN by design below the reentry floor where the density model is out
    of its domain. A NaN there is an honest statement that no density was
    evaluated. Their count is reported so the skip is visible; a series with no
    finite sample at all fails, since then nothing was checked.
    """
    densities = result.density_kg_m3
    if not densities:
        raise ValueError("check_density_positive needs at least one density sample; got none")

    finite = [d for d in densities if math.isfinite(d)]
    skipped = len(densities) - len(finite)
    if not finite:
        return CheckOutcome(
            name="density_positive",
            passed=False,
            detail=(
                f"all {len(densities)} density samples are non-finite, so the "
                "invariant was never actually tested. Treat this as a failure "
                "rather than a vacuous pass."
            ),
            severity="error",
            measured_value=None,
            threshold=0.0,
        )

    violations = [d for d in finite if d <= 0.0]
    minimum = min(finite)
    passed = not violations

    coverage = (
        f"{len(finite)} finite samples, {skipped} non-finite samples skipped: the "
        "propagator writes NaN below the reentry floor, where the density model "
        "has no domain"
    )
    if passed:
        detail = f"minimum finite density {minimum:.6g} kg/m^3 ({coverage})."
    else:
        detail = (
            f"{len(violations)} density samples are non-positive, the smallest "
            f"being {minimum:.6g} kg/m^3. Drag scales linearly with density, so a "
            f"non-positive sample accelerates the vehicle ({coverage})."
        )
    return CheckOutcome(
        name="density_positive",
        passed=passed,
        detail=detail,
        severity="error",
        measured_value=minimum,
        threshold=0.0,
    )


def check_altitude_bounds(
    result: PropagationResult, floor_m: float, ceiling_m: float
) -> CheckOutcome:
    """Every altitude sample must lie inside an explicit band.

    The band is a parameter, not a constant, because what counts as out of
    bounds depends on the question. For a decay run the floor is the reentry
    threshold — below it the atmosphere model is outside its domain and any
    density it returns is extrapolation — and the ceiling catches a state that
    has been flung outward by a bad step long before it becomes obvious.
    """
    if not math.isfinite(floor_m) or not math.isfinite(ceiling_m):
        raise ValueError(
            f"altitude bounds must be finite, got floor={floor_m}, ceiling={ceiling_m}"
        )
    if floor_m >= ceiling_m:
        raise ValueError(
            f"altitude floor must be below the ceiling, got floor={floor_m} m "
            f"and ceiling={ceiling_m} m"
        )
    altitudes = result.altitudes_m
    if not altitudes:
        raise ValueError("check_altitude_bounds needs at least one altitude sample; got none")

    unusable = _count_unusable(altitudes)
    if unusable:
        return CheckOutcome(
            name="altitude_bounds",
            passed=False,
            detail=(
                f"{unusable} of {len(altitudes)} altitude samples are not finite. "
                "A non-finite position is an integration failure, not an "
                "out-of-band trajectory."
            ),
            severity="error",
            measured_value=float(unusable),
            threshold=0.0,
        )

    lowest = min(altitudes)
    highest = max(altitudes)
    below = sum(1 for h in altitudes if h < floor_m)
    above = sum(1 for h in altitudes if h > ceiling_m)

    if not below and not above:
        return CheckOutcome(
            name="altitude_bounds",
            passed=True,
            detail=(
                f"all {len(altitudes)} samples lie within "
                f"{floor_m / 1000.0:.1f}-{ceiling_m / 1000.0:.1f} km; the run "
                f"spans {lowest / 1000.0:.1f}-{highest / 1000.0:.1f} km."
            ),
            severity="error",
            measured_value=lowest,
            threshold=floor_m,
        )

    # Report the end that was breached hardest, so the number shown is the one
    # that matters. Both counts appear in the detail regardless.
    below_excursion = floor_m - lowest if below else 0.0
    above_excursion = highest - ceiling_m if above else 0.0
    if below_excursion >= above_excursion:
        measured, threshold, edge = lowest, floor_m, "below the floor"
    else:
        measured, threshold, edge = highest, ceiling_m, "above the ceiling"

    return CheckOutcome(
        name="altitude_bounds",
        passed=False,
        detail=(
            f"{below} sample(s) below the {floor_m / 1000.0:.1f} km floor and "
            f"{above} above the {ceiling_m / 1000.0:.1f} km ceiling; the run "
            f"spans {lowest / 1000.0:.1f}-{highest / 1000.0:.1f} km. The worst "
            f"excursion is {edge}."
        ),
        severity="error",
        measured_value=measured,
        threshold=threshold,
    )


def check_species_closure(
    number_densities: NumberDensities, model_total: float, tolerance: float
) -> CheckOutcome:
    """Compare the species sum against the model's own total mass density.

    A FAILURE HERE IS A FINDING ABOUT THE DATA, NOT AN ERROR TO CORRECT. The two
    quantities are allowed to differ, and the difference carries information:
    the provider may withhold species, account anomalous oxygen separately, or
    include constituents it does not enumerate. The correct response is to
    report the residual and its sign and to state which of the two densities the
    analysis actually used. Rescaling either number to force closure would
    destroy exactly the diagnostic this check exists to surface, which is why
    the severity is ``warning``: the run remains usable, and the residual must
    travel with it.
    """
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"closure tolerance must be finite and non-negative, got {tolerance}")

    fraction = density_closure_fraction(number_densities, model_total)
    species_sum = total_species_mass_density(number_densities)
    # Species with a None value are MISSING, not zero, and are not counted as
    # supplied — the distinction is what makes the residual interpretable.
    supplied = sum(1 for value in number_densities.values() if value is not None)
    passed = abs(fraction) <= tolerance

    detail = (
        f"species sum {species_sum:.6e} kg/m^3 differs from the model total "
        f"{float(model_total):.6e} kg/m^3 by {fraction:+.4%} "
        f"(tolerance +/-{tolerance:.4%}, {supplied} species supplied)."
    )
    if passed:
        detail += " The two densities close to within the stated tolerance."
    else:
        detail += (
            " THIS IS A FINDING ABOUT THE DATA, NOT AN ERROR TO CORRECT: report "
            "the residual and its sign, and state which density the analysis "
            "used. Do not reconcile it by rescaling either quantity."
        )
    return CheckOutcome(
        name="species_closure",
        passed=passed,
        detail=detail,
        severity="warning",
        measured_value=fraction,
        threshold=tolerance,
    )


def run_all_checks(
    result: PropagationResult,
    *,
    floor_m: float = REENTRY_ALTITUDE_M,
    ceiling_m: float = MAX_ALTITUDE_M,
) -> tuple[CheckOutcome, ...]:
    """Run every invariant that a ``PropagationResult`` alone can answer.

    Species closure is deliberately absent: it needs a model's own total mass
    density, which a propagation result does not carry. Including it with an
    invented total would manufacture a diagnostic rather than report one.

    The altitude band defaults to the propagator's own reentry floor and escape
    ceiling, so the default screening asks exactly the question the integrator's
    terminating events ask.
    """
    return (
        check_energy_monotonic(result),
        check_perigee_monotonic(result),
        check_density_positive(result),
        check_altitude_bounds(result, floor_m, ceiling_m),
    )


def summarise(outcomes: Sequence[CheckOutcome]) -> str:
    """Render outcomes as a plain-text verdict, worst news first.

    The leading line states whether the run may be believed, because that is the
    only sentence a reader is guaranteed to read.
    """
    if not outcomes:
        raise ValueError("summarise() needs at least one outcome; none were supplied")

    failed_errors = [o for o in outcomes if not o.passed and o.severity == "error"]
    failed_warnings = [o for o in outcomes if not o.passed and o.severity == "warning"]
    passed = [o for o in outcomes if o.passed]

    if failed_errors:
        headline = (
            f"VERDICT: DO NOT BELIEVE THIS RUN - {len(failed_errors)} of "
            f"{len(outcomes)} invariants failed at error severity."
        )
    elif failed_warnings:
        headline = (
            f"VERDICT: invariants hold, but {len(failed_warnings)} finding(s) are "
            "recorded and must travel with the result."
        )
    else:
        headline = f"VERDICT: all {len(outcomes)} invariants hold."

    lines = [
        headline,
        (
            f"{len(passed)} passed, {len(failed_errors)} failed (error), "
            f"{len(failed_warnings)} failed (warning)."
        ),
    ]
    ordered = [*failed_errors, *failed_warnings, *passed]
    for outcome in ordered:
        status = "PASS" if outcome.passed else "FAIL"
        lines.append(f"  [{status} {outcome.severity}] {outcome.name}: {outcome.detail}")
    return "\n".join(lines)
