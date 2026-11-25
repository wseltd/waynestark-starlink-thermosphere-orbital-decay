"""Propagating input uncertainty through to the four headline numbers.

``uq.sensitivity`` answers "which unmeasured input is the answer most sensitive
to?". It does not answer "so what is the answer, then?" — it returns variance
shares, not bounds on the reported quantity. This module answers the second
question: it takes the same justified input ranges, pushes them through the
whole decay chain, and reports for each headline quantity a minimum, a nominal
value, a maximum, and the input configuration that produced each extreme.

WHAT THESE NUMBERS ARE, AND THE ONE THING THEY ARE NOT
-----------------------------------------------------
Every range this module produces is a SENSITIVITY RANGE: the span the quantity
takes as the inputs move over ranges whose provenance is recorded beside them.
It is the answer to "how far does this move if the inputs move over the box we
are prepared to defend?".

It is NOT a probability statement of any kind. Turning it into one would need a
joint distribution over the inputs, and there is not one: the input ranges come
from a third-party mass estimate, a quoting precision, a counterfactual choice
and a published model spread. Putting a density over that box would be
inventing the very thing the range exists to avoid claiming. So the extremes
reported here are the extremes OF THE SAMPLED BOX, they carry no coverage
probability, and :data:`SENSITIVITY_RANGE_CAVEAT` says so on every artefact.
:func:`render_markdown` refuses to emit a report that has lost that sentence.

WHAT IS SWEPT, AND WHAT TURNS OUT NOT TO MATTER
-----------------------------------------------
:class:`UncertaintyCase` carries every swept input. Three of them are worth
flagging in advance because their effect is structural rather than numerical:

  * **Mass, projected area and drag coefficient cancel.** The trajectory
    depends on the inverse ballistic coefficient C_d*A/m and on nothing else
    about the vehicle: :meth:`SpacecraftProperties.inverse_ballistic_coefficient`
    is the only place they enter the equations of motion. So a 275-303 kg mass
    range moves the decay time by exactly zero at fixed C_d*A/m, and it moves
    the *implied area* by 10%. This is not a null result to be buried; it is
    the identifiability statement in ``domain.drag_parameters`` showing up as a
    number, and :func:`degeneracy_evidence` measures it rather than asserting it.

  * **Neutral wind enters as an equivalent drag scale.** The propagator has no
    wind term, and this module does not add one. Drag depends on the relative
    speed through ``B * rho * |v_rel| * v_rel``, so a constant wind is
    equivalent, to first order in wind/orbital speed, to multiplying C_d*A/m by
    the mean of ``|v_rel,wind|^2 / |v_rel,0|^2`` around the orbit.
    :func:`wind_equivalent_drag_scale` computes that mean from
    ``providers.wind.relative_velocity_with_wind`` on the actual insertion
    orbit. It is an approximation, it is recorded as one, and its direct
    consequence — that the wind cancels out of the storm-cost DIFFERENCE
    because it scales the storm and quiet branches identically — is reported
    rather than left implicit.

  * **The published-envelope provider cannot propagate, and is right not to.**
    ``providers.published_envelope`` refuses altitudes and epochs its sources
    do not cover, and a decay run leaves both within hours. So the
    atmosphere-model branch splits in two: the published bands are compared
    against the peak-enhancement quantity at 210 km inside their window, where
    they are valid, and the decay quantities use the scalar stress test with
    its prohibition on being labelled with a physics model's name.
    :func:`published_envelope_propagation_refusal` records the actual exception
    text, so "we could not use it" is evidence and not a claim.

Nearest canonical ticket: P06_08_T039, which scoped the uncertainty stage and
left execution out. ``uq.sensitivity`` is the variance-decomposition half of
what that ticket anticipated; this is the interval-reporting half.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import functools
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from ..drivers.space_weather import F107Series, KpApSeries, drivers_at
from ..events import starlink_g4_7 as event
from ..molecular import composition as comp
from ..propagation import propagator as prop
from ..propagation.propagator import (
    DragPropagator,
    OrbitState,
    SpacecraftProperties,
    greenwich_sidereal_time,
    state_from_perigee_apogee,
)
from ..protocols.atmosphere import AtmosphereProvider
from ..providers import published_envelope as envelope
from ..providers.scaled_storm import ScalarDensityStressTestProvider
from ..providers.wind import relative_velocity_with_wind
from ..scenarios import ballistic_inversion as inversion
from ..scenarios.fixed_path import quiet_baseline_drivers
from ..scenarios.starlink_analysis import AtmosphereDensityFunction
from .sensitivity import UncertainParameter

__all__ = [
    "DECAY_DATE_CONVENTIONS",
    "DECAY_GRID_M2_KG",
    "DECAY_SOURCE_DIRECT",
    "DECAY_SOURCE_INTERPOLATED",
    "ENHANCEMENT_PARAMETERS",
    "NOMINAL_CASE",
    "PROPAGATION_PARAMETERS",
    "SENSITIVITY_RANGE_CAVEAT",
    "CaseOutcome",
    "RangeSample",
    "SensitivityRange",
    "UncertaintyCase",
    "case_from_sample",
    "curve_from_rows",
    "decay_days_at",
    "decay_point",
    "degeneracy_evidence",
    "nominal_sample_row",
    "observed_decay_targets",
    "outcome_samples",
    "outcomes_from_curves",
    "peak_density_enhancement",
    "published_envelope_enhancements",
    "published_envelope_propagation_refusal",
    "reentry_threshold_m",
    "render_markdown",
    "sample_row",
    "sensitivity_range",
    "wind_equivalent_drag_scale",
]

#: Printed on every artefact this module produces, and required by
#: :func:`render_markdown` to be present in the rendered text. The phrasing is
#: deliberate: it states what the range IS rather than only what it is not, so
#: that a reader who skims still comes away with the right object.
SENSITIVITY_RANGE_CAVEAT = (
    "SENSITIVITY RANGE, NOT A PROBABILITY STATEMENT. Each minimum and maximum "
    "below is the extreme this quantity reached over the sampled box of input "
    "ranges recorded beside it. No probability distribution has been justified "
    "for any of those inputs — the ranges come from a third-party mass "
    "estimate, the quoting precision of published elements, a counterfactual "
    "baseline choice and the spread between published storm-response models — "
    "so these bounds carry NO coverage probability and must never be reported "
    "as though they did. A wider range here means the answer depends more on "
    "choices nobody has measured, not that the answer is known to lie within "
    "it with any stated likelihood."
)

#: Grid of inverse ballistic coefficients (C_d*A/m, m^2/kg) at which the forward
#: decay curve is computed before it is inverted for the observed decay dates.
#:
#: Chosen to BRACKET the catalogued decay window with margin under every case in
#: the sweep, not to be economical: the fastest catalogued decay is 3 d and the
#: slowest 13 d, and the grid spans roughly 20 d down to 1.8 d under nominal
#: inputs. An inversion target outside the computed range is not interpolated,
#: it is dropped with a note by ``ballistic_inversion.invert_decay``, and a
#: dropped target would silently bias the median — so the margin is the point.
#:
#: Interpolation between grid points is log-log linear, which is near-exact
#: here: decay time goes as C_d*A/m to the power -1.035 over this span, so
#: subsampling the 28-point curve committed in ``evidence/`` down to a grid of
#: this size moves the median inferred value by a few tenths of a percent. That
#: figure is measured, not assumed, in ``tests/uq/test_propagation.py``.
#:
#: A few tenths of a percent is fine for the fitted C_d*A/m and NOT fine for the
#: storm cost, which is a ~0.47 d DIFFERENCE between two ~7.5 d lifetimes: a
#: 0.5% error on each branch is a 15% error on their difference. That is why
#: :func:`outcomes_from_curves` accepts directly propagated lifetimes and why
#: the sweep supplies them — the curve locates the fitted parameter, and two
#: further propagations at that parameter measure the cost without
#: interpolating it.
DECAY_GRID_M2_KG: tuple[float, ...] = (
    0.0095,
    0.01661,
    0.02905,
    0.05081,
    0.08885,
    0.15537,
)

#: The two defensible ways to turn a catalogued decay DATE into an elapsed time,
#: kept as an explicit categorical branch because they differ by 0.77 d against
#: a 3-13 d observable and the committed analysis uses the first.
#:
#:   ``days_from_launch_date``: midnight at the start of the launch date to the
#:     recorded decay date. This is ``GroundTruthDecay.days_to_decay`` and it is
#:     what produced every number in ``evidence/analysis_report.json``.
#:   ``days_from_deployment``: deployment epoch (2022-02-03 18:28 UTC, the
#:     propagation's own initial condition) to the recorded decay date. This is
#:     the quantity a propagated lifetime should actually be compared against.
DECAY_DATE_CONVENTIONS: tuple[str, str] = (
    "days_from_launch_date",
    "days_from_deployment",
)

_SECONDS_PER_DAY = 86_400.0

# Longitude at which the density field is sampled for the peak-enhancement
# quantity, matching scripts/run_full_analysis.py so the nominal case of this
# module reproduces the committed figure rather than a near neighbour of it.
_ENHANCEMENT_LONGITUDE_DEG = 0.0
_ENHANCEMENT_START_UTC = _dt.datetime(2022, 2, 2, tzinfo=_dt.UTC)
_ENHANCEMENT_END_UTC = _dt.datetime(2022, 2, 8, tzinfo=_dt.UTC)

#: True anomalies at which the wind-equivalent drag scale is averaged. Twenty-
#: four points around the orbit: the projection of a zonal wind onto the
#: along-track direction varies between cos(53.22 deg) and 1 over a revolution,
#: so a single-point evaluation would misstate the scale by up to 40%.
_WIND_AVERAGING_ANOMALIES_RAD: tuple[float, ...] = tuple(
    float(x) for x in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
)


# ---------------------------------------------------------------------------
# The swept input set
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UncertaintyCase:
    """One fully specified input configuration for the decay chain.

    Every field defaults to the value the committed analysis used, so
    ``UncertaintyCase(case_id="nominal")`` is the reference run and any other
    case is describable as "the reference, with these fields changed". That
    matters for reporting: the configuration that produced an extreme has to be
    quotable in a sentence, and :meth:`differences_from_nominal` produces it.
    """

    case_id: str
    perigee_altitude_m: float = event.INSERTION_PERIGEE_ALTITUDE_M
    apogee_altitude_m: float = event.INSERTION_APOGEE_ALTITUDE_M
    inclination_deg: float = event.INSERTION_INCLINATION_DEG
    true_anomaly_deg: float = 0.0
    quiet_ap: float = 4.0
    storm_density_factor: float = 1.0
    decay_date_offset_days: float = 0.0
    decay_date_convention: str = DECAY_DATE_CONVENTIONS[0]
    mass_kg: float = event.MASS_KG_NOMINAL
    projected_area_m2: float | None = None
    drag_coefficient: float = 2.4
    wind_east_m_s: float = 0.0
    reentry_threshold_m: float = prop.REENTRY_ALTITUDE_M
    rtol: float = 1e-9
    atol: float = 1e-6
    max_step_s: float = 90.0
    sample_interval_s: float = 3600.0
    duration_days: float = 45.0
    thrust_delta_v_m_s: float = 0.0
    thrust_epoch_days: float = 0.0
    # The point at which the density-field diagnostic is evaluated. Carried on
    # the case rather than passed separately so that a reported extreme of the
    # peak-enhancement quantity names the altitude and latitude that produced
    # it, in the same object as everything else that produced it.
    enhancement_altitude_m: float = 210_000.0
    enhancement_latitude_deg: float = event.INSERTION_INCLINATION_DEG

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be a non-empty label; it names the run in the report")
        if self.apogee_altitude_m < self.perigee_altitude_m:
            raise ValueError(
                f"{self.case_id}: apogee {self.apogee_altitude_m} m is below perigee "
                f"{self.perigee_altitude_m} m"
            )
        if self.decay_date_convention not in DECAY_DATE_CONVENTIONS:
            raise ValueError(
                f"{self.case_id}: decay_date_convention must be one of "
                f"{DECAY_DATE_CONVENTIONS}, got {self.decay_date_convention!r}"
            )
        if self.reentry_threshold_m <= 0.0:
            raise ValueError(
                f"{self.case_id}: reentry threshold must be positive, got "
                f"{self.reentry_threshold_m} m"
            )
        if self.thrust_delta_v_m_s < 0.0:
            raise ValueError(
                f"{self.case_id}: thrust_delta_v_m_s must be non-negative; the sweep "
                "models an along-track orbit-raising impulse, and a negative value "
                "would be a de-orbit burn nobody has alleged"
            )
        if self.thrust_delta_v_m_s > 0.0 and self.thrust_epoch_days <= 0.0:
            raise ValueError(
                f"{self.case_id}: a thrust case must place the impulse strictly after "
                f"deployment, got thrust_epoch_days={self.thrust_epoch_days}"
            )
        if self.thrust_epoch_days >= self.duration_days:
            raise ValueError(
                f"{self.case_id}: thrust epoch {self.thrust_epoch_days} d is not inside "
                f"the {self.duration_days} d propagation window"
            )
        for name in ("rtol", "atol", "max_step_s", "sample_interval_s", "duration_days"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{self.case_id}: {name} must be positive")

    def differences_from_nominal(self) -> dict[str, Any]:
        """The fields that differ from the committed reference configuration.

        This, not the whole field set, is what a report should quote beside an
        extreme: "the maximum arose at storm_density_factor=3.5, quiet_ap=2.0"
        is readable, and a twenty-field dump is not.
        """
        reference = asdict(replace(NOMINAL_CASE, case_id=self.case_id))
        mine = asdict(self)
        return {key: mine[key] for key in mine if mine[key] != reference[key] and key != "case_id"}

    def initial_state(self) -> OrbitState:
        """The deployment-epoch Cartesian state implied by this case's elements."""
        position, velocity = state_from_perigee_apogee(
            self.perigee_altitude_m,
            self.apogee_altitude_m,
            math.radians(self.inclination_deg),
            true_anomaly_rad=math.radians(self.true_anomaly_deg),
        )
        return OrbitState(
            event.DEPLOYMENT_UTC,
            (float(position[0]), float(position[1]), float(position[2])),
            (float(velocity[0]), float(velocity[1]), float(velocity[2])),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready field mapping, including the swept-field summary."""
        payload: dict[str, Any] = asdict(self)
        payload["differences_from_nominal"] = self.differences_from_nominal()
        return payload


#: The committed reference configuration: every field at the value that produced
#: ``evidence/analysis_report.json``.
NOMINAL_CASE = UncertaintyCase(case_id="nominal")


# ---------------------------------------------------------------------------
# Justified ranges, reusing the sensitivity layer's gate on unjustified ones
# ---------------------------------------------------------------------------

#: Continuous inputs entered into the variance decomposition of the DECAY-based
#: quantities. Each one is an :class:`~.sensitivity.UncertainParameter`, so the
#: constructor gate applies and no range reaches a sampler without its reason.
#:
#: Mass, area, drag coefficient and accommodation are deliberately ABSENT. They
#: are swept — see :func:`degeneracy_evidence` and the categorical cases — but
#: they cannot enter a variance decomposition of decay time, because they cancel
#: exactly out of the equations of motion at fixed C_d*A/m and would contribute
#: a structural zero that a reader could mistake for "measured to be
#: unimportant".
PROPAGATION_PARAMETERS: tuple[UncertainParameter, ...] = (
    UncertainParameter(
        name="perigee_altitude_km",
        low=208.0,
        high=212.0,
        units="km",
        justification=(
            "Insertion perigee published as 210 km with NO formal uncertainty "
            "(event manifest insertion_perigee_altitude notes; Guarnieri et al. 2025, "
            "doi:10.5194/npg-32-75-2025). +/-2 km spans the quoting precision of a "
            "value given to the kilometre plus the dispersion of a single-burn LEO "
            "insertion. ASSUMED range, not a measured bound."
        ),
    ),
    UncertainParameter(
        name="apogee_altitude_km",
        low=335.0,
        high=341.0,
        units="km",
        justification=(
            "Insertion apogee published as 338 km with no formal uncertainty "
            "(Guarnieri et al. 2025). +/-3 km, wider than the perigee band because "
            "apogee dispersion of an insertion burn exceeds perigee dispersion. "
            "ASSUMED range."
        ),
    ),
    UncertainParameter(
        name="inclination_deg",
        low=53.12,
        high=53.32,
        units="degrees",
        justification=(
            "Insertion inclination published as 53.22 deg (Guarnieri et al. 2025), "
            "quoted to 0.01 deg. +/-0.1 deg covers the quoting precision and launch "
            "dispersion; it changes which latitudes the low passes sample. ASSUMED range."
        ),
    ),
    UncertainParameter(
        name="true_anomaly_deg",
        low=0.0,
        high=360.0,
        units="degrees",
        justification=(
            "The orbital phase at deployment is NOT published. The full circle is "
            "therefore the honest range: it decides whether the vehicle starts at "
            "perigee, at apogee or between, and hence how deep its first passes are."
        ),
    ),
    UncertainParameter(
        name="quiet_ap",
        low=2.0,
        high=7.0,
        units="nT",
        justification="counterfactual baseline choice, Kp 0+ to 2-",
    ),
    UncertainParameter(
        name="storm_density_factor",
        low=1.0,
        high=3.5,
        units="dimensionless",
        justification=(
            "He et al. 2023: MSIS ~20-25% storm response vs WAM-IPE ~40% and MAGE "
            "~57-75% at 200/300 km. Applied by the SCALAR STRESS TEST provider; a "
            "factor of 2.0 is NOT a WAM-IPE result and 3.5 is NOT a MAGE result."
        ),
    ),
    UncertainParameter(
        name="decay_date_offset_days",
        low=-0.5,
        high=0.5,
        units="days",
        justification=(
            "Catalogue decay dates carry DATE resolution only "
            f"(events.starlink_g4_7.DECAY_DATE_RESOLUTION_S = "
            f"{event.DECAY_DATE_RESOLUTION_S:g} s), so the decay instant is somewhere "
            "inside the recorded day and the elapsed time is known to +/-0.5 d about "
            "the midpoint of that day. Interpolated from the constant rather than "
            "restated, so a rename cannot leave this justification citing a symbol "
            "that no longer exists."
        ),
    ),
)

#: Continuous inputs for the PEAK DENSITY ENHANCEMENT quantity, which is a
#: density-field diagnostic rather than a trajectory and is therefore cheap
#: enough to decompose at a design size where the indices actually converge.
ENHANCEMENT_PARAMETERS: tuple[UncertainParameter, ...] = (
    UncertainParameter(
        name="altitude_km",
        low=200.0,
        high=220.0,
        units="km",
        justification=(
            "The enhancement is quoted at the 210 km staging altitude; +/-10 km "
            "covers the published insertion perigee's quoting precision and the "
            "altitude band over which the staging orbit's low passes actually sit."
        ),
    ),
    UncertainParameter(
        name="latitude_deg",
        low=-53.22,
        high=53.22,
        units="degrees",
        justification=(
            "A 53.22 deg orbit samples every latitude between +/-53.22 deg. The "
            "committed figure is evaluated at +53.22 deg only, so the range is the "
            "latitudes the vehicle actually flew over."
        ),
    ),
    UncertainParameter(
        name="quiet_ap",
        low=2.0,
        high=7.0,
        units="nT",
        justification="counterfactual baseline choice, Kp 0+ to 2-",
    ),
    UncertainParameter(
        name="storm_density_factor",
        low=1.0,
        high=3.5,
        units="dimensionless",
        justification=(
            "He et al. 2023: MSIS ~20-25% storm response vs WAM-IPE ~40% and MAGE "
            "~57-75% at 200/300 km. SCALAR STRESS TEST factor, not a physics model."
        ),
    ),
)


def case_from_sample(
    values: Sequence[float], *, case_id: str, parameters: Sequence[UncertainParameter] = ()
) -> UncertaintyCase:
    """Build a case from one row of a SALib design.

    The parameter order is the design's column order, which is why the
    parameter set is passed in rather than assumed: SALib indexes columns by
    position, so a case builder that hard-coded the order would silently
    mis-assign every column the day a parameter is inserted.
    """
    chosen = tuple(parameters) or PROPAGATION_PARAMETERS
    if len(values) != len(chosen):
        raise ValueError(
            f"{case_id}: got {len(values)} sample values for {len(chosen)} parameters "
            f"{[p.name for p in chosen]}; the design and the case builder disagree"
        )
    named = dict(zip((p.name for p in chosen), (float(v) for v in values), strict=True))
    unknown = sorted(set(named) - _CASE_FIELD_FOR_PARAMETER.keys())
    if unknown:
        raise ValueError(
            f"{case_id}: no UncertaintyCase field is registered for parameter(s) "
            f"{unknown}; add it to _CASE_FIELD_FOR_PARAMETER rather than letting the "
            "sampled value be discarded silently"
        )
    overrides: dict[str, float] = {}
    for name, value in named.items():
        field_name, transform = _CASE_FIELD_FOR_PARAMETER[name]
        overrides[field_name] = transform(value)
    return replace(NOMINAL_CASE, case_id=case_id, **overrides)  # type: ignore[arg-type]


def nominal_sample_row(parameters: Sequence[UncertainParameter]) -> list[float]:
    """The committed reference configuration, expressed in the sampler's units.

    A one-at-a-time sweep needs to hold every parameter but one at its nominal
    value, and the sampler's units are not always the case field's units
    (kilometres against metres). Deriving the row from the case rather than
    writing it out twice is what keeps the two from drifting apart.
    """
    row: list[float] = []
    for parameter in parameters:
        if parameter.name not in _CASE_FIELD_FOR_PARAMETER:
            raise ValueError(
                f"no UncertaintyCase field is registered for parameter {parameter.name!r}"
            )
        field_name, _transform = _CASE_FIELD_FOR_PARAMETER[parameter.name]
        value = float(getattr(NOMINAL_CASE, field_name))
        row.append(value / 1000.0 if parameter.units == "km" else value)
    return row


def sample_row(
    overrides: Mapping[str, float], parameters: Sequence[UncertainParameter]
) -> list[float]:
    """A design row equal to the nominal one except for the named parameters."""
    names = [p.name for p in parameters]
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise ValueError(
            f"cannot override parameter(s) {unknown}: the design covers {names}"
        )
    row = nominal_sample_row(parameters)
    for name, value in overrides.items():
        row[names.index(name)] = float(value)
    return row


def _kilometres_to_metres(value: float) -> float:
    return value * 1000.0


def _identity(value: float) -> float:
    return value


#: Maps a SALib parameter name to the :class:`UncertaintyCase` field it sets and
#: the unit conversion between them. Held as data so that adding a swept
#: parameter is one entry here and one entry in the parameter tuple, and so that
#: a parameter with nowhere to go raises in :func:`case_from_sample` instead of
#: being sampled and thrown away.
_CASE_FIELD_FOR_PARAMETER: dict[str, tuple[str, Callable[[float], float]]] = {
    "perigee_altitude_km": ("perigee_altitude_m", _kilometres_to_metres),
    "apogee_altitude_km": ("apogee_altitude_m", _kilometres_to_metres),
    "inclination_deg": ("inclination_deg", _identity),
    "true_anomaly_deg": ("true_anomaly_deg", _identity),
    "quiet_ap": ("quiet_ap", _identity),
    "storm_density_factor": ("storm_density_factor", _identity),
    "decay_date_offset_days": ("decay_date_offset_days", _identity),
    "altitude_km": ("enhancement_altitude_m", _kilometres_to_metres),
    "latitude_deg": ("enhancement_latitude_deg", _identity),
}


# ---------------------------------------------------------------------------
# Neutral wind as an equivalent drag scale
# ---------------------------------------------------------------------------


def wind_equivalent_drag_scale(
    wind_east_m_s: float,
    *,
    case: UncertaintyCase = NOMINAL_CASE,
    wind_north_m_s: float = 0.0,
) -> float:
    """Factor by which a constant zonal wind multiplies drag at fixed C_d*A/m.

    Drag acceleration is ``-(1/2) * (C_d A/m) * rho * |v_rel| * v_rel``, so
    replacing the co-rotation-only relative velocity with the wind-corrected one
    scales the drag magnitude by ``|v_rel,wind|^2 / |v_rel,0|^2``. Averaged
    around the orbit, that single number is what a constant wind does to the
    trajectory, to first order in wind/orbital speed.

    The average is over true anomaly rather than taken at one point because the
    eastward unit vector projects onto the along-track direction with a factor
    between cos(53.22 deg) = 0.60 and 1.00 over a revolution.

    Two things this is NOT. It is not a wind model — ``providers.wind`` says so
    at length, and a constant wind is a sensitivity instrument. And it is not
    exact: it captures the change in relative SPEED and neglects the small
    rotation of the drag vector's direction, which enters at second order in
    wind/orbital speed (500/7400, so of order 0.5%).
    """
    if not math.isfinite(wind_east_m_s) or not math.isfinite(wind_north_m_s):
        raise ValueError(
            f"wind components must be finite, got east={wind_east_m_s}, north={wind_north_m_s}"
        )
    if wind_east_m_s == 0.0 and wind_north_m_s == 0.0:
        return 1.0

    gmst = greenwich_sidereal_time(event.DEPLOYMENT_UTC)
    ratios: list[float] = []
    for anomaly in _WIND_AVERAGING_ANOMALIES_RAD:
        position, velocity = state_from_perigee_apogee(
            case.perigee_altitude_m,
            case.apogee_altitude_m,
            math.radians(case.inclination_deg),
            true_anomaly_rad=anomaly,
        )
        still = relative_velocity_with_wind(velocity, position, 0.0, 0.0, gmst)
        blown = relative_velocity_with_wind(
            velocity, position, wind_east_m_s, wind_north_m_s, gmst
        )
        ratios.append(float(np.dot(blown, blown) / np.dot(still, still)))
    return float(np.mean(ratios))


# ---------------------------------------------------------------------------
# Reentry threshold, which is a module-level policy constant
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def reentry_threshold_m(threshold_m: float) -> Iterator[float]:
    """Temporarily set the propagator's reentry floor, restoring it on exit.

    The floor is ``propagation.propagator.REENTRY_ALTITUDE_M``, a module
    constant, because for the propagator it is a fixed convention (100 km, the
    Karman line) rather than a per-call parameter — and that is the right design
    for a propagator. It is the wrong design for an uncertainty sweep whose job
    is to ask what the convention costs, so this rebinds it for the duration of
    a call and puts it back.

    PROCESS-LOCAL AND NOT THREAD-SAFE, stated rather than discovered: it mutates
    module state. The sweep runs each case in its own multiprocessing worker,
    where that is exactly what is wanted; two threads sharing an interpreter
    would corrupt each other's runs.
    """
    if not math.isfinite(threshold_m) or threshold_m <= 0.0:
        raise ValueError(f"reentry threshold must be positive and finite, got {threshold_m}")
    original = prop.REENTRY_ALTITUDE_M
    prop.REENTRY_ALTITUDE_M = float(threshold_m)
    try:
        yield float(threshold_m)
    finally:
        prop.REENTRY_ALTITUDE_M = original


# ---------------------------------------------------------------------------
# One forward run
# ---------------------------------------------------------------------------


def _density_function(
    case: UncertaintyCase,
    provider: AtmosphereProvider,
    kp_ap: KpApSeries,
    f107: F107Series,
    *,
    mode: str,
) -> AtmosphereDensityFunction:
    """Atmosphere callable for one case, wrapped by the stress test if asked.

    The stress test amplifies the storm-minus-quiet difference, so in the quiet
    branch it is a no-op by construction: the drivers it differences against are
    already the quiet ones. Wrapping both branches identically is therefore
    correct and keeps the two branches differing in the atmosphere alone.
    """
    quiet_for = functools.partial(quiet_baseline_drivers, quiet_ap=case.quiet_ap)
    effective: AtmosphereProvider = provider
    if case.storm_density_factor != 1.0:
        effective = ScalarDensityStressTestProvider(
            provider, quiet_for, enhancement_factor=case.storm_density_factor
        )
    return AtmosphereDensityFunction(
        effective, kp_ap, f107, mode=mode, quiet_ap=case.quiet_ap
    )


def decay_point(
    case: UncertaintyCase,
    inverse_ballistic_coefficient: float,
    *,
    mode: str,
    provider: AtmosphereProvider,
    kp_ap: KpApSeries,
    f107: F107Series,
) -> inversion.DecayPoint:
    """Propagate one case at one inverse ballistic coefficient.

    Returns the same :class:`~.ballistic_inversion.DecayPoint` that
    ``ballistic_inversion.decay_for_inverse_ballistic_coefficient`` returns, so
    the curves this builds feed straight into ``invert_decay`` and
    ``storm_cost_days`` without a second representation of a decay curve
    existing anywhere.

    It is a separate function from that one because the sweep has to reach four
    knobs the library entry point does not expose — integrator tolerances, the
    reentry floor, the wind-equivalent drag scale and an optional thrust impulse
    — and reaching them through a wrapper is what keeps the physics in the
    propagator instead of copied into the sweep.
    """
    if inverse_ballistic_coefficient <= 0:
        raise ValueError(
            "inverse ballistic coefficient must be positive, got "
            f"{inverse_ballistic_coefficient}"
        )
    if mode not in ("storm", "quiet"):
        raise ValueError(f"mode must be 'storm' or 'quiet', got {mode!r}")

    scale = wind_equivalent_drag_scale(case.wind_east_m_s, case=case)
    effective_b = inverse_ballistic_coefficient * scale
    area = case.projected_area_m2
    if area is None:
        area = effective_b * case.mass_kg / case.drag_coefficient
        drag_coefficient = case.drag_coefficient
    else:
        # A projected area was pinned, so the drag coefficient is whatever
        # reproduces the requested C_d*A/m at that area and mass. Solving for
        # C_d rather than silently keeping it and changing C_d*A/m is what makes
        # the mass/area degeneracy measurable instead of merely asserted.
        drag_coefficient = effective_b * case.mass_kg / area

    spacecraft = SpacecraftProperties(
        mass_kg=case.mass_kg,
        reference_area_m2=area,
        drag_coefficient=drag_coefficient,
        name=f"{case.case_id} B={inverse_ballistic_coefficient:.5f}",
    )
    density = _density_function(case, provider, kp_ap, f107, mode=mode)

    with reentry_threshold_m(case.reentry_threshold_m):
        propagator = DragPropagator(density, spacecraft, include_j2=True)
        result = propagator.propagate(
            case.initial_state(),
            case.duration_days,
            sample_interval_s=case.sample_interval_s,
            rtol=case.rtol,
            atol=case.atol,
            max_step_s=case.max_step_s,
        )
        if case.thrust_delta_v_m_s > 0.0 and not _reentered_before(result, case.thrust_epoch_days):
            result = _continue_after_impulse(case, result, propagator)

    return inversion.DecayPoint(
        inverse_ballistic_coefficient=inverse_ballistic_coefficient,
        decay_days=result.decay_days,
        final_perigee_km=result.perigee_altitude_m[-1] / 1000.0,
        reentered=result.reentered,
    )


def _reentered_before(result: prop.PropagationResult, epoch_days: float) -> bool:
    decay = result.decay_days
    return decay is not None and decay <= epoch_days


def _continue_after_impulse(
    case: UncertaintyCase,
    first_leg: prop.PropagationResult,
    propagator: DragPropagator,
) -> prop.PropagationResult:
    """Re-propagate from an along-track impulse applied mid-flight.

    The propagator integrates a ballistic trajectory, which is the right model
    for the safe-mode vehicles SpaceX described. Thrust is therefore modelled
    the only way it can be without adding a thrust term to the equations of
    motion: as an instantaneous along-track velocity change at a stated epoch,
    with the trajectory re-integrated from the new state.

    That is a real approximation and it is recorded as one. A finite burn
    spreads the same impulse over minutes and raises the orbit slightly less
    efficiently than an impulse at the same point.
    """
    target_seconds = case.thrust_epoch_days * _SECONDS_PER_DAY
    epoch = first_leg.times_utc[0]
    index = min(
        range(len(first_leg.times_utc)),
        key=lambda i: abs((first_leg.times_utc[i] - epoch).total_seconds() - target_seconds),
    )
    position = np.asarray(first_leg.positions_m[index], dtype=float)
    velocity = np.asarray(first_leg.velocities_m_s[index], dtype=float)
    speed = float(np.linalg.norm(velocity))
    if speed == 0.0:
        raise RuntimeError(
            f"{case.case_id}: zero velocity at the thrust epoch; the trajectory is "
            "not usable as an impulse start point"
        )
    boosted = velocity + case.thrust_delta_v_m_s * velocity / speed
    remaining_days = case.duration_days - case.thrust_epoch_days
    return propagator.propagate(
        OrbitState(
            first_leg.times_utc[index],
            (float(position[0]), float(position[1]), float(position[2])),
            (float(boosted[0]), float(boosted[1]), float(boosted[2])),
        ),
        remaining_days,
        sample_interval_s=case.sample_interval_s,
        rtol=case.rtol,
        atol=case.atol,
        max_step_s=case.max_step_s,
    )


def curve_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[inversion.DecayPoint, ...]:
    """Rebuild a decay curve from serialised rows, sorted by C_d*A/m."""
    points = [
        inversion.DecayPoint(
            inverse_ballistic_coefficient=float(row["inverse_ballistic_coefficient"]),
            decay_days=None if row["decay_days"] is None else float(row["decay_days"]),
            final_perigee_km=float(row.get("final_perigee_km", float("nan"))),
            reentered=bool(row["reentered"]),
        )
        for row in rows
    ]
    return tuple(sorted(points, key=lambda p: p.inverse_ballistic_coefficient))


# ---------------------------------------------------------------------------
# From curves to the reported quantities
# ---------------------------------------------------------------------------


def observed_decay_targets(case: UncertaintyCase) -> tuple[float, ...]:
    """Catalogued decay times this case inverts, under its date convention.

    The offset is applied to every object identically because it represents the
    catalogue's date resolution, which is a property of the catalogue and not of
    the individual object: shifting each object independently would model
    measurement noise the catalogue does not have.
    """
    losses = [g for g in event.CATALOGUED_DECAYS if g.is_storm_loss]
    if case.decay_date_convention == "days_from_deployment":
        raw = [g.days_from_deployment for g in losses]
    else:
        raw = [g.days_to_decay for g in losses]
    values = sorted(
        float(value) + case.decay_date_offset_days for value in raw if value is not None
    )
    if not values:
        raise ValueError(f"{case.case_id}: no catalogued storm-window decays to invert")
    return tuple(values)


def decay_days_at(
    curve: Sequence[inversion.DecayPoint], inverse_ballistic_coefficient: float
) -> float | None:
    """Decay time at one C_d*A/m, by log-log interpolation of a forward curve.

    The single-curve form of the interpolation
    ``ballistic_inversion.storm_cost_days`` performs on a PAIR of curves. It is
    needed because the reported storm cost is quoted both as days and as a
    fraction of orbital life, and the fraction needs the quiet-branch lifetime
    as its denominator, which the paired function differences away. A test
    pins the two against each other on the committed curves so they cannot
    drift into disagreeing.

    Returns None when the value is outside the curve's reentering span, rather
    than extrapolating a power law past the last point that was actually
    computed.
    """
    usable = sorted(
        (p for p in curve if p.reentered and p.decay_days is not None),
        key=lambda p: p.inverse_ballistic_coefficient,
    )
    if len(usable) < 2:
        return None
    for low, high in zip(usable[:-1], usable[1:], strict=True):
        low_b = low.inverse_ballistic_coefficient
        high_b = high.inverse_ballistic_coefficient
        if not low_b <= inverse_ballistic_coefficient <= high_b:
            continue
        low_days, high_days = low.decay_days, high.decay_days
        if low_days is None or high_days is None:
            raise RuntimeError("decay curve contains a non-reentering point after filtering")
        span = math.log(high_b) - math.log(low_b)
        if span == 0.0:
            return float(low_days)
        weight = (math.log(inverse_ballistic_coefficient) - math.log(low_b)) / span
        return float(
            math.exp(math.log(low_days) + weight * (math.log(high_days) - math.log(low_days)))
        )
    return None


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """The four headline quantities as one case produced them.

    ``n_inverted`` and ``n_targets`` are carried because an inversion target
    outside the computed curve is DROPPED by ``invert_decay``, and a median over
    a silently shrunken set is a different statistic. A case where the two
    differ is reported, not averaged.
    """

    case_id: str
    inferred_b_median_m2_kg: float | None
    inferred_b_min_m2_kg: float | None
    inferred_b_max_m2_kg: float | None
    storm_decay_days: float | None
    quiet_decay_days: float | None
    storm_cost_days: float | None
    storm_cost_fraction: float | None
    n_targets: int
    n_inverted: int
    decay_source: str = "log-log interpolation of the forward curve"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


#: How the two branch lifetimes behind a reported storm cost were obtained.
DECAY_SOURCE_INTERPOLATED = "log-log interpolation of the forward curve"
DECAY_SOURCE_DIRECT = "direct propagation at the fitted C_d*A/m"


def outcomes_from_curves(
    case: UncertaintyCase,
    storm_curve: Sequence[inversion.DecayPoint],
    quiet_curve: Sequence[inversion.DecayPoint],
    *,
    direct_storm_decay_days: float | None = None,
    direct_quiet_decay_days: float | None = None,
) -> CaseOutcome:
    """Invert the catalogued decays and report the storm cost at the fitted value.

    Reproduces the chain ``scripts/run_full_analysis.py`` runs — invert each
    catalogued decay against the storm curve, take the median of the inferred
    values, and evaluate both atmosphere branches there.

    The two branch lifetimes may be supplied directly. They should be: the cost
    is a fraction of a day out of seven and a half, so interpolating each branch
    to within half a percent still leaves the difference wrong by of order ten
    percent. Passing lifetimes obtained by propagating at the fitted parameter
    removes that entirely, and the outcome records which route was taken so a
    reader can tell the two apart.
    """
    targets = observed_decay_targets(case)
    results = [inversion.invert_decay(target, tuple(storm_curve)) for target in targets]
    inferred = [
        r.inferred_inverse_ballistic_coefficient
        for r in results
        if r.inferred_inverse_ballistic_coefficient is not None
    ]
    notes = tuple(
        sorted({r.note for r in results if r.inferred_inverse_ballistic_coefficient is None})
    )
    if not inferred:
        return CaseOutcome(
            case_id=case.case_id,
            inferred_b_median_m2_kg=None,
            inferred_b_min_m2_kg=None,
            inferred_b_max_m2_kg=None,
            storm_decay_days=None,
            quiet_decay_days=None,
            storm_cost_days=None,
            storm_cost_fraction=None,
            n_targets=len(targets),
            n_inverted=0,
            decay_source=DECAY_SOURCE_INTERPOLATED,
            notes=notes,
        )

    median = float(np.median(inferred))
    if direct_storm_decay_days is not None and direct_quiet_decay_days is not None:
        storm_days: float | None = float(direct_storm_decay_days)
        quiet_days: float | None = float(direct_quiet_decay_days)
        source = DECAY_SOURCE_DIRECT
    else:
        storm_days = decay_days_at(storm_curve, median)
        quiet_days = decay_days_at(quiet_curve, median)
        source = DECAY_SOURCE_INTERPOLATED
    cost = None if storm_days is None or quiet_days is None else quiet_days - storm_days
    fraction = None if cost is None or not quiet_days else cost / quiet_days
    return CaseOutcome(
        case_id=case.case_id,
        inferred_b_median_m2_kg=median,
        inferred_b_min_m2_kg=float(min(inferred)),
        inferred_b_max_m2_kg=float(max(inferred)),
        storm_decay_days=storm_days,
        quiet_decay_days=quiet_days,
        storm_cost_days=cost,
        storm_cost_fraction=fraction,
        n_targets=len(targets),
        n_inverted=len(inferred),
        decay_source=source,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Peak density enhancement, and the published bands it is checked against
# ---------------------------------------------------------------------------


def peak_density_enhancement(
    case: UncertaintyCase,
    provider: AtmosphereProvider,
    kp_ap: KpApSeries,
    f107: F107Series,
    *,
    step_hours: float = 1.0,
) -> dict[str, Any]:
    """Peak and mean storm-over-quiet mass-density enhancement at one point.

    Sampled hourly across 2-8 February 2022 at the case's evaluation altitude
    and latitude and at zero longitude, which is what
    ``scripts/run_full_analysis.py`` does; the nominal case therefore reproduces
    the committed 17.1% rather than approximating it.

    The density differenced is the SPECIES SUM, not the model's own total, for
    the same reason ``scenarios.starlink_analysis`` uses the species sum: it is
    the quantity the composition diagnostics describe, and mixing the two would
    fold the density-closure fraction into the enhancement.
    """
    if step_hours <= 0.0:
        raise ValueError(f"step_hours must be positive, got {step_hours}")
    altitude_m = case.enhancement_altitude_m
    latitude_deg = case.enhancement_latitude_deg
    quiet_for = functools.partial(quiet_baseline_drivers, quiet_ap=case.quiet_ap)
    effective: AtmosphereProvider = provider
    if case.storm_density_factor != 1.0:
        effective = ScalarDensityStressTestProvider(
            provider, quiet_for, enhancement_factor=case.storm_density_factor
        )

    enhancements: list[float] = []
    peak_epoch = ""
    peak = -math.inf
    when = _ENHANCEMENT_START_UTC
    step = _dt.timedelta(hours=step_hours)
    while when <= _ENHANCEMENT_END_UTC:
        observed = drivers_at(when, kp_ap, f107)
        quiet_drivers = quiet_for(observed)
        storm_state = effective.state_at(
            when, altitude_m, latitude_deg, _ENHANCEMENT_LONGITUDE_DEG, observed
        )
        quiet_state = provider.state_at(
            when, altitude_m, latitude_deg, _ENHANCEMENT_LONGITUDE_DEG, quiet_drivers
        )
        storm_rho = comp.total_species_mass_density(storm_state.number_densities)
        quiet_rho = comp.total_species_mass_density(quiet_state.number_densities)
        value = storm_rho / quiet_rho - 1.0
        enhancements.append(value)
        if value > peak:
            peak = value
            peak_epoch = when.isoformat()
        when += step

    return {
        "peak_fraction": float(peak),
        "mean_fraction": float(np.mean(enhancements)),
        "minimum_fraction": float(min(enhancements)),
        "samples": float(len(enhancements)),
        "altitude_m": float(altitude_m),
        "latitude_deg": float(latitude_deg),
        "quiet_ap": float(case.quiet_ap),
        "storm_density_factor": float(case.storm_density_factor),
        "peak_time_utc": peak_epoch,
    }


def published_envelope_enhancements(
    *, altitude_m: float = 210_000.0, when: _dt.datetime | None = None
) -> list[dict[str, Any]]:
    """Every published enhancement band that covers this altitude and epoch.

    This is the atmosphere-model branch for the peak-enhancement quantity, and
    it is the branch where the published-envelope provider is doing what it was
    built for: reporting a cited number inside the coverage its source supports.
    Each row carries the citation, so a range assembled from them can be read
    back to a page in a paper.
    """
    moment = when if when is not None else event.DEPLOYMENT_UTC
    rows: list[dict[str, Any]] = []
    for band in envelope.bands_covering(altitude_m / 1000.0, moment):
        row: dict[str, Any] = {
            "citation": band.citation,
            "doi_or_url": band.doi_or_url,
            "provenance_kind": band.provenance_kind,
            "altitude_km": altitude_m / 1000.0,
            "coverage_km": [band.altitude_min_km, band.altitude_max_km],
            "note": band.note,
        }
        row["supplies_altitude_profile"] = band.supplies_altitude_profile
        selections: tuple[envelope.Selection, ...] = ("low", "nominal", "high")
        for selection in selections:
            try:
                row[f"enhancement_{selection}"] = band.enhancement_at(
                    altitude_m / 1000.0, selection
                )
            except envelope.UnsupportedAltitudeProfileError as exc:
                # Recorded, not skipped. A null with the provider's own refusal
                # text beside it is evidence that the source states no value
                # here; an omitted key would read as an oversight.
                row[f"enhancement_{selection}"] = None
                row[f"enhancement_{selection}_refusal"] = str(exc)
        if row["supplies_altitude_profile"]:
            row["nominal_status"] = envelope.DERIVED_INTERPOLATION_LABEL
        else:
            row["nominal_status"] = (
                "REFUSED: the source supplies no altitude profile, so no value "
                "is produced at this altitude. The low and high figures are the "
                "published extremes over the whole reported domain."
            )
        rows.append(row)
    return rows


def published_envelope_propagation_refusal(
    provider: AtmosphereProvider,
    kp_ap: KpApSeries,
    f107: F107Series,
    *,
    altitude_m: float = 150_000.0,
    when: _dt.datetime | None = None,
) -> dict[str, str]:
    """Record WHY the published-envelope provider cannot drive a decay run.

    A decay from 210 km leaves both the altitude coverage (200-400 km at
    widest) and the time coverage (3-5 February 2022) of every encoded band
    within hours, and the provider raises rather than extrapolating. That is
    correct behaviour and it is the reason the decay quantities use the scalar
    stress test instead.

    So the refusal is PROVOKED here and its message stored, rather than
    described. A note saying "we could not use it" is a claim; the exception
    text is evidence, and it will change if the provider's coverage ever does.
    """
    moment = when if when is not None else _dt.datetime(2022, 2, 10, tzinfo=_dt.UTC)
    # An ENDPOINT_PAIR band, chosen by name rather than by index. The probe is
    # about COVERAGE - the provider refusing an altitude and an epoch its
    # source does not reach - so it must not be run on a band that would refuse
    # earlier, and for a different reason, because it supplies no altitude
    # profile at all.
    band = envelope.band_for("MSIS2.0")
    subject = envelope.PublishedEventDensityEnvelopeProvider(
        provider, quiet_baseline_drivers, band=band, selection="nominal"
    )
    outcome: dict[str, str] = {
        "provider": subject.name,
        "probe_altitude_m": f"{altitude_m:.0f}",
        "probe_time_utc": moment.isoformat(),
    }
    try:
        subject.state_at(moment, altitude_m, 53.22, 0.0, drivers_at(moment, kp_ap, f107))
    except envelope.OutsidePublishedCoverageError as exc:
        outcome["refused"] = "true"
        outcome["message"] = str(exc)
    else:
        outcome["refused"] = "false"
        outcome["message"] = (
            "The provider answered a probe outside the coverage recorded when this "
            "sweep was written. Its coverage has changed; re-examine whether the "
            "decay branch can now use it directly."
        )
    return outcome


# ---------------------------------------------------------------------------
# The mass / area / drag-coefficient degeneracy, measured
# ---------------------------------------------------------------------------


def degeneracy_evidence(
    inverse_ballistic_coefficient: float,
    *,
    mass_range_kg: tuple[float, float] = event.MASS_KG_RANGE,
    area_range_m2: tuple[float, float] = (2.0, 12.0),
    drag_coefficient_range: tuple[float, float] = event.CD_SENSITIVITY_RANGE,
) -> dict[str, Any]:
    """What mass, area and C_d uncertainty DOES change, given they cancel in decay.

    They do not change the trajectory: ``SpacecraftProperties`` reduces them to
    ``C_d*A/m`` before the equations of motion see them, and this function shows
    that reduction is exact by evaluating it across the corners of the three
    ranges. What they do change is the physical INTERPRETATION of a fitted
    C_d*A/m — the implied area at an assumed mass and drag coefficient — and
    that span is what is reported.
    """
    corners = [
        (mass, area, cd)
        for mass in mass_range_kg
        for area in area_range_m2
        for cd in drag_coefficient_range
    ]
    reduced = [
        SpacecraftProperties(
            mass_kg=mass, reference_area_m2=area, drag_coefficient=cd
        ).inverse_ballistic_coefficient
        for mass, area, cd in corners
    ]
    implied_area = [
        inverse_ballistic_coefficient * mass / cd
        for mass in mass_range_kg
        for cd in drag_coefficient_range
    ]
    implied_cd = [
        inverse_ballistic_coefficient * mass / area
        for mass in mass_range_kg
        for area in area_range_m2
    ]
    return {
        "inverse_ballistic_coefficient_m2_kg": inverse_ballistic_coefficient,
        "trajectory_depends_on": "C_d*A/m only",
        "analytic_decay_sensitivity_to_mass_at_fixed_cd_a_over_m": 0.0,
        "analytic_decay_sensitivity_to_area_at_fixed_cd_a_over_m": 0.0,
        "analytic_decay_sensitivity_to_drag_coefficient_at_fixed_cd_a_over_m": 0.0,
        "analytic_sensitivity_caveat": (
            "The three zeros above are properties of the EQUATIONS, not measurements "
            "of a run. Reconstructing the same C_d*A/m from a different (mass, area, "
            "C_d) triple is exact in algebra and inexact in binary floating point, "
            "and a multi-day integration through a binned density field amplifies "
            "that last-bit difference. The size of the amplification is measured by "
            "propagating the mass and area cases in full; see the numerical noise "
            "floor reported beside this."
        ),
        "corner_count": len(corners),
        "reduced_parameter_min_m2_kg": float(min(reduced)),
        "reduced_parameter_max_m2_kg": float(max(reduced)),
        "implied_area_min_m2": float(min(implied_area)),
        "implied_area_max_m2": float(max(implied_area)),
        "implied_area_span_fraction": float(
            max(implied_area) / min(implied_area) - 1.0
        ),
        "implied_drag_coefficient_min": float(min(implied_cd)),
        "implied_drag_coefficient_max": float(max(implied_cd)),
        "mass_range_kg": list(mass_range_kg),
        "area_range_m2": list(area_range_m2),
        "drag_coefficient_range": list(drag_coefficient_range),
        "statement": (
            "Mass, projected area and drag coefficient do not enter the equations "
            "of motion separately: SpacecraftProperties reduces them to C_d*A/m "
            "before the propagator sees them, so at fixed C_d*A/m their analytic "
            "effect on decay time is zero. What remains when they are actually "
            "swept is floating-point round-off in that reduction, amplified by the "
            "integration, and it is reported as a measured noise floor rather than "
            "asserted away. They are not thereby unimportant: the same fitted "
            "C_d*A/m implies an area spanning the range above once a mass and a "
            "drag coefficient are assumed, and none of the three is constrained by "
            "a decay curve. This is the identifiability limit recorded in "
            "domain.drag_parameters, quantified."
        ),
    }


# ---------------------------------------------------------------------------
# Ranges and reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RangeSample:
    """One (label, configuration, value) triple entering a reported range.

    A plain (case, value) pair would not do: the peak-enhancement quantity is
    also evaluated under alternative ATMOSPHERE MODELS, whose "configuration" is
    a citation rather than a set of case fields. Carrying the configuration as a
    mapping lets both kinds of input into the same range without either of them
    arriving unlabelled.
    """

    label: str
    configuration: Mapping[str, Any]
    value: float | None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("a range sample must be labelled; an unlabelled bound is unusable")


def outcome_samples(
    outcomes: Sequence[tuple[UncertaintyCase, CaseOutcome]], attribute: str
) -> list[RangeSample]:
    """Pull one reported quantity out of a set of case outcomes."""
    samples: list[RangeSample] = []
    for case, outcome in outcomes:
        if not hasattr(outcome, attribute):
            raise AttributeError(
                f"CaseOutcome has no attribute {attribute!r}; available: "
                f"{sorted(f for f in outcome.__slots__ if not f.startswith('_'))}"
            )
        samples.append(
            RangeSample(
                label=case.case_id,
                configuration=case.differences_from_nominal(),
                value=getattr(outcome, attribute),
            )
        )
    return samples


@dataclass(frozen=True, slots=True)
class SensitivityRange:
    """The span one quantity took over the sweep, with the cases at each end.

    ``minimum_case_id`` and ``maximum_case_id`` are required rather than
    optional. A bound without the configuration that produced it cannot be
    checked, cannot be reproduced, and invites the reader to treat it as a
    property of the world rather than of a particular corner of the input box.
    """

    quantity: str
    units: str
    nominal: float | None
    minimum: float
    maximum: float
    median: float | None
    minimum_case_id: str
    maximum_case_id: str
    minimum_configuration: dict[str, Any]
    maximum_configuration: dict[str, Any]
    n_cases: int
    n_missing: int

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = asdict(self)
        payload["caveat"] = SENSITIVITY_RANGE_CAVEAT
        return payload


def sensitivity_range(
    quantity: str,
    units: str,
    samples: Sequence[RangeSample],
    *,
    nominal: float | None = None,
    report_median: bool = True,
) -> SensitivityRange:
    """Collapse a set of labelled samples into a reported range.

    ``None`` values are COUNTED, not dropped quietly: a case whose propagation
    did not reenter inside the window, or whose inversion targets fell off the
    computed curve, is a real outcome and ``n_missing`` carries it into the
    report. A range quoted over an unstated subset of the cases that were
    attempted is the failure this argument exists to prevent.
    """
    if not samples:
        raise ValueError(f"{quantity}: no samples supplied; there is no range to report")
    usable = [(s, float(s.value)) for s in samples if s.value is not None]
    missing = len(samples) - len(usable)
    if not usable:
        raise ValueError(
            f"{quantity}: all {len(samples)} sampled configurations returned no value. "
            "The range is undefined and must be reported as such rather than as a number."
        )
    low_sample, low_value = min(usable, key=lambda item: item[1])
    high_sample, high_value = max(usable, key=lambda item: item[1])
    return SensitivityRange(
        quantity=quantity,
        units=units,
        nominal=nominal,
        minimum=low_value,
        maximum=high_value,
        median=float(np.median([value for _s, value in usable])) if report_median else None,
        minimum_case_id=low_sample.label,
        maximum_case_id=high_sample.label,
        minimum_configuration=dict(low_sample.configuration),
        maximum_configuration=dict(high_sample.configuration),
        n_cases=len(samples),
        n_missing=missing,
    )


#: Phrase that must never appear in a generated uncertainty artefact, because
#: none of these ranges is one and the two are not interchangeable. Held here so
#: the renderer and the test that guards it read the same string.
FORBIDDEN_INTERVAL_PHRASE = "confidence interval"


def _format_value(value: float | None, units: str) -> str:
    if value is None:
        return "not established"
    if units == "dimensionless fraction":
        return f"{value:.4%}"
    if units == "m^2/kg":
        return f"{value:.5f}"
    return f"{value:.4f}"


def _range_rows(ranges: Sequence[SensitivityRange]) -> list[str]:
    lines = [
        "| Quantity | Units | Minimum | Nominal | Maximum | Median | Cases | No value |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in ranges:
        lines.append(
            f"| {item.quantity} | {item.units} | "
            f"{_format_value(item.minimum, item.units)} | "
            f"{_format_value(item.nominal, item.units)} | "
            f"{_format_value(item.maximum, item.units)} | "
            f"{_format_value(item.median, item.units)} | "
            f"{item.n_cases} | {item.n_missing} |"
        )
    return lines


def _extreme_rows(ranges: Sequence[SensitivityRange]) -> list[str]:
    lines: list[str] = []
    for item in ranges:
        lines.extend(
            [
                "",
                f"#### {item.quantity}",
                "",
                f"- Minimum {_format_value(item.minimum, item.units)} at case "
                f"`{item.minimum_case_id}`, which differs from the committed reference "
                f"configuration in: {_configuration_text(item.minimum_configuration)}",
                f"- Maximum {_format_value(item.maximum, item.units)} at case "
                f"`{item.maximum_case_id}`, which differs from the committed reference "
                f"configuration in: {_configuration_text(item.maximum_configuration)}",
            ]
        )
    return lines


def _configuration_text(configuration: Mapping[str, Any]) -> str:
    if not configuration:
        return "nothing (this IS the reference configuration)"
    return ", ".join(f"`{key}` = {value!r}" for key, value in sorted(configuration.items()))


def render_markdown(
    ranges: Sequence[SensitivityRange],
    *,
    sections: Sequence[tuple[str, str]] = (),
    title: str = "Uncertainty propagation for the February 2022 Group 4-7 analysis",
) -> str:
    """Render the report, refusing to emit one that has lost its own caveat.

    Two checks, both of which have to be structural rather than editorial. The
    caveat has to be PRESENT, because a table of minima and maxima with no
    statement of what they are will be read as a probability statement by
    default. And the phrase "confidence interval" has to be ABSENT, because
    nothing here has a justified distribution behind it, so the phrase would be
    false wherever it appeared — including in a section supplied by a caller,
    which is why the check runs over the assembled text and not only over the
    part this function wrote.
    """
    if not ranges:
        raise ValueError("no sensitivity ranges supplied; there is nothing to report")
    lines: list[str] = [
        f"# {title}",
        "",
        "## What these numbers are",
        "",
        SENSITIVITY_RANGE_CAVEAT,
        "",
        "## Headline quantities",
        "",
    ]
    lines.extend(_range_rows(ranges))
    lines.extend(["", "### Where each extreme came from"])
    lines.extend(_extreme_rows(ranges))
    for heading, body in sections:
        lines.extend(["", f"## {heading}", "", body])
    lines.append("")
    text = "\n".join(lines)

    if SENSITIVITY_RANGE_CAVEAT not in text:
        raise RuntimeError(
            "the rendered report does not contain SENSITIVITY_RANGE_CAVEAT. A table "
            "of minima and maxima without it will be read as a probability "
            "statement, which is exactly what these ranges are not."
        )
    if FORBIDDEN_INTERVAL_PHRASE in text.casefold():
        raise RuntimeError(
            f"the rendered report contains the phrase {FORBIDDEN_INTERVAL_PHRASE!r}. "
            "No input to this analysis has a justified probability distribution, so "
            "no interval here has a coverage probability and the phrase would be "
            "false wherever it appears. Say 'sensitivity range' and state the "
            "configuration that produced each bound."
        )
    return text
