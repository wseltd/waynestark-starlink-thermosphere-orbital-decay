"""Counterfactual attribution decomposition for the Group 4-7 losses.

WHY THIS MODULE EXISTS. An earlier version of this analysis compared one quiet
run against one storm run, found the storm cost roughly half a day out of seven,
and concluded that the storm was "a contributing factor, not the primary cause".
That conclusion does not follow from that comparison. A single pairwise
difference establishes the size of ONE effect; calling any factor primary is a
statement about the ORDERING of several effects, and an ordering computed at one
point in parameter space says nothing about whether it survives the uncertainty
surrounding that point.

So the claim is rebuilt here as something checkable. Each candidate factor is
given an explicit counterfactual pair — two propagations identical in every
respect but one — and the resulting lifetime difference is its contribution.
The contributions are then ranked, and the ranking is recomputed at every corner
of the plausible uncertainty envelope. If the ranking is the same everywhere,
the top factor is dominant and can be named with its numbers. If the ranking
changes, what exists is not a cause but a REGIME BOUNDARY, and this module
reports the boundary rather than choosing the more quotable side of it.

WHAT IS AND IS NOT ESTABLISHED HERE.

  CAN: "under this atmosphere model, moving the insertion perigee from 210 km
       to 320 km changes modelled lifetime by X days, and that change is larger
       than the change from the storm at every corner tested."
  CANNOT: "the storm was not the cause." Ranking modelled sensitivities is not
       the same as apportioning responsibility for a real event, in which the
       factors were not independent: a low insertion is chosen in the
       expectation of being able to orbit-raise, and the storm is what removed
       that ability.
  CANNOT: anything about absolute density. Decay depends on the product of
       density and C_d*A/m, so an error in one is absorbed by the other.

CENSORING IS FIRST-CLASS. Several counterfactuals do not reenter inside the
simulated window. "Did not decay in 114 days" is not a lifetime and must not be
turned into one, so every lifetime carries a ``censored`` flag, differences are
computed as INTERVALS, and a factor whose interval overlaps another's is
reported as unresolved rather than silently ordered.

WINDOW LENGTH. The window is bounded by DATA, not by taste: the GFZ Kp/ap series
in this repository ends 2022-05-31T03:00Z and the driver lookup refuses to
extrapolate past it. 114 days from deployment is the longest window that never
asks for a driver value that does not exist.

TERMINOLOGY. ``inverse_ballistic_coefficient_m2_kg`` is C_d*A/m in m^2/kg. It is
NOT a ballistic coefficient; the conventional ballistic coefficient m/(C_d*A) in
kg/m^2 is its reciprocal and is reported alongside it in every row. See
domain.drag_parameters for why this project spells that out every time.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from ..domain.drag_parameters import to_ballistic_coefficient
from ..events import starlink_g4_7 as event
from ..propagation.propagator import (
    REENTRY_ALTITUDE_M,
    DensityFunction,
    DragPropagator,
    OrbitState,
    PropagationResult,
    SpacecraftProperties,
    geodetic_altitude,
    geodetic_latlon,
    greenwich_sidereal_time,
    state_from_perigee_apogee,
)
from ..providers.wind import (
    WIND_SENSITIVITY_CASES_M_S,
    corotation_velocity,
    relative_velocity_with_wind,
)

# --------------------------------------------------------------------------
# Assumptions. Every value below is a CHOICE, recorded as one.
# --------------------------------------------------------------------------

#: Longest window the loaded drivers support. The GFZ Kp/ap extended series ends
#: 2022-05-31T03:00Z and KpApSeries refuses to extrapolate, so a longer window
#: would raise rather than quietly invent geomagnetic activity.
SIMULATION_WINDOW_DAYS = 114.0

#: The documented insertion orbit (Guarnieri et al. 2025). Not an assumption.
DOCUMENTED_PERIGEE_ALTITUDE_M = event.INSERTION_PERIGEE_ALTITUDE_M
DOCUMENTED_APOGEE_ALTITUDE_M = event.INSERTION_APOGEE_ALTITUDE_M

#: The "safer" comparison orbit. ASSUMPTION, not an operator plan: SpaceX has
#: not published what insertion it would have chosen instead and this repository
#: has no basis for guessing one. 320x340 km is used because it sits just above
#: the band where storm-time density enhancement is largest while remaining a
#: plausible low-insertion strategy. Nothing in the METHOD depends on the exact
#: figure; the SIZE of the insertion contribution does, which is why the number
#: is stated here rather than buried in a script.
RAISED_PERIGEE_ALTITUDE_M = 320_000.0
RAISED_APOGEE_ALTITUDE_M = 340_000.0

#: Effective C_d*A/m values obtained by inverting the catalogued decay dates
#: (scenarios.ballistic_inversion, ten storm-window objects). Literals here so
#: this module performs no I/O; the consuming script asserts they still match
#: evidence/analysis_report.json and now FAILS rather than warning if they do
#: not. They drifted once, silently, by about 0.1%: the warning was printed at
#: the end of a long run and a warning nobody reads is not a check.
#:
#: THESE ARE A MIXTURE. evidence/PER_OBJECT_CALIBRATION.md shows the ten
#: objects are two populations -- payloads 0.0463-0.2110 and debris
#: 0.0150-0.0193 m^2/kg, with no overlap -- so this range averages across both
#: and its median describes neither. It is retained because the published
#: comparison is stated against it, not because it characterises a spacecraft.
FITTED_LOW_DRAG_INVERSE_BALLISTIC_COEFFICIENT_M2_KG = 0.01477287878849512
FITTED_MEDIAN_INVERSE_BALLISTIC_COEFFICIENT_M2_KG = 0.026246221431224966
FITTED_HIGH_AREA_INVERSE_BALLISTIC_COEFFICIENT_M2_KG = 0.06024043425638363

#: Externally published inverse ballistic coefficient for a Capella Space
#: Whitney SAR satellite in operational flight near 525 km (Shambaugh 2024,
#: arXiv:2406.08342; NOT SpaceX Starlink). Included as a labelled external
#: reference case only, and kept out of the fitted range because it describes
#: a different spacecraft family and orbital regime.
OPERATOR_INVERSE_BALLISTIC_COEFFICIENT_M2_KG = (
    event.OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG
)

#: Storm-response amplification factors applied by providers.scaled_storm. 1.0 is
#: NRLMSISE-00 as published; 2.0 is roughly the WAM-IPE level and 3.5 roughly the
#: MAGE level reported by He et al. 2023 at 200/300 km.
DENSITY_ENHANCEMENT_FACTORS: tuple[float, ...] = (1.0, 2.0, 3.5)

#: Along-track acceleration for the counterfactual orbit-raising cases, m/s^2.
#: ASSUMPTION with three parts, all load-bearing:
#:
#:   1. The real vehicles were in safe mode and did NOT orbit-raise. Every
#:      thrust case here is counterfactual BY CONSTRUCTION and none of them
#:      describes anything that happened.
#:   2. SpaceX has published no thrust figure for the v1.5 bus, so no value here
#:      can be sourced. 2e-5 m/s^2 is about 5.9 mN on a 297 kg vehicle, the low
#:      end of small electric propulsion.
#:   3. It is deliberately LOW-AUTHORITY: at the median fitted C_d*A/m it slows
#:      the decay without arresting it, so the cost of delaying it is a finite
#:      number rather than the uninformative "both cases survived the window".
#:      Choosing a magnitude for measurability is a choice, so the
#:      high-authority regime is reported alongside it as a separate, unranked
#:      row rather than left out.
NOMINAL_THRUST_ACCELERATION_M_S2 = 2.0e-5

#: A magnitude at which prompt thrust escapes the decay outright, about 30 mN on
#: 297 kg. Used only for the secondary, unranked regime row.
HIGH_AUTHORITY_THRUST_ACCELERATION_M_S2 = 1.0e-4

#: Thrust start delays in days after deployment. ASSUMPTIONS: 0.25 d is a
#: post-deployment checkout followed immediately by orbit-raising; 2.0 d is a
#: safe-mode recovery after the geomagnetic activity subsides.
NOMINAL_THRUST_START_DELAY_DAYS = 0.25
DELAYED_THRUST_START_DELAY_DAYS = 2.0

#: Drag coefficient used to split C_d*A/m into an area at fixed mass. The split
#: is cosmetic — only the product enters the equations of motion — but the
#: propagator's constructor wants the three factors separately.
REFERENCE_DRAG_COEFFICIENT = 2.4

#: Zonal neutral-wind cases swept as the third envelope axis, in m/s eastward.
#: Taken from providers.wind, which sources the magnitudes from this
#: repository's own zero-wind caveat (HWM14 storm-time winds at 210 km of
#: several hundred m/s) and from the storm-time circulation Fang et al. 2022
#: simulate for this event. Both signs matter: on a prograde orbit an eastward
#: wind moves the gas with the spacecraft and REDUCES the drag-relevant relative
#: speed, a westward wind increases it, so sweeping one sign would report half
#: the sensitivity. 0.0 is carried as one branch of the sweep rather than as the
#: reference the others are measured from — it is the case in which the wind
#: term is omitted, which is a choice like any other.
WIND_EAST_CASES_M_S: tuple[float, ...] = tuple(WIND_SENSITIVITY_CASES_M_S)

# --------------------------------------------------------------------------
# Factor names. Constants because they are compared for equality across
# envelope corners, and a typo in a ranking key is a silent wrong answer.
# --------------------------------------------------------------------------

FACTOR_INSERTION_ALTITUDE = "insertion altitude"
FACTOR_STORM_DENSITY = "storm density"
FACTOR_ATTITUDE = "attitude C_d*A/m"
FACTOR_THRUST_DELAY = "thrust delay"
FACTOR_ATMOSPHERE_MODEL = "atmosphere-model choice"

RANKED_FACTORS: tuple[str, ...] = (
    FACTOR_INSERTION_ALTITUDE,
    FACTOR_STORM_DENSITY,
    FACTOR_ATTITUDE,
    FACTOR_THRUST_DELAY,
    FACTOR_ATMOSPHERE_MODEL,
)

#: Bucket name used when the TOP of a corner's ranking is itself contested, so
#: no factor can be said to lead there. A sentinel rather than None because the
#: regime map is keyed by factor name and a missing key would read as "no such
#: corner" instead of "no answer at this corner".
UNRESOLVED_LEADER = "UNRESOLVED"

#: Phrases that assert causal primacy. A conclusion may contain one only when the
#: ordering was stable at every corner tested.
PRIMACY_PHRASES: tuple[str, ...] = (
    "primary cause",
    "primary driver",
    "principal cause",
    "main cause",
    "root cause",
    "the cause of",
)


class AttributionError(ValueError):
    """Raised when a decomposition is asked for something it cannot support."""


# --------------------------------------------------------------------------
# Interval arithmetic, needed because censored lifetimes make differences bounds
# rather than numbers.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    """A closed interval on the extended reals, ``low <= high``.

    Infinite endpoints are permitted and meaningful: they are how "this run did
    not reenter, so the true value is at least X" travels through the arithmetic
    instead of being rounded to a convenient finite number.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if math.isnan(self.low) or math.isnan(self.high):
            raise AttributionError(
                f"interval endpoints must not be NaN, got [{self.low}, {self.high}]"
            )
        if self.low > self.high:
            raise AttributionError(
                f"interval low must not exceed high, got [{self.low}, {self.high}]"
            )

    @property
    def is_exact(self) -> bool:
        return self.low == self.high

    def __sub__(self, other: Interval) -> Interval:
        return Interval(self.low - other.high, self.high - other.low)

    def scaled(self, factor: float) -> Interval:
        """Multiply by a positive scalar, preserving orientation."""
        if factor <= 0:
            raise AttributionError(f"interval scale factor must be positive, got {factor}")
        return Interval(self.low * factor, self.high * factor)

    @property
    def magnitude(self) -> Interval:
        """Range of |x| over the interval, used for ranking by effect size."""
        if self.low <= 0.0 <= self.high:
            return Interval(0.0, max(abs(self.low), abs(self.high)))
        return Interval(min(abs(self.low), abs(self.high)), max(abs(self.low), abs(self.high)))

    def overlaps(self, other: Interval) -> bool:
        return not (self.low > other.high or other.low > self.high)


@dataclass(frozen=True, slots=True)
class Lifetime:
    """Modelled orbital lifetime in days, possibly right-censored.

    ``censored`` True means the run reached the end of its window, or left the
    integration domain upward, without reentering: ``days`` is then the window
    length and the true lifetime is strictly greater. The flag is not a nicety.
    Treating a censored run as though it decayed at the window edge would
    understate exactly the counterfactuals that matter most, because the
    counterfactuals that save the vehicle are the ones that fail to decay.
    """

    days: float
    censored: bool = False
    termination: str = "reentered"

    def __post_init__(self) -> None:
        if not math.isfinite(self.days) or self.days <= 0.0:
            raise AttributionError(f"lifetime must be finite and positive days, got {self.days}")

    @property
    def interval(self) -> Interval:
        return Interval(self.days, math.inf if self.censored else self.days)

    def describe(self) -> str:
        return f">{self.days:.2f} d" if self.censored else f"{self.days:.2f} d"


# --------------------------------------------------------------------------
# Case specification and propagation.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThrustProfile:
    """Constant along-track acceleration switched on after a fixed delay.

    Crude on purpose. A real orbit-raise has a duty cycle, a throttle, an
    attitude schedule and a target altitude at which it stops; none of that is
    published for this bus, so modelling it would be invention dressed as
    detail. What a constant along-track term does capture is the only thing the
    decomposition needs: the rate at which propulsion adds orbital energy, and
    therefore what a delay in starting costs.
    """

    acceleration_m_s2: float
    start_delay_days: float

    def __post_init__(self) -> None:
        if self.acceleration_m_s2 <= 0.0:
            raise AttributionError(
                "thrust acceleration must be positive; use thrust=None for a "
                f"coasting case, got {self.acceleration_m_s2}"
            )
        if self.start_delay_days < 0.0:
            raise AttributionError(
                f"thrust start delay must be non-negative, got {self.start_delay_days}"
            )

    @property
    def start_delay_s(self) -> float:
        return self.start_delay_days * 86400.0

    def thrust_newtons(self, mass_kg: float) -> float:
        """The force this acceleration implies at a given mass, for reporting."""
        if mass_kg <= 0:
            raise AttributionError(f"mass must be positive, got {mass_kg}")
        return self.acceleration_m_s2 * mass_kg


class CounterfactualPropagator(DragPropagator):
    """DragPropagator plus two counterfactual terms: neutral wind and thrust.

    Subclassed rather than folded into the propagator because neither term is
    part of the physics this project validated. Every result already in the
    repository was produced by DragPropagator exactly as it stands, and adding
    optional branches to it would put never-exercised code in the middle of
    results that have been exercised. These cases are counterfactual by
    construction, so their propagator can be too.

    The wind term is applied as a CORRECTION to the parent's drag rather than by
    reimplementing drag. The parent computes drag against the co-rotating
    atmosphere; this class computes what drag would be against the co-rotating
    atmosphere PLUS the wind and adds the difference. That keeps gravity, J2, the
    reentry floor and the drag formula itself in one place, so the two
    trajectories cannot drift apart through a transcription error in a formula
    that appears twice.
    """

    def __init__(
        self,
        density_function: DensityFunction,
        spacecraft: SpacecraftProperties,
        *,
        thrust: ThrustProfile | None = None,
        wind_east_m_s: float = 0.0,
        wind_north_m_s: float = 0.0,
        include_j2: bool = True,
        include_atmosphere_rotation: bool = True,
    ) -> None:
        super().__init__(
            density_function,
            spacecraft,
            include_j2=include_j2,
            include_atmosphere_rotation=include_atmosphere_rotation,
        )
        if not math.isfinite(wind_east_m_s) or not math.isfinite(wind_north_m_s):
            raise AttributionError(
                f"wind components must be finite, got east={wind_east_m_s}, "
                f"north={wind_north_m_s}"
            )
        if (wind_east_m_s or wind_north_m_s) and not include_atmosphere_rotation:
            raise AttributionError(
                "a wind case requires include_atmosphere_rotation=True: the wind "
                "term is computed as a correction to the co-rotating relative "
                "velocity, so switching co-rotation off would leave the "
                "correction measured from a baseline that was never used"
            )
        self.thrust = thrust
        self.wind_east_m_s = float(wind_east_m_s)
        self.wind_north_m_s = float(wind_north_m_s)

    @property
    def has_wind(self) -> bool:
        return self.wind_east_m_s != 0.0 or self.wind_north_m_s != 0.0

    def _wind_drag_correction(
        self, t_s: float, state: np.ndarray, epoch: _dt.datetime
    ) -> np.ndarray:
        """Drag against gas that is moving, minus drag against gas that is not."""
        position = state[:3]
        velocity = state[3:]
        if geodetic_altitude(position) < REENTRY_ALTITUDE_M:
            return np.zeros(3)

        when = epoch + _dt.timedelta(seconds=float(t_s))
        altitude = geodetic_altitude(position)
        gmst = greenwich_sidereal_time(when)
        latitude, longitude = geodetic_latlon(position, gmst)
        density = self.density_function(when, altitude, latitude, longitude)
        if density <= 0.0:
            return np.zeros(3)

        with_wind = relative_velocity_with_wind(
            velocity, position, self.wind_east_m_s, self.wind_north_m_s, gmst
        )
        without_wind = velocity - corotation_velocity(position)
        b = self.spacecraft.inverse_ballistic_coefficient
        drag_with = -0.5 * b * density * float(np.linalg.norm(with_wind)) * with_wind
        drag_without = (
            -0.5 * b * density * float(np.linalg.norm(without_wind)) * without_wind
        )
        return drag_with - drag_without

    def _derivative(self, t_s: float, state: np.ndarray, epoch: _dt.datetime) -> np.ndarray:
        derivative = super()._derivative(t_s, state, epoch)
        acceleration = derivative[3:]
        if self.has_wind:
            acceleration = acceleration + self._wind_drag_correction(t_s, state, epoch)
        if self.thrust is not None and t_s >= self.thrust.start_delay_s:
            velocity = state[3:]
            speed = float(np.linalg.norm(velocity))
            if speed > 0.0:
                acceleration = (
                    acceleration + self.thrust.acceleration_m_s2 * velocity / speed
                )
        return np.concatenate([derivative[:3], acceleration])


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """One counterfactual: an atmosphere, an orbit, a C_d*A/m, a wind and a thrust.

    ``letter`` is the reviewer's label. ``physical_key`` is derived from the
    physics alone, so two letters describing the same propagation — F is B at the
    median C_d*A/m — share one run instead of being integrated twice and then
    differing by integrator noise.
    """

    letter: str
    title: str
    atmosphere_mode: str
    enhancement_factor: float
    perigee_altitude_m: float
    apogee_altitude_m: float
    inverse_ballistic_coefficient_m2_kg: float
    thrust: ThrustProfile | None = None
    wind_east_m_s: float = 0.0
    wind_north_m_s: float = 0.0
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.wind_east_m_s) or not math.isfinite(self.wind_north_m_s):
            raise AttributionError(
                f"wind components must be finite, got east={self.wind_east_m_s}, "
                f"north={self.wind_north_m_s}"
            )
        if self.atmosphere_mode not in ("storm", "quiet"):
            raise AttributionError(
                f"atmosphere_mode must be 'storm' or 'quiet', got {self.atmosphere_mode!r}"
            )
        if self.enhancement_factor <= 0:
            raise AttributionError(
                f"enhancement factor must be positive, got {self.enhancement_factor}"
            )
        if self.atmosphere_mode == "quiet" and self.enhancement_factor != 1.0:
            raise AttributionError(
                "quiet cases must carry enhancement_factor 1.0: the scaled-storm "
                "provider amplifies a storm-minus-quiet difference that is "
                "identically zero under quiet drivers, so a non-unit factor would "
                f"change nothing and misdescribe the run (got {self.enhancement_factor})"
            )
        if self.inverse_ballistic_coefficient_m2_kg <= 0:
            raise AttributionError(
                "inverse ballistic coefficient C_d*A/m must be positive, got "
                f"{self.inverse_ballistic_coefficient_m2_kg} m^2/kg"
            )
        if self.apogee_altitude_m < self.perigee_altitude_m:
            raise AttributionError(
                f"apogee ({self.apogee_altitude_m} m) is below perigee "
                f"({self.perigee_altitude_m} m)"
            )

    @property
    def ballistic_coefficient_kg_m2(self) -> float:
        """The CONVENTIONAL ballistic coefficient m/(C_d*A), in kg/m^2."""
        return to_ballistic_coefficient(self.inverse_ballistic_coefficient_m2_kg)

    @property
    def physical_key(self) -> str:
        thrust = (
            "coast"
            if self.thrust is None
            else f"a{self.thrust.acceleration_m_s2:g}d{self.thrust.start_delay_days:g}"
        )
        return (
            f"{self.atmosphere_mode}|k{self.enhancement_factor:g}"
            f"|{self.perigee_altitude_m / 1000.0:.0f}x{self.apogee_altitude_m / 1000.0:.0f}km"
            f"|CdA_m{self.inverse_ballistic_coefficient_m2_kg:.5f}"
            f"|wind{self.wind_east_m_s:+.0f}E{self.wind_north_m_s:+.0f}N"
            f"|{thrust}"
        )

    @property
    def display_name(self) -> str:
        """Human label naming every parameter that distinguishes this case."""
        thrust = (
            "coasting"
            if self.thrust is None
            else (
                f"thrust {self.thrust.acceleration_m_s2:g} m/s^2 from "
                f"{self.thrust.start_delay_days:g} d"
            )
        )
        return (
            f"{self.letter} [{self.atmosphere_mode} k={self.enhancement_factor:g}, "
            f"{self.perigee_altitude_m / 1000.0:.0f}x{self.apogee_altitude_m / 1000.0:.0f} km, "
            f"C_d*A/m={self.inverse_ballistic_coefficient_m2_kg:.4f} m^2/kg, "
            f"wind {self.wind_east_m_s:+.0f} m/s E, {thrust}]"
        )

    def area_m2(
        self,
        *,
        mass_kg: float = event.MASS_KG_NOMINAL,
        drag_coefficient: float = REFERENCE_DRAG_COEFFICIENT,
    ) -> float:
        """Area implied by C_d*A/m at a GIVEN C_d and mass. Not a measurement."""
        if mass_kg <= 0:
            raise AttributionError(f"mass must be positive, got {mass_kg}")
        if drag_coefficient <= 0:
            raise AttributionError(f"drag coefficient must be positive, got {drag_coefficient}")
        return self.inverse_ballistic_coefficient_m2_kg * mass_kg / drag_coefficient

    def initial_state(self) -> OrbitState:
        """Cartesian state at deployment for this case's orbit."""
        position, velocity = state_from_perigee_apogee(
            self.perigee_altitude_m,
            self.apogee_altitude_m,
            math.radians(event.INSERTION_INCLINATION_DEG),
        )
        return OrbitState(
            event.DEPLOYMENT_UTC,
            (float(position[0]), float(position[1]), float(position[2])),
            (float(velocity[0]), float(velocity[1]), float(velocity[2])),
        )


@dataclass(frozen=True, slots=True)
class CaseLifetime:
    """One propagated case: its specification, its lifetime and its end state."""

    case: CaseSpec
    lifetime: Lifetime
    final_perigee_altitude_km: float
    final_apogee_altitude_km: float
    window_days: float
    notes: tuple[str, ...] = ()

    @property
    def physical_key(self) -> str:
        return self.case.physical_key


def lifetime_from_propagation(
    result: PropagationResult, window_days: float
) -> tuple[Lifetime, tuple[str, ...]]:
    """Read a lifetime, censored where appropriate, out of a propagation.

    Three endings are distinguished because they mean different things. A
    reentry is a lifetime. Reaching the end of the window is a lower bound.
    Leaving through the top of the integration domain — which a thrusting case
    does — is also a lower bound, but for the opposite reason, so it keeps its
    own termination string instead of being blurred into the second.
    """
    if window_days <= 0:
        raise AttributionError(f"window must be positive days, got {window_days}")

    decay = result.decay_days
    if decay is not None:
        return Lifetime(days=decay, censored=False, termination="reentered"), ()

    final_altitude_km = result.altitudes_m[-1] / 1000.0
    elapsed_days = (result.times_utc[-1] - result.times_utc[0]).total_seconds() / 86400.0
    if elapsed_days < window_days - 1e-6:
        note = (
            f"Integration stopped at {elapsed_days:.2f} d at {final_altitude_km:.0f} km "
            "without reentry: the orbit left the propagator's 2000 km ceiling while "
            f"climbing. Lifetime is censored at the full {window_days:.0f} d window, "
            "which is still a valid lower bound because an orbit at that altitude does "
            "not decay within it."
        )
        return Lifetime(days=window_days, censored=True, termination="escaped_upward"), (note,)

    note = (
        f"Did not reenter within the {window_days:.0f} d window; final altitude "
        f"{final_altitude_km:.0f} km. The lifetime is a lower bound, never a value."
    )
    return Lifetime(days=window_days, censored=True, termination="window_elapsed"), (note,)


def run_case(
    case: CaseSpec,
    density_function: DensityFunction,
    *,
    mass_kg: float = event.MASS_KG_NOMINAL,
    drag_coefficient: float = REFERENCE_DRAG_COEFFICIENT,
    window_days: float = SIMULATION_WINDOW_DAYS,
    sample_interval_s: float = 3600.0,
    max_step_s: float = 90.0,
) -> CaseLifetime:
    """Propagate one case and return its lifetime.

    ``density_function`` must already encode ``case.atmosphere_mode`` and
    ``case.enhancement_factor``. This function cannot check that, because a
    density callable is opaque by design — that opacity is what lets the same
    trajectory run against a quiet and a storm atmosphere with nothing else
    changed. Callers are expected to build the callable from the case in one
    place so the coupling has a single site.

    A case with neither wind nor thrust runs on the unmodified DragPropagator,
    not on the counterfactual subclass with both terms switched off. The two are
    numerically identical, but routing the plain cases through the class the rest
    of the repository already validated keeps the comparison honest.
    """
    spacecraft = SpacecraftProperties(
        mass_kg=mass_kg,
        reference_area_m2=case.area_m2(mass_kg=mass_kg, drag_coefficient=drag_coefficient),
        drag_coefficient=drag_coefficient,
        name=f"Starlink v1.5 (case {case.letter}: {case.title})",
    )
    propagator: DragPropagator
    if case.thrust is None and case.wind_east_m_s == 0.0 and case.wind_north_m_s == 0.0:
        propagator = DragPropagator(density_function, spacecraft, include_j2=True)
    else:
        propagator = CounterfactualPropagator(
            density_function,
            spacecraft,
            thrust=case.thrust,
            wind_east_m_s=case.wind_east_m_s,
            wind_north_m_s=case.wind_north_m_s,
            include_j2=True,
        )

    result = propagator.propagate(
        case.initial_state(),
        window_days,
        sample_interval_s=sample_interval_s,
        max_step_s=max_step_s,
    )
    lifetime, notes = lifetime_from_propagation(result, window_days)
    return CaseLifetime(
        case=case,
        lifetime=lifetime,
        final_perigee_altitude_km=result.perigee_altitude_m[-1] / 1000.0,
        final_apogee_altitude_km=result.apogee_altitude_m[-1] / 1000.0,
        window_days=window_days,
        notes=notes,
    )


# --------------------------------------------------------------------------
# The ten reviewer cases, the secondary cases, and the envelope grid.
# --------------------------------------------------------------------------

_SAFE_MODE_ASSUMPTION = (
    "COUNTERFACTUAL BY CONSTRUCTION: the real Group 4-7 vehicles were in safe "
    "mode and did NOT orbit-raise. No thrust case here describes anything that "
    "happened."
)
_RAISED_ORBIT_ASSUMPTION = (
    "ASSUMPTION, not an operator plan: 320x340 km is used as the 'safer' "
    "comparison insertion. SpaceX has not published an alternative it would "
    "have flown."
)
_QUIET_ASSUMPTION = (
    "COUNTERFACTUAL: geomagnetic activity held at ap=4 (Kp~1) with solar flux "
    "at observed values. This orbit was never flown."
)
_OPERATOR_ATTITUDE_ASSUMPTION = (
    "The external reference 0.085 m^2/kg (Shambaugh 2024, arXiv:2406.08342, "
    "p. 6) describes a Capella Space SAR (Whitney) satellite, not SpaceX "
    "hardware or the safe-mode configuration of Starlink Group 4-7."
)


def _storm(
    letter: str,
    title: str,
    *,
    enhancement_factor: float,
    perigee_m: float,
    apogee_m: float,
    inverse_ballistic_coefficient_m2_kg: float,
    thrust: ThrustProfile | None = None,
    wind_east_m_s: float = 0.0,
    assumptions: tuple[str, ...] = (),
) -> CaseSpec:
    return CaseSpec(
        letter=letter,
        title=title,
        atmosphere_mode="storm",
        enhancement_factor=enhancement_factor,
        perigee_altitude_m=perigee_m,
        apogee_altitude_m=apogee_m,
        inverse_ballistic_coefficient_m2_kg=inverse_ballistic_coefficient_m2_kg,
        thrust=thrust,
        wind_east_m_s=wind_east_m_s,
        assumptions=assumptions,
    )


def _quiet(
    letter: str,
    title: str,
    *,
    perigee_m: float,
    apogee_m: float,
    inverse_ballistic_coefficient_m2_kg: float,
    thrust: ThrustProfile | None = None,
    wind_east_m_s: float = 0.0,
    assumptions: tuple[str, ...] = (),
) -> CaseSpec:
    return CaseSpec(
        letter=letter,
        title=title,
        atmosphere_mode="quiet",
        enhancement_factor=1.0,
        perigee_altitude_m=perigee_m,
        apogee_altitude_m=apogee_m,
        inverse_ballistic_coefficient_m2_kg=inverse_ballistic_coefficient_m2_kg,
        thrust=thrust,
        wind_east_m_s=wind_east_m_s,
        assumptions=(_QUIET_ASSUMPTION, *assumptions),
    )


def nominal_thrust(
    *, acceleration_m_s2: float = NOMINAL_THRUST_ACCELERATION_M_S2
) -> ThrustProfile:
    """Thrust starting after the post-deployment checkout."""
    return ThrustProfile(
        acceleration_m_s2=acceleration_m_s2,
        start_delay_days=NOMINAL_THRUST_START_DELAY_DAYS,
    )


def delayed_thrust(
    *, acceleration_m_s2: float = NOMINAL_THRUST_ACCELERATION_M_S2
) -> ThrustProfile:
    """Thrust starting only after a two-day safe-mode recovery."""
    return ThrustProfile(
        acceleration_m_s2=acceleration_m_s2,
        start_delay_days=DELAYED_THRUST_START_DELAY_DAYS,
    )


def ten_cases(
    *,
    enhancement_factor: float = 1.0,
    inverse_ballistic_coefficient_m2_kg: float = (
        FITTED_MEDIAN_INVERSE_BALLISTIC_COEFFICIENT_M2_KG
    ),
    wind_east_m_s: float = 0.0,
) -> tuple[CaseSpec, ...]:
    """The ten cases A-J at one point of the envelope.

    E, F and G are the attitude sweep, so they carry the low, median and high
    fitted C_d*A/m rather than the argument; every other case carries the
    argument. F is therefore the same propagation as B whenever the argument is
    the median, which is why cases are deduplicated by ``physical_key``.
    """
    b = inverse_ballistic_coefficient_m2_kg
    k = enhancement_factor
    w = wind_east_m_s
    return (
        _quiet(
            "A",
            "quiet atmosphere, documented insertion orbit",
            perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
            apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=b,
        ),
        _storm(
            "B",
            "storm atmosphere, documented insertion orbit",
            enhancement_factor=k,
            perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
            apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=b,
        ),
        _quiet(
            "C",
            "quiet atmosphere, raised-perigee orbit",
            perigee_m=RAISED_PERIGEE_ALTITUDE_M,
            apogee_m=RAISED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=b,
            assumptions=(_RAISED_ORBIT_ASSUMPTION,),
        ),
        _storm(
            "D",
            "storm atmosphere, raised-perigee orbit",
            enhancement_factor=k,
            perigee_m=RAISED_PERIGEE_ALTITUDE_M,
            apogee_m=RAISED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=b,
            assumptions=(_RAISED_ORBIT_ASSUMPTION,),
        ),
        _storm(
            "E",
            "storm atmosphere, LOW-DRAG attitude (small C_d*A/m)",
            enhancement_factor=k,
            perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
            apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=(
                FITTED_LOW_DRAG_INVERSE_BALLISTIC_COEFFICIENT_M2_KG
            ),
        ),
        _storm(
            "F",
            "storm atmosphere, NOMINAL attitude (median fitted C_d*A/m)",
            enhancement_factor=k,
            perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
            apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=(
                FITTED_MEDIAN_INVERSE_BALLISTIC_COEFFICIENT_M2_KG
            ),
        ),
        _storm(
            "G",
            "storm atmosphere, HIGH-AREA / unstable attitude (large C_d*A/m)",
            enhancement_factor=k,
            perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
            apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=(
                FITTED_HIGH_AREA_INVERSE_BALLISTIC_COEFFICIENT_M2_KG
            ),
        ),
        _quiet(
            "H",
            "quiet atmosphere, delayed thrust",
            perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
            apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=b,
            thrust=delayed_thrust(),
            assumptions=(_SAFE_MODE_ASSUMPTION,),
        ),
        _storm(
            "I",
            "storm atmosphere, delayed thrust",
            enhancement_factor=k,
            perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
            apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=b,
            thrust=delayed_thrust(),
            assumptions=(_SAFE_MODE_ASSUMPTION,),
        ),
        _storm(
            "J",
            "storm atmosphere, nominal thrust",
            enhancement_factor=k,
            perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
            apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
            wind_east_m_s=w,
            inverse_ballistic_coefficient_m2_kg=b,
            thrust=nominal_thrust(),
            assumptions=(_SAFE_MODE_ASSUMPTION,),
        ),
    )


@dataclass(frozen=True, slots=True)
class EnvelopeCorner:
    """One corner of the plausible uncertainty envelope.

    Three axes are swept: the storm-response amplification, the fitted C_d*A/m
    range, and the zonal neutral wind. The wind axis exists because
    providers.wind supplies both the physics and the swept magnitudes; a caller
    that passes only 0.0 has not swept it, and the report is expected to say so
    rather than let a single-valued axis read as an explored one.
    """

    enhancement_factor: float
    inverse_ballistic_coefficient_m2_kg: float
    wind_east_m_s: float = 0.0

    def __post_init__(self) -> None:
        if self.enhancement_factor <= 0:
            raise AttributionError(
                f"enhancement factor must be positive, got {self.enhancement_factor}"
            )
        if self.inverse_ballistic_coefficient_m2_kg <= 0:
            raise AttributionError(
                "inverse ballistic coefficient C_d*A/m must be positive, got "
                f"{self.inverse_ballistic_coefficient_m2_kg} m^2/kg"
            )
        if not math.isfinite(self.wind_east_m_s):
            raise AttributionError(
                f"zonal wind must be finite, got {self.wind_east_m_s} m/s"
            )

    @property
    def label(self) -> str:
        return (
            f"k={self.enhancement_factor:g}, "
            f"C_d*A/m={self.inverse_ballistic_coefficient_m2_kg:.4f} m^2/kg, "
            f"wind={self.wind_east_m_s:+.0f} m/s E"
        )

    @property
    def ballistic_coefficient_kg_m2(self) -> float:
        return to_ballistic_coefficient(self.inverse_ballistic_coefficient_m2_kg)


def envelope_corners(
    *,
    enhancement_factors: Sequence[float] = DENSITY_ENHANCEMENT_FACTORS,
    inverse_ballistic_coefficients_m2_kg: Sequence[float] = (
        FITTED_LOW_DRAG_INVERSE_BALLISTIC_COEFFICIENT_M2_KG,
        FITTED_MEDIAN_INVERSE_BALLISTIC_COEFFICIENT_M2_KG,
        FITTED_HIGH_AREA_INVERSE_BALLISTIC_COEFFICIENT_M2_KG,
    ),
    wind_east_m_s_values: Sequence[float] = WIND_EAST_CASES_M_S,
) -> tuple[EnvelopeCorner, ...]:
    """Every corner of the swept envelope, in a deterministic order."""
    if (
        not enhancement_factors
        or not inverse_ballistic_coefficients_m2_kg
        or not wind_east_m_s_values
    ):
        raise AttributionError(
            "envelope needs at least one value on every axis; got "
            f"{len(enhancement_factors)} enhancement factors, "
            f"{len(inverse_ballistic_coefficients_m2_kg)} C_d*A/m values, "
            f"{len(wind_east_m_s_values)} wind cases"
        )
    return tuple(
        EnvelopeCorner(
            enhancement_factor=float(k),
            inverse_ballistic_coefficient_m2_kg=float(b),
            wind_east_m_s=float(wind),
        )
        for k in enhancement_factors
        for b in inverse_ballistic_coefficients_m2_kg
        for wind in wind_east_m_s_values
    )


def corner_cases(corner: EnvelopeCorner) -> dict[str, CaseSpec]:
    """Every case one corner's decomposition refers to, keyed by letter.

    The ten reviewer cases plus three secondary ones: G_op is the external
    Capella Space Whitney SAR reference C_d*A/m (Shambaugh 2024, arXiv:2406.08342;
    not SpaceX Starlink), and I_hi/J_hi repeat the thrust-delay pair at a thrust
    magnitude high enough to arrest the decay outright. The secondary cases exist
    so that the effect of the unsourced thrust assumption is visible in the
    output rather than only in this docstring.
    """
    cases = {
        case.letter: case
        for case in ten_cases(
            enhancement_factor=corner.enhancement_factor,
            inverse_ballistic_coefficient_m2_kg=corner.inverse_ballistic_coefficient_m2_kg,
            wind_east_m_s=corner.wind_east_m_s,
        )
    }
    cases["G_op"] = _storm(
        "G_op",
        "storm atmosphere, Capella Space Whitney reference C_d*A/m (Shambaugh 2024; not Starlink)",
        enhancement_factor=corner.enhancement_factor,
        perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
        apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
        inverse_ballistic_coefficient_m2_kg=OPERATOR_INVERSE_BALLISTIC_COEFFICIENT_M2_KG,
        wind_east_m_s=corner.wind_east_m_s,
        assumptions=(_OPERATOR_ATTITUDE_ASSUMPTION,),
    )
    cases["I_hi"] = _storm(
        "I_hi",
        "storm atmosphere, delayed HIGH-AUTHORITY thrust",
        enhancement_factor=corner.enhancement_factor,
        perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
        apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
        inverse_ballistic_coefficient_m2_kg=corner.inverse_ballistic_coefficient_m2_kg,
        thrust=delayed_thrust(acceleration_m_s2=HIGH_AUTHORITY_THRUST_ACCELERATION_M_S2),
        wind_east_m_s=corner.wind_east_m_s,
        assumptions=(_SAFE_MODE_ASSUMPTION,),
    )
    cases["J_hi"] = _storm(
        "J_hi",
        "storm atmosphere, prompt HIGH-AUTHORITY thrust",
        enhancement_factor=corner.enhancement_factor,
        perigee_m=DOCUMENTED_PERIGEE_ALTITUDE_M,
        apogee_m=DOCUMENTED_APOGEE_ALTITUDE_M,
        inverse_ballistic_coefficient_m2_kg=corner.inverse_ballistic_coefficient_m2_kg,
        thrust=nominal_thrust(acceleration_m_s2=HIGH_AUTHORITY_THRUST_ACCELERATION_M_S2),
        wind_east_m_s=corner.wind_east_m_s,
        assumptions=(_SAFE_MODE_ASSUMPTION,),
    )
    return cases


def envelope_case_grid(corners: Sequence[EnvelopeCorner]) -> tuple[CaseSpec, ...]:
    """Every distinct propagation the corner decompositions need.

    Deduplicated by ``physical_key``: F is B at the median C_d*A/m, quiet cases
    do not depend on the enhancement factor, and the attitude cases are shared
    between corners that differ only in C_d*A/m. Running each physical
    configuration once is not merely faster — it also guarantees that two
    references to "the same case" cannot disagree by integrator noise.
    """
    if not corners:
        raise AttributionError("no envelope corners supplied; nothing to run")
    seen: dict[str, CaseSpec] = {}
    for corner in corners:
        for case in corner_cases(corner).values():
            seen.setdefault(case.physical_key, case)
    return tuple(seen.values())


# --------------------------------------------------------------------------
# Decomposition.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorPair:
    """The named counterfactual pair that defines one factor's contribution.

    Both members are ``physical_key`` strings and both human names travel with
    them, because a contribution reported without saying which two runs produced
    it is not reproducible — that omission is the specific defect this module was
    written to repair.
    """

    factor: str
    baseline_key: str
    baseline_name: str
    counterfactual_key: str
    counterfactual_name: str
    ranked: bool = True
    note: str = ""

    @property
    def comparison(self) -> str:
        return f"{self.counterfactual_name} vs {self.baseline_name}"


@dataclass(frozen=True, slots=True)
class FactorContribution:
    """One factor's effect on modelled lifetime, as an interval in days.

    Sign convention, uniform across factors: ``delta = counterfactual -
    baseline``, so a positive contribution means the counterfactual bought
    orbital life and a negative one means it cost orbital life. The fraction is
    ``delta / baseline_days`` and is None whenever the baseline is itself
    censored, because a ratio of two lower bounds is not a fraction.
    """

    factor: str
    comparison: str
    baseline_name: str
    counterfactual_name: str
    baseline: Lifetime
    counterfactual: Lifetime
    delta_days: Interval
    fraction: Interval | None
    ranked: bool = True
    note: str = ""

    @property
    def censored(self) -> bool:
        return self.baseline.censored or self.counterfactual.censored

    @property
    def magnitude_days(self) -> Interval:
        return self.delta_days.magnitude

    def describe_days(self) -> str:
        if self.delta_days.is_exact:
            return f"{self.delta_days.low:+.2f} d"
        low = "-inf" if self.delta_days.low == -math.inf else f"{self.delta_days.low:+.2f}"
        high = "+inf" if self.delta_days.high == math.inf else f"{self.delta_days.high:+.2f}"
        return f"[{low}, {high}] d"

    def describe_fraction(self) -> str:
        if self.fraction is None:
            return "n/a (baseline censored)"
        if self.fraction.is_exact:
            return f"{self.fraction.low:+.1%}"
        low = "-inf" if self.fraction.low == -math.inf else f"{self.fraction.low:+.1%}"
        high = "+inf" if self.fraction.high == math.inf else f"{self.fraction.high:+.1%}"
        return f"[{low}, {high}]"


def contribution(pair: FactorPair, lifetimes: Mapping[str, Lifetime]) -> FactorContribution:
    """Compute one factor's contribution from a table of case lifetimes."""
    for key, role in (
        (pair.baseline_key, "baseline"),
        (pair.counterfactual_key, "counterfactual"),
    ):
        if key not in lifetimes:
            raise AttributionError(
                f"factor {pair.factor!r}: no lifetime for the {role} case {key!r}. "
                "A contribution cannot be reported for a case that was not run."
            )
    baseline = lifetimes[pair.baseline_key]
    counterfactual = lifetimes[pair.counterfactual_key]
    delta = counterfactual.interval - baseline.interval
    fraction = None if baseline.censored else delta.scaled(1.0 / baseline.days)
    return FactorContribution(
        factor=pair.factor,
        comparison=pair.comparison,
        baseline_name=pair.baseline_name,
        counterfactual_name=pair.counterfactual_name,
        baseline=baseline,
        counterfactual=counterfactual,
        delta_days=delta,
        fraction=fraction,
        ranked=pair.ranked,
        note=pair.note,
    )


def decompose(
    pairs: Sequence[FactorPair], lifetimes: Mapping[str, Lifetime]
) -> tuple[FactorContribution, ...]:
    """Compute every named factor contribution from one lifetime table."""
    if not pairs:
        raise AttributionError("no factor pairs supplied; nothing to decompose")
    return tuple(contribution(pair, lifetimes) for pair in pairs)


def factor_pairs_for_corner(
    corner: EnvelopeCorner,
    *,
    enhancement_factors: Sequence[float] = DENSITY_ENHANCEMENT_FACTORS,
) -> tuple[FactorPair, ...]:
    """The five ranked factor pairs at one corner, plus four unranked checks.

    Two of the five ranked factors are themselves envelope axes, and their
    definitions say so. The attitude factor is the SPAN of the fitted C_d*A/m
    range evaluated at the corner's atmosphere, so it does not use the corner's
    own C_d*A/m; the atmosphere-model factor is the span of the enhancement
    sweep evaluated at the corner's C_d*A/m, so it does not use the corner's own
    enhancement factor. That is what "the contribution of this axis" means, and
    stating it here stops it looking like a wiring mistake.
    """
    if not enhancement_factors:
        raise AttributionError("enhancement_factors must not be empty")
    cases = corner_cases(corner)
    low_k = min(float(k) for k in enhancement_factors)
    high_k = max(float(k) for k in enhancement_factors)
    if low_k == high_k:
        raise AttributionError(
            "the atmosphere-model factor needs at least two distinct enhancement "
            f"factors to span; got only {low_k:g}"
        )
    b_low_k = corner_cases(replace(corner, enhancement_factor=low_k))["B"]
    b_high_k = corner_cases(replace(corner, enhancement_factor=high_k))["B"]

    def pair(
        factor: str,
        baseline: CaseSpec,
        counterfactual: CaseSpec,
        note: str,
        *,
        ranked: bool = True,
    ) -> FactorPair:
        return FactorPair(
            factor=factor,
            baseline_key=baseline.physical_key,
            baseline_name=baseline.display_name,
            counterfactual_key=counterfactual.physical_key,
            counterfactual_name=counterfactual.display_name,
            ranked=ranked,
            note=note,
        )

    return (
        pair(
            FACTOR_INSERTION_ALTITUDE,
            cases["B"],
            cases["D"],
            "Matched storm atmosphere and matched C_d*A/m; only the insertion "
            "orbit differs (210x338 km -> 320x340 km).",
        ),
        pair(
            FACTOR_STORM_DENSITY,
            cases["B"],
            cases["A"],
            "Matched documented orbit and matched C_d*A/m; only the geomagnetic "
            "drivers differ (observed -> ap=4 counterfactual).",
        ),
        pair(
            FACTOR_ATTITUDE,
            cases["E"],
            cases["G"],
            "Matched storm atmosphere and matched orbit; only C_d*A/m differs, "
            "across the range fitted from the catalogued decays "
            f"({FITTED_LOW_DRAG_INVERSE_BALLISTIC_COEFFICIENT_M2_KG:.4f} -> "
            f"{FITTED_HIGH_AREA_INVERSE_BALLISTIC_COEFFICIENT_M2_KG:.4f} m^2/kg).",
        ),
        pair(
            FACTOR_THRUST_DELAY,
            cases["I"],
            cases["J"],
            "Matched storm atmosphere, orbit and C_d*A/m; only the thrust start "
            f"time differs ({DELAYED_THRUST_START_DELAY_DAYS:g} d -> "
            f"{NOMINAL_THRUST_START_DELAY_DAYS:g} d). Both sides are "
            "counterfactual: the real vehicles never thrusted.",
        ),
        pair(
            FACTOR_ATMOSPHERE_MODEL,
            b_low_k,
            b_high_k,
            "Matched documented orbit and matched C_d*A/m; only the assumed "
            f"storm-response amplification differs (x{low_k:g} -> x{high_k:g}), "
            "which is the scalar density stress test already in this repository.",
        ),
        pair(
            "insertion altitude (quiet-atmosphere check)",
            cases["A"],
            cases["C"],
            "The same insertion comparison under the quiet counterfactual "
            "atmosphere, to show the factor is not an artefact of the storm.",
            ranked=False,
        ),
        pair(
            "storm density (raised-orbit check)",
            cases["D"],
            cases["C"],
            "The same storm comparison from the raised orbit, to show how the "
            "storm's effect changes with insertion altitude.",
            ranked=False,
        ),
        pair(
            "attitude C_d*A/m (Capella reference value)",
            cases["E"],
            cases["G_op"],
            "The low-drag case against the external Capella SAR reference "
            "0.085 m^2/kg (Shambaugh 2024, arXiv:2406.08342), which lies well "
            "above the fitted range and describes a different spacecraft.",
            ranked=False,
        ),
        pair(
            "thrust delay (high-authority regime)",
            cases["I_hi"],
            cases["J_hi"],
            "The same delay at "
            f"{HIGH_AUTHORITY_THRUST_ACCELERATION_M_S2:g} m/s^2 instead of "
            f"{NOMINAL_THRUST_ACCELERATION_M_S2:g}, showing how much of the "
            "ranked thrust contribution is an artefact of the assumed magnitude.",
            ranked=False,
        ),
    )


def rank_factors(contributions: Sequence[FactorContribution]) -> tuple[str, ...]:
    """Factor names ordered by effect size, largest first.

    Ranking is on |delta|, because a factor that costs orbital life and a factor
    that buys it are both explanations of the outcome and the question is which
    explains more. Censored contributions are ordered by the lower bound of their
    magnitude; whether that ordering is actually DETERMINED by the data is a
    separate question, answered by ``unresolved_factor_pairs``.
    """
    ranked = [c for c in contributions if c.ranked]
    if not ranked:
        raise AttributionError(
            "no ranked contributions supplied; every contribution was marked "
            "ranked=False, so there is no ordering to compute"
        )
    names = [c.factor for c in ranked]
    if len(set(names)) != len(names):
        raise AttributionError(f"duplicate ranked factor names: {sorted(names)}")
    return tuple(
        c.factor
        for c in sorted(
            ranked, key=lambda c: (-c.magnitude_days.low, -c.magnitude_days.high, c.factor)
        )
    )


def unresolved_factor_pairs(
    contributions: Sequence[FactorContribution],
) -> tuple[tuple[str, str], ...]:
    """Factor pairs whose magnitude intervals overlap, so their order is undetermined.

    Two exact contributions overlap only when they are equal, which is a genuine
    tie. A censored contribution has an unbounded magnitude interval and overlaps
    anything at least as large, which is the honest answer: the run did not
    establish which of the two is bigger.
    """
    ranked = [c for c in contributions if c.ranked]
    out: set[tuple[str, str]] = set()
    for index, first in enumerate(ranked):
        for second in ranked[index + 1 :]:
            if first.magnitude_days.overlaps(second.magnitude_days):
                names = sorted((first.factor, second.factor))
                out.add((names[0], names[1]))
    return tuple(sorted(out))


# --------------------------------------------------------------------------
# The key output: is the ordering stable across the envelope?
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderingStability:
    """Whether the factor ranking survives the uncertainty envelope.

    ``stable`` is True only when every corner produces the same ranking AND no
    corner leaves a pair unresolved. Both conditions are necessary: an ordering
    that is identical everywhere but rests on overlapping intervals has not been
    established, it has merely been printed in the same order each time.
    """

    corner_rankings: dict[str, tuple[str, ...]]
    reference_corner: str
    reference_ranking: tuple[str, ...]
    stable: bool
    flip_corners: tuple[str, ...]
    unresolved_pairs: dict[str, tuple[tuple[str, str], ...]]
    dominant_by_corner: dict[str, str | None]
    dominant_factor: str | None
    dominant_margin_days: float | None
    corners_tested: int
    wind_cases_tested: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def corners_by_dominant_factor(self) -> dict[str, tuple[str, ...]]:
        """The regime map: which factor dominates under which conditions.

        Corners where the top of the ranking is itself contested are collected
        under ``UNRESOLVED_LEADER`` rather than being silently credited to
        whichever factor happened to sort first.
        """
        out: dict[str, list[str]] = {}
        for corner, factor in self.dominant_by_corner.items():
            out.setdefault(UNRESOLVED_LEADER if factor is None else factor, []).append(corner)
        return {name: tuple(corners) for name, corners in out.items()}

    def rank_positions(self) -> dict[str, tuple[int, ...]]:
        """Every rank each factor takes across the corners, 1-based and sorted.

        A factor whose tuple has one element held the same position everywhere.
        That is a weaker claim than a stable full ordering but a real one, and it
        is often the only claim the data support — for instance that a factor is
        the smallest contributor at every corner even though the top of the
        ranking moves.
        """
        positions: dict[str, set[int]] = {}
        for ranking in self.corner_rankings.values():
            for index, factor in enumerate(ranking):
                positions.setdefault(factor, set()).add(index + 1)
        return {factor: tuple(sorted(ranks)) for factor, ranks in sorted(positions.items())}


def ordering_is_stable(
    contributions_by_corner: Mapping[str, Sequence[FactorContribution]],
    *,
    reference_corner: str | None = None,
    wind_cases_tested: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> OrderingStability:
    """Decide whether the factor ranking is the same at every envelope corner.

    Returns the ranking per corner, the boolean, and the corners where it flips.
    A False here is not a failure of the analysis: it is the analysis's actual
    finding, namely that the data support a regime boundary and not a single
    dominant cause.
    """
    if not contributions_by_corner:
        raise AttributionError(
            "no envelope corners supplied; stability across an empty envelope is "
            "not a result"
        )

    factor_sets = {
        corner: frozenset(c.factor for c in contributions if c.ranked)
        for corner, contributions in contributions_by_corner.items()
    }
    corners = list(contributions_by_corner)
    first_corner = corners[0]
    for corner in corners[1:]:
        if factor_sets[corner] != factor_sets[first_corner]:
            raise AttributionError(
                "corners rank different factor sets, so no ordering can be "
                f"compared: {first_corner!r} ranks {sorted(factor_sets[first_corner])}, "
                f"{corner!r} ranks {sorted(factor_sets[corner])}"
            )

    if reference_corner is None:
        reference_corner = first_corner
    elif reference_corner not in contributions_by_corner:
        raise AttributionError(
            f"reference corner {reference_corner!r} is not among the corners "
            f"supplied: {corners}"
        )

    rankings = {
        corner: rank_factors(contributions)
        for corner, contributions in contributions_by_corner.items()
    }
    unresolved = {
        corner: unresolved_factor_pairs(contributions)
        for corner, contributions in contributions_by_corner.items()
    }

    dominant_by_corner: dict[str, str | None] = {}
    for corner, ranking in rankings.items():
        top = ranking[0]
        contested = any(top in names for names in unresolved[corner])
        dominant_by_corner[corner] = None if contested else top

    reference_ranking = rankings[reference_corner]
    flips = tuple(
        corner for corner, ranking in rankings.items() if ranking != reference_ranking
    )
    any_unresolved = any(pairs for pairs in unresolved.values())
    stable = not flips and not any_unresolved

    dominant: str | None = None
    margin: float | None = None
    if stable:
        dominant = reference_ranking[0]
        margins: list[float] = []
        for contributions in contributions_by_corner.values():
            by_name = {c.factor: c for c in contributions if c.ranked}
            ranking = rank_factors(contributions)
            if len(ranking) < 2:
                continue
            margins.append(
                by_name[ranking[0]].magnitude_days.low - by_name[ranking[1]].magnitude_days.high
            )
        margin = min(margins) if margins else None

    return OrderingStability(
        corner_rankings=rankings,
        reference_corner=reference_corner,
        reference_ranking=reference_ranking,
        stable=stable,
        flip_corners=flips,
        unresolved_pairs=unresolved,
        dominant_by_corner=dominant_by_corner,
        dominant_factor=dominant,
        dominant_margin_days=margin,
        corners_tested=len(contributions_by_corner),
        wind_cases_tested=tuple(wind_cases_tested),
        notes=tuple(notes),
    )


def validate_conclusion(text: str, stability: OrderingStability) -> None:
    """Refuse a conclusion that claims primacy the ordering did not establish.

    Called on the generated narrative before it is written out, so the rule is
    enforced by code rather than by the discipline of whoever edits the markdown
    next.
    """
    if stability.stable:
        return
    lowered = text.lower()
    for phrase in PRIMACY_PHRASES:
        if phrase in lowered:
            raise AttributionError(
                f"conclusion claims causal primacy ({phrase!r}) but the factor "
                "ordering was NOT stable across the envelope: it flips at "
                f"{len(stability.flip_corners)} of {stability.corners_tested} "
                "corners. Report the regime boundary instead."
            )


def conclusion_text(stability: OrderingStability) -> str:
    """The conclusion the decomposition actually supports, and nothing more."""
    if stability.stable:
        margin = stability.dominant_margin_days
        margin_text = "" if margin is None else f", by a margin of at least {margin:.2f} d"
        return (
            f"ORDERING IS STABLE across all {stability.corners_tested} envelope "
            f"corners tested. The dominant factor is {stability.dominant_factor}"
            f"{margin_text}. The ranking is "
            + " > ".join(stability.reference_ranking)
            + "."
        )
    regimes = stability.corners_by_dominant_factor
    total = stability.corners_tested
    parts = [
        f"{factor} leads at {len(corner_labels)} of {total} corners"
        for factor, corner_labels in sorted(regimes.items())
        if factor != UNRESOLVED_LEADER
    ]
    contested = regimes.get(UNRESOLVED_LEADER, ())
    if contested:
        parts.append(
            f"at {len(contested)} of {total} corners no factor leads at all, "
            "because a counterfactual there did not reenter inside the window and "
            "its contribution is a bound rather than a value"
        )
    return (
        "ORDERING IS NOT STABLE. The ranking changes at "
        f"{len(stability.flip_corners)} of {total} envelope corners. What the "
        "decomposition establishes is a REGIME BOUNDARY, not a single dominant "
        "factor: " + "; ".join(parts) + ". This analysis therefore declines to "
        "name one factor as decisive, because the data do not order the factors "
        "the same way everywhere and choosing a corner to quote would be choosing "
        "a narrative."
    )
