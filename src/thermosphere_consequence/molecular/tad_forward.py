"""Forward travelling-atmospheric-disturbance arrival times: WHEN, not HOW FAST.

:mod:`~thermosphere_consequence.molecular.disturbances` solves the INVERSE
problem. Given a Hovmoller diagram it measures a meridional phase speed, or
refuses to state one. This module solves the FORWARD problem, which is the one
an operator actually asks: a storm has just started at a known place and a
known time -- when does the disturbance reach MY latitude, and how much density
enhancement does it still carry when it gets there?

The two are deliberately separate modules and deliberately share their
machinery. The published band this module predicts from
(:data:`~thermosphere_consequence.molecular.disturbances.PUBLISHED_PHASE_SPEED_M_S`),
the arc-distance convention, the signed perturbation record
(:class:`~thermosphere_consequence.molecular.disturbances.DensityPerturbation`)
and the caveat that agreement with a modelled band is agreement between models
are all imported from there rather than restated, so the forward prediction and
the inverse measurement cannot drift into using different numbers for the same
physical quantity. A round-trip test drives that consistency: a speed goes into
:func:`arrival_time`, the predicted arrival times are used to build a synthetic
Hovmoller, and
:func:`~thermosphere_consequence.molecular.disturbances.estimate_phase_speed`
is asked to recover the speed that started the chain.

WHAT THIS MODULE IS. Arithmetic on a distance and a speed, plus the bookkeeping
that stops the answer being read as more than that. It computes:

  * the great-circle arc from a disturbance source to a target latitude
    (:func:`great_circle_distance_m`), at Earth radius plus the altitude the
    disturbance travels at;
  * an arrival instant for ONE stated speed, returned WITH the distance and the
    travel time so a reader can redo the division (:func:`arrival_time`);
  * an arrival WINDOW from a speed RANGE (:func:`arrival_window`), which is the
    honest form of the answer, because no single phase speed is knowable in
    advance;
  * the fractional density enhancement expected on arrival, under an
    attenuation law the caller states (:func:`predicted_enhancement_at_arrival`);
  * whether an observed perturbation series actually fell inside the predicted
    window (:func:`compare_forward_to_observed`);
  * the local sound speed, as the ceiling a freely propagating gravity wave
    does not much exceed (:func:`sound_speed_bound`).

WHAT IT IS NOT, stated because a timestamp is the most quotable thing this
repository produces:

  * A PREDICTION IS NOT A DETECTION. :func:`arrival_time` will return an
    instant for any positive speed it is given, including one no gravity wave
    can have. The number is a consequence of the speed that was assumed, and
    :data:`FORWARD_PREDICTION_IS_NOT_A_DETECTION` travels with every window.

  * IT IS NOT VALIDATED AGAINST ANYTHING IN THIS REPOSITORY. The one
    travelling-disturbance signal measured here -- ``evidence/TRAVELLING_DISTURBANCES.md``
    -- was found NOT to be a travelling atmospheric disturbance: it runs
    POLEWARD in both hemispheres, it is supersonic at 1070-2192 m/s against a
    675-773 m/s sound speed, and in the southern hemisphere it is present at
    the same sign and magnitude in a geomagnetically QUIETER control window.
    Checking this estimator against that signal would be tuning a forward model
    to a structure that is not the thing it models. See
    :data:`NOT_VALIDATED_AGAINST_THE_MEASURED_SIGNAL`.

  * IT DOES NOT KNOW THE SPEED. That is why :func:`arrival_window` exists and
    why the single-speed :func:`arrival_time` is the lower-level function
    rather than the headline one.

THE ONE PLACE THE TWO MODULES DISAGREE, AND IT IS ON PURPOSE.
:func:`~thermosphere_consequence.molecular.disturbances.meridional_distance_m`
measures on the sphere of mean Earth radius; a disturbance at 210 km travels on
a sphere 3.3 per cent larger. ``altitude_m`` here defaults to 0.0 so the two
agree by default and are directly comparable, and a caller who wants the arc at
the level the disturbance really travels passes it. The choice is an argument
rather than a constant precisely so it cannot be made silently.

Canonical tickets: the forward half of the P06_08 travelling-disturbance group.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..domain.constants import AMU, K_B, R_EARTH_MEAN
from ..domain.temporal import ensure_utc
from .disturbances import (
    AGREEMENT_IS_NOT_OBSERVATIONAL_SUPPORT,
    PUBLISHED_PHASE_SPEED_M_S,
    PUBLISHED_REFERENCE,
    DensityPerturbation,
)

__all__ = [
    "AP_STORM_ONSET_PROXY",
    "AP_STORM_ONSET_PROXY_CITATION",
    "DEFAULT_DETECTION_THRESHOLD",
    "FORWARD_PREDICTION_IS_NOT_A_DETECTION",
    "NOT_VALIDATED_AGAINST_THE_MEASURED_SIGNAL",
    "PATH_CHORD",
    "PATH_GREAT_CIRCLE",
    "THERMOSPHERIC_GAMMA_RANGE",
    "UNATTENUATED",
    "ArrivalPrediction",
    "ArrivalVerdict",
    "ArrivalWindow",
    "AttenuationLaw",
    "EnhancementPrediction",
    "ForwardComparison",
    "Hemisphere",
    "SoundSpeedBound",
    "TadSource",
    "arrival_time",
    "arrival_window",
    "compare_forward_to_observed",
    "exponential_attenuation",
    "great_circle_distance_m",
    "predicted_enhancement_at_arrival",
    "sound_speed_bound",
]

_SECONDS_PER_HOUR = 3600.0

#: A worked default onset proxy: the 3-hourly ap index crossing 48. Offered as
#: a value a caller may pass, never as a default on :class:`TadSource`, which
#: requires the proxy to be stated -- an onset with no stated marker is a time
#: somebody chose, and the arrival time inherits every hour of that choice.
AP_STORM_ONSET_PROXY = "3-hourly ap crossing 48 (Kp 5o)"

AP_STORM_ONSET_PROXY_CITATION = (
    "NOAA Space Weather Scales place the G1 minor-storm threshold at Kp 5, "
    "which is ap 48 on the GFZ Kp-to-ap conversion table "
    "(https://kp.gfz-potsdam.de/en/data). The February 2022 window analysed in "
    "this repository reaches ap 56 and Kp 5.3, so this threshold is crossed "
    "within it (evidence/TRAVELLING_DISTURBANCES.md)."
)

FORWARD_PREDICTION_IS_NOT_A_DETECTION = (
    "This is a PREDICTION computed from an assumed phase speed and an assumed "
    "onset time. It says when a disturbance travelling at that speed from that "
    "onset would arrive; it does not say that a disturbance was launched, that "
    "it survived to this latitude, or that anything arrived at all. Nothing "
    "was measured to produce it. A window that turns out to contain an "
    "observed perturbation is consistent with the assumption, and consistency "
    "with an assumption is not evidence for it."
)

NOT_VALIDATED_AGAINST_THE_MEASURED_SIGNAL = (
    "This estimator is NOT validated against the travelling-disturbance signal "
    "measured in this repository. evidence/TRAVELLING_DISTURBANCES.md reports "
    "a coherent meridional phase progression at 210 km that is POLEWARD in "
    "both hemispheres, supersonic at 1070-2192 m/s against a 675-773 m/s local "
    "sound speed, and -- in the southern hemisphere -- present at the same "
    "sign and comparable magnitude in a quieter control window. That signal is "
    "therefore not a travelling atmospheric disturbance, and a forward "
    "estimator checked against it would be checked against the wrong thing. "
    "The only quantitative anchors used here are the published 500-800 m/s "
    "band and the arithmetic of distance over speed."
)

#: What counts as a disturbance having arrived, as a fraction of the background
#: density. A STATED choice and not a measured one, which is why it is an
#: argument to :func:`compare_forward_to_observed` and is carried on the result:
#: a threshold picked after seeing the series would decide the verdict.
DEFAULT_DETECTION_THRESHOLD = 0.02

#: Ratio of specific heats at thermospheric altitudes, as a RANGE rather than a
#: value. At 210 km the gas is atomic-oxygen dominated -- monatomic, 5/3 --
#: with a diatomic N2 admixture at 7/5, so the mixture's gamma lies between
#: them and is not worth pretending to know better. The sound-speed bound is
#: evaluated at both ends. These are the same limits behind the 675-773 m/s
#: figure in evidence/TRAVELLING_DISTURBANCES.md.
THERMOSPHERIC_GAMMA_RANGE: tuple[float, float] = (1.40, 1.67)

PATH_GREAT_CIRCLE = (
    "great-circle arc along the meridian, on a sphere of mean Earth radius plus "
    "the stated propagation altitude"
)

PATH_CHORD = (
    "straight chord between the two points, which passes THROUGH the sphere. "
    "Shorter than the arc, and not a path a disturbance in the atmosphere can "
    "take; offered only so the size of the spherical correction is visible"
)


class Hemisphere(StrEnum):
    """Which auroral oval launched the disturbance.

    Carried explicitly rather than inferred from the sign of the latitude,
    because it is the claim being made -- a storm launches a disturbance in
    BOTH ovals, and a prediction is about one of them.
    """

    NORTHERN = "northern"
    SOUTHERN = "southern"


class ArrivalVerdict(StrEnum):
    """Where an observed onset fell relative to a predicted arrival window."""

    #: The perturbation was already there before the window opened.
    BEFORE = "before"

    #: The perturbation crossed the threshold inside the window.
    INSIDE = "inside"

    #: The perturbation did not cross the threshold until after the window shut.
    AFTER = "after"

    #: Nothing in the series crossed the threshold at all. Distinct from
    #: ``AFTER``: the disturbance was not late, it was absent.
    NO_SIGNAL = "no_signal"


def _checked_latitude(value: float, field: str) -> float:
    """A finite latitude inside the closed range the sphere has."""
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value}")
    if value < -90.0 or value > 90.0:
        raise ValueError(f"{field} must lie in [-90, 90] degrees, got {value}")
    return float(value)


@dataclass(frozen=True, slots=True)
class TadSource:
    """Where and when a disturbance was launched, and what says so.

    ``driver_proxy`` is the OBSERVABLE that marks the onset -- an ap or AE
    threshold crossing, a sudden storm commencement, a substorm expansion --
    and ``driver_proxy_citation`` is why that marker was chosen. Both are
    required and both are checked non-blank, because the onset time is the
    largest uncontrolled term in every arrival time this module produces: a
    proxy that fires an hour early moves every predicted arrival an hour early,
    and there is nothing downstream that can recover it. A source that records
    only an instant hides that entirely.

    ``hemisphere`` must agree with the sign of ``source_latitude_deg``. It is
    stored anyway rather than derived, so a run record says which oval was
    meant instead of leaving a reader to infer it from a number.
    """

    onset_utc: _dt.datetime
    source_latitude_deg: float
    source_longitude_deg: float
    hemisphere: Hemisphere
    driver_proxy: str
    driver_proxy_citation: str

    def __post_init__(self) -> None:
        if self.onset_utc.tzinfo is None:
            raise ValueError(
                "TadSource.onset_utc must be timezone-aware; an onset without an "
                "offset is an instant nobody can place against a driver index"
            )
        _checked_latitude(self.source_latitude_deg, "source_latitude_deg")
        if not math.isfinite(self.source_longitude_deg):
            raise ValueError(
                f"source_longitude_deg must be finite, got {self.source_longitude_deg}"
            )
        if not isinstance(self.hemisphere, Hemisphere):
            raise TypeError(
                f"hemisphere must be a Hemisphere, got {type(self.hemisphere).__name__}"
            )
        if self.source_latitude_deg > 0.0 and self.hemisphere is not Hemisphere.NORTHERN:
            raise ValueError(
                f"source_latitude_deg {self.source_latitude_deg} is northern but "
                f"hemisphere says {self.hemisphere.value}. The two must agree, or "
                "the run record names an oval the source is not in."
            )
        if self.source_latitude_deg < 0.0 and self.hemisphere is not Hemisphere.SOUTHERN:
            raise ValueError(
                f"source_latitude_deg {self.source_latitude_deg} is southern but "
                f"hemisphere says {self.hemisphere.value}. The two must agree, or "
                "the run record names an oval the source is not in."
            )
        if not self.driver_proxy.strip():
            raise ValueError(
                "driver_proxy must say what marks the onset -- an ap or AE "
                "threshold crossing, a sudden storm commencement. An onset with "
                "no stated marker is a time somebody chose, and every arrival "
                "time computed from it inherits that choice invisibly."
            )
        if not self.driver_proxy_citation.strip():
            raise ValueError(
                "driver_proxy_citation must say where the onset marker comes "
                "from. A threshold with no source is a threshold that was tuned."
            )


def great_circle_distance_m(
    from_latitude_deg: float,
    to_latitude_deg: float,
    *,
    altitude_m: float = 0.0,
    great_circle: bool = True,
) -> float:
    """Unsigned distance between two latitudes on the same meridian, in metres.

    ``great_circle=True`` (the default, and the only physical one) is the arc
    ``R * dtheta`` on a sphere of mean Earth radius plus ``altitude_m`` -- the
    path a disturbance propagating through the atmosphere actually follows.

    ``great_circle=False`` is the CHORD, ``2 R sin(dtheta/2)``, the straight
    line between the two points through the interior of the sphere. Nothing in
    the atmosphere travels along it. It exists so the size of the spherical
    correction is visible rather than asserted: over 40 degrees of latitude the
    chord is about 2 per cent short of the arc, which is well inside the
    500-800 m/s speed range and therefore not the term that limits a forward
    prediction.

    ``altitude_m`` defaults to 0.0, which puts this on the SAME sphere as
    :func:`~thermosphere_consequence.molecular.disturbances.meridional_distance_m`
    so a forward prediction and an inverse measurement are directly comparable.
    At 210 km the arc is 3.3 per cent longer than at the surface.
    """
    if not math.isfinite(altitude_m) or altitude_m < 0.0:
        raise ValueError(
            f"altitude_m must be finite and non-negative, got {altitude_m}"
        )
    radius = R_EARTH_MEAN.value + altitude_m
    separation = abs(
        math.radians(
            _checked_latitude(from_latitude_deg, "from_latitude_deg")
            - _checked_latitude(to_latitude_deg, "to_latitude_deg")
        )
    )
    if great_circle:
        return radius * separation
    return 2.0 * radius * math.sin(separation / 2.0)


@dataclass(frozen=True, slots=True)
class ArrivalPrediction:
    """One arrival instant, with the distance and the time that produced it.

    The distance and the travel time are returned beside the arrival because
    the whole calculation is one division, and a reader who can see both
    operands can check it in their head. An arrival instant on its own is the
    same number with the arithmetic hidden.
    """

    source: TadSource
    target_latitude_deg: float
    phase_speed_m_s: float
    distance_m: float
    travel_time_s: float
    arrival_utc: _dt.datetime
    altitude_m: float
    path: str
    caveat: str = FORWARD_PREDICTION_IS_NOT_A_DETECTION

    @property
    def travel_time_hours(self) -> float:
        """The travel time in hours, the unit the published 3-4 h is quoted in."""
        return self.travel_time_s / _SECONDS_PER_HOUR


def arrival_time(
    source: TadSource,
    target_latitude_deg: float,
    *,
    phase_speed_m_s: float,
    great_circle: bool = True,
    altitude_m: float = 0.0,
) -> ArrivalPrediction:
    """When a disturbance leaving ``source`` at ``phase_speed_m_s`` reaches a latitude.

    Distance over speed, on the meridian through the source. The target is
    taken to sit at the source's longitude, which is what makes the path a
    meridian and the distance a pure latitude difference; a target at another
    longitude is a different arc and this function does not pretend to cover it.

    THREE REFUSALS, each a different mistake:

      * a target in the OPPOSITE hemisphere. A storm launches a disturbance
        from both ovals, and the southern one is modelled with a southern
        source. The arc computed here would run through the equator, where the
        premise -- propagation away from one oval -- has already failed.
      * a target at the SAME latitude as the source. There is no distance
        between them, so there is no travel time to state.
      * a target POLEWARD of the source. The published band describes
        propagation AWAY from the auroral source, and this repository's own
        measurement (evidence/TRAVELLING_DISTURBANCES.md) rejected a poleward
        progression on exactly that ground. Returning a number for it would
        make this module the thing that analysis refused to be.

    The speed is NOT checked against the local sound speed here, because that
    check needs a temperature and a mean particle mass this function is not
    given. Run :func:`sound_speed_bound` on the speed before quoting an arrival
    computed from it.
    """
    if not math.isfinite(phase_speed_m_s) or phase_speed_m_s <= 0.0:
        raise ValueError(
            f"phase_speed_m_s must be finite and positive, got {phase_speed_m_s}. "
            "A zero or negative propagation speed has no arrival time; a "
            "disturbance travelling the other way is a different source."
        )
    target = _checked_latitude(target_latitude_deg, "target_latitude_deg")
    if source.hemisphere is Hemisphere.NORTHERN and target < 0.0:
        raise ValueError(
            f"target_latitude_deg {target} is southern but the source is in the "
            f"{source.hemisphere.value} hemisphere. Model the southern "
            "disturbance with a southern source: the meridional arc used here "
            "crosses the equator, where 'propagation away from the oval' is no "
            "longer what is being computed."
        )
    if source.hemisphere is Hemisphere.SOUTHERN and target > 0.0:
        raise ValueError(
            f"target_latitude_deg {target} is northern but the source is in the "
            f"{source.hemisphere.value} hemisphere. Model the northern "
            "disturbance with a northern source: the meridional arc used here "
            "crosses the equator, where 'propagation away from the oval' is no "
            "longer what is being computed."
        )
    if abs(target) == abs(source.source_latitude_deg):
        raise ValueError(
            f"target_latitude_deg {target} is the source latitude "
            f"{source.source_latitude_deg}; there is no distance between them "
            "and so no travel time to state."
        )
    if abs(target) > abs(source.source_latitude_deg):
        raise ValueError(
            f"target_latitude_deg {target} is POLEWARD of the source at "
            f"{source.source_latitude_deg}. A disturbance launched in the "
            "auroral oval travels equatorward, and this repository's own "
            "analysis (evidence/TRAVELLING_DISTURBANCES.md) rejected a "
            "poleward progression as not being a travelling atmospheric "
            "disturbance at all. Pass an equatorward target."
        )

    distance = great_circle_distance_m(
        source.source_latitude_deg,
        target,
        altitude_m=altitude_m,
        great_circle=great_circle,
    )
    travel_time = distance / phase_speed_m_s
    return ArrivalPrediction(
        source=source,
        target_latitude_deg=target,
        phase_speed_m_s=float(phase_speed_m_s),
        distance_m=distance,
        travel_time_s=travel_time,
        arrival_utc=ensure_utc(source.onset_utc, "onset_utc")
        + _dt.timedelta(seconds=travel_time),
        altitude_m=float(altitude_m),
        path=PATH_GREAT_CIRCLE if great_circle else PATH_CHORD,
    )


@dataclass(frozen=True, slots=True)
class ArrivalWindow:
    """An INTERVAL of arrival instants, because one speed is not knowable.

    ``earliest`` is the prediction from the FASTEST speed in the range and
    ``latest`` the one from the SLOWEST: a faster disturbance covers the same
    arc sooner. Both whole predictions are kept rather than only their
    instants, so the distance and travel time behind each endpoint stay
    visible.
    """

    source: TadSource
    target_latitude_deg: float
    speed_range_m_s: tuple[float, float]
    distance_m: float
    earliest: ArrivalPrediction
    latest: ArrivalPrediction
    published_reference: str = PUBLISHED_REFERENCE
    caveat: str = FORWARD_PREDICTION_IS_NOT_A_DETECTION
    agreement_caveat: str = AGREEMENT_IS_NOT_OBSERVATIONAL_SUPPORT
    validation_caveat: str = NOT_VALIDATED_AGAINST_THE_MEASURED_SIGNAL

    @property
    def earliest_utc(self) -> _dt.datetime:
        """The arrival instant implied by the fastest speed in the range."""
        return self.earliest.arrival_utc

    @property
    def latest_utc(self) -> _dt.datetime:
        """The arrival instant implied by the slowest speed in the range."""
        return self.latest.arrival_utc

    @property
    def width_s(self) -> float:
        """How wide the window is, in seconds. Zero only for a degenerate range."""
        return (self.latest_utc - self.earliest_utc).total_seconds()

    def contains(self, moment: _dt.datetime) -> bool:
        """Whether an instant falls inside the window, endpoints included."""
        instant = ensure_utc(moment, "moment")
        return self.earliest_utc <= instant <= self.latest_utc


def arrival_window(
    source: TadSource,
    target_latitude_deg: float,
    *,
    speed_range_m_s: tuple[float, float] = PUBLISHED_PHASE_SPEED_M_S,
    great_circle: bool = True,
    altitude_m: float = 0.0,
) -> ArrivalWindow:
    """The interval a disturbance could arrive in, given a RANGE of speeds.

    This is the honest form of a forward arrival prediction and
    :func:`arrival_time` is the lower-level piece it is built from. A single
    phase speed is not knowable in advance: it depends on the heating profile,
    the background wind and the thermospheric state, none of which is in hand
    at onset. Quoting one instant would state a precision that does not exist.

    The default range is the published 500-800 m/s band -- imported from
    :data:`~thermosphere_consequence.molecular.disturbances.PUBLISHED_PHASE_SPEED_M_S`
    rather than restated, so the forward predictor and the inverse estimator
    compare against the same two numbers. Those are MODELLED figures
    (Fuller-Rowell et al. 2008, carried into Fang et al. 2022 and Guarnieri et
    al. 2025), which is why
    :data:`~thermosphere_consequence.molecular.disturbances.AGREEMENT_IS_NOT_OBSERVATIONAL_SUPPORT`
    is on every window this returns.

    A degenerate range where both endpoints are equal is permitted and produces
    a window of zero width, which is the same claim as a single speed and
    should be read as one.
    """
    slowest, fastest = speed_range_m_s
    if not math.isfinite(slowest) or not math.isfinite(fastest):
        raise ValueError(f"speed_range_m_s must be finite, got {speed_range_m_s}")
    if slowest <= 0.0:
        raise ValueError(
            f"speed_range_m_s must be positive, got {speed_range_m_s}; a "
            "non-positive speed has no arrival time"
        )
    if slowest > fastest:
        raise ValueError(
            f"speed_range_m_s must be (slowest, fastest), got {speed_range_m_s}. "
            "Reversing them would swap the endpoints of the window and report "
            "the fast arrival as the late one."
        )
    earliest = arrival_time(
        source,
        target_latitude_deg,
        phase_speed_m_s=fastest,
        great_circle=great_circle,
        altitude_m=altitude_m,
    )
    latest = arrival_time(
        source,
        target_latitude_deg,
        phase_speed_m_s=slowest,
        great_circle=great_circle,
        altitude_m=altitude_m,
    )
    return ArrivalWindow(
        source=source,
        target_latitude_deg=earliest.target_latitude_deg,
        speed_range_m_s=(float(slowest), float(fastest)),
        distance_m=earliest.distance_m,
        earliest=earliest,
        latest=latest,
    )


# --------------------------------------------------------------------------
# how much of the disturbance is left when it gets there
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttenuationLaw:
    """A stated rule for how much amplitude survives a distance, and its cost.

    The callable is not enough on its own. A decay law is the single most
    load-bearing assumption in a forward amplitude prediction and the easiest
    to hide inside a constant, so this type refuses to carry one without
    ``assumes`` and ``hides`` in words -- and those words travel onto every
    :class:`EnhancementPrediction` the law produces.

    ``surviving_fraction`` maps a distance in metres to the fraction of the
    source amplitude still present, and must return a value in [0, 1].
    """

    name: str
    assumes: str
    hides: str
    surviving_fraction: Callable[[float], float]

    def __post_init__(self) -> None:
        for field_name in ("name", "assumes", "hides"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(
                    f"AttenuationLaw.{field_name} must be stated. An attenuation "
                    "law with no description is a decay constant hidden in a "
                    "function, which is the failure this type exists to prevent."
                )
        if not callable(self.surviving_fraction):
            raise TypeError(
                "AttenuationLaw.surviving_fraction must be callable, got "
                f"{type(self.surviving_fraction).__name__}"
            )


#: The default, and it asserts NOTHING. Every unit of amplitude that left the
#: source arrives, so the prediction is an UPPER BOUND on the enhancement and
#: must be quoted as one. It is the default because the alternative -- an
#: e-folding distance picked to look reasonable -- would put an unsourced
#: number at the centre of every amplitude this module reports, and a reader
#: would have no way to see it. A caller who has a decay scale states it with
#: :func:`exponential_attenuation`.
UNATTENUATED = AttenuationLaw(
    name="unattenuated",
    assumes=(
        "Nothing is lost between the source and the target: the fractional "
        "density enhancement that left the auroral oval is the fractional "
        "enhancement that arrives. The result is an UPPER BOUND, not an "
        "estimate."
    ),
    hides=(
        "Everything that removes amplitude from a real large-scale gravity "
        "wave: geometric spreading as the wavefront lengthens away from the "
        "oval, molecular and ion-drag dissipation, dispersion of the packet, "
        "and refraction by the background wind. Each of them reduces the "
        "arriving enhancement, so a measurement falling BELOW this bound says "
        "nothing at all, while one above it is a real disagreement."
    ),
    surviving_fraction=lambda distance_m: 1.0,
)


def exponential_attenuation(e_folding_distance_m: float) -> AttenuationLaw:
    """An ``exp(-d / L)`` decay with the scale ``L`` named by the caller.

    The e-folding distance is a REQUIRED argument with no default on purpose.
    This module has no measured decay scale for a travelling atmospheric
    disturbance -- none is held anywhere in this repository -- so any default
    would be a number invented here and then quoted from downstream as though
    it had come from somewhere. Passing it explicitly puts the assumption in
    the caller's code, where a reviewer reads it.

    The returned law records ``L`` in its own name, so an amplitude prediction
    carries the scale that produced it.
    """
    if not math.isfinite(e_folding_distance_m) or e_folding_distance_m <= 0.0:
        raise ValueError(
            f"e_folding_distance_m must be finite and positive, got "
            f"{e_folding_distance_m}. A zero or negative scale is not a decay."
        )
    scale = float(e_folding_distance_m)
    return AttenuationLaw(
        name=f"exponential, e-folding distance {scale:.3g} m",
        assumes=(
            f"Amplitude falls as exp(-d / {scale:.3g} m), so a single "
            "distance-independent scale describes every loss mechanism at once "
            "and the disturbance keeps its shape while shrinking."
        ),
        hides=(
            "That the scale is an INPUT, not a measurement: this repository "
            "holds no observed decay scale for a travelling atmospheric "
            "disturbance. It also hides the distance dependence of the real "
            "losses -- geometric spreading falls as a power of distance, not "
            "exponentially -- so agreement at one range does not carry to "
            "another."
        ),
        surviving_fraction=lambda distance_m: math.exp(-distance_m / scale),
    )


@dataclass(frozen=True, slots=True)
class EnhancementPrediction:
    """The fractional density enhancement expected on arrival, and its assumptions.

    ``predicted_fractional_enhancement`` is SIGNED and follows the sign of the
    amplitude at the source, so a source depletion stays a depletion -- the
    same convention
    :class:`~thermosphere_consequence.molecular.disturbances.DensityPerturbation`
    uses, and for the same reason: a trough is as real as a crest.
    """

    source_fractional_enhancement: float
    distance_m: float
    surviving_fraction: float
    predicted_fractional_enhancement: float
    attenuation_law: str
    attenuation_assumes: str
    attenuation_hides: str
    caveat: str = FORWARD_PREDICTION_IS_NOT_A_DETECTION


def predicted_enhancement_at_arrival(
    arrival: ArrivalPrediction,
    *,
    source_fractional_enhancement: float,
    attenuation: AttenuationLaw = UNATTENUATED,
) -> EnhancementPrediction:
    """How much fractional density enhancement is left when the disturbance lands.

    Takes the whole :class:`ArrivalPrediction` rather than a bare distance, so
    the amplitude and the path it decayed over are guaranteed to belong to the
    same prediction.

    ``attenuation`` defaults to :data:`UNATTENUATED`, which returns the source
    amplitude unchanged and is therefore an upper bound rather than an
    estimate. That is deliberate: see :data:`UNATTENUATED` for what it assumes
    and what it hides, both of which are copied onto the result so they cannot
    be separated from the number.

    A law that returns a surviving fraction outside [0, 1] is refused by name.
    Above 1 it would be amplification, which no passive propagation does; below
    0 it would flip the sign of the perturbation, turning a predicted
    enhancement into a predicted depletion through a bug in a decay function.
    """
    if not isinstance(attenuation, AttenuationLaw):
        raise TypeError(
            "attenuation must be an AttenuationLaw, so the assumptions travel "
            f"with the number; got {type(attenuation).__name__}"
        )
    if not math.isfinite(source_fractional_enhancement):
        raise ValueError(
            f"source_fractional_enhancement must be finite, got "
            f"{source_fractional_enhancement}"
        )
    if source_fractional_enhancement <= -1.0:
        raise ValueError(
            f"source_fractional_enhancement {source_fractional_enhancement} is at "
            "or below -1, which is a density at or below zero. A fractional "
            "perturbation is density/background - 1 and cannot reach -1."
        )
    surviving = float(attenuation.surviving_fraction(arrival.distance_m))
    if not math.isfinite(surviving) or surviving < 0.0 or surviving > 1.0:
        raise ValueError(
            f"attenuation law {attenuation.name!r} returned a surviving fraction "
            f"of {surviving} over {arrival.distance_m:.3g} m, which is outside "
            "[0, 1]. Above 1 is amplification and below 0 flips the sign of the "
            "perturbation; neither is propagation."
        )
    return EnhancementPrediction(
        source_fractional_enhancement=float(source_fractional_enhancement),
        distance_m=arrival.distance_m,
        surviving_fraction=surviving,
        predicted_fractional_enhancement=source_fractional_enhancement * surviving,
        attenuation_law=attenuation.name,
        attenuation_assumes=attenuation.assumes,
        attenuation_hides=attenuation.hides,
    )


# --------------------------------------------------------------------------
# did anything actually arrive when it was predicted to?
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForwardComparison:
    """Where an observed perturbation onset fell against a predicted window.

    ``margin_s`` is SIGNED and is zero exactly when the verdict is ``INSIDE``:
    negative means the onset preceded the window by that many seconds, positive
    means it followed it. ``inside_margin_s`` is filled only for ``INSIDE`` and
    says how far the onset sat from the NEARER edge, because an onset one
    second inside a three-hour window and one in the middle of it are not the
    same claim, and a verdict alone cannot tell them apart.
    """

    verdict: ArrivalVerdict
    predicted_earliest_utc: _dt.datetime
    predicted_latest_utc: _dt.datetime
    detection_threshold: float
    observed_onset_utc: _dt.datetime | None
    observed_fractional_perturbation: float | None
    margin_s: float | None
    inside_margin_s: float | None
    peak_fractional_perturbation: float
    samples_considered: int
    note: str
    caveat: str = FORWARD_PREDICTION_IS_NOT_A_DETECTION

    @property
    def agrees(self) -> bool:
        """True only for ``INSIDE``. ``NO_SIGNAL`` is not agreement."""
        return self.verdict is ArrivalVerdict.INSIDE


def compare_forward_to_observed(
    predicted_window: ArrivalWindow,
    observed_perturbation_series: Sequence[DensityPerturbation],
    *,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
) -> ForwardComparison:
    """Did the observed perturbation arrive when the window said it would?

    ``observed_perturbation_series`` is a time-ordered run of
    :class:`~thermosphere_consequence.molecular.disturbances.DensityPerturbation`
    records at the target latitude -- the same record type
    :meth:`~thermosphere_consequence.molecular.disturbances.PerturbationField.peak`
    and
    :meth:`~thermosphere_consequence.molecular.disturbances.Hovmoller.peak`
    return, so a series taken out of a Hovmoller can be handed straight here
    without a conversion that could reinterpret the sign.

    THE ONSET IS THE FIRST CROSSING, NOT THE PEAK. A disturbance ARRIVES when
    its leading edge gets there; a peak-based rule would report an arrival
    systematically late by a quarter of the wave period, and the published 3-4
    h propagation time this would be compared against is an onset-to-onset
    figure. The peak is reported alongside so a reader can see how far apart
    the two rules would have put the answer.

    The crossing is tested on the MAGNITUDE of the fractional perturbation, so
    a disturbance whose leading signature at this altitude is a depletion is
    detected rather than waited out -- the same reason
    :meth:`~thermosphere_consequence.molecular.disturbances.Hovmoller.peak`
    ranks by magnitude.

    ``NO_SIGNAL`` and ``AFTER`` are kept apart deliberately: one says the
    disturbance was late, the other says nothing crossed the threshold at all,
    and collapsing them would let an absence be reported as a near miss.

    An EMPTY series raises rather than returning ``NO_SIGNAL``: no data is not
    the same finding as no signal, and a comparison that cannot tell them apart
    would report an unrun measurement as a negative result.
    """
    if not math.isfinite(detection_threshold) or detection_threshold <= 0.0:
        raise ValueError(
            f"detection_threshold must be finite and positive, got "
            f"{detection_threshold}; it is a fraction of the background density"
        )
    if not observed_perturbation_series:
        raise ValueError(
            "observed_perturbation_series is empty. An empty series is NO DATA, "
            "which is not the same finding as NO_SIGNAL -- returning a verdict "
            "here would report a measurement that was never taken as a negative "
            "result."
        )
    moments = [
        ensure_utc(sample.time_utc, "observed_perturbation_series[].time_utc")
        for sample in observed_perturbation_series
    ]
    for earlier, later in zip(moments, moments[1:], strict=False):
        if later <= earlier:
            raise ValueError(
                f"observed_perturbation_series must be strictly increasing in "
                f"time; {later.isoformat()} does not follow {earlier.isoformat()}. "
                "An onset read off an unordered series is whichever sample came "
                "first in the list, not whichever came first in time."
            )

    peak = max(
        abs(sample.fractional_perturbation) for sample in observed_perturbation_series
    )
    onset_index: int | None = None
    for index, sample in enumerate(observed_perturbation_series):
        if abs(sample.fractional_perturbation) >= detection_threshold:
            onset_index = index
            break

    earliest = predicted_window.earliest_utc
    latest = predicted_window.latest_utc
    if onset_index is None:
        return ForwardComparison(
            verdict=ArrivalVerdict.NO_SIGNAL,
            predicted_earliest_utc=earliest,
            predicted_latest_utc=latest,
            detection_threshold=float(detection_threshold),
            observed_onset_utc=None,
            observed_fractional_perturbation=None,
            margin_s=None,
            inside_margin_s=None,
            peak_fractional_perturbation=peak,
            samples_considered=len(observed_perturbation_series),
            note=(
                f"no sample in the series reached a fractional perturbation of "
                f"{detection_threshold:g} in magnitude; the largest was {peak:.4g}. "
                "Nothing arrived to be early or late, so no margin is stated."
            ),
        )

    onset = moments[onset_index]
    observed = observed_perturbation_series[onset_index].fractional_perturbation
    if onset < earliest:
        margin = (onset - earliest).total_seconds()
        return ForwardComparison(
            verdict=ArrivalVerdict.BEFORE,
            predicted_earliest_utc=earliest,
            predicted_latest_utc=latest,
            detection_threshold=float(detection_threshold),
            observed_onset_utc=onset,
            observed_fractional_perturbation=observed,
            margin_s=margin,
            inside_margin_s=None,
            peak_fractional_perturbation=peak,
            samples_considered=len(observed_perturbation_series),
            note=(
                f"the perturbation was already present {abs(margin) / _SECONDS_PER_HOUR:.2f} h "
                "before the window opened, so it cannot have been carried there "
                "by a disturbance leaving the source at any speed in the range. "
                "Either the onset proxy fired late or this is a different structure."
            ),
        )
    if onset > latest:
        margin = (onset - latest).total_seconds()
        return ForwardComparison(
            verdict=ArrivalVerdict.AFTER,
            predicted_earliest_utc=earliest,
            predicted_latest_utc=latest,
            detection_threshold=float(detection_threshold),
            observed_onset_utc=onset,
            observed_fractional_perturbation=observed,
            margin_s=margin,
            inside_margin_s=None,
            peak_fractional_perturbation=peak,
            samples_considered=len(observed_perturbation_series),
            note=(
                f"the perturbation crossed the threshold {margin / _SECONDS_PER_HOUR:.2f} h "
                "after the window shut, so it is slower than the slowest speed "
                "in the range, or it is not the predicted disturbance."
            ),
        )
    from_open = (onset - earliest).total_seconds()
    from_shut = (latest - onset).total_seconds()
    return ForwardComparison(
        verdict=ArrivalVerdict.INSIDE,
        predicted_earliest_utc=earliest,
        predicted_latest_utc=latest,
        detection_threshold=float(detection_threshold),
        observed_onset_utc=onset,
        observed_fractional_perturbation=observed,
        margin_s=0.0,
        inside_margin_s=min(from_open, from_shut),
        peak_fractional_perturbation=peak,
        samples_considered=len(observed_perturbation_series),
        note=(
            "the onset fell inside the predicted window, which is CONSISTENT "
            "with a disturbance leaving the source at a speed in the range. It "
            "is not evidence that one did: any structure whose onset lands in a "
            f"{predicted_window.width_s / _SECONDS_PER_HOUR:.2f} h window does the same."
        ),
    )


# --------------------------------------------------------------------------
# the ceiling: what a gravity wave cannot exceed
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SoundSpeedBound:
    """The local sound speed as an interval over the ratio of specific heats.

    An interval and not a value, because gamma at 210 km is not one number: the
    gas is a mixture of monatomic and diatomic species and the mixture's gamma
    lies somewhere between 7/5 and 5/3. Reporting a single sound speed would
    pick one end of that and hide the choice.
    """

    temperature_k: float
    mean_particle_mass_kg: float
    gamma_range: tuple[float, float]
    lowest_m_s: float
    highest_m_s: float

    def is_physically_propagable(self, speed_m_s: float) -> bool:
        """Whether a phase speed is one a freely propagating gravity wave could have.

        Tested against the HIGHEST end of the bound -- the most permissive one
        -- so a False here means the speed exceeds the sound speed under EVERY
        gamma in the range, not merely under the strictest choice. A rejection
        is then a statement about the gas rather than about which gamma was
        picked.

        Sign is ignored: a poleward disturbance is bounded by the same sound
        speed as an equatorward one. Direction is a separate question, and
        :func:`~thermosphere_consequence.molecular.disturbances.estimate_phase_speed`
        reports it separately.

        PASSING IS NOT EVIDENCE. A subsonic phase speed is consistent with a
        gravity wave; so is a subsonic diurnal pattern that is not a wave at
        all. This is a necessary condition and nothing more -- which is exactly
        how evidence/TRAVELLING_DISTURBANCES.md used it, to REJECT a 2192 m/s
        measurement rather than to accept anything.
        """
        if not math.isfinite(speed_m_s):
            raise ValueError(f"speed_m_s must be finite, got {speed_m_s}")
        return abs(speed_m_s) <= self.highest_m_s

    def excess_m_s(self, speed_m_s: float) -> float:
        """How far a speed exceeds the most permissive sound speed; 0.0 if it does not."""
        if self.is_physically_propagable(speed_m_s):
            return 0.0
        return abs(speed_m_s) - self.highest_m_s


def sound_speed_bound(
    temperature_k: float,
    mean_particle_mass_kg: float,
    *,
    gamma_range: tuple[float, float] = THERMOSPHERIC_GAMMA_RANGE,
) -> SoundSpeedBound:
    """``sqrt(gamma k T / m)`` evaluated at both ends of the gamma range.

    The ceiling a freely propagating gravity wave does not much exceed, and the
    check this repository already used to reject its own travelling-disturbance
    measurement: 1070-2192 m/s against a 675-773 m/s sound speed at 210 km
    (evidence/TRAVELLING_DISTURBANCES.md).

    ``mean_particle_mass_kg`` is the composition-weighted mean particle mass --
    what
    :func:`~thermosphere_consequence.molecular.composition.mean_particle_mass`
    returns -- not the mass of any one species. At 210 km it runs about 20-24
    atomic mass units, and using the mass of atomic oxygen alone would put the
    bound about 10 per cent high.

    The 675-773 m/s figure in the evidence file came from CORRELATED (T, m)
    pairs sampled at four latitudes and two times, so it is narrower than the
    interval this function returns for the independent extremes of the same
    ranges. That is the expected relationship and not a disagreement: a bound
    over unpaired extremes must contain a bound over paired samples.
    """
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError(
            f"temperature_k must be finite and positive, got {temperature_k}"
        )
    if not math.isfinite(mean_particle_mass_kg) or mean_particle_mass_kg <= 0.0:
        raise ValueError(
            f"mean_particle_mass_kg must be finite and positive, got "
            f"{mean_particle_mass_kg}. This is a mean particle mass in "
            f"kilograms; one atomic mass unit is {AMU.value:g} kg."
        )
    low_gamma, high_gamma = gamma_range
    if not math.isfinite(low_gamma) or not math.isfinite(high_gamma):
        raise ValueError(f"gamma_range must be finite, got {gamma_range}")
    if low_gamma <= 1.0:
        raise ValueError(
            f"gamma_range must exceed 1, got {gamma_range}; the ratio of "
            "specific heats of an ideal gas is 1 + 2/f and f is finite"
        )
    if low_gamma > high_gamma:
        raise ValueError(
            f"gamma_range must be (lowest, highest), got {gamma_range}"
        )
    thermal = K_B.value * temperature_k / mean_particle_mass_kg
    return SoundSpeedBound(
        temperature_k=float(temperature_k),
        mean_particle_mass_kg=float(mean_particle_mass_kg),
        gamma_range=(float(low_gamma), float(high_gamma)),
        lowest_m_s=math.sqrt(low_gamma * thermal),
        highest_m_s=math.sqrt(high_gamma * thermal),
    )
