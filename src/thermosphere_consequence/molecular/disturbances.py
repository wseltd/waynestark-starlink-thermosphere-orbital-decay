"""Travelling atmospheric disturbances, measured out of a gridded density field.

Storm-time Joule heating in the auroral oval launches large-scale gravity waves
that carry the heating signature away from the source. In the neutral density
they appear as a travelling atmospheric disturbance (TAD): a band of enhanced
and depleted density that sweeps equatorward from the oval. The canonical
numbers this module exists to be COMPARED AGAINST -- never fitted to -- are
phase speeds of 500-800 m/s and high-latitude to mid-latitude propagation in
roughly 3-4 hours (``PUBLISHED_PHASE_SPEED_M_S``,
``PUBLISHED_HIGH_TO_MID_DELAY_HOURS``).

WHAT THIS MODULE IS. Pure array kernels plus three functions that pull samples
from a ``TotalDensityProvider``. Nothing here knows which provider answered, so
the same diagnostic runs over the retrieved WAM-IPE grid, over a point model,
or over a synthetic field written by a test. It computes:

  * a BACKGROUND, by one of two named methods the caller must choose between
    explicitly (``background_field``);
  * the SIGNED fractional departure from it (``perturbation_field``), never
    clamped, because a depletion is as real a disturbance as an enhancement;
  * the latitude-time Hovmoller diagram (``latitude_time_hovmoller``), which is
    the diagnostic that makes a propagating front visible at all;
  * a phase speed with an uncertainty and a correlation quality
    (``estimate_phase_speed``), or a refusal when the correlation does not
    support a number;
  * the delay between two latitudes (``propagation_delay``);
  * a comparison against the published band (``compare_to_literature``).

WHAT IT DOES NOT DO, stated because the shape of the output invites the wrong
reading:

  * It does NOT identify a TAD. It measures the meridional phase progression of
    whatever coherent structure dominates the field it is given. At a fixed
    longitude, the diurnal density cycle also has a latitude-dependent phase,
    and it will be measured just as happily. Separating the two needs
    CONTROLS: the same measurement over a geomagnetically quieter window, the
    hemispheric direction -- a storm-launched disturbance travels equatorward
    in BOTH hemispheres, so a poleward progression in either is not one -- and
    the speed of sound, which a freely propagating gravity wave does not much
    exceed. ``scripts/run_tad_analysis.py`` runs all three and reports them.

  * It does NOT establish that a model's disturbance is the real atmosphere's.
    Agreement with a band that was itself derived from model runs is agreement
    between models. ``AGREEMENT_IS_NOT_OBSERVATIONAL_SUPPORT`` says so on every
    comparison this module returns.

  * It does NOT interpolate across a hole in an archive. Sampling is delegated
    to the provider, and a provider that refuses a gap (as
    ``providers.storm_grid`` does) keeps refusing here.

RESOLUTION, which decides what can be measured at all. A lag between two
latitude bands can only be recovered from a sampled series to a fraction of the
sampling interval, and only up to one period of the signal's own oscillation.
On an hourly archive, adjacent 2-degree bands are separated by 222 km, which a
600 m/s disturbance crosses in 0.10 h -- a tenth of one sample. The estimator
below recovers such lags by interpolating the correlation peak, and it refuses
when the peak lands at the edge of the range in which the lag is unambiguous.
Widening the bands makes each lag larger and easier to see, at the cost of
fewer of them; the analysis script reports the whole spacing sweep rather than
one spacing chosen after the fact.

Canonical tickets: P06_08_T022 through T029.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from ..domain.constants import R_EARTH_MEAN
from ..domain.temporal import ensure_utc
from ..protocols.atmosphere import DriverState, TotalDensityProvider

__all__ = [
    "AGREEMENT_IS_NOT_OBSERVATIONAL_SUPPORT",
    "Axis",
    "DEFAULT_BACKGROUND_WINDOW_HOURS",
    "DEFAULT_LAG_RESOLUTION_SAMPLES",
    "PHASE_SPEED_METHOD",
    "PUBLISHED_HIGH_TO_MID_DELAY_HOURS",
    "PUBLISHED_PHASE_SPEED_M_S",
    "PUBLISHED_REFERENCE",
    "BackgroundMethod",
    "DensityPerturbation",
    "Hovmoller",
    "PerturbationField",
    "PhaseSpeedEstimate",
    "PropagationDelay",
    "background_field",
    "compare_to_literature",
    "estimate_phase_speed",
    "fractional_departure",
    "latitude_time_hovmoller",
    "meridional_distance_m",
    "perturbation_field",
    "propagation_delay",
]

#: Large-scale gravity-wave phase speeds, for COMPARISON. Fuller-Rowell et al.
#: 2008, as carried into the two papers this project already cites for the
#: February 2022 event. Nothing in this module or its callers is tuned to land
#: inside this band.
PUBLISHED_PHASE_SPEED_M_S: tuple[float, float] = (500.0, 800.0)

#: High-latitude to mid-latitude propagation time from the same source.
PUBLISHED_HIGH_TO_MID_DELAY_HOURS: tuple[float, float] = (3.0, 4.0)

PUBLISHED_REFERENCE = (
    "Fuller-Rowell et al. 2008, as cited in Fang et al. 2022 "
    "(doi:10.1029/2022SW003193) and Guarnieri et al. 2025 "
    "(doi:10.5194/npg-32-75-2025)"
)

AGREEMENT_IS_NOT_OBSERVATIONAL_SUPPORT = (
    "The published 500-800 m/s band and the 3-4 h propagation time are "
    "MODELLED quantities. A measurement taken from another model that lands "
    "inside them is agreement between models, not observational support: it "
    "says the two simulations behave alike, and says nothing about whether "
    "either resembles the real thermosphere. No in-situ measurement of a "
    "travelling atmospheric disturbance is held anywhere in this repository, "
    "so neither agreement nor disagreement here can be called validation."
)

#: Centred running-mean window used by the analysis script. At a one-hour
#: cadence this is the sample plus the three either side. See
#: ``background_field`` for what that choice keeps and what it hides.
DEFAULT_BACKGROUND_WINDOW_HOURS = 7.0

_SECONDS_PER_HOUR = 3600.0

#: A coordinate axis, however the caller happens to hold it. Plain sequences
#: and numpy arrays are both accepted because a caller with a grid in hand has
#: an array and a caller writing a case by hand has a list, and neither should
#: have to convert to talk to this module.
Axis = Sequence[float] | NDArray[np.float64]

#: Axes are compared to this tolerance in their own units before being treated
#: as uniformly spaced. Float arithmetic on a decimal step lands a few ulp away
#: from it, so the comparison cannot be exact.
_UNIFORM_TOLERANCE = 1e-9

_RUNNING_MEAN_RULE = (
    "Background = the centred running mean of the density over a stated "
    "window, evaluated only where the FULL window fits. Positions without a "
    "full window are NaN rather than a mean of whatever samples happen to be "
    "there, because a shortened window biases the perturbation exactly at the "
    "ends of the record, which is where a reader is most likely to look for "
    "an onset."
)

_QUIET_REFERENCE_RULE = (
    "Background = the density at the SAME UTC time of day on a stated quiet "
    "reference day, at the same point. Holding the UTC time of day fixed holds "
    "local solar time fixed at a fixed longitude, so the diurnal cycle divides "
    "out and what is left is the difference between the two days."
)


class BackgroundMethod(StrEnum):
    """How the slowly-varying background is separated from the disturbance.

    There is no default anywhere in this module. Both methods are defensible
    and they hide different things, so the caller states which one produced
    every number, and every returned object carries the choice.
    """

    #: Centred running mean in time. See ``background_field``.
    RUNNING_MEAN = "running_mean"

    #: Same UTC time of day on a quiet reference day. See ``background_field``.
    QUIET_REFERENCE_DAY = "quiet_reference_day"


def meridional_distance_m(from_latitude_deg: float, to_latitude_deg: float) -> float:
    """Signed arc length along a meridian, in metres, on a spherical Earth.

    Positive when ``to_latitude_deg`` is the more equatorward of the two in the
    NORTHERN hemisphere sense of decreasing latitude; callers that care about
    equatorward versus poleward read the sign from the latitude ORDER they
    passed, not from the hemisphere. The mean Earth radius is used because this
    is a propagation-distance diagnostic, not a geodetic calculation; the
    difference between mean and equatorial radius is 0.1 per cent and is far
    below the uncertainty of any speed measured here.
    """
    return R_EARTH_MEAN.value * math.radians(from_latitude_deg - to_latitude_deg)


@dataclass(frozen=True, slots=True)
class DensityPerturbation:
    """One point's departure from its background density.

    ``fractional_perturbation`` is SIGNED and is never clamped: a depletion is
    as real a disturbance as an enhancement, and a diagnostic that reported
    only enhancements would turn the trough of every wave into a zero.
    """

    time_utc: _dt.datetime
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    density: float
    background_density: float
    fractional_perturbation: float

    def __post_init__(self) -> None:
        if self.time_utc.tzinfo is None:
            raise ValueError("DensityPerturbation.time_utc must be timezone-aware")
        if not math.isfinite(self.density) or self.density < 0.0:
            raise ValueError(f"density must be finite and non-negative, got {self.density}")
        if not math.isfinite(self.background_density) or self.background_density <= 0.0:
            raise ValueError(
                f"background_density must be finite and positive, got "
                f"{self.background_density}; a fractional perturbation against a "
                "zero or negative background is not a ratio"
            )
        implied = self.density / self.background_density - 1.0
        if not math.isclose(self.fractional_perturbation, implied, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"fractional_perturbation {self.fractional_perturbation} does not "
                f"match density/background - 1 = {implied}. This record would "
                "state a perturbation the densities beside it do not support."
            )

    @classmethod
    def from_densities(
        cls,
        *,
        time_utc: _dt.datetime,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
        density: float,
        background_density: float,
    ) -> DensityPerturbation:
        """Build one record, computing the signed fraction from the densities."""
        moment = ensure_utc(time_utc, "time_utc")
        if not math.isfinite(background_density) or background_density <= 0.0:
            raise ValueError(
                f"background_density must be finite and positive, got {background_density}"
            )
        return cls(
            time_utc=moment,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            altitude_m=altitude_m,
            density=density,
            background_density=background_density,
            fractional_perturbation=density / background_density - 1.0,
        )

    @property
    def is_depletion(self) -> bool:
        """True when the density is BELOW its background."""
        return self.fractional_perturbation < 0.0


@dataclass(frozen=True, slots=True)
class PerturbationField:
    """The signed fractional perturbation over a latitude/longitude slice."""

    time_utc: _dt.datetime
    altitude_m: float
    latitudes_deg: tuple[float, ...]
    longitudes_deg: tuple[float, ...]
    density: NDArray[np.float64]
    background: NDArray[np.float64]
    perturbation: NDArray[np.float64]
    method: BackgroundMethod
    background_rule: str
    window_hours: float | None = None
    quiet_reference_day: _dt.date | None = None

    def peak(self) -> DensityPerturbation:
        """The largest departure by MAGNITUDE, sign preserved.

        By magnitude and not by value, so a storm whose strongest signature at
        this altitude is a depletion is not reported as its largest enhancement.
        """
        flat = int(np.nanargmax(np.abs(self.perturbation)))
        row, column = np.unravel_index(flat, self.perturbation.shape)
        return DensityPerturbation.from_densities(
            time_utc=self.time_utc,
            latitude_deg=self.latitudes_deg[int(row)],
            longitude_deg=self.longitudes_deg[int(column)],
            altitude_m=self.altitude_m,
            density=float(self.density[row, column]),
            background_density=float(self.background[row, column]),
        )


@dataclass(frozen=True, slots=True)
class Hovmoller:
    """Perturbation on a (time, latitude) grid at one longitude and altitude.

    This is the standard diagnostic for a propagating front: a disturbance that
    sweeps equatorward appears as a tilted stripe, and the tilt IS the phase
    speed. ``perturbation`` is what ``estimate_phase_speed`` consumes.
    """

    times_utc: tuple[_dt.datetime, ...]
    latitudes_deg: tuple[float, ...]
    longitude_deg: float
    altitude_m: float
    density: NDArray[np.float64]
    background: NDArray[np.float64]
    perturbation: NDArray[np.float64]
    method: BackgroundMethod
    background_rule: str
    window_hours: float | None = None
    quiet_reference_day: _dt.date | None = None

    @property
    def step_hours(self) -> float:
        """The uniform sampling interval of the time axis, in hours."""
        return _uniform_step_hours(self.times_utc)

    def peak(self) -> DensityPerturbation:
        """The largest departure by MAGNITUDE over the whole diagram."""
        flat = int(np.nanargmax(np.abs(self.perturbation)))
        row, column = np.unravel_index(flat, self.perturbation.shape)
        return DensityPerturbation.from_densities(
            time_utc=self.times_utc[int(row)],
            latitude_deg=self.latitudes_deg[int(column)],
            longitude_deg=self.longitude_deg,
            altitude_m=self.altitude_m,
            density=float(self.density[row, column]),
            background_density=float(self.background[row, column]),
        )


# --------------------------------------------------------------------------
# background and perturbation
# --------------------------------------------------------------------------


def _uniform_step_hours(times: Sequence[_dt.datetime]) -> float:
    """The sampling interval of a strictly increasing, uniformly spaced axis."""
    if len(times) < 2:
        raise ValueError(
            f"a time axis needs at least two instants to have a step, got {len(times)}"
        )
    steps = [
        (b - a).total_seconds() / _SECONDS_PER_HOUR
        for a, b in zip(times, times[1:], strict=False)
    ]
    if min(steps) <= 0.0:
        raise ValueError("times must be strictly increasing")
    if max(steps) - min(steps) > _UNIFORM_TOLERANCE:
        raise ValueError(
            f"times must be uniformly spaced; the step ranges from {min(steps):g} to "
            f"{max(steps):g} h. A lag measured on an uneven axis would be in units "
            "that change along the axis."
        )
    return steps[0]


def _half_window_steps(window_hours: float, step_hours: float) -> int:
    """Samples either side of the centre, for a centred window of odd length."""
    if window_hours <= 0.0:
        raise ValueError(f"window_hours must be positive, got {window_hours}")
    if step_hours <= 0.0:
        raise ValueError(f"step_hours must be positive, got {step_hours}")
    span = window_hours / step_hours
    half = (span - 1.0) / 2.0
    rounded = int(round(half))
    if rounded < 1 or abs(half - rounded) > 1e-6:
        raise ValueError(
            f"a window of {window_hours:g} h at a {step_hours:g} h step is not an odd "
            "number of samples wide, so it has no centre. Use a window of "
            f"(2n+1) x {step_hours:g} h, for example "
            f"{(2 * max(rounded, 1) + 1) * step_hours:g} h."
        )
    return rounded


def background_field(
    values: NDArray[np.float64],
    *,
    method: BackgroundMethod,
    step_hours: float | None = None,
    window_hours: float | None = None,
    reference: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """The slowly-varying background of ``values``, by an explicitly chosen method.

    ``values`` is sampled along axis 0 in time; the remaining axes are spatial
    and are treated independently. ``method`` has NO default: both methods
    below are defensible, they hide different things, and picking one silently
    would bury that choice inside a number.

    ``BackgroundMethod.RUNNING_MEAN`` -- the centred running mean over
    ``window_hours``, requiring ``step_hours``.

      WHY: it needs nothing but the record itself, so it applies unchanged to a
      storm window and to a quiet control window, which is what makes the two
      comparable. It is a high-pass filter, and the window sets the cut: a
      7-hour window at hourly cadence passes the 2-4 h periods that hourly
      sampling can resolve at all, which is the band a travelling atmospheric
      disturbance occupies.

      WHAT IT HIDES: everything slower than the window. The storm's own
      multi-hour density build-up is REMOVED by construction, so a perturbation
      computed this way is not the storm enhancement and must never be quoted
      as one. It also leaks: a running mean does not annihilate the diurnal
      cycle, it attenuates it, and the residual is coherent and has its own
      latitude-dependent phase. A 7-hour window at hourly cadence passes about
      13 per cent of a 24-hour oscillation; a 3-hour window passes about 2 per
      cent but is close to the sampling limit. That leakage is precisely why a
      quiet-window control is not optional.

    ``BackgroundMethod.QUIET_REFERENCE_DAY`` -- the field at the same UTC time
    of day on a quiet day, supplied as ``reference``.

      WHY: it keeps the storm's slow build-up in the perturbation instead of
      filtering it away, so the result is the storm's whole signature rather
      than only its fast part.

      WHAT IT HIDES: the reference day carries its own weather. Anything the
      two days share -- including any disturbance active on the reference day
      -- divides out, so this method reports a LOWER bound on the departure.
      It also cannot be applied to a quiet window as a control without picking
      a second reference day, which is why the control in this project uses the
      running mean.

    Positions where a running-mean window does not fit come back as NaN. Supply
    margin (extra leading and trailing samples) rather than accepting a
    shortened window at the ends.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError(f"values must have at least one sample on axis 0, got {array.shape}")

    if method is BackgroundMethod.RUNNING_MEAN:
        if reference is not None:
            raise ValueError(
                "reference is only used by BackgroundMethod.QUIET_REFERENCE_DAY; "
                "passing it with RUNNING_MEAN would silently ignore it"
            )
        if window_hours is None or step_hours is None:
            raise ValueError(
                "BackgroundMethod.RUNNING_MEAN needs both window_hours and step_hours"
            )
        half = _half_window_steps(window_hours, step_hours)
        width = 2 * half + 1
        if array.shape[0] < width:
            raise ValueError(
                f"a {window_hours:g} h window is {width} samples wide but only "
                f"{array.shape[0]} were supplied; there is nowhere the full window fits"
            )
        out = np.full(array.shape, np.nan, dtype=float)
        cumulative = np.cumsum(array, axis=0)
        totals = np.concatenate(
            [cumulative[width - 1 : width], cumulative[width:] - cumulative[:-width]], axis=0
        )
        out[half : array.shape[0] - half] = totals / width
        return out

    if method is BackgroundMethod.QUIET_REFERENCE_DAY:
        if window_hours is not None:
            raise ValueError(
                "window_hours is only used by BackgroundMethod.RUNNING_MEAN; the "
                "quiet-reference-day background has no window"
            )
        if reference is None:
            raise ValueError(
                "BackgroundMethod.QUIET_REFERENCE_DAY needs the reference-day field "
                "in `reference`, sampled at the same points and the same UTC times "
                "of day as `values`"
            )
        reference_array = np.asarray(reference, dtype=float)
        if reference_array.shape != array.shape:
            raise ValueError(
                f"reference has shape {reference_array.shape} but values has "
                f"{array.shape}; the reference day must be sampled at the same points"
            )
        return reference_array

    raise ValueError(f"unknown background method {method!r}")


def fractional_departure(
    values: NDArray[np.float64], background: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Signed ``values / background - 1``, never clamped.

    Raises rather than returning infinity where the background is not positive:
    a fractional departure from a zero background is not a ratio, and the one
    thing this diagnostic must not do is turn a missing background into a
    spectacular perturbation.
    """
    array = np.asarray(values, dtype=float)
    base = np.asarray(background, dtype=float)
    if array.shape != base.shape:
        raise ValueError(
            f"values has shape {array.shape} but background has {base.shape}"
        )
    finite = ~np.isnan(base)
    if bool(np.any(base[finite] <= 0.0)):
        raise ValueError(
            f"{int(np.count_nonzero(base[finite] <= 0.0))} background values are "
            "non-positive; a fractional departure from them is not defined"
        )
    return array / base - 1.0


def _sample_row(
    provider: TotalDensityProvider,
    when: _dt.datetime,
    altitude_m: float,
    latitudes: Axis,
    longitude_deg: float,
    drivers: DriverState,
) -> NDArray[np.float64]:
    """Density along one meridian at one instant."""
    return np.asarray(
        [
            provider.total_mass_density(when, altitude_m, float(lat), longitude_deg, drivers)
            for lat in latitudes
        ],
        dtype=float,
    )


def _sample_slice(
    provider: TotalDensityProvider,
    when: _dt.datetime,
    altitude_m: float,
    latitudes: Axis,
    longitudes: Axis,
    drivers: DriverState,
) -> NDArray[np.float64]:
    """Density on a (latitude, longitude) slice at one instant."""
    return np.asarray(
        [
            [
                provider.total_mass_density(
                    when, altitude_m, float(lat), float(lon), drivers
                )
                for lon in longitudes
            ]
            for lat in latitudes
        ],
        dtype=float,
    )


def _reference_instant(when: _dt.datetime, day: _dt.date) -> _dt.datetime:
    """The same UTC time of day, on ``day``."""
    return _dt.datetime.combine(day, when.timetz())


def _window_instants(
    centre: _dt.datetime, half: int, step_hours: float
) -> list[_dt.datetime]:
    """The full centred window around ``centre``, earliest first."""
    step = _dt.timedelta(hours=step_hours)
    return [centre + (offset - half) * step for offset in range(2 * half + 1)]


def perturbation_field(
    provider: TotalDensityProvider,
    when: _dt.datetime,
    altitude_m: float,
    latitudes: Axis,
    longitudes: Axis,
    *,
    drivers: DriverState,
    method: BackgroundMethod,
    window_hours: float | None = None,
    step_hours: float = 1.0,
    quiet_reference_day: _dt.date | None = None,
) -> PerturbationField:
    """The signed fractional departure from the background across one slice.

    For ``RUNNING_MEAN`` the provider is sampled over the whole centred window
    around ``when``, so the background at every point is a full window and the
    returned field carries no edge bias. Those extra instants are real requests
    to the provider: a provider that refuses a gap in its archive refuses here,
    which is the intended behaviour -- a perturbation computed across a hole
    would be a picture of the hole.
    """
    moment = ensure_utc(when, "when")
    if len(latitudes) == 0 or len(longitudes) == 0:
        raise ValueError("latitudes and longitudes must both be non-empty")

    if method is BackgroundMethod.RUNNING_MEAN:
        if window_hours is None:
            raise ValueError("BackgroundMethod.RUNNING_MEAN needs window_hours")
        half = _half_window_steps(window_hours, step_hours)
        block = np.stack(
            [
                _sample_slice(provider, instant, altitude_m, latitudes, longitudes, drivers)
                for instant in _window_instants(moment, half, step_hours)
            ]
        )
        background_block = background_field(
            block,
            method=method,
            step_hours=step_hours,
            window_hours=window_hours,
        )
        density = block[half]
        background = background_block[half]
        rule = _RUNNING_MEAN_RULE
        reference_day = None
    elif method is BackgroundMethod.QUIET_REFERENCE_DAY:
        if quiet_reference_day is None:
            raise ValueError(
                "BackgroundMethod.QUIET_REFERENCE_DAY needs quiet_reference_day"
            )
        if window_hours is not None:
            raise ValueError(
                "window_hours is only used by BackgroundMethod.RUNNING_MEAN"
            )
        density = _sample_slice(provider, moment, altitude_m, latitudes, longitudes, drivers)
        background = _sample_slice(
            provider,
            _reference_instant(moment, quiet_reference_day),
            altitude_m,
            latitudes,
            longitudes,
            drivers,
        )
        rule = _QUIET_REFERENCE_RULE
        reference_day = quiet_reference_day
    else:
        raise ValueError(f"unknown background method {method!r}")

    return PerturbationField(
        time_utc=moment,
        altitude_m=altitude_m,
        latitudes_deg=tuple(float(lat) for lat in latitudes),
        longitudes_deg=tuple(float(lon) for lon in longitudes),
        density=density,
        background=background,
        perturbation=fractional_departure(density, background),
        method=method,
        background_rule=rule,
        window_hours=window_hours,
        quiet_reference_day=reference_day,
    )


def latitude_time_hovmoller(
    provider: TotalDensityProvider,
    altitude_m: float,
    longitude_deg: float,
    times: Sequence[_dt.datetime],
    latitudes: Axis,
    *,
    drivers: DriverState,
    method: BackgroundMethod,
    window_hours: float | None = None,
    quiet_reference_day: _dt.date | None = None,
) -> Hovmoller:
    """Perturbation on a (time, latitude) grid: the diagnostic that shows a front.

    ``times`` must be uniformly spaced and is the axis the result is returned
    on. For ``RUNNING_MEAN`` the provider is additionally sampled over the
    margin the window needs at each end, so every returned row has a full
    window behind it and none of them is an edge artefact.

    The provider is sampled time-outermost, which matters in practice: a
    gridded provider that caches whole hours then loads each hour once instead
    of once per latitude.
    """
    moments = [ensure_utc(t, "times") for t in times]
    if len(moments) < 2:
        raise ValueError(f"a Hovmoller needs at least two times, got {len(moments)}")
    if len(latitudes) < 2:
        raise ValueError(f"a Hovmoller needs at least two latitudes, got {len(latitudes)}")
    step_hours = _uniform_step_hours(moments)

    if method is BackgroundMethod.RUNNING_MEAN:
        if window_hours is None:
            raise ValueError("BackgroundMethod.RUNNING_MEAN needs window_hours")
        half = _half_window_steps(window_hours, step_hours)
        step = _dt.timedelta(hours=step_hours)
        extended = (
            [moments[0] - (half - offset) * step for offset in range(half)]
            + moments
            + [moments[-1] + (offset + 1) * step for offset in range(half)]
        )
        block = np.stack(
            [
                _sample_row(provider, instant, altitude_m, latitudes, longitude_deg, drivers)
                for instant in extended
            ]
        )
        background_block = background_field(
            block, method=method, step_hours=step_hours, window_hours=window_hours
        )
        density = block[half : half + len(moments)]
        background = background_block[half : half + len(moments)]
        rule = _RUNNING_MEAN_RULE
        reference_day = None
    elif method is BackgroundMethod.QUIET_REFERENCE_DAY:
        if quiet_reference_day is None:
            raise ValueError(
                "BackgroundMethod.QUIET_REFERENCE_DAY needs quiet_reference_day"
            )
        if window_hours is not None:
            raise ValueError("window_hours is only used by BackgroundMethod.RUNNING_MEAN")
        density = np.stack(
            [
                _sample_row(provider, instant, altitude_m, latitudes, longitude_deg, drivers)
                for instant in moments
            ]
        )
        background = np.stack(
            [
                _sample_row(
                    provider,
                    _reference_instant(instant, quiet_reference_day),
                    altitude_m,
                    latitudes,
                    longitude_deg,
                    drivers,
                )
                for instant in moments
            ]
        )
        rule = _QUIET_REFERENCE_RULE
        reference_day = quiet_reference_day
    else:
        raise ValueError(f"unknown background method {method!r}")

    return Hovmoller(
        times_utc=tuple(moments),
        latitudes_deg=tuple(float(lat) for lat in latitudes),
        longitude_deg=float(longitude_deg),
        altitude_m=float(altitude_m),
        density=density,
        background=background,
        perturbation=fractional_departure(density, background),
        method=method,
        background_rule=rule,
        window_hours=window_hours,
        quiet_reference_day=reference_day,
    )


# --------------------------------------------------------------------------
# lag estimation
# --------------------------------------------------------------------------

PHASE_SPEED_METHOD = (
    "Each adjacent pair of latitude bands is cross-correlated. The lag search "
    "is restricted to the range in which the lag is UNAMBIGUOUS -- out to the "
    "first zero crossing of the bands' own autocorrelation, plus one sample so "
    "the peak can be interpolated -- because the correlation of an oscillatory "
    "signal repeats, and a peak found beyond that range is one period of the "
    "wave, not one crossing of the band. A peak that lands on the edge of that "
    "range, or that is attained at more than one lag -- which is what two bands "
    "in exact antiphase do -- is reported as UNRESOLVED rather than refined "
    "into a number, and so is a band whose variation is no larger than the "
    "rounding of its own magnitude. The "
    "integer peak is then refined by cosine interpolation over the three "
    "samples around it, which is exact for a sinusoidal correlation function "
    "and recovers lags well below one sampling interval; parabolic "
    "interpolation is used only where the cosine form does not apply. "
    "Adjacent-band lags are accumulated and a straight line is fitted to "
    "cumulative lag against cumulative distance; the phase speed is the "
    "reciprocal of its slope and the uncertainty is the standard error of that "
    "slope propagated through the reciprocal. That standard error is FLOORED "
    "at the lag resolution -- one tenth of a sampling interval per band pair, "
    "accumulated in quadrature along the chain -- because the scatter of a "
    "smooth field understates how well a lag can really be read off an hourly "
    "record, and without the floor a pattern whose lags are zero to machine "
    "precision comes back as a speed measured to absurd precision instead of "
    "as no speed at all."
)

#: Per-band-pair lag uncertainty floor, as a fraction of one sampling
#: interval. Sub-sample interpolation of a correlation peak is good to a few
#: per cent of a sample on a clean signal and worse on a real one; a tenth of a
#: sample is a deliberately conservative floor and it is what stops a
#: perfectly smooth field from reporting a speed with no uncertainty.
DEFAULT_LAG_RESOLUTION_SAMPLES = 0.1


#: A band counts as having no variance when its spread is this small relative
#: to its own magnitude. An absolute test would be fooled by a constant series
#: whose samples differ in the last bit: the rounding noise correlates with
#: itself perfectly and would be read as a signal.
_VARIANCE_FLOOR = 1e-12

#: Two lags whose correlations differ by less than this are treated as tied,
#: and a tied maximum makes the lag undecided rather than pointing at whichever
#: of them the search happened to reach first. Small enough that it catches
#: exact degeneracy -- antiphase bands, a constant field -- without touching a
#: real correlation peak, which is never flat to nine decimal places.
_TIE_TOLERANCE = 1e-9


def _has_variance(values: NDArray[np.float64], spread: float) -> bool:
    """Whether a series varies by more than the rounding of its own magnitude."""
    scale = max(abs(float(np.mean(values))), float(np.max(np.abs(values))))
    return spread > _VARIANCE_FLOOR * scale


def _correlation_at_lag(a: NDArray[np.float64], b: NDArray[np.float64], lag: int) -> float:
    """Pearson correlation of ``a(t)`` against ``b(t + lag)`` on their overlap."""
    if lag >= 0:
        x, y = a[: a.size - lag], b[lag:]
    else:
        x, y = a[-lag:], b[: b.size + lag]
    if x.size < 4:
        return float("nan")
    x_spread = float(np.std(x))
    y_spread = float(np.std(y))
    if not _has_variance(x, x_spread) or not _has_variance(y, y_spread):
        return float("nan")
    covariance = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
    return covariance / (x_spread * y_spread)


def _first_zero_crossing(series: NDArray[np.float64], cap: int) -> int:
    """Smallest lag at which a series stops correlating with itself."""
    for lag in range(1, cap + 1):
        value = _correlation_at_lag(series, series, lag)
        if math.isnan(value) or value <= 0.0:
            return lag
    return cap


def _refine_peak(correlations: NDArray[np.float64], index: int) -> float:
    """Sub-sample offset of a correlation peak from its integer position."""
    left = float(correlations[index - 1])
    centre = float(correlations[index])
    right = float(correlations[index + 1])
    if math.isnan(left) or math.isnan(right) or centre == 0.0:
        return 0.0
    ratio = (left + right) / (2.0 * centre)
    if -1.0 < ratio < 1.0:
        omega = math.acos(ratio)
        if omega > 1e-9:
            theta = math.atan2(left - right, 2.0 * centre * math.sin(omega))
            return -theta / omega
    curvature = left - 2.0 * centre + right
    if curvature == 0.0:
        return 0.0
    return -0.5 * (right - left) / curvature


def _band_lag(
    earlier: NDArray[np.float64], later: NDArray[np.float64], cap: int
) -> tuple[float | None, float, str]:
    """Lag in SAMPLES of ``later`` behind ``earlier``, with its peak correlation.

    Returns ``(None, correlation, reason)`` when the peak sits at the edge of
    the unambiguous range, which is the honest answer for a pair whose lag is
    longer than the signal's own coherence.
    """
    limit = max(_first_zero_crossing(earlier, cap), _first_zero_crossing(later, cap))
    search = max(1, min(limit + 1, cap))
    lags = np.arange(-search, search + 1)
    correlations = np.asarray(
        [_correlation_at_lag(earlier, later, int(lag)) for lag in lags], dtype=float
    )
    if bool(np.all(np.isnan(correlations))):
        return None, float("nan"), "no variance in one of the bands"
    index = int(np.nanargmax(correlations))
    peak = float(correlations[index])
    if index == 0 or index == lags.size - 1:
        return None, peak, "correlation peak at the edge of the unambiguous lag range"
    tied = int(np.count_nonzero(correlations >= peak - _TIE_TOLERANCE))
    if tied > 1:
        return (
            None,
            peak,
            f"the correlation reaches its maximum at {tied} different lags, so which "
            "one is the propagation lag is not decided by the data. Two bands in "
            "exact antiphase do this: the peak sits half a period either side of "
            "zero and the sign of the lag is a coin toss",
        )
    return float(lags[index]) + _refine_peak(correlations, index), peak, ""


@dataclass(frozen=True, slots=True)
class PhaseSpeedEstimate:
    """A meridional phase speed, or a refusal to state one.

    ``speed_m_s`` is None whenever the correlation does not support a number,
    and ``refusal_reason`` then says which check failed. A detector that always
    returns a speed is not a detector, so the refusals are part of the result
    type rather than an exception: a caller sweeping a parameter wants the
    refusals in the table beside the measurements.

    The sign follows the ORDER of the latitudes that were passed. Latitudes
    ordered from the pole toward the equator make a positive speed equatorward,
    which is the direction a storm-launched disturbance travels;
    ``sequence_direction`` records which it was.
    """

    speed_m_s: float | None
    uncertainty_m_s: float | None
    refusal_reason: str | None
    mean_correlation: float
    minimum_correlation: float
    r_squared: float
    slope_significance: float
    band_pair_count: int
    unresolved_band_pairs: tuple[str, ...]
    distances_m: tuple[float, ...]
    cumulative_lag_hours: tuple[float, ...]
    sequence_direction: str
    method: str = PHASE_SPEED_METHOD

    @property
    def measured(self) -> bool:
        """True when a speed was measured rather than refused."""
        return self.speed_m_s is not None


def _sequence_direction(latitudes: NDArray[np.float64]) -> str:
    """Whether the latitude order runs equatorward, poleward or across."""
    magnitudes = np.abs(latitudes)
    if bool(np.all(np.diff(magnitudes) < 0.0)):
        return "equatorward"
    if bool(np.all(np.diff(magnitudes) > 0.0)):
        return "poleward"
    return "crosses the equator"


def _checked_axes(
    hovmoller: NDArray[np.float64],
    latitudes: Axis,
    times: Sequence[_dt.datetime],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Validate a Hovmoller and its axes; return them with the time step."""
    field = np.asarray(hovmoller, dtype=float)
    bands = np.asarray([float(lat) for lat in latitudes], dtype=float)
    if field.ndim != 2:
        raise ValueError(
            f"hovmoller must be two-dimensional (time, latitude), got shape {field.shape}"
        )
    if field.shape != (len(times), bands.size):
        raise ValueError(
            f"hovmoller has shape {field.shape} but the axes are "
            f"({len(times)}, {bands.size})"
        )
    if not bool(np.all(np.isfinite(field))):
        raise ValueError(
            f"hovmoller holds {int(np.count_nonzero(~np.isfinite(field)))} "
            "non-finite values. Supply the margin the background needs instead "
            "of correlating across NaN."
        )
    steps = np.diff(bands)
    if not (bool(np.all(steps > 0.0)) or bool(np.all(steps < 0.0))):
        raise ValueError("latitudes must be strictly monotonic")
    step_hours = _uniform_step_hours([ensure_utc(t, "times") for t in times])
    return field, bands, step_hours


def estimate_phase_speed(
    hovmoller: NDArray[np.float64],
    latitudes: Axis,
    times: Sequence[_dt.datetime],
    *,
    minimum_correlation: float = 0.5,
    minimum_r_squared: float = 0.5,
    minimum_slope_significance: float = 2.0,
    lag_resolution_samples: float = DEFAULT_LAG_RESOLUTION_SAMPLES,
) -> PhaseSpeedEstimate:
    """Meridional phase speed from the lag between adjacent latitude bands.

    ``hovmoller`` is the 2-D perturbation array, ``(time, latitude)`` -- the
    ``perturbation`` attribute of a ``Hovmoller``, or any array a test builds
    by hand. The method is stated in full in ``PhaseSpeedEstimate.method``.

    FOUR WAYS TO GET NO NUMBER, and each of them is a real answer:

      * too few band pairs produced a lag at all, because their correlation
        peaks were ambiguous, off the end of the unambiguous range, or the
        bands did not vary -- a slope needs at least two surviving pairs;

      * the bands do not track a common signal at all (mean peak correlation
        below ``minimum_correlation``) -- noise, or two unrelated structures;
      * the lags do not grow linearly with distance (fit R-squared below
        ``minimum_r_squared``) -- several structures at once, or a wave that
        changes speed within the window;
      * the fitted slope is not distinguishable from zero (below
        ``minimum_slope_significance`` standard errors) -- a standing
        oscillation, where every band peaks at the same time and the implied
        speed is infinite rather than large.

    The thresholds are arguments so a caller can state them; they are not tuned
    against any particular field.
    """
    field, bands, step_hours = _checked_axes(hovmoller, latitudes, times)
    direction = _sequence_direction(bands)
    cap = max(2, field.shape[0] // 4)

    distances: list[float] = [0.0]
    cumulative: list[float] = [0.0]
    peaks: list[float] = []
    unresolved: list[str] = []
    running_lag = 0.0
    running_distance = 0.0
    for index in range(bands.size - 1):
        lag, peak, band_reason = _band_lag(field[:, index], field[:, index + 1], cap)
        span = abs(meridional_distance_m(float(bands[index]), float(bands[index + 1])))
        if lag is None:
            unresolved.append(
                f"{bands[index]:.1f} to {bands[index + 1]:.1f} deg: {band_reason}"
            )
            continue
        running_lag += lag * step_hours
        running_distance += span
        distances.append(running_distance)
        cumulative.append(running_lag)
        peaks.append(peak)

    if len(distances) < 3:
        return PhaseSpeedEstimate(
            speed_m_s=None,
            uncertainty_m_s=None,
            refusal_reason=(
                f"only {len(distances) - 1} of {bands.size - 1} band pairs produced a "
                "resolved lag; a slope and its uncertainty need at least two"
            ),
            mean_correlation=float(np.mean(peaks)) if peaks else float("nan"),
            minimum_correlation=float(np.min(peaks)) if peaks else float("nan"),
            r_squared=float("nan"),
            slope_significance=float("nan"),
            band_pair_count=len(peaks),
            unresolved_band_pairs=tuple(unresolved),
            distances_m=tuple(distances),
            cumulative_lag_hours=tuple(cumulative),
            sequence_direction=direction,
        )

    x = np.asarray(distances, dtype=float)
    y = np.asarray(cumulative, dtype=float)
    design = np.vstack([x, np.ones_like(x)]).T
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope = float(coefficients[0])
    residual = y - design @ coefficients
    spread = float(np.sum((x - x.mean()) ** 2))
    variance = float(np.sum(residual**2)) / (x.size - 2)
    scatter_error = math.sqrt(variance / spread) if spread > 0.0 else float("inf")
    # The floor: each accumulated lag inherits at least the lag resolution from
    # every band pair behind it, which adds in quadrature along the chain.
    resolution_hours = lag_resolution_samples * step_hours * math.sqrt(len(peaks))
    resolution_error = resolution_hours / math.sqrt(spread) if spread > 0.0 else float("inf")
    slope_error = max(scatter_error, resolution_error)
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 0.0
    significance = abs(slope) / slope_error if slope_error > 0.0 else float("inf")
    mean_peak = float(np.mean(peaks))
    lowest_peak = float(np.min(peaks))

    reason: str | None = None
    if mean_peak < minimum_correlation:
        reason = (
            f"mean peak correlation between adjacent bands is {mean_peak:.2f}, below "
            f"the {minimum_correlation:.2f} required. The bands are not tracking one "
            "structure, so the lags between them are not a propagation time."
        )
    elif r_squared < minimum_r_squared:
        reason = (
            f"lag against distance has R-squared {r_squared:.2f}, below the "
            f"{minimum_r_squared:.2f} required. The lags do not grow linearly with "
            "distance, so no single phase speed describes them."
        )
    elif significance < minimum_slope_significance:
        reason = (
            f"the fitted slope is {significance:.1f} standard errors from zero, below "
            f"the {minimum_slope_significance:.1f} required. Lag does not grow with "
            "distance: the pattern is standing or stationary, not propagating."
        )

    if reason is not None:
        return PhaseSpeedEstimate(
            speed_m_s=None,
            uncertainty_m_s=None,
            refusal_reason=reason,
            mean_correlation=mean_peak,
            minimum_correlation=lowest_peak,
            r_squared=r_squared,
            slope_significance=significance,
            band_pair_count=len(peaks),
            unresolved_band_pairs=tuple(unresolved),
            distances_m=tuple(distances),
            cumulative_lag_hours=tuple(cumulative),
            sequence_direction=direction,
        )

    speed = 1.0 / (slope * _SECONDS_PER_HOUR)
    uncertainty = abs(slope_error / slope**2) / _SECONDS_PER_HOUR
    return PhaseSpeedEstimate(
        speed_m_s=speed,
        uncertainty_m_s=uncertainty,
        refusal_reason=None,
        mean_correlation=mean_peak,
        minimum_correlation=lowest_peak,
        r_squared=r_squared,
        slope_significance=significance,
        band_pair_count=len(peaks),
        unresolved_band_pairs=tuple(unresolved),
        distances_m=tuple(distances),
        cumulative_lag_hours=tuple(cumulative),
        sequence_direction=direction,
    )


@dataclass(frozen=True, slots=True)
class PropagationDelay:
    """How long a disturbance took to get from one latitude to another.

    ``delay_hours`` is the DIRECT cross-correlation lag between the two bands.
    ``chained_delay_hours`` sums the lags of the adjacent band pairs between
    them. The two answer the same question differently: the direct lag needs
    the far band to still resemble the near one, which a decaying or dispersing
    wave does not guarantee, while the chained sum only needs each neighbour to
    resemble the next. They are both reported because a disagreement between
    them is information about the wave, not a bug.
    """

    from_latitude_deg: float
    to_latitude_deg: float
    distance_m: float
    delay_hours: float | None
    correlation: float
    implied_speed_m_s: float | None
    chained_delay_hours: float | None
    note: str = ""

    @property
    def resolved(self) -> bool:
        """True when the direct lag was measurable."""
        return self.delay_hours is not None


def propagation_delay(
    hovmoller: NDArray[np.float64],
    latitudes: Axis,
    times: Sequence[_dt.datetime],
    *,
    from_lat: float,
    to_lat: float,
    minimum_correlation: float = 0.5,
) -> PropagationDelay:
    """Hours for the disturbance to travel from ``from_lat`` to ``to_lat``.

    The nearest band in ``latitudes`` is used for each; the band actually used
    is reported, so a request for 70 degrees answered by the 70.0-degree row of
    a 2-degree grid cannot be confused with one answered by a 68-degree row.

    A positive delay means the disturbance reached ``to_lat`` later.

    ``delay_hours`` is None when the two bands do not track a common structure
    well enough to time it -- either because the correlation peak sits at the
    edge of the unambiguous lag range, or because the peak correlation is below
    ``minimum_correlation``. Two bands tens of degrees apart routinely fail
    that test even when each neighbouring pair between them passes, which is
    what ``chained_delay_hours`` is for.
    """
    field, bands, step_hours = _checked_axes(hovmoller, latitudes, times)
    start = int(np.argmin(np.abs(bands - from_lat)))
    end = int(np.argmin(np.abs(bands - to_lat)))
    if start == end:
        raise ValueError(
            f"from_lat {from_lat:g} and to_lat {to_lat:g} both fall on the "
            f"{bands[start]:g} deg band; there is no distance between them"
        )
    cap = max(2, field.shape[0] // 4)
    lag, peak, reason = _band_lag(field[:, start], field[:, end], cap)
    if lag is not None and (math.isnan(peak) or peak < minimum_correlation):
        lag = None
        reason = (
            f"peak correlation {peak:.2f} between the two bands is below the "
            f"{minimum_correlation:.2f} required; they are not tracking a common "
            "structure, so the lag between them is not a propagation time"
        )
    distance = meridional_distance_m(float(bands[start]), float(bands[end]))

    step = 1 if end > start else -1
    chained = 0.0
    chain_broken = False
    for index in range(start, end, step):
        first, second = (index, index + 1) if step == 1 else (index, index - 1)
        link_lag, _link_peak, _link_reason = _band_lag(
            field[:, first], field[:, second], cap
        )
        if link_lag is None:
            chain_broken = True
            break
        chained += link_lag * step_hours

    delay = None if lag is None else lag * step_hours
    implied: float | None = None
    if delay is not None and delay != 0.0:
        implied = abs(distance) / (delay * _SECONDS_PER_HOUR)
    return PropagationDelay(
        from_latitude_deg=float(bands[start]),
        to_latitude_deg=float(bands[end]),
        distance_m=abs(distance),
        delay_hours=delay,
        correlation=peak,
        implied_speed_m_s=implied,
        chained_delay_hours=None if chain_broken else chained,
        note=reason,
    )


def compare_to_literature(
    speed_m_s: float | None, delay_hours: float | None
) -> dict[str, object]:
    """Place a measurement against the published TAD numbers.

    Returns a plain dictionary so the comparison can be written straight into
    an evidence file. Each verdict is BELOW, WITHIN, ABOVE, or NOT MEASURED,
    and the caveat that agreement with a modelled band is agreement between
    models travels with every one of them.
    """
    comparison: dict[str, object] = {
        "published_phase_speed_m_s": list(PUBLISHED_PHASE_SPEED_M_S),
        "published_high_to_mid_delay_hours": list(PUBLISHED_HIGH_TO_MID_DELAY_HOURS),
        "published_reference": PUBLISHED_REFERENCE,
        "measured_phase_speed_m_s": speed_m_s,
        "measured_delay_hours": delay_hours,
        "phase_speed_verdict": _verdict(speed_m_s, PUBLISHED_PHASE_SPEED_M_S),
        "delay_verdict": _verdict(delay_hours, PUBLISHED_HIGH_TO_MID_DELAY_HOURS),
        "agreement_caveat": AGREEMENT_IS_NOT_OBSERVATIONAL_SUPPORT,
    }
    notes: list[str] = []
    if speed_m_s is not None and speed_m_s < 0.0:
        notes.append(
            "The measured phase speed is NEGATIVE, meaning the phase progresses "
            "along the reverse of the latitude order it was measured on. The "
            "published band describes EQUATORWARD propagation away from the "
            "auroral source, so this measurement is not a slow version of the "
            "published disturbance; it is a different thing pointing the other "
            "way, and the numerical verdict below should be read with that in "
            "mind."
        )
    if delay_hours is not None and delay_hours < 0.0:
        notes.append(
            "The measured delay is NEGATIVE: the disturbance reached the "
            "lower-latitude band FIRST, which is the opposite of the "
            "high-latitude-to-mid-latitude sequence the published time describes."
        )
    if speed_m_s is None:
        notes.append(
            "No phase speed was measured, so no comparison against the published "
            "band is possible. A refusal is not a value outside the band."
        )
    if delay_hours is None:
        notes.append(
            "No propagation delay was measured, so no comparison against the "
            "published 3-4 h is possible."
        )
    comparison["notes"] = notes
    return comparison


def _verdict(value: float | None, band: tuple[float, float]) -> str:
    """BELOW, WITHIN, ABOVE or NOT MEASURED, against a published band."""
    if value is None or not math.isfinite(value):
        return "NOT MEASURED"
    low, high = band
    if value < low:
        return "BELOW"
    if value > high:
        return "ABOVE"
    return "WITHIN"
