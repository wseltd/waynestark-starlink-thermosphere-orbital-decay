"""Horizontal neutral wind, and the relative velocity that drag actually sees.

Drag depends on the velocity of the spacecraft RELATIVE to the gas, not on its
inertial velocity. This project already models one part of that difference --
the atmosphere co-rotating with the Earth, worth ~340 m/s at the equator -- and
has until now set the other part, the neutral wind, to exactly zero. Zero is a
defensible engineering choice and an indefensible scientific claim: the winds
at 210 km during a geomagnetic storm are not small, and the repository's own
caveats already say so.

    ``providers.nrlmsise00.ZeroWindProvider``: "HWM14 storm-time winds at
    210 km reach several hundred m/s and would alter the drag-relevant
    relative velocity by a few percent."

    ``scenarios.starlink_analysis``: "Neutral wind is zero. Storm-time winds
    at 210 km reach several hundred m/s and would alter relative velocity by a
    few percent."

    Fang et al. 2022 (doi:10.1029/2022SW003193) simulate this exact event with
    WAM-IPE, in which the storm-time thermospheric circulation -- the same
    circulation that redistributes the density this project is measuring -- is
    the mechanism producing those winds.

This module turns that caveat into a swept parameter. ``ZeroWindProvider`` is
re-exported here so the whole wind question has one place to look, and the zero
case appears in ``WIND_SENSITIVITY_CASES_M_S`` as ONE BRANCH of the sweep. It
is not the centre of the sweep and it is not the truth the other cases are
perturbations from; it is the case in which the wind term is omitted.

``Hwm14WindProvider`` is the real model the sweep stands in for. It is OPTIONAL
and it is not a default, for a licence reason rather than a physics one: the
NRL Fortran and its three coefficient files carry no explicit terms at all, so
``config/licence_policy.yaml`` records HWM14 as AMBER and excluded from the
production path under blocker BLK-HWM14-TERMS. Where it is present, it says
what the bounded sweep can only bracket, and ``evidence/WIND_MODEL_AUDIT.md``
records how the two compare along the February 2022 insertion trajectory.

Canonical ticket lineage: P03_05_T024 (the labelled zero-wind default).
"""

from __future__ import annotations

import datetime as _dt
import importlib
import math
import os
import pathlib
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from ..domain.constants import OMEGA_EARTH
from ..domain.temporal import require_aware_datetime
from ..propagation.propagator import geodetic_latlon
from ..protocols.atmosphere import DriverState
from .nrlmsise00 import ZeroWindProvider

__all__ = [
    "HWM14_COEFFICIENT_FILES",
    "HWM14_MODULE_NAME",
    "HWM14_UNAVAILABLE",
    "WIND_SENSITIVITY_CASES_M_S",
    "ConstantAlongTrackWindProvider",
    "Hwm14UnavailableError",
    "Hwm14WindProvider",
    "ZeroWindProvider",
    "corotation_velocity",
    "hwm14_available",
    "relative_velocity_with_wind",
    "wind_sweep_providers",
]

#: The zonal wind speeds swept when reporting wind sensitivity, in m/s,
#: positive eastward.
#:
#: Magnitudes, not guesses. The repository's own zero-wind caveat records HWM14
#: storm-time winds at 210 km reaching "several hundred m/s", and Fang et al.
#: 2022 simulate the storm-time circulation for this specific event. +/-500 m/s
#: is the upper end of "several hundred"; +/-250 m/s is a moderately disturbed
#: value inside it; 0.0 is the omission this project has been making, carried
#: as one branch so it can be compared against the others rather than assumed.
#:
#: Both signs are swept because the sign decides whether the wind ADDS to or
#: SUBTRACTS from the drag-relevant relative speed. On a prograde orbit an
#: eastward (positive) wind moves the gas with the spacecraft and reduces the
#: relative speed; a westward (negative) wind increases it. Sweeping only one
#: sign would report half the sensitivity.
WIND_SENSITIVITY_CASES_M_S: tuple[float, float, float, float, float] = (
    -500.0,
    -250.0,
    0.0,
    250.0,
    500.0,
)


def corotation_velocity(position_m: Sequence[float] | np.ndarray) -> np.ndarray:
    """Velocity of the co-rotating atmosphere at an ECI position, in m/s.

    ``omega x r`` with omega along the ECI z axis. Separated from the wind term
    so a report can state the two contributions to the gas velocity apart from
    one another: at 210 km co-rotation is ~340-480 m/s and the storm wind is a
    few hundred, so neither dominates the other and neither can be dropped on
    the grounds that the other is bigger.
    """
    position = np.asarray(position_m, dtype=float)
    if position.shape != (3,):
        raise ValueError(f"position must have three components, got shape {position.shape}")
    omega = np.array([0.0, 0.0, OMEGA_EARTH.value])
    return np.cross(omega, position)


def relative_velocity_with_wind(
    velocity_eci: Sequence[float] | np.ndarray,
    position: Sequence[float] | np.ndarray,
    wind_east_m_s: float,
    wind_north_m_s: float,
    gmst_rad: float,
) -> np.ndarray:
    """Drag-relevant relative velocity in ECI, including the neutral wind.

    The gas velocity is the co-rotation term plus the horizontal neutral wind:

        v_gas = omega x r + w_east * e_east + w_north * e_north
        v_rel = v_spacecraft - v_gas

    The local east and north unit vectors are built in the INERTIAL frame from
    the inertial longitude ``theta = longitude + gmst`` and the geodetic
    latitude, so no separate ECEF round trip is needed and the geodetic
    latitude comes from the same routine the rest of the project uses:

        e_east  = (-sin theta,  cos theta, 0)
        e_north = (-sin phi cos theta, -sin phi sin theta, cos phi)

    Sign convention, which is the part that is easy to get backwards: for a
    prograde orbit an EASTWARD wind moves the gas in the same sense as the
    spacecraft and REDUCES the relative speed, exactly as co-rotation does. A
    westward wind increases it. Over a 53.22 degree orbit the eastward unit
    vector projects onto the along-track direction with a factor between
    cos(53.22 deg) = 0.60 at the equator and 1.00 at the orbit's highest
    latitude, so a zonal wind of W m/s changes the relative speed by between
    0.60 W and 1.00 W depending on where in the orbit it is evaluated.
    """
    velocity = np.asarray(velocity_eci, dtype=float)
    location = np.asarray(position, dtype=float)
    if velocity.shape != (3,):
        raise ValueError(f"velocity_eci must have three components, got shape {velocity.shape}")
    if location.shape != (3,):
        raise ValueError(f"position must have three components, got shape {location.shape}")
    if not math.isfinite(wind_east_m_s) or not math.isfinite(wind_north_m_s):
        raise ValueError(
            f"wind components must be finite, got east={wind_east_m_s}, "
            f"north={wind_north_m_s}"
        )
    if not math.isfinite(gmst_rad):
        raise ValueError(f"gmst_rad must be finite, got {gmst_rad}")

    latitude_deg, longitude_deg = geodetic_latlon(location, gmst_rad)
    phi = math.radians(latitude_deg)
    theta = math.radians(longitude_deg) + gmst_rad

    east = np.array([-math.sin(theta), math.cos(theta), 0.0])
    north = np.array(
        [
            -math.sin(phi) * math.cos(theta),
            -math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ]
    )
    gas = corotation_velocity(location) + wind_east_m_s * east + wind_north_m_s * north
    return velocity - gas


class ConstantAlongTrackWindProvider:
    """A constant horizontal neutral wind, swept as a sensitivity parameter.

    The swept quantity is the ZONAL (eastward) component, because that is the
    component that projects most strongly onto the along-track direction of a
    prograde orbit and therefore does most to change the drag-relevant relative
    speed. The meridional component is settable and defaults to zero: on a
    53.22 degree orbit a north-south wind is close to cross-track over much of
    the revolution and changes the relative SPEED only at second order, so
    sweeping it alongside the zonal component would spend runs on a smaller
    effect.

    This is a CONSTANT wind, which no real thermosphere has. It is a
    sensitivity instrument: it answers "if the neutral gas were moving at W m/s
    eastward everywhere, how much would the drag-relevant relative speed and
    the drag acceleration change?" It is not a wind model, it is not HWM14, and
    a result computed with it must not be reported as a wind-resolved result.
    """

    version = "1.0"

    def __init__(self, wind_east_m_s: float, *, wind_north_m_s: float = 0.0) -> None:
        if not math.isfinite(wind_east_m_s) or not math.isfinite(wind_north_m_s):
            raise ValueError(
                f"wind components must be finite, got east={wind_east_m_s}, "
                f"north={wind_north_m_s}"
            )
        self.wind_east_m_s = float(wind_east_m_s)
        self.wind_north_m_s = float(wind_north_m_s)
        self.name = (
            f"constant_wind_east{self.wind_east_m_s:+.0f}_"
            f"north{self.wind_north_m_s:+.0f}_m_s"
        )
        self.note = (
            f"CONSTANT neutral wind of {self.wind_east_m_s:+.0f} m/s eastward and "
            f"{self.wind_north_m_s:+.0f} m/s northward, applied at every point and "
            "every instant. A SENSITIVITY CASE, not a wind model: the real "
            "thermospheric wind field varies with latitude, local time, altitude "
            "and storm phase. Magnitudes are taken from the storm-time wind scale "
            "this repository already cites (HWM14 storm-time winds at 210 km of "
            "several hundred m/s; the storm-time circulation simulated for this "
            "event by Fang et al. 2022, doi:10.1029/2022SW003193)."
        )

    def wind_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> tuple[float, float]:
        """Return (zonal eastward, meridional northward) wind in m/s.

        Constant by construction, so none of the arguments is consulted. They
        are present because the WindProvider protocol requires them: a caller
        must be able to swap this for a real wind model without changing the
        call site, which is the entire point of the protocol.
        """
        return (self.wind_east_m_s, self.wind_north_m_s)


def wind_sweep_providers(
    cases_m_s: Sequence[float] = WIND_SENSITIVITY_CASES_M_S,
) -> tuple[ConstantAlongTrackWindProvider, ...]:
    """One provider per swept zonal wind case, in the order given.

    The zero case is included as a provider like any other rather than being
    special-cased, so a report that iterates the sweep cannot accidentally
    present it as the reference the others are measured against.
    """
    if not cases_m_s:
        raise ValueError("the wind sweep needs at least one case; none were supplied")
    return tuple(ConstantAlongTrackWindProvider(float(w)) for w in cases_m_s)


# --------------------------------------------------------------------------
# HWM14: the real wind model, when it is present
# --------------------------------------------------------------------------

#: The extension module this provider drives. Not an invented name: it is the
#: module the MIT-licensed wrapper at https://github.com/lgpedersen/hwm14
#: declares in its own setup.py --
#:
#:     Extension(name="hwm14fort", sources=["src/hwm14.f90"],
#:               f2py_options=["only:", "hwm14", ":"])
#:
#: so an installation of that wrapper and a direct f2py build of the NRL source
#: both satisfy this import with the same call signature.
HWM14_MODULE_NAME: Final[str] = "hwm14fort"

#: The three NRL coefficient files hwm14.f90 opens at first call. Named here
#: because "HWM14 did not initialise" is useless and "gd2qd.dat is not in
#: HWMPATH" is actionable.
HWM14_COEFFICIENT_FILES: Final[tuple[str, ...]] = (
    "hwm123114.bin",
    "dwm07b104i.dat",
    "gd2qd.dat",
)

#: Stated once so the error a caller sees and the test that pins it cannot drift.
HWM14_UNAVAILABLE = (
    f"HWM14 is not available: the {HWM14_MODULE_NAME!r} extension module could not be "
    "imported. HWM14 is not installable from PyPI as of this writing -- both maintained "
    "wrappers (pyhwm2014 on PyPI, and https://github.com/lgpedersen/hwm14) build through "
    "numpy.distutils, which NumPy removed at 1.26 for Python 3.12, so pip install fails "
    "before any compilation starts. Build it from the NRL Fortran instead: obtain "
    "hwm14.f90 and the coefficient files "
    f"({', '.join(HWM14_COEFFICIENT_FILES)}) from https://github.com/lgpedersen/hwm14 "
    "and run `python -m numpy.f2py -c hwm14.f90 -m hwm14fort only: hwm14 :` with gfortran, "
    "meson and ninja present. The build recipe this repository used, and the licence "
    "position on the result, are recorded in evidence/WIND_MODEL_AUDIT.md. Until then use "
    "ConstantAlongTrackWindProvider over WIND_SENSITIVITY_CASES_M_S, which is a bounded "
    "sensitivity sweep and not a wind model."
)

#: NRL's own stated vertical extent for the quiet-time model, "0 to ~450+ km"
#: (hwm14.f90 header). Above 250 km the model is an exponential decay with a
#: 60 km scale height rather than fitted structure, so 500 km is taken as the
#: outer edge of the range this project is willing to call a wind model.
HWM14_MIN_ALTITUDE_M: Final[float] = 0.0
HWM14_MAX_ALTITUDE_M: Final[float] = 500_000.0


class Hwm14UnavailableError(RuntimeError):
    """Raised when HWM14 or its coefficient files cannot be reached.

    A distinct type so a caller can offer the bounded sweep as a fallback
    without also swallowing a genuine numerical failure inside the model.
    """


def hwm14_available() -> bool:
    """True when the HWM14 extension module can be imported in this interpreter.

    Says nothing about the coefficient files: constructing
    :class:`Hwm14WindProvider` is what checks those, because that is where a
    directory can still be supplied.
    """
    try:
        importlib.import_module(HWM14_MODULE_NAME)
    except ImportError:
        return False
    return True


def _resolve_coefficient_dir(coefficient_dir: str | os.PathLike[str] | None) -> pathlib.Path:
    """Locate the three NRL coefficient files, or say precisely which is missing.

    ``hwm14.f90``'s ``findandopen`` looks in the working directory, in
    ``../Meta/`` and in ``$HWMPATH``. Only the last of those is reproducible
    from a library, so this resolves an explicit directory or ``HWMPATH`` and
    refuses anything else: a provider that silently succeeded because the
    caller happened to be in the right working directory would fail in the
    next process for reasons invisible in the run record.
    """
    if coefficient_dir is not None:
        directory = pathlib.Path(coefficient_dir)
        origin = "the coefficient_dir argument"
    else:
        from_env = os.environ.get("HWMPATH")
        if not from_env:
            raise Hwm14UnavailableError(
                "HWM14 coefficient files were not located: no coefficient_dir was given "
                "and HWMPATH is unset. HWM14 cannot run without "
                f"{', '.join(HWM14_COEFFICIENT_FILES)}; see evidence/WIND_MODEL_AUDIT.md "
                "for where they come from."
            )
        directory = pathlib.Path(from_env)
        origin = "HWMPATH"

    missing = [name for name in HWM14_COEFFICIENT_FILES if not (directory / name).is_file()]
    if missing:
        raise Hwm14UnavailableError(
            f"HWM14 coefficient files missing from {directory} (taken from {origin}): "
            f"{', '.join(missing)}. All of {', '.join(HWM14_COEFFICIENT_FILES)} are "
            "required; the model reads them once, at its first call."
        )
    return directory


def _hwm14_ap_3hr(drivers: DriverState) -> float:
    """The current 3-hourly ap, which is the only driver HWM14 consumes.

    NRL's README is explicit: STL, F107 and F107A are ignored, and "only the
    second element of the AP argument (the 3-hour ap) is used to compute
    disturbance winds". ``DriverState.ap_history`` carries the NRLMSISE-00
    seven-element array whose second entry is exactly that value; when no
    history is present the daily Ap is used and the substitution is a stated
    approximation, not a silent one -- a daily average understates a storm's
    peak 3-hour bin, so the disturbance wind comes out low rather than high.
    """
    ap = drivers.ap_history[1] if drivers.ap_history is not None else drivers.ap_daily
    if not math.isfinite(ap):
        raise ValueError(f"the 3-hourly ap must be finite, got {ap}")
    if ap < 0.0:
        raise ValueError(
            f"the 3-hourly ap must be non-negative, got {ap}. HWM14 treats a NEGATIVE "
            "ap(2) as a request for quiet-time winds only, so passing one through would "
            "silently drop the DWM07 storm response. Ask for that explicitly with "
            "Hwm14WindProvider(include_disturbance_winds=False)."
        )
    return float(ap)


class Hwm14WindProvider:
    """HWM14 (+DWM07) horizontal neutral wind. The real model, not a sweep case.

    This is the provider ``ConstantAlongTrackWindProvider`` is a stand-in for.
    Where the constant provider answers "if the gas moved at W m/s everywhere",
    this answers "what does the NRL empirical model say the gas is doing here,
    at this local time, at this magnetic latitude, under this ap".

    LICENCE, which is why this is not a default. The Python wrappers are
    MIT-licensed and the NRL Fortran and its coefficient files carry NO explicit
    terms at all -- not a restriction, an absence. ``config/licence_policy.yaml``
    records ``hwm14`` and ``hwm14_coefficients`` as AMBER and
    ``excluded_by_policy`` under blocker BLK-HWM14-TERMS, so a number from this
    provider is a RESEARCH-mode number until NRL confirms the terms in writing.
    Nothing in the production path may construct it by default.

    VALIDITY, from NRL's own README rather than from inference:
      * the quiet-time part represents average conditions for ap <= 12; the
        disturbed part is an AVERAGE perturbation for the given ap, so it is a
        storm climatology and not this storm.
      * the disturbance winds represent the upper thermosphere (above 225 km),
        are held constant with height, and are cut off smoothly below 125 km.
        At 210 km they are therefore an above-225-km value applied downward.
      * there is no solar-activity dependence at all: F10.7 is ignored. NRL note
        the effect is small by day and significant at night.
      * the Kp dependence is constant above Kp = 8, so the most disturbed bins
        saturate.

    UNITS AND ORDER. HWM14 returns ``w(1)`` = meridional (+north) and ``w(2)``
    = zonal (+east), which is the OPPOSITE order from the ``(east, north)``
    pair the WindProvider protocol requires. The swap happens here, once, and
    is pinned against NRL's own published check output in the tests.
    """

    name = "hwm14"
    version = "HWM14.123114+DWM07"

    def __init__(
        self,
        *,
        coefficient_dir: str | os.PathLike[str] | None = None,
        include_disturbance_winds: bool = True,
    ) -> None:
        try:
            module = importlib.import_module(HWM14_MODULE_NAME)
        except ImportError as exc:
            raise Hwm14UnavailableError(HWM14_UNAVAILABLE) from exc

        self.coefficient_dir = _resolve_coefficient_dir(coefficient_dir)
        # hwm14.f90 reads HWMPATH inside findandopen(), on the first call only:
        # `hwminit` latches after that, so a second provider with a different
        # directory in the same process gets the first one's coefficients. The
        # directory is recorded on the instance so a run report can state which
        # one actually took effect rather than which one was asked for.
        os.environ["HWMPATH"] = str(self.coefficient_dir)

        self._hwm14: Any = module.hwm14
        self.include_disturbance_winds = bool(include_disturbance_winds)
        self.name = "hwm14" if self.include_disturbance_winds else "hwm14-quiet"
        self.note = (
            "NRL Horizontal Wind Model 2014"
            + (" with DWM07 disturbance winds" if self.include_disturbance_winds else "")
            + ", driven by the current 3-hourly ap alone (HWM14 ignores F10.7 and solar "
            "local time). The disturbance component is an AVERAGE storm response for that "
            "ap, valid above 225 km and held constant with height, not a simulation of "
            "this storm. LICENCE: the NRL source and coefficient files carry no explicit "
            "terms (BLK-HWM14-TERMS); recorded AMBER and excluded from the production "
            "path in config/licence_policy.yaml, so results from it are research-mode."
        )

    def wind_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> tuple[float, float]:
        """Return (zonal eastward, meridional northward) wind in m/s."""
        when = require_aware_datetime(time_utc, "time_utc").astimezone(_dt.UTC)

        if not math.isfinite(altitude_m):
            raise ValueError(f"altitude must be finite, got {altitude_m}")
        if not HWM14_MIN_ALTITUDE_M <= altitude_m <= HWM14_MAX_ALTITUDE_M:
            raise ValueError(
                f"altitude {altitude_m} m is outside HWM14's range "
                f"[{HWM14_MIN_ALTITUDE_M}, {HWM14_MAX_ALTITUDE_M}] m. NRL state the model "
                "covers 0 to ~450+ km; above 250 km it is an exponential decay with a "
                "60 km scale height, so beyond 500 km it is extrapolation, not a fit."
            )
        if not math.isfinite(latitude_deg) or not -90.0 <= latitude_deg <= 90.0:
            raise ValueError(f"geodetic latitude must lie in [-90, 90], got {latitude_deg}")
        if not math.isfinite(longitude_deg):
            raise ValueError(f"longitude must be finite, got {longitude_deg}")

        ap_3hr = _hwm14_ap_3hr(drivers) if self.include_disturbance_winds else -1.0

        day_of_year = when.timetuple().tm_yday
        # yyddd, which is the documented form; hwm14.f90 uses `mod(iyd,1000)`
        # and consumes the day of year only, so the year digits are carried for
        # legibility and have no effect on the result.
        iyd = (when.year % 100) * 1000 + day_of_year
        seconds_ut = (
            when.hour * 3600.0 + when.minute * 60.0 + when.second + when.microsecond * 1e-6
        )
        solar_local_time_hr = (seconds_ut / 3600.0 + longitude_deg / 15.0) % 24.0

        # stl, f107a and f107 are ignored by HWM14; stl is computed anyway so
        # the call is well formed and the placeholders are visibly placeholders.
        wind = self._hwm14(
            iyd,
            seconds_ut,
            altitude_m / 1000.0,
            latitude_deg,
            longitude_deg,
            solar_local_time_hr,
            -1.0,
            -1.0,
            [-1.0, ap_3hr],
        )
        meridional_north = float(wind[0])
        zonal_east = float(wind[1])
        if not math.isfinite(zonal_east) or not math.isfinite(meridional_north):
            raise ValueError(
                f"HWM14 returned a non-finite wind at {when.isoformat()}, "
                f"{altitude_m} m, {latitude_deg} deg N, {longitude_deg} deg E: "
                f"east={zonal_east}, north={meridional_north}"
            )
        return (zonal_east, meridional_north)
