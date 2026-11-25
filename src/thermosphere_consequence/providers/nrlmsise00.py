"""NRLMSISE-00 species-resolved atmosphere provider.

Licence position, which is the reason this file names version 0 explicitly at
every call site: pymsis ships an MIT-licensed wrapper over several MSIS
generations. NRLMSISE-00 (the 2001 model, ``version=0``) is the generation this
project uses. Its compiled artefact carries NO licence terms at all -- see
MODEL_LICENCE below and third_party/licences/NRLMSISE-00-PROVENANCE.md; that
absence is why the resource is research-only rather than cleared for
production. NRLMSIS 2.0/2.1 is governed by a US Naval
Research Laboratory licence granting use "for academic, non-commercial,
purposes only" and covered by US Patent 10,641,925. This module therefore
pins version 0 and a guard test asserts no other version can be reached.

Scientific position: NRLMSISE-00 is the model SpaceX itself used for Starlink
drag estimation (Fang et al. 2022), which makes it the right instrument for
asking what the operator's own model would have predicted. It is NOT ground
truth for storm-time density. Empirical MSIS models are documented as
underestimating rapid storm-time density enhancements relative to accelerometer
observations and physics-based models during main-phase storms (e.g., Fang et
al. 2022; He et al. 2023 report MSIS 2.0 underestimating Swarm-A orbit-averaged
enhancements near 438 km by ~50-70% on 3-4 Feb 2022, alongside global-mean
peaks of ~10-15% at 200 km and ~20-25% at 300 km), though bias depends on
altitude, storm phase, and quiet baseline so outputs are not an established
universal lower bound.

Canonical tickets: P03_05_T026, P03_05_T027, P03_05_T028.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

from ..domain.temporal import require_aware_datetime
from ..protocols.atmosphere import AtmosphereState, DriverState

# pymsis output column order for calculate(); index 9 (NO) is not produced by
# NRLMSISE-00 and comes back as NaN, which is why it is excluded below rather
# than being read and defended against downstream.
_PYMSIS_SPECIES_COLUMNS: tuple[tuple[int, str], ...] = (
    (1, "N2"),
    (2, "O2"),
    (3, "O"),
    (4, "He"),
    (5, "H"),
    (6, "Ar"),
    (7, "N"),
    (8, "anomalous_O"),
)
_TOTAL_MASS_DENSITY_COLUMN = 0
_TEMPERATURE_COLUMN = 10

MODEL_VERSION = 0
MODEL_NAME = "NRLMSISE-00"
MODEL_REFERENCE = "Picone et al. 2002, doi:10.1029/2002JA009430"
# CORRECTED 2026-08-16. This previously read "public domain
# (COPYING.NRLMSISE-00)". That citation was for a DIFFERENT artefact:
# COPYING.NRLMSISE-00 accompanies Dominik Brodowski's independent C port, not
# the NRL Fortran this project compiles. pymsis fetches NRLMSISE-00.FOR at
# build time from map.nrl.navy.mil and compiles it into msis00f; that file was
# retrieved and searched, and carries NO licence text of any kind. See
# third_party/licences/NRLMSISE-00-PROVENANCE.md.
#
# An absence of terms is not a grant. Nothing forbidding use was found, and
# nothing granting it either, so the artefact is research-only pending primary
# evidence (BLK-NRLMSISE00-ARTEFACT-PROVENANCE).
MODEL_LICENCE = (
    "no licence terms located for the compiled artefact; research-only pending "
    "verification (see third_party/licences/NRLMSISE-00-PROVENANCE.md)"
)
MODEL_SOURCE_SHA256 = "cce0420e90781c256bc6705c4cc8056b054812d6308adccb7081bf09af0d44cb"

# Documented storm-underestimation caveat, attached to every state this
# provider returns so it travels with the number rather than living only in
# documentation.
_STORM_UNDERESTIMATE_NOTE = (
    "NRLMSISE-00 is an empirical model documented to underestimate rapid "
    "storm-time density enhancements relative to accelerometer observations "
    "and physics-based models during main-phase storms (e.g., Fang et al. "
    "2022; He et al. 2023 report MSIS 2.0 underestimating Swarm-A enhancements "
    "near 438 km by ~50-70% on 3-4 Feb 2022), though bias varies with "
    "altitude, storm phase, and quiet baseline; not an established universal "
    "lower bound."
)


class Nrlmsise00Provider:
    """Species-resolved neutral atmosphere from the 2001 MSIS generation.

    Research-only: no licence terms were located for the artefact this
    compiles. See MODEL_LICENCE.
    """

    name = MODEL_NAME
    version = "pymsis/version=0"

    def __init__(self, *, include_anomalous_oxygen: bool = True) -> None:
        self._include_anomalous_oxygen = include_anomalous_oxygen

    def state_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> AtmosphereState:
        """Return the neutral atmosphere at one point.

        Altitude is geodetic and supplied in metres; pymsis wants kilometres,
        so the conversion happens here at the boundary rather than leaking a
        kilometre convention into the rest of the system.
        """
        time_utc = require_aware_datetime(time_utc, "time_utc")
        if altitude_m < 0:
            raise ValueError(f"altitude must be non-negative, got {altitude_m}")
        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError(f"latitude out of range: {latitude_deg}")

        raw = self._raw(time_utc, altitude_m, latitude_deg, longitude_deg, drivers)

        number_densities: dict[str, float] = {}
        for column, symbol in _PYMSIS_SPECIES_COLUMNS:
            if symbol == "anomalous_O" and not self._include_anomalous_oxygen:
                continue
            value = float(raw[column])
            # A NaN means the model did not report this species. It must stay
            # absent: a zero would be a claim that the species is not there.
            if np.isnan(value):
                continue
            number_densities[symbol] = value

        total = float(raw[_TOTAL_MASS_DENSITY_COLUMN])
        temperature = float(raw[_TEMPERATURE_COLUMN])
        if np.isnan(temperature):
            raise ValueError(
                f"{MODEL_NAME} returned NaN temperature at {altitude_m} m; "
                "inputs are outside the model's valid domain"
            )

        return AtmosphereState(
            time_utc=time_utc,
            altitude_m=altitude_m,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            temperature_k=temperature,
            number_densities=number_densities,
            total_mass_density=None if np.isnan(total) else total,
            drivers=drivers,
            provider_name=MODEL_NAME,
            provider_version=self.version,
            notes=(_STORM_UNDERESTIMATE_NOTE,),
        )

    def _raw(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> np.ndarray:
        """Call pymsis, pinned to the 2001 model (version=0)."""
        from pymsis import msis

        # NRLMSISE-00 ignores the 7-element ap array unless the geomagnetic
        # switch is set to storm-time mode. Supplying the history WITHOUT this
        # switch silently falls back to daily-Ap mode, which smears a
        # multi-hour ap=56 spike into a daily mean near 25 and understates the
        # storm response by more than a factor of two at 210 km. That failure
        # is invisible in the output — the model returns a perfectly plausible
        # density — so the switch is set here and asserted by a test.
        if drivers.ap_history is not None:
            aps = [list(drivers.ap_history)]
            options = msis.create_options(geomagnetic_activity=-1)
        else:
            aps = [[drivers.ap_daily] * 7]
            options = None

        out = msis.calculate(
            np.datetime64(time_utc.replace(tzinfo=None), "s"),
            longitude_deg,
            latitude_deg,
            altitude_m / 1000.0,
            drivers.f107,
            drivers.f107a,
            aps,
            options=options,
            version=MODEL_VERSION,
        )
        return np.asarray(out).ravel()


class ZeroWindProvider:
    """The explicit, labelled zero-wind default.

    Neutral wind at 210 km is not zero — storm-time winds reach hundreds of
    m/s and modify the relative velocity that drag depends on. This provider
    exists so that "we neglected wind" is a named, greppable choice appearing
    in the provenance of every run, rather than an unstated omission.

    Canonical ticket: P03_05_T024.
    """

    name = "zero-wind"
    version = "1.0"
    note = (
        "Neutral wind set to exactly zero. This is a modelling choice, not a "
        "measurement: HWM14 storm-time winds at 210 km reach several hundred "
        "m/s and would alter the drag-relevant relative velocity by a few "
        "percent. Recorded so the omission is visible in every run's provenance."
    )

    def wind_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> tuple[float, float]:
        return (0.0, 0.0)
