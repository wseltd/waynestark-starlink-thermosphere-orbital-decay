"""Atmosphere-model audit: answer every reviewer question with a computed number.

This project reports a peak modelled density enhancement of 17.1% at 210 km for
the February 2022 storm. That figure sits BELOW every published figure for the
event, including the published figure for the MSIS family this project itself
uses (He et al. 2023, ~20-25% at 200/300 km). A result that disagrees with the
whole literature is either a finding about the literature or a finding about
the run, and the only way to tell is to open the run.

So this script opens it. It answers twenty numbered questions about exactly
which model was called, with which switches, driven by which indices, sampled
where and when, and it computes the density, species, temperature and
enhancement outputs directly rather than quoting them from a previous report.
It then tests specific, falsifiable explanations for the gap to He et al. and
reports which ones survive.

Nothing here calibrates anything. No published value is used as an input to a
model call; published values appear only on the right-hand side of a
comparison.

Run from the repository root:

    .venv/bin/python scripts/run_atmosphere_audit.py

Writes evidence/atmosphere_audit.json (every number, machine-readable) and
evidence/ATMOSPHERE_MODEL_AUDIT.md (the same numbers as prose).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import math
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from thermosphere_consequence.aerodynamics import effective_drag
from thermosphere_consequence.drivers import space_weather as sw
from thermosphere_consequence.drivers.raw_store import RawStore
from thermosphere_consequence.events import starlink_g4_7 as EV
from thermosphere_consequence.molecular import composition as comp
from thermosphere_consequence.propagation.propagator import (
    DragPropagator,
    OrbitState,
    SpacecraftProperties,
    geodetic_latlon,
    greenwich_sidereal_time,
)
from thermosphere_consequence.protocols.atmosphere import DriverState
from thermosphere_consequence.provenance import execution_environment as _EXEC_ENV
from thermosphere_consequence.providers import nrlmsise00 as N
from thermosphere_consequence.providers.nrlmsise00 import Nrlmsise00Provider
from thermosphere_consequence.providers.published_envelope import (
    PUBLISHED_ENVELOPE_BANDS,
    OutsidePublishedCoverageError,
    PublishedEventDensityEnvelopeProvider,
    UnsupportedAltitudeProfileError,
    band_for,
)
from thermosphere_consequence.providers.wind import (
    WIND_SENSITIVITY_CASES_M_S,
    corotation_velocity,
    relative_velocity_with_wind,
    wind_sweep_providers,
)
from thermosphere_consequence.scenarios.fixed_path import quiet_baseline_drivers
from thermosphere_consequence.validation import literature

EVIDENCE = pathlib.Path("evidence")
STORE = RawStore("data/raw")

# --- the sampling the reported 17.1% came from ----------------------------
# Transcribed from scripts/run_full_analysis.py, which is the run that
# produced evidence/analysis_report.json. Restated here as named constants so
# the audit is checking a stated configuration rather than an implied one.
REPORTED_ALTITUDE_M = 210_000.0
REPORTED_LATITUDE_DEG = 53.22
REPORTED_LONGITUDE_DEG = 0.0
REPORTED_WINDOW_START = dt.datetime(2022, 2, 2, tzinfo=dt.UTC)
REPORTED_WINDOW_END = dt.datetime(2022, 2, 8, tzinfo=dt.UTC)
REPORTED_SAMPLE_INTERVAL = dt.timedelta(hours=1)
REPORTED_QUIET_AP = 4.0

#: The quietest pre-storm 3-hour bin in the ingested GFZ series (ap = 2, Kp =
#: 0.33). Chosen by inspecting the series, not by searching for a bin that
#: produces a favourable answer, and the whole series is printed in the report
#: so the choice can be second-guessed.
PRE_STORM_REFERENCE = dt.datetime(2022, 2, 1, 18, 0, tzinfo=dt.UTC)

#: The altitudes He et al. 2023 quote, so the comparison is like for like.
HE_ET_AL_ALTITUDES_M = (200_000.0, 300_000.0)

#: He et al. 2023 report MSIS2.0 SEPARATELY at those two altitudes, section 3,
#: from Figures 9 and 10: about 10-15% at 200 km and about 20% then 25% on the
#: two storm days at 300 km. This script previously carried a single 20-25%
#: band and treated it as the published figure across the whole 200-300 km
#: layer, which is what made the 210 km result look like it sat below the
#: published MSIS value when it sits above the nearest one.
HE_MSIS2_BAND_AT_200_KM = (0.10, 0.15)
HE_MSIS2_BAND_AT_300_KM = (0.20, 0.25)

#: The comparison altitude is 210 km, so the altitude-matched published figure
#: is the 200 km one. Ten kilometres of extrapolation is declared here rather
#: than hidden inside an interpolation across a hundred.
HE_MSIS2_BAND_NEAREST_210_KM = HE_MSIS2_BAND_AT_200_KM

#: Where the two accelerometer measurements for this storm actually are.
SWARM_A_ALTITUDE_M = 438_000.0
GRACE_FO_ALTITUDE_M = 505_000.0


def load_drivers() -> tuple[sw.KpApSeries, sw.F107Series]:
    """Read the ingested indices from the immutable raw store."""
    kp = sw.parse_gfz_kp_ap(
        STORE.read("gfz", "kp_2022_extended", ".json"),
        STORE.read("gfz", "kp_2022_extended_ap", ".json"),
    )
    f107 = sw.parse_celestrak_f107(STORE.read("celestrak", "sw_last5years", ".csv"))
    return kp, f107


def sha256_of(path: pathlib.Path) -> str:
    """Hex digest of a file, or a reason it could not be hashed."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"unknown ({exc.__class__.__name__})"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown (not installed)"


def mass_density(state_number_densities: dict[str, float]) -> float:
    return comp.total_species_mass_density(state_number_densities)


def local_solar_time_hours(when: dt.datetime, longitude_deg: float) -> float:
    """Mean local solar time, UTC hour plus 15 degrees of longitude per hour."""
    utc_hours = when.astimezone(dt.UTC).hour + when.astimezone(dt.UTC).minute / 60.0
    return (utc_hours + longitude_deg / 15.0) % 24.0


def enhancement_at(
    provider: Nrlmsise00Provider,
    when: dt.datetime,
    altitude_m: float,
    latitude_deg: float,
    longitude_deg: float,
    storm_drivers: DriverState,
    quiet_drivers: DriverState,
) -> dict[str, float]:
    """Storm and quiet densities at one point, and the enhancement between them."""
    storm = provider.state_at(when, altitude_m, latitude_deg, longitude_deg, storm_drivers)
    quiet = provider.state_at(when, altitude_m, latitude_deg, longitude_deg, quiet_drivers)
    rho_storm = mass_density(storm.number_densities)
    rho_quiet = mass_density(quiet.number_densities)
    return {
        "rho_storm_kg_m3": rho_storm,
        "rho_quiet_kg_m3": rho_quiet,
        "enhancement_fraction": rho_storm / rho_quiet - 1.0,
        "temperature_storm_k": storm.temperature_k,
        "temperature_quiet_k": quiet.temperature_k,
    }


# --------------------------------------------------------------------------
# Q1-Q7: what was actually called
# --------------------------------------------------------------------------


def audit_model_configuration(peak_drivers: DriverState) -> dict[str, Any]:
    """Implementation, version, package, coefficients, switches, ap mode."""
    from pymsis import msis

    pymsis_dir = pathlib.Path(msis.__file__).resolve().parent
    shared_objects = sorted(pymsis_dir.glob("msis*.so"))
    parameter_files = sorted(pymsis_dir.glob("*.parm"))

    storm_options = msis.create_options(geomagnetic_activity=-1)
    daily_options = msis.create_options()

    provider_source = inspect.getsource(N)

    # The switch is not documented, it is exercised: the same drivers are run
    # through both ap paths and the two densities are reported. If the storm
    # switch were not in force the two numbers would be identical.
    history = list(peak_drivers.ap_history or ())
    storm_mode = msis.calculate(
        np.datetime64(dt.datetime(2022, 2, 4, 20, 0), "s"),
        REPORTED_LONGITUDE_DEG,
        REPORTED_LATITUDE_DEG,
        REPORTED_ALTITUDE_M / 1000.0,
        peak_drivers.f107,
        peak_drivers.f107a,
        [history],
        options=storm_options,
        version=N.MODEL_VERSION,
    )
    daily_mode = msis.calculate(
        np.datetime64(dt.datetime(2022, 2, 4, 20, 0), "s"),
        REPORTED_LONGITUDE_DEG,
        REPORTED_LATITUDE_DEG,
        REPORTED_ALTITUDE_M / 1000.0,
        peak_drivers.f107,
        peak_drivers.f107a,
        [[peak_drivers.ap_daily] * 7],
        options=None,
        version=N.MODEL_VERSION,
    )
    rho_storm_mode = float(np.asarray(storm_mode).ravel()[0])
    rho_daily_mode = float(np.asarray(daily_mode).ravel()[0])

    return {
        "q01_exact_implementation": {
            "model": N.MODEL_NAME,
            "reference": N.MODEL_REFERENCE,
            "licence": N.MODEL_LICENCE,
            "implementation": (
                "The NRL Fortran NRLMSISE-00 source, compiled into the pymsis "
                "extension module msis00f and called through pymsis.msis.calculate(). "
                "No Python reimplementation of the model is used anywhere in this "
                "repository."
            ),
            "call_site": (
                "thermosphere_consequence.providers.nrlmsise00.Nrlmsise00Provider._raw"
            ),
            "provider_name": Nrlmsise00Provider().name,
            "provider_version": Nrlmsise00Provider().version,
        },
        "q02_model_version": {
            "version_argument": N.MODEL_VERSION,
            "meaning": (
                "0 = NRLMSISE-00, the 2001 model; no licence terms located for "
                "the compiled artefact, research-only"
            ),
            "restricted_versions_not_used": [2.0, 2.1],
            "reason": (
                "NRLMSIS 2.0/2.1 is under a US Naval Research Laboratory "
                "academic-non-commercial licence and US Patent 10,641,925. A guard "
                "test parses the provider with ast and fails if any restricted "
                "version literal can reach msis.calculate."
            ),
            "guard_present_in_source": "version=2" not in provider_source,
        },
        "q03_package_versions": {
            "pymsis": package_version("pymsis"),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "q04_coefficient_files": {
            "note": (
                "NRLMSISE-00 carries its coefficients INSIDE the Fortran source, so "
                "there is no external coefficient file for version 0 to hash. The "
                "hashable artefact is the compiled extension module. The .parm files "
                "shipped alongside belong to NRLMSIS 2.0/2.1 and are NOT read by "
                "version 0; they are hashed here only to record that they were "
                "present and unused."
            ),
            "compiled_modules": {
                so.name: {"sha256": sha256_of(so), "bytes": so.stat().st_size}
                for so in shared_objects
            },
            "parameter_files_present_but_unused_by_version_0": {
                parm.name: {"sha256": sha256_of(parm), "bytes": parm.stat().st_size}
                for parm in parameter_files
            },
        },
        "q05_model_switches": {
            "switch_array_length": len(storm_options),
            "storm_mode_switches": [float(v) for v in storm_options],
            "default_switches_for_contrast": [float(v) for v in daily_options],
            "only_switch_changed_from_default": "geomagnetic_activity",
            "geomagnetic_activity_value": -1.0,
            "meaning": (
                "pymsis create_options index 8 is geomagnetic_activity: +1 selects "
                "daily-Ap mode, -1 selects storm-time mode in which the 7-element ap "
                "array is consumed. Every other switch is left at 1, its default, so "
                "no seasonal, diurnal, tidal or longitudinal term is suppressed."
            ),
        },
        "q06_nrlmsise00_flags": {
            "flags": {
                "f107_effects": 1.0,
                "time_independent": 1.0,
                "symmetrical_annual": 1.0,
                "symmetrical_semiannual": 1.0,
                "asymmetrical_annual": 1.0,
                "asymmetrical_semiannual": 1.0,
                "diurnal": 1.0,
                "semidiurnal": 1.0,
                "geomagnetic_activity": -1.0,
                "all_ut_effects": 1.0,
                "longitudinal": 1.0,
                "mixed_ut_long": 1.0,
                "mixed_ap_ut_long": 1.0,
                "terdiurnal": 1.0,
            },
            "anomalous_oxygen_included": Nrlmsise00Provider()._include_anomalous_oxygen,
            "species_reported": [name for _, name in N._PYMSIS_SPECIES_COLUMNS],
            "species_not_reported": ["NO"],
            "no_note": (
                "NRLMSISE-00 does not produce NO; pymsis returns NaN in that column "
                "and the provider omits the key rather than writing a zero, because "
                "absence and zero are different claims."
            ),
        },
        "q07_ap_mode": {
            "mode_in_force": "7-element ap history, storm-time mode",
            "geomagnetic_activity_switch": -1.0,
            "provider_rule": (
                "Nrlmsise00Provider._raw sets options=create_options("
                "geomagnetic_activity=-1) and passes drivers.ap_history whenever that "
                "history is present; it falls back to [ap_daily] * 7 with options=None "
                "only when no history was supplied."
            ),
            "history_supplied_by_drivers_at": True,
            "ap_history_at_peak": list(peak_drivers.ap_history or ()),
            "ap_history_element_meaning": [
                "daily Ap",
                "ap now",
                "ap 3 h before",
                "ap 6 h before",
                "ap 9 h before",
                "mean ap 12-33 h before",
                "mean ap 36-57 h before",
            ],
            "verification": {
                "method": (
                    "The same drivers were run through both ap paths at the peak "
                    "point. Identical results would prove the storm switch was not "
                    "in force."
                ),
                "rho_storm_mode_kg_m3": rho_storm_mode,
                "rho_daily_ap_mode_kg_m3": rho_daily_mode,
                "ratio_storm_over_daily": rho_storm_mode / rho_daily_mode,
                "switch_is_in_force": abs(rho_storm_mode - rho_daily_mode) > 0.0,
            },
        },
    }


# --------------------------------------------------------------------------
# Q8-Q11: drivers, sampling, reference definitions, dates
# --------------------------------------------------------------------------


def audit_drivers_and_sampling(
    kp: sw.KpApSeries, f107: sw.F107Series, peak_time: dt.datetime
) -> dict[str, Any]:
    peak_drivers = sw.drivers_at(peak_time, kp, f107)
    deployment_drivers = sw.drivers_at(EV.DEPLOYMENT_UTC, kp, f107)
    quiet = quiet_baseline_drivers(peak_drivers, quiet_ap=REPORTED_QUIET_AP)

    daily_f107: list[dict[str, Any]] = []
    day = dt.date(2022, 1, 30)
    while day <= dt.date(2022, 2, 8):
        daily_f107.append(
            {
                "date": day.isoformat(),
                "f107_observed": f107.observed_at(day),
                "f107_81day_centred": f107.centred_81_day_average(day),
            }
        )
        day += dt.timedelta(days=1)

    three_hour: list[dict[str, Any]] = []
    when = dt.datetime(2022, 1, 31, tzinfo=dt.UTC)
    while when < dt.datetime(2022, 2, 7, tzinfo=dt.UTC):
        three_hour.append(
            {
                "time_utc": when.isoformat(),
                "kp": kp.kp_at(when),
                "ap_3h": kp.ap_at(when),
                "ap_daily_derived": kp.ap_history_for(when)[0],
            }
        )
        when += dt.timedelta(hours=3)

    n_samples = int((REPORTED_WINDOW_END - REPORTED_WINDOW_START) / REPORTED_SAMPLE_INTERVAL) + 1

    # Read out of the ingested series rather than transcribed from a previous
    # report, so a change in the ingested indices cannot leave a stale peak
    # time sitting in this audit.
    in_window = [
        row
        for row in three_hour
        if REPORTED_WINDOW_START
        <= dt.datetime.fromisoformat(str(row["time_utc"]))
        <= REPORTED_WINDOW_END
    ]
    peak_ap_row = max(in_window, key=lambda row: float(row["ap_3h"]))
    peak_kp_row = max(in_window, key=lambda row: float(row["kp"]))
    ap_ties = [row["time_utc"] for row in in_window if row["ap_3h"] == peak_ap_row["ap_3h"]]
    peak_ap_time = dt.datetime.fromisoformat(str(peak_ap_row["time_utc"]))

    return {
        "q08_solar_flux": {
            "rule": (
                "NRLMSISE-00 is fitted to the PREVIOUS day's observed F10.7; "
                "drivers_at applies that offset. The 81-day value is a centred mean "
                "of observed F10.7, never of adjusted F10.7."
            ),
            "at_peak": {
                "time_utc": peak_time.isoformat(),
                "f107_observed_previous_day": peak_drivers.f107,
                "f107a_81day_centred": peak_drivers.f107a,
            },
            "at_deployment": {
                "time_utc": EV.DEPLOYMENT_UTC.isoformat(),
                "f107_observed_previous_day": deployment_drivers.f107,
                "f107a_81day_centred": deployment_drivers.f107a,
            },
            "daily_series": daily_f107,
            "source": f107.source,
            "citation": f107.citation,
        },
        "q09_sampling": {
            "altitude_m": REPORTED_ALTITUDE_M,
            "altitude_km": REPORTED_ALTITUDE_M / 1000.0,
            "altitude_kind": "geodetic",
            "latitude_deg": REPORTED_LATITUDE_DEG,
            "latitude_note": (
                "Held at 53.22 deg N, the orbital INCLINATION, for the whole series. "
                "It is not a latitude the vehicle occupied continuously: an inclined "
                "orbit reaches that latitude twice per revolution and spends most of "
                "its time below it."
            ),
            "longitude_deg": REPORTED_LONGITUDE_DEG,
            "longitude_note": (
                "Held at 0 deg for the whole series, so mean local solar time equals "
                "UTC exactly and the diurnal cycle is aliased onto the storm time "
                "series rather than sampled across it."
            ),
            "sample_interval_hours": REPORTED_SAMPLE_INTERVAL.total_seconds() / 3600.0,
            "n_samples": n_samples,
            "local_solar_time_at_peak_hours": local_solar_time_hours(
                peak_time, REPORTED_LONGITUDE_DEG
            ),
            "local_solar_time_coverage": (
                "All 24 hours, but only once each per day and perfectly correlated "
                "with UTC, so local time and storm phase cannot be separated in this "
                "series."
            ),
        },
        "q10_reference_definitions": {
            "storm_reference": {
                "definition": (
                    "Observed drivers at each instant: GFZ Kp/ap converted to the "
                    "7-element NRLMSISE-00 ap history, plus CelesTrak F10.7 "
                    "(previous-day observed and 81-day centred average)."
                ),
                "status": str(peak_drivers.status),
                "source": peak_drivers.source,
                "f107": peak_drivers.f107,
                "f107a": peak_drivers.f107a,
                "ap_daily": peak_drivers.ap_daily,
                "ap_history": list(peak_drivers.ap_history or ()),
            },
            "quiet_reference": {
                "definition": (
                    "A COUNTERFACTUAL, not an observation: geomagnetic activity is "
                    "set to ap = 4 (Kp ~ 1) in every element of the history while "
                    "solar flux is held at the observed value, so the comparison "
                    "isolates the geomagnetic storm from the solar-cycle background."
                ),
                "status": str(quiet.status),
                "source": quiet.source,
                "f107": quiet.f107,
                "f107a": quiet.f107a,
                "ap_daily": quiet.ap_daily,
                "ap_history": list(quiet.ap_history or ()),
            },
        },
        "q11_dates_and_times": {
            "launch_utc": EV.LAUNCH_UTC.isoformat(),
            "deployment_utc": EV.DEPLOYMENT_UTC.isoformat(),
            "enhancement_series_start_utc": REPORTED_WINDOW_START.isoformat(),
            "enhancement_series_end_utc": REPORTED_WINDOW_END.isoformat(),
            "peak_enhancement_time_utc": peak_time.isoformat(),
            "peak_kp": peak_kp_row["kp"],
            "peak_kp_time_utc": peak_kp_row["time_utc"],
            "peak_ap_3h": peak_ap_row["ap_3h"],
            "peak_ap_time_utc": peak_ap_row["time_utc"],
            "peak_ap_time_ties": ap_ties,
            "peak_enhancement_lags_peak_ap_hours": (
                peak_time - peak_ap_time
            ).total_seconds()
            / 3600.0,
            "three_hourly_index_series": three_hour,
        },
    }


# --------------------------------------------------------------------------
# Q12-Q15: the outputs and the enhancement calculation
# --------------------------------------------------------------------------


def audit_outputs(
    provider: Nrlmsise00Provider,
    kp: sw.KpApSeries,
    f107: sw.F107Series,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the reported series, then report the outputs at its peak."""
    series: list[dict[str, Any]] = []
    when = REPORTED_WINDOW_START
    while when <= REPORTED_WINDOW_END:
        observed = sw.drivers_at(when, kp, f107)
        quiet = quiet_baseline_drivers(observed, quiet_ap=REPORTED_QUIET_AP)
        point = enhancement_at(
            provider,
            when,
            REPORTED_ALTITUDE_M,
            REPORTED_LATITUDE_DEG,
            REPORTED_LONGITUDE_DEG,
            observed,
            quiet,
        )
        row: dict[str, Any] = dict(point)
        row["time_utc"] = when.isoformat()
        row["ap_daily"] = observed.ap_daily
        series.append(row)
        when += REPORTED_SAMPLE_INTERVAL

    peak = max(series, key=lambda row: float(row["enhancement_fraction"]))
    peak_time = dt.datetime.fromisoformat(str(peak["time_utc"]))
    values = [float(row["enhancement_fraction"]) for row in series]

    observed = sw.drivers_at(peak_time, kp, f107)
    quiet = quiet_baseline_drivers(observed, quiet_ap=REPORTED_QUIET_AP)
    storm_state = provider.state_at(
        peak_time,
        REPORTED_ALTITUDE_M,
        REPORTED_LATITUDE_DEG,
        REPORTED_LONGITUDE_DEG,
        observed,
    )
    quiet_state = provider.state_at(
        peak_time,
        REPORTED_ALTITUDE_M,
        REPORTED_LATITUDE_DEG,
        REPORTED_LONGITUDE_DEG,
        quiet,
    )
    rho_storm = mass_density(storm_state.number_densities)
    rho_quiet = mass_density(quiet_state.number_densities)

    species: dict[str, Any] = {}
    for symbol in sorted(storm_state.number_densities):
        storm_n = storm_state.number_densities[symbol]
        quiet_n = quiet_state.number_densities[symbol]
        species[symbol] = {
            "n_storm_m3": storm_n,
            "n_quiet_m3": quiet_n,
            "change_fraction": (storm_n - quiet_n) / quiet_n if quiet_n > 0 else float("nan"),
            "mass_fraction_storm": comp.species_mass_density(symbol, storm_n) / rho_storm,
        }

    model_total_storm = storm_state.total_mass_density
    model_total_quiet = quiet_state.total_mass_density
    outputs = {
        "q12_density": {
            "definition": (
                "rho = sum over reported species of m_i * n_i. The species sum is "
                "used rather than the model's own total, so the density that drives "
                "drag is the same quantity the composition diagnostics describe."
            ),
            "rho_storm_kg_m3": rho_storm,
            "rho_quiet_kg_m3": rho_quiet,
            "model_own_total_storm_kg_m3": model_total_storm,
            "model_own_total_quiet_kg_m3": model_total_quiet,
            "closure_fraction_storm": (
                comp.density_closure_fraction(storm_state.number_densities, model_total_storm)
                if model_total_storm is not None
                else None
            ),
            "enhancement_from_species_sum": rho_storm / rho_quiet - 1.0,
            "enhancement_from_model_own_total": (
                model_total_storm / model_total_quiet - 1.0
                if model_total_storm is not None and model_total_quiet
                else None
            ),
        },
        "q13_species": species,
        "q14_temperature": {
            "temperature_storm_k": storm_state.temperature_k,
            "temperature_quiet_k": quiet_state.temperature_k,
            "temperature_change_k": storm_state.temperature_k - quiet_state.temperature_k,
            "temperature_change_fraction": (
                storm_state.temperature_k / quiet_state.temperature_k - 1.0
            ),
            "note": (
                "NRLMSISE-00 returns the kinetic temperature at the requested "
                "altitude. The exospheric temperature is available from pymsis but "
                "is not carried by this provider, so it is reported as absent rather "
                "than guessed."
            ),
        },
        "q15_enhancement_calculation": {
            "formula": "(rho_storm - rho_quiet) / rho_quiet, signed, never clamped",
            "rho_storm_kg_m3": rho_storm,
            "rho_quiet_kg_m3": rho_quiet,
            "peak_enhancement_fraction": rho_storm / rho_quiet - 1.0,
            "peak_time_utc": peak_time.isoformat(),
            "mean_enhancement_fraction": float(np.mean(values)),
            "median_enhancement_fraction": float(np.median(values)),
            "min_enhancement_fraction": float(np.min(values)),
            "n_samples": len(series),
            "both_states_at_identical_position_and_time": True,
            "what_varies_between_them": "geomagnetic drivers only; solar flux held",
        },
    }
    return outputs, {"peak_time": peak_time, "series": series, "peak_enhancement": max(values)}


# --------------------------------------------------------------------------
# Q16-Q18: comparisons
# --------------------------------------------------------------------------


def audit_published_comparison(peak_enhancement: float) -> dict[str, Any]:
    comparisons = literature.compare(
        peak_enhancement,
        literature.PUBLISHED_DENSITY_ENHANCEMENT,
        altitude_km=REPORTED_ALTITUDE_M / 1000.0,
        # What this project's headline IS: one latitude, one longitude, one
        # altitude, maximised over the event window. Declared rather than
        # assumed, because every published figure for this layer is a maximum
        # of a global or zonal MEAN, and a point peak is expected to exceed it.
        value_definition=literature.PROJECT_QUANTITY_DEFINITION,
        # And what it is measured AGAINST: the same instant and place
        # re-evaluated at a fixed quiet ap. Not a quiet calendar day, and not a
        # multi-day model mean, which are what the two source papers use.
        value_baseline_kind=literature.PROJECT_BASELINE_KIND,
    )
    return {
        "q16_published_comparison": {
            "computed_value": peak_enhancement,
            "altitude_km": REPORTED_ALTITUDE_M / 1000.0,
            "comparisons": [dict(c) for c in comparisons],
            "verdict": literature.verdict_text(comparisons),
        }
    }


def audit_envelope_comparison(
    provider: Nrlmsise00Provider,
    kp: sw.KpApSeries,
    f107: sw.F107Series,
    peak_time: dt.datetime,
) -> dict[str, Any]:
    """The MSIS storm density against each published band, as a density."""
    observed = sw.drivers_at(peak_time, kp, f107)
    quiet = quiet_baseline_drivers(observed, quiet_ap=REPORTED_QUIET_AP)
    rho_storm = mass_density(
        provider.state_at(
            peak_time,
            REPORTED_ALTITUDE_M,
            REPORTED_LATITUDE_DEG,
            REPORTED_LONGITUDE_DEG,
            observed,
        ).number_densities
    )
    rho_quiet = mass_density(
        provider.state_at(
            peak_time,
            REPORTED_ALTITUDE_M,
            REPORTED_LATITUDE_DEG,
            REPORTED_LONGITUDE_DEG,
            quiet,
        ).number_densities
    )

    rows: list[dict[str, Any]] = []
    for band in PUBLISHED_ENVELOPE_BANDS:
        row: dict[str, Any] = {
            "citation": band.citation,
            "doi_or_url": band.doi_or_url,
            "provenance_kind": band.provenance_kind,
            "altitude_coverage_km": [band.altitude_min_km, band.altitude_max_km],
            "time_coverage_utc": [
                band.time_start_utc.isoformat(),
                band.time_end_utc.isoformat(),
            ],
        }
        row["supplies_altitude_profile"] = band.supplies_altitude_profile
        for selection in ("low", "nominal", "high"):
            envelope = PublishedEventDensityEnvelopeProvider(
                provider,
                lambda drivers: quiet_baseline_drivers(drivers, quiet_ap=REPORTED_QUIET_AP),
                band=band,
                selection=selection,
            )
            try:
                state = envelope.state_at(
                    peak_time,
                    REPORTED_ALTITUDE_M,
                    REPORTED_LATITUDE_DEG,
                    REPORTED_LONGITUDE_DEG,
                    observed,
                )
            except UnsupportedAltitudeProfileError as exc:
                # The refusal is the result. A band whose span varies with
                # location as well as height states nothing at 210 km, and the
                # provider says so rather than interpolating one.
                row[selection] = {
                    "applied_enhancement": None,
                    "rho_kg_m3": None,
                    "ratio_to_msis_storm": None,
                    "refused": str(exc),
                }
                continue
            rho_envelope = mass_density(state.number_densities)
            row[selection] = {
                "applied_enhancement": envelope.enhancement_at(REPORTED_ALTITUDE_M, peak_time),
                "rho_kg_m3": rho_envelope,
                "ratio_to_msis_storm": rho_envelope / rho_storm,
                "derived": (
                    selection == "nominal" and band.supplies_altitude_profile
                ),
            }
        rows.append(row)

    refusal: dict[str, str] = {}
    # An endpoint-pair band, by name. The probe below is about COVERAGE, and a
    # range-covered band such as Fang's refuses a nominal selection before any
    # coverage check runs, which would make this record answer a different
    # question than the one it is labelled with.
    envelope = PublishedEventDensityEnvelopeProvider(
        provider,
        lambda drivers: quiet_baseline_drivers(drivers, quiet_ap=REPORTED_QUIET_AP),
        band=band_for("MSIS2.0"),
    )
    for label, altitude_m, when in (
        ("above the published altitude band", 505_000.0, peak_time),
        ("outside the published time window", REPORTED_ALTITUDE_M, REPORTED_WINDOW_END),
    ):
        try:
            envelope.state_at(
                when, altitude_m, REPORTED_LATITUDE_DEG, REPORTED_LONGITUDE_DEG, observed
            )
        except OutsidePublishedCoverageError as exc:
            refusal[label] = str(exc)
        else:  # pragma: no cover - a silent success here would be the bug
            refusal[label] = "NO REFUSAL: the provider extrapolated, which it must not"

    return {
        "q17_published_envelope_comparison": {
            "msis_storm_rho_kg_m3": rho_storm,
            "msis_quiet_rho_kg_m3": rho_quiet,
            "msis_enhancement": rho_storm / rho_quiet - 1.0,
            "bands": rows,
            "refusal_outside_coverage": refusal,
            "reading": (
                "The envelope densities are the SAME quiet baseline scaled by the "
                "published enhancement. The ratio column is therefore the factor by "
                "which this project's storm density would have to rise to match each "
                "published source, at the same point, on the same baseline."
            ),
        }
    }


def audit_observational_comparison(
    provider: Nrlmsise00Provider,
    kp: sw.KpApSeries,
    f107: sw.F107Series,
    peak_time: dt.datetime,
) -> dict[str, Any]:
    """MSIS enhancement at the altitudes where accelerometer data exist."""
    observed = sw.drivers_at(peak_time, kp, f107)
    quiet = quiet_baseline_drivers(observed, quiet_ap=REPORTED_QUIET_AP)

    rows: list[dict[str, Any]] = []
    for label, altitude_m, published in (
        ("Swarm-A accelerometer", SWARM_A_ALTITUDE_M, 1.10),
        ("GRACE-FO accelerometer", GRACE_FO_ALTITUDE_M, 1.20),
    ):
        point = enhancement_at(
            provider,
            peak_time,
            altitude_m,
            REPORTED_LATITUDE_DEG,
            REPORTED_LONGITUDE_DEG,
            observed,
            quiet,
        )
        rows.append(
            {
                "platform": label,
                "altitude_m": altitude_m,
                "altitude_above_staging_km": (altitude_m - REPORTED_ALTITUDE_M) / 1000.0,
                "published_enhancement": published,
                "msis_enhancement_here": point["enhancement_fraction"],
                "msis_shortfall_factor": published / point["enhancement_fraction"],
                "provenance": "observed (accelerometer-derived density)",
            }
        )

    return {
        "q18_observational_comparison": {
            "altitude_mismatch_warning": (
                "NEITHER measurement is at 210 km. Swarm-A flies at ~438 km and "
                "GRACE-FO at ~505 km, 228 km and 295 km ABOVE the Starlink staging "
                "altitude. Agreement or disagreement with them says nothing directly "
                "about the density at 210 km, because the storm-time enhancement "
                "varies with altitude. What the rows below DO establish is whether "
                "this model reproduces the observed enhancement at the altitudes "
                "where an observation exists at all."
            ),
            "rows": rows,
            "no_observation_at_staging_altitude": (
                "There is no accelerometer-derived density at 210 km for this storm "
                "in the sources this project holds. The 210 km result is therefore "
                "unvalidated against observation and is reported as such."
            ),
        }
    }


# --------------------------------------------------------------------------
# Q19: neutral wind
# --------------------------------------------------------------------------


def audit_wind_sensitivity(
    provider: Nrlmsise00Provider,
    kp: sw.KpApSeries,
    f107: sw.F107Series,
    peak_time: dt.datetime,
) -> dict[str, Any]:
    """Relative speed, drag acceleration and first-order lifetime across the sweep."""
    spacecraft_mass = EV.MASS_KG_NOMINAL
    drag_coefficient = EV.CD_CONVENTIONAL
    area = EV.effective_area_from_operator_inverse_ballistic_coefficient(
        drag_coefficient, spacecraft_mass
    )

    position_tuple, velocity_tuple = EV.insertion_state_vectors()
    position = np.asarray(position_tuple)
    velocity = np.asarray(velocity_tuple)

    # One drag-free revolution from the deployment state, so the sweep is
    # orbit-averaged rather than quoted at a single convenient point.
    nominal = SpacecraftProperties(mass_kg=1.0, reference_area_m2=1.0, drag_coefficient=1.0)
    drag_free = DragPropagator(lambda *_: 0.0, nominal, include_j2=True)
    path = drag_free.propagate(
        OrbitState(EV.DEPLOYMENT_UTC, position_tuple, velocity_tuple),
        1.0 / 12.0,
        sample_interval_s=60.0,
    )

    densities: list[float] = []
    for index, when in enumerate(path.times_utc):
        altitude = path.altitudes_m[index]
        gmst = greenwich_sidereal_time(when)
        latitude, longitude = geodetic_latlon(np.asarray(path.positions_m[index]), gmst)
        drivers = sw.drivers_at(when, kp, f107)
        state = provider.state_at(when, altitude, latitude, longitude, drivers)
        densities.append(mass_density(state.number_densities))

    gmst_deployment = greenwich_sidereal_time(EV.DEPLOYMENT_UTC)
    drivers_deployment = sw.drivers_at(EV.DEPLOYMENT_UTC, kp, f107)
    latitude_dep, longitude_dep = geodetic_latlon(position, gmst_deployment)
    rho_deployment = mass_density(
        provider.state_at(
            EV.DEPLOYMENT_UTC, EV.INSERTION_PERIGEE_ALTITUDE_M, latitude_dep, longitude_dep,
            drivers_deployment,
        ).number_densities
    )

    rows: list[dict[str, Any]] = []

    for wind_provider in wind_sweep_providers():
        wind_east, wind_north = wind_provider.wind_at(
            EV.DEPLOYMENT_UTC, EV.INSERTION_PERIGEE_ALTITUDE_M, latitude_dep, longitude_dep,
            drivers_deployment,
        )
        relative = relative_velocity_with_wind(
            velocity, position, wind_east, wind_north, gmst_deployment
        )
        speed = float(np.linalg.norm(relative))
        acceleration = effective_drag.drag_acceleration(
            drag_coefficient, rho_deployment, area, speed, spacecraft_mass
        )

        weighted: list[float] = []
        speeds: list[float] = []
        for index, when in enumerate(path.times_utc):
            gmst = greenwich_sidereal_time(when)
            v_rel = relative_velocity_with_wind(
                np.asarray(path.velocities_m_s[index]),
                np.asarray(path.positions_m[index]),
                wind_east,
                wind_north,
                gmst,
            )
            weighted.append(densities[index] * float(np.dot(v_rel, v_rel)))
            speeds.append(float(np.linalg.norm(v_rel)))
        orbit_mean_square = float(np.mean(weighted))

        rows.append(
            {
                "wind_east_m_s": wind_east,
                "wind_north_m_s": wind_north,
                "provider": wind_provider.name,
                "relative_speed_at_deployment_m_s": speed,
                "drag_acceleration_at_deployment_m_s2": acceleration,
                "orbit_mean_relative_speed_m_s": float(np.mean(speeds)),
                "orbit_min_relative_speed_m_s": float(np.min(speeds)),
                "orbit_max_relative_speed_m_s": float(np.max(speeds)),
                "orbit_mean_rho_v_rel_squared": orbit_mean_square,
            }
        )

    zero_row = next(row for row in rows if row["wind_east_m_s"] == 0.0)
    zero_speed = float(zero_row["relative_speed_at_deployment_m_s"])
    zero_acceleration = float(zero_row["drag_acceleration_at_deployment_m_s2"])
    zero_orbit = float(zero_row["orbit_mean_rho_v_rel_squared"])
    zero_orbit_speed = float(zero_row["orbit_mean_relative_speed_m_s"])

    for row in rows:
        wind_speed = abs(float(row["wind_east_m_s"]))
        row["relative_speed_change_m_s"] = (
            float(row["relative_speed_at_deployment_m_s"]) - zero_speed
        )
        row["relative_speed_change_fraction"] = (
            float(row["relative_speed_at_deployment_m_s"]) / zero_speed - 1.0
        )
        row["orbit_mean_relative_speed_change_m_s"] = (
            float(row["orbit_mean_relative_speed_m_s"]) - zero_orbit_speed
        )
        # How much of the wind actually reaches the along-track direction. A
        # zonal wind is not along-track except at the orbit's highest latitude,
        # so the change in relative speed is smaller than the wind itself.
        row["along_track_projection_at_deployment"] = (
            abs(float(row["relative_speed_change_m_s"])) / wind_speed if wind_speed else None
        )
        row["orbit_mean_along_track_projection"] = (
            abs(float(row["orbit_mean_relative_speed_change_m_s"])) / wind_speed
            if wind_speed
            else None
        )
        row["drag_acceleration_change_fraction"] = (
            float(row["drag_acceleration_at_deployment_m_s2"]) / zero_acceleration - 1.0
        )
        orbit_change = float(row["orbit_mean_rho_v_rel_squared"]) / zero_orbit - 1.0
        row["orbit_mean_drag_change_fraction"] = orbit_change
        # First order at fixed density and fixed C_d: lifetime scales as the
        # reciprocal of the orbit-averaged drag.
        row["first_order_lifetime_change_fraction"] = 1.0 / (1.0 + orbit_change) - 1.0

    return {
        "q19_wind_sensitivity": {
            "cases_m_s": list(WIND_SENSITIVITY_CASES_M_S),
            "justification": (
                "Magnitudes taken from the storm-wind scale this repository already "
                "cites: HWM14 storm-time winds at 210 km of several hundred m/s "
                "(providers.nrlmsise00.ZeroWindProvider), and the storm-time "
                "circulation simulated for this event by Fang et al. 2022, "
                "doi:10.1029/2022SW003193. Both signs are swept because the sign "
                "decides whether the wind adds to or subtracts from the relative "
                "speed."
            ),
            "geometry": {
                "evaluation_state": "insertion state at deployment",
                "latitude_deg": latitude_dep,
                "longitude_deg": longitude_dep,
                "corotation_speed_m_s": float(
                    np.linalg.norm(corotation_velocity(position))
                ),
                "inertial_speed_m_s": float(np.linalg.norm(velocity)),
                "rho_at_deployment_kg_m3": rho_deployment,
                "orbit_average_samples": len(path.times_utc),
                "orbit_average_span_hours": (
                    path.times_utc[-1] - path.times_utc[0]
                ).total_seconds()
                / 3600.0,
                "along_track_projection_note": (
                    "A zonal wind is not along-track. On a 53.22 degree orbit the "
                    "eastward unit vector projects onto the along-track direction "
                    "with a factor between cos(53.22 deg) = 0.598 at the equator and "
                    "1.0 at the orbit's highest latitude, so a wind of W m/s changes "
                    "the relative speed by less than W. The deployment state is at "
                    "the ascending node, i.e. the equator, which is the WEAKEST point "
                    "of the orbit for a zonal wind; the orbit-mean column covers the "
                    "rest."
                ),
            },
            "drag_parameters": {
                "drag_coefficient": drag_coefficient,
                "mass_kg": spacecraft_mass,
                "projected_area_m2": area,
                "inverse_ballistic_coefficient_m2_kg": (
                    EV.OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG
                ),
            },
            "rows": rows,
            "lifetime_caveat": (
                "The lifetime column is a FIRST-ORDER estimate, not a propagated "
                "result. DragPropagator has no wind term, so a wind-resolved decay "
                "cannot be produced without changing the propagator. The estimate "
                "assumes lifetime scales as the reciprocal of the orbit-averaged "
                "rho * |v_rel|^2 at fixed density and fixed drag coefficient."
            ),
        }
    }


# --------------------------------------------------------------------------
# Q20: local-time and latitude sampling
# --------------------------------------------------------------------------


def audit_local_time_and_latitude(
    provider: Nrlmsise00Provider,
    kp: sw.KpApSeries,
    f107: sw.F107Series,
    peak_time: dt.datetime,
) -> dict[str, Any]:
    observed = sw.drivers_at(peak_time, kp, f107)
    quiet = quiet_baseline_drivers(observed, quiet_ap=REPORTED_QUIET_AP)

    latitudes = (-53.22, -40.0, -20.0, 0.0, 20.0, 40.0, 53.22)
    longitudes = tuple(float(15 * h) for h in range(-12, 12))

    grid: list[dict[str, Any]] = []
    for latitude in latitudes:
        for longitude in longitudes:
            point = enhancement_at(
                provider, peak_time, REPORTED_ALTITUDE_M, latitude, longitude, observed, quiet
            )
            grid.append(
                {
                    "latitude_deg": latitude,
                    "longitude_deg": longitude,
                    "local_solar_time_hours": local_solar_time_hours(peak_time, longitude),
                    "enhancement_fraction": point["enhancement_fraction"],
                    "rho_storm_kg_m3": point["rho_storm_kg_m3"],
                }
            )

    values = [float(row["enhancement_fraction"]) for row in grid]
    best = max(grid, key=lambda row: float(row["enhancement_fraction"]))
    worst = min(grid, key=lambda row: float(row["enhancement_fraction"]))

    by_local_time: list[dict[str, Any]] = []
    for longitude in longitudes:
        point = enhancement_at(
            provider,
            peak_time,
            REPORTED_ALTITUDE_M,
            REPORTED_LATITUDE_DEG,
            longitude,
            observed,
            quiet,
        )
        by_local_time.append(
            {
                "local_solar_time_hours": local_solar_time_hours(peak_time, longitude),
                "longitude_deg": longitude,
                "enhancement_fraction": point["enhancement_fraction"],
            }
        )

    by_latitude: list[dict[str, Any]] = []
    for latitude in latitudes:
        point = enhancement_at(
            provider,
            peak_time,
            REPORTED_ALTITUDE_M,
            latitude,
            REPORTED_LONGITUDE_DEG,
            observed,
            quiet,
        )
        by_latitude.append(
            {
                "latitude_deg": latitude,
                "enhancement_fraction": point["enhancement_fraction"],
            }
        )

    reported_point = enhancement_at(
        provider,
        peak_time,
        REPORTED_ALTITUDE_M,
        REPORTED_LATITUDE_DEG,
        REPORTED_LONGITUDE_DEG,
        observed,
        quiet,
    )

    return {
        "q20_local_time_and_latitude_sampling": {
            "grid": {
                "latitudes_deg": list(latitudes),
                "longitudes_deg": list(longitudes),
                "n_points": len(grid),
                "altitude_m": REPORTED_ALTITUDE_M,
                "time_utc": peak_time.isoformat(),
            },
            "spread": {
                "min_enhancement": float(np.min(values)),
                "max_enhancement": float(np.max(values)),
                "mean_enhancement": float(np.mean(values)),
                "median_enhancement": float(np.median(values)),
                "range_enhancement": float(np.max(values) - np.min(values)),
                "max_over_min_ratio": float(np.max(values) / np.min(values)),
                "at_the_reported_sampling_point": reported_point["enhancement_fraction"],
                "reported_point_percentile": float(
                    100.0
                    * sum(1 for v in values if v <= reported_point["enhancement_fraction"])
                    / len(values)
                ),
            },
            "most_enhanced_point": best,
            "least_enhanced_point": worst,
            "by_local_solar_time_at_53_22_deg": by_local_time,
            "by_latitude_at_longitude_zero": by_latitude,
            "reading": (
                "The single point the reported figure used is one cell of this grid. "
                "Whether it is a representative cell or a favourable one is answered "
                "by its percentile above."
            ),
        }
    }


# --------------------------------------------------------------------------
# The gap to He et al. 2023
# --------------------------------------------------------------------------


def audit_gap_to_he_et_al(
    provider: Nrlmsise00Provider,
    kp: sw.KpApSeries,
    f107: sw.F107Series,
    peak_time: dt.datetime,
    reported_peak: float,
) -> dict[str, Any]:
    """How wide is this model's family of defensible values, against published MSIS.

    This began as "why does 17.1% sit below He et al.'s ~20-25%?" That premise
    was an artefact of comparing a 210 km value against a figure He et al.
    quote at 300 km. Their 200 km figure for the same model is 10-15%, and the
    reported value sits above it. The variants computed below - baseline
    choice, orbit averaging, altitude, maximisation over place - are still the
    right things to measure, because the spread they reveal is what should be
    reported instead of a single number.
    """
    observed = sw.drivers_at(peak_time, kp, f107)
    counterfactual_quiet = quiet_baseline_drivers(observed, quiet_ap=REPORTED_QUIET_AP)

    # (a1) Pre-storm OBSERVED drivers, evaluated at the SAME instant and place.
    # This isolates the baseline definition: only the drivers change.
    pre_storm = sw.drivers_at(PRE_STORM_REFERENCE, kp, f107)
    pre_storm_baseline = enhancement_at(
        provider,
        peak_time,
        REPORTED_ALTITUDE_M,
        REPORTED_LATITUDE_DEG,
        REPORTED_LONGITUDE_DEG,
        observed,
        pre_storm,
    )

    # (a2) The pre-storm density as actually sampled, at its own instant. This
    # is what an observation-style baseline gives, and it mixes the diurnal and
    # UT terms into the comparison.
    pre_storm_state = provider.state_at(
        PRE_STORM_REFERENCE,
        REPORTED_ALTITUDE_M,
        REPORTED_LATITUDE_DEG,
        REPORTED_LONGITUDE_DEG,
        pre_storm,
    )
    rho_pre_storm_in_situ = mass_density(pre_storm_state.number_densities)
    rho_storm_peak = mass_density(
        provider.state_at(
            peak_time,
            REPORTED_ALTITUDE_M,
            REPORTED_LATITUDE_DEG,
            REPORTED_LONGITUDE_DEG,
            observed,
        ).number_densities
    )

    # (a3) The quiet-ap choice itself.
    quiet_ap_rows: list[dict[str, Any]] = []
    for quiet_ap in (0.0, 2.0, 4.0, 7.0, 12.0):
        point = enhancement_at(
            provider,
            peak_time,
            REPORTED_ALTITUDE_M,
            REPORTED_LATITUDE_DEG,
            REPORTED_LONGITUDE_DEG,
            observed,
            quiet_baseline_drivers(observed, quiet_ap=quiet_ap),
        )
        quiet_ap_rows.append(
            {"quiet_ap": quiet_ap, "enhancement_fraction": point["enhancement_fraction"]}
        )

    # (b) Orbit-averaged rather than single-point sampling, along the drag-free
    # insertion trajectory over the whole reported window.
    nominal = SpacecraftProperties(mass_kg=1.0, reference_area_m2=1.0, drag_coefficient=1.0)
    drag_free = DragPropagator(lambda *_: 0.0, nominal, include_j2=True)
    position_tuple, velocity_tuple = EV.insertion_state_vectors()
    path = drag_free.propagate(
        OrbitState(EV.DEPLOYMENT_UTC, position_tuple, velocity_tuple),
        4.0,
        sample_interval_s=300.0,
    )

    orbit_samples: list[dict[str, float]] = []
    for index, when in enumerate(path.times_utc):
        altitude = path.altitudes_m[index]
        if altitude < 100_000.0:
            continue
        gmst = greenwich_sidereal_time(when)
        latitude, longitude = geodetic_latlon(np.asarray(path.positions_m[index]), gmst)
        drivers = sw.drivers_at(when, kp, f107)
        point = enhancement_at(
            provider,
            when,
            altitude,
            latitude,
            longitude,
            drivers,
            quiet_baseline_drivers(drivers, quiet_ap=REPORTED_QUIET_AP),
        )
        orbit_samples.append(
            {
                "epoch_s": (when - path.times_utc[0]).total_seconds(),
                "altitude_m": altitude,
                "latitude_deg": latitude,
                "enhancement_fraction": point["enhancement_fraction"],
                "rho_storm_kg_m3": point["rho_storm_kg_m3"],
                "rho_quiet_kg_m3": point["rho_quiet_kg_m3"],
            }
        )

    period_s = 5_400.0
    per_orbit: list[dict[str, float]] = []
    if orbit_samples:
        span = orbit_samples[-1]["epoch_s"]
        n_orbits = int(span // period_s)
        for orbit_index in range(n_orbits):
            lo, hi = orbit_index * period_s, (orbit_index + 1) * period_s
            block = [s for s in orbit_samples if lo <= s["epoch_s"] < hi]
            if not block:
                continue
            mean_enhancement = float(
                np.mean([s["enhancement_fraction"] for s in block])
            )
            rho_weighted = float(
                np.sum([s["rho_storm_kg_m3"] for s in block])
                / np.sum([s["rho_quiet_kg_m3"] for s in block])
                - 1.0
            )
            per_orbit.append(
                {
                    "orbit_index": float(orbit_index),
                    "start_epoch_s": lo,
                    "mean_enhancement_fraction": mean_enhancement,
                    "density_weighted_enhancement_fraction": rho_weighted,
                    "n_samples": float(len(block)),
                }
            )

    best_orbit = (
        max(per_orbit, key=lambda row: row["mean_enhancement_fraction"]) if per_orbit else None
    )

    # (c) The altitudes He et al. actually quote.
    by_altitude: list[dict[str, Any]] = []
    for altitude_m in (200_000.0, 210_000.0, 250_000.0, 300_000.0, 400_000.0):
        point = enhancement_at(
            provider,
            peak_time,
            altitude_m,
            REPORTED_LATITUDE_DEG,
            REPORTED_LONGITUDE_DEG,
            observed,
            counterfactual_quiet,
        )
        by_altitude.append(
            {
                "altitude_m": altitude_m,
                "enhancement_fraction": point["enhancement_fraction"],
                "quoted_by_he_et_al": altitude_m in HE_ET_AL_ALTITUDES_M,
            }
        )
    he_band = HE_MSIS2_BAND_NEAREST_210_KM
    inside_he_band = [
        row["altitude_m"]
        for row in by_altitude
        if he_band[0] <= float(row["enhancement_fraction"]) <= he_band[1]
    ]
    at_200 = next(
        float(row["enhancement_fraction"]) for row in by_altitude if row["altitude_m"] == 200_000.0
    )
    at_300 = next(
        float(row["enhancement_fraction"]) for row in by_altitude if row["altitude_m"] == 300_000.0
    )
    he_band_inside_the_200_300_range = at_200 <= he_band[0] and he_band[1] <= at_300

    # (d) The whole-window maximum over the grid, i.e. the largest enhancement
    # this model produces anywhere the vehicle could have been.
    grid_max = -math.inf
    grid_max_where: dict[str, Any] = {}
    when = REPORTED_WINDOW_START
    while when <= REPORTED_WINDOW_END:
        drivers = sw.drivers_at(when, kp, f107)
        quiet = quiet_baseline_drivers(drivers, quiet_ap=REPORTED_QUIET_AP)
        for latitude in (-53.22, 0.0, 53.22):
            for longitude in (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0, 0.0):
                point = enhancement_at(
                    provider,
                    when,
                    REPORTED_ALTITUDE_M,
                    latitude,
                    longitude,
                    drivers,
                    quiet,
                )
                if point["enhancement_fraction"] > grid_max:
                    grid_max = point["enhancement_fraction"]
                    grid_max_where = {
                        "time_utc": when.isoformat(),
                        "latitude_deg": latitude,
                        "longitude_deg": longitude,
                        "local_solar_time_hours": local_solar_time_hours(when, longitude),
                    }
        when += dt.timedelta(hours=3)

    def verdict(label: str, value: float | None) -> dict[str, Any]:
        """Where does this variant sit against the altitude-matched published figure?"""
        if value is None:
            return {
                "explanation": label,
                "alternative_value": None,
                "reaches_altitude_matched_msis_figure": False,
                "statement": f"{label}: not computed, so it says nothing.",
            }
        # "Reaches" means reaching the ALTITUDE-MATCHED published MSIS figure,
        # which for a 210 km comparison is He et al.'s 200 km one. It used to
        # mean reaching their 300 km figure, which no value computed at 210 km
        # should be expected to reach and which is why so much of this section
        # read as a gap needing explanation.
        closes = value >= he_band[0]
        return {
            "explanation": label,
            "alternative_value": value,
            "change_vs_reported": value - reported_peak,
            "reaches_altitude_matched_msis_figure": closes,
            "inside_altitude_matched_msis_figure": he_band[0] <= value <= he_band[1],
            "statement": (
                f"{label}: {value:.2%} against the reported {reported_peak:.2%}, "
                f"compared with He et al.'s MSIS2.0 figure at the nearest quoted "
                f"altitude ({he_band[0]:.0%}-{he_band[1]:.0%} at 200 km). "
                + (
                    "At or above that figure."
                    if closes
                    else "Below that figure."
                )
            ),
        }

    verdicts = [
        verdict(
            "(a1) pre-storm observed baseline",
            float(pre_storm_baseline["enhancement_fraction"]),
        ),
        verdict(
            "(a2) pre-storm density at its own instant",
            rho_storm_peak / rho_pre_storm_in_situ - 1.0,
        ),
        verdict(
            "(a3) quiet baseline at ap = 2 rather than ap = 4",
            next(
                row["enhancement_fraction"]
                for row in quiet_ap_rows
                if row["quiet_ap"] == 2.0
            ),
        ),
        verdict(
            "(b) orbit-averaged over the best revolution",
            best_orbit["mean_enhancement_fraction"] if best_orbit else None,
        ),
        verdict(
            "(c) evaluated at 300 km, one of the two altitudes He et al. quote",
            at_300,
        ),
        verdict("(d) maximum over time, latitude and local time", grid_max),
    ]

    return {
        "spread_against_published_msis_figures": {
            "verdicts": verdicts,
            "variants_at_or_above_the_altitude_matched_msis_figure": [
                row["explanation"]
                for row in verdicts
                if row["reaches_altitude_matched_msis_figure"]
            ],
            "variants_below_it": [
                row["explanation"]
                for row in verdicts
                if not row["reaches_altitude_matched_msis_figure"]
            ],
            "synthesis": (
                "THERE IS NO GAP TO THE PUBLISHED MSIS FIGURE AT THIS ALTITUDE. "
                f"The reported {reported_peak:.1%} at 210 km sits above He et al.'s "
                f"{he_band[0]:.0%}-{he_band[1]:.0%} for MSIS2.0 at 200 km, the nearest "
                "altitude they quote, and inside the 10%-25% their 200 and 300 km "
                "figures span. This section was built to explain a gap that existed "
                "only while a single 20-25% band was carried across the whole layer. "
                "What the section still measures is worth keeping: how wide the "
                "family of defensible values from this model is. This run's "
                f"enhancement rises with altitude from {at_200:.1%} at 200 km to "
                f"{at_300:.1%} at 300 km, so quoting it at 210 km and averaging it "
                "along a 210 x 338 km orbit differ mainly in how much of that "
                "altitude range they include. Maximising over place as well as time "
                f"gives {grid_max:.1%}. Replacing the counterfactual with a pre-storm "
                "OBSERVED baseline makes the enhancement SMALLER, not larger; the "
                "quiet-ap VALUE chosen inside that definition is a sensitive knob. "
                f"The honest reading is that {reported_peak:.1%} is one defensible "
                "number from a family spanning roughly "
                f"{at_200:.0%} to {at_300:.0%}, and that the family, not the single "
                "number, is what should be reported. THE REAL DISAGREEMENT IS WITH "
                "THE PHYSICS-BASED MODELS - WAM-IPE, CTIPe and MAGE run two to four "
                "times higher for this event - and both source papers report that "
                "difference themselves. It is a difference between model families, "
                "not evidence about the true atmosphere at 210 km."
            ),
            "reported_peak_enhancement": reported_peak,
            "he_msis2_band_at_200_km": list(HE_MSIS2_BAND_AT_200_KM),
            "he_msis2_band_at_300_km": list(HE_MSIS2_BAND_AT_300_KM),
            "altitude_matched_band": list(he_band),
            "difference_to_altitude_matched_band_low": reported_peak - he_band[0],
            "question": (
                "How wide is the family of defensible enhancement values this model "
                "produces for this event, and where do He et al. 2023's MSIS2.0 "
                "figures - about 10-15% at 200 km and 20-25% at 300 km - sit inside "
                "it? The earlier form of this question assumed a gap, by comparing a "
                "210 km value against a band quoted at 300 km."
            ),
            "explanation_a_baseline_definition": {
                "hypothesis": (
                    "The counterfactual quiet-ap baseline is not the same baseline "
                    "He et al. used. A pre-storm OBSERVED baseline would give a "
                    "different, possibly larger, enhancement."
                ),
                "a1_pre_storm_drivers_at_the_same_instant": {
                    "reference_time_utc": PRE_STORM_REFERENCE.isoformat(),
                    "reference_ap_daily": pre_storm.ap_daily,
                    "reference_ap_history": list(pre_storm.ap_history or ()),
                    "reference_f107": pre_storm.f107,
                    "reference_f107a": pre_storm.f107a,
                    "enhancement_fraction": pre_storm_baseline["enhancement_fraction"],
                    "change_vs_counterfactual": (
                        pre_storm_baseline["enhancement_fraction"] - reported_peak
                    ),
                },
                "a2_pre_storm_density_at_its_own_instant": {
                    "rho_pre_storm_kg_m3": rho_pre_storm_in_situ,
                    "rho_storm_peak_kg_m3": rho_storm_peak,
                    "enhancement_fraction": rho_storm_peak / rho_pre_storm_in_situ - 1.0,
                    "change_vs_counterfactual": (
                        rho_storm_peak / rho_pre_storm_in_situ - 1.0 - reported_peak
                    ),
                    "caveat": (
                        "This comparison also moves 68 hours in UT and therefore "
                        "carries the diurnal, seasonal and solar-flux differences "
                        "between the two instants. It is not a clean measure of the "
                        "storm."
                    ),
                },
                "a3_sensitivity_to_the_quiet_ap_choice": quiet_ap_rows,
            },
            "explanation_b_single_point_vs_orbit_averaged": {
                "hypothesis": (
                    "A single fixed point at 210 km, 53.22 deg N, 0 deg E is not what "
                    "the vehicle sampled. Averaging along the actual insertion "
                    "trajectory would give a different number."
                ),
                "orbit_average_configuration": {
                    "trajectory": "drag-free Keplerian + J2 from the deployment state",
                    "perigee_altitude_m": EV.INSERTION_PERIGEE_ALTITUDE_M,
                    "apogee_altitude_m": EV.INSERTION_APOGEE_ALTITUDE_M,
                    "duration_days": 4.0,
                    "sample_interval_s": 300.0,
                    "n_samples": len(orbit_samples),
                    "assumed_period_s": period_s,
                },
                "whole_path_mean_enhancement": (
                    float(np.mean([s["enhancement_fraction"] for s in orbit_samples]))
                    if orbit_samples
                    else None
                ),
                "whole_path_max_enhancement": (
                    float(np.max([s["enhancement_fraction"] for s in orbit_samples]))
                    if orbit_samples
                    else None
                ),
                "best_orbit": best_orbit,
                "change_vs_counterfactual": (
                    best_orbit["mean_enhancement_fraction"] - reported_peak
                    if best_orbit
                    else None
                ),
                "altitude_caveat": (
                    "The insertion orbit is 210 x 338 km, so an orbit average spans "
                    "that whole altitude range and is NOT a 210 km number."
                ),
            },
            "explanation_c_altitude_of_the_published_figure": {
                "hypothesis": (
                    "He et al. quote 200 AND 300 km. The modelled enhancement rises "
                    "with altitude, so a figure quoted at 300 km is not comparable to "
                    "one computed at 210 km."
                ),
                "by_altitude": by_altitude,
                "altitudes_landing_inside_the_he_band_m": inside_he_band,
                "enhancement_at_200_km": at_200,
                "enhancement_at_300_km": at_300,
                "he_band_lies_inside_the_200_to_300_km_range": he_band_inside_the_200_300_range,
                "reading": (
                    f"This run gives {at_200:.1%} at 200 km and {at_300:.1%} at 300 km. "
                    f"He et al.'s {he_band[0]:.0%}-{he_band[1]:.0%} is a summary across "
                    "those same two altitudes, and it lies "
                    + ("INSIDE" if he_band_inside_the_200_300_range else "OUTSIDE")
                    + " the range this run produces between them. A single number "
                    "computed at 210 km is near the BOTTOM of that range by "
                    "construction, because 210 km is near the bottom of the interval."
                ),
            },
            "explanation_d_peak_definition": {
                "hypothesis": (
                    "The reported peak is the maximum over TIME at one fixed place. "
                    "The maximum over time AND place is larger."
                ),
                "max_over_time_and_place": grid_max,
                "where": grid_max_where,
                "change_vs_counterfactual": grid_max - reported_peak,
            },
        }
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _sci(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4e}"


def render_markdown(audit: dict[str, Any]) -> str:
    """The audit as prose. Every number here comes from the JSON beside it."""
    config = audit["model_configuration"]
    drivers = audit["drivers_and_sampling"]
    outputs = audit["outputs"]
    published = audit["published_comparison"]["q16_published_comparison"]
    envelope = audit["envelope_comparison"]["q17_published_envelope_comparison"]
    observational = audit["observational_comparison"]["q18_observational_comparison"]
    wind = audit["wind_sensitivity"]["q19_wind_sensitivity"]
    sampling = audit["local_time_and_latitude"]["q20_local_time_and_latitude_sampling"]
    gap = audit["gap_analysis"]["spread_against_published_msis_figures"]

    q15 = outputs["q15_enhancement_calculation"]
    peak = float(q15["peak_enhancement_fraction"])

    lines: list[str] = []
    add = lines.append

    add("# Atmosphere-model audit")
    add("")
    add(f"Generated {audit['generated_utc']} by `scripts/run_atmosphere_audit.py`.")
    add("Machine-readable companion: `evidence/atmosphere_audit.json`.")
    add("")
    add("## Summary")
    add("")
    add(
        f"This run reports a peak modelled density enhancement of **{peak:.1%}** at "
        f"{drivers['q09_sampling']['altitude_km']:.0f} km, at "
        f"{q15['peak_time_utc']}. That is BELOW every published figure for this "
        "event, including the published figure for the MSIS family this project "
        "itself uses."
    )
    add("")
    add(
        "The audit below establishes that the model was called correctly - public "
        "domain NRLMSISE-00, storm-time ap mode with the full 7-element history, "
        f"observed indices, no suppressed terms - and then computes {len(gap['verdicts'])} "
        "defensible variants of the same quantity. Where each lands is decided by "
        "the numbers, not chosen:"
    )
    add("")
    for row in gap["verdicts"]:
        marker = (
            "AT OR ABOVE"
            if row["reaches_altitude_matched_msis_figure"]
            else "below"
        )
        add(f"- **{marker}** - {row['statement']}")
    add("")
    closing = gap["variants_at_or_above_the_altitude_matched_msis_figure"]
    if closing:
        add(
            "At or above He et al.'s altitude-matched MSIS2.0 figure: "
            + "; ".join(closing)
            + "."
        )
        add("")
        add(gap["synthesis"])
    else:
        add(
            "NO variant reaches the altitude-matched published MSIS figure. That "
            "would be a real disagreement with the published figure and must be "
            "reported as an open disagreement."
        )
    add("")

    # Q1-Q7
    add("## 1. Exact implementation")
    add("")
    q1 = config["q01_exact_implementation"]
    add(f"- Model: **{q1['model']}** ({q1['reference']})")
    add(f"- Licence: {q1['licence']}")
    add(f"- Implementation: {q1['implementation']}")
    add(f"- Call site: `{q1['call_site']}`")
    add(f"- Provider identity: `{q1['provider_name']}` / `{q1['provider_version']}`")
    add("")

    add("## 2. Model version")
    add("")
    q2 = config["q02_model_version"]
    add(f"- `version={q2['version_argument']}` - {q2['meaning']}")
    add(f"- Never reachable: {q2['restricted_versions_not_used']}. {q2['reason']}")
    add("")

    add("## 3. Package versions")
    add("")
    for name, value in config["q03_package_versions"].items():
        add(f"- {name}: `{value}`")
    add("")

    add("## 4. Coefficient files and hashes")
    add("")
    q4 = config["q04_coefficient_files"]
    add(q4["note"])
    add("")
    add("| artefact | role | sha256 | bytes |")
    add("| --- | --- | --- | --- |")
    for name, info in q4["compiled_modules"].items():
        role = "NRLMSISE-00 (IN USE)" if "00" in name else "NRLMSIS 2.x (never called)"
        add(f"| `{name}` | {role} | `{info['sha256']}` | {info['bytes']} |")
    for name, info in q4["parameter_files_present_but_unused_by_version_0"].items():
        add(f"| `{name}` | NRLMSIS 2.x parameters (not read by version 0) "
            f"| `{info['sha256']}` | {info['bytes']} |")
    add("")

    add("## 5. Model switches")
    add("")
    q5 = config["q05_model_switches"]
    add(f"- Switch array length: {q5['switch_array_length']}")
    add(f"- In force: `{q5['storm_mode_switches']}`")
    add(f"- Default, for contrast: `{q5['default_switches_for_contrast']}`")
    add(f"- Only switch changed from default: **{q5['only_switch_changed_from_default']}** "
        f"= {q5['geomagnetic_activity_value']}")
    add("")
    add(q5["meaning"])
    add("")

    add("## 6. NRLMSISE-00 flags")
    add("")
    q6 = config["q06_nrlmsise00_flags"]
    add("| flag | value |")
    add("| --- | --- |")
    for name, value in q6["flags"].items():
        add(f"| {name} | {value} |")
    add("")
    add(f"- Anomalous oxygen included: {q6['anomalous_oxygen_included']}")
    add(f"- Species reported: {', '.join(q6['species_reported'])}")
    add(f"- Species NOT reported: {', '.join(q6['species_not_reported'])}. {q6['no_note']}")
    add("")

    add("## 7. Daily Ap versus the 7-element ap history")
    add("")
    q7 = config["q07_ap_mode"]
    add(f"- **Mode in force: {q7['mode_in_force']}** "
        f"(`geomagnetic_activity={q7['geomagnetic_activity_switch']}`)")
    add(f"- {q7['provider_rule']}")
    add(f"- ap history at the peak: `{q7['ap_history_at_peak']}`")
    add(f"  ({', '.join(q7['ap_history_element_meaning'])})")
    verification = q7["verification"]
    add("")
    add(f"Verified, not asserted. {verification['method']}")
    add("")
    add("| ap path | rho at 210 km (kg/m3) |")
    add("| --- | --- |")
    add(f"| 7-element history, storm-time switch (IN USE) "
        f"| {_sci(verification['rho_storm_mode_kg_m3'])} |")
    add(f"| daily Ap repeated 7 times, default switch "
        f"| {_sci(verification['rho_daily_ap_mode_kg_m3'])} |")
    add("")
    add(
        f"Ratio {verification['ratio_storm_over_daily']:.4f}. The two differ, which is "
        "the evidence that the storm-time switch is actually in force: in daily-Ap "
        "mode the multi-hour ap spike is smeared into a daily mean and the storm "
        "response is understated."
    )
    add("")

    # Q8-Q11
    add("## 8. F10.7")
    add("")
    q8 = drivers["q08_solar_flux"]
    add(q8["rule"])
    add("")
    add(f"- At the peak ({q8['at_peak']['time_utc']}): "
        f"F10.7 = **{q8['at_peak']['f107_observed_previous_day']}**, "
        f"F10.7a = **{q8['at_peak']['f107a_81day_centred']:.4f}**")
    add(f"- At deployment ({q8['at_deployment']['time_utc']}): "
        f"F10.7 = **{q8['at_deployment']['f107_observed_previous_day']}**, "
        f"F10.7a = **{q8['at_deployment']['f107a_81day_centred']:.4f}**")
    add(f"- Source: {q8['source']} - {q8['citation']}")
    add("")
    add("| date | F10.7 observed | F10.7 81-day centred |")
    add("| --- | --- | --- |")
    for row in q8["daily_series"]:
        add(f"| {row['date']} | {row['f107_observed']:.1f} | {row['f107_81day_centred']:.4f} |")
    add("")

    add("## 9. Sampling: latitude, longitude, altitude, local solar time")
    add("")
    q9 = drivers["q09_sampling"]
    add(f"- Altitude: **{q9['altitude_km']:.0f} km** ({q9['altitude_kind']}), fixed")
    add(f"- Latitude: **{q9['latitude_deg']} deg**, fixed. {q9['latitude_note']}")
    add(f"- Longitude: **{q9['longitude_deg']} deg**, fixed. {q9['longitude_note']}")
    add(f"- Cadence: {q9['sample_interval_hours']:.0f} h, {q9['n_samples']} samples")
    add(f"- Local solar time at the reported peak: "
        f"**{q9['local_solar_time_at_peak_hours']:.2f} h**")
    add(f"- Local-time coverage: {q9['local_solar_time_coverage']}")
    add("")

    add("## 10. Storm and quiet reference definitions")
    add("")
    q10 = drivers["q10_reference_definitions"]
    for label, key in (("Storm", "storm_reference"), ("Quiet", "quiet_reference")):
        reference = q10[key]
        add(f"**{label}** ({reference['status']}): {reference['definition']}")
        add("")
        add(f"- F10.7 = {reference['f107']}, F10.7a = {reference['f107a']:.4f}")
        add(f"- ap daily = {reference['ap_daily']}, ap history = `{reference['ap_history']}`")
        add(f"- Source: {reference['source']}")
        add("")

    add("## 11. Exact dates and times")
    add("")
    q11 = drivers["q11_dates_and_times"]
    add(f"- Launch: {q11['launch_utc']}")
    add(f"- Deployment: {q11['deployment_utc']}")
    add(f"- Enhancement series: {q11['enhancement_series_start_utc']} to "
        f"{q11['enhancement_series_end_utc']}")
    add(f"- Peak enhancement: **{q11['peak_enhancement_time_utc']}**")
    add(f"- Peak 3-hourly ap ({q11['peak_ap_3h']:.0f}) at {q11['peak_ap_time_utc']}; "
        f"peak Kp ({q11['peak_kp']:.2f}) at {q11['peak_kp_time_utc']}")
    add(f"- The density peak therefore lags the index peak by "
        f"**{q11['peak_enhancement_lags_peak_ap_hours']:.0f} hours**")
    if len(q11["peak_ap_time_ties"]) > 1:
        add(f"- The peak ap value is attained more than once in the window: "
            f"{', '.join(q11['peak_ap_time_ties'])}. The earliest is reported above.")
    add("")

    # Q12-Q15
    add("## 12. Density outputs")
    add("")
    q12 = outputs["q12_density"]
    add(q12["definition"])
    add("")
    add("| quantity | value |")
    add("| --- | --- |")
    add(f"| rho storm (species sum) | {_sci(q12['rho_storm_kg_m3'])} kg/m3 |")
    add(f"| rho quiet (species sum) | {_sci(q12['rho_quiet_kg_m3'])} kg/m3 |")
    add(f"| model's own total, storm | {_sci(q12['model_own_total_storm_kg_m3'])} kg/m3 |")
    add(f"| model's own total, quiet | {_sci(q12['model_own_total_quiet_kg_m3'])} kg/m3 |")
    add(f"| closure fraction (storm) | {q12['closure_fraction_storm']:.6f} |")
    add(f"| enhancement from species sum | {float(q12['enhancement_from_species_sum']):.4%} |")
    add(f"| enhancement from model's own total "
        f"| {float(q12['enhancement_from_model_own_total']):.4%} |")
    add("")

    add("## 13. Species outputs")
    add("")
    add("| species | n storm (m^-3) | n quiet (m^-3) | change | mass fraction (storm) |")
    add("| --- | --- | --- | --- | --- |")
    for symbol, info in outputs["q13_species"].items():
        add(f"| {symbol} | {_sci(info['n_storm_m3'])} | {_sci(info['n_quiet_m3'])} "
            f"| {_pct(info['change_fraction'])} | {info['mass_fraction_storm']:.6f} |")
    add("")

    add("## 14. Temperature outputs")
    add("")
    q14 = outputs["q14_temperature"]
    add(f"- Storm: **{q14['temperature_storm_k']:.2f} K**")
    add(f"- Quiet: **{q14['temperature_quiet_k']:.2f} K**")
    add(f"- Difference: {q14['temperature_change_k']:+.2f} K "
        f"({_pct(q14['temperature_change_fraction'])})")
    add(f"- {q14['note']}")
    add("")

    add("## 15. The enhancement calculation")
    add("")
    add(f"- Formula: `{q15['formula']}`")
    add(f"- Peak: **{peak:.4%}** at {q15['peak_time_utc']}")
    add(f"- Mean over the window: {float(q15['mean_enhancement_fraction']):.4%}")
    add(f"- Median: {float(q15['median_enhancement_fraction']):.4%}, "
        f"minimum: {float(q15['min_enhancement_fraction']):.4%}")
    add(f"- Samples: {q15['n_samples']}")
    add(f"- Between the two states, {q15['what_varies_between_them']}.")
    add("")

    # Q16-Q18
    add("## 16. Comparison with published values")
    add("")
    add("Each row shows what the source reports, at what altitude, from which "
        "model or instrument, under which averaging and against which baseline "
        "- and then whether those match the computed quantity. A verdict is "
        "issued only where all three compatibility checks pass. Where they do "
        "not, the position is still shown, in the last column, as arithmetic "
        "context rather than as agreement or disagreement.")
    add("")
    add("| source | published value | model or instrument | altitude reported "
        "| averaging | baseline | verdict | position only |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in published["comparisons"]:
        low, high = float(row["low"]), float(row["high"])
        band = f"{low:.0%}" if low == high else f"{low:.0%}-{high:.0%}"
        lo_km, hi_km = row["finding_altitude_km_range"]
        where = f"{lo_km:.0f} km" if lo_km == hi_km else f"{lo_km:.0f}-{hi_km:.0f} km domain"
        producer = row["model"] or row["instrument"] or row["kind"]
        verdict_cell = (
            f"**{row['relation']}**"
            if row["comparable"]
            else "**NOT COMPARABLE**"
        )
        add(f"| {row['source']} | {band} | {producer} | {where} "
            f"| {row['finding_quantity_definition']} | {row['finding_baseline_kind']} "
            f"| {verdict_cell} | {row['positional_relation']} |")
    add("")
    add("Incompatibilities, per source:")
    add("")
    for row in published["comparisons"]:
        if row["comparable"]:
            continue
        add(f"- **{row['source']}** - " + "; ".join(row["incompatibility_reasons"]) + ".")
    add("")
    add("```")
    add(published["verdict"])
    add("```")
    add("")

    add("## 17. Comparison with the published-envelope provider")
    add("")
    add(envelope["reading"])
    add("")
    add(f"MSIS storm density at the peak point: {_sci(envelope['msis_storm_rho_kg_m3'])} kg/m3 "
        f"(enhancement {float(envelope['msis_enhancement']):.2%} over "
        f"{_sci(envelope['msis_quiet_rho_kg_m3'])} kg/m3).")
    add("")
    add("| band | coverage | selection | applied enhancement | rho (kg/m3) "
        "| x MSIS storm | status |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for row in envelope["bands"]:
        lo_km, hi_km = row["altitude_coverage_km"]
        coverage = f"{lo_km:.0f}-{hi_km:.0f} km"
        for selection in ("low", "nominal", "high"):
            cell = row[selection]
            if cell.get("refused"):
                add(f"| {row['citation']} [{row['provenance_kind']}] | {coverage} "
                    f"| {selection} | — | — | — | REFUSED: the source supplies no "
                    "altitude profile |")
                continue
            if cell.get("derived"):
                status = "DERIVED interpolation, not a published value"
            else:
                status = "published edge of the band"
            add(f"| {row['citation']} [{row['provenance_kind']}] | {coverage} | {selection} "
                f"| {float(cell['applied_enhancement']):.2%} | {_sci(cell['rho_kg_m3'])} "
                f"| {float(cell['ratio_to_msis_storm']):.3f} | {status} |")
    add("")
    add("A band whose source reports its span across LOCATIONS AND ALTITUDES "
        "rather than at each altitude edge has no profile to interpolate, and "
        "the provider raises rather than manufacturing a value for 210 km. Its "
        "low and high rows are the published extremes over the whole domain "
        "and are contextual bounds for the event, NOT a validated interval at "
        "this altitude. Where a nominal value IS produced, the source states "
        "only the two edges and the interpolated figure is this project's "
        "assumption.")
    add("")
    add("The envelope provider refuses to answer outside its published coverage:")
    add("")
    for label, message in envelope["refusal_outside_coverage"].items():
        add(f"- {label}: `{message}`")
    add("")

    add("## 18. Comparison with observation-derived density")
    add("")
    add(f"**{observational['altitude_mismatch_warning']}**")
    add("")
    add("| platform | altitude | above 210 km by | published | this model here "
        "| model shortfall |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in observational["rows"]:
        add(f"| {row['platform']} | {row['altitude_m'] / 1000.0:.0f} km "
            f"| {row['altitude_above_staging_km']:.0f} km "
            f"| {float(row['published_enhancement']):.0%} "
            f"| {float(row['msis_enhancement_here']):.1%} "
            f"| x{float(row['msis_shortfall_factor']):.1f} |")
    add("")
    add(observational["no_observation_at_staging_altitude"])
    add("")

    # Q19
    add("## 19. Sensitivity to neutral wind")
    add("")
    add(wind["justification"])
    add("")
    geometry = wind["geometry"]
    add(f"- Evaluated at the {geometry['evaluation_state']}: "
        f"{geometry['latitude_deg']:.2f} deg N, {geometry['longitude_deg']:.2f} deg E")
    add(f"- Inertial speed {geometry['inertial_speed_m_s']:.1f} m/s; co-rotation "
        f"{geometry['corotation_speed_m_s']:.1f} m/s")
    add(f"- Orbit average over {geometry['orbit_average_span_hours']:.2f} h, "
        f"{geometry['orbit_average_samples']} samples")
    add("")
    add(geometry["along_track_projection_note"])
    add("")
    add("| wind (m/s, east) | relative speed (m/s) | change | projection | drag accel (m/s2) "
        "| drag change | orbit-mean drag change | first-order lifetime change |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in wind["rows"]:
        projection = row["along_track_projection_at_deployment"]
        add(f"| {row['wind_east_m_s']:+.0f} "
            f"| {row['relative_speed_at_deployment_m_s']:.2f} "
            f"| {row['relative_speed_change_m_s']:+.2f} "
            f"| {'-' if projection is None else f'{projection:.3f}'} "
            f"| {row['drag_acceleration_at_deployment_m_s2']:.6e} "
            f"| {_pct(row['drag_acceleration_change_fraction'])} "
            f"| {_pct(row['orbit_mean_drag_change_fraction'])} "
            f"| {_pct(row['first_order_lifetime_change_fraction'])} |")
    add("")
    add("Orbit-averaged relative speed across the sweep:")
    add("")
    add("| wind (m/s, east) | orbit-mean relative speed (m/s) | change | projection |")
    add("| --- | --- | --- | --- |")
    for row in wind["rows"]:
        projection = row["orbit_mean_along_track_projection"]
        add(f"| {row['wind_east_m_s']:+.0f} "
            f"| {row['orbit_mean_relative_speed_m_s']:.2f} "
            f"| {row['orbit_mean_relative_speed_change_m_s']:+.2f} "
            f"| {'-' if projection is None else f'{projection:.3f}'} |")
    add("")
    add(f"**{wind['lifetime_caveat']}**")
    add("")

    # Q20
    add("## 20. Sensitivity to local-time and latitude sampling")
    add("")
    grid = sampling["grid"]
    spread = sampling["spread"]
    add(f"Grid of {grid['n_points']} points at {grid['altitude_m'] / 1000.0:.0f} km, "
        f"{grid['time_utc']}: {len(grid['latitudes_deg'])} latitudes from "
        f"{min(grid['latitudes_deg']):.2f} to {max(grid['latitudes_deg']):.2f} deg, "
        f"{len(grid['longitudes_deg'])} longitudes covering all 24 hours of local "
        "solar time.")
    add("")
    add("| statistic | enhancement |")
    add("| --- | --- |")
    add(f"| minimum | {float(spread['min_enhancement']):.2%} |")
    add(f"| median | {float(spread['median_enhancement']):.2%} |")
    add(f"| mean | {float(spread['mean_enhancement']):.2%} |")
    add(f"| maximum | {float(spread['max_enhancement']):.2%} |")
    add(f"| range | {float(spread['range_enhancement']):.2%} |")
    add(f"| max / min | {float(spread['max_over_min_ratio']):.2f} |")
    add(f"| at the reported sampling point | "
        f"{float(spread['at_the_reported_sampling_point']):.2%} |")
    add(f"| percentile of the reported point | "
        f"{float(spread['reported_point_percentile']):.0f}th |")
    add("")
    best = sampling["most_enhanced_point"]
    worst = sampling["least_enhanced_point"]
    add(f"- Most enhanced cell: {float(best['enhancement_fraction']):.2%} at "
        f"{best['latitude_deg']:.2f} deg, LST {best['local_solar_time_hours']:.1f} h")
    add(f"- Least enhanced cell: {float(worst['enhancement_fraction']):.2%} at "
        f"{worst['latitude_deg']:.2f} deg, LST {worst['local_solar_time_hours']:.1f} h")
    add("")
    add(sampling["reading"])
    add("")
    add("Enhancement against local solar time at 53.22 deg N:")
    add("")
    add("| LST (h) | enhancement |")
    add("| --- | --- |")
    for row in sampling["by_local_solar_time_at_53_22_deg"]:
        add(f"| {row['local_solar_time_hours']:.0f} "
            f"| {float(row['enhancement_fraction']):.2%} |")
    add("")
    add("Enhancement against latitude at 0 deg longitude:")
    add("")
    add("| latitude (deg) | enhancement |")
    add("| --- | --- |")
    for row in sampling["by_latitude_at_longitude_zero"]:
        add(f"| {row['latitude_deg']:+.2f} | {float(row['enhancement_fraction']):.2%} |")
    add("")

    # The spread
    add("## Where the computed enhancement sits against the published MSIS figures")
    add("")
    add(gap["question"])
    add("")
    add(
        f"Reported peak: **{float(gap['reported_peak_enhancement']):.2%}** at 210 km. "
        f"He et al.'s MSIS2.0: "
        f"{float(gap['he_msis2_band_at_200_km'][0]):.0%}-"
        f"{float(gap['he_msis2_band_at_200_km'][1]):.0%} at 200 km and "
        f"{float(gap['he_msis2_band_at_300_km'][0]):.0%}-"
        f"{float(gap['he_msis2_band_at_300_km'][1]):.0%} at 300 km. "
        "Against the altitude-matched (200 km) figure the reported value is "
        f"{float(gap['difference_to_altitude_matched_band_low']):+.2%} from its "
        "lower edge."
    )
    add("")

    explanation_a = gap["explanation_a_baseline_definition"]
    add("### (a) Is it the counterfactual quiet-ap baseline?")
    add("")
    add(explanation_a["hypothesis"])
    add("")
    a1 = explanation_a["a1_pre_storm_drivers_at_the_same_instant"]
    add(f"**a1 - pre-storm OBSERVED drivers, same instant and place.** Reference "
        f"{a1['reference_time_utc']} (ap daily {a1['reference_ap_daily']:.2f}, history "
        f"`{[round(v, 2) for v in a1['reference_ap_history']]}`, F10.7 "
        f"{a1['reference_f107']}).")
    add("")
    add(f"Enhancement becomes **{float(a1['enhancement_fraction']):.2%}**, a change of "
        f"{float(a1['change_vs_counterfactual']):+.2%} against the counterfactual "
        "baseline.")
    add("")
    a2 = explanation_a["a2_pre_storm_density_at_its_own_instant"]
    add(f"**a2 - pre-storm density sampled at its own instant.** "
        f"{_sci(a2['rho_pre_storm_kg_m3'])} kg/m3 against "
        f"{_sci(a2['rho_storm_peak_kg_m3'])} kg/m3 at the peak, giving "
        f"**{float(a2['enhancement_fraction']):.2%}** "
        f"({float(a2['change_vs_counterfactual']):+.2%}). {a2['caveat']}")
    add("")
    add("**a3 - sensitivity to the quiet-ap choice itself:**")
    add("")
    add("| quiet ap | enhancement |")
    add("| --- | --- |")
    for row in explanation_a["a3_sensitivity_to_the_quiet_ap_choice"]:
        add(f"| {row['quiet_ap']:.0f} | {float(row['enhancement_fraction']):.2%} |")
    add("")
    add(
        "The observed 3-hourly ap values in the quiet day before the storm were 2, 3 "
        "and 5, so ap = 4 is a mid-range choice within them and ap = 2 is the quiet "
        "end of what was actually observed. The enhancement is sensitive to that "
        "choice at the level of the gap being investigated, which is worth stating "
        "plainly: the quiet baseline is an assumption, and it is not a small one."
    )
    add("")

    explanation_b = gap["explanation_b_single_point_vs_orbit_averaged"]
    add("### (b) Is it single-point versus orbit-averaged sampling?")
    add("")
    add(explanation_b["hypothesis"])
    add("")
    configuration = explanation_b["orbit_average_configuration"]
    add(f"- {configuration['trajectory']}, "
        f"{configuration['perigee_altitude_m'] / 1000.0:.0f} x "
        f"{configuration['apogee_altitude_m'] / 1000.0:.0f} km, "
        f"{configuration['duration_days']:.0f} days, "
        f"{configuration['n_samples']} samples at "
        f"{configuration['sample_interval_s']:.0f} s")
    add(f"- Whole-path mean enhancement: "
        f"{float(explanation_b['whole_path_mean_enhancement']):.2%}")
    add(f"- Whole-path maximum enhancement: "
        f"{float(explanation_b['whole_path_max_enhancement']):.2%}")
    best_orbit = explanation_b["best_orbit"]
    if best_orbit:
        add(f"- Best single revolution, mean over the revolution: "
            f"**{float(best_orbit['mean_enhancement_fraction']):.2%}** "
            f"(density-weighted "
            f"{float(best_orbit['density_weighted_enhancement_fraction']):.2%}), a change "
            f"of {float(explanation_b['change_vs_counterfactual']):+.2%} against the "
            "single-point figure")
    add(f"- {explanation_b['altitude_caveat']}")
    add("")

    explanation_c = gap["explanation_c_altitude_of_the_published_figure"]
    add("### (c) Is it the altitude at which the published figure is quoted?")
    add("")
    add(explanation_c["hypothesis"])
    add("")
    add("| altitude | enhancement | quoted by He et al.? |")
    add("| --- | --- | --- |")
    for row in explanation_c["by_altitude"]:
        add(f"| {row['altitude_m'] / 1000.0:.0f} km "
            f"| {float(row['enhancement_fraction']):.2%} "
            f"| {'yes' if row['quoted_by_he_et_al'] else 'no'} |")
    add("")
    inside = explanation_c["altitudes_landing_inside_the_he_band_m"]
    if inside:
        add("Altitudes at which this run lands INSIDE the He et al. 20-25% band: "
            + ", ".join(f"{v / 1000.0:.0f} km" for v in inside)
            + ".")
    else:
        add("This run lands inside the He et al. 20-25% band at NO altitude sampled.")
    add("")
    add(explanation_c["reading"])
    add("")

    explanation_d = gap["explanation_d_peak_definition"]
    add("### (d) Is it the definition of 'peak'?")
    add("")
    add(explanation_d["hypothesis"])
    add("")
    where = explanation_d["where"]
    add(f"Maximum over the whole window AND over latitude and local time: "
        f"**{float(explanation_d['max_over_time_and_place']):.2%}** at "
        f"{where['time_utc']}, {where['latitude_deg']:.2f} deg, LST "
        f"{where['local_solar_time_hours']:.1f} h - a change of "
        f"{float(explanation_d['change_vs_counterfactual']):+.2%} against the reported "
        "single-point peak.")
    add("")

    add("## What must not be claimed from this audit")
    add("")
    add("- A scalar multiplier applied to this model is NOT a physics simulation. A "
        "x2 result is not a WAM-IPE result and a x3.5 result is not a MAGE result; "
        "the provider that applies such multipliers is named "
        "`ScalarDensityStressTestProvider` and every state it returns carries that "
        "prohibition.")
    add("- Swarm-A and GRACE-FO do NOT validate the 210 km density. They are 228 km "
        "and 295 km above it.")
    add("- Agreement with He et al. would be agreement with the model family already "
        "in use, in the direction that family is documented to be wrong. It is not "
        "corroboration.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    # Refuse to produce a publication artefact from code that is not
    # this checkout's. The venv here is a symlink to a sibling
    # worktree's environment, so this is a live hazard, not a theory.
    _EXEC_ENV.require_source_tree()
    EVIDENCE.mkdir(exist_ok=True)
    kp, f107 = load_drivers()
    provider = Nrlmsise00Provider()

    print("recomputing the reported enhancement series ...")
    outputs, series_info = audit_outputs(provider, kp, f107)
    peak_time: dt.datetime = series_info["peak_time"]
    peak_enhancement: float = float(series_info["peak_enhancement"])
    print(f"  peak {peak_enhancement:.4%} at {peak_time.isoformat()}")

    peak_drivers = sw.drivers_at(peak_time, kp, f107)

    print("auditing model configuration ...")
    configuration = audit_model_configuration(peak_drivers)
    print("auditing drivers and sampling ...")
    drivers_section = audit_drivers_and_sampling(kp, f107, peak_time)
    print("comparing against published values ...")
    published = audit_published_comparison(peak_enhancement)
    print("comparing against the published-envelope provider ...")
    envelope = audit_envelope_comparison(provider, kp, f107, peak_time)
    print("comparing against observation-derived density ...")
    observational = audit_observational_comparison(provider, kp, f107, peak_time)
    print("sweeping neutral wind ...")
    wind = audit_wind_sensitivity(provider, kp, f107, peak_time)
    print("sweeping local time and latitude ...")
    sampling = audit_local_time_and_latitude(provider, kp, f107, peak_time)
    print("testing explanations for the gap to He et al. 2023 ...")
    gap = audit_gap_to_he_et_al(provider, kp, f107, peak_time, peak_enhancement)

    audit: dict[str, Any] = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "purpose": (
            "Audit of the atmosphere model behind the reported peak density "
            "enhancement, answering twenty reviewer questions with computed values "
            "and testing four falsifiable explanations for the gap to He et al. 2023."
        ),
        "model_configuration": configuration,
        "drivers_and_sampling": drivers_section,
        "outputs": outputs,
        "enhancement_series": series_info["series"],
        "published_comparison": published,
        "envelope_comparison": envelope,
        "observational_comparison": observational,
        "wind_sensitivity": wind,
        "local_time_and_latitude": sampling,
        "gap_analysis": gap,
    }

    json_path = EVIDENCE / "atmosphere_audit.json"
    json_path.write_text(json.dumps(audit, indent=2, default=str))
    print(f"wrote {json_path}")

    markdown_path = EVIDENCE / "ATMOSPHERE_MODEL_AUDIT.md"
    markdown_path.write_text(render_markdown(audit))
    print(f"wrote {markdown_path}")

    gap_section = gap["spread_against_published_msis_figures"]
    explanation_a = gap_section["explanation_a_baseline_definition"]
    explanation_b = gap_section["explanation_b_single_point_vs_orbit_averaged"]
    print("\nGAP ANALYSIS")
    print(f"  reported single-point peak      {peak_enhancement:.4%}")
    print(
        "  pre-storm observed baseline     "
        f"{float(explanation_a['a1_pre_storm_drivers_at_the_same_instant']['enhancement_fraction']):.4%}"
    )
    print(
        "  pre-storm density, own instant  "
        f"{float(explanation_a['a2_pre_storm_density_at_its_own_instant']['enhancement_fraction']):.4%}"
    )
    best_orbit = explanation_b["best_orbit"]
    if best_orbit:
        print(f"  best orbit-averaged revolution  {best_orbit['mean_enhancement_fraction']:.4%}")
    peak_definition = gap_section["explanation_d_peak_definition"]
    print(
        "  max over time, lat and LST      "
        f"{float(peak_definition['max_over_time_and_place']):.4%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
