"""Tabular outputs: a trajectory as rows, and a scenario set as one summary row.

The unit convention changes at this boundary and only here. Inside the package
everything is SI metres; a table that a human reads or that goes into a paper is
in kilometres, because "perigee 2.1e5" is a number nobody checks by eye. The
conversion is applied once, in this module, and the column names carry the unit
so a stray metres value cannot hide in a kilometres column.

Two formats, for two different jobs, deliberately not one:

  * CSV is for reading, grepping and pasting into anything. It is text, so a
    timestamp becomes a string and a float becomes its repr.
  * Parquet is for exactness. Timezone-aware timestamps, float64 and the
    missing/NaN distinction survive a write/read cycle unchanged, which CSV
    cannot promise.

Ragged input is rejected rather than truncated. A :class:`PropagationResult`
whose parallel series have different lengths is a corrupted result, and a table
built by zipping them would silently pair the wrong density with the wrong time
— an error that produces a plausible plot and a wrong conclusion.

Nearest canonical tickets: P09_10_T030, P09_10_T031 (per-time-step results
schema and record assembly).
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Mapping

import pandas as pd

from ..domain.temporal import ensure_utc
from ..propagation.propagator import PropagationResult

__all__ = [
    "DECAY_SUMMARY_COLUMNS",
    "TRAJECTORY_COLUMNS",
    "decay_summary_table",
    "propagation_to_dataframe",
    "write_csv",
    "write_parquet",
]

#: Column order of the per-time-step trajectory table. Pinned: downstream
#: readers and figure code index by name, but a stable order keeps file diffs
#: reviewable and makes a schema change visible in one place.
TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "time_utc",
    "altitude_km",
    "perigee_km",
    "apogee_km",
    "semi_major_axis_km",
    "eccentricity",
    "density_kg_m3",
    "drag_accel_m_s2",
)

#: Column order of the one-row-per-scenario decay summary.
DECAY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "scenario",
    "inverse_ballistic_coefficient",
    "reentered",
    "decay_days",
    "final_perigee_km",
)

_M_PER_KM = 1000.0


def _series_lengths(result: PropagationResult) -> dict[str, int]:
    return {
        "times_utc": len(result.times_utc),
        "altitudes_m": len(result.altitudes_m),
        "perigee_altitude_m": len(result.perigee_altitude_m),
        "apogee_altitude_m": len(result.apogee_altitude_m),
        "semi_major_axis_m": len(result.semi_major_axis_m),
        "eccentricity": len(result.eccentricity),
        "density_kg_m3": len(result.density_kg_m3),
        "drag_acceleration_m_s2": len(result.drag_acceleration_m_s2),
    }


def propagation_to_dataframe(result: PropagationResult) -> pd.DataFrame:
    """Convert a propagation result to a per-time-step table in kilometres.

    Altitudes are converted from metres exactly once, here. Density and drag
    acceleration stay in SI: they are read by machines and compared against
    model output, never eyeballed, so a "convenient" unit would only invite a
    conversion error.

    ``NaN`` in ``density_kg_m3`` and ``drag_accel_m_s2`` is preserved rather
    than filled. Below the reentry floor the atmosphere model is out of its
    domain and the propagator records no density; substituting zero would be a
    physical claim the model never made.
    """
    lengths = _series_lengths(result)
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"PropagationResult series have mismatched lengths: {lengths}. "
            "Rows cannot be assembled from ragged series without silently "
            "pairing a sample with another sample's time."
        )
    count = next(iter(lengths.values()))
    if count == 0:
        raise ValueError(
            "PropagationResult contains no samples; there is nothing to "
            "tabulate. An empty trajectory is a failed integration, not a "
            "valid result."
        )

    times = [ensure_utc(when, "PropagationResult.times_utc") for when in result.times_utc]
    frame = pd.DataFrame(
        {
            "time_utc": times,
            "altitude_km": [value / _M_PER_KM for value in result.altitudes_m],
            "perigee_km": [value / _M_PER_KM for value in result.perigee_altitude_m],
            "apogee_km": [value / _M_PER_KM for value in result.apogee_altitude_m],
            "semi_major_axis_km": [value / _M_PER_KM for value in result.semi_major_axis_m],
            "eccentricity": list(result.eccentricity),
            "density_kg_m3": list(result.density_kg_m3),
            "drag_accel_m_s2": list(result.drag_acceleration_m_s2),
        }
    )
    return frame[list(TRAJECTORY_COLUMNS)]


def decay_summary_table(results: Mapping[str, PropagationResult]) -> pd.DataFrame:
    """One row per scenario: what happened, and under which inverse ballistic coefficient.

    Scenario order follows the mapping's insertion order, which is the order the
    caller ran them (typically quiet before storm). Sorting alphabetically would
    reorder a counterfactual pair for no reason.

    ``decay_days`` is ``NaN`` for a scenario that had not reentered when the
    integration horizon ran out. The fact is carried by ``reentered``: a NaN
    here means "did not reenter within the horizon", never "reentered at an
    unknown time".

    ``inverse_ballistic_coefficient`` is C_d*A/m in m^2/kg. It is reported because mass,
    area and drag coefficient are not separately identifiable from a decay
    curve — only their combination is — so it is the only drag parameter a
    reader can meaningfully compare between scenarios.
    """
    if not results:
        raise ValueError(
            "decay_summary_table requires at least one scenario; an empty "
            "summary would state nothing while looking like a result."
        )

    rows: list[dict[str, object]] = []
    for scenario, result in results.items():
        if not result.perigee_altitude_m:
            raise ValueError(
                f"scenario {scenario!r} has no samples, so it has no final "
                "perigee to report."
            )
        decay_days = result.decay_days
        rows.append(
            {
                "scenario": scenario,
                "inverse_ballistic_coefficient": result.spacecraft.inverse_ballistic_coefficient,
                "reentered": bool(result.reentered),
                "decay_days": math.nan if decay_days is None else decay_days,
                "final_perigee_km": result.perigee_altitude_m[-1] / _M_PER_KM,
            }
        )
    return pd.DataFrame(rows)[list(DECAY_SUMMARY_COLUMNS)]


def write_csv(df: pd.DataFrame, path: pathlib.Path | str) -> pathlib.Path:
    """Write a table as UTF-8 CSV without the index, returning the path.

    The index is dropped because it carries no information here — the row
    identity is the timestamp or the scenario name, both already columns — and
    an unnamed leading column is a reliable source of confusion downstream.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8")
    return target


def write_parquet(df: pd.DataFrame, path: pathlib.Path | str) -> pathlib.Path:
    """Write a table as Parquet without the index, returning the path.

    This is the format to reload from: dtypes, the UTC offset on timestamps and
    the NaN/present distinction all survive the round trip, so a reloaded table
    is the same table rather than a re-parsed approximation of it.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False, engine="pyarrow")
    return target
