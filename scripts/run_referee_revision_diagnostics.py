#!/usr/bin/env python3
"""Deterministic Round-3 referee-revision calculations for Papers I, II and III.

Reads frozen 08c8b86 evidence and raw drivers strictly read-only without modifying
any existing source or historical evidence file, and writes a deterministic,
byte-repeatable (no wall-clock runtime keys) SHA-256-verifiable artefact
``evidence/referee_revision_diagnostics.json`` resolving all Round-1, Round-2 and Round-3 findings:
  - Paper I (R-I-1..R-I-7, R3-1, R3-4):
      * Same-UTC-hour vs whole-day-mean WAM/MSIS baselines across altitudes & days,
      * Inertial orbital-energy loss vs atmosphere-relative dissipation (bridged & contiguous),
      * Full-vector 1D log-s cubic-spline V3-pair orbital propagation at 0, 30 and 90 deg,
        including explicit 90 deg V2 collimated-area quiet/storm trajectories (fixing R3-1),
        retaining FULL drag + normal/lift vectors and verified across a dense 101-temperature
        [550, 1400 K] grid (fixing R3-4),
      * Explicit 30 s vs 60 s max-step convergence verification on all 0, 30 and 90 deg V2/V3-pair
        pairs (12 tightened convergence trajectories),
      * Minimum Knudsen number sampled on the accepted dense-output trajectory over t <= t_170
        (sampling spacing <= 30 s plus accepted solver knots; not a continuous lower-bound proof).
  - Paper II (R-II-1..R-II-4, R3-5):
      * 20-period drag-free central+J2 control invariants (E, h_z, mean a, mean e, mean perigee)
        for fixed-Keplerian nu, E-matched nu, E-matched omega, and same-orbit J2-transported phase,
      * Uncached deterministic NRLMSISE-00 brentq refits across all 4 conventions plus coupled
        release-epoch shifts (conditional four-state same-orbit initial-phase sensitivity),
      * Exact 652-case ensemble & 10-object SGP4-initialized catalogue summaries.
  - Paper III (R1..R6, R3-2):
      * Exact analytic-horizontal (erfc-safe) + adaptive mean-centered 1D normal quadrature for
        V3-pair across all 181 integer attitudes (0..180 deg) + near-zero probes,
      * Replicated scrambled Sobol paired exact-pair control variate (R=16 replicates x 2^16 = 2^20
        paired samples per angle) across ALL 7 cardinal angles {0, 15, 30, 45, 60, 75, 90} deg
        reporting mean, SE, 95% CI, SNR and resolution status (fixing R3-2),
      * Exact printed Walker-Mehta-Koller CLL Eq. (8) specular-endpoint residuals at (0,0) across
        all 6 CLL species, plus 145-h temperature-vs-composition OAT decomposition.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from pymsis import msis as _pymsis_mod  # noqa: E402
from pymsis import msis00f as _msis00f  # noqa: E402
from scipy.integrate import quad, solve_ivp  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import erf, ndtr, ndtri  # noqa: E402
from scipy.stats import qmc, t as student_t  # noqa: E402

from thermosphere_consequence.aerodynamics.cll_constants import (  # noqa: E402
    CLL_SUPPORTED_SPECIES,
    cll_constants,
)
from thermosphere_consequence.aerodynamics.geometry import self_shadowing_visibility  # noqa: E402
from thermosphere_consequence.aerodynamics.proxy_geometry import starlink_like_proxy  # noqa: E402
from thermosphere_consequence.domain.constants import MU_EARTH, OMEGA_EARTH  # noqa: E402
from thermosphere_consequence.drivers import space_weather as sw  # noqa: E402
from thermosphere_consequence.drivers.raw_store import RawStore  # noqa: E402
from thermosphere_consequence.events import starlink_g4_7 as EV  # noqa: E402
from thermosphere_consequence.molecular import composition as comp  # noqa: E402
from thermosphere_consequence.molecular.kinetics import (  # noqa: E402
    hard_sphere_mean_free_path,
    knudsen_number,
)
from thermosphere_consequence.propagation import energy_ledger as EL  # noqa: E402
from thermosphere_consequence.propagation.propagator import (  # noqa: E402
    DragPropagator,
    OrbitState,
    REENTRY_ALTITUDE_M,
    SpacecraftProperties,
    geodetic_altitude,
    geodetic_latlon,
    greenwich_sidereal_time,
    j2_acceleration,
)
from thermosphere_consequence.providers.nrlmsise00 import Nrlmsise00Provider  # noqa: E402
from thermosphere_consequence.providers.storm_grid import (  # noqa: E402
    GriddedStormDensityProvider,
    OutsideGridCoverageError,
)
from thermosphere_consequence.scenarios.fixed_path import quiet_baseline_drivers  # noqa: E402

KB = 1.380649e-23
RE_WGS84 = 6378137.0
J2_CONST = 1.08262668e-3
SQRT_PI = math.sqrt(math.pi)
SQRT_2 = math.sqrt(2.0)

_EPOCH_2022_START_TS = int(dt.datetime(2022, 1, 1, 0, 0, tzinfo=dt.UTC).timestamp())
_DEPLOY_TS = int(EV.DEPLOYMENT_UTC.timestamp())


# ---------------------------------------------------------------------------
# Bit-exact fast scalar NRLMSISE-00 evaluator (verified 0.0-ulp vs Nrlmsise00Provider)
# ---------------------------------------------------------------------------

class FastMsis00Evaluator:
    """Calls ``pymsis.msis00f.pymsiscalc`` with identical float32 inputs and storm options
    to ``Nrlmsise00Provider._raw``, pre-indexing 3-hour space-weather slots.
    """

    def __init__(self, kp: Any, f107: Any, *, quiet_ap: float | None = None) -> None:
        self.options = _pymsis_mod.create_options(geomagnetic_activity=-1)
        _msis00f.pyinitswitch(self.options, parmpath=_pymsis_mod._MSIS_PARAMETER_PATH)
        _msis00f._last_used_options = self.options

        self.base_slot_ts = int(dt.datetime(2022, 2, 1, 0, 0, tzinfo=dt.UTC).timestamp())
        n_slots = 35 * 8
        self.slot_f107 = np.empty(n_slots, dtype=np.float32)
        self.slot_f107a = np.empty(n_slots, dtype=np.float32)
        self.slot_aps = np.empty((n_slots, 7), dtype=np.float32)
        for idx in range(n_slots):
            when = dt.datetime.fromtimestamp(self.base_slot_ts + idx * 10800, tz=dt.UTC)
            drv = sw.drivers_at(when, kp, f107)
            if quiet_ap is not None:
                drv = quiet_baseline_drivers(drv, quiet_ap=quiet_ap)
            self.slot_f107[idx] = drv.f107
            self.slot_f107a[idx] = drv.f107a
            self.slot_aps[idx, :] = drv.ap_history if drv.ap_history is not None else [drv.ap_daily] * 7

        self._buf = np.empty((1, 14), dtype=np.float32, order="F")
        self.col_masses = np.array(
            [comp.species_mass_density(sym, 1.0) for _, sym in (
                (1, "N2"), (2, "O2"), (3, "O"), (4, "He"),
                (5, "H"), (6, "Ar"), (7, "N"), (8, "anomalous_O"),
            )],
            dtype=np.float64,
        )

    def eval_unix(self, unix_s_float: float, alt_m: float, lat_deg: float, lon_deg: float) -> np.ndarray:
        ts_int = int(math.floor(unix_s_float))
        sec_since_y = ts_int - _EPOCH_2022_START_TS
        doy = float(sec_since_y // 86400 + 1)
        dsec = float(sec_since_y % 86400)
        slot = (ts_int - self.base_slot_ts) // 10800

        b = self._buf
        b[0, 0] = doy
        b[0, 1] = dsec
        b[0, 2] = lon_deg
        b[0, 3] = lat_deg
        b[0, 4] = alt_m * 0.001
        b[0, 5] = self.slot_f107[slot]
        b[0, 6] = self.slot_f107a[slot]
        b[0, 7:] = self.slot_aps[slot]

        out = _msis00f.pymsiscalc(
            b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5], b[:, 6], b[:, 7:]
        )
        row = out[0]
        row[(row >= 9.9e-38) & (row < 1e-37)] = np.nan
        return row

    def species_mass_density_unix(self, unix_s_float: float, alt_m: float, lat_deg: float, lon_deg: float) -> float:
        row = self.eval_unix(unix_s_float, alt_m, lat_deg, lon_deg)
        nds = np.asarray(row[1:9], dtype=np.float64)
        nds = np.where(np.isnan(nds), 0.0, nds)
        return float(nds @ self.col_masses)


def _gmst_from_unix(unix_s: float) -> float:
    jd = 2440587.5 + unix_s / 86400.0
    t = (jd - 2451545.0) / 36525.0
    gmst_s = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    return math.radians((gmst_s % 86400.0) / 240.0)


MSIS00_FIELD_SPECS: tuple[tuple[int, str, str, bool], ...] = (
    (0, "total_mass_density_kg_m3", "kg/m^3", True),
    (1, "n_N2_m3", "m^-3", True),
    (2, "n_O2_m3", "m^-3", True),
    (3, "n_O_m3", "m^-3", True),
    (4, "n_He_m3", "m^-3", True),
    (5, "n_H_m3", "m^-3", True),
    (6, "n_Ar_m3", "m^-3", True),
    (7, "n_N_m3", "m^-3", True),
    (8, "n_anomalous_O_m3", "m^-3", True),
    (9, "n_NO_m3", "m^-3", False),
    (10, "temperature_K", "K", True),
)

WAM_IPE_KNOWN_ARCHIVE_GAPS_UTC: tuple[str, ...] = (
    "2022-02-05T10:00:00Z",
    "2022-02-05T11:00:00Z",
    "2022-02-05T12:00:00Z",
    "2022-02-05T13:00:00Z",
    "2022-02-05T14:00:00Z",
    "2022-02-05T15:00:00Z",
    "2022-02-07T10:00:00Z",
    "2022-02-07T11:00:00Z",
    "2022-02-07T12:00:00Z",
    "2022-02-07T13:00:00Z",
    "2022-02-07T14:00:00Z",
    "2022-02-07T15:00:00Z",
)

CONSUMED_RAW_DRIVER_RELPATHS: tuple[str, ...] = (
    "data/raw/gfz/kp_2022_extended.json",
    "data/raw/gfz/kp_2022_extended_ap.json",
    "data/raw/celestrak/sw_last5years.csv",
    "data/raw/celestrak/satcat_2022-010.json",
)


def _map_tasks(fn: Any, tasks: list[Any], max_workers: int = 1) -> list[Any]:
    """Execute tasks serially in-process when max_workers <= 1 (safe shared-host default)."""
    if int(max_workers) <= 1 or len(tasks) <= 1:
        return [fn(item) for item in tasks]
    with ProcessPoolExecutor(max_workers=min(int(max_workers), len(tasks))) as pool:
        return list(pool.map(fn, tasks))


def _ieee754_float64_ulp_distance(a: float, b: float) -> int:
    """Exact IEEE-754 binary64 ULP distance between two finite floats (-0.0 == +0.0 -> 0)."""
    fa, fb = float(a), float(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise ValueError(f"ULP distance requires finite float64 values, got {fa!r}, {fb!r}")
    if fa == 0.0 and fb == 0.0:
        return 0
    ua = struct.unpack(">Q", struct.pack(">d", fa))[0]
    ub = struct.unpack(">Q", struct.pack(">d", fb))[0]
    oa = (ua ^ 0xFFFFFFFFFFFFFFFF) if (ua >> 63) else (ua | 0x8000000000000000)
    ob = (ub ^ 0xFFFFFFFFFFFFFFFF) if (ub >> 63) else (ub | 0x8000000000000000)
    return abs(oa - ob)


def _ieee754_float32_ulp_distance(a: np.float32, b: np.float32) -> int:
    """Exact IEEE-754 binary32 ULP distance between two finite float32 values (-0.0 == +0.0 -> 0)."""
    fa, fb = float(np.float32(a)), float(np.float32(b))
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise ValueError(f"ULP distance requires finite float32 values, got {fa!r}, {fb!r}")
    if fa == 0.0 and fb == 0.0:
        return 0
    ua = struct.unpack(">I", struct.pack(">f", fa))[0]
    ub = struct.unpack(">I", struct.pack(">f", fb))[0]
    oa = (ua ^ 0xFFFFFFFF) if (ua >> 31) else (ua | 0x80000000)
    ob = (ub ^ 0xFFFFFFFF) if (ub >> 31) else (ub | 0x80000000)
    return abs(oa - ob)


def verify_fast_msis_bit_identity(
    kp: Any,
    f107: Any,
    msis_ref: Nrlmsise00Provider,
    *,
    fast_evaluator_cls: Any = None,
) -> dict[str, Any]:
    """Verify exact mask parity (NaN/Inf/finite in both directions), same-dtype parity
    (binary32 float32 from pymsis.msis00f or binary64 float64), physical-unit max_abs_diff,
    and exact same-dtype IEEE-754 ULP distance across all 11 NRLMSISE-00 output fields
    over 20 storm/quiet test states.
    """
    eval_cls = fast_evaluator_cls if fast_evaluator_cls is not None else FastMsis00Evaluator
    fast_s = eval_cls(kp, f107, quiet_ap=None)
    fast_q = eval_cls(kp, f107, quiet_ap=4.0)

    per_field: dict[str, dict[str, Any]] = {}
    for _idx, fname, units, expect_finite in MSIS00_FIELD_SPECS:
        per_field[fname] = {
            "units": units,
            "expected_finite_in_msis00": expect_finite,
            "dtype_ref": None,
            "dtype_cand": None,
            "n_states": 0,
            "n_finite_ref": 0,
            "n_finite_cand": 0,
            "n_nan_ref": 0,
            "n_nan_cand": 0,
            "n_inf_ref": 0,
            "n_inf_cand": 0,
            "mask_equal": True,
            "max_abs_diff": 0.0,
            "max_ulp_diff": 0,
        }

    tested_states = 0
    state_grid: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    max_ulp_diff = 0
    observed_dtypes: set[str] = set()
    same_dtype_verified = True
    nan_mask_equal = True
    inf_mask_equal = True
    finite_mask_equal = True
    mismatches: list[str] = []

    for h_off in (0.0, 13.37, 48.91, 125.04, 310.77):
        when = EV.DEPLOYMENT_UTC + dt.timedelta(hours=h_off)
        ts = when.timestamp()
        for alt_m, lat_d, lon_d in ((210_000.0, 53.22, 0.0), (175_432.1, -31.4, 142.7)):
            drv_s = sw.drivers_at(when, kp, f107)
            drv_q = quiet_baseline_drivers(drv_s, quiet_ap=4.0)
            for forcing_label, drv, fast_ev in (("storm", drv_s, fast_s), ("quiet", drv_q, fast_q)):
                tested_states += 1
                state_grid.append({
                    "hours_after_deployment": h_off,
                    "alt_m": alt_m,
                    "lat_deg": lat_d,
                    "lon_deg": lon_d,
                    "forcing": forcing_label,
                })
                ref_arr = np.asarray(msis_ref._raw(when, alt_m, lat_d, lon_d, drv))
                got_arr = np.asarray(fast_ev.eval_unix(ts, alt_m, lat_d, lon_d))
                if ref_arr.shape != (11,) or got_arr.shape != (11,):
                    raise AssertionError(
                        f"MSIS array shape mismatch at state {tested_states}: ref={ref_arr.shape}, cand={got_arr.shape}"
                    )
                observed_dtypes.add(str(ref_arr.dtype))
                observed_dtypes.add(str(got_arr.dtype))
                if ref_arr.dtype != got_arr.dtype or ref_arr.dtype not in (np.float32, np.float64):
                    same_dtype_verified = False
                    mismatches.append(
                        f"state {tested_states} dtype mismatch: ref={ref_arr.dtype}, cand={got_arr.dtype}"
                    )

                ref_nan = np.isnan(ref_arr)
                got_nan = np.isnan(got_arr)
                ref_inf = np.isinf(ref_arr)
                got_inf = np.isinf(got_arr)
                ref_fin = np.isfinite(ref_arr)
                got_fin = np.isfinite(got_arr)

                if not bool(np.array_equal(ref_nan, got_nan)):
                    nan_mask_equal = False
                if not bool(np.array_equal(ref_inf, got_inf)):
                    inf_mask_equal = False
                if not bool(np.array_equal(ref_fin, got_fin)):
                    finite_mask_equal = False

                use_f32 = (ref_arr.dtype == np.float32 and got_arr.dtype == np.float32)

                for idx, fname, _units, expect_finite in MSIS00_FIELD_SPECS:
                    rec = per_field[fname]
                    rec["dtype_ref"] = str(ref_arr.dtype)
                    rec["dtype_cand"] = str(got_arr.dtype)
                    rec["n_states"] += 1
                    r_nan, g_nan = bool(ref_nan[idx]), bool(got_nan[idx])
                    r_inf, g_inf = bool(ref_inf[idx]), bool(got_inf[idx])
                    r_fin, g_fin = bool(ref_fin[idx]), bool(got_fin[idx])
                    rec["n_finite_ref"] += int(r_fin)
                    rec["n_finite_cand"] += int(g_fin)
                    rec["n_nan_ref"] += int(r_nan)
                    rec["n_nan_cand"] += int(g_nan)
                    rec["n_inf_ref"] += int(r_inf)
                    rec["n_inf_cand"] += int(g_inf)

                    if r_nan != g_nan or r_inf != g_inf or r_fin != g_fin:
                        rec["mask_equal"] = False
                        mismatches.append(
                            f"state {tested_states} ({forcing_label}, h={h_off}, alt={alt_m}) field {fname}: "
                            f"mask mismatch ref=(fin={r_fin},nan={r_nan},inf={r_inf}) vs "
                            f"cand=(fin={g_fin},nan={g_nan},inf={g_inf})"
                        )
                        continue

                    if expect_finite and not (r_fin and g_fin):
                        rec["mask_equal"] = False
                        mismatches.append(
                            f"state {tested_states} field {fname}: expected finite output in NRLMSISE-00, got non-finite"
                        )
                        continue
                    if (not expect_finite) and not (r_nan and g_nan):
                        rec["mask_equal"] = False
                        mismatches.append(
                            f"state {tested_states} field {fname}: expected NaN in NRLMSISE-00, got non-NaN "
                            f"(ref_nan={r_nan}, cand_nan={g_nan}, ref_inf={r_inf}, cand_inf={g_inf}, ref_fin={r_fin}, cand_fin={g_fin})"
                        )
                        continue

                    if r_fin and g_fin:
                        rv = float(ref_arr[idx])
                        gv = float(got_arr[idx])
                        abs_d = abs(rv - gv)
                        ulp_d = (
                            _ieee754_float32_ulp_distance(ref_arr[idx], got_arr[idx])
                            if use_f32
                            else _ieee754_float64_ulp_distance(rv, gv)
                        )
                        raw_bits_ref = struct.pack(">f", float(np.float32(ref_arr[idx]))) if use_f32 else struct.pack(">d", rv)
                        raw_bits_cand = struct.pack(">f", float(np.float32(got_arr[idx]))) if use_f32 else struct.pack(">d", gv)
                        raw_bits_equal = (raw_bits_ref == raw_bits_cand)
                        if abs_d > rec["max_abs_diff"]:
                            rec["max_abs_diff"] = abs_d
                        if ulp_d > rec["max_ulp_diff"]:
                            rec["max_ulp_diff"] = ulp_d
                        if abs_d > max_abs_diff:
                            max_abs_diff = abs_d
                        if ulp_d > max_ulp_diff:
                            max_ulp_diff = ulp_d
                        if ulp_d != 0 or abs_d != 0.0 or not raw_bits_equal:
                            mismatches.append(
                                f"state {tested_states} ({forcing_label}, h={h_off}) field {fname}: "
                                f"ref={rv!r} (0x{raw_bits_ref.hex()}) != cand={gv!r} (0x{raw_bits_cand.hex()}) "
                                f"(abs_diff={abs_d:.3e}, ulp={ulp_d}, sign_bit_equal={raw_bits_equal})"
                            )
                    elif r_inf and g_inf:
                        if math.copysign(1.0, float(ref_arr[idx])) != math.copysign(1.0, float(got_arr[idx])):
                            inf_mask_equal = False
                            rec["mask_equal"] = False
                            mismatches.append(f"state {tested_states} field {fname}: opposite-sign infinity")

    if len(observed_dtypes) != 1:
        same_dtype_verified = False
    dtype_label = sorted(observed_dtypes)[0] if len(observed_dtypes) == 1 else ",".join(sorted(observed_dtypes))

    if mismatches or (not nan_mask_equal) or (not inf_mask_equal) or (not finite_mask_equal) or (not same_dtype_verified):
        raise AssertionError(
            "FastMsis00Evaluator equivalence verification failed: "
            + "; ".join(mismatches[:6])
        )

    return {
        "tested_states": tested_states,
        "dtype_observed": dtype_label,
        "ulp_representation": "ieee754_binary32" if dtype_label == "float32" else "ieee754_binary64",
        "same_dtype_verified": same_dtype_verified,
        "bit_equality_includes_sign_bit": True,
        "signed_zero_bit_equality_verified": True,
        "fields_in_scope": [spec[1] for spec in MSIS00_FIELD_SPECS],
        "finite_fields_in_scope": [spec[1] for spec in MSIS00_FIELD_SPECS if spec[3]],
        "expected_nan_fields": [spec[1] for spec in MSIS00_FIELD_SPECS if not spec[3]],
        "nan_mask_equal": nan_mask_equal,
        "inf_mask_equal": inf_mask_equal,
        "finite_mask_equal": finite_mask_equal,
        "exact_finite_equality": max_abs_diff == 0.0,
        "bit_identical_same_dtype": bool(same_dtype_verified and max_ulp_diff == 0 and nan_mask_equal and inf_mask_equal),
        "ulp_computed": True,
        "max_abs_diff": max_abs_diff,
        "max_ulp_diff": max_ulp_diff,
        "state_grid": state_grid,
        "per_field_results": per_field,
    }


def run_fast_msis_verifier_controls() -> dict[str, Any]:
    """Self-contained unit controls for verify_fast_msis_bit_identity (zero external assets required).

    Verifies:
      1. equal_control (10 finite float32 fields + index-9 NO == NaN) -> PASS (max_abs_diff=0.0, max_ulp_diff=0)
      2. ref_finite_cand_nan (candidate sets O to NaN) -> REJECTED
      3. ref_nan_cand_finite (reference sets O to NaN) -> REJECTED
      4. ref_nan_no_cand_finite_no (candidate sets NO to finite while ref is NaN) -> REJECTED
      5. unequal_finite (candidate perturbs O by +1e-5) -> REJECTED
      6. dtype_mismatch (float32 ref vs float64 cand) -> REJECTED
    """
    base = np.array(
        [1.2e-10, 2.1e15, 1.7e14, 3.0e15, 1.2e13, 1.1e11, 1.6e12, 3.8e13, 0.03, np.nan, 924.0],
        dtype=np.float32,
    )

    class _MockRef:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def _raw(self, *_args: Any) -> np.ndarray:
            return self._arr.copy()

    def _make_cand_cls(arr: np.ndarray) -> type:
        class _MockCand:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def eval_unix(self, *_args: Any) -> np.ndarray:
                return arr.copy()

        return _MockCand

    orig_drivers_at = sw.drivers_at
    orig_quiet = quiet_baseline_drivers
    sw.drivers_at = lambda *_a, **_k: None  # type: ignore[assignment]
    globals()["quiet_baseline_drivers"] = lambda *_a, **_k: None
    try:
        eq_res = verify_fast_msis_bit_identity(None, None, _MockRef(base), fast_evaluator_cls=_make_cand_cls(base))  # type: ignore[arg-type]
        assert eq_res["max_abs_diff"] == 0.0 and eq_res["max_ulp_diff"] == 0 and eq_res["nan_mask_equal"] is True

        controls: dict[str, Any] = {
            "equal_control": {
                "passed": True,
                "dtype_observed": eq_res["dtype_observed"],
                "tested_states": eq_res["tested_states"],
                "max_abs_diff": eq_res["max_abs_diff"],
                "max_ulp_diff": eq_res["max_ulp_diff"],
                "nan_mask_equal": eq_res["nan_mask_equal"],
            }
        }
        perturbed_fin = base.copy()
        perturbed_fin[3] = np.float32(float(base[3]) * (1.0 + 1e-5))
        cand_nan_o = base.copy()
        cand_nan_o[3] = np.nan
        cand_fin_no = base.copy()
        cand_fin_no[9] = np.float32(0.0)
        both_inf_no = base.copy()
        both_inf_no[9] = np.float32(np.inf)
        pos_zero = base.copy()
        pos_zero[8] = np.float32(0.0)
        neg_zero = base.copy()
        neg_zero[8] = np.float32(-0.0)
        fault_cases = [
            ("ref_finite_cand_nan", base.copy(), cand_nan_o),
            ("ref_nan_cand_finite", cand_nan_o, base.copy()),
            ("ref_nan_no_cand_finite_no", base.copy(), cand_fin_no),
            ("no_positive_infinity_both_sides", both_inf_no.copy(), both_inf_no.copy()),
            ("plus_zero_vs_minus_zero", pos_zero, neg_zero),
            ("unequal_finite", base.copy(), perturbed_fin),
            ("dtype_mismatch", base.copy(), base.astype(np.float64)),
        ]
        for label, r_arr, c_arr in fault_cases:
            rejected = False
            msg = ""
            try:
                verify_fast_msis_bit_identity(None, None, _MockRef(r_arr), fast_evaluator_cls=_make_cand_cls(c_arr))  # type: ignore[arg-type]
            except AssertionError as exc:
                rejected = True
                msg = str(exc)
            if not rejected:
                raise AssertionError(f"Negative control {label!r} was NOT rejected by verify_fast_msis_bit_identity")
            controls[label] = {"rejected": True, "error": msg}
        return controls
    finally:
        sw.drivers_at = orig_drivers_at  # type: ignore[assignment]
        globals()["quiet_baseline_drivers"] = orig_quiet


def verify_consumed_raw_inputs(
    repo_root: Path | None = None,
    *,
    manifest_relpath: str = "evidence/DATA_MANIFEST.json",
    raw_root: Path | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    """Verify every consumed raw driver and WAM-IPE NetCDF slice against evidence/DATA_MANIFEST.json
    BEFORE running science. Distinguishes the 12 documented CCMC archive gaps from unexpected missing
    or extra files and digest mismatches.
    """
    root = Path(repo_root) if repo_root is not None else REPO
    rroot = Path(raw_root) if raw_root is not None else (root / "data/raw")
    manifest_path = root / manifest_relpath
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing pinned raw-data manifest: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))

    by_rel: dict[str, str] = {}
    for asset in manifest.get("assets", []):
        paths = asset.get("payload_paths") or []
        shas = asset.get("computed_sha256") or []
        rec_sha = str(asset.get("recorded_sha256") or "").strip().lower()
        for idx, rel in enumerate(paths):
            sha = str(shas[idx] if idx < len(shas) else rec_sha).strip().lower()
            by_rel[str(rel)] = sha

    # Expected 192 hourly timestamps from 2022-02-01 00:00Z to 2022-02-08 23:00Z minus 12 known CCMC archive gaps
    known_gaps_set = set(WAM_IPE_KNOWN_ARCHIVE_GAPS_UTC)
    expected_wam_rels: list[str] = []
    t_cur = dt.datetime(2022, 2, 1, 0, 0, tzinfo=dt.UTC)
    t_end = dt.datetime(2022, 2, 8, 23, 0, tzinfo=dt.UTC)
    slot_count_total = 0
    while t_cur <= t_end:
        slot_count_total += 1
        iso = t_cur.strftime("%Y-%m-%dT%H:%M:%SZ")
        if iso not in known_gaps_set:
            expected_wam_rels.append(f"data/raw/ccmc/wam_ipe_fixed_height_{t_cur:%Y%m%d_%H%M%S}.nc")
        t_cur += dt.timedelta(hours=1)

    unexpected_missing: list[str] = []
    unexpected_extra: list[str] = []
    digest_mismatches: list[dict[str, str]] = []
    driver_assets: dict[str, str] = {}
    wam_verified_pairs: list[list[str]] = []
    ordered_wam_paths: list[Path] = []

    def _resolve_raw(rel: str) -> Path:
        if rel.startswith("data/raw/"):
            return rroot / rel.removeprefix("data/raw/")
        return root / rel

    for rel in CONSUMED_RAW_DRIVER_RELPATHS:
        exp_sha = by_rel.get(rel)
        if not exp_sha:
            unexpected_missing.append(f"{rel} (missing from {manifest_relpath})")
            continue
        p = _resolve_raw(rel)
        if not p.is_file():
            unexpected_missing.append(rel)
            continue
        actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_sha != exp_sha:
            digest_mismatches.append({"path": rel, "expected_sha256": exp_sha, "actual_sha256": actual_sha})
        else:
            driver_assets[rel] = actual_sha

    for rel in expected_wam_rels:
        exp_sha = by_rel.get(rel)
        if not exp_sha:
            unexpected_missing.append(f"{rel} (missing from {manifest_relpath})")
            continue
        p = _resolve_raw(rel)
        if not p.is_file():
            unexpected_missing.append(rel)
            continue
        actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_sha != exp_sha:
            digest_mismatches.append({"path": rel, "expected_sha256": exp_sha, "actual_sha256": actual_sha})
        else:
            wam_verified_pairs.append([rel, actual_sha])
            ordered_wam_paths.append(p)

    ccmc_dir = rroot / "ccmc"
    expected_wam_names = {Path(rel).name for rel in expected_wam_rels}
    if ccmc_dir.is_dir():
        for p in sorted(ccmc_dir.glob("*.nc")):
            if p.name not in expected_wam_names:
                unexpected_extra.append(f"data/raw/ccmc/{p.name}")

    wam_agg_sha = hashlib.sha256(
        json.dumps(wam_verified_pairs, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    inventory = {
        "manifest_path": manifest_relpath,
        "manifest_sha256": manifest_sha256,
        "consumed_count": len(driver_assets) + len(wam_verified_pairs),
        "driver_assets": driver_assets,
        "wam_ipe_slot_count_total": slot_count_total,
        "wam_ipe_expected_count": len(expected_wam_rels),
        "wam_ipe_verified_count": len(wam_verified_pairs),
        "wam_ipe_aggregate_sha256": wam_agg_sha,
        "known_archive_gaps": list(WAM_IPE_KNOWN_ARCHIVE_GAPS_UTC),
        "unexpected_missing_files": unexpected_missing,
        "unexpected_extra_files": unexpected_extra,
        "digest_mismatches": digest_mismatches,
        "input_hashes": {
            **driver_assets,
            "wam_ipe_180_slice_aggregate": wam_agg_sha,
        },
    }

    if unexpected_missing or unexpected_extra or digest_mismatches:
        raise RuntimeError(
            "Consumed raw-input verification failed before science: "
            f"missing={len(unexpected_missing)}, extra={len(unexpected_extra)}, "
            f"digest_mismatches={len(digest_mismatches)}; first_detail="
            f"{(digest_mismatches or unexpected_missing or unexpected_extra)[0]!r}"
        )

    return ordered_wam_paths, inventory


# ---------------------------------------------------------------------------
# 1. Paper I: WAM-IPE vs NRLMSISE-00 baselines, sample counts & path integrals
# ---------------------------------------------------------------------------

def compute_paper1_wam_and_path(
    store: RawStore,
    kp: Any,
    f107: Any,
    msis: Nrlmsise00Provider,
    *,
    wam_files: list[Path] | None = None,
) -> dict[str, Any]:
    nc_files = list(wam_files) if wam_files is not None else verify_consumed_raw_inputs(REPO)[0]
    altitudes_km = [200.0, 210.0, 250.0, 300.0, 350.0, 400.0]
    series_by_alt: dict[float, list[dict[str, Any]]] = {a: [] for a in altitudes_km}

    for p in nc_files:
        when = dt.datetime.strptime(p.stem[-15:], "%Y%m%d_%H%M%S").replace(tzinfo=dt.UTC)
        drv = sw.drivers_at(when, kp, f107)
        with netCDF4.Dataset(p) as ds:
            lat = np.asarray(ds["lat"][:], dtype=float)
            il = int(np.searchsorted(lat, 53.22) - 1)
            w_lat = (53.22 - lat[il]) / (lat[il + 1] - lat[il])
            ha = np.asarray(ds["hlevs"][:], dtype=float)
            ix = int(np.where(np.isclose(ds["lon"][:], 0.0))[0][0])
            den_arr = ds["den"]
            for akm in altitudes_km:
                iz = int(np.searchsorted(ha, akm) - 1)
                wz = (akm - ha[iz]) / (ha[iz + 1] - ha[iz])
                sl = np.asarray(den_arr[0, iz:iz + 2, il:il + 2, ix], dtype=float)
                vals_z = np.exp((1.0 - wz) * np.log(sl[0]) + wz * np.log(sl[1]))
                rho_wam = float((1.0 - w_lat) * vals_z[0] + w_lat * vals_z[1])
                st = msis.state_at(when, akm * 1000.0, 53.22, 0.0, drv)
                rho_msis = comp.total_species_mass_density(st.number_densities)
                series_by_alt[akm].append({
                    "utc": when,
                    "wam": rho_wam,
                    "msis": rho_msis,
                    "msis_native": float(st.total_mass_density or rho_msis),
                })

    rows_210 = series_by_alt[210.0]
    abs_ratios_180 = sorted(r["wam"] / r["msis"] for r in rows_210)
    win_start = dt.datetime(2022, 2, 2, 0, 0, tzinfo=dt.UTC)
    win_end = dt.datetime(2022, 2, 8, 0, 0, tzinfo=dt.UTC)
    rows_210_win133 = [r for r in rows_210 if win_start <= r["utc"] <= win_end]
    abs_ratios_133 = sorted(r["wam"] / r["msis"] for r in rows_210_win133)

    ref_days_210: list[dict[str, Any]] = []
    for d_num in (1, 2, 8):
        ref_date = dt.date(2022, 2, d_num)
        ref_map = {r["utc"].hour: r for r in rows_210 if r["utc"].date() == ref_date}
        non_ref = [r for r in rows_210 if r["utc"].date() != ref_date]
        entry: dict[str, Any] = {"reference_date": str(ref_date), "n_eval_hours": len(non_ref), "n_ref_hours": len(ref_map)}
        for key in ("wam", "msis"):
            day_mean = float(np.mean([x[key] for x in ref_map.values()]))
            sh_val, sh_t = max(((x[key] / ref_map[x["utc"].hour][key] - 1.0, x["utc"]) for x in non_ref), key=lambda z: z[0])
            dm_val, dm_t = max(((x[key] / day_mean - 1.0, x["utc"]) for x in non_ref), key=lambda z: z[0])
            entry[key] = {
                "same_utc_hour_peak_pct": 100.0 * sh_val,
                "same_utc_hour_peak_utc": sh_t.isoformat(),
                "whole_day_mean_peak_pct": 100.0 * dm_val,
                "whole_day_mean_peak_utc": dm_t.isoformat(),
                "reference_day_mean_kg_m3": day_mean,
            }
        entry["same_utc_hour_ratio"] = entry["wam"]["same_utc_hour_peak_pct"] / entry["msis"]["same_utc_hour_peak_pct"]
        entry["whole_day_mean_ratio"] = entry["wam"]["whole_day_mean_peak_pct"] / entry["msis"]["whole_day_mean_peak_pct"]
        ref_days_210.append(entry)

    alt_profile_feb1: list[dict[str, Any]] = []
    ref_feb1 = dt.date(2022, 2, 1)
    for akm in altitudes_km:
        rlist = series_by_alt[akm]
        rmap = {r["utc"].hour: r for r in rlist if r["utc"].date() == ref_feb1}
        nref = [r for r in rlist if r["utc"].date() != ref_feb1]
        w_sh = max(x["wam"] / rmap[x["utc"].hour]["wam"] - 1.0 for x in nref)
        m_sh = max(x["msis"] / rmap[x["utc"].hour]["msis"] - 1.0 for x in nref)
        w_mean = float(np.mean([x["wam"] for x in rmap.values()]))
        m_mean = float(np.mean([x["msis"] for x in rmap.values()]))
        w_dm = max(x["wam"] / w_mean - 1.0 for x in nref)
        m_dm = max(x["msis"] / m_mean - 1.0 for x in nref)
        alt_profile_feb1.append({
            "altitude_km": akm,
            "same_utc_hour_wam_pct": 100.0 * w_sh,
            "same_utc_hour_msis_pct": 100.0 * m_sh,
            "same_utc_hour_ratio": w_sh / m_sh,
            "whole_day_mean_wam_pct": 100.0 * w_dm,
            "whole_day_mean_msis_pct": 100.0 * m_dm,
            "whole_day_mean_ratio": w_dm / m_dm,
        })

    day_mean_ratios_200_300: list[float] = []
    same_hour_ratios_200_300: list[float] = []
    for d_num in (1, 2, 8):
        rdate = dt.date(2022, 2, d_num)
        for akm in (200.0, 210.0, 250.0, 300.0):
            rlist = series_by_alt[akm]
            rmap = {r["utc"].hour: r for r in rlist if r["utc"].date() == rdate}
            nref = [r for r in rlist if r["utc"].date() != rdate]
            w_sh = max(x["wam"] / rmap[x["utc"].hour]["wam"] - 1.0 for x in nref)
            m_sh = max(x["msis"] / rmap[x["utc"].hour]["msis"] - 1.0 for x in nref)
            w_dm = max(x["wam"] / float(np.mean([x["wam"] for x in rmap.values()])) - 1.0 for x in nref)
            m_dm = max(x["msis"] / float(np.mean([x["msis"] for x in rmap.values()])) - 1.0 for x in nref)
            same_hour_ratios_200_300.append(w_sh / m_sh)
            day_mean_ratios_200_300.append(w_dm / m_dm)

    grid = GriddedStormDensityProvider(store, composition_baseline=msis)
    nominal_sc = SpacecraftProperties(mass_kg=1.0, reference_area_m2=1.0, drag_coefficient=1.0)
    drag_free = DragPropagator(lambda *_: 0.0, nominal_sc, include_j2=True)
    pos0, vel0 = EV.insertion_state_vectors()
    path = drag_free.propagate(OrbitState(EV.DEPLOYMENT_UTC, pos0, vel0), 96.0 / 24.0, sample_interval_s=300.0)
    gamma_fit = 0.026246221431224966

    s_times: list[dt.datetime] = []
    s_pos: list[np.ndarray] = []
    s_vel: list[np.ndarray] = []
    rho_msis_s: list[float] = []
    rho_msis_q: list[float] = []
    rho_wam_s: list[float] = []
    refused = 0
    for i, when in enumerate(path.times_utc):
        alt_m = path.altitudes_m[i]
        pos = np.asarray(path.positions_m[i])
        vel = np.asarray(path.velocities_m_s[i])
        lat_d, lon_d = geodetic_latlon(pos, greenwich_sidereal_time(when))
        obs = sw.drivers_at(when, kp, f107)
        qui = quiet_baseline_drivers(obs)
        try:
            rw = grid.total_mass_density(when, alt_m, lat_d, lon_d, obs)
        except OutsideGridCoverageError:
            refused += 1
            continue
        st_s = msis.state_at(when, alt_m, lat_d, lon_d, obs)
        st_q = msis.state_at(when, alt_m, lat_d, lon_d, qui)
        s_times.append(when)
        s_pos.append(pos)
        s_vel.append(vel)
        rho_msis_s.append(comp.total_species_mass_density(st_s.number_densities))
        rho_msis_q.append(comp.total_species_mass_density(st_q.number_densities))
        rho_wam_s.append(rw)

    def _eval_ledger_pair(times: list[dt.datetime], pos_l: list[np.ndarray], vel_l: list[np.ndarray],
                          rs: list[float], rq: list[float], rw: list[float]) -> dict[str, Any]:
        ls = EL.integrate_ledger(times, pos_l, vel_l, rs, gamma_fit)
        lq = EL.integrate_ledger(times, pos_l, vel_l, rq, gamma_fit)
        lw = EL.integrate_ledger(times, pos_l, vel_l, rw, gamma_fit)
        orb_s = abs(ls.inertial_orbital_energy_change_j_kg)
        orb_q = abs(lq.inertial_orbital_energy_change_j_kg)
        orb_w = abs(lw.inertial_orbital_energy_change_j_kg)
        dis_s = ls.atmosphere_relative_dissipation_j_kg
        dis_q = lq.atmosphere_relative_dissipation_j_kg
        dis_w = lw.atmosphere_relative_dissipation_j_kg
        return {
            "n_samples": len(times),
            "inertial_orbital_loss_J_kg": {
                "msis_storm": orb_s, "msis_quiet": orb_q, "wam_storm": orb_w,
                "wam_over_msis_excess_pct": 100.0 * (orb_w / orb_s - 1.0),
                "msis_storm_over_quiet_excess_pct": 100.0 * (orb_s / orb_q - 1.0),
            },
            "atmosphere_relative_dissipation_J_kg": {
                "msis_storm": dis_s, "msis_quiet": dis_q, "wam_storm": dis_w,
                "wam_over_msis_excess_pct": 100.0 * (dis_w / dis_s - 1.0),
                "msis_storm_over_quiet_excess_pct": 100.0 * (dis_s / dis_q - 1.0),
            },
        }

    bridged_985 = _eval_ledger_pair(s_times, s_pos, s_vel, rho_msis_s, rho_msis_q, rho_wam_s)

    seg_indices: list[tuple[int, int]] = []
    seg_start = 0
    gaps_detail: list[dict[str, Any]] = []
    for idx in range(1, len(s_times)):
        dt_s = (s_times[idx] - s_times[idx - 1]).total_seconds()
        if dt_s > 301.0:
            seg_indices.append((seg_start, idx))
            gaps_detail.append({
                "gap_start_utc": s_times[idx - 1].isoformat(),
                "gap_end_utc": s_times[idx].isoformat(),
                "gap_duration_minutes": dt_s / 60.0,
                "start_altitude_km": float(geodetic_altitude(s_pos[idx - 1]) / 1000.0),
                "end_altitude_km": float(geodetic_altitude(s_pos[idx]) / 1000.0),
            })
            seg_start = idx
    seg_indices.append((seg_start, len(s_times)))

    orb_tot = {"msis_storm": 0.0, "msis_quiet": 0.0, "wam_storm": 0.0}
    dis_tot = {"msis_storm": 0.0, "msis_quiet": 0.0, "wam_storm": 0.0}
    segments_summary = []
    for a, b in seg_indices:
        sub = _eval_ledger_pair(s_times[a:b], s_pos[a:b], s_vel[a:b], rho_msis_s[a:b], rho_msis_q[a:b], rho_wam_s[a:b])
        segments_summary.append({
            "start_utc": s_times[a].isoformat(),
            "end_utc": s_times[b - 1].isoformat(),
            **sub,
        })
        for k in orb_tot:
            orb_tot[k] += sub["inertial_orbital_loss_J_kg"][k]
            dis_tot[k] += sub["atmosphere_relative_dissipation_J_kg"][k]

    contiguous_sum = {
        "n_segments": len(seg_indices),
        "gaps_excluded": gaps_detail,
        "segments": segments_summary,
        "inertial_orbital_loss_J_kg": {
            **orb_tot,
            "wam_over_msis_excess_pct": 100.0 * (orb_tot["wam_storm"] / orb_tot["msis_storm"] - 1.0),
            "msis_storm_over_quiet_excess_pct": 100.0 * (orb_tot["msis_storm"] / orb_tot["msis_quiet"] - 1.0),
        },
        "atmosphere_relative_dissipation_J_kg": {
            **dis_tot,
            "wam_over_msis_excess_pct": 100.0 * (dis_tot["wam_storm"] / dis_tot["msis_storm"] - 1.0),
            "msis_storm_over_quiet_excess_pct": 100.0 * (dis_tot["msis_storm"] / dis_tot["msis_quiet"] - 1.0),
        },
    }

    return {
        "total_grid_points_96h_5min": len(path.times_utc),
        "retained_path_samples": len(s_times),
        "missing_gap_samples": refused,
        "wam_peak_210km_kg_m3": max(r["wam"] for r in rows_210),
        "wam_peak_210km_utc": max(rows_210, key=lambda r: r["wam"])["utc"].isoformat(),
        "absolute_wam_over_msis00_180h": {
            "n_hours": len(abs_ratios_180),
            "min": float(min(abs_ratios_180)),
            "max": float(max(abs_ratios_180)),
            "median_standard": float(np.median(abs_ratios_180)),
            "median_upper_middle": float(abs_ratios_180[len(abs_ratios_180) // 2]),
        },
        "absolute_wam_over_msis00_window_133h": {
            "n_hours": len(abs_ratios_133),
            "min": float(min(abs_ratios_133)),
            "max": float(max(abs_ratios_133)),
            "median_standard": float(np.median(abs_ratios_133)),
        },
        "reference_day_operator_comparison_210km": ref_days_210,
        "altitude_profile_feb1_both_operators": alt_profile_feb1,
        "envelope_200_to_300km_all_3_ref_days": {
            "same_utc_hour_ratio_min": min(same_hour_ratios_200_300),
            "same_utc_hour_ratio_max": max(same_hour_ratios_200_300),
            "whole_day_mean_ratio_min": min(day_mean_ratios_200_300),
            "whole_day_mean_ratio_max": max(day_mean_ratios_200_300),
        },
        "path_energy_bridged_985": bridged_985,
        "path_energy_contiguous_non_gap": contiguous_sum,
    }


# ---------------------------------------------------------------------------
# 2. Paper III & Paper I: Exact Analytic-Horizontal + Adaptive 1D V3-pair,
#    Full-8-Facet 3D Polyhedral QMC V3-full, CLL (0,0) Endpoints, & Full-Vector Orbital Runs
# ---------------------------------------------------------------------------

def _sentman_cd_cl(s: float, mu: float, alpha: float, tw: float, tinf: float) -> tuple[float, float]:
    ell = math.sqrt(max(0.0, 1.0 - mu * mu))
    p = math.exp(-((mu * s) ** 2)) / s
    g = 0.5 / (s * s)
    z = math.erfc(-mu * s)
    r = math.sqrt(max(0.0, 0.5 * (1.0 - alpha) + alpha * tw / (tinf * s * s)))
    cd = p / SQRT_PI + mu * (1.0 + g) * z + 0.5 * mu * r * (mu * SQRT_PI * z + p)
    cl = (ell * g * z + 0.5 * ell * r * (mu * SQRT_PI * z + p)) if ell > 1e-12 else 0.0
    return cd, cl


def _trunc_gauss_moments(lo: float, hi: float, h: float) -> tuple[float, float, float]:
    """Exact 0th, 1st, 2nd moments of pi^(-1/2) * exp(-(x - h)^2) on [lo, hi], erfc-safe."""
    a = lo - h
    b = hi - h
    if a >= 0.0:
        z = 0.5 * (math.erfc(a) - math.erfc(b))
    elif b <= 0.0:
        z = 0.5 * (math.erfc(-b) - math.erfc(-a))
    else:
        z = 0.5 * (math.erf(b) - math.erf(a))
    ea = math.exp(-(a * a)) if abs(a) < 27.0 else 0.0
    eb = math.exp(-(b * b)) if abs(b) < 27.0 else 0.0
    m1 = h * z + (ea - eb) / (2.0 * SQRT_PI)
    m2 = (h * h + 0.5) * z + ((lo + h) * ea - (hi + h) * eb) / (2.0 * SQRT_PI)
    return z, m1, m2


def _cavity_horizontal_expectations(u: float, h: float) -> tuple[float, float]:
    """Exact closed-form horizontal Gaussian expectations E[phi_x * phi_y] and E[c_x * phi_x * phi_y]
    for parallel plates L_x=2.8, W_b=1.4, W_a=8.0, dz=0.9 at normal speed u > 0 and drift h = -s cos(theta).
    """
    a = (2.8 / 0.9) * u
    if a <= 1e-13:
        return 0.0, 0.0
    b = (3.3 / 0.9) * u
    c = (4.7 / 0.9) * u
    z_l, m1_l, m2_l = _trunc_gauss_moments(-a, 0.0, h)
    z_r, m1_r, m2_r = _trunc_gauss_moments(0.0, a, h)
    ex = z_l + z_r + (m1_l - m1_r) / a
    exx = m1_l + m1_r + (m2_l - m2_r) / a
    if b >= 26.0:
        ey = 1.0
    else:
        ec = math.exp(-(c * c)) if c < 27.0 else 0.0
        eb = math.exp(-(b * b)) if b < 27.0 else 0.0
        erfc_diff = math.erfc(b) - math.erfc(c)
        ey = math.erf(b) + (c * erfc_diff + (ec - eb) / SQRT_PI) / (c - b)
    return ex * ey, exx * ey


class ExactCavityPairQuadrature:
    """Exact analytic horizontal Gaussian moments + adaptive mean-centered 1D normal quadrature
    for the parallel bus-top (B) / array-underside (A) cavity (`V3-pair`, Proposition 4).

    Eliminates the fixed u_max=11 truncation defect across ALL speed ratios s > 0 and pitches
    theta in [0, 180] deg, and separates incoming moments (Gamma, p_in, fx_in) that depend
    solely on (s, theta) from diffuse wall recoil r_j(s, alpha, T_w/T_inf).
    """

    def __init__(self, epsabs: float = 1e-13, epsrel: float = 1e-12, limit: int = 200) -> None:
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = limit

    def incoming_moments(self, s: float, pitch_deg: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return ((gamma_b, p_in_b, fx_in_b), (gamma_a, p_in_a, fx_in_a)) depending only on (s, pitch_deg)."""
        rad = math.radians(pitch_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        h = -s * ca
        inv_s = 1.0 / s
        inv_s2 = inv_s * inv_s
        out: list[tuple[float, float, float]] = []
        for face_sign, ratio in ((1.0, 1.0), (-1.0, 0.175)):
            mean = face_sign * s * sa
            upper = max(14.0, mean + 12.0)
            pts = sorted({0.0, upper} | {x for x in (mean - 6.0, mean - 2.0, mean, mean + 2.0, mean + 6.0) if 0.0 < x < upper})
            if mean < -18.0:
                # Incoming Maxwellian tail < exp(-324) ~ 1e-141
                out.append((0.0, 0.0, 0.0))
                continue

            def _int_one(kind: int) -> float:
                def integrand(u: float) -> float:
                    du = u - mean
                    if du * du > 700.0:
                        return 0.0
                    dens = math.exp(-(du * du)) / SQRT_PI
                    blk, bmom = _cavity_horizontal_expectations(u, h)
                    if kind == 0:
                        return (u * inv_s) * dens * (1.0 - ratio * blk)
                    if kind == 1:
                        return (2.0 * u * u * inv_s2) * dens * (1.0 - ratio * blk)
                    return (2.0 * u * inv_s2) * dens * (h - ratio * bmom)

                total = 0.0
                for lo, hi in zip(pts[:-1], pts[1:]):
                    val, _ = quad(integrand, lo, hi, epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit)
                    total += val
                return total

            out.append((_int_one(0), _int_one(1), _int_one(2)))
        return out[0], out[1]

    def evaluate_cavity_pair_vectors(
        self, s: float, pitch_deg: float, alpha: float, tw: float, tinf: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return full dimensionless body-frame force vectors (f_bus_top, f_arr_bot)."""
        (gb, pb, fxb), (ga, pa, fxa) = self.incoming_moments(s, pitch_deg)
        r = math.sqrt(max(0.0, 0.5 * (1.0 - alpha) + alpha * tw / (tinf * s * s)))
        f_bus = np.array([fxb, 0.0, -(pb + r * SQRT_PI * gb)], dtype=float)
        f_arr = np.array([fxa, 0.0, +(pa + r * SQRT_PI * ga)], dtype=float)
        return f_bus, f_arr


class CavityPairSplineSurrogate:
    """Exact 1D log-s cubic spline surrogate for fixed pitch theta over s in [0.25, 30.0].
    Preserves BOTH drag AND normal/lift body-frame vector components with exact runtime r_j.
    """

    def __init__(self, pitch_deg: float, s_min: float = 0.25, s_max: float = 30.0, n_pts: int = 120) -> None:
        self.pitch_deg = float(pitch_deg)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.log_s = np.linspace(math.log(s_min), math.log(s_max), n_pts)
        q = ExactCavityPairQuadrature(epsabs=1e-13, epsrel=1e-12)
        arr = np.empty((n_pts, 6), dtype=np.float64)
        for i, ls in enumerate(self.log_s):
            (gb, pb, fxb), (ga, pa, fxa) = q.incoming_moments(math.exp(float(ls)), self.pitch_deg)
            arr[i, :] = (gb, pb, fxb, ga, pa, fxa)
        self.data = arr
        self._splines = [CubicSpline(self.log_s, arr[:, k], bc_type="not-a-knot") for k in range(6)]

    def to_payload(self) -> dict[str, Any]:
        return {
            "pitch_deg": self.pitch_deg,
            "s_min": self.s_min,
            "s_max": self.s_max,
            "log_s": self.log_s.tolist(),
            "data": self.data.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CavityPairSplineSurrogate":
        obj = cls.__new__(cls)
        obj.pitch_deg = float(payload["pitch_deg"])
        obj.s_min = float(payload["s_min"])
        obj.s_max = float(payload["s_max"])
        obj.log_s = np.asarray(payload["log_s"], dtype=np.float64)
        obj.data = np.asarray(payload["data"], dtype=np.float64)
        obj._splines = [CubicSpline(obj.log_s, obj.data[:, k], bc_type="not-a-knot") for k in range(6)]
        return obj

    def eval_body_vectors(self, s: float, alpha: float, tw: float, tinf: float) -> tuple[np.ndarray, np.ndarray]:
        if s < self.s_min or s > self.s_max:
            raise ValueError(f"Runtime speed ratio s={s:.6g} outside validated spline domain [{self.s_min}, {self.s_max}]")
        ls = math.log(s)
        gb = float(self._splines[0](ls))
        pb = float(self._splines[1](ls))
        fxb = float(self._splines[2](ls))
        ga = float(self._splines[3](ls))
        pa = float(self._splines[4](ls))
        fxa = float(self._splines[5](ls))
        r = math.sqrt(max(0.0, 0.5 * (1.0 - alpha) + alpha * tw / (tinf * s * s)))
        return (
            np.array([fxb, 0.0, -(pb + r * SQRT_PI * gb)], dtype=float),
            np.array([fxa, 0.0, +(pa + r * SQRT_PI * ga)], dtype=float),
        )


def _worker_v3_full_paired_replicated(
    args: tuple[float, dict[str, float], dict[str, float], float, float, list[float], int, int],
) -> dict[str, Any]:
    """Replicated scrambled Sobol paired exact-pair control variate for `V3-full - V3-pair` (`R3-2`).

    Evaluates ONLY the exact incremental polyhedral occlusion beyond `V3-pair`:
      1. Overhanging array occlusion on exterior bus faces (`bus_+x`, `bus_-x`, `bus_+y`, `bus_-y`);
      2. Additional 3D solid-box thickness occlusion (`blk_box & ~blk_flat_top`) on `arr_-z`.
    `bus_+z` has zero incremental difference (array occlusion against z=1.0 is identical in `V3-pair`
    and `V3-full`), cancelling 100% of `V3-pair` Monte Carlo variance on the cavity top face.
    Uses ``n_replicates`` independent fixed-seed scrambled Sobol sequences of ``2**n_pow2`` points.
    """
    pitch_deg, species_nd, masses_kg, tinf, vrel, f_pair_body_list, n_replicates, n_pow2 = args
    if n_replicates < 2:
        raise ValueError(f"n_replicates must be >= 2 for replicated Student-t CI, got {n_replicates}")
    alpha, tw = 0.998, 300.0
    rad = math.radians(pitch_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    u_sc = np.array([ca, 0.0, sa], dtype=float)
    f_pair_body = np.asarray(f_pair_body_list, dtype=float)
    d_pair = float(-(f_pair_body @ u_sc))

    # Only facets with non-zero incremental occlusion between V3-full and V3-pair
    inc_specs = [
        # (name, centroid, normal, t1, t2, L1, L2, area, kind)
        ("bus_+x", (1.4, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 1.4, 0.2, 0.28, "arr"),
        ("bus_-x", (-1.4, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 1.4, 0.2, 0.28, "arr"),
        ("bus_+y", (0.0, 0.7, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.8, 0.2, 0.56, "arr"),
        ("bus_-y", (0.0, -0.7, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.8, 0.2, 0.56, "arr"),
        ("arr_-z", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.8, 8.0, 22.40, "box_extra"),
    ]
    all_names = ["bus_+x", "bus_-x", "bus_+y", "bus_-y", "bus_+z", "bus_-z", "arr_+z", "arr_-z"]

    replicate_seeds: list[int] = []
    rep_delta_drag_N: list[float] = []
    rep_delta_pct: list[float] = []
    rep_full_drag_N: list[float] = []
    rep_lift_N: list[float] = []
    rep_facet_blocked: dict[str, list[float]] = {nm: [] for nm in all_names}

    for rep in range(n_replicates):
        rep_seed = 20261123 + rep * 1009 + int(round(pitch_deg * 10))
        replicate_seeds.append(rep_seed)
        sampler = qmc.Sobol(d=5, scramble=True, rng=rep_seed)
        u = np.clip(sampler.random_base2(m=n_pow2), 1e-12, 1.0 - 1e-12)
        u1 = u[:, 0] - 0.5
        u2 = u[:, 1] - 0.5
        u3 = u[:, 2]
        z_t1 = ndtri(u[:, 3]) / SQRT_2
        z_t2 = ndtri(u[:, 4]) / SQRT_2

        df_body_rep = np.zeros(3, dtype=float)
        blk_rep = {nm: 0.0 for nm in all_names}

        for sym, nd in species_nd.items():
            if nd <= 0.0 or sym not in masses_kg:
                continue
            m_j = masses_kg[sym]
            s_j = vrel * math.sqrt(m_j / (2.0 * KB * tinf))
            q_j = 0.5 * nd * m_j * (vrel ** 2)
            r_j = math.sqrt(max(0.0, 0.5 * (1.0 - alpha) + alpha * tw / (tinf * s_j * s_j)))
            c_mean = -s_j * u_sc

            for name, cen, nrm, t1, t2, L1, L2, area, kind in inc_specs:
                n_v = np.asarray(nrm, dtype=float)
                mu = float(n_v @ u_sc)
                mu = 0.0 if abs(mu) <= 1e-12 else mu
                m_n = s_j * mu
                p_z = float(ndtr(SQRT_2 * m_n))
                if p_z <= 1e-30:
                    continue
                t1_v = np.asarray(t1, dtype=float)
                t2_v = np.asarray(t2, dtype=float)
                m_t1 = float(c_mean @ t1_v)
                m_t2 = float(c_mean @ t2_v)

                z_n = -ndtri(u3 * p_z)
                c_n = m_n + z_n / SQRT_2
                c_t1 = m_t1 + z_t1
                c_t2 = m_t2 + z_t2

                cx = c_t1 * t1_v[0] + c_t2 * t2_v[0] - c_n * n_v[0]
                cy = c_t1 * t1_v[1] + c_t2 * t2_v[1] - c_n * n_v[1]
                cz = c_t1 * t1_v[2] + c_t2 * t2_v[2] - c_n * n_v[2]
                dx, dy, dz = -cx, -cy, -cz

                x0 = cen[0] + u1 * L1 * t1_v[0] + u2 * L2 * t2_v[0]
                y0 = cen[1] + u1 * L1 * t1_v[1] + u2 * L2 * t2_v[1]
                z0 = cen[2] + u1 * L1 * t1_v[2] + u2 * L2 * t2_v[2]

                if kind == "arr":
                    t_hit = np.where(dz > 1e-15, (1.0 - z0) / np.where(dz > 1e-15, dz, 1.0), -1.0)
                    xh = x0 + t_hit * dx
                    yh = y0 + t_hit * dy
                    blk = (t_hit > 1e-9) & (np.abs(xh) <= 1.4) & (np.abs(yh) <= 4.0)
                else:
                    # Paired difference on arr_-z: solid 3D bus box MINUS flat top rectangle z=0.1
                    t_top = np.where(dz < -1e-15, (0.1 - z0) / np.where(dz < -1e-15, dz, -1.0), -1.0)
                    blk_top = (t_top > 1e-9) & (np.abs(x0 + t_top * dx) <= 1.4) & (np.abs(y0 + t_top * dy) <= 0.7)

                    inv_dx = np.where(np.abs(dx) > 1e-15, 1.0 / np.where(np.abs(dx) > 1e-15, dx, 1.0), np.inf)
                    inv_dy = np.where(np.abs(dy) > 1e-15, 1.0 / np.where(np.abs(dy) > 1e-15, dy, 1.0), np.inf)
                    inv_dz = np.where(np.abs(dz) > 1e-15, 1.0 / np.where(np.abs(dz) > 1e-15, dz, 1.0), np.inf)

                    tx1 = np.where(np.abs(dx) > 1e-15, (-1.4 - x0) * inv_dx, np.where(np.abs(x0) <= 1.4, -np.inf, np.inf))
                    tx2 = np.where(np.abs(dx) > 1e-15, (1.4 - x0) * inv_dx, np.where(np.abs(x0) <= 1.4, np.inf, -np.inf))
                    ty1 = np.where(np.abs(dy) > 1e-15, (-0.7 - y0) * inv_dy, np.where(np.abs(y0) <= 0.7, -np.inf, np.inf))
                    ty2 = np.where(np.abs(dy) > 1e-15, (0.7 - y0) * inv_dy, np.where(np.abs(y0) <= 0.7, np.inf, -np.inf))
                    tz1 = np.where(np.abs(dz) > 1e-15, (-0.1 - z0) * inv_dz, np.where((z0 >= -0.1) & (z0 <= 0.1), -np.inf, np.inf))
                    tz2 = np.where(np.abs(dz) > 1e-15, (0.1 - z0) * inv_dz, np.where((z0 >= -0.1) & (z0 <= 0.1), np.inf, -np.inf))

                    tmin = np.maximum(np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2)), np.minimum(tz1, tz2))
                    tmax = np.minimum(np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2)), np.maximum(tz1, tz2))
                    blk_box = (tmax > np.maximum(tmin, 1e-9))
                    blk = blk_box & (~blk_top)

                if not np.any(blk):
                    continue
                w_blk = blk.astype(np.float64)
                inv_n = p_z / float(len(w_blk))
                d_gamma = float(np.sum(w_blk * c_n)) * (inv_n / s_j)
                coef_in = 2.0 * inv_n / (s_j * s_j)
                df_in = np.array([
                    float(np.sum(w_blk * c_n * cx)) * coef_in,
                    float(np.sum(w_blk * c_n * cy)) * coef_in,
                    float(np.sum(w_blk * c_n * cz)) * coef_in,
                ], dtype=float)
                df_re = -(r_j * SQRT_PI * d_gamma) * n_v
                df_blk = area * q_j * (df_in + df_re)
                df_body_rep -= df_blk
                blk_rep[name] += float(-(df_blk @ u_sc))

        f_full_rep = f_pair_body + df_body_rep
        d_full_rep = float(-(f_full_rep @ u_sc))
        dd_rep = d_full_rep - d_pair
        rep_full_drag_N.append(d_full_rep)
        rep_delta_drag_N.append(dd_rep)
        rep_delta_pct.append(100.0 * dd_rep / d_pair)
        rep_lift_N.append(float(-f_full_rep[0] * sa + f_full_rep[2] * ca))
        for nm in all_names:
            rep_facet_blocked[nm].append(blk_rep[nm])

    # Exact Student-t critical value for df = n_replicates - 1 (95% two-sided approximate randomized-QMC CI; R4-7).
    df = n_replicates - 1
    t_crit = float(student_t.ppf(0.975, df=df))

    def _stats(vals: list[float]) -> tuple[float, float]:
        arr = np.asarray(vals, dtype=float)
        if len(arr) < 2:
            raise ValueError(f"Cannot compute replicated standard error with {len(arr)} replicate(s); require >= 2")
        mean = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
        return mean, se

    d_full_m, d_full_se = _stats(rep_full_drag_N)
    dp_m, dp_se = _stats(rep_delta_pct)
    dd_m, dd_se = _stats(rep_delta_drag_N)
    lift_m, lift_se = _stats(rep_lift_N)
    snr: float | None = (abs(dp_m) / dp_se) if dp_se > 1e-30 else None
    rho_tot = sum(nd * masses_kg[sym] for sym, nd in species_nd.items() if sym in masses_kg)
    q_tot = 0.5 * rho_tot * (vrel ** 2)

    extra_by_facet_mean: dict[str, float] = {}
    extra_by_facet_se: dict[str, float] = {}
    for nm in all_names:
        fm, fse = _stats(rep_facet_blocked[nm])
        extra_by_facet_mean[nm] = fm
        extra_by_facet_se[nm] = fse

    return {
        "pitch_deg": pitch_deg,
        "status": "completed",
        "n_replicates": n_replicates,
        "degrees_of_freedom": df,
        "t_critical_95": t_crit,
        "ci_method": "approximate_student_t_df_R_minus_1",
        "ci_qualification": (
            "Approximate two-sided 95% Student-t confidence interval (df = R - 1) across R independent "
            "LMS+shift scrambled Sobol replicates (Owen 1998; scipy.stats.qmc.Sobol); statistical uncertainty "
            "estimate, not a deterministic error bound."
        ),
        "samples_per_replicate": 1 << n_pow2,
        "total_paired_samples": n_replicates * (1 << n_pow2),
        "replicate_seeds": replicate_seeds,
        "replicate_delta_drag_N": rep_delta_drag_N,
        "replicate_full_minus_pair_pct": rep_delta_pct,
        "drag_v3_pair_N": d_pair,
        "drag_v3_full_N": d_full_m,
        "drag_v3_full_se_N": d_full_se,
        "gamma_v3_full_m2_kg": d_full_m / (q_tot * 297.0),
        "lift_z_ram_v3_full_N": lift_m,
        "lift_z_ram_v3_full_se_N": lift_se,
        "delta_full_minus_pair_N": dd_m,
        "delta_full_minus_pair_se_N": dd_se,
        "full_minus_pair_pct": dp_m,
        "full_minus_pair_pct_se": dp_se,
        "full_minus_pair_pct_ci95_halfwidth": t_crit * dp_se,
        "signal_to_error_ratio": snr,
        "resolved_at_3sigma": bool(snr is not None and snr >= 3.0),
        "extra_blocked_drag_by_facet_N": extra_by_facet_mean,
        "extra_blocked_drag_by_facet_se_N": extra_by_facet_se,
    }


def _x_slit_gaussian_moments(m_x: float, t: np.ndarray, width: float = 2.8) -> tuple[np.ndarray, np.ndarray]:
    """Exact E[(1 - t|cx|/W)_+] and E[cx*(1 - t|cx|/W)_+] for cx ~ (1/sqrt(pi)) exp(-(cx - m_x)^2)."""
    lim = width / t
    a = -lim - m_x
    b = lim - m_x
    c = -m_x

    def _g0(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
        return 0.5 * (erf(q) - erf(p))

    def _g1(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
        return (np.exp(-p * p) - np.exp(-q * q)) / (2.0 * SQRT_PI)

    def _g2(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
        return 0.5 * _g0(p, q) + (p * np.exp(-p * p) - q * np.exp(-q * q)) / (2.0 * SQRT_PI)

    ratio = t / width
    g0_n, g1_n, g2_n = _g0(a, c), _g1(a, c), _g2(a, c)
    g0_p, g1_p, g2_p = _g0(c, b), _g1(c, b), _g2(c, b)
    m0 = (1.0 + ratio * m_x) * g0_n + ratio * g1_n + (1.0 - ratio * m_x) * g0_p - ratio * g1_p
    m1 = (
        m_x * (1.0 + ratio * m_x) * g0_n + (1.0 + 2.0 * ratio * m_x) * g1_n + ratio * g2_n
        + m_x * (1.0 - ratio * m_x) * g0_p + (1.0 - 2.0 * ratio * m_x) * g1_p - ratio * g2_p
    )
    return m0, m1


def _evaluate_v3_full_exact_2d_quadrature(
    pitch_deg: float,
    species_nd: dict[str, float],
    masses_kg: dict[str, float],
    tinf: float,
    vrel: float,
    *,
    alpha: float = 0.998,
    tw: float = 300.0,
    nz: int = 48,
    nu: int = 160,
) -> tuple[np.ndarray, dict[str, float]]:
    """Exact deterministic 2D Gauss-Legendre evaluation of V3-full minus V3-pair body force vector.

    Analytically integrates x0, cx, and transverse velocity (cy/v) in closed form using
    Gaussian error functions, leaving smooth 2D integrals over z in [-0.1, 0.1] and vertical
    molecular speed u > 0 for the lateral bus walls (bus_+y, bus_-y) and the incremental
    bus-box vertical-wall occlusion on arr_-z (blk_box & ~blk_top). Zero rare-event sampling
    bias even for heavy species (O, N2) at exact 0 deg edge-on incidence (`F37`).
    """
    rad = math.radians(pitch_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    u_sc = np.array([ca, 0.0, sa], dtype=float)
    gz, wz = np.polynomial.legendre.leggauss(nz)
    z_nodes = 0.1 * gz
    z_wts = 0.5 * wz
    gu, wu = np.polynomial.legendre.leggauss(nu)

    all_names = ("bus_+x", "bus_-x", "bus_+y", "bus_-y", "bus_+z", "bus_-z", "arr_+z", "arr_-z")
    by_facet = {nm: 0.0 for nm in all_names}
    df_body = np.zeros(3, dtype=float)

    for sym, nd in species_nd.items():
        if nd <= 0.0 or sym not in masses_kg:
            continue
        m_j = masses_kg[sym]
        s_j = vrel * math.sqrt(m_j / (2.0 * KB * tinf))
        q_j = 0.5 * nd * m_j * (vrel ** 2)
        r_j = math.sqrt(max(0.0, 0.5 * (1.0 - alpha) + alpha * tw / (tinf * s_j * s_j)))
        m_x = -s_j * ca

        # 1. Lateral bus walls bus_+y and bus_-y (area = 0.56 m^2 each; symmetric in drag/lift, y cancels)
        m_u_bus = s_j * sa
        u_lo = max(0.0, m_u_bus - 10.0)
        u_hi = max(12.0, m_u_bus + 10.0)
        segs = [(0.0, u_lo), (u_lo, u_hi)] if u_lo > 1e-12 else [(0.0, u_hi)]
        f_bus_side = np.zeros(3, dtype=float)
        for a_u, b_u in segs:
            u_n = 0.5 * (b_u - a_u) * gu + 0.5 * (b_u + a_u)
            u_w = 0.5 * (b_u - a_u) * wu
            u_grid = u_n[None, :]
            z_grid = z_nodes[:, None]
            w2 = z_wts[:, None] * u_w[None, :]
            t_hit = (1.0 - z_grid) / u_grid
            v_lim = 3.3 / t_hit
            i1 = -np.expm1(-v_lim * v_lim) / (2.0 * SQRT_PI)
            m0, m1 = _x_slit_gaussian_moments(m_x, t_hit, 2.8)
            pu = np.exp(-((u_grid - m_u_bus) ** 2)) / SQRT_PI
            fx = (2.0 / (s_j * s_j)) * float(np.sum(w2 * pu * i1 * m1))
            fz = (2.0 / (s_j * s_j)) * float(np.sum(w2 * pu * (-u_grid) * i1 * m0))
            f_bus_side[0] += 0.56 * q_j * fx
            f_bus_side[2] += 0.56 * q_j * fz
        d_bus_side = float(-(f_bus_side @ u_sc))
        by_facet["bus_+y"] += d_bus_side
        by_facet["bus_-y"] += d_bus_side
        df_body -= 2.0 * f_bus_side

        # 2. Incremental solid bus-box vertical-side occlusion on arr_-z (blk_box & ~blk_top).
        # Area 22.4 m^2, both +/-y strips contribute 2*(0.2/8.0)*22.4 = 1.12 m^2; u = cz > 0, m_u = -s_j*sa <= 0.
        m_u_arr = -s_j * sa
        u_hi_arr = max(1e-6, math.sqrt(m_u_arr * m_u_arr + 70.0) - abs(m_u_arr))
        u_n = 0.5 * u_hi_arr * (gu + 1.0)
        u_w = 0.5 * u_hi_arr * wu
        u_grid = u_n[None, :]
        z_grid = z_nodes[:, None]
        w2 = z_wts[:, None] * u_w[None, :]
        t_hit = (1.0 - z_grid) / u_grid
        v_lim = 3.3 / t_hit
        i1 = -np.expm1(-v_lim * v_lim) / (2.0 * SQRT_PI)
        m0, m1 = _x_slit_gaussian_moments(m_x, t_hit, 2.8)
        pu = np.exp(-((u_grid - m_u_arr) ** 2)) / SQRT_PI
        fx_arr = 1.12 * q_j * (2.0 / (s_j * s_j)) * float(np.sum(w2 * pu * i1 * m1))
        fz_in_arr = 1.12 * q_j * (2.0 / (s_j * s_j)) * float(np.sum(w2 * pu * u_grid * i1 * m0))
        fz_re_arr = 1.12 * q_j * (r_j * SQRT_PI / s_j) * float(np.sum(w2 * pu * i1 * m0))
        f_arr = np.array([fx_arr, 0.0, fz_in_arr + fz_re_arr], dtype=float)
        by_facet["arr_-z"] += float(-(f_arr @ u_sc))
        df_body -= f_arr

    return df_body, by_facet


def _worker_v3_full_exact_and_importance_mc(task: tuple[Any, ...]) -> dict[str, Any]:
    """Primary exact 2D Gauss-Legendre evaluation (with 48x160 vs 96x320 grid-doubling check)
    cross-checked by an independent importance-sampled 5D polyhedral ray-tracing Monte Carlo
    across R scrambled Sobol replicates (`F37`).
    """
    pitch_deg, species_nd, masses_kg, tinf, vrel, f_pair_body_list, n_replicates, n_pow2 = task
    if int(n_replicates) < 2:
        raise ValueError(f"n_replicates must be >= 2 for independent MC SE, got {n_replicates}")

    rad = math.radians(pitch_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    u_sc = np.array([ca, 0.0, sa], dtype=float)
    f_pair_body = np.asarray(f_pair_body_list, dtype=float)
    d_pair = float(-(f_pair_body @ u_sc))

    # 1. Primary exact deterministic 2D Gauss-Legendre (coarse 48x160 and refined 96x320)
    df_coarse, _ = _evaluate_v3_full_exact_2d_quadrature(
        pitch_deg, species_nd, masses_kg, tinf, vrel, nz=48, nu=160
    )
    df_fine, extra_by_facet_exact = _evaluate_v3_full_exact_2d_quadrature(
        pitch_deg, species_nd, masses_kg, tinf, vrel, nz=96, nu=320
    )
    dd_coarse = float(-(df_coarse @ u_sc))
    dd_exact = float(-(df_fine @ u_sc))
    quad_ref_rel = abs(dd_coarse - dd_exact) / max(abs(dd_exact), 1e-30)
    f_full_exact = f_pair_body + df_fine
    d_full_exact = float(-(f_full_exact @ u_sc))
    lift_exact = float(-f_full_exact[0] * sa + f_full_exact[2] * ca)
    dp_exact_pct = 100.0 * dd_exact / d_pair
    rho_tot = sum(nd * masses_kg[sym] for sym, nd in species_nd.items() if sym in masses_kg)
    q_tot = 0.5 * rho_tot * (vrel ** 2)

    # 2. Independent importance-sampled 5D polyhedral ray-tracing Monte Carlo cross-check
    # Uses direct 3D plane & slab ray-box intersection on bus_+y, bus_-y, and arr_-z with
    # exponential-tilting importance sampling in (cx, u) and active-strip sampling on arr_-z
    # so every species including atomic oxygen O gets abundant ray hits even at 0 deg.
    replicate_seeds = [20261201 + 1009 * r_idx for r_idx in range(n_replicates)]
    rep_mc_deltas: list[float] = []
    rep_mc_pcts: list[float] = []
    stratum_total_hits: dict[str, int] = {}
    k_slope = 2.8 / 0.95

    for seed in replicate_seeds:
        u_raw = np.clip(qmc.Sobol(d=5, scramble=True, rng=seed).random_base2(m=n_pow2), 1e-12, 1.0 - 1e-12)
        u1 = u_raw[:, 0] - 0.5
        u2 = u_raw[:, 1] - 0.5
        u3 = u_raw[:, 2]
        u4 = u_raw[:, 3]
        u5 = u_raw[:, 4]

        df_mc = np.zeros(3, dtype=float)
        for sym, nd in species_nd.items():
            if nd <= 0.0 or sym not in masses_kg:
                continue
            m_j = masses_kg[sym]
            s_j = vrel * math.sqrt(m_j / (2.0 * KB * tinf))
            q_j = 0.5 * nd * m_j * (vrel ** 2)
            r_j = math.sqrt(max(0.0, 0.5 * (1.0 - 0.998) + 0.998 * 300.0 / (tinf * s_j * s_j)))
            m_x = -s_j * ca
            m_z = -s_j * sa

            for side_sign, fname in ((1.0, "bus_+y"), (-1.0, "bus_-y")):
                m_u = -m_z
                if abs(m_x) > k_slope * max(m_u, 0.0):
                    u_star = max(0.5, (m_u + k_slope * (-m_x)) / (1.0 + k_slope * k_slope))
                    cx_star = -k_slope * u_star
                else:
                    u_star = max(0.5, m_u)
                    cx_star = m_x
                p_u_is = float(ndtr(SQRT_2 * u_star))
                u_s = u_star - ndtri(u4 * p_u_is) / SQRT_2
                cx_s = cx_star + ndtri(u5) / SQRT_2
                v_s = -ndtri(u3 * 0.5) / SQRT_2
                log_w = -((u_s - m_u) ** 2 - (u_s - u_star) ** 2) - ((cx_s - m_x) ** 2 - (cx_s - cx_star) ** 2)
                w_is = 0.5 * p_u_is * np.exp(np.clip(log_w, -700.0, 50.0))

                cz_s = -u_s
                cy_s = -side_sign * v_s
                dx, dy, dz = -cx_s, -cy_s, -cz_s
                x0 = u1 * 2.8
                y0 = np.full_like(x0, 0.7 * side_sign)
                z0 = u2 * 0.2

                t_hit = (1.0 - z0) / dz
                xh = x0 + t_hit * dx
                yh = y0 + t_hit * dy
                blk = (t_hit > 1e-9) & (np.abs(xh) <= 1.4) & (np.abs(yh) <= 4.0)
                hits = int(np.sum(blk))
                sk = f"{sym}_{fname}"
                stratum_total_hits[sk] = stratum_total_hits.get(sk, 0) + hits
                if hits > 0:
                    w_eff = np.where(blk, w_is, 0.0) / float(len(w_is))
                    n_v = np.array([0.0, side_sign, 0.0], dtype=float)
                    d_gamma = float(np.sum(w_eff * v_s)) / s_j
                    coef_in = 2.0 / (s_j * s_j)
                    df_in = np.array([
                        float(np.sum(w_eff * v_s * cx_s)) * coef_in,
                        float(np.sum(w_eff * v_s * cy_s)) * coef_in,
                        float(np.sum(w_eff * v_s * cz_s)) * coef_in,
                    ], dtype=float)
                    df_re = -(r_j * SQRT_PI * d_gamma) * n_v
                    df_mc -= 0.56 * q_j * (df_in + df_re)

            # arr_-z incremental box-height occlusion: defensive mixture importance sampling in
            # y_gap = |y0| - 0.7 on [0, 3.3] (90% focused on [0.9/u*cy, 1.1/u*cy] + 10% uniform floor
            # over [0, 3.3]), tested strictly via full 3D ray-box slab intersection (`F37`).
            m_u_a = m_z
            u_star_a = max(0.5, (m_u_a + k_slope * (-m_x)) / (1.0 + k_slope * k_slope))
            cx_star_a = -k_slope * u_star_a
            p_u_a = float(ndtr(SQRT_2 * u_star_a))
            u_s = u_star_a - ndtri(u3 * p_u_a) / SQRT_2
            cx_s = cx_star_a + ndtri(u4) / SQRT_2
            cy_mag = -ndtri(u5 * 0.5) / SQRT_2  # > 0, half-normal factor 0.5
            # Map u2 in [-0.5, 0.5]: sign picks +/-y wing, v_y in (0, 1) draws from defensive mixture
            strip_sign = np.where(u2 >= 0.0, 1.0, -1.0)
            v_y = np.abs(u2) * 2.0
            a_lo = np.minimum(3.3, (0.9 / u_s) * cy_mag)
            a_hi = np.minimum(3.3, (1.1 / u_s) * cy_mag)
            win_w = np.maximum(a_hi - a_lo, 0.0)
            has_win = win_w > 1e-12
            p_focus = np.where(has_win, 0.9, 0.0)
            in_focus_draw = v_y < p_focus
            y_gap = np.where(
                in_focus_draw,
                a_lo + (v_y / np.where(has_win, 0.9, 1.0)) * win_w,
                ((v_y - p_focus) / np.where(has_win, 0.1, 1.0)) * 3.3,
            )
            in_win = has_win & (y_gap >= a_lo) & (y_gap <= a_hi)
            # True mixture density q(y_gap) on [0, 3.3]; target uniform density on [0, 3.3] is 1/3.3.
            q_mix = np.where(has_win, 0.1 / 3.3, 1.0 / 3.3) + np.where(in_win, 0.9 / np.where(has_win, win_w, 1.0), 0.0)
            w_y = (1.0 / 3.3) / q_mix
            y0 = strip_sign * (0.7 + y_gap)
            cy_s = strip_sign * cy_mag  # so dy = -cy_s points inward toward bus |y|<=0.7!
            log_w = -((u_s - m_u_a) ** 2 - (u_s - u_star_a) ** 2) - ((cx_s - m_x) ** 2 - (cx_s - cx_star_a) ** 2)
            w_is = (6.6 / 8.0) * 0.5 * p_u_a * w_y * np.exp(np.clip(log_w, -700.0, 50.0))

            dx, dy, dz = -cx_s, -cy_s, -u_s
            x0 = u1 * 2.8
            z0 = np.ones_like(x0)

            t_top = (0.1 - z0) / dz
            blk_top = (t_top > 1e-9) & (np.abs(x0 + t_top * dx) <= 1.4) & (np.abs(y0 + t_top * dy) <= 0.7)
            inv_dx = np.where(np.abs(dx) > 1e-15, 1.0 / np.where(np.abs(dx) > 1e-15, dx, 1.0), np.inf)
            inv_dy = np.where(np.abs(dy) > 1e-15, 1.0 / np.where(np.abs(dy) > 1e-15, dy, 1.0), np.inf)
            inv_dz = 1.0 / dz
            tx1 = np.where(np.abs(dx) > 1e-15, (-1.4 - x0) * inv_dx, np.where(np.abs(x0) <= 1.4, -np.inf, np.inf))
            tx2 = np.where(np.abs(dx) > 1e-15, (1.4 - x0) * inv_dx, np.where(np.abs(x0) <= 1.4, np.inf, np.inf))
            ty1 = np.where(np.abs(dy) > 1e-15, (-0.7 - y0) * inv_dy, np.where(np.abs(y0) <= 0.7, -np.inf, np.inf))
            ty2 = np.where(np.abs(dy) > 1e-15, (0.7 - y0) * inv_dy, np.where(np.abs(y0) <= 0.7, np.inf, np.inf))
            tz1 = (-0.1 - z0) * inv_dz
            tz2 = (0.1 - z0) * inv_dz
            tmin = np.maximum(np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2)), np.minimum(tz1, tz2))
            tmax = np.minimum(np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2)), np.maximum(tz1, tz2))
            blk = (tmax > np.maximum(tmin, 1e-9)) & (~blk_top)
            hits = int(np.sum(blk))
            sk = f"{sym}_arr_-z"
            stratum_total_hits[sk] = stratum_total_hits.get(sk, 0) + hits
            if hits > 0:
                w_eff = np.where(blk, w_is, 0.0) / float(len(w_is))
                n_v = np.array([0.0, 0.0, -1.0], dtype=float)
                d_gamma = float(np.sum(w_eff * u_s)) / s_j
                coef_in = 2.0 / (s_j * s_j)
                df_in = np.array([
                    float(np.sum(w_eff * u_s * cx_s)) * coef_in,
                    float(np.sum(w_eff * u_s * cy_s)) * coef_in,
                    float(np.sum(w_eff * u_s * u_s)) * coef_in,
                ], dtype=float)
                df_re = -(r_j * SQRT_PI * d_gamma) * n_v
                df_mc -= 22.4 * q_j * (df_in + df_re)

        dd_mc = float(-(df_mc @ u_sc))
        rep_mc_deltas.append(dd_mc)
        rep_mc_pcts.append(100.0 * dd_mc / d_pair)

    df_dof = n_replicates - 1
    t_crit = float(student_t.ppf(0.975, df=df_dof))
    arr_mc = np.asarray(rep_mc_deltas, dtype=float)
    mc_mean = float(np.mean(arr_mc))
    mc_se = float(np.std(arr_mc, ddof=1) / math.sqrt(len(arr_mc)))
    arr_pct = np.asarray(rep_mc_pcts, dtype=float)
    mc_pct_mean = float(np.mean(arr_pct))
    mc_pct_se = float(np.std(arr_pct, ddof=1) / math.sqrt(len(arr_pct)))
    z_score = abs(dd_exact - mc_mean) / max(mc_se, 1e-30)
    min_hits_all = min(stratum_total_hits.values()) if stratum_total_hits else 0

    return {
        "pitch_deg": pitch_deg,
        "status": "completed",
        "primary_method": "deterministic_2d_gauss_legendre",
        "independent_method": "importance_sampled_5d_polyhedral_ray_mc",
        "quadrature_coarse_48x160_delta_N": dd_coarse,
        "quadrature_fine_96x320_delta_N": dd_exact,
        "quadrature_refinement_rel_diff": quad_ref_rel,
        "drag_v3_pair_N": d_pair,
        "drag_v3_full_N": d_full_exact,
        "gamma_v3_full_m2_kg": d_full_exact / (q_tot * 297.0),
        "lift_z_ram_v3_full_N": lift_exact,
        "delta_full_minus_pair_N": dd_exact,
        "full_minus_pair_pct": dp_exact_pct,
        "extra_blocked_drag_by_facet_N": extra_by_facet_exact,
        "n_replicates": n_replicates,
        "degrees_of_freedom": df_dof,
        "t_critical_95": t_crit,
        "samples_per_replicate": 1 << n_pow2,
        "total_paired_samples": n_replicates * (1 << n_pow2),
        "replicate_seeds": replicate_seeds,
        "independent_5d_ray_mc_delta_N": mc_mean,
        "independent_5d_ray_mc_se_N": mc_se,
        "independent_5d_ray_mc_ci95_halfwidth_N": t_crit * mc_se,
        "independent_5d_ray_mc_pct_mean": mc_pct_mean,
        "independent_5d_ray_mc_pct_se": mc_pct_se,
        "independent_5d_ray_mc_pct_ci95_halfwidth": t_crit * mc_pct_se,
        "independent_5d_ray_mc_z_score": z_score,
        "independent_5d_ray_mc_replicate_delta_N": rep_mc_deltas,
        "replicate_full_minus_pair_pct": rep_mc_pcts,
        "full_minus_pair_pct_se": mc_pct_se,
        "full_minus_pair_pct_ci95_halfwidth": t_crit * mc_pct_se,
        "independent_5d_ray_mc_stratum_total_hits": stratum_total_hits,
        "independent_5d_ray_mc_min_stratum_hits": min_hits_all,
        "resolved_at_3sigma": bool(abs(mc_mean) / max(mc_se, 1e-30) >= 3.0 and quad_ref_rel < 1e-10 and z_score <= 3.0),
    }


def _evaluate_full_8facet_polyhedral_qmc(
    species_nd: dict[str, float],
    masses_kg: dict[str, float],
    tinf: float,
    vrel: float,
    pitch_deg: float,
    *,
    alpha: float = 0.998,
    tw: float = 300.0,
    n_pow2: int = 16,
    pair_only_test: bool = False,
) -> dict[str, Any]:
    """Deterministic Sobol QMC control-variate evaluation of full-8-facet 3D polyhedral ray occlusion
    (`V3-full`, fixing R3): exact analytic unoccluded Sentman minus deterministic QMC blocked flux
    against the solar array rectangle (z=1.0, [-1.4,1.4]x[-4,4]) and solid 3D bus box
    ([-1.4,1.4]x[-0.7,0.7]x[-0.1,0.1]).
    """
    proxy = starlink_like_proxy()
    rad = math.radians(pitch_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    u_sc = np.array([ca, 0.0, sa], dtype=float)

    sampler = qmc.Sobol(d=5, scramble=False)
    u = sampler.random_base2(m=n_pow2)
    # Clamp strictly inside (1e-12, 1 - 1e-12) so ndtri is finite on every Sobol point
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    u1 = u[:, 0] - 0.5
    u2 = u[:, 1] - 0.5
    u3 = u[:, 2]
    z_t1 = ndtri(u[:, 3]) / SQRT_2
    z_t2 = ndtri(u[:, 4]) / SQRT_2

    # Define each facet's exact tangent frame (t1, t2, L1, L2) and which occluders can shadow it.
    facet_specs = [
        # (name, centroid, normal, t1, t2, L1, L2, area, can_hit_array, can_hit_bus_box, pair_member)
        ("bus_+x", (1.4, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 1.4, 0.2, 0.28, True, False, False),
        ("bus_-x", (-1.4, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 1.4, 0.2, 0.28, True, False, False),
        ("bus_+y", (0.0, 0.7, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.8, 0.2, 0.56, True, False, False),
        ("bus_-y", (0.0, -0.7, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.8, 0.2, 0.56, True, False, False),
        ("bus_+z", (0.0, 0.0, 0.1), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.8, 1.4, 3.92, True, False, True),
        ("bus_-z", (0.0, 0.0, -0.1), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.8, 1.4, 3.92, False, False, False),
        ("arr_+z", (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.8, 8.0, 22.40, False, False, False),
        ("arr_-z", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.8, 8.0, 22.40, False, True, True),
    ]

    total_f_body = np.zeros(3, dtype=float)
    blocked_drag_by_facet = {spec[0]: 0.0 for spec in facet_specs}

    for sym, nd in species_nd.items():
        if nd <= 0.0 or sym not in masses_kg:
            continue
        m_j = masses_kg[sym]
        s_j = vrel * math.sqrt(m_j / (2.0 * KB * tinf))
        q_j = 0.5 * nd * m_j * (vrel ** 2)
        r_j = math.sqrt(max(0.0, 0.5 * (1.0 - alpha) + alpha * tw / (tinf * s_j * s_j)))
        c_mean = -s_j * u_sc

        for name, cen, nrm, t1, t2, L1, L2, area, hit_arr, hit_box, is_pair in facet_specs:
            n_v = np.asarray(nrm, dtype=float)
            mu = float(n_v @ u_sc)
            mu = 0.0 if abs(mu) <= 1e-12 else mu
            ell = math.sqrt(max(0.0, 1.0 - mu * mu))
            cd, cl = _sentman_cd_cl(s_j, mu, alpha, tw, tinf)
            if ell > 1e-12:
                w_hat = (n_v - mu * u_sc) / ell
                f_sent = area * q_j * (-cd * u_sc - cl * w_hat)
            else:
                f_sent = area * q_j * (-cd * u_sc)
            total_f_body += f_sent

            if pair_only_test and not is_pair:
                continue
            if not (hit_arr or hit_box):
                continue

            m_n = s_j * mu
            p_z = float(ndtr(SQRT_2 * m_n))
            if p_z <= 1e-30:
                continue
            t1_v = np.asarray(t1, dtype=float)
            t2_v = np.asarray(t2, dtype=float)
            m_t1 = float(c_mean @ t1_v)
            m_t2 = float(c_mean @ t2_v)

            z_n = -ndtri(u3 * p_z)
            c_n = m_n + z_n / SQRT_2
            c_t1 = m_t1 + z_t1
            c_t2 = m_t2 + z_t2

            # Body-frame molecular velocity c_tilde and backward ray direction d = -c_tilde
            cx = c_t1 * t1_v[0] + c_t2 * t2_v[0] - c_n * n_v[0]
            cy = c_t1 * t1_v[1] + c_t2 * t2_v[1] - c_n * n_v[1]
            cz = c_t1 * t1_v[2] + c_t2 * t2_v[2] - c_n * n_v[2]
            dx, dy, dz = -cx, -cy, -cz

            x0 = cen[0] + u1 * L1 * t1_v[0] + u2 * L2 * t2_v[0]
            y0 = cen[1] + u1 * L1 * t1_v[1] + u2 * L2 * t2_v[1]
            z0 = cen[2] + u1 * L1 * t1_v[2] + u2 * L2 * t2_v[2]

            if hit_arr:
                # Backward ray hits z=1.0 array iff dz > 0 and hit (xh, yh) in [-1.4, 1.4] x [-4.0, 4.0]
                t_hit = np.where(dz > 1e-15, (1.0 - z0) / np.where(dz > 1e-15, dz, 1.0), -1.0)
                xh = x0 + t_hit * dx
                yh = y0 + t_hit * dy
                blk = (t_hit > 1e-9) & (np.abs(xh) <= 1.4) & (np.abs(yh) <= 4.0)
            else:
                if pair_only_test:
                    # Flat top rectangle z=0.1 only
                    t_hit = np.where(dz < -1e-15, (0.1 - z0) / np.where(dz < -1e-15, dz, -1.0), -1.0)
                    xh = x0 + t_hit * dx
                    yh = y0 + t_hit * dy
                    blk = (t_hit > 1e-9) & (np.abs(xh) <= 1.4) & (np.abs(yh) <= 0.7)
                else:
                    # Full 3D solid bus box [-1.4, 1.4] x [-0.7, 0.7] x [-0.1, 0.1] via exact slab intersection
                    inv_dx = np.where(np.abs(dx) > 1e-15, 1.0 / np.where(np.abs(dx) > 1e-15, dx, 1.0), np.inf)
                    inv_dy = np.where(np.abs(dy) > 1e-15, 1.0 / np.where(np.abs(dy) > 1e-15, dy, 1.0), np.inf)
                    inv_dz = np.where(np.abs(dz) > 1e-15, 1.0 / np.where(np.abs(dz) > 1e-15, dz, 1.0), np.inf)

                    tx1 = np.where(np.abs(dx) > 1e-15, (-1.4 - x0) * inv_dx, np.where(np.abs(x0) <= 1.4, -np.inf, np.inf))
                    tx2 = np.where(np.abs(dx) > 1e-15, (1.4 - x0) * inv_dx, np.where(np.abs(x0) <= 1.4, np.inf, -np.inf))
                    ty1 = np.where(np.abs(dy) > 1e-15, (-0.7 - y0) * inv_dy, np.where(np.abs(y0) <= 0.7, -np.inf, np.inf))
                    ty2 = np.where(np.abs(dy) > 1e-15, (0.7 - y0) * inv_dy, np.where(np.abs(y0) <= 0.7, np.inf, -np.inf))
                    tz1 = np.where(np.abs(dz) > 1e-15, (-0.1 - z0) * inv_dz, np.where((z0 >= -0.1) & (z0 <= 0.1), -np.inf, np.inf))
                    tz2 = np.where(np.abs(dz) > 1e-15, (0.1 - z0) * inv_dz, np.where((z0 >= -0.1) & (z0 <= 0.1), np.inf, -np.inf))

                    tmin = np.maximum(np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2)), np.minimum(tz1, tz2))
                    tmax = np.minimum(np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2)), np.maximum(tz1, tz2))
                    blk = (tmax > np.maximum(tmin, 1e-9))

            if not np.any(blk):
                continue
            w_blk = blk.astype(np.float64)
            inv_n = p_z / float(len(w_blk))
            d_gamma = float(np.sum(w_blk * c_n)) * (inv_n / s_j)
            coef_in = 2.0 * inv_n / (s_j * s_j)
            df_in = np.array([
                float(np.sum(w_blk * c_n * cx)) * coef_in,
                float(np.sum(w_blk * c_n * cy)) * coef_in,
                float(np.sum(w_blk * c_n * cz)) * coef_in,
            ], dtype=float)
            df_re = -(r_j * SQRT_PI * d_gamma) * n_v
            df_blk = area * q_j * (df_in + df_re)
            total_f_body -= df_blk
            blocked_drag_by_facet[name] += float(-(df_blk @ u_sc))

    drag_full = float(-(total_f_body @ u_sc))
    lift_z_ram = float(-total_f_body[0] * sa + total_f_body[2] * ca)
    rho_tot = sum(nd * masses_kg[sym] for sym, nd in species_nd.items() if sym in masses_kg)
    q_tot = 0.5 * rho_tot * (vrel ** 2)
    return {
        "pitch_deg": pitch_deg,
        "n_sobol_samples": 1 << n_pow2,
        "drag_v3_full_N": drag_full,
        "gamma_v3_full_m2_kg": drag_full / (q_tot * 297.0),
        "lift_z_ram_v3_full_N": lift_z_ram,
        "blocked_drag_by_facet_N": blocked_drag_by_facet,
    }


def evaluate_proxy_three_models(
    species_nd: dict[str, float],
    masses_kg: dict[str, float],
    tinf: float,
    vrel: float,
    pitch_deg: float,
    *,
    alpha: float = 0.998,
    tw: float = 300.0,
    kin_quad: ExactCavityPairQuadrature | None = None,
) -> dict[str, Any]:
    proxy = starlink_like_proxy()
    rad = math.radians(pitch_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    u_sc = np.array([ca, 0.0, sa], dtype=float)
    u_flow = (-ca, 0.0, -sa)

    vis_v1_raw = self_shadowing_visibility(proxy.facets, u_flow)

    f_v1 = np.zeros(3, dtype=float)
    f_v2 = np.zeros(3, dtype=float)
    f_v3 = np.zeros(3, dtype=float)

    for sym, nd in species_nd.items():
        if nd <= 0.0 or sym not in masses_kg:
            continue
        m_j = masses_kg[sym]
        s_j = vrel * math.sqrt(m_j / (2.0 * KB * tinf))
        q_j = 0.5 * nd * m_j * (vrel ** 2)
        kin_pair = kin_quad.evaluate_cavity_pair_vectors(s_j, pitch_deg, alpha, tw, tinf) if kin_quad else None

        for idx, fac in enumerate(proxy.facets):
            n_vec = np.asarray(fac.normal_body, dtype=float)
            mu = float(n_vec @ u_sc)
            if abs(mu) <= 1e-12:
                mu = 0.0
            ell = math.sqrt(max(0.0, 1.0 - mu * mu))
            cd, cl = _sentman_cd_cl(s_j, mu, alpha, tw, tinf)
            area = fac.area_m2
            if ell > 1e-12:
                w_hat = (n_vec - mu * u_sc) / ell
                f_sent = area * q_j * (-cd * u_sc - cl * w_hat)
            else:
                f_sent = area * q_j * (-cd * u_sc)

            is_bus_top = (abs(fac.centroid_body[2] - 0.1) < 1e-6 and fac.normal_body[2] > 0.9)
            is_arr_bot = (abs(fac.centroid_body[2] - 1.0) < 1e-6 and fac.normal_body[2] < -0.9)

            chi_v1 = 1.0 if (mu < 0.0 or vis_v1_raw[idx]) else 0.0
            f_v1 += chi_v1 * f_sent

            if is_bus_top and sa > 1e-12:
                chi_v2 = min(1.0, 0.9 * abs(ca / sa) / 2.8)
            elif is_arr_bot and sa < -1e-12:
                chi_v2 = 1.0 - 0.175 * max(0.0, 1.0 - 0.9 * abs(ca / sa) / 2.8)
            else:
                chi_v2 = 1.0
            f_v2 += chi_v2 * f_sent

            if kin_pair is not None and is_bus_top:
                f_v3 += area * q_j * kin_pair[0]
            elif kin_pair is not None and is_arr_bot:
                f_v3 += area * q_j * kin_pair[1]
            else:
                f_v3 += f_sent

    d_v1 = float(-(f_v1 @ u_sc))
    d_v2 = float(-(f_v2 @ u_sc))
    d_v3 = float(-(f_v3 @ u_sc)) if kin_quad else None
    l_v3 = float(-f_v3[0] * sa + f_v3[2] * ca) if kin_quad else None

    rho_tot = sum(nd * masses_kg[sym] for sym, nd in species_nd.items() if sym in masses_kg)
    q_tot = 0.5 * rho_tot * (vrel ** 2)
    mass_nom = 297.0
    return {
        "pitch_deg": pitch_deg,
        "drag_v1_legacy_centroid_N": d_v1,
        "drag_v2_collimated_area_N": d_v2,
        "drag_v3_kinetic_cavity_N": d_v3,
        "lift_z_ram_v3_N": l_v3,
        "body_force_v3_N": f_v3.tolist() if kin_quad else None,
        "gamma_v1_m2_kg": d_v1 / (q_tot * mass_nom),
        "gamma_v2_m2_kg": d_v2 / (q_tot * mass_nom),
        "gamma_v3_m2_kg": (d_v3 / (q_tot * mass_nom)) if d_v3 is not None else None,
        "v1_visibility_mask": list(vis_v1_raw),
    }


def _worker_orbital_case(args: tuple[float, str, str, float, dict[str, float], dict[str, Any] | None]) -> dict[str, Any]:
    pitch_deg, mode, forcing, max_step_s, msis_masses, spline_payload = args
    store = RawStore(REPO / "data/raw")
    kp = sw.parse_gfz_kp_ap(store.read("gfz", "kp_2022_extended", ".json"), store.read("gfz", "kp_2022_extended_ap", ".json"))
    f107 = sw.parse_celestrak_f107(store.read("celestrak", "sw_last5years", ".csv"))
    fast_msis = FastMsis00Evaluator(kp, f107, quiet_ap=4.0 if forcing == "quiet" else None)

    surrogate = CavityPairSplineSurrogate.from_payload(spline_payload) if spline_payload is not None else None
    proxy = starlink_like_proxy()
    pos0, vel0 = EV.insertion_state_vectors()
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH.value])
    mass_kg = EV.MASS_KG_NOMINAL
    rad = math.radians(pitch_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    u_sc_body = np.array([ca, 0.0, sa], dtype=float)
    vis_v1 = self_shadowing_visibility(proxy.facets, (-ca, 0.0, -sa))
    sigma_n2 = math.pi * (3.64e-10 ** 2)

    sp_order = ["N2", "O2", "O", "He", "H", "Ar", "N", "anomalous_O"]
    m_arr = np.array([msis_masses[s] for s in sp_order], dtype=np.float64)
    s_seen_min = math.inf
    s_seen_max = 0.0

    def rhs(t_s: float, y: np.ndarray) -> np.ndarray:
        nonlocal s_seen_min, s_seen_max
        r = y[:3]
        v = y[3:]
        alt_m = geodetic_altitude(r)
        acc_grav = -MU_EARTH.value * r / (np.linalg.norm(r) ** 3) + j2_acceleration(r)
        if alt_m < REENTRY_ALTITUDE_M:
            return np.concatenate([v, acc_grav])

        unix_s = _DEPLOY_TS + float(t_s)
        gmst = _gmst_from_unix(unix_s)
        lat_d, lon_d = geodetic_latlon(r, gmst)
        raw = fast_msis.eval_unix(unix_s, alt_m, lat_d, lon_d)
        nds = np.asarray(raw[1:9], dtype=np.float64)
        nds = np.where(np.isnan(nds), 0.0, nds)
        tk = float(raw[10])

        v_rel = v - np.cross(omega_vec, r)
        speed = float(np.linalg.norm(v_rel))
        x_ram = v_rel / speed
        r_hat = r / np.linalg.norm(r)
        z_raw = r_hat - float(r_hat @ x_ram) * x_ram
        z_ram = z_raw / np.linalg.norm(z_raw)

        f_body = np.zeros(3, dtype=float)
        for k_sp, sym in enumerate(sp_order):
            nd = float(nds[k_sp])
            if nd <= 0.0:
                continue
            m_j = float(m_arr[k_sp])
            s_j = speed * math.sqrt(m_j / (2.0 * KB * tk))
            if s_j < s_seen_min:
                s_seen_min = s_j
            if s_j > s_seen_max:
                s_seen_max = s_j
            q_j = 0.5 * nd * m_j * (speed ** 2)

            v3_vecs = surrogate.eval_body_vectors(s_j, 0.998, 300.0, tk) if (mode == "v3_kinetic_cavity" and surrogate is not None) else None

            for idx, fac in enumerate(proxy.facets):
                n_vec = np.asarray(fac.normal_body, dtype=float)
                mu = float(n_vec @ u_sc_body)
                mu = 0.0 if abs(mu) <= 1e-12 else mu
                ell = math.sqrt(max(0.0, 1.0 - mu * mu))
                is_bus_top = (abs(fac.centroid_body[2] - 0.1) < 1e-6 and fac.normal_body[2] > 0.9)
                is_arr_bot = (abs(fac.centroid_body[2] - 1.0) < 1e-6 and fac.normal_body[2] < -0.9)

                # FULL 3D VECTOR (drag AND normal/lift) for V3-pair cavity faces (fixes R-I-1 & R6!)
                if v3_vecs is not None and is_bus_top:
                    f_body += fac.area_m2 * q_j * v3_vecs[0]
                    continue
                if v3_vecs is not None and is_arr_bot:
                    f_body += fac.area_m2 * q_j * v3_vecs[1]
                    continue

                cd, cl = _sentman_cd_cl(s_j, mu, 0.998, 300.0, tk)
                if mode == "v1_legacy_centroid":
                    chi = 1.0 if (mu < 0.0 or vis_v1[idx]) else 0.0
                else:
                    chi = min(1.0, 0.9 * abs(ca / sa) / 2.8) if (is_bus_top and sa > 1e-12) else 1.0
                if chi == 0.0:
                    continue
                if ell > 1e-12:
                    w_hat = (n_vec - mu * u_sc_body) / ell
                    f_body += fac.area_m2 * q_j * chi * (-cd * u_sc_body - cl * w_hat)
                else:
                    f_body += fac.area_m2 * q_j * chi * (-cd * u_sc_body)

        f_x_ram = float(f_body[0] * ca + f_body[2] * sa)
        f_z_ram = float(-f_body[0] * sa + f_body[2] * ca)
        acc_aero = (f_x_ram * x_ram + f_z_ram * z_ram) / mass_kg
        return np.concatenate([v, acc_grav + acc_aero])

    def ev_170(t_s: float, y: np.ndarray) -> float:
        return geodetic_altitude(y[:3]) - 170_000.0
    ev_170.direction = -1.0  # type: ignore[attr-defined]
    ev_170.terminal = False  # type: ignore[attr-defined]

    def ev_100(t_s: float, y: np.ndarray) -> float:
        return geodetic_altitude(y[:3]) - 100_000.0
    ev_100.direction = -1.0  # type: ignore[attr-defined]
    ev_100.terminal = True  # type: ignore[attr-defined]

    y0 = np.concatenate([np.asarray(pos0, dtype=float), np.asarray(vel0, dtype=float)])
    sol = solve_ivp(rhs, (0.0, 25.0 * 86400.0), y0, method="DOP853", rtol=1e-9, atol=1e-6,
                    max_step=max_step_s, events=[ev_170, ev_100], dense_output=True)
    t170_s = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    t100_s = float(sol.t_events[1][0]) if len(sol.t_events[1]) else None

    # Minimum Knudsen number sampled on the accepted dense-output trajectory over t <= t_170
    # (sampling spacing <= 30 s plus accepted solver knots; not a continuous lower-bound proof).
    min_kn_accepted = math.inf
    min_kn_time_d = None
    min_kn_alt_km = None
    n_kn_samples = 0
    if t170_s is not None and sol.sol is not None:
        n_dense = max(500, int(math.ceil(t170_s / 30.0)) + 1)
        t_grid_kn = np.unique(np.concatenate([
            np.linspace(0.0, t170_s, n_dense),
            sol.t[sol.t <= t170_s],
            np.array([t170_s]),
        ]))
        for ts_eval in t_grid_kn:
            st_y = sol.sol(float(ts_eval))
            r_eval = st_y[:3]
            alt_eval = geodetic_altitude(r_eval)
            if alt_eval < 169_999.9:
                continue
            n_kn_samples += 1
            unix_s = _DEPLOY_TS + float(ts_eval)
            lat_d, lon_d = geodetic_latlon(r_eval, _gmst_from_unix(unix_s))
            raw = fast_msis.eval_unix(unix_s, alt_eval, lat_d, lon_d)
            nds = np.asarray(raw[1:9], dtype=np.float64)
            nds = np.where(np.isnan(nds), 0.0, nds)
            n_tot = float(np.sum(nds))
            kn_val = knudsen_number(hard_sphere_mean_free_path(n_tot, sigma_n2), 8.0)
            if kn_val is not None and kn_val < min_kn_accepted:
                min_kn_accepted = kn_val
                min_kn_time_d = float(ts_eval / 86400.0)
                min_kn_alt_km = float(alt_eval / 1000.0)

    return {
        "pitch_deg": pitch_deg,
        "mode": mode,
        "forcing": forcing,
        "max_step_s": max_step_s,
        "t170_days": (t170_s / 86400.0) if t170_s is not None else None,
        "t100_days": (t100_s / 86400.0) if t100_s is not None else None,
        "min_kn_sampled_pre_t170": min_kn_accepted,
        "min_kn_accepted_pre_t170": min_kn_accepted,
        "kn_sample_count": n_kn_samples,
        "kn_max_spacing_s": 30.0,
        "min_kn_time_days": min_kn_time_d,
        "min_kn_alt_km": min_kn_alt_km,
        "runtime_s_min": s_seen_min,
        "runtime_s_max": s_seen_max,
    }


def compute_cll_specular_endpoint_residuals(species_nd: dict[str, float], masses_kg: dict[str, float], tinf: float, vrel: float) -> list[dict[str, Any]]:
    """Exact symbolic/arithmetic evaluation of printed Walker-Mehta-Koller CLL Eq. (8) at
    (alpha_N, sigma_T) = (0, 0) for mu = 0.5, T_w = 300 K across all 6 CLL species (`R1`, `F25`).
    """
    tw = 300.0
    mu = 0.5
    rows = []
    for sym in CLL_SUPPORTED_SPECIES:
        m_j = masses_kg[sym]
        s = vrel * math.sqrt(m_j / (2.0 * KB * tinf))
        c = cll_constants(sym, 0.0)
        e_term = math.exp(-((mu * s) ** 2))
        z_term = math.erfc(-mu * s)
        g1 = (mu * s * e_term + 0.5 * SQRT_PI * (1.0 + 2.0 * (mu * s) ** 2) * z_term) / SQRT_PI
        g2 = (e_term + SQRT_PI * mu * s * z_term) / SQRT_PI
        specular_cp = 2.0 * g1 / (s * s)
        fitted_extra_cp = (SQRT_PI * c.zeta / (2.0 * (s ** 3))) * math.exp(-c.beta) * ((tw / tinf) ** (c.delta + 0.5)) * g2
        rows.append({
            "species": sym,
            "speed_ratio": s,
            "mu": mu,
            "beta": c.beta,
            "gamma": c.gamma,
            "delta": c.delta,
            "zeta": c.zeta,
            "exact_specular_cp": specular_cp,
            "fitted_extra_cp": fitted_extra_cp,
            "fitted_total_cp": specular_cp + fitted_extra_cp,
            "excess_over_specular_pct": 100.0 * fitted_extra_cp / specular_cp,
        })
    return rows


def compute_paper3_and_paired_orbital(*, max_workers: int = 1) -> dict[str, Any]:
    gsi_dir = REPO / "results/species_resolved_gsi"
    sp_rows = list(csv.DictReader((gsi_dir / "species_force_summary.csv").open()))
    manifest = json.loads((gsi_dir / "run_manifest.json").read_text())
    tinf = float(manifest["accommodation"]["factorial_isotherm_temperature_k"])
    vrel = float(manifest["flow_convention"]["relative_speed_m_s"])
    species_nd = {r["species"]: float(r["number_density_m3"]) for r in sp_rows}
    masses_kg = {r["species"]: float(r["molecular_mass_kg"]) for r in sp_rows}

    kin_quad = ExactCavityPairQuadrature(epsabs=1e-13, epsrel=1e-12)
    kin_quad_tight = ExactCavityPairQuadrature(epsabs=1e-14, epsrel=1e-13)

    # 1. FULL 181 integer attitudes (0..180 deg) + near-zero probes (fixes R5!)
    near_zero_probes = [1e-5, 1e-3, 1e-2, 0.1]
    all_181_angles = [float(a) for a in range(181)]
    probe_angles = sorted(set(all_181_angles + near_zero_probes))

    full_185 = [evaluate_proxy_three_models(species_nd, masses_kg, tinf, vrel, a, kin_quad=kin_quad) for a in probe_angles]
    by_a = {r["pitch_deg"]: r for r in full_185}

    cardinal_angles = [0.0, 1e-5, 1e-3, 1e-2, 0.1, 1.0, 15.0, 30.0, 32.0, 33.0, 45.0, 60.0, 75.0, 90.0]
    sweep_cardinal = [by_a[a] for a in cardinal_angles]

    # Symmetry & positivity check across all 181 integer attitudes
    d_v1_181 = [by_a[float(a)]["drag_v1_legacy_centroid_N"] for a in range(181)]
    d_v2_181 = [by_a[float(a)]["drag_v2_collimated_area_N"] for a in range(181)]
    d_v3_181 = [by_a[float(a)]["drag_v3_kinetic_cavity_N"] for a in range(181)]
    max_sym_diff_v3 = max(abs(by_a[float(a)]["drag_v3_kinetic_cavity_N"] - by_a[float(180 - a)]["drag_v3_kinetic_cavity_N"]) for a in range(91))

    # 2. Quadrature refinement & high-speed limit checks (fixes R2!)
    quad_refinement = []
    for a in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
        ref_tight = evaluate_proxy_three_models(species_nd, masses_kg, tinf, vrel, a, kin_quad=kin_quad_tight)
        d_std = by_a[a]["drag_v3_kinetic_cavity_N"]
        d_tight = ref_tight["drag_v3_kinetic_cavity_N"]
        quad_refinement.append({
            "pitch_deg": a,
            "drag_v3_eps12_N": d_std,
            "drag_v3_eps13_N": d_tight,
            "rel_diff": abs(d_std - d_tight) / abs(d_tight),
        })

    high_speed_45deg = []
    ca45 = math.sqrt(0.5)
    r_inf_45 = math.sqrt(0.5 * (1.0 - 0.998))
    true_s_inf_bus_cd = 2.0 * ca45 * (1.0 + 0.5 * SQRT_PI * ca45 * r_inf_45) * (0.9 / 2.8)
    for s_val in (10.0, 20.0, 40.0):
        fb, _ = kin_quad.evaluate_cavity_pair_vectors(s_val, 45.0, 0.998, 300.0, tinf)
        cd_v3_bus = float(-(fb[0] * ca45 + fb[2] * ca45))
        cd_collimated_bus = _sentman_cd_cl(s_val, ca45, 0.998, 300.0, tinf)[0] * (0.9 / 2.8)
        high_speed_45deg.append({
            "speed_ratio": s_val,
            "v3_pair_bus_cd": cd_v3_bus,
            "collimated_v2_bus_cd": cd_collimated_bus,
            "rel_diff_pct": 100.0 * (cd_v3_bus / cd_collimated_bus - 1.0),
            "true_s_inf_limit_cd": true_s_inf_bus_cd,
            "v3_pair_excess_over_true_s_inf_pct": 100.0 * (cd_v3_bus / true_s_inf_bus_cd - 1.0),
            "collimated_v2_excess_over_true_s_inf_pct": 100.0 * (cd_collimated_bus / true_s_inf_bus_cd - 1.0),
        })

    # 3. Full-8-facet 3D polyhedral ray-occlusion kinetic check (`V3-full`, fixing F37!):
    # Primary: exact deterministic 2D Gauss-Legendre (48x160 vs 96x320 grid-doubling check < 1e-12 rel)
    # Cross-check: independent importance-sampled 5D polyhedral ray-tracing MC (R=16 x 2^15 per angle)
    # across ALL 7 cardinal angles {0, 15, 30, 45, 60, 75, 90} deg, eliminating crude-Sobol rare-event bias!
    qmc_val_0 = _evaluate_full_8facet_polyhedral_qmc(species_nd, masses_kg, tinf, vrel, 0.0, n_pow2=16, pair_only_test=True)
    qmc_val_30 = _evaluate_full_8facet_polyhedral_qmc(species_nd, masses_kg, tinf, vrel, 30.0, n_pow2=16, pair_only_test=True)
    cardinal_7 = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    paired_qmc_tasks = [
        (a, species_nd, masses_kg, tinf, vrel, by_a[a]["body_force_v3_N"], 16, 15)
        for a in cardinal_7
    ]
    v3_full_sweep = _map_tasks(_worker_v3_full_exact_and_importance_mc, paired_qmc_tasks, max_workers=max_workers)
    v3_full_by_a = {r["pitch_deg"]: r for r in v3_full_sweep}

    # 4. CLL (alpha_N=0, sigma_T=0) endpoint residuals (fixes R1!)
    cll_endpoint_residuals = compute_cll_specular_endpoint_residuals(species_nd, masses_kg, tinf, vrel)

    jump_v1_pct = 100.0 * (by_a[1e-5]["drag_v1_legacy_centroid_N"] / by_a[0.0]["drag_v1_legacy_centroid_N"] - 1.0)
    jump_v2_pct = 100.0 * (by_a[1e-5]["drag_v2_collimated_area_N"] / by_a[0.0]["drag_v2_collimated_area_N"] - 1.0)
    jump_v3_pct = 100.0 * (by_a[1e-5]["drag_v3_kinetic_cavity_N"] / by_a[0.0]["drag_v3_kinetic_cavity_N"] - 1.0)

    # 5. 145-h temperature vs composition one-at-a-time decomposition (legacy V1, unchanged physics)
    ts_rows = list(csv.DictReader((gsi_dir / "species_force_timeseries.csv").open()))
    peak_w = {r["species"]: float(r["mass_fraction"]) for r in sp_rows}
    proxy = starlink_like_proxy()
    u_30 = np.array([math.cos(math.radians(30.0)), 0.0, math.sin(math.radians(30.0))])
    vis_30 = self_shadowing_visibility(proxy.facets, (-u_30[0], 0.0, -u_30[2]))

    def _per_unit_mass_coeff(temp_k: float) -> dict[str, float]:
        out_c = {}
        for sym, m_j in masses_kg.items():
            s_j = vrel * math.sqrt(m_j / (2.0 * KB * temp_k))
            tot_cd_a = 0.0
            for idx, fac in enumerate(proxy.facets):
                mu = float(np.asarray(fac.normal_body) @ u_30)
                mu = 0.0 if abs(mu) <= 1e-12 else mu
                if mu >= 0.0 and not vis_30[idx]:
                    continue
                cd, _ = _sentman_cd_cl(s_j, mu, 0.998, 300.0, temp_k)
                tot_cd_a += fac.area_m2 * cd
            out_c[sym] = tot_cd_a
        return out_c

    k_peak = _per_unit_mass_coeff(tinf)
    series_joint, series_t_only, series_c_only = [], [], []
    temps_145 = []
    for row in ts_rows:
        tk = float(row["temperature_k"])
        temps_145.append(tk)
        kt = _per_unit_mass_coeff(tk)
        wt_raw = {k: float(row[f"fraction_{k}"]) / kt[k] for k in kt}
        w_sum = sum(wt_raw.values())
        wt = {k: v / w_sum for k, v in wt_raw.items()}
        series_joint.append(sum(wt[k] * kt[k] for k in kt))
        series_t_only.append(sum(peak_w[k] * kt[k] for k in kt))
        series_c_only.append(sum(wt[k] * k_peak[k] for k in kt))

    def _rel_width_pct(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        return float(100.0 * np.ptp(arr) / np.median(arr))

    # 6. Build validated 1D log-s cubic spline surrogates for 0 deg, 30 deg, 90 deg & verify FULL VECTOR error
    # across a dense 101-temperature grid in [550, 1400 K] (fixing R3-4; fixed diagnostic velocity & composition)!
    msis_masses = {s: masses_kg[s] for s in ("N2", "O2", "O", "He", "H", "Ar", "N", "anomalous_O")}
    splines = {p: CavityPairSplineSurrogate(p, s_min=0.25, s_max=30.0, n_pts=120) for p in (0.0, 30.0, 90.0)}

    dense_temps_101 = np.linspace(550.0, 1400.0, 101).tolist()
    surrogate_vector_validation = []
    for p in (0.0, 30.0, 90.0):
        rad = math.radians(p)
        ca, sa = math.cos(rad), math.sin(rad)
        u_sc = np.array([ca, 0.0, sa], dtype=float)

        def _eval_vec_err(tk_test: float) -> tuple[np.ndarray, np.ndarray, float]:
            f_dir = np.zeros(3, dtype=float)
            f_sur = np.zeros(3, dtype=float)
            for sym, nd in species_nd.items():
                if sym not in msis_masses or nd <= 0.0:
                    continue
                m_j = msis_masses[sym]
                s_j = vrel * math.sqrt(m_j / (2.0 * KB * tk_test))
                q_j = 0.5 * nd * m_j * (vrel ** 2)
                d_b, d_a = kin_quad.evaluate_cavity_pair_vectors(s_j, p, 0.998, 300.0, tk_test)
                s_b, s_a = splines[p].eval_body_vectors(s_j, 0.998, 300.0, tk_test)
                for fac in proxy.facets:
                    n_vec = np.asarray(fac.normal_body, dtype=float)
                    mu = float(n_vec @ u_sc)
                    mu = 0.0 if abs(mu) <= 1e-12 else mu
                    ell = math.sqrt(max(0.0, 1.0 - mu * mu))
                    is_bt = (abs(fac.centroid_body[2] - 0.1) < 1e-6 and fac.normal_body[2] > 0.9)
                    is_ab = (abs(fac.centroid_body[2] - 1.0) < 1e-6 and fac.normal_body[2] < -0.9)
                    if is_bt:
                        f_dir += fac.area_m2 * q_j * d_b
                        f_sur += fac.area_m2 * q_j * s_b
                    elif is_ab:
                        f_dir += fac.area_m2 * q_j * d_a
                        f_sur += fac.area_m2 * q_j * s_a
                    else:
                        cd, cl = _sentman_cd_cl(s_j, mu, 0.998, 300.0, tk_test)
                        w_hat = (n_vec - mu * u_sc) / ell if ell > 1e-12 else np.zeros(3)
                        ff = fac.area_m2 * q_j * (-cd * u_sc - cl * w_hat)
                        f_dir += ff
                        f_sur += ff
            return f_dir, f_sur, float(np.linalg.norm(f_sur - f_dir) / np.linalg.norm(f_dir))

        f_dir_diag, f_sur_diag, rel_diag = _eval_vec_err(923.9803476408316)
        worst_rel_vec = 0.0
        worst_temp_k = 550.0
        for tk_test in dense_temps_101:
            _, _, rel_v = _eval_vec_err(float(tk_test))
            if rel_v > worst_rel_vec:
                worst_rel_vec = rel_v
                worst_temp_k = float(tk_test)
        surrogate_vector_validation.append({
            "pitch_deg": p,
            "diagnostic_T_K": 923.9803476408316,
            "direct_body_vector_N": f_dir_diag.tolist(),
            "surrogate_body_vector_N": f_sur_diag.tolist(),
            "diagnostic_vec_rel_error": rel_diag,
            "n_test_temperatures": len(dense_temps_101),
            "temperature_grid_K": [550.0, 1400.0],
            "max_vec_rel_error_101pt_550_to_1400K": worst_rel_vec,
            "max_error_temperature_K": worst_temp_k,
            "scope_note": (
                "Observed maximum relative full-vector error on 101 uniformly spaced temperatures in [550, 1400] K "
                "at fixed diagnostic velocity (7495.2346 m/s) and fixed peak-state composition (not a continuum bound)."
            ),
        })

    # 7. Propagate ALL 10 quiet/storm pairs at max_step=60s (including explicit 90 deg V2, fixing R3-1!)
    #    + 6 tightened step-convergence pairs (12 trajectories) at max_step=30s (0, 30 AND 90 deg V2/V3-pair)!
    case_specs_60 = [
        (0.0, "v1_legacy_centroid"),
        (1e-3, "v1_legacy_centroid"),
        (0.0, "v2_collimated_area"),
        (0.0, "v3_kinetic_cavity"),
        (30.0, "v1_legacy_centroid"),
        (30.0, "v2_collimated_area"),
        (30.0, "v3_kinetic_cavity"),
        (90.0, "v1_legacy_centroid"),
        (90.0, "v2_collimated_area"),
        (90.0, "v3_kinetic_cavity"),
    ]
    step30_specs = [
        (0.0, "v2_collimated_area"),
        (0.0, "v3_kinetic_cavity"),
        (30.0, "v2_collimated_area"),
        (30.0, "v3_kinetic_cavity"),
        (90.0, "v2_collimated_area"),
        (90.0, "v3_kinetic_cavity"),
    ]
    spline_payloads = {p: splines[p].to_payload() for p in splines}

    tasks = []
    for pitch_deg, mode in case_specs_60:
        sp_pl = spline_payloads[pitch_deg] if mode == "v3_kinetic_cavity" else None
        for forcing in ("quiet", "storm"):
            tasks.append((pitch_deg, mode, forcing, 60.0, msis_masses, sp_pl))
    for pitch_deg, mode in step30_specs:
        sp_pl = spline_payloads[pitch_deg] if mode == "v3_kinetic_cavity" else None
        for forcing in ("quiet", "storm"):
            tasks.append((pitch_deg, mode, forcing, 30.0, msis_masses, sp_pl))

    raw_results = _map_tasks(_worker_orbital_case, tasks, max_workers=min(12, max_workers) if max_workers > 1 else 1)

    res_map = {(r["pitch_deg"], r["mode"], r["forcing"], r["max_step_s"]): r for r in raw_results}
    paired_msis00_visibility_study: list[dict[str, Any]] = []
    for pitch_deg, mode in case_specs_60:
        q_res = res_map[(pitch_deg, mode, "quiet", 60.0)]
        s_res = res_map[(pitch_deg, mode, "storm", 60.0)]
        q_d, s_d = q_res["t170_days"], s_res["t170_days"]
        diff_h = 24.0 * (q_d - s_d) if (q_d and s_d) else None
        diff_pct = 100.0 * (q_d - s_d) / q_d if (q_d and s_d) else None
        paired_msis00_visibility_study.append({
            "pitch_deg": pitch_deg,
            "visibility_mode": mode,
            "status": "completed",
            "atmosphere": "NRLMSISE-00 (in-repo research provider)",
            "quiet_170km_days": q_d,
            "storm_170km_days": s_d,
            "storm_advance_hours": diff_h,
            "storm_advance_pct": diff_pct,
            "quiet_100km_days": q_res["t100_days"],
            "storm_100km_days": s_res["t100_days"],
            "min_kn_sampled_pre_t170_quiet": q_res["min_kn_sampled_pre_t170"],
            "min_kn_sampled_pre_t170_storm": s_res["min_kn_sampled_pre_t170"],
            "min_kn_accepted_pre_t170_quiet": q_res["min_kn_accepted_pre_t170"],
            "min_kn_accepted_pre_t170_storm": s_res["min_kn_accepted_pre_t170"],
            "kn_sample_count_quiet": q_res["kn_sample_count"],
            "kn_sample_count_storm": s_res["kn_sample_count"],
            "kn_max_spacing_s": 30.0,
            "min_kn_storm_time_days": s_res["min_kn_time_days"],
            "min_kn_storm_alt_km": s_res["min_kn_alt_km"],
            "runtime_s_range": [min(q_res["runtime_s_min"], s_res["runtime_s_min"]),
                                max(q_res["runtime_s_max"], s_res["runtime_s_max"])],
        })

    # Explicit 90 deg broadside three-model comparison (R3-1)
    by_mode_90 = {r["visibility_mode"]: r for r in paired_msis00_visibility_study if r["pitch_deg"] == 90.0}
    d90_v1 = by_a[90.0]["drag_v1_legacy_centroid_N"]
    d90_v2 = by_a[90.0]["drag_v2_collimated_area_N"]
    d90_v3 = by_a[90.0]["drag_v3_kinetic_cavity_N"]
    broadside_90deg_summary = {
        "static_drag_N": {"V1": d90_v1, "V2": d90_v2, "V3_pair": d90_v3},
        "static_v2_over_v1_pct": 100.0 * (d90_v2 / d90_v1 - 1.0),
        "static_v3_pair_over_v2_pct": 100.0 * (d90_v3 / d90_v2 - 1.0),
        "static_v3_pair_over_v1_pct": 100.0 * (d90_v3 / d90_v1 - 1.0),
        "orbital_quiet_170km_days": {
            "V1": by_mode_90["v1_legacy_centroid"]["quiet_170km_days"],
            "V2": by_mode_90["v2_collimated_area"]["quiet_170km_days"],
            "V3_pair": by_mode_90["v3_kinetic_cavity"]["quiet_170km_days"],
        },
        "orbital_storm_170km_days": {
            "V1": by_mode_90["v1_legacy_centroid"]["storm_170km_days"],
            "V2": by_mode_90["v2_collimated_area"]["storm_170km_days"],
            "V3_pair": by_mode_90["v3_kinetic_cavity"]["storm_170km_days"],
        },
        "orbital_storm_100km_days": {
            "V1": by_mode_90["v1_legacy_centroid"]["storm_100km_days"],
            "V2": by_mode_90["v2_collimated_area"]["storm_100km_days"],
            "V3_pair": by_mode_90["v3_kinetic_cavity"]["storm_100km_days"],
        },
        "v2_minus_v1_quiet_170km_s": (by_mode_90["v2_collimated_area"]["quiet_170km_days"] - by_mode_90["v1_legacy_centroid"]["quiet_170km_days"]) * 86400.0,
        "v2_minus_v1_storm_170km_s": (by_mode_90["v2_collimated_area"]["storm_170km_days"] - by_mode_90["v1_legacy_centroid"]["storm_170km_days"]) * 86400.0,
        "v2_minus_v1_storm_100km_min": (by_mode_90["v2_collimated_area"]["storm_100km_days"] - by_mode_90["v1_legacy_centroid"]["storm_100km_days"]) * 1440.0,
        "v3_minus_v2_storm_100km_min": (by_mode_90["v3_kinetic_cavity"]["storm_100km_days"] - by_mode_90["v2_collimated_area"]["storm_100km_days"]) * 1440.0,
        "v3_minus_v1_storm_100km_min": (by_mode_90["v3_kinetic_cavity"]["storm_100km_days"] - by_mode_90["v1_legacy_centroid"]["storm_100km_days"]) * 1440.0,
    }

    step30_checks = []
    for pitch_deg, mode in step30_specs:
        q60 = res_map[(pitch_deg, mode, "quiet", 60.0)]
        s60 = res_map[(pitch_deg, mode, "storm", 60.0)]
        q30 = res_map[(pitch_deg, mode, "quiet", 30.0)]
        s30 = res_map[(pitch_deg, mode, "storm", 30.0)]
        adv60 = 24.0 * (q60["t170_days"] - s60["t170_days"])
        adv30 = 24.0 * (q30["t170_days"] - s30["t170_days"])
        step30_checks.append({
            "pitch_deg": pitch_deg,
            "visibility_mode": mode,
            "quiet_t170_diff_s": abs(q30["t170_days"] - q60["t170_days"]) * 86400.0,
            "storm_t170_diff_s": abs(s30["t170_days"] - s60["t170_days"]) * 86400.0,
            "advance_diff_s": abs(adv30 - adv60) * 3600.0,
        })

    cll_table = []
    for sp_sym in CLL_SUPPORTED_SPECIES:
        if sp_sym in ("He", "H"):
            for alpha_rep in (0.97, 0.92, 0.70, 0.25):
                c = cll_constants(sp_sym, alpha_rep)
                cll_table.append({
                    "species": sp_sym, "alpha_n_interval": list(c.alpha_n_interval),
                    "beta": c.beta, "gamma": c.gamma, "delta": c.delta, "zeta": c.zeta,
                })
        else:
            c = cll_constants(sp_sym, 0.90)
            cll_table.append({
                "species": sp_sym, "alpha_n_interval": [0.0, 1.0],
                "beta": c.beta, "gamma": c.gamma, "delta": c.delta, "zeta": c.zeta,
            })

    full_181_rows = [
        {
            "pitch_deg": float(a),
            "status": "completed",
            "drag_v1_legacy_centroid_N": by_a[float(a)]["drag_v1_legacy_centroid_N"],
            "drag_v2_collimated_area_N": by_a[float(a)]["drag_v2_collimated_area_N"],
            "drag_v3_kinetic_cavity_N": by_a[float(a)]["drag_v3_kinetic_cavity_N"],
        }
        for a in range(181)
    ]

    return {
        "visibility_sweep": sweep_cardinal,
        "full_181_integer_attitude_rows": full_181_rows,
        "full_181_integer_attitude_summary": {
            "n_integer_attitudes": 181,
            "all_v1_positive": all(d > 0.0 for d in d_v1_181),
            "all_v2_positive": all(d > 0.0 for d in d_v2_181),
            "all_v3_pair_positive": all(d > 0.0 for d in d_v3_181),
            "min_drag_v1_N": min(d_v1_181),
            "max_drag_v1_N": max(d_v1_181),
            "min_drag_v2_N": min(d_v2_181),
            "max_drag_v2_N": max(d_v2_181),
            "min_drag_v3_pair_N": min(d_v3_181),
            "max_drag_v3_pair_N": max(d_v3_181),
            "max_v3_symmetry_diff_0_180_N": max_sym_diff_v3,
        },
        "v3_pair_quadrature_refinement": quad_refinement,
        "v3_pair_high_speed_45deg": high_speed_45deg,
        "v3_full_8facet_polyhedral": {
            "method": "deterministic_2d_gauss_legendre_crosschecked_by_importance_sampled_5d_ray_mc",
            "primary_method": "deterministic_2d_gauss_legendre",
            "independent_method": "importance_sampled_5d_polyhedral_ray_mc",
            "quadrature_grids": {"coarse": [48, 160], "fine": [96, 320]},
            "ci_method": "approximate_student_t_df_R_minus_1_independent_is_mc_crosscheck",
            "n_replicates": 16,
            "degrees_of_freedom": 15,
            "t_critical_95": float(student_t.ppf(0.975, df=15)),
            "samples_per_replicate": 32768,
            "total_paired_samples_per_angle": 524288,
            "uniform_replication_across_cardinal_angles": True,
            "multi_resolution_refinement_claimed": True,
            "reciprocity_0deg_bus_y_sum_N": (
                v3_full_by_a[0.0]["extra_blocked_drag_by_facet_N"]["bus_+y"]
                + v3_full_by_a[0.0]["extra_blocked_drag_by_facet_N"]["bus_-y"]
            ),
            "reciprocity_0deg_arr_z_N": v3_full_by_a[0.0]["extra_blocked_drag_by_facet_N"]["arr_-z"],
            "reciprocity_0deg_rel_diff": abs(
                (
                    v3_full_by_a[0.0]["extra_blocked_drag_by_facet_N"]["bus_+y"]
                    + v3_full_by_a[0.0]["extra_blocked_drag_by_facet_N"]["bus_-y"]
                )
                - v3_full_by_a[0.0]["extra_blocked_drag_by_facet_N"]["arr_-z"]
            )
            / max(abs(v3_full_by_a[0.0]["extra_blocked_drag_by_facet_N"]["arr_-z"]), 1e-30),
            "qmc_pair_only_validation_0deg_rel_diff_pct": 100.0 * abs(qmc_val_0["drag_v3_full_N"] - by_a[0.0]["drag_v3_kinetic_cavity_N"]) / by_a[0.0]["drag_v3_kinetic_cavity_N"],
            "qmc_pair_only_validation_30deg_rel_diff_pct": 100.0 * abs(qmc_val_30["drag_v3_full_N"] - by_a[30.0]["drag_v3_kinetic_cavity_N"]) / by_a[30.0]["drag_v3_kinetic_cavity_N"],
            "sweep": v3_full_sweep,
            "span_90_over_0": v3_full_by_a[90.0]["drag_v3_full_N"] / v3_full_by_a[0.0]["drag_v3_full_N"],
        },
        "broadside_90deg_summary": broadside_90deg_summary,
        "surrogate_vector_validation": surrogate_vector_validation,
        "near_zero_jump_pct": {
            "v1_legacy_centroid_0_to_1e-5_deg": jump_v1_pct,
            "v2_collimated_area_0_to_1e-5_deg": jump_v2_pct,
            "v3_kinetic_cavity_0_to_1e-5_deg": jump_v3_pct,
        },
        "attitude_90_over_0_ratio": {
            "v1_legacy_centroid": by_a[90.0]["drag_v1_legacy_centroid_N"] / by_a[0.0]["drag_v1_legacy_centroid_N"],
            "v1_at_1e-5_deg": by_a[90.0]["drag_v1_legacy_centroid_N"] / by_a[1e-5]["drag_v1_legacy_centroid_N"],
            "v2_collimated_area": by_a[90.0]["drag_v2_collimated_area_N"] / by_a[0.0]["drag_v2_collimated_area_N"],
            "v3_pair_kinetic_cavity": by_a[90.0]["drag_v3_kinetic_cavity_N"] / by_a[0.0]["drag_v3_kinetic_cavity_N"],
            "v3_full_8facet_polyhedral": v3_full_by_a[90.0]["drag_v3_full_N"] / v3_full_by_a[0.0]["drag_v3_full_N"],
        },
        "cll_specular_endpoint_residuals": cll_endpoint_residuals,
        "timeseries_145h_decomposition": {
            "t_min_k": min(temps_145),
            "t_max_k": max(temps_145),
            "joint_actual_width_pct": _rel_width_pct(series_joint),
            "temperature_only_fixed_peak_comp_width_pct": _rel_width_pct(series_t_only),
            "composition_only_fixed_peak_temp_width_pct": _rel_width_pct(series_c_only),
        },
        "paired_orbital_visibility_and_knudsen_msis00": paired_msis00_visibility_study,
        "msis00_step30_convergence_checks": step30_checks,
        "cll_constants_table": cll_table,
    }


# ---------------------------------------------------------------------------
# 3. Paper II: Drag-Free J2 Controls, Same-Orbit J2-Transported Phase Sweep & Uncached Refits
# ---------------------------------------------------------------------------

def _conic_state(a_m: float, e: float, inc_deg: float, raan_deg: float, omega_deg: float, nu_deg: float) -> tuple[np.ndarray, np.ndarray]:
    p = a_m * (1.0 - e * e)
    nu = math.radians(nu_deg)
    rmag = p / (1.0 + e * math.cos(nu))
    r_pqw = np.array([rmag * math.cos(nu), rmag * math.sin(nu), 0.0])
    v_pqw = math.sqrt(MU_EARTH.value / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    O = math.radians(raan_deg)
    w = math.radians(omega_deg)
    i = math.radians(inc_deg)
    cO, sO = math.cos(O), math.sin(O)
    cw, sw_ = math.cos(w), math.sin(w)
    ci, si = math.cos(i), math.sin(i)
    R = np.array([
        [cO * cw - sO * sw_ * ci, -cO * sw_ - sO * cw * ci, sO * si],
        [sO * cw + cO * sw_ * ci, -sO * sw_ + cO * cw * ci, -cO * si],
        [sw_ * si, cw * si, ci],
    ])
    return R @ r_pqw, R @ v_pqw


def _j2_potential_j_kg(r: np.ndarray) -> float:
    rmag = float(np.linalg.norm(r))
    sin_lat = r[2] / rmag
    return float((MU_EARTH.value / rmag) * J2_CONST * ((RE_WGS84 / rmag) ** 2) * 0.5 * (3.0 * sin_lat * sin_lat - 1.0))


def _total_mech_energy_j_kg(r: np.ndarray, v: np.ndarray) -> float:
    rmag = float(np.linalg.norm(r))
    return float(0.5 * (v @ v) - MU_EARTH.value / rmag + _j2_potential_j_kg(r))


def _drag_free_j2_rhs(t_s: float, y: np.ndarray) -> np.ndarray:
    r = y[:3]
    rn = float(np.linalg.norm(r))
    acc = -MU_EARTH.value * r / (rn ** 3) + j2_acceleration(r)
    return np.concatenate([y[3:], acc])


def _evaluate_drag_free_j2_control(
    convention: str, angle_deg: float, r0: np.ndarray, v0: np.ndarray,
    a0: float, e_ref: float, hz_ref: float,
) -> dict[str, Any]:
    """20-period drag-free central+J2 control quantifying conserved invariants (E, h_z) and
    orbital-shape means (time-mean osculating a, e, perigee altitude, and min actual radius).
    """
    period_s = 2.0 * math.pi * math.sqrt((a0 ** 3) / MU_EARTH.value)
    t_eval = np.linspace(0.0, 20.0 * period_s, 4001)
    sol = solve_ivp(_drag_free_j2_rhs, (0.0, 20.0 * period_s), np.concatenate([r0, v0]),
                    method="DOP853", rtol=2e-11, atol=1e-8, max_step=50.0, t_eval=t_eval)
    rr = sol.y[:3].T
    vv = sol.y[3:].T
    rn = np.linalg.norm(rr, axis=1)
    h = np.cross(rr, vv)
    e_vecs = np.cross(vv, h) / MU_EARTH.value - rr / rn[:, None]
    ee = np.linalg.norm(e_vecs, axis=1)
    aa = 1.0 / (2.0 / rn - np.sum(vv * vv, axis=1) / MU_EARTH.value)
    hp_km = (aa * (1.0 - ee) - RE_WGS84) / 1000.0
    e0_val = _total_mech_energy_j_kg(r0, v0)
    hz0 = float(np.cross(r0, v0)[2])
    return {
        "convention": convention,
        "angle_deg": float(angle_deg),
        "energy_diff_vs_ref_J_kg": float(e0_val - e_ref),
        "hz_rel_diff_pct": float(100.0 * (hz0 / hz_ref - 1.0)),
        "time_mean_osculating_a_km": float(np.trapezoid(aa, sol.t) / (20.0 * period_s) / 1000.0),
        "time_mean_osculating_e": float(np.trapezoid(ee, sol.t) / (20.0 * period_s)),
        "time_mean_osculating_perigee_alt_km": float(np.trapezoid(hp_km, sol.t) / (20.0 * period_s)),
        "min_actual_radius_alt_km": float((rn.min() - RE_WGS84) / 1000.0),
        "energy_conservation_max_J_kg": float(max(abs(_total_mech_energy_j_kg(rx, vx) - e0_val) for rx, vx in zip(rr, vv))),
    }


def _generate_same_orbit_j2_states(r_ref: np.ndarray, v_ref: np.ndarray, a0: float) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """Transport (r_ref, v_ref) along ONE drag-free central+J2 orbit to argument-of-latitude
    phases u in {0, 90, 180, 270} deg so all 4 initial states belong to the EXACT SAME J2 orbit
    (identical E, identical h_z, identical 20-period mean perigee & eccentricity; fixes R-II-1!).
    """
    period_s = 2.0 * math.pi * math.sqrt((a0 ** 3) / MU_EARTH.value)
    sol = solve_ivp(_drag_free_j2_rhs, (0.0, 1.15 * period_s), np.concatenate([r_ref, v_ref]),
                    method="DOP853", rtol=1e-12, atol=1e-9, max_step=20.0, dense_output=True)

    # Track continuous argument of latitude u(t) measured in instantaneous orbital plane from ascending node
    def _arg_lat_rad(t_s: float) -> float:
        y = sol.sol(t_s)
        r = y[:3]
        v = y[3:]
        h = np.cross(r, v)
        n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h)
        n_hat = n_vec / np.linalg.norm(n_vec)
        q_hat = np.cross(h / np.linalg.norm(h), n_hat)
        ang = math.atan2(float(r @ q_hat), float(r @ n_hat))
        if ang < 0.0:
            ang += 2.0 * math.pi
        return ang

    states: list[tuple[float, float, np.ndarray, np.ndarray]] = [(0.0, 0.0, r_ref.copy(), v_ref.copy())]
    for target_deg in (90.0, 180.0, 270.0):
        target_rad = math.radians(target_deg)
        # Bracket around fractional Keplerian period
        t_guess = (target_deg / 360.0) * period_s
        t_lo = max(10.0, t_guess - 400.0)
        t_hi = min(1.12 * period_s, t_guess + 400.0)
        t_star = float(brentq(lambda ts: _arg_lat_rad(ts) - target_rad, t_lo, t_hi, xtol=1e-9))
        y_star = sol.sol(t_star)
        states.append((target_deg, t_star, y_star[:3].copy(), y_star[3:].copy()))
    return states


def _decay_days_for_state(
    r0_l: list[float],
    v0_l: list[float],
    epoch_unix: float,
    gamma: float,
    fast_msis: FastMsis00Evaluator,
) -> float:
    """Propagate scalar Cowell DOP853 (rtol=1e-9, atol=1e-6, max_step=90s) to first 100 km crossing."""
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH.value])
    y0 = np.array([*r0_l, *v0_l], dtype=float)

    def rhs(t_s: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:]
        rmag = float(np.linalg.norm(r))
        acc = -MU_EARTH.value * r / (rmag ** 3) + j2_acceleration(r)
        alt_m = geodetic_altitude(r)
        if alt_m < REENTRY_ALTITUDE_M:
            return np.concatenate([v, acc])
        unix_s = epoch_unix + float(t_s)
        lat_d, lon_d = geodetic_latlon(r, _gmst_from_unix(unix_s))
        rho = fast_msis.species_mass_density_unix(unix_s, alt_m, lat_d, lon_d)
        v_rel = v - np.cross(omega_vec, r)
        acc = acc - 0.5 * gamma * rho * float(np.linalg.norm(v_rel)) * v_rel
        return np.concatenate([v, acc])

    def ev_100(t_s: float, y: np.ndarray) -> float:
        return geodetic_altitude(y[:3]) - REENTRY_ALTITUDE_M

    ev_100.terminal = True  # type: ignore[attr-defined]
    ev_100.direction = -1.0  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        (0.0, 20.0 * 86400.0),
        y0,
        method="DOP853",
        rtol=1e-9,
        atol=1e-6,
        max_step=90.0,
        events=[ev_100],
    )
    if len(sol.t_events[0]) == 0:
        return 20.0
    return float(sol.t_events[0][0] / 86400.0)


def _worker_invert_gamma(args: tuple[str, float, list[float], list[float], float]) -> tuple[str, float]:
    tag, target_days, r0_l, v0_l, epoch_unix = args
    store = RawStore(REPO / "data/raw")
    kp = sw.parse_gfz_kp_ap(store.read("gfz", "kp_2022_extended", ".json"), store.read("gfz", "kp_2022_extended_ap", ".json"))
    f107 = sw.parse_celestrak_f107(store.read("celestrak", "sw_last5years", ".csv"))
    fast_msis = FastMsis00Evaluator(kp, f107, quiet_ap=None)
    g_root = float(
        brentq(
            lambda g: _decay_days_for_state(r0_l, v0_l, epoch_unix, g, fast_msis) - target_days,
            0.015,
            0.045,
            xtol=1e-7,
            rtol=1e-7,
        )
    )
    return tag, g_root


def evaluate_paper2_same_orbit_residuals(
    same_orbit_rows: list[dict[str, Any]],
    kp: Any,
    f107: Any,
    *,
    target_days: float = 8.0,
    acceptable_time_tolerance_s: float = 10.0,
    bracket_delta_gamma: float = 1e-7,
    always_check_brackets: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate target-space crossing time T(gamma_fit), signed residual (s), and bracketing
    sensitivity for the 4 published same-orbit drag-free J2-transported phase fits (Item 8).
    """
    rp0 = RE_WGS84 + 210_000.0
    ra0 = RE_WGS84 + 338_000.0
    a0 = 0.5 * (rp0 + ra0)
    e0 = (ra0 - rp0) / (ra0 + rp0)
    inc0 = 53.22
    r_ref, v_ref = _conic_state(a0, e0, inc0, 0.0, 0.0, 0.0)
    dep_ts = EV.DEPLOYMENT_UTC.timestamp()
    states_by_u = {
        float(u_deg): (float(t_trans), r_s.tolist(), v_s.tolist())
        for u_deg, t_trans, r_s, v_s in _generate_same_orbit_j2_states(r_ref, v_ref, a0)
    }
    fast_msis = FastMsis00Evaluator(kp, f107, quiet_ap=None)

    records: list[dict[str, Any]] = []
    for row in same_orbit_rows:
        u_deg = float(row["arg_lat_deg"])
        g_fit = float(row["fitted_gamma_m2_kg_target_8d"])
        t_trans, r0_l, v0_l = states_by_u[u_deg]
        achieved_days = _decay_days_for_state(r0_l, v0_l, dep_ts, g_fit, fast_msis)
        signed_res_days = achieved_days - float(target_days)
        signed_res_s = signed_res_days * 86400.0
        abs_res_s = abs(signed_res_s)
        within_tol = abs_res_s <= float(acceptable_time_tolerance_s)

        rec: dict[str, Any] = {
            "label": f"same_orbit_u_{int(round(u_deg))}deg",
            "arg_lat_deg": u_deg,
            "drag_free_transport_time_s": t_trans,
            "fitted_gamma_m2_kg": g_fit,
            "target_days": float(target_days),
            "achieved_days": achieved_days,
            "signed_residual_days": signed_res_days,
            "signed_residual_s": signed_res_s,
            "abs_residual_s": abs_res_s,
            "acceptable_time_tolerance_s": float(acceptable_time_tolerance_s),
            "within_tolerance": within_tol,
            "discontinuity_disclosed": False,
        }
        if always_check_brackets or not within_tol:
            g_lo = g_fit - float(bracket_delta_gamma)
            g_hi = g_fit + float(bracket_delta_gamma)
            t_lo_days = _decay_days_for_state(r0_l, v0_l, dep_ts, g_lo, fast_msis)
            t_hi_days = _decay_days_for_state(r0_l, v0_l, dep_ts, g_hi, fast_msis)
            span_s = abs(t_lo_days - t_hi_days) * 86400.0
            # An orbital-pass switch across delta_gamma=1e-7 jumps by a sizeable fraction of an orbit (~1000-5400 s)
            pass_switch = bool(span_s > 300.0)
            rec["bracket_delta_gamma"] = float(bracket_delta_gamma)
            rec["bracket_low_gamma"] = g_lo
            rec["bracket_high_gamma"] = g_hi
            rec["bracket_low_achieved"] = t_lo_days
            rec["bracket_high_achieved"] = t_hi_days
            rec["bracket_span_s"] = span_s
            rec["pass_switch_detected"] = pass_switch
            if pass_switch or not within_tol:
                rec["discontinuity_disclosed"] = True
        records.append(rec)
    return records


def compute_paper2_phase_and_catalogue(*, max_workers: int = 1) -> dict[str, Any]:
    rp0 = RE_WGS84 + 210_000.0
    ra0 = RE_WGS84 + 338_000.0
    a0 = 0.5 * (rp0 + ra0)
    e0 = (ra0 - rp0) / (ra0 + rp0)
    inc0 = 53.22

    r_ref, v_ref = _conic_state(a0, e0, inc0, 0.0, 0.0, 0.0)
    E_ref = _total_mech_energy_j_kg(r_ref, v_ref)
    hz_ref = float(np.cross(r_ref, v_ref)[2])
    dep_ts = EV.DEPLOYMENT_UTC.timestamp()

    inv_jobs: list[tuple[str, float, list[float], list[float], float]] = []
    drag_free_controls: list[dict[str, Any]] = []

    # 1) Osculating nu (fixed Keplerian a0, e0)
    osc_meta = []
    for nu_deg in (0.0, 90.0, 180.0, 270.0):
        r_i, v_i = _conic_state(a0, e0, inc0, 0.0, 0.0, nu_deg)
        h_vec = np.cross(r_i, v_i)
        e_vec = np.cross(v_i, h_vec) / MU_EARTH.value - r_i / np.linalg.norm(r_i)
        e_tot = _total_mech_energy_j_kg(r_i, v_i)
        u_j2 = _j2_potential_j_kg(r_i)
        delta_a_eq_km = (2.0 * (a0 ** 2) / MU_EARTH.value) * (e_tot - E_ref) / 1000.0
        tag = f"osc_nu_{int(nu_deg)}"
        inv_jobs.append((tag, 8.0, r_i.tolist(), v_i.tolist(), dep_ts))
        osc_meta.append((tag, nu_deg, e_vec, u_j2, e_tot - E_ref, delta_a_eq_km))
        drag_free_controls.append(_evaluate_drag_free_j2_control("osc_nu", nu_deg, r_i, v_i, a0, E_ref, hz_ref))

    # 2) J2-energy-matched nu (varies a only -> still leaves h_z & orbital-shape spread!)
    j2_nu_meta = []
    for nu_deg in (0.0, 90.0, 180.0, 270.0):
        a_match = a0 if nu_deg == 0.0 else float(brentq(
            lambda a_try: _total_mech_energy_j_kg(*_conic_state(a_try, e0, inc0, 0.0, 0.0, nu_deg)) - E_ref,
            a0 - 30_000.0, a0 + 30_000.0, xtol=1e-8,
        ))
        r_m, v_m = _conic_state(a_match, e0, inc0, 0.0, 0.0, nu_deg)
        tag = f"j2_nu_{int(nu_deg)}"
        if nu_deg != 0.0:
            inv_jobs.append((tag, 8.0, r_m.tolist(), v_m.tolist(), dep_ts))
        j2_nu_meta.append((tag, nu_deg, (a_match - a0) / 1000.0))
        drag_free_controls.append(_evaluate_drag_free_j2_control("matched_nu", nu_deg, r_m, v_m, a0, E_ref, hz_ref))

    # 3) NEW Same-Orbit Drag-Free J2-Transported Phase Sweep (fixes R-II-1!)
    same_orbit_states = _generate_same_orbit_j2_states(r_ref, v_ref, a0)
    same_orbit_meta = []
    for u_deg, t_trans, r_s, v_s in same_orbit_states:
        tag = f"same_orb_{int(u_deg)}"
        if u_deg != 0.0:
            inv_jobs.append((tag, 8.0, r_s.tolist(), v_s.tolist(), dep_ts))
        same_orbit_meta.append((tag, u_deg, t_trans))
        drag_free_controls.append(_evaluate_drag_free_j2_control("same_orbit_j2_phase", u_deg, r_s, v_s, a0, E_ref, hz_ref))

    # 4) Argument-of-perigee omega rotation (fixed Keplerian & E-matched)
    om_meta = []
    for om_deg in (0.0, 90.0, 180.0, 270.0):
        r_o, v_o = _conic_state(a0, e0, inc0, 0.0, om_deg, 0.0)
        h_vec = np.cross(r_o, v_o)
        e_vec = np.cross(v_o, h_vec) / MU_EARTH.value - r_o / np.linalg.norm(r_o)
        a_om = a0 if om_deg == 0.0 else float(brentq(
            lambda a_try: _total_mech_energy_j_kg(*_conic_state(a_try, e0, inc0, 0.0, om_deg, 0.0)) - E_ref,
            a0 - 30_000.0, a0 + 30_000.0, xtol=1e-8,
        ))
        r_om, v_om = _conic_state(a_om, e0, inc0, 0.0, om_deg, 0.0)
        tag_fix = f"om_fix_{int(om_deg)}"
        tag_j2 = f"om_j2_{int(om_deg)}"
        if om_deg != 0.0:
            inv_jobs.append((tag_fix, 8.0, r_o.tolist(), v_o.tolist(), dep_ts))
            inv_jobs.append((tag_j2, 8.0, r_om.tolist(), v_om.tolist(), dep_ts))
        om_meta.append((tag_fix, tag_j2, om_deg, e_vec, math.degrees(math.asin(r_o[2] / np.linalg.norm(r_o))), (a_om - a0) / 1000.0))
        drag_free_controls.append(_evaluate_drag_free_j2_control("matched_omega", om_deg, r_om, v_om, a0, E_ref, hz_ref))

    # 5) Coupled release-epoch shift (+/- 6 h) + uncached aligned baseline
    ep_meta = []
    aligned_base_target = 8.0 - ((18.0 + 28.0 / 60.0) / 24.0 - 0.5)
    for dh in (-6.0, 0.0, +6.0):
        ep_ts = dep_ts + dh * 3600.0
        t_leg = 8.0 - (dh / 24.0)
        t_aln = aligned_base_target - (dh / 24.0)
        tag_l = f"ep_leg_{int(dh)}"
        tag_a = f"ep_aln_{int(dh)}"
        if dh != 0.0:
            inv_jobs.append((tag_l, t_leg, r_ref.tolist(), v_ref.tolist(), ep_ts))
        inv_jobs.append((tag_a, t_aln, r_ref.tolist(), v_ref.tolist(), ep_ts))
        ep_meta.append((tag_l, tag_a, dh, t_leg, t_aln))

    inv_out = dict(_map_tasks(_worker_invert_gamma, inv_jobs, max_workers=min(12, max_workers) if max_workers > 1 else 1))

    g0 = inv_out["osc_nu_0"]
    inv_out["j2_nu_0"] = g0
    inv_out["same_orb_0"] = g0
    inv_out["om_fix_0"] = g0
    inv_out["om_j2_0"] = g0
    inv_out["ep_leg_0"] = g0

    osculating_nu_uncached = [
        {
            "nu_deg": nu_deg,
            "omega_deg": 0.0,
            "eccentricity_vector": [float(x) for x in e_vec],
            "u_j2_kJ_kg": u_j2 / 1000.0,
            "delta_mech_energy_vs_0deg_kJ_kg": de / 1000.0,
            "equivalent_mean_sma_shift_km": da_km,
            "fitted_gamma_m2_kg_target_8d": inv_out[tag],
        }
        for tag, nu_deg, e_vec, u_j2, de, da_km in osc_meta
    ]
    j2_energy_matched_nu = [
        {
            "nu_deg": nu_deg,
            "matched_osculating_sma_delta_km": da_km,
            "fitted_gamma_m2_kg_target_8d": inv_out[tag],
        }
        for tag, nu_deg, da_km in j2_nu_meta
    ]
    same_orbit_j2_phase = [
        {
            "arg_lat_deg": u_deg,
            "drag_free_transport_time_s": t_trans,
            "fitted_gamma_m2_kg_target_8d": inv_out[tag],
        }
        for tag, u_deg, t_trans in same_orbit_meta
    ]
    omega_perigee_rotation = [
        {
            "omega_deg": om_deg,
            "nu_deg": 0.0,
            "eccentricity_vector": [float(x) for x in e_vec],
            "perigee_geocentric_lat_deg": lat_p,
            "matched_osculating_sma_delta_km": da_km,
            "fitted_gamma_fixed_keplerian_m2_kg": inv_out[t_fix],
            "fitted_gamma_j2_energy_matched_m2_kg": inv_out[t_j2],
        }
        for t_fix, t_j2, om_deg, e_vec, lat_p, da_km in om_meta
    ]
    coupled_epoch = [
        {
            "epoch_shift_hours": dh,
            "coupled_legacy_target_days": t_leg,
            "fitted_gamma_coupled_legacy_m2_kg": inv_out[t_l],
            "coupled_aligned_target_days": t_aln,
            "fitted_gamma_coupled_aligned_m2_kg": inv_out[t_a],
        }
        for t_l, t_a, dh, t_leg, t_aln in ep_meta
    ]

    def _mid_spread_pct(vals: list[float]) -> float:
        lo_v, hi_v = min(vals), max(vals)
        return 100.0 * (hi_v - lo_v) / (0.5 * (hi_v + lo_v))

    # Summarise drag-free control spreads by convention
    control_summaries: dict[str, dict[str, float]] = {}
    for conv in ("osc_nu", "matched_nu", "same_orbit_j2_phase", "matched_omega"):
        rows_c = [r for r in drag_free_controls if r["convention"] == conv]
        hps = [r["time_mean_osculating_perigee_alt_km"] for r in rows_c]
        es = [r["time_mean_osculating_e"] for r in rows_c]
        hzs = [r["hz_rel_diff_pct"] for r in rows_c]
        ens = [r["energy_diff_vs_ref_J_kg"] for r in rows_c]
        control_summaries[conv] = {
            "mean_perigee_alt_min_km": min(hps),
            "mean_perigee_alt_max_km": max(hps),
            "mean_perigee_alt_spread_km": max(hps) - min(hps),
            "mean_e_min": min(es),
            "mean_e_max": max(es),
            "mean_e_spread": max(es) - min(es),
            "hz_rel_spread_pct": max(hzs) - min(hzs),
            "energy_spread_J_kg": max(ens) - min(ens),
        }

    u_doc = json.loads((REPO / "evidence/uncertainty_analysis.json").read_text())
    u_cases = u_doc["cases"]
    costs = [float(c["outcome"]["storm_cost_days"]) for c in u_cases]
    fracs = [float(c["outcome"]["storm_cost_fraction"]) for c in u_cases]
    max_cost_row = max(u_cases, key=lambda c: float(c["outcome"]["storm_cost_days"]))
    max_frac_row = max(u_cases, key=lambda c: float(c["outcome"]["storm_cost_fraction"]))

    poc_doc = json.loads((REPO / "evidence/per_object_calibration.json").read_text())
    obs_by_id = {o["object_id"]: o for o in poc_doc["observations"]}
    per_obj_table = []
    for fit_row in poc_doc["per_object_fits"]:
        oid = fit_row["object_id"]
        ob = obs_by_id[oid]
        per_obj_table.append({
            "object_id": oid,
            "norad_id": fit_row["norad_id"],
            "object_class": fit_row["object_class"],
            "n_elements": fit_row["n_elements"],
            "first_element_epoch_utc": ob["first_element_epoch_utc"],
            "decay_date": ob["decay_date"],
            "observed_decay_days": fit_row["observed_decay_days"],
            "initial_perigee_km": fit_row["initial_perigee_altitude_m"] / 1000.0,
            "initial_apogee_km": fit_row["initial_apogee_altitude_m"] / 1000.0,
            "fitted_gamma_m2_kg": fit_row["fitted_inverse_ballistic_coefficient_m2_kg"],
            "curve_decay_range_days": fit_row["curve_decay_range_days"],
        })

    return {
        "uncached_vs_legacy_cached_0deg": {
            "legacy_5deg_cached_m2_kg": 0.023745284948988015,
            "uncached_deterministic_m2_kg": g0,
            "uncached_epoch_aligned_m2_kg": inv_out["ep_aln_0"],
            "relative_diff_pct": 100.0 * (g0 / 0.023745284948988015 - 1.0),
        },
        "drag_free_j2_controls": drag_free_controls,
        "drag_free_j2_control_summaries": control_summaries,
        "osculating_nu_uncached": osculating_nu_uncached,
        "osculating_nu_uncached_midpoint_spread_pct": _mid_spread_pct([r["fitted_gamma_m2_kg_target_8d"] for r in osculating_nu_uncached]),
        "j2_energy_matched_nu": j2_energy_matched_nu,
        "j2_energy_matched_nu_midpoint_spread_pct": _mid_spread_pct([r["fitted_gamma_m2_kg_target_8d"] for r in j2_energy_matched_nu]),
        "same_orbit_j2_phase": same_orbit_j2_phase,
        "same_orbit_j2_phase_midpoint_spread_pct": _mid_spread_pct([r["fitted_gamma_m2_kg_target_8d"] for r in same_orbit_j2_phase]),
        "omega_perigee_rotation": omega_perigee_rotation,
        "omega_fixed_keplerian_midpoint_spread_pct": _mid_spread_pct([r["fitted_gamma_fixed_keplerian_m2_kg"] for r in omega_perigee_rotation]),
        "omega_j2_matched_midpoint_spread_pct": _mid_spread_pct([r["fitted_gamma_j2_energy_matched_m2_kg"] for r in omega_perigee_rotation]),
        "coupled_release_epoch_shift": coupled_epoch,
        "coupled_legacy_midpoint_spread_pct": _mid_spread_pct([r["fitted_gamma_coupled_legacy_m2_kg"] for r in coupled_epoch]),
        "coupled_aligned_midpoint_spread_pct": _mid_spread_pct([r["fitted_gamma_coupled_aligned_m2_kg"] for r in coupled_epoch]),
        "uncertainty_652_exact": {
            "n_cases": len(u_cases),
            "cost_min_days": min(costs),
            "cost_max_days": max(costs),
            "cost_median_days": float(np.median(costs)),
            "n_below_0_5_days": sum(1 for c in costs if c < 0.5),
            "pct_below_0_5_days": 100.0 * sum(1 for c in costs if c < 0.5) / len(u_cases),
            "fraction_min_pct": 100.0 * min(fracs),
            "fraction_max_pct": 100.0 * max(fracs),
            "fraction_median_pct": 100.0 * float(np.median(fracs)),
            "max_cost_case_id": max_cost_row["case"]["case_id"],
            "max_cost_case": max_cost_row["case"],
            "max_cost_outcome": max_cost_row["outcome"],
            "max_fraction_case_id": max_frac_row["case"]["case_id"],
            "max_fraction_case": max_frac_row["case"],
            "max_fraction_outcome": max_frac_row["outcome"],
        },
        "per_object_grid": poc_doc["grid"],
        "per_object_10_table": per_obj_table,
    }


def run_serial_fixture_test() -> dict[str, Any]:
    """Exercise the serial _map_tasks(max_workers=1) path on a small deterministic fixture
    while asserting that ProcessPoolExecutor is never instantiated (Item 4).
    """
    orig_pool = globals()["ProcessPoolExecutor"]

    def _forbidden_pool(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("ProcessPoolExecutor must NOT be instantiated when max_workers=1")

    globals()["ProcessPoolExecutor"] = _forbidden_pool
    try:
        q = ExactCavityPairQuadrature()

        def _eval_angle(angle_deg: float) -> dict[str, float]:
            fb, fa = q.evaluate_cavity_pair_vectors(10.0, angle_deg, 0.998, 300.0, 923.9803476408316)
            return {"pitch_deg": angle_deg, "bus_fx": float(fb[0]), "bus_fz": float(fb[2]), "arr_fz": float(fa[2])}

        rows = _map_tasks(_eval_angle, [0.0, 30.0, 45.0], max_workers=1)
        assert len(rows) == 3
        return {
            "serial_default_verified": True,
            "process_pool_spawned": False,
            "fixture_angles_evaluated": [r["pitch_deg"] for r in rows],
            "sample_45deg_bus_fx": rows[2]["bus_fx"],
        }
    finally:
        globals()["ProcessPoolExecutor"] = orig_pool


def _build_provenance(raw_inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    tracked_inputs = [
        "config/events/starlink_group_4_7_2022.yaml",
        "evidence/DATA_MANIFEST.json",
        "evidence/per_object_calibration.json",
        "evidence/uncertainty_analysis.json",
        "results/species_resolved_gsi/run_manifest.json",
        "results/species_resolved_gsi/species_force_summary.csv",
        "results/species_resolved_gsi/species_force_timeseries.csv",
    ]
    input_hashes = {
        rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        for rel in tracked_inputs
        if (REPO / rel).is_file()
    }
    prov: dict[str, Any] = {
        "generator_path": "scripts/run_referee_revision_diagnostics.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_hashes": input_hashes,
    }
    if raw_inventory is not None:
        prov["consumed_raw_input_summary"] = {
            "manifest_path": raw_inventory.get("manifest_path"),
            "manifest_sha256": raw_inventory.get("manifest_sha256"),
            "consumed_count": raw_inventory.get("consumed_count"),
            "wam_ipe_aggregate_sha256": raw_inventory.get("wam_ipe_aggregate_sha256"),
        }
    return prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum worker processes across all parallel sections (default: 1 = serial in-process).",
    )
    ap.add_argument("--verify-fast-msis-only", action="store_true",
                    help="Run negative controls + real 20-state NRLMSISE-00 equivalence check only.")
    ap.add_argument("--verify-raw-inputs-only", action="store_true",
                    help="Verify pinned GFZ, CelesTrak, and 180 WAM-IPE raw inputs only.")
    ap.add_argument("--verify-paper2-residuals-only", action="store_true",
                    help="Evaluate Paper II same-orbit phase target-space residuals only (serial).")
    ap.add_argument("--serial-fixture-test", action="store_true",
                    help="Run offline serial-default fixture check with ProcessPoolExecutor disabled.")
    ap.add_argument(
        "--update-diagnostics-metadata",
        action="store_true",
        help=(
            "Run only the bounded targeted checks (fast-MSIS 20-state mask+ULP verification, "
            "pinned raw-input inventory, and Paper II same-orbit target-space residuals) and "
            "update those verification blocks in evidence/referee_revision_diagnostics.json "
            "without rerunning full orbital sweeps (only allowed if fast-MSIS agrees identically)."
        ),
    )
    args = ap.parse_args(argv)
    if args.max_workers < 1:
        ap.error("--max-workers must be >= 1")

    if args.serial_fixture_test:
        print(json.dumps(run_serial_fixture_test(), indent=2))
        return 0

    t0 = time.perf_counter()

    if args.verify_raw_inputs_only:
        _wam, inv = verify_consumed_raw_inputs(REPO)
        print(json.dumps(inv, indent=2))
        return 0

    # Verify consumed raw inputs BEFORE science in every raw-driver mode
    wam_files, raw_inventory = verify_consumed_raw_inputs(REPO)
    store = RawStore(REPO / "data/raw")
    kp = sw.parse_gfz_kp_ap(
        store.read("gfz", "kp_2022_extended", ".json"),
        store.read("gfz", "kp_2022_extended_ap", ".json"),
    )
    f107 = sw.parse_celestrak_f107(store.read("celestrak", "sw_last5years", ".csv"))
    msis = Nrlmsise00Provider()

    if args.verify_fast_msis_only:
        controls = run_fast_msis_verifier_controls()
        real_check = verify_fast_msis_bit_identity(kp, f107, msis)
        print(json.dumps({"negative_controls": controls, "real_20_state_check": real_check}, indent=2))
        return 0

    out_path = REPO / "evidence" / "referee_revision_diagnostics.json"

    if args.verify_paper2_residuals_only:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        res_rows = evaluate_paper2_same_orbit_residuals(existing["paper2"]["same_orbit_j2_phase"], kp, f107)
        print(json.dumps(res_rows, indent=2))
        return 0

    if args.update_diagnostics_metadata:
        controls = run_fast_msis_verifier_controls()
        bit_check = verify_fast_msis_bit_identity(kp, f107, msis)
        if not (bit_check["bit_identical_same_dtype"] and bit_check["max_abs_diff"] == 0.0 and bit_check["max_ulp_diff"] == 0):
            raise RuntimeError("FastMsis00Evaluator did not match Nrlmsise00Provider identically; full rerun required.")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        res_rows = evaluate_paper2_same_orbit_residuals(existing["paper2"]["same_orbit_j2_phase"], kp, f107)
        existing["provenance"] = _build_provenance(raw_inventory)
        existing["fast_msis_equivalence_check"] = bit_check
        existing["fast_msis_verifier_controls"] = controls
        existing["consumed_raw_input_inventory"] = raw_inventory
        existing["paper2"]["same_orbit_j2_phase_fit_residuals"] = res_rows
        raw_bytes = (json.dumps(existing, indent=2, sort_keys=True) + "\n").encode("utf-8")
        out_path.write_bytes(raw_bytes)
        print(json.dumps({
            "mode": "update_diagnostics_metadata",
            "output": str(out_path),
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "fast_msis_bit_identical": bit_check["bit_identical_same_dtype"],
            "raw_inputs_consumed_count": raw_inventory["consumed_count"],
            "paper2_same_orbit_max_abs_residual_s": max(r["abs_residual_s"] for r in res_rows),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }, indent=2))
        return 0

    controls = run_fast_msis_verifier_controls()
    bit_check = verify_fast_msis_bit_identity(kp, f107, msis)
    print(f"[0/3] FastMsis00Evaluator equivalence verified (max_workers={args.max_workers}): "
          f"states={bit_check['tested_states']}, max_abs_diff={bit_check['max_abs_diff']}, max_ulp_diff={bit_check['max_ulp_diff']}", flush=True)

    print("[1/3] Computing Paper I WAM-IPE operators & contiguous path integrals...", flush=True)
    p1 = compute_paper1_wam_and_path(store, kp, f107, msis, wam_files=wam_files)
    print(f"      Done ({time.perf_counter() - t0:.1f}s).", flush=True)

    print(f"[2/3] Computing Paper III exact V3-pair (181 deg), V3-full 8-facet QMC, CLL endpoints & full-vector DOP853 runs (max_workers={args.max_workers})...", flush=True)
    p3 = compute_paper3_and_paired_orbital(max_workers=args.max_workers)
    print(f"      Done ({time.perf_counter() - t0:.1f}s). Spans: {p3['attitude_90_over_0_ratio']}", flush=True)

    print(f"[3/3] Computing Paper II drag-free J2 controls, same-orbit J2 phase sweep & uncached refits (max_workers={args.max_workers})...", flush=True)
    p2 = compute_paper2_phase_and_catalogue(max_workers=args.max_workers)
    p2["same_orbit_j2_phase_fit_residuals"] = evaluate_paper2_same_orbit_residuals(p2["same_orbit_j2_phase"], kp, f107)
    print(f"      Done ({time.perf_counter() - t0:.1f}s). Spreads (%): osc_nu={p2['osculating_nu_uncached_midpoint_spread_pct']:.2f}, "
          f"E_matched_nu={p2['j2_energy_matched_nu_midpoint_spread_pct']:.2f}, "
          f"same_orbit_J2={p2['same_orbit_j2_phase_midpoint_spread_pct']:.2f}, "
          f"omega_E_matched={p2['omega_j2_matched_midpoint_spread_pct']:.2f}", flush=True)

    provenance = _build_provenance(raw_inventory)
    # Deterministic payload: ZERO wall-clock elapsed_seconds keys so reruns yield identical SHA-256 (`S-2`, `F29`).
    payload = {
        "schema": "starlink-referee-revision-diagnostics/2",
        "generated_by": "scripts/run_referee_revision_diagnostics.py",
        "provenance": provenance,
        "fast_msis_equivalence_check": bit_check,
        "fast_msis_verifier_controls": controls,
        "consumed_raw_input_inventory": raw_inventory,
        "paper1": p1,
        "paper2": p2,
        "paper3": p3,
    }
    raw_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    out_path.write_bytes(raw_bytes)
    print(f"Wrote {out_path} ({len(raw_bytes)} bytes, sha256={hashlib.sha256(raw_bytes).hexdigest()}) in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
