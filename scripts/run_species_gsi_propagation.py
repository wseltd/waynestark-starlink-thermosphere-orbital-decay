"""First-orbit path sampling and the declared molecular propagation set.

TWO JOBS, both downstream of the validated fixed-point reanalysis:

1. FIRST-ORBIT FIXED PATH. The insertion orbit is propagated for one
   revolution under the scalar model (operator gamma_eff, NRLMSIS 2.1 storm
   atmosphere), and the species-resolved molecular force is EVALUATED along
   the resulting trajectory samples with a ram-fixed attitude — the same
   "trajectory held constant so differences are attributable to the model"
   construction the frozen study used for atmospheres, now applied to the
   force model.

2. DECLARED PROPAGATION SET (four cases, stated in full here and nowhere
   widened): scalar baseline (gamma_eff = 0.085 m^2/kg external Capella SAR
   reference from Shambaugh 2024, NRLMSIS 2.1 storm); molecular Sentman-nominal at the three named proxy
   attitudes (edge-on, 30 deg, broadside), ram-fixed. The molecular
   integrator below mirrors DragPropagator's integration contract exactly —
   DOP853, the same reentry/escape events, the same reentry floor, the same
   tolerances — so the comparison isolates the FORCE MODEL and nothing else.
   NRLMSIS 2.1 supplies every atmosphere; storm-time ap history throughout.

ATTITUDE IN PROPAGATION. Ram-fixed: body x along v_rel; body z chosen along
the component of the radial direction perpendicular to v_rel; body y
completes the right-handed frame (approximately the orbit normal). The named
pitch cases rotate the proxy about body y exactly as at the fixed point.
None of this is a claim about the flown attitude — the three cases bracket.

Energy accounting keeps the Version B conventions: the inertial
orbital-energy rate is dot(a_drag, v); the atmosphere-relative dissipation
is -dot(a_drag, v_rel); integrals are reported over the common 36 h horizon
(every case is still in orbit there), with decay time carrying the terminal
behaviour separately. Nothing is called "drag work".

Model-conditioned decay times are NEVER the observed spacecraft lifetime.
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.integrate import solve_ivp

from thermosphere_consequence.aerodynamics.faceted import FacetedFreeMolecularModel
from thermosphere_consequence.aerodynamics.gsi_accommodation import (
    PilinskiAltitudeAccommodation,
)
from thermosphere_consequence.aerodynamics.proxy_geometry import starlink_like_proxy
from thermosphere_consequence.domain.constants import MU_EARTH, OMEGA_EARTH
from thermosphere_consequence.drivers import space_weather as sw
from thermosphere_consequence.events import starlink_g4_7 as EV
from thermosphere_consequence.licence.policy_loader import load_licence_policy
from thermosphere_consequence.molecular import composition as comp
from thermosphere_consequence.propagation import energy_ledger as ledger
from thermosphere_consequence.propagation.propagator import (
    MAX_ALTITUDE_M,
    REENTRY_ALTITUDE_M,
    OrbitState,
    geodetic_altitude,
    geodetic_latlon,
    greenwich_sidereal_time,
    j2_acceleration,
)
from thermosphere_consequence.protocols.atmosphere import AtmosphereState, DriverState
from thermosphere_consequence.providers.nrlmsis2 import NRLMSIS21Provider
from thermosphere_consequence.scenarios import starlink_analysis as SA

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "species_resolved_gsi"
FIGURES_DIR = RESULTS_DIR / "figures"

#: Named attitude cases (deg pitch from edge-on), identical to the
#: fixed-point script's cases.
ATTITUDE_CASES = {
    "min_drag_edge_on": 0.0,
    "plausible_nominal_30deg": 30.0,
    "max_drag_broadside": 90.0,
}

#: The Version B common energy-ledger horizon [s].
LEDGER_HORIZON_S = 36.0 * 3600.0

#: Propagation cap, as in the frozen scenarios.
#: The lowest altitude at which EVERY bracketed reading of the Knudsen number
#: is still strongly free molecular (Kn > 10), from
#: results/species_resolved_gsi/flow_regime/. Crossing times are reported here
#: because the force model is supported here; the 100 km floor below is where
#: the integration is allowed to run on, and any time taken from it is an
#: extrapolation beyond the model's regime.
SUPPORTED_THRESHOLD_ALTITUDE_M = 170_000.0

DURATION_DAYS = 25.0
SAMPLE_INTERVAL_S = 600.0
MAX_STEP_S = 60.0

#: Accommodation wall temperature for the nominal propagation model.
NOMINAL_WALL_K = 300.0


def _load_audit_module() -> types.ModuleType:
    path = REPO_ROOT / "scripts" / "run_atmosphere_audit.py"
    spec = importlib.util.spec_from_file_location("run_atmosphere_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audit module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_roots() -> tuple[pathlib.Path, pathlib.Path]:
    missing = [n for n in ("NRLMSIS20_ROOT", "NRLMSIS21_ROOT") if not os.environ.get(n)]
    if missing:
        raise SystemExit("REFUSING TO RUN: " + ", ".join(missing) + " not set.")
    return (
        pathlib.Path(os.environ["NRLMSIS20_ROOT"]),
        pathlib.Path(os.environ["NRLMSIS21_ROOT"]),
    )


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _pitch_quaternion(pitch_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(pitch_deg) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def _ram_frame(position: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
    """Rows are the body axes in inertial coordinates (x = ram)."""
    x_body = v_rel / np.linalg.norm(v_rel)
    radial = position / np.linalg.norm(position)
    z_raw = radial - float(radial @ x_body) * x_body
    z_norm = float(np.linalg.norm(z_raw))
    if z_norm < 1e-12:
        raise RuntimeError("degenerate ram frame: v_rel parallel to the radial")
    z_body = z_raw / z_norm
    y_body = np.cross(z_body, x_body)
    return np.vstack([x_body, y_body, z_body])


@dataclass
class MolecularCase:
    """One declared molecular propagation case and its raw outputs."""

    name: str
    pitch_deg: float
    times_s: list[float]
    altitudes_m: list[float]
    positions: list[np.ndarray]
    velocities: list[np.ndarray]
    drag_accels: list[np.ndarray]
    reentered: bool
    reentry_time_s: float | None
    #: First crossing of the FREE-MOLECULAR SUPPORTED threshold, recorded from
    #: the same trajectory as `reentry_time_s` rather than from a second run.
    #: Below this altitude the force model is outside its own flow regime, so a
    #: time that depends on integrating past it is an extrapolated diagnostic
    #: and is labelled as one. None means the run never reached the threshold.
    supported_threshold_m: float = SUPPORTED_THRESHOLD_ALTITUDE_M
    supported_crossing_s: float | None = None


class MolecularPropagator:
    """DragPropagator's integration contract with the faceted force model.

    Mirrors propagator.DragPropagator deliberately: DOP853, the same
    terminal events at the same altitude floor and ceiling, gravity + J2,
    co-rotating atmosphere. The only difference IS the science under test:
    the acceleration comes from the species-and-facet-resolved model
    evaluated in a ram-fixed frame, divided by the event's nominal mass.
    """

    def __init__(
        self,
        model: FacetedFreeMolecularModel,
        provider: NRLMSIS21Provider,
        drivers_at: object,
        attitude: tuple[float, float, float, float],
        mass_kg: float,
    ) -> None:
        self.model = model
        self.provider = provider
        self.drivers_at = drivers_at
        self.attitude = attitude
        self.mass_kg = mass_kg
        self._state_cache: tuple[tuple[float, bytes], AtmosphereState] | None = None

    def _atmosphere(
        self, when: dt.datetime, t_s: float, altitude: float, position: np.ndarray
    ) -> AtmosphereState:
        # KEYED ON TIME AND STATE, not time alone. The atmosphere depends on
        # altitude, latitude and longitude, all of which come from `position`,
        # so a key of t_s alone is wrong by construction: an adaptive solver is
        # entitled to call the derivative at one time with several trial
        # states, and this cache would have returned the first one's density
        # for all of them. A direct probe confirmed it: two positions 52 km
        # apart at the same t_s produced one provider call.
        #
        # It did not corrupt any published result - instrumenting a full decay
        # found 1,951 cache hits and zero at a differing position, because
        # DOP853's repeats are step restarts at an identical (t, y) - but a
        # cache that is safe only by the accident of one solver's call pattern
        # is not a cache this can rely on.
        key = (t_s, position.tobytes())
        if self._state_cache is not None and self._state_cache[0] == key:
            return self._state_cache[1]
        gmst = greenwich_sidereal_time(when)
        latitude, longitude = geodetic_latlon(position, gmst)
        drivers: DriverState = self.drivers_at(when)  # type: ignore[operator]
        state = self.provider.state_at(when, altitude, latitude, longitude, drivers)
        self._state_cache = (key, state)
        return state

    def drag_acceleration(
        self, when: dt.datetime, t_s: float, position: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        altitude = geodetic_altitude(position)
        if altitude < REENTRY_ALTITUDE_M:
            return np.zeros(3)
        omega = np.array([0.0, 0.0, OMEGA_EARTH.value])
        v_rel = velocity - np.cross(omega, position)
        state = self._atmosphere(when, t_s, altitude, position)
        frame = _ram_frame(position, v_rel)
        speed = float(np.linalg.norm(v_rel))
        result = self.model.evaluate(
            dict(state.number_densities),
            state.temperature_k,
            (speed, 0.0, 0.0),
            attitude_quaternion=self.attitude,
        )
        force_body = np.asarray(result.force_n)
        force_inertial = frame.T @ force_body
        return force_inertial / self.mass_kg

    def propagate(self, initial: OrbitState, name: str, pitch_deg: float) -> MolecularCase:
        epoch = initial.time_utc
        y0 = np.array([*initial.position_m, *initial.velocity_m_s], dtype=float)
        t_end = DURATION_DAYS * 86400.0

        def derivative(t_s: float, state: np.ndarray) -> np.ndarray:
            position = state[:3]
            velocity = state[3:]
            r = float(np.linalg.norm(position))
            acceleration = -MU_EARTH.value * position / (r**3) + j2_acceleration(
                position
            )
            when = epoch + dt.timedelta(seconds=float(t_s))
            acceleration = acceleration + self.drag_acceleration(
                when, t_s, position, velocity
            )
            return np.concatenate([velocity, acceleration])

        def reentry_event(t_s: float, state: np.ndarray) -> float:
            return geodetic_altitude(state[:3]) - REENTRY_ALTITUDE_M

        reentry_event.terminal = True  # type: ignore[attr-defined]
        reentry_event.direction = -1.0  # type: ignore[attr-defined]

        def supported_threshold_event(t_s: float, state: np.ndarray) -> float:
            return geodetic_altitude(state[:3]) - SUPPORTED_THRESHOLD_ALTITUDE_M

        # NOT terminal: the run continues to the floor so the extrapolated
        # diagnostic remains available, and both times come from ONE trajectory
        # rather than from two runs that could differ for other reasons.
        supported_threshold_event.terminal = False  # type: ignore[attr-defined]
        supported_threshold_event.direction = -1.0  # type: ignore[attr-defined]

        def escape_event(t_s: float, state: np.ndarray) -> float:
            return MAX_ALTITUDE_M - geodetic_altitude(state[:3])

        escape_event.terminal = True  # type: ignore[attr-defined]
        escape_event.direction = -1.0  # type: ignore[attr-defined]

        t_eval = np.arange(0.0, t_end, SAMPLE_INTERVAL_S)
        if t_eval.size == 0 or t_eval[-1] < t_end:
            t_eval = np.append(t_eval, t_end)

        solution = solve_ivp(
            fun=derivative,
            t_span=(0.0, t_end),
            y0=y0,
            method="DOP853",
            t_eval=t_eval,
            events=[reentry_event, escape_event, supported_threshold_event],
            rtol=1e-9,
            atol=1e-6,
            max_step=MAX_STEP_S,
            dense_output=False,
        )
        if not solution.success and solution.status != 1:
            raise RuntimeError(f"molecular integration failed: {solution.message}")

        reentered = len(solution.t_events[0]) > 0
        crossings = solution.t_events[2]
        # FIRST crossing. A grazing pass can dip below the threshold and climb
        # back out, and the quantity reported is the first time the spacecraft
        # reaches it, not the last.
        supported_crossing = float(crossings[0]) if len(crossings) else None
        times: list[float] = []
        altitudes: list[float] = []
        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        drag_accels: list[np.ndarray] = []
        for i, t_s in enumerate(solution.t):
            position = solution.y[:3, i]
            velocity = solution.y[3:, i]
            times.append(float(t_s))
            altitudes.append(geodetic_altitude(position))
            positions.append(position.copy())
            velocities.append(velocity.copy())
            when = epoch + dt.timedelta(seconds=float(t_s))
            drag_accels.append(
                self.drag_acceleration(when, float(t_s), position, velocity)
            )
        return MolecularCase(
            name=name,
            pitch_deg=pitch_deg,
            times_s=times,
            altitudes_m=altitudes,
            positions=positions,
            velocities=velocities,
            drag_accels=drag_accels,
            reentered=reentered,
            supported_crossing_s=supported_crossing,
            reentry_time_s=(
                float(solution.t_events[0][0]) if reentered else None
            ),
        )


def _energy_over_horizon(
    times_s: Sequence[float],
    positions: Sequence[np.ndarray],
    velocities: Sequence[np.ndarray],
    drag_accels: Sequence[np.ndarray],
) -> dict[str, float]:
    keep = [i for i, t in enumerate(times_s) if t <= LEDGER_HORIZON_S]
    inertial: list[float] = []
    dissipation: list[float] = []
    delta_v: list[float] = []
    kept_times: list[float] = []
    for i in keep:
        v_rel = ledger.relative_velocity_m_s(positions[i], velocities[i])
        a_drag = drag_accels[i]
        inertial.append(ledger.inertial_orbital_energy_rate_w_kg(a_drag, velocities[i]))
        dissipation.append(
            ledger.atmosphere_relative_dissipation_rate_w_kg(a_drag, v_rel)
        )
        delta_v.append(float(np.linalg.norm(a_drag)))
        kept_times.append(times_s[i])
    return {
        ledger.INERTIAL_KEY: float(np.trapezoid(inertial, kept_times)),
        ledger.DISSIPATION_KEY: float(np.trapezoid(dissipation, kept_times)),
        ledger.DELTA_V_KEY: float(np.trapezoid(delta_v, kept_times)),
        "integration_horizon_hours": LEDGER_HORIZON_S / 3600.0,
    }


def main() -> int:
    audit = _load_audit_module()
    os.chdir(REPO_ROOT)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _root20, root21 = _require_roots()
    policy = load_licence_policy(REPO_ROOT / "config" / "licence_policy.yaml")
    msis21 = NRLMSIS21Provider(
        policy=policy, mode="open_noncommercial_research", root=root21
    )
    kp, f107 = audit.load_drivers()

    def observed(when: dt.datetime) -> DriverState:
        return sw.drivers_at(when, kp, f107)

    proxy = starlink_like_proxy()
    mass_kg = EV.MASS_KG_NOMINAL
    accommodation = PilinskiAltitudeAccommodation(
        altitude_m=210_000.0,
        activity="solar_maximum",
        wall_temperature_k=NOMINAL_WALL_K,
    )
    model = FacetedFreeMolecularModel(
        proxy.facets,
        reference_area_m2=proxy.reference_area_m2,
        accommodation=accommodation,
        gas_surface_model="sentman_diffuse",
    )

    position, velocity = EV.insertion_state_vectors()
    initial = OrbitState(EV.DEPLOYMENT_UTC, position, velocity)

    # ---- first-orbit path sampling -----------------------------------------
    print("first-orbit path under the scalar model, molecular force sampled ...")
    gamma = EV.OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG
    scalar_first = SA.run_scenario(
        msis21,
        kp,
        f107,
        label="first-orbit path (scalar carrier)",
        mode="storm",
        drag_coefficient=2.2,
        duration_days=0.075,
        sample_interval_s=60.0,
        max_step_s=30.0,
    )
    path_rows: list[dict[str, object]] = []
    prop = scalar_first.propagation
    for i, when in enumerate(prop.times_utc):
        pos = np.asarray(prop.positions_m[i])
        vel = np.asarray(prop.velocities_m_s[i])
        v_rel = ledger.relative_velocity_m_s(pos, vel)
        altitude = prop.altitudes_m[i]
        gmst = greenwich_sidereal_time(when)
        latitude, longitude = geodetic_latlon(pos, gmst)
        state = msis21.state_at(when, altitude, latitude, longitude, observed(when))
        result = model.evaluate(
            dict(state.number_densities),
            state.temperature_k,
            (float(np.linalg.norm(v_rel)), 0.0, 0.0),
            attitude_quaternion=_pitch_quaternion(30.0),
        )
        force = np.asarray(result.force_n)
        flow = np.asarray(result.flow_unit_body)
        drag_n = float(force @ flow)
        row: dict[str, object] = {
            "time_utc": when.isoformat(),
            "altitude_m": altitude,
            "latitude_deg": latitude,
            "longitude_deg": longitude,
            "v_rel_m_s": float(np.linalg.norm(v_rel)),
            "rho_kg_m3": comp.total_species_mass_density(
                dict(state.number_densities)
            ),
            "drag_force_n": drag_n,
        }
        weighted = {
            symbol: value
            * state.number_densities.get(symbol, 0.0)
            * comp.species_mass_density(symbol, 1.0)
            for symbol, value in result.per_species_drag_coefficient.items()
        }
        total = sum(weighted.values()) or 1.0
        for symbol in sorted(weighted):
            row[f"fraction_{symbol}"] = weighted[symbol] / total
        path_rows.append(row)

    with (RESULTS_DIR / "first_orbit_path.csv").open("w", newline="") as fh:
        fieldnames = list(path_rows[0])
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(path_rows)
    print(f"  wrote first_orbit_path.csv ({len(path_rows)} rows)")

    # ---- declared propagation set ------------------------------------------
    print("declared propagation set (scalar + three molecular attitudes) ...")
    summary_rows: list[dict[str, object]] = []

    scalar_case = SA.run_scenario(
        msis21,
        kp,
        f107,
        label="scalar gamma_operator",
        mode="storm",
        drag_coefficient=2.2,
        duration_days=DURATION_DAYS,
        sample_interval_s=SAMPLE_INTERVAL_S,
        max_step_s=MAX_STEP_S,
    )
    sp = scalar_case.propagation
    scalar_positions = [np.asarray(p) for p in sp.positions_m]
    scalar_velocities = [np.asarray(v) for v in sp.velocities_m_s]
    scalar_times = [(t - sp.times_utc[0]).total_seconds() for t in sp.times_utc]
    scalar_drags: list[np.ndarray] = []
    for i in range(len(scalar_times)):
        v_rel = ledger.relative_velocity_m_s(scalar_positions[i], scalar_velocities[i])
        scalar_drags.append(
            ledger.drag_acceleration_m_s2(gamma, sp.density_kg_m3[i], v_rel)
        )
    energy = _energy_over_horizon(
        scalar_times, scalar_positions, scalar_velocities, scalar_drags
    )
    summary_rows.append(
        {
            "case": "scalar_gamma_operator_0.085",
            "pitch_deg": "",
            "decay_days": scalar_case.decay_days,
            "reentered": sp.reentered,
            "min_altitude_km": min(sp.altitudes_m) / 1000.0,
            "min_perigee_36h_km": min(
                per
                for per, t in zip(sp.perigee_altitude_m, scalar_times, strict=True)
                if t <= LEDGER_HORIZON_S
            )
            / 1000.0,
            "sma_start_km": sp.semi_major_axis_m[0] / 1000.0,
            "apogee_start_km": sp.apogee_altitude_m[0] / 1000.0,
            "termination": "reentry" if sp.reentered else "duration cap",
            **{k: v for k, v in energy.items()},
        }
    )
    print(f"  scalar: decay {scalar_case.decay_days}")

    series_paths: dict[str, list[tuple[float, float]]] = {
        "scalar_gamma_operator_0.085": list(
            zip(scalar_times, sp.altitudes_m, strict=True)
        )
    }
    for name, pitch in ATTITUDE_CASES.items():
        propagator = MolecularPropagator(
            model, msis21, observed, _pitch_quaternion(pitch), mass_kg
        )
        case = propagator.propagate(initial, name, pitch)
        energy = _energy_over_horizon(
            case.times_s, case.positions, case.velocities, case.drag_accels
        )
        decay_days = (
            case.reentry_time_s / 86400.0 if case.reentry_time_s is not None else None
        )
        summary_rows.append(
            {
                "case": f"molecular_sentman_{name}",
                "pitch_deg": pitch,
                "decay_days": decay_days,
                "reentered": case.reentered,
                "min_altitude_km": min(case.altitudes_m) / 1000.0,
                "min_perigee_36h_km": "",
                "sma_start_km": "",
                "apogee_start_km": "",
                "termination": "reentry" if case.reentered else "duration cap",
                **{k: v for k, v in energy.items()},
            }
        )
        series_paths[f"molecular_sentman_{name}"] = list(
            zip(case.times_s, case.altitudes_m, strict=True)
        )
        decay_text = f"{decay_days:.2f} d" if decay_days is not None else "no reentry"
        print(f"  molecular {name}: {decay_text}")

    with (RESULTS_DIR / "propagation_summary.csv").open("w", newline="") as fh:
        fieldnames = list(summary_rows[0])
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  wrote propagation_summary.csv ({len(summary_rows)} rows)")

    manifest_path = RESULTS_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["propagation"] = {
        "generated_by": "scripts/run_species_gsi_propagation.py",
        "git_commit": _git_commit(),
        "declared_cases": [str(row["case"]) for row in summary_rows],
        "duration_cap_days": DURATION_DAYS,
        "energy_horizon_hours": LEDGER_HORIZON_S / 3600.0,
        "note": (
            "model-conditioned decay times, never the observed spacecraft "
            "lifetime; the molecular integrator mirrors DragPropagator's "
            "contract so only the force model differs"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("  updated run_manifest.json")

    # ---- figures ------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "species-resolved-gsi"
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, series in series_paths.items():
        ax.plot(
            [t / 86400.0 for t, _ in series],
            [a / 1000.0 for _, a in series],
            label=name,
        )
    ax.set_xlabel("days since deployment")
    ax.set_ylabel("geodetic altitude (km)")
    ax.set_title("Model-conditioned decay: scalar vs molecular attitude cases")
    ax.legend(fontsize=8)
    fig.savefig(
        FIGURES_DIR / "propagation_comparison.svg",
        format="svg",
        metadata={"Date": None},
    )
    plt.close(fig)
    print("  wrote figures/propagation_comparison.svg")

    species = sorted(
        k.removeprefix("fraction_") for k in path_rows[0] if k.startswith("fraction_")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    altitudes_km = [float(str(r["altitude_m"])) / 1000.0 for r in path_rows]
    for symbol in species:
        ax.plot(
            altitudes_km,
            [float(str(r[f"fraction_{symbol}"])) for r in path_rows],
            ".",
            label=symbol,
            markersize=3,
        )
    ax.set_xlabel("altitude (km)")
    ax.set_ylabel("drag-force fraction")
    ax.set_title("Species force contribution vs altitude (first orbit, ram-fixed)")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(
        FIGURES_DIR / "species_contribution_vs_altitude.svg",
        format="svg",
        metadata={"Date": None},
    )
    plt.close(fig)
    print("  wrote figures/species_contribution_vs_altitude.svg")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
