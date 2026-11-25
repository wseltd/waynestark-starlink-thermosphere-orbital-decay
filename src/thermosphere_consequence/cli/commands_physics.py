"""The `physics-diagnostics` subcommand: gas-kinetic state at a point.

A thin adapter. It resolves arguments, calls library functions and formats the
result; it contains no scientific logic, and an AST test enforces that by
rejecting arithmetic in this module.

Why it exists as a command rather than only as a library call: the quantities
here — mean particle mass, species scale heights, speed ratios, Knudsen number,
impact flux — are what let a reader see WHY drag changed rather than only that
it did. Having to write a script to see them means nobody looks.

Canonical ticket: P00_02_T049.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Declare the subcommand's arguments beside its implementation."""
    parser.add_argument(
        "--time", default="2022-02-03T18:28:00+00:00",
        help="UTC instant, ISO-8601 and timezone-aware (default: deployment epoch)",
    )
    parser.add_argument("--altitude-km", type=float, default=210.0)
    parser.add_argument("--latitude-deg", type=float, default=53.22)
    parser.add_argument("--longitude-deg", type=float, default=0.0)
    parser.add_argument(
        "--speed-m-s", type=float, default=None,
        help="bulk speed for the speed ratio (default: circular speed at altitude)",
    )
    parser.add_argument(
        "--mode", choices=("storm", "quiet"), default="storm",
        help="observed drivers, or the counterfactual quiet baseline",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--store-root", default="data/raw",
        help="raw store to read drivers from; run scripts/fetch_drivers.py first",
    )
    return parser


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    from ..aerodynamics import sentman
    from ..domain.temporal import parse_aware_datetime
    from ..domain.units import to_si
    from ..drivers import space_weather as sw
    from ..drivers.raw_store import RawStore
    from ..molecular import composition as comp
    from ..molecular import impact, kinetics
    from ..providers.nrlmsise00 import Nrlmsise00Provider
    from ..scenarios.fixed_path import quiet_baseline_drivers

    when = parse_aware_datetime(args.time)
    # Converted through the unit registry rather than by a literal here: an
    # adapter multiplying by 1000 is the second place a unit can be got wrong.
    altitude_m = to_si(args.altitude_km, "kilometer", "length")

    store = RawStore(args.store_root)
    kp = sw.parse_gfz_kp_ap(
        store.read("gfz", "kp_2022_extended", ".json"),
        store.read("gfz", "kp_2022_extended_ap", ".json"),
    )
    f107 = sw.parse_celestrak_f107(store.read("celestrak", "sw_last5years", ".csv"))

    observed = sw.drivers_at(when, kp, f107)
    drivers = observed if args.mode == "storm" else quiet_baseline_drivers(observed)

    state = Nrlmsise00Provider().state_at(
        when, altitude_m, args.latitude_deg, args.longitude_deg, drivers
    )
    speed = (
        args.speed_m_s
        if args.speed_m_s is not None
        else kinetics.circular_orbital_speed(altitude_m)
    )
    gravity = kinetics.local_gravity(altitude_m)
    densities = state.number_densities

    species: dict[str, dict[str, float]] = {}
    fractions = comp.mass_fractions(densities)
    for symbol, number_density in sorted(densities.items()):
        entry: dict[str, float] = {
            "number_density_m3": number_density,
            "mass_fraction": fractions.get(symbol, float("nan")),
            "speed_ratio": sentman.molecular_speed_ratio(speed, state.temperature_k, symbol),
            "scale_height_m": kinetics.species_scale_height(
                symbol, state.temperature_k, gravity
            ),
            "impact_flux_m2_s": impact.molecular_impact_flux(number_density, speed),
        }
        species[symbol] = entry

    return {
        "time_utc": when.isoformat(),
        "altitude_m": altitude_m,
        "latitude_deg": args.latitude_deg,
        "longitude_deg": args.longitude_deg,
        "mode": args.mode,
        "epistemic_status": "MODELLED",
        "provider": state.provider_name,
        "provider_version": state.provider_version,
        "drivers": {
            "f107": drivers.f107,
            "f107a": drivers.f107a,
            "ap_daily": drivers.ap_daily,
            "status": str(drivers.status),
            "source": drivers.source,
        },
        "temperature_k": state.temperature_k,
        "mass_density_kg_m3": comp.total_species_mass_density(densities),
        "total_number_density_m3": comp.total_number_density(densities),
        "mean_particle_mass_kg": comp.mean_particle_mass(densities),
        "bulk_speed_m_s": speed,
        "local_gravity_m_s2": gravity,
        "drag_coefficient_plate": sentman.species_weighted_drag_coefficient(
            densities, state.temperature_k, speed, geometry="plate"
        ),
        "drag_coefficient_sphere": sentman.species_weighted_drag_coefficient(
            densities, state.temperature_k, speed, geometry="sphere"
        ),
        "species": species,
        "caveats": list(state.notes),
    }


def _render_text(payload: dict[str, Any]) -> str:
    lines = [
        f"gas-kinetic diagnostics  {payload['time_utc']}  "
        f"{payload['altitude_m'] / 1000:.1f} km  "
        f"{payload['latitude_deg']:.2f} deg  [{payload['mode']}]",
        f"  provider          {payload['provider']} ({payload['provider_version']})",
        f"  epistemic status  {payload['epistemic_status']}",
        f"  drivers           F10.7 {payload['drivers']['f107']:.1f}, "
        f"81-day {payload['drivers']['f107a']:.1f}, ap {payload['drivers']['ap_daily']:.1f}",
        f"  temperature       {payload['temperature_k']:.1f} K",
        f"  mass density      {payload['mass_density_kg_m3']:.4e} kg/m^3",
        f"  number density    {payload['total_number_density_m3']:.4e} m^-3",
        f"  mean particle m   {payload['mean_particle_mass_kg']:.4e} kg",
        f"  bulk speed        {payload['bulk_speed_m_s']:.1f} m/s",
        f"  C_d plate         {payload['drag_coefficient_plate']:.4f}",
        f"  C_d sphere        {payload['drag_coefficient_sphere']:.4f}",
        "",
        f"  {'species':<12} {'n (m^-3)':>12} {'mass frac':>10} {'s':>7} "
        f"{'H (km)':>9} {'flux (m^-2 s^-1)':>18}",
    ]
    for symbol, entry in payload["species"].items():
        lines.append(
            f"  {symbol:<12} {entry['number_density_m3']:>12.4e} "
            f"{entry['mass_fraction']:>10.5f} {entry['speed_ratio']:>7.2f} "
            f"{entry['scale_height_m'] / 1000:>9.2f} {entry['impact_flux_m2_s']:>18.4e}"
        )
    if payload["caveats"]:
        lines.append("")
        for caveat in payload["caveats"]:
            lines.append(f"  CAVEAT  {caveat}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    """Execute the subcommand. Returns a process exit status."""
    try:
        payload = _collect(args)
    except FileNotFoundError as error:
        print(
            f"required driver data is missing: {error}\n"
            "Run scripts/fetch_drivers.py first; this command reads only from "
            "the raw store and never the network.",
        )
        return 2
    except ValueError as error:
        print(f"invalid request: {error}")
        return 2

    if args.format == "json":
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(_render_text(payload))
    return 0
