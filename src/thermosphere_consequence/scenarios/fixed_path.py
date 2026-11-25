"""Fixed-path atmospheric comparison.

The methodological device at the centre of this project. To ask "how much did
the storm change the atmosphere?" without confounding it with "how much did
the orbit change?", the trajectory is held FIXED and only the drivers vary.
The same sequence of positions and times is evaluated twice — once under the
observed drivers and once under a counterfactual quiet baseline — so every
difference in the answer is attributable to the atmosphere and to nothing else.

This is what separates a defensible attribution from a plausible story. A free
propagation under two atmospheres also diverges in POSITION, and by the second
day the two vehicles are sampling different altitudes at different local times,
so the density difference between them is no longer a clean measure of the
storm. The fixed path has no such contamination.

What the comparison can and cannot support:

  CAN: "along this trajectory, NRLMSISE-00 reports X% more mass density under
       the observed drivers than under the quiet baseline."
  CANNOT: "the storm increased drag by X%" — drag also depends on the drag
       coefficient, which itself shifts with composition and temperature. That
       is computed separately and reported separately.
  CANNOT: "composition alone increased drag" — a prohibited claim. The
       composition-invariance control in the aerodynamics tests exists to keep
       this honest.

Canonical tickets: P06_08_T001 through T014.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np

from ..aerodynamics import sentman
from ..domain.provenance import ValueStatus
from ..molecular import composition as comp
from ..molecular import kinetics
from ..propagation.propagator import (
    OrbitState,
    geodetic_latlon,
    greenwich_sidereal_time,
)
from ..protocols.atmosphere import (
    AtmosphereProvider,
    AtmosphereState,
    DriverLookup,
    DriverState,
)


@dataclass(frozen=True, slots=True)
class PathSample:
    """One point on the fixed path, with both atmospheres evaluated there."""

    time_utc: _dt.datetime
    altitude_m: float
    latitude_deg: float
    longitude_deg: float
    speed_m_s: float
    quiet: AtmosphereState
    storm: AtmosphereState

    @property
    def quiet_mass_density(self) -> float:
        return comp.total_species_mass_density(self.quiet.number_densities)

    @property
    def storm_mass_density(self) -> float:
        return comp.total_species_mass_density(self.storm.number_densities)

    @property
    def density_enhancement_fraction(self) -> float:
        """(storm - quiet) / quiet, signed. Never clamped."""
        q = self.quiet_mass_density
        if q <= 0:
            raise ValueError("quiet mass density is non-positive; enhancement undefined")
        return (self.storm_mass_density - q) / q


@dataclass(frozen=True, slots=True)
class FixedPathComparison:
    """The full comparison plus the caveats that must travel with it."""

    samples: tuple[PathSample, ...]
    quiet_drivers: DriverState
    provider_name: str
    provider_version: str
    caveats: tuple[str, ...]

    @property
    def mean_enhancement_fraction(self) -> float:
        return float(np.mean([s.density_enhancement_fraction for s in self.samples]))

    @property
    def peak_enhancement_fraction(self) -> float:
        return float(np.max([s.density_enhancement_fraction for s in self.samples]))

    def enhancement_percentiles(
        self, percentiles: tuple[float, ...] = (5, 25, 50, 75, 95)
    ) -> dict[float, float]:
        values = [s.density_enhancement_fraction for s in self.samples]
        return {p: float(np.percentile(values, p)) for p in percentiles}

    def by_altitude_band(self, edges_km: tuple[float, ...]) -> dict[str, float]:
        """Mean enhancement within each altitude band, for comparison with the literature.

        The published bands are what make this checkable: Fang et al. 2022 give
        50-125% at 200-400 km from WAM-IPE, Girgis et al. 2023 give 35-60% at
        200-300 km.
        """
        out: dict[str, float] = {}
        for lo, hi in zip(edges_km[:-1], edges_km[1:], strict=True):
            band = [
                s.density_enhancement_fraction
                for s in self.samples
                if lo * 1000.0 <= s.altitude_m < hi * 1000.0
            ]
            if band:
                out[f"{lo:.0f}-{hi:.0f} km"] = float(np.mean(band))
        return out


def quiet_baseline_drivers(
    observed: DriverState,
    *,
    quiet_ap: float = 4.0,
    keep_solar_flux: bool = True,
) -> DriverState:
    """Construct the counterfactual quiet-geomagnetic baseline.

    Solar flux is held at the OBSERVED value by default and only the
    geomagnetic drivers are quieted. This is the scientifically meaningful
    counterfactual: it asks "what if the geomagnetic storm had not happened,
    on this same day of this same solar cycle?" Quieting F10.7 as well would
    conflate the storm with the solar-cycle background and inflate the answer.

    ap = 4 corresponds to Kp ~ 1, a genuinely quiet geomagnetic state. It is a
    CHOICE, recorded as an assumption rather than presented as a measurement.
    """
    if quiet_ap < 0:
        raise ValueError(f"quiet ap must be non-negative, got {quiet_ap}")
    return DriverState(
        f107=observed.f107 if keep_solar_flux else 70.0,
        f107a=observed.f107a if keep_solar_flux else 70.0,
        ap_daily=quiet_ap,
        ap_history=(quiet_ap,) * 7,
        source=f"counterfactual quiet baseline (ap={quiet_ap}) over observed solar flux",
        source_url="",
        status=ValueStatus.ASSUMED,
    )


def sample_fixed_path(
    initial: OrbitState,
    duration_hours: float,
    provider: AtmosphereProvider,
    observed_drivers_at: DriverLookup,
    *,
    quiet_ap: float = 4.0,
    sample_interval_s: float = 120.0,
) -> FixedPathComparison:
    """Evaluate quiet and storm atmospheres along one unperturbed trajectory.

    The path is generated by Keplerian + J2 motion with NO drag, deliberately.
    Adding drag would make the path depend on the atmosphere, which is exactly
    the confound this function exists to remove.
    """
    from ..propagation.propagator import DragPropagator, SpacecraftProperties

    if duration_hours <= 0:
        raise ValueError(f"duration must be positive, got {duration_hours}")

    # A nominal spacecraft is required by the propagator's constructor but is
    # irrelevant here: the density function returns zero, so no drag is applied
    # and the trajectory is purely gravitational.
    nominal = SpacecraftProperties(mass_kg=1.0, reference_area_m2=1.0, drag_coefficient=1.0)
    drag_free = DragPropagator(lambda *_: 0.0, nominal, include_j2=True)
    path = drag_free.propagate(
        initial, duration_hours / 24.0, sample_interval_s=sample_interval_s
    )

    samples: list[PathSample] = []
    quiet_reference: DriverState | None = None

    for i, when in enumerate(path.times_utc):
        altitude = path.altitudes_m[i]
        if altitude < 100_000.0:
            continue

        position = np.asarray(path.positions_m[i])
        speed = float(np.linalg.norm(path.velocities_m_s[i]))
        gmst = greenwich_sidereal_time(when)
        latitude, longitude = geodetic_latlon(position, gmst)

        observed = observed_drivers_at(when)
        quiet = quiet_baseline_drivers(observed, quiet_ap=quiet_ap)
        if quiet_reference is None:
            quiet_reference = quiet

        samples.append(
            PathSample(
                time_utc=when,
                altitude_m=altitude,
                latitude_deg=latitude,
                longitude_deg=longitude,
                speed_m_s=speed,
                quiet=provider.state_at(when, altitude, latitude, longitude, quiet),
                storm=provider.state_at(when, altitude, latitude, longitude, observed),
            )
        )

    if not samples:
        raise ValueError("fixed path produced no samples above the reentry floor")
    if quiet_reference is None:
        raise RuntimeError("samples were produced without a quiet baseline")

    return FixedPathComparison(
        samples=tuple(samples),
        quiet_drivers=quiet_reference,
        provider_name=provider.name,
        provider_version=provider.version,
        caveats=(
            "Trajectory is drag-free by construction so the density difference "
            "is attributable to drivers alone. It is NOT the trajectory the "
            "vehicle actually flew.",
            "NRLMSISE-00 is an empirical model documented to underestimate rapid "
            "storm-time density enhancements relative to accelerometer "
            "observations and physics-based models during main-phase storms "
            "(e.g., Fang et al. 2022; He et al. 2023 report MSIS 2.0 "
            "underestimating Swarm-A enhancements near 438 km by ~50-70% on "
            "3-4 Feb 2022), though bias varies with altitude, storm phase, and "
            "quiet baseline; not an established universal lower bound.",
            f"Quiet baseline is a counterfactual assumption (ap={quiet_ap}), not "
            "an observation. Solar flux held at observed values so the "
            "comparison isolates the geomagnetic storm.",
        ),
    )


def diagnose_sample(
    sample: PathSample, *, accommodation: float = sentman.NOMINAL_ACCOMMODATION
) -> dict[str, float]:
    """Full gas-kinetic and aerodynamic diagnostics at one path point.

    Reports the quantities that let a reader see WHY drag changed, not merely
    that it did: mean particle mass, scale heights, Knudsen number, per-species
    impact flux, and the composition-weighted drag coefficient under both
    atmospheres.
    """
    gravity = kinetics.local_gravity(sample.altitude_m)

    def _one_atmosphere(state: AtmosphereState) -> tuple[float, float, dict[str, float]]:
        """Return (mass density, plate C_d, the rest of the diagnostics).

        The two quantities the caller goes on to combine are returned as named
        values rather than fished back out of the dictionary by key. Reading a
        dynamically-built key back by literal name is how a renamed label turns
        into a KeyError at runtime instead of a type error at edit time.
        """
        nd = state.number_densities
        rho = comp.total_species_mass_density(nd)
        cd_plate = sentman.species_weighted_drag_coefficient(
            nd, state.temperature_k, sample.speed_m_s,
            accommodation=accommodation, geometry="plate",
        )
        extra: dict[str, float] = {
            "temperature_k": state.temperature_k,
            "mean_particle_mass": comp.mean_particle_mass(nd),
            "total_number_density": comp.total_number_density(nd),
            "cd_sphere": sentman.species_weighted_drag_coefficient(
                nd, state.temperature_k, sample.speed_m_s,
                accommodation=accommodation, geometry="sphere",
            ),
        }
        for species in ("O", "N2"):
            if species in nd:
                extra[f"scale_height_{species}"] = kinetics.species_scale_height(
                    species, state.temperature_k, gravity
                )
                extra[f"flux_{species}"] = nd[species] * sample.speed_m_s
        return rho, cd_plate, extra

    quiet_rho, quiet_cd, quiet_extra = _one_atmosphere(sample.quiet)
    storm_rho, storm_cd, storm_extra = _one_atmosphere(sample.storm)

    out: dict[str, float] = {
        "quiet_mass_density": quiet_rho,
        "storm_mass_density": storm_rho,
        "quiet_cd_plate": quiet_cd,
        "storm_cd_plate": storm_cd,
        "density_enhancement_fraction": sample.density_enhancement_fraction,
        "cd_change_fraction": (storm_cd - quiet_cd) / quiet_cd,
        # The PRODUCT is what drag actually scales with. Reporting it separately
        # stops the density enhancement being read as the drag enhancement,
        # which is the specific misreading this project exists to prevent.
        "drag_enhancement_fraction": (storm_rho * storm_cd) / (quiet_rho * quiet_cd) - 1.0,
    }
    for prefix, extra in (("quiet", quiet_extra), ("storm", storm_extra)):
        for key, value in extra.items():
            out[f"{prefix}_{key}"] = value
    return out
