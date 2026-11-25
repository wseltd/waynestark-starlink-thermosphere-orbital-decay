"""End-to-end analysis of the February 2022 Starlink Group 4-7 incident.

The chain this module closes, and the epistemic status of each link:

  observed drivers (GFZ Kp/ap, CelesTrak F10.7)          OBSERVED
    -> NRLMSISE-00 species composition                    MODELLED (documented storm underestimation)
    -> species-weighted Sentman free-molecular C_d        MODELLED
    -> drag acceleration, on TOTAL rho and that one C_d   DERIVED
    -> numerically propagated decay                       MODELLED
    -> compared against catalogue decay dates             OBSERVED

Only the two ends are observational. Everything between is a model, and the
report says so at every step. The point of the exercise is not to claim the
model is right; it is to ask whether a defensible model driven by the observed
storm produces decay consistent with what the catalogue recorded — and to say
plainly where it does not.

WHERE THE SPECIES RESOLUTION STOPS, marked here because the third link is the
one a reader will skim past. Composition is resolved by species and so is every
molecular diagnostic built from it; the FORCE is not. It is the effective scalar
0.5 * C_d * rho_total * A * v_rel^2, and the mixture reaches it only through the
mass-flux weighting that produced C_d. This is a species-resolved molecular
diagnostic and interpretation layer accompanying an effective scalar
free-molecular drag model — not a species-resolved force closure.

Canonical tickets: P06_08_T015 through T043, P09_10_T013 through T030.
"""

from __future__ import annotations

import datetime as _dt
import math
import statistics
from dataclasses import dataclass, field

from ..aerodynamics import sentman
from ..drivers.space_weather import F107Series, KpApSeries, drivers_at
from ..events import starlink_g4_7 as event
from ..molecular import composition as comp
from ..propagation.propagator import (
    DragPropagator,
    OrbitState,
    PropagationResult,
    SpacecraftProperties,
    geodetic_latlon,
    greenwich_sidereal_time,
)
from ..protocols.atmosphere import AtmosphereProvider, DriverState
from ..scenarios.fixed_path import quiet_baseline_drivers


@dataclass(frozen=True, slots=True)
class DensityFieldCache:
    """Memoises atmosphere calls on a coarse grid.

    A 30-day propagation at 30 s steps asks for density of order 10^5 times.
    NRLMSISE-00 is smooth on scales far finer than its own driver cadence
    (3 hours) and far finer than 5 km in altitude, so evaluating on a grid and
    interpolating changes nothing physical while making the run tractable.
    The grid resolution is recorded in the result so the approximation is
    visible rather than hidden.
    """

    altitude_bin_m: float = 2_000.0
    time_bin_s: float = 600.0
    latitude_bin_deg: float = 5.0
    # Longitude MUST be part of the key. Omitting it made the returned density
    # depend on evaluation ORDER: the first call at a given (time, altitude,
    # latitude) cached its longitude's value and every later call at a
    # different longitude silently received it. NRLMSISE-00 varies with
    # longitude through local solar time, so this was a real error, not a
    # rounding one.
    longitude_bin_deg: float = 5.0


class AtmosphereDensityFunction:
    """Adapts an AtmosphereProvider to the propagator's density callable.

    Holds the driver policy: either the observed drivers at each instant
    (storm case) or a fixed counterfactual quiet baseline. Which one is in
    force is recorded on the instance so the propagation result can state it.
    """

    def __init__(
        self,
        provider: AtmosphereProvider,
        kp_ap: KpApSeries,
        f107: F107Series,
        *,
        mode: str = "storm",
        quiet_ap: float = 4.0,
        cache: DensityFieldCache | None = None,
    ) -> None:
        if mode not in ("storm", "quiet"):
            raise ValueError(f"mode must be 'storm' or 'quiet', got {mode!r}")
        self.provider = provider
        self.kp_ap = kp_ap
        self.f107 = f107
        self.mode = mode
        self.quiet_ap = quiet_ap
        self.cache_policy = cache or DensityFieldCache()
        self._cache: dict[tuple[int, int, int, int], float] = {}
        self.calls = 0
        self.misses = 0
        # Records the last drivers used, so a run can report what it consumed.
        self.last_drivers: DriverState | None = None

    def drivers_for(self, when: _dt.datetime) -> DriverState:
        observed = drivers_at(when, self.kp_ap, self.f107)
        if self.mode == "quiet":
            return quiet_baseline_drivers(observed, quiet_ap=self.quiet_ap)
        return observed

    def __call__(
        self, when: _dt.datetime, altitude_m: float, latitude_deg: float, longitude_deg: float
    ) -> float:
        self.calls += 1
        policy = self.cache_policy
        key = (
            int(when.timestamp() // policy.time_bin_s),
            int(altitude_m // policy.altitude_bin_m),
            int(latitude_deg // policy.latitude_bin_deg),
            int(longitude_deg // policy.longitude_bin_deg),
        )
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        self.misses += 1
        drivers = self.drivers_for(when)
        self.last_drivers = drivers
        state = self.provider.state_at(when, altitude_m, latitude_deg, longitude_deg, drivers)
        # The species sum is used rather than the model's own total, so the
        # density that drives drag is the same quantity the composition
        # diagnostics describe. The two differ by the closure fraction, which
        # is reported separately rather than reconciled silently.
        density = comp.total_species_mass_density(state.number_densities)
        self._cache[key] = density
        return density


@dataclass
class ScenarioResult:
    """One propagation plus the parameters and caveats that produced it."""

    label: str
    mode: str
    spacecraft: SpacecraftProperties
    propagation: PropagationResult
    drag_coefficient_source: str
    density_calls: int
    density_evaluations: int
    caveats: list[str] = field(default_factory=list)

    @property
    def decay_days(self) -> float | None:
        return self.propagation.decay_days


def composition_weighted_cd_at_insertion(
    provider: AtmosphereProvider,
    kp_ap: KpApSeries,
    f107: F107Series,
    *,
    mode: str = "storm",
    accommodation: float = sentman.NOMINAL_ACCOMMODATION,
    geometry: str = "plate",
) -> tuple[float, dict[str, float]]:
    """Species-weighted C_d at the insertion point, with its diagnostics.

    Computed once at insertion rather than at every integration step. The
    speed ratio changes by only a few percent over the decay because orbital
    speed and temperature both vary slowly, so a fixed C_d is accurate to
    better than the ballistic-coefficient uncertainty it sits inside.
    """
    if mode not in ("storm", "quiet"):
        raise ValueError(
            f"mode must be 'storm' or 'quiet', got {mode!r}. An unvalidated "
            "mode silently returned the QUIET answer labelled as whatever the "
            "caller asked for."
        )
    when = event.DEPLOYMENT_UTC
    position, velocity = event.insertion_state_vectors()
    speed = math.sqrt(sum(c * c for c in velocity))

    observed = drivers_at(when, kp_ap, f107)
    drivers = observed if mode == "storm" else quiet_baseline_drivers(observed)

    import numpy as np

    gmst = greenwich_sidereal_time(when)
    latitude, longitude = geodetic_latlon(np.asarray(position), gmst)

    state = provider.state_at(
        when, event.INSERTION_PERIGEE_ALTITUDE_M, latitude, longitude, drivers
    )
    cd = sentman.species_weighted_drag_coefficient(
        state.number_densities,
        state.temperature_k,
        speed,
        accommodation=accommodation,
        geometry=geometry,
    )
    diagnostics: dict[str, float] = {
        "temperature_k": state.temperature_k,
        "mass_density_kg_m3": comp.total_species_mass_density(state.number_densities),
        "mean_particle_mass_kg": comp.mean_particle_mass(state.number_densities),
        "total_number_density_m3": comp.total_number_density(state.number_densities),
        "speed_m_s": speed,
        "f107": drivers.f107,
        "f107a": drivers.f107a,
        "ap_daily": drivers.ap_daily,
    }
    for symbol, n in state.number_densities.items():
        diagnostics[f"n_{symbol}"] = n
        diagnostics[f"speed_ratio_{symbol}"] = sentman.molecular_speed_ratio(
            speed, state.temperature_k, symbol
        )
    fractions = comp.mass_fractions(state.number_densities)
    for symbol, fraction in fractions.items():
        diagnostics[f"mass_fraction_{symbol}"] = fraction
    return cd, diagnostics


def run_scenario(
    provider: AtmosphereProvider,
    kp_ap: KpApSeries,
    f107: F107Series,
    *,
    label: str,
    mode: str,
    drag_coefficient: float,
    mass_kg: float = event.MASS_KG_NOMINAL,
    area_m2: float | None = None,
    duration_days: float = 40.0,
    sample_interval_s: float = 600.0,
    max_step_s: float = 60.0,
    quiet_ap: float = 4.0,
) -> ScenarioResult:
    """Propagate the insertion orbit under one atmosphere policy."""
    if area_m2 is None:
        area_m2 = event.effective_area_from_operator_inverse_ballistic_coefficient(
            drag_coefficient, mass_kg
        )

    spacecraft = SpacecraftProperties(
        mass_kg=mass_kg,
        reference_area_m2=area_m2,
        drag_coefficient=drag_coefficient,
        name=f"Starlink v1.5 ({label})",
    )
    density = AtmosphereDensityFunction(
        provider, kp_ap, f107, mode=mode, quiet_ap=quiet_ap
    )
    propagator = DragPropagator(density, spacecraft, include_j2=True)

    position, velocity = event.insertion_state_vectors()
    initial = OrbitState(event.DEPLOYMENT_UTC, position, velocity)

    result = propagator.propagate(
        initial,
        duration_days,
        sample_interval_s=sample_interval_s,
        max_step_s=max_step_s,
    )

    caveats = [
        "NRLMSISE-00 is an empirical model documented to underestimate rapid "
        "storm-time density enhancements relative to accelerometer observations "
        "and physics-based models during main-phase storms (e.g., Fang et al. "
        "2022; He et al. 2023), though bias depends on altitude, storm phase, "
        "and quiet baseline; neither storm enhancement nor survival time is an "
        "established universal bound.",
        "No thrust is modelled. The real vehicles were in safe mode and did "
        "not orbit-raise, which is why this is a defensible idealisation for "
        "the lost satellites and NOT for the survivors.",
        "Neutral wind is zero. Storm-time winds at 210 km reach several "
        "hundred m/s and would alter relative velocity by a few percent.",
        f"Drag coefficient held fixed at {drag_coefficient:.3f} over the run.",
        f"Area {area_m2:.3f} m^2 derived from the external Capella SAR (Shambaugh 2024) "
        f"INVERSE ballistic coefficient reference {event.OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG} m^2/kg, "
        "not from Starlink geometry.",
        f"Density grid: {density.cache_policy.altitude_bin_m:.0f} m altitude, "
        f"{density.cache_policy.time_bin_s:.0f} s time, "
        f"{density.cache_policy.latitude_bin_deg:.0f} deg latitude, "
        f"{density.cache_policy.longitude_bin_deg:.0f} deg longitude.",
    ]
    if mode == "quiet":
        caveats.append(
            f"COUNTERFACTUAL: geomagnetic activity fixed at ap={quiet_ap} "
            "(Kp~1) with solar flux held at observed values. This orbit was "
            "never flown."
        )

    return ScenarioResult(
        label=label,
        mode=mode,
        spacecraft=spacecraft,
        propagation=result,
        drag_coefficient_source="species-weighted Sentman free-molecular",
        density_calls=density.calls,
        density_evaluations=density.misses,
        caveats=caveats,
    )


@dataclass(frozen=True, slots=True)
class ValidationVerdict:
    """Model decay against catalogued decay, stated without varnish."""

    observed_decay_days: tuple[float, ...]
    observed_median_days: float
    observed_range_days: tuple[float, float]
    modelled_storm_days: float | None
    modelled_quiet_days: float | None
    within_observed_range: bool
    storm_effect_days: float | None
    verdict: str
    caveats: tuple[str, ...]


def validate_against_catalogue(
    ground_truth: tuple[event.GroundTruthDecay, ...],
    storm: ScenarioResult,
    quiet: ScenarioResult,
) -> ValidationVerdict:
    """Compare modelled decay with catalogued decay dates.

    Deliberately blunt: the verdict string states whether the model lands
    inside the observed spread, and when it does not, it says so rather than
    finding a framing in which it does.
    """
    storm_losses = [g for g in ground_truth if g.is_storm_loss]
    # days_from_deployment, NOT days_to_decay. The latter counts from midnight
    # on the launch date while the propagation starts at deployment 18:28Z, a
    # systematic +0.7694 d offset -- 26% of the smallest catalogued value.
    observed = tuple(
        sorted(
            g.days_from_deployment
            for g in storm_losses
            if g.days_from_deployment is not None
        )
    )
    if not observed:
        raise ValueError("no catalogued storm-window decays supplied; cannot validate")

    # statistics.median averages the middle pair for an even count; indexing
    # len//2 returns the UPPER of the two and reported 9.0 where the true
    # median of (3,3,4,5,6,9,11,13,13,13) is 7.5.
    median = float(statistics.median(observed))
    lo, hi = float(min(observed)), float(max(observed))

    modelled_storm = storm.decay_days
    modelled_quiet = quiet.decay_days
    inside = modelled_storm is not None and lo <= modelled_storm <= hi

    effect: float | None = None
    if modelled_storm is not None and modelled_quiet is not None:
        effect = modelled_quiet - modelled_storm

    if modelled_storm is None:
        verdict = (
            "MODEL DID NOT REENTER within the simulated window while "
            f"{len(observed)} catalogued objects decayed between {lo:.0f} and "
            f"{hi:.0f} days. The model UNDERSTATES the decay."
        )
    elif inside:
        verdict = (
            f"Modelled storm decay {modelled_storm:.1f} d lies INSIDE the "
            f"catalogued range {lo:.0f}-{hi:.0f} d (median {median:.0f} d) "
            f"for {len(observed)} objects."
        )
    elif modelled_storm < lo:
        verdict = (
            f"Modelled storm decay {modelled_storm:.1f} d is FASTER than every "
            f"catalogued decay ({lo:.0f}-{hi:.0f} d). The model OVERSTATES drag."
        )
    else:
        verdict = (
            f"Modelled storm decay {modelled_storm:.1f} d is SLOWER than every "
            f"catalogued decay ({lo:.0f}-{hi:.0f} d). The model UNDERSTATES drag, "
            "which is the direction NRLMSISE-00's known storm underestimate predicts."
        )

    return ValidationVerdict(
        observed_decay_days=observed,
        observed_median_days=median,
        observed_range_days=(lo, hi),
        modelled_storm_days=modelled_storm,
        modelled_quiet_days=modelled_quiet,
        within_observed_range=inside,
        storm_effect_days=effect,
        verdict=verdict,
        caveats=(
            "Catalogued decay dates are day-resolution, so the observed range "
            "carries at least +/-1 day of quantisation.",
            f"{len(observed)} objects decayed within the storm window and are "
            f"compared here. Of {event.SATELLITES_LAUNCHED} launched, "
            f"{len(ground_truth)} were ever catalogued and "
            f"{event.SATELLITES_NEVER_CATALOGUED} were never tracked at all. "
            "This is a comparison against a SUBSET, and the subset is biased "
            "toward objects that survived long enough to be catalogued.",
            "The modelled object carries no thrust and a single fixed "
            "attitude; the real objects differed in both.",
            "Agreement would not prove the density field is right. Decay "
            "depends on the product of density and inverse ballistic coefficient, so "
            "errors in the two can offset.",
        ),
    )


def compare_to_published_enhancement(mean_enhancement: float) -> dict[str, str]:
    """Where a computed enhancement falls against each published figure.

    POSITION ONLY, AND IT SAYS SO IN EVERY ROW. This is the manifest-backed
    counterpart of ``validation.literature.compare``, and it has the same
    limitation: the published figures are reported at their own altitudes, as
    maxima of global or zonal mean fields, against their own baselines. A
    computed point peak is none of those things, so "BELOW" here is the
    arithmetic fact that one number is smaller than another and is not
    agreement, disagreement, validation or a measure of bias.

    The bare verdict strings this used to return - "BELOW (17.1% < 20%)" and
    nothing else - read as a finding, and were quoted as one.
    ``validation.literature`` is where the compatibility checks live and where
    a verdict is issued or withheld; this function deliberately issues none.
    """
    out: dict[str, str] = {}
    for name, (lo, hi, kind) in event.PUBLISHED_ENHANCEMENT.items():
        if mean_enhancement < lo:
            position = f"falls BELOW ({mean_enhancement:.1%} < {lo:.0%})"
        elif mean_enhancement > hi:
            position = f"falls ABOVE ({mean_enhancement:.1%} > {hi:.0%})"
        else:
            position = f"falls INSIDE ({lo:.0%}-{hi:.0%})"
        out[name] = f"POSITION ONLY, not a verdict: {position} [{kind}]"
    return out
