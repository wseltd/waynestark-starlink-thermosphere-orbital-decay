"""Gas-kinetic diagnostics: gravity, scale heights, thermal speeds, flow regime.

Pure functions on SI floats. Canonical tickets: P00_02_T010, T011, T012, T013,
T014, T015, T016, T017.
"""

from __future__ import annotations

import math

from ..domain.constants import K_B, MU_EARTH, R_EARTH_MEAN
from ..domain.species import species_mass

__all__ = [
    "local_gravity",
    "circular_orbital_speed",
    "species_scale_height",
    "species_scale_heights",
    "isothermal_number_density",
    "scale_height_density_ratio",
    "hydrostatic_number_density",
    "most_probable_speed",
    "mean_thermal_speed",
    "rms_thermal_speed",
    "molecular_speed_ratio",
    "hard_sphere_mean_free_path",
    "knudsen_number",
    "FlowRegime",
    "classify_flow_regime",
]


def _positive(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} is not finite ({v})")
    if v <= 0.0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


# --------------------------------------------------------------------------
# gravity and orbital speed
# --------------------------------------------------------------------------


def local_gravity(altitude_m: float, radius_m: float = R_EARTH_MEAN.value) -> float:
    """g(h) = mu_E / (R_E + h)^2   [m/s^2].

    A scalar diagnostic evaluated on mean Earth radius. The orbit propagator
    uses the full position vector and the geodetic model, not this.
    Monotonically decreasing in altitude; always positive.
    """
    r = _positive("radius", radius_m) + float(altitude_m)
    if r <= 0.0:
        raise ValueError(f"geocentric radius must be positive, got {r}")
    return MU_EARTH.value / (r * r)


def circular_orbital_speed(altitude_m: float, radius_m: float = R_EARTH_MEAN.value) -> float:
    """v_circ = sqrt(mu_E / (R_E + h))   [m/s].

    Permitted in unit tests and explanatory examples only. Production drag uses
    the propagated state vector relative to the moving atmosphere; the insertion
    orbit is eccentric and a circular speed would misstate it.
    """
    r = _positive("radius", radius_m) + float(altitude_m)
    return math.sqrt(MU_EARTH.value / r)


# --------------------------------------------------------------------------
# scale heights and barometric diagnostics
# --------------------------------------------------------------------------


def species_scale_height(symbol: str, temperature_k: float, gravity_m_s2: float) -> float:
    """H_i = k_B T / (m_i g)   [m].

    Proportional to temperature at fixed gravity, and inversely proportional to
    particle mass — hence H_He > H_O > H_N2 > H_O2 > H_Ar at equal T and g.
    """
    t = _positive("temperature", temperature_k)
    g = _positive("gravity", gravity_m_s2)
    return K_B.value * t / (species_mass(symbol) * g)


def species_scale_heights(
    symbols: list[str] | tuple[str, ...], temperature_k: float, gravity_m_s2: float
) -> dict[str, float]:
    return {s: species_scale_height(s, temperature_k, gravity_m_s2) for s in symbols}


def isothermal_number_density(n0: float, delta_h_m: float, scale_height_m: float) -> float:
    """n(h) = n(h0) exp(-(h - h0) / H)   [1/m^3].

    Isothermal diffusive-equilibrium DIAGNOSTIC. It is not the storm atmosphere
    and must never be presented as one.
    """
    h = _positive("scale height", scale_height_m)
    n = float(n0)
    if n < 0.0:
        raise ValueError(f"reference number density is negative ({n})")
    return n * math.exp(-float(delta_h_m) / h)


def scale_height_density_ratio(
    delta_h_m: float, scale_height_quiet_m: float, scale_height_storm_m: float
) -> float:
    """Density ratio attributable to scale-height change alone [dimensionless].

    ratio = exp(delta_h * (1/H_quiet - 1/H_storm))

    A diagnostic of how much of an observed enhancement simple thermal
    expansion can account for. The residual against the model ratio is not
    attributable to any single mechanism.
    """
    hq = _positive("quiet scale height", scale_height_quiet_m)
    hs = _positive("storm scale height", scale_height_storm_m)
    return math.exp(float(delta_h_m) * (1.0 / hq - 1.0 / hs))


def hydrostatic_number_density(
    symbol: str,
    n0: float,
    altitudes_m: list[float] | tuple[float, ...],
    temperatures_k: list[float] | tuple[float, ...],
    radius_m: float = R_EARTH_MEAN.value,
) -> float:
    """Variable-temperature hydrostatic diagnostic.

    n(h1) = n(h0) * T(h0)/T(h1) * exp( -integral m_i g(h) / (k_B T(h)) dh )

    Integrated numerically by the trapezoidal rule over the supplied profile.
    Equilibrium only: excludes horizontal transport, vertical acceleration,
    storm-time compression, travelling atmospheric disturbances and
    time-dependent species chemistry.
    """
    if len(altitudes_m) != len(temperatures_k):
        raise ValueError("altitude and temperature profiles must be the same length")
    if len(altitudes_m) < 2:
        raise ValueError("hydrostatic integration needs at least two profile points")
    m_i = species_mass(symbol)
    integrand = [
        m_i * local_gravity(h, radius_m) / (K_B.value * _positive("temperature", t))
        for h, t in zip(altitudes_m, temperatures_k, strict=True)
    ]
    integral = math.fsum(
        0.5 * (integrand[k] + integrand[k + 1]) * (altitudes_m[k + 1] - altitudes_m[k])
        for k in range(len(altitudes_m) - 1)
    )
    t0 = _positive("temperature", temperatures_k[0])
    t1 = _positive("temperature", temperatures_k[-1])
    return float(n0) * (t0 / t1) * math.exp(-integral)


# --------------------------------------------------------------------------
# thermal speeds and speed ratio
# --------------------------------------------------------------------------


def most_probable_speed(symbol: str, temperature_k: float) -> float:
    """v_mp = sqrt(2 k_B T / m_i)   [m/s]."""
    t = _positive("temperature", temperature_k)
    return math.sqrt(2.0 * K_B.value * t / species_mass(symbol))


def mean_thermal_speed(symbol: str, temperature_k: float) -> float:
    """v_mean = sqrt(8 k_B T / (pi m_i))   [m/s]."""
    t = _positive("temperature", temperature_k)
    return math.sqrt(8.0 * K_B.value * t / (math.pi * species_mass(symbol)))


def rms_thermal_speed(symbol: str, temperature_k: float) -> float:
    """v_rms = sqrt(3 k_B T / m_i)   [m/s]. Always v_mp < v_mean < v_rms."""
    t = _positive("temperature", temperature_k)
    return math.sqrt(3.0 * K_B.value * t / species_mass(symbol))


def molecular_speed_ratio(symbol: str, relative_speed_m_s: float, temperature_k: float) -> float:
    """S_i = v_rel / v_mp_i   [dimensionless].

    Tests whether orbital motion dominates random thermal motion. S >> 1
    justifies the directed hyperthermal-beam approximation; when it is not
    large, a fuller free-molecular flux model is required.
    """
    return _positive("relative speed", relative_speed_m_s) / most_probable_speed(
        symbol, temperature_k
    )


# --------------------------------------------------------------------------
# mean free path and flow regime
# --------------------------------------------------------------------------


def hard_sphere_mean_free_path(total_number_density_m3: float, sigma_eff_m2: float) -> float:
    """lambda = 1 / (sqrt(2) n sigma_eff)   [m].

    A BOUNDING approximation with an explicitly supplied cross-section. No
    universal cross-section covers O-O, O-N2, O-O2, N2-N2 and O2-O2 across
    thermospheric temperatures, so the caller must supply a cited value and
    the resulting Knudsen number is coefficient-model dependent.
    """
    n = _positive("total number density", total_number_density_m3)
    sigma = _positive("effective cross-section", sigma_eff_m2)
    return 1.0 / (math.sqrt(2.0) * n * sigma)


def knudsen_number(mean_free_path_m: float | None, characteristic_length_m: float) -> float | None:
    """Kn = lambda / L   [dimensionless], or None when lambda is unavailable.

    Returns None rather than fabricating a value from altitude alone.
    """
    if mean_free_path_m is None:
        return None
    return _positive("mean free path", mean_free_path_m) / _positive(
        "characteristic length", characteristic_length_m
    )


class FlowRegime:
    """Flow-regime labels. ``UNKNOWN`` when the Knudsen number is unavailable."""

    CONTINUUM_LIKE = "continuum_like"
    TRANSITIONAL = "transitional"
    FREE_MOLECULAR = "strongly_free_molecular"
    UNKNOWN = "unknown"


def classify_flow_regime(knudsen: float | None) -> str:
    """Kn < 0.1 continuum-like; 0.1 <= Kn <= 10 transitional; Kn > 10 free molecular."""
    if knudsen is None:
        return FlowRegime.UNKNOWN
    kn = _positive("knudsen number", knudsen)
    if kn < 0.1:
        return FlowRegime.CONTINUUM_LIKE
    if kn <= 10.0:
        return FlowRegime.TRANSITIONAL
    return FlowRegime.FREE_MOLECULAR
