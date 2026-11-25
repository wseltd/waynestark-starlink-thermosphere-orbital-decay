"""Particle impact: directed flux, cumulative fluence, translational energy.

Canonical tickets: P00_02_T018, T019, T020.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..domain.constants import E_CHARGE
from ..domain.species import species_mass

__all__ = [
    "molecular_impact_flux",
    "molecular_impact_fluxes",
    "cumulative_fluence",
    "particle_impact_energy_j",
    "particle_impact_energy_ev",
    "particle_impact_energies_ev",
]

NumberDensities = Mapping[str, float | None]


def molecular_impact_flux(number_density_m3: float, relative_speed_m_s: float) -> float:
    """Phi_i = n_i * v_rel   [1/(m^2 s)].

    Directed-beam approximation: the number of particles of one species
    striking a square metre of forward-facing surface each second. Valid where
    the molecular speed ratio is large; check it with
    ``molecular_speed_ratio`` before relying on this.
    """
    n = float(number_density_m3)
    v = float(relative_speed_m_s)
    if not math.isfinite(n) or n < 0.0:
        raise ValueError(f"number density must be finite and non-negative, got {n}")
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"relative speed must be finite and non-negative, got {v}")
    return n * v


def molecular_impact_fluxes(n_i: NumberDensities, relative_speed_m_s: float) -> dict[str, float]:
    """Per-species impact flux. Missing species stay missing."""
    return {
        s: molecular_impact_flux(v, relative_speed_m_s)
        for s, v in n_i.items()
        if v is not None
    }


def cumulative_fluence(fluxes: Sequence[float], times_s: Sequence[float]) -> float:
    """Time integral of flux   [1/m^2], by the trapezoidal rule.

    Monotonically non-decreasing for non-negative flux. This quantity may later
    support atomic-oxygen surface-exposure studies, but it must NOT be
    converted into erosion or damage without validated reaction yields,
    material data, incidence angle and surface temperature.
    """
    if len(fluxes) != len(times_s):
        raise ValueError("flux and time series must be the same length")
    if len(fluxes) < 2:
        raise ValueError("fluence integration needs at least two samples")
    for k in range(len(times_s) - 1):
        if times_s[k + 1] < times_s[k]:
            raise ValueError("time series must be non-decreasing")
    return math.fsum(
        0.5
        * (float(fluxes[k]) + float(fluxes[k + 1]))
        * (float(times_s[k + 1]) - float(times_s[k]))
        for k in range(len(fluxes) - 1)
    )


def particle_impact_energy_j(symbol: str, relative_speed_m_s: float) -> float:
    """E_i = 0.5 m_i v_rel^2   [J].

    Translational kinetic energy of one incident particle. A diagnostic only:
    it does not license any claim about surface damage, which additionally
    requires surface material, reaction yield, incidence angle, surface
    temperature, fluence and accommodation data.
    """
    v = float(relative_speed_m_s)
    if not math.isfinite(v):
        raise ValueError(f"relative speed is not finite ({v})")
    return 0.5 * species_mass(symbol) * v * v


def particle_impact_energy_ev(symbol: str, relative_speed_m_s: float) -> float:
    """Particle impact energy in electronvolts [eV]."""
    return particle_impact_energy_j(symbol, relative_speed_m_s) / E_CHARGE.value


def particle_impact_energies_ev(
    symbols: Sequence[str], relative_speed_m_s: float
) -> dict[str, float]:
    return {s: particle_impact_energy_ev(s, relative_speed_m_s) for s in symbols}
