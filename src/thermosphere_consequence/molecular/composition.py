"""Species composition: mass density, closure, mean particle mass, fractions.

All inputs are SI floats already normalised at the boundary. A species that the
atmosphere provider did not supply is **missing**, never zero: a missing helium
density and a genuine helium density of zero are different physical statements,
and collapsing them would silently corrupt the mean particle mass and the
density-closure diagnostic.

Canonical tickets: P00_02_T006, T007, T008, T009.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..domain.species import species_mass

__all__ = [
    "species_mass_density",
    "species_mass_densities",
    "total_species_mass_density",
    "total_number_density",
    "mean_particle_mass",
    "mean_particle_mass_amu",
    "mass_fractions",
    "density_closure_fraction",
    "storm_contribution",
]

NumberDensities = Mapping[str, float | None]


def _validate(symbol: str, n: float) -> float:
    if n is None:
        raise ValueError(f"{symbol}: number density is missing, not a value")
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        raise TypeError(f"{symbol}: number density must be a real number")
    n = float(n)
    if not math.isfinite(n):
        raise ValueError(f"{symbol}: number density is not finite ({n})")
    if n < 0.0:
        raise ValueError(f"{symbol}: number density is negative ({n}); rejected")
    return n


def _present(n_i: NumberDensities) -> dict[str, float]:
    """Species actually supplied. Absent keys and None values stay missing."""
    return {s: _validate(s, v) for s, v in n_i.items() if v is not None}


def species_mass_density(symbol: str, number_density: float) -> float:
    """rho_i = n_i * m_i   [kg/m^3], from n_i [1/m^3] and m_i [kg]."""
    return _validate(symbol, number_density) * species_mass(symbol)


def species_mass_densities(n_i: NumberDensities) -> dict[str, float]:
    """Per-species mass density for every supplied species."""
    return {s: n * species_mass(s) for s, n in _present(n_i).items()}


def total_species_mass_density(n_i: NumberDensities) -> float:
    """rho_species_sum = sum_i n_i m_i   [kg/m^3]. Supplied species only."""
    return math.fsum(species_mass_densities(n_i).values())


def total_number_density(n_i: NumberDensities) -> float:
    """n_total = sum_i n_i   [1/m^3]. Supplied species only."""
    return math.fsum(_present(n_i).values())


def mean_particle_mass(n_i: NumberDensities) -> float:
    """m_mean = rho_species_sum / n_total   [kg].

    Number-weighted, so it is the mass of the average particle in the supplied
    mixture — not an average of the species masses.
    """
    n_total = total_number_density(n_i)
    if n_total <= 0.0:
        raise ValueError("mean particle mass is undefined for zero total number density")
    return total_species_mass_density(n_i) / n_total


def mean_particle_mass_amu(n_i: NumberDensities) -> float:
    """Mean particle mass expressed in atomic mass units [dimensionless]."""
    from ..domain.constants import AMU

    return mean_particle_mass(n_i) / AMU.value


def mass_fractions(n_i: NumberDensities, of_total: float | None = None) -> dict[str, float]:
    """Species mass fractions [dimensionless].

    By default fractions are of the species sum and therefore sum to 1. Pass
    ``of_total`` to express them against a separately supplied total mass
    density, in which case they need not sum to 1 — and that discrepancy is
    exactly the closure information the caller should be looking at.
    """
    rho = species_mass_densities(n_i)
    denominator = total_species_mass_density(n_i) if of_total is None else float(of_total)
    if denominator <= 0.0:
        raise ValueError("mass fractions are undefined for non-positive total density")
    return {s: r / denominator for s, r in rho.items()}


def density_closure_fraction(n_i: NumberDensities, rho_total: float) -> float:
    """(sum_i rho_i - rho_total) / rho_total   [dimensionless].

    Reported, never silently corrected. A non-zero residual can mean the
    provider withheld species, that anomalous oxygen is accounted separately,
    that the fields use different interpolation, or that the total includes
    constituents not enumerated. The sign carries meaning and is preserved.
    """
    rho_total = float(rho_total)
    if not math.isfinite(rho_total):
        raise ValueError(f"total mass density is not finite ({rho_total})")
    if rho_total <= 0.0:
        raise ValueError(f"total mass density must be positive, got {rho_total}")
    return (total_species_mass_density(n_i) - rho_total) / rho_total


def storm_contribution(
    n_i_quiet: NumberDensities,
    n_i_storm: NumberDensities,
    rho_total_quiet: float | None = None,
    rho_total_storm: float | None = None,
) -> dict[str, float]:
    """Per-species share of the storm-time density change [dimensionless].

    contribution_i = (rho_i_storm - rho_i_quiet) / (rho_total_storm - rho_total_quiet)

    Negative contributions are physically real — a species can be depleted
    while total density rises — and are never clamped to zero.
    """
    rho_q = species_mass_densities(n_i_quiet)
    rho_s = species_mass_densities(n_i_storm)
    total_q = float(rho_total_quiet) if rho_total_quiet is not None else math.fsum(rho_q.values())
    total_s = float(rho_total_storm) if rho_total_storm is not None else math.fsum(rho_s.values())
    delta_total = total_s - total_q
    if delta_total == 0.0:
        raise ValueError("storm contribution is undefined for zero total density change")
    shared = sorted(set(rho_q) & set(rho_s))
    if not shared:
        raise ValueError("quiet and storm samples share no species")
    return {s: (rho_s[s] - rho_q[s]) / delta_total for s in shared}
