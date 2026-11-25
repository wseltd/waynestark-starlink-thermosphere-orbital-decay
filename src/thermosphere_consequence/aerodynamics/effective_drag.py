"""Effective-C_d free-molecular drag: force, acceleration, inverse ballistic coefficient.

The conventional engineering drag model, used as the primary force model for
the first release. C_d = 2.2 is a CONVENTION traceable to Cook (1965) — who
himself wrote that it "may be too low, perhaps by as much as 10-15 percent" —
not a universal physical constant. It varies with composition, speed ratio,
surface temperature, accommodation, geometry and attitude, so it is always a
scenario parameter carried through the sensitivity envelope, never a fact.

Canonical tickets: P00_02_T021, T022, T023, T024, T025.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "CD_CONVENTIONAL",
    "CD_SENSITIVITY_RANGE",
    "drag_force",
    "drag_acceleration",
    "drag_acceleration_vector",
    "ballistic_coefficient_kg_m2",
    "inverse_ballistic_coefficient_m2_kg",
    "species_drag_decomposition",
    "momentum_factor_from_cd",
    "cd_from_momentum_factor",
]

#: Conventional engineering value. A scenario assumption, not a measurement.
CD_CONVENTIONAL = 2.2

#: Minimum sensitivity range. The lower bound is the convention; the upper
#: bound accommodates the external Capella Space Whitney reference inverse
#: ballistic coefficient (Shambaugh 2024, arXiv:2406.08342; not Starlink).
#: Not a probability distribution.
CD_SENSITIVITY_RANGE = (1.8, 3.0)


def _non_negative(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} is not finite ({v})")
    if v < 0.0:
        raise ValueError(f"{name} must be non-negative, got {v}")
    return v


def _positive(name: str, value: float) -> float:
    v = _non_negative(name, value)
    if v == 0.0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


def drag_force(
    drag_coefficient: float,
    mass_density_kg_m3: float,
    projected_area_m2: float,
    relative_speed_m_s: float,
) -> float:
    """F = 0.5 C_d rho A v_rel^2   [N].

    Linear in density, area and C_d; quadratic in relative speed. Zero density
    or zero area gives exactly zero force.
    """
    cd = _non_negative("drag coefficient", drag_coefficient)
    rho = _non_negative("mass density", mass_density_kg_m3)
    area = _non_negative("projected area", projected_area_m2)
    v = _non_negative("relative speed", relative_speed_m_s)
    return 0.5 * cd * rho * area * v * v


def drag_acceleration(
    drag_coefficient: float,
    mass_density_kg_m3: float,
    projected_area_m2: float,
    relative_speed_m_s: float,
    spacecraft_mass_kg: float,
) -> float:
    """a_drag = F / m   [m/s^2]. Inversely proportional to spacecraft mass."""
    m = _positive("spacecraft mass", spacecraft_mass_kg)
    return drag_force(drag_coefficient, mass_density_kg_m3, projected_area_m2,
                      relative_speed_m_s) / m


def drag_acceleration_vector(
    drag_coefficient: float,
    mass_density_kg_m3: float,
    projected_area_m2: float,
    relative_velocity_m_s: Sequence[float],
    spacecraft_mass_kg: float,
) -> tuple[float, float, float]:
    """Vector drag acceleration [m/s^2], opposing the atmosphere-relative velocity.

    a_vec = -0.5 (C_d A / m) rho |v_rel| v_rel

    Sign convention: the result is antiparallel to ``relative_velocity_m_s``,
    so its dot product with that velocity is non-positive by construction.
    """
    if len(relative_velocity_m_s) != 3:
        raise ValueError("relative velocity must be a 3-vector")
    vx, vy, vz = (float(c) for c in relative_velocity_m_s)
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if speed == 0.0:
        return (0.0, 0.0, 0.0)
    m = _positive("spacecraft mass", spacecraft_mass_kg)
    cd = _non_negative("drag coefficient", drag_coefficient)
    rho = _non_negative("mass density", mass_density_kg_m3)
    area = _non_negative("projected area", projected_area_m2)
    k = -0.5 * cd * area * rho * speed / m
    return (k * vx, k * vy, k * vz)


def ballistic_coefficient_kg_m2(
    spacecraft_mass_kg: float, drag_coefficient: float, projected_area_m2: float
) -> float:
    """B = m / (C_d A)   [kg/m^2] — the CONVENTIONAL ballistic coefficient.

    Larger means heavier per unit drag area, hence slower decay.

    From orbital decay alone the identifiable quantity is the COMBINATION
    C_d * A / m. Do not claim to have independently inferred C_d, A and m from
    one decay record without independent constraints.
    """
    return _positive("spacecraft mass", spacecraft_mass_kg) / (
        _positive("drag coefficient", drag_coefficient)
        * _positive("projected area", projected_area_m2)
    )


def inverse_ballistic_coefficient_m2_kg(
    spacecraft_mass_kg: float, drag_coefficient: float, projected_area_m2: float
) -> float:
    """C_d A / m   [m^2/kg] — the quantity orbital decay actually constrains.

    This is NOT the ballistic coefficient; it is its reciprocal. A value in
    m^2/kg may never be reported under the name "ballistic coefficient".
    """
    return 1.0 / ballistic_coefficient_kg_m2(
        spacecraft_mass_kg, drag_coefficient, projected_area_m2
    )


def species_drag_decomposition(
    species_mass_densities: Mapping[str, float],
    drag_coefficient: float,
    projected_area_m2: float,
    relative_speed_m_s: float,
) -> dict[str, float]:
    """Per-species drag force under one COMMON drag coefficient [N].

    F_i = 0.5 C_d rho_i A v_rel^2, so sum(F_i) = 0.5 C_d sum(rho_i) A v_rel^2.

    This shows each species' share of the mass flux. It does NOT show that
    composition changes drag at fixed total density: with total density, C_d,
    area and speed all fixed, redistributing mass between species leaves the
    total drag exactly unchanged. Composition matters because it changes total
    mass density, and because species differ in gas-surface interaction — the
    latter is the advanced model's business, not this one's.
    """
    return {
        s: drag_force(drag_coefficient, rho, projected_area_m2, relative_speed_m_s)
        for s, rho in species_mass_densities.items()
    }


def momentum_factor_from_cd(drag_coefficient: float) -> float:
    """momentum_factor = C_d / 2   [dimensionless].

    For the simplified per-particle momentum convention
    ``impact_pressure_i = momentum_factor * rho_i * v_rel^2``, consistency with
    the engineering drag pressure ``0.5 C_d rho v_rel^2`` forces this identity.
    C_d = 2.2 corresponds to 1.1; ideal normal reversal (2.0) corresponds to
    C_d = 4.0. The two are never independent estimates.
    """
    return _non_negative("drag coefficient", drag_coefficient) / 2.0


def cd_from_momentum_factor(momentum_factor: float) -> float:
    """C_d = 2 * momentum_factor   [dimensionless]. Exact inverse of the above."""
    return 2.0 * _non_negative("momentum factor", momentum_factor)
