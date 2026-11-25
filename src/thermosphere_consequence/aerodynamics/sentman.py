"""Free-molecular aerodynamics: Sentman flat plate and Schaaf-Chambre sphere.

Why implemented here rather than imported: no permissively licensed,
maintained Sentman/DRIA/CLL tool exists for commercial use. ADBSat is GPLv3
AND requires MATLAB's Aerospace Toolbox, so it is doubly disqualified. The
equations are closed form and given in the primary literature, so the honest
option is to implement them from that literature and validate against limits
the literature also fixes.

Validity: free-molecular flow, Knudsen number >> 1. At 210 km the mean free
path is of order a kilometre against a metre-scale spacecraft, so Kn ~ 10^3
and the regime assumption holds comfortably. DSMC would add cost and nothing
else.

Two independent geometries are implemented on purpose. The sphere has a
closed-form solution (Schaaf & Chambre) that is completely independent of the
plate algebra, so agreement between them on the physics they share — the
hyperthermal limit, the accommodation dependence, the temperature-ratio term
— is a real check rather than a restatement.

Per-species treatment is the point of the module: the speed ratio
s = V / sqrt(2kT/m) differs per constituent, so light and heavy species sit in
different flow regimes at the same altitude. Atomic oxygen dominates the mass
flux at 210 km and is also the species that adsorbs and drives accommodation
toward unity, which is why the physical C_d there exceeds the conventional 2.2.

Primary sources:
  Sentman, L.H. (1961) "Free molecule flow theory and its application to the
    determination of aerodynamic forces", LMSC-448514.
  Schaaf, S.A. & Chambre, P.L. (1961) "Flow of Rarefied Gases", Princeton.
  Cook, G.E. (1965) "Satellite drag coefficients", Planet. Space Sci. 13:929,
    doi:10.1016/0032-0633(65)90150-9.
  Moe, K. & Moe, M.M. (2005) Planet. Space Sci. 53:793.
  Mehta, P.M. et al. (2014) Adv. Space Res. 54:1590.

Canonical tickets: P11_12_T004, P11_12_T005, P11_12_T006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain.constants import K_B
from ..domain.species import species_mass

# Moe & Moe (2005) find near-complete energy accommodation on surfaces with
# adsorbed atomic oxygen, the situation below ~500 km. 0.9 is the nominal the
# toolchain research recommends. It is a scenario parameter, never a
# measurement; the sensitivity envelope sweeps the range.
NOMINAL_ACCOMMODATION = 0.9
ACCOMMODATION_SENSITIVITY_RANGE = (0.7, 1.0)

# Representative spacecraft surface temperature in LEO. Enters only through
# the re-emission term, whose contribution is O(1/s); at s ~ 8 it moves C_d by
# a few percent, so it is exposed rather than buried.
NOMINAL_WALL_TEMPERATURE_K = 300.0

# Angular cosine below which a facet counts as edge-on. Well above the
# ~6e-17 that cos(pi/2) actually returns, and far below any physically
# meaningful incidence.
_EDGE_ON_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class PlateGeometry:
    """One flat facet: its area and the angle from its outward normal to the flow."""

    area_m2: float
    theta_rad: float

    def __post_init__(self) -> None:
        if self.area_m2 <= 0:
            raise ValueError(f"facet area must be positive, got {self.area_m2}")


def most_probable_speed(temperature_k: float, species: str) -> float:
    """sqrt(2kT/m), the normalising speed in every free-molecular expression."""
    if temperature_k <= 0:
        raise ValueError(f"temperature must be positive, got {temperature_k}")
    return math.sqrt(2.0 * K_B.value * temperature_k / species_mass(species))


def molecular_speed_ratio(
    bulk_speed_m_s: float, temperature_k: float, species: str
) -> float:
    """Speed ratio s = V / sqrt(2kT/m) for one species.

    The single parameter deciding whether flow is hyperthermal (s >> 1,
    molecules effectively at rest relative to the vehicle) or thermal (s ~ 1,
    random motion comparable to flight speed).
    """
    if bulk_speed_m_s <= 0:
        raise ValueError(f"bulk speed must be positive, got {bulk_speed_m_s}")
    return bulk_speed_m_s / most_probable_speed(temperature_k, species)


#: The two published re-emission constants. Koppenwallner (HTG-TN-08-11,
#: 2009) uses 1/2 and is what ADBSat implements and what its published sphere
#: benchmark (C_d = 2.095 at 200 km, alpha = 1) was produced with; Moe et al.
#: (2004, doi:10.2514/1.2090) use 2/3. The two agree EXACTLY at alpha = 1 and
#: differ by (1 - alpha)/6 inside the square of V_re/V_inf below it — a
#: kernel-variant uncertainty this project reports rather than hides. The
#: half factor is the production default for benchmark continuity.
REEMISSION_FACTORS: dict[str, float] = {
    "koppenwallner_half": 0.5,
    "moe_two_thirds": 2.0 / 3.0,
}


def _reemission_speed_ratio(
    accommodation: float,
    speed_ratio: float,
    wall_temperature_k: float,
    freestream_temperature_k: float,
    reemission_factor: str = "koppenwallner_half",
) -> float:
    """V_re / V_inf, the ratio of re-emitted to incident speed.

    Under partial accommodation the surface re-emits at a temperature between
    the incident kinetic temperature and the wall temperature. With alpha = 0
    the molecule leaves as fast as it arrived (specular-like); with alpha = 1
    it leaves in equilibrium with the wall. Both published variants share

        (V_re/V_inf)^2 = c*(1 - alpha) + alpha * T_w/(T_inf * s^2)

    with c the variant constant of :data:`REEMISSION_FACTORS`.
    """
    if not 0.0 <= accommodation <= 1.0:
        raise ValueError(f"accommodation must lie in [0, 1], got {accommodation}")
    constant = REEMISSION_FACTORS.get(reemission_factor)
    if constant is None:
        raise ValueError(
            f"unknown re-emission factor {reemission_factor!r}; expected one "
            f"of {sorted(REEMISSION_FACTORS)}"
        )
    ratio = wall_temperature_k / (freestream_temperature_k * speed_ratio * speed_ratio)
    inner = constant * (1.0 - accommodation) + accommodation * ratio
    # Numerical floor against a vanishing negative from rounding.
    return math.sqrt(max(inner, 0.0))


def sentman_plate_drag_coefficient(
    speed_ratio: float,
    theta_rad: float,
    accommodation: float,
    wall_temperature_k: float,
    freestream_temperature_k: float,
    *,
    reemission_factor: str = "koppenwallner_half",
) -> float:
    """Sentman drag coefficient for a flat plate, referenced to its own area.

    Standard closed form (Mehta et al. 2014, eq. 2; Walker et al. 2014):

        C_D = P/sqrt(pi) + gamma*Q*Z + (gamma/2)*(V_re/V_inf)*(gamma*sqrt(pi)*Z + P)

    with gamma = cos(theta), P = exp(-(gamma*s)^2)/s, Q = 1 + 1/(2 s^2),
    Z = 1 + erf(gamma*s).

    The hyperthermal, fully accommodating, cold-wall limit is exactly 2, which
    is the analytic anchor the tests pin.
    """
    if speed_ratio <= 0:
        raise ValueError(f"speed ratio must be positive, got {speed_ratio}")
    if wall_temperature_k <= 0 or freestream_temperature_k <= 0:
        raise ValueError("temperatures must be positive")

    gamma = math.cos(theta_rad)
    # THE TANGENTIAL TERM AT EXACTLY GRAZING IS PHYSICS, NOT NOISE. An earlier
    # version of this function returned 0.0 for gamma <= 1e-12, on the grounds
    # that P/sqrt(pi) (~0.069 at s = 8.15) was a "spurious tangential residual"
    # appearing across a discontinuity. It is not spurious: it is the drag
    # exerted by the finite normal thermal flux onto a plate whose MEAN flow is
    # parallel, and it equals 1/(sqrt(pi) s) exactly, which is what this
    # expression returns in the limit. The discontinuity was created by the
    # guard, not by the physics; the expression itself is continuous through
    # gamma = 0 and decays by thirty orders of magnitude by gamma = -0.5, so no
    # cutoff is needed to suppress genuinely leeward facets.
    #
    # Facet SELECTION - which faces the gas reaches at all - belongs to the
    # geometry layer and is made in faceted.py, not here.

    s = speed_ratio
    p = math.exp(-((gamma * s) ** 2)) / s
    q = 1.0 + 1.0 / (2.0 * s * s)
    z = 1.0 + math.erf(gamma * s)
    v_re = _reemission_speed_ratio(
        accommodation,
        s,
        wall_temperature_k,
        freestream_temperature_k,
        reemission_factor,
    )

    return (
        p / math.sqrt(math.pi)
        + gamma * q * z
        + (gamma / 2.0) * v_re * (gamma * math.sqrt(math.pi) * z + p)
    )


def schaaf_chambre_sphere_drag_coefficient(
    speed_ratio: float,
    accommodation: float,
    wall_temperature_k: float,
    freestream_temperature_k: float,
) -> float:
    """Free-molecular sphere drag coefficient, referenced to frontal area.

    Independent closed form used to cross-check the plate algebra:

        C_D = (2s^2+1)/(sqrt(pi) s^3) exp(-s^2)
            + (4s^4 + 4s^2 - 1)/(2 s^4) erf(s)
            + (2 sqrt(pi))/(3 s) * (V_re/V_inf)

    Hyperthermal limit is 2 plus the re-emission term, which is the origin of
    the conventional C_d ~ 2.2 for compact bodies.
    """
    if speed_ratio <= 0:
        raise ValueError(f"speed ratio must be positive, got {speed_ratio}")
    s = speed_ratio
    v_re = _reemission_speed_ratio(
        accommodation, s, wall_temperature_k, freestream_temperature_k
    )
    term_exp = (2.0 * s * s + 1.0) / (math.sqrt(math.pi) * s**3) * math.exp(-(s**2))
    term_erf = (4.0 * s**4 + 4.0 * s * s - 1.0) / (2.0 * s**4) * math.erf(s)
    term_reemit = (2.0 * math.sqrt(math.pi)) / (3.0 * s) * v_re
    return term_exp + term_erf + term_reemit


def species_weighted_drag_coefficient(
    number_densities: dict[str, float],
    temperature_k: float,
    bulk_speed_m_s: float,
    *,
    accommodation: float = NOMINAL_ACCOMMODATION,
    wall_temperature_k: float = NOMINAL_WALL_TEMPERATURE_K,
    theta_rad: float = 0.0,
    geometry: str = "plate",
) -> float:
    """Composition-weighted C_d: ONE scalar, informed by the mixture.

    Weighting is by species MASS flux, not number density, because drag is a
    momentum transfer: a helium atom and an oxygen atom arriving at equal
    rates do not deliver equal momentum. So composition is what decides which
    coefficient comes out, and a mixture shifted toward atomic oxygen returns a
    different C_d from one shifted toward N2.

    WHAT THE RETURN VALUE IS, and is not. It is a single number, and the force
    built from it is ``0.5 * C_d * rho_total * A * v_rel^2`` against the TOTAL
    mass density. Species composition therefore INFORMS the effective C_d; it
    does not give each constituent its own force. Calling this a species-
    resolved force closure would be wrong in a way this docstring exists to
    prevent: the per-species coefficients are collapsed here, deliberately, and
    what survives the collapse is their mass-flux-weighted mean. The per-facet,
    per-species force VECTOR lives in
    :mod:`~thermosphere_consequence.aerodynamics.faceted` and is a comparison
    model, not the propagator's.

    A species present in the mapping but absent from the registry raises,
    rather than being skipped, so a provider that invents a species is caught
    here instead of quietly biasing the mean.
    """
    if not number_densities:
        raise ValueError("no species supplied; cannot weight a drag coefficient")
    if geometry not in ("plate", "sphere"):
        raise ValueError(f"unknown geometry {geometry!r}; expected 'plate' or 'sphere'")

    total_mass_flux = 0.0
    weighted = 0.0
    for symbol, n in number_densities.items():
        mass = species_mass(symbol)  # raises on an unknown species, by design
        if n <= 0:
            continue
        s = molecular_speed_ratio(bulk_speed_m_s, temperature_k, symbol)
        if geometry == "plate":
            cd_i = sentman_plate_drag_coefficient(
                s, theta_rad, accommodation, wall_temperature_k, temperature_k
            )
        else:
            cd_i = schaaf_chambre_sphere_drag_coefficient(
                s, accommodation, wall_temperature_k, temperature_k
            )
        mass_flux = n * mass
        weighted += cd_i * mass_flux
        total_mass_flux += mass_flux

    if total_mass_flux <= 0:
        raise ValueError("total mass flux is zero; cannot form a weighted mean")
    return weighted / total_mass_flux


def inverse_ballistic_coefficient(mass_kg: float, area_m2: float, drag_coefficient: float) -> float:
    """B = C_d A / m in m^2/kg — the combination decay actually depends on.

    Reported alongside every decay result because mass, area and C_d are not
    separately identifiable from a decay curve: only their product is. The
    external literature reference C_B = 0.085 m^2/kg (Shambaugh 2024,
    arXiv:2406.08342, p. 6) is for a Capella Space SAR (Whitney) satellite
    rather than Starlink, and serves only as a fixed scalar benchmark.
    """
    if mass_kg <= 0:
        raise ValueError(f"mass must be positive, got {mass_kg}")
    if area_m2 <= 0:
        raise ValueError(f"area must be positive, got {area_m2}")
    if drag_coefficient <= 0:
        raise ValueError(f"drag coefficient must be positive, got {drag_coefficient}")
    return drag_coefficient * area_m2 / mass_kg


def projected_area(facets: tuple[PlateGeometry, ...]) -> float:
    """Sum of facet areas projected onto the flow direction."""
    return sum(f.area_m2 * max(0.0, math.cos(f.theta_rad)) for f in facets)
