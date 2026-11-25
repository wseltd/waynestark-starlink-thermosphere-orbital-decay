"""Species-fitted constants for the modified Walker-Mehta-Koller CLL model.

WHERE THESE NUMBERS COME FROM, precisely. Walker, Mehta & Koller (2014,
J. Spacecraft & Rockets 51(5):1544, doi:10.2514/1.A32677) fitted closed-form
flat-plate expressions to CLL DSMC behaviour and published species-dependent
constants (beta_j, gamma_j, delta_j, zeta_j). The primary paper is paywalled;
this table was taken from and verified value-by-value against TWO independent
published reproductions read in full for this project:

  * Sinpetru et al., "ADBSat: Methodology of a novel panel method tool for
    aerodynamic analysis of satellites", arXiv:2104.05543v2 (eqs. 11-14 and
    the species constants reference [35]);
  * Macario-Rojas et al., "Atmospheric Interaction with Nanosatellites from
    Observed Orbital Decay", University of Manchester open repository
    (Table 1, explicitly reproduced from Walker et al. [39]).

The verification record is planning/species_gsi_worklog.md section 2.

WHAT THE CONSTANTS ARE NOT. They are gas-SPECIES constants of the fitted
closed form, not material properties: the surface enters through alpha_N,
sigma_T and T_w, which this repository treats as SWEPT parameters with stated
envelopes, never as known Starlink values.

WHICH SPECIES EXIST. Constants were fitted for He, O, N2, O2, H and N only.
Ar, anomalous oxygen and NO have no published constants (ADBSat's own CLL
implementation zeroes Ar and anomalous O; NO postdates the fit), so this
module REFUSES them rather than inventing numbers, and the faceted model
reports the excluded mass fraction so a CLL result always states how much of
the atmosphere it could not represent.

He and H change constants with the normal energy accommodation interval. The
published intervals are open ("0.95 < alpha_n < 1.00"); at an interior
boundary this module assigns the boundary to the HIGHER interval (0.95 maps
to the 0.95-1.00 row), a convention it states rather than hides. alpha_N = 1
exactly never reaches this table: the Walker model switches to the
Schaaf-Chambre form there (see kernels.cll).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CLL_SUPPORTED_SPECIES",
    "CllConstants",
    "CllUnsupportedSpeciesError",
    "cll_constants",
    "is_cll_supported",
]

#: The complete species set the Walker fit covers. Anything else is refused.
CLL_SUPPORTED_SPECIES: tuple[str, ...] = ("He", "O", "N2", "O2", "H", "N")


class CllUnsupportedSpeciesError(ValueError):
    """Raised for a species the published CLL fit has no constants for."""


@dataclass(frozen=True, slots=True)
class CllConstants:
    """One row of the Walker fitted-constant table, with its provenance."""

    species: str
    beta: float
    gamma: float
    delta: float
    zeta: float
    alpha_n_interval: tuple[float, float]
    source: str = (
        "Walker, Mehta & Koller (2014) doi:10.2514/1.A32677, verified against "
        "the Manchester reproduction (Macario-Rojas et al., Table 1) and the "
        "ADBSat methodology paper (arXiv:2104.05543v2); see "
        "planning/species_gsi_worklog.md"
    )


#: Species whose constants are independent of alpha_N.
_FIXED: dict[str, tuple[float, float, float, float]] = {
    "O2": (6.300, 0.260, 0.420, 20.500),
    "N2": (6.600, 0.220, 0.480, 35.000),
    "O": (5.850, 0.200, 0.480, 31.000),
    "N": (4.900, 0.320, 0.420, 8.000),
}

#: (interval_low, interval_high, beta, gamma, delta, zeta), highest first.
#: Boundaries belong to the higher interval, stated in the module docstring.
_INTERVAL_ROWS: dict[str, tuple[tuple[float, float, float, float, float, float], ...]] = {
    "He": (
        (0.95, 1.00, 6.200, 0.380, 3.300, 0.740),
        (0.90, 0.95, 3.800, 0.520, 3.400, 1.120),
        (0.50, 0.90, 3.450, 0.520, 2.400, 0.930),
        (0.00, 0.50, 0.080, 0.520, 4.200, 1.100),
    ),
    "H": (
        (0.95, 1.00, 3.900, 0.195, 1.400, 0.300),
        (0.90, 0.95, 3.500, 0.420, 2.000, 0.720),
        (0.50, 0.90, 3.450, 0.520, 2.400, 0.930),
        (0.00, 0.50, 0.095, 0.465, 2.900, 0.920),
    ),
}


def is_cll_supported(species: str) -> bool:
    """Whether the published fit has constants for this species."""
    return species in CLL_SUPPORTED_SPECIES


def cll_constants(species: str, alpha_n: float) -> CllConstants:
    """The fitted constants for one species at one normal accommodation.

    ``alpha_n`` must lie in [0, 1). The value 1 exactly is refused here
    because the Walker model does not use this table there — it switches to
    the Schaaf-Chambre pressure form, and a caller asking this table for
    alpha_n = 1 has taken a wrong branch.
    """
    if not 0.0 <= alpha_n < 1.0:
        raise ValueError(
            f"alpha_n must lie in [0, 1) for the fitted-constant table, got "
            f"{alpha_n}; at alpha_n = 1 the Walker model uses the "
            f"Schaaf-Chambre form instead (kernels.cll handles the switch)"
        )
    fixed = _FIXED.get(species)
    if fixed is not None:
        beta, gamma, delta, zeta = fixed
        return CllConstants(
            species=species,
            beta=beta,
            gamma=gamma,
            delta=delta,
            zeta=zeta,
            alpha_n_interval=(0.0, 1.0),
        )
    rows = _INTERVAL_ROWS.get(species)
    if rows is None:
        raise CllUnsupportedSpeciesError(
            f"the published CLL fit has no constants for species {species!r}; "
            f"supported species are {CLL_SUPPORTED_SPECIES}. Constants are "
            f"not invented for unsupported species — use the Sentman/DRIA "
            f"baseline, which covers every species, and report the excluded "
            f"mass fraction."
        )
    for low, high, beta, gamma, delta, zeta in rows:
        if low <= alpha_n < high:
            return CllConstants(
                species=species,
                beta=beta,
                gamma=gamma,
                delta=delta,
                zeta=zeta,
                alpha_n_interval=(low, high),
            )
    raise AssertionError(
        f"interval rows for {species!r} do not tile [0, 1); alpha_n={alpha_n}"
    )
