"""Versioned physical constants, each carrying its source.

Scale heights, thermal speeds and impact energies all scale directly with these
values, so a constant that changes silently changes every downstream number.
Each is an immutable record naming its source and revision; there is no bare
float anywhere in this module's public surface.

Canonical ticket: P00_02_T004.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PhysicalConstant",
    "CONSTANTS",
    "K_B",
    "AMU",
    "E_CHARGE",
    "MU_EARTH",
    "R_EARTH_MEAN",
    "R_EARTH_EQUATORIAL",
    "OMEGA_EARTH",
    "J2_EARTH",
    "G0",
]

_NIST = "https://physics.nist.gov/cuu/Constants/"
_WGS84 = "https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84"


@dataclass(frozen=True, slots=True)
class PhysicalConstant:
    """One constant with its provenance. Immutable by construction."""

    symbol: str
    value: float
    unit: str
    source: str
    source_url: str
    revision: str
    exact: bool
    note: str = ""


#: Boltzmann constant — exact by the 2019 SI redefinition.
K_B = PhysicalConstant(
    symbol="k_B", value=1.380649e-23, unit="joule / kelvin",
    source="SI defining constant (2019 redefinition)", source_url=_NIST,
    revision="SI-2019", exact=True,
    note="Exact by definition; no uncertainty.",
)

#: Atomic mass constant — CODATA 2018 measured value.
AMU = PhysicalConstant(
    symbol="amu", value=1.66053906660e-27, unit="kilogram",
    source="CODATA 2018 atomic mass constant", source_url=_NIST,
    revision="CODATA-2018", exact=False,
    note="Relative standard uncertainty 3.0e-10.",
)

#: Elementary charge — exact by the 2019 SI redefinition.
E_CHARGE = PhysicalConstant(
    symbol="e", value=1.602176634e-19, unit="coulomb",
    source="SI defining constant (2019 redefinition)", source_url=_NIST,
    revision="SI-2019", exact=True,
    note="Used to express particle impact energy in electronvolts.",
)

#: Earth gravitational parameter GM.
MU_EARTH = PhysicalConstant(
    symbol="mu_E", value=3.986004418e14, unit="meter ** 3 / second ** 2",
    source="EGM96 / WGS-84 adopted value", source_url=_WGS84,
    revision="WGS-84", exact=False,
)

#: IUGG mean Earth radius. Deliberately NOT the equatorial radius.
R_EARTH_MEAN = PhysicalConstant(
    symbol="R_E", value=6.371e6, unit="meter",
    source="IUGG mean Earth radius", source_url=_WGS84,
    revision="IUGG", exact=False,
    note="Mean radius, used for scalar-altitude diagnostics. Propagation uses "
         "the geodetic model, not this scalar.",
)

R_EARTH_EQUATORIAL = PhysicalConstant(
    symbol="R_E_eq", value=6.378137e6, unit="meter",
    source="WGS-84 semi-major axis", source_url=_WGS84,
    revision="WGS-84", exact=True,
    note="Exact by WGS-84 definition.",
)

OMEGA_EARTH = PhysicalConstant(
    symbol="omega_E", value=7.2921150e-5, unit="radian / second",
    source="WGS-84 Earth rotation rate", source_url=_WGS84,
    revision="WGS-84", exact=False,
    note="Drives atmospheric corotation velocity.",
)

J2_EARTH = PhysicalConstant(
    symbol="J2", value=1.08262668e-3, unit="dimensionless",
    source="EGM96 second zonal harmonic", source_url=_WGS84,
    revision="EGM96", exact=False,
)

G0 = PhysicalConstant(
    symbol="g0", value=9.80665, unit="meter / second ** 2",
    source="Standard acceleration of gravity (defined)", source_url=_NIST,
    revision="CGPM-1901", exact=True,
    note="Definitional constant used in rocket-equation mass flow, NOT local "
         "gravity. Local gravity is computed from mu_E and radius.",
)

CONSTANTS: dict[str, PhysicalConstant] = {
    c.symbol: c
    for c in (
        K_B, AMU, E_CHARGE, MU_EARTH, R_EARTH_MEAN,
        R_EARTH_EQUATORIAL, OMEGA_EARTH, J2_EARTH, G0,
    )
}
