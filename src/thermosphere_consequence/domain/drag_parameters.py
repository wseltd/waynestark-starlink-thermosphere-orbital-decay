"""Drag-parameter definitions, with the units error this module exists to prevent.

TERMINOLOGY, stated once and enforced everywhere.

The CONVENTIONAL ballistic coefficient is

    B = m / (C_d * A)          units kg m^-2

A LARGER B means a *more* massive body per unit drag area, so it decays MORE
SLOWLY. This is the quantity that appears in the ballistics and re-entry
literature and it is what a reader assumes when they see "ballistic
coefficient".

Its reciprocal is

    B_inv = C_d * A / m        units m^2 kg^-1

which is the quantity that appears directly in the drag acceleration

    a_drag = -(1/2) * B_inv * rho * |v_rel| * v_rel

A LARGER B_inv means MORE drag per unit mass, so it decays MORE QUICKLY.

These are reciprocals and they are not interchangeable. An earlier version of
this project reported C_d*A/m and called it the ballistic coefficient, which is
wrong in both name and units. That error is the reason this module exists and
the reason every drag parameter in the codebase now carries its unit in its
name. A quantity in m^2/kg may never be called a ballistic coefficient.

The external literature reference value this project compares against —
Shambaugh's (2024) fitted 0.085 m^2/kg for a Capella Space SAR (Whitney)
satellite (arXiv:2406.08342, p. 6; NOT SpaceX and NOT Starlink) — is likewise
an INVERSE ballistic coefficient, because its units are m^2/kg. The reciprocal,
11.76 kg/m^2, is the conventional ballistic coefficient of that reference bus.

IDENTIFIABILITY. Neither form separates mass, area and drag coefficient. A
decay curve constrains only the product C_d*A/m, so a change in the fitted
value may come from projected area, drag coefficient, gas-surface
accommodation, mass, attitude, atmosphere-model bias, initial-state bias, or
any combination. Attributing a fitted change to area alone requires
independently constraining the other two, which this project does not do.

Canonical tickets: P00_02_T023, P00_02_T025, P11_12_T020.
"""

from __future__ import annotations

from dataclasses import dataclass

# The units string used wherever a drag parameter is serialised, so a consumer
# reading JSON cannot mistake one form for the other.
INVERSE_BALLISTIC_COEFFICIENT_UNITS = "m^2/kg"
BALLISTIC_COEFFICIENT_UNITS = "kg/m^2"


class DragParameterUnitError(ValueError):
    """Raised when a drag parameter is used in the wrong sense or unit."""


def inverse_ballistic_coefficient_m2_kg(
    drag_coefficient: float, area_m2: float, mass_kg: float
) -> float:
    """C_d * A / m in m^2 kg^-1 — the drag-area-to-mass parameter.

    This is what multiplies density in the drag acceleration. It is NOT the
    ballistic coefficient; it is its reciprocal.
    """
    if drag_coefficient <= 0:
        raise ValueError(f"drag coefficient must be positive, got {drag_coefficient}")
    if area_m2 <= 0:
        raise ValueError(f"area must be positive, got {area_m2}")
    if mass_kg <= 0:
        raise ValueError(f"mass must be positive, got {mass_kg}")
    return drag_coefficient * area_m2 / mass_kg


def ballistic_coefficient_kg_m2(
    drag_coefficient: float, area_m2: float, mass_kg: float
) -> float:
    """m / (C_d * A) in kg m^-2 — the CONVENTIONAL ballistic coefficient.

    Larger means heavier per unit drag area, hence slower decay.
    """
    return 1.0 / inverse_ballistic_coefficient_m2_kg(drag_coefficient, area_m2, mass_kg)


def to_ballistic_coefficient(inverse_m2_kg: float) -> float:
    """Convert C_d*A/m (m^2/kg) to the conventional B (kg/m^2)."""
    if inverse_m2_kg <= 0:
        raise DragParameterUnitError(
            f"inverse ballistic coefficient must be positive, got {inverse_m2_kg} "
            f"{INVERSE_BALLISTIC_COEFFICIENT_UNITS}"
        )
    return 1.0 / inverse_m2_kg


def to_inverse_ballistic_coefficient(ballistic_kg_m2: float) -> float:
    """Convert the conventional B (kg/m^2) to C_d*A/m (m^2/kg)."""
    if ballistic_kg_m2 <= 0:
        raise DragParameterUnitError(
            f"ballistic coefficient must be positive, got {ballistic_kg_m2} "
            f"{BALLISTIC_COEFFICIENT_UNITS}"
        )
    return 1.0 / ballistic_kg_m2


@dataclass(frozen=True, slots=True)
class DragParameter:
    """A drag parameter carrying BOTH forms and its identifiability caveat.

    Constructed from the inverse form because that is what a decay fit
    produces, and it exposes the conventional form so a reader who wants a
    ballistic coefficient gets one in the right units rather than the
    reciprocal under the wrong name.
    """

    inverse_ballistic_coefficient_m2_kg: float
    source: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.inverse_ballistic_coefficient_m2_kg <= 0:
            raise DragParameterUnitError(
                "inverse ballistic coefficient must be positive, got "
                f"{self.inverse_ballistic_coefficient_m2_kg} "
                f"{INVERSE_BALLISTIC_COEFFICIENT_UNITS}"
            )

    @property
    def ballistic_coefficient_kg_m2(self) -> float:
        """The conventional B = m/(C_d A)."""
        return 1.0 / self.inverse_ballistic_coefficient_m2_kg

    def implied_area_m2(self, drag_coefficient: float, mass_kg: float) -> float:
        """Area implied at a GIVEN C_d and mass.

        Both must be supplied because the fit does not constrain them. The
        returned area is conditional on those two choices and is not a
        measurement of the spacecraft.
        """
        if drag_coefficient <= 0:
            raise ValueError(f"drag coefficient must be positive, got {drag_coefficient}")
        if mass_kg <= 0:
            raise ValueError(f"mass must be positive, got {mass_kg}")
        return self.inverse_ballistic_coefficient_m2_kg * mass_kg / drag_coefficient

    def as_dict(self) -> dict[str, object]:
        """Serialise BOTH forms with explicit units."""
        return {
            "inverse_ballistic_coefficient": self.inverse_ballistic_coefficient_m2_kg,
            "inverse_ballistic_coefficient_units": INVERSE_BALLISTIC_COEFFICIENT_UNITS,
            "ballistic_coefficient": self.ballistic_coefficient_kg_m2,
            "ballistic_coefficient_units": BALLISTIC_COEFFICIENT_UNITS,
            "source": self.source,
            "note": self.note,
            "identifiability_caveat": (
                "A decay curve constrains only the product C_d*A/m. A change in "
                "this value may arise from projected area, drag coefficient, "
                "gas-surface accommodation, mass, attitude, atmosphere-model bias "
                "or initial-state bias, or any combination. It is not a "
                "measurement of any one of them."
            ),
        }


# The external literature reference figure (Shambaugh 2024, arXiv:2406.08342,
# p. 6, for a Capella Space SAR Whitney-generation satellite — NOT SpaceX and
# NOT Starlink), recorded in BOTH forms so it can never be compared against
# the wrong one. Legacy constant names are retained solely for backward API
# compatibility.
OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG = 0.085
OPERATOR_FITTED_BALLISTIC_COEFFICIENT_KG_M2 = 1.0 / 0.085
OPERATOR_FIGURE_SOURCE = (
    "Shambaugh (2024), arXiv:2406.08342, p. 6 (Capella Space SAR Whitney satellite; not SpaceX/Starlink)"
)
OPERATOR_FIGURE_CAVEAT = (
    "External literature reference describing a Capella Space SAR (Whitney) "
    "satellite in Shambaugh (2024), NOT SpaceX hardware and NOT the Starlink "
    "Group 4-7 satellites."
)
