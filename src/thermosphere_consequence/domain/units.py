"""Unit handling: Pint at the boundaries, documented SI floats inside kernels.

Mixing kilometres with metres in a scale-height calculation is one of the
specific errors this project is required to prevent. Units are therefore
enforced where values enter and leave the system, while numerical kernels
operate on plain floats already normalised to SI — Pint objects inside an ODE
right-hand side would be both slow and awkward to serialise.

Canonical ticket: P00_02_T003.
"""

from __future__ import annotations

import math
from typing import Any

import pint

__all__ = [
    "DimensionalityError",
    "SI",
    "get_unit_registry",
    "to_si",
    "from_si",
]

_REGISTRY: pint.UnitRegistry | None = None


class DimensionalityError(ValueError):
    """Raised when a quantity does not match the expected dimensionality."""


#: Canonical SI unit for each project quantity. These strings are the contract
#: between the boundary layer and the kernels: a kernel documented as taking
#: ``length`` receives metres, always.
SI: dict[str, str] = {
    "length": "meter",
    "time": "second",
    "mass": "kilogram",
    "temperature": "kelvin",
    "number_density": "1 / meter ** 3",
    "mass_density": "kilogram / meter ** 3",
    "velocity": "meter / second",
    "acceleration": "meter / second ** 2",
    "force": "newton",
    "pressure": "newton / meter ** 2",
    "energy": "joule",
    "specific_energy": "joule / kilogram",
    "area": "meter ** 2",
    "angle": "radian",
    "magnetic_flux_density": "tesla",
    # Both drag parameters, each under its own name with its own unit. The
    # mechanical rename that introduced the corrected names briefly mapped
    # "inverse_ballistic_coefficient" to kg/m^2 -- the CONVENTIONAL B's unit
    # under the INVERSE name, i.e. the very inversion the rename existed to
    # remove. Both are pinned here so the registry itself cannot express it.
    "ballistic_coefficient": "kilogram / meter ** 2",
    "inverse_ballistic_coefficient": "meter ** 2 / kilogram",
    "flux": "1 / (meter ** 2 * second)",
    "fluence": "1 / meter ** 2",
    "dimensionless": "dimensionless",
}


def get_unit_registry() -> pint.UnitRegistry:
    """Return the process-wide registry.

    Pint refuses to operate across registries, so exactly one instance must
    exist; this is the only place it is constructed.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = pint.UnitRegistry()
    return _REGISTRY


def _resolve(expected: str | None) -> str | None:
    if expected is None:
        return None
    return SI.get(expected, expected)


def to_si(value: Any, unit: str | None = None, expected: str | None = None) -> float:
    """Convert a quantity or a value-plus-unit into a plain SI float.

    ``expected`` may be a project quantity name (e.g. ``"length"``) or any
    dimensionally equivalent unit spelling. A mismatch raises rather than
    coercing, because a silent kilometre-to-metre slip is the failure this
    function exists to catch.
    """
    ureg = get_unit_registry()
    target = _resolve(expected)

    if isinstance(value, ureg.Quantity):
        quantity = value
    elif unit is not None:
        try:
            quantity = ureg.Quantity(value, unit)
        except Exception as exc:  # pint raises several distinct types
            raise DimensionalityError(f"unit {unit!r} is not resolvable") from exc
    else:
        raise ValueError("to_si requires either a Quantity or an explicit unit")

    if target is not None:
        try:
            quantity = quantity.to(target)
        except pint.DimensionalityError as exc:
            raise DimensionalityError(
                f"expected {target!r} ({expected!r}) but got "
                f"{quantity.units!r} with dimensionality {quantity.dimensionality}"
            ) from exc

    result = float(quantity.magnitude)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value after conversion: {result!r}")
    return result


def from_si(value: float, unit: str) -> Any:
    """Attach units to an SI float for presentation or reporting."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"from_si expects a real number, got {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite value: {value!r}")
    ureg = get_unit_registry()
    return ureg.Quantity(float(value), _resolve(unit) or unit)
