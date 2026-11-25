"""Gas-surface energy accommodation, per species, with its provenance.

The accommodation coefficient is the single largest free parameter in a
free-molecular force model. It is also the one most often quietly invented.
This module exists so that every accommodation value used anywhere in the
package arrives through an object that can say where it came from.

WHAT THIS REPOSITORY CAN SOURCE, and what it cannot.

  SOURCED: a single ENERGY accommodation coefficient, nominal 0.9 over the
  sensitivity range 0.7-1.0. Moe & Moe (2005) find near-complete energy
  accommodation on surfaces carrying an adsorbed atomic-oxygen layer, which is
  the condition below roughly 500 km, and the project's canonical toolchain
  research recommends 0.9 as the nominal with the range resolved by the
  sensitivity envelope.

  NOT SOURCED: per-species values. The same research states plainly that
  "Species-specific gas-surface accommodation coefficients: no single open
  dataset; assemble from Moe & Moe (2005) and Mehta et al. (2014); introduces
  +/-10-30% Cd uncertainty" (open-source-toolchain-1.md, item B3). Moe & Moe's
  finding about atomic oxygen is qualitative — "near-complete" — and turning
  that adjective into a number would be inventing a coefficient.

  :class:`LiteratureAccommodation` therefore ships with NO built-in per-species
  overrides. It returns the sourced nominal for every species and says so. The
  override mechanism is real and typed, and it REQUIRES a source string, so a
  user who does hold a defensible per-species value can supply it and the
  value carries its citation with it. That is the honest shape: the fallback is
  documented, the extension point exists, and no number appears without a
  provenance.

RE-EMISSION TEMPERATURE. Under partial accommodation the re-emitted flux does
not leave at the wall temperature; it leaves at a blend of the wall and the
incident stream. Both temperatures are inputs here — the wall temperature on
the provider, the free-stream temperature and speed ratio at the call — and
the blend is computed by
:func:`~thermosphere_consequence.aerodynamics.kernels.reemission_temperature_k`
so there is one implementation of that relation in the package rather than two.

Primary sources:
  Moe, K. & Moe, M.M. (2005) Planet. Space Sci. 53:793,
    https://www.sciencedirect.com/science/article/abs/pii/S0032063305000486
  Mehta, P.M. et al. (2014) Adv. Space Res. 54:1590.
  Cook, G.E. (1965) Planet. Space Sci. 13:929, doi:10.1016/0032-0633(65)90150-9.

Canonical tickets: P11_12_T005, P11_12_T006.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from ..domain.species import get_species
from .kernels import reemission_temperature_k
from .sentman import (
    ACCOMMODATION_SENSITIVITY_RANGE,
    NOMINAL_ACCOMMODATION,
    NOMINAL_WALL_TEMPERATURE_K,
)

__all__ = [
    "ACCOMMODATION_SENSITIVITY_RANGE",
    "AccommodationValue",
    "LiteratureAccommodation",
    "NOMINAL_ACCOMMODATION",
    "NO_SOURCED_SPECIES_OVERRIDES",
    "SpeciesAccommodationProvider",
]

#: The nominal energy accommodation, as Moe & Moe (2005) support it and the
#: project's toolchain research recommends it. Re-exported from ``sentman`` so
#: the scalar and faceted models cannot end up disagreeing about the nominal.
_NOMINAL_SOURCE = (
    "Moe & Moe (2005) Planet. Space Sci. 53:793 — near-complete energy "
    "accommodation on adsorbed atomic oxygen below ~500 km; 0.9 nominal per "
    "the project's canonical toolchain research (item B3)."
)


@dataclass(frozen=True, slots=True)
class AccommodationValue:
    """One accommodation coefficient and the citation that justifies it.

    ``source`` is mandatory and non-empty. A value without a source is exactly
    the thing this module exists to prevent, so it is rejected at construction
    rather than caught in review.
    """

    value: float
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                f"accommodation must lie in [0, 1], got {self.value}"
            )
        if not self.source.strip():
            raise ValueError(
                "an accommodation value must carry a non-empty source; an "
                "unsourced coefficient is not admissible in a force model"
            )


#: Empty, and empty ON PURPOSE. See the module docstring: this repository's
#: canonical research records that no open dataset gives per-species
#: gas-surface accommodation coefficients, so there is nothing defensible to
#: put here. It is a named constant rather than a bare ``{}`` so that the
#: absence is visible, testable and explained where a reader will look for the
#: values.
NO_SOURCED_SPECIES_OVERRIDES: Mapping[str, AccommodationValue] = MappingProxyType({})


@runtime_checkable
class SpeciesAccommodationProvider(Protocol):
    """Anything that can state the surface's thermal response to one species.

    A Protocol rather than a base class so a thermal model, a fitted surface
    table or the literature default all satisfy it without inheriting
    anything — the same seam the atmosphere providers use.
    """

    name: str
    wall_temperature_k: float

    def energy_accommodation(self, species: str) -> float:
        """Energy accommodation coefficient in [0, 1] for one species."""
        ...

    def reemission_temperature_k(
        self, species: str, speed_ratio: float, freestream_temperature_k: float
    ) -> float:
        """Temperature the flux of that species re-emits at [K]."""
        ...


class LiteratureAccommodation:
    """The sourced literature default: one nominal, applied to every species.

    Constructed with the surface's WALL TEMPERATURE, because the re-emission
    temperature under partial accommodation depends on it and hiding it would
    make a scenario parameter look like a constant.

    ``overrides`` accepts per-species :class:`AccommodationValue` records for a
    caller who holds a sourced value. Nothing is supplied by default: see
    :data:`NO_SOURCED_SPECIES_OVERRIDES`.

    An unknown species symbol is REJECTED rather than silently given the
    nominal, so a provider that invents a species is caught here instead of
    quietly biasing a force.
    """

    def __init__(
        self,
        *,
        wall_temperature_k: float = NOMINAL_WALL_TEMPERATURE_K,
        nominal: float = NOMINAL_ACCOMMODATION,
        overrides: Mapping[str, AccommodationValue] | None = None,
    ) -> None:
        if wall_temperature_k <= 0.0:
            raise ValueError(
                f"wall temperature must be positive, got {wall_temperature_k}"
            )
        if not 0.0 <= nominal <= 1.0:
            raise ValueError(f"nominal accommodation must lie in [0, 1], got {nominal}")
        self.name = "Moe & Moe (2005) nominal energy accommodation"
        self.wall_temperature_k = float(wall_temperature_k)
        self.nominal = AccommodationValue(
            value=float(nominal),
            source=_NOMINAL_SOURCE,
            note=(
                "A SCENARIO PARAMETER, never a measurement. The sensitivity "
                f"range {ACCOMMODATION_SENSITIVITY_RANGE} is what bounds it."
            ),
        )
        self._overrides: dict[str, AccommodationValue] = dict(
            overrides if overrides is not None else NO_SOURCED_SPECIES_OVERRIDES
        )

    def accommodation_record(self, species: str) -> AccommodationValue:
        """The full record — value, source and note — used for this species."""
        get_species(species)  # unknown symbol raises rather than taking the nominal
        return self._overrides.get(species, self.nominal)

    def energy_accommodation(self, species: str) -> float:
        """Energy accommodation coefficient for one species."""
        return self.accommodation_record(species).value

    def reemission_temperature_k(
        self, species: str, speed_ratio: float, freestream_temperature_k: float
    ) -> float:
        """Re-emission temperature for one species at this speed ratio [K].

        alpha = 1 returns the wall temperature exactly; alpha = 0 returns half
        the incident kinetic temperature, which is the molecule leaving as
        energetic as it arrived.
        """
        return reemission_temperature_k(
            self.energy_accommodation(species),
            speed_ratio,
            self.wall_temperature_k,
            freestream_temperature_k,
        )

    def sensitivity_range(self) -> tuple[float, float]:
        """The declared range the nominal must be swept over, (low, high)."""
        return ACCOMMODATION_SENSITIVITY_RANGE

    def provenance(self) -> dict[str, str]:
        """Everything a report needs to say where these numbers came from."""
        return {
            "provider": self.name,
            "nominal": f"{self.nominal.value}",
            "nominal_source": self.nominal.source,
            "sensitivity_range": f"{ACCOMMODATION_SENSITIVITY_RANGE}",
            "wall_temperature_k": f"{self.wall_temperature_k}",
            "species_overrides": (
                ", ".join(sorted(self._overrides)) if self._overrides else "none"
            ),
            "species_override_blocker": (
                "No per-species energy accommodation coefficients are supplied "
                "by default: the project's canonical research records that no "
                "open dataset provides them (open-source-toolchain-1.md item "
                "B3). Every species therefore uses the sourced nominal."
            ),
        }
