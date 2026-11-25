"""Neutral thermospheric species registry.

Species masses are exact multiples of the atomic mass constant, so they are
derived here rather than transcribed — a transcribed mass is a place for a typo
to hide.

Anomalous oxygen shares the mass of atomic oxygen but is a distinct species:
NRLMSISE-00 reports it separately and it participates in the density-closure
check, so collapsing the two would silently break closure.

Canonical ticket: P00_02_T005.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .constants import AMU

__all__ = ["Species", "UnknownSpeciesError", "ALL_SPECIES", "get_species", "species_mass"]


class UnknownSpeciesError(KeyError):
    """Raised when a species symbol is not in the registry."""


@dataclass(frozen=True, slots=True)
class Species:
    """One neutral species. Immutable."""

    symbol: str
    name: str
    mass_number: float
    mass_kg: float
    note: str = ""


def _make(symbol: str, name: str, mass_number: float, note: str = "") -> Species:
    return Species(
        symbol=symbol, name=name, mass_number=mass_number,
        mass_kg=mass_number * AMU.value, note=note,
    )


_SPECIES: dict[str, Species] = {
    s.symbol: s
    for s in (
        _make("H", "atomic hydrogen", 1.0),
        _make("He", "helium", 4.0),
        _make("N", "atomic nitrogen", 14.0),
        _make("O", "atomic oxygen", 16.0,
              "Dominant by number density near 210 km; near-complete "
              "accommodation on adsorbed surfaces."),
        _make("N2", "molecular nitrogen", 28.0),
        _make("O2", "molecular oxygen", 32.0),
        _make("Ar", "argon", 40.0),
        _make("anomalous_O", "anomalous oxygen", 16.0,
              "Same particle mass as atomic oxygen but reported separately by "
              "NRLMSISE-00; kept distinct so density closure stays honest."),
        _make("NO", "nitric oxide", 30.0,
              "Reported by NRLMSIS 2.1 only. NRLMSISE-00 and NRLMSIS 2.0 do "
              "not model it and must report it ABSENT, never zero."),
    )
}

#: Read-only view; mutation attempts raise.
ALL_SPECIES: Mapping[str, Species] = MappingProxyType(_SPECIES)


def get_species(symbol: str) -> Species:
    """Return the species record, or raise ``UnknownSpeciesError``."""
    try:
        return _SPECIES[symbol]
    except KeyError as exc:
        raise UnknownSpeciesError(
            f"unknown species {symbol!r}; known symbols: {sorted(_SPECIES)}"
        ) from exc


def species_mass(symbol: str) -> float:
    """Particle mass in kilograms (SI)."""
    return get_species(symbol).mass_kg
