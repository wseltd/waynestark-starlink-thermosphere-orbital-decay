"""Structural contracts for atmosphere and wind providers.

A provider answers one question: what is the neutral atmosphere at this place
and time? Nothing in the physics kernels may know *which* provider answered,
so that swapping NRLMSISE-00 for a gridded physics model changes the result
and never the call site.

Two rules the protocol enforces by shape rather than by documentation:

  * A provider returns species number densities it actually models. A species
    it does not model is ABSENT from the mapping, never zero. Zero is a
    physical claim; absence is an epistemic one, and the difference decides
    whether a density-closure check is meaningful.

  * Every returned state carries the driver values that produced it. A density
    without its F10.7 and ap is not reproducible, and this system's entire
    purpose is provenance.

Canonical tickets: P03_05_T022, P03_05_T026.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..domain.provenance import ValueStatus


@dataclass(frozen=True, slots=True)
class DriverState:
    """The space-weather drivers an atmosphere model consumed.

    Kept beside every density so a number can be traced to the indices that
    produced it. ``ap_history`` is the 7-element array NRLMSISE-00 consumes
    (daily Ap, then 3-hourly ap at 0, 3, 6, 9 h before, then two averages);
    it is stored whole because the storm response depends on the history, not
    only on the instantaneous value.
    """

    f107: float
    f107a: float
    ap_daily: float
    ap_history: tuple[float, ...] | None = None
    source: str = ""
    source_url: str = ""
    status: ValueStatus = ValueStatus.OBSERVED

    def __post_init__(self) -> None:
        if self.f107 <= 0 or self.f107a <= 0:
            raise ValueError(
                f"solar flux must be positive, got f107={self.f107}, f107a={self.f107a}"
            )
        if self.ap_daily < 0:
            raise ValueError(f"ap must be non-negative, got {self.ap_daily}")
        if self.ap_history is not None and len(self.ap_history) != 7:
            raise ValueError(
                f"NRLMSISE-00 requires exactly 7 ap history entries, got {len(self.ap_history)}"
            )


@dataclass(frozen=True, slots=True)
class AtmosphereState:
    """Neutral atmosphere at one point in space and time.

    ``number_densities`` is in m^-3 keyed by the species symbols of
    domain.species. A species the model does not report is absent from the
    mapping. ``total_mass_density`` is the model's own total where the model
    provides one, which is NOT generally equal to the sum over the reported
    species — the difference is what density_closure_fraction measures and is
    a diagnostic, not an error to be corrected away.
    """

    time_utc: _dt.datetime
    altitude_m: float
    latitude_deg: float
    longitude_deg: float
    temperature_k: float
    number_densities: dict[str, float]
    total_mass_density: float | None = None
    exospheric_temperature_k: float | None = None
    drivers: DriverState | None = None
    provider_name: str = ""
    provider_version: str = ""
    status: ValueStatus = ValueStatus.MODELLED
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.time_utc.tzinfo is None:
            raise ValueError("AtmosphereState.time_utc must be timezone-aware")
        if self.temperature_k <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature_k}")
        if self.total_mass_density is not None and self.total_mass_density < 0:
            raise ValueError(f"mass density must be non-negative, got {self.total_mass_density}")
        for symbol, n in self.number_densities.items():
            if n < 0:
                raise ValueError(f"{symbol}: number density is negative ({n}); rejected")


@runtime_checkable
class AtmosphereProvider(Protocol):
    """Anything that can state the neutral atmosphere at a point."""

    name: str
    version: str

    def state_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> AtmosphereState:
        """Return the atmospheric state at one point."""
        ...


@runtime_checkable
class TotalDensityProvider(Protocol):
    """A model that supplies total mass density but no species breakdown.

    Separate from AtmosphereProvider on purpose: JB2008 and most gridded
    storm products give total density only, and a caller that needs species
    resolution must fail to type-check against this rather than silently
    receive an empty species mapping.
    """

    name: str
    version: str

    def total_mass_density(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> float:
        """Return total neutral mass density in kg m^-3."""
        ...


@runtime_checkable
class WindProvider(Protocol):
    """Horizontal neutral wind. Zero-wind is an explicit implementation."""

    name: str
    version: str

    def wind_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> tuple[float, float]:
        """Return (zonal_eastward, meridional_northward) wind in m/s."""
        ...


@runtime_checkable
class DriverLookup(Protocol):
    """Anything that can state the observed drivers at an instant.

    A Protocol rather than a base class, so a plain function, a bound method
    or a closure over a loaded index series all satisfy it without inheriting
    anything. This is the seam that lets a fixed-path comparison be driven by
    real ingested indices in production and by a recorded fixture in tests,
    with no branch in the physics.
    """

    def __call__(self, when: _dt.datetime) -> DriverState:
        """Return the observed DriverState at ``when``."""
        ...
