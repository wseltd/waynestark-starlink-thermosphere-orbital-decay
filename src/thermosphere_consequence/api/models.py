"""Wire shapes for the HTTP adapter.

These models are the boundary contract, not a second home for physics. They
carry three responsibilities and no others:

  * reject malformed input at the edge, with a message that names the field, so
    a caller learns what is wrong from the response rather than from a stack
    trace buried in a server log;
  * state the shape of every payload the API can emit, so the OpenAPI schema is
    generated from the same declarations the runtime validates against;
  * force every payload through :class:`ProvenanceEnvelope`, which is the only
    reason this module is worth having.

Every model is frozen. A response object that can be mutated after it has been
assembled is a response whose provenance can be edited away from the number it
describes, which defeats the point of attaching provenance at all. ``extra`` is
forbidden for the same reason in the other direction: a misspelled request field
must be an error, never a silently ignored one that leaves a default in place of
the value the caller believed they sent.

Validation that the domain layer already performs is NOT restated here. Unknown
geometries, accommodation outside [0, 1] and unknown species symbols are all
rejected by the library functions themselves, and their messages reach the
caller through the adapter's ValueError handler. Only boundary conditions that
the caller needs located to a *field* are checked here.

Canonical ticket: P00_02_T048 (thin-adapter discipline, shared with the CLI).
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

from ..domain.temporal import require_aware_datetime

__all__ = [
    "AP_HISTORY_LENGTH",
    "AtmosphereRequest",
    "AtmosphereResponse",
    "DragRequest",
    "DragResponse",
    "DriverRequest",
    "EpistemicStatus",
    "ProvenanceEnvelope",
]

#: NRLMSISE-00 consumes exactly seven ap entries in storm mode: daily Ap, the
#: 3-hourly ap at 0/3/6/9 hours before, and two multi-hour averages. Six is not
#: a shorter history, it is a different array, and the model would read it wrong.
AP_HISTORY_LENGTH = 7

_MIN_LATITUDE_DEG = -90.0
_MAX_LATITUDE_DEG = 90.0


class EpistemicStatus(StrEnum):
    """How a payload came to exist, in the coarse vocabulary the wire uses.

    Deliberately coarser than :class:`~thermosphere_consequence.domain.provenance.ValueStatus`,
    which distinguishes eleven origins. A consumer of the API needs to know
    whether a number was measured, produced by a model, computed from other
    numbers, or chosen in the absence of data; the finer domain vocabulary
    travels inside the payload (see ``AtmosphereResponse.status``) for consumers
    that need it. Two vocabularies with one direction of translation is honest;
    one vocabulary stretched across both jobs would not be.
    """

    OBSERVED = "OBSERVED"
    MODELLED = "MODELLED"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"


class _ApiModel(BaseModel):
    """Frozen, closed base for every wire model.

    Declared once rather than per class so that no model can be added later that
    quietly forgets either property.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class DriverRequest(_ApiModel):
    """The space-weather drivers the caller wants the model to consume.

    Supplied by the caller rather than fetched: this API performs no network
    I/O, so it cannot look an index up, and pretending otherwise by defaulting
    to a plausible F10.7 would produce a number with no traceable input.
    """

    f107: float
    f107a: float
    ap_daily: float
    ap_history: list[float] | None = None

    @field_validator("f107", "f107a")
    @classmethod
    def _flux_must_be_positive(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0.0:
            raise ValueError(f"{info.field_name} must be positive, got {value}")
        return value

    @field_validator("ap_daily")
    @classmethod
    def _ap_must_be_non_negative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError(f"ap_daily must be non-negative, got {value}")
        return value

    @field_validator("ap_history")
    @classmethod
    def _ap_history_must_be_seven_long(cls, value: list[float] | None) -> list[float] | None:
        """Reject any history that is not exactly seven entries.

        The domain's ``DriverState`` enforces this too. It is restated at the
        boundary so the caller gets a 422 that points at ``ap_history`` instead
        of a 500-shaped surprise from deeper in the call chain.
        """
        if value is not None and len(value) != AP_HISTORY_LENGTH:
            raise ValueError(
                f"ap_history must have exactly {AP_HISTORY_LENGTH} entries "
                f"(NRLMSISE-00 storm mode), got {len(value)}"
            )
        return value


class AtmosphereRequest(_ApiModel):
    """One point in space and time, plus the drivers to evaluate it with."""

    time_utc: _dt.datetime
    altitude_m: float
    latitude_deg: float
    longitude_deg: float
    drivers: DriverRequest

    @field_validator("time_utc")
    @classmethod
    def _time_must_be_timezone_aware(cls, value: _dt.datetime) -> _dt.datetime:
        """Reject naive timestamps. UTC is never assumed.

        A naive instant is ambiguous by hours, which is the width of the storm
        onset this package exists to study.
        """
        return require_aware_datetime(value, "time_utc")

    @field_validator("altitude_m")
    @classmethod
    def _altitude_must_be_non_negative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError(f"altitude_m must be non-negative, got {value}")
        return value

    @field_validator("latitude_deg")
    @classmethod
    def _latitude_must_be_on_the_globe(cls, value: float) -> float:
        if not _MIN_LATITUDE_DEG <= value <= _MAX_LATITUDE_DEG:
            raise ValueError(
                f"latitude_deg must lie in "
                f"[{_MIN_LATITUDE_DEG}, {_MAX_LATITUDE_DEG}], got {value}"
            )
        return value


class AtmosphereResponse(_ApiModel):
    """The neutral atmosphere at the requested point.

    ``number_densities`` omits any species the model does not report. An absent
    species is an epistemic statement ("this model says nothing about it"); a
    zero would be a physical claim, and the two must not be confused by a
    serialisation step.

    ``status`` is the domain's finer ``ValueStatus`` vocabulary, carried through
    so a consumer can see that the provider labelled this state ``modelled``
    even though the envelope's coarser status says the same thing.
    """

    temperature_k: float
    number_densities: dict[str, float]
    mass_density_kg_m3: float
    mean_particle_mass_kg: float
    provider_name: str
    provider_version: str
    status: str
    caveats: list[str]

    @field_validator("temperature_k")
    @classmethod
    def _temperature_must_be_positive(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError(f"temperature_k must be positive, got {value}")
        return value


class DragRequest(_ApiModel):
    """A free-molecular drag evaluation.

    ``accommodation`` and ``geometry`` are required, not defaulted. Both are
    modelling choices rather than measurements, and a default would let a caller
    receive a drag coefficient without ever having decided the two assumptions
    that most strongly determine it.
    """

    number_densities: dict[str, float]
    temperature_k: float
    speed_m_s: float
    accommodation: float
    geometry: str
    mass_kg: float
    area_m2: float

    @field_validator("temperature_k")
    @classmethod
    def _temperature_must_be_positive(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError(f"temperature_k must be positive, got {value}")
        return value

    @field_validator("number_densities")
    @classmethod
    def _at_least_one_species(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError(
                "number_densities must name at least one species; a drag "
                "coefficient cannot be mass-weighted over an empty mixture"
            )
        return value


class DragResponse(_ApiModel):
    """A composition-weighted drag coefficient and the inverse ballistic coefficient.

    Both are reported because neither is sufficient alone: C_d is the physics,
    while B = C_d A / m is the combination an observed decay curve can actually
    constrain.
    """

    drag_coefficient: float
    inverse_ballistic_coefficient_m2_kg: float
    geometry: str
    caveats: list[str]


class ProvenanceEnvelope[DataT](_ApiModel):
    """The wrapper every response body must travel in.

    A number leaving the process without its status and caveats is exactly the
    failure this whole system exists to prevent. A density is not a fact; it is
    a model output with a stated origin, known limitations and a named source,
    and the moment those are dropped the number starts being quoted as though it
    were measured. Wrapping is therefore mandatory rather than conventional:
    there is no route out of this API that emits a bare payload.

    ``epistemic_status`` says how the payload came to exist. ``sources`` names
    what backs it. ``caveats`` states what remains wrong with it even so.

    The construction rule mirrors ``ProvenancedValue`` in the domain layer: a
    payload that is not ``ASSUMED`` must name a source, and an ``ASSUMED``
    payload must instead carry at least one caveat explaining the assumption.
    Neither may be satisfied with blank strings.
    """

    data: DataT
    epistemic_status: EpistemicStatus
    sources: list[str]
    caveats: list[str]

    @field_validator("sources", "caveats")
    @classmethod
    def _entries_must_carry_text(cls, value: list[str], info: ValidationInfo) -> list[str]:
        name = info.field_name
        for index, entry in enumerate(value):
            if not entry.strip():
                raise ValueError(f"{name}[{index}] is blank; an empty {name} entry says nothing")
        return value

    @model_validator(mode="after")
    def _must_name_a_source_or_explain_the_assumption(self) -> Self:
        if self.epistemic_status is EpistemicStatus.ASSUMED:
            if not self.caveats:
                raise ValueError(
                    "an ASSUMED payload must carry at least one caveat explaining "
                    "the assumption"
                )
        elif not self.sources:
            raise ValueError(
                f"epistemic_status {self.epistemic_status.value} requires at least one source"
            )
        return self
