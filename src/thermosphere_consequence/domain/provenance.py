"""ProvenancedValue: no critical quantity exists as an unlabelled float.

The distinction between an observation, a model output, a literature estimate
and an assumption is the difference between a defensible result and an
unsupportable claim. Encoding it in the type makes that distinction impossible
to lose in transit.

Deliberately absent: arithmetic operators. Combining two provenanced values is
a decision about provenance, not a numeric convenience, so it must be explicit
at the call site.

Canonical ticket: P00_02_T033.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .temporal import ensure_utc, format_utc, parse_aware_datetime
from .units import SI

__all__ = ["ValueStatus", "QualityFlag", "ProvenancedValue"]


class ValueStatus(StrEnum):
    """How a value came to exist. There is no default; the caller must say."""

    OBSERVED = "observed"
    OPERATOR_SUPPLIED = "operator_supplied"
    MODELLED = "modelled"
    CATALOGUE_DERIVED = "catalogue_derived"
    LITERATURE_ESTIMATE = "literature_estimate"
    ASSUMED = "assumed"
    CALCULATED = "calculated"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    PROVISIONAL = "provisional"
    DEFINITIVE = "definitive"


class QualityFlag(StrEnum):
    """Conditions that travel with a value from its upstream source."""

    MISSING = "missing"
    OBSERVED = "observed"
    MODELLED = "modelled"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    PROVISIONAL = "provisional"
    DEFINITIVE = "definitive"
    CATALOGUE_DERIVED = "catalogue_derived"
    OPERATOR_SUPPLIED = "operator_supplied"
    ASSUMED = "assumed"
    OUT_OF_DOMAIN = "out_of_domain"
    COVERAGE_GAP = "coverage_gap"
    LICENCE_RESTRICTED = "licence_restricted"
    AUTHENTICATION_REQUIRED = "authentication_required"
    QUALITY_UNKNOWN = "quality_unknown"


@dataclass(frozen=True, slots=True)
class ProvenancedValue:
    """A number that knows where it came from.

    Invariants enforced at construction:
      * ``status`` is from the controlled vocabulary and is never defaulted;
      * timestamps are timezone-aware (naive input raises);
      * uncertainty bounds, when present, bracket the value;
      * the unit string is a recognised SI quantity name or unit;
      * a value that is not ``assumed`` names a source; an ``assumed`` value
        must instead carry a note explaining the assumption.
    """

    value: float | None
    unit: str
    status: ValueStatus
    source_name: str = ""
    source_reference: str = ""
    valid_time_utc: _dt.datetime | None = None
    retrieval_time_utc: _dt.datetime | None = None
    uncertainty_low: float | None = None
    uncertainty_high: float | None = None
    quality_flags: tuple[QualityFlag, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, ValueStatus):
            raise ValueError(
                f"status must be a ValueStatus, got {self.status!r}; "
                f"permitted: {[s.value for s in ValueStatus]}"
            )
        if not str(self.unit).strip():
            raise ValueError("unit must be a non-empty string")
        if self.unit not in SI and self.unit not in SI.values():
            # Allow any pint-resolvable spelling, but reject empty/garbage early.
            from .units import get_unit_registry

            try:
                get_unit_registry().Unit(self.unit)
            except Exception as exc:
                raise ValueError(f"unit {self.unit!r} is not resolvable") from exc

        for name in ("valid_time_utc", "retrieval_time_utc"):
            ts = getattr(self, name)
            if ts is not None:
                object.__setattr__(self, name, ensure_utc(ts, name))

        lo, hi, val = self.uncertainty_low, self.uncertainty_high, self.value
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"uncertainty bounds inverted: {lo} > {hi}")
        if val is not None:
            if lo is not None and lo > val:
                raise ValueError(f"uncertainty_low {lo} exceeds value {val}")
            if hi is not None and hi < val:
                raise ValueError(f"uncertainty_high {hi} is below value {val}")

        for f in self.quality_flags:
            if not isinstance(f, QualityFlag):
                raise ValueError(f"unknown quality flag {f!r}")

        has_source = bool(str(self.source_name).strip() and str(self.source_reference).strip())
        if self.status is ValueStatus.ASSUMED:
            if not str(self.notes).strip():
                raise ValueError(
                    "an 'assumed' value must carry a note explaining the assumption"
                )
        elif not has_source:
            raise ValueError(
                f"status {self.status.value!r} requires both source_name and "
                f"source_reference"
            )

    # ---- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "status": self.status.value,
            "source_name": self.source_name,
            "source_reference": self.source_reference,
            "valid_time_utc": format_utc(self.valid_time_utc) if self.valid_time_utc else None,
            "retrieval_time_utc": (
                format_utc(self.retrieval_time_utc) if self.retrieval_time_utc else None
            ),
            "uncertainty_low": self.uncertainty_low,
            "uncertainty_high": self.uncertainty_high,
            "quality_flags": [f.value for f in self.quality_flags],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenancedValue:
        def _ts(key: str) -> _dt.datetime | None:
            raw = data.get(key)
            return parse_aware_datetime(raw, key) if raw else None

        return cls(
            value=data.get("value"),
            unit=data["unit"],
            status=ValueStatus(data["status"]),
            source_name=data.get("source_name", ""),
            source_reference=data.get("source_reference", ""),
            valid_time_utc=_ts("valid_time_utc"),
            retrieval_time_utc=_ts("retrieval_time_utc"),
            uncertainty_low=data.get("uncertainty_low"),
            uncertainty_high=data.get("uncertainty_high"),
            quality_flags=tuple(QualityFlag(f) for f in data.get("quality_flags", ())),
            notes=data.get("notes", ""),
        )
