"""Timezone-aware temporal handling.

A naive timestamp in a space-weather context is ambiguous by hours, which is
comparable to the entire storm-onset window this project studies. Every instant
entering the system is therefore required to be timezone-aware; there is
deliberately no "assume UTC" path, because silently assuming UTC is exactly the
error this module exists to prevent.

Canonical ticket: P00_02_T034.
"""

from __future__ import annotations

import datetime as _dt

__all__ = [
    "NaiveDatetimeError",
    "require_aware_datetime",
    "ensure_utc",
    "parse_aware_datetime",
    "format_utc",
]

_UTC = _dt.UTC


class NaiveDatetimeError(ValueError):
    """Raised when a datetime lacks a usable UTC offset.

    Subclasses ``ValueError`` so callers validating untrusted input can catch
    the broad case, while still being able to distinguish this one.
    """


def require_aware_datetime(value: _dt.datetime, field: str = "datetime") -> _dt.datetime:
    """Return ``value`` unchanged if it is timezone-aware, else raise.

    A tzinfo whose ``utcoffset()`` returns ``None`` is treated as naive: it
    carries no usable offset and would be just as ambiguous.
    """
    if not isinstance(value, _dt.datetime):
        raise TypeError(f"{field}: expected datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise NaiveDatetimeError(
            f"{field}: naive datetime {value!r} is not accepted. Supply an "
            f"aware datetime; UTC is never assumed."
        )
    return value


def ensure_utc(value: _dt.datetime, field: str = "datetime") -> _dt.datetime:
    """Normalise an aware datetime to UTC, preserving the absolute instant."""
    return require_aware_datetime(value, field).astimezone(_UTC)


def parse_aware_datetime(text: str, field: str = "datetime") -> _dt.datetime:
    """Parse an ISO-8601 string that carries an explicit offset or ``Z``.

    A string without an offset is rejected rather than assumed UTC.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field}: expected a non-empty ISO-8601 string")
    candidate = text.strip()
    normalised = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
    try:
        parsed = _dt.datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise ValueError(f"{field}: {text!r} is not a valid ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise NaiveDatetimeError(
            f"{field}: {text!r} has no UTC offset. Supply an offset or a "
            f"trailing 'Z'; UTC is never assumed."
        )
    return parsed.astimezone(_UTC)


def format_utc(value: _dt.datetime, field: str = "datetime") -> str:
    """Serialise to canonical RFC 3339 UTC with a trailing ``Z``."""
    aware = ensure_utc(value, field)
    return aware.strftime("%Y-%m-%dT%H:%M:%S") + (
        f".{aware.microsecond:06d}" if aware.microsecond else ""
    ) + "Z"
