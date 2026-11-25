"""Load the event manifest from configuration, and refuse to load a bad one.

The event constants used to live as module-level literals in
:mod:`thermosphere_consequence.events.starlink_g4_7`. That is a fine place for
a number and a poor place for its provenance: the citation ends up in a comment
beside it, the units end up in the variable name, and nothing checks that
either one is still true. This loader moves the values into
``config/events/*.yaml`` and makes the checks executable.

What it refuses to load, rather than repairing quietly:

* a value with no source citation — provenance is not optional, and a default
  citation would be a fabricated one;
* a timestamp without a UTC offset — naive timestamps are ambiguous by hours in
  a context where the storm onset is the thing being studied, and "assume UTC"
  is exactly the silent default this project forbids;
* a unit that is not SI-canonical — kilometres, degrees and nanotesla are
  rejected in favour of metres, radians and tesla, so the conversion a reader
  expects is visible in the manifest instead of hidden in a constant's name;
* a status or authority outside the controlled vocabulary;
* an unknown key anywhere — a typo must not become a silently ignored field;
* a drag parameter whose name and units disagree about which of the two
  reciprocal quantities it is. This one goes through
  :func:`thermosphere_consequence.schemas.claim_gate.guard_value`, so the
  specific error this project made — publishing C_d*A/m in m^2/kg as a
  "ballistic coefficient" — cannot be reintroduced through configuration
  either.

The loader is the authority, and it is the only source.
:mod:`~thermosphere_consequence.events.starlink_g4_7` holds no literals: every
one of its module-level names is read from here through
:func:`load_default_event_manifest` and the typed accessors on
:class:`EventManifest`, at import time. Editing the YAML therefore moves the
scenario, which is the property that makes the provenance recorded in it worth
anything. ``tests/events/test_manifest_loader.py`` pins the manifest's own
contents against the values those constants have always carried, so the file
cannot drift either.

Canonical tickets: P09_10_T001 through T012.
"""

from __future__ import annotations

import datetime as _dt
import functools
import math
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..domain.temporal import NaiveDatetimeError, parse_aware_datetime, require_aware_datetime
from ..domain.units import get_unit_registry
from ..schemas.claim_gate import (
    INVERSE_DRAG_UNITS,
    CalibrationStatus,
    ClaimContext,
    UncertaintyBasis,
    enforce,
    guard_value,
)
from ..schemas.modes import AnalysisMode, DataStatus, ModelMode, SourceAuthority

__all__ = [
    "DecayRecord",
    "EventManifest",
    "EventManifestError",
    "ManifestValue",
    "MissingProvenanceError",
    "NaiveTimestampError",
    "NonSIUnitError",
    "PublishedEnhancement",
    "SourceStatus",
    "UnknownFieldError",
    "default_manifest_path",
    "load_default_event_manifest",
    "load_event_manifest",
]

SCHEMA_VERSION = 1

#: Units that name something other than a physical quantity. Everything else
#: must be SI-canonical.
UTC_UNITS = "utc"
IDENTIFIER_UNITS = "identifier"


class EventManifestError(ValueError):
    """Base class for a manifest this loader will not accept."""


class MissingProvenanceError(EventManifestError):
    """A value is missing its citation, its status or another required field."""


class NaiveTimestampError(EventManifestError):
    """A timestamp carries no UTC offset, and UTC is never assumed."""


class NonSIUnitError(EventManifestError):
    """A unit is unresolvable, or resolvable but not the SI form."""


class UnknownFieldError(EventManifestError):
    """A key the schema does not define — most often a typo."""


class SourceStatus(StrEnum):
    """How a manifest value came to be known.

    Narrower than :class:`thermosphere_consequence.domain.provenance.ValueStatus`
    on purpose: a configuration file is read by people, and four words they can
    keep straight are worth more than eleven they cannot.
    """

    OBSERVED = "observed"
    REPORTED = "reported"
    DERIVED = "derived"
    ASSUMED = "assumed"

    @property
    def data_status(self) -> DataStatus:
        """The run-level :class:`DataStatus` this value contributes."""
        return {
            SourceStatus.OBSERVED: DataStatus.OBSERVED,
            SourceStatus.REPORTED: DataStatus.OBSERVED,
            SourceStatus.DERIVED: DataStatus.DERIVED,
            SourceStatus.ASSUMED: DataStatus.ASSUMED,
        }[self]


_VALUE_REQUIRED = ("value", "units", "status", "authority", "source", "source_url")
#: ``source_sha256`` names the exact bytes a value was read from, when the
#: source document is held in the repository's literature store. It is optional
#: because most entries cite a source nobody has pinned - which is precisely the
#: gap it exists to close. A value carrying it can be traced to a file; a value
#: without it rests on whatever its ``source`` string says, and this project has
#: now had one case where that string named the wrong paper, the wrong author
#: and the wrong spacecraft for four months without anything noticing.
_VALUE_OPTIONAL = (
    "uncertainty_low",
    "uncertainty_high",
    "notes",
    "module_constant",
    "source_sha256",
)

_TOP_LEVEL_REQUIRED = (
    "schema_version",
    "event_id",
    "title",
    "analysis_mode",
    "scenario_label",
    "values",
    "decay_table",
    "published_enhancement",
)

_DECAY_TABLE_REQUIRED = (
    "status",
    "authority",
    "source",
    "source_url",
    "launch_date",
    "records",
)
_DECAY_RECORD_REQUIRED = ("norad_id", "object_id", "decay_date")

_ENHANCEMENT_REQUIRED = (
    "reference",
    "fraction_low",
    "fraction_high",
    "basis",
    "status",
    "authority",
    "source",
    "source_url",
)


@dataclass(frozen=True, slots=True)
class ManifestValue:
    """One event constant, with everything needed to defend it."""

    name: str
    value: float | int | str | _dt.datetime
    units: str
    status: SourceStatus
    authority: SourceAuthority
    source: str
    source_url: str
    uncertainty_low: float | None = None
    uncertainty_high: float | None = None
    notes: str = ""
    module_constant: str = ""
    #: SHA-256 of the source document, when it is held in the literature store.
    #: Empty means nobody has pinned the bytes this value was read from.
    source_sha256: str = ""

    @property
    def number(self) -> float:
        """The value as a float, or a clear error if it is not a number."""
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise EventManifestError(
                f"{self.name!r} holds {self.value!r} ({type(self.value).__name__}), "
                f"which is not a number"
            )
        return float(self.value)

    @property
    def timestamp(self) -> _dt.datetime:
        """The value as an aware UTC datetime, or a clear error."""
        if not isinstance(self.value, _dt.datetime):
            raise EventManifestError(
                f"{self.name!r} holds {self.value!r} ({type(self.value).__name__}), "
                f"which is not a timestamp"
            )
        return self.value

    @property
    def text(self) -> str:
        """The value as a string identifier, or a clear error."""
        if isinstance(self.value, _dt.datetime) or not isinstance(self.value, str | int):
            raise EventManifestError(
                f"{self.name!r} holds {self.value!r} ({type(self.value).__name__}), "
                f"which is not an identifier"
            )
        return str(self.value)

    @property
    def uncertainty_range(self) -> tuple[float, float]:
        """The declared bounds, or a clear error if there are none."""
        if self.uncertainty_low is None or self.uncertainty_high is None:
            raise EventManifestError(f"{self.name!r} declares no uncertainty range")
        return (self.uncertainty_low, self.uncertainty_high)


@dataclass(frozen=True, slots=True)
class DecayRecord:
    """One row of the public catalogue.

    Deliberately inert: it carries what the catalogue recorded and nothing
    else. The rule that turns a decay date into a "storm loss" is a modelling
    decision and lives with the analysis, not with the data.
    """

    norad_id: int
    object_id: str
    decay_date: _dt.date | None


@dataclass(frozen=True, slots=True)
class PublishedEnhancement:
    """A published density-enhancement figure, kept for comparison only."""

    reference: str
    fraction_low: float
    fraction_high: float
    basis: str
    status: SourceStatus
    authority: SourceAuthority
    source: str
    source_url: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class EventManifest:
    """A loaded, validated event manifest."""

    event_id: str
    title: str
    analysis_mode: AnalysisMode
    scenario_label: str
    values: Mapping[str, ManifestValue]
    decays: tuple[DecayRecord, ...]
    decay_source: str
    decay_source_url: str
    launch_date: _dt.date
    published_enhancement: tuple[PublishedEnhancement, ...]
    source_path: pathlib.Path

    # ---- accessors -------------------------------------------------------

    def value(self, name: str) -> ManifestValue:
        """The named value, or an error listing what the manifest does hold."""
        try:
            return self.values[name]
        except KeyError as exc:
            raise EventManifestError(
                f"{self.source_path.name}: no value named {name!r}; "
                f"available: {sorted(self.values)}"
            ) from exc

    def number_of(self, name: str) -> float:
        """The named value as a float."""
        return self.value(name).number

    def timestamp_of(self, name: str) -> _dt.datetime:
        """The named value as an aware UTC datetime."""
        return self.value(name).timestamp

    def text_of(self, name: str) -> str:
        """The named value as a string identifier."""
        return self.value(name).text

    def module_constants(self) -> Mapping[str, str]:
        """Module constant name -> manifest value name, for the migration.

        Only values that declare ``module_constant`` appear. It is how a caller
        (and the equivalence test) can tell which literals this manifest is
        able to replace.
        """
        return MappingProxyType(
            {v.module_constant: name for name, v in self.values.items() if v.module_constant}
        )


# --- validation helpers ---------------------------------------------------


def _mapping(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EventManifestError(f"{where}: expected a mapping, got {type(raw).__name__}")
    return raw


def _check_keys(
    payload: Mapping[str, Any],
    required: tuple[str, ...],
    optional: tuple[str, ...],
    where: str,
) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise MissingProvenanceError(
            f"{where}: missing required field(s) {missing}; required: {list(required)}"
        )
    unknown = [key for key in payload if key not in required and key not in optional]
    if unknown:
        raise UnknownFieldError(
            f"{where}: unknown field(s) {sorted(unknown)}. The schema defines "
            f"{sorted(set(required) | set(optional))}; an unrecognised key is a "
            f"typo, not a default"
        )


def _timestamp(raw: Any, where: str) -> _dt.datetime:
    """Parse a manifest timestamp, rejecting anything without an offset."""
    if isinstance(raw, _dt.datetime):
        try:
            return require_aware_datetime(raw, where).astimezone(_dt.UTC)
        except NaiveDatetimeError as exc:
            raise NaiveTimestampError(
                f"{where}: {raw!r} is not timezone-aware. Write the timestamp as "
                f"a quoted ISO-8601 string with an explicit offset or 'Z'; UTC is "
                f"never assumed"
            ) from exc
    if not isinstance(raw, str):
        raise EventManifestError(f"{where}: expected an ISO-8601 timestamp, got {raw!r}")
    try:
        return parse_aware_datetime(raw, where)
    except NaiveDatetimeError as exc:
        raise NaiveTimestampError(
            f"{where}: {raw!r} is not timezone-aware. Supply an explicit offset "
            f"or a trailing 'Z'; UTC is never assumed"
        ) from exc
    except ValueError as exc:
        raise EventManifestError(f"{where}: {raw!r} is not a valid ISO-8601 timestamp") from exc


def _date(raw: Any, where: str) -> _dt.date:
    if isinstance(raw, _dt.datetime):
        raise EventManifestError(
            f"{where}: expected a calendar date, got the timestamp {raw!r}. A "
            f"catalogue decay date has no time of day; do not invent one"
        )
    if isinstance(raw, _dt.date):
        return raw
    if not isinstance(raw, str):
        raise EventManifestError(f"{where}: expected an ISO-8601 date, got {raw!r}")
    try:
        return _dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise EventManifestError(f"{where}: {raw!r} is not a valid ISO-8601 date") from exc


def _check_si_units(units: str, where: str) -> None:
    """Require the unit to be the SI form for its dimensionality.

    The test is general rather than a list: one of the unit converted to base
    units must still be one. Kilometres give 1000, degrees give 0.0175, degrees
    Celsius carry an offset — all rejected, and no allowlist has to be
    maintained as new quantities appear.
    """
    registry = get_unit_registry()
    try:
        base = registry.Quantity(1.0, units).to_base_units()
    except Exception as exc:
        raise NonSIUnitError(f"{where}: unit {units!r} is not resolvable") from exc
    if not math.isclose(float(base.magnitude), 1.0, rel_tol=1e-12, abs_tol=0.0):
        raise NonSIUnitError(
            f"{where}: unit {units!r} is not SI-canonical — one {units} is "
            f"{base.magnitude!r} {base.units}. Convert the value and declare the "
            f"SI unit, so the conversion is visible here rather than implied by a "
            f"constant's name"
        )


def _enum(value: Any, enum_type: type[StrEnum], where: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as exc:
        raise MissingProvenanceError(
            f"{where}: {value!r} is not a known {enum_type.__name__}; "
            f"permitted: {[m.value for m in enum_type]}"
        ) from exc


def _check_citation(payload: Mapping[str, Any], where: str) -> tuple[str, str]:
    source = str(payload.get("source", "") or "").strip()
    if not source:
        raise MissingProvenanceError(
            f"{where}: no source citation. Every value must name where it came "
            f"from; there is no default and an uncited value is not loadable"
        )
    url = str(payload.get("source_url", "") or "").strip()
    if not url and not str(payload.get("notes", "") or "").strip():
        raise MissingProvenanceError(
            f"{where}: source_url is empty and no note explains why. An absent "
            f"link is acceptable — an unexplained absent link is not, because a "
            f"plausible-looking URL is easier to invent than to verify"
        )
    return source, url


def _number(raw: Any, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise EventManifestError(f"{where}: expected a number, got {raw!r}")
    return float(raw)


def _parse_value(name: str, payload: Mapping[str, Any], where: str) -> ManifestValue:
    _check_keys(payload, _VALUE_REQUIRED, _VALUE_OPTIONAL, where)
    source, url = _check_citation(payload, where)
    status = _enum(payload["status"], SourceStatus, f"{where}.status")
    authority = _enum(payload["authority"], SourceAuthority, f"{where}.authority")
    units = str(payload["units"]).strip()
    raw = payload["value"]

    value: float | int | str | _dt.datetime
    if units == UTC_UNITS:
        value = _timestamp(raw, where)
    elif units == IDENTIFIER_UNITS:
        if isinstance(raw, bool) or not isinstance(raw, str | int):
            raise EventManifestError(
                f"{where}: an identifier must be a string or an integer key, got {raw!r}"
            )
        value = raw
    else:
        _check_si_units(units, where)
        value = _number(raw, where)

    low = None if payload.get("uncertainty_low") is None else _number(
        payload["uncertainty_low"], f"{where}.uncertainty_low"
    )
    high = None if payload.get("uncertainty_high") is None else _number(
        payload["uncertainty_high"], f"{where}.uncertainty_high"
    )
    if low is not None and high is not None and low > high:
        raise EventManifestError(f"{where}: uncertainty bounds inverted: {low} > {high}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if low is not None and low > float(value):
            raise EventManifestError(f"{where}: uncertainty_low {low} exceeds value {value}")
        if high is not None and high < float(value):
            raise EventManifestError(f"{where}: uncertainty_high {high} is below value {value}")
        # The gate that makes the units error impossible to reintroduce.
        guard_value(name, float(value), units)

    return ManifestValue(
        name=name,
        value=value,
        units=units,
        status=status,
        authority=authority,
        source=source,
        source_url=url,
        uncertainty_low=low,
        uncertainty_high=high,
        notes=str(payload.get("notes", "") or "").strip(),
        module_constant=str(payload.get("module_constant", "") or "").strip(),
        source_sha256=str(payload.get("source_sha256", "") or "").strip(),
    )


def _parse_decay_table(
    payload: Mapping[str, Any], where: str
) -> tuple[tuple[DecayRecord, ...], str, str, _dt.date]:
    _check_keys(payload, _DECAY_TABLE_REQUIRED, ("notes",), where)
    source, url = _check_citation(payload, where)
    _enum(payload["status"], SourceStatus, f"{where}.status")
    _enum(payload["authority"], SourceAuthority, f"{where}.authority")
    launch_date = _date(payload["launch_date"], f"{where}.launch_date")

    rows = payload["records"]
    if not isinstance(rows, list) or not rows:
        raise EventManifestError(f"{where}.records: expected a non-empty list")

    records: list[DecayRecord] = []
    seen: set[int] = set()
    for index, row in enumerate(rows):
        row_where = f"{where}.records[{index}]"
        row_map = _mapping(row, row_where)
        _check_keys(row_map, _DECAY_RECORD_REQUIRED, (), row_where)
        norad = row_map["norad_id"]
        if isinstance(norad, bool) or not isinstance(norad, int):
            raise EventManifestError(f"{row_where}.norad_id: expected an integer, got {norad!r}")
        if norad in seen:
            raise EventManifestError(f"{row_where}.norad_id: {norad} appears twice")
        seen.add(norad)
        decay = row_map["decay_date"]
        records.append(
            DecayRecord(
                norad_id=norad,
                object_id=str(row_map["object_id"]),
                decay_date=None if decay is None else _date(decay, f"{row_where}.decay_date"),
            )
        )
    return tuple(records), source, url, launch_date


def _parse_enhancement(payload: Mapping[str, Any], where: str) -> PublishedEnhancement:
    _check_keys(payload, _ENHANCEMENT_REQUIRED, ("notes",), where)
    source, url = _check_citation(payload, where)
    low = _number(payload["fraction_low"], f"{where}.fraction_low")
    high = _number(payload["fraction_high"], f"{where}.fraction_high")
    if low > high:
        raise EventManifestError(f"{where}: fraction bounds inverted: {low} > {high}")
    return PublishedEnhancement(
        reference=str(payload["reference"]),
        fraction_low=low,
        fraction_high=high,
        basis=str(payload["basis"]),
        status=_enum(payload["status"], SourceStatus, f"{where}.status"),
        authority=_enum(payload["authority"], SourceAuthority, f"{where}.authority"),
        source=source,
        source_url=url,
        notes=str(payload.get("notes", "") or "").strip(),
    )


def _strictest_context(analysis_mode: AnalysisMode) -> ClaimContext:
    """The most restrictive reading of a manifest, for label enforcement.

    A manifest describes an event, not a run, so it cannot know which model
    mode or calibration state will consume it. The label therefore has to hold
    up under the strictest one: nothing ran, nothing is measured, nothing is
    validated, and the drag parameter is in m^2/kg.
    """
    return ClaimContext(
        analysis_mode=analysis_mode,
        model_mode=ModelMode.NATIVE_SURYA_NOT_RUN,
        data_status=DataStatus.MODELLED,
        source_authority=SourceAuthority.DERIVED,
        uncertainty_basis=UncertaintyBasis.SCENARIO,
        has_probabilistic_model=False,
        value_units=INVERSE_DRAG_UNITS,
        attribution_ordering_stable=None,
        calibration_status=CalibrationStatus.CALIBRATED_IN_SAMPLE,
    )


def load_event_manifest(path: str | pathlib.Path) -> EventManifest:
    """Load and validate an event manifest.

    Raises:
        EventManifestError: and its subclasses, for anything the schema does
            not accept. Nothing is defaulted, repaired or skipped: a manifest
            either states what it needs to state, or it does not load.
    """
    manifest_path = pathlib.Path(path)
    if not manifest_path.is_file():
        raise EventManifestError(f"event manifest not found: {manifest_path}")

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    where = manifest_path.name
    document = _mapping(raw, where)
    _check_keys(document, _TOP_LEVEL_REQUIRED, (), where)

    version = document["schema_version"]
    if version != SCHEMA_VERSION:
        raise EventManifestError(
            f"{where}: schema_version {version!r} is not supported by this loader "
            f"(expected {SCHEMA_VERSION}). Migrate the file rather than loosening "
            f"the check"
        )

    analysis_mode = _enum(document["analysis_mode"], AnalysisMode, f"{where}.analysis_mode")
    scenario_label = str(document["scenario_label"]).strip()
    if analysis_mode is AnalysisMode.SCENARIO and not scenario_label:
        raise MissingProvenanceError(
            f"{where}: analysis_mode is 'scenario' but scenario_label is empty. A "
            f"scenario that does not say it is one reads as a reconstruction"
        )
    title = str(document["title"]).strip()
    if not title:
        raise MissingProvenanceError(f"{where}: title is empty")

    context = _strictest_context(analysis_mode)
    enforce(title, context=context, source=f"{where}:title")
    enforce(scenario_label, context=context, source=f"{where}:scenario_label")

    values_payload = _mapping(document["values"], f"{where}.values")
    if not values_payload:
        raise EventManifestError(f"{where}.values: an event manifest with no values is empty")
    values = {}
    for name, payload in values_payload.items():
        value_where = f"{where}.values.{name}"
        values[name] = _parse_value(name, _mapping(payload, value_where), value_where)

    decays, decay_source, decay_url, launch_date = _parse_decay_table(
        _mapping(document["decay_table"], f"{where}.decay_table"), f"{where}.decay_table"
    )

    enhancement_payload = document["published_enhancement"]
    if not isinstance(enhancement_payload, list) or not enhancement_payload:
        raise EventManifestError(f"{where}.published_enhancement: expected a non-empty list")
    enhancements = tuple(
        _parse_enhancement(
            _mapping(item, f"{where}.published_enhancement[{index}]"),
            f"{where}.published_enhancement[{index}]",
        )
        for index, item in enumerate(enhancement_payload)
    )

    return EventManifest(
        event_id=str(document["event_id"]),
        title=title,
        analysis_mode=analysis_mode,
        scenario_label=scenario_label,
        values=MappingProxyType(values),
        decays=decays,
        decay_source=decay_source,
        decay_source_url=decay_url,
        launch_date=launch_date,
        published_enhancement=enhancements,
        source_path=manifest_path,
    )


def default_manifest_path(event_id: str = "starlink_group_4_7_2022") -> pathlib.Path:
    """Where an event manifest lives, relative to the repository root."""
    root = pathlib.Path(__file__).resolve().parents[3]
    return root / "config" / "events" / f"{event_id}.yaml"


@functools.lru_cache(maxsize=8)
def load_default_event_manifest(event_id: str = "starlink_group_4_7_2022") -> EventManifest:
    """Load the shipped manifest for ``event_id``, once per process.

    Cached because the manifest is immutable and validating it involves parsing
    YAML and building a unit registry; a module sourcing its constants from
    here should not pay that per import.
    """
    return load_event_manifest(default_manifest_path(event_id))
