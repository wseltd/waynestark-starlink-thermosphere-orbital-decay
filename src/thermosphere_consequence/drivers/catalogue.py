"""Orbit-catalogue ingestion: CelesTrak GP element sets and SATCAT records.

Two CelesTrak endpoints answer two different questions, and conflating them is
the mistake this module exists to prevent:

  * ``gp.php`` serves GP (general perturbations) mean element sets for objects
    that are CURRENTLY ON ORBIT. An object that has reentered simply stops
    appearing. The absence of a GP record is therefore not evidence of decay —
    it is equally consistent with a tracking gap or a query mistake.

  * ``satcat/records.php`` serves the satellite catalogue, which carries
    ``DECAY_DATE``. That field is this project's GROUND TRUTH for whether and
    when an object reentered. Every decay claim in a result traces back to it,
    not to the disappearance of an element set and not to an integration that
    happened to hit the reentry floor.

Two provenance rules are enforced by shape rather than by convention:

  * A missing ``DECAY_DATE`` stays ``None``. It is never widened to a sentinel
    date, never zero, never "today". ``None`` means "this object has not been
    recorded as decayed"; any date would be a claim the catalogue did not make.

  * A record the catalogue cannot supply in full is REJECTED, loudly, naming
    the offending index and field. Silently skipping a malformed record would
    quietly change the denominator of a survival statistic, which is the one
    number this project is built to report.

Network access lives behind an injected fetcher, as everywhere else in
``drivers``, so the module is fully testable offline. The two URLs are fixed
templates and the only caller-supplied component is an international
designator matched against ``INTERNATIONAL_DESIGNATOR``, so there is no code
path here that can be steered at a different host.

Canonical tickets: P03_05_T030, P03_05_T031, P03_05_T033.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.temporal import ensure_utc, parse_aware_datetime
from .raw_store import RawStore
from .space_weather import CELESTRAK_LICENCE, CELESTRAK_LICENCE_URL, Fetcher

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_SATCAT_URL = "https://celestrak.org/satcat/records.php"
GP_CITATION = "CelesTrak GP element sets (Kelso), Space-Track-derived"
SATCAT_CITATION = "CelesTrak SATCAT (Kelso), Space-Track-derived"

# COSPAR international designator: launch year, launch number of that year and
# an optional piece suffix (2022-010, 2022-010A, 1998-067A).
INTERNATIONAL_DESIGNATOR = re.compile(r"^\d{4}-\d{3}[A-Z]{0,3}$")

# SATCAT records a decay to CALENDAR-DAY resolution and nothing finer. A decay
# date is represented as 00:00:00Z on the recorded day — the START of that day
# — so the true reentry lies somewhere in the following 86400 s. Code that
# compares a modelled reentry epoch against a catalogue decay date must widen
# by this amount rather than pretend to a precision the source does not have.
DECAY_DATE_RESOLUTION_S = 86_400.0

# SATCAT operational-status codes, spelled out so downstream code never has to
# guess what a bare "+" or "D" meant. Source: CelesTrak SATCAT field
# documentation.
OPS_STATUS_MEANINGS: Mapping[str, str] = {
    "+": "operational",
    "-": "nonoperational",
    "P": "partially operational",
    "B": "backup or standby",
    "S": "spare",
    "X": "extended mission",
    "D": "decayed",
    "?": "unknown",
    "": "unstated",
}


class CatalogueParseError(ValueError):
    """Raised when a catalogue record cannot be parsed in full.

    Subclasses ``ValueError`` so a caller validating an untrusted payload can
    catch the broad case, while still being able to distinguish a catalogue
    defect from any other bad input.
    """


# --------------------------------------------------------------------------
# record types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CatalogueObject:
    """One SATCAT record: identity, launch, geometry and decay ground truth.

    ``decay_date`` is the authoritative statement that an object reentered.
    ``ops_status`` is a curated label that can lag the catalogue by days and is
    therefore kept beside the decay date rather than substituted for it — the
    two are allowed to disagree, and ``survivors``/``decayed`` partition on the
    decay date alone.

    ``launch_date`` and ``decay_date`` are day-resolution values from the
    source, carried as timezone-aware UTC instants at the start of the recorded
    day; see ``DECAY_DATE_RESOLUTION_S``.
    """

    norad_id: int
    object_id: str
    object_name: str
    launch_date: _dt.datetime
    decay_date: _dt.datetime | None
    period_min: float
    inclination_deg: float
    apogee_km: float
    perigee_km: float
    ops_status: str
    object_type: str = ""
    owner: str = ""
    source: str = "CelesTrak SATCAT"
    licence: str = CELESTRAK_LICENCE
    citation: str = SATCAT_CITATION

    def __post_init__(self) -> None:
        object.__setattr__(self, "launch_date", ensure_utc(self.launch_date, "launch_date"))
        if self.decay_date is not None:
            object.__setattr__(self, "decay_date", ensure_utc(self.decay_date, "decay_date"))
            if self.decay_date < self.launch_date:
                raise ValueError(
                    f"{self.object_id}: decay date {self.decay_date.date()} precedes "
                    f"launch date {self.launch_date.date()}; the catalogue record is "
                    "internally inconsistent and is rejected rather than reordered"
                )
        if self.period_min <= 0:
            raise ValueError(f"{self.object_id}: orbital period must be positive")
        if not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError(
                f"{self.object_id}: inclination {self.inclination_deg} is outside 0-180 degrees"
            )
        if self.apogee_km < self.perigee_km:
            raise ValueError(
                f"{self.object_id}: apogee {self.apogee_km} km is below perigee "
                f"{self.perigee_km} km"
            )

    @property
    def has_decayed(self) -> bool:
        """True only when the catalogue states a decay date."""
        return self.decay_date is not None

    @property
    def days_on_orbit(self) -> float | None:
        """Recorded lifetime in days, or ``None`` while the object is on orbit.

        ``None`` rather than "days so far", because a censored lifetime and a
        completed one are different quantities and averaging them together is
        the classic way to understate a constellation's losses.
        """
        if self.decay_date is None:
            return None
        return (self.decay_date - self.launch_date).total_seconds() / 86400.0

    @property
    def ops_status_description(self) -> str:
        """The SATCAT status code in words, or the raw code if unrecognised."""
        return OPS_STATUS_MEANINGS.get(self.ops_status, f"unrecognised code {self.ops_status!r}")


@dataclass(frozen=True, slots=True)
class ElementSet:
    """One GP (SGP4 mean) element set, as served in CCSDS OMM form.

    These are MEAN elements in the specific sense SGP4 defines: they are only
    meaningful when propagated by SGP4 itself, and feeding them to a numerical
    integrator as if they were osculating elements is wrong by kilometres.
    ``ephemeris_type`` 0 is the SGP4/SDP4 declaration; any other value means the
    set is not SGP4 material at all.

    ``tle_line1``/``tle_line2`` are ``None`` when the source does not carry the
    two-line text — CelesTrak's ``gp.php?FORMAT=json`` returns OMM fields only,
    whereas Space-Track's GP class also returns ``TLE_LINE1``/``TLE_LINE2``.
    ``None`` records that absence instead of an empty string, so a caller
    cannot hand a blank line to a TLE parser and get zeros back.
    """

    norad_id: int
    object_id: str
    epoch_utc: _dt.datetime
    mean_motion: float
    eccentricity: float
    inclination_deg: float
    bstar: float
    raan_deg: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_dot: float
    mean_motion_ddot: float
    tle_line1: str | None = None
    tle_line2: str | None = None
    object_name: str = ""
    element_set_no: int | None = None
    rev_at_epoch: int | None = None
    ephemeris_type: int = 0
    classification: str = "U"
    source: str = "CelesTrak GP"
    licence: str = CELESTRAK_LICENCE
    citation: str = GP_CITATION

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch_utc", ensure_utc(self.epoch_utc, "epoch_utc"))
        if self.mean_motion <= 0:
            raise ValueError(
                f"{self.object_id}: mean motion must be positive, got {self.mean_motion}"
            )
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError(
                f"{self.object_id}: eccentricity {self.eccentricity} is not a closed orbit"
            )
        if not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError(
                f"{self.object_id}: inclination {self.inclination_deg} is outside 0-180 degrees"
            )

    @property
    def period_minutes(self) -> float:
        """Nodal period implied by the mean motion, in minutes."""
        return 1440.0 / self.mean_motion

    @property
    def two_line_elements(self) -> tuple[str, str]:
        """The two-line text, raising when the source did not supply it.

        A property rather than an attribute access so the failure lands here,
        with the object's identity in the message, instead of inside an SGP4
        parser that would read a blank line as a satellite at the origin.
        """
        if self.tle_line1 is None or self.tle_line2 is None:
            raise CatalogueParseError(
                f"{self.object_id} (NORAD {self.norad_id}) has no two-line text: this "
                "element set came from a source that returns OMM fields only "
                "(CelesTrak gp.php?FORMAT=json). Fetch FORMAT=tle, or propagate "
                "from the OMM fields instead."
            )
        return self.tle_line1, self.tle_line2


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def validate_international_designator(int_des: str) -> str:
    """Normalise and check a COSPAR designator before it enters a URL.

    An unchecked designator is interpolated straight into a query string, and
    CelesTrak answers a malformed ``INTDES`` with an error page rather than an
    empty list. Stored verbatim, that page would sit in the raw store wearing
    the provenance of the launch that was actually requested.
    """
    text = str(int_des).strip().upper()
    if not INTERNATIONAL_DESIGNATOR.match(text):
        raise ValueError(
            f"{int_des!r} is not a COSPAR international designator. Expected "
            "YYYY-NNN with an optional piece suffix, e.g. '2022-010' or '1998-067A'."
        )
    return text


def fetch_celestrak_gp(
    int_des: str,
    *,
    fetcher: Fetcher,
    store: RawStore | None = None,
    asset_id: str | None = None,
) -> tuple[ElementSet, ...]:
    """Fetch GP element sets for a launch, storing the raw JSON verbatim.

    Returns only objects CURRENTLY ON ORBIT: CelesTrak drops an object from the
    GP feed once it reenters. Use ``fetch_celestrak_satcat`` for decay status.
    """
    designator = validate_international_designator(int_des)
    url = f"{CELESTRAK_GP_URL}?INTDES={designator}&FORMAT=json"
    payload = fetcher(url)
    if store is not None:
        store.write(
            source="celestrak",
            asset_id=asset_id or f"gp_{designator}",
            payload=payload,
            url=url,
            licence=CELESTRAK_LICENCE,
            licence_url=CELESTRAK_LICENCE_URL,
            citation=GP_CITATION,
            content_type="application/json",
            ext=".json",
        )
    return parse_gp(payload)


def fetch_celestrak_satcat(
    int_des: str,
    *,
    fetcher: Fetcher,
    store: RawStore | None = None,
    asset_id: str | None = None,
) -> tuple[CatalogueObject, ...]:
    """Fetch SATCAT records for a launch, storing the raw JSON verbatim.

    This is the endpoint that carries ``DECAY_DATE``, so it is the one every
    decay claim in this project is traceable to. It lists every piece of the
    launch, decayed or not, which is what makes a survival fraction computable.
    """
    designator = validate_international_designator(int_des)
    url = f"{CELESTRAK_SATCAT_URL}?INTDES={designator}&FORMAT=json"
    payload = fetcher(url)
    if store is not None:
        store.write(
            source="celestrak",
            asset_id=asset_id or f"satcat_{designator}",
            payload=payload,
            url=url,
            licence=CELESTRAK_LICENCE,
            licence_url=CELESTRAK_LICENCE_URL,
            citation=SATCAT_CITATION,
            content_type="application/json",
            ext=".json",
        )
    return parse_satcat(payload)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _records(payload: bytes, what: str) -> list[dict[str, Any]]:
    """Decode a CelesTrak JSON payload to a list of records, or raise."""
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CatalogueParseError(
            f"{what} payload is not valid JSON ({exc}). CelesTrak answers a bad "
            "query with an HTML error page; check the designator and the stored "
            "raw asset before re-parsing."
        ) from exc
    if isinstance(document, dict):
        document = [document]
    if not isinstance(document, list):
        raise CatalogueParseError(
            f"{what} payload decoded to {type(document).__name__}, expected a list of records"
        )
    for index, record in enumerate(document):
        if not isinstance(record, dict):
            raise CatalogueParseError(
                f"{what} record {index} is a {type(record).__name__}, expected an object"
            )
    return list(document)


def _present(record: Mapping[str, Any], key: str, what: str, index: int) -> Any:
    """Return a required field, raising when it is absent, null or blank."""
    if key not in record:
        raise CatalogueParseError(
            f"{what} record {index} is missing the required field {key!r}; "
            f"present fields: {sorted(record)}"
        )
    value = record[key]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise CatalogueParseError(
            f"{what} record {index}: {key!r} is empty. The record is rejected rather "
            "than admitted with a substituted value."
        )
    return value


def _text(record: Mapping[str, Any], key: str, what: str, index: int) -> str:
    return str(_present(record, key, what, index)).strip()


def _number(record: Mapping[str, Any], key: str, what: str, index: int) -> float:
    value = _present(record, key, what, index)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CatalogueParseError(
            f"{what} record {index}: {key!r} is {value!r}, which is not a number"
        ) from exc


def _integer(record: Mapping[str, Any], key: str, what: str, index: int) -> int:
    value = _present(record, key, what, index)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CatalogueParseError(
            f"{what} record {index}: {key!r} is {value!r}, which is not an integer"
        ) from exc


def _optional_integer(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_line(record: Mapping[str, Any], key: str) -> str | None:
    """A TLE line if the source carried one, else ``None`` — never ``''``."""
    value = record.get(key)
    if value is None or not str(value).strip():
        return None
    return str(value).rstrip("\r\n")


def _calendar_day_utc(text: str, what: str, key: str, index: int) -> _dt.datetime:
    """Convert a SATCAT ``YYYY-MM-DD`` field to the start of that UTC day.

    This does not violate the project's "UTC is never assumed" rule. That rule
    guards against a naive INSTANT, whose true offset could be anything. A
    SATCAT date is not an instant: it is a calendar day that the catalogue
    itself keeps in UTC, and it is widened here to the day's first moment with
    ``DECAY_DATE_RESOLUTION_S`` recording how much later the event may lie.
    """
    try:
        day = _dt.date.fromisoformat(str(text).strip())
    except ValueError as exc:
        raise CatalogueParseError(
            f"{what} record {index}: {key!r} is {text!r}, which is not an ISO "
            "calendar date (YYYY-MM-DD)"
        ) from exc
    return ensure_utc(_dt.datetime.combine(day, _dt.time(), tzinfo=_dt.UTC), key)


def _omm_epoch(text: str, what: str, index: int) -> _dt.datetime:
    """Parse a CCSDS OMM ``EPOCH`` into an aware UTC instant.

    OMM declares its time system in the message; for GP element sets it is
    always UTC, and CelesTrak accordingly writes the epoch without an offset.
    An offset-bearing string is parsed by the shared temporal helper; a bare
    one is attached to UTC because the FORMAT specifies it, which is a
    different thing from assuming it.
    """
    candidate = str(text).strip()
    if candidate.endswith("Z") or re.search(r"[+-]\d{2}:?\d{2}$", candidate):
        return parse_aware_datetime(candidate, "EPOCH")
    try:
        naive = _dt.datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise CatalogueParseError(
            f"{what} record {index}: 'EPOCH' is {text!r}, which is not an ISO-8601 datetime"
        ) from exc
    return ensure_utc(naive.replace(tzinfo=_dt.UTC), "EPOCH")


def parse_satcat(payload: bytes) -> tuple[CatalogueObject, ...]:
    """Parse a CelesTrak SATCAT JSON payload into catalogue objects.

    Every record is parsed or the whole payload is rejected. Skipping a
    malformed record would silently shrink the launch's piece count, and the
    survival fraction of a launch is exactly a ratio of piece counts.
    """
    what = "SATCAT"
    objects: list[CatalogueObject] = []
    for index, record in enumerate(_records(payload, what)):
        decay_raw = record.get("DECAY_DATE")
        decay = (
            None
            if decay_raw is None or not str(decay_raw).strip()
            else _calendar_day_utc(str(decay_raw), what, "DECAY_DATE", index)
        )
        ops_raw = record.get("OPS_STATUS_CODE")
        objects.append(
            CatalogueObject(
                norad_id=_integer(record, "NORAD_CAT_ID", what, index),
                object_id=_text(record, "OBJECT_ID", what, index),
                object_name=_text(record, "OBJECT_NAME", what, index),
                launch_date=_calendar_day_utc(
                    _text(record, "LAUNCH_DATE", what, index), what, "LAUNCH_DATE", index
                ),
                decay_date=decay,
                period_min=_number(record, "PERIOD", what, index),
                inclination_deg=_number(record, "INCLINATION", what, index),
                apogee_km=_number(record, "APOGEE", what, index),
                perigee_km=_number(record, "PERIGEE", what, index),
                ops_status="" if ops_raw is None else str(ops_raw).strip(),
                object_type=str(record.get("OBJECT_TYPE") or "").strip(),
                owner=str(record.get("OWNER") or "").strip(),
            )
        )
    return tuple(objects)


def parse_gp(payload: bytes) -> tuple[ElementSet, ...]:
    """Parse a CelesTrak/Space-Track GP JSON payload into element sets.

    ``TLE_LINE1``/``TLE_LINE2`` are carried through when the source supplies
    them and left as ``None`` when it does not, so a caller can tell "this feed
    has no two-line text" from "this object has none".
    """
    what = "GP"
    sets: list[ElementSet] = []
    for index, record in enumerate(_records(payload, what)):
        sets.append(
            ElementSet(
                norad_id=_integer(record, "NORAD_CAT_ID", what, index),
                object_id=_text(record, "OBJECT_ID", what, index),
                epoch_utc=_omm_epoch(_text(record, "EPOCH", what, index), what, index),
                mean_motion=_number(record, "MEAN_MOTION", what, index),
                eccentricity=_number(record, "ECCENTRICITY", what, index),
                inclination_deg=_number(record, "INCLINATION", what, index),
                bstar=_number(record, "BSTAR", what, index),
                raan_deg=_number(record, "RA_OF_ASC_NODE", what, index),
                arg_perigee_deg=_number(record, "ARG_OF_PERICENTER", what, index),
                mean_anomaly_deg=_number(record, "MEAN_ANOMALY", what, index),
                mean_motion_dot=_number(record, "MEAN_MOTION_DOT", what, index),
                mean_motion_ddot=_number(record, "MEAN_MOTION_DDOT", what, index),
                tle_line1=_optional_line(record, "TLE_LINE1"),
                tle_line2=_optional_line(record, "TLE_LINE2"),
                object_name=str(record.get("OBJECT_NAME") or "").strip(),
                element_set_no=_optional_integer(record, "ELEMENT_SET_NO"),
                rev_at_epoch=_optional_integer(record, "REV_AT_EPOCH"),
                ephemeris_type=_optional_integer(record, "EPHEMERIS_TYPE") or 0,
                classification=str(record.get("CLASSIFICATION_TYPE") or "U").strip(),
            )
        )
    return tuple(sets)


# --------------------------------------------------------------------------
# partitions
# --------------------------------------------------------------------------


def survivors(objects: Iterable[CatalogueObject]) -> tuple[CatalogueObject, ...]:
    """Objects the catalogue has NOT recorded as decayed.

    Partitioned on ``decay_date`` rather than ``ops_status``: the status code is
    a curated label that lags, whereas the decay date is the recorded fact.
    """
    return tuple(obj for obj in objects if not obj.has_decayed)


def decayed(objects: Iterable[CatalogueObject]) -> tuple[CatalogueObject, ...]:
    """Objects the catalogue has recorded a decay date for."""
    return tuple(obj for obj in objects if obj.has_decayed)


def decay_epochs(objects: Iterable[CatalogueObject]) -> dict[str, _dt.datetime]:
    """Map international designator to recorded decay epoch.

    Only decayed objects appear. A survivor is absent rather than mapped to a
    placeholder, so ``in`` is a truthful test of "did this piece reenter?" and
    no caller can accidentally read a sentinel as a date.
    """
    epochs: dict[str, _dt.datetime] = {}
    for obj in objects:
        if obj.decay_date is None:
            continue
        if obj.object_id in epochs and epochs[obj.object_id] != obj.decay_date:
            raise CatalogueParseError(
                f"{obj.object_id} appears twice with different decay dates "
                f"({epochs[obj.object_id].date()} and {obj.decay_date.date()}); "
                "refusing to pick one"
            )
        epochs[obj.object_id] = obj.decay_date
    return epochs
