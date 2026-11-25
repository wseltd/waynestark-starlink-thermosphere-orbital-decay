"""NASA OMNI HRO ingestion: the lawful substitute for WDC Kyoto SYM-H/ASY-H/AE.

This module exists for a LICENCE reason before a technical one.

WDC Kyoto publishes Dst, SYM-H and AE, and its terms of use state plainly that
"The WDC Kyoto does not allow commercial applications of the geomagnetic
indices" (https://wdc.kugi.kyoto-u.ac.jp/wdc/Sec3.html). This product is
commercial, so a Kyoto endpoint is not a source it may use — not with
attribution, not cached, not "for validation only" inside a shipped result.

NASA/GSFC republishes SYM-H, ASY-H and AE inside the OMNI high-resolution
dataset as US-Government open data, alongside the solar-wind and IMF variables
that come from the same aggregation. OMNI is therefore the substitute, and this
module is the only route by which those indices enter the project.

The guarantee is structural rather than documentary: every URL built here is
passed through the SAME ``_guard_url`` implementation that
``space_weather`` uses — imported, not copied, so there is exactly one refusal
in the codebase and it cannot drift — and no function in this module accepts a
caller-supplied host, path or dataset id. There is no code path from here to
``wdc.kugi.kyoto-u.ac.jp``.

Registry note: ``planning/registry/resources.yaml`` records an open legal
question (``BLK-OMNI-KYOTO-FIELDS``) about whether the Kyoto-DERIVED fields
inside the NASA aggregation inherit NASA's terms in full. That question is
about provenance of the upstream indices, not about the Kyoto endpoints, which
are unconditionally excluded either way.

Transport note: CDAWeb HAPI reports request failures INSIDE the response body
with HTTP 200, so a rejected request is indistinguishable from a served one at
the transport layer. ``raise_for_hapi_status`` reads the status document out of
the body at the fetch boundary and refuses it there, before anything is stored
or parsed. See the comment above ``HAPI_OK_STATUS_CODES``.

Network access lives behind an injected fetcher, as everywhere else in
``drivers``, so the module is fully testable offline. The test suite blocks
sockets outright, so an accidental live call fails loudly.

Canonical tickets: P03_05_T011, P03_05_T013.
"""

from __future__ import annotations

import bisect
import datetime as _dt
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from ..domain.temporal import ensure_utc, parse_aware_datetime
from .raw_store import RawStore
from .space_weather import Fetcher, _guard_url

CDAWEB_HAPI_DATA_URL = "https://cdaweb.gsfc.nasa.gov/hapi/data"
OMNI_DATASET_ID = "OMNI_HRO_1MIN"
OMNI_LICENCE = "US Government open data (NASA OMNI)"
OMNI_LICENCE_URL = "https://omniweb.gsfc.nasa.gov/"
OMNI_CITATION = "NASA/GSFC OMNI 1-min HRO"

# The native cadence of OMNI_HRO_1MIN. Recorded as a constant so that a caller
# who wants hourly numbers has to resample DELIBERATELY, downstream, with the
# gaps visible. Resampling on load would silently interpolate across data gaps
# and turn "no measurement" into a plausible-looking value.
OMNI_CADENCE_S = 60.0

# The DATASET'S OWN declared order, of which this is a contiguous-in-meaning
# subset, read from https://cdaweb.gsfc.nasa.gov/hapi/info?id=OMNI_HRO_1MIN.
# It is a property of the server, not a preference of this module.
#
# CDAWeb HAPI requires a parameter subset to be listed in the order the dataset
# declares it, and refuses any other ordering with status 1411, "Parameter out
# of order". It sends that refusal as HTTP 200 with an error document in the
# body, so nothing raises at the transport layer.
#
# This tuple previously read SYM_H, ASY_H, AE_INDEX, BZ_GSM, flow_speed,
# proton_density — a request the server rejected on every live call, which no
# offline test could catch because the recorded fixture had been written in the
# same wrong order. ``scripts/fetch_event_context.py`` reconciled the two
# orderings after the fact. Asking in the served order removes the mismatch at
# its source: the served CSV columns and this tuple are now the same sequence,
# so ``parse_omni_csv``'s positional mapping is correct against the raw bytes
# and no reordering step stands between the wire and the parsed series.
OMNI_PARAMETERS: tuple[str, ...] = (
    "BZ_GSM",
    "flow_speed",
    "proton_density",
    "AE_INDEX",
    "SYM_H",
    "ASY_H",
)

# Written into every sidecar this module produces. A stored CSV whose column
# order is not recorded is unreadable evidence: a future reader has no way to
# tell which column is SYM-H, and guessing wrong relabels IMF Bz as a
# geomagnetic index without changing anything that looks suspicious.
OMNI_COLUMN_ORDER_NOTE = (
    "Columns are the CDAWeb HAPI served order "
    f"({','.join(OMNI_PARAMETERS)}) after the leading time column. That is the "
    "dataset's own declared order and the only order the server accepts, and it "
    "is also the order drivers/omni.py:OMNI_PARAMETERS requests and "
    "parse_omni_csv maps positionally, so the stored bytes map onto the parsed "
    "series column for column with no reordering step in between. The bytes "
    "here are verbatim as served."
)

# OMNI marks a missing measurement with a per-variable numeric SENTINEL rather
# than an empty field, and the sentinels differ between variables (see the OMNI
# HRO format documentation at https://omniweb.gsfc.nasa.gov/). Reading one as
# data is not a small error, it is a catastrophic one:
#
#   * SYM-H fill 99999 nT read as data turns the 3-4 February 2022 storm, whose
#     true SYM-H minimum is a few tens of nT negative, into a positive
#     excursion four orders of magnitude too large. A single fill in a mean
#     dominates every real sample in the window.
#   * flow_speed fill 99999.9 km/s is a third of the speed of light, and any
#     dynamic pressure or coupling function derived from it is nonsense.
#   * BZ_GSM fill 9999.99 nT would swamp the southward-Bz term that drives
#     every storm-coupling estimate in this project.
#
# So a fill becomes None here, at the parse boundary, and None means "OMNI
# recorded no measurement for this minute". It is never 0.0, never carried
# forward from the previous minute, and never interpolated.
FILL_VALUES: Mapping[str, tuple[float, ...]] = {
    "SYM_H": (99999.0, 99999.9),
    "ASY_H": (99999.0, 99999.9),
    "AE_INDEX": (99999.0, 99999.9),
    "BZ_GSM": (9999.99, 9999.9),
    "flow_speed": (99999.9, 99999.0),
    "proton_density": (999.99, 999.9),
}

# Sentinels are exact decimal literals in the source text, so this only absorbs
# float round-tripping, not a genuine measurement that happens to be nearby.
_FILL_ABS_TOLERANCE = 1e-6


def is_fill(parameter: str, value: float) -> bool:
    """True when ``value`` is OMNI's "no measurement" sentinel for ``parameter``.

    Keyed by parameter rather than tested against one global sentinel set,
    because the sentinels differ per variable and a global set would either
    miss one or reject a real reading in another column.
    """
    try:
        fills = FILL_VALUES[parameter]
    except KeyError:
        raise KeyError(
            f"no OMNI fill value is documented for {parameter!r}; refusing to "
            "guess. Add it to FILL_VALUES from the OMNI HRO format "
            "documentation before ingesting the column."
        ) from None
    return any(
        math.isclose(value, fill, rel_tol=0.0, abs_tol=_FILL_ABS_TOLERANCE) for fill in fills
    )


@dataclass(frozen=True, slots=True)
class OmniSeries:
    """OMNI 1-minute geomagnetic indices and solar-wind state, never resampled.

    Every value column is ``float | None``. ``None`` is the only representation
    of a missing measurement: the OMNI fill sentinels are converted at parse
    time and never survive into this object, because a fill read as data is a
    silent, order-of-magnitude corruption of a storm result (see
    ``FILL_VALUES``).

    Timestamps are timezone-aware and strictly increasing. The ordering is
    validated rather than assumed, because ``storm_onset`` and ``min_sym_h``
    both read the series in order and would answer confidently and wrongly on
    an unsorted one.
    """

    times_utc: tuple[_dt.datetime, ...]
    sym_h_nt: tuple[float | None, ...]
    asy_h_nt: tuple[float | None, ...]
    ae_nt: tuple[float | None, ...]
    bz_gsm_nt: tuple[float | None, ...]
    flow_speed_km_s: tuple[float | None, ...]
    proton_density_cm3: tuple[float | None, ...]
    source: str = "NASA OMNI"
    licence: str = OMNI_LICENCE
    citation: str = OMNI_CITATION

    def __post_init__(self) -> None:
        n = len(self.times_utc)
        lengths = {
            "sym_h_nt": len(self.sym_h_nt),
            "asy_h_nt": len(self.asy_h_nt),
            "ae_nt": len(self.ae_nt),
            "bz_gsm_nt": len(self.bz_gsm_nt),
            "flow_speed_km_s": len(self.flow_speed_km_s),
            "proton_density_cm3": len(self.proton_density_cm3),
        }
        ragged = {name: length for name, length in lengths.items() if length != n}
        if ragged:
            raise ValueError(
                f"OmniSeries fields must all have the same length as times_utc "
                f"({n}); these disagree: {ragged}"
            )
        if n == 0:
            raise ValueError("OmniSeries is empty")
        for t in self.times_utc:
            if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
                raise ValueError("OmniSeries times must be timezone-aware")
        for earlier, later in zip(self.times_utc, self.times_utc[1:], strict=False):
            if later <= earlier:
                raise ValueError(
                    f"OmniSeries times must be strictly increasing; "
                    f"{later.isoformat()} does not follow {earlier.isoformat()}"
                )

    def sym_h_at(self, when: _dt.datetime) -> float | None:
        """SYM-H for the 1-minute sample containing ``when``.

        ``None`` means OMNI recorded no SYM-H for that minute. That is a
        different statement from 0 nT — a quiet magnetosphere — and callers
        must handle it rather than coerce it.
        """
        return self.sym_h_nt[self._index(when)]

    def _index(self, when: _dt.datetime) -> int:
        """The sample covering ``when``, refusing to read outside the series.

        Every lookup funnels through here so the aware-datetime rule is
        enforced once. ``.astimezone()`` on a naive datetime applies the SYSTEM
        timezone, which under US Eastern would select a sample five hours from
        the intended one — longer than the storm onset this project studies.
        """
        moment = ensure_utc(when, "when")
        if moment < self.times_utc[0]:
            raise ValueError(
                f"{moment.isoformat()} precedes the loaded OMNI series "
                f"(starts {self.times_utc[0].isoformat()})"
            )
        last = self.times_utc[-1] + _dt.timedelta(seconds=OMNI_CADENCE_S)
        if moment >= last:
            raise ValueError(
                f"{moment.isoformat()} is past the loaded OMNI series "
                f"(ends {last.isoformat()}). Extending would fabricate drivers."
            )
        return bisect.bisect_right(self.times_utc, moment) - 1

    def min_sym_h(self, start: _dt.datetime, end: _dt.datetime) -> tuple[_dt.datetime, float]:
        """The most negative observed SYM-H in ``[start, end]``, with its time.

        The minimum, not the extremum by magnitude: a geomagnetic storm is a
        NEGATIVE SYM-H excursion, and its depth is the number every intensity
        classification is stated against.

        Missing minutes are skipped, never treated as 0 nT. A window in which
        every minute is missing raises instead of returning a fabricated
        minimum. Ties resolve to the earliest sample.
        """
        window_start = ensure_utc(start, "start")
        window_end = ensure_utc(end, "end")
        if window_end < window_start:
            raise ValueError(
                f"min_sym_h window ends ({window_end.isoformat()}) before it "
                f"starts ({window_start.isoformat()})"
            )
        observed = [
            (t, v)
            for t, v in zip(self.times_utc, self.sym_h_nt, strict=True)
            if v is not None and window_start <= t <= window_end
        ]
        if not observed:
            raise ValueError(
                f"no observed SYM-H between {window_start.isoformat()} and "
                f"{window_end.isoformat()}: the window is either outside the "
                "loaded series or entirely fill. Refusing to report a minimum "
                "of nothing."
            )
        return min(observed, key=lambda row: row[1])

    def storm_onset(self, threshold_nt: float = -50.0) -> _dt.datetime | None:
        """The first instant SYM-H crosses below ``threshold_nt``, or ``None``.

        A crossing requires the previous OBSERVED sample to have been at or
        above the threshold, so a run of missing minutes cannot manufacture an
        onset and a sample that merely stays below one does not re-trigger it.

        If the series already opens below the threshold the crossing happened
        BEFORE the loaded window, and this raises rather than returning the
        first timestamp: reporting the edge of a fetch window as a storm onset
        would be an artefact of where the download started, presented as a
        measured time.

        ``None`` means SYM-H never went below the threshold anywhere in the
        loaded series — no storm by this criterion, stated as such.
        """
        previous: float | None = None
        for when, value in zip(self.times_utc, self.sym_h_nt, strict=True):
            if value is None:
                continue
            if value < threshold_nt:
                if previous is None:
                    raise ValueError(
                        f"SYM-H is already {value} nT at the first observed sample "
                        f"({when.isoformat()}), below the {threshold_nt} nT threshold: "
                        "the crossing precedes the loaded series. Fetch an earlier "
                        "window rather than reading its first timestamp as an onset."
                    )
                return when
            previous = value
        return None


# --- HAPI status documents -------------------------------------------------
#
# HAPI reports request-level failures INSIDE the response body, and CDAWeb sends
# them with HTTP 200, so ``urllib`` raises nothing and the bytes arrive looking
# like a successful fetch. That is not hypothetical. Asking for this dataset's
# parameters in the wrong order returned, with HTTP 200:
#
#     ,
#     "status": {"code": 1411, "message": "HAPI error 1411: Parameter out of order"}
#     }
#
# and the client failed only later, in the parser, complaining about a column
# count — a message that named neither the real cause nor the server's own
# explanation of it.
#
# So the detection below sits at the FETCH boundary, and it is generic over HAPI
# status codes rather than special-cased to 1411, because the same
# 200-with-an-error-document shape carries every other HAPI failure: an unknown
# dataset id, an unknown parameter, an unparseable time bound. Each would
# otherwise reach the parser disguised as a malformed data response.
#
# The scan covers the WHOLE payload rather than a prefix. CDAWeb streams CSV, so
# a failure part-way through a long window arrives as valid rows followed by a
# status object; reading only the head would accept those rows and return a
# series silently truncated at an arbitrary minute.

# HAPI's success statuses: 1200 is OK, 1201 is OK with no data in the requested
# range. Everything else is a failure — including a code this module has never
# seen, because waving an unrecognised status through means parsing a document
# whose meaning is unknown.
HAPI_OK_STATUS_CODES: frozenset[int] = frozenset({1200, 1201})

_HAPI_STATUS_OBJECT = re.compile(r'"status"\s*:\s*\{(?P<body>[^{}]*)\}')
_HAPI_STATUS_CODE = re.compile(r'"code"\s*:\s*(?P<code>-?\d+)')
_HAPI_STATUS_MESSAGE = re.compile(r'"message"\s*:\s*"(?P<message>(?:[^"\\]|\\.)*)"')

# Reported when a status object carries a message but no code. Negative, so it
# can never collide with a real HAPI code and can never be mistaken for one.
HAPI_UNSTATED_CODE = -1


class HapiStatusError(ValueError):
    """A HAPI response whose body declares a status that is not success.

    Subclasses ``ValueError`` so a caller already catching the broad case still
    catches this, while ``code`` and ``message`` let a fetch log record what the
    server actually said rather than re-deriving it from the formatted text.
    """

    def __init__(self, code: int, message: str, *, url: str = "") -> None:
        self.code = code
        self.message = message
        self.url = url
        location = f" to {url}" if url else ""
        super().__init__(
            f"CDAWeb HAPI answered the request{location} with HTTP 200 carrying an "
            f"error document rather than data: HAPI status {code}: "
            f"{message or '(the document carried no message)'}. Nothing was stored "
            "or parsed. A 200 that is really an error must not pass silently."
        )


def hapi_status(payload: bytes) -> tuple[int, str] | None:
    """The status code and message a HAPI response declares, if it declares one.

    ``None`` means the payload carries no status object, which is the normal
    shape of a CSV data response requested without a header block.

    Undecodable bytes are reported as carrying no status rather than raising:
    this runs ahead of the parser, and the parser has its own, more specific
    complaint about a payload that is not UTF-8.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    found = _HAPI_STATUS_OBJECT.search(text)
    if found is None:
        return None
    body = found.group("body")
    code_match = _HAPI_STATUS_CODE.search(body)
    message_match = _HAPI_STATUS_MESSAGE.search(body)
    message = "" if message_match is None else message_match.group("message")
    if code_match is None:
        # A status object with no code is itself a broken response. It is
        # reported under a sentinel code rather than ignored, so the caller
        # still refuses it: ignoring it would hand the parser a document that
        # is, whatever else it is, not data.
        return (HAPI_UNSTATED_CODE, message or "the HAPI status object carried no code")
    return (int(code_match.group("code")), message)


def raise_for_hapi_status(payload: bytes, *, url: str = "") -> None:
    """Raise ``HapiStatusError`` when ``payload`` is a HAPI error document.

    Returns quietly for a plain data response and for the two HAPI success
    statuses, so a response that includes a header block is not refused merely
    for saying it succeeded.
    """
    status = hapi_status(payload)
    if status is None:
        return
    code, message = status
    if code in HAPI_OK_STATUS_CODES:
        return
    raise HapiStatusError(code, message, url=url)


def build_omni_hapi_url(start: _dt.datetime, end: _dt.datetime) -> str:
    """Build the CDAWeb HAPI request for the fixed OMNI parameter set.

    The dataset id, the parameter list and the host are all literals here.
    Nothing a caller supplies reaches the URL except the time window, which is
    formatted from validated aware datetimes, so the request cannot be steered
    at a different — and possibly non-commercial — source.

    The parameters are listed in ``OMNI_PARAMETERS``, which is the order the
    dataset itself declares. HAPI rejects any other ordering, and rejects it
    with HTTP 200, so getting this wrong does not fail at the fetch.
    """
    window_start = ensure_utc(start, "start")
    window_end = ensure_utc(end, "end")
    if window_end <= window_start:
        raise ValueError(
            f"OMNI window ends ({window_end.isoformat()}) at or before it starts "
            f"({window_start.isoformat()}); an empty window would store an empty "
            "asset under a name that claims to cover a period"
        )
    return (
        f"{CDAWEB_HAPI_DATA_URL}?id={OMNI_DATASET_ID}"
        f"&time.min={window_start:%Y-%m-%dT%H:%M:%SZ}"
        f"&time.max={window_end:%Y-%m-%dT%H:%M:%SZ}"
        f"&parameters={','.join(OMNI_PARAMETERS)}"
        f"&format=csv"
    )


def fetch_omni_hro(
    start: _dt.datetime,
    end: _dt.datetime,
    *,
    fetcher: Fetcher,
    store: RawStore | None = None,
    asset_id: str | None = None,
) -> OmniSeries:
    """Fetch OMNI 1-minute HRO data, storing the raw CSV verbatim.

    The stored sidecar carries NASA's open-data licence and citation, so a
    result regenerated years later can still show which terms the SYM-H behind
    it was obtained under — which is the whole reason this source replaced WDC
    Kyoto.

    An error document is refused BEFORE anything is stored. HAPI sends its
    failures with HTTP 200, so without this check a status object would be
    written into the raw store wearing a sidecar that claims NASA OMNI data
    under NASA's licence, and the only symptom would be a confusing complaint
    from the parser about column counts. Nothing is stored for a request the
    server did not serve.
    """
    window_start = ensure_utc(start, "start")
    window_end = ensure_utc(end, "end")
    url = build_omni_hapi_url(window_start, window_end)
    payload = fetcher(_guard_url(url))
    raise_for_hapi_status(payload, url=url)
    if store is not None:
        store.write(
            source="omni",
            asset_id=asset_id
            or f"omni_hro_1min_{window_start:%Y%m%dT%H%M%SZ}_{window_end:%Y%m%dT%H%M%SZ}",
            payload=payload,
            url=url,
            licence=OMNI_LICENCE,
            licence_url=OMNI_LICENCE_URL,
            citation=OMNI_CITATION,
            content_type="text/csv",
            ext=".csv",
            notes=OMNI_COLUMN_ORDER_NOTE,
        )
    return parse_omni_csv(payload)


def _parse_value(parameter: str, raw: str, line_number: int) -> float | None:
    """One CSV cell: a number, or ``None`` for an empty or fill-valued cell."""
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"OMNI HAPI CSV line {line_number}: {parameter} is {raw!r}, which is "
            "not a number"
        ) from exc
    return None if is_fill(parameter, value) else value


def parse_omni_csv(payload: bytes) -> OmniSeries:
    """Parse a HAPI CSV response into an ``OmniSeries`` at its native cadence.

    HAPI CSV is headerless by default: the first column is the ISO-8601 time
    and the remaining columns follow the requested parameter order. Rows are
    kept exactly as served — no resampling, no gap filling, no reordering — so
    the 1-minute cadence and every data gap in it survive ingestion.

    A row whose column count does not match the request is rejected rather than
    padded: a short row means the response is not the one that was asked for,
    and guessing which column is missing would silently relabel SYM-H as ASY-H.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "OMNI HAPI payload is not UTF-8 text. HAPI answers a bad request "
            "with a JSON status document; check the stored raw asset before "
            "re-parsing."
        ) from exc

    expected_fields = 1 + len(OMNI_PARAMETERS)
    times: list[_dt.datetime] = []
    columns: dict[str, list[float | None]] = {name: [] for name in OMNI_PARAMETERS}

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        # HAPI emits an optional commented header block when include=header is
        # requested; skip it rather than fail, but never treat it as data.
        if not stripped or stripped.startswith("#"):
            continue
        fields = [field.strip() for field in stripped.split(",")]
        if len(fields) != expected_fields:
            raise ValueError(
                f"OMNI HAPI CSV line {line_number} has {len(fields)} fields, "
                f"expected {expected_fields} (time plus "
                f"{', '.join(OMNI_PARAMETERS)}). The response does not match the "
                "request; refusing to guess which column is which."
            )
        times.append(parse_aware_datetime(fields[0], f"OMNI HAPI CSV line {line_number} time"))
        for parameter, raw in zip(OMNI_PARAMETERS, fields[1:], strict=True):
            columns[parameter].append(_parse_value(parameter, raw, line_number))

    if not times:
        raise ValueError(
            "OMNI HAPI CSV payload contains no data rows. An empty response is "
            "not an empty magnetosphere; check the window and the stored asset."
        )

    return OmniSeries(
        times_utc=tuple(times),
        sym_h_nt=tuple(columns["SYM_H"]),
        asy_h_nt=tuple(columns["ASY_H"]),
        ae_nt=tuple(columns["AE_INDEX"]),
        bz_gsm_nt=tuple(columns["BZ_GSM"]),
        flow_speed_km_s=tuple(columns["flow_speed"]),
        proton_density_cm3=tuple(columns["proton_density"]),
    )
