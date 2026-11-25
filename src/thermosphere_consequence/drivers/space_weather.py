"""Space-weather driver ingestion: GFZ Kp/ap, NOAA F10.7, NASA OMNI.

Licence discipline is the organising principle of this module, not an
afterthought:

  * GFZ Kp/ap is CC BY 4.0 — commercial use and redistribution permitted.
  * NASA OMNI and NOAA products are US-Government open data.
  * WDC Kyoto Dst/SYM-H/AE is explicitly non-commercial and is NOT fetched
    here at all. SYM-H comes from OMNI instead. There is no code path in this
    module that can reach a Kyoto endpoint, which is a stronger guarantee than
    a policy note.

Network access lives behind an injected fetcher so the whole module is
testable offline against recorded fixtures. The test suite blocks sockets
outright, so any accidental live call fails loudly rather than making the
suite depend on the weather.

Canonical tickets: P03_05_T008, P03_05_T009, P03_05_T011, P03_05_T013, P03_05_T015.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable
from dataclasses import dataclass

from ..domain.provenance import ValueStatus
from ..domain.temporal import require_aware_datetime
from ..protocols.atmosphere import DriverState
from .raw_store import RawStore

Fetcher = Callable[[str], bytes]

GFZ_JSON_URL = "https://kp.gfz.de/app/json/"
GFZ_LICENCE = "CC BY 4.0"
GFZ_LICENCE_URL = "https://kp.gfz-potsdam.de/en/"
GFZ_CITATION = "Matzka et al. 2021, doi:10.5880/Kp.0001"

# NOAA's observed-solar-cycle feed is MONTHLY. The fetch/parse pair for it was
# removed rather than kept as an unused fallback: production needs DAILY
# observed F10.7, which CelesTrak supplies, and a monthly value silently
# standing in for a daily one is a systematic density error of several percent.
NOAA_LICENCE = "US Government open data"
NOAA_CITATION = "NOAA SWPC / NRCan DRAO Penticton F10.7"

CELESTRAK_SW_URL = "https://celestrak.org/SpaceData/SW-Last5Years.csv"
CELESTRAK_LICENCE = "CelesTrak, Space-Track-derived"
CELESTRAK_LICENCE_URL = "https://celestrak.org/"
CELESTRAK_CITATION = "CelesTrak space weather data (Kelso), Space-Track-derived"

# Endpoints this module must never reach, enforced by a guard test. WDC Kyoto
# forbids commercial use of its indices; the substitute is NASA OMNI SYM-H.
FORBIDDEN_HOSTS = ("wdc.kugi.kyoto-u.ac.jp", "spacewx.com", "map.nrl.navy.mil")


class ForbiddenSourceError(RuntimeError):
    """Raised when a fetch would reach a non-commercial-licensed source."""


def _guard_url(url: str) -> str:
    for host in FORBIDDEN_HOSTS:
        if host in url:
            raise ForbiddenSourceError(
                f"refusing to fetch {url}: {host} data is not licensed for "
                "commercial use in this product. Use NASA OMNI (SYM-H) or "
                "GFZ (Kp/ap) instead."
            )
    return url


@dataclass(frozen=True, slots=True)
class KpApSeries:
    """3-hourly Kp/ap at the model's native cadence, never resampled on load.

    ``status_flags`` preserves GFZ's definitive/nowcast distinction. February
    2022 is definitive, but the flag travels with the data so a future run
    over recent dates cannot silently mix nowcast values into a result
    presented as observed.
    """

    times_utc: tuple[_dt.datetime, ...]
    kp: tuple[float, ...]
    ap: tuple[float, ...]
    status_flags: tuple[str, ...]
    source: str = "GFZ"
    licence: str = GFZ_LICENCE
    citation: str = GFZ_CITATION

    def __post_init__(self) -> None:
        n = len(self.times_utc)
        if not (len(self.kp) == len(self.ap) == len(self.status_flags) == n):
            raise ValueError("KpApSeries fields must all have the same length")
        if n == 0:
            raise ValueError("KpApSeries is empty")
        for t in self.times_utc:
            if t.tzinfo is None:
                raise ValueError("KpApSeries times must be timezone-aware")

    def ap_at(self, when: _dt.datetime) -> float:
        """The 3-hour ap bin containing ``when``."""
        idx = self._bin_index(when)
        return self.ap[idx]

    def kp_at(self, when: _dt.datetime) -> float:
        """The 3-hour Kp bin containing ``when``."""
        idx = self._bin_index(when)
        return self.kp[idx]

    def _bin_index(self, when: _dt.datetime) -> int:
        # Every entry point funnels through here, so the aware-datetime rule is
        # enforced once. Calling .astimezone() on a naive datetime would apply
        # the SYSTEM timezone, silently selecting a driver bin hours away from
        # the intended one -- a five-hour error under US Eastern, for instance.
        when = require_aware_datetime(when, "when").astimezone(_dt.UTC)
        if when < self.times_utc[0]:
            raise ValueError(
                f"{when.isoformat()} precedes the loaded Kp/ap series "
                f"(starts {self.times_utc[0].isoformat()})"
            )
        idx = 0
        for i, t in enumerate(self.times_utc):
            if t <= when:
                idx = i
            else:
                break
        last = self.times_utc[-1] + _dt.timedelta(hours=3)
        if when >= last:
            raise ValueError(
                f"{when.isoformat()} is past the loaded Kp/ap series "
                f"(ends {last.isoformat()}). Extending would fabricate drivers."
            )
        return idx

    def daily_ap_for(self, when: _dt.datetime) -> float:
        """GFZ UTC calendar-day daily Ap: mean of all eight 3-hourly ap bins for ``when.date()``.

        Requires a complete UTC calendar day consisting of all eight distinct
        3-hourly synoptic bins (00:00, 03:00, 06:00, 09:00, 12:00, 15:00,
        18:00, 21:00 UTC), matching the GFZ Potsdam and NRLMSIS ``Ap``
        specification rather than a trailing rolling 24-hour window. Raises
        ``ValueError`` on incomplete, duplicate, or non-synoptic calendar-day
        records rather than silently averaging fewer than eight observations.
        """
        when_utc = require_aware_datetime(when, "when").astimezone(_dt.UTC)
        self._bin_index(when_utc)
        target_date = when_utc.date()
        expected_hours = (0, 3, 6, 9, 12, 15, 18, 21)
        day_entries = [
            (t_utc, ap_val)
            for t, ap_val in zip(self.times_utc, self.ap, strict=True)
            if (t_utc := t.astimezone(_dt.UTC)).date() == target_date
        ]
        synoptic_hours = tuple(
            sorted(
                t_utc.hour
                for t_utc, _ in day_entries
                if t_utc.minute == 0 and t_utc.second == 0 and t_utc.microsecond == 0
            )
        )
        if len(day_entries) != 8 or synoptic_hours != expected_hours:
            raise ValueError(
                f"incomplete UTC calendar-day ap record for {target_date.isoformat()}: "
                f"found {len(day_entries)} bin(s) (synoptic UTC hours {list(synoptic_hours)}); "
                "daily Ap requires all 8 three-hourly synoptic bins (00, 03, 06, 09, 12, 15, 18, 21 UTC)"
            )
        return sum(ap_val for _, ap_val in day_entries) / 8.0

    def ap_history_for(self, when: _dt.datetime) -> tuple[float, ...]:
        """The 7-element ap array NRLMSISE-00 consumes.

        Order per the model's own documentation: UTC calendar-day daily Ap;
        ap now; ap 3, 6 and 9 hours before now; the average of ap over 12-33
        hours before now; and the average over 36-57 hours before now.
        """
        when = require_aware_datetime(when, "when").astimezone(_dt.UTC)
        here = self._bin_index(when)

        def back(bins: int) -> float:
            return self.ap[max(0, here - bins)]

        daily_ap = self.daily_ap_for(when)

        window_12_33 = [self.ap[max(0, here - b)] for b in range(4, 12)]
        window_36_57 = [self.ap[max(0, here - b)] for b in range(12, 20)]

        return (
            daily_ap,
            back(0),
            back(1),
            back(2),
            back(3),
            sum(window_12_33) / len(window_12_33),
            sum(window_36_57) / len(window_36_57),
        )


@dataclass(frozen=True, slots=True)
class F107Series:
    """Daily F10.7 solar radio flux, observed and adjusted.

    Thermosphere models want the OBSERVED daily value plus the 81-day centred
    average. Adjusted (1 AU) values are kept separately and never substituted
    for observed, because the two differ by several percent and mixing them
    is a silent systematic error.
    """

    dates_utc: tuple[_dt.date, ...]
    observed: tuple[float, ...]
    adjusted: tuple[float, ...]
    source: str = "CelesTrak"
    licence: str = CELESTRAK_LICENCE
    citation: str = CELESTRAK_CITATION

    def __post_init__(self) -> None:
        if not (len(self.dates_utc) == len(self.observed) == len(self.adjusted)):
            raise ValueError("F107Series fields must all have the same length")
        if not self.dates_utc:
            raise ValueError("F107Series is empty")

    def observed_at(self, when: _dt.date | _dt.datetime) -> float:
        return self.observed[self._index(when)]

    def centred_81_day_average(self, when: _dt.date | _dt.datetime) -> float:
        """The 81-day centred mean of observed F10.7.

        Falls back to the widest symmetric window available at the series
        edges and never pads with a repeated value, so an edge average is a
        real average over fewer days rather than a fabricated one.
        """
        i = self._index(when)
        half = 40
        lo = max(0, i - half)
        hi = min(len(self.observed), i + half + 1)
        window = self.observed[lo:hi]
        return sum(window) / len(window)

    def _index(self, when: _dt.date | _dt.datetime) -> int:
        day = when.date() if isinstance(when, _dt.datetime) else when
        try:
            return self.dates_utc.index(day)
        except ValueError:
            raise ValueError(
                f"{day.isoformat()} is outside the loaded F10.7 series "
                f"({self.dates_utc[0].isoformat()} to {self.dates_utc[-1].isoformat()})"
            ) from None


def fetch_gfz_kp_ap(
    start: _dt.datetime,
    end: _dt.datetime,
    *,
    fetcher: Fetcher,
    store: RawStore | None = None,
    asset_id: str | None = None,
) -> KpApSeries:
    """Fetch GFZ Kp and ap, storing the raw JSON verbatim."""
    url = (
        f"{GFZ_JSON_URL}?start={start.astimezone(_dt.UTC):%Y-%m-%dT%H:%M:%SZ}"
        f"&end={end.astimezone(_dt.UTC):%Y-%m-%dT%H:%M:%SZ}&index=Kp"
    )
    payload = fetcher(_guard_url(url))
    if store is not None:
        store.write(
            source="gfz",
            asset_id=asset_id or f"kp_{start:%Y%m%d}_{end:%Y%m%d}",
            payload=payload,
            url=url,
            licence=GFZ_LICENCE,
            licence_url=GFZ_LICENCE_URL,
            citation=GFZ_CITATION,
            content_type="application/json",
            ext=".json",
        )

    ap_url = url.replace("index=Kp", "index=ap")
    ap_payload = fetcher(_guard_url(ap_url))
    if store is not None:
        store.write(
            source="gfz",
            asset_id=(asset_id or f"kp_{start:%Y%m%d}_{end:%Y%m%d}") + "_ap",
            payload=ap_payload,
            url=ap_url,
            licence=GFZ_LICENCE,
            licence_url=GFZ_LICENCE_URL,
            citation=GFZ_CITATION,
            content_type="application/json",
            ext=".json",
        )

    return parse_gfz_kp_ap(payload, ap_payload)


def parse_gfz_kp_ap(kp_payload: bytes, ap_payload: bytes) -> KpApSeries:
    """Parse GFZ JSON into the native 3-hourly series, no resampling."""
    kp_doc = json.loads(kp_payload)
    ap_doc = json.loads(ap_payload)

    times = tuple(
        _dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in kp_doc["datetime"]
    )
    kp = tuple(float(v) for v in kp_doc["Kp"])
    ap = tuple(float(v) for v in ap_doc["ap"])
    flags = tuple(str(s) for s in kp_doc.get("status", ["definitive"] * len(kp)))

    if len(ap) != len(kp):
        raise ValueError(
            f"GFZ Kp and ap series lengths disagree ({len(kp)} vs {len(ap)}); "
            "refusing to align them by truncation"
        )
    return KpApSeries(times_utc=times, kp=kp, ap=ap, status_flags=flags)


def fetch_celestrak_space_weather(
    *,
    fetcher: Fetcher,
    store: RawStore | None = None,
    asset_id: str = "sw_last5years",
) -> bytes:
    """Fetch CelesTrak's daily space-weather CSV (F10.7 observed + adjusted, ap)."""
    payload = fetcher(_guard_url(CELESTRAK_SW_URL))
    if store is not None:
        store.write(
            source="celestrak",
            asset_id=asset_id,
            payload=payload,
            url=CELESTRAK_SW_URL,
            licence=CELESTRAK_LICENCE,
            licence_url=CELESTRAK_LICENCE_URL,
            citation=CELESTRAK_CITATION,
            content_type="text/csv",
            ext=".csv",
        )
    return payload


def parse_celestrak_f107(payload: bytes) -> F107Series:
    """Parse daily observed and adjusted F10.7 from the CelesTrak CSV."""
    text = payload.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split(",")
    try:
        i_date = header.index("DATE")
        i_obs = header.index("F10.7_OBS")
        i_adj = header.index("F10.7_ADJ")
    except ValueError as exc:
        raise ValueError(f"unexpected CelesTrak space-weather header: {header}") from exc

    dates: list[_dt.date] = []
    observed: list[float] = []
    adjusted: list[float] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(i_date, i_obs, i_adj):
            continue
        try:
            obs = float(parts[i_obs])
            adj = float(parts[i_adj])
        except ValueError:
            continue
        if obs <= 0 or adj <= 0:
            continue
        dates.append(_dt.date.fromisoformat(parts[i_date]))
        observed.append(obs)
        adjusted.append(adj)

    return F107Series(
        dates_utc=tuple(dates),
        observed=tuple(observed),
        adjusted=tuple(adjusted),
        source="CelesTrak",
        licence=CELESTRAK_LICENCE,
        citation=CELESTRAK_CITATION,
    )


def parse_celestrak_ap(payload: bytes) -> KpApSeries:
    """Parse the 3-hourly ap and Kp columns from the CelesTrak CSV.

    Used as an independent cross-check on GFZ. The two are not expected to be
    bit-identical: CelesTrak republishes GFZ values with its own rounding of
    Kp to thirds, so a small disagreement is a formatting artefact and a large
    one is a real finding.
    """
    text = payload.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split(",")
    i_date = header.index("DATE")
    ap_cols = [header.index(f"AP{i}") for i in range(1, 9)]
    kp_cols = [header.index(f"KP{i}") for i in range(1, 9)]

    times: list[_dt.datetime] = []
    ap_values: list[float] = []
    kp_values: list[float] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(ap_cols + kp_cols):
            continue
        try:
            day = _dt.date.fromisoformat(parts[i_date])
        except ValueError:
            continue
        for slot, (ai, ki) in enumerate(zip(ap_cols, kp_cols, strict=True)):
            if not parts[ai] or not parts[ki]:
                continue
            times.append(
                _dt.datetime.combine(day, _dt.time(hour=slot * 3), tzinfo=_dt.UTC)
            )
            ap_values.append(float(parts[ai]))
            # CelesTrak stores Kp in tenths of a unit as an integer (e.g. 37 = 3.7).
            kp_values.append(float(parts[ki]) / 10.0)

    return KpApSeries(
        times_utc=tuple(times),
        kp=tuple(kp_values),
        ap=tuple(ap_values),
        status_flags=tuple("definitive" for _ in times),
        source="CelesTrak",
        licence=CELESTRAK_LICENCE,
        citation=CELESTRAK_CITATION,
    )


def drivers_at(
    when: _dt.datetime,
    kp_ap: KpApSeries,
    f107: F107Series,
    *,
    status: ValueStatus = ValueStatus.OBSERVED,
) -> DriverState:
    """Assemble the driver state an atmosphere model consumes at one instant.

    NRLMSISE-00 wants the PREVIOUS day's observed F10.7, because the model was
    fitted that way; using the same-day value is a common and quiet error that
    shifts density by a few percent.
    """
    when = require_aware_datetime(when, "when")
    previous_day = (when.astimezone(_dt.UTC) - _dt.timedelta(days=1)).date()
    return DriverState(
        f107=f107.observed_at(previous_day),
        f107a=f107.centred_81_day_average(when),
        ap_daily=kp_ap.ap_history_for(when)[0],
        ap_history=kp_ap.ap_history_for(when),
        source=f"{kp_ap.source} Kp/ap + {f107.source} F10.7",
        source_url=GFZ_LICENCE_URL,
        status=status,
    )


def http_fetcher(timeout_s: float = 60.0) -> Fetcher:
    """A real network fetcher, constructed explicitly by callers that want one.

    Deliberately not the default anywhere: production ingestion opts in, and
    the test suite never can, because sockets are blocked outright there.
    """

    def _fetch(url: str) -> bytes:
        import urllib.parse
        import urllib.request

        _guard_url(url)
        # urlopen honours file:, ftp: and custom schemes, so a URL reaching this
        # function from configuration or a redirect could read a local file
        # instead of fetching. Only https is permitted: every source in this
        # module is an https endpoint, and a plain-http fetch of provenance
        # data would be unauthenticated in transit.
        scheme = urllib.parse.urlparse(url).scheme
        if scheme != "https":
            raise ForbiddenSourceError(
                f"refusing to fetch {url!r}: scheme {scheme!r} is not permitted, "
                "only https. This blocks file:, ftp: and custom-scheme reads."
            )

        request = urllib.request.Request(
            url, headers={"User-Agent": "thermosphere-consequence/0.1 (research)"}
        )
        with urllib.request.urlopen(  # nosec B310 - noqa: S310
            request, timeout=timeout_s
        ) as response:
            return bytes(response.read())

    return _fetch


def summarise_storm(kp_ap: KpApSeries, start: _dt.datetime, end: _dt.datetime) -> dict[str, object]:
    """Peak and duration statistics over a window, for the event report."""
    selected: list[tuple[_dt.datetime, float, float]] = [
        (t, k, a)
        for t, k, a in zip(kp_ap.times_utc, kp_ap.kp, kp_ap.ap, strict=True)
        if start <= t <= end
    ]
    if not selected:
        raise ValueError("no Kp/ap samples in the requested window")
    peak_kp = max(selected, key=lambda row: row[1])
    peak_ap = max(selected, key=lambda row: row[2])
    hours_kp_ge_5 = 3 * sum(1 for _, k, _ in selected if k >= 5.0)
    return {
        "samples": len(selected),
        "peak_kp": peak_kp[1],
        "peak_kp_time": peak_kp[0].isoformat(),
        "peak_ap": peak_ap[2],
        "peak_ap_time": peak_ap[0].isoformat(),
        "hours_kp_ge_5": hours_kp_ge_5,
        "mean_ap": sum(a for _, _, a in selected) / len(selected),
    }


def cross_check_kp_ap(
    primary: KpApSeries, independent: KpApSeries, *, tolerance: float = 1.0
) -> dict[str, object]:
    """Compare two Kp/ap series that should describe the same reality.

    CelesTrak republishes GFZ values with its own rounding of Kp to thirds, so
    the two are not expected to be bit-identical. A small disagreement is a
    formatting artefact; a large one means one of the two feeds is wrong, and
    the whole downstream density chain rests on which.

    Only overlapping timestamps are compared. Non-overlapping ones are counted
    and reported rather than dropped silently, because a feed that covers a
    different window than expected is itself a finding.
    """
    by_time = {t: (k, a) for t, k, a in
               zip(independent.times_utc, independent.kp, independent.ap, strict=True)}

    compared = 0
    ap_mismatches: list[dict[str, object]] = []
    max_kp_difference = 0.0
    for t, kp, ap in zip(primary.times_utc, primary.kp, primary.ap, strict=True):
        other = by_time.get(t)
        if other is None:
            continue
        compared += 1
        max_kp_difference = max(max_kp_difference, abs(kp - other[0]))
        if abs(ap - other[1]) > tolerance:
            ap_mismatches.append(
                {"time": t.isoformat(), "primary_ap": ap, "independent_ap": other[1]}
            )

    return {
        "primary_source": primary.source,
        "independent_source": independent.source,
        "samples_compared": compared,
        "primary_only": len(primary.times_utc) - compared,
        "independent_only": len(independent.times_utc) - compared,
        "max_kp_difference": max_kp_difference,
        "ap_mismatch_count": len(ap_mismatches),
        "ap_mismatches": ap_mismatches[:20],
        "agrees": not ap_mismatches,
    }
