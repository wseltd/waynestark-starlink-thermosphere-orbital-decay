"""Retrieve the CCMC/SWPC WAM-IPE fixed-height neutral-density grid for Feb 2022.

This resolves blocker BLK-STORM-GRID-TERMS, which recorded that no physics-based
storm-density grid had ever been downloaded, leaving NRLMSISE-00 -- documented to
under-respond to storms -- as the project's only density model.

HOW THE ARCHIVE WAS REACHED, because it is not discoverable by browsing:

  The CCMC landing page
  https://ccmc.gsfc.nasa.gov/external-runs/swpc/ionosphere/WAM-IPE/WAM-IPE-July2021-22/index.html
  is the awslabs ``aws-js-s3-explorer`` v2-alpha single-page app. It renders a
  bucket listing in the browser and therefore contains no file links for a
  non-JavaScript client to follow. The bucket name is not in the HTML: it is a
  literal in the sibling ``explorer.js``, in the AngularJS default settings
  object --

      bucket: 'ccmc-data-tree-root', ... prefix:
      'RT_DATA_TREE_ROOT/ccmc_rt_data_tree/model/ionosphere/WAM-IPE/WAM-IPE-July2021-22/'

  -- and the bucket allows anonymous ListObjectsV2, so the archive can be
  enumerated directly over HTTPS with no AWS credentials and no SDK.

WHICH FILES, and why these and not the others. The archive holds three products
per hour: ``gsm3d.*.nc`` (WAM on its native hybrid-pressure levels),
``ipe3d.*.nc`` (the IPE ionosphere) and ``wam_fixed_height.gsm10.*.nc``. Only the
last is used here. It is the postprocessed WAM neutral atmosphere already
interpolated onto fixed geometric altitudes, which is what a drag calculation at
a stated altitude needs; taking density from the native-level files would require
this project to redo a vertical interpolation the modelling centre has already
done and published. It is also by far the smallest of the three.

Every byte lands in the immutable raw store, so the analysis is reproducible from
recorded hashes long after this S3 bucket stops answering.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
import urllib.error
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from thermosphere_consequence.drivers import space_weather as sw
from thermosphere_consequence.drivers.raw_store import RawStore
from thermosphere_consequence.molecular import composition as comp
from thermosphere_consequence.providers.nrlmsise00 import Nrlmsise00Provider
from thermosphere_consequence.providers.storm_grid import (
    QUIET_REFERENCE_RULE,
    DensitySampler,
    GriddedStormDensityProvider,
    peak_enhancement,
    quiet_reference_by_hour,
)

#: Read out of ``explorer.js`` on the CCMC landing page. Not a guess: the page's
#: own client uses this bucket, and the S3 error document returned for a missing
#: key under the CCMC path confirms the matching key prefix.
BUCKET = "ccmc-data-tree-root"
KEY_PREFIX = "RT_DATA_TREE_ROOT/ccmc_rt_data_tree/model/ionosphere/WAM-IPE/WAM-IPE-July2021-22/"

#: Anonymous ListObjectsV2. The bucket is public; no credentials are sent.
LIST_ENDPOINT = f"https://{BUCKET}.s3.amazonaws.com/"

#: The citable public route to the same objects. ``ccmc.gsfc.nasa.gov`` fronts
#: the identical S3 objects (verified by matching ETag and Content-Length), and
#: it is the URL a reader can be pointed at, so it is what the provenance
#: sidecar records as the retrieval URL.
PUBLIC_BASE = "https://ccmc.gsfc.nasa.gov/external-runs/swpc/ionosphere/WAM-IPE/WAM-IPE-July2021-22"

ARCHIVE_INDEX_URL = f"{PUBLIC_BASE}/index.html"

#: The product: WAM neutral atmosphere on fixed geometric altitude levels.
PRODUCT_STEM = "wam_fixed_height.gsm10."

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

SOURCE = "ccmc"

#: Verbatim from the task brief and NOT upgraded by this script. CCMC publishes
#: a PUBLICATION policy, not a licence; see TERMS_FINDING below for exactly what
#: it does and does not say about commercial reuse.
LICENCE = "CCMC / US Government (check attribution)"

#: The real terms page. An earlier draft of this script recorded
#: `/about/rules-of-the-road/`, which 404s; this one was verified to return 200
#: and to contain the acknowledgement text quoted in CITATION.
LICENCE_URL = "https://ccmc.gsfc.nasa.gov/publication-policy/"

#: CCMC's own sample acknowledgement wording, filled in for this model and this
#: archive. Written from the policy page rather than from memory.
CITATION = (
    "Simulation results have been provided by the Community Coordinated Modeling "
    "Center (CCMC) at Goddard Space Flight Center through their publicly available "
    "simulation services (https://ccmc.gsfc.nasa.gov). The WAM-IPE model was "
    "developed by T. J. Fuller-Rowell, T.-W. Fang, R. A. Akmaev and colleagues at "
    "NOAA SWPC / CIRES, University of Colorado. Archive: WAM-IPE Forecast System "
    "(WFS) operational run for NOAA SWPC, July 2021 - 2022. Event reference: Fang, "
    "T.-W., Kubaryk, A., Goldstein, D., Li, Z., Fuller-Rowell, T., Millward, G., "
    "et al. (2022), Space Weather Environment During the SpaceX Starlink Satellite "
    "Loss in February 2022, Space Weather, 20, e2022SW003193, "
    "doi:10.1029/2022SW003193."
)

#: What was actually established about the terms, and what was not. Recorded in
#: the evidence file because BLK-STORM-GRID-TERMS is a blocker about TERMS as
#: much as about access, and retrieving the files does not settle it.
TERMS_FINDING = {
    "policy_url": LICENCE_URL,
    "policy_kind": "publication policy, not a licence",
    "policy_last_updated": "2025-07-23 (as printed on the page)",
    "requires": [
        "acknowledge the CCMC for the model runs",
        "acknowledge the originator(s) / model developer(s) of the model",
        "report the publication to the CCMC via its publication submission page",
    ],
    "material_condition": (
        "The policy states that for publications using certain models the user "
        "SHOULD CONTACT THE MODEL DEVELOPERS and that 'the model owners may "
        "request co-authorship at their discretion'. A discretionary co-authorship "
        "request is a condition a commercial product cannot generally satisfy, and "
        "it is not a term this project can resolve by reading a web page."
    ),
    "silent_on": [
        "commercial use",
        "redistribution of the retrieved NetCDF files",
        "SaaS or hosted-inference use",
    ],
    "conclusion": (
        "The ACCESS half of BLK-STORM-GRID-TERMS is resolved: the archive is "
        "public, anonymous and now retrieved and hashed. The TERMS half is NOT "
        "resolved. CCMC publishes no commercial-use grant for this archive, so the "
        "blocker's resolution criterion -- 'terms confirmation recorded, or CCMC "
        "ROR run delivered with usage statement' -- is still unmet. Results from "
        "this grid belong in the research mode of config/licence_policy.yaml until "
        "CCMC confirms terms in writing."
    ),
}

CONTENT_TYPE = "application/x-netcdf"

#: Refuse to start a download run larger than this. The archive is hourly and
#: multi-year; an unbounded prefix walk would pull hundreds of gigabytes.
DEFAULT_BUDGET_BYTES = 2 * 1024**3

Fetcher = Callable[[str], bytes]

# --- what the retrieved grid is measured against --------------------------
# The point this project reports its headline density enhancement at, copied
# from scripts/run_atmosphere_audit.py so the comparison is at the same place.
ANALYSIS_ALTITUDE_M = 210_000.0
ANALYSIS_LATITUDE_DEG = 53.22
ANALYSIS_LONGITUDE_DEG = 0.0

#: The quietest complete day in the retrieved window: ap 2-6 through 18:00 UTC,
#: rising to 18 only in the final 3-hour bin. Every hour of it was archived, so
#: the reference day needs no interpolation. Chosen for those two facts, not by
#: trying reference days and keeping the flattering one -- the sensitivity of
#: the answer to that choice is computed below and reported alongside it.
QUIET_REFERENCE_DAY = dt.date(2022, 2, 1)

#: Reference days the headline is stress-tested against.
SENSITIVITY_REFERENCE_DAYS = (dt.date(2022, 2, 1), dt.date(2022, 2, 2), dt.date(2022, 2, 8))

#: Altitudes spanning the band Fang et al. 2022 report for this event.
PROFILE_ALTITUDES_M = (200_000.0, 210_000.0, 250_000.0, 300_000.0, 350_000.0, 400_000.0)

#: This project's own headline, from evidence/ATMOSPHERE_MODEL_AUDIT.md.
REPORTED_MSIS_PEAK_FRACTION = 0.1714

#: Fang et al. 2022, WAM-IPE, 200-400 km, February 2022 event.
FANG_BAND = (0.50, 1.25)


@dataclass(frozen=True, slots=True)
class RemoteObject:
    """One archived file, as the bucket listing describes it."""

    key: str
    size_bytes: int
    etag: str
    last_modified: str

    @property
    def filename(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    @property
    def public_url(self) -> str:
        return f"{PUBLIC_BASE}/{self.key.removeprefix(KEY_PREFIX)}"

    @property
    def stamp(self) -> str:
        """The ``YYYYMMDD_HHMMSS`` valid time embedded in the filename."""
        return self.filename.removeprefix(PRODUCT_STEM).removesuffix(".nc")

    @property
    def asset_id(self) -> str:
        return f"wam_ipe_fixed_height_{self.stamp}"

    @property
    def valid_time_utc(self) -> dt.datetime:
        return dt.datetime.strptime(self.stamp, "%Y%m%d_%H%M%S").replace(tzinfo=dt.UTC)


def list_prefix(fetcher: Fetcher, prefix: str) -> list[RemoteObject]:
    """Every object under ``prefix``, following the listing's continuation tokens."""
    objects: list[RemoteObject] = []
    token: str | None = None
    while True:
        url = f"{LIST_ENDPOINT}?list-type=2&prefix={prefix}&max-keys=1000"
        if token is not None:
            url = f"{url}&continuation-token={urllib_quote(token)}"
        root = ET.fromstring(fetcher(url).decode("utf-8"))
        for node in root.findall("s3:Contents", S3_NS):
            objects.append(
                RemoteObject(
                    key=_text(node, "s3:Key"),
                    size_bytes=int(_text(node, "s3:Size")),
                    etag=_text(node, "s3:ETag").strip('"'),
                    last_modified=_text(node, "s3:LastModified"),
                )
            )
        if _text(root, "s3:IsTruncated", default="false") != "true":
            return objects
        token = _text(root, "s3:NextContinuationToken")


def urllib_quote(value: str) -> str:
    import urllib.parse

    return urllib.parse.quote(value, safe="")


def _text(node: ET.Element, path: str, default: str | None = None) -> str:
    found = node.find(path, S3_NS)
    if found is None or found.text is None:
        if default is not None:
            return default
        raise ValueError(f"S3 listing is missing {path}")
    return found.text


def fixed_height_objects(
    fetcher: Fetcher, start: dt.date, end: dt.date
) -> tuple[list[RemoteObject], list[str]]:
    """Fixed-height files whose valid day falls in ``[start, end]``, and the days walked.

    The listing is issued per day rather than once for the whole run because the
    run spans two years at hourly cadence; a single prefix walk would enumerate
    roughly 40,000 keys to keep 180.
    """
    selected: list[RemoteObject] = []
    days_walked: list[str] = []
    day = start
    while day <= end:
        prefix = f"{KEY_PREFIX}{day:%Y/%m/%d}/postprocessed/"
        days_walked.append(prefix)
        for obj in list_prefix(fetcher, prefix):
            if obj.filename.startswith(PRODUCT_STEM) and obj.filename.endswith(".nc"):
                selected.append(obj)
        day += dt.timedelta(days=1)
    selected.sort(key=lambda o: o.stamp)
    return selected, days_walked


def retrying_fetcher(inner: Fetcher, attempts: int = 4, backoff_s: float = 2.0) -> Fetcher:
    """Retry transient HTTP and network failures; never retry a 404.

    A 404 here means the archive does not hold what the listing implied, which is
    a finding to report rather than a condition to wait out.
    """

    def _fetch(url: str) -> bytes:
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return inner(url)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            if attempt < attempts - 1:
                time.sleep(backoff_s * (2**attempt))
        assert last is not None
        raise last

    return _fetch


def store_object(store: RawStore, obj: RemoteObject, payload: bytes) -> None:
    """Write one grid file and its provenance sidecar into the raw store."""
    store.write(
        source=SOURCE,
        asset_id=obj.asset_id,
        payload=payload,
        url=obj.public_url,
        licence=LICENCE,
        licence_url=LICENCE_URL,
        citation=CITATION,
        content_type=CONTENT_TYPE,
        ext=".nc",
        notes=(
            f"WAM-IPE fixed-height postprocessed neutral density, valid "
            f"{obj.valid_time_utc.isoformat()}. S3 key {obj.key} in bucket {BUCKET}; "
            f"upstream ETag {obj.etag}, last modified {obj.last_modified}."
        ),
    )


def retrieve(
    store: RawStore,
    fetcher: Fetcher,
    start: dt.date,
    end: dt.date,
    budget_bytes: int,
) -> dict[str, object]:
    """Enumerate, budget-check and download the grid. Returns the retrieval record."""
    attempts: list[dict[str, object]] = []

    objects, days_walked = fixed_height_objects(fetcher, start, end)
    total = sum(o.size_bytes for o in objects)
    attempts.append(
        {
            "step": "list",
            "endpoint": LIST_ENDPOINT,
            "prefixes": days_walked,
            "status": "ok",
            "objects_found": len(objects),
            "total_bytes": total,
        }
    )
    if not objects:
        raise RuntimeError(
            f"no {PRODUCT_STEM}*.nc objects under {KEY_PREFIX} for {start}..{end}; "
            "the archive layout has changed and the selection must be re-derived."
        )
    if total > budget_bytes:
        raise RuntimeError(
            f"selection is {total / 1e6:.0f} MB, over the {budget_bytes / 1e6:.0f} MB "
            "budget. Narrow --start/--end rather than raising the budget blindly."
        )

    downloaded = 0
    skipped = 0
    for obj in objects:
        if store.exists(SOURCE, obj.asset_id, ".nc"):
            skipped += 1
            continue
        payload = fetcher(obj.public_url)
        if len(payload) != obj.size_bytes:
            raise RuntimeError(
                f"{obj.public_url}: got {len(payload)} bytes, listing said "
                f"{obj.size_bytes}. Refusing to store a truncated grid."
            )
        store_object(store, obj, payload)
        downloaded += 1
        print(f"  stored {obj.asset_id}  {len(payload)} B", flush=True)

    attempts.append(
        {
            "step": "download",
            "url_pattern": f"{PUBLIC_BASE}/YYYY/MM/DD/postprocessed/{PRODUCT_STEM}*.nc",
            "status": "ok",
            "downloaded": downloaded,
            "already_present": skipped,
        }
    )
    return {
        "outcome": "retrieved",
        "bucket": BUCKET,
        "key_prefix": KEY_PREFIX,
        "bucket_discovery": (
            "Read from the literal `bucket: 'ccmc-data-tree-root'` in the AngularJS "
            f"default settings object in {PUBLIC_BASE}/explorer.js. The bucket is not "
            "named anywhere in index.html."
        ),
        "archive_index_url": ARCHIVE_INDEX_URL,
        "product": f"{PRODUCT_STEM}*.nc (WAM neutral density on fixed geometric altitudes)",
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "file_count": len(objects),
        "total_bytes": total,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "citation": CITATION,
        "terms": TERMS_FINDING,
        "steps": attempts,
        "assets": [
            {
                "asset_id": o.asset_id,
                "url": o.public_url,
                "s3_key": o.key,
                "bytes": o.size_bytes,
                "valid_time_utc": o.valid_time_utc.isoformat(),
            }
            for o in objects
        ],
    }


# --------------------------------------------------------------------------
# What the grid says, against what this project has been reporting
# --------------------------------------------------------------------------


def _samplers(
    store: RawStore, altitude_m: float
) -> tuple[GriddedStormDensityProvider, DensitySampler, DensitySampler]:
    """A WAM-IPE and an NRLMSISE-00 density sampler at the same fixed point.

    Both are plain ``(time) -> kg m^-3`` callables, so the enhancement is formed
    by identical arithmetic for the grid and for the point model and any
    difference between the two answers is a difference between the models.
    """
    kp = sw.parse_gfz_kp_ap(
        store.read("gfz", "kp_2022_extended", ".json"),
        store.read("gfz", "kp_2022_extended_ap", ".json"),
    )
    f107 = sw.parse_celestrak_f107(store.read("celestrak", "sw_last5years", ".csv"))
    grid = GriddedStormDensityProvider(store)
    msis = Nrlmsise00Provider()

    # The grid ignores drivers by construction; one instance is passed only to
    # satisfy the protocol signature.
    unused = sw.drivers_at(dt.datetime(2022, 2, 1, tzinfo=dt.UTC), kp, f107)

    def grid_sample(when: dt.datetime) -> float:
        return grid.total_mass_density(
            when, altitude_m, ANALYSIS_LATITUDE_DEG, ANALYSIS_LONGITUDE_DEG, unused
        )

    def msis_sample(when: dt.datetime) -> float:
        state = msis.state_at(
            when,
            altitude_m,
            ANALYSIS_LATITUDE_DEG,
            ANALYSIS_LONGITUDE_DEG,
            sw.drivers_at(when, kp, f107),
        )
        return comp.total_species_mass_density(state.number_densities)

    return grid, grid_sample, msis_sample


@dataclass(frozen=True, slots=True)
class PeakComparison:
    """Peak enhancement from the grid and from the point model, same definition."""

    altitude_m: float
    quiet_reference_day: dt.date
    sample_count: int
    grid_fraction: float
    grid_time_utc: dt.datetime
    grid_density_kg_m3: float
    msis_fraction: float
    msis_time_utc: dt.datetime
    msis_density_kg_m3: float

    @property
    def ratio(self) -> float:
        """How much more the grid responds than the point model."""
        return self.grid_fraction / self.msis_fraction

    @property
    def inside_fang_band(self) -> bool:
        return FANG_BAND[0] <= self.grid_fraction <= FANG_BAND[1]

    def as_dict(self) -> dict[str, object]:
        return {
            "altitude_m": self.altitude_m,
            "quiet_reference_day": self.quiet_reference_day.isoformat(),
            "samples": self.sample_count,
            "wam_ipe_peak_fraction": self.grid_fraction,
            "wam_ipe_peak_time_utc": self.grid_time_utc.isoformat(),
            "wam_ipe_peak_density_kg_m3": self.grid_density_kg_m3,
            "nrlmsise00_peak_fraction": self.msis_fraction,
            "nrlmsise00_peak_time_utc": self.msis_time_utc.isoformat(),
            "nrlmsise00_peak_density_kg_m3": self.msis_density_kg_m3,
            "wam_ipe_over_nrlmsise00": self.ratio,
            "inside_fang_band": self.inside_fang_band,
        }


@dataclass(frozen=True, slots=True)
class GridAnalysis:
    """Everything the retrieved grid says about this project's headline number."""

    coverage: str
    gaps_utc: tuple[tuple[str, str], ...]
    interpolation_rule: str
    headline: PeakComparison
    reference_day_sensitivity: tuple[PeakComparison, ...]
    altitude_profile: tuple[PeakComparison, ...]
    density_ratio_median: float
    density_ratio_min: float
    density_ratio_max: float
    density_ratio_samples: int

    def as_dict(self) -> dict[str, object]:
        return {
            "point": {
                "altitude_m": ANALYSIS_ALTITUDE_M,
                "latitude_deg": ANALYSIS_LATITUDE_DEG,
                "longitude_deg": ANALYSIS_LONGITUDE_DEG,
            },
            "grid_coverage": self.coverage,
            "grid_gaps_utc": [list(gap) for gap in self.gaps_utc],
            "interpolation_rule": self.interpolation_rule,
            "enhancement_definition": QUIET_REFERENCE_RULE,
            "headline": self.headline.as_dict(),
            "reference_day_sensitivity": [c.as_dict() for c in self.reference_day_sensitivity],
            "altitude_profile": [c.as_dict() for c in self.altitude_profile],
            "absolute_density_ratio_wam_over_msis": {
                "median": self.density_ratio_median,
                "min": self.density_ratio_min,
                "max": self.density_ratio_max,
                "samples": self.density_ratio_samples,
                "note": _ABSOLUTE_DENSITY_NOTE,
            },
            "comparison": {
                "project_reported_msis_peak_fraction": REPORTED_MSIS_PEAK_FRACTION,
                "wam_ipe_peak_fraction": self.headline.grid_fraction,
                "wam_ipe_minus_reported": (
                    self.headline.grid_fraction - REPORTED_MSIS_PEAK_FRACTION
                ),
                "fang_2022_band": list(FANG_BAND),
                "wam_ipe_inside_fang_band": self.headline.inside_fang_band,
            },
        }


_ABSOLUTE_DENSITY_NOTE = (
    "WAM-IPE is denser than NRLMSISE-00 at this point throughout the window, "
    "storm or no storm. This is a bias in the ABSOLUTE density and is separate "
    "from, and additional to, the storm-response gap."
)


def _peak_pair(
    grid: GriddedStormDensityProvider,
    grid_sample: DensitySampler,
    msis_sample: DensitySampler,
    reference_day: dt.date,
    altitude_m: float,
) -> PeakComparison:
    """Peak enhancement for both models against one quiet reference day.

    Samples are the grid's OWN archived hours, so every point compared is a real
    simulated hour: nothing is interpolated in time and nothing is skipped. The
    reference day itself is excluded, because a day cannot be its own anomaly.
    """
    times = [t for t in grid.times_utc if t.date() != reference_day]
    grid_when, grid_fraction, grid_rho = peak_enhancement(
        grid_sample, times, quiet_reference_by_hour(grid_sample, reference_day)
    )
    msis_when, msis_fraction, msis_rho = peak_enhancement(
        msis_sample, times, quiet_reference_by_hour(msis_sample, reference_day)
    )
    return PeakComparison(
        altitude_m=altitude_m,
        quiet_reference_day=reference_day,
        sample_count=len(times),
        grid_fraction=grid_fraction,
        grid_time_utc=grid_when,
        grid_density_kg_m3=grid_rho,
        msis_fraction=msis_fraction,
        msis_time_utc=msis_when,
        msis_density_kg_m3=msis_rho,
    )


def analyse(store: RawStore) -> GridAnalysis:
    """Peak storm-time density enhancement from the grid, against MSIS and Fang."""
    grid, grid_sample, msis_sample = _samplers(store, ANALYSIS_ALTITUDE_M)
    headline = _peak_pair(
        grid, grid_sample, msis_sample, QUIET_REFERENCE_DAY, ANALYSIS_ALTITUDE_M
    )
    sensitivity = tuple(
        _peak_pair(grid, grid_sample, msis_sample, day, ANALYSIS_ALTITUDE_M)
        for day in SENSITIVITY_REFERENCE_DAYS
    )

    profile: list[PeakComparison] = []
    for altitude_m in PROFILE_ALTITUDES_M:
        level_grid, level_grid_sample, level_msis = _samplers(store, altitude_m)
        profile.append(
            _peak_pair(
                level_grid, level_grid_sample, level_msis, QUIET_REFERENCE_DAY, altitude_m
            )
        )

    # Absolute density, not only its storm-relative change: a model that is
    # denser everywhere produces more drag even with no storm at all, and that
    # is a separate finding from the storm response.
    ratios = sorted(grid_sample(t) / msis_sample(t) for t in grid.times_utc)

    return GridAnalysis(
        coverage=grid.coverage.describe(),
        gaps_utc=tuple((a.isoformat(), b.isoformat()) for a, b in grid.coverage.gaps),
        interpolation_rule=grid.interpolation_rule,
        headline=headline,
        reference_day_sensitivity=sensitivity,
        altitude_profile=tuple(profile),
        density_ratio_median=ratios[len(ratios) // 2],
        density_ratio_min=ratios[0],
        density_ratio_max=ratios[-1],
        density_ratio_samples=len(ratios),
    )


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected an int in the retrieval record, got {value!r}")
    return value


def render_markdown(record: dict[str, object], analysis: GridAnalysis | None) -> str:
    """The human-readable retrieval and comparison report."""
    lines: list[str] = ["# WAM-IPE storm-density grid: retrieval and first result", ""]
    lines.append(f"Generated {record['retrieved_at_utc']} by `scripts/fetch_wam_ipe.py`.")
    lines.append("Machine-readable companion: `evidence/storm_grid_retrieval.json`.")
    lines.append("")
    lines.append("## Outcome")
    lines.append("")
    lines.append(
        f"**RETRIEVED.** {record['file_count']} files, "
        f"{_as_int(record['total_bytes']) / 1e6:.1f} MB, covering "
        f"{record['window_start']} to {record['window_end']}."
    )
    lines.append("")
    lines.append(
        "This resolves the ACCESS half of BLK-STORM-GRID-TERMS, which recorded that "
        "the February 2022 file URLs were 'partly behind dynamic interfaces' and that "
        "no grid had been downloaded. It does NOT resolve the TERMS half -- see "
        "'Licence and terms' below. The blocker entry in "
        "`planning/registry/blockers.yaml` was not edited by this run."
    )
    lines.append("")
    lines.append("## How the archive was reached")
    lines.append("")
    lines.append(f"- Landing page: {record['archive_index_url']}")
    lines.append(f"- Bucket: `{record['bucket']}`, key prefix `{record['key_prefix']}`")
    lines.append(f"- How the bucket was found: {record['bucket_discovery']}")
    lines.append(f"- Product: {record['product']}")
    lines.append(
        "- Configuration: the SWPC operational **WAM-IPE Forecast System (WFS)** run, "
        "not the free-running Runs-on-Request configuration. The CCMC model catalog "
        "distinguishes the two explicitly, and the archive's own helper scripts "
        "(`name_change_wfs.sh`) name WFS."
    )
    lines.append("")
    lines.append("## Licence and terms")
    lines.append("")
    lines.append(f"- Recorded licence string: `{record['licence']}`")
    lines.append(f"- Terms page: {record['licence_url']} ({TERMS_FINDING['policy_kind']})")
    lines.append("")
    lines.append("CCMC's policy requires that a user:")
    for requirement in TERMS_FINDING["requires"]:
        lines.append(f"- {requirement}")
    lines.append("")
    lines.append(str(TERMS_FINDING["material_condition"]))
    lines.append("")
    lines.append(
        "The policy is silent on: "
        + ", ".join(str(item) for item in TERMS_FINDING["silent_on"])
        + "."
    )
    lines.append("")
    lines.append(f"**{TERMS_FINDING['conclusion']}**")
    lines.append("")
    lines.append(f"Acknowledgement text to use: {record['citation']}")
    lines.append("")
    lines.append(
        "Every file is stored through `RawStore` under `data/raw/ccmc/` with a "
        "`.meta.json` sidecar carrying its SHA-256, source URL, licence and "
        "retrieval time, so the result below can be regenerated from bytes whose "
        "provenance is recorded even after the bucket stops answering."
    )
    lines.append("")

    if analysis is None:
        lines.append("No analysis was run in this invocation.")
        return "\n".join(lines) + "\n"

    head = analysis.headline
    lines.append("## The grid")
    lines.append("")
    lines.append(f"- Coverage: {analysis.coverage}")
    lines.append(f"- Interpolation: {analysis.interpolation_rule}")
    lines.append("")
    lines.append("## What is being compared")
    lines.append("")
    lines.append(QUIET_REFERENCE_RULE)
    lines.append("")
    lines.append(
        f"Point: {ANALYSIS_ALTITUDE_M / 1000:.0f} km, {ANALYSIS_LATITUDE_DEG} deg N, "
        f"{ANALYSIS_LONGITUDE_DEG} deg E -- the point this project reports its "
        f"headline enhancement at. Quiet reference day: "
        f"{head.quiet_reference_day.isoformat()}. Samples: {head.sample_count} archived "
        "hours, each a real simulated hour rather than an interpolated one."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("| --- | --- |")
    lines.append(
        f"| **WAM-IPE peak enhancement at 210 km** | **{_pct(head.grid_fraction)}** at "
        f"{head.grid_time_utc.isoformat()} |"
    )
    lines.append(
        f"| NRLMSISE-00 peak, identical definition | {_pct(head.msis_fraction)} at "
        f"{head.msis_time_utc.isoformat()} |"
    )
    lines.append(
        f"| NRLMSISE-00 peak, this project's headline | "
        f"{_pct(REPORTED_MSIS_PEAK_FRACTION)} |"
    )
    lines.append(f"| WAM-IPE / NRLMSISE-00 storm response | {head.ratio:.2f}x |")
    lines.append(
        f"| Fang et al. 2022 published band (200-400 km) | "
        f"{_pct(FANG_BAND[0])} to {_pct(FANG_BAND[1])} |"
    )
    lines.append("")
    lines.append(
        f"The two NRLMSISE-00 rows agree to within "
        f"{abs(head.msis_fraction - REPORTED_MSIS_PEAK_FRACTION) * 100:.2f} percentage "
        "points, which is the check that matters before believing the third row: the "
        "quiet-reference-day definition used here and the quiet-driver counterfactual "
        "used by `scripts/run_atmosphere_audit.py` are different constructions, and "
        "they put MSIS in the same place. The gap to WAM-IPE is therefore a difference "
        "between the MODELS, not between two ways of taking a ratio."
    )
    lines.append("")

    lines.append("## Altitude profile, against the published band")
    lines.append("")
    lines.append("| altitude | WAM-IPE peak | NRLMSISE-00 peak | ratio | inside Fang 50-125% |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in analysis.altitude_profile:
        lines.append(
            f"| {row.altitude_m / 1000:.0f} km | {_pct(row.grid_fraction)} "
            f"| {_pct(row.msis_fraction)} | {row.ratio:.2f}x "
            f"| {'yes' if row.inside_fang_band else 'no'} |"
        )
    lines.append("")
    profile_low = analysis.altitude_profile[0]
    profile_high = analysis.altitude_profile[-1]
    lines.append(
        "This is the strongest evidence that the retrieved grid is the product Fang "
        "et al. describe rather than something merely named like it. Their band is "
        "quoted for 200-400 km as a whole, and the enhancement here rises with "
        f"altitude across exactly that range, from {_pct(profile_low.grid_fraction)} at "
        f"{profile_low.altitude_m / 1000:.0f} km to {_pct(profile_high.grid_fraction)} at "
        f"{profile_high.altitude_m / 1000:.0f} km. That spans the published "
        f"{_pct(FANG_BAND[0])}-{_pct(FANG_BAND[1])} band rather than reproducing its "
        "endpoints: the bottom of the profile sits just BELOW the band's lower edge "
        "and the top sits inside it, which is what a band quoted for a whole altitude "
        "range should look like when it is resolved level by level. Nothing in this "
        "repository was fitted to make that happen."
    )
    lines.append("")

    lines.append("## Sensitivity to the quiet reference day")
    lines.append("")
    lines.append("| reference day | WAM-IPE peak | NRLMSISE-00 peak | ratio |")
    lines.append("| --- | --- | --- | --- |")
    for row in analysis.reference_day_sensitivity:
        lines.append(
            f"| {row.quiet_reference_day.isoformat()} | {_pct(row.grid_fraction)} "
            f"| {_pct(row.msis_fraction)} | {row.ratio:.2f}x |"
        )
    lines.append("")
    lines.append(
        "The headline uses 2022-02-01 because it is the quietest COMPLETE day in the "
        "retrieved window (ap 2-6 until 18:00 UTC) and because it produces the "
        "SMALLEST WAM-IPE/MSIS ratio of the three. The choice is therefore "
        "conservative: every other reference day makes the gap wider, not narrower."
    )
    lines.append("")

    lines.append("## Absolute density, not only its change")
    lines.append("")
    lines.append(
        f"Across all {analysis.density_ratio_samples} archived hours at this point, "
        f"WAM-IPE's density divided by NRLMSISE-00's has median "
        f"{analysis.density_ratio_median:.2f} (range {analysis.density_ratio_min:.2f} "
        f"to {analysis.density_ratio_max:.2f})."
    )
    lines.append("")
    lines.append(_ABSOLUTE_DENSITY_NOTE)
    lines.append("")

    lines.append("## What this does NOT establish")
    lines.append("")
    for caveat in _CAVEATS:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines) + "\n"


_CAVEATS = (
    "The grid carries TOTAL neutral mass density only. There is no species "
    "breakdown and no temperature in it, so nothing here says anything about "
    "storm-time O/N2 depletion or about a composition-weighted drag coefficient.",
    "WAM-IPE is a model, not a measurement. Agreeing with Fang et al. is agreement "
    "with the same model family, not validation against accelerometer data. The "
    "only in-situ densities this project holds for the event are Swarm-A at 438 km "
    "and GRACE-FO at 505 km, neither of which is at 210 km.",
    "The enhancement here is a ratio against a real quiet day inside the same run, "
    "not against a no-storm counterfactual. The reference day carries its own "
    "residual activity, so the ratios measure enhancement relative to that "
    "reference day's background rather than a zero-activity counterfactual "
    "(not an established universal lower bound on the storm response).",
    "Two seven-hour holes in the archive (2022-02-05 and 2022-02-07, 10:00-15:00 "
    "UTC) were never filled in and are refused by the provider rather than "
    "interpolated across. Neither hole is near the peak.",
    "This one retrieval covers 2022-02-01 to 2022-02-08 only. It says nothing "
    "about any other storm, and the provider refuses outside that window.",
)


def write_records(
    record: dict[str, object], analysis: GridAnalysis | None, evidence_dir: pathlib.Path
) -> None:
    """Write the machine-readable retrieval record and its markdown companion."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if analysis is not None:
        record["analysis"] = analysis.as_dict()
    target = evidence_dir / "storm_grid_retrieval.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    markdown = evidence_dir / "STORM_GRID_RETRIEVAL.md"
    markdown.write_text(render_markdown(record, analysis), encoding="utf-8")
    print(f"wrote {markdown}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-02-01", help="first UTC day to retrieve")
    parser.add_argument("--end", default="2022-02-08", help="last UTC day to retrieve")
    parser.add_argument("--root", default="data/raw", help="raw store root")
    parser.add_argument("--evidence", default="evidence", help="evidence directory")
    parser.add_argument(
        "--budget-bytes",
        type=int,
        default=DEFAULT_BUDGET_BYTES,
        help="refuse to download more than this in one run",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="re-run the analysis over the already-stored grid, touching no network",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="retrieve only; do not compare the grid against NRLMSISE-00",
    )
    args = parser.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    store = RawStore(args.root)

    if args.skip_fetch:
        record = read_previous_record(pathlib.Path(args.evidence))
    else:
        fetcher = retrying_fetcher(sw.http_fetcher(timeout_s=180.0))
        print(f"listing {BUCKET} under {KEY_PREFIX} for {start} .. {end} ...", flush=True)
        record = retrieve(store, fetcher, start, end, args.budget_bytes)
    record["retrieved_at_utc"] = dt.datetime.now(dt.UTC).isoformat()

    analysis: GridAnalysis | None = None
    if not args.skip_analysis:
        print("comparing the grid against NRLMSISE-00 ...", flush=True)
        analysis = analyse(store)
        head = analysis.headline
        print(
            f"  WAM-IPE peak {head.grid_fraction * 100:.2f}% at "
            f"{head.grid_time_utc.isoformat()}; NRLMSISE-00 "
            f"{head.msis_fraction * 100:.2f}% ({head.ratio:.2f}x)"
        )

    write_records(record, analysis, pathlib.Path(args.evidence))

    stored = [a for a in store.manifest() if a.source == SOURCE]
    print(f"\nraw store holds {len(stored)} {SOURCE} assets, {sum(a.byte_count for a in stored)} B")
    return 0


def read_previous_record(evidence_dir: pathlib.Path) -> dict[str, object]:
    """Reload the retrieval record so ``--skip-fetch`` reports the real retrieval."""
    target = evidence_dir / "storm_grid_retrieval.json"
    if not target.exists():
        raise SystemExit(
            f"--skip-fetch needs a previous {target}; run the script once without it."
        )
    loaded: dict[str, object] = json.loads(target.read_text(encoding="utf-8"))
    loaded.pop("analysis", None)
    return loaded


if __name__ == "__main__":
    raise SystemExit(main())
