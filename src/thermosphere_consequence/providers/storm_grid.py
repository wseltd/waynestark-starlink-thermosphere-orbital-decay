"""Neutral mass density read from a retrieved WAM-IPE storm grid.

This is the project's first PHYSICS-BASED alternative to NRLMSISE-00's storm
response. Everything else available here is either NRLMSISE-00 itself, a scalar
multiplier applied to it (``providers.scaled_storm`` -- explicitly not a model),
or the published enhancement bands transcribed from papers
(``providers.published_envelope`` -- a citation, not a simulation). This
provider reads the density field a coupled whole-atmosphere model actually
produced, hour by hour, on a global grid.

WHAT THE GRID IS. The CCMC/SWPC WAM-IPE operational archive run, product
``wam_fixed_height.gsm10.*.nc``: WAM's neutral atmosphere postprocessed onto
fixed geometric altitudes. One file per hour, each holding a single variable
``den`` -- total neutral mass density in kg m^-3 -- on a
(time, altitude, latitude, longitude) grid. Retrieved and hashed by
``scripts/fetch_wam_ipe.py``; read here through the immutable raw store, so
every number this provider returns is traceable to bytes with a recorded
SHA-256.

WHAT IT DOES NOT CONTAIN, stated here because the shape of the file invites the
wrong assumption:

  * NO species. The fixed-height product carries total mass density only. There
    is therefore no storm-time composition here -- no O/N2 depletion, no
    per-species number density. ``total_mass_density`` is this provider's
    native answer and is the one to prefer. ``state_at`` exists so the provider
    can stand in for an ``AtmosphereProvider`` at a call site, and it requires
    a ``composition_baseline`` to be supplied explicitly, because the species
    breakdown it returns comes from that baseline and NOT from WAM-IPE.

  * NO temperature. ``state_at`` reports the baseline's temperature, labelled
    as the baseline's on every state it returns.

  * NO counterfactual. The archive holds the storm as it was run. There is no
    quiet twin of this run, so a "storm enhancement" computed from it is a
    ratio against another time in the SAME run, not against a no-storm control.
    That is a different quantity from NRLMSISE-00's storm/quiet ratio and must
    not be quoted as if it were the same one.

WHAT IT REFUSES. Outside the grid's time span, altitude span or valid latitude
span it raises ``OutsideGridCoverageError`` rather than extrapolating. A
gridded model extrapolated past its own domain is no longer that model's
prediction, and the whole reason to hold real model output is to stop
substituting a guess for one.

Longitude is the one axis that is NOT bounded, because it is periodic: the grid
spans 0-356 degrees at 4-degree spacing and the interval from 356 back to 0 is
interpolated across the wrap. That is interpolation, not extrapolation.

THE ARCHIVE HAS HOLES. The February 2022 retrieval is hourly but not complete:
2022-02-05 and 2022-02-07 are each missing 10:00-15:00 UTC. Those hours were
never archived, so this provider REFUSES them too. Bridging a seven-hour hole
during a geomagnetic storm with a straight line would produce a number that
looks like model output and is not, which is the specific failure this module
exists to avoid; a request that lands inside a hole raises and names the hole.

THE POLE ROWS ARE NOT DATA. In this product the latitude rows at -90 and +90
are identically zero at every altitude and every longitude -- a postprocessing
artefact, since a zero neutral density at 200 km is not a physical claim any
model makes. Those rows are dropped on load and the provider's valid latitude
coverage narrows accordingly, so a request near the pole is REFUSED rather than
answered by interpolating towards zero.

Related blocker: BLK-STORM-GRID-TERMS.
"""

from __future__ import annotations

import datetime as _dt
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import netCDF4
import numpy as np
from numpy.typing import NDArray

from ..domain.provenance import ValueStatus
from ..domain.temporal import ensure_utc
from ..drivers.raw_store import RawStore
from ..molecular import composition as comp
from ..protocols.atmosphere import AtmosphereProvider, AtmosphereState, DriverState

__all__ = [
    "ASSET_PREFIX",
    "DENSITY_VARIABLE",
    "GRID_SOURCE",
    "MODEL_LICENCE",
    "MODEL_NAME",
    "MODEL_REFERENCE",
    "QUIET_REFERENCE_RULE",
    "DensitySampler",
    "GridCoverage",
    "GridInconsistencyError",
    "GriddedStormDensityProvider",
    "MissingCompositionBaselineError",
    "OutsideGridCoverageError",
    "enhancement_series",
    "peak_enhancement",
    "quiet_reference_by_hour",
]

#: Raw-store source and asset-id prefix written by ``scripts/fetch_wam_ipe.py``.
GRID_SOURCE = "ccmc"
ASSET_PREFIX = "wam_ipe_fixed_height_"

MODEL_NAME = "WAM-IPE"
MODEL_REFERENCE = "Fang et al. 2022, doi:10.1029/2022SW003193 (WAM-IPE at NOAA SWPC)"
MODEL_LICENCE = "CCMC / US Government (check attribution)"

#: Variable names in the fixed-height product.
DENSITY_VARIABLE = "den"
_ALTITUDE_VARIABLE = "hlevs"
_LATITUDE_VARIABLE = "lat"
_LONGITUDE_VARIABLE = "lon"
_TIME_VARIABLE = "time"

#: The product stores altitude in kilometres and time in days since the epoch.
#: These are CHECKED against each file's own units attributes rather than
#: assumed: a product archived in metres, or in hours since the epoch, would
#: otherwise be read as a plausible number that is wrong by a factor of 1000.
_ALTITUDE_UNITS_KM_TO_M = 1000.0
_TIME_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)
_EXPECTED_UNITS = {
    _ALTITUDE_VARIABLE: "km",
    _TIME_VARIABLE: "days since 1970-01-01",
    DENSITY_VARIABLE: "kg m^-3",
}

#: A filename-derived valid time may disagree with the file's own time variable
#: by no more than this. Beyond it the archive is mislabelled and the provider
#: refuses rather than indexing the grid by a name it cannot trust.
_TIME_LABEL_TOLERANCE = _dt.timedelta(seconds=1)

#: Coordinate axes are compared between files to this tolerance in their own
#: units (km, degrees). Float32 storage of a decimal level such as 210 km lands
#: about 1.5e-5 km away from it, so the comparison cannot be exact.
_AXIS_TOLERANCE = 1e-3

_INTERPOLATION_RULE = (
    "Density is interpolated log-linearly in altitude (density falls close to "
    "exponentially with height, so interpolating the density itself between "
    "10 km levels would bias low), then bilinearly in latitude and longitude, "
    "then linearly in time between the two bracketing hourly files. Longitude "
    "wraps across the 356-to-0 degree gap; every other axis refuses outside "
    "its range."
)

_NO_COMPOSITION_NOTE = (
    "COMPOSITION IS NOT WAM-IPE's: the fixed-height product carries total "
    "neutral mass density only. The species number densities returned here are "
    "the composition baseline's, rescaled by a single factor so their mass sums "
    "to WAM-IPE's total. Storm-time O/N2 depletion is therefore NOT represented, "
    "and any species-weighted quantity computed from this state reflects the "
    "baseline model's composition, not the grid's."
)

_NO_TEMPERATURE_NOTE = (
    "TEMPERATURE IS NOT WAM-IPE's: the fixed-height product carries no "
    "temperature. The value returned is the composition baseline's, unmodified."
)

_DRIVERS_NOT_CONSUMED_NOTE = (
    "DRIVERS NOT CONSUMED: this provider did not use the DriverState it was "
    "passed. The grid is a completed simulation that already consumed its own "
    "operational forcing at run time; the drivers are carried on the returned "
    "state for provenance only and changing them changes nothing."
)


class OutsideGridCoverageError(ValueError):
    """Raised when a point falls outside the grid's own domain.

    A ``ValueError`` subclass so a caller validating inputs catches the broad
    case, while a caller that wants to distinguish "the model was never run
    here" from "that argument is malformed" can catch this specifically.
    """


class GridInconsistencyError(RuntimeError):
    """Raised when the stored files do not form one coherent grid."""


class MissingCompositionBaselineError(RuntimeError):
    """Raised by ``state_at`` when no composition baseline was supplied."""


@dataclass(frozen=True, slots=True)
class GridCoverage:
    """The domain the loaded grid actually speaks for."""

    time_start_utc: _dt.datetime
    time_end_utc: _dt.datetime
    time_count: int
    altitude_min_m: float
    altitude_max_m: float
    altitude_level_count: int
    latitude_min_deg: float
    latitude_max_deg: float
    latitude_count: int
    longitude_count: int
    cadence_s: float
    gaps: tuple[tuple[_dt.datetime, _dt.datetime], ...] = ()
    longitude_is_periodic: bool = True

    def describe(self) -> str:
        gaps = (
            "no gaps"
            if not self.gaps
            else "; ".join(
                f"GAP {a.isoformat()} to {b.isoformat()}" for a, b in self.gaps
            )
        )
        return (
            f"{self.time_start_utc.isoformat()} to {self.time_end_utc.isoformat()} "
            f"({self.time_count} files at {self.cadence_s / 3600:g} h cadence, {gaps}), "
            f"{self.altitude_min_m / 1000:.0f}-{self.altitude_max_m / 1000:.0f} km "
            f"({self.altitude_level_count} levels), "
            f"{self.latitude_min_deg:.0f} to {self.latitude_max_deg:.0f} deg latitude "
            f"({self.latitude_count} rows), {self.longitude_count} longitudes (periodic)"
        )


@dataclass(frozen=True, slots=True)
class _Slice:
    """One hour of the grid: density on (altitude, latitude, longitude)."""

    time_utc: _dt.datetime
    asset_id: str
    density_kg_m3: NDArray[np.float64]


def _bracket(
    axis: NDArray[np.float64], value: float, name: str, unit: str
) -> tuple[int, int, float]:
    """Indices either side of ``value`` and the fractional weight of the upper one."""
    low = float(axis[0])
    high = float(axis[-1])
    if not low <= value <= high:
        raise OutsideGridCoverageError(
            f"{name} {value:g} {unit} is outside the grid's coverage "
            f"[{low:g}, {high:g}] {unit}. This provider refuses to extrapolate "
            "a gridded model beyond the domain it was run over."
        )
    upper = int(np.searchsorted(axis, value, side="right"))
    if upper >= axis.size:
        return axis.size - 2, axis.size - 1, 1.0
    lower = upper - 1
    span = float(axis[upper]) - float(axis[lower])
    weight = 0.0 if span == 0.0 else (value - float(axis[lower])) / span
    return lower, upper, weight


def _bracket_periodic(
    axis: NDArray[np.float64], value: float, period: float
) -> tuple[int, int, float]:
    """As ``_bracket`` but across a periodic axis, so nothing is ever outside."""
    wrapped = value % period
    if wrapped < float(axis[0]):
        wrapped += period
    last = axis.size - 1
    if wrapped >= float(axis[last]):
        span = period - float(axis[last]) + float(axis[0])
        return last, 0, (wrapped - float(axis[last])) / span
    upper = int(np.searchsorted(axis, wrapped, side="right"))
    lower = upper - 1
    span = float(axis[upper]) - float(axis[lower])
    weight = 0.0 if span == 0.0 else (wrapped - float(axis[lower])) / span
    return lower, upper, weight


def _time_from_asset_id(asset_id: str) -> _dt.datetime:
    """The valid time a stored asset's name claims, as a UTC instant."""
    stamp = asset_id.removeprefix(ASSET_PREFIX)
    try:
        return _dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=_dt.UTC)
    except ValueError as exc:
        raise GridInconsistencyError(
            f"stored asset {asset_id!r} does not carry a YYYYMMDD_HHMMSS valid time"
        ) from exc


class GriddedStormDensityProvider:
    """Total neutral mass density interpolated from a stored WAM-IPE grid.

    Satisfies ``TotalDensityProvider`` natively. It also satisfies
    ``AtmosphereProvider``, but only usefully when constructed with a
    ``composition_baseline``: the grid has no species and no temperature, and
    this class will not invent either.

    Files are opened lazily and the most recently used hours are cached, so
    holding a week of hourly global grids costs a few tens of megabytes rather
    than the half-gigabyte the whole archive occupies on disk. Every load goes
    through ``RawStore.read``, which verifies the payload against its recorded
    SHA-256, so a grid altered on disk raises instead of quietly changing an
    answer.
    """

    name = MODEL_NAME
    version = "SWPC operational archive July2021-22, wam_fixed_height.gsm10 (CCMC external run)"

    def __init__(
        self,
        store: RawStore,
        *,
        source: str = GRID_SOURCE,
        asset_prefix: str = ASSET_PREFIX,
        composition_baseline: AtmosphereProvider | None = None,
        cache_slices: int = 6,
        max_interpolation_gap: _dt.timedelta | None = None,
    ) -> None:
        if max_interpolation_gap is not None and max_interpolation_gap <= _dt.timedelta(0):
            raise ValueError(
                f"max_interpolation_gap must be positive, got {max_interpolation_gap}"
            )
        if cache_slices < 2:
            raise ValueError(
                f"cache_slices must be at least 2 so a time interpolation can hold "
                f"both bracketing hours, got {cache_slices}"
            )
        self._store = store
        self._source = source
        self._asset_prefix = asset_prefix
        self.composition_baseline = composition_baseline
        self._cache_slices = cache_slices
        self._cache: OrderedDict[str, _Slice] = OrderedDict()

        self._asset_ids: tuple[str, ...] = tuple(self._discover_assets())
        if not self._asset_ids:
            raise GridInconsistencyError(
                f"no {asset_prefix}* assets under source {source!r} in {store.root}. "
                "Run scripts/fetch_wam_ipe.py to retrieve the grid."
            )
        self._times: tuple[_dt.datetime, ...] = tuple(
            _time_from_asset_id(a) for a in self._asset_ids
        )
        if len(set(self._times)) != len(self._times):
            raise GridInconsistencyError(
                "two stored assets claim the same valid time; the grid's time axis "
                "must be strictly increasing"
            )
        self._time_axis: NDArray[np.float64] = np.asarray(
            [t.timestamp() for t in self._times], dtype=float
        )
        self._cadence = self._native_cadence()
        # An archive with holes must not be bridged silently: anything wider
        # than the native cadence is a gap in the simulation record, not an
        # interval to draw a line across.
        self._max_gap = max_interpolation_gap or self._cadence
        self._gaps: tuple[tuple[_dt.datetime, _dt.datetime], ...] = tuple(
            (a, b)
            for a, b in zip(self._times, self._times[1:], strict=False)
            if b - a > self._max_gap
        )

        # The first file defines the grid. Every later file is checked against
        # it on load, so a mixed-resolution archive cannot be interpolated
        # across without anyone noticing.
        first, altitudes_m, latitudes, longitudes, block = self._read_raw(self._asset_ids[0])
        for axis, label in (
            (altitudes_m, "altitude"),
            (latitudes, "latitude"),
            (longitudes, "longitude"),
        ):
            if axis.size < 2 or not bool(np.all(np.diff(axis) > 0.0)):
                raise GridInconsistencyError(
                    f"{self._asset_ids[0]}: {label} axis is not strictly increasing"
                )
        self._altitudes_m: NDArray[np.float64] = altitudes_m
        self._latitudes_deg: NDArray[np.float64] = latitudes
        self._longitudes_deg: NDArray[np.float64] = longitudes
        self._cache[self._asset_ids[0]] = _Slice(
            time_utc=first, asset_id=self._asset_ids[0], density_kg_m3=block
        )

    # -- construction helpers ---------------------------------------------

    def _native_cadence(self) -> _dt.timedelta:
        """The archive's intended sampling interval: the most common time step.

        Taken from the data rather than assumed, so a product archived at a
        different cadence is handled without editing a constant here.
        """
        if len(self._times) < 2:
            return _dt.timedelta(0)
        steps = [b - a for a, b in zip(self._times, self._times[1:], strict=False)]
        if min(steps) <= _dt.timedelta(0):
            raise GridInconsistencyError("the grid's time axis is not increasing")
        return min(steps, key=lambda step: (-steps.count(step), step))

    def _discover_assets(self) -> Iterator[str]:
        """Stored asset ids for this grid, in valid-time order."""
        directory = self._store.root / self._source
        if not directory.is_dir():
            return
        found = sorted(
            path.name.removesuffix(".meta.json")
            for path in directory.glob(f"{self._asset_prefix}*.meta.json")
        )
        yield from found

    def _read_raw(
        self, asset_id: str
    ) -> tuple[
        _dt.datetime,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Load one hour: its valid time, its three axes and its density block."""
        payload = self._store.read(self._source, asset_id, ".nc")
        dataset = netCDF4.Dataset(asset_id, mode="r", memory=payload)
        try:
            for variable, expected in _EXPECTED_UNITS.items():
                actual = getattr(dataset.variables[variable], "units", None)
                if actual != expected:
                    raise GridInconsistencyError(
                        f"{asset_id}: variable {variable!r} is in units {actual!r}, "
                        f"not the expected {expected!r}. Reading it anyway would "
                        "silently rescale every density this provider returns."
                    )
            altitudes_m = (
                np.asarray(dataset.variables[_ALTITUDE_VARIABLE][:], dtype=float)
                * _ALTITUDE_UNITS_KM_TO_M
            )
            latitudes = np.asarray(dataset.variables[_LATITUDE_VARIABLE][:], dtype=float)
            longitudes = np.asarray(dataset.variables[_LONGITUDE_VARIABLE][:], dtype=float)
            days = np.asarray(dataset.variables[_TIME_VARIABLE][:], dtype=float)
            density = np.asarray(dataset.variables[DENSITY_VARIABLE][:], dtype=float)
        finally:
            dataset.close()

        if days.size != 1:
            raise GridInconsistencyError(
                f"{asset_id}: expected one time step per file, found {days.size}"
            )
        recorded = _TIME_EPOCH + _dt.timedelta(days=float(days[0]))
        labelled = _time_from_asset_id(asset_id)
        if abs(recorded - labelled) > _TIME_LABEL_TOLERANCE:
            raise GridInconsistencyError(
                f"{asset_id}: the file's own valid time {recorded.isoformat()} "
                f"disagrees with the name's {labelled.isoformat()}. The grid cannot "
                "be indexed by a filename that does not describe its contents."
            )

        if density.ndim != 4 or density.shape[0] != 1:
            raise GridInconsistencyError(
                f"{asset_id}: expected density shaped (1, altitude, latitude, longitude), "
                f"found {density.shape}"
            )
        keep = self._valid_latitude_mask(density[0], latitudes, asset_id)
        latitudes = latitudes[keep]
        block: NDArray[np.float64] = np.ascontiguousarray(density[0][:, keep, :])
        if not bool(np.all(block > 0.0)):
            raise GridInconsistencyError(
                f"{asset_id}: {int(np.count_nonzero(block <= 0.0))} non-positive "
                "densities inside the retained latitude block. A zero neutral "
                "density is not a physical value and will not be interpolated through."
            )
        if block.shape != (altitudes_m.size, latitudes.size, longitudes.size):
            raise GridInconsistencyError(
                f"{asset_id}: density shape {block.shape} does not match its "
                "own coordinate axes"
            )
        return labelled, altitudes_m, latitudes, longitudes, block

    def _read_file(self, asset_id: str) -> _Slice:
        """One hour, checked against the axes the first file established."""
        labelled, altitudes_m, latitudes, longitudes, block = self._read_raw(asset_id)
        self._check_axes(asset_id, altitudes_m, latitudes, longitudes)
        return _Slice(time_utc=labelled, asset_id=asset_id, density_kg_m3=block)

    @staticmethod
    def _valid_latitude_mask(
        density: NDArray[np.float64], latitudes: NDArray[np.float64], asset_id: str
    ) -> NDArray[np.bool_]:
        """Drop latitude rows that are identically zero at every level and longitude."""
        keep: NDArray[np.bool_] = np.any(density > 0.0, axis=(0, 2))
        if not keep.any():
            raise GridInconsistencyError(f"{asset_id}: every latitude row is empty")
        first = int(np.argmax(keep))
        last = int(keep.size - 1 - np.argmax(keep[::-1]))
        if not keep[first : last + 1].all():
            dropped = latitudes[first : last + 1][~keep[first : last + 1]]
            raise GridInconsistencyError(
                f"{asset_id}: empty latitude row(s) at {dropped.tolist()} lie inside "
                "the populated band. Only whole rows at the edges of the grid can be "
                "treated as a pole artefact; a hole in the middle is a corrupt file."
            )
        return keep

    def _check_axes(
        self,
        asset_id: str,
        altitudes_m: NDArray[np.float64],
        latitudes: NDArray[np.float64],
        longitudes: NDArray[np.float64],
    ) -> None:
        """Every file after the first must sit on the grid the first established."""
        for actual, expected, label in (
            (altitudes_m, self._altitudes_m, "altitude"),
            (latitudes, self._latitudes_deg, "latitude"),
            (longitudes, self._longitudes_deg, "longitude"),
        ):
            if actual.shape != expected.shape or not bool(
                np.allclose(actual, expected, atol=_AXIS_TOLERANCE, rtol=0.0)
            ):
                raise GridInconsistencyError(
                    f"{asset_id}: {label} axis differs from the grid's first file. "
                    "Files on different axes cannot be interpolated between."
                )

    def _slice(self, asset_id: str) -> _Slice:
        """One hour of the grid, from the cache when it is there."""
        cached = self._cache.get(asset_id)
        if cached is not None:
            self._cache.move_to_end(asset_id)
            return cached
        loaded = self._read_file(asset_id)
        self._cache[asset_id] = loaded
        while len(self._cache) > self._cache_slices:
            self._cache.popitem(last=False)
        return loaded

    # -- public surface ----------------------------------------------------

    @property
    def coverage(self) -> GridCoverage:
        """The domain outside which this provider refuses to answer."""
        return GridCoverage(
            time_start_utc=self._times[0],
            time_end_utc=self._times[-1],
            time_count=len(self._times),
            altitude_min_m=float(self._altitudes_m[0]),
            altitude_max_m=float(self._altitudes_m[-1]),
            altitude_level_count=int(self._altitudes_m.size),
            latitude_min_deg=float(self._latitudes_deg[0]),
            latitude_max_deg=float(self._latitudes_deg[-1]),
            latitude_count=int(self._latitudes_deg.size),
            longitude_count=int(self._longitudes_deg.size),
            cadence_s=self._cadence.total_seconds(),
            gaps=self._gaps,
        )

    @property
    def asset_ids(self) -> tuple[str, ...]:
        """Every stored hour this provider will read, in valid-time order."""
        return self._asset_ids

    @property
    def times_utc(self) -> tuple[_dt.datetime, ...]:
        """The grid's own time axis."""
        return self._times

    @property
    def interpolation_rule(self) -> str:
        """The interpolation, in prose, so a number can be checked by hand."""
        return _INTERPOLATION_RULE

    def total_mass_density(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> float:
        """Total neutral mass density in kg m^-3 at one point.

        ``drivers`` is accepted to satisfy the protocol and is NOT consumed:
        the grid is a finished simulation that already used its own forcing.
        """
        del drivers
        moment = ensure_utc(time_utc, "time_utc")
        if len(self._times) == 1:
            if moment != self._times[0]:
                raise OutsideGridCoverageError(
                    f"{moment.isoformat()} is outside the grid's coverage, which is the "
                    f"single instant {self._times[0].isoformat()}."
                )
            return self._sample(self._slice(self._asset_ids[0]), altitude_m, latitude_deg,
                                longitude_deg)

        if not self._times[0] <= moment <= self._times[-1]:
            raise OutsideGridCoverageError(
                f"{moment.isoformat()} is outside the grid's coverage "
                f"{self._times[0].isoformat()} to {self._times[-1].isoformat()}. "
                "This provider refuses to extrapolate a gridded model in time."
            )
        earlier, later, weight = _bracket(
            self._time_axis, moment.timestamp(), "time", "s"
        )
        # An instant that IS an archived hour is answered from that hour alone,
        # whichever side of the bracket it landed on. Checking this before the
        # gap check matters: the hours at the edges of a hole are real model
        # output and must not be refused along with the hole itself.
        if weight == 0.0:
            return self._sample(
                self._slice(self._asset_ids[earlier]), altitude_m, latitude_deg, longitude_deg
            )
        if weight == 1.0:
            return self._sample(
                self._slice(self._asset_ids[later]), altitude_m, latitude_deg, longitude_deg
            )
        step = self._times[later] - self._times[earlier]
        if step > self._max_gap:
            raise OutsideGridCoverageError(
                f"{moment.isoformat()} falls in a gap in the archive: the nearest "
                f"files either side are {self._times[earlier].isoformat()} and "
                f"{self._times[later].isoformat()}, {step} apart, against a native "
                f"cadence of {self._cadence}. Those hours were never archived, and "
                "this provider will not draw a line across a hole in a storm "
                "simulation and present the result as model output."
            )
        low = self._sample(
            self._slice(self._asset_ids[earlier]), altitude_m, latitude_deg, longitude_deg
        )
        high = self._sample(
            self._slice(self._asset_ids[later]), altitude_m, latitude_deg, longitude_deg
        )
        return low * (1.0 - weight) + high * weight

    def state_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> AtmosphereState:
        """WAM-IPE's total mass density, with species and temperature borrowed.

        Requires a ``composition_baseline``. The grid has no species and no
        temperature; rather than emit an empty species mapping (which the
        atmosphere protocol reserves for "this model does not report it") or a
        fabricated temperature, this raises when no baseline was supplied.
        """
        moment = ensure_utc(time_utc, "time_utc")
        baseline = self.composition_baseline
        if baseline is None:
            raise MissingCompositionBaselineError(
                "the WAM-IPE fixed-height grid carries total mass density only, so "
                "state_at cannot report species or temperature without a "
                "composition_baseline. Construct GriddedStormDensityProvider with "
                "composition_baseline=..., or call total_mass_density(), which is "
                "this provider's native answer."
            )
        rho = self.total_mass_density(moment, altitude_m, latitude_deg, longitude_deg, drivers)
        base = baseline.state_at(moment, altitude_m, latitude_deg, longitude_deg, drivers)
        base_rho = comp.total_species_mass_density(base.number_densities)
        if base_rho <= 0.0:
            raise ValueError(
                f"composition baseline {baseline.name!r} reported a non-positive mass "
                f"density ({base_rho}); it cannot set the composition shape."
            )
        factor = rho / base_rho
        return AtmosphereState(
            time_utc=moment,
            altitude_m=altitude_m,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            temperature_k=base.temperature_k,
            number_densities={s: n * factor for s, n in base.number_densities.items()},
            total_mass_density=rho,
            exospheric_temperature_k=base.exospheric_temperature_k,
            drivers=drivers,
            provider_name=self.name,
            provider_version=self.version,
            status=ValueStatus.MODELLED,
            notes=(
                _INTERPOLATION_RULE,
                _NO_COMPOSITION_NOTE.replace(
                    "the composition baseline's", f"{baseline.name}'s"
                ),
                _NO_TEMPERATURE_NOTE.replace(
                    "the composition baseline's", f"{baseline.name}'s"
                ),
                _DRIVERS_NOT_CONSUMED_NOTE,
            ),
        )

    # -- interpolation -----------------------------------------------------

    def _sample(
        self, slice_: _Slice, altitude_m: float, latitude_deg: float, longitude_deg: float
    ) -> float:
        """Density at one point of one hour, by the documented interpolation rule."""
        k0, k1, wk = _bracket(self._altitudes_m, altitude_m, "altitude", "m")
        j0, j1, wj = _bracket(self._latitudes_deg, latitude_deg, "latitude", "deg")
        i0, i1, wi = _bracket_periodic(self._longitudes_deg, longitude_deg, 360.0)

        corners = slice_.density_kg_m3[np.ix_([k0, k1], [j0, j1], [i0, i1])]
        log_corners = np.log(corners)
        # Altitude first, in log space: density is near-exponential in height.
        at_altitude = np.exp(log_corners[0] * (1.0 - wk) + log_corners[1] * wk)
        # Then bilinear in longitude and latitude on the density itself.
        in_longitude = at_altitude[:, 0] * (1.0 - wi) + at_altitude[:, 1] * wi
        return float(in_longitude[0] * (1.0 - wj) + in_longitude[1] * wj)


# --------------------------------------------------------------------------
# Comparing a gridded storm response with a point model, like for like
# --------------------------------------------------------------------------
#
# The archive holds the storm as it ran and no quiet twin of it, so WAM-IPE's
# storm response cannot be formed the way NRLMSISE-00's is -- by re-running the
# same instant with quiet drivers. The comparable quantity is a ratio against a
# QUIET REFERENCE DAY at the same UTC hour, which holds local solar time and
# season fixed so what remains is the storm rather than the diurnal cycle.
#
# The helpers below take a plain ``(time) -> density`` callable rather than a
# provider, which is the point: the SAME function computes the reference day and
# the peak for the grid and for the point model, so the two enhancements differ
# only in the model that produced the densities.

#: Total neutral mass density in kg m^-3 at one instant, at a point fixed by
#: the caller.
DensitySampler = Callable[[_dt.datetime], float]

QUIET_REFERENCE_RULE = (
    "Enhancement is density at time t divided by the density at the SAME UTC "
    "hour on a quiet reference day, minus one. Holding the UTC hour fixed holds "
    "local solar time fixed at a fixed longitude, so the diurnal cycle divides "
    "out and what is left is the storm. This is NOT the same quantity as a "
    "storm-versus-quiet-driver ratio from a point model re-run at one instant: "
    "the reference day carries its own real (if low) activity, so an "
    "enhancement measured this way is relative to that reference day's "
    "residual background rather than a zero-activity counterfactual (not an "
    "established universal lower bound on the physical storm response)."
)


def quiet_reference_by_hour(
    sample: DensitySampler, day: _dt.date, hours: Sequence[int] = tuple(range(24))
) -> dict[int, float]:
    """Density at each UTC hour of one quiet reference day, keyed by hour."""
    reference: dict[int, float] = {}
    for hour in hours:
        when = _dt.datetime.combine(day, _dt.time(hour), tzinfo=_dt.UTC)
        density = sample(when)
        if density <= 0.0:
            raise ValueError(
                f"reference density at {when.isoformat()} is {density}; a quiet "
                "baseline must be positive"
            )
        reference[hour] = density
    return reference


def enhancement_series(
    sample: DensitySampler,
    times_utc: Sequence[_dt.datetime],
    baseline_by_hour: dict[int, float],
) -> list[tuple[_dt.datetime, float, float]]:
    """``(time, enhancement_fraction, density)`` at every requested instant.

    Raises rather than skipping when a sample has no reference hour: a quietly
    shortened series would move a peak without saying so.
    """
    missing = sorted({t.hour for t in times_utc} - set(baseline_by_hour))
    if missing:
        raise ValueError(
            f"the quiet reference day does not cover UTC hour(s) {missing}, which "
            "the requested samples need. Extend the reference day or drop those samples."
        )
    out: list[tuple[_dt.datetime, float, float]] = []
    for when in times_utc:
        rho = sample(when)
        out.append((when, rho / baseline_by_hour[when.hour] - 1.0, rho))
    return out


def peak_enhancement(
    sample: DensitySampler,
    times_utc: Sequence[_dt.datetime],
    baseline_by_hour: dict[int, float],
) -> tuple[_dt.datetime, float, float]:
    """The largest enhancement in the series, as ``(when, fraction, density)``."""
    series = enhancement_series(sample, times_utc, baseline_by_hour)
    if not series:
        raise ValueError("times_utc is empty; there is nothing to take a peak over")
    return max(series, key=lambda row: row[1])
