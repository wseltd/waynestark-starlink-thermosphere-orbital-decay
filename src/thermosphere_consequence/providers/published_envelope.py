"""Storm-time density from the PUBLISHED event-specific enhancement bands.

The alternative to multiplying NRLMSISE-00's storm response by a made-up
constant. Instead of inventing a factor and naming it after a physics model,
this provider encodes what the literature actually reported for the February
2022 event -- the number, its citation, the altitudes it covers, the times it
covers, and what it explicitly cannot say -- and applies exactly that to a
quiet-baseline atmosphere.

Three properties make it a model rather than a lookup:

  * It is CITED. Every band carries a source and a DOI or URL, and every state
    it returns names the band that produced it. A density from this provider
    can be traced to a page in a paper.

  * It has an explicit DOMAIN. A band speaks for the altitudes and the time
    window its source covers, and for nothing else. Asked outside that domain,
    this provider RAISES. It never extends a published band to an altitude the
    source did not study or to an hour outside the storm the source simulated,
    because a published number quietly extrapolated is no longer a published
    number.

  * It has an explicit INTERPOLATION RULE, stated in prose on every state it
    returns, so that the value at 210 km can be checked by hand against the
    band edges printed in the paper.

What it CANNOT do, stated here because the temptation is to forget it:

  * It does not model storm-time COMPOSITION change. Storms deplete the O/N2
    ratio at auroral latitudes, and none of the sources encoded here report a
    per-species breakdown. The enhancement is therefore applied uniformly to
    every species, which bounds TOTAL mass density and leaves the composition
    exactly as the quiet baseline reported it. A composition-dependent result
    -- a species-weighted drag coefficient, for instance -- computed from this
    provider reflects the QUIET composition scaled up, not the storm's.

  * It does not resolve local solar time or latitude. Every band encoded here
    is a peak or a range reported for the event as a whole. Applying it at one
    point is applying a peak everywhere, which overstates the enhancement away
    from the storm's most disturbed sector.

  * It does not produce a time series. The band is the enhancement REPORTED
    for the storm, applied uniformly across the source's time coverage. It is
    an envelope in time, not a profile through it.

  * It says nothing about temperature. The sources give density; the returned
    temperature is the quiet baseline's, unmodified, and is labelled as such.

Related canonical ticket: P06_08_T040 (envelope comparison against published
enhancement bands).
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Literal, get_args

from ..domain.provenance import ValueStatus
from ..domain.temporal import ensure_utc, require_aware_datetime
from ..protocols.atmosphere import AtmosphereProvider, AtmosphereState, DriverState

__all__ = [
    "ALTITUDE_INTERPOLATION_RULE",
    "AltitudeSemantics",
    "DERIVED_INTERPOLATION_LABEL",
    "PUBLISHED_ENVELOPE_BANDS",
    "EnvelopeBand",
    "OutsidePublishedCoverageError",
    "ProvenanceKind",
    "PublishedEventDensityEnvelopeProvider",
    "Selection",
    "UnsupportedAltitudeProfileError",
    "band_for",
    "bands_covering",
]

ProvenanceKind = Literal["modelled", "observed", "mixed"]
Selection = Literal["low", "nominal", "high"]

#: How to read a band's two enhancement values against its altitude range.
#: ``endpoint_pair`` means the source reports the low value AT the lower
#: altitude and the high value AT the upper one, so linear interpolation
#: between them is what the source supports. ``range_covered`` means the source
#: reports the span as applying across the whole layer, varying with location
#: as much as with height - interpolating it by altitude alone is an ENCODING
#: CONVENTION of this project and is not a published height profile. The field
#: exists because that distinction was not recorded, and a band whose span is a
#: spatial range was read for three review rounds as though it were a profile.
AltitudeSemantics = Literal["endpoint_pair", "range_covered"]

#: Prefixed to every value this module derives rather than reads. An
#: ``endpoint_pair`` band interpolated between its two published altitudes
#: produces a number no paper states, and the number must travel with that
#: fact attached wherever it is reported.
DERIVED_INTERPOLATION_LABEL = (
    "DERIVED BY INTERPOLATION BETWEEN THE SOURCE'S TWO REPORTED ALTITUDES - "
    "an assumption of this project, not a value the paper states"
)

_PROVENANCE_KINDS: tuple[str, ...] = get_args(ProvenanceKind)
_SELECTIONS: tuple[str, ...] = get_args(Selection)
_ALTITUDE_SEMANTICS: tuple[str, ...] = get_args(AltitudeSemantics)

#: Stated on every state a ``range_covered`` band produces, and the reason
#: such a band has no nominal selection at all.
_RANGE_COVERED_CAVEAT = (
    "CONTEXTUAL RANGE OVER A DOMAIN, NOT A PUBLISHED PROFILE: this band's span "
    "is the range the source reports ACROSS LOCATIONS AND ALTITUDES, not a "
    "value at the lower edge and a value at the upper edge. There is therefore "
    "no profile to interpolate, and the 'nominal' selection RAISES rather than "
    "inventing one. The 'low' and 'high' selections return the published "
    "extremes of the whole domain, which are contextual bounds for the event "
    "and NOT a validated interval for any single altitude."
)


class UnsupportedAltitudeProfileError(ValueError):
    """Raised when a band is asked for a value at one altitude it cannot supply.

    A ``range_covered`` band's two numbers are the extremes the source reports
    ACROSS a domain of locations and altitudes. They are not a value at the
    bottom of the layer and a value at the top, so there is no profile to
    interpolate and no published value at any single height. Returning one
    anyway is what put "the published figure at 210 km" at 53.75 % when Fang et
    al. state no figure for 210 km at all.

    A ``ValueError`` subclass, like its sibling below, so a caller validating
    inputs catches the broad case while a caller that specifically wants to
    know "this source supplies no profile" can distinguish it.
    """


class OutsidePublishedCoverageError(ValueError):
    """Raised when a band is asked for a value its source does not support.

    A ``ValueError`` subclass so a caller validating inputs can catch the broad
    case, while a caller that specifically wants to know "the literature does
    not cover this point" can distinguish it.
    """


#: The altitude interpolation rule, stated once and attached to every state.
#:
#: Linear in geodetic altitude from ``enhancement_low`` at the band's lower
#: altitude edge to ``enhancement_high`` at its upper edge. The DIRECTION is
#: not arbitrary: for this event the relative density enhancement grows with
#: height, which is why the two accelerometer figures for the same storm --
#: Swarm-A ~110% at 438 km and GRACE-FO ~120% at 505 km, both from He et al.
#: 2023 -- sit above the 50-125% band reported for 200-400 km. The band edges
#: are the SPREAD the source reported, not an uncertainty that has been
#: resolved; "nominal" places a monotone reading inside that spread and claims
#: nothing more.
#:
#: WHETHER THE INTERPOLATION IS SUPPORTED DEPENDS ON THE BAND, and until the
#: sources were read this was not recorded anywhere. For an ``endpoint_pair``
#: band the source states a value at each altitude edge and interpolating is
#: what it supports. For a ``range_covered`` band -- Fang et al.'s 50-125%, which
#: the paper gives as varying "depending on location and altitude" -- the span
#: is not a height profile at all, and the interpolant is this project's own
#: convention. See ``_RANGE_COVERED_CAVEAT``, which every such band carries.
ALTITUDE_INTERPOLATION_RULE = (
    "For an ENDPOINT_PAIR band, whose source states a value at each altitude "
    "edge: enhancement is linear in geodetic altitude from enhancement_low at "
    "altitude_min_km to enhancement_high at altitude_max_km. The interpolated "
    "value is DERIVED BY THIS PROJECT and is not a figure the paper states; "
    "only the two edges are published. Selections 'low' and 'high' return those "
    "published edges. "
    "For a RANGE_COVERED band, whose span varies with location as well as "
    "height: there is no profile, the 'nominal' selection RAISES "
    "UnsupportedAltitudeProfileError, and only 'low' and 'high' are available, "
    "as contextual bounds over the whole domain. "
    "Outside the band's altitude range the enhancement is UNDEFINED and this "
    "provider raises rather than extrapolating."
)

_COMPOSITION_LIMITATION = (
    "COMPOSITION NOT MODELLED: the enhancement is applied uniformly to every "
    "species because none of the encoded sources reports a per-species storm "
    "response. This bounds TOTAL neutral mass density. Storm-time O/N2 "
    "depletion is NOT represented; the composition returned is the quiet "
    "baseline's composition scaled by a single factor."
)

_TEMPERATURE_LIMITATION = (
    "TEMPERATURE NOT MODELLED: the published bands constrain density only. The "
    "temperature returned is the quiet baseline's, unmodified, and must not be "
    "read as a storm-time temperature."
)

_TIME_ENVELOPE_LIMITATION = (
    "TIME ENVELOPE, NOT A TIME SERIES: the band is the enhancement the source "
    "reports for the event, applied uniformly across the source's time "
    "coverage. It does not trace the storm's rise and recovery."
)


@dataclass(frozen=True, slots=True)
class EnvelopeBand:
    """One published density-enhancement finding, with its whole domain.

    Every field exists because omitting it would let the number be quoted
    somewhere it does not apply. ``altitude_min_km``/``altitude_max_km`` bound
    where the source looked; ``time_start_utc``/``time_end_utc`` bound when;
    ``latitude_applicability`` and ``local_time_limitation`` record the two
    dimensions this encoding deliberately does not resolve, so that a reader
    cannot mistake a global peak for a local value.

    ``enhancement_low``/``enhancement_high`` are fractions of the quiet-time
    neutral mass density: 0.50 means "50% more than quiet", i.e. a factor 1.50.
    """

    citation: str
    doi_or_url: str
    locator: str
    altitude_min_km: float
    altitude_max_km: float
    altitude_semantics: AltitudeSemantics
    time_start_utc: _dt.datetime
    time_end_utc: _dt.datetime
    latitude_applicability: str
    local_time_limitation: str
    enhancement_low: float
    enhancement_high: float
    provenance_kind: ProvenanceKind
    note: str

    def __post_init__(self) -> None:
        if not self.citation:
            raise ValueError("EnvelopeBand.citation must name a source")
        if not self.doi_or_url:
            raise ValueError(f"{self.citation}: doi_or_url must be recorded")
        if not self.doi_or_url.startswith("doi:"):
            raise ValueError(
                f"{self.citation}: doi_or_url must be a resolvable DOI, got "
                f"{self.doi_or_url!r}. A descriptive phrase in this field is how "
                "a figure came to be attributed to a paper that does not report "
                "it; a locator belongs in `locator`."
            )
        if not self.locator:
            raise ValueError(
                f"{self.citation}: locator must name the page, section, figure or "
                "table the band was read from"
            )
        if self.altitude_semantics not in _ALTITUDE_SEMANTICS:
            raise ValueError(
                f"{self.citation}: altitude_semantics must be one of "
                f"{_ALTITUDE_SEMANTICS}, got {self.altitude_semantics!r}"
            )
        if not math.isfinite(self.altitude_min_km) or not math.isfinite(self.altitude_max_km):
            raise ValueError(f"{self.citation}: altitude bounds must be finite")
        if self.altitude_min_km <= 0.0:
            raise ValueError(
                f"{self.citation}: altitude_min_km must be positive, got {self.altitude_min_km}"
            )
        if self.altitude_max_km <= self.altitude_min_km:
            raise ValueError(
                f"{self.citation}: altitude_max_km ({self.altitude_max_km}) must exceed "
                f"altitude_min_km ({self.altitude_min_km}); a degenerate altitude range "
                "cannot be interpolated, and a point-altitude finding belongs in "
                "validation.literature rather than in an interpolating envelope"
            )
        require_aware_datetime(self.time_start_utc, f"{self.citation}: time_start_utc")
        require_aware_datetime(self.time_end_utc, f"{self.citation}: time_end_utc")
        if self.time_end_utc <= self.time_start_utc:
            raise ValueError(
                f"{self.citation}: time_end_utc ({self.time_end_utc.isoformat()}) must be "
                f"after time_start_utc ({self.time_start_utc.isoformat()})"
            )
        if not math.isfinite(self.enhancement_low) or not math.isfinite(self.enhancement_high):
            raise ValueError(f"{self.citation}: enhancement bounds must be finite")
        if self.enhancement_high < self.enhancement_low:
            raise ValueError(
                f"{self.citation}: enhancement_high ({self.enhancement_high}) is below "
                f"enhancement_low ({self.enhancement_low})"
            )
        if self.enhancement_low <= -1.0:
            raise ValueError(
                f"{self.citation}: enhancement_low {self.enhancement_low} would drive "
                "density to zero or below; density enhancements are fractions above -1"
            )
        if self.provenance_kind not in _PROVENANCE_KINDS:
            raise ValueError(
                f"{self.citation}: provenance_kind must be one of {_PROVENANCE_KINDS}, "
                f"got {self.provenance_kind!r}"
            )
        for field_name in ("latitude_applicability", "local_time_limitation", "note"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"{self.citation}: {field_name} must record what this finding "
                    "can and cannot support"
                )

    @property
    def supplies_altitude_profile(self) -> bool:
        """Whether the source states a value AT each edge, so a profile exists.

        True only for ``endpoint_pair``. This is the single predicate that
        decides whether a value may be read at an intermediate altitude, and it
        is derived from ``altitude_semantics`` rather than stored separately so
        the two can never disagree.
        """
        return self.altitude_semantics == "endpoint_pair"

    def covers_altitude(self, altitude_km: float) -> bool:
        """Whether the source's stated domain contains this altitude.

        Coverage is NOT the same as supplying a value here: a
        ``range_covered`` band covers 200-400 km and states nothing at any
        height inside it. ``supplies_altitude_profile`` is the predicate that
        governs whether a number may be produced.
        """
        return self.altitude_min_km <= altitude_km <= self.altitude_max_km

    def covers_time(self, when: _dt.datetime) -> bool:
        """Whether the source's reported window contains this instant."""
        moment = ensure_utc(when, "when")
        return self.time_start_utc <= moment <= self.time_end_utc

    def enhancement_at(self, altitude_km: float, selection: Selection = "nominal") -> float:
        """The published enhancement at one altitude, per the interpolation rule.

        Raises ``OutsidePublishedCoverageError`` outside the band's altitude
        coverage. Clamping to the nearest edge instead would silently turn a
        400 km band into a 600 km claim.
        """
        if selection not in _SELECTIONS:
            raise ValueError(f"selection must be one of {_SELECTIONS}, got {selection!r}")
        if not math.isfinite(altitude_km):
            raise ValueError(f"altitude_km must be finite, got {altitude_km}")
        if not self.covers_altitude(altitude_km):
            raise OutsidePublishedCoverageError(
                f"{altitude_km:.1f} km is outside the published altitude coverage of "
                f"{self.citation} ({self.altitude_min_km:.0f}-{self.altitude_max_km:.0f} km). "
                "This provider does not extrapolate a published band beyond what its "
                "source supports."
            )
        if selection == "low":
            return self.enhancement_low
        if selection == "high":
            return self.enhancement_high
        if not self.supplies_altitude_profile:
            raise UnsupportedAltitudeProfileError(
                f"{self.citation} supplies no altitude profile: its "
                f"{self.enhancement_low:.0%}-{self.enhancement_high:.0%} span is "
                f"the range reported across {self.altitude_min_km:.0f}-"
                f"{self.altitude_max_km:.0f} km AND across locations "
                f"({self.locator}), so the source states no value at "
                f"{altitude_km:.1f} km and this provider will not manufacture "
                "one. Use selection 'low' or 'high', which are the published "
                "extremes of the domain, and report them as a contextual range "
                "rather than as a value at this altitude."
            )
        span = self.altitude_max_km - self.altitude_min_km
        fraction = (altitude_km - self.altitude_min_km) / span
        return self.enhancement_low + fraction * (self.enhancement_high - self.enhancement_low)

    def describe(self) -> str:
        """One line naming the band, its bounds, provenance and where it came from."""
        return (
            f"{self.citation} [{self.provenance_kind}, {self.altitude_semantics}] "
            f"{self.enhancement_low:.0%}-{self.enhancement_high:.0%} over "
            f"{self.altitude_min_km:.0f}-{self.altitude_max_km:.0f} km, "
            f"{self.time_start_utc.isoformat()} to {self.time_end_utc.isoformat()} "
            f"({self.doi_or_url}, {self.locator})"
        )


# The window over which the encoded sources report the February 2022 storm
# response. The storm's main phase ran through 3 February 2022 (peak Kp 5.3,
# peak ap 56 at 09:00 UTC in this project's own ingested GFZ series) and the
# published enhancements describe the 3-4 February disturbance. Beyond this
# window the sources report nothing, so the bands report nothing.
PUBLISHED_WINDOW_START_UTC = _dt.datetime(2022, 2, 3, 0, 0, tzinfo=_dt.UTC)
PUBLISHED_WINDOW_END_UTC = _dt.datetime(2022, 2, 5, 0, 0, tzinfo=_dt.UTC)

_PEAK_OVER_LOCAL_TIME = (
    "The reported figure is a storm peak, not a local-time-resolved value. "
    "This encoding does not resolve local solar time, so the enhancement it "
    "returns is the peak applied at every local time. It therefore OVERSTATES "
    "the enhancement outside the storm's most disturbed local-time sector and "
    "must never be quoted as 'the enhancement at local solar time X'."
)

PUBLISHED_ENVELOPE_BANDS: tuple[EnvelopeBand, ...] = (
    EnvelopeBand(
        citation="Fang et al. 2022 (WAM-IPE)",
        doi_or_url="doi:10.1029/2022SW003193",
        locator="Abstract (p. 1); Conclusions (p. 13)",
        altitude_min_km=200.0,
        altitude_max_km=400.0,
        altitude_semantics="range_covered",
        time_start_utc=PUBLISHED_WINDOW_START_UTC,
        time_end_utc=PUBLISHED_WINDOW_END_UTC,
        latitude_applicability=(
            "Reported for the storm as simulated globally by WAM-IPE. This "
            "encoding applies it at every latitude, which overstates it at low "
            "latitudes and may understate it in the auroral zone where Joule "
            "heating concentrates."
        ),
        local_time_limitation=_PEAK_OVER_LOCAL_TIME,
        enhancement_low=0.50,
        enhancement_high=1.25,
        provenance_kind="modelled",
        note=(
            "WAM-IPE simulation of the February 2022 storm; the band contains the "
            "210 km Starlink staging altitude and is therefore altitude-comparable "
            "there. It is a MODEL result: agreement with it is agreement between "
            "two models, not observational corroboration. " + _RANGE_COVERED_CAVEAT
        ),
    ),
    EnvelopeBand(
        citation="He et al. 2023 (CTIPe/MAGE average)",
        doi_or_url="doi:10.1029/2023SW003521",
        locator="Section 3 (p. 15); restated in the Abstract and Key Points (p. 1)",
        altitude_min_km=200.0,
        altitude_max_km=300.0,
        altitude_semantics="endpoint_pair",
        time_start_utc=PUBLISHED_WINDOW_START_UTC,
        time_end_utc=PUBLISHED_WINDOW_END_UTC,
        latitude_applicability=(
            "A global-averaged maximum of a zonal-mean field, so it carries no "
            "latitude breakdown this encoding can reproduce. Applied at every "
            "latitude here, which overstates it away from the disturbed sector."
        ),
        local_time_limitation=_PEAK_OVER_LOCAL_TIME,
        enhancement_low=0.35,
        enhancement_high=0.60,
        provenance_kind="modelled",
        note=(
            "THIS BAND WAS PREVIOUSLY ATTRIBUTED TO 'Girgis et al. 2023' with no "
            "DOI, and its provenance kind was recorded as 'mixed'. Both were "
            "wrong: the figure is He et al. 2023's, and it is the average of that "
            "paper's two THEORETICAL models, CTIPe and MAGE - purely modelled, "
            "and not a consensus of its six models. 35% is the value AT 200 km "
            "and 60% the value AT 300 km, so the two edges are published and a "
            "value between them is DERIVED: 37.5% at 210 km is this project's "
            "interpolation assumption, not a figure He et al. state."
        ),
    ),
    EnvelopeBand(
        citation="He et al. 2023 (MSIS2.0)",
        doi_or_url="doi:10.1029/2023SW003521",
        locator="Section 3 (pp. 14-15), discussion of Figures 9 (200 km) and 10 (300 km)",
        altitude_min_km=200.0,
        altitude_max_km=300.0,
        altitude_semantics="endpoint_pair",
        time_start_utc=PUBLISHED_WINDOW_START_UTC,
        time_end_utc=PUBLISHED_WINDOW_END_UTC,
        latitude_applicability=(
            "A global-averaged maximum of a zonal-mean field, reported for 200 and "
            "300 km without a latitude breakdown. Applied at every latitude here."
        ),
        local_time_limitation=_PEAK_OVER_LOCAL_TIME,
        enhancement_low=0.10,
        enhancement_high=0.25,
        provenance_kind="modelled",
        note=(
            "The MSIS family's OWN published storm response for this event. He "
            "et al. report ~10%-15% for MSIS2.0 at 200 km and ~20%-25% at "
            "300 km, and those two rows are what "
            "validation.literature carries. This band exists only to scale a "
            "baseline atmosphere between them, and ANY VALUE IT RETURNS AT AN "
            "INTERMEDIATE ALTITUDE IS DERIVED: about 11%-16% at 210 km is this "
            "project's interpolation assumption, not a figure He et al. state. "
            "An earlier encoding put 0.20 at the 200 km edge, which the source "
            "does not support and which placed the supposed published value at "
            "210 km near 20.5%. He et al. report MSIS2.0 as underestimating the "
            "enhancement relative to their theoretical models, so landing inside "
            "this band is not corroboration: it is agreement with the model "
            "family already in use, in the direction that family is reported to "
            "be wrong."
        ),
    ),
)


def band_for(citation_fragment: str) -> EnvelopeBand:
    """The single published band whose citation contains ``citation_fragment``.

    Raises on no match and on an ambiguous match, rather than picking one: a
    silently chosen band would put the wrong citation on every number
    downstream.
    """
    if not citation_fragment:
        raise ValueError("citation_fragment must be a non-empty string")
    needle = citation_fragment.casefold()
    matches = [b for b in PUBLISHED_ENVELOPE_BANDS if needle in b.citation.casefold()]
    if not matches:
        known = ", ".join(b.citation for b in PUBLISHED_ENVELOPE_BANDS)
        raise KeyError(f"no published band matches {citation_fragment!r}; known bands: {known}")
    if len(matches) > 1:
        named = ", ".join(b.citation for b in matches)
        raise KeyError(f"{citation_fragment!r} is ambiguous; it matches {named}")
    return matches[0]


def bands_covering(altitude_km: float, when: _dt.datetime) -> tuple[EnvelopeBand, ...]:
    """Every published band whose altitude AND time coverage contains the point."""
    moment = ensure_utc(when, "when")
    return tuple(
        b
        for b in PUBLISHED_ENVELOPE_BANDS
        if b.covers_altitude(altitude_km) and b.covers_time(moment)
    )


class PublishedEventDensityEnvelopeProvider:
    """Quiet-baseline atmosphere scaled by ONE published enhancement band.

    The band is chosen at construction and never inferred, because the citation
    is part of the answer: "density from Fang et al. 2022's band" and "density
    from He et al. 2023's band" are different claims and must not be blended
    into an unattributed number.

    The baseline is evaluated under QUIET drivers, obtained by passing the
    caller's observed drivers through ``quiet_drivers_for``. This matters: the
    published enhancement is defined as the storm-time density divided by the
    quiet-time density. Applying it to a baseline that already carries
    NRLMSISE-00's own storm response would count the storm twice.
    """

    def __init__(
        self,
        baseline: AtmosphereProvider,
        quiet_drivers_for: object,
        *,
        band: EnvelopeBand,
        selection: Selection = "nominal",
    ) -> None:
        if selection not in _SELECTIONS:
            raise ValueError(f"selection must be one of {_SELECTIONS}, got {selection!r}")
        self.baseline = baseline
        self.quiet_drivers_for = quiet_drivers_for
        self.band = band
        self.selection: Selection = selection
        self.name = f"published_envelope[{band.citation}]_{selection}"
        self.version = f"{baseline.version}+published_envelope/{selection}"

    def enhancement_at(self, altitude_m: float, time_utc: _dt.datetime) -> float:
        """The published enhancement fraction this provider would apply.

        Exposed separately from ``state_at`` so a report can quote the applied
        enhancement without also constructing an atmosphere, and so the
        coverage check can be exercised on its own.
        """
        moment = ensure_utc(time_utc, "time_utc")
        if not self.band.covers_time(moment):
            raise OutsidePublishedCoverageError(
                f"{moment.isoformat()} is outside the published time coverage of "
                f"{self.band.citation} ({self.band.time_start_utc.isoformat()} to "
                f"{self.band.time_end_utc.isoformat()}). This provider does not "
                "extrapolate a published band beyond what its source supports."
            )
        return self.band.enhancement_at(altitude_m / 1000.0, self.selection)

    def state_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> AtmosphereState:
        """Quiet baseline at this point, multiplied by the published enhancement.

        Raises ``OutsidePublishedCoverageError`` outside the band's altitude or
        time coverage. Both checks run before the baseline model is called, so
        an out-of-coverage request costs nothing and cannot leave a half-formed
        state behind.
        """
        moment = ensure_utc(time_utc, "time_utc")
        if altitude_m < 0:
            raise ValueError(f"altitude must be non-negative, got {altitude_m}")
        enhancement = self.enhancement_at(altitude_m, moment)
        factor = 1.0 + enhancement

        quiet_drivers = self.quiet_drivers_for(drivers)  # type: ignore[operator]
        quiet = self.baseline.state_at(
            time_utc, altitude_m, latitude_deg, longitude_deg, quiet_drivers
        )

        scaled = {symbol: n * factor for symbol, n in quiet.number_densities.items()}
        # The model's own total scales by exactly the same factor as every
        # species, so carrying it through keeps the density-closure fraction
        # identical to the baseline's rather than silently changing a
        # diagnostic that has nothing to do with the storm.
        total = None if quiet.total_mass_density is None else quiet.total_mass_density * factor

        return AtmosphereState(
            time_utc=moment,
            altitude_m=altitude_m,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            temperature_k=quiet.temperature_k,
            number_densities=scaled,
            total_mass_density=total,
            exospheric_temperature_k=quiet.exospheric_temperature_k,
            drivers=quiet_drivers,
            provider_name=self.name,
            provider_version=self.version,
            status=ValueStatus.MODELLED,
            notes=(
                f"PUBLISHED ENVELOPE, status MODELLED: quiet baseline from "
                f"{self.baseline.name} multiplied by {factor:.4f}, i.e. a "
                f"density enhancement of {enhancement:.1%} from "
                f"{self.band.citation} ({self.band.doi_or_url}), selection "
                f"'{self.selection}'."
                + (
                    f" {DERIVED_INTERPOLATION_LABEL}: the published values are "
                    f"{self.band.enhancement_low:.0%} at "
                    f"{self.band.altitude_min_km:.0f} km and "
                    f"{self.band.enhancement_high:.0%} at "
                    f"{self.band.altitude_max_km:.0f} km."
                    if self.selection == "nominal"
                    else " This is a published edge of the band, read directly."
                ),
                f"Band used: {self.band.describe()}",
                f"Interpolation rule: {ALTITUDE_INTERPOLATION_RULE}",
                f"Local-time limitation: {self.band.local_time_limitation}",
                f"Latitude applicability: {self.band.latitude_applicability}",
                _TIME_ENVELOPE_LIMITATION,
                _COMPOSITION_LIMITATION,
                _TEMPERATURE_LIMITATION,
                f"Source note: {self.band.note}",
            ),
        )
