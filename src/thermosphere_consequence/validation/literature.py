"""Comparison of a computed storm-time density enhancement against published figures.

These findings are what a result is checked AGAINST. They are never inputs: no
value here is used to calibrate, scale or select anything in the model, and the
comparison produces prose and a relation, not a score.

WHY THIS TABLE HOLDS ORIGINAL ALTITUDE ROWS
--------------------------------------------
He et al. 2023 report MSIS2.0 at 200 km and at 300 km as two separate results.
An earlier encoding merged them into one 200-300 km band, which let a linear
interpolation produce a "published value at 210 km" that no paper states. That
number then carried the weight of a source figure. Both altitude rows are now
kept as the source reports them, degenerate in altitude, and nothing in this
module interpolates between published altitudes. Where an interpolation is
genuinely wanted for a density-scaling context it lives in
``providers.published_envelope``, where it is labelled a derived assumption.

Fang et al.'s 50 %-125 % is not two altitude endpoints at all: the paper gives
it as varying "depending on location and altitude" over 200-400 km. It is a
CONTEXTUAL RANGE over a domain, and this module refuses to read a value for any
single altitude out of it.

WHAT A COMPARISON HAS TO CLEAR BEFORE IT MEANS ANYTHING
--------------------------------------------------------
Three separate things must match before a BELOW/WITHIN/ABOVE relation is
evidence about a model rather than an artefact of definitions:

  * **Altitude.** The published figure must be reported at the altitude being
    compared. Nearby is not the same: the storm-time enhancement varies with
    height, which is the whole reason the two accelerometer records for this
    storm (Swarm-A ~438 km, GRACE-FO ~505 km) cannot certify 210 km.

  * **Quantity.** This project's headline is a PEAK OVER TIME AT ONE POINT.
    Every modelled published figure for this layer is a maximum of a global or
    zonal MEAN field. A point peak is expected to exceed the maximum of a mean
    field containing it, so their difference measures neither bias nor its
    direction.

  * **Baseline.** An enhancement is a ratio to something. This project uses a
    fixed quiet-ap counterfactual; He et al. use the 1-2 February reference
    days; Fang et al. subtract a 10-day model mean at the same UT. Three
    different denominators.

``compare()`` checks all three from recorded metadata. When any fails it
returns ``relation = "NOT_COMPARABLE"`` and states which check failed. The
positional fact - whether the number happens to fall below, inside or above the
published interval - is still reported, in ``positional_relation``, because
suppressing evidence is its own distortion; but it is not a verdict and the
summary says so.

Related canonical ticket: P06_08_T040 (envelope comparison against published
enhancement bands).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, get_args

__all__ = [
    "AltitudeSemantics",
    "BaselineKind",
    "Comparison",
    "EvidenceBasis",
    "FindingKind",
    "MeasureKind",
    "PUBLISHED_DENSITY_ENHANCEMENT",
    "PROJECT_BASELINE_KIND",
    "PROJECT_QUANTITY_DEFINITION",
    "PublicationStatus",
    "PublishedFinding",
    "QuantityDefinition",
    "SOURCES",
    "STARLINK_STAGING_ALTITUDE_KM",
    "SUPPORT_VERIFICATION_NOTE",
    "SourceRecord",
    "compare",
    "independent_source_count",
    "source_for",
    "verdict_text",
]

FindingKind = Literal["modelled", "observed", "mixed"]
VALID_KINDS: tuple[str, ...] = get_args(FindingKind)

PublicationStatus = Literal["peer_reviewed", "preprint", "software", "dataset", "internal"]
VALID_STATUSES: tuple[str, ...] = get_args(PublicationStatus)

#: What a published number actually measures. Two numbers with the same units
#: at the same altitude are still not comparable if one is a point peak and the
#: other the maximum of a global mean.
QuantityDefinition = Literal[
    # One location, one altitude, maximised over the event window. This is what
    # THIS project computes.
    "point_peak",
    # The largest value in a global field at one instant and altitude.
    "global_field_max",
    # The largest value, over the storm days, of a global- or zonal-MEAN field.
    "global_mean_max",
    # The span of values the source reports across locations and altitudes; not
    # a single quantity at all, and not readable at one altitude.
    "spatial_range",
    # Accelerometer-derived density sampled along one spacecraft's orbit.
    "in_situ_orbit_max",
]
VALID_DEFINITIONS: tuple[str, ...] = get_args(QuantityDefinition)

#: What the enhancement is measured RELATIVE TO. An enhancement is a ratio, and
#: two ratios with different denominators are different quantities.
BaselineKind = Literal[
    # This project: the same instant and place re-evaluated with a fixed quiet
    # ap, holding solar flux. A counterfactual, not an observation.
    "fixed_quiet_ap_counterfactual",
    # Named quiet calendar days before the storm.
    "quiet_reference_days",
    # A rolling multi-day mean of the same model at the same UT.
    "n_day_model_mean",
    # A pre-storm in-situ measurement from the same instrument.
    "pre_storm_in_situ",
]
VALID_BASELINES: tuple[str, ...] = get_args(BaselineKind)

#: What kind of number this is. A 50 % increase, a ratio of 1.5, "150 % of
#: baseline" and a 50-percentage-point difference are four different statements
#: and have been confused for one another in this literature before.
MeasureKind = Literal[
    "fractional_increase",
    "density_ratio",
    "fraction_of_baseline",
    "percentage_point_difference",
]
VALID_MEASURES: tuple[str, ...] = get_args(MeasureKind)

#: What produced the number. The PAPER is the citable source; the instrument,
#: model or dataset is what generated the value the paper reports. Collapsing
#: these is how an accelerometer came to be cited as though it were a
#: publication.
EvidenceBasis = Literal["model_simulation", "instrument_measurement", "derived_dataset"]
VALID_BASES: tuple[str, ...] = get_args(EvidenceBasis)

#: How to read ``low``/``high`` against ``altitude_km_range``.
AltitudeSemantics = Literal[
    # A degenerate range: one altitude, one value or one reported interval at
    # that altitude. Every modelled row in this table is of this kind, because
    # the sources report per-altitude results and this module keeps them that
    # way.
    "single_altitude",
    # The source reports this span as applying ACROSS the whole range, varying
    # with location as much as with height. It has no value at any one
    # altitude, and nothing here will produce one.
    "range_covered",
]
VALID_ALTITUDE_SEMANTICS: tuple[str, ...] = get_args(AltitudeSemantics)

#: Insertion perigee of Starlink Group 4-7, the altitude every comparison is
#: ultimately about. Repeated here as a plain number rather than imported from
#: the event module so this table stays usable for any event.
STARLINK_STAGING_ALTITUDE_KM = 210.0

#: What this project's own headline enhancement is, declared once so callers
#: cannot drift from it.
PROJECT_QUANTITY_DEFINITION: QuantityDefinition = "point_peak"
PROJECT_BASELINE_KIND: BaselineKind = "fixed_quiet_ap_counterfactual"

ENHANCEMENT_QUANTITY = "storm-time neutral mass density enhancement"
ENHANCEMENT_UNITS = "fraction of the quiet-time neutral mass density"

#: The honest state of citation-to-claim verification, recorded rather than
#: asserted. Every figure below was read at source from the stored PDF by the
#: agent that wrote this table. That is verification by the AUTHORING role, not
#: by an independent one, and it does not satisfy a protocol rule requiring a
#: verifier independent of the author.
SUPPORT_VERIFICATION_NOTE = (
    "Every figure in this table was read at source from the PDF pinned in "
    "data/raw/literature by the agent that wrote the table. Grade I0_SELF: this "
    "is the AUTHORING role verifying its own citations, which establishes that "
    "the number is in the paper and does NOT discharge a requirement for an "
    "independent verifier. No finding here may claim a higher grade without "
    "naming the external verifier who assigned it."
)

SELF_VERIFICATION_GRADE = "I0_SELF"


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Identity and retrieval record for one cited work.

    The fields mirror ``drivers.raw_store.RawAssetMetadata`` deliberately: a
    paper this project's conclusions rest on is evidence, and it is pinned the
    same way a driver file is. ``sha256`` is of the stored PDF, so a reader who
    re-fetches can confirm they are reading the bytes that were read here.
    """

    key: str
    citation: str
    doi: str
    publication_status: PublicationStatus
    retrieval_url: str
    retrieved_at_utc: str
    stored_asset: str
    sha256: str
    licence: str

    def __post_init__(self) -> None:
        for field_name in (
            "key",
            "citation",
            "doi",
            "retrieval_url",
            "retrieved_at_utc",
            "stored_asset",
            "licence",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"SourceRecord.{field_name} must be recorded, got empty")
        if self.publication_status not in VALID_STATUSES:
            raise ValueError(
                f"{self.key}: publication_status must be one of {VALID_STATUSES}, "
                f"got {self.publication_status!r}"
            )
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError(
                f"{self.key}: sha256 must be 64 lowercase hex characters of the "
                f"stored payload, got {self.sha256!r}"
            )


SOURCES: dict[str, SourceRecord] = {
    "fang_2022": SourceRecord(
        key="fang_2022",
        citation=(
            "Fang, T.-W., Kubaryk, A., Goldstein, D., Li, Z., Fuller-Rowell, T., "
            "Millward, G., et al. (2022). Space weather environment during the "
            "SpaceX Starlink satellite loss in February 2022. Space Weather, 20, "
            "e2022SW003193."
        ),
        doi="10.1029/2022SW003193",
        publication_status="peer_reviewed",
        retrieval_url="https://repository.library.noaa.gov/view/noaa/53091/noaa_53091_DS1.pdf",
        retrieved_at_utc="2026-09-11T11:49:00+00:00",
        stored_asset="data/raw/literature/fang_2022_sw003193.pdf",
        sha256="8404d1889e2fd56a042d9bdde217d3c242e2ad660e78654b6ecd81dcec85e7f3",
        licence="CC BY 4.0",
    ),
    "he_2023": SourceRecord(
        key="he_2023",
        citation=(
            "He, J., Astafyeva, E., Yue, X., Pedatella, N. M., Lin, D., "
            "Fuller-Rowell, T. J., et al. (2023). Comparison of empirical and "
            "theoretical models of the thermospheric density enhancement during "
            "the 3-4 February 2022 geomagnetic storm. Space Weather, 21, "
            "e2023SW003521."
        ),
        doi="10.1029/2023SW003521",
        publication_status="peer_reviewed",
        retrieval_url="https://hal.science/hal-04248456/document",
        retrieved_at_utc="2026-09-11T11:49:00+00:00",
        stored_asset="data/raw/literature/he_2023_sw003521.pdf",
        sha256="802f9012162195fab4ea341d5c7095b76ff6f0ecbc29e61b0b6dfd563e3e1d67",
        licence="CC BY-NC-ND 4.0 (publisher of record)",
    ),
}


def source_for(key: str) -> SourceRecord:
    """The source record for a key, raising with the known keys if absent."""
    try:
        return SOURCES[key]
    except KeyError:
        known = ", ".join(sorted(SOURCES))
        raise KeyError(f"no source record for {key!r}; known sources: {known}") from None


@dataclass(frozen=True, slots=True)
class PublishedFinding:
    """One published figure, with everything needed to know what it can support.

    ``low`` and ``high`` may be equal: an accelerometer-derived enhancement is a
    single reported number, not a band, and widening it into an interval would
    invent a spread the source never claimed. Where they differ at a single
    altitude the interval is the source's own - typically its two storm days -
    and not an altitude profile.

    ``locator`` and ``quoted_text`` exist so that a reader can check the number
    against the paper without re-reading the paper: the locator says where to
    look and the quotation says what is there. A finding whose quotation does
    not contain its own figures is the defect this pair is here to prevent.

    ``evidence_basis``, ``instrument``, ``model`` and ``dataset_or_method``
    separate four things that collapsed into one another in the previous
    encoding: the PAPER that reports a result, the INSTRUMENT that measured it,
    the MODEL that produced it, and the processing that turned raw counts into
    a density. Two findings from one paper are one citable source and two
    pieces of evidence; ``independent_source_count`` exists so that is never
    counted the other way round.
    """

    source: str
    source_key: str
    quantity: str
    low: float
    high: float
    units: str
    measure_kind: MeasureKind
    kind: FindingKind
    evidence_basis: EvidenceBasis
    instrument: str
    model: str
    dataset_or_method: str
    altitude_km_range: tuple[float, float]
    altitude_semantics: AltitudeSemantics
    quantity_definition: QuantityDefinition
    baseline_kind: BaselineKind
    baseline_definition: str
    averaging_domains: str
    locator: str
    quoted_text: str
    support_verified_by: str
    support_verification_grade: str
    note: str

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("PublishedFinding.source must carry a citation")
        source_for(self.source_key)
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError(f"{self.source}: bounds must be finite, got {self.low}, {self.high}")
        if self.high < self.low:
            raise ValueError(
                f"{self.source}: high bound {self.high} is below low bound {self.low}"
            )
        for field_name, value, allowed in (
            ("kind", self.kind, VALID_KINDS),
            ("measure_kind", self.measure_kind, VALID_MEASURES),
            ("evidence_basis", self.evidence_basis, VALID_BASES),
            ("quantity_definition", self.quantity_definition, VALID_DEFINITIONS),
            ("baseline_kind", self.baseline_kind, VALID_BASELINES),
            ("altitude_semantics", self.altitude_semantics, VALID_ALTITUDE_SEMANTICS),
        ):
            if value not in allowed:
                raise ValueError(
                    f"{self.source}: {field_name} must be one of {allowed}, got {value!r}"
                )
        lo, hi = self.altitude_km_range
        if hi < lo:
            raise ValueError(f"{self.source}: altitude range {self.altitude_km_range} is inverted")
        if (lo == hi) != (self.altitude_semantics == "single_altitude"):
            raise ValueError(
                f"{self.source}: a degenerate altitude range and 'single_altitude' "
                "semantics must be used together; got range "
                f"{self.altitude_km_range} with {self.altitude_semantics!r}. A "
                "non-degenerate range must be declared 'range_covered', which "
                "means no value can be read from it at one altitude."
            )
        if self.evidence_basis == "instrument_measurement" and not self.instrument:
            raise ValueError(f"{self.source}: an instrument measurement must name its instrument")
        if self.evidence_basis == "model_simulation" and not self.model:
            raise ValueError(f"{self.source}: a model simulation must name its model")
        for field_name in (
            "baseline_definition",
            "averaging_domains",
            "dataset_or_method",
            "locator",
            "quoted_text",
            "support_verified_by",
            "support_verification_grade",
            "note",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{self.source}: {field_name} must be recorded, got empty")

    @property
    def reports_a_single_altitude(self) -> bool:
        """Whether this figure is stated at one altitude the source names."""
        return self.altitude_semantics == "single_altitude"

    @property
    def reported_altitude_km(self) -> float | None:
        """The altitude the source states, or None for a domain-wide range."""
        lo, hi = self.altitude_km_range
        return lo if lo == hi else None

    def contains_altitude(self, altitude_km: float) -> bool:
        """Whether this finding states a value AT the given altitude.

        For a ``range_covered`` finding this is always False even inside the
        range: the source reports a span over a domain, not a value at a
        height, and treating "inside the domain" as "reported at this altitude"
        is exactly the error that produced a published value at 210 km which no
        paper states.
        """
        if self.altitude_semantics == "range_covered":
            return False
        lo, hi = self.altitude_km_range
        return lo <= altitude_km <= hi

    def altitude_gap_km(self, altitude_km: float) -> float:
        """Distance from an altitude to the nearest edge of the reported range."""
        lo, hi = self.altitude_km_range
        if altitude_km < lo:
            return lo - altitude_km
        if altitude_km > hi:
            return altitude_km - hi
        return 0.0


def independent_source_count(findings: Sequence[PublishedFinding]) -> int:
    """How many distinct PAPERS these findings come from.

    Six findings from two papers are two independent sources. Reporting them as
    six would overstate the weight of the comparison, and the previous encoding
    invited exactly that by giving three of He et al.'s results three different
    source strings, one of them another author's name.
    """
    return len({f.source_key for f in findings})


_HE_REFERENCE_DAYS = (
    "Enhancement relative to the 1 and 2 February 2022 reference days (He et "
    "al. section 3: 'The reference day is again chosen as 1 and 2 February'). "
    "This project's quiet reference is a fixed quiet-ap COUNTERFACTUAL at the "
    "same instant and place, which is a different denominator."
)

_FANG_TEN_DAY_MEAN = (
    "Enhancement relative to a 10-DAY AVERAGED density from the same model at "
    "the same UT (Fang et al. section 4: 'The anomalies were calculated by "
    "subtracting the current values from the 10-day averaged densities from "
    "each model at the same UT'). This is neither this project's quiet-ap "
    "counterfactual nor He et al.'s reference days."
)

_HE_GLOBAL_MEAN_MAX = (
    "Averaged over SPACE (a zonal mean, then a global average of that field), "
    "then MAXIMISED over the two storm days, 3 and 4 February. Not averaged "
    "over models, and not a local value at any latitude or longitude."
)

_OBSERVATIONAL_ALTITUDE_CAVEAT = (
    "Swarm-A and GRACE-FO together place the observational evidence for this "
    "storm 230-300 km ABOVE the 210 km Starlink staging altitude, so they "
    "validate the storm RESPONSE and cannot certify density at 210 km."
)

_SELF = "implementing agent, authoring role (read at source from the stored PDF)"

_HE_ACCELEROMETER_QUOTE = (
    "Observations at higher altitudes, by Swarm-A (~438 km, 09/21 Local Time "
    "[LT]) and the Gravity Recovery and Climate Experiment Follow-On (~505 km, "
    "06/18 LT) missions show that during the main phase of the storms the "
    "neutral mass density increased by 110% and 120%, respectively."
)

PUBLISHED_DENSITY_ENHANCEMENT: tuple[PublishedFinding, ...] = (
    PublishedFinding(
        source="Fang et al. 2022 (WAM-IPE, 200-400 km domain), doi:10.1029/2022SW003193",
        source_key="fang_2022",
        quantity=ENHANCEMENT_QUANTITY,
        low=0.50,
        high=1.25,
        units=ENHANCEMENT_UNITS,
        measure_kind="fractional_increase",
        kind="modelled",
        evidence_basis="model_simulation",
        instrument="",
        model="WAM-IPE (Whole Atmosphere Model - Ionosphere Plasmasphere Electrodynamics)",
        dataset_or_method=(
            "Operational WAM-IPE output on a fixed-height grid at 10 km altitude "
            "spacing, 4 deg longitude by 2 deg latitude, standardised to 10-minute "
            "cadence; total density is the sum of O, O2 and N2 mass densities."
        ),
        altitude_km_range=(200.0, 400.0),
        altitude_semantics="range_covered",
        quantity_definition="spatial_range",
        baseline_kind="n_day_model_mean",
        baseline_definition=_FANG_TEN_DAY_MEAN,
        averaging_domains=(
            "NOT AVERAGED. The 50-125 % span is the RANGE the simulated "
            "enhancement takes across locations and altitudes over the event. "
            "It is not a mean, not a profile, and it has no value at any single "
            "altitude."
        ),
        locator="Abstract (p. 1); Conclusions (p. 13)",
        quoted_text=(
            "Based on WAM-IPE simulations, the neutral density enhancement for "
            "this event ranged from 50% to 125%, depending on location and "
            "altitude."
        ),
        support_verified_by=_SELF,
        support_verification_grade=SELF_VERIFICATION_GRADE,
        note=(
            "CONTEXTUAL RANGE OVER A DOMAIN, NOT AN ALTITUDE PROFILE. The source "
            "gives the span as varying with location AND altitude, so no value "
            "for 210 km can be read out of it and none is produced. It is "
            "retained because it is the headline figure of the paper that "
            "studied this event, and it is what makes the model-family "
            "difference visible: the same paper reports NRLMSISE-00 at up to "
            "20 % for the same event."
        ),
    ),
    PublishedFinding(
        source="Fang et al. 2022 (NRLMSISE-00 at 210 km), doi:10.1029/2022SW003193",
        source_key="fang_2022",
        quantity=ENHANCEMENT_QUANTITY,
        low=0.0,
        high=0.20,
        units=ENHANCEMENT_UNITS,
        measure_kind="fractional_increase",
        kind="modelled",
        evidence_basis="model_simulation",
        instrument="",
        model="NRLMSISE-00",
        dataset_or_method=(
            "NRLMSISE-00 driven by daily F10.7 and ap converted from real-time "
            "3-hourly Kp, on the same fixed-height grid as the WAM-IPE output; "
            "anomaly field shown at 210 km for 11:50 UT on 4 February 2022."
        ),
        altitude_km_range=(210.0, 210.0),
        altitude_semantics="single_altitude",
        quantity_definition="global_field_max",
        baseline_kind="n_day_model_mean",
        baseline_definition=_FANG_TEN_DAY_MEAN,
        averaging_domains=(
            "NOT AVERAGED over space or time: 'up to 20 %' is the largest value "
            "reached anywhere in the global anomaly field at one instant, "
            "11:50 UT on 4 February 2022."
        ),
        locator="Section 4 (p. 7), discussion of Figure 5 (210 km, 11:50 UT 4 Feb 2022)",
        quoted_text=(
            "In the anomaly plots, WAM suggested at least a 50% density increase "
            "in most regions and more than 100% increase in some specific areas "
            "but MSIS-00 only showed up to 20% enhancement in the density."
        ),
        support_verified_by=_SELF,
        support_verification_grade=SELF_VERIFICATION_GRADE,
        note=(
            "The only published figure in this table stated at exactly the "
            "210 km staging altitude, and the only one for the same model this "
            "project runs. Reported as an upper bound ('up to 20 %'), hence a "
            "0-20 % interval. Its baseline is a 10-day model mean and its "
            "quantity is the maximum of a global field at one instant, so it is "
            "still not like-for-like with a point peak over time against a "
            "quiet-ap counterfactual."
        ),
    ),
    PublishedFinding(
        source="He et al. 2023 (MSIS2.0 at 200 km), doi:10.1029/2023SW003521",
        source_key="he_2023",
        quantity=ENHANCEMENT_QUANTITY,
        low=0.10,
        high=0.15,
        units=ENHANCEMENT_UNITS,
        measure_kind="fractional_increase",
        kind="modelled",
        evidence_basis="model_simulation",
        instrument="",
        model="NRLMSIS 2.0",
        dataset_or_method=(
            "NRLMSIS 2.0 driven by F10.7 and the 7-value ap array on a 2.5 deg "
            "by 2.5 deg, 10 km, 10-minute grid; zonal-mean enhancement field at "
            "200 km (He et al. Figure 9)."
        ),
        altitude_km_range=(200.0, 200.0),
        altitude_semantics="single_altitude",
        quantity_definition="global_mean_max",
        baseline_kind="quiet_reference_days",
        baseline_definition=_HE_REFERENCE_DAYS,
        averaging_domains=_HE_GLOBAL_MEAN_MAX,
        locator="Section 3 (p. 14), discussion of Figure 9 (200 km)",
        quoted_text=(
            "For MSIS2.0, it shows ~10%-15% enhancement during the two storm "
            "days, while no obvious enhancement is shown in the recovery phase."
        ),
        support_verified_by=_SELF,
        support_verification_grade=SELF_VERIFICATION_GRADE,
        note=(
            "The 200 km row as the source reports it. AN EARLIER ENCODING MERGED "
            "THIS WITH THE 300 km ROW into a single 200-300 km band and read "
            "20 % at this altitude, which the source does not state. The two "
            "rows are kept separate and nothing interpolates between them."
        ),
    ),
    PublishedFinding(
        source="He et al. 2023 (MSIS2.0 at 300 km), doi:10.1029/2023SW003521",
        source_key="he_2023",
        quantity=ENHANCEMENT_QUANTITY,
        low=0.20,
        high=0.25,
        units=ENHANCEMENT_UNITS,
        measure_kind="fractional_increase",
        kind="modelled",
        evidence_basis="model_simulation",
        instrument="",
        model="NRLMSIS 2.0",
        dataset_or_method=(
            "As the 200 km row; zonal-mean enhancement field at 300 km (He et "
            "al. Figure 10). The 20 % and 25 % are the 3 and 4 February values."
        ),
        altitude_km_range=(300.0, 300.0),
        altitude_semantics="single_altitude",
        quantity_definition="global_mean_max",
        baseline_kind="quiet_reference_days",
        baseline_definition=_HE_REFERENCE_DAYS,
        averaging_domains=_HE_GLOBAL_MEAN_MAX,
        locator="Section 3 (pp. 14-15), discussion of Figure 10 (300 km)",
        quoted_text=(
            "During the two storm days, the global averaged maximum density "
            "enhancement of DTM2013 can reach ~50% and ~45% on 3 and 4 February, "
            "respectively. They are ~20% and ~25% for MSIS2.0."
        ),
        support_verified_by=_SELF,
        support_verification_grade=SELF_VERIFICATION_GRADE,
        note=(
            "The 300 km row as the source reports it, 90 km above the staging "
            "altitude. He et al. report MSIS2.0 as UNDERESTIMATING the "
            "enhancement relative to their theoretical models, so agreement with "
            "it would be agreement with the model family already in use, in the "
            "direction that family is reported to be wrong."
        ),
    ),
    PublishedFinding(
        source="He et al. 2023 (CTIPe/MAGE model average at 200 km), doi:10.1029/2023SW003521",
        source_key="he_2023",
        quantity=ENHANCEMENT_QUANTITY,
        low=0.35,
        high=0.35,
        units=ENHANCEMENT_UNITS,
        measure_kind="fractional_increase",
        kind="modelled",
        evidence_basis="model_simulation",
        instrument="",
        model="mean of CTIPe and MAGE",
        dataset_or_method=(
            "The arithmetic mean of two first-principle simulations, CTIPe and "
            "MAGE, selected by the authors as the two that best track the "
            "observed density and GOLD temperature changes."
        ),
        altitude_km_range=(200.0, 200.0),
        altitude_semantics="single_altitude",
        quantity_definition="global_mean_max",
        baseline_kind="quiet_reference_days",
        baseline_definition=_HE_REFERENCE_DAYS,
        averaging_domains=(
            "THREE SEPARATE OPERATIONS, and the source is explicit about each. "
            "Averaged over MODELS (two of them, CTIPe and MAGE); each model's "
            "field is itself averaged over SPACE (zonal mean, then global "
            "average); and the result is MAXIMISED over TIME across the two "
            "storm days. 'CTIPe/MAGE average' names only the first of the three."
        ),
        locator="Section 3 (p. 15); restated in the Abstract and Key Points (p. 1)",
        quoted_text=(
            "we can estimate that the global averaged maximum density "
            "enhancement based on the average value of both the CTIPe and the "
            "MAGE models, which can be up to ~35% at 200 and ~60% at 300 km "
            "during the two storm days."
        ),
        support_verified_by=_SELF,
        support_verification_grade=SELF_VERIFICATION_GRADE,
        note=(
            "THIS FIGURE WAS PREVIOUSLY ATTRIBUTED TO 'Girgis et al. 2023' and "
            "recorded as a 35-60 % band of mixed provenance over 200-300 km. All "
            "three were wrong: it is He et al.'s, it is purely modelled, and 35 "
            "and 60 are the values AT 200 and AT 300 km, which are kept as two "
            "rows. The paper's abstract does restate the pair as '~35%-60%' for "
            "200-300 km; that is a summary of the two endpoints, not a claim "
            "about the layer between them."
        ),
    ),
    PublishedFinding(
        source="He et al. 2023 (CTIPe/MAGE model average at 300 km), doi:10.1029/2023SW003521",
        source_key="he_2023",
        quantity=ENHANCEMENT_QUANTITY,
        low=0.60,
        high=0.60,
        units=ENHANCEMENT_UNITS,
        measure_kind="fractional_increase",
        kind="modelled",
        evidence_basis="model_simulation",
        instrument="",
        model="mean of CTIPe and MAGE",
        dataset_or_method=(
            "As the 200 km row: the arithmetic mean of the CTIPe and MAGE "
            "simulations at 300 km."
        ),
        altitude_km_range=(300.0, 300.0),
        altitude_semantics="single_altitude",
        quantity_definition="global_mean_max",
        baseline_kind="quiet_reference_days",
        baseline_definition=_HE_REFERENCE_DAYS,
        averaging_domains=(
            "Averaged over MODELS (CTIPe and MAGE), over SPACE (zonal then "
            "global mean), and maximised over TIME across the two storm days."
        ),
        locator="Section 3 (p. 15); restated in the Abstract and Key Points (p. 1)",
        quoted_text=(
            "we can estimate that the global averaged maximum density "
            "enhancement based on the average value of both the CTIPe and the "
            "MAGE models, which can be up to ~35% at 200 and ~60% at 300 km "
            "during the two storm days."
        ),
        support_verified_by=_SELF,
        support_verification_grade=SELF_VERIFICATION_GRADE,
        note=(
            "The 300 km end of the CTIPe/MAGE pair, 90 km above the staging "
            "altitude. Reported as a single value, not an interval, which is "
            "why low and high are equal."
        ),
    ),
    PublishedFinding(
        source="He et al. 2023 (Swarm-A accelerometer), doi:10.1029/2023SW003521",
        source_key="he_2023",
        quantity=ENHANCEMENT_QUANTITY,
        low=1.10,
        high=1.10,
        units=ENHANCEMENT_UNITS,
        measure_kind="fractional_increase",
        kind="observed",
        evidence_basis="instrument_measurement",
        instrument="Swarm-A accelerometer",
        model="",
        dataset_or_method=(
            "Accelerometer-derived neutral mass density along the Swarm-A orbit "
            "in the 09/21 LT sector. A DIFFERENT INSTRUMENT ON A DIFFERENT "
            "SPACECRAFT from the GRACE-FO row: one paper reports both, and they "
            "remain two pieces of evidence."
        ),
        altitude_km_range=(438.0, 438.0),
        altitude_semantics="single_altitude",
        quantity_definition="in_situ_orbit_max",
        baseline_kind="quiet_reference_days",
        baseline_definition=_HE_REFERENCE_DAYS,
        averaging_domains=(
            "Sampled along one orbit, in one local-time sector, and taken at the "
            "storms' main phase. Not averaged over space."
        ),
        locator="Abstract and Key Points (p. 1)",
        quoted_text=_HE_ACCELEROMETER_QUOTE,
        support_verified_by=_SELF,
        support_verification_grade=SELF_VERIFICATION_GRADE,
        note=(
            "Accelerometer-derived density at the Swarm-A orbit, ~438 km, in the "
            "09/21 LT sector, which is 228 km ABOVE the 210 km Starlink staging "
            "altitude. ALTITUDE MISMATCH: this is a genuine measurement and it "
            "establishes that the thermosphere responded to the storm, but it "
            "cannot certify the density at 210 km because the enhancement varies "
            "with altitude. It is kept at its own sampling altitude and no "
            "mapping to 210 km is applied. " + _OBSERVATIONAL_ALTITUDE_CAVEAT
        ),
    ),
    PublishedFinding(
        source="He et al. 2023 (GRACE-FO accelerometer), doi:10.1029/2023SW003521",
        source_key="he_2023",
        quantity=ENHANCEMENT_QUANTITY,
        low=1.20,
        high=1.20,
        units=ENHANCEMENT_UNITS,
        measure_kind="fractional_increase",
        kind="observed",
        evidence_basis="instrument_measurement",
        instrument="GRACE-FO accelerometer",
        model="",
        dataset_or_method=(
            "Accelerometer-derived neutral mass density along the GRACE-FO orbit "
            "in the 06/18 LT sector. A different spacecraft, instrument, "
            "altitude and local-time sector from the Swarm-A row."
        ),
        altitude_km_range=(505.0, 505.0),
        altitude_semantics="single_altitude",
        quantity_definition="in_situ_orbit_max",
        baseline_kind="quiet_reference_days",
        baseline_definition=_HE_REFERENCE_DAYS,
        averaging_domains=(
            "Sampled along one orbit, in one local-time sector, at the storms' "
            "main phase. Not averaged over space."
        ),
        locator="Abstract and Key Points (p. 1)",
        quoted_text=_HE_ACCELEROMETER_QUOTE,
        support_verified_by=_SELF,
        support_verification_grade=SELF_VERIFICATION_GRADE,
        note=(
            "Accelerometer-derived density at the GRACE-FO orbit, ~505 km, in the "
            "06/18 LT sector, which is 295 km ABOVE the 210 km Starlink staging "
            "altitude. ALTITUDE MISMATCH, and the largest in this table: the "
            "measurement is sound and it cannot certify the density at 210 km. "
            "Kept at its own sampling altitude. " + _OBSERVATIONAL_ALTITUDE_CAVEAT
        ),
    ),
)


class Comparison(TypedDict):
    """One value placed against one finding.

    A ``TypedDict`` rather than a plain ``dict`` so that the keys a report
    generator depends on are checked statically; at runtime it is an ordinary
    dict and serialises straight to JSON.

    ``relation`` is the VERDICT and is ``NOT_COMPARABLE`` unless every
    compatibility check passed. ``positional_relation`` is the bare fact of
    where the number falls relative to the published interval, always present,
    and never a verdict on its own.
    """

    source: str
    source_doi: str
    quantity: str
    value: float
    value_definition: str
    value_baseline_kind: str
    low: float
    high: float
    units: str
    measure_kind: str
    kind: str
    evidence_basis: str
    instrument: str
    model: str
    relation: str
    positional_relation: str
    altitude_km: float
    finding_altitude_km_range: tuple[float, float]
    reported_altitude_km: float | None
    altitude_gap_km: float
    altitude_comparable: bool
    finding_quantity_definition: str
    definition_comparable: bool
    finding_baseline_kind: str
    baseline_comparable: bool
    comparable: bool
    incompatibility_reasons: list[str]
    baseline_definition: str
    averaging_domains: str
    locator: str
    quoted_text: str
    note: str
    summary: str


def _format_band(low: float, high: float) -> str:
    """Render a published band as a percentage, collapsing a degenerate range."""
    if low == high:
        return f"{low:.0%}"
    return f"{low:.0%}-{high:.0%}"


def _format_altitude(low: float, high: float) -> str:
    if low == high:
        return f"{low:.0f} km"
    return f"{low:.0f}-{high:.0f} km"


def compare(
    value: float,
    findings: Sequence[PublishedFinding],
    *,
    altitude_km: float,
    value_definition: QuantityDefinition,
    value_baseline_kind: BaselineKind,
) -> list[Comparison]:
    """Place one computed value against each published finding.

    ``value_definition`` and ``value_baseline_kind`` are REQUIRED and have no
    defaults. They say what the computed value is and what it is measured
    against, and they are what makes the compatibility checks mean anything. A
    default would be accepted unexamined by every caller, and the reason these
    arguments exist is that nobody examined it for three review rounds.

    THE RELATION IS A VERDICT AND IS WITHHELD UNLESS IT IS EARNED. Altitude,
    quantity definition and baseline must all match; if any does not,
    ``relation`` is ``NOT_COMPARABLE``, ``incompatibility_reasons`` says which,
    and the bare positional fact moves to ``positional_relation`` so that no
    reader mistakes it for agreement or disagreement.

    All findings must share the same units AND the same measure kind: a
    fractional increase and a density ratio are different numbers even when
    both are written as a percentage, so mixing them raises rather than
    silently producing rows.
    """
    if not math.isfinite(value):
        raise ValueError(f"the computed value must be finite, got {value}")
    if not findings:
        raise ValueError("compare() needs at least one published finding; none were supplied")
    if not math.isfinite(altitude_km) or altitude_km <= 0.0:
        raise ValueError(f"altitude_km must be finite and positive, got {altitude_km}")
    if value_definition not in VALID_DEFINITIONS:
        raise ValueError(
            f"value_definition must be one of {VALID_DEFINITIONS}, got {value_definition!r}"
        )
    if value_baseline_kind not in VALID_BASELINES:
        raise ValueError(
            f"value_baseline_kind must be one of {VALID_BASELINES}, got {value_baseline_kind!r}"
        )

    distinct_units = sorted({f.units for f in findings})
    if len(distinct_units) > 1:
        raise ValueError(
            "all findings must share the same units to be compared against one "
            f"value; got {distinct_units}"
        )
    distinct_measures = sorted({f.measure_kind for f in findings})
    if len(distinct_measures) > 1:
        raise ValueError(
            "all findings must share the same measure kind; a fractional "
            "increase and a density ratio are different numbers even when both "
            f"are printed as a percentage. Got {distinct_measures}"
        )

    comparisons: list[Comparison] = []
    for finding in findings:
        if value < finding.low:
            positional = "BELOW"
        elif value > finding.high:
            positional = "ABOVE"
        else:
            positional = "WITHIN"

        altitude_ok = finding.contains_altitude(altitude_km)
        definition_ok = finding.quantity_definition == value_definition
        baseline_ok = finding.baseline_kind == value_baseline_kind

        reasons: list[str] = []
        if not altitude_ok:
            if finding.altitude_semantics == "range_covered":
                domain = _format_altitude(*finding.altitude_km_range)
                reasons.append(
                    f"the source reports a range over {domain} that varies with "
                    "location as well as height, so it states no value at "
                    f"{altitude_km:.0f} km and none can be read from it"
                )
            else:
                reasons.append(
                    f"reported at {_format_altitude(*finding.altitude_km_range)}, "
                    f"{finding.altitude_gap_km(altitude_km):.0f} km from the "
                    f"{altitude_km:.0f} km comparison altitude"
                )
        if not definition_ok:
            reasons.append(
                f"the computed value is a '{value_definition}' and this finding "
                f"is a '{finding.quantity_definition}'"
            )
        if not baseline_ok:
            reasons.append(
                f"the computed value is measured against a "
                f"'{value_baseline_kind}' and this finding against a "
                f"'{finding.baseline_kind}'"
            )

        comparable = not reasons
        relation = positional if comparable else "NOT_COMPARABLE"

        band = _format_band(finding.low, finding.high)
        where = _format_altitude(*finding.altitude_km_range)
        if comparable:
            summary = (
                f"{finding.source}: {value:.1%} is {relation} the published "
                f"{band} [{finding.kind}, {where}] - comparable on altitude, "
                "quantity and baseline."
            )
        else:
            summary = (
                f"{finding.source}: NOT COMPARABLE with {value:.1%} - "
                + "; ".join(reasons)
                + f". For context only, {value:.1%} falls {positional} the "
                f"published {band} [{finding.kind}, {where}]; that is a "
                "position, not agreement, disagreement or a measure of bias."
            )

        comparisons.append(
            Comparison(
                source=finding.source,
                source_doi=source_for(finding.source_key).doi,
                quantity=finding.quantity,
                value=float(value),
                value_definition=value_definition,
                value_baseline_kind=value_baseline_kind,
                low=finding.low,
                high=finding.high,
                units=finding.units,
                measure_kind=finding.measure_kind,
                kind=finding.kind,
                evidence_basis=finding.evidence_basis,
                instrument=finding.instrument,
                model=finding.model,
                relation=relation,
                positional_relation=positional,
                altitude_km=float(altitude_km),
                finding_altitude_km_range=finding.altitude_km_range,
                reported_altitude_km=finding.reported_altitude_km,
                altitude_gap_km=finding.altitude_gap_km(altitude_km),
                altitude_comparable=altitude_ok,
                finding_quantity_definition=finding.quantity_definition,
                definition_comparable=definition_ok,
                finding_baseline_kind=finding.baseline_kind,
                baseline_comparable=baseline_ok,
                comparable=comparable,
                incompatibility_reasons=reasons,
                baseline_definition=finding.baseline_definition,
                averaging_domains=finding.averaging_domains,
                locator=finding.locator,
                quoted_text=finding.quoted_text,
                note=finding.note,
                summary=summary,
            )
        )
    return comparisons


def _relation_sentence(kind_label: str, relations: list[str], count: int) -> str:
    """One sentence describing how a value sits against a group of findings."""
    unique = set(relations)
    if unique == {"BELOW"}:
        return (
            f"The computed value lies below every {kind_label} finding it was "
            f"compared against ({count} of {count}). That is a disagreement with "
            "the published figures and is reported as one."
        )
    if unique == {"ABOVE"}:
        return (
            f"The computed value lies above every {kind_label} finding it was "
            f"compared against ({count} of {count})."
        )
    if unique == {"WITHIN"}:
        return (
            f"The computed value lies within every {kind_label} band it was "
            f"compared against ({count} of {count})."
        )
    below = relations.count("BELOW")
    within = relations.count("WITHIN")
    above = relations.count("ABOVE")
    return (
        f"Against the {count} {kind_label} findings the computed value is below "
        f"{below}, within {within} and above {above}. There is no single verdict "
        "to state."
    )


#: The statement the comparison supports when no finding clears every
#: compatibility check. It is a constant rather than assembled prose because it
#: is the sentence this whole module exists to get right, and it must read the
#: same in every artefact that quotes it.
NO_DIRECT_VALIDATION_STATEMENT = (
    "The previous claim that our fixed-point enhancement lay below all "
    "published modelled findings was unsupported. Corrected citations and "
    "altitude assignments place the result in its proper model-family context, "
    "but differences in sampling, averaging and baseline definitions prevent a "
    "direct numerical validation."
)


def verdict_text(comparisons: Sequence[Comparison]) -> str:
    """Render comparisons as prose, without a favourable framing.

    Two rules this function exists to enforce. If the value sits below every
    comparable modelled finding, the text says so in those words: a result that
    disagrees with the published literature is the most important thing a
    reader can be told about it, and the temptation is always to find the one
    band it does fall inside and lead with that. And if NO finding clears the
    compatibility checks, the text refuses to issue any verdict at all rather
    than issuing a weak one.
    """
    if not comparisons:
        raise ValueError("verdict_text() needs at least one comparison; none were supplied")

    values = {c["value"] for c in comparisons}
    if len(values) > 1:
        raise ValueError(
            "verdict_text() received comparisons of different values "
            f"({sorted(values)}); they must all describe one computed value"
        )

    first = comparisons[0]
    value = first["value"]
    altitude_km = first["altitude_km"]
    units = first["units"]
    quantity = first["quantity"]
    value_definition = first["value_definition"]
    value_baseline = first["value_baseline_kind"]

    paragraphs = [
        f"Computed {quantity}: {value:.3f} ({value:.1%}), measured as a "
        f"{units}, at {altitude_km:.0f} km, defined as a '{value_definition}' "
        f"against a '{value_baseline}' baseline."
    ]

    comparable = [c for c in comparisons if c["comparable"]]
    if not comparable:
        failed_altitude = sum(1 for c in comparisons if not c["altitude_comparable"])
        failed_definition = sum(1 for c in comparisons if not c["definition_comparable"])
        failed_baseline = sum(1 for c in comparisons if not c["baseline_comparable"])
        paragraphs.append(
            f"NO DIRECT COMPARISON IS AVAILABLE. None of the {len(comparisons)} "
            "published findings matches the computed value on all three of "
            f"altitude, quantity definition and baseline: {failed_altitude} fail "
            f"on altitude, {failed_definition} on quantity definition and "
            f"{failed_baseline} on baseline. No agreement, disagreement, "
            "validation or bias classification is issued, and the per-finding "
            "positions below are context only. "
            + NO_DIRECT_VALIDATION_STATEMENT
        )
    else:
        paragraphs.append(
            f"{len(comparable)} of {len(comparisons)} published findings match "
            "the computed value on altitude, quantity definition and baseline. "
            "Only those carry a verdict; the rest are context."
        )
        modelled = [c for c in comparable if c["kind"] == "modelled"]
        if modelled:
            paragraphs.append(
                _relation_sentence(
                    "comparable modelled", [str(c["relation"]) for c in modelled], len(modelled)
                )
                + " These are model results, so agreement with them would be "
                "agreement between models rather than observational support."
            )
        observational = [c for c in comparable if c["kind"] in ("observed", "mixed")]
        if observational:
            paragraphs.append(
                _relation_sentence(
                    "comparable observed or mixed-provenance",
                    [str(c["relation"]) for c in observational],
                    len(observational),
                )
            )

    sources = sorted({c["source_doi"] for c in comparisons})
    paragraphs.append(
        f"Provenance: {len(comparisons)} findings from {len(sources)} distinct "
        f"papers ({', '.join(sources)}). Findings are not independent sources; "
        "several of these are reported by one paper."
    )

    paragraphs.append("Per-finding detail:")
    paragraphs.extend(f"  - {c['summary']}" for c in comparisons)
    return "\n".join(paragraphs)
