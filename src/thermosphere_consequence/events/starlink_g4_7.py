"""Starlink Group 4-7: the February 2022 event constants, read from configuration.

Every name below is READ FROM ``config/events/starlink_group_4_7_2022.yaml`` at
import time, through :func:`~.manifest_loader.load_default_event_manifest`. None
of them is a literal any more.

That distinction is the whole point. These constants used to be literals here,
with the manifest alongside them as a second copy that two tests kept equal.
A copy that agrees is not a source: editing the YAML moved nothing, so the
provenance recorded in it described the analysis only for as long as nobody
edited one side without the other. Now the YAML *is* the analysis's input, and
an edit to it reaches every propagation, report and API response that reads
these names.

If the manifest is missing, malformed, uncited, non-SI or otherwise refused by
the loader, **importing this module fails**, loudly, with the loader's own
error. There is deliberately no fallback to a literal: a fallback would restore
exactly the defect described above, silently, at the moment the configuration
stopped being trustworthy.

Where the literature and the operator disagree, both are recorded with their
provenance rather than one being silently preferred — SpaceX's own early
statement said "up to 40" reentered, while the peer-reviewed figure is 38, and
the difference is real information about how the count firmed up over time. The
manifest carries both, and so do the names below.

Correction carried from the audited research: an earlier brief assumed NORAD
range 51089-51137. That is WRONG. Group 4-7 is COSPAR 2022-010, catalogue
numbers approximately 51456-51476, with 2022-010A = NORAD 51456.

UNITS. The manifest is SI throughout, so the conversions a reader expects are
visible here rather than hidden in a constant's name: the manifest holds
inclination in radians and this module exposes ``..._DEG``; it holds SYM-H in
tesla and this module exposes ``..._NT``; it holds the Kp duration in seconds
and this module exposes ``..._HOURS``. Each conversion is one expression below
and is asserted in ``tests/events/test_manifest_loader.py``.

Canonical tickets: P09_10_T001 through T012.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

from .manifest_loader import load_default_event_manifest

# The loaded, validated manifest. Private because it is an implementation
# detail of the names below: a caller that wants the manifest itself, with its
# per-value units, statuses and citations, should call
# load_default_event_manifest() rather than reach through this module.
#
# There is no import cycle to break: manifest_loader depends on domain/ and
# schemas/ only, and on nothing in events/. That is why this is an ordinary
# module-level import rather than a function-local one that would hide the
# dependency from a reader and from the import graph.
_MANIFEST = load_default_event_manifest()

# --- Launch and insertion -------------------------------------------------
LAUNCH_UTC = _MANIFEST.timestamp_of("launch_utc")
DEPLOYMENT_UTC = _MANIFEST.timestamp_of("deployment_utc")
INSERTION_PERIGEE_ALTITUDE_M = _MANIFEST.number_of("insertion_perigee_altitude")
INSERTION_APOGEE_ALTITUDE_M = _MANIFEST.number_of("insertion_apogee_altitude")
INSERTION_INCLINATION_DEG = math.degrees(_MANIFEST.number_of("insertion_inclination"))
INTENDED_OPERATIONAL_ALTITUDE_M = _MANIFEST.number_of("intended_operational_altitude")

COSPAR_ID_PREFIX = _MANIFEST.text_of("cospar_id_prefix")
FIRST_NORAD_ID = int(_MANIFEST.text_of("first_norad_id"))

# --- Fleet outcome --------------------------------------------------------
# Catalogue decay dates carry DATE resolution only: a decay recorded on
# 2022-02-06 occurred somewhere within that day. Declared in SI seconds so the
# conversion is visible rather than implied by a constant name.
DECAY_DATE_RESOLUTION_S = _MANIFEST.number_of("decay_date_resolution_s")

SATELLITES_LAUNCHED = int(_MANIFEST.number_of("satellites_launched"))
SATELLITES_LOST_PEER_REVIEWED = int(_MANIFEST.number_of("satellites_lost_peer_reviewed"))
SATELLITES_SURVIVED_PEER_REVIEWED = int(
    _MANIFEST.number_of("satellites_survived_peer_reviewed")
)
SATELLITES_NEVER_CATALOGUED = int(_MANIFEST.number_of("satellites_never_catalogued"))
SPACEX_EARLY_ESTIMATE_LOST = int(_MANIFEST.number_of("spacex_early_estimate_lost"))

# --- Spacecraft -----------------------------------------------------------
# Mass is third-party/derived (KeepTrack), not an operator disclosure, and is
# therefore carried in the manifest with an uncertainty range, which is what
# the sensitivity envelope sweeps.
MASS_KG_RANGE = _MANIFEST.value("mass_nominal").uncertainty_range
MASS_KG_NOMINAL = _MANIFEST.number_of("mass_nominal")

# Shambaugh (2024, arXiv:2406.08342, p. 6) reported 0.085 m^2/kg for a
# Capella Space SAR (Whitney) satellite (NOT SpaceX and NOT Starlink).
#
# UNITS: m^2/kg means this is C_d*A/m, the INVERSE ballistic coefficient, NOT
# the conventional B = m/(C_d*A) which would be 11.76 kg/m^2. The two are
# reciprocals and move in opposite directions. The manifest declares both, each
# in its own units, and the loader puts the name and units of the first through
# the claim gate — so a rename that dropped "inverse" fails to load rather than
# reaching this module.
#
# SCOPE: external literature reference for a different spacecraft class
# (Capella Space SAR), retained under legacy symbol names for compatibility.
OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG = _MANIFEST.number_of(
    "external_reference_inverse_ballistic_coefficient"
)
OPERATOR_FITTED_BALLISTIC_COEFFICIENT_KG_M2 = _MANIFEST.number_of(
    "external_reference_ballistic_coefficient"
)

# Conventional C_d, retained as a floor rather than a truth. Cook (1965)
# himself wrote that 2.2 "is subject to some uncertainty and may be too low,
# perhaps by as much as 10-15 percent".
CD_CONVENTIONAL = _MANIFEST.number_of("cd_conventional")
CD_SENSITIVITY_RANGE = _MANIFEST.value("cd_conventional").uncertainty_range

def _enhancement_source(reference_fragment: str) -> str:
    """The citation of the published-enhancement entry naming ``reference_fragment``.

    Matched on the reference text rather than on a list position, so reordering
    the manifest's ``published_enhancement`` section cannot quietly move a
    citation onto a different study. A fragment that matches nothing, or more
    than one entry, raises here at import rather than mislabelling a comparison.
    """
    matches = [
        entry for entry in _MANIFEST.published_enhancement if reference_fragment in entry.reference
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{reference_fragment!r} matches {len(matches)} published_enhancement "
            f"entries in {_MANIFEST.source_path.name}; expected exactly one. "
            f"References available: {[e.reference for e in _MANIFEST.published_enhancement]}"
        )
    return matches[0].source


# The citation table, assembled from the citations the manifest already carries
# on the values themselves. It is a convenience index by topic, not a second
# place a citation is written down: change a source in the YAML and the entry
# here changes with it. Qualifications that used to be parenthesised into these
# strings — that the SpaceX text is held via secondary quotation, that the mass
# figure is a third-party estimate rather than an operator disclosure — live in
# the notes of the manifest value they qualify, beside the number itself.
SOURCES: dict[str, str] = {
    "insertion": _MANIFEST.value("insertion_perigee_altitude").source,
    "counts": _MANIFEST.value("satellites_launched").source,
    "spacex_statement": _MANIFEST.value("spacex_early_estimate_lost").source,
    "inverse_ballistic_coefficient": _MANIFEST.value(
        "external_reference_inverse_ballistic_coefficient"
    ).source,
    "mass": _MANIFEST.value("mass_nominal").source,
    "cd_convention": _MANIFEST.value("cd_conventional").source,
    # Fragments must be unambiguous. They became ambiguous the moment the
    # published-enhancement table stopped holding one entry per paper, and the
    # guard in _enhancement_source refused the import rather than picking one --
    # which is the behaviour that was wanted when it was written. They became
    # ambiguous a second time when each source's ORIGINAL ALTITUDE ROWS were
    # restored, so every fragment below names an altitude.
    "density_enhancement_wamipe": _enhancement_source("WAM-IPE (200-400 km domain range)"),
    "density_enhancement_msis_published": _enhancement_source("NRLMSISE-00 (210 km)"),
    # Was "density_enhancement_observed", pointing at an entry credited to
    # Girgis et al. 2023. The figure is He et al.'s and it is modelled, not
    # observed; the accelerometer entries below are the observed ones.
    "density_enhancement_theoretical_models_200km": _enhancement_source(
        "CTIPe/MAGE model average (200 km)"
    ),
    "density_enhancement_theoretical_models_300km": _enhancement_source(
        "CTIPe/MAGE model average (300 km)"
    ),
    "density_enhancement_observed_swarm": _enhancement_source("Swarm-A accelerometer"),
    "density_enhancement_observed_grace": _enhancement_source("GRACE-FO accelerometer"),
    "msis_published_200km": _enhancement_source("MSIS2.0 (200 km)"),
    "msis_published_300km": _enhancement_source("MSIS2.0 (300 km)"),
}

# --- Published enhancement figures, for comparison not for fitting --------
# These are what the result is checked AGAINST. They are never used as inputs.
PUBLISHED_ENHANCEMENT: dict[str, tuple[float, float, str]] = {
    entry.reference: (entry.fraction_low, entry.fraction_high, entry.basis)
    for entry in _MANIFEST.published_enhancement
}

# Storm indices as reported in the literature, used to sanity-check ingestion.
# The manifest holds them in SI (tesla, seconds); the conversion to the units
# these names declare is the one expression below.
REPORTED_PEAK_SYM_H_NT = _MANIFEST.number_of("reported_peak_sym_h") * 1e9
REPORTED_PEAK_KP = _MANIFEST.number_of("reported_peak_kp")
REPORTED_KP_GE_5_HOURS = _MANIFEST.number_of("reported_kp_ge_5_duration") / 3600.0


@dataclass(frozen=True, slots=True)
class GroundTruthDecay:
    """An observed decay date from the public catalogue.

    This is the only genuinely observational anchor in the whole chain. It is
    not a model output and not a literature claim: it is what the catalogue
    recorded happening.
    """

    norad_id: int
    object_id: str
    decay_date: _dt.date | None
    launch_date: _dt.date = _dt.date(2022, 2, 3)

    @property
    def days_to_decay(self) -> float | None:
        """Days from LAUNCH DATE midnight to the catalogued decay date.

        Retained for reporting because it is what the catalogue supports, but
        it is NOT the right quantity to compare a propagation against: see
        days_from_deployment.
        """
        if self.decay_date is None:
            return None
        return float((self.decay_date - self.launch_date).days)

    @property
    def days_from_deployment(self) -> float | None:
        """Days from DEPLOYMENT to the catalogued decay date.

        Propagation starts at deployment, 2022-02-03 18:28 UTC, which is
        0.7694 d after midnight on the launch date. Comparing a propagated
        lifetime against days_to_decay therefore carried a systematic 0.77 d
        offset against a 3-13 d observable -- roughly a fifth of the smallest
        value being compared. This property removes it.

        The catalogue gives DATE resolution only, so the decay instant is
        taken as midnight at the START of the recorded day and the true value
        lies within +1 d of it. DECAY_DATE_RESOLUTION_S records that.
        """
        if self.decay_date is None:
            return None
        decay_instant = _dt.datetime.combine(
            self.decay_date, _dt.time(0, 0), tzinfo=_dt.UTC
        )
        return (decay_instant - DEPLOYMENT_UTC).total_seconds() / 86400.0

    @property
    def is_storm_loss(self) -> bool:
        """Decayed within 30 days — attributable to the insertion-altitude event.

        The two 2022-010 objects that decayed in 2023 and 2024 reached
        operational altitude first and were lost for unrelated reasons; folding
        them into the storm-loss count would overstate the event.
        """
        d = self.days_to_decay
        return d is not None and d <= 30.0


def _catalogued_decays() -> tuple[GroundTruthDecay, ...]:
    """The manifest's catalogue table, in the order a reader reads it.

    The manifest lists the rows by catalogue number, because that is how the
    SATCAT query returns them and a table sorted by its key is a table nobody
    has to trust. A report reads them chronologically, so they are ordered here
    by decay date with the catalogue number breaking ties, and the objects that
    have not decayed last. Ordering is a presentation choice and belongs on
    this side of the seam; the rows, the dates and the launch date are the
    manifest's and are not touched.
    """
    launch_date = _MANIFEST.launch_date
    ordered = sorted(
        _MANIFEST.decays,
        key=lambda record: (
            record.decay_date is None,
            record.decay_date or _dt.date.max,
            record.norad_id,
        ),
    )
    return tuple(
        GroundTruthDecay(
            norad_id=record.norad_id,
            object_id=record.object_id,
            decay_date=record.decay_date,
            launch_date=launch_date,
        )
        for record in ordered
    )


# Catalogued decay dates for COSPAR 2022-010, retrieved from the CelesTrak
# SATCAT (https://celestrak.org/satcat/records.php?INTDES=2022-010) and carried
# in the manifest's decay_table section.
#
# This is the only genuinely observational anchor in the entire chain: not a
# model output, not a literature claim, but what the public catalogue recorded
# happening. Twenty-one of the forty-nine launched objects ever received public
# catalogue numbers; the remaining thirty-two were never tracked, so any
# comparison against this table is a comparison against a SUBSET, and one
# biased toward objects that survived long enough to be catalogued.
#
# The two 2023/2024 decays reached operational altitude first and were lost for
# unrelated reasons. is_storm_loss excludes them.
CATALOGUED_DECAYS: tuple[GroundTruthDecay, ...] = _catalogued_decays()


def storm_window_losses() -> tuple[GroundTruthDecay, ...]:
    """Catalogued objects that decayed within thirty days of launch."""
    return tuple(g for g in CATALOGUED_DECAYS if g.is_storm_loss)


def observed_decay_days() -> tuple[float, ...]:
    """Sorted days-to-decay for the storm-window losses."""
    return tuple(
        sorted(g.days_to_decay for g in storm_window_losses() if g.days_to_decay is not None)
    )


def insertion_state_vectors(
    *, true_anomaly_rad: float = 0.0
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Cartesian state at deployment, from the published insertion elements."""
    from ..propagation.propagator import state_from_perigee_apogee

    r, v = state_from_perigee_apogee(
        INSERTION_PERIGEE_ALTITUDE_M,
        INSERTION_APOGEE_ALTITUDE_M,
        math.radians(INSERTION_INCLINATION_DEG),
        true_anomaly_rad=true_anomaly_rad,
    )
    return (float(r[0]), float(r[1]), float(r[2])), (float(v[0]), float(v[1]), float(v[2]))


def effective_area_from_operator_inverse_ballistic_coefficient(
    drag_coefficient: float, mass_kg: float = MASS_KG_NOMINAL
) -> float:
    """Area implied by the Capella Space Whitney reference C_d*A/m (Shambaugh 2024,
    arXiv:2406.08342; not SpaceX Starlink), at a GIVEN C_d and mass.

    Retained for historical scenario scaling: mass, area and C_d are not
    separately identifiable from a decay curve, so anchoring on a fixed
    external reference C_B and solving for area keeps the product C_d*A/m
    fixed when testing sensitivity to C_d or mass.
    """
    if drag_coefficient <= 0:
        raise ValueError(f"drag coefficient must be positive, got {drag_coefficient}")
    if mass_kg <= 0:
        raise ValueError(f"mass must be positive, got {mass_kg}")
    return OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG * mass_kg / drag_coefficient


#: The seven statements frozen as prohibited by
#: ``evidence/FINAL_SCIENTIFIC_STATUS.md``, held here in the document's own
#: wording.
#:
#: The document is where a reader checks what this project may and may not be
#: said to have shown; this tuple is why that list is worth checking. Each
#: sentence is unreachable through two independent exits: the runtime gate in
#: :mod:`thermosphere_consequence.schemas.claim_gate`, which reasons about the
#: declared mode wherever prose is generated with a context in hand, and this
#: register, which the report generator and the API scan at the point of
#: emission where no context exists. Neither is a single point of failure, and
#: a sentence removed from the document without being removed here would make
#: the document wrong rather than the code permissive.
FROZEN_STATUS_CLAIMS: tuple[str, ...] = (
    "the geomagnetic storm was proved to be the unique cause",
    "the storm shortened the real spacecraft lifetime by exactly 0.4664 days",
    "the fitted C_d*A/m uniquely determines attitude or projected area",
    "all 49 trajectories were reconstructed",
    "SURYA predicts re-entry from solar imagery",
    "the system is validated for operational Starlink lifetime prediction",
    "modelled density is measured density",
)


def prohibited_claims() -> tuple[str, ...]:
    """Statements this repository must never make, with the reason.

    Kept as executable data rather than prose so the report generator can
    assert none of them appear in generated output.

    Two groups, one register. The first were written against the specific
    errors this analysis was at risk of making as it was built. The second is
    :data:`FROZEN_STATUS_CLAIMS`, the list frozen by
    ``evidence/FINAL_SCIENTIFIC_STATUS.md``; it is spliced in from there rather
    than copied, so the document's list and the enforced list cannot drift
    apart.
    """
    written_during_the_build = (
        "composition alone increased drag",
        "SURYA predicted the Starlink losses",
        "SURYA would have prevented the loss",
        "this model reproduces the observed density at 210 km",
        "NRLMSISE-00 storm density is ground truth",
    )
    return (*written_during_the_build, *FROZEN_STATUS_CLAIMS)
