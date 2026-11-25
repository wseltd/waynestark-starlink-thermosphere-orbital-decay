"""Load the catalogued decay dates from the raw store, not from a literal.

The catalogued decay dates are the ONLY observational anchor in this project's
entire causal chain. Everything else between the space-weather indices and the
final number is a model. Until this module existed they were the one input
with no provenance: hard-coded literals typed into events/starlink_g4_7.py,
never routed through the CelesTrak client written to fetch them, so
starlink_object_dataset.csv had to record their retrieval time and SHA-256 as
`unknown` while every other external input carried both.

That is now fixed. This module reads them from data/raw through the same
hash-verified RawStore as every other external input, so the anchor is as
traceable as the things it anchors.

The literals in events/starlink_g4_7.CATALOGUED_DECAYS are retained as a
CROSS-CHECK, not as the source. ``verify_against_literals`` compares the two
and raises on any disagreement, so a silent divergence between what was typed
and what the catalogue says is impossible. If CelesTrak revises a decay date,
the run fails loudly rather than quietly using a stale number.

Canonical tickets: P03_05_T030, P09_10_T009.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from ..drivers.catalogue import parse_satcat
from ..drivers.raw_store import RawStore
from .starlink_g4_7 import CATALOGUED_DECAYS, GroundTruthDecay

SATCAT_SOURCE = "celestrak"
SATCAT_ASSET_ID = "satcat_2022-010"


class GroundTruthMismatchError(RuntimeError):
    """Raised when the fetched catalogue disagrees with the recorded literals."""


@dataclass(frozen=True, slots=True)
class GroundTruthProvenance:
    """Where the observational anchor came from, and whether it still agrees."""

    source: str
    asset_id: str
    url: str
    sha256: str
    retrieved_at_utc: str
    licence: str
    citation: str
    object_count: int
    decayed_count: int
    verified_against_literals: bool


def load_catalogued_decays(
    store: RawStore | None = None,
) -> tuple[tuple[GroundTruthDecay, ...], GroundTruthProvenance]:
    """Read the catalogued decay dates from the hash-verified raw store.

    Returns the decays AND their provenance, together, so a caller cannot
    obtain the numbers without also obtaining the record of where they came
    from. RawStore.read re-hashes the payload on every read, so an altered
    file on disk raises rather than being silently used.
    """
    store = store or RawStore("data/raw")
    payload = store.read(SATCAT_SOURCE, SATCAT_ASSET_ID, ".json")
    metadata = store.read_metadata(SATCAT_SOURCE, SATCAT_ASSET_ID)
    objects = parse_satcat(payload)

    decays = tuple(
        GroundTruthDecay(
            norad_id=obj.norad_id,
            object_id=obj.object_id,
            decay_date=obj.decay_date.date() if obj.decay_date else None,
        )
        for obj in sorted(objects, key=lambda o: o.norad_id)
    )
    provenance = GroundTruthProvenance(
        source=metadata.source,
        asset_id=metadata.asset_id,
        url=metadata.url,
        sha256=metadata.sha256,
        retrieved_at_utc=metadata.retrieved_at_utc,
        licence=metadata.licence,
        citation=metadata.citation,
        object_count=len(decays),
        decayed_count=sum(1 for d in decays if d.decay_date is not None),
        verified_against_literals=False,
    )
    return decays, provenance


def verify_against_literals(
    fetched: tuple[GroundTruthDecay, ...],
) -> tuple[str, ...]:
    """Compare the fetched catalogue with the recorded literals.

    Returns the list of disagreements. An empty tuple means the literals are
    exactly what the catalogue says. This is a real check rather than a
    formality: a decay date the catalogue has since revised, or a typo made
    when the literals were entered, shows up here instead of propagating
    silently into a fitted parameter.
    """
    by_id_fetched = {d.norad_id: d for d in fetched}
    by_id_literal = {d.norad_id: d for d in CATALOGUED_DECAYS}

    problems: list[str] = []
    for norad_id in sorted(set(by_id_fetched) | set(by_id_literal)):
        got = by_id_fetched.get(norad_id)
        want = by_id_literal.get(norad_id)
        if got is None:
            problems.append(f"{norad_id}: in the literals but ABSENT from the catalogue")
            continue
        if want is None:
            problems.append(
                f"{norad_id} ({got.object_id}): in the catalogue but ABSENT from the "
                "literals — the recorded set is incomplete"
            )
            continue
        if got.object_id != want.object_id:
            problems.append(
                f"{norad_id}: object_id differs, catalogue {got.object_id!r} vs "
                f"literal {want.object_id!r}"
            )
        if got.decay_date != want.decay_date:
            problems.append(
                f"{norad_id} ({got.object_id}): decay date differs, catalogue "
                f"{got.decay_date} vs literal {want.decay_date}"
            )
    return tuple(problems)


def load_verified_ground_truth(
    store: RawStore | None = None,
) -> tuple[tuple[GroundTruthDecay, ...], GroundTruthProvenance]:
    """Load the anchor and refuse to return it if it disagrees with the literals.

    This is the function analysis code should call. It fails closed: a
    disagreement raises rather than being reported as a warning a caller might
    not read.
    """
    decays, provenance = load_catalogued_decays(store)
    problems = verify_against_literals(decays)
    if problems:
        raise GroundTruthMismatchError(
            "the fetched catalogue disagrees with the recorded literals in "
            "events/starlink_g4_7.CATALOGUED_DECAYS:\n  "
            + "\n  ".join(problems)
            + "\nThe catalogue is authoritative; update the literals, and treat "
            "any change to a decay date as a change to the observational anchor "
            "that every fitted parameter depends on."
        )
    return decays, GroundTruthProvenance(
        source=provenance.source,
        asset_id=provenance.asset_id,
        url=provenance.url,
        sha256=provenance.sha256,
        retrieved_at_utc=provenance.retrieved_at_utc,
        licence=provenance.licence,
        citation=provenance.citation,
        object_count=provenance.object_count,
        decayed_count=provenance.decayed_count,
        verified_against_literals=True,
    )


def storm_window_losses(
    decays: tuple[GroundTruthDecay, ...],
    *,
    window_days: float = 30.0,
) -> tuple[GroundTruthDecay, ...]:
    """Objects that decayed within the window, from the FETCHED set.

    The two 2022-010 objects that decayed in 2023 and 2024 reached operational
    altitude first and were lost for unrelated reasons; including them would
    overstate the event.
    """
    selected: list[GroundTruthDecay] = []
    for decay in decays:
        days = decay.days_from_deployment
        if days is not None and 0.0 < days <= window_days:
            selected.append(decay)
    return tuple(sorted(selected, key=lambda d: (d.decay_date or _dt.date.min, d.norad_id)))
