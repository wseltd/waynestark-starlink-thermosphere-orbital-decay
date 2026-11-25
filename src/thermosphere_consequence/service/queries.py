"""The read operations the API and the CLI both need, answered once.

Every function here returns a :class:`QueryResult`: a value paired with the
epistemic status of that value, the sources behind it and the caveats that
survive them. Nothing returns a bare number. A density that leaves this layer
without its status and its caveats is a density that will be quoted as though
it had been measured, and preventing exactly that is what the whole system is
for.

NO SCIENTIFIC LOGIC LIVES HERE. Every physical quantity is produced by the
library function that already implements it — the same function the API, the
CLI, the analysis scripts and the notebooks call — so an answer cannot differ
between entry points. A structural test asserts the absence of arithmetic in
this package, which is what makes that claim checkable rather than aspirational.
The rule bites in an unobvious place: even ``path / "name"`` is arithmetic to
the parser, so paths here are built with ``joinpath``.

THE MODE IS ENFORCED, NOT ANNOTATED. :func:`ground_truth` refuses to answer in
:attr:`~thermosphere_consequence.schemas.modes.AnalysisMode.SCENARIO`, because
the catalogued decay dates are a claim about what specific real objects did and
scenario mode explicitly does not license that claim. A mode that only appeared
in a report header would be a label; refusing the query is what makes it a
licence.

THE OBSERVATIONAL ANCHOR GOES THROUGH THE VERIFIED LOADER. :func:`ground_truth`
calls
:func:`~thermosphere_consequence.events.ground_truth_loader.load_verified_ground_truth`,
so the hash re-verification of the stored catalogue and the cross-check against
the recorded literals both sit on the service path. Reading the literals
directly would be shorter and would silently bypass the only place in the chain
where an observation is checked against its source.

AUTHORISATION CANNOT BE FORGOTTEN, BECAUSE IT CANNOT BE OMITTED. Every query
takes ``principal`` as a KEYWORD-ONLY parameter WITH NO DEFAULT, so a call site
that does not pass one does not run at all — it fails at the call, in every
caller, at import-time review and at test time. A default principal would have
been the convenient choice and it is the one thing that must not exist here: it
would make "authorised" the property of whichever principal happened to be the
default, and the check would pass on every path that forgot to think about it.
The permission is checked BEFORE any work is done. That ordering is load-bearing
rather than tidy: :func:`latest_run_provenance` raises
:class:`FileNotFoundError` for an absent artefact and
:class:`~thermosphere_consequence.service.authorisation.NotAuthorisedError` for
a refused caller, so checking the file first would let an unauthorised caller
learn which run records exist by reading the exception type.

THE MODE THE PRINCIPAL IS CHECKED AGAINST IS THE COMPOSITION'S, NOT AN
ARGUMENT. ``providers.mode`` is the licence-gate mode the provider set was
actually built and cleared under, so the authorisation decision and the licence
decision cannot be made against two different modes.

THIS IS NOT THE API KEY. :mod:`thermosphere_consequence.api.app` has its own
``X-API-Key`` gate and this layer sits BELOW it: the key says a request reached
a deployment it is entitled to reach, and a
:class:`~thermosphere_consequence.service.authorisation.Principal` says who is
asking and what they may do. The CLI and a notebook never see the
HTTP gate and do go through this one, which is the reason it lives here rather
than in the adapter.

EVERY QUERY EMITS EXACTLY ONE AUDIT EVENT, including the ones that refuse. See
:mod:`~thermosphere_consequence.service.observability`.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..aerodynamics.sentman import (
    ACCOMMODATION_SENSITIVITY_RANGE,
    NOMINAL_ACCOMMODATION,
    species_weighted_drag_coefficient,
)
from ..domain.drag_parameters import (
    INVERSE_BALLISTIC_COEFFICIENT_UNITS,
    inverse_ballistic_coefficient_m2_kg,
)
from ..drivers.raw_store import RawStore
from ..events.ground_truth_loader import (
    GroundTruthProvenance,
    load_verified_ground_truth,
    storm_window_losses,
)
from ..events.manifest_loader import EventManifest, load_default_event_manifest
from ..events.starlink_g4_7 import GroundTruthDecay
from ..molecular.composition import mean_particle_mass, total_species_mass_density
from ..protocols.atmosphere import AtmosphereState, DriverState
from ..schemas.modes import AnalysisMode, DataStatus, SourceAuthority
from .authorisation import (
    UNTENANTED_RESOURCE_IS_SHARED,
    NotAuthorisedError,
    Permission,
    Principal,
    authorise,
    authorise_tenant,
)
from .composition_root import REPOSITORY_ROOT, ProviderSet
from .observability import ObservabilityConfig, OperationContext, observed_operation

__all__ = [
    "DEFAULT_RAW_STORE_PATH",
    "DEFAULT_RUN_PROVENANCE_PATH",
    "RUN_RECORD_TENANT_FIELD",
    "STORM_WINDOW_DAYS",
    "AtmosphereAnswer",
    "DragAnswer",
    "GroundTruthAnswer",
    "ModeNotLicensedError",
    "QueryResult",
    "atmosphere_at",
    "drag_at",
    "event_manifest",
    "ground_truth",
    "latest_run_provenance",
]

#: The key a stored run record uses to name the tenant it belongs to. A record
#: without it names no owner; see
#: :data:`~thermosphere_consequence.service.authorisation.UNTENANTED_RESOURCE_IS_SHARED`.
RUN_RECORD_TENANT_FIELD = "tenant_id"

#: The hash-verified store the observational anchor is read from. Anchored on
#: the installed package, not the working directory.
DEFAULT_RAW_STORE_PATH = REPOSITORY_ROOT.joinpath("data", "raw")

#: The stored record of the last full analysis run.
DEFAULT_RUN_PROVENANCE_PATH = REPOSITORY_ROOT.joinpath("evidence", "run_provenance.json")

#: Days after deployment within which a decay is attributed to the storm. The
#: two objects that decayed in 2023 and 2024 reached operational altitude first
#: and were lost for unrelated reasons.
STORM_WINDOW_DAYS = 30.0

_ANALYSIS_MODES_LICENSING_CATALOGUE_CLAIMS = (
    AnalysisMode.TRACKED_OBJECT,
    AnalysisMode.OPERATOR_TRUTH,
)

_ATMOSPHERE_SPECIES_SUM_CAVEAT = (
    "mass_density_kg_m3 is the sum over the species the model reported. The "
    "model's own total density is not identical to that sum, and the residual "
    "is the density-closure diagnostic rather than an error to correct away."
)

_DRAG_CAVEATS = (
    "Free-molecular closed forms, valid for Knudsen number much greater than 1. "
    "At 210 km the mean free path is of order a kilometre, so the regime holds; "
    "it does not hold through the lower thermosphere.",
    "Energy accommodation is a scenario parameter, never a measurement. This "
    f"package's nominal is {NOMINAL_ACCOMMODATION} and the sensitivity envelope "
    f"sweeps {ACCOMMODATION_SENSITIVITY_RANGE}.",
    "Weighting is by species mass flux, so the answer belongs to the "
    "composition supplied with it and cannot be reused at another altitude.",
    f"inverse_ballistic_coefficient_m2_kg is C_d*A/m in "
    f"{INVERSE_BALLISTIC_COEFFICIENT_UNITS}. It is NOT a ballistic "
    f"coefficient, which is its reciprocal in kg/m^2 and moves the other way.",
    "A decay curve constrains only the product C_d*A/m. A change in it may "
    "arise from area, drag coefficient, accommodation, mass, attitude, "
    "atmosphere-model bias or initial-state bias, or any combination.",
)

_MANIFEST_CAVEATS = (
    "Every value in the manifest carries its own status, authority and source. "
    "The status on this envelope is the manifest as a whole; a consumer "
    "quoting one value must quote that value's own status.",
    "The operator's early statement and the peer-reviewed count disagree. Both "
    "are recorded with their provenance rather than one being silently "
    "preferred.",
    "published_enhancement figures are what results are checked AGAINST. They "
    "are never used as inputs.",
)

_GROUND_TRUTH_CAVEATS = (
    "A comparison against this table is a comparison against a SUBSET, biased "
    "toward objects that survived long enough to be catalogued: some of the "
    "launched objects never received a public catalogue number at all.",
    "The 2023 and 2024 decays reached operational altitude first and were lost "
    "for unrelated reasons; the storm window excludes them, and folding them "
    "in would overstate the event.",
    "A null decay date means the object was still on orbit at the time of "
    "retrieval, not that it survived the storm window.",
)

_RUN_PROVENANCE_CAVEATS = (
    "Served verbatim from the stored artefact. It is a record of the run that "
    "produced it, not a live computation, and may predate the current code.",
    "Regenerate with scripts/run_full_analysis.py before quoting it against a "
    "changed model or changed inputs.",
)


class ModeNotLicensedError(RuntimeError):
    """Raised when a query is refused because the analysis mode does not license it."""


@dataclass(frozen=True)
class QueryResult[ValueT]:
    """A value that cannot travel without the reasons to believe it.

    Construction fails without at least one source and one caveat. Both
    refusals are deliberate. A sourceless answer cannot be traced back to
    anything; a caveatless one asserts that nothing at all remains wrong with
    it, and no answer in this system has ever been in that position.
    """

    value: ValueT
    epistemic_status: DataStatus
    authority: SourceAuthority
    sources: tuple[str, ...]
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError(
                "a query result must name at least one source; an unsourced "
                "answer cannot be traced back to anything"
            )
        if not self.caveats:
            raise ValueError(
                "a query result must carry at least one caveat; an answer with "
                "none claims that nothing remains wrong with it"
            )
        for label, entries in (("sources", self.sources), ("caveats", self.caveats)):
            for index, entry in enumerate(entries):
                if not entry.strip():
                    raise ValueError(
                        f"{label}[{index}] is blank; an empty {label} entry says nothing"
                    )

    @property
    def licence(self) -> str:
        """What a caller holding this result is entitled to claim."""
        return self.epistemic_status.licence


@dataclass(frozen=True, slots=True)
class AtmosphereAnswer:
    """The neutral atmosphere at a point, with the two derived scalars.

    ``mass_density_kg_m3`` is the sum over the species the provider REPORTED.
    It is not the provider's own total, which is carried on ``state``; the
    difference is a diagnostic, not a discrepancy to be reconciled.
    """

    state: AtmosphereState
    mass_density_kg_m3: float
    mean_particle_mass_kg: float


@dataclass(frozen=True, slots=True)
class DragAnswer:
    """A composition-weighted C_d and the drag parameter decay constrains.

    Both are reported because neither is sufficient alone: C_d is the physics,
    while C_d*A/m is the combination an observed decay curve can constrain.
    """

    drag_coefficient: float
    inverse_ballistic_coefficient_m2_kg: float
    geometry: str


@dataclass(frozen=True, slots=True)
class GroundTruthAnswer:
    """The catalogued decays, the storm-window subset, and where they came from."""

    decays: tuple[GroundTruthDecay, ...]
    storm_window: tuple[GroundTruthDecay, ...]
    provenance: GroundTruthProvenance


#: Everything this layer treats as a policy REFUSAL rather than a fault, for the
#: audit stream. :class:`ModeNotLicensedError` belongs here as much as an
#: authorisation refusal does: both are the system correctly declining, and a
#: dashboard that counted them as errors would report an enforced policy as an
#: outage.
_REFUSAL_TYPES: tuple[type[BaseException], ...] = (
    NotAuthorisedError,
    ModeNotLicensedError,
)


def _observed(
    operation: str,
    providers: ProviderSet,
    principal: Principal,
    observability: ObservabilityConfig | None,
) -> contextlib.AbstractContextManager[OperationContext]:
    """The audit context every query runs inside.

    One helper rather than the same seven arguments repeated at five call
    sites: the mode an operation is audited under must be the mode the provider
    set was cleared under, and threading ``providers.mode`` by hand five times
    is five chances for one of them to be something else.
    """
    settings = observability if observability is not None else ObservabilityConfig()
    return observed_operation(
        operation=operation,
        principal=principal,
        mode=providers.mode,
        logger=settings.logger,
        counters=settings.counters,
        credentials=settings.credentials,
        correlation_id=settings.correlation_id,
        refusal_types=_REFUSAL_TYPES,
    )


def _provider_provenance(providers: ProviderSet) -> tuple[str, ...]:
    """The composition's own provenance lines, plus any required attribution."""
    lines = list(providers.sources())
    for clearance in (providers.atmosphere_clearance, providers.wind_clearance):
        record = clearance.record
        if record is not None:
            lines.append(
                f"{record.resource_name}: {record.attribution_text} ({record.source_url})"
            )
    return tuple(lines)


def atmosphere_at(
    providers: ProviderSet,
    *,
    principal: Principal,
    time_utc: _dt.datetime,
    altitude_m: float,
    latitude_deg: float,
    longitude_deg: float,
    drivers: DriverState,
    observability: ObservabilityConfig | None = None,
) -> QueryResult[AtmosphereAnswer]:
    """Evaluate the composed atmosphere provider at one point.

    Requires :attr:`~thermosphere_consequence.service.authorisation.Permission.READ_ATMOSPHERE`.

    Drivers are supplied by the caller rather than fetched. This layer performs
    no network I/O, so it cannot look an index up, and defaulting to a
    plausible F10.7 would produce a number with no traceable input.
    """
    with _observed("atmosphere_at", providers, principal, observability):
        authorise(principal, Permission.READ_ATMOSPHERE, mode=providers.mode)
        state = providers.atmosphere.state_at(
            time_utc, altitude_m, latitude_deg, longitude_deg, drivers
        )
        answer = AtmosphereAnswer(
            state=state,
            mass_density_kg_m3=total_species_mass_density(state.number_densities),
            mean_particle_mass_kg=mean_particle_mass(state.number_densities),
        )
        sources = (
            *_provider_provenance(providers),
            f"drivers: {drivers.source or 'supplied by the caller'} "
            f"(status {drivers.status.value})",
        )
        return QueryResult(
            value=answer,
            epistemic_status=DataStatus.MODELLED,
            authority=SourceAuthority.DERIVED,
            sources=sources,
            caveats=(*providers.caveats, *state.notes, _ATMOSPHERE_SPECIES_SUM_CAVEAT),
        )


def drag_at(
    providers: ProviderSet,
    *,
    principal: Principal,
    number_densities: Mapping[str, float],
    temperature_k: float,
    speed_m_s: float,
    mass_kg: float,
    area_m2: float,
    accommodation: float,
    geometry: str,
    observability: ObservabilityConfig | None = None,
) -> QueryResult[DragAnswer]:
    """Composition-weighted free-molecular C_d and the inverse ballistic coefficient.

    Requires :attr:`~thermosphere_consequence.service.authorisation.Permission.READ_DRAG`.

    ``accommodation`` and ``geometry`` are required rather than defaulted. Both
    are modelling choices, not measurements, and a default would let a caller
    receive a drag coefficient without ever having decided the two assumptions
    that most strongly determine it.
    """
    with _observed("drag_at", providers, principal, observability):
        authorise(principal, Permission.READ_DRAG, mode=providers.mode)
        drag_coefficient = species_weighted_drag_coefficient(
            dict(number_densities),
            temperature_k,
            speed_m_s,
            accommodation=accommodation,
            geometry=geometry,
        )
        answer = DragAnswer(
            drag_coefficient=drag_coefficient,
            inverse_ballistic_coefficient_m2_kg=inverse_ballistic_coefficient_m2_kg(
                drag_coefficient, area_m2, mass_kg
            ),
            geometry=geometry,
        )
        return QueryResult(
            value=answer,
            epistemic_status=DataStatus.DERIVED,
            authority=SourceAuthority.DERIVED,
            sources=(
                *_provider_provenance(providers),
                "Sentman, L.H. (1961), LMSC-448514",
                "Schaaf, S.A. & Chambre, P.L. (1961), Flow of Rarefied Gases, Princeton",
                "Mehta, P.M. et al. (2014), Adv. Space Res. 54:1590",
                "Moe, K. & Moe, M.M. (2005), Planet. Space Sci. 53:793",
            ),
            caveats=(*providers.caveats, *_DRAG_CAVEATS),
        )


def event_manifest(
    providers: ProviderSet,
    *,
    principal: Principal,
    event_id: str = "starlink_group_4_7_2022",
    observability: ObservabilityConfig | None = None,
) -> QueryResult[EventManifest]:
    """The validated event manifest, with the source of every number.

    Requires :attr:`~thermosphere_consequence.service.authorisation.Permission.READ_EVENT`.
    """
    with _observed("event_manifest", providers, principal, observability):
        authorise(principal, Permission.READ_EVENT, mode=providers.mode)
        manifest = load_default_event_manifest(event_id)
        sources = [f"event manifest {manifest.event_id} ({manifest.source_path})"]
        sources.extend(sorted({value.source for value in manifest.values.values()}))
        sources.append(
            f"catalogued decays: {manifest.decay_source} ({manifest.decay_source_url})"
        )
        return QueryResult(
            value=manifest,
            epistemic_status=DataStatus.OBSERVED,
            authority=SourceAuthority.PRIMARY,
            sources=tuple(sources),
            caveats=(
                *providers.caveats,
                f"Manifest analysis mode: {manifest.analysis_mode.value}. "
                f"{manifest.analysis_mode.licence}",
                *_MANIFEST_CAVEATS,
            ),
        )


def ground_truth(
    providers: ProviderSet,
    *,
    principal: Principal,
    store: RawStore | None = None,
    observability: ObservabilityConfig | None = None,
) -> QueryResult[GroundTruthAnswer]:
    """The catalogued decay dates: the only observational anchor in the chain.

    Requires
    :attr:`~thermosphere_consequence.service.authorisation.Permission.READ_GROUND_TRUTH`,
    which ``reader`` does not hold. The permission ladder and the analysis-mode
    gate below are two independent conditions on the same query and both must
    pass: the permission says whether this principal may make a claim about
    specific catalogued objects, and the mode says whether this RUN is entitled
    to make one at all.

    Refused in scenario mode. The decay dates are a claim about what specific
    catalogued objects actually did, and
    :attr:`~thermosphere_consequence.schemas.modes.AnalysisMode.SCENARIO`
    licenses no such claim; serving them anyway would let a bounded
    construction be reported beside real outcomes as though the two were the
    same kind of statement.

    Loaded through the verified loader, so the store's hash check and the
    cross-check against the recorded literals both run before the numbers are
    returned. A disagreement raises rather than being reported as a warning
    nobody reads.
    """
    with _observed("ground_truth", providers, principal, observability):
        authorise(principal, Permission.READ_GROUND_TRUTH, mode=providers.mode)
        if providers.analysis_mode not in _ANALYSIS_MODES_LICENSING_CATALOGUE_CLAIMS:
            raise ModeNotLicensedError(
                f"the catalogued decay dates are refused in analysis mode "
                f"{providers.analysis_mode.value!r}. "
                f"{providers.analysis_mode.licence} "
                f"Re-enter with one of "
                f"{[m.value for m in _ANALYSIS_MODES_LICENSING_CATALOGUE_CLAIMS]}, "
                f"which is what licenses a claim tied to a specific catalogued object."
            )

        decays, provenance = load_verified_ground_truth(
            store if store is not None else RawStore(DEFAULT_RAW_STORE_PATH)
        )
        answer = GroundTruthAnswer(
            decays=decays,
            storm_window=storm_window_losses(decays, window_days=STORM_WINDOW_DAYS),
            provenance=provenance,
        )
        return QueryResult(
            value=answer,
            epistemic_status=DataStatus.OBSERVED,
            authority=SourceAuthority.PRIMARY,
            sources=(
                *_provider_provenance(providers),
                f"{provenance.source} asset {provenance.asset_id} ({provenance.url})",
                f"sha256 {provenance.sha256}, retrieved {provenance.retrieved_at_utc}",
                f"licence: {provenance.licence}",
                f"citation: {provenance.citation}",
            ),
            caveats=(
                *providers.caveats,
                f"Verified against the recorded literals: "
                f"{provenance.verified_against_literals}.",
                f"Storm window: {STORM_WINDOW_DAYS} days from deployment.",
                *_GROUND_TRUTH_CAVEATS,
            ),
        )


def latest_run_provenance(
    providers: ProviderSet,
    *,
    principal: Principal,
    path: pathlib.Path | None = None,
    observability: ObservabilityConfig | None = None,
) -> QueryResult[dict[str, Any]]:
    """The stored record of the last full analysis run, or a plain refusal.

    Requires
    :attr:`~thermosphere_consequence.service.authorisation.Permission.READ_PROVENANCE`
    AND, where the record names an owner, membership of that tenant. The two
    are separate checks because the permission says a principal may read run
    records and says nothing about WHOSE — a run record carries the inputs, the
    provider choices and the commit of somebody's analysis, and a competitor
    holding the same role must not receive it.

    A record with no ``tenant_id`` names no owner and is served to any
    principal holding the permission, carrying
    :data:`~thermosphere_consequence.service.authorisation.UNTENANTED_RESOURCE_IS_SHARED`
    as a caveat. That constant states plainly what the choice costs; the
    alternative, refusing every untenanted record, would make this
    repository's own committed evidence unreadable and would be removed rather
    than complied with.

    A missing artefact raises :class:`FileNotFoundError` naming the script that
    regenerates it. This layer serves the stored record and never fabricates
    one: an invented provenance record is worse than none, because it looks
    like evidence.
    """
    with _observed("latest_run_provenance", providers, principal, observability):
        authorise(principal, Permission.READ_PROVENANCE, mode=providers.mode)
        resolved = path if path is not None else DEFAULT_RUN_PROVENANCE_PATH
        if not resolved.is_file():
            raise FileNotFoundError(
                f"no run provenance record at {resolved}. Run "
                f"scripts/run_full_analysis.py to generate it. This service serves "
                f"the stored artefact and never fabricates one."
            )
        record: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
        owner = record.get(RUN_RECORD_TENANT_FIELD)
        authorise_tenant(
            principal,
            resource_tenant_id=owner,
            resource_description=f"the run provenance record at {resolved}",
        )
        sources = [str(resolved)]
        run_id = record.get("run_id")
        if run_id:
            sources.append(f"run_id: {run_id}")
        commit = record.get("git_commit")
        if commit:
            sources.append(f"git commit: {commit}")
        ownership_caveat = (
            f"This run record belongs to tenant {owner!r}, which is the "
            f"requesting principal's tenant."
            if owner is not None
            else UNTENANTED_RESOURCE_IS_SHARED
        )
        return QueryResult(
            value=record,
            epistemic_status=DataStatus.DERIVED,
            authority=SourceAuthority.DERIVED,
            sources=tuple(sources),
            caveats=(*providers.caveats, *_RUN_PROVENANCE_CAVEATS, ownership_caveat),
        )
