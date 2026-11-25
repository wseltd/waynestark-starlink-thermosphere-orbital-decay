"""The controlled vocabulary a run declares itself in.

A mode is not a label for a report header. It is a licence: it fixes the set of
sentences the run is entitled to write. Every member below therefore carries a
``licence`` string saying, in plain words, what a caller may claim while in that
mode — and by omission, what it may not.

The reason this exists as data rather than prose is that
:mod:`thermosphere_consequence.schemas.claim_gate` evaluates predicates over
these members at emission time. A run that wants to say "native SURYA" has to
be *in* a native SURYA mode; it cannot get there by rewording the sentence.

Canonical tickets: P00_02_T033, P11_12_T030.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["AnalysisMode", "ForceClosure", "ModelMode", "DataStatus", "SourceAuthority"]


class AnalysisMode(StrEnum):
    """What the numbers describe: a construction, an object, or the truth.

    This is the mode that separates "we propagated a published insertion state"
    from "we reconstructed what this spacecraft did". The distinction is the
    whole difference between a scenario and a history, and it is the one most
    easily lost between a notebook and a headline.
    """

    SCENARIO = "scenario"
    TRACKED_OBJECT = "tracked_object"
    OPERATOR_TRUTH = "operator_truth"

    @property
    def licence(self) -> str:
        """What a caller in this mode is entitled to claim."""
        return _ANALYSIS_MODE_LICENCE[self]


_ANALYSIS_MODE_LICENCE: dict[AnalysisMode, str] = {
    AnalysisMode.SCENARIO: (
        "Licenses claims about a bounded, self-declared construction: what the "
        "implemented models produce from stated initial conditions and stated "
        "assumptions. Does NOT license any claim that the result is what a real "
        "object did, that a path is an actual trajectory, or that a re-entry "
        "time is the one that occurred."
    ),
    AnalysisMode.TRACKED_OBJECT: (
        "Licenses claims tied to a specific catalogued object over the interval "
        "for which public element sets or catalogue records exist, including "
        "comparisons against catalogued decay. Does NOT license claims about "
        "intervals or objects with no tracking data, nor about the operator's "
        "own state estimate."
    ),
    AnalysisMode.OPERATOR_TRUTH: (
        "Licenses claims sourced from the operator's own telemetry, ephemeris "
        "or state estimate, attributed to the operator. Requires that the "
        "operator record actually be in hand; it is not a mode a modelling run "
        "may enter by inference."
    ),
}


class ModelMode(StrEnum):
    """Which engine actually produced a model output — including "none".

    ``native_surya`` means the published model was executed by this project.
    Everything else is a stand-in of some kind, and the difference has to
    survive into the report, because "SURYA says" and "our heuristic says" are
    not the same sentence.
    """

    NATIVE_SURYA = "native_surya"
    CACHED_NATIVE_SURYA = "cached_native_surya"
    HEURISTIC = "heuristic"
    MOCK = "mock"
    STRUCTURAL_ADAPTER = "structural_adapter"
    NATIVE_SURYA_NOT_RUN = "native_surya_not_run"

    @property
    def licence(self) -> str:
        """What a caller in this mode is entitled to claim."""
        return _MODEL_MODE_LICENCE[self]

    @property
    def is_native_surya(self) -> bool:
        """True only when the output came from the published model itself.

        Cached native output counts: the weights and the code that produced it
        were the real ones, and the cache is a transport detail. Nothing else
        counts, however faithful its interface.
        """
        return self in (ModelMode.NATIVE_SURYA, ModelMode.CACHED_NATIVE_SURYA)


_MODEL_MODE_LICENCE: dict[ModelMode, str] = {
    ModelMode.NATIVE_SURYA: (
        "Licenses attributing the output to SURYA: that native inference ran in "
        "this project, on stated inputs, at a stated version."
    ),
    ModelMode.CACHED_NATIVE_SURYA: (
        "Licenses attributing the output to SURYA, provided the cache key and "
        "the originating run are recorded. The inference was native; only its "
        "execution is historical."
    ),
    ModelMode.HEURISTIC: (
        "Licenses claims about a hand-written approximation defined in this "
        "repository. Does NOT license naming SURYA, or any published model, as "
        "the source of the number."
    ),
    ModelMode.MOCK: (
        "Licenses claims about interface behaviour and control flow only. Does "
        "NOT license any physical or predictive claim whatsoever: the values "
        "are fixtures."
    ),
    ModelMode.STRUCTURAL_ADAPTER: (
        "Licenses claims about shapes, dtypes, units and call contracts — that "
        "the integration is wired correctly. Does NOT license any claim about "
        "model skill, because no model was consulted."
    ),
    ModelMode.NATIVE_SURYA_NOT_RUN: (
        "Licenses one claim and no others: that native SURYA inference was not "
        "run by this project. It is the honest default, and it forbids every "
        "SURYA-attributed result."
    ),
}


class ForceClosure(StrEnum):
    """How much of the gas-surface momentum transfer the applied FORCE resolves.

    This is the mode that separates a rich diagnostic from a force closure, and
    the two are easy to conflate because they draw on the same species state.
    The engine resolves the thermosphere by constituent — number density, mass
    density, mean particle mass, scale height, thermal speed, speed ratio, mean
    free path, Knudsen number, impact flux, impact energy — and it also mass-flux
    weights those constituents into ONE drag coefficient
    (:func:`~thermosphere_consequence.aerodynamics.sentman.species_weighted_drag_coefficient`).
    But the force put on the orbit is then the scalar
    ``F = 0.5 * C_d * rho_total * A * v_rel^2``: one coefficient, one total
    density, one projected area. Composition INFORMS that coefficient; it does
    not enter the force any other way, and at fixed total density redistributing
    mass between species leaves the force exactly unchanged.

    Declaring which of these a run is in is what stops "species-resolved
    diagnostics" from being read, or written, as "species-resolved force".
    """

    EFFECTIVE_SCALAR = "effective_scalar"
    FACETED_DIAGNOSTIC = "faceted_diagnostic"
    SPECIES_RESOLVED_FACETED = "species_resolved_faceted"

    @property
    def licence(self) -> str:
        """What a caller in this mode is entitled to claim."""
        return _FORCE_CLOSURE_LICENCE[self]


_FORCE_CLOSURE_LICENCE: dict[ForceClosure, str] = {
    ForceClosure.EFFECTIVE_SCALAR: (
        "Licenses 'effective scalar free-molecular drag', and licenses saying "
        "that species composition informs the effective C_d by mass-flux "
        "weighting. Does NOT license 'species-resolved force', 'per-species "
        "force' or any phrasing in which individual constituents determine the "
        "applied force independently: the force is one C_d against one total "
        "density."
    ),
    ForceClosure.FACETED_DIAGNOSTIC: (
        "Licenses claims about a per-facet, per-species force VECTOR computed "
        "as a comparison — the sum over facets and species that "
        "``aerodynamics.faceted`` performs. Does NOT license calling it a "
        "closure: it does not drive the orbit, and it carries one sourced "
        "energy accommodation coefficient applied to every species, because no "
        "open dataset supplies per-species values."
    ),
    ForceClosure.SPECIES_RESOLVED_FACETED: (
        "Licenses 'species-resolved force closure': that the force integrated "
        "into the trajectory is itself the sum over species and facets, with "
        "species-dependent accommodation. Requires that closure actually be the "
        "force model of the run, with per-species coefficients that carry a "
        "source. It is not a mode this repository has reached."
    ),
}


class DataStatus(StrEnum):
    """How a number came to exist, from the reader's point of view.

    A narrower vocabulary than
    :class:`thermosphere_consequence.domain.provenance.ValueStatus`, which
    tracks per-value provenance in fine detail. This one is the run-level
    summary the claim gate reasons over: it answers "may this run use the word
    *measured*?".
    """

    OBSERVED = "observed"
    MODELLED = "modelled"
    DERIVED = "derived"
    ASSUMED = "assumed"
    PROVISIONAL = "provisional"
    DEFINITIVE = "definitive"

    @property
    def licence(self) -> str:
        """What a caller in this mode is entitled to claim."""
        return _DATA_STATUS_LICENCE[self]


_DATA_STATUS_LICENCE: dict[DataStatus, str] = {
    DataStatus.OBSERVED: (
        "Licenses 'measured', 'observed' and 'catalogued' for values that an "
        "instrument or a public catalogue recorded, with the instrument or "
        "catalogue named."
    ),
    DataStatus.MODELLED: (
        "Licenses 'modelled', 'the model gives' and 'predicted by'. Does NOT "
        "license 'measured' or 'observed' for the same quantity, however well "
        "the model is regarded."
    ),
    DataStatus.DERIVED: (
        "Licenses 'derived', 'inferred' and 'computed from', with the inputs "
        "named. The status of the weakest input governs the strength of the "
        "conclusion."
    ),
    DataStatus.ASSUMED: (
        "Licenses 'assumed' and 'taken as', with the assumption stated. Does "
        "NOT license presenting the value as an outcome of the analysis."
    ),
    DataStatus.PROVISIONAL: (
        "Licenses use of the value with an explicit provisional marker, and "
        "requires that the marker travel with it into any derived figure."
    ),
    DataStatus.DEFINITIVE: (
        "Licenses use of the value as the final published version from its "
        "source, with the source's own version identifier recorded."
    ),
}


class SourceAuthority(StrEnum):
    """How much weight a citation can carry.

    Kept separate from :class:`DataStatus` because they vary independently: an
    operator statement is a strong authority for a weak status (an early
    estimate), and a third-party fit can be a weak authority for something
    presented as observed.
    """

    PRIMARY = "primary"
    PEER_REVIEWED = "peer_reviewed"
    OPERATOR = "operator"
    THIRD_PARTY = "third_party"
    #: A published value for a DIFFERENT vehicle, borrowed as a reference point.
    #:
    #: Added 11 September 2026, after a value carried here as ``OPERATOR`` for
    #: the spacecraft under study turned out to be a third party's figure for an
    #: unrelated satellite of a different class. Every other authority in this
    #: enum answers "how much weight does the citation carry"; none of them
    #: could express "the citation is sound, and it is about something else",
    #: so that failure had no vocabulary and went unrecorded. It has one now.
    #: A value at this authority may be used as a labelled external reference
    #: and may NEVER be described as characterising the vehicle under study.
    THIRD_PARTY_OTHER_VEHICLE = "third_party_other_vehicle"
    DERIVED = "derived"
    UNVERIFIED = "unverified"

    @property
    def licence(self) -> str:
        """What a caller citing a source of this authority may claim."""
        return _SOURCE_AUTHORITY_LICENCE[self]


_SOURCE_AUTHORITY_LICENCE: dict[SourceAuthority, str] = {
    SourceAuthority.PRIMARY: (
        "Licenses citing the value as the record itself — a public catalogue "
        "entry, an archived data file, an instrument product — identified well "
        "enough to be retrieved again."
    ),
    SourceAuthority.PEER_REVIEWED: (
        "Licenses citing the value as published and reviewed, with DOI. Does "
        "NOT license presenting it as the project's own result."
    ),
    SourceAuthority.OPERATOR: (
        "Licenses citing the value as the operator's own statement, attributed "
        "to the operator and dated. Does NOT license treating it as "
        "independently verified, and early statements must keep their date."
    ),
    SourceAuthority.THIRD_PARTY: (
        "Licenses citing the value as a third-party estimate, named as such. "
        "Does NOT license presenting it as an operator disclosure or as a "
        "measurement."
    ),
    SourceAuthority.THIRD_PARTY_OTHER_VEHICLE: (
        "Licenses citing the value as a published figure FOR A DIFFERENT "
        "VEHICLE, used as a labelled external reference point and named as "
        "belonging to that vehicle. Does NOT license describing it as "
        "characterising the spacecraft under study, does NOT license calling "
        "it an operator disclosure for this study's subject, and does NOT "
        "license using it as an out-of-sample anchor without checking what it "
        "was itself fitted against."
    ),
    SourceAuthority.DERIVED: (
        "Licenses citing the value as computed within this project from other "
        "manifest values, with those inputs named."
    ),
    SourceAuthority.UNVERIFIED: (
        "Licenses nothing beyond repeating the claim with its origin and the "
        "fact that it was not verified. It must never be an input to a result."
    ),
}
