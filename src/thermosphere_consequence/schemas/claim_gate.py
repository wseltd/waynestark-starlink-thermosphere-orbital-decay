"""The claim gate: prohibited statements enforced at emission time.

A list of forbidden sentences checked against finished prose is a spell
checker. It catches the wording somebody happened to use last time and misses
the same claim phrased differently, and — worse — it says nothing at all about
values, which is where the error this module was written for actually occurred.

So the gate here is built the other way round. A prohibited claim is a
*predicate over the declared mode set*: "actual trajectory" is not a banned
phrase, it is a phrase that is banned **while the run is in scenario mode**,
and permitted while the run is in tracked-object mode. The same list therefore
serves both directions — it says what may not be said, and it says what would
have to be true for the sentence to be allowed.

Two entry points, for two different moments:

* :func:`check_text` / :func:`enforce` run over text at the point it is
  generated, with the run's context in hand.
* :func:`guard_value` runs over a *number* at the point it is emitted, where
  its units are still known. This is the one that matters most. The specific
  error this project made was publishing C_d*A/m — 0.085 m^2/kg — under the
  name "ballistic coefficient", which is the reciprocal quantity in reciprocal
  units. No amount of prose scanning prevents that; a units check at the
  emission point does, and after this module exists it is impossible to do
  again without the gate raising.

Canonical tickets: P00_02_T023, P11_12_T020, P11_12_T030.
"""

from __future__ import annotations

import functools
import re
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .modes import AnalysisMode, DataStatus, ForceClosure, ModelMode, SourceAuthority

__all__ = [
    "CalibrationStatus",
    "ClaimContext",
    "ClaimOccurrence",
    "ClaimViolation",
    "ProhibitedClaim",
    "UncertaintyBasis",
    "CONVENTIONAL_DRAG_UNITS",
    "INVERSE_DRAG_UNITS",
    "PROHIBITED_CLAIMS",
    "check_text",
    "enforce",
    "guard_value",
    "normalise_units",
]


class UncertaintyBasis(StrEnum):
    """What an uncertainty range on a result actually is.

    A spread produced by re-running a model over a few chosen parameter values
    is a scenario envelope. It is not a confidence interval, it carries no
    probability, and it cannot support a statement about when something will
    happen or how likely it is.
    """

    SCENARIO = "scenario"
    SAMPLED_POSTERIOR = "sampled_posterior"
    MEASURED_COVARIANCE = "measured_covariance"
    NONE = "none"


class CalibrationStatus(StrEnum):
    """Whether the parameters were fitted on the data now being compared.

    ``calibrated_in_sample`` is the honest description of a fit whose agreement
    with the data is a property of the fitting, not evidence about the world.
    Calling that "validated" is the claim this status exists to block.
    """

    CALIBRATED_IN_SAMPLE = "calibrated_in_sample"
    OUT_OF_SAMPLE_VALIDATED = "out_of_sample_validated"
    NOT_CALIBRATED = "not_calibrated"


#: The two drag-parameter unit strings, canonicalised. They are reciprocals of
#: each other and are never interchangeable; see
#: :mod:`thermosphere_consequence.domain.drag_parameters`.
INVERSE_DRAG_UNITS = "m^2/kg"
CONVENTIONAL_DRAG_UNITS = "kg/m^2"

_INVERSE_DRAG_SPELLINGS = frozenset(
    {"m^2/kg", "m2/kg", "m^2kg^-1", "m^2*kg^-1", "m^2.kg^-1", "m^2·kg^-1", "m^2perkg"}
)
_CONVENTIONAL_DRAG_SPELLINGS = frozenset(
    {"kg/m^2", "kg/m2", "kgm^-2", "kg*m^-2", "kg.m^-2", "kg·m^-2", "kgperm^2"}
)


def normalise_units(units: str) -> str:
    """Fold the spellings of a unit string onto one canonical form.

    Only the drag-parameter units are canonicalised, because they are the pair
    this gate has to tell apart; anything else is returned in its whitespace-
    stripped lowercase form so that callers can compare it if they wish.
    """
    text = units.strip().lower()
    for old, new in (
        ("meters", "m"),
        ("metres", "m"),
        ("meter", "m"),
        ("metre", "m"),
        ("kilograms", "kg"),
        ("kilogram", "kg"),
        ("**", "^"),
        (" ", ""),
    ):
        text = text.replace(old, new)
    if text in _INVERSE_DRAG_SPELLINGS:
        return INVERSE_DRAG_UNITS
    if text in _CONVENTIONAL_DRAG_SPELLINGS:
        return CONVENTIONAL_DRAG_UNITS
    return text


@dataclass(frozen=True, slots=True)
class ClaimContext:
    """The declared position a run is speaking from.

    The four mode fields have no defaults: a caller must state where it stands
    before the gate will judge what it says. The remaining fields default to
    the most restrictive value, so an under-specified context forbids more
    rather than less — an omission cannot buy permission.
    """

    analysis_mode: AnalysisMode
    model_mode: ModelMode
    data_status: DataStatus
    source_authority: SourceAuthority
    uncertainty_basis: UncertaintyBasis = UncertaintyBasis.SCENARIO
    has_probabilistic_model: bool = False
    value_units: str | None = None
    attribution_ordering_stable: bool | None = None
    calibration_status: CalibrationStatus = CalibrationStatus.CALIBRATED_IN_SAMPLE
    force_closure: ForceClosure = ForceClosure.EFFECTIVE_SCALAR

    def __post_init__(self) -> None:
        for field_name, enum_type in (
            ("analysis_mode", AnalysisMode),
            ("model_mode", ModelMode),
            ("data_status", DataStatus),
            ("source_authority", SourceAuthority),
            ("uncertainty_basis", UncertaintyBasis),
            ("calibration_status", CalibrationStatus),
            ("force_closure", ForceClosure),
        ):
            raw = getattr(self, field_name)
            try:
                object.__setattr__(self, field_name, enum_type(raw))
            except ValueError as exc:
                permitted = [m.value for m in enum_type]
                raise ValueError(
                    f"{field_name}: {raw!r} is not a known {enum_type.__name__}; "
                    f"permitted: {permitted}"
                ) from exc
        stable = self.attribution_ordering_stable
        if stable is not None and not isinstance(stable, bool):
            raise ValueError(
                "attribution_ordering_stable must be True, False or None "
                f"(not established), got {stable!r}"
            )

    @property
    def drag_units_are_inverse(self) -> bool:
        """True when the value under discussion is C_d*A/m in m^2/kg."""
        return (
            self.value_units is not None
            and normalise_units(self.value_units) == INVERSE_DRAG_UNITS
        )

    def summary(self) -> str:
        """A one-line rendering of the declared position, for error messages."""
        return (
            f"analysis_mode={self.analysis_mode.value}, "
            f"model_mode={self.model_mode.value}, "
            f"data_status={self.data_status.value}, "
            f"source_authority={self.source_authority.value}, "
            f"uncertainty_basis={self.uncertainty_basis.value}, "
            f"has_probabilistic_model={self.has_probabilistic_model}, "
            f"value_units={self.value_units!r}, "
            f"attribution_ordering_stable={self.attribution_ordering_stable}, "
            f"calibration_status={self.calibration_status.value}, "
            f"force_closure={self.force_closure.value}"
        )


@dataclass(frozen=True, slots=True)
class ClaimOccurrence:
    """One place a prohibited phrase appears, with the text around it.

    ``preceding`` and ``following`` are windows into the *whole* document, not
    the line, because prose wraps: "it is not the ballistic coefficient\\nB =
    m/(C_d*A)" is one statement split across two lines, and a line-bounded view
    would misread it.
    """

    line_number: int
    line_text: str
    matched_text: str
    column: int
    preceding: str
    following: str


class ClaimViolation(Exception):
    """A claim that the declared context does not permit.

    Used two ways: raised by :func:`enforce` and :func:`guard_value`, and
    returned as a value by :func:`check_text`, which reports every violation in
    a document rather than stopping at the first.
    """

    def __init__(
        self,
        *,
        claim_pattern: str,
        reason: str,
        permitted_alternative: str,
        matched_text: str = "",
        line_number: int = 0,
        line_text: str = "",
        column: int = 0,
        context_summary: str = "",
        source: str = "",
        also_found: Sequence[str] = (),
    ) -> None:
        self.claim_pattern = claim_pattern
        self.reason = reason
        self.permitted_alternative = permitted_alternative
        self.matched_text = matched_text
        self.line_number = line_number
        self.line_text = line_text
        self.column = column
        self.context_summary = context_summary
        self.source = source
        self.also_found = tuple(also_found)
        super().__init__(self._message())

    def _message(self) -> str:
        where = f"{self.source}:{self.line_number}: " if self.source else ""
        if not where and self.line_number:
            where = f"line {self.line_number}: "
        matched = f" (matched {self.matched_text!r})" if self.matched_text else ""
        parts = [
            f"{where}prohibited claim {self.claim_pattern!r}{matched}",
            f"reason: {self.reason}",
            f"permitted instead: {self.permitted_alternative}",
        ]
        if self.context_summary:
            parts.append(f"declared context: {self.context_summary}")
        if self.also_found:
            parts.append(f"also found: {', '.join(repr(p) for p in self.also_found)}")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class ProhibitedClaim:
    """A claim, the condition that forbids it, and what to say instead.

    ``forbidden_when`` is the heart of it: a predicate over
    :class:`ClaimContext`. The phrase is not wrong in itself — it is wrong from
    a particular position — so the gate records the position rather than the
    sentence.

    ``occurrence_exempt`` is a narrow structural escape for phrases whose
    *correct* use contains the prohibited substring. "Inverse ballistic
    coefficient" is the required term for C_d*A/m; flagging it would make the
    gate punish the fix it exists to enforce.
    """

    claim_pattern: str
    forbidden_when: Callable[[ClaimContext], bool]
    reason: str
    permitted_alternative: str
    aliases: tuple[str, ...] = ()
    occurrence_exempt: Callable[[ClaimOccurrence], bool] | None = None

    def phrases(self) -> tuple[str, ...]:
        """The canonical phrase plus any recorded variants."""
        return (self.claim_pattern, *self.aliases)


@functools.lru_cache(maxsize=256)
def _phrase_regex(phrase: str) -> re.Pattern[str]:
    """Compile a phrase to a whitespace-tolerant, case-insensitive matcher.

    Spaces become ``\\s+`` so a phrase broken across a line in wrapped
    markdown still matches; the word boundaries stop "revalidated" from
    counting as "validated".
    """
    tokens = [re.escape(token) for token in phrase.split()]
    return re.compile(r"\b" + r"[\s]+".join(tokens) + r"\b", re.IGNORECASE)


# --- the predicates -------------------------------------------------------
#
# Named functions rather than lambdas so the error messages, the tests and the
# audit report can all refer to the same thing by name.


def _analysis_is_scenario(context: ClaimContext) -> bool:
    return context.analysis_mode is AnalysisMode.SCENARIO


def _surya_did_not_run(context: ClaimContext) -> bool:
    return not context.model_mode.is_native_surya


def _quantity_is_modelled(context: ClaimContext) -> bool:
    return context.data_status is DataStatus.MODELLED


def _uncertainty_is_scenario(context: ClaimContext) -> bool:
    return context.uncertainty_basis is UncertaintyBasis.SCENARIO


def _no_probabilistic_model(context: ClaimContext) -> bool:
    return not context.has_probabilistic_model


def _units_are_inverse_drag(context: ClaimContext) -> bool:
    return context.drag_units_are_inverse


def _attribution_ordering_not_established(context: ClaimContext) -> bool:
    # None means "not established", which forbids the claim exactly as False
    # does. Only an explicit True licenses it.
    return context.attribution_ordering_stable is not True


def _calibrated_on_the_same_data(context: ClaimContext) -> bool:
    return context.calibration_status is CalibrationStatus.CALIBRATED_IN_SAMPLE


def _force_is_not_species_resolved(context: ClaimContext) -> bool:
    # Both EFFECTIVE_SCALAR and FACETED_DIAGNOSTIC forbid it, for different
    # reasons: the first does not resolve species in the force at all, and the
    # second resolves them in a force it never applies to the orbit. Only the
    # closure itself, driving the trajectory, licenses the word.
    return context.force_closure is not ForceClosure.SPECIES_RESOLVED_FACETED


def _no_established_predictive_skill(context: ClaimContext) -> bool:
    """True unless the run declares everything a claim of *prediction* needs.

    A conjunction rather than a single field, because prediction is not one
    property. It needs a probabilistic model to produce a statement about an
    unseen case, an uncertainty basis that carries coverage rather than a swept
    envelope, and an out-of-sample evaluation to show the agreement is skill
    and not the fit. Any one of the three missing forbids the word, so the
    claim cannot be unlocked by relabelling a single mode. This repository is
    missing all three.
    """
    return (
        not context.has_probabilistic_model
        or context.uncertainty_basis is UncertaintyBasis.SCENARIO
        or context.calibration_status is not CalibrationStatus.OUT_OF_SAMPLE_VALIDATED
    )


# --- structural exemptions ------------------------------------------------

_INVERSE_PREFIX = re.compile(r"inverse[\s\-_]*$", re.IGNORECASE)
_CONVENTIONAL_FORM = re.compile(
    r"^[\s_]*(?:"
    r"b\s*=\s*m\s*/|"  # "B = m/(C_d*A)"
    r"m\s*/\s*\(?\s*c_?d|"  # "m/(C_d*A)" given without the "B ="
    r"_?kg[\s_/·*]*m\^?\s*-?\s*2|"  # "_kg_m2", "kg/m^2", "kg m^-2"
    r"\(\s*kg\s*/\s*m\^?2\s*\)|"
    r"in\s+kg\s*/\s*m\^?2"
    r")",
    re.IGNORECASE,
)


def _is_correct_drag_terminology(occurrence: ClaimOccurrence) -> bool:
    """True when this occurrence is the *right* use of the words.

    Two cases are legitimate and must not be reported:

    * the phrase is qualified as INVERSE, which is the required name for
      C_d*A/m in m^2/kg;
    * the phrase is immediately followed by the conventional definition or its
      units, i.e. the text is genuinely talking about B = m/(C_d*A) in kg/m^2.
    """
    if _INVERSE_PREFIX.search(occurrence.preceding):
        return True
    return bool(_CONVENTIONAL_FORM.search(occurrence.following[:64]))


#: The reviewer's list. Each entry is keyed to the mode that forbids it, so the
#: same table doubles as documentation of what would have to be true for the
#: claim to become permissible.
PROHIBITED_CLAIMS: tuple[ProhibitedClaim, ...] = (
    ProhibitedClaim(
        claim_pattern="actual trajectory",
        forbidden_when=_analysis_is_scenario,
        reason=(
            "a scenario run propagates a stated initial condition under stated "
            "assumptions; it does not reconstruct what any object did"
        ),
        permitted_alternative=(
            "'modelled scenario path from the published insertion state' — or "
            "declare analysis_mode=tracked_object and cite the element sets"
        ),
        aliases=("actual trajectories", "true trajectory", "true trajectories"),
    ),
    ProhibitedClaim(
        claim_pattern="native SURYA",
        forbidden_when=_surya_did_not_run,
        reason="native SURYA inference was not executed in this model mode",
        permitted_alternative=(
            "'native SURYA inference was not run by this project' — or declare "
            "model_mode=native_surya/cached_native_surya and record the version"
        ),
        aliases=("SURYA inference was run", "we ran SURYA"),
    ),
    ProhibitedClaim(
        claim_pattern="SURYA ran",
        forbidden_when=_surya_did_not_run,
        reason="native SURYA inference was not executed in this model mode",
        permitted_alternative=(
            "'SURYA is carried as context only and was not run' — or declare "
            "model_mode=native_surya and record the version and inputs"
        ),
        aliases=("SURYA was run", "SURYA has run"),
    ),
    ProhibitedClaim(
        claim_pattern="measured density",
        forbidden_when=_quantity_is_modelled,
        reason=(
            "the density here is an empirical-model output, not an instrument "
            "product; no accelerometer or mass spectrometer measured it"
        ),
        permitted_alternative=(
            "'modelled density' — or cite the accelerometer-derived density "
            "and declare data_status=observed"
        ),
        aliases=("measured densities", "observed density at 210 km"),
    ),
    ProhibitedClaim(
        claim_pattern="exact re-entry time",
        forbidden_when=_uncertainty_is_scenario,
        reason=(
            "a scenario envelope carries no probability, so it cannot support a "
            "point statement about when re-entry occurred or will occur"
        ),
        permitted_alternative=(
            "'catalogued decay date' for the observed fact, or 'modelled decay "
            "time under the stated scenario' for the computed one"
        ),
        aliases=("exact reentry time", "precise re-entry time", "exact re-entry epoch"),
    ),
    ProhibitedClaim(
        claim_pattern="probability of loss",
        forbidden_when=_no_probabilistic_model,
        reason=(
            "no probabilistic model is defined, so no number in this project "
            "carries a probability of anything"
        ),
        permitted_alternative=(
            "'fraction of the launched fleet that was lost' for the counted "
            "outcome, or state the probabilistic model and its assumptions"
        ),
        aliases=("loss probability", "probability of failure"),
    ),
    ProhibitedClaim(
        claim_pattern="ballistic coefficient",
        forbidden_when=_units_are_inverse_drag,
        reason=(
            "the value is in m^2/kg, which is C_d*A/m — the INVERSE ballistic "
            "coefficient. The ballistic coefficient B = m/(C_d*A) is its "
            "reciprocal in kg/m^2 and moves in the opposite direction"
        ),
        permitted_alternative=(
            "'inverse ballistic coefficient (C_d*A/m, m^2/kg)' — or convert and "
            "report B = m/(C_d*A) in kg/m^2"
        ),
        aliases=("ballistic coefficients",),
        occurrence_exempt=_is_correct_drag_terminology,
    ),
    ProhibitedClaim(
        claim_pattern="primary cause",
        forbidden_when=_attribution_ordering_not_established,
        reason=(
            "attributing primacy requires the ordering of the competing causes "
            "to hold across the whole uncertainty envelope; that has not been "
            "established, so attribution_ordering_stable is not True"
        ),
        permitted_alternative=(
            "'contributed, alongside the insertion altitude' — or establish "
            "ordering stability across the envelope and set the flag"
        ),
        aliases=("main cause", "dominant cause", "root cause"),
    ),
    ProhibitedClaim(
        claim_pattern="validated",
        forbidden_when=_calibrated_on_the_same_data,
        reason=(
            "the parameter was fitted on the same data it is then compared "
            "with; agreement is in-sample fit, which is a property of the "
            "fitting rather than evidence about the world"
        ),
        permitted_alternative=(
            "'calibrated in-sample' — or hold data out, record the holdout, and "
            "declare calibration_status=out_of_sample_validated"
        ),
        aliases=("validation confirms", "independently validated"),
    ),
    ProhibitedClaim(
        claim_pattern="species-resolved force",
        forbidden_when=_force_is_not_species_resolved,
        reason=(
            "the force integrated into the trajectory is "
            "0.5*C_d*rho_total*A*v_rel^2 — ONE effective scalar drag "
            "coefficient against the TOTAL mass density. Species composition "
            "informs that coefficient, by mass-flux weighting in "
            "aerodynamics.sentman.species_weighted_drag_coefficient, but no "
            "constituent enters the force independently: at fixed total "
            "density, redistributing mass between species leaves the force "
            "exactly unchanged. The species resolution is in the diagnostics "
            "and in the coefficient, not in the closure"
        ),
        permitted_alternative=(
            "'a species-resolved molecular diagnostic and interpretation layer "
            "accompanying an effective scalar free-molecular drag model, whose "
            "effective C_d is informed by species composition through mass-flux "
            "weighting' — or build the closure, drive the orbit with it, and "
            "declare force_closure=species_resolved_faceted"
        ),
        # "per-species force" is deliberately NOT here, and the narrower
        # "per-species force closure" is. A per-species force term is a real
        # object in this package — effective_drag.species_drag_decomposition
        # computes one under a common C_d, and aerodynamics.faceted sums one
        # over facets as a comparison — so gating the bare phrase would flag
        # the accurate descriptions of both and push the documentation into
        # denying something the code does. What is forbidden is the CLOSURE:
        # the assertion that such terms are what the trajectory is integrated
        # under.
        aliases=(
            "species resolved force",
            "species-resolved drag force",
            "species-resolved free-molecular force",
            "species-resolved momentum transfer",
            "species-resolved momentum-transfer closure",
            "complete species-resolved closure",
            "per-species force closure",
            "per species force closure",
        ),
    ),
    # --- the seven statements frozen by evidence/FINAL_SCIENTIFIC_STATUS.md ---
    #
    # That document freezes what this project may and may not be said to have
    # shown. Two of the seven were already unreachable through entries above —
    # "modelled density is measured density" through `measured density`, and
    # the exactness of a re-entry through `exact re-entry time` — so only the
    # remaining five need new entries here, plus the reconstruction claim,
    # which the manifest register held as a sentence but the runtime gate did
    # not. Each is keyed to the mode that forbids it, exactly as above: the
    # document's list is a statement about executable behaviour, not prose.
    ProhibitedClaim(
        claim_pattern="unique cause",
        forbidden_when=_attribution_ordering_not_established,
        reason=(
            "naming one cause as the only one requires the ordering of the "
            "competing causes to survive the whole uncertainty envelope. It "
            "does not: the ordering flips at 35 of the 45 envelope corners "
            "tested, so attribution_ordering_stable is not True. A storm "
            "cannot be proved to be the unique cause of a decay whose "
            "atmosphere, drag state, attitude and propulsion history are all "
            "unobserved over the interval that mattered"
        ),
        permitted_alternative=(
            "'the storm contributed, alongside the insertion altitude and the "
            "fitted drag state' — or establish ordering stability across the "
            "envelope and set attribution_ordering_stable=True"
        ),
        aliases=(
            "sole cause",
            "only cause",
            "proved to be the cause",
            "proven cause",
            "uniquely caused",
            "uniquely responsible",
        ),
    ),
    ProhibitedClaim(
        claim_pattern="real spacecraft lifetime",
        forbidden_when=_quantity_is_modelled,
        reason=(
            "the storm cost is the difference between two model branches at one "
            "parameter choice — 0.4664 d, evidence/analysis_report.json — not a "
            "quantity anything measured. No spacecraft was flown twice, once "
            "with the storm and once without. The difference is 0.93 of the "
            "+/-0.5 d half-width of the calendar-day decay records that anchor "
            "it, so no catalogue of those records could resolve it at all"
        ),
        permitted_alternative=(
            "'the modelled lifetime difference between the storm and quiet "
            "branches, 0.4664 d under the stated parameter choice, inside a "
            "0.183-2.720 d sensitivity envelope that carries no coverage "
            "probability' — or cite an observed lifetime penalty and declare "
            "data_status=observed"
        ),
        aliases=(
            "actual spacecraft lifetime",
            "true spacecraft lifetime",
            "measured storm penalty",
            "observed storm penalty",
            "measured lifetime penalty",
            "actual lifetime penalty",
        ),
    ),
    ProhibitedClaim(
        claim_pattern="determines the projected area",
        forbidden_when=_units_are_inverse_drag,
        reason=(
            "the value under discussion is C_d*A/m in m^2/kg, which is a lumped "
            "product and not a property of the vehicle. A decay time constrains "
            "only rho * C_d*A/m: density scale factors 0.25-4 reproduce the same "
            "9.000 d decay while C_d*A/m moves from 0.005271 to 0.084334 m^2/kg. "
            "One fixed geometry swept through attitude spans 0.0013-0.0850 "
            "m^2/kg, which brackets the whole fitted range, so the fitted value "
            "singles out neither an attitude nor an area"
        ),
        permitted_alternative=(
            "'the fitted C_d*A/m is consistent with a wide range of attitudes "
            "and projected areas and singles out none of them' — or bring an "
            "independent constraint on the area, and declare its own units "
            "rather than the drag parameter's"
        ),
        aliases=(
            "determines projected area",
            "determines the attitude",
            "determines attitude",
            "fixes the projected area",
            "fixes the attitude",
            "implies the projected area",
            "threefold area reduction",
        ),
    ),
    ProhibitedClaim(
        claim_pattern="trajectories were reconstructed",
        forbidden_when=_analysis_is_scenario,
        reason=(
            "a scenario run propagates one published insertion state; it "
            "reconstructs no object's history. 21 of the 49 launched objects "
            "were ever catalogued and 32 were never tracked at all, and the "
            "earliest element set for any of them postdates deployment by 1.92 "
            "to 4.99 days — an interval that contains every post-launch hour at "
            "Kp >= 5"
        ),
        permitted_alternative=(
            "'modelled scenario paths from the published insertion state' — or "
            "declare analysis_mode=tracked_object, name the objects "
            "individually, and cite the element sets that exist for each"
        ),
        aliases=(
            "trajectory was reconstructed",
            "reconstructed the trajectories",
            "reconstructed trajectories",
            "reconstructed trajectory",
        ),
    ),
    ProhibitedClaim(
        claim_pattern="SURYA predicts re-entry",
        forbidden_when=_no_established_predictive_skill,
        reason=(
            "the SURYA layer emits a flare score and is barred from density, "
            "drag and propagation by an import ban the test suite enforces, so "
            "it predicts no orbital quantity whatsoever; and the incident window "
            "that was scored holds 5 samples carrying 1 positive, which has no "
            "power to detect skill in either direction. Native inference having "
            "run is a fact about execution, not about prediction"
        ),
        permitted_alternative=(
            "'native SURYA inference was executed and its flare scores are "
            "carried as labelled context; SURYA drives no density, drag or "
            "orbit result in this study' — or state a probabilistic model, "
            "evaluate it out of sample on a window with power, and declare all "
            "three of has_probabilistic_model, uncertainty_basis and "
            "calibration_status accordingly"
        ),
        aliases=(
            "SURYA predicts reentry",
            "SURYA predicted re-entry",
            "SURYA forecasts re-entry",
            "SURYA predicts orbital decay",
            "SURYA predicts the decay",
        ),
    ),
    ProhibitedClaim(
        claim_pattern="operational Starlink lifetime prediction",
        forbidden_when=_no_established_predictive_skill,
        reason=(
            "operational use is a claim of skill on cases not yet seen. The drag "
            "parameter is fitted on the same decay dates it is then compared "
            "with, only 4 of the 49 objects had enough clean element history to "
            "be scored on a held-out interval at all, and the sensitivity range "
            "carries no coverage probability. One archived event, calibrated "
            "in-sample, licenses no operational claim"
        ),
        permitted_alternative=(
            "'a research engine for model comparison, counterfactual analysis, "
            "sensitivity analysis and inference-limit auditing on one archived "
            "event' — or hold data out, state a probabilistic model, and declare "
            "calibration_status=out_of_sample_validated with an uncertainty "
            "basis that carries coverage"
        ),
        aliases=(
            "operational lifetime prediction",
            "operational lifetime predictor",
            "operational re-entry prediction",
            "operational decay prediction",
            "operationally deployable",
            "ready for operational use",
        ),
    ),
)


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for index, character in enumerate(text):
        if character == "\n":
            starts.append(index + 1)
    return starts


def _line_of(starts: Sequence[int], position: int) -> int:
    return bisect_right(starts, position)


def check_text(
    text: str,
    *,
    context: ClaimContext,
    source: str = "",
    claims: Iterable[ProhibitedClaim] | None = None,
    window: int = 120,
) -> list[ClaimViolation]:
    """Return every prohibited claim ``text`` makes from ``context``.

    Reports all of them rather than the first, because a document is fixed once
    and a caller needs the whole list. Ordering is by position in the text.
    """
    catalogue = tuple(PROHIBITED_CLAIMS if claims is None else claims)
    starts = _line_starts(text)
    lines = text.split("\n")
    found: list[tuple[int, ClaimViolation]] = []

    for claim in catalogue:
        if not claim.forbidden_when(context):
            continue
        for phrase in claim.phrases():
            for match in _phrase_regex(phrase).finditer(text):
                line_number = _line_of(starts, match.start())
                occurrence = ClaimOccurrence(
                    line_number=line_number,
                    line_text=lines[line_number - 1],
                    matched_text=match.group(0),
                    column=match.start() - starts[line_number - 1],
                    preceding=text[max(0, match.start() - window) : match.start()],
                    following=text[match.end() : match.end() + window],
                )
                if claim.occurrence_exempt is not None and claim.occurrence_exempt(occurrence):
                    continue
                found.append(
                    (
                        match.start(),
                        ClaimViolation(
                            claim_pattern=claim.claim_pattern,
                            reason=claim.reason,
                            permitted_alternative=claim.permitted_alternative,
                            matched_text=occurrence.matched_text,
                            line_number=occurrence.line_number,
                            line_text=occurrence.line_text,
                            column=occurrence.column,
                            context_summary=context.summary(),
                            source=source,
                        ),
                    )
                )

    found.sort(key=lambda pair: pair[0])
    return [violation for _, violation in found]


def enforce(
    text: str,
    *,
    context: ClaimContext,
    source: str = "",
    claims: Iterable[ProhibitedClaim] | None = None,
) -> None:
    """Raise :class:`ClaimViolation` if ``text`` makes a prohibited claim.

    The raised exception describes the first violation in the text and names
    the rest, so a caller sees the whole problem without having to re-run.
    """
    violations = check_text(text, context=context, source=source, claims=claims)
    if not violations:
        return
    first = violations[0]
    raise ClaimViolation(
        claim_pattern=first.claim_pattern,
        reason=first.reason,
        permitted_alternative=first.permitted_alternative,
        matched_text=first.matched_text,
        line_number=first.line_number,
        line_text=first.line_text,
        column=first.column,
        context_summary=first.context_summary,
        source=source,
        also_found=tuple(v.claim_pattern for v in violations[1:]),
    )


_BALLISTIC_TOKENS = ("ballistic_coefficient", "ballistic coefficient", "ballisticcoefficient")


def guard_value(
    name: str,
    value: float,
    units: str,
    *,
    claimed_as: str = "",
) -> float:
    """Check a number against the name it is about to be emitted under.

    This is the gate for the error that was actually made: 0.085 m^2/kg is
    C_d*A/m, and publishing it as a "ballistic coefficient" inverts the
    physical meaning — a larger number becomes slower decay instead of faster.
    Both directions are checked, because mislabelling kg/m^2 as *inverse* is
    the same mistake with the sign of the mistake flipped.

    Returns ``value`` so the guard can wrap an expression at the emission
    point:

        report["b_inv"] = guard_value("inverse_ballistic_coefficient", b, "m^2/kg")

    Raises:
        ClaimViolation: if the name and the units disagree about which of the
            two reciprocal quantities is being reported.
    """
    label = f"{name} {claimed_as}".strip()
    lowered = label.lower()
    mentions_ballistic = any(token in lowered for token in _BALLISTIC_TOKENS)
    if not mentions_ballistic:
        return value

    says_inverse = "inverse" in lowered
    canonical = normalise_units(units)

    if canonical == INVERSE_DRAG_UNITS and not says_inverse:
        raise ClaimViolation(
            claim_pattern="ballistic coefficient",
            reason=(
                f"{name!r} carries {value!r} in {units!r}, which normalises to "
                f"{INVERSE_DRAG_UNITS}. That is C_d*A/m, the INVERSE ballistic "
                f"coefficient. The ballistic coefficient B = m/(C_d*A) is its "
                f"reciprocal in {CONVENTIONAL_DRAG_UNITS}; the two move in "
                f"opposite directions and are not interchangeable"
            ),
            permitted_alternative=(
                f"rename to 'inverse_{name}' (or a name containing 'inverse'), "
                f"or emit the reciprocal 1/{value!r} in {CONVENTIONAL_DRAG_UNITS}"
            ),
            matched_text=name,
            context_summary=f"value_units={units!r}, claimed_as={claimed_as!r}",
        )

    if canonical == CONVENTIONAL_DRAG_UNITS and says_inverse:
        raise ClaimViolation(
            claim_pattern="inverse ballistic coefficient",
            reason=(
                f"{name!r} carries {value!r} in {units!r}, which normalises to "
                f"{CONVENTIONAL_DRAG_UNITS}. That is the CONVENTIONAL ballistic "
                f"coefficient B = m/(C_d*A); the inverse form C_d*A/m is its "
                f"reciprocal in {INVERSE_DRAG_UNITS}"
            ),
            permitted_alternative=(
                "drop 'inverse' from the name, or emit the reciprocal in "
                f"{INVERSE_DRAG_UNITS}"
            ),
            matched_text=name,
            context_summary=f"value_units={units!r}, claimed_as={claimed_as!r}",
        )

    if canonical not in (INVERSE_DRAG_UNITS, CONVENTIONAL_DRAG_UNITS):
        raise ClaimViolation(
            claim_pattern="ballistic coefficient",
            reason=(
                f"{name!r} names a ballistic coefficient but carries units "
                f"{units!r}, which is neither {INVERSE_DRAG_UNITS} nor "
                f"{CONVENTIONAL_DRAG_UNITS}. A drag parameter whose units are "
                f"unstated or wrong cannot be read in either sense"
            ),
            permitted_alternative=(
                f"emit C_d*A/m in {INVERSE_DRAG_UNITS} under a name containing "
                f"'inverse', or m/(C_d*A) in {CONVENTIONAL_DRAG_UNITS}"
            ),
            matched_text=name,
            context_summary=f"value_units={units!r}, claimed_as={claimed_as!r}",
        )

    return value
