"""The licence record: a licence FACT and a policy CHOICE, held apart by force.

The single error this module exists to prevent is the conflation of two
questions that read alike and are not alike:

  1. What does the licence permit?            -> ``commercial_use``,
                                                 ``classification``
  2. What has this project decided to do?     -> ``policy_decision``

They diverge in both directions, and each direction has burned real projects:

  * ADBSat is GPL-3.0 and Nyx is AGPL-3.0. Both licences PERMIT commercial use;
    what they attach is a copyleft obligation. This project declines that
    obligation, so both are ``excluded_by_policy``. Recording them as
    "prohibited" would be a false statement about somebody else's licence.
  * DTM2020 is under a CNES academic-non-commercial licence and NRLMSIS 2.x
    under an equivalent US Naval Research Laboratory one (US Patent 10,641,925
    recorded separately, because a patent is not a licence status). Those
    licences genuinely forbid commercial use, so they are
    ``prohibited_by_licence``. Recording that as a mere policy preference would
    understate a hard legal bar, and a future reviewer could "relax the policy"
    and ship it. NRLMSIS 2.x is currently one step further out again, at
    ``licence_pending``, for the reason set out below.

The invariant is therefore enforced in code, not in review:

    policy_decision in {"prohibited_by_licence", "licence_pending"}
                                                <=>  commercial_use == "prohibited"
    classification  == "red"                    <=>  commercial_use == "prohibited"

so a record cannot claim a legal prohibition it has not recorded, nor record a
prohibition while claiming a lesser status.

``licence_pending`` sits on the prohibited side of that biconditional rather
than weakening it, and the reason is worth stating exactly. The NRLMSIS 2.0/2.1
terms as WRITTEN do forbid commercial use; nothing about that is in doubt. What
is outstanding is a different question — whether non-revenue, openly published
research is separately permitted — put to NRL in writing and not yet answered.
A record may therefore say "the answer to a further question has not arrived"
without downgrading what the terms already say. Silence is not permission, so a
``licence_pending`` record must carry ``fail_closed=True`` and a
``pending_request`` whose ``reply_received`` is false, and the gate refuses it
in EVERY mode. When the reply lands, the record may no longer be
``licence_pending``: it must be re-decided explicitly against the scope of what
was actually granted.

``permitted_research_only`` records a fourth situation, and it is neither of the
two above. NRLMSISE-00 is the case it exists for. The Fortran this project
actually compiles — ``NRLMSISE-00.FOR``, fetched from NRL at build time by
pymsis — carries NO licence text of any kind: no copyright notice, no
public-domain dedication, no terms, and equally no restriction. That is an
ABSENCE OF TERMS, and an absence of terms is not a grant. It is also not a
prohibition, so recording it as ``prohibited_by_licence`` would be a false
statement about somebody else's position in the opposite direction from the one
this module usually guards against. The honest answer is that ``commercial_use``
is genuinely ``unknown``, which is why the vocabulary carries that word: a
resource may be run in research, where nothing turns on the commercial
question, and refused in production, where everything does.

Because ``unknown`` is the one commercial position that no reviewed decision
should ever be allowed to hide, it is tied to the decision in both directions:

    policy_decision == "permitted_research_only"  <=>  commercial_use == "unknown"
    policy_decision == "permitted_research_only"   =>  classification  == "amber"
    policy_decision == "permitted_research_only"   =>  verification_gap is populated
    policy_decision == "permitted"                 =>  verification_gap is empty

The ``verification_gap`` requirement is the point of the decision rather than
paperwork attached to it. A record that says "research only" without saying what
evidence is missing and what would close it cannot be discharged by anyone, and
degrades within a release or two into a status nobody remembers the reason for.
The last rule is its mirror: a fully ``permitted`` record may not carry an open
gap, because that is exactly the state — production running on terms still
outstanding — that this decision exists to make unrepresentable.

A ``permitted_research_only`` record is NOT frozen. Freezing is for a resource
whose terms are known to be restrictive with a further question outstanding;
here nothing forbidding use has been located at all, and research mode is
precisely the mode entitled to proceed on that footing.

Vocabulary for ``classification`` (the FACT about the licence):

    green   commercial use permitted with no material obligation beyond
            attribution or a NOTICE file
    amber   commercial use permitted, but under a material condition: copyleft,
            a no-modification clause, per-user credentials, redistribution
            limits, or terms this project has not been able to verify
    red     the licence does not permit commercial use

Vocabulary for ``policy_decision`` (this project's CHOICE):

    permitted               may appear in the production path
    permitted_research_only usable in research; REFUSED in production, pending
                            primary evidence of what the terms actually are
    excluded_by_policy      the licence would allow it; this project declines
    prohibited_by_licence   the licence forbids commercial use
    licence_pending         the licence forbids commercial use AND a written
                            question about a further permission is outstanding;
                            frozen in every mode until the answer arrives
    research_noncommercial_open_publication_permitted
                            the licence forbids commercial use AND the rights
                            holder has confirmed in writing that the specific
                            non-commercial, openly published research use this
                            project described is permitted under the existing
                            licence terms. NOT a waiver, NOT a relicensing and
                            NOT a commercial grant: commercial_use stays
                            "prohibited" and the classification stays red. The
                            decision is admissible ONLY in a mode whose table
                            lists it by name — production never does, and the
                            loader's production invariant makes that
                            structural. The written reply is evidence, so the
                            record must carry the answered request and the
                            confirmation fields, or it fails to construct.

Field values arrive from YAML, so they are typed ``str`` and validated at
runtime rather than typed as ``Literal``: a ``Literal`` annotation on data read
from a file would have to be reached by a ``cast``, which asserts the very
property that has not yet been checked.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Final

from ..domain.temporal import NaiveDatetimeError, parse_aware_datetime

__all__ = [
    "CLASSIFICATIONS",
    "COMMERCIAL_PROHIBITION_DECISIONS",
    "COMMERCIAL_UNKNOWN_DECISIONS",
    "COMMERCIAL_USE_VALUES",
    "LICENCE_PENDING",
    "PERMISSION_VALUES",
    "PERMITTED_RESEARCH_ONLY",
    "POLICY_DECISIONS",
    "RESEARCH_NONCOMMERCIAL_OPEN_PUBLICATION_PERMITTED",
    "RECORD_FIELDS",
    "RECORD_OPTIONAL_FIELDS",
    "RECORD_REQUIRED_FIELDS",
    "RESOURCE_KINDS",
    "SUBSTITUTABLE_DECISIONS",
    "LicenceRecord",
    "LicenceRecordError",
    "PendingLicenceRequest",
]


class LicenceRecordError(ValueError):
    """Raised when a licence record is internally inconsistent or malformed."""


#: What kind of thing is being licensed. The distinction matters because the
#: terms differ by kind even inside one distribution: pymsis is MIT-licensed
#: SOFTWARE that can reach a MODEL whose COEFFICIENT_FILE (msis21.parm) carries
#: the NRL restriction. One record per licensed thing, never per package.
RESOURCE_KINDS: Final[frozenset[str]] = frozenset(
    {"software", "dataset", "model_weights", "coefficient_file"}
)

#: Vocabulary for ``commercial_use``. "conditional" means the licence permits
#: commercial use only under a condition this project must actively satisfy
#: (credentials, acknowledgement, no redistribution). "unknown" means no primary
#: evidence of the terms is held in either direction — neither a grant nor a
#: restriction has been located. Neither value ever means "probably fine".
COMMERCIAL_USE_VALUES: Final[frozenset[str]] = frozenset(
    {"permitted", "prohibited", "conditional", "unknown"}
)

#: Vocabulary for the per-activity permission fields. ``not_applicable`` is a
#: real answer (raw-data caching for a software library) and is distinct from
#: ``conditional``, which means "permitted if a condition holds", and from
#: ``unknown``, which means the terms governing that activity have not been
#: located at all.
PERMISSION_VALUES: Final[frozenset[str]] = frozenset(
    {"permitted", "prohibited", "conditional", "not_applicable", "unknown"}
)

CLASSIFICATIONS: Final[frozenset[str]] = frozenset({"green", "amber", "red"})

#: The decision taken while a written licence question is outstanding.
LICENCE_PENDING: Final = "licence_pending"

#: The decision taken when no primary evidence of the terms is held at all.
PERMITTED_RESEARCH_ONLY: Final = "permitted_research_only"

#: The decision taken when a commercially prohibited licence has been joined by
#: a WRITTEN confirmation that this project's described non-commercial, openly
#: published research use is permitted under the existing terms. The name is
#: deliberately narrow: it is not "commercial_use_permitted", not
#: "all_uses_permitted", and it never widens the licence — the confirmation
#: authorises the described research workflow and nothing else.
RESEARCH_NONCOMMERCIAL_OPEN_PUBLICATION_PERMITTED: Final = (
    "research_noncommercial_open_publication_permitted"
)

POLICY_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "permitted",
        PERMITTED_RESEARCH_ONLY,
        "excluded_by_policy",
        "prohibited_by_licence",
        LICENCE_PENDING,
        RESEARCH_NONCOMMERCIAL_OPEN_PUBLICATION_PERMITTED,
    }
)

#: The decisions that assert the licence itself does not permit commercial use.
#: Both sides of the biconditional are checked against this set, so adding a
#: decision here without also meaning ``commercial_use == "prohibited"`` is not
#: possible: the record would fail to construct.
#: RESEARCH_NONCOMMERCIAL_OPEN_PUBLICATION_PERMITTED belongs here because the
#: written confirmation proceeds UNDER the licence terms rather than replacing
#: them: the commercial prohibition is untouched, and a record that claimed
#: otherwise would be recording the confirmation as a waiver, which it is not.
COMMERCIAL_PROHIBITION_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "prohibited_by_licence",
        LICENCE_PENDING,
        RESEARCH_NONCOMMERCIAL_OPEN_PUBLICATION_PERMITTED,
    }
)

#: The decisions that assert nothing is known about commercial use. Checked in
#: both directions, exactly like the prohibition set above, so ``unknown`` can
#: never be carried by a decision production admits.
COMMERCIAL_UNKNOWN_DECISIONS: Final[frozenset[str]] = frozenset({PERMITTED_RESEARCH_ONLY})

#: The decisions a resource may hold and still be named as somebody else's
#: substitute. Deliberately narrow: a substitute is a replacement a caller can
#: actually reach in some declared mode, so a resource this project declines or
#: is legally barred from cannot be one. ``permitted_research_only`` qualifies
#: because research mode really does admit it — the gate then has to say in
#: which modes, which is why the check is not simply "any substitute will do".
SUBSTITUTABLE_DECISIONS: Final[frozenset[str]] = frozenset(
    {"permitted", PERMITTED_RESEARCH_ONLY}
)

# Fields whose value must be a permission word from PERMISSION_VALUES.
_PERMISSION_FIELDS: Final[tuple[str, ...]] = (
    "saas_use",
    "on_premise_redistribution",
    "raw_data_caching",
    "raw_data_redistribution",
    "derived_output_redistribution",
    "modification_permitted",
)

# Fields that must carry text. ``note`` is deliberately absent: its key is still
# required in the source data, but a record with nothing further to say may
# leave it empty rather than pad it with filler.
_NON_EMPTY_TEXT_FIELDS: Final[tuple[str, ...]] = (
    "resource_name",
    "spdx_or_licence_name",
    "licence_url",
    "copyright_holder",
    "source_url",
    "version",
    "attribution_text",
    "policy_reason",
    "evidence_url",
)

_BOOL_FIELDS: Final[tuple[str, ...]] = (
    "credential_required",
    "legal_review_required",
    "notice_required",
    "fail_closed",
)


@dataclass(frozen=True, slots=True)
class PendingLicenceRequest:
    """One written licence question put to a rights holder, and its answer or absence.

    The point of recording the question rather than a summary of it is that a
    reply answers a specific thing. A grant covering "non-commercial research"
    is not a grant covering redistribution, and a reader who only has "we asked
    NRL and they said yes" cannot tell the two apart. So the questions travel
    with the record, and ``reply_scope`` — required the moment a reply is
    recorded — has to say what the answer actually covered.

    ``reply_utc`` and ``reply_scope`` must be EMPTY while no reply has been
    received. An unanswered request that carries a scope is the exact shape of
    inferring permission from silence, and it fails to construct.
    """

    requested_of: str
    requested_utc: str
    questions_asked: tuple[str, ...]
    reply_received: bool
    reply_utc: str
    reply_scope: str

    def __post_init__(self) -> None:
        if not isinstance(self.requested_of, str) or not self.requested_of.strip():
            raise LicenceRecordError(
                f"pending licence request: 'requested_of' must name who was asked, got "
                f"{self.requested_of!r}"
            )
        _require_utc_text(self.requested_utc, "requested_utc")

        if not isinstance(self.questions_asked, tuple):
            raise LicenceRecordError(
                f"pending licence request to {self.requested_of!r}: 'questions_asked' must be "
                f"a tuple, got {type(self.questions_asked).__name__}"
            )
        if not self.questions_asked:
            raise LicenceRecordError(
                f"pending licence request to {self.requested_of!r}: 'questions_asked' must "
                f"record at least one question. A reply is only ever a reply to something."
            )
        for question in self.questions_asked:
            if not isinstance(question, str) or not question.strip():
                raise LicenceRecordError(
                    f"pending licence request to {self.requested_of!r}: every question must be "
                    f"a non-empty string, got {question!r}"
                )

        if not isinstance(self.reply_received, bool):
            raise LicenceRecordError(
                f"pending licence request to {self.requested_of!r}: 'reply_received' must be a "
                f"boolean, got {self.reply_received!r}"
            )
        for name in ("reply_utc", "reply_scope"):
            if not isinstance(getattr(self, name), str):
                raise LicenceRecordError(
                    f"pending licence request to {self.requested_of!r}: {name!r} must be a "
                    f"string, got {getattr(self, name)!r}"
                )

        if self.reply_received:
            _require_utc_text(self.reply_utc, "reply_utc")
            if not self.reply_scope.strip():
                raise LicenceRecordError(
                    f"pending licence request to {self.requested_of!r}: reply_received is true, "
                    f"so 'reply_scope' must state what the reply actually covered. A recorded "
                    f"reply with no scope reads as blanket permission and is not one."
                )
        else:
            for name in ("reply_utc", "reply_scope"):
                if getattr(self, name).strip():
                    raise LicenceRecordError(
                        f"pending licence request to {self.requested_of!r}: reply_received is "
                        f"false, so {name!r} must be empty, got {getattr(self, name)!r}. No "
                        f"reply has been received; nothing about one may be recorded."
                    )

    @property
    def summary_line(self) -> str:
        """One line naming who was asked, when, and whether they have answered."""
        state = f"reply received {self.reply_utc}" if self.reply_received else "NO REPLY RECEIVED"
        return f"{self.requested_of}, sent {self.requested_utc}: {state}"


@dataclass(frozen=True, slots=True)
class LicenceRecord:
    """One licensed resource, as reviewed, with the policy decision taken on it.

    Frozen because a licence determination is evidence: code that could mutate
    a record in flight could also make the production gate agree to something
    the audit never reviewed.
    """

    resource_name: str
    resource_kind: str
    spdx_or_licence_name: str
    licence_url: str
    copyright_holder: str
    source_url: str
    version: str
    commercial_use: str
    saas_use: str
    on_premise_redistribution: str
    raw_data_caching: str
    raw_data_redistribution: str
    derived_output_redistribution: str
    modification_permitted: str
    credential_required: bool
    legal_review_required: bool
    notice_required: bool
    attribution_text: str
    classification: str
    policy_decision: str
    policy_reason: str
    evidence_url: str
    note: str
    # The three fields below carry defaults because their absence has a single
    # unambiguous meaning — no freeze, no outstanding question, no open gap in
    # the licence evidence — and because defaulting them cannot open a hole: the
    # decisions that depend on them, ``licence_pending`` and
    # ``permitted_research_only``, refuse to construct unless they are set.
    fail_closed: bool = False
    pending_request: PendingLicenceRequest | None = None
    # -- written-confirmation fields -----------------------------------------
    # Populated only by records whose policy_decision is
    # RESEARCH_NONCOMMERCIAL_OPEN_PUBLICATION_PERMITTED, and REQUIRED for
    # those: _validate_written_confirmation refuses the decision without them.
    # Defaults are empty so every pre-existing record parses unchanged.
    #
    # ``model_version`` lives in the existing ``version`` field,
    # ``authorised_source`` in ``source_url``, ``derived_output_status`` in
    # ``derived_output_redistribution``, ``modification_status`` in
    # ``modification_permitted`` — the fields below are the ones with no
    # existing home.
    written_confirmation_status: str = ""
    permitted_project_mode: str = ""
    hosted_service_status: str = ""
    customer_distribution_status: str = ""
    source_redistribution_status: str = ""
    coefficient_redistribution_status: str = ""
    translation_status: str = ""
    wrapper_status: str = ""
    citation_requirements: str = ""
    acknowledgement_requirements: str = ""
    permission_basis: str = ""
    #: What evidence is missing, and what would close it. Required by
    #: ``permitted_research_only`` and forbidden to ``permitted``.
    verification_gap: str = ""

    def __post_init__(self) -> None:
        self._validate_vocabulary()
        self._validate_fact_and_policy_agree()
        self._validate_verification_gap()
        self._validate_freeze()
        self._validate_written_confirmation()

    # -- written-confirmation validation --------------------------------------

    def _validate_written_confirmation(self) -> None:
        """Tie the confirmation decision to the evidence that justifies it.

        The decision asserts a written reply exists. A record carrying the
        decision without the answered request, or without stating what each
        prohibited use-mode's status is, would let the strongest claim in the
        vocabulary rest on the least evidence, so every field is required and
        the request must record ``reply_received: true`` — the reply IS the
        evidence, and an unanswered request cannot support this decision.

        The converse direction is enforced too: confirmation fields on a
        record with any other decision would let a later edit flip the
        decision without re-establishing the evidence, so they must be empty
        everywhere else.
        """
        name = self.resource_name
        confirmation_fields = (
            "written_confirmation_status",
            "permitted_project_mode",
            "hosted_service_status",
            "customer_distribution_status",
            "source_redistribution_status",
            "coefficient_redistribution_status",
            "translation_status",
            "wrapper_status",
            "citation_requirements",
            "acknowledgement_requirements",
            "permission_basis",
        )
        if self.policy_decision == RESEARCH_NONCOMMERCIAL_OPEN_PUBLICATION_PERMITTED:
            for field_name in confirmation_fields:
                value = getattr(self, field_name)
                if not isinstance(value, str) or not value.strip():
                    raise LicenceRecordError(
                        f"licence record {name!r}: policy_decision "
                        f"{self.policy_decision!r} requires a non-empty "
                        f"{field_name!r}. The confirmation authorises a described "
                        f"workflow, and a record that does not describe it grants "
                        f"by omission."
                    )
            if self.pending_request is None or not self.pending_request.reply_received:
                raise LicenceRecordError(
                    f"licence record {name!r}: policy_decision {self.policy_decision!r} "
                    f"asserts a written reply exists, so the record must carry the "
                    f"answered request (pending_request with reply_received: true, the "
                    f"reply date and its scope). An unanswered or absent request cannot "
                    f"support this decision — that would be inferring permission from "
                    f"silence with extra steps."
                )
            if self.fail_closed:
                raise LicenceRecordError(
                    f"licence record {name!r}: fail_closed freezes a record in every "
                    f"mode, which contradicts a decision whose whole content is that one "
                    f"named research mode admits it. The prohibited modes are refused "
                    f"structurally — production admits only 'permitted' and no other "
                    f"mode lists this decision unless it says so by name."
                )
        else:
            for field_name in confirmation_fields:
                if getattr(self, field_name).strip():
                    raise LicenceRecordError(
                        f"licence record {name!r}: field {field_name!r} is populated but "
                        f"policy_decision is {self.policy_decision!r}. Confirmation "
                        f"fields may only accompany "
                        f"'research_noncommercial_open_publication_permitted', so that "
                        f"the decision can never be flipped without re-establishing the "
                        f"evidence beside it."
                    )

    # -- validation ----------------------------------------------------------

    def _validate_vocabulary(self) -> None:
        name = self.resource_name if isinstance(self.resource_name, str) else "<unnamed>"

        for field_name in _NON_EMPTY_TEXT_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise LicenceRecordError(
                    f"licence record {name!r}: field {field_name!r} must be a non-empty "
                    f"string, got {value!r}"
                )

        for field_name in ("note", "verification_gap"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise LicenceRecordError(
                    f"licence record {name!r}: field {field_name!r} must be a string, "
                    f"got {value!r}"
                )

        for field_name in _BOOL_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise LicenceRecordError(
                    f"licence record {name!r}: field {field_name!r} must be a boolean, "
                    f"got {value!r}"
                )

        if self.pending_request is not None and not isinstance(
            self.pending_request, PendingLicenceRequest
        ):
            raise LicenceRecordError(
                f"licence record {name!r}: field 'pending_request' must be a "
                f"PendingLicenceRequest or None, got {self.pending_request!r}"
            )

        _require_in(name, "resource_kind", self.resource_kind, RESOURCE_KINDS)
        _require_in(name, "commercial_use", self.commercial_use, COMMERCIAL_USE_VALUES)
        _require_in(name, "classification", self.classification, CLASSIFICATIONS)
        _require_in(name, "policy_decision", self.policy_decision, POLICY_DECISIONS)
        for field_name in _PERMISSION_FIELDS:
            _require_in(name, field_name, getattr(self, field_name), PERMISSION_VALUES)

    def _validate_fact_and_policy_agree(self) -> None:
        """Enforce the fact/policy invariants that give the vocabulary meaning.

        Without these checks the ``policy_decision`` values collapse into
        interchangeable labels, and the distinction the whole module exists to
        preserve is a naming convention rather than a property.

        Order is load-bearing in one place. The ``unknown`` rules run BEFORE the
        red and green classification rules, so a record whose terms are
        unlocated is told the specific thing that is wrong with it — "amber, or
        you are claiming something nobody has established" — rather than the
        generic red/green complaint that would also be true of it. The
        prohibition rules stay first of all: a record claiming a bar it has not
        recorded is answered on that, not on its colour.
        """
        name = self.resource_name
        prohibited = self.commercial_use == "prohibited"

        if self.policy_decision in COMMERCIAL_PROHIBITION_DECISIONS and not prohibited:
            raise LicenceRecordError(
                f"licence record {name!r}: policy_decision {self.policy_decision!r} claims the "
                f"LICENCE forbids commercial use, but commercial_use is "
                f"{self.commercial_use!r}. A component this project merely declines to use is "
                f"'excluded_by_policy'; 'prohibited_by_licence' is reserved for "
                f"commercial_use == 'prohibited', and 'licence_pending' does not soften that — "
                f"it records a FURTHER question outstanding on top of it."
            )

        if prohibited and self.policy_decision not in COMMERCIAL_PROHIBITION_DECISIONS:
            raise LicenceRecordError(
                f"licence record {name!r}: commercial_use is 'prohibited', so policy_decision "
                f"must be 'prohibited_by_licence' or 'licence_pending', not "
                f"{self.policy_decision!r}. Recording a legal bar as a policy preference "
                f"invites a later reviewer to relax it."
            )

        self._validate_unknown_and_policy_agree()

        if (self.classification == "red") != prohibited:
            raise LicenceRecordError(
                f"licence record {name!r}: classification {self.classification!r} disagrees with "
                f"commercial_use {self.commercial_use!r}. 'red' means and only means that the "
                f"licence does not permit commercial use."
            )

        if self.classification == "green" and self.commercial_use != "permitted":
            raise LicenceRecordError(
                f"licence record {name!r}: classification 'green' requires commercial_use "
                f"'permitted', got {self.commercial_use!r}. A resource whose commercial use is "
                f"conditional is 'amber'."
            )

    def _validate_unknown_and_policy_agree(self) -> None:
        """Tie ``commercial_use == "unknown"`` to the one decision that may carry it.

        The failure guarded here is quieter than a false prohibition and worse:
        a record whose terms nobody has located, sitting on a decision the
        production gate admits. Both directions are checked, so ``unknown``
        cannot be attached to ``permitted``, and ``permitted_research_only``
        cannot be used to soft-pedal terms that are actually known.
        """
        name = self.resource_name
        research_only = self.policy_decision in COMMERCIAL_UNKNOWN_DECISIONS
        unknown = self.commercial_use == "unknown"

        if research_only and not unknown:
            raise LicenceRecordError(
                f"licence record {name!r}: policy_decision {self.policy_decision!r} records that "
                f"the terms have NOT been established, so commercial_use must be 'unknown', got "
                f"{self.commercial_use!r}. A resource whose terms ARE known belongs to a decision "
                f"that states them: 'permitted', 'excluded_by_policy' or 'prohibited_by_licence'."
            )

        if unknown and not research_only:
            raise LicenceRecordError(
                f"licence record {name!r}: commercial_use is 'unknown', so policy_decision must "
                f"be {PERMITTED_RESEARCH_ONLY!r}, not {self.policy_decision!r}. An unknown "
                f"commercial position may not be carried by a decision the production gate "
                f"admits: nothing has been located that permits it, and nothing that forbids it."
            )

        if research_only and self.classification != "amber":
            raise LicenceRecordError(
                f"licence record {name!r}: policy_decision {self.policy_decision!r} requires "
                f"classification 'amber' — the arm of that word covering terms this project has "
                f"not been able to verify — got {self.classification!r}. 'green' would claim a "
                f"verified permission and 'red' a prohibition; neither has been established."
            )

    def _validate_verification_gap(self) -> None:
        """A research-only record must say what is missing; a permitted one must not.

        Both halves matter. Without the first, ``permitted_research_only``
        becomes a status with no discharge condition that outlives everyone who
        remembers why it was set. Without the second, a record could be
        ``permitted`` — admitted straight into production — while still
        carrying an open gap in the evidence behind it.
        """
        name = self.resource_name
        gap = self.verification_gap.strip()

        if self.policy_decision == PERMITTED_RESEARCH_ONLY and not gap:
            raise LicenceRecordError(
                f"licence record {name!r}: policy_decision {PERMITTED_RESEARCH_ONLY!r} requires a "
                f"populated 'verification_gap' naming what evidence is missing and what would "
                f"close it. A record cannot claim research-only status without saying why it is "
                f"not fully permitted, or nobody can ever discharge it."
            )

        if self.policy_decision == "permitted" and gap:
            raise LicenceRecordError(
                f"licence record {name!r}: policy_decision 'permitted' carries a "
                f"verification_gap {self.verification_gap!r}. A resource with an open gap in its "
                f"licence evidence is {PERMITTED_RESEARCH_ONLY!r}, not 'permitted': production "
                f"must not run on terms that are still outstanding."
            )

    def _validate_freeze(self) -> None:
        """A pending licence question must be frozen, evidenced, and unanswered.

        Three separate ways of turning "we have not been told yes" into "we may
        proceed" are refused here, because each has a plausible-looking shape:
        a pending record that is not fail-closed, a pending record with no
        request behind it, and a pending record left in place after the reply
        arrived.
        """
        name = self.resource_name
        pending = self.pending_request

        if self.policy_decision == LICENCE_PENDING:
            if not self.fail_closed:
                raise LicenceRecordError(
                    f"licence record {name!r}: policy_decision 'licence_pending' requires "
                    f"fail_closed=True. The whole content of a pending decision is that the "
                    f"resource is refused in EVERY mode until the answer arrives; a pending "
                    f"record that did not fail closed would infer permission from silence."
                )
            if pending is None:
                raise LicenceRecordError(
                    f"licence record {name!r}: policy_decision 'licence_pending' requires a "
                    f"pending_request naming who was asked, when, and what was asked. Without "
                    f"one the record claims an outstanding question it cannot produce."
                )
            if pending.reply_received:
                raise LicenceRecordError(
                    f"licence record {name!r}: pending_request records a reply received "
                    f"{pending.reply_utc}, so policy_decision may no longer be "
                    f"'licence_pending'. A reply forces an explicit re-decision — 'permitted', "
                    f"'excluded_by_policy' or 'prohibited_by_licence' — taken against the "
                    f"recorded reply_scope {pending.reply_scope!r} rather than left to lapse."
                )
        elif self.fail_closed and pending is None:
            raise LicenceRecordError(
                f"licence record {name!r}: fail_closed=True with policy_decision "
                f"{self.policy_decision!r} and no pending_request. A resource frozen in every "
                f"mode must say what it is waiting for, or nobody can ever unfreeze it."
            )

    # -- derived views -------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        """True when the gate must refuse this record in every mode.

        Distinct from ``not is_permitted_in_production``: an
        ``excluded_by_policy`` record is refused in production and admitted in
        research, which is what research mode is for. A frozen record is
        admitted nowhere.
        """
        return self.fail_closed or self.policy_decision == LICENCE_PENDING

    @property
    def is_permitted_in_production(self) -> bool:
        """True only for records this project has cleared for the product."""
        return self.policy_decision == "permitted"

    @property
    def is_research_only(self) -> bool:
        """True for a record usable in research and refused in production.

        Distinct from ``not is_permitted_in_production``, which is also true of
        every excluded, prohibited and frozen record. This one names the single
        reason production refuses it: the terms have not been established.
        """
        return self.policy_decision == PERMITTED_RESEARCH_ONLY

    @property
    def is_available_as_substitute(self) -> bool:
        """True when this record could stand in for a refused one in SOME mode.

        Not the same as "is permitted": a research-only record is a genuine
        substitute for a caller working in research, and pretending otherwise
        would leave the gate with nothing to point at. It IS narrower than "any
        substitute will do" — a declined, prohibited or frozen record can never
        qualify, so a restricted resource can never be pointed at another
        restricted one. Whether a given MODE admits it is a separate question,
        answered by the gate at the point of refusal.
        """
        return not self.is_frozen and self.policy_decision in SUBSTITUTABLE_DECISIONS

    def summary_line(self) -> str:
        """One-line human summary, used in gate errors and generated notices."""
        line = (
            f"{self.resource_name} ({self.resource_kind}) — {self.spdx_or_licence_name}; "
            f"commercial use {self.commercial_use}; classification {self.classification}; "
            f"policy {self.policy_decision}"
        )
        if self.verification_gap.strip():
            line = f"{line}; verification gap: {self.verification_gap}"
        if self.pending_request is None:
            return line
        return f"{line}; pending request: {self.pending_request.summary_line}"


def _require_in(record_name: str, field_name: str, value: object, allowed: frozenset[str]) -> None:
    if value not in allowed:
        raise LicenceRecordError(
            f"licence record {record_name!r}: field {field_name!r} has unknown value {value!r}; "
            f"allowed values are {sorted(allowed)}"
        )


def _require_utc_text(value: object, field_name: str) -> None:
    """Require an ISO-8601 instant carrying an explicit offset.

    A licence question dated ``2026-07-24`` with no offset is ambiguous by a
    day at the boundaries, and the age of an unanswered request is exactly what
    a reviewer will be reading it for.
    """
    if not isinstance(value, str) or not value.strip():
        raise LicenceRecordError(
            f"pending licence request: {field_name!r} must be a non-empty ISO-8601 instant, "
            f"got {value!r}"
        )
    try:
        parse_aware_datetime(value, field_name)
    except (NaiveDatetimeError, ValueError) as exc:
        raise LicenceRecordError(
            f"pending licence request: {field_name!r} must be an ISO-8601 instant with an "
            f"explicit UTC offset, got {value!r} ({exc})"
        ) from exc


#: Every field of a record, in declaration order.
RECORD_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in fields(LicenceRecord))

#: Every field a source record MUST supply. The loader derives its
#: missing-key check from here, so adding a field without a default to the
#: dataclass makes existing policy files fail loudly instead of defaulting.
RECORD_REQUIRED_FIELDS: Final[tuple[str, ...]] = tuple(
    f.name for f in fields(LicenceRecord) if f.default is MISSING
)

#: Fields a source record MAY supply. Only the freeze fields and
#: ``verification_gap`` live here, and only because their absence has one
#: meaning — not frozen, nothing outstanding, no open gap — while the decisions
#: that depend on them refuse to construct unless they are written down.
RECORD_OPTIONAL_FIELDS: Final[tuple[str, ...]] = tuple(
    f.name for f in fields(LicenceRecord) if f.default is not MISSING
)
