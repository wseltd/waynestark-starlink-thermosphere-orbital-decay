"""The production gate: what may enter a commercial deliverable, decided by code.

Licence discipline that lives in review comments survives exactly as long as
the reviewers do. This module is the machine-checked form: a call that names a
resource either returns its reviewed record or raises, and no third outcome
exists.

Two properties are deliberate and are tested rather than asserted:

**Fail-closed on the unknown.** A resource that is not in the policy is
REJECTED, in every mode. The tempting alternative — allow what has not been
reviewed, block only what has — inverts the burden of proof: a component enters
the product by nobody noticing it. An undeclared resource is not an unusual
input to this gate, it is the input the gate exists for.

**Keyed on the model, not the package.** ``pymsis`` is MIT-licensed and
permitted. The same MIT package can reach NRLMSIS 2.x, which is under a US
Naval Research Laboratory academic-non-commercial licence. A gate keyed on
distribution names would wave that through, because the distribution is
genuinely permissive. So models, datasets and coefficient files are separate
resources with their own records: ``check_resource("pymsis", ...)`` passes in
production while ``check_resource("nrlmsis_2_1", ...)`` is refused there — and
in research — because the written NRL permission it carries is admitted by the
``open_noncommercial_research`` mode alone.

Mode semantics come from the policy file, with one fixed floor enforced by the
loader: production admits ``policy_decision == "permitted"`` and nothing else.
Research mode is internal, non-commercial and non-shipping; it may admit
academically licensed and copyleft components, which is precisely why a
research result must never be published from a production build.

**Frozen beats mode.** One class of record is refused before the mode is
consulted at all: a record that is ``fail_closed``, which every
``licence_pending`` record must be. Research mode exists to admit things whose
terms are known and non-commercial; it cannot admit something whose terms carry
an unanswered written question, because the answer might be no. That makes the
freeze STRICTER than ``excluded_by_policy``, and it is checked here rather than
in the mode table so that widening a mode in the policy file cannot thaw it.

**A substitute is only a substitute in a mode that admits it.** The recorded
replacement for a refused resource is checked against the SAME mode the caller
asked about, by the same admission rule, rather than being named unconditionally
because the policy file lists it. NRLMSISE-00 is why: it is the recorded
substitute for NRLMSIS 2.x, msis21.parm and DTM2020, and it is itself
``permitted_research_only`` — a real substitute in research and no substitute at
all in production. Naming it in a production refusal without saying so would
send a caller to a second refusal, and the honest message says instead that the
recorded substitute is ALSO unavailable here, and why.
"""

from __future__ import annotations

import pathlib
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .policy_loader import (
    PRODUCTION_MODE,
    GateMode,
    LicencePolicy,
    SoftwareManifest,
    load_software_manifest,
    normalise_distribution_name,
)
from .record import LicenceRecord

__all__ = [
    "DependencyAudit",
    "ProductionGateError",
    "check_all_declared_dependencies",
    "check_resource",
    "parse_runtime_dependency_names",
]


class ProductionGateError(RuntimeError):
    """Raised when a resource may not be used in the requested mode."""


# A PEP 508 requirement begins with the distribution name; everything from the
# first extras bracket, version specifier, marker or whitespace onward is not
# part of the name.
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def check_resource(name: str, policy: LicencePolicy, *, mode: str) -> LicenceRecord:
    """Return the reviewed record for ``name``, or raise if ``mode`` forbids it.

    Args:
        name: the resource name as declared in the licence policy — the model,
            dataset or coefficient file, not necessarily the package that
            carries it.
        policy: a policy loaded by
            :func:`~thermosphere_consequence.licence.policy_loader.load_licence_policy`.
        mode: ``"production"``, ``"research"`` or any other mode the policy
            declares. An unrecognised mode is rejected rather than treated as
            permissive.

    Raises:
        ProductionGateError: if the mode is unknown, if the resource is
            undeclared, if the record is frozen pending a written licence
            reply, or if the mode does not admit the resource's
            ``policy_decision``. The message names the resource, its licence,
            the recorded reason and the recorded substitute where one exists —
            saying so explicitly when that substitute is unavailable here too.
    """
    gate_mode = policy.mode(mode)
    if gate_mode is None:
        raise ProductionGateError(
            f"unknown licence gate mode {mode!r}; policy {policy.policy_version} declares "
            f"{sorted(policy.modes)}. An unrecognised mode is refused rather than assumed "
            f"permissive."
        )

    record = policy.record_for(name)
    if record is None:
        raise ProductionGateError(
            f"resource {name!r} is REJECTED in {mode} mode: it is not declared in licence policy "
            f"{policy.policy_version} ({policy.source_path}). This gate is fail-closed — an "
            f"undeclared resource has no reviewed licence, so it cannot be cleared. Add a record "
            f"for {name!r} to the policy, with evidence, before using it."
        )

    # Checked before the mode, and deliberately: a frozen record is refused in
    # research too, so consulting the mode first would make the answer look
    # configurable when it is not.
    if record.is_frozen:
        raise ProductionGateError(_frozen_refusal_message(record, policy, gate_mode))

    if _mode_admits(record, gate_mode):
        return record

    raise ProductionGateError(_refusal_message(record, policy, gate_mode))


def _mode_admits(record: LicenceRecord, gate_mode: GateMode) -> bool:
    """The single admission rule: frozen beats mode, otherwise the mode decides.

    Written once and used both to clear a resource and to decide whether its
    recorded substitute is reachable, so the two can never come to disagree
    about the same record in the same mode.
    """
    return not record.is_frozen and gate_mode.allows(record.policy_decision)


def _substitute_clause(record: LicenceRecord, policy: LicencePolicy, gate_mode: GateMode) -> str:
    """Name the recorded replacement, or say why it is no help in THIS mode."""
    substitute = policy.substitute_for(record.resource_name)
    if substitute is None:
        return (
            "No permitted substitute is recorded for this resource; see the recorded reason above."
        )
    if _mode_admits(substitute, gate_mode):
        return (
            f"Permitted substitute: {substitute.resource_name!r} "
            f"({substitute.spdx_or_licence_name}, {substitute.source_url})."
        )

    admitting = sorted(
        name for name, candidate in policy.modes.items() if _mode_admits(substitute, candidate)
    )
    where = (
        f"It is admitted in {admitting} mode only."
        if admitting
        else "It is admitted in no declared mode."
    )
    gap = substitute.verification_gap.strip() or substitute.policy_reason.strip()
    return (
        f"The recorded substitute {substitute.resource_name!r} "
        f"({substitute.spdx_or_licence_name}, {substitute.source_url}) is ALSO UNAVAILABLE in "
        f"{gate_mode.name} mode: its policy decision is {substitute.policy_decision}. "
        f"Verification gap: {gap} {where} There is therefore no cleared replacement for "
        f"{record.resource_name!r} in {gate_mode.name} mode."
    )


def _refusal_message(record: LicenceRecord, policy: LicencePolicy, gate_mode: GateMode) -> str:
    return (
        f"resource {record.resource_name!r} is REJECTED in {gate_mode.name} mode. "
        f"Licence: {record.spdx_or_licence_name} ({record.licence_url}). "
        f"Commercial use: {record.commercial_use}; classification: {record.classification}; "
        f"policy decision: {record.policy_decision}. "
        f"Reason: {record.policy_reason} "
        f"{_verification_gap_clause(record)}"
        f"{_substitute_clause(record, policy, gate_mode)} "
        f"Evidence: {record.evidence_url}."
    )


def _verification_gap_clause(record: LicenceRecord) -> str:
    """The gap, stated in the refusal, because it is the thing that would lift it."""
    gap = record.verification_gap.strip()
    if not gap:
        return ""
    return f"Verification gap: {gap} "


def _frozen_refusal_message(
    record: LicenceRecord, policy: LicencePolicy, gate_mode: GateMode
) -> str:
    """The refusal for a record frozen pending a written reply.

    It names who was asked, on what date, what was asked and that nothing has
    come back, because a reader who is being told "no" needs to know exactly
    which fact would change the answer — and needs to be unable to mistake the
    absence of an answer for one.
    """
    pending = record.pending_request
    if pending is None:
        outstanding = (
            "The record is marked fail_closed, so it is refused in every mode until the "
            "record itself is changed."
        )
    elif pending.reply_received:
        outstanding = (
            f"A written request was sent to {pending.requested_of} on {pending.requested_utc} "
            f"and answered on {pending.reply_utc}, covering: {pending.reply_scope}"
        )
    else:
        questions = " ".join(
            f"({number}) {question}"
            for number, question in enumerate(pending.questions_asked, start=1)
        )
        outstanding = (
            f"A written request was sent to {pending.requested_of} on {pending.requested_utc} "
            f"asking: {questions} NO REPLY HAS BEEN RECEIVED. Permission is not inferred from "
            f"silence."
        )
    return (
        f"resource {record.resource_name!r} is REJECTED in {gate_mode.name} mode: its licence "
        f"position is UNRESOLVED and the record is FROZEN. "
        f"{outstanding} "
        f"While frozen the resource is not executed, not redistributed, its source and "
        f"coefficient files are not bundled, and no new result generated from it is published — "
        f"in EVERY mode. This is STRICTER than 'excluded_by_policy', which research mode admits. "
        f"Licence: {record.spdx_or_licence_name} ({record.licence_url}). "
        f"Commercial use: {record.commercial_use}; classification: {record.classification}; "
        f"policy decision: {record.policy_decision}. "
        f"Reason: {record.policy_reason} "
        f"{_substitute_clause(record, policy, gate_mode)} "
        f"Evidence: {record.evidence_url}."
    )


@dataclass(frozen=True, slots=True)
class DependencyAudit:
    """Result of cross-checking ``pyproject.toml`` against the software manifest."""

    policy_version: str
    pyproject_path: pathlib.Path
    manifest_path: pathlib.Path
    declared: tuple[str, ...]
    recorded: tuple[str, ...]
    unrecorded: tuple[str, ...]
    not_permitted: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        return not self.unrecorded and not self.not_permitted

    def findings(self) -> tuple[str, ...]:
        """Human-readable findings, empty when the audit is clean."""
        lines: list[str] = []
        for name in self.unrecorded:
            lines.append(
                f"runtime dependency {name!r} is declared in {self.pyproject_path.name} but has "
                f"no record in {self.manifest_path.name}"
            )
        for name, decision in self.not_permitted:
            lines.append(
                f"runtime dependency {name!r} is recorded with policy_decision {decision!r}, "
                f"which is not permitted in production"
            )
        return tuple(lines)

    def raise_for_findings(self) -> None:
        """Raise :class:`ProductionGateError` if the audit found anything."""
        findings = self.findings()
        if findings:
            raise ProductionGateError(
                "declared runtime dependencies failed the licence audit: " + "; ".join(findings)
            )


def parse_runtime_dependency_names(pyproject_path: str | pathlib.Path) -> tuple[str, ...]:
    """Normalised distribution names from ``[project].dependencies``.

    Only the runtime set: optional groups are development and deployment
    choices that are audited in the manifest but are not part of the installed
    library, and folding them in here would hide a real runtime addition among
    them.
    """
    path = pathlib.Path(pyproject_path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project")
    if not isinstance(project, Mapping):
        raise ProductionGateError(f"{path}: no [project] table to audit")
    declared = project.get("dependencies", [])
    if not isinstance(declared, Sequence) or isinstance(declared, str):
        raise ProductionGateError(f"{path}: [project].dependencies must be a list")

    names: list[str] = []
    for requirement in declared:
        if not isinstance(requirement, str):
            raise ProductionGateError(f"{path}: dependency entry {requirement!r} is not a string")
        match = _REQUIREMENT_NAME.match(requirement)
        if match is None:
            raise ProductionGateError(
                f"{path}: cannot read a distribution name from {requirement!r}"
            )
        names.append(normalise_distribution_name(match.group(1)))
    return tuple(sorted(names))


def check_all_declared_dependencies(
    policy: LicencePolicy,
    *,
    pyproject_path: str | pathlib.Path | None = None,
    manifest_path: str | pathlib.Path | None = None,
    manifest: SoftwareManifest | None = None,
) -> DependencyAudit:
    """Report every runtime dependency that the software manifest does not record.

    The failure this catches is the ordinary one: a dependency is added to
    ``pyproject.toml`` to make something work, and the licence review that
    should accompany it never happens. Anything declared but unrecorded is
    reported, as is anything recorded with a policy decision that production
    does not admit.

    Paths default to the repository that contains the loaded policy file, so the
    common call is ``check_all_declared_dependencies(policy)``; both may be
    overridden, and a pre-loaded manifest may be passed to avoid re-reading it.
    """
    root = _repository_root(policy)
    resolved_pyproject = pathlib.Path(pyproject_path) if pyproject_path else root / "pyproject.toml"
    resolved_manifest = (
        pathlib.Path(manifest_path)
        if manifest_path
        else root / "third_party" / "software_manifest.yaml"
    )
    software_manifest = manifest or load_software_manifest(resolved_manifest, policy)

    declared = parse_runtime_dependency_names(resolved_pyproject)
    by_distribution = software_manifest.by_distribution()

    recorded: list[str] = []
    unrecorded: list[str] = []
    not_permitted: list[tuple[str, str]] = []
    for name in declared:
        component = by_distribution.get(name)
        if component is None:
            unrecorded.append(name)
            continue
        recorded.append(name)
        # Ask the gate rather than re-read the record: the audit and the gate
        # then cannot drift into disagreeing about the same resource.
        try:
            check_resource(component.resource, policy, mode=PRODUCTION_MODE)
        except ProductionGateError:
            not_permitted.append((name, policy.records[component.resource].policy_decision))

    return DependencyAudit(
        policy_version=policy.policy_version,
        pyproject_path=resolved_pyproject,
        manifest_path=resolved_manifest,
        declared=declared,
        recorded=tuple(recorded),
        unrecorded=tuple(unrecorded),
        not_permitted=tuple(not_permitted),
    )


def _repository_root(policy: LicencePolicy) -> pathlib.Path:
    """The repository that owns the loaded policy file.

    Anchored on the policy's own path rather than on this module's location, so
    it is right for a source checkout, an editable install and a test fixture
    alike, and wrong loudly rather than silently when it is wrong.
    """
    source = policy.source_path.resolve()
    for candidate in source.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ProductionGateError(
        f"cannot locate a repository root above {source}: no pyproject.toml found. Pass "
        f"pyproject_path and manifest_path explicitly."
    )
