"""Loading the licence policy: completely, strictly, or not at all.

A licence gate is only as trustworthy as the file it reads. Three loading
behaviours that are ordinary elsewhere would each be a silent hole here, so
none of them exist in this module:

  * **No defaulted fields.** A record missing ``commercial_use`` does not become
    "unknown" or "permitted"; the load fails. A default would let a resource
    enter production carrying an answer nobody wrote down.
  * **No ignored keys.** An unrecognised key in a record is an error, not
    decoration. ``comercial_use`` misspelled would otherwise leave the real
    field missing and the typo invisible.
  * **No partial policy.** Every validation runs at load time, including the
    variable-level table, so a caller cannot hold a ``LicencePolicy`` that has
    not been checked end to end.

One invariant is not configurable, by design: the ``production`` mode must
allow exactly ``policy_decision == "permitted"``. The mode roster and the
records are configuration; the production floor is not, because a gate that can
be opened by editing the file it validates is not a gate.

PyYAML ships no ``py.typed`` marker and this repository does not vendor the
stub package, so the import carries a narrow ignore rather than a project-wide
override.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import yaml  # type: ignore[import-untyped]

from .record import (
    POLICY_DECISIONS,
    RECORD_OPTIONAL_FIELDS,
    RECORD_REQUIRED_FIELDS,
    SUBSTITUTABLE_DECISIONS,
    LicenceRecord,
    LicenceRecordError,
    PendingLicenceRequest,
)

__all__ = [
    "DATA_MANIFEST_FIELDS",
    "PENDING_REQUEST_FIELDS",
    "PRODUCTION_MODE",
    "RESEARCH_MODE",
    "SOFTWARE_MANIFEST_FIELDS",
    "DataManifest",
    "DataResource",
    "GateMode",
    "LicencePolicy",
    "LicencePolicyError",
    "SoftwareComponent",
    "SoftwareManifest",
    "load_data_manifest",
    "load_licence_policy",
    "load_software_manifest",
    "normalise_distribution_name",
]


class LicencePolicyError(ValueError):
    """Raised when a policy or manifest file is malformed, incomplete or inconsistent."""


PRODUCTION_MODE: Final = "production"
RESEARCH_MODE: Final = "research"

_POLICY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "policy_version",
        "permissive_allowlist",
        "excluded_licences",
        "modes",
        "substitutes",
        "resources",
        "data_fields",
    }
)

_MODE_KEYS: Final[frozenset[str]] = frozenset({"description", "allowed_policy_decisions"})

#: Keys required of a ``pending_request`` block. Exact, like every other block
#: in this file: a misspelled ``reply_recieved`` would otherwise leave the real
#: key missing and default a record to "unanswered" or "answered" silently.
PENDING_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "requested_of",
        "requested_utc",
        "questions_asked",
        "reply_received",
        "reply_utc",
        "reply_scope",
    }
)

#: Keys required of every software-manifest component.
SOFTWARE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {"distribution", "resource", "direct", "role"}
)

#: Keys required of every data-manifest entry.
DATA_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset({"resource", "role", "used_for"})

_MANIFEST_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {"manifest_version", "policy_version", "components"}
)

_DATA_MANIFEST_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {"manifest_version", "policy_version", "datasets"}
)

_NAME_SEPARATORS = re.compile(r"[-_.]+")


def normalise_distribution_name(name: str) -> str:
    """Normalise a Python distribution name per PEP 503.

    ``PyYAML``, ``netCDF4`` and ``SALib`` are spelled three different ways in
    ``pyproject.toml`` and a fourth way on PyPI. Comparing raw spellings would
    report a recorded dependency as unrecorded.
    """
    return _NAME_SEPARATORS.sub("-", name.strip().lower())


@dataclass(frozen=True, slots=True)
class GateMode:
    """One named execution mode and the policy decisions it tolerates."""

    name: str
    description: str
    allowed_policy_decisions: frozenset[str]

    def allows(self, policy_decision: str) -> bool:
        return policy_decision in self.allowed_policy_decisions


@dataclass(frozen=True, slots=True)
class SoftwareComponent:
    """A software component in the shipped-or-declined bill of materials.

    ``distribution`` is the PyPI distribution name where one exists and ``None``
    where the component is not a Python package (a Fortran model, a MATLAB
    tool). Only the former can be cross-checked against ``pyproject.toml``.
    """

    distribution: str | None
    resource: str
    direct: bool
    role: str


@dataclass(frozen=True, slots=True)
class DataResource:
    """A dataset in the data manifest, with the role it plays in the pipeline."""

    resource: str
    role: str
    used_for: str


@dataclass(frozen=True, slots=True)
class SoftwareManifest:
    """The software bill of materials, resolved against the licence policy."""

    manifest_version: str
    policy_version: str
    components: tuple[SoftwareComponent, ...]

    def by_distribution(self) -> dict[str, SoftwareComponent]:
        """Map normalised distribution name -> component, skipping non-Python entries."""
        return {
            normalise_distribution_name(c.distribution): c
            for c in self.components
            if c.distribution is not None
        }


@dataclass(frozen=True, slots=True)
class DataManifest:
    """The dataset bill of materials, resolved against the licence policy."""

    manifest_version: str
    policy_version: str
    datasets: tuple[DataResource, ...]


@dataclass(frozen=True, slots=True)
class LicencePolicy:
    """A fully validated licence policy: records, modes and substitutions."""

    policy_version: str
    permissive_allowlist: frozenset[str]
    excluded_licences: frozenset[str]
    modes: Mapping[str, GateMode]
    records: Mapping[str, LicenceRecord]
    substitutes: Mapping[str, str]
    data_field_entries: tuple[Mapping[str, Any], ...]
    source_path: pathlib.Path

    def record_for(self, resource_name: str) -> LicenceRecord | None:
        """The record for a resource, or ``None`` if it was never declared.

        Returning ``None`` rather than raising is deliberate: the decision about
        what an undeclared resource means belongs to the gate, which rejects it,
        not to the lookup.
        """
        return self.records.get(resource_name)

    def substitute_for(self, resource_name: str) -> LicenceRecord | None:
        """The replacement recorded for a resource, if any.

        Returns the record whatever mode the caller is in, because whether a
        substitute is REACHABLE is a question about a mode and this lookup does
        not have one. The loader has already guaranteed the record is usable in
        some mode; the gate decides whether it is usable in the mode being
        asked about, and says so when it is not.
        """
        substitute = self.substitutes.get(resource_name)
        if substitute is None:
            return None
        return self.records.get(substitute)

    def mode(self, name: str) -> GateMode | None:
        return self.modes.get(name)

    @property
    def resource_names(self) -> tuple[str, ...]:
        return tuple(self.records)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_licence_policy(path: str | pathlib.Path) -> LicencePolicy:
    """Read, validate and return the licence policy at ``path``.

    Raises :class:`LicencePolicyError` on an unknown ``resource_kind``, a
    missing or unexpected field, a ``policy_decision`` inconsistent with
    ``commercial_use``, a substitute that is not itself usable in any mode, a
    production mode that would admit more than ``permitted``, or a malformed
    variable table. There is no partially valid outcome.
    """
    policy_path = pathlib.Path(path)
    raw = _read_mapping(policy_path, "licence policy")
    _require_exact_keys(raw, _POLICY_KEYS, context="licence policy", path=policy_path)

    policy_version = _require_text(raw["policy_version"], "policy_version", policy_path)
    allowlist = _require_string_set(
        raw["permissive_allowlist"], "permissive_allowlist", policy_path
    )
    excluded = _require_string_set(raw["excluded_licences"], "excluded_licences", policy_path)
    modes = _load_modes(raw["modes"], policy_path)
    records = _load_records(raw["resources"], policy_path)
    _validate_excluded_licences_are_declined(records, excluded, policy_path)
    substitutes = _load_substitutes(raw["substitutes"], records, policy_path)
    data_field_entries = _load_data_field_entries(raw["data_fields"], policy_path)

    policy = LicencePolicy(
        policy_version=policy_version,
        permissive_allowlist=allowlist,
        excluded_licences=excluded,
        modes=MappingProxyType(modes),
        records=MappingProxyType(records),
        substitutes=MappingProxyType(substitutes),
        data_field_entries=data_field_entries,
        source_path=policy_path,
    )

    # Imported here rather than at module scope: ``data_field`` depends on this
    # module for the policy type, and validating the variable table at load time
    # is what makes "a loaded policy is a checked policy" true without turning
    # that dependency into an import cycle.
    from .data_field import parse_variable_allowlist

    parse_variable_allowlist(policy)
    return policy


def load_software_manifest(
    path: str | pathlib.Path, policy: LicencePolicy
) -> SoftwareManifest:
    """Read the software manifest and resolve every entry against ``policy``.

    The manifest names components and points at policy records; it never
    restates a licence. Two files stating the same licence fact would eventually
    disagree, and the disagreement would be invisible.
    """
    manifest_path = pathlib.Path(path)
    raw = _read_mapping(manifest_path, "software manifest")
    _require_exact_keys(raw, _MANIFEST_TOP_KEYS, context="software manifest", path=manifest_path)
    _require_matching_policy_version(raw, policy, manifest_path, "software manifest")

    entries = _require_sequence(raw["components"], "components", manifest_path)
    components: list[SoftwareComponent] = []
    seen_resources: set[str] = set()
    seen_distributions: set[str] = set()
    for index, entry in enumerate(entries):
        mapping = _require_entry_mapping(entry, index, "components", manifest_path)
        _require_exact_keys(
            mapping,
            SOFTWARE_MANIFEST_FIELDS,
            context=f"software manifest component #{index}",
            path=manifest_path,
        )
        resource = _require_text(
            mapping["resource"], f"components[{index}].resource", manifest_path
        )
        record = policy.records.get(resource)
        if record is None:
            raise LicencePolicyError(
                f"{manifest_path}: component #{index} references resource {resource!r}, which is "
                f"not declared in licence policy {policy.source_path}"
            )
        if record.resource_kind == "dataset":
            raise LicencePolicyError(
                f"{manifest_path}: component #{index} references {resource!r}, whose "
                f"resource_kind is 'dataset'. Datasets belong in the data manifest."
            )
        if resource in seen_resources:
            raise LicencePolicyError(
                f"{manifest_path}: resource {resource!r} appears more than once"
            )
        seen_resources.add(resource)

        distribution = mapping["distribution"]
        if distribution is not None:
            distribution = _require_text(
                distribution, f"components[{index}].distribution", manifest_path
            )
            key = normalise_distribution_name(distribution)
            if key in seen_distributions:
                raise LicencePolicyError(
                    f"{manifest_path}: distribution {distribution!r} appears more than once"
                )
            seen_distributions.add(key)

        direct = mapping["direct"]
        if not isinstance(direct, bool):
            raise LicencePolicyError(
                f"{manifest_path}: components[{index}].direct must be a boolean, got {direct!r}"
            )

        components.append(
            SoftwareComponent(
                distribution=distribution,
                resource=resource,
                direct=direct,
                role=_require_text(mapping["role"], f"components[{index}].role", manifest_path),
            )
        )

    return SoftwareManifest(
        manifest_version=_require_text(raw["manifest_version"], "manifest_version", manifest_path),
        policy_version=policy.policy_version,
        components=tuple(components),
    )


def load_data_manifest(path: str | pathlib.Path, policy: LicencePolicy) -> DataManifest:
    """Read the data manifest and resolve every dataset against ``policy``."""
    manifest_path = pathlib.Path(path)
    raw = _read_mapping(manifest_path, "data manifest")
    _require_exact_keys(raw, _DATA_MANIFEST_TOP_KEYS, context="data manifest", path=manifest_path)
    _require_matching_policy_version(raw, policy, manifest_path, "data manifest")

    entries = _require_sequence(raw["datasets"], "datasets", manifest_path)
    datasets: list[DataResource] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        mapping = _require_entry_mapping(entry, index, "datasets", manifest_path)
        _require_exact_keys(
            mapping,
            DATA_MANIFEST_FIELDS,
            context=f"data manifest entry #{index}",
            path=manifest_path,
        )
        resource = _require_text(mapping["resource"], f"datasets[{index}].resource", manifest_path)
        record = policy.records.get(resource)
        if record is None:
            raise LicencePolicyError(
                f"{manifest_path}: dataset #{index} references resource {resource!r}, which is "
                f"not declared in licence policy {policy.source_path}"
            )
        if record.resource_kind == "software":
            raise LicencePolicyError(
                f"{manifest_path}: dataset #{index} references {resource!r}, whose resource_kind "
                f"is 'software'. Software components belong in the software manifest."
            )
        if resource in seen:
            raise LicencePolicyError(
                f"{manifest_path}: resource {resource!r} appears more than once"
            )
        seen.add(resource)
        datasets.append(
            DataResource(
                resource=resource,
                role=_require_text(mapping["role"], f"datasets[{index}].role", manifest_path),
                used_for=_require_text(
                    mapping["used_for"], f"datasets[{index}].used_for", manifest_path
                ),
            )
        )

    return DataManifest(
        manifest_version=_require_text(raw["manifest_version"], "manifest_version", manifest_path),
        policy_version=policy.policy_version,
        datasets=tuple(datasets),
    )


# --------------------------------------------------------------------------
# section loaders
# --------------------------------------------------------------------------


def _load_modes(raw_modes: Any, path: pathlib.Path) -> dict[str, GateMode]:
    if not isinstance(raw_modes, Mapping):
        raise LicencePolicyError(
            f"{path}: 'modes' must be a mapping, got {type(raw_modes).__name__}"
        )

    modes: dict[str, GateMode] = {}
    for name, body in raw_modes.items():
        if not isinstance(name, str) or not name:
            raise LicencePolicyError(f"{path}: mode names must be non-empty strings, got {name!r}")
        if not isinstance(body, Mapping):
            raise LicencePolicyError(f"{path}: mode {name!r} must be a mapping")
        _require_exact_keys(body, _MODE_KEYS, context=f"mode {name!r}", path=path)
        decisions = _require_string_set(
            body["allowed_policy_decisions"], f"modes.{name}.allowed_policy_decisions", path
        )
        unknown = decisions - POLICY_DECISIONS
        if unknown:
            raise LicencePolicyError(
                f"{path}: mode {name!r} allows unknown policy decision(s) {sorted(unknown)}; "
                f"allowed values are {sorted(POLICY_DECISIONS)}"
            )
        modes[name] = GateMode(
            name=name,
            description=_require_text(body["description"], f"modes.{name}.description", path),
            allowed_policy_decisions=decisions,
        )

    for required in (PRODUCTION_MODE, RESEARCH_MODE):
        if required not in modes:
            raise LicencePolicyError(f"{path}: 'modes' must define {required!r}")

    production = modes[PRODUCTION_MODE]
    if production.allowed_policy_decisions != frozenset({"permitted"}):
        raise LicencePolicyError(
            f"{path}: the production mode must allow exactly ['permitted'], not "
            f"{sorted(production.allowed_policy_decisions)}. The production floor is an "
            f"invariant of this gate, not a configurable value: a mode that admits "
            f"'excluded_by_policy' or 'prohibited_by_licence' would ship a component the "
            f"audit refused."
        )
    return modes


def _load_records(raw_records: Any, path: pathlib.Path) -> dict[str, LicenceRecord]:
    entries = _require_sequence(raw_records, "resources", path)
    records: dict[str, LicenceRecord] = {}
    required = frozenset(RECORD_REQUIRED_FIELDS)
    optional = frozenset(RECORD_OPTIONAL_FIELDS)
    for index, entry in enumerate(entries):
        mapping = _require_entry_mapping(entry, index, "resources", path)
        _require_exact_keys(
            mapping, required, optional=optional, context=f"resource #{index}", path=path
        )
        values: dict[str, Any] = {name: mapping[name] for name in RECORD_REQUIRED_FIELDS}
        # Optional fields are copied generically so that a field added to the
        # dataclass cannot be silently dropped here: an earlier version of this
        # loop named each optional field by hand, which meant a record could
        # carry a key the schema accepted and the loader discarded — the record
        # then failed validation for a value its source file plainly contained.
        for name in RECORD_OPTIONAL_FIELDS:
            if name == "pending_request" or name not in mapping:
                continue
            values[name] = mapping[name]
        if "pending_request" in mapping:
            values["pending_request"] = _load_pending_request(
                mapping["pending_request"], index, path
            )
        try:
            record = LicenceRecord(**values)
        except LicenceRecordError as exc:
            raise LicencePolicyError(f"{path}: resource #{index} is invalid: {exc}") from exc
        if record.resource_name in records:
            raise LicencePolicyError(
                f"{path}: resource {record.resource_name!r} is declared more than once"
            )
        records[record.resource_name] = record
    if not records:
        raise LicencePolicyError(f"{path}: 'resources' declares no records")
    return records


def _load_pending_request(
    raw: Any, index: int, path: pathlib.Path
) -> PendingLicenceRequest | None:
    """Build the outstanding-question sub-record, or ``None`` where there is none.

    ``questions_asked`` becomes a tuple here rather than in the dataclass: the
    record is frozen evidence, and a list read out of YAML would still be a
    mutable object held inside it.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise LicencePolicyError(
            f"{path}: resources[{index}].pending_request must be a mapping or null, got "
            f"{type(raw).__name__}"
        )
    _require_exact_keys(
        raw,
        PENDING_REQUEST_FIELDS,
        context=f"resources[{index}].pending_request",
        path=path,
    )
    questions = raw["questions_asked"]
    if isinstance(questions, str) or not isinstance(questions, Sequence):
        raise LicencePolicyError(
            f"{path}: resources[{index}].pending_request.questions_asked must be a list, got "
            f"{type(questions).__name__}"
        )
    try:
        return PendingLicenceRequest(
            requested_of=raw["requested_of"],
            requested_utc=raw["requested_utc"],
            questions_asked=tuple(questions),
            reply_received=raw["reply_received"],
            reply_utc=raw["reply_utc"],
            reply_scope=raw["reply_scope"],
        )
    except LicenceRecordError as exc:
        raise LicencePolicyError(
            f"{path}: resources[{index}].pending_request is invalid: {exc}"
        ) from exc


def _validate_excluded_licences_are_declined(
    records: Mapping[str, LicenceRecord], excluded: frozenset[str], path: pathlib.Path
) -> None:
    """A licence on the excluded list may not carry a ``permitted`` decision.

    This is the check that keeps the excluded list from becoming documentation:
    adding ``GPL-3.0-only`` to it and then marking a GPL component permitted
    fails the load rather than passing review.
    """
    for record in records.values():
        if record.spdx_or_licence_name in excluded and record.policy_decision == "permitted":
            raise LicencePolicyError(
                f"{path}: resource {record.resource_name!r} carries licence "
                f"{record.spdx_or_licence_name!r}, which is on the excluded-by-policy list, but "
                f"its policy_decision is 'permitted'"
            )


def _load_substitutes(
    raw_substitutes: Any, records: Mapping[str, LicenceRecord], path: pathlib.Path
) -> dict[str, str]:
    if not isinstance(raw_substitutes, Mapping):
        raise LicencePolicyError(
            f"{path}: 'substitutes' must be a mapping, got {type(raw_substitutes).__name__}"
        )
    substitutes: dict[str, str] = {}
    for key, value in raw_substitutes.items():
        name = _require_text(key, "substitutes key", path)
        replacement = _require_text(value, f"substitutes.{name}", path)
        if name not in records:
            raise LicencePolicyError(
                f"{path}: substitutes names undeclared resource {name!r}"
            )
        target = records.get(replacement)
        if target is None:
            raise LicencePolicyError(
                f"{path}: substitute {replacement!r} for {name!r} is not a declared resource"
            )
        if not target.is_available_as_substitute:
            raise LicencePolicyError(
                f"{path}: substitute {replacement!r} for {name!r} has policy_decision "
                f"{target.policy_decision!r}. A substitute must itself be usable — "
                f"{sorted(SUBSTITUTABLE_DECISIONS)} — and not frozen, or the gate would point a "
                f"caller at a second refusal. A resource this project declines, is legally "
                f"barred from, or has frozen pending a written reply cannot stand in for another."
            )
        substitutes[name] = replacement
    return substitutes


def _load_data_field_entries(raw_entries: Any, path: pathlib.Path) -> tuple[Mapping[str, Any], ...]:
    """Shape-check the variable table; ``data_field`` checks its vocabulary."""
    entries = _require_sequence(raw_entries, "data_fields", path)
    checked: list[Mapping[str, Any]] = []
    for index, entry in enumerate(entries):
        mapping = _require_entry_mapping(entry, index, "data_fields", path)
        checked.append(MappingProxyType(dict(mapping)))
    return tuple(checked)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def _read_mapping(path: pathlib.Path, what: str) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise LicencePolicyError(
            f"{path}: {what} must be a YAML mapping, got {type(loaded).__name__}"
        )
    return loaded


def _require_exact_keys(
    mapping: Mapping[str, Any],
    required: frozenset[str],
    *,
    context: str,
    path: pathlib.Path,
    optional: frozenset[str] = frozenset(),
) -> None:
    present = frozenset(str(key) for key in mapping)
    missing = required - present
    if missing:
        raise LicencePolicyError(
            f"{path}: {context} is missing required field(s) {sorted(missing)}"
        )
    unexpected = present - required - optional
    if unexpected:
        expected = (
            f"exactly {sorted(required)}"
            if not optional
            else f"{sorted(required)}, optionally with {sorted(optional)}"
        )
        raise LicencePolicyError(
            f"{path}: {context} has unexpected field(s) {sorted(unexpected)}; expected {expected}"
        )


def _require_text(value: Any, field_name: str, path: pathlib.Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LicencePolicyError(
            f"{path}: {field_name} must be a non-empty string, got {value!r}"
        )
    return value


def _require_string_set(value: Any, field_name: str, path: pathlib.Path) -> frozenset[str]:
    items = _require_sequence(value, field_name, path)
    out: set[str] = set()
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise LicencePolicyError(
                f"{path}: {field_name} must contain non-empty strings, got {item!r}"
            )
        if item in out:
            raise LicencePolicyError(f"{path}: {field_name} lists {item!r} more than once")
        out.add(item)
    if not out:
        raise LicencePolicyError(f"{path}: {field_name} must not be empty")
    return frozenset(out)


def _require_sequence(value: Any, field_name: str, path: pathlib.Path) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise LicencePolicyError(
            f"{path}: {field_name} must be a list, got {type(value).__name__}"
        )
    return value


def _require_entry_mapping(
    entry: Any, index: int, field_name: str, path: pathlib.Path
) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise LicencePolicyError(
            f"{path}: {field_name}[{index}] must be a mapping, got {type(entry).__name__}"
        )
    return entry


def _require_matching_policy_version(
    raw: Mapping[str, Any], policy: LicencePolicy, path: pathlib.Path, what: str
) -> None:
    declared = _require_text(raw["policy_version"], "policy_version", path)
    if declared != policy.policy_version:
        raise LicencePolicyError(
            f"{path}: {what} declares policy_version {declared!r} but the loaded policy is "
            f"{policy.policy_version!r}. The manifest and the policy must be revised together."
        )
