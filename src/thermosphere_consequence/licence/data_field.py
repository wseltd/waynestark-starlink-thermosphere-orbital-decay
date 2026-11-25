"""Per-variable data licensing, because a dataset can carry mixed terms.

Treating a dataset as one licensed object is the mistake this module exists to
stop. Two cases in this project's own dependency set prove the point:

  * **NASA OMNI.** The aggregation is US-Government open data. Inside it,
    SYM-H, ASY-H and AE are Kyoto-DERIVED indices, and WDC Kyoto's own terms
    state that it "does not allow commercial applications of the geomagnetic
    indices". Whether NASA's terms cover the derived fields in full is an OPEN
    QUESTION, tracked as ``BLK-OMNI-KYOTO-FIELDS`` in
    ``planning/registry/blockers.yaml`` and documented in
    ``drivers/omni.py``. This module records it as ``conditional`` and keeps
    those variables out of production. It does not assert that they are fine,
    and it does not assert that they are forbidden — writing down either answer
    would be inventing a legal determination nobody made.
  * **TU Delft.** The GRACE-FO density and crosswind DATA are CC BY 4.0. The
    geometry, aerodynamic and radiation-pressure MODELS from the same group are
    CC BY-NC-SA 4.0 — non-commercial. A dataset-level record would have to pick
    one, and either choice would be wrong about half the material.

So the unit of licensing here is ``(dataset, variable)``, and the gate is
fail-closed in the same way as the resource gate: a variable nobody recorded is
refused, not waved through.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Final

from .policy_loader import PRODUCTION_MODE, RESEARCH_MODE, LicencePolicy, LicencePolicyError
from .production_gate import ProductionGateError
from .record import COMMERCIAL_USE_VALUES, LicenceRecordError

__all__ = [
    "DATA_FIELD_FIELDS",
    "DataFieldLicence",
    "VariableAllowlist",
    "check_variable",
    "parse_variable_allowlist",
]

#: A variable is identified by its dataset and its name within that dataset.
VariableKey = tuple[str, str]

#: What :func:`parse_variable_allowlist` returns and :func:`check_variable` reads.
VariableAllowlist = dict[VariableKey, "DataFieldLicence"]

_SUBSTITUTE_SEPARATOR: Final = "."


@dataclass(frozen=True, slots=True)
class DataFieldLicence:
    """The licence position for one variable inside one dataset."""

    dataset: str
    variable_name: str
    licence: str
    commercial_use: str
    production_allowed: bool
    substitute_variable: str | None
    note: str

    def __post_init__(self) -> None:
        label = f"{self.dataset}.{self.variable_name}"
        for field_name in ("dataset", "variable_name", "licence"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise LicenceRecordError(
                    f"data field {label!r}: {field_name!r} must be a non-empty string, "
                    f"got {value!r}"
                )
        if self.commercial_use not in COMMERCIAL_USE_VALUES:
            raise LicenceRecordError(
                f"data field {label!r}: commercial_use {self.commercial_use!r} is not one of "
                f"{sorted(COMMERCIAL_USE_VALUES)}"
            )
        if not isinstance(self.production_allowed, bool):
            raise LicenceRecordError(
                f"data field {label!r}: production_allowed must be a boolean, "
                f"got {self.production_allowed!r}"
            )
        if self.substitute_variable is not None and (
            not isinstance(self.substitute_variable, str) or not self.substitute_variable.strip()
        ):
            raise LicenceRecordError(
                f"data field {label!r}: substitute_variable must be null or a non-empty string, "
                f"got {self.substitute_variable!r}"
            )
        if not isinstance(self.note, str):
            raise LicenceRecordError(f"data field {label!r}: note must be a string")

        if self.commercial_use == "prohibited" and self.production_allowed:
            raise LicenceRecordError(
                f"data field {label!r}: commercial_use is 'prohibited' but production_allowed is "
                f"True. Production is a commercial context; the two cannot both hold."
            )
        if self.commercial_use == "conditional" and not self.note.strip():
            raise LicenceRecordError(
                f"data field {label!r}: commercial_use 'conditional' requires a note stating the "
                f"condition. 'Conditional' without the condition written down is an unrecorded "
                f"assumption."
            )

    @property
    def key(self) -> VariableKey:
        return (self.dataset, self.variable_name)


#: Keys every ``data_fields`` entry in the policy file must supply.
DATA_FIELD_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in fields(DataFieldLicence))


def parse_variable_allowlist(policy: LicencePolicy) -> VariableAllowlist:
    """Build and fully validate the per-variable table from ``policy``.

    Raises:
        LicencePolicyError: on a missing or unexpected key, an unknown
            vocabulary value, a duplicate variable, a dataset that is not a
            declared resource, a variable marked production-allowed inside a
            dataset the resource gate refuses, or a substitute that does not
            resolve to a production-allowed variable.
    """
    path = policy.source_path
    required = frozenset(DATA_FIELD_FIELDS)
    allowlist: VariableAllowlist = {}

    for index, entry in enumerate(policy.data_field_entries):
        present = frozenset(str(key) for key in entry)
        missing = required - present
        if missing:
            raise LicencePolicyError(
                f"{path}: data_fields[{index}] is missing required field(s) {sorted(missing)}"
            )
        unexpected = present - required
        if unexpected:
            raise LicencePolicyError(
                f"{path}: data_fields[{index}] has unexpected field(s) {sorted(unexpected)}; "
                f"expected exactly {sorted(required)}"
            )
        try:
            field_licence = DataFieldLicence(**{name: entry[name] for name in DATA_FIELD_FIELDS})
        except LicenceRecordError as exc:
            raise LicencePolicyError(f"{path}: data_fields[{index}] is invalid: {exc}") from exc

        record = policy.record_for(field_licence.dataset)
        if record is None:
            raise LicencePolicyError(
                f"{path}: data_fields[{index}] names dataset {field_licence.dataset!r}, which is "
                f"not a declared resource"
            )
        if record.resource_kind == "software":
            raise LicencePolicyError(
                f"{path}: data_fields[{index}] names {field_licence.dataset!r}, whose "
                f"resource_kind is 'software'. Variables belong to data resources."
            )
        if field_licence.production_allowed and not record.is_permitted_in_production:
            raise LicencePolicyError(
                f"{path}: data_fields[{index}] marks {field_licence.variable_name!r} as "
                f"production-allowed, but its dataset {field_licence.dataset!r} has "
                f"policy_decision {record.policy_decision!r}. A variable cannot be cleared for "
                f"production inside a dataset the resource gate refuses."
            )
        if field_licence.key in allowlist:
            raise LicencePolicyError(
                f"{path}: data_fields declares {field_licence.dataset}."
                f"{field_licence.variable_name} more than once"
            )
        allowlist[field_licence.key] = field_licence

    _validate_substitutes(allowlist, policy)
    return allowlist


def _validate_substitutes(allowlist: VariableAllowlist, policy: LicencePolicy) -> None:
    """Every recorded substitute must exist and must itself be usable.

    A substitute pointing at a second refused variable would send a caller in a
    circle, which is worse than recording no substitute at all.
    """
    for field_licence in allowlist.values():
        target_name = field_licence.substitute_variable
        if target_name is None:
            continue
        target_key = _resolve_substitute_key(field_licence.dataset, target_name)
        target = allowlist.get(target_key)
        if target is None:
            raise LicencePolicyError(
                f"{policy.source_path}: {field_licence.dataset}.{field_licence.variable_name} "
                f"names substitute {target_name!r}, which is not a declared variable. Use "
                f"'dataset{_SUBSTITUTE_SEPARATOR}variable', or a bare variable name in the same "
                f"dataset."
            )
        if not target.production_allowed:
            raise LicencePolicyError(
                f"{policy.source_path}: {field_licence.dataset}.{field_licence.variable_name} "
                f"names substitute {target_name!r}, which is itself not production-allowed"
            )


def _resolve_substitute_key(dataset: str, substitute: str) -> VariableKey:
    if _SUBSTITUTE_SEPARATOR in substitute:
        other_dataset, _, variable = substitute.partition(_SUBSTITUTE_SEPARATOR)
        return (other_dataset, variable)
    return (dataset, substitute)


def check_variable(
    dataset: str,
    variable: str,
    *,
    mode: str,
    allowlist: Mapping[VariableKey, DataFieldLicence],
) -> DataFieldLicence:
    """Return the licence position for one variable, or raise if ``mode`` forbids it.

    ``allowlist`` is the table produced by :func:`parse_variable_allowlist`; it
    is passed in rather than re-derived so that a check performs no I/O and
    cannot silently validate a different policy than the caller loaded.

    Raises:
        ProductionGateError: if the mode is unrecognised, if the variable is not
            recorded (fail-closed), or if production mode does not allow it. The
            message names the dataset, the variable, its licence position, the
            recorded note and the substitute where one exists.
    """
    if mode not in (PRODUCTION_MODE, RESEARCH_MODE):
        raise ProductionGateError(
            f"unknown data-licence mode {mode!r}; expected {PRODUCTION_MODE!r} or "
            f"{RESEARCH_MODE!r}. An unrecognised mode is refused rather than assumed permissive."
        )

    field_licence = allowlist.get((dataset, variable))
    if field_licence is None:
        raise ProductionGateError(
            f"variable {dataset}.{variable} is REJECTED in {mode} mode: it has no per-variable "
            f"licence record. Datasets carry mixed terms, so an unrecorded variable has no "
            f"reviewed position and this gate is fail-closed. Add it to the policy's data_fields "
            f"table before reading it."
        )

    if mode == PRODUCTION_MODE and not field_licence.production_allowed:
        substitute = (
            f"Permitted substitute: {field_licence.substitute_variable}."
            if field_licence.substitute_variable is not None
            else "No permitted substitute is recorded."
        )
        raise ProductionGateError(
            f"variable {dataset}.{variable} is REJECTED in production mode. "
            f"Licence: {field_licence.licence}. Commercial use: {field_licence.commercial_use}. "
            f"Note: {field_licence.note} {substitute}"
        )

    return field_licence
