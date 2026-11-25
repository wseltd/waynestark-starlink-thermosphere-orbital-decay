"""Licence and data-policy enforcement: the reviewed position, made executable.

This package turns the project's licence review from prose into a gate that
code must pass:

    record           one licensed resource; keeps the licence FACT
                     (``classification``, ``commercial_use``) apart from this
                     project's CHOICE (``policy_decision``)
    policy_loader    reads ``config/licence_policy.yaml`` and the two manifests,
                     rejecting anything incomplete, unknown or self-contradictory
    production_gate  ``check_resource`` — fail-closed, keyed on the model rather
                     than the package — and the pyproject cross-check
    data_field       per-variable terms, because one dataset can mix licences

Typical use::

    from thermosphere_consequence.licence import check_resource, load_licence_policy

    policy = load_licence_policy("config/licence_policy.yaml")
    check_resource("pymsis", policy, mode="production")       # returns its record
    check_resource("nrlmsise00", policy, mode="research")     # returns its record
    check_resource("nrlmsise00", policy, mode="production")   # raises: research-only
    check_resource("nrlmsis_2_1", policy, mode="research")    # raises: only the
    #   'open_noncommercial_research' mode admits the written NRL permission

Importing this package reads no files and touches no network; every load is an
explicit call with an explicit path.
"""

from __future__ import annotations

from .data_field import (
    DataFieldLicence,
    VariableAllowlist,
    check_variable,
    parse_variable_allowlist,
)
from .policy_loader import (
    PRODUCTION_MODE,
    RESEARCH_MODE,
    DataManifest,
    GateMode,
    LicencePolicy,
    LicencePolicyError,
    SoftwareManifest,
    load_data_manifest,
    load_licence_policy,
    load_software_manifest,
)
from .production_gate import (
    DependencyAudit,
    ProductionGateError,
    check_all_declared_dependencies,
    check_resource,
    parse_runtime_dependency_names,
)
from .record import LicenceRecord, LicenceRecordError

__all__ = [
    "PRODUCTION_MODE",
    "RESEARCH_MODE",
    "DataFieldLicence",
    "DataManifest",
    "DependencyAudit",
    "GateMode",
    "LicencePolicy",
    "LicencePolicyError",
    "LicenceRecord",
    "LicenceRecordError",
    "ProductionGateError",
    "SoftwareManifest",
    "VariableAllowlist",
    "check_all_declared_dependencies",
    "check_resource",
    "check_variable",
    "load_data_manifest",
    "load_licence_policy",
    "load_software_manifest",
    "parse_runtime_dependency_names",
    "parse_variable_allowlist",
]
