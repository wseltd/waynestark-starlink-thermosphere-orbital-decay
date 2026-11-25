"""Where providers are chosen, cleared and assembled — and nowhere else.

Every other module in this package receives a provider set; this one is the
only place that builds it. Concentrating the choice here is what makes two
properties checkable rather than conventional: that nothing enters a run
without passing the licence gate, and that a run can name every component it
used.

FAIL CLOSED, IN THREE PLACES.

**An unknown provider name is refused.** The registry below is a closed table.
A name that is not in it raises :class:`UnknownProviderError` naming the name
and listing what this service does compose. The tempting alternative — fall
back to the default provider when the name is unrecognised — turns a typo into
a silently different physical model, and the run reports the model it fell back
to as though it had been asked for.

**An unknown mode is refused.** A mode the loaded policy does not declare
raises rather than being treated as permissive. This is checked here as well as
inside the gate, because an in-house provider never reaches the gate and would
otherwise be composed under a mode nobody defined.

**A resource the licence gate refuses is refused.** Clearance is delegated to
:func:`~thermosphere_consequence.licence.production_gate.check_resource` rather
than reimplemented, so this module and the dependency audit cannot drift into
disagreeing about the same resource. Production admits ``permitted`` and
nothing else; research additionally admits the academically licensed and
copyleft records, and the records whose terms have not been located at all,
which is what those positions themselves allow — and is precisely why a
research result must never be published from a production build.

PRODUCTION HAS NO ATMOSPHERE PROVIDER, AND THAT IS THE ANSWER RATHER THAN A
BUG. Every registered atmosphere model is refused in production: NRLMSIS 2.0
and 2.1 carry the written NRL permission for open non-commercial research and
are admitted by the ``open_noncommercial_research`` mode ALONE — refused in
production AND in research, because the confirmation covers the described
open-publication workflow and nothing wider — DTM2020 is prohibited by a CNES
academic licence, and NRLMSISE-00 is ``permitted_research_only`` because the
Fortran this project compiles carries no licence text of any kind — an absence
of terms, which is not a grant. This project is open non-commercial research
and has never had a verified production atmosphere. Rather than let that arrive
as unrelated refusals that each look like a missing feature, a refusal
that leaves a provider kind with nothing composable says so in as many words,
naming the mode, the verification gap and the way out.

IN-HOUSE PROVIDERS ARE NOT GATED, AND THAT IS NOT A HOLE. ``ZeroWindProvider``
and ``ConstantAlongTrackWindProvider`` are implemented in this repository under
its own licence; there is no third-party resource for a gate to clear. The
licence policy takes the same position for the components whose replacement is
this project's own code. Such a registration is marked in the table, carries
its reason, and that reason travels into the composed set — so "not gated"
appears in the run record instead of being invisible.

A REGISTERED NAME WITH NO IMPLEMENTATION IS A DIFFERENT REFUSAL FROM AN UNKNOWN
NAME. ``dtm2020`` is registered deliberately with no builder. It is a real
model a caller might reasonably ask for, it is refused in production by its
licence, and it does not ship here. Dropping it from the table would collapse
three distinguishable answers — "no such thing", "you may not use it", "you may
use it but this repository does not implement it" — into one, and the first is
the only one of the three that is a mistake by the caller. ``nrlmsis_2_0`` and
``nrlmsis_2_1`` moved OUT of that category on 2026-08-23: implementations now
ship (an original Apache-2.0 driver over the separately installed official
sources), and their builders refuse with instructions when the external
installation or the compiled driver is absent — an environment fact, distinct
from both the licence answer and the registry answer.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..licence.policy_loader import LicencePolicy, load_licence_policy
from ..licence.production_gate import ProductionGateError, check_resource
from ..licence.record import LicenceRecord
from ..protocols.atmosphere import AtmosphereProvider, WindProvider
from ..providers.nrlmsis2 import (
    Nrlmsis2AssetError,
    NRLMSIS20Provider,
    NRLMSIS21Provider,
    load_asset_manifest_entry,
)
from ..providers.nrlmsise00 import Nrlmsise00Provider, ZeroWindProvider
from ..providers.wind import ConstantAlongTrackWindProvider
from ..schemas.modes import AnalysisMode

__all__ = [
    "ATMOSPHERE_PROVIDERS",
    "DEFAULT_LICENCE_POLICY_PATH",
    "REPOSITORY_ROOT",
    "WIND_PROVIDERS",
    "ProviderClearance",
    "ProviderNotImplementedError",
    "ProviderRegistration",
    "ProviderSet",
    "ServiceConfig",
    "UnknownModeError",
    "UnknownProviderError",
    "build_provider_set",
    "load_policy",
    "resolve_provider_licence",
]

#: Anchored on the installed package rather than the working directory, so the
#: service composes the same way whichever directory a process was started
#: from. ``parents[3]`` is ``service`` -> package -> ``src`` -> repository.
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: The register the gate reads. An explicit field on :class:`ServiceConfig`
#: rather than a hidden default lookup, because which policy cleared a run is
#: part of the run.
DEFAULT_LICENCE_POLICY_PATH = REPOSITORY_ROOT.joinpath("config", "licence_policy.yaml")

_ATMOSPHERE_KIND = "atmosphere"
_WIND_KIND = "wind"


class UnknownProviderError(ValueError):
    """Raised when a configuration names a provider this service does not compose."""


class UnknownModeError(ValueError):
    """Raised when a configuration names a mode the loaded policy does not declare."""


class ProviderNotImplementedError(RuntimeError):
    """Raised when a registered provider has no implementation in this repository.

    Deliberately distinct from :class:`UnknownProviderError` and from
    :class:`~thermosphere_consequence.licence.production_gate.ProductionGateError`.
    Reaching this error means the licence gate ADMITTED the resource and the
    refusal is an engineering one, which is a different fact about the run and
    must not be reported as a licence refusal.
    """


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """Everything the composition root needs, stated rather than discovered.

    No field is defaulted to a value that would change the physics. The
    provider names and the mode are required, because a run that did not choose
    its atmosphere model or declare the mode it is entitled to speak in is not
    a run whose output can be quoted.
    """

    atmosphere_provider: str
    wind_provider: str
    mode: str
    licence_policy_path: pathlib.Path = DEFAULT_LICENCE_POLICY_PATH
    analysis_mode: AnalysisMode = AnalysisMode.SCENARIO
    wind_east_m_s: float = 0.0
    wind_north_m_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("atmosphere_provider", "wind_provider", "mode"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """One entry of the closed provider table.

    ``licence_resource`` is ``None`` only for a provider implemented in this
    repository, and then ``in_house_reason`` must say so. ``build`` is ``None``
    for a resource this repository deliberately does not implement, and then
    ``unavailable_reason`` must say why.
    """

    licence_resource: str | None
    build: Callable[[ServiceConfig], Any] | None
    description: str
    in_house_reason: str = ""
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        if self.licence_resource is None and not self.in_house_reason.strip():
            raise ValueError(
                "a registration with no licence resource must state why it is "
                "in-house; an ungated provider with no recorded reason is "
                "indistinguishable from one that was never reviewed"
            )
        if self.build is None and not self.unavailable_reason.strip():
            raise ValueError(
                "a registration with no implementation must state why; a "
                "silent absence reads to a caller as a licence refusal"
            )


def _build_nrlmsise00(config: ServiceConfig) -> Nrlmsise00Provider:
    """The 2001 reference model, reached through pymsis pinned to version 0."""
    return Nrlmsise00Provider()


def _build_nrlmsis2[
    ProviderT: (NRLMSIS20Provider, NRLMSIS21Provider)
](
    config: ServiceConfig,
    provider_class: type[ProviderT],
    logical_id: str,
    version_label: str,
) -> ProviderT:
    """Construct one NRLMSIS 2.x provider over the separately installed official package.

    The ONLY thing read from the environment is the installation-root path,
    under the variable the public asset manifest names for this package
    (``NRLMSIS20_ROOT`` for 2.0, ``NRLMSIS21_ROOT`` for 2.1 — names documented
    in ``.env.example``, values never committed). The licence policy and the
    gate mode arrive from the configuration, so no environment variable can
    widen what the gate admits; an unset root is an environment refusal with
    instructions, never a silent fallback.
    """
    entry = load_asset_manifest_entry(logical_id)
    root = os.environ.get(entry.environment_variable, "").strip()
    if not root:
        raise Nrlmsis2AssetError(
            f"atmosphere provider 'nrlmsis_{version_label.replace('.', '_')}' needs the "
            f"{entry.environment_variable} environment variable, and it is not set. "
            f"Install the official NRLMSIS {version_label} package from the authorised "
            f"source {entry.authorised_source} OUTSIDE this repository, export "
            f"{entry.environment_variable}=<installation directory>, then build the "
            f"driver with `scripts/build_nrlmsis2_driver.sh {version_label}`. Nothing "
            f"is downloaded automatically."
        )
    return provider_class(
        policy=load_policy(config), mode=config.mode, root=pathlib.Path(root)
    )


def _build_nrlmsis20(config: ServiceConfig) -> NRLMSIS20Provider:
    """NRLMSIS 2.0 over the external official install; root from NRLMSIS20_ROOT."""
    return _build_nrlmsis2(config, NRLMSIS20Provider, "nrlmsis_2_0_package", "2.0")


def _build_nrlmsis21(config: ServiceConfig) -> NRLMSIS21Provider:
    """NRLMSIS 2.1 over the external official install; root from NRLMSIS21_ROOT."""
    return _build_nrlmsis2(config, NRLMSIS21Provider, "nrlmsis_2_1_package", "2.1")


def _build_zero_wind(config: ServiceConfig) -> ZeroWindProvider:
    """The explicit, labelled zero-wind choice — never an unstated omission."""
    return ZeroWindProvider()


def _build_constant_wind(config: ServiceConfig) -> ConstantAlongTrackWindProvider:
    """A constant wind sensitivity case, taking its magnitude from the config."""
    return ConstantAlongTrackWindProvider(
        config.wind_east_m_s, wind_north_m_s=config.wind_north_m_s
    )


#: Atmosphere providers this service will compose, keyed on the name a caller
#: writes in a configuration.
ATMOSPHERE_PROVIDERS: Mapping[str, ProviderRegistration] = MappingProxyType(
    {
        "nrlmsise00": ProviderRegistration(
            licence_resource="nrlmsise00",
            build=_build_nrlmsise00,
            description=(
                "NRLMSISE-00 (Picone et al. 2002), species-resolved, reached "
                "through pymsis in version=0 mode. RESEARCH ONLY: the NRL "
                "Fortran compiled here carries no licence text of any kind, so "
                "its terms are unverified in both directions and it is refused "
                "in production (BLK-NRLMSISE00-ARTEFACT-PROVENANCE)."
            ),
        ),
        "nrlmsis_2_0": ProviderRegistration(
            licence_resource="nrlmsis_2_0",
            build=_build_nrlmsis20,
            description=(
                "NRLMSIS 2.0 (release 26 MAY 2020), US Naval Research "
                "Laboratory, species-resolved, NO reported ABSENT. Runs the "
                "OFFICIAL, UNMODIFIED sources from a separately installed, "
                "hash-verified external root (NRLMSIS20_ROOT) through this "
                "project's original Apache-2.0 driver. Written NRL permission "
                "for open non-commercial research: admitted in the "
                "'open_noncommercial_research' mode only, refused everywhere "
                "else including research."
            ),
        ),
        "nrlmsis_2_1": ProviderRegistration(
            licence_resource="nrlmsis_2_1",
            build=_build_nrlmsis21,
            description=(
                "NRLMSIS 2.1 (release 04 APR 2022, adds NO density), US Naval "
                "Research Laboratory, species-resolved. Runs the OFFICIAL, "
                "UNMODIFIED sources from a separately installed, hash-verified "
                "external root (NRLMSIS21_ROOT) through this project's "
                "original Apache-2.0 driver. Written NRL permission for open "
                "non-commercial research: admitted in the "
                "'open_noncommercial_research' mode only, refused everywhere "
                "else including research."
            ),
        ),
        "dtm2020": ProviderRegistration(
            licence_resource="dtm2020",
            build=None,
            description="DTM2020 / MCM (SWAMI), CNES.",
            unavailable_reason=(
                "CNES grants DTM2020 for academic, non-commercial purposes "
                "only, so no implementation ships in this repository."
            ),
        ),
    }
)

#: Wind providers this service will compose.
WIND_PROVIDERS: Mapping[str, ProviderRegistration] = MappingProxyType(
    {
        "zero_wind": ProviderRegistration(
            licence_resource=None,
            build=_build_zero_wind,
            description="Neutral wind set to exactly zero, as a named choice.",
            in_house_reason=(
                "Implemented in this repository under its own licence. There is "
                "no third-party resource for the gate to clear. Note that zero "
                "wind is a modelling choice and not a measurement: storm-time "
                "winds at 210 km reach several hundred m/s."
            ),
        ),
        "constant_along_track": ProviderRegistration(
            licence_resource=None,
            build=_build_constant_wind,
            description=(
                "A constant horizontal wind, swept as a sensitivity parameter. "
                "A sensitivity instrument, not a wind model."
            ),
            in_house_reason=(
                "Implemented in this repository under its own licence. No "
                "third-party resource is involved."
            ),
        ),
        "hwm14": ProviderRegistration(
            licence_resource="hwm14",
            build=None,
            description="HWM14 horizontal wind model with DWM07 disturbance winds.",
            unavailable_reason=(
                "HWM14 is excluded from production because the original NRL "
                "terms have not been located in writing, so no implementation "
                "ships here. The recorded fallback is an explicit zero-wind or "
                "bounded-wind scenario, never an invisible default."
            ),
        ),
    }
)

_REGISTRIES: Mapping[str, Mapping[str, ProviderRegistration]] = MappingProxyType(
    {_ATMOSPHERE_KIND: ATMOSPHERE_PROVIDERS, _WIND_KIND: WIND_PROVIDERS}
)


@dataclass(frozen=True, slots=True)
class ProviderClearance:
    """Why one provider slot is allowed to be filled, recorded with the answer.

    ``record`` is the reviewed licence record where a third-party resource was
    cleared, and ``None`` where the provider is implemented in this repository.
    ``basis`` states which of those two it was in words, so a reader of a run
    record does not have to infer it from a null.
    """

    provider_name: str
    licence_resource: str | None
    record: LicenceRecord | None
    basis: str


@dataclass(frozen=True, slots=True)
class ProviderSet:
    """A composed, cleared set of providers, with the record of its clearance."""

    atmosphere: AtmosphereProvider
    wind: WindProvider
    atmosphere_clearance: ProviderClearance
    wind_clearance: ProviderClearance
    mode: str
    analysis_mode: AnalysisMode
    policy_version: str
    policy_path: pathlib.Path
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def sources(self) -> tuple[str, ...]:
        """The provenance lines every answer built from this set must carry."""
        return (
            f"atmosphere provider: {self.atmosphere.name} {self.atmosphere.version}",
            f"wind provider: {self.wind.name} {self.wind.version}",
            self.atmosphere_clearance.basis,
            self.wind_clearance.basis,
            f"licence policy {self.policy_version} ({self.policy_path})",
            f"licence gate mode: {self.mode}",
            f"analysis mode: {self.analysis_mode.value}",
        )


def load_policy(config: ServiceConfig) -> LicencePolicy:
    """Load and fully validate the licence policy this configuration names.

    Separate from :func:`build_provider_set` so a caller that wants to ask what
    a mode would admit, without composing anything, does not have to build
    providers to find out.
    """
    return load_licence_policy(config.licence_policy_path)


def _registration(name: str, kind: str) -> ProviderRegistration:
    registry = _REGISTRIES[kind]
    registration = registry.get(name)
    if registration is None:
        raise UnknownProviderError(
            f"unknown {kind} provider {name!r}. This service composes only "
            f"{sorted(registry)}. An unrecognised name is refused rather than "
            f"resolved to a default: falling back silently would run a "
            f"different physical model from the one that was asked for, and "
            f"report the fallback as though it had been chosen."
        )
    return registration


def resolve_provider_licence(
    name: str, *, kind: str, policy: LicencePolicy, mode: str
) -> LicenceRecord | None:
    """Clear one provider against the licence policy, or raise.

    Returns the reviewed record for a third-party resource, or ``None`` for a
    provider implemented in this repository. Raises
    :class:`UnknownProviderError` for a name outside the registry, and
    :class:`~thermosphere_consequence.licence.production_gate.ProductionGateError`
    when the mode does not admit the resource — with a message naming the
    resource, its licence, the recorded reason and the permitted substitute
    where one exists.

    This is the only licence decision in the service, and it is delegated: the
    verdict comes from ``check_resource`` rather than from a second copy of the
    mode rules living here.
    """
    registration = _registration(name, kind)
    if registration.licence_resource is None:
        return None
    return check_resource(registration.licence_resource, policy, mode=mode)


#: Said in full when the licence gate leaves the atmosphere slot with nothing
#: composable. Written out rather than assembled from the individual refusals,
#: because three separate "no"s read to a caller as three separate defects,
#: and the fact worth conveying is the single one underneath them.
_NO_ATMOSPHERE_NOTE = (
    "This project is open non-commercial research and has NEVER had a verified "
    "production atmosphere. The absence is the licence position itself, not a "
    "missing feature and not a defect to be worked around: NRLMSISE-00 is held "
    "research-only because the NRL Fortran compiled here carries no licence "
    "text of any kind — an absence of terms, which is not a grant — and the "
    "alternatives are restricted by their own licences. Run in 'research' "
    "mode, which admits the research-only record, or close the verification "
    "gap by filing primary evidence under third_party/licences/ (see "
    "third_party/licences/NRLMSISE-00-PROVENANCE.md and blocker "
    "BLK-NRLMSISE00-ARTEFACT-PROVENANCE)."
)

_NO_PROVIDER_NOTES: Mapping[str, str] = MappingProxyType({_ATMOSPHERE_KIND: _NO_ATMOSPHERE_NOTE})


def _provider_availability(
    kind: str, *, policy: LicencePolicy, mode: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """What this service could build in ``mode`` for ``kind``, and why not for the rest.

    Both lists come from one pass, so the reasons can never end up describing a
    different set from the names they are meant to explain. Two conditions have
    to hold for a name to be composable, and they are different kinds of fact:
    the licence gate must admit the resource AND an implementation must ship
    here. A registered model that clears the gate with no code behind it is not
    an available provider, and offering it as one would leave a caller cycling
    through names that cannot run.
    """
    composable: list[str] = []
    refused: list[str] = []
    for name, registration in _REGISTRIES[kind].items():
        if registration.build is None:
            refused.append(f"{name} (no implementation ships in this repository)")
            continue
        if registration.licence_resource is None:
            composable.append(name)
            continue
        try:
            check_resource(registration.licence_resource, policy, mode=mode)
        except ProductionGateError:
            record = policy.record_for(registration.licence_resource)
            decision = record.policy_decision if record is not None else "undeclared"
            refused.append(f"{name} (policy decision {decision})")
            continue
        composable.append(name)
    return tuple(composable), tuple(refused)


def _no_provider_message(
    refusal: ProductionGateError, *, kind: str, mode: str, refused: tuple[str, ...]
) -> str:
    """The refusal, plus the fact that nothing else in this kind would work either."""
    return (
        f"{refusal} "
        f"NO {kind} provider is available in {mode!r} mode at all, so nothing was composed: "
        f"{'; '.join(refused)}. {_NO_PROVIDER_NOTES.get(kind, '')}"
    ).strip()


def _clearance(
    name: str, *, kind: str, policy: LicencePolicy, mode: str
) -> ProviderClearance:
    registration = _registration(name, kind)
    try:
        record = resolve_provider_licence(name, kind=kind, policy=policy, mode=mode)
    except ProductionGateError as exc:
        composable, refused = _provider_availability(kind, policy=policy, mode=mode)
        if composable:
            raise
        raise ProductionGateError(
            _no_provider_message(exc, kind=kind, mode=mode, refused=refused)
        ) from exc
    if record is None:
        basis = (
            f"{kind} provider {name!r}: no third-party licence resource. "
            f"{registration.in_house_reason}"
        )
    else:
        basis = (
            f"{kind} provider {name!r}: resource {record.resource_name!r} "
            f"({record.spdx_or_licence_name}) cleared in {mode} mode under "
            f"licence policy {policy.policy_version}; policy decision "
            f"{record.policy_decision}."
        )
    return ProviderClearance(
        provider_name=name,
        licence_resource=registration.licence_resource,
        record=record,
        basis=basis,
    )


def _construct(name: str, *, kind: str, config: ServiceConfig) -> Any:
    registration = _registration(name, kind)
    if registration.build is None:
        raise ProviderNotImplementedError(
            f"{kind} provider {name!r} passed the licence gate in "
            f"{config.mode!r} mode but has no implementation in this "
            f"repository. {registration.unavailable_reason} This is an "
            f"engineering refusal, not a licence refusal, and must not be "
            f"reported as one."
        )
    return registration.build(config)


def build_provider_set(config: ServiceConfig) -> ProviderSet:
    """Compose the providers this configuration names, or refuse to compose any.

    Nothing is constructed before every slot has been cleared. Constructing an
    atmosphere provider and only then discovering that the wind provider is
    refused would leave a caller holding half a run.
    """
    policy = load_policy(config)
    if policy.mode(config.mode) is None:
        raise UnknownModeError(
            f"unknown licence gate mode {config.mode!r}; licence policy "
            f"{policy.policy_version} ({policy.source_path}) declares "
            f"{sorted(policy.modes)}. An unrecognised mode is refused rather "
            f"than assumed permissive."
        )

    atmosphere_clearance = _clearance(
        config.atmosphere_provider, kind=_ATMOSPHERE_KIND, policy=policy, mode=config.mode
    )
    wind_clearance = _clearance(
        config.wind_provider, kind=_WIND_KIND, policy=policy, mode=config.mode
    )

    atmosphere = _construct(
        config.atmosphere_provider, kind=_ATMOSPHERE_KIND, config=config
    )
    wind = _construct(config.wind_provider, kind=_WIND_KIND, config=config)

    gate_mode = policy.mode(config.mode)
    caveats: list[str] = [
        ATMOSPHERE_PROVIDERS[config.atmosphere_provider].description,
        WIND_PROVIDERS[config.wind_provider].description,
        config.analysis_mode.licence,
    ]
    if gate_mode is not None:
        caveats.append(f"Licence gate mode {config.mode!r}: {gate_mode.description}")
    registration = WIND_PROVIDERS[config.wind_provider]
    if registration.in_house_reason:
        caveats.append(registration.in_house_reason)

    return ProviderSet(
        atmosphere=atmosphere,
        wind=wind,
        atmosphere_clearance=atmosphere_clearance,
        wind_clearance=wind_clearance,
        mode=config.mode,
        analysis_mode=config.analysis_mode,
        policy_version=policy.policy_version,
        policy_path=policy.source_path,
        caveats=tuple(caveats),
    )
