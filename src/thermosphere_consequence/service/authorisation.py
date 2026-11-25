"""Who may perform which operation, per tenant, and why a refusal happened.

This sits BELOW the HTTP adapter and answers a different question from it.
:mod:`thermosphere_consequence.api.app` asks "is this request from someone who
holds the deployment's API key?" -- one shared secret, one bit of information,
and every caller who has it is the same caller. This module asks "may THIS
principal perform THIS operation in THIS mode, on data belonging to THIS
tenant?" Neither substitutes for the other, and this one is deliberately not
reachable from the wire: it is enforced on the service path, so the CLI, a
notebook and a batch job go through the same gate the API does rather than
through a gate that only exists in one adapter.

FAIL CLOSED, IN FOUR PLACES. Each of these is the safe answer to a question the
system cannot answer, and each is separately tested:

**An unknown role grants nothing.** ``Principal.roles`` holds plain strings
rather than :class:`Role` members precisely so an unrecognised role CAN be
constructed and must then be handled -- a typo, a role from a newer deployment,
a role that was renamed. It contributes no permissions, and the refusal says
that none of the principal's roles is one this service defines. The tempting
alternative, resolving an unrecognised role to the least-privileged one, turns
a deployment mismatch into a silent partial grant.

**An empty role set grants nothing.** Authorisation here is a whitelist. A
principal that was never given a role is refused rather than defaulted, because
the state "we have not decided what this principal may do" must not read as
"anything a reader may do".

**A principal without production rights is refused in production mode**, even
when its role would allow the operation. Production is where a result gets
published, and the licence gate has already narrowed what may be composed
there; this narrows who may ask.

**An unrecognised MODE is treated as production.** This module does not keep
its own list of modes -- :mod:`~thermosphere_consequence.service.composition_root`
refuses an unknown mode where the mode is chosen, and a second table here would
drift from the licence policy. What it keeps instead is the list of modes that
do NOT require production rights (:data:`NON_PRODUCTION_MODES`), so anything
else, including a typo, lands on the restrictive side. Had it been written the
other way round -- "special-case the string 'production'" -- a mode spelled
``"Production"`` would have skipped the check entirely.

TENANT ISOLATION IS SEPARATE FROM PERMISSION. Holding
:attr:`Permission.READ_PROVENANCE` says a principal may read run records; it
does not say WHOSE. :func:`authorise_tenant` answers the second question, and
it is a distinct call rather than an extra argument to :func:`authorise` so
that a resource with an owner cannot be served by a code path that only ever
checked the permission.

NOTHING HERE AUTHENTICATES. A :class:`Principal` is an assertion that some
layer above has already established an identity. This module decides what that
identity may do; it has no idea whether it is genuine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

__all__ = [
    "NON_PRODUCTION_MODES",
    "PRODUCTION_MODE",
    "ROLE_PERMISSIONS",
    "UNTENANTED_RESOURCE_IS_SHARED",
    "NotAuthorisedError",
    "Permission",
    "Principal",
    "Role",
    "TenantIsolationError",
    "authorise",
    "authorise_tenant",
    "granted_permissions",
    "unknown_roles",
]

#: The licence-gate mode in which results may be published, and the one this
#: module treats as privileged. The string matches ``config/licence_policy.yaml``
#: and :class:`~thermosphere_consequence.service.composition_root.ServiceConfig`.
PRODUCTION_MODE = "production"

#: Modes that do NOT require a principal to hold production rights. Stated as
#: the permissive list rather than the restrictive one, so that any mode this
#: module has never heard of -- a typo, a mode added to the licence policy
#: after this file was written -- requires production rights and is refused for
#: a principal that lacks them. See the module docstring.
NON_PRODUCTION_MODES: frozenset[str] = frozenset({"research"})

UNTENANTED_RESOURCE_IS_SHARED = (
    "This resource names no owning tenant, so it is treated as a SHARED "
    "artefact of the repository and is readable by any principal holding the "
    "permission. That is a real hole with a real cost: a record that SHOULD "
    "belong to one tenant and was written without a tenant_id is readable by "
    "all of them. It is stated on the answer rather than hidden because the "
    "alternative -- refusing every untenanted record -- would make the "
    "repository's own committed evidence unreadable, and an operator would "
    "then remove the check rather than stamp the records."
)


class Permission(StrEnum):
    """One operation the service layer can be asked to perform.

    Keyed on the OPERATION and not on the object, because that is the grain at
    which the answer differs: reading a modelled density and reading the
    catalogued decay dates are both reads, and they are not the same claim.
    """

    #: Evaluate the composed atmosphere provider at a point.
    READ_ATMOSPHERE = "read_atmosphere"

    #: Compute a composition-weighted drag coefficient and C_d*A/m.
    READ_DRAG = "read_drag"

    #: Read the validated event manifest.
    READ_EVENT = "read_event"

    #: Read the catalogued decay dates. The only observational anchor in the
    #: chain, and a claim about what specific real objects did.
    READ_GROUND_TRUTH = "read_ground_truth"

    #: Read the stored record of a previous analysis run.
    READ_PROVENANCE = "read_provenance"

    #: Propagate an orbit.
    RUN_PROPAGATION = "run_propagation"

    #: Re-run a full reproduction, which regenerates committed artefacts.
    RUN_REPRODUCTION = "run_reproduction"

    #: Ask what the licence policy admits, without composing anything.
    INSPECT_LICENCE = "inspect_licence"


class Role(StrEnum):
    """A named bundle of permissions.

    Three of them, and the boundaries are where the CLAIM changes rather than
    where the code gets more expensive:

    ``reader`` may see what the models say. Every permission it holds returns a
    MODELLED or DERIVED answer, so the worst a reader can do with the output is
    quote a simulation as a simulation.

    ``analyst`` additionally may read the observational anchor, read previous
    run records and propagate orbits. The step up is not compute, it is that
    the catalogued decay dates are a statement about specific real objects and
    a run record is a statement about what this repository has already
    published.

    ``operator`` additionally may run a reproduction, which REGENERATES
    committed artefacts. That is the only permission here that changes what the
    repository asserts, which is why it sits alone at the top.
    """

    READER = "reader"
    ANALYST = "analyst"
    OPERATOR = "operator"

    @property
    def permissions(self) -> frozenset[Permission]:
        """Everything this role grants, including what it inherits."""
        return ROLE_PERMISSIONS[self]


_READER_PERMISSIONS: frozenset[Permission] = frozenset(
    {
        Permission.READ_ATMOSPHERE,
        Permission.READ_DRAG,
        Permission.READ_EVENT,
        Permission.INSPECT_LICENCE,
    }
)

_ANALYST_PERMISSIONS: frozenset[Permission] = _READER_PERMISSIONS.union(
    {
        Permission.READ_GROUND_TRUTH,
        Permission.READ_PROVENANCE,
        Permission.RUN_PROPAGATION,
    }
)

_OPERATOR_PERMISSIONS: frozenset[Permission] = _ANALYST_PERMISSIONS.union(
    {Permission.RUN_REPRODUCTION}
)

#: The closed table. Read-only at runtime so a caller cannot widen a role in
#: place and have every principal holding it silently gain the permission.
ROLE_PERMISSIONS: Mapping[Role, frozenset[Permission]] = MappingProxyType(
    {
        Role.READER: _READER_PERMISSIONS,
        Role.ANALYST: _ANALYST_PERMISSIONS,
        Role.OPERATOR: _OPERATOR_PERMISSIONS,
    }
)

_ROLE_BY_NAME: Mapping[str, Role] = MappingProxyType({role.value: role for role in Role})


class NotAuthorisedError(RuntimeError):
    """Raised when a principal may not perform the operation it asked for.

    Deliberately NOT a subclass of :class:`PermissionError`, which is an
    :class:`OSError`: a broad ``except OSError`` around file handling is common
    and would swallow an authorisation refusal as though it were a failed read.
    """


class TenantIsolationError(NotAuthorisedError):
    """Raised when a principal asks for a resource belonging to another tenant.

    A subclass so a caller that only cares about "refused" catches both, while
    a caller that must distinguish "you may not do this at all" from "you may,
    but not to this" can. The two are different operational incidents.
    """


@dataclass(frozen=True, slots=True)
class Principal:
    """An identity some layer above has already established, and what it may do.

    ``roles`` is a frozenset of plain STRINGS, not of :class:`Role` members,
    and that is the point rather than an omission: a role this service does not
    define must be REPRESENTABLE so that it can be explicitly refused. Typing
    the field as ``frozenset[Role]`` would move the failure to construction
    time in this process and leave a deployment that sends an unknown role with
    no defined behaviour at all.

    ``may_operate_in_production`` defaults to ``False``. The default is the
    fail-closed one: a principal built without deciding this question does not
    get production rights by omission.
    """

    principal_id: str
    tenant_id: str
    roles: frozenset[str]
    may_operate_in_production: bool = False

    def __post_init__(self) -> None:
        for field_name in ("principal_id", "tenant_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string; an anonymous "
                    "principal cannot be named in a refusal or in an audit line"
                )
        if not isinstance(self.roles, frozenset):
            raise TypeError(
                f"roles must be a frozenset, got {type(self.roles).__name__}. A "
                "mutable role collection on a frozen principal could be widened "
                "after the principal was authorised, so the check would have "
                "been made against a set that no longer exists."
            )
        for role in self.roles:
            if not isinstance(role, str) or not role.strip():
                raise ValueError(
                    "every role must be a non-empty string; a blank role is "
                    "indistinguishable from a missing one and grants nothing "
                    "either way, so it is refused where it is written rather "
                    "than where it fails"
                )

    @property
    def has_no_roles(self) -> bool:
        """True when this principal was never given a role at all."""
        return not self.roles


def granted_permissions(principal: Principal) -> frozenset[Permission]:
    """Everything ``principal`` may do, from the roles this service recognises.

    A role outside :class:`Role` contributes NOTHING and is skipped in silence
    here; :func:`authorise` names it in the refusal, which is where a caller
    can act on it. Returning a partial grant for an unrecognised role, or
    raising here, would both be worse: the first is a silent privilege
    decision, the second would stop a principal that holds one good role and
    one stale one from doing anything.
    """
    granted: set[Permission] = set()
    for name in principal.roles:
        role = _ROLE_BY_NAME.get(name)
        if role is not None:
            granted.update(ROLE_PERMISSIONS[role])
    return frozenset(granted)


def unknown_roles(principal: Principal) -> frozenset[str]:
    """The roles this principal holds that this service does not define."""
    return frozenset(name for name in principal.roles if name not in _ROLE_BY_NAME)


def _roles_granting(permission: Permission) -> list[str]:
    """Every defined role that would grant ``permission``, for the refusal text."""
    return sorted(
        role.value for role, allowed in ROLE_PERMISSIONS.items() if permission in allowed
    )


def _requires_production_rights(mode: str) -> bool:
    """Whether ``mode`` demands a principal with production rights.

    Anything not explicitly listed in :data:`NON_PRODUCTION_MODES` does. See
    the module docstring for why the list runs this way round.
    """
    return mode not in NON_PRODUCTION_MODES


def authorise(principal: Principal, permission: Permission, *, mode: str) -> None:
    """Refuse unless ``principal`` may perform ``permission`` in ``mode``.

    Returns ``None`` on success and raises :class:`NotAuthorisedError`
    otherwise, rather than returning a boolean. A boolean can be ignored by
    accident and a missing ``if`` is invisible in review; an exception cannot be
    dropped without writing something that says it was dropped.

    THE ORDER OF THE TWO CHECKS IS DELIBERATE. The role check runs first, so a
    principal that lacks the role is told which role it needs -- the actionable
    answer -- rather than being told it lacks production rights it would not
    have been able to use anyway. A principal whose role DOES grant the
    operation then meets the production check, which is the case the
    fail-closed rule exists for: the mode refuses it even though the role
    allowed it.
    """
    if not isinstance(permission, Permission):
        raise TypeError(
            f"permission must be a Permission, got {type(permission).__name__}. A "
            "bare string would let a permission this service does not define "
            "pass a membership test against a role's permission set."
        )
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(
            "mode must be a non-empty string naming the licence-gate mode this "
            "operation runs in -- ProviderSet.mode carries it. An unstated mode "
            "cannot be checked for production rights."
        )

    granted = granted_permissions(principal)
    if permission not in granted:
        held = sorted(principal.roles)
        detail = (
            f"principal {principal.principal_id!r} of tenant "
            f"{principal.tenant_id!r} may not perform {permission.value!r}. It "
            f"holds roles {held}; that permission is granted by role(s) "
            f"{_roles_granting(permission)}."
        )
        if principal.has_no_roles:
            raise NotAuthorisedError(
                f"{detail} It holds NO roles at all. An empty role set grants "
                "nothing: authorisation here is a whitelist, so a principal "
                "that was never given a role is refused rather than defaulted "
                "to the least-privileged one."
            )
        unrecognised = unknown_roles(principal)
        if unrecognised == principal.roles:
            raise NotAuthorisedError(
                f"{detail} None of the roles it holds is defined by this "
                f"service, which defines {sorted(_ROLE_BY_NAME)}. An unknown "
                "role grants NOTHING and is never resolved to a default: doing "
                "so would turn a renamed role or a typo into a silent partial "
                "grant."
            )
        raise NotAuthorisedError(detail)

    if _requires_production_rights(mode) and not principal.may_operate_in_production:
        raise NotAuthorisedError(
            f"principal {principal.principal_id!r} of tenant "
            f"{principal.tenant_id!r} holds a role granting {permission.value!r} "
            f"but is not permitted to operate in mode {mode!r}, which requires "
            f"production rights. Production is where a result may be published, "
            f"so the role is not enough on its own. Modes that do not require "
            f"production rights: {sorted(NON_PRODUCTION_MODES)}; any other mode, "
            f"including one this service does not recognise, requires them."
        )


def authorise_tenant(
    principal: Principal, *, resource_tenant_id: str | None, resource_description: str
) -> None:
    """Refuse unless ``principal`` belongs to the tenant that owns the resource.

    A SEPARATE call from :func:`authorise` on purpose. Holding
    :attr:`Permission.READ_PROVENANCE` says a principal may read run records;
    it does not say whose. Folding the owner check into the permission check
    would let a resource with an owner be served by any path that remembered to
    check the permission -- and every path remembers to check the permission,
    which is exactly why that would not be noticed.

    ``resource_tenant_id`` of ``None`` means the resource names no owner. It is
    served, and :data:`UNTENANTED_RESOURCE_IS_SHARED` says what that costs; see
    that constant for why the alternative was rejected.

    A BLANK owner is not the same as no owner and is refused. A record carrying
    ``tenant_id: ""`` was written by something that meant to name an owner and
    failed, and serving it to whoever asks is the failure mode this whole
    function exists to prevent.
    """
    if not resource_description.strip():
        raise ValueError(
            "resource_description must name what is being protected; a refusal "
            "that does not say which resource was refused is unactionable"
        )
    if resource_tenant_id is None:
        return
    if not isinstance(resource_tenant_id, str) or not resource_tenant_id.strip():
        raise TenantIsolationError(
            f"{resource_description} names an owning tenant that is not a "
            f"non-empty string (it is {type(resource_tenant_id).__name__}, "
            f"blank or otherwise unusable). That is not the same as carrying "
            f"none: it was written by something that meant to name an owner and "
            f"did not, so it is refused rather than treated as shared. Principal "
            f"{principal.principal_id!r} of tenant {principal.tenant_id!r} is "
            f"not served it."
        )
    if resource_tenant_id != principal.tenant_id:
        raise TenantIsolationError(
            f"principal {principal.principal_id!r} belongs to tenant "
            f"{principal.tenant_id!r} and {resource_description} belongs to "
            f"tenant {resource_tenant_id!r}. A principal is served only its own "
            f"tenant's records; no role grants access across tenants, because a "
            f"role says what an operation is and a tenant says whose data it is."
        )
