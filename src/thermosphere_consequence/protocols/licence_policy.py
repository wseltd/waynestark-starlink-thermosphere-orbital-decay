"""Structural contract for the register a licence gate interrogates.

The production gate decides what may enter a commercial deliverable, and it
decides it by asking a policy three questions: is this resource declared, what
does this mode admit, and if the answer is no, what may be used instead. This
contract is those three questions and nothing else.

WHAT THE CONTRACT BUYS. A caller that accepts a ``LicencePolicyProvider`` can
be handed the YAML register loaded from ``config/licence_policy.yaml``, a
narrower fixture built for one test, or a future register assembled from a
review system — and the gate logic is untouched. That matters because the gate
is the component whose behaviour must be identical everywhere: a test that
exercised a different gate against a stand-in policy would be testing the
stand-in.

WHAT IT DELIBERATELY DOES NOT DECIDE.
:meth:`LicencePolicyProvider.record_for` returns ``None`` for an undeclared
resource rather than raising. Turning absence into a refusal is the GATE's
decision — and it is fail-closed, in every mode, because the alternative
inverts the burden of proof and lets a component enter the product by nobody
noticing it. Encoding that refusal here would put the policy in a position to
soften it by returning a permissive default, which is exactly the failure the
separation prevents.

Nor does the contract expose a "is this allowed" method. Admission is a
property of a (record, mode) pair evaluated by
:func:`~thermosphere_consequence.licence.production_gate.check_resource`, and a
policy that answered it directly could disagree with the gate about the same
resource.

MODE SEMANTICS ARE DATA; THE PRODUCTION FLOOR IS NOT. A policy declares its own
mode roster through :meth:`LicencePolicyProvider.mode`, and an unrecognised mode
is refused rather than treated as permissive. The one invariant no provider may
express differently is that production admits ``policy_decision == "permitted"``
and nothing else; that is enforced when a policy is loaded, not negotiated
through this interface, because a gate that can be opened by editing the file it
validates is not a gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pathlib

    from ..licence.policy_loader import GateMode
    from ..licence.record import LicenceRecord

__all__ = ["LicencePolicyProvider"]


@runtime_checkable
class LicencePolicyProvider(Protocol):
    """Anything that can answer the three questions a licence gate asks."""

    @property
    def policy_version(self) -> str:
        """The revision of the register.

        Read-only, and declared that way on purpose: a loaded policy is an
        immutable record, and a gate that could restamp the version it cleared
        a resource under would make the clearance untraceable.
        """
        ...

    @property
    def source_path(self) -> pathlib.Path:
        """Where the register was read from.

        Named in every refusal, because "add a record for this resource" is
        only actionable if the operator is told which file to add it to.
        """
        ...

    def record_for(self, resource_name: str) -> LicenceRecord | None:
        """The reviewed record for a resource, or ``None`` if never declared.

        Keyed on the MODEL, dataset or coefficient file — not on the package
        that carries it. ``pymsis`` is MIT and permitted; the NRLMSIS 2.x model
        the same package can reach is not, and a register keyed on distribution
        names could not express the difference.
        """
        ...

    def substitute_for(self, resource_name: str) -> LicenceRecord | None:
        """The permitted replacement recorded for a resource, if any.

        Returns a record that is itself permitted, or ``None``. A gate must
        never be able to answer a refusal by pointing at a second refusal.
        """
        ...

    def mode(self, name: str) -> GateMode | None:
        """The named execution mode, or ``None`` if this policy declares no such mode."""
        ...
