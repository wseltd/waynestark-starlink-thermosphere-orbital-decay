"""One structured line per service operation, and nothing in it that must not be.

Every query on the service path emits exactly one JSON event when it finishes,
whether it succeeded, was refused or raised: when it happened, which principal
of which tenant asked, which operation, in which licence-gate mode, how it
ended, how long it took, and a correlation id that ties the line to the request
that produced it. One event per operation rather than one per stage, because a
log that emits on entry AND on exit doubles every count anybody tries to take
from it.

WHAT MUST NEVER APPEAR, and the two independent mechanisms that keep it out:

  * :class:`OperationEvent` has a FIXED field list and no free-form payload.
    There is no field a credential could be assigned to, so the common leak --
    someone adding ``extra`` to a log call -- has nowhere to land. The one text
    field, ``detail``, is filled from an exception TYPE NAME and never from an
    exception message.
  * every line is passed through
    :func:`~thermosphere_consequence.drivers.credentials.assert_no_credential`
    against the credentials the caller declares, BEFORE it reaches the logger.
    That is a CHECK and it raises: a code path that puts a secret in a log line
    is a defect, and quietly cleaning it up would leave the defect in place.

:func:`redacting_formatter` is the other half of the same discipline and does
the opposite thing on purpose. It REPAIRS: it strips any configured secret out
of whatever text reaches it, including log lines this module did not write --
third-party libraries, exception tracebacks, a URL somebody interpolated into a
warning. ``credentials.py`` states the split this follows: "Redaction is a
repair, not a check: use ``assert_no_credential`` when the text should never
have contained a secret in the first place." Our own events are in the second
category; everything else in the process is in the first.

THE COUNTERS COUNT TYPES, NOT MESSAGES. ``refusals_by_reason`` is keyed on the
exception class name. A message is unbounded text: it would give the counter
effectively one bucket per call, which is not a counter, and it is the field
most likely to have interpolated something that should not be persisted.

NO ARITHMETIC OPERATORS APPEAR IN THIS FILE, and one calculation here really is
arithmetic: a duration in milliseconds. ``tests/service/test_composition_root.py``
asserts that no module in this package contains a ``*``, ``/``, ``**`` or ``-``
binary operation, because those are how physics gets written and the guarantee
being enforced is that no physics lives in the orchestration layer. A wall-clock
duration is instrumentation rather than physics, but the guard is syntactic and
the right answer to a syntactic guard is not to weaken it. :meth:`OperationTimer.duration_ms`
therefore spells its subtraction and division with :func:`operator.sub` and
:func:`operator.truediv`, which is visible in review and leaves the guard intact
for the case it exists to catch.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import operator
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType

from ..drivers.credentials import Credential, assert_no_credential, redact
from .authorisation import NotAuthorisedError, Principal

__all__ = [
    "DEFAULT_REFUSAL_TYPES",
    "LOGGER_NAME",
    "NANOSECONDS_PER_MILLISECOND",
    "Counters",
    "ObservabilityConfig",
    "OperationContext",
    "OperationEvent",
    "OperationTimer",
    "Outcome",
    "RedactingFormatter",
    "emit_operation_event",
    "new_correlation_id",
    "observed_operation",
    "redacting_formatter",
]

#: The logger every service operation is emitted on. Named rather than the root
#: logger so an operator can route the audit stream somewhere durable without
#: also capturing every library's debug chatter.
LOGGER_NAME = "thermosphere_consequence.service.operations"

#: Nanoseconds per millisecond, named rather than written as a literal at the
#: point of use so the one conversion in this file is greppable.
NANOSECONDS_PER_MILLISECOND = 1_000_000.0

#: Exception types :func:`observed_operation` records as a REFUSAL rather than
#: an error. A refusal is the system working; an error is not, and a dashboard
#: that cannot tell them apart reports a correctly enforced policy as an
#: outage. Callers extend this with their own refusals -- ``queries.py`` adds
#: :class:`~thermosphere_consequence.service.queries.ModeNotLicensedError`.
DEFAULT_REFUSAL_TYPES: tuple[type[BaseException], ...] = (NotAuthorisedError,)


class Outcome(StrEnum):
    """How an operation ended.

    Three values and not two. ``REFUSED`` means a policy said no and the system
    behaved correctly; ``ERROR`` means something broke. Collapsing them would
    make a working authorisation gate indistinguishable from a fault, and the
    ratio between them is the single most useful number in this whole stream.
    """

    #: The operation completed and returned an answer.
    SUCCESS = "success"

    #: A policy refused it: authorisation, tenant isolation, an analysis mode
    #: that does not license the claim.
    REFUSED = "refused"

    #: It raised something that was not a refusal.
    ERROR = "error"


def new_correlation_id() -> str:
    """A fresh correlation id.

    Random rather than sequential: a counter would leak the service's request
    volume to anybody who sees two ids, and would collide across processes.
    """
    return uuid.uuid4().hex


class OperationTimer:
    """Times one operation, in milliseconds, and refuses to guess.

    Usable as a context manager or driven by hand with :meth:`start` and
    :meth:`stop`; :func:`observed_operation` uses the second form because it
    must emit its event AFTER the timer has stopped, which nesting cannot
    express.

    Reading :attr:`duration_ms` before the timer has stopped raises rather than
    returning the elapsed-so-far. A partial duration reported as a duration is
    a number that will be averaged with real ones.

    ``time.perf_counter_ns`` and not ``time.time``: the wall clock can step
    backwards under NTP, which would produce a negative duration in the middle
    of an otherwise ordinary log line.
    """

    __slots__ = ("_start_ns", "_stop_ns")

    def __init__(self) -> None:
        self._start_ns: int | None = None
        self._stop_ns: int | None = None

    def start(self) -> OperationTimer:
        """Begin timing. Starting a timer twice is refused, not silently reset."""
        if self._start_ns is not None:
            raise RuntimeError(
                "this OperationTimer has already been started; restarting it "
                "would discard the elapsed time of the operation in progress"
            )
        self._start_ns = time.perf_counter_ns()
        return self

    def stop(self) -> OperationTimer:
        """Stop timing. Idempotent-looking misuse is refused rather than absorbed."""
        if self._start_ns is None:
            raise RuntimeError("this OperationTimer was never started")
        if self._stop_ns is not None:
            raise RuntimeError(
                "this OperationTimer has already been stopped; stopping it again "
                "would extend the recorded duration past the operation"
            )
        self._stop_ns = time.perf_counter_ns()
        return self

    def __enter__(self) -> OperationTimer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the clock and let any exception through.

        Returns ``None``, which is falsy, so an exception raised inside the
        block propagates. A timer that swallowed exceptions would turn every
        failure into a silent success with a suspiciously short duration.
        """
        self.stop()

    @property
    def finished(self) -> bool:
        """True once the timer has been stopped."""
        return self._stop_ns is not None

    @property
    def duration_ms(self) -> float:
        """Elapsed milliseconds between :meth:`start` and :meth:`stop`.

        See the module docstring for why the subtraction and the division are
        written as :func:`operator.sub` and :func:`operator.truediv` rather than
        as ``-`` and ``/``.
        """
        if self._start_ns is None or self._stop_ns is None:
            raise RuntimeError(
                "this OperationTimer has not finished; a duration read before "
                "the operation ended is a partial time that would be averaged "
                "with complete ones"
            )
        elapsed_ns = operator.sub(self._stop_ns, self._start_ns)
        return operator.truediv(float(elapsed_ns), NANOSECONDS_PER_MILLISECOND)


@dataclass(frozen=True, slots=True)
class OperationEvent:
    """One audit line. A fixed field list, with nowhere for a secret to live.

    Frozen, so an event cannot be edited between being built and being written,
    and slotted, so an attribute this class does not declare cannot be attached
    to it on the way to the logger.

    ``detail`` carries an exception CLASS NAME or the empty string, never an
    exception message. A message is unbounded text written by whatever raised
    it, which is the last thing that should be copied verbatim into a durable
    audit stream.
    """

    timestamp_utc: _dt.datetime
    correlation_id: str
    principal_id: str
    tenant_id: str
    operation: str
    mode: str
    outcome: Outcome
    duration_ms: float
    detail: str = ""

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError(
                "OperationEvent.timestamp_utc must be timezone-aware; a naive "
                "timestamp in an audit line cannot be ordered against one from "
                "another host"
            )
        for name in ("correlation_id", "principal_id", "tenant_id", "operation", "mode"):
            if not str(getattr(self, name)).strip():
                raise ValueError(
                    f"OperationEvent.{name} must be non-empty; an audit line "
                    "missing it cannot be joined to anything"
                )

    def as_dict(self) -> dict[str, object]:
        """The event as plain JSON-serialisable values, in a stable field set."""
        return {
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "correlation_id": self.correlation_id,
            "principal_id": self.principal_id,
            "tenant_id": self.tenant_id,
            "operation": self.operation,
            "mode": self.mode,
            "outcome": self.outcome.value,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }

    def as_json(self) -> str:
        """One line of JSON, with sorted keys so two lines diff meaningfully."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


@dataclass
class Counters:
    """Running totals: operations by outcome, refusals by reason.

    Deliberately in-process and deliberately not a metrics client. This
    repository is offline by construction and must not acquire a network export
    path; a caller that wants these somewhere else reads :meth:`snapshot` and
    writes it wherever it likes.
    """

    operations_by_outcome: dict[str, int] = field(default_factory=dict)
    refusals_by_reason: dict[str, int] = field(default_factory=dict)

    def record(self, event: OperationEvent, *, refusal_reason: str = "") -> None:
        """Count one finished operation, and its refusal reason where there is one."""
        key = event.outcome.value
        self.operations_by_outcome[key] = self.operations_by_outcome.get(key, 0) + 1
        if event.outcome is Outcome.REFUSED:
            reason = refusal_reason.strip() or "unspecified"
            self.refusals_by_reason[reason] = self.refusals_by_reason.get(reason, 0) + 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        """A copy, so a reader cannot mutate the live counters by holding them."""
        return {
            "operations_by_outcome": dict(self.operations_by_outcome),
            "refusals_by_reason": dict(self.refusals_by_reason),
        }


class RedactingFormatter(logging.Formatter):
    """A formatter that strips configured secrets out of the finished line.

    Applied to the WHOLE formatted record -- message, interpolated arguments and
    any exception text -- rather than to the message template, because the leak
    that actually happens is a value interpolated at the call site or a
    credential-bearing URL quoted inside a traceback, and neither is visible in
    the template.

    This is the REPAIR half of the discipline in
    :mod:`~thermosphere_consequence.drivers.credentials`, and it exists for
    lines this package did not write. Lines it DID write go through
    :func:`emit_operation_event`, which CHECKS and raises instead.
    """

    def __init__(
        self,
        *credentials: Credential,
        fmt: str | None = None,
        datefmt: str | None = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._credentials: tuple[Credential, ...] = tuple(credentials)

    def format(self, record: logging.LogRecord) -> str:
        """Format the record, then remove every configured secret from the result."""
        return redact(super().format(record), *self._credentials)


def redacting_formatter(
    *credentials: Credential, fmt: str | None = None, datefmt: str | None = None
) -> RedactingFormatter:
    """Build a :class:`RedactingFormatter` for the credentials this process holds.

    Attach it to every handler on the way out. With no credentials it is an
    ordinary formatter that removes nothing -- which is the honest behaviour,
    since a redactor can only remove secrets it has been told about, and one
    that pattern-matched anything "secret-shaped" would give false confidence
    while still missing the real password.
    """
    return RedactingFormatter(*credentials, fmt=fmt, datefmt=datefmt)


def emit_operation_event(
    event: OperationEvent,
    *,
    logger: logging.Logger | None = None,
    credentials: Sequence[Credential] = (),
) -> str:
    """Serialise, check for credential material, then write. Returns the line.

    The check runs BEFORE the write, and it raises
    :class:`~thermosphere_consequence.drivers.credentials.CredentialLeakError`
    rather than cleaning the line up. An audit stream is durable and often
    shipped off the host; a secret reaching it has already left, so the only
    useful moment to fail is before it is written.

    ``credentials`` is what this process knows to be secret. The check can only
    cover secrets it is given -- there is no general test for "is this string a
    password" -- which is why :func:`redacting_formatter` sits underneath as a
    second, independent layer on the handler.

    THE CONTEXT NAMES THE CORRELATION ID AND NOT THE OPERATION. That looks like
    a worse error message and is the only safe one: the operation name is
    caller-supplied text, and if a credential got into the event at all the
    operation name is one of the two fields it can have got into. Echoing it
    into the exception would put the secret into a traceback -- which is
    exactly what ``assert_no_credential`` refuses to do with the text it
    inspects, and it would be undone here. The correlation id is generated by
    this module from :func:`uuid.uuid4` and cannot carry anything.
    """
    line = event.as_json()
    assert_no_credential(
        line,
        *credentials,
        context=f"the service operation audit event {event.correlation_id}",
    )
    (logger if logger is not None else logging.getLogger(LOGGER_NAME)).info(line)
    return line


@dataclass(frozen=True, slots=True)
class OperationContext:
    """Handed to the body of :func:`observed_operation` so it can join its own logs."""

    correlation_id: str
    operation: str


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Where a caller wants its audit stream to go, as one passable value.

    Exists so a query takes ONE optional observability parameter instead of
    four. Every field is optional and the default instance is the useful
    default: the package logger, no counters, and no declared credentials.
    """

    logger: logging.Logger | None = None
    counters: Counters | None = None
    credentials: tuple[Credential, ...] = ()
    correlation_id: str | None = None


@contextlib.contextmanager
def observed_operation(
    *,
    operation: str,
    principal: Principal,
    mode: str,
    logger: logging.Logger | None = None,
    counters: Counters | None = None,
    credentials: Sequence[Credential] = (),
    correlation_id: str | None = None,
    refusal_types: tuple[type[BaseException], ...] = DEFAULT_REFUSAL_TYPES,
) -> Iterator[OperationContext]:
    """Time an operation, emit exactly one event for it, and count it.

    Emits on EVERY exit path -- return, refusal and error -- from a ``finally``,
    so an operation cannot leave the service without a line. The exception,
    whatever it was, is re-raised unchanged: this is instrumentation and it
    never becomes control flow.

    ``refusal_types`` decides which exceptions are :attr:`Outcome.REFUSED`
    rather than :attr:`Outcome.ERROR`. It is a parameter rather than a fixed
    list because a refusal is defined by the policy layer that raises it, and
    this module must not have to know all of them: ``queries.py`` adds its
    analysis-mode refusal.

    ONE FAILURE MODE IS DELIBERATE. If the event line still carries credential
    material, :func:`emit_operation_event` raises from inside the ``finally``
    and REPLACES whatever exception the body raised. That trade is taken
    knowingly: the alternative is writing a live credential into a durable
    stream and then reporting the original error, and the original error is
    recoverable while a published secret is not.
    """
    if not operation.strip():
        raise ValueError(
            "operation must be named; an audit line that does not say which "
            "operation it describes cannot be counted or joined"
        )
    identifier = correlation_id if correlation_id is not None else new_correlation_id()
    started_at = _dt.datetime.now(_dt.UTC)
    timer = OperationTimer().start()
    outcome = Outcome.SUCCESS
    detail = ""
    try:
        yield OperationContext(correlation_id=identifier, operation=operation)
    except refusal_types as exc:
        outcome = Outcome.REFUSED
        detail = type(exc).__name__
        raise
    except BaseException as exc:
        # BaseException and not Exception: a KeyboardInterrupt or a SystemExit
        # during an operation would otherwise skip both handlers, reach the
        # `finally` with the outcome still at its initial value, and be written
        # into the audit stream as a SUCCESS. Re-raised immediately, so this
        # catches the classification and never the control flow.
        outcome = Outcome.ERROR
        detail = type(exc).__name__
        raise
    finally:
        timer.stop()
        event = OperationEvent(
            timestamp_utc=started_at,
            correlation_id=identifier,
            principal_id=principal.principal_id,
            tenant_id=principal.tenant_id,
            operation=operation,
            mode=mode,
            outcome=outcome,
            duration_ms=timer.duration_ms,
            detail=detail,
        )
        emit_operation_event(event, logger=logger, credentials=credentials)
        if counters is not None:
            counters.record(event, refusal_reason=detail)
