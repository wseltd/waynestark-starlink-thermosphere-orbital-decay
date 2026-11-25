"""What a run says about itself before it is allowed to say anything else.

The manifest is the run's declared position: which modes it is in, which code
and inputs produced it, and — the field that decides most of the argument —
whether its parameters were fitted on the data it then compares against.

It exists because the claim gate needs a context, and a context assembled ad
hoc at the call site is a context that can be assembled favourably. Building it
once, from a record that also has to carry the git commit and the input hashes,
means the epistemic claim and the reproducibility claim stand or fall together.

The invariant worth stating plainly:
``calibration_status="out_of_sample_validated"`` requires a
:class:`HoldoutRecord`. "Validated" is the strongest word this project can use,
so it is the one that must be impossible to write down without the evidence
attached.

Canonical tickets: P11_12_T030, P11_12_T048.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..domain.temporal import ensure_utc, format_utc
from .claim_gate import CalibrationStatus, ClaimContext, UncertaintyBasis
from .modes import AnalysisMode, DataStatus, ForceClosure, ModelMode, SourceAuthority

__all__ = ["HoldoutRecord", "RunManifest"]

#: A digest, optionally prefixed with its algorithm ("sha256:...").
_DIGEST = re.compile(r"^(?:[a-z0-9_]+:)?[0-9a-f]{32,128}$")

#: A git object name, or the explicit admission that there is not one.
_COMMIT = re.compile(r"^(?:[0-9a-f]{7,40}(?:-dirty)?|uncommitted)$")

ParameterValue = float | int | str | bool


@dataclass(frozen=True, slots=True)
class HoldoutRecord:
    """The evidence that "validated" is entitled to be said.

    Names what was fitted on and what was held back, so a reader can check that
    the two sets are actually disjoint rather than taking the word for it. The
    constructor checks it too.
    """

    description: str
    fitted_on: tuple[str, ...]
    held_out: tuple[str, ...]
    evaluated_utc: _dt.datetime
    metric_name: str
    metric_value: float

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("a holdout record must describe how the split was made")
        object.__setattr__(self, "fitted_on", tuple(self.fitted_on))
        object.__setattr__(self, "held_out", tuple(self.held_out))
        if not self.fitted_on:
            raise ValueError("a holdout record must name the identifiers fitted on")
        if not self.held_out:
            raise ValueError("a holdout record must name the identifiers held out")
        overlap = set(self.fitted_on) & set(self.held_out)
        if overlap:
            raise ValueError(
                "holdout is not out of sample: "
                f"{sorted(overlap)} appear in both fitted_on and held_out"
            )
        if not self.metric_name.strip():
            raise ValueError("a holdout record must name the metric it was scored on")
        object.__setattr__(self, "evaluated_utc", ensure_utc(self.evaluated_utc, "evaluated_utc"))

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the run report."""
        return {
            "description": self.description,
            "fitted_on": list(self.fitted_on),
            "held_out": list(self.held_out),
            "evaluated_utc": format_utc(self.evaluated_utc),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
        }


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The declared identity of one run.

    Mappings are stored behind :class:`types.MappingProxyType` so a frozen
    manifest is frozen all the way down: a caller cannot add an input hash
    after the fact and present it as what the run consumed.
    """

    analysis_mode: AnalysisMode
    model_mode: ModelMode
    data_status: DataStatus
    source_authority: SourceAuthority
    code_version: str
    git_commit: str
    input_hashes: Mapping[str, str]
    parameter_set: Mapping[str, ParameterValue]
    calibration_status: CalibrationStatus
    attribution_ordering_stable: bool | None = None
    uncertainty_basis: UncertaintyBasis = UncertaintyBasis.SCENARIO
    has_probabilistic_model: bool = False
    force_closure: ForceClosure = ForceClosure.EFFECTIVE_SCALAR
    holdout: HoldoutRecord | None = None
    generated_utc: _dt.datetime | None = None
    notes: str = field(default="")

    def __post_init__(self) -> None:
        self._coerce_enums()
        self._check_identity()
        self._check_inputs()
        self._check_calibration()
        stable = self.attribution_ordering_stable
        if stable is not None and not isinstance(stable, bool):
            raise ValueError(
                "attribution_ordering_stable must be True, False or None "
                f"(not established), got {stable!r}"
            )
        if self.generated_utc is not None:
            object.__setattr__(
                self, "generated_utc", ensure_utc(self.generated_utc, "generated_utc")
            )

    # ---- construction checks --------------------------------------------

    def _coerce_enums(self) -> None:
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
                raise ValueError(
                    f"{field_name}: {raw!r} is not a known {enum_type.__name__}; "
                    f"permitted: {[m.value for m in enum_type]}"
                ) from exc

    def _check_identity(self) -> None:
        if not self.code_version.strip():
            raise ValueError("code_version must be recorded; a run without one is not reproducible")
        commit = self.git_commit.strip().lower()
        if not _COMMIT.fullmatch(commit):
            raise ValueError(
                f"git_commit {self.git_commit!r} is not a git object name. Record "
                f"the commit, or the literal 'uncommitted' to admit there is not one"
            )
        object.__setattr__(self, "git_commit", commit)

    def _check_inputs(self) -> None:
        if not self.input_hashes:
            raise ValueError(
                "input_hashes is empty: a run that does not record what it "
                "consumed cannot be reproduced or audited"
            )
        for key, digest in self.input_hashes.items():
            if not str(key).strip():
                raise ValueError("input_hashes contains an empty key")
            if not _DIGEST.fullmatch(str(digest).strip().lower()):
                raise ValueError(
                    f"input_hashes[{key!r}] = {digest!r} is not a hex digest "
                    f"(optionally prefixed 'sha256:')"
                )
        for key, value in self.parameter_set.items():
            if not str(key).strip():
                raise ValueError("parameter_set contains an empty key")
            if not isinstance(value, float | int | str | bool):
                raise ValueError(
                    f"parameter_set[{key!r}] = {value!r} is not a scalar; a "
                    f"parameter set must be flat enough to compare between runs"
                )
        normalised = {str(k): str(v).strip().lower() for k, v in self.input_hashes.items()}
        object.__setattr__(self, "input_hashes", MappingProxyType(normalised))
        object.__setattr__(self, "parameter_set", MappingProxyType(dict(self.parameter_set)))

    def _check_calibration(self) -> None:
        validated = self.calibration_status is CalibrationStatus.OUT_OF_SAMPLE_VALIDATED
        if validated and self.holdout is None:
            raise ValueError(
                "calibration_status 'out_of_sample_validated' requires a holdout "
                "record naming what was fitted on and what was held back. "
                "Without one the agreement is in-sample fit; declare "
                "'calibrated_in_sample' instead"
            )
        if self.holdout is not None and not validated:
            raise ValueError(
                f"a holdout record was supplied but calibration_status is "
                f"{self.calibration_status.value!r}. A manifest must not carry "
                f"holdout evidence it does not claim, nor claim less than it can "
                f"show: use 'out_of_sample_validated'"
            )

    # ---- use -------------------------------------------------------------

    def claim_context(self, *, value_units: str | None = None) -> ClaimContext:
        """The claim-gate context implied by this manifest.

        ``value_units`` names the units of the quantity the text under scrutiny
        is describing, which is what lets the gate tell an inverse ballistic
        coefficient from a ballistic coefficient.
        """
        return ClaimContext(
            analysis_mode=self.analysis_mode,
            model_mode=self.model_mode,
            data_status=self.data_status,
            source_authority=self.source_authority,
            uncertainty_basis=self.uncertainty_basis,
            has_probabilistic_model=self.has_probabilistic_model,
            value_units=value_units,
            attribution_ordering_stable=self.attribution_ordering_stable,
            calibration_status=self.calibration_status,
            force_closure=self.force_closure,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the run report."""
        return {
            "analysis_mode": self.analysis_mode.value,
            "model_mode": self.model_mode.value,
            "data_status": self.data_status.value,
            "source_authority": self.source_authority.value,
            "code_version": self.code_version,
            "git_commit": self.git_commit,
            "input_hashes": dict(self.input_hashes),
            "parameter_set": dict(self.parameter_set),
            "calibration_status": self.calibration_status.value,
            "attribution_ordering_stable": self.attribution_ordering_stable,
            "uncertainty_basis": self.uncertainty_basis.value,
            "has_probabilistic_model": self.has_probabilistic_model,
            "force_closure": self.force_closure.value,
            "holdout": self.holdout.to_dict() if self.holdout else None,
            "generated_utc": format_utc(self.generated_utc) if self.generated_utc else None,
            "notes": self.notes,
        }
