"""Audit of the drag-parameter calibration: separating fitting from validating.

WHAT THIS MODULE EXISTS TO CORRECT.

The inversion in ``scenarios.ballistic_inversion`` takes an observed decay time
and returns the inverse ballistic coefficient (C_d*A/m, m^2/kg) that reproduces
it under a given atmosphere. Run once per catalogued object, that is a
CALIBRATION: one free parameter is fitted to each observation. The model is
then saturated — ten parameters for ten observations, zero residual degrees of
freedom — so the residual is identically zero by construction and reporting
that agreement as "validation" is circular. It measures nothing except that a
monotone curve can be inverted.

The vocabulary this module uses, and does not interchange:

  CALIBRATION            estimating a parameter from data.
  IN-SAMPLE FIT          residuals against the data used to estimate it.
  OUT-OF-SAMPLE          residuals against data withheld from the estimation.
  VALIDATION
  COUNTERFACTUAL         re-running the same fitted model under a changed input
  SIMULATION             (here: a quiet atmosphere) to attribute an effect. It
                         produces no residual and cannot validate anything.
  EXTERNAL VALIDATION    comparison against an independent observation that
                         entered no part of the estimation. This module does
                         not perform one; nothing in the repository does.

WHAT AN HONEST HOLD-OUT REQUIRES. A per-object parameter cannot be held out:
removing the object removes the only datum constraining its own parameter, and
the fold is undefined. A hold-out therefore requires a SHARED parameter — one
inverse ballistic coefficient for the whole fleet — which is also the only form
in which the fit has any degrees of freedom to spend. That is what
``fit_shared_inverse_ballistic_coefficient`` estimates and what
``leave_one_out_fit`` and ``train_test_split_fit`` evaluate.

A consequence worth stating in advance rather than discovering in the numbers:
every object here is propagated from the SAME published insertion state through
the SAME atmosphere, so a shared-parameter model predicts ONE decay time for
all of them. Its out-of-sample prediction is therefore mathematically identical
to the training-set mean, and its skill against a constant-mean baseline is
exactly zero. The physics adds no per-object information because no per-object
information was ever supplied to it.

IDENTIFIABILITY. Drag acceleration contains density and the inverse ballistic
coefficient only through the product ``B_inv * rho``. Scaling density by k and
dividing B_inv by k leaves the equations of motion unchanged, hence leaves the
decay time unchanged. ``identifiability_profile`` demonstrates this numerically
and reports ``k * B_inv`` as the quantity the data actually constrain.

TERMINOLOGY. ``inverse_ballistic_coefficient_m2_kg`` is C_d*A/m in m^2/kg. It
is NOT the ballistic coefficient; the conventional B = m/(C_d*A) in kg/m^2 is
its reciprocal. See ``domain.drag_parameters``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.optimize import minimize_scalar

from ..scenarios.ballistic_inversion import DecayPoint

__all__ = [
    "CalibrationVerdict",
    "CurveCoverageError",
    "DEFAULT_LOG_XATOL",
    "DEFAULT_QUANTISATION_HALF_WIDTH_DAYS",
    "DensityDegeneracyPoint",
    "IdentifiabilityProfile",
    "LeaveOneOutRecord",
    "LeaveOneOutResult",
    "MATERIAL_DEGRADATION_FRACTION",
    "NumericSensitivity",
    "NumericVariant",
    "NumericVariantFit",
    "ObjectiveProfile",
    "ObservedDecay",
    "QuantisationEnvelope",
    "REENTRY_THRESHOLD_VARIANTS_KM",
    "ResidualStats",
    "SOLVER_RTOL_VARIANTS",
    "SharedFit",
    "SplitFitResult",
    "TRUE_ANOMALY_VARIANTS_DEG",
    "calibration_verdict",
    "decay_date_quantisation_envelope",
    "fit_shared_inverse_ballistic_coefficient",
    "identifiability_profile",
    "leave_one_out_fit",
    "objective_profile",
    "predicted_decay_days",
    "scale_curve_density",
    "sensitivity_to_numerics",
    "train_test_split_fit",
]

# Absolute tolerance on ln(B_inv) for the bounded scalar minimisation. At
# B_inv ~ 0.02 m^2/kg this is a parameter resolution of ~2e-12 m^2/kg, five
# orders finer than any uncertainty in the problem, so the optimiser never
# contributes to the reported spread.
DEFAULT_LOG_XATOL = 1e-10

# Catalogued decay dates carry calendar-day resolution, so the true decay
# instant lies anywhere within the recorded day: +/- half a day about its
# midpoint.
DEFAULT_QUANTISATION_HALF_WIDTH_DAYS = 0.5

# Out-of-sample error is called materially worse than in-sample above this
# fractional degradation. Stated as a constant so the threshold is auditable
# rather than buried in a comparison.
MATERIAL_DEGRADATION_FRACTION = 0.10

# The numerical settings the audit is required to vary. Producing a decay curve
# under each is the caller's job (it costs a propagation sweep); this module
# consumes the resulting curves. The constants live here so the protocol is
# stated once and the script cannot silently audit a different set.
REENTRY_THRESHOLD_VARIANTS_KM: tuple[float, ...] = (80.0, 100.0, 120.0)
SOLVER_RTOL_VARIANTS: tuple[float, ...] = (1e-7, 1e-9, 1e-11)
TRUE_ANOMALY_VARIANTS_DEG: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

# scipy's bounded Brent method takes no user initial guess: its first probe is
# the golden-section point of the bracket. Recorded so "what was the initial
# guess?" has a concrete answer instead of "none".
_GOLDEN_SECTION_FRACTION = 0.3819660112501051


class CurveCoverageError(ValueError):
    """Raised when a decay curve cannot bracket the quantity being fitted."""


@dataclass(frozen=True, slots=True)
class ObservedDecay:
    """One catalogued object and the decay time attributed to it.

    ``label`` must be unique within a set: it identifies the object in every
    hold-out record and in the deterministic train/test split.
    """

    label: str
    observed_decay_days: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("observed decay requires a non-empty label")
        if self.observed_decay_days <= 0:
            raise ValueError(
                f"observed decay must be positive, got {self.observed_decay_days} "
                f"days for {self.label}"
            )


@dataclass(frozen=True, slots=True)
class _MonotoneCurve:
    """A decay curve reduced to two aligned float arrays, validated once.

    Carrying the pair separately rather than re-filtering ``DecayPoint`` at
    every call keeps the optional ``decay_days`` out of the arithmetic, so a
    non-reentering point cannot reach a logarithm as a ``None``.
    """

    inverse_ballistic_coefficients_m2_kg: tuple[float, ...]
    decay_days: tuple[float, ...]

    @property
    def b_bounds(self) -> tuple[float, float]:
        return (
            self.inverse_ballistic_coefficients_m2_kg[0],
            self.inverse_ballistic_coefficients_m2_kg[-1],
        )

    @property
    def decay_bounds(self) -> tuple[float, float]:
        """Shortest and longest decay time on the curve, in days."""
        return (self.decay_days[-1], self.decay_days[0])


def _prepare(curve: Sequence[DecayPoint]) -> _MonotoneCurve:
    """Filter, sort and validate a decay curve for use as an interpolant.

    The fit is only well posed when decay time is a strictly decreasing
    function of the inverse ballistic coefficient, because that is what makes
    the least-squares minimiser unique and interior. A curve that violates it
    is rejected rather than quietly interpolated, since a non-monotone decay
    curve is evidence of a propagation problem, not a modelling choice.
    """
    usable = sorted(
        (p for p in curve if p.reentered and p.decay_days is not None),
        key=lambda p: p.inverse_ballistic_coefficient,
    )
    if len(usable) < 2:
        raise CurveCoverageError(
            f"decay curve needs at least two reentering points, got {len(usable)}"
        )
    b_values: list[float] = []
    d_values: list[float] = []
    for point in usable:
        decay = point.decay_days
        if decay is None:  # pragma: no cover - excluded by the filter above
            raise CurveCoverageError("decay curve contains a non-reentering point")
        if point.inverse_ballistic_coefficient <= 0:
            raise CurveCoverageError(
                "inverse ballistic coefficient must be positive, got "
                f"{point.inverse_ballistic_coefficient}"
            )
        if decay <= 0:
            raise CurveCoverageError(f"decay time must be positive, got {decay}")
        b_values.append(point.inverse_ballistic_coefficient)
        d_values.append(decay)

    for i in range(1, len(b_values)):
        if b_values[i] <= b_values[i - 1]:
            raise CurveCoverageError(
                "decay curve has a repeated inverse ballistic coefficient at "
                f"{b_values[i]} m^2/kg"
            )
        if d_values[i] >= d_values[i - 1]:
            raise CurveCoverageError(
                "decay time must decrease strictly with the inverse ballistic "
                f"coefficient; {d_values[i]} d at {b_values[i]} m^2/kg does not "
                f"fall below {d_values[i - 1]} d at {b_values[i - 1]} m^2/kg"
            )
    return _MonotoneCurve(tuple(b_values), tuple(d_values))


def predicted_decay_days(
    curve: Sequence[DecayPoint], inverse_ballistic_coefficient_m2_kg: float
) -> float:
    """Decay time the curve implies at one inverse ballistic coefficient.

    Log-log interpolation between bracketing forward runs, which is the same
    estimator ``ballistic_inversion.invert_decay`` uses in the other direction.
    Using the same interpolant matters: the audit must measure the estimator
    the project actually reported, not a better one introduced for the audit.
    """
    table = _prepare(curve)
    b_lo, b_hi = table.b_bounds
    b = inverse_ballistic_coefficient_m2_kg
    if not b_lo <= b <= b_hi:
        raise CurveCoverageError(
            f"inverse ballistic coefficient {b} m^2/kg lies outside the curve "
            f"range {b_lo}-{b_hi} m^2/kg"
        )
    values = table.inverse_ballistic_coefficients_m2_kg
    decays = table.decay_days
    for i in range(1, len(values)):
        if values[i - 1] <= b <= values[i]:
            span = math.log(values[i]) - math.log(values[i - 1])
            t = (math.log(b) - math.log(values[i - 1])) / span
            log_lo, log_hi = math.log(decays[i - 1]), math.log(decays[i])
            return math.exp(log_lo + t * (log_hi - log_lo))
    # Unreachable: the bounds check above guarantees a bracket exists.
    raise CurveCoverageError(f"no bracketing interval found for {b} m^2/kg")


@dataclass(frozen=True, slots=True)
class SharedFit:
    """One shared-parameter calibration, with everything needed to reproduce it.

    The optimisation metadata is carried rather than described in prose because
    "what was the objective, the method, the bounds, the initial guess and the
    convergence criterion?" are exactly the questions a reviewer asks and the
    questions the previous report could not answer from its own artefacts.
    """

    fitted_inverse_ballistic_coefficient_m2_kg: float
    n_observations: int
    predicted_decay_days: float
    sum_of_squared_residuals_days2: float
    rmse_days: float
    objective: str
    method: str
    parameter_bounds_m2_kg: tuple[float, float]
    initial_probe_m2_kg: float
    convergence_tolerance_log_b: float
    iterations: int
    converged: bool

    @property
    def ballistic_coefficient_kg_m2(self) -> float:
        """The conventional B = m/(C_d*A), for readers who expect that form."""
        return 1.0 / self.fitted_inverse_ballistic_coefficient_m2_kg


def fit_shared_inverse_ballistic_coefficient(
    observed_days: Sequence[float],
    curve: Sequence[DecayPoint],
    *,
    xatol: float = DEFAULT_LOG_XATOL,
) -> SharedFit:
    """Fit ONE inverse ballistic coefficient shared by every observation.

    Objective: S(B) = sum_i ( D(B) - d_i )^2, with D(B) the modelled decay time
    interpolated from ``curve`` and d_i the observed decay times in days.
    Squared days, unweighted, no prior, no regularisation.

    Method: bounded Brent (scipy ``minimize_scalar(method="bounded")``) over
    ln B rather than B, because decay time is close to a power law in B and the
    objective is far better conditioned in the log variable.

    Bounds: the endpoints of the supplied curve. Outside them the model has not
    been evaluated, and extrapolating a propagation is not a fit.

    The minimiser is interior whenever the mean observation lies strictly
    inside the curve's decay range, because D is a continuous strictly
    decreasing bijection onto that range; that condition is checked up front,
    so a value pinned to a bound cannot be reported as a fit.
    """
    if not observed_days:
        raise ValueError("cannot fit a shared parameter to zero observations")
    if xatol <= 0:
        raise ValueError(f"convergence tolerance must be positive, got {xatol}")
    for d in observed_days:
        if d <= 0:
            raise ValueError(f"observed decay must be positive, got {d} days")

    table = _prepare(curve)
    b_lo, b_hi = table.b_bounds
    d_short, d_long = table.decay_bounds
    mean_observed = math.fsum(observed_days) / len(observed_days)
    if not d_short < mean_observed < d_long:
        raise CurveCoverageError(
            f"mean observed decay {mean_observed:.4f} d lies outside the curve's "
            f"decay range {d_short:.4f}-{d_long:.4f} d; the least-squares "
            "minimiser would sit on a bound and would not be a fit"
        )

    def objective(log_b: float) -> float:
        modelled = predicted_decay_days(curve, math.exp(log_b))
        return math.fsum((modelled - d) ** 2 for d in observed_days)

    lo, hi = math.log(b_lo), math.log(b_hi)
    result = minimize_scalar(
        objective, bounds=(lo, hi), method="bounded", options={"xatol": xatol}
    )
    fitted = math.exp(float(result.x))
    modelled = predicted_decay_days(curve, fitted)
    sse = float(result.fun)
    return SharedFit(
        fitted_inverse_ballistic_coefficient_m2_kg=fitted,
        n_observations=len(observed_days),
        predicted_decay_days=modelled,
        sum_of_squared_residuals_days2=sse,
        rmse_days=math.sqrt(sse / len(observed_days)),
        objective="sum of squared residuals in decay days, unweighted",
        method="scipy minimize_scalar, bounded Brent, over ln(B_inv)",
        parameter_bounds_m2_kg=(b_lo, b_hi),
        initial_probe_m2_kg=math.exp(lo + _GOLDEN_SECTION_FRACTION * (hi - lo)),
        convergence_tolerance_log_b=xatol,
        iterations=int(result.nit),
        converged=bool(result.success),
    )


@dataclass(frozen=True, slots=True)
class ResidualStats:
    """Residual summary. Residual = predicted minus observed, in days."""

    n: int
    rmse_days: float
    mean_absolute_error_days: float
    mean_signed_error_days: float
    max_absolute_error_days: float


def _residual_stats(residuals: Sequence[float]) -> ResidualStats:
    if not residuals:
        raise ValueError("cannot summarise an empty residual set")
    n = len(residuals)
    return ResidualStats(
        n=n,
        rmse_days=math.sqrt(math.fsum(r * r for r in residuals) / n),
        mean_absolute_error_days=math.fsum(abs(r) for r in residuals) / n,
        mean_signed_error_days=math.fsum(residuals) / n,
        max_absolute_error_days=max(abs(r) for r in residuals),
    )


@dataclass(frozen=True, slots=True)
class CalibrationVerdict:
    """Whether withheld data are predicted as well as fitted data.

    ``n_train`` is the number of observations used to estimate the parameter;
    ``n_test`` the number of observations scored without having contributed to
    it. For leave-one-out these are N-1 and N, because every object is withheld
    exactly once from a fit of the other N-1.
    """

    in_sample_rmse_days: float
    out_of_sample_rmse_days: float
    n_train: int
    n_test: int
    out_of_sample_to_in_sample_ratio: float
    verdict: str


def calibration_verdict(
    *,
    in_sample_rmse_days: float,
    out_of_sample_rmse_days: float,
    n_train: int,
    n_test: int,
    material_fraction: float = MATERIAL_DEGRADATION_FRACTION,
) -> CalibrationVerdict:
    """State plainly whether out-of-sample error is materially worse."""
    if in_sample_rmse_days < 0 or out_of_sample_rmse_days < 0:
        raise ValueError("RMSE cannot be negative")
    if n_train < 1 or n_test < 1:
        raise ValueError(f"need at least one train and one test point, got {n_train}/{n_test}")

    if in_sample_rmse_days == 0.0:
        ratio = math.inf if out_of_sample_rmse_days > 0 else 1.0
        verdict = (
            f"In-sample RMSE is exactly zero over n_train={n_train} because the "
            "model was saturated — one free parameter per observation — so the "
            "in-sample fit is a tautology and cannot be compared. Out-of-sample "
            f"RMSE over n_test={n_test} is {out_of_sample_rmse_days:.3f} d, and "
            "that number is the only one carrying information."
        )
        return CalibrationVerdict(
            in_sample_rmse_days=in_sample_rmse_days,
            out_of_sample_rmse_days=out_of_sample_rmse_days,
            n_train=n_train,
            n_test=n_test,
            out_of_sample_to_in_sample_ratio=ratio,
            verdict=verdict,
        )

    ratio = out_of_sample_rmse_days / in_sample_rmse_days
    degradation = ratio - 1.0
    if degradation > material_fraction:
        judgement = (
            f"Out-of-sample error IS materially worse than in-sample: "
            f"{degradation:.1%} higher, above the {material_fraction:.0%} threshold."
        )
    else:
        judgement = (
            f"Out-of-sample error is NOT materially worse than in-sample: "
            f"{degradation:+.1%}, within the {material_fraction:.0%} threshold."
        )
    verdict = (
        f"{judgement} In-sample RMSE {in_sample_rmse_days:.3f} d over "
        f"n_train={n_train}; out-of-sample RMSE {out_of_sample_rmse_days:.3f} d "
        f"over n_test={n_test}. A small degradation is not evidence of a good "
        "model: it also arises when the model carries no per-object information, "
        "in which case both errors equal the spread of the observations."
    )
    return CalibrationVerdict(
        in_sample_rmse_days=in_sample_rmse_days,
        out_of_sample_rmse_days=out_of_sample_rmse_days,
        n_train=n_train,
        n_test=n_test,
        out_of_sample_to_in_sample_ratio=ratio,
        verdict=verdict,
    )


@dataclass(frozen=True, slots=True)
class LeaveOneOutRecord:
    """One fold: fitted on the other N-1, scored on the object held out."""

    label: str
    training_fitted_inverse_ballistic_coefficient_m2_kg: float
    predicted_decay_days: float
    observed_decay_days: float
    residual_days: float
    relative_residual: float


@dataclass(frozen=True, slots=True)
class LeaveOneOutResult:
    """Leave-one-out cross-validation of the shared-parameter calibration."""

    records: tuple[LeaveOneOutRecord, ...]
    full_sample_fit: SharedFit
    in_sample: ResidualStats
    out_of_sample: ResidualStats
    fitted_spread_m2_kg: tuple[float, float]
    fitted_relative_spread: float
    verdict: CalibrationVerdict


def _unique_labels(objects: Sequence[ObservedDecay]) -> None:
    seen: set[str] = set()
    for obj in objects:
        if obj.label in seen:
            raise ValueError(f"duplicate object label {obj.label!r}; labels must be unique")
        seen.add(obj.label)


def leave_one_out_fit(
    objects: Sequence[ObservedDecay],
    curve: Sequence[DecayPoint],
    *,
    xatol: float = DEFAULT_LOG_XATOL,
) -> LeaveOneOutResult:
    """Withhold each object in turn, fit on the rest, predict the one withheld.

    This is the exercise the original report omitted. Because the parameter is
    shared, withholding an object genuinely removes information from the fit,
    and the residual it produces is an OUT-OF-SAMPLE residual rather than a
    restatement of the calibration.

    The in-sample comparator is the fit over all N objects, scored against
    those same N. Both are reported so the two can be set side by side without
    either being mistaken for the other.
    """
    if len(objects) < 2:
        raise ValueError(
            f"leave-one-out needs at least two objects, got {len(objects)}; with "
            "one object the training set is empty"
        )
    _unique_labels(objects)

    all_days = [o.observed_decay_days for o in objects]
    full = fit_shared_inverse_ballistic_coefficient(all_days, curve, xatol=xatol)
    in_sample = _residual_stats([full.predicted_decay_days - d for d in all_days])

    records: list[LeaveOneOutRecord] = []
    for i, held_out in enumerate(objects):
        training = [o.observed_decay_days for j, o in enumerate(objects) if j != i]
        fit = fit_shared_inverse_ballistic_coefficient(training, curve, xatol=xatol)
        residual = fit.predicted_decay_days - held_out.observed_decay_days
        records.append(
            LeaveOneOutRecord(
                label=held_out.label,
                training_fitted_inverse_ballistic_coefficient_m2_kg=(
                    fit.fitted_inverse_ballistic_coefficient_m2_kg
                ),
                predicted_decay_days=fit.predicted_decay_days,
                observed_decay_days=held_out.observed_decay_days,
                residual_days=residual,
                relative_residual=residual / held_out.observed_decay_days,
            )
        )

    out_of_sample = _residual_stats([r.residual_days for r in records])
    fitted = [r.training_fitted_inverse_ballistic_coefficient_m2_kg for r in records]
    lo, hi = min(fitted), max(fitted)
    return LeaveOneOutResult(
        records=tuple(records),
        full_sample_fit=full,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        fitted_spread_m2_kg=(lo, hi),
        fitted_relative_spread=(hi - lo) / full.fitted_inverse_ballistic_coefficient_m2_kg,
        verdict=calibration_verdict(
            in_sample_rmse_days=in_sample.rmse_days,
            out_of_sample_rmse_days=out_of_sample.rmse_days,
            n_train=len(objects) - 1,
            n_test=len(objects),
        ),
    )


@dataclass(frozen=True, slots=True)
class SplitFitResult:
    """One deterministic train/test split of the calibration set."""

    seed: int
    train_labels: tuple[str, ...]
    test_labels: tuple[str, ...]
    fit: SharedFit
    train_residuals: ResidualStats
    test_residuals: ResidualStats
    test_records: tuple[LeaveOneOutRecord, ...]
    verdict: CalibrationVerdict


def train_test_split_fit(
    objects: Sequence[ObservedDecay],
    curve: Sequence[DecayPoint],
    *,
    n_train: int,
    seed: int,
    xatol: float = DEFAULT_LOG_XATOL,
) -> SplitFitResult:
    """Split once, fit on the training part, score the withheld part.

    The split is deterministic in ``seed`` AND in the object set: objects are
    sorted by label before shuffling, so the assignment does not depend on the
    order the caller happened to build the sequence in. Ten objects is far too
    few for a single split to be stable, which is why ``leave_one_out_fit`` is
    the primary result and this is the corroborating one.
    """
    if len(objects) < 2:
        raise ValueError(f"a train/test split needs at least two objects, got {len(objects)}")
    if not 1 <= n_train < len(objects):
        raise ValueError(
            f"n_train must be at least 1 and leave at least one test object; got "
            f"n_train={n_train} with {len(objects)} objects"
        )
    _unique_labels(objects)

    ordered = sorted(objects, key=lambda o: o.label)
    indices = list(range(len(ordered)))
    random.Random(seed).shuffle(indices)
    train = [ordered[i] for i in sorted(indices[:n_train])]
    test = [ordered[i] for i in sorted(indices[n_train:])]

    fit = fit_shared_inverse_ballistic_coefficient(
        [o.observed_decay_days for o in train], curve, xatol=xatol
    )
    train_residuals = _residual_stats(
        [fit.predicted_decay_days - o.observed_decay_days for o in train]
    )
    test_records = tuple(
        LeaveOneOutRecord(
            label=o.label,
            training_fitted_inverse_ballistic_coefficient_m2_kg=(
                fit.fitted_inverse_ballistic_coefficient_m2_kg
            ),
            predicted_decay_days=fit.predicted_decay_days,
            observed_decay_days=o.observed_decay_days,
            residual_days=fit.predicted_decay_days - o.observed_decay_days,
            relative_residual=(
                (fit.predicted_decay_days - o.observed_decay_days) / o.observed_decay_days
            ),
        )
        for o in test
    )
    test_residuals = _residual_stats([r.residual_days for r in test_records])

    return SplitFitResult(
        seed=seed,
        train_labels=tuple(o.label for o in train),
        test_labels=tuple(o.label for o in test),
        fit=fit,
        train_residuals=train_residuals,
        test_residuals=test_residuals,
        test_records=test_records,
        verdict=calibration_verdict(
            in_sample_rmse_days=train_residuals.rmse_days,
            out_of_sample_rmse_days=test_residuals.rmse_days,
            n_train=len(train),
            n_test=len(test),
        ),
    )


def scale_curve_density(curve: Sequence[DecayPoint], factor: float) -> tuple[DecayPoint, ...]:
    """The decay curve that a uniformly k-scaled atmosphere would produce.

    Drag acceleration is -(1/2) * B_inv * rho * |v_rel| * v_rel, so density and
    the inverse ballistic coefficient enter the equations of motion ONLY as the
    product B_inv*rho. Replacing rho by k*rho and B_inv by B_inv/k leaves every
    derivative unchanged, hence the whole trajectory and the decay time. A
    forward run at (k*rho, B_inv/k) therefore reproduces the run at
    (rho, B_inv) exactly, and the curve under k-scaled density is the original
    curve with its abscissa divided by k.

    This is an exact algebraic consequence of the equations of motion, not an
    approximation. It holds for a UNIFORM scaling; an altitude- or time-
    dependent density error is not exactly degenerate with a constant B_inv,
    and this transformation does not claim it is.
    """
    if factor <= 0:
        raise ValueError(f"density scale factor must be positive, got {factor}")
    return tuple(
        DecayPoint(
            inverse_ballistic_coefficient=p.inverse_ballistic_coefficient / factor,
            decay_days=p.decay_days,
            final_perigee_km=p.final_perigee_km,
            reentered=p.reentered,
        )
        for p in curve
    )


@dataclass(frozen=True, slots=True)
class DensityDegeneracyPoint:
    """One (density scale factor, refitted parameter) pair on the ridge."""

    density_scale_factor: float
    inverse_ballistic_coefficient_m2_kg: float
    ballistic_coefficient_kg_m2: float
    decay_days_under_scaled_density: float
    decay_reproduction_error_days: float
    product_scale_factor_times_inverse_ballistic_coefficient: float


@dataclass(frozen=True, slots=True)
class IdentifiabilityProfile:
    """Numerical demonstration that density and C_d*A/m are not separable."""

    observed_decay_days: float
    points: tuple[DensityDegeneracyPoint, ...]
    constrained_product_m2_kg: float
    product_relative_spread: float
    max_decay_reproduction_error_days: float
    separately_identifiable: bool
    verdict: str


def identifiability_profile(
    curve: Sequence[DecayPoint],
    observed_days: float,
    *,
    density_scale_factors: Sequence[float],
) -> IdentifiabilityProfile:
    """Refit the drag parameter under a scaled atmosphere and show the ridge.

    For each density scale factor k, find the inverse ballistic coefficient
    that reproduces ``observed_days`` when the atmosphere is k times denser.
    The answer is B_inv(k) = B_inv(1)/k, every one of them reproduces the SAME
    decay time to interpolation precision, and the product k*B_inv is constant.

    That product — equivalently the combination rho*C_d*A/m — is what the decay
    observation constrains. The two factors are not separately identifiable
    from decay data at all: no decay time, however precisely observed, can
    distinguish a dense atmosphere and a slippery satellite from a thin
    atmosphere and a draggy one. Reporting a fitted C_d*A/m as though it were a
    property of the vehicle silently assigns the entire density error to the
    spacecraft.
    """
    if observed_days <= 0:
        raise ValueError(f"observed decay must be positive, got {observed_days} days")
    if not density_scale_factors:
        raise ValueError("identifiability profile needs at least one density scale factor")

    points: list[DensityDegeneracyPoint] = []
    for k in density_scale_factors:
        scaled = scale_curve_density(curve, k)
        fit = fit_shared_inverse_ballistic_coefficient([observed_days], scaled)
        b = fit.fitted_inverse_ballistic_coefficient_m2_kg
        reproduced = predicted_decay_days(scaled, b)
        points.append(
            DensityDegeneracyPoint(
                density_scale_factor=k,
                inverse_ballistic_coefficient_m2_kg=b,
                ballistic_coefficient_kg_m2=1.0 / b,
                decay_days_under_scaled_density=reproduced,
                decay_reproduction_error_days=reproduced - observed_days,
                product_scale_factor_times_inverse_ballistic_coefficient=k * b,
            )
        )

    products = [p.product_scale_factor_times_inverse_ballistic_coefficient for p in points]
    mean_product = math.fsum(products) / len(products)
    spread = (max(products) - min(products)) / mean_product
    max_error = max(abs(p.decay_reproduction_error_days) for p in points)
    # The ridge is flat to interpolation precision, so any non-zero spread here
    # is numerical, not physical. The threshold is deliberately generous: a
    # separately identifiable pair would show a spread of order one, not 1e-6.
    separable = spread > 1e-6
    verdict = (
        f"Density scale factors {min(density_scale_factors):g}-"
        f"{max(density_scale_factors):g} all reproduce the same "
        f"{observed_days:.3f} d decay to within {max_error:.2e} d, with "
        f"C_d*A/m varying from {min(p.inverse_ballistic_coefficient_m2_kg for p in points):.6f} "
        f"to {max(p.inverse_ballistic_coefficient_m2_kg for p in points):.6f} m^2/kg while the "
        f"product k*C_d*A/m stays at {mean_product:.6f} m^2/kg "
        f"(relative spread {spread:.2e}). Density and the inverse ballistic "
        "coefficient are NOT separately identifiable from a decay time; only "
        "their product is constrained."
    )
    if separable:  # pragma: no cover - would indicate the algebra had broken
        verdict = (
            "UNEXPECTED: the product k*C_d*A/m varied by "
            f"{spread:.2e} across density scale factors. The degeneracy argument "
            "predicts an exactly constant product; investigate the curve before "
            "reading anything into the fitted values."
        )
    return IdentifiabilityProfile(
        observed_decay_days=observed_days,
        points=tuple(points),
        constrained_product_m2_kg=mean_product,
        product_relative_spread=spread,
        max_decay_reproduction_error_days=max_error,
        separately_identifiable=separable,
        verdict=verdict,
    )


@dataclass(frozen=True, slots=True)
class NumericVariant:
    """A decay curve computed under one changed numerical setting.

    ``axis`` names the setting that was varied and ``setting`` its value, both
    as strings, because the axes are heterogeneous (kilometres, dimensionless
    tolerances, degrees) and forcing them into one numeric type would invite
    exactly the unit confusion this repository exists to avoid.
    """

    axis: str
    setting: str
    curve: tuple[DecayPoint, ...]


@dataclass(frozen=True, slots=True)
class NumericVariantFit:
    """The shared-parameter fit obtained under one numerical setting."""

    axis: str
    setting: str
    fitted_inverse_ballistic_coefficient_m2_kg: float
    predicted_decay_days: float
    rmse_days: float


@dataclass(frozen=True, slots=True)
class NumericSensitivity:
    """Spread of the fitted parameter across one axis of numerical settings."""

    axis: str
    fits: tuple[NumericVariantFit, ...]
    minimum_fitted_m2_kg: float
    maximum_fitted_m2_kg: float
    relative_spread: float
    note: str


def sensitivity_to_numerics(
    observed_days: Sequence[float],
    variants: Sequence[NumericVariant],
) -> tuple[NumericSensitivity, ...]:
    """Refit under each numerical setting and report the spread per axis.

    The curves are supplied rather than computed here: producing one costs a
    propagation sweep, which belongs in a script with a process pool, not in a
    library function. What belongs here is the refit and the comparison, which
    are pure and testable.

    A fitted parameter that moves when the re-entry threshold, the solver
    tolerance or the initial orbital phase moves is a parameter partly
    absorbing a numerical choice. The spread is therefore a lower bound on the
    systematic uncertainty of the calibration, and it is reported as a fraction
    of the fitted value so it can be set against the catalogue-date and
    density-model uncertainties on the same footing.
    """
    if not variants:
        raise ValueError("numerical sensitivity needs at least one variant curve")

    axes: list[str] = []
    grouped: dict[str, list[NumericVariant]] = {}
    for variant in variants:
        if variant.axis not in grouped:
            axes.append(variant.axis)
            grouped[variant.axis] = []
        grouped[variant.axis].append(variant)

    out: list[NumericSensitivity] = []
    for axis in axes:
        fits: list[NumericVariantFit] = []
        for variant in grouped[axis]:
            fit = fit_shared_inverse_ballistic_coefficient(observed_days, variant.curve)
            fits.append(
                NumericVariantFit(
                    axis=axis,
                    setting=variant.setting,
                    fitted_inverse_ballistic_coefficient_m2_kg=(
                        fit.fitted_inverse_ballistic_coefficient_m2_kg
                    ),
                    predicted_decay_days=fit.predicted_decay_days,
                    rmse_days=fit.rmse_days,
                )
            )
        values = [f.fitted_inverse_ballistic_coefficient_m2_kg for f in fits]
        lo, hi = min(values), max(values)
        mid = 0.5 * (lo + hi)
        spread = (hi - lo) / mid
        out.append(
            NumericSensitivity(
                axis=axis,
                fits=tuple(fits),
                minimum_fitted_m2_kg=lo,
                maximum_fitted_m2_kg=hi,
                relative_spread=spread,
                note=(
                    f"Varying {axis} over {len(fits)} settings moves the fitted "
                    f"C_d*A/m from {lo:.6f} to {hi:.6f} m^2/kg, a relative spread "
                    f"of {spread:.2%} about the midpoint."
                ),
            )
        )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class QuantisationEnvelope:
    """Fitted-parameter range implied by day-resolution decay dates."""

    half_width_days: float
    nominal_mean_observed_days: float
    fitted_nominal_m2_kg: float
    fitted_from_earliest_dates_m2_kg: float
    fitted_from_latest_dates_m2_kg: float
    lower_m2_kg: float
    upper_m2_kg: float
    relative_width: float
    note: str


def decay_date_quantisation_envelope(
    observed_days: Sequence[float],
    curve: Sequence[DecayPoint],
    *,
    half_width_days: float = DEFAULT_QUANTISATION_HALF_WIDTH_DAYS,
) -> QuantisationEnvelope:
    """Refit with every decay shifted earlier and later by the half width.

    The catalogue records a decay DATE, not a decay time, so the true instant
    lies anywhere inside the recorded day. Shifting every observation by the
    same half day is the worst case and therefore the honest envelope: an
    independent per-object shift would partly cancel in the mean and would
    understate the systematic component, which is what a common transcription
    or epoch convention would produce.

    Earlier decays mean faster decay, hence a LARGER inverse ballistic
    coefficient, so the earliest-date fit is the upper edge of the envelope.
    """
    if not observed_days:
        raise ValueError("quantisation envelope needs at least one observation")
    if half_width_days <= 0:
        raise ValueError(f"half width must be positive, got {half_width_days} days")
    earliest = [d - half_width_days for d in observed_days]
    if any(d <= 0 for d in earliest):
        raise ValueError(
            f"shifting by {half_width_days} days sends an observed decay to zero "
            "or below; the envelope is not defined there"
        )

    nominal = fit_shared_inverse_ballistic_coefficient(observed_days, curve)
    early = fit_shared_inverse_ballistic_coefficient(earliest, curve)
    late = fit_shared_inverse_ballistic_coefficient(
        [d + half_width_days for d in observed_days], curve
    )
    lo = min(
        early.fitted_inverse_ballistic_coefficient_m2_kg,
        late.fitted_inverse_ballistic_coefficient_m2_kg,
    )
    hi = max(
        early.fitted_inverse_ballistic_coefficient_m2_kg,
        late.fitted_inverse_ballistic_coefficient_m2_kg,
    )
    nominal_value = nominal.fitted_inverse_ballistic_coefficient_m2_kg
    return QuantisationEnvelope(
        half_width_days=half_width_days,
        nominal_mean_observed_days=math.fsum(observed_days) / len(observed_days),
        fitted_nominal_m2_kg=nominal_value,
        fitted_from_earliest_dates_m2_kg=early.fitted_inverse_ballistic_coefficient_m2_kg,
        fitted_from_latest_dates_m2_kg=late.fitted_inverse_ballistic_coefficient_m2_kg,
        lower_m2_kg=lo,
        upper_m2_kg=hi,
        relative_width=(hi - lo) / nominal_value,
        note=(
            f"Day-resolution decay dates at +/-{half_width_days:g} d place the "
            f"fitted C_d*A/m between {lo:.6f} and {hi:.6f} m^2/kg, a width of "
            f"{(hi - lo) / nominal_value:.1%} of the nominal {nominal_value:.6f} "
            "m^2/kg."
        ),
    )


@dataclass(frozen=True, slots=True)
class ObjectiveProfile:
    """The objective sampled across the parameter range, plus its curvature.

    With one parameter there is no covariance matrix to report. What can be
    reported is the shape of the objective and the standard error implied by
    its curvature at the minimum, which is the one-parameter analogue.
    """

    inverse_ballistic_coefficients_m2_kg: tuple[float, ...]
    sum_of_squared_residuals_days2: tuple[float, ...]
    minimiser_m2_kg: float
    minimum_sum_of_squared_residuals_days2: float
    residual_standard_deviation_days: float
    curvature_standard_error_m2_kg: float
    note: str


def objective_profile(
    observed_days: Sequence[float],
    curve: Sequence[DecayPoint],
    *,
    n_points: int = 201,
) -> ObjectiveProfile:
    """Sample S(B) across the curve and derive a curvature standard error.

    The standard error comes from the usual Gauss-Newton result for a
    one-parameter least-squares problem: sigma_B = s / (|dD/dB| * sqrt(N)),
    with s the residual standard deviation on N-1 degrees of freedom and dD/dB
    evaluated at the minimum by central difference on the interpolant. It is a
    PRECISION statement about the fit under the assumption that the residuals
    are independent noise. That assumption is false here — the residuals are
    dominated by real object-to-object differences the shared model does not
    represent — so the number is reported as a lower bound on the parameter
    uncertainty and never as its total.
    """
    if n_points < 3:
        raise ValueError(f"objective profile needs at least three points, got {n_points}")
    fit = fit_shared_inverse_ballistic_coefficient(observed_days, curve)
    table = _prepare(curve)
    b_lo, b_hi = table.b_bounds

    log_lo, log_hi = math.log(b_lo), math.log(b_hi)
    step = (log_hi - log_lo) / (n_points - 1)
    # exp(log(b)) does not round-trip exactly, so the last grid point can land a
    # few ulp above the bound and be rejected as an extrapolation. Clamping is
    # the honest fix: the endpoints ARE the bounds, to the precision available.
    grid = [min(max(math.exp(log_lo + i * step), b_lo), b_hi) for i in range(n_points)]
    values = [
        math.fsum((predicted_decay_days(curve, b) - d) ** 2 for d in observed_days) for b in grid
    ]

    n = len(observed_days)
    dof = max(n - 1, 1)
    s = math.sqrt(fit.sum_of_squared_residuals_days2 / dof)
    b_star = fit.fitted_inverse_ballistic_coefficient_m2_kg
    h = 1e-4 * b_star
    slope = (
        predicted_decay_days(curve, min(b_star + h, b_hi))
        - predicted_decay_days(curve, max(b_star - h, b_lo))
    ) / (min(b_star + h, b_hi) - max(b_star - h, b_lo))
    sigma = s / (abs(slope) * math.sqrt(n))
    return ObjectiveProfile(
        inverse_ballistic_coefficients_m2_kg=tuple(grid),
        sum_of_squared_residuals_days2=tuple(values),
        minimiser_m2_kg=b_star,
        minimum_sum_of_squared_residuals_days2=fit.sum_of_squared_residuals_days2,
        residual_standard_deviation_days=s,
        curvature_standard_error_m2_kg=sigma,
        note=(
            f"One parameter, so no covariance matrix exists. Curvature standard "
            f"error {sigma:.6f} m^2/kg ({sigma / b_star:.1%} of the fitted value) "
            f"from residual standard deviation {s:.3f} d on {dof} degrees of "
            "freedom. This is a PRECISION statement only: it assumes the "
            "residuals are independent noise, which they are not, and it "
            "excludes the density degeneracy entirely, along which the "
            "objective is exactly flat."
        ),
    )
