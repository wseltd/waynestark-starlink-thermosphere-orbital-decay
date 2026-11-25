"""Global sensitivity analysis: which unmeasured number is driving the answer.

The Group 4-7 decay calculation rests on six quantities nobody published for
these vehicles in this configuration. Quoting a single decay time from a single
choice of those six would present an arbitrary point in a six-dimensional box as
though it were a result. The honest alternative is to sample the box and ask
which axis the answer actually moves along.

Two methods, chosen for what they cost and what they can say:

  * Sobol' variance decomposition (Saltelli estimator) apportions output
    variance between each input alone (first order, ``S1``) and each input
    together with every interaction it takes part in (total order, ``ST``). It
    is the quantitative answer and it costs ``N*(2D+2)`` model evaluations. For
    a model whose single evaluation is a multi-day orbit propagation that is the
    entire compute budget, so :func:`n_model_evaluations` exists to put the
    price on the table before anyone commits to paying it.
  * Morris elementary effects (Campolongo screening) costs ``N*(D+1)``.
    ``mu_star`` ranks influence and ``sigma`` flags non-linearity or
    interaction. It cannot apportion variance, but it can identify which
    parameters are safe to freeze before a Sobol' run is funded.

Two claims this module deliberately does NOT make:

  * A Sobol index is a property of a MODEL under an ASSUMED input distribution.
    It is not a measurement of the atmosphere. A parameter that ranks low is not
    thereby physically unimportant — it means this model is insensitive to it
    over the stated range, and a model can be insensitive to something precisely
    because the mechanism that would make it matter is missing.
  * Putting a uniform distribution over a range is itself a modelling claim, not
    a neutral default. That is why :class:`UncertainParameter` will not accept a
    range without a justification: an unjustified interval would enter the
    variance decomposition carrying exactly the same weight as a measured bound
    and leave it indistinguishable from one. The repository's standing position
    that sensitivity envelopes are not confidence intervals (README,
    "Limitations") is unchanged by this module; what is decomposed here is the
    variance of the model over ranges whose provenance is recorded alongside the
    indices.

Implementation is SALib (MIT), not a hand-rolled estimator: the Saltelli
cross-sampling matrices and the total-index estimator are exactly the kind of
index-arithmetic that is easy to get subtly wrong and hard to notice, and the
package is the reference implementation. What is validated locally is that this
wrapper drives it correctly — the test suite runs the Ishigami function, whose
first-order indices are known in closed form, and checks the numbers that come
back against those analytic values.

Primary sources:
  Sobol', I.M. (2001) "Global sensitivity indices for nonlinear mathematical
    models and their Monte Carlo estimates", Math. Comput. Simul. 55:271,
    doi:10.1016/S0378-4754(00)00270-6.
  Saltelli, A. et al. (2010) "Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index",
    Comput. Phys. Commun. 181:259, doi:10.1016/j.cpc.2009.09.018.
  Morris, M.D. (1991) "Factorial sampling plans for preliminary computational
    experiments", Technometrics 33:161, doi:10.1080/00401706.1991.10484804.
  Campolongo, F., Cariboni, J. & Saltelli, A. (2007) "An effective screening
    design for sensitivity analysis of large models", Environ. Model. Softw.
    22:1509, doi:10.1016/j.envsoft.2006.10.004.
  Herman, J. & Usher, W. (2017) "SALib: an open-source Python library for
    sensitivity analysis", J. Open Source Softw. 2(9):97,
    doi:10.21105/joss.00097.

Nearest canonical ticket: P06_08_T039, which scoped the uncertainty stage as an
interface gated on justified distributions and left execution out. This module
is the executing layer that ticket anticipated, and it keeps the gate: the
justification is a required constructor argument, so no range reaches a sampler
without its reason attached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from SALib.analyze import morris as _morris_analyser
from SALib.analyze import sobol as _sobol_analyser
from SALib.sample import morris as _morris_sampler
from SALib.sample import sobol as _sobol_sampler

__all__ = [
    "MODEL_RELATIVE_CAVEAT",
    "MORRIS_NUM_LEVELS",
    "STARLINK_PARAMETERS",
    "UncertainParameter",
    "analyse_morris",
    "analyse_sobol",
    "build_salib_problem",
    "morris_sample",
    "n_model_evaluations",
    "rank_parameters",
    "screening_report",
    "sobol_sample",
]

#: The only distributions accepted, and their SALib spellings. Both are
#: parameterised by nothing except the low and high bound, which is the whole of
#: what :class:`UncertainParameter` records. A normal or triangular distribution
#: would need a mean, a mode or a spread that the sources cited in these
#: justifications do not supply, so accepting one would mean inventing it.
_SALIB_DISTRIBUTION: dict[str, str] = {
    "uniform": "unif",
    "loguniform": "logunif",
}

#: Morris grid resolution. Campolongo et al. (2007) recommend an even number of
#: levels with a step of ``num_levels / 2``; 4 is their worked default and
#: SALib's.
MORRIS_NUM_LEVELS = 4

#: Default seed for the bootstrap that produces the interval on each index.
#: Fixed so that two analyses of the same outputs return identical intervals;
#: the value is the Group 4-7 deployment date, arbitrary but traceable.
_BOOTSTRAP_SEED = 20220203

#: Number of bootstrap resamples behind each reported interval.
_BOOTSTRAP_RESAMPLES = 100

#: Stated on every screening report. Held as a constant rather than inlined so
#: that the report cannot be regenerated without it and a test can pin it.
MODEL_RELATIVE_CAVEAT = (
    "Sobol indices are computed against a MODEL, under the input distributions "
    "assumed above. They are not measurements of the atmosphere. A parameter "
    "that ranks low here is NOT thereby physically unimportant: it means THIS "
    "MODEL is insensitive to it over the stated range. A model can be "
    "insensitive to a quantity precisely because the mechanism through which it "
    "would matter is absent from the model, so a low rank is a prompt to check "
    "the model, not a licence to stop measuring the parameter."
)

#: Type of SALib's problem dictionary. It carries the four keys SALib reads
#: (``num_vars``, ``names``, ``bounds``, ``dists``) plus ``units`` and
#: ``justifications``, which SALib ignores and the report renders — so the
#: reason for each range travels with the sample design instead of being left
#: behind in the source file.
SalibProblem = dict[str, Any]


@dataclass(frozen=True, slots=True)
class UncertainParameter:
    """One quantity the analysis does not know, with the reason it does not.

    ``justification`` is required and is checked for content. A range without a
    stated reason is a guess wearing a lab coat: once it is inside a variance
    decomposition it is weighted identically to a bound that somebody measured,
    and nothing downstream can tell the two apart. Requiring the citation at
    construction is the only point in the chain where the distinction still
    exists.
    """

    name: str
    low: float
    high: float
    units: str
    justification: str
    distribution: str = "uniform"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name must not be empty")
        if self.low >= self.high:
            raise ValueError(
                f"{self.name}: low must be strictly below high, got low={self.low} "
                f"and high={self.high}. A collapsed or inverted interval is not an "
                "uncertainty, and SALib would scale samples onto it silently."
            )
        if not self.justification.strip():
            raise ValueError(
                f"{self.name}: a justification is required. A range without a stated "
                "reason is a guess wearing a lab coat — it would enter the variance "
                "decomposition with the same weight as a measured bound and leave it "
                "indistinguishable from one."
            )
        if self.distribution not in _SALIB_DISTRIBUTION:
            raise ValueError(
                f"{self.name}: unsupported distribution {self.distribution!r}; "
                f"expected one of {sorted(_SALIB_DISTRIBUTION)}. Anything else needs a "
                "mean, mode or spread that a low/high pair does not carry."
            )
        if self.distribution == "loguniform" and self.low <= 0.0:
            raise ValueError(
                f"{self.name}: a loguniform range must start above zero, got "
                f"low={self.low}"
            )


#: The uncertainty set for the February 2022 Starlink Group 4-7 event.
#:
#: Every bound traces to the event manifest or to the primary literature cited
#: in its justification. These are the six axes of the box the decay answer
#: lives in; nothing else about the vehicles is unknown to a degree that matters
#: against these.
STARLINK_PARAMETERS: tuple[UncertainParameter, ...] = (
    UncertainParameter(
        name="mass_kg",
        low=275.0,
        high=303.0,
        units="kg",
        justification="KeepTrack third-party estimate; not an operator disclosure",
    ),
    UncertainParameter(
        name="drag_coefficient",
        low=1.8,
        high=3.9,
        units="dimensionless",
        justification=(
            "Cook 1965 convention 2.2 as floor; Moe & Moe 2005 and Mehta et al. 2014 "
            "show physical Cd exceeds it at 200-400 km; SpaceX fitted Cd~2.7 "
            "(arXiv:2406.08342)"
        ),
    ),
    UncertainParameter(
        name="accommodation",
        low=0.7,
        high=1.0,
        units="dimensionless",
        justification=(
            "Moe & Moe 2005 near-complete accommodation on adsorbed atomic oxygen"
        ),
    ),
    UncertainParameter(
        name="projected_area_m2",
        low=2.0,
        high=12.0,
        units="m^2",
        justification=(
            "edge-on safe-mode attitude to full-face; SpaceX reported commanding "
            "edge-on flight"
        ),
    ),
    UncertainParameter(
        name="density_scale_factor",
        low=1.0,
        high=3.5,
        units="dimensionless",
        justification=(
            "He et al. 2023: MSIS ~20-25% storm response vs WAM-IPE ~40% and MAGE "
            "~57-75% at 200/300 km"
        ),
    ),
    UncertainParameter(
        name="quiet_ap",
        low=2.0,
        high=7.0,
        units="nT",
        justification="counterfactual baseline choice, Kp 0+ to 2-",
    ),
)


def _checked_parameters(
    parameters: Sequence[UncertainParameter],
) -> tuple[UncertainParameter, ...]:
    """Reject parameter sets a sampler would misread rather than refuse."""
    checked = tuple(parameters)
    if not checked:
        raise ValueError(
            "at least one uncertain parameter is required; a sensitivity analysis "
            "over an empty set has nothing to decompose"
        )
    names = [p.name for p in checked]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate parameter name(s) {duplicates}. SALib indexes columns by "
            "position and reports them by name, so a repeated name would attribute "
            "one parameter's variance share to another without any error."
        )
    return checked


def _validate_base_samples(n_base: int) -> int:
    """Require a positive power of two, which is what the Sobol' sequence needs."""
    if n_base < 1:
        raise ValueError(f"n_base must be a positive integer, got {n_base}")
    if n_base & (n_base - 1):
        lower = 1 << (n_base.bit_length() - 1)
        raise ValueError(
            f"n_base must be a power of two, got {n_base}. The Saltelli design draws "
            f"from a Sobol' sequence whose balance properties hold only at powers of "
            f"two ({lower} or {lower * 2} here); off a power of two the estimates "
            "degrade with no visible symptom."
        )
    return n_base


def _checked_outputs(outputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Flatten model outputs to float64 and refuse a run that did not finish."""
    values = np.asarray(outputs, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("no model outputs supplied; there is nothing to decompose")
    if not bool(np.all(np.isfinite(values))):
        bad = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(
            f"{bad} of {values.size} model outputs are not finite. A NaN or infinity "
            "propagates through the index arithmetic without raising, so the analysis "
            "is refused: a model evaluation that failed has to be fixed, not averaged."
        )
    return values


def build_salib_problem(parameters: Sequence[UncertainParameter]) -> SalibProblem:
    """Assemble SALib's problem dictionary from a justified parameter set.

    Beyond the four keys SALib reads, the dictionary carries ``units`` and
    ``justifications``. SALib ignores them; :func:`screening_report` renders
    them, so the reason each range was chosen arrives in the same object as the
    indices computed over it rather than having to be looked up separately.
    """
    checked = _checked_parameters(parameters)
    return {
        "num_vars": len(checked),
        "names": [p.name for p in checked],
        "bounds": [[p.low, p.high] for p in checked],
        "dists": [_SALIB_DISTRIBUTION[p.distribution] for p in checked],
        "units": [p.units for p in checked],
        "justifications": [p.justification for p in checked],
    }


def n_model_evaluations(parameters: Sequence[UncertainParameter], n_base: int) -> int:
    """Cost of a second-order Sobol' run: ``N*(2D+2)`` model evaluations.

    Exposed separately from the sampler so a caller can price the run before
    starting it. With a model whose single evaluation is an orbit propagation,
    the difference between ``N=512`` and ``N=4096`` over six parameters is the
    difference between a coffee break and a weekend.

    ``n_base`` is validated exactly as :func:`sobol_sample` validates it, so a
    cost this function quotes is always a cost that can actually be incurred.
    """
    dimension = len(_checked_parameters(parameters))
    _validate_base_samples(n_base)
    return n_base * (2 * dimension + 2)


def sobol_sample(
    parameters: Sequence[UncertainParameter],
    n_base: int,
    *,
    seed: int,
) -> tuple[npt.NDArray[np.float64], SalibProblem]:
    """Draw a Saltelli design over the parameter ranges.

    Returns the sample matrix and the problem dictionary it was built from,
    because :func:`analyse_sobol` needs the same dictionary and reconstructing
    it at the call site is how the two silently drift apart.

    ``seed`` is keyword-only and has no default: the design is a random draw,
    and a run whose seed was never stated cannot be reproduced. ``scramble`` is
    passed explicitly rather than left to SALib's default so that an upstream
    change of default cannot alter results computed by this package.

    The row count is ``n_base * (2D + 2)`` — see :func:`n_model_evaluations`.
    """
    problem = build_salib_problem(parameters)
    _validate_base_samples(n_base)
    samples = _sobol_sampler.sample(
        problem,
        n_base,
        calc_second_order=True,
        scramble=True,
        seed=seed,
    )
    return np.asarray(samples, dtype=np.float64), problem


def morris_sample(
    parameters: Sequence[UncertainParameter],
    n_trajectories: int,
    *,
    seed: int,
    num_levels: int = MORRIS_NUM_LEVELS,
) -> tuple[npt.NDArray[np.float64], SalibProblem]:
    """Draw Morris trajectories for elementary-effects screening.

    The row count is ``n_trajectories * (D + 1)``, an order of magnitude below
    the Sobol' cost, which is the point: Morris is what decides whether the
    Sobol' run is worth funding.

    ``seed`` is keyword-only and required, for the same reason as in
    :func:`sobol_sample`.
    """
    problem = build_salib_problem(parameters)
    if n_trajectories < 2:
        raise ValueError(
            f"n_trajectories must be at least 2, got {n_trajectories}. mu_star and "
            "sigma are a mean and a spread over trajectories; one trajectory gives a "
            "spread of zero that means 'not measured', not 'linear'."
        )
    if num_levels < 2 or num_levels % 2:
        raise ValueError(
            f"num_levels must be an even integer of at least 2, got {num_levels}. "
            "The Morris grid steps by num_levels/2, which is only a lattice point "
            "for an even number of levels (Campolongo et al. 2007)."
        )
    samples = _morris_sampler.sample(problem, n_trajectories, num_levels=num_levels, seed=seed)
    return np.asarray(samples, dtype=np.float64), problem


def analyse_sobol(
    problem: SalibProblem,
    outputs: npt.ArrayLike,
    *,
    num_resamples: int = _BOOTSTRAP_RESAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Decompose output variance into first-order and total-order indices.

    Returns ``{parameter_name: {"S1", "S1_conf", "ST", "ST_conf"}}``. The
    ``_conf`` entries are bootstrap half-widths at the 95% level on the INDEX
    ESTIMATE — they describe the sampling error of the estimator at this ``N``,
    and say nothing about any physical quantity. A ``_conf`` comparable to its
    index means the run was too small to rank that parameter, not that the
    parameter is uncertain.

    ``S1`` can come back slightly negative for a genuinely inactive parameter:
    the Saltelli estimator is unbiased, not non-negative, and a small negative
    value is Monte-Carlo noise around zero. It is passed through unaltered
    rather than clipped, because clipping would hide exactly the signal that
    tells you the run was too small.
    """
    values = _checked_outputs(outputs)
    dimension = int(problem["num_vars"])
    block = 2 * dimension + 2
    if values.size % block:
        raise ValueError(
            f"got {values.size} model outputs, which is not a multiple of 2D+2 = "
            f"{block} for D = {dimension} parameters. The Saltelli design is read in "
            "blocks of that size, so a partial block would pair outputs with the "
            "wrong rows of the design."
        )

    result = _sobol_analyser.analyze(
        problem,
        values,
        calc_second_order=True,
        num_resamples=num_resamples,
        seed=seed,
        print_to_console=False,
    )
    return {
        name: {
            "S1": float(result["S1"][index]),
            "S1_conf": float(result["S1_conf"][index]),
            "ST": float(result["ST"][index]),
            "ST_conf": float(result["ST_conf"][index]),
        }
        for index, name in enumerate(problem["names"])
    }


def analyse_morris(
    problem: SalibProblem,
    samples: npt.ArrayLike,
    outputs: npt.ArrayLike,
    *,
    num_levels: int = MORRIS_NUM_LEVELS,
    num_resamples: int = _BOOTSTRAP_RESAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Elementary-effects screening statistics per parameter.

    Returns ``{parameter_name: {"mu", "mu_star", "mu_star_conf", "sigma"}}``.

    ``mu_star`` is the mean ABSOLUTE elementary effect and is the ranking
    statistic; ``mu`` keeps its sign, so ``mu`` far below ``mu_star`` means the
    effect changes direction across the range. A large ``sigma`` means the
    effect is non-linear or interacts, and is the signal that Morris alone
    cannot settle the question and a Sobol' run is needed.

    ``num_levels`` must match the value used for :func:`morris_sample`; the
    elementary effect is a difference divided by the grid step, so a mismatch
    rescales every result.
    """
    design = np.asarray(samples, dtype=np.float64)
    values = _checked_outputs(outputs)
    dimension = int(problem["num_vars"])
    if design.ndim != 2 or design.shape[1] != dimension:
        raise ValueError(
            f"samples must be a 2-D array with {dimension} columns, one per "
            f"parameter, got shape {design.shape}"
        )
    if design.shape[0] != values.size:
        raise ValueError(
            f"{design.shape[0]} sample rows against {values.size} model outputs. "
            "Morris pairs each output with the row that produced it, so unequal "
            "lengths would compute effects from mismatched pairs."
        )

    result = _morris_analyser.analyze(
        problem,
        design,
        values,
        num_levels=num_levels,
        num_resamples=num_resamples,
        seed=seed,
        print_to_console=False,
    )
    return {
        name: {
            "mu": float(result["mu"][index]),
            "mu_star": float(result["mu_star"][index]),
            "mu_star_conf": float(result["mu_star_conf"][index]),
            "sigma": float(result["sigma"][index]),
        }
        for index, name in enumerate(problem["names"])
    }


def rank_parameters(
    indices: Mapping[str, Mapping[str, float]],
    *,
    by: str = "ST",
) -> list[tuple[str, float]]:
    """Order parameters by one index, largest first.

    Defaults to ``ST``: the total-order index is what answers "can I freeze
    this?", because a parameter with a small ``S1`` and a large ``ST`` matters
    only through its interactions and freezing it would still change the answer.
    Pass ``by="mu_star"`` to rank a :func:`analyse_morris` result.

    Ties break on the parameter name, so the order is a function of the input
    alone and two runs of the same analysis produce the same report.
    """
    if not indices:
        raise ValueError("no indices supplied; there is nothing to rank")
    missing = sorted(name for name, values in indices.items() if by not in values)
    if missing:
        raise ValueError(
            f"index {by!r} is not present for parameter(s) {missing}; available "
            f"keys are {sorted(set().union(*(set(v) for v in indices.values())))}"
        )
    return sorted(
        ((name, float(values[by])) for name, values in indices.items()),
        key=lambda item: (-item[1], item[0]),
    )


def screening_report(
    problem: SalibProblem,
    indices: Mapping[str, Mapping[str, float]],
) -> str:
    """Render the Sobol result as markdown, with its ranges and its caveat.

    The justifications are printed underneath the table on purpose. A ranking
    read without them invites the reader to treat the top row as a physical
    finding, when it is a statement about how wide the top row's range was
    allowed to be and why.
    """
    required = ("names", "bounds", "units", "justifications")
    absent = [key for key in required if key not in problem]
    if absent:
        raise ValueError(
            f"problem dictionary is missing {absent}; build it with "
            "build_salib_problem() so the ranges and their justifications travel "
            "with the design"
        )

    names = list(problem["names"])
    unknown = [name for name in names if name not in indices]
    if unknown:
        raise ValueError(
            f"no indices for parameter(s) {unknown}; the problem and the analysis "
            "describe different parameter sets"
        )

    lines = [
        "### Sobol' sensitivity screening",
        "",
        f"{len(names)} parameters. S1 is the variance share attributable to a "
        "parameter alone; ST additionally includes every interaction it takes part "
        "in. The +/- columns are bootstrap 95% half-widths on the INDEX ESTIMATE, "
        "i.e. the sampling error of the estimator at this design size, not an "
        "uncertainty band on a physical quantity.",
        "",
        "| Parameter | Range | Units | S1 | S1 +/- | ST | ST +/- |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, bounds, units in zip(names, problem["bounds"], problem["units"], strict=True):
        entry = indices[name]
        low, high = float(bounds[0]), float(bounds[1])
        lines.append(
            f"| {name} | {low:g} to {high:g} | {units} | "
            f"{entry['S1']:.4f} | {entry['S1_conf']:.4f} | "
            f"{entry['ST']:.4f} | {entry['ST_conf']:.4f} |"
        )

    ranking = rank_parameters({name: indices[name] for name in names}, by="ST")
    lines += [
        "",
        "Ranked by total order: "
        + ", ".join(f"{name} ({value:.4f})" for name, value in ranking)
        + ".",
        "",
        "Why these ranges:",
    ]
    lines += [
        f"- {name}: {reason}"
        for name, reason in zip(names, problem["justifications"], strict=True)
    ]
    lines += ["", MODEL_RELATIVE_CAVEAT]
    return "\n".join(lines)
