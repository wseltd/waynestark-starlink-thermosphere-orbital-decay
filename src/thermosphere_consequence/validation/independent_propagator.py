"""A SECOND, in-house orbit propagator, written to disagree with the first.

WHAT "INDEPENDENT" MEANS HERE, EXACTLY
--------------------------------------
This module is an in-house second implementation. It is not a wrapper around a
third-party tool and it is not a port of ``propagation.propagator``. It shares
no line of code with that module. It also does not share its *formulation*:

  * ``propagation.propagator.DragPropagator`` is a Cowell integrator. Its state
    is Cartesian position and velocity in an Earth-centred inertial frame, its
    perturbations are Cartesian accelerations, and its integrator is SciPy's
    adaptive-step DOP853.

  * ``GaussPropagator`` below is a variation-of-parameters integrator. Its state
    is the six classical orbital elements, its perturbations enter through
    Gauss's variational equations in the rotating RSW (radial / along-track /
    cross-track) frame, and its integrator is a fixed-step classical RK4
    written out by hand in this file. Its geodetic-altitude routine is
    Heikkinen's closed-form inversion, not Bowring's approximation.

Because the two differ in state variables, in the algebraic form of every
perturbation, and in the integrator, an agreement between them is evidence that
neither contains an implementation error: a sign slip, a transposed rotation, a
missing factor of two or a unit mistake would have to occur identically in two
different algebras to cancel.

WHAT IT DOES NOT ESTABLISH
--------------------------
Both implementations rest on the SAME physics literature: Newtonian two-body
gravity, the J2 zonal term of the geopotential, and a free-molecular drag law
``a = -(1/2) (C_d A / m) rho |v_rel| v_rel`` with a rigidly co-rotating
atmosphere. Both neglect J3 and higher zonals, third-body attraction, solar
radiation pressure, thermospheric winds and relativistic terms. Both treat the
integration frame as inertial.

Agreement between them therefore catches IMPLEMENTATION error and says nothing
at all about a shared CONCEPTUAL error. If the drag law itself is the wrong
model for this vehicle, or the neglected thermospheric wind matters at the
percent level, both propagators are wrong together and this comparison will
report a clean pass. That limitation is structural and cannot be removed by
tightening any tolerance in this module.

The physical CONSTANTS are deliberately shared with the rest of the package
rather than restated here. A restated constant that drifted by a digit would
show up as a spurious disagreement between the propagators and would be read as
a propagation bug, which is the opposite of what this module is for. Constants
are data; independence is claimed for the code and the formulation.

THE KNOWN SINGULARITIES ARE GUARDED, NOT HIDDEN
-----------------------------------------------
Classical elements are singular for circular orbits (the argument of perigee is
undefined) and for equatorial orbits (the node is undefined). That is a
property of the coordinates, not of the physics, and it is exactly why
equinoctial elements exist. Rather than silently returning nonsense near those
limits, the propagator raises with a message naming the singularity. Runs are
therefore set up away from them, and the minimum eccentricity actually reached
is recorded in every result so a reader can see how close a run came.

PRE-DECLARED TOLERANCES
-----------------------
``PRE_DECLARED_TOLERANCES`` below fixes, in one place, the pass/fail limit for
every validation comparison together with the reasoning that set it. The test
suite and the reporting script both read it, so the number asserted in a test
and the number printed in the evidence report cannot drift apart, and neither
can be edited after a measurement without editing the justification beside it.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..domain.constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH_EQUATORIAL

__all__ = [
    "ClassicalElements",
    "ConstantAtmosphere",
    "ExponentialAtmosphere",
    "GaussPropagationResult",
    "GaussPropagator",
    "PRE_DECLARED_TOLERANCES",
    "PreDeclaredTolerance",
    "SingularElementsError",
    "TrajectorySamples",
    "cartesian_from_elements",
    "compare_trajectories",
    "earth_rotation_angle_rad",
    "elements_from_cartesian",
    "geodetic_altitude_heikkinen",
    "geodetic_latitude_longitude_heikkinen",
    "greenwich_mean_sidereal_time_rad",
    "j2_rsw_acceleration",
    "j2_secular_rates",
    "relative_velocity_rsw",
    "secular_rate_fit",
]

MU = MU_EARTH.value
J2 = J2_EARTH.value
R_EQUATORIAL_M = R_EARTH_EQUATORIAL.value
OMEGA_EARTH_RAD_S = OMEGA_EARTH.value

#: WGS-84 flattening. Definitional, exact, and restated rather than imported
#: because ``domain.constants`` carries no ellipsoid shape parameter.
FLATTENING = 1.0 / 298.257223563
R_POLAR_M = R_EQUATORIAL_M * (1.0 - FLATTENING)
ECCENTRICITY_SQUARED = FLATTENING * (2.0 - FLATTENING)
SECOND_ECCENTRICITY_SQUARED = (
    R_EQUATORIAL_M * R_EQUATORIAL_M - R_POLAR_M * R_POLAR_M
) / (R_POLAR_M * R_POLAR_M)

#: Matches ``propagation.propagator.REENTRY_ALTITUDE_M``. Restated rather than
#: imported so the two propagators do not inherit a shared threshold by
#: accident; ``test_independent_propagator`` asserts the two values are equal,
#: which turns the duplication into a checked invariant instead of a risk.
DEFAULT_REENTRY_ALTITUDE_M = 100_000.0

#: Above this geodetic altitude the run is refusing to be a decay study.
MAX_ALTITUDE_M = 2_000_000.0

#: Below this eccentricity the argument of perigee is not defined and the
#: 1/e terms in Gauss's equations diverge. Guarded, not clamped: clamping would
#: turn an undefined quantity into a plausible number.
MIN_ECCENTRICITY = 1.0e-6

#: Below this |sin i| the right ascension of the ascending node is not defined
#: and the 1/sin(i) terms in Gauss's equations diverge.
MIN_SIN_INCLINATION = 1.0e-6

#: Below this eccentricity the omega/nu split is still defined but badly
#: conditioned: the two 1/(h e) terms are individually large and nearly cancel,
#: and d(omega)/dt approaches the mean motion. Results carry a note when a run
#: goes below it.
ILL_CONDITIONED_ECCENTRICITY = 1.0e-4

#: Bisection on the terminating altitude event stops at this bracket width.
#: One microsecond of LEO motion is 8 micrometres, far below anything compared.
EVENT_BRACKET_TOLERANCE_S = 1.0e-6

DensityFunction = Callable[[_dt.datetime, float, float, float], float]

#: J2000.0, the epoch both sidereal-time formulations are referred to.
J2000_EPOCH = _dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=_dt.UTC)
JULIAN_DATE_J2000 = 2451545.0
JULIAN_CENTURY_DAYS = 36525.0
ARCSECONDS_PER_RADIAN = 648000.0 / math.pi


class SingularElementsError(RuntimeError):
    """Raised when a state reaches a coordinate singularity of classical elements."""


@dataclass(frozen=True, slots=True)
class PreDeclaredTolerance:
    """One pass/fail limit, fixed before any measurement was taken.

    ``justification`` is not decoration. A tolerance without a stated reason can
    be widened after a failure and nobody can tell; a tolerance whose reason
    names the physical effect that sets it cannot be widened without the edit
    being visible in the same commit.
    """

    test_id: str
    quantity: str
    limit: float
    unit: str
    justification: str


def _tolerance(
    test_id: str, quantity: str, limit: float, unit: str, justification: str
) -> tuple[str, PreDeclaredTolerance]:
    key = f"{test_id}.{quantity}"
    return key, PreDeclaredTolerance(test_id, quantity, limit, unit, justification)


PRE_DECLARED_TOLERANCES: dict[str, PreDeclaredTolerance] = dict(
    [
        _tolerance(
            "T1",
            "scipy_energy_drift_relative",
            1.0e-8,
            "relative",
            "Two-body only: specific energy -mu/(2a) is an exact invariant, so any "
            "drift is integrator error. 1e-8 relative is 6.6 cm of semi-major axis on "
            "a 6.6e6 m orbit, three orders below the metre-scale differences the "
            "cross-propagator comparison is built to resolve, and it is the bound the "
            "existing suite already holds DOP853 to over one day.",
        ),
        _tolerance(
            "T1",
            "gauss_energy_drift_relative",
            1.0e-14,
            "relative",
            "In the variational formulation the semi-major axis IS a state variable "
            "and its derivative is a product with the perturbing acceleration, so with "
            "drag and J2 disabled that derivative is exactly 0.0 and a is bitwise "
            "constant. The limit is set at machine epsilon rather than at zero only to "
            "leave room for the accumulation of an exact zero, and a value far above "
            "it would indicate the perturbation switches are not actually off.",
        ),
        _tolerance(
            "T2",
            "secular_rate_vs_analytic_relative",
            5.0e-3,
            "relative",
            "The analytic node and perigee rates are FIRST order in J2. The neglected "
            "second-order terms are O(J2) = 1.08e-3 relative with an O(1) coefficient "
            "that first-order theory does not pin down, and the least-squares fit to a "
            "finite arc of OSCULATING angles carries a residual short-period term. "
            "5e-3 is roughly five times the leading neglected term: loose enough not "
            "to fail on the known truncation, tight enough that a wrong factor "
            "(3/2 vs 3/4, cos i vs cos^2 i) could not pass.",
        ),
        _tolerance(
            "T2",
            "secular_rate_propagator_agreement_relative",
            1.0e-4,
            "relative",
            "The two propagators integrate the same J2 field, so their fitted secular "
            "rates must agree far more closely than either agrees with first-order "
            "theory. 1e-4 relative leaves room for the different fitting residuals "
            "(one propagator's angles come from osculating elements recovered from "
            "Cartesian samples, the other's are integrated directly) while still being "
            "an order of magnitude inside the analytic comparison.",
        ),
        _tolerance(
            "T3",
            "drag_da_dt_vs_analytic_relative",
            1.0e-3,
            "relative",
            "The closed form da/dt = -(C_d A/m) rho sqrt(mu a) is exact for a circular "
            "orbit under uniform density. The test orbit carries e = 5e-3, whose "
            "orbit-averaged correction is (1 + 0.75 e^2) = 1 + 1.9e-5; evaluating the "
            "analytic rate at the mean semi-major axis of the window removes the "
            "first-order effect of a decaying by da, leaving O((da/a)^2) ~ 1e-7. 1e-3 "
            "is fifty times the largest identified error term and would still catch a "
            "factor-of-two or a missing sqrt.",
        ),
        _tolerance(
            "T4",
            "rk4_observed_order",
            0.4,
            "orders (absolute deviation from 4)",
            "Classical RK4 is fourth-order accurate, so the Richardson estimate "
            "log2(|y_h - y_h/2| / |y_h/2 - y_h/4|) must approach 4. An allowance of "
            "0.4 covers the step sizes still being finite (the asymptotic regime is "
            "approached, not reached) without admitting a third- or fifth-order "
            "result, which is what a mis-weighted RK4 stage would produce.",
        ),
        _tolerance(
            "T4",
            "rk4_error_reduction_factor",
            8.0,
            "factor (minimum)",
            "Each halving of the step must shrink the difference between successive "
            "solutions by at least 8x. Fourth order predicts 16x; the floor is set at "
            "half that so a not-yet-asymptotic coarsest pair cannot fail the check, "
            "while a second-order (4x) or first-order (2x) method could not pass it.",
        ),
        _tolerance(
            "T5",
            "scipy_tightest_pair_position_m",
            5.0,
            "metre",
            "Difference in final position between the two tightest tolerance settings, "
            "with max_step relaxed to 600 s so that rtol/atol -- and not the step cap "
            "-- is what actually controls the step. A per-step relative error of 1e-10 "
            "on a 6.6e6 m radius is 6.6e-4 m and along-track error grows by roughly "
            "3 n t = 74 over four orbits, so a fraction of a metre is expected. 5 m is "
            "a ten-fold allowance on that order-of-magnitude estimate.",
        ),
        _tolerance(
            "T5",
            "scipy_convergence_ratio",
            10.0,
            "factor (minimum)",
            "Agreement alone does not show convergence: two settings could agree "
            "because both are stuck. With max_step relaxed to 600 s the loosest pair "
            "of settings must differ by at least 10x more than the tightest pair, "
            "which is the signature of a solution actually approaching a limit.",
        ),
        _tolerance(
            "T5",
            "scipy_production_setting_spread_m",
            5.0,
            "metre",
            "Separate statement about the settings the science runs actually use. At "
            "max_step = 30 s, DOP853 (an eighth-order method) already takes steps far "
            "finer than any of these tolerances demands, so the tolerance should be "
            "INERT: sweeping rtol/atol over six orders of magnitude must move the final "
            "position by less than 5 m. If it moved more, the production runs would be "
            "silently tolerance-limited.",
        ),
        _tolerance(
            "T6",
            "threshold_spread_fraction_of_decay",
            0.10,
            "relative",
            "The scientific question is whether the reentry threshold is a free "
            "parameter that moves the answer. Under an exponential atmosphere the "
            "closed form (H / (B rho0 sqrt(mu a))) [exp((h1-h0)/H) - exp((h2-h0)/H)] "
            "puts the 120 km -> 80 km descent at [exp(-3.25) - exp(-4.25)] / "
            "[1 - exp(-3.75)] = 2.5% of the decay from 250 km. The limit is set at four "
            "times that to allow for the effects the closed form omits, named in the "
            "tolerance below.",
        ),
        _tolerance(
            "T6",
            "threshold_spread_vs_analytic_relative",
            0.60,
            "relative",
            "Comparison of the measured (spread / decay time) against the closed-form "
            "ratio, with the expected value referred to PERIGEE altitude because the "
            "threshold is crossed at perigee: the start altitude entering the "
            "denominator is a(1-e) - Re, about a e = 6.6 km below the mean. The RATIO "
            "rather than the raw spread is compared so the common factor "
            "H / (B rho0 sqrt(mu a)) drops out. 60% is wide, and deliberately so: the "
            "closed form is ORBIT-AVERAGED and quasi-static, and near the floor the "
            "semi-major axis falls by kilometres per revolution, so the last forty "
            "kilometres are covered in only a few revolutions. The J2 short-period "
            "radius oscillation (~3 km) displaces the crossings further. The check "
            "remains meaningful -- a mishandled threshold would be wrong by a factor, "
            "not by 60% -- but it is an order-of-magnitude anchor and is reported as "
            "one.",
        ),
        _tolerance(
            "T7",
            "geodetic_altitude_agreement_m",
            1.0,
            "metre",
            "Bowring's non-iterative latitude approximation and Heikkinen's exact "
            "closed-form inversion solve the same equation by different routes, so any "
            "difference is one method's truncation error. 1 m is the requirement level "
            "for this comparison; at a 40 km density scale height 1 m of altitude is "
            "2.5e-5 in density, which is negligible against every other uncertainty "
            "here.",
        ),
        _tolerance(
            "T7",
            "heikkinen_roundtrip_m",
            1.0e-6,
            "metre",
            "Independent anchor for the new routine that does not involve the old one: "
            "a point placed at height h along the ellipsoid normal at a known geodetic "
            "latitude must invert back to h. Heikkinen's solution is algebraically "
            "exact, so the residual is pure floating-point round-off, ~1e-9 m on a "
            "6.4e6 m radius. 1e-6 m is a thousand-fold allowance on that.",
        ),
        _tolerance(
            "T7",
            "sidereal_time_agreement_rad",
            1.0e-6,
            "radian",
            "The IAU-1982 GMST polynomial and the IAU-2006 Earth-rotation-angle plus "
            "precession-in-right-ascension form are different constructions of the "
            "same angle and agree at the millisecond-of-time level, i.e. ~7e-8 rad. "
            "1e-6 rad is 6.4 m on the ground, below any longitude structure a density "
            "model resolves.",
        ),
        _tolerance(
            "T8",
            "corotation_speed_m_s",
            1.0e-6,
            "metre/second",
            "For an eastward circular equatorial orbit the co-rotation subtraction is "
            "exactly |v| - omega r, a scalar identity, so the only admissible error is "
            "floating-point round-off. 1e-6 m/s is far above round-off (1e-10 m/s) and "
            "far below the 480 m/s effect being verified.",
        ),
        _tolerance(
            "T8",
            "corotation_hand_value_m_s",
            0.1,
            "metre/second",
            "Hand check on the magnitude itself, independent of any code: "
            "omega r = 7.2921150e-5 rad/s x 6588137 m = 480.4 m/s at 210 km. 0.1 m/s "
            "is the rounding of that hand arithmetic to four significant figures.",
        ),
        _tolerance(
            "T9",
            "sgp4_max_separation_m",
            10_000.0,
            "metre",
            "Kinematic sanity only. SGP4 carries J3, J4 and a fitted B*; the "
            "independent propagator carries J2 alone, so the two dynamical models "
            "genuinely differ and the comparison bounds the size, shape and "
            "orientation of the orbit rather than its precision. Over a one-hour arc "
            "the J4 contribution alone is ~300 m along-track and SGP4's own truncation "
            "is kilometre-class; 10 km is 1.5e-3 of the orbit circumference and would "
            "still catch a wrong inclination, a swapped node or a units error.",
        ),
        _tolerance(
            "T9",
            "sgp4_rms_radial_m",
            3_000.0,
            "metre",
            "Radial separation is the component that reflects a difference in "
            "ALTITUDE, which is what a drag or geopotential disagreement looks like; "
            "the total is dominated by along-track phasing. J3 and J4 produce radial "
            "differences of order hundreds of metres over an hour, so 3 km is a "
            "ten-fold allowance and is an order of magnitude tighter than the total.",
        ),
        _tolerance(
            "T10",
            "cross_semi_major_axis_m",
            100.0,
            "metre",
            "The two propagators integrate identical physics with the same injected "
            "density function, so the only admissible differences are integration "
            "error (RK4 at 5 s gives sub-metre over a day, DOP853 at rtol 1e-11 far "
            "less) and the ~1 m disagreement between the two geodetic-altitude "
            "routines, which at a 40 km scale height perturbs density by 2.5e-5 and "
            "the day's decay by well under a metre. 100 m is 1.5e-5 relative and is a "
            "two-order allowance over the identified terms.",
        ),
        _tolerance(
            "T10",
            "cross_perigee_altitude_m",
            500.0,
            "metre",
            "Perigee and apogee combine the semi-major-axis difference with the "
            "eccentricity difference, and for e ~ 1e-2 an eccentricity difference of "
            "1e-5 already moves each apsis by 66 m. 500 m allows for both while "
            "staying far inside the 20-40 km apsis changes the decay itself produces.",
        ),
        _tolerance(
            "T10",
            "cross_apogee_altitude_m",
            500.0,
            "metre",
            "The same two element differences as the perigee bound, entering with the "
            "opposite sign: a(1+e) rather than a(1-e). It is declared separately rather "
            "than folded into one apsis bound because drag acts asymmetrically -- it "
            "lowers apogee far faster than perigee -- so a defect in the drag model "
            "would show at apogee first, and a single combined bound would let the "
            "quiet apsis mask the moving one.",
        ),
        _tolerance(
            "T10",
            "cross_radial_position_m",
            500.0,
            "metre",
            "Radial separation, unlike total separation, is not inflated by along-track "
            "phasing. It receives the semi-major-axis difference directly (<= 100 m) "
            "plus the projection of the phase offset onto the orbit's own radial "
            "variation: at e ~ 1e-2 the radius swings by a e = 66 km per revolution, "
            "so a 3e-3 rad phase offset contributes ~200 m. 500 m covers both.",
        ),
        _tolerance(
            "T10",
            "cross_total_position_m",
            20_000.0,
            "metre",
            "DERIVED FROM the semi-major-axis bound, not chosen: a semi-major-axis "
            "difference da makes the mean motions differ by 1.5 n da / a, so the "
            "along-track separation after time t is about 1.5 n da t. With da growing "
            "linearly to the declared 100 m over one day this is 1.5 x 1.16e-3 x 50 x "
            "86400 = 7.5 km. 20 km carries the factor-of-two-plus margin appropriate "
            "to an order-of-magnitude derivation. This number bounds PHASING, not "
            "altitude, and on its own it is a weak statement -- which is why the "
            "semi-major-axis, apsis, radial and energy bounds are declared separately.",
        ),
        _tolerance(
            "T10",
            "cross_total_velocity_m_s",
            30.0,
            "metre/second",
            "Also phase-dominated and derived from the position bound: two states "
            "separated along-track by s have velocity vectors rotated by s/a, so "
            "|dv| ~ v s / a = 7800 x 20000 / 6.6e6 = 23.6 m/s at the declared position "
            "limit. 30 m/s is that with rounding up.",
        ),
        _tolerance(
            "T10",
            "cross_speed_m_s",
            2.0,
            "metre/second",
            "The SPEED difference, unlike the velocity-vector difference, is nearly "
            "free of phasing: it takes the semi-major-axis difference through vis-viva "
            "(~0.06 m/s at 100 m) plus the projection of the phase offset onto the "
            "orbit's own speed variation (e v = 78 m/s per revolution at e = 1e-2, so "
            "~0.23 m/s at a 3e-3 rad offset). 2 m/s is a five-fold allowance.",
        ),
        _tolerance(
            "T10",
            "cross_energy_loss_relative",
            0.01,
            "relative",
            "Energy lost to drag is the integral of the drag work, so it is the "
            "quantity the whole counterfactual depends on. The two propagators see the "
            "same density function at positions differing by ~1 m in altitude "
            "(2.5e-5 in density), so 1% is a four-hundred-fold allowance over the "
            "identified difference and would fail loudly on any real disagreement in "
            "the drag law.",
        ),
        _tolerance(
            "T10",
            "cross_decay_time_relative",
            0.01,
            "relative",
            "The decay time is the headline output. It is set by the integrated energy "
            "loss, so its admissible difference follows the energy bound: 1%. On a "
            "seven-day decay that is 100 minutes, and the analytic bracket this "
            "comparison replaces was 3.16-8.78 days -- a factor of 2.8 -- so 1% is "
            "roughly two hundred times stronger a statement.",
        ),
    ]
)


# --------------------------------------------------------------------------
# Closed-form geodesy: Heikkinen, not Bowring
# --------------------------------------------------------------------------


def geodetic_altitude_heikkinen(position_m: np.ndarray) -> float:
    """Geodetic altitude above WGS-84 by Heikkinen's exact closed-form inversion.

    Bowring's method, used by the Cowell propagator, obtains the geodetic
    latitude from a single application of a very good approximation and is
    therefore accurate but truncated. Heikkinen's construction instead solves
    the quartic that relates the geocentric and geodetic radii exactly, in
    closed form, with no iteration and no series truncation.

    Using a different inversion is deliberate. Altitude is the argument of the
    density model, so an altitude error is a density error is a decay-rate
    error; two routes to the same number is the only way to notice that the
    first route was wrong.
    """
    x, y, z = float(position_m[0]), float(position_m[1]), float(position_m[2])
    a = R_EQUATORIAL_M
    b = R_POLAR_M
    e2 = ECCENTRICITY_SQUARED
    e4 = e2 * e2

    p = math.hypot(x, y)
    f_term = 54.0 * b * b * z * z
    g_term = p * p + (1.0 - e2) * z * z - e2 * (a * a - b * b)
    if g_term == 0.0:
        raise ValueError(
            f"position {(x, y, z)} lies on the ellipsoid's focal disc, where the "
            "closed-form inversion is undefined; no orbital state can reach it"
        )
    c_term = e4 * f_term * p * p / (g_term**3)
    s_term = (1.0 + c_term + math.sqrt(c_term * c_term + 2.0 * c_term)) ** (1.0 / 3.0)
    k_term = s_term + 1.0 + 1.0 / s_term
    p_term = f_term / (3.0 * k_term * k_term * g_term * g_term)
    q_term = math.sqrt(1.0 + 2.0 * e4 * p_term)

    inner = (
        0.5 * a * a * (1.0 + 1.0 / q_term)
        - p_term * (1.0 - e2) * z * z / (q_term * (1.0 + q_term))
        - 0.5 * p_term * p * p
    )
    r0 = -(p_term * e2 * p) / (1.0 + q_term) + math.sqrt(max(inner, 0.0))

    u_term = math.sqrt((p - e2 * r0) ** 2 + z * z)
    v_term = math.sqrt((p - e2 * r0) ** 2 + (1.0 - e2) * z * z)
    if v_term == 0.0:
        raise ValueError(f"position {(x, y, z)} is at the Earth's centre; altitude is undefined")
    return u_term * (1.0 - b * b / (a * v_term))


def geodetic_latitude_longitude_heikkinen(
    position_m: np.ndarray, sidereal_time_rad: float
) -> tuple[float, float]:
    """Geodetic latitude and east longitude in degrees, by the same closed form.

    Longitude is the inertial right ascension minus the sidereal rotation of the
    Greenwich meridian; latitude comes from Heikkinen's intermediates rather
    than from a second, differently-approximated routine.
    """
    x, y, z = float(position_m[0]), float(position_m[1]), float(position_m[2])
    a = R_EQUATORIAL_M
    b = R_POLAR_M
    e2 = ECCENTRICITY_SQUARED
    e4 = e2 * e2
    ep2 = SECOND_ECCENTRICITY_SQUARED

    p = math.hypot(x, y)
    f_term = 54.0 * b * b * z * z
    g_term = p * p + (1.0 - e2) * z * z - e2 * (a * a - b * b)
    c_term = e4 * f_term * p * p / (g_term**3)
    s_term = (1.0 + c_term + math.sqrt(c_term * c_term + 2.0 * c_term)) ** (1.0 / 3.0)
    k_term = s_term + 1.0 + 1.0 / s_term
    p_term = f_term / (3.0 * k_term * k_term * g_term * g_term)
    q_term = math.sqrt(1.0 + 2.0 * e4 * p_term)
    inner = (
        0.5 * a * a * (1.0 + 1.0 / q_term)
        - p_term * (1.0 - e2) * z * z / (q_term * (1.0 + q_term))
        - 0.5 * p_term * p * p
    )
    r0 = -(p_term * e2 * p) / (1.0 + q_term) + math.sqrt(max(inner, 0.0))
    v_term = math.sqrt((p - e2 * r0) ** 2 + (1.0 - e2) * z * z)
    z0 = b * b * z / (a * v_term)

    latitude_rad = math.atan2(z + ep2 * z0, p)
    longitude_rad = math.atan2(y, x) - sidereal_time_rad
    longitude_deg = (math.degrees(longitude_rad) + 180.0) % 360.0 - 180.0
    return math.degrees(latitude_rad), longitude_deg


def _days_since_j2000(when: _dt.datetime) -> float:
    """Days from J2000.0, straight from the datetime difference.

    ``datetime`` already implements the proleptic Gregorian calendar exactly, so
    a hand-rolled Julian-date polynomial would only add a place to make an
    arithmetic mistake.
    """
    if when.tzinfo is None:
        raise ValueError(f"timestamp {when!r} is naive; validation requires explicit UTC")
    return (when.astimezone(_dt.UTC) - J2000_EPOCH).total_seconds() / 86400.0


def earth_rotation_angle_rad(when: _dt.datetime) -> float:
    """IAU-2000 Earth rotation angle: a linear function of UT1 by definition."""
    d = _days_since_j2000(when)
    turns = 0.7790572732640 + 1.00273781191135448 * d
    return 2.0 * math.pi * (turns % 1.0)


def greenwich_mean_sidereal_time_rad(when: _dt.datetime) -> float:
    """GMST via the IAU-2006 route: rotation angle plus precession in right ascension.

    The Cowell propagator uses the IAU-1982 GMST polynomial. This is the other
    standard construction of the same angle, and the two are compared in the
    test suite rather than assumed equal -- an error here would displace the
    sub-satellite longitude and, for any longitude-dependent density model,
    silently move the vehicle into the wrong atmosphere.
    """
    t = _days_since_j2000(when) / JULIAN_CENTURY_DAYS
    precession_arcsec = (
        0.014506
        + 4612.156534 * t
        + 1.3915817 * t * t
        - 0.00000044 * t**3
        - 0.000029956 * t**4
    )
    gmst = earth_rotation_angle_rad(when) + precession_arcsec / ARCSECONDS_PER_RADIAN
    return gmst % (2.0 * math.pi)


# --------------------------------------------------------------------------
# Classical elements and the RSW frame
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassicalElements:
    """The six classical orbital elements, in metres and radians.

    ``true_anomaly_rad`` is stored UNWRAPPED during integration. Wrapping a fast
    angle inside an RK4 stage would put a 2-pi discontinuity in the middle of a
    step and quietly destroy the method's order.
    """

    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    argument_of_perigee_rad: float
    true_anomaly_rad: float

    def __post_init__(self) -> None:
        if self.semi_major_axis_m <= 0.0:
            raise ValueError(
                f"semi-major axis must be positive, got {self.semi_major_axis_m}"
            )
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError(
                f"eccentricity must lie in [0, 1) for a bound orbit, got {self.eccentricity}"
            )
        if self.eccentricity < MIN_ECCENTRICITY:
            raise ValueError(
                f"eccentricity {self.eccentricity} is below {MIN_ECCENTRICITY}: the "
                "argument of perigee is undefined for a circular orbit and Gauss's "
                "variational equations carry a 1/e factor. This is a singularity of "
                "classical elements, not of the physics; use an eccentricity above "
                "the threshold or a set of equinoctial elements."
            )
        if abs(math.sin(self.inclination_rad)) < MIN_SIN_INCLINATION:
            raise ValueError(
                f"inclination {self.inclination_rad} rad has |sin i| below "
                f"{MIN_SIN_INCLINATION}: the ascending node is undefined for an "
                "equatorial orbit and Gauss's variational equations carry a 1/sin(i) "
                "factor. This is a singularity of classical elements, not of the "
                "physics."
            )

    @property
    def semi_latus_rectum_m(self) -> float:
        return self.semi_major_axis_m * (1.0 - self.eccentricity**2)

    @property
    def angular_momentum_m2_s(self) -> float:
        return math.sqrt(MU * self.semi_latus_rectum_m)

    @property
    def radius_m(self) -> float:
        return self.semi_latus_rectum_m / (
            1.0 + self.eccentricity * math.cos(self.true_anomaly_rad)
        )

    @property
    def argument_of_latitude_rad(self) -> float:
        return self.argument_of_perigee_rad + self.true_anomaly_rad

    @property
    def mean_motion_rad_s(self) -> float:
        return math.sqrt(MU / self.semi_major_axis_m**3)

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.mean_motion_rad_s

    @property
    def specific_energy_j_kg(self) -> float:
        return -MU / (2.0 * self.semi_major_axis_m)

    @property
    def perigee_altitude_m(self) -> float:
        return self.semi_major_axis_m * (1.0 - self.eccentricity) - R_EQUATORIAL_M

    @property
    def apogee_altitude_m(self) -> float:
        return self.semi_major_axis_m * (1.0 + self.eccentricity) - R_EQUATORIAL_M

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.semi_major_axis_m,
                self.eccentricity,
                self.inclination_rad,
                self.raan_rad,
                self.argument_of_perigee_rad,
                self.true_anomaly_rad,
            ],
            dtype=float,
        )


def _rsw_basis(
    raan_rad: float, inclination_rad: float, argument_of_latitude_rad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial, along-track and cross-track unit vectors in the inertial frame.

    Built directly as the three columns of the 3-1-3 rotation rather than by
    composing rotation matrices, so the sign of every entry is visible.
    """
    cos_o, sin_o = math.cos(raan_rad), math.sin(raan_rad)
    cos_i, sin_i = math.cos(inclination_rad), math.sin(inclination_rad)
    cos_u, sin_u = math.cos(argument_of_latitude_rad), math.sin(argument_of_latitude_rad)

    radial = np.array(
        [
            cos_o * cos_u - sin_o * sin_u * cos_i,
            sin_o * cos_u + cos_o * sin_u * cos_i,
            sin_u * sin_i,
        ]
    )
    along = np.array(
        [
            -cos_o * sin_u - sin_o * cos_u * cos_i,
            -sin_o * sin_u + cos_o * cos_u * cos_i,
            cos_u * sin_i,
        ]
    )
    cross = np.array([sin_o * sin_i, -cos_o * sin_i, cos_i])
    return radial, along, cross


def cartesian_from_elements(elements: ClassicalElements) -> tuple[np.ndarray, np.ndarray]:
    """Inertial position and velocity from the six classical elements.

    Assembled from the RSW basis rather than from a perifocal vector rotated by
    a fused matrix: the radial and along-track velocity components of a Keplerian
    orbit are (mu/h) e sin(nu) and h/r, and writing them in that frame keeps the
    construction in the same algebra the perturbations are expressed in.
    """
    radial, along, _ = _rsw_basis(
        elements.raan_rad, elements.inclination_rad, elements.argument_of_latitude_rad
    )
    r = elements.radius_m
    h = elements.angular_momentum_m2_s
    v_radial = MU / h * elements.eccentricity * math.sin(elements.true_anomaly_rad)
    v_along = h / r
    return r * radial, v_radial * radial + v_along * along


def elements_from_cartesian(
    position_m: np.ndarray, velocity_m_s: np.ndarray
) -> ClassicalElements:
    """Classical elements from an inertial Cartesian state.

    Uses the node vector and the eccentricity vector, with every angle recovered
    through ``atan2`` of an in-plane pair rather than through ``acos`` plus a
    sign test: ``acos`` loses precision near 0 and pi, which is exactly where a
    near-circular decaying orbit spends its time.
    """
    r_vec = np.asarray(position_m, dtype=float)
    v_vec = np.asarray(velocity_m_s, dtype=float)
    r = float(np.linalg.norm(r_vec))
    v = float(np.linalg.norm(v_vec))
    if r == 0.0:
        raise ValueError("position vector is zero; no orbit is defined")

    energy = 0.5 * v * v - MU / r
    if energy >= 0.0:
        raise ValueError(
            f"specific orbital energy is {energy:.6e} J/kg, which is not negative; "
            "the state is unbound and has no semi-major axis"
        )
    a = -MU / (2.0 * energy)

    h_vec = np.cross(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))
    h_hat = h_vec / h
    inclination = math.acos(max(-1.0, min(1.0, float(h_hat[2]))))

    node_vec = np.array([-float(h_vec[1]), float(h_vec[0]), 0.0])
    node = float(np.linalg.norm(node_vec))
    if node == 0.0:
        raise ValueError(
            "the orbit is exactly equatorial, so the ascending node is undefined; "
            "classical elements are singular here"
        )
    node_hat = node_vec / node

    e_vec = ((v * v - MU / r) * r_vec - float(np.dot(r_vec, v_vec)) * v_vec) / MU
    eccentricity = float(np.linalg.norm(e_vec))
    if eccentricity == 0.0:
        raise ValueError(
            "the orbit is exactly circular, so the argument of perigee is undefined; "
            "classical elements are singular here"
        )
    e_hat = e_vec / eccentricity

    raan = math.atan2(float(node_hat[1]), float(node_hat[0]))
    in_plane = np.cross(h_hat, node_hat)
    argument_of_perigee = math.atan2(float(np.dot(e_hat, in_plane)), float(np.dot(e_hat, node_hat)))
    ahead_of_perigee = np.cross(h_hat, e_hat)
    true_anomaly = math.atan2(
        float(np.dot(r_vec, ahead_of_perigee)), float(np.dot(r_vec, e_hat))
    )
    return ClassicalElements(
        semi_major_axis_m=a,
        eccentricity=eccentricity,
        inclination_rad=inclination,
        raan_rad=raan % (2.0 * math.pi),
        argument_of_perigee_rad=argument_of_perigee % (2.0 * math.pi),
        true_anomaly_rad=true_anomaly % (2.0 * math.pi),
    )


# --------------------------------------------------------------------------
# Perturbations, in RSW components
# --------------------------------------------------------------------------


def j2_rsw_acceleration(
    radius_m: float, inclination_rad: float, argument_of_latitude_rad: float
) -> tuple[float, float, float]:
    """J2 perturbing acceleration resolved in the RSW frame.

    Derived, not transcribed. Writing the geopotential's J2 term as

        U_J2 = -(mu J2 Re^2 / 2) (3 z^2 / r^5 - 1 / r^3)

    and taking its gradient gives

        a_J2 = (3 mu J2 Re^2 / (2 r^4)) [(5 sin^2(phi) - 1) r_hat - 2 sin(phi) z_hat]

    where sin(phi) = z/r = sin(i) sin(u). Projecting z_hat onto the RSW basis --
    z_hat . r_hat = sin(i) sin(u), z_hat . s_hat = sin(i) cos(u), z_hat . w_hat =
    cos(i) -- collapses that to the three components below.

    This is the same physical field the Cowell propagator applies, reached
    through a completely different algebra: no rotation matrix is inverted and
    no Cartesian component ever appears.
    """
    factor = 1.5 * J2 * MU * R_EQUATORIAL_M * R_EQUATORIAL_M / radius_m**4
    sin_i = math.sin(inclination_rad)
    sin_u = math.sin(argument_of_latitude_rad)
    cos_u = math.cos(argument_of_latitude_rad)
    radial = -factor * (1.0 - 3.0 * sin_i * sin_i * sin_u * sin_u)
    along = -factor * 2.0 * sin_i * sin_i * sin_u * cos_u
    cross = -factor * 2.0 * sin_i * math.cos(inclination_rad) * sin_u
    return radial, along, cross


def relative_velocity_rsw(
    elements: ClassicalElements, *, include_atmosphere_rotation: bool
) -> tuple[float, float, float]:
    """Velocity of the vehicle relative to a rigidly co-rotating atmosphere, in RSW.

    The inertial velocity has components ((mu/h) e sin(nu), h/r, 0). The
    co-rotation term omega x r, with omega along the inertial z-axis, resolves
    as omega r (cos(i) s_hat - sin(i) cos(u) w_hat) -- a cross-track component
    appears because the atmosphere's motion is not confined to the orbit plane
    unless the orbit is equatorial. Dropping that component is a common and
    invisible error; it is why this is written out rather than assumed parallel
    to the along-track direction.
    """
    r = elements.radius_m
    h = elements.angular_momentum_m2_s
    v_radial = MU / h * elements.eccentricity * math.sin(elements.true_anomaly_rad)
    v_along = h / r
    v_cross = 0.0
    if include_atmosphere_rotation:
        omega_r = OMEGA_EARTH_RAD_S * r
        v_along -= omega_r * math.cos(elements.inclination_rad)
        v_cross += omega_r * math.sin(elements.inclination_rad) * math.cos(
            elements.argument_of_latitude_rad
        )
    return v_radial, v_along, v_cross


def gauss_variational_derivatives(
    elements: ClassicalElements, radial: float, along: float, cross: float
) -> np.ndarray:
    """Gauss's variational equations for the six classical elements.

    With p the semi-latus rectum, h the specific angular momentum, r the radius
    and u = omega + nu the argument of latitude, for a perturbing acceleration
    (R, S, W) in the radial / along-track / cross-track frame:

        da/dt  = (2 a^2 / h) [e sin(nu) R + (p/r) S]
        de/dt  = (1/h) [p sin(nu) R + ((p + r) cos(nu) + r e) S]
        di/dt  = (r cos(u) / h) W
        dRAAN/dt = (r sin(u) / (h sin i)) W
        domega/dt = (1/(h e)) [-p cos(nu) R + (p + r) sin(nu) S]
                    - (r sin(u) cos(i) / (h sin i)) W
        dnu/dt = h/r^2 + (1/(h e)) [p cos(nu) R - (p + r) sin(nu) S]

    Note that the two 1/(he) terms in domega/dt and dnu/dt are equal and
    opposite, so their sum -- the argument of latitude, which is what actually
    positions the vehicle -- stays well conditioned even where omega alone does
    not.
    """
    e = elements.eccentricity
    p = elements.semi_latus_rectum_m
    h = elements.angular_momentum_m2_s
    r = elements.radius_m
    sin_nu = math.sin(elements.true_anomaly_rad)
    cos_nu = math.cos(elements.true_anomaly_rad)
    u = elements.argument_of_latitude_rad
    sin_i = math.sin(elements.inclination_rad)
    cos_i = math.cos(elements.inclination_rad)

    a_dot = 2.0 * elements.semi_major_axis_m**2 / h * (e * sin_nu * radial + (p / r) * along)
    e_dot = (p * sin_nu * radial + ((p + r) * cos_nu + r * e) * along) / h
    i_dot = r * math.cos(u) / h * cross
    raan_dot = r * math.sin(u) / (h * sin_i) * cross
    in_plane = (-p * cos_nu * radial + (p + r) * sin_nu * along) / (h * e)
    argp_dot = in_plane - r * math.sin(u) * cos_i / (h * sin_i) * cross
    nu_dot = h / (r * r) - in_plane
    return np.array([a_dot, e_dot, i_dot, raan_dot, argp_dot, nu_dot])


def j2_secular_rates(
    semi_major_axis_m: float, eccentricity: float, inclination_rad: float
) -> tuple[float, float]:
    """First-order secular node and perigee rates under J2, in rad/s.

        dRAAN/dt   = -(3/2) n J2 (Re/p)^2 cos(i)
        domega/dt  =  (3/4) n J2 (Re/p)^2 (5 cos^2(i) - 1)

    These are the SECULAR half of this module's J2 treatment; the numerical
    integration above carries the secular, long-period and short-period terms
    together. They are reported so a run can be read against theory without
    re-deriving it, and the test suite restates them independently rather than
    calling this function, so a mistake here cannot validate itself.
    """
    n = math.sqrt(MU / semi_major_axis_m**3)
    p = semi_major_axis_m * (1.0 - eccentricity**2)
    shape = J2 * (R_EQUATORIAL_M / p) ** 2
    cos_i = math.cos(inclination_rad)
    raan_dot = -1.5 * n * shape * cos_i
    argp_dot = 0.75 * n * shape * (5.0 * cos_i * cos_i - 1.0)
    return raan_dot, argp_dot


# --------------------------------------------------------------------------
# Analytic atmospheres, for controlled comparison only
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstantAtmosphere:
    """An altitude-independent density, so decay depends only on C_d A/m and rho.

    VALIDATION ONLY. This is not a thermosphere model and must never reach a
    science result; it exists because the closed-form circular decay law assumes
    uniform density, and comparing against that law requires an atmosphere that
    actually is uniform.
    """

    density_kg_m3: float

    def __call__(
        self, when: _dt.datetime, altitude_m: float, latitude_deg: float, longitude_deg: float
    ) -> float:
        return self.density_kg_m3


@dataclass(frozen=True, slots=True)
class ExponentialAtmosphere:
    """A single-scale-height exponential atmosphere. VALIDATION ONLY.

    Used where the comparison needs the density to actually rise as the vehicle
    falls -- reentry-threshold sensitivity is meaningless under a uniform
    atmosphere, because there the descent rate through the last forty kilometres
    is the same as through the first forty.
    """

    reference_altitude_m: float
    reference_density_kg_m3: float
    scale_height_m: float

    def __post_init__(self) -> None:
        if self.reference_density_kg_m3 <= 0.0:
            raise ValueError(
                f"reference density must be positive, got {self.reference_density_kg_m3}"
            )
        if self.scale_height_m <= 0.0:
            raise ValueError(f"scale height must be positive, got {self.scale_height_m}")

    def __call__(
        self, when: _dt.datetime, altitude_m: float, latitude_deg: float, longitude_deg: float
    ) -> float:
        exponent = -(altitude_m - self.reference_altitude_m) / self.scale_height_m
        return self.reference_density_kg_m3 * math.exp(min(exponent, 700.0))


# --------------------------------------------------------------------------
# The propagator
# --------------------------------------------------------------------------


@dataclass(slots=True)
class GaussPropagationResult:
    """A trajectory in elements, plus what is needed to judge whether to trust it."""

    epoch_utc: _dt.datetime
    times_utc: list[_dt.datetime]
    elements: list[ClassicalElements]
    positions_m: list[tuple[float, float, float]]
    velocities_m_s: list[tuple[float, float, float]]
    altitudes_m: list[float]
    density_kg_m3: list[float]
    reentered: bool
    reentry_time_utc: _dt.datetime | None
    step_s: float
    steps_taken: int
    minimum_eccentricity: float
    reentry_altitude_m: float
    notes: list[str] = field(default_factory=list)

    @property
    def semi_major_axis_m(self) -> list[float]:
        return [element.semi_major_axis_m for element in self.elements]

    @property
    def eccentricity(self) -> list[float]:
        return [element.eccentricity for element in self.elements]

    @property
    def perigee_altitude_m(self) -> list[float]:
        return [element.perigee_altitude_m for element in self.elements]

    @property
    def apogee_altitude_m(self) -> list[float]:
        return [element.apogee_altitude_m for element in self.elements]

    @property
    def specific_energy_j_kg(self) -> list[float]:
        return [element.specific_energy_j_kg for element in self.elements]

    @property
    def decay_days(self) -> float | None:
        if not self.reentered or self.reentry_time_utc is None:
            return None
        return (self.reentry_time_utc - self.epoch_utc).total_seconds() / 86400.0


class GaussPropagator:
    """Fixed-step RK4 integration of Gauss's variational equations.

    The integrator is written out here rather than delegated. That is the point:
    ``scipy.integrate.solve_ivp`` is the one component the Cowell propagator and
    this one could otherwise still share, and a defect in a shared integrator --
    an event located on the wrong side of a step, a tolerance not doing what it
    claims -- would be invisible to a comparison that used it twice.

    A fixed step also makes the convergence study meaningful: halving a declared
    step and watching the answer improve by the predicted factor is a statement
    about this integrator, whereas halving an adaptive solver's tolerance is a
    statement about its error estimator.
    """

    def __init__(
        self,
        density_function: DensityFunction,
        inverse_ballistic_coefficient_m2_kg: float,
        *,
        include_j2: bool = True,
        include_drag: bool = True,
        include_atmosphere_rotation: bool = True,
        step_s: float = 5.0,
        reentry_altitude_m: float = DEFAULT_REENTRY_ALTITUDE_M,
    ) -> None:
        if inverse_ballistic_coefficient_m2_kg < 0.0:
            raise ValueError(
                "inverse ballistic coefficient C_d*A/m must be non-negative, got "
                f"{inverse_ballistic_coefficient_m2_kg}"
            )
        if step_s <= 0.0:
            raise ValueError(f"step must be positive, got {step_s}")
        if reentry_altitude_m <= 0.0:
            raise ValueError(f"reentry altitude must be positive, got {reentry_altitude_m}")
        self.density_function = density_function
        self.inverse_ballistic_coefficient_m2_kg = inverse_ballistic_coefficient_m2_kg
        self.include_j2 = include_j2
        self.include_drag = include_drag
        self.include_atmosphere_rotation = include_atmosphere_rotation
        self.step_s = step_s
        self.reentry_altitude_m = reentry_altitude_m
        self.density_evaluations = 0

    # -- right-hand side ---------------------------------------------------

    def _perturbation_rsw(
        self, elements: ClassicalElements, when: _dt.datetime
    ) -> tuple[float, float, float]:
        radial = along = cross = 0.0
        if self.include_j2:
            j2_r, j2_s, j2_w = j2_rsw_acceleration(
                elements.radius_m, elements.inclination_rad, elements.argument_of_latitude_rad
            )
            radial += j2_r
            along += j2_s
            cross += j2_w

        if not self.include_drag or self.inverse_ballistic_coefficient_m2_kg == 0.0:
            return radial, along, cross

        position, _ = cartesian_from_elements(elements)
        altitude = geodetic_altitude_heikkinen(position)
        if altitude < self.reentry_altitude_m:
            # Below the floor the density model is out of its domain and the
            # terminating event has already fired; return gravity and J2 only.
            return radial, along, cross

        sidereal = greenwich_mean_sidereal_time_rad(when)
        latitude, longitude = geodetic_latitude_longitude_heikkinen(position, sidereal)
        density = self.density_function(when, altitude, latitude, longitude)
        self.density_evaluations += 1
        if density <= 0.0:
            return radial, along, cross

        v_r, v_s, v_w = relative_velocity_rsw(
            elements, include_atmosphere_rotation=self.include_atmosphere_rotation
        )
        speed = math.sqrt(v_r * v_r + v_s * v_s + v_w * v_w)
        coefficient = -0.5 * self.inverse_ballistic_coefficient_m2_kg * density * speed
        return radial + coefficient * v_r, along + coefficient * v_s, cross + coefficient * v_w

    def _derivative(self, state: np.ndarray, when: _dt.datetime) -> np.ndarray:
        elements = _elements_from_state(state)
        radial, along, cross = self._perturbation_rsw(elements, when)
        return gauss_variational_derivatives(elements, radial, along, cross)

    # -- the integrator ----------------------------------------------------

    def _rk4_step(self, state: np.ndarray, when: _dt.datetime, step_s: float) -> np.ndarray:
        """One classical Runge-Kutta step, written out stage by stage."""
        half = step_s / 2.0
        mid = when + _dt.timedelta(seconds=half)
        end = when + _dt.timedelta(seconds=step_s)
        k1 = self._derivative(state, when)
        k2 = self._derivative(state + half * k1, mid)
        k3 = self._derivative(state + half * k2, mid)
        k4 = self._derivative(state + step_s * k3, end)
        return state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def propagate(
        self,
        initial: ClassicalElements,
        epoch: _dt.datetime,
        duration_days: float,
        *,
        sample_interval_s: float = 60.0,
    ) -> GaussPropagationResult:
        """Integrate until the reentry altitude is crossed or the duration elapses.

        The sample grid is required to be an exact sub-multiple of the duration
        and an exact multiple of the step. That refusal to interpolate is
        deliberate: this propagator exists to be differenced sample-by-sample
        against another one, and an interpolated sample would put the
        interpolator's error into the comparison and let it be read as a physics
        disagreement.
        """
        if duration_days <= 0.0:
            raise ValueError(f"duration must be positive, got {duration_days}")
        if epoch.tzinfo is None:
            raise ValueError(f"epoch {epoch!r} is naive; validation requires explicit UTC")
        if sample_interval_s <= 0.0:
            raise ValueError(f"sample interval must be positive, got {sample_interval_s}")

        steps_per_sample = sample_interval_s / self.step_s
        if abs(steps_per_sample - round(steps_per_sample)) > 1.0e-9:
            raise ValueError(
                f"sample interval {sample_interval_s} s is not an integer multiple of "
                f"the step {self.step_s} s; samples would have to be interpolated and "
                "the interpolation error would enter the comparison"
            )
        duration_s = duration_days * 86400.0
        samples = duration_s / sample_interval_s
        if abs(samples - round(samples)) > 1.0e-9:
            raise ValueError(
                f"duration {duration_s} s is not an integer multiple of the sample "
                f"interval {sample_interval_s} s; the final sample would not land on "
                "the grid the other propagator uses"
            )

        steps_per_sample = int(round(steps_per_sample))
        total_samples = int(round(samples))

        state = initial.as_array()
        times = [epoch]
        elements_out = [initial]
        position, velocity = cartesian_from_elements(initial)
        positions = [_triple(position)]
        velocities = [_triple(velocity)]
        altitudes = [geodetic_altitude_heikkinen(position)]
        densities = [self._sample_density(epoch, position, altitudes[0])]
        minimum_eccentricity = initial.eccentricity

        reentered = False
        reentry_time: _dt.datetime | None = None
        steps_taken = 0

        for sample_index in range(total_samples):
            for step_index in range(steps_per_sample):
                elapsed = (sample_index * steps_per_sample + step_index) * self.step_s
                when = epoch + _dt.timedelta(seconds=elapsed)
                previous = state
                state = self._rk4_step(state, when, self.step_s)
                steps_taken += 1
                _validate_state(state, when)
                minimum_eccentricity = min(minimum_eccentricity, float(state[1]))

                altitude = _altitude_of(state)
                if altitude < self.reentry_altitude_m:
                    offset = self._refine_crossing(previous, when, self.step_s)
                    reentered = True
                    reentry_time = when + _dt.timedelta(seconds=offset)
                    break
                if altitude > MAX_ALTITUDE_M:
                    raise RuntimeError(
                        f"geodetic altitude reached {altitude / 1000:.1f} km at "
                        f"{when.isoformat()}, above the {MAX_ALTITUDE_M / 1000:.0f} km "
                        "ceiling; this is not a decay scenario and the run is stopped "
                        "rather than reported"
                    )
            if reentered:
                break

            elements = _elements_from_state(state)
            when = epoch + _dt.timedelta(seconds=(sample_index + 1) * sample_interval_s)
            position, velocity = cartesian_from_elements(elements)
            altitude = geodetic_altitude_heikkinen(position)
            times.append(when)
            elements_out.append(elements)
            positions.append(_triple(position))
            velocities.append(_triple(velocity))
            altitudes.append(altitude)
            densities.append(self._sample_density(when, position, altitude))

        notes = [
            "Independent second implementation: Gauss variational equations in "
            "classical elements, hand-written fixed-step RK4, Heikkinen closed-form "
            "geodetic altitude. Shares no code with the Cowell propagator.",
            "Same physics literature as the Cowell propagator (two-body + J2 + "
            "free-molecular drag on a rigidly co-rotating atmosphere), so agreement "
            "tests implementation, NOT the shared model assumptions.",
            "Integration frame treated as inertial; precession, nutation and polar "
            "motion omitted.",
            "Atmosphere co-rotation "
            + ("included" if self.include_atmosphere_rotation else "OMITTED")
            + ".",
            f"J2 {'included' if self.include_j2 else 'OMITTED'}.",
            f"Drag {'included' if self.include_drag else 'OMITTED'}.",
            f"Reentry threshold {self.reentry_altitude_m / 1000:.0f} km geodetic.",
            f"Fixed RK4 step {self.step_s} s; {steps_taken} steps taken.",
            f"Minimum eccentricity reached: {minimum_eccentricity:.3e}.",
            f"Density evaluations during integration: {self.density_evaluations}.",
        ]
        if minimum_eccentricity < ILL_CONDITIONED_ECCENTRICITY:
            notes.append(
                f"WARNING: eccentricity fell to {minimum_eccentricity:.3e}, below the "
                f"{ILL_CONDITIONED_ECCENTRICITY:.0e} conditioning threshold. The "
                "argument of perigee is poorly determined there and its rate "
                "approaches the mean motion; treat omega from this run as indicative."
            )

        return GaussPropagationResult(
            epoch_utc=epoch,
            times_utc=times,
            elements=elements_out,
            positions_m=positions,
            velocities_m_s=velocities,
            altitudes_m=altitudes,
            density_kg_m3=densities,
            reentered=reentered,
            reentry_time_utc=reentry_time,
            step_s=self.step_s,
            steps_taken=steps_taken,
            minimum_eccentricity=minimum_eccentricity,
            reentry_altitude_m=self.reentry_altitude_m,
            notes=notes,
        )

    def propagate_from_cartesian(
        self,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        epoch: _dt.datetime,
        duration_days: float,
        *,
        sample_interval_s: float = 60.0,
    ) -> GaussPropagationResult:
        """Same integration, seeded from a Cartesian state.

        The conversion runs through this module's own element recovery, so a
        cross-propagator comparison started from a shared Cartesian state does
        not import the other propagator's idea of what those elements are.
        """
        initial = elements_from_cartesian(position_m, velocity_m_s)
        return self.propagate(
            initial, epoch, duration_days, sample_interval_s=sample_interval_s
        )

    def _sample_density(
        self, when: _dt.datetime, position: np.ndarray, altitude_m: float
    ) -> float:
        if not self.include_drag or altitude_m < self.reentry_altitude_m:
            return float("nan")
        sidereal = greenwich_mean_sidereal_time_rad(when)
        latitude, longitude = geodetic_latitude_longitude_heikkinen(position, sidereal)
        return self.density_function(when, altitude_m, latitude, longitude)

    def _refine_crossing(
        self, before: np.ndarray, when: _dt.datetime, step_s: float
    ) -> float:
        """Bisect inside the step that crossed the reentry altitude.

        The bracket is re-integrated from the last good state at each trial
        rather than interpolated, so the located crossing lies on the trajectory
        the integrator actually produced instead of on a chord across it.
        """
        low, high = 0.0, step_s
        while high - low > EVENT_BRACKET_TOLERANCE_S:
            middle = 0.5 * (low + high)
            trial = self._rk4_step(before, when, middle)
            if _altitude_of(trial) > self.reentry_altitude_m:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)


def _triple(vector: np.ndarray) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _elements_from_state(state: np.ndarray) -> ClassicalElements:
    return ClassicalElements(
        semi_major_axis_m=float(state[0]),
        eccentricity=float(state[1]),
        inclination_rad=float(state[2]),
        raan_rad=float(state[3]),
        argument_of_perigee_rad=float(state[4]),
        true_anomaly_rad=float(state[5]),
    )


def _altitude_of(state: np.ndarray) -> float:
    position, _ = cartesian_from_elements(_elements_from_state(state))
    return geodetic_altitude_heikkinen(position)


def _validate_state(state: np.ndarray, when: _dt.datetime) -> None:
    if not np.all(np.isfinite(state)):
        raise RuntimeError(
            f"element state went non-finite at {when.isoformat()}: {state.tolist()}"
        )
    if float(state[1]) < MIN_ECCENTRICITY:
        raise SingularElementsError(
            f"eccentricity fell to {float(state[1]):.3e} at {when.isoformat()}, below "
            f"the {MIN_ECCENTRICITY:.0e} floor. Classical elements are singular for a "
            "circular orbit: the argument of perigee is undefined and Gauss's "
            "equations carry a 1/e factor. The run is stopped rather than continued "
            "into coordinates that no longer mean anything."
        )
    if abs(math.sin(float(state[2]))) < MIN_SIN_INCLINATION:
        raise SingularElementsError(
            f"|sin(inclination)| fell below {MIN_SIN_INCLINATION:.0e} at "
            f"{when.isoformat()}. Classical elements are singular for an equatorial "
            "orbit: the ascending node is undefined and Gauss's equations carry a "
            "1/sin(i) factor."
        )


# --------------------------------------------------------------------------
# Comparison machinery
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrajectorySamples:
    """A trajectory reduced to the quantities two propagators can be differenced on.

    Both trajectories are reduced through THIS module's element recovery, never
    through each propagator's own. Otherwise a disagreement between two
    element-extraction routines would be reported as a disagreement between two
    propagators, which is a different and much less interesting finding.
    """

    times_s: tuple[float, ...]
    positions_m: tuple[tuple[float, float, float], ...]
    velocities_m_s: tuple[tuple[float, float, float], ...]
    semi_major_axis_m: tuple[float, ...]
    eccentricity: tuple[float, ...]
    perigee_altitude_m: tuple[float, ...]
    apogee_altitude_m: tuple[float, ...]
    specific_energy_j_kg: tuple[float, ...]

    @classmethod
    def from_states(
        cls,
        epoch: _dt.datetime,
        times_utc: Sequence[_dt.datetime],
        positions_m: Sequence[tuple[float, float, float]],
        velocities_m_s: Sequence[tuple[float, float, float]],
    ) -> TrajectorySamples:
        if not (len(times_utc) == len(positions_m) == len(velocities_m_s)):
            raise ValueError(
                f"times, positions and velocities have lengths {len(times_utc)}, "
                f"{len(positions_m)} and {len(velocities_m_s)}; they must match"
            )
        if not times_utc:
            raise ValueError("no samples to reduce")
        elements = [
            elements_from_cartesian(np.asarray(p, dtype=float), np.asarray(v, dtype=float))
            for p, v in zip(positions_m, velocities_m_s, strict=True)
        ]
        positions = tuple(
            (float(p[0]), float(p[1]), float(p[2])) for p in positions_m
        )
        velocities = tuple(
            (float(v[0]), float(v[1]), float(v[2])) for v in velocities_m_s
        )
        return cls(
            times_s=tuple((t - epoch).total_seconds() for t in times_utc),
            positions_m=positions,
            velocities_m_s=velocities,
            semi_major_axis_m=tuple(e.semi_major_axis_m for e in elements),
            eccentricity=tuple(e.eccentricity for e in elements),
            perigee_altitude_m=tuple(e.perigee_altitude_m for e in elements),
            apogee_altitude_m=tuple(e.apogee_altitude_m for e in elements),
            specific_energy_j_kg=tuple(e.specific_energy_j_kg for e in elements),
        )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def compare_trajectories(
    first: TrajectorySamples,
    second: TrajectorySamples,
    *,
    time_tolerance_s: float = 1.0e-6,
) -> dict[str, float | int]:
    """Difference two trajectories sample by sample on a common time grid.

    Where the two runs stop at different times -- which is the normal case when
    one reenters a few minutes before the other -- the comparison is made over
    the common prefix and the number of samples actually used is reported, so a
    short overlap cannot masquerade as good agreement. Overlapping samples whose
    timestamps do not match are an error, not something to interpolate away.
    """
    common = min(len(first.times_s), len(second.times_s))
    if common == 0:
        raise ValueError("no overlapping samples to compare")
    for index in range(common):
        offset = abs(first.times_s[index] - second.times_s[index])
        if offset > time_tolerance_s:
            raise ValueError(
                f"sample {index} timestamps differ by {offset:.9f} s, beyond the "
                f"{time_tolerance_s} s tolerance; the two runs are not on a common "
                "grid and differencing them would measure the grid, not the physics"
            )

    position_diff: list[float] = []
    radial_diff: list[float] = []
    velocity_diff: list[float] = []
    speed_diff: list[float] = []
    for index in range(common):
        p1, p2 = first.positions_m[index], second.positions_m[index]
        v1, v2 = first.velocities_m_s[index], second.velocities_m_s[index]
        position_diff.append(_norm((p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])))
        radial_diff.append(_norm(p1) - _norm(p2))
        velocity_diff.append(_norm((v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])))
        speed_diff.append(_norm(v1) - _norm(v2))

    sma_diff = [first.semi_major_axis_m[i] - second.semi_major_axis_m[i] for i in range(common)]
    perigee_diff = [
        first.perigee_altitude_m[i] - second.perigee_altitude_m[i]
        for i in range(common)
    ]
    apogee_diff = [first.apogee_altitude_m[i] - second.apogee_altitude_m[i] for i in range(common)]

    loss_first = first.specific_energy_j_kg[0] - first.specific_energy_j_kg[common - 1]
    loss_second = second.specific_energy_j_kg[0] - second.specific_energy_j_kg[common - 1]
    loss_difference = abs(loss_first - loss_second)
    loss_scale = max(abs(loss_first), abs(loss_second))

    return {
        "compared_samples": common,
        "samples_first": len(first.times_s),
        "samples_second": len(second.times_s),
        "span_s": first.times_s[common - 1] - first.times_s[0],
        "max_position_difference_m": max(position_diff),
        "rms_position_difference_m": math.sqrt(
            math.fsum(value * value for value in position_diff) / common
        ),
        "max_radial_difference_m": max(abs(value) for value in radial_diff),
        "max_velocity_difference_m_s": max(velocity_diff),
        "max_speed_difference_m_s": max(abs(value) for value in speed_diff),
        "max_semi_major_axis_difference_m": max(abs(value) for value in sma_diff),
        "final_semi_major_axis_difference_m": abs(sma_diff[common - 1]),
        "max_perigee_altitude_difference_m": max(abs(value) for value in perigee_diff),
        "max_apogee_altitude_difference_m": max(abs(value) for value in apogee_diff),
        "energy_loss_first_j_kg": loss_first,
        "energy_loss_second_j_kg": loss_second,
        "energy_loss_difference_j_kg": loss_difference,
        "energy_loss_relative_difference": (
            loss_difference / loss_scale if loss_scale > 0.0 else 0.0
        ),
    }


def secular_rate_fit(
    times_s: Sequence[float],
    angles_rad: Sequence[float],
    *,
    samples_per_orbit: int | None = None,
) -> float:
    """Least-squares secular rate of an angle, in rad/s.

    The angle is unwrapped first. Fitting a wrapped angle would read every 2-pi
    rollover as a jump and return a rate near zero, which looks like "no
    precession" rather than like a bug.

    ``samples_per_orbit`` averages the series in consecutive whole-orbit blocks
    before fitting, and is not optional in practice for the argument of perigee.
    First-order J2 theory is written in MEAN elements; both propagators report
    OSCULATING ones. The osculating argument of perigee carries a short-period
    term of amplitude roughly J2/e radians, which at e = 0.02 is larger than the
    secular change over a whole day, and a straight least-squares slope through
    it returns a rate biased by whatever phase the arc happened to end on --
    measured here at between 0.05% and 1.4% depending on the arc length, which
    is not a repeatable comparison. Averaging over whole orbits removes the
    short-period term to the accuracy with which the sampling period matches the
    true anomalistic period, leaving a bias two orders smaller.
    """
    if len(times_s) != len(angles_rad):
        raise ValueError(
            f"times and angles have lengths {len(times_s)} and {len(angles_rad)}; "
            "they must match"
        )
    if len(times_s) < 2:
        raise ValueError("at least two samples are needed to fit a rate")
    t = np.asarray(times_s, dtype=float)
    y = np.unwrap(np.asarray(angles_rad, dtype=float))

    if samples_per_orbit is not None:
        if samples_per_orbit < 1:
            raise ValueError(f"samples per orbit must be at least 1, got {samples_per_orbit}")
        blocks = len(t) // samples_per_orbit
        if blocks < 2:
            raise ValueError(
                f"{len(t)} samples at {samples_per_orbit} per orbit give {blocks} whole "
                "orbits; at least two are needed to fit an orbit-averaged rate"
            )
        used = blocks * samples_per_orbit
        t = t[:used].reshape(blocks, samples_per_orbit).mean(axis=1)
        y = y[:used].reshape(blocks, samples_per_orbit).mean(axis=1)

    slope, _intercept = np.polyfit(t, y, 1)
    return float(slope)
