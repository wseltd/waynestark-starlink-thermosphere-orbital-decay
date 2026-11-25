"""Numerical orbit propagation with drag, J2 and a reentry event.

Cowell formulation: integrate Cartesian position and velocity in an
Earth-centred inertial frame under two-body gravity plus J2 plus atmospheric
drag. This is the right method here precisely because SGP4 is not — SGP4 is an
analytic mean-element theory whose drag is a fitted B* term, so it cannot
answer "what would this orbit have done under a different atmosphere?", which
is the entire question. SGP4 is retained separately, for validating catalogued
element sets against reality, never for the counterfactual.

Frame handling is deliberately modest and stated rather than hidden: the
integration frame is treated as inertial and Earth rotation enters only
through the co-rotating atmosphere in the relative-velocity term. Precession,
nutation and polar motion are omitted. Over the days-to-weeks decay timescales
here, and against the far larger uncertainty in density and ballistic
coefficient, that omission is negligible — but it is an omission, and it is
recorded in the result rather than assumed away.

The atmosphere co-rotation term matters and is included: at 210 km the surface
rotation speed is ~340 m/s at the equator against an orbital speed of
~7.8 km/s, so neglecting it biases drag by of order 4%.

Canonical tickets: P03_05_T040 through T052, P06_08_T001 through T012.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from ..domain.constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH_EQUATORIAL

# Below this geodetic altitude the vehicle is treated as having reentered.
# 100 km is the Karman line convention; the exact value barely affects decay
# time because the final descent from 150 km takes hours, not days.
REENTRY_ALTITUDE_M = 100_000.0

# Integration fails rather than silently producing nonsense if the state goes
# non-finite or the vehicle escapes.
MAX_ALTITUDE_M = 2_000_000.0


@dataclass(frozen=True, slots=True)
class SpacecraftProperties:
    """The three parameters drag depends on, plus their identifiability caveat.

    mass, area and drag coefficient are not separately identifiable from a
    decay curve — only the inverse ballistic coefficient C_d*A/m is. They are carried
    separately because the physics computes C_d from composition, but every
    reported result also states the resulting inverse ballistic coefficient.
    """

    mass_kg: float
    reference_area_m2: float
    drag_coefficient: float
    name: str = "spacecraft"

    def __post_init__(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError(f"mass must be positive, got {self.mass_kg}")
        if self.reference_area_m2 <= 0:
            raise ValueError(f"area must be positive, got {self.reference_area_m2}")
        if self.drag_coefficient <= 0:
            raise ValueError(f"drag coefficient must be positive, got {self.drag_coefficient}")

    @property
    def inverse_ballistic_coefficient(self) -> float:
        """C_d * A / m in m^2/kg."""
        return self.drag_coefficient * self.reference_area_m2 / self.mass_kg


@dataclass(frozen=True, slots=True)
class OrbitState:
    """Cartesian state in the integration frame, metres and metres per second."""

    time_utc: _dt.datetime
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]

    @property
    def radius_m(self) -> float:
        return math.sqrt(sum(c * c for c in self.position_m))

    @property
    def speed_m_s(self) -> float:
        return math.sqrt(sum(c * c for c in self.velocity_m_s))

    @property
    def geodetic_altitude_m(self) -> float:
        return geodetic_altitude(np.asarray(self.position_m))


@dataclass(slots=True)
class PropagationResult:
    """A trajectory plus everything needed to judge whether to believe it."""

    times_utc: list[_dt.datetime]
    altitudes_m: list[float]
    semi_major_axis_m: list[float]
    eccentricity: list[float]
    perigee_altitude_m: list[float]
    apogee_altitude_m: list[float]
    density_kg_m3: list[float]
    drag_acceleration_m_s2: list[float]
    reentered: bool
    reentry_time_utc: _dt.datetime | None
    spacecraft: SpacecraftProperties
    # Full Cartesian states are retained rather than recomputed. A consumer
    # that needs the sub-satellite point must not have to re-derive position
    # from elements, because that reconstruction loses the orbital phase and
    # would silently put every sample at the same longitude.
    positions_m: list[tuple[float, float, float]] = field(default_factory=list)
    velocities_m_s: list[tuple[float, float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def decay_days(self) -> float | None:
        if not self.reentered or self.reentry_time_utc is None:
            return None
        return (self.reentry_time_utc - self.times_utc[0]).total_seconds() / 86400.0


def geodetic_altitude(position_m: np.ndarray) -> float:
    """Geodetic altitude above the WGS-84 ellipsoid, by Bowring's method.

    Spherical altitude would be wrong by up to ~21 km at these inclinations
    because the Earth's flattening is comparable to the whole decay margin.
    """
    a = R_EARTH_EQUATORIAL.value
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = f * (2.0 - f)
    ep2 = (a * a - b * b) / (b * b)

    x, y, z = float(position_m[0]), float(position_m[1]), float(position_m[2])
    p = math.hypot(x, y)
    if p < 1.0:
        return abs(z) - b

    theta = math.atan2(z * a, p * b)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - e2 * a * math.cos(theta) ** 3,
    )
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    return p / math.cos(lat) - n


def geodetic_latlon(position_m: np.ndarray, gmst_rad: float) -> tuple[float, float]:
    """Geodetic latitude and east longitude in degrees, given Greenwich sidereal time."""
    a = R_EARTH_EQUATORIAL.value
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = f * (2.0 - f)
    ep2 = (a * a - b * b) / (b * b)

    x, y, z = float(position_m[0]), float(position_m[1]), float(position_m[2])
    p = math.hypot(x, y)
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - e2 * a * math.cos(theta) ** 3,
    )
    lon_inertial = math.atan2(y, x)
    lon = math.degrees(lon_inertial - gmst_rad)
    lon = (lon + 180.0) % 360.0 - 180.0
    return math.degrees(lat), lon


def greenwich_sidereal_time(when: _dt.datetime) -> float:
    """GMST in radians (IAU 1982), adequate for locating the sub-satellite point."""
    when = when.astimezone(_dt.UTC)
    jd = julian_date(when)
    t = (jd - 2451545.0) / 36525.0
    gmst_s = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    return math.radians((gmst_s % 86400.0) / 240.0)


def julian_date(when: _dt.datetime) -> float:
    when = when.astimezone(_dt.UTC)
    y, m = when.year, when.month
    d = (
        when.day
        + (when.hour + (when.minute + (when.second + when.microsecond / 1e6) / 60.0) / 60.0)
        / 24.0
    )
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5


def classical_elements(
    position_m: np.ndarray, velocity_m_s: np.ndarray
) -> tuple[float, float, float]:
    """Semi-major axis, eccentricity and inclination from a Cartesian state."""
    mu = MU_EARTH.value
    r = float(np.linalg.norm(position_m))
    v = float(np.linalg.norm(velocity_m_s))
    energy = 0.5 * v * v - mu / r
    if energy >= 0:
        raise ValueError("state is not bound; specific orbital energy is non-negative")
    a = -mu / (2.0 * energy)
    h_vec = np.cross(position_m, velocity_m_s)
    h = float(np.linalg.norm(h_vec))
    e_vec = np.cross(velocity_m_s, h_vec) / mu - position_m / r
    e = float(np.linalg.norm(e_vec))
    inc = math.acos(max(-1.0, min(1.0, float(h_vec[2]) / h)))
    return a, e, inc


def state_from_perigee_apogee(
    perigee_altitude_m: float,
    apogee_altitude_m: float,
    inclination_rad: float,
    *,
    raan_rad: float = 0.0,
    argument_of_perigee_rad: float = 0.0,
    true_anomaly_rad: float = 0.0,
    body_radius_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian state at a given true anomaly, from perigee/apogee altitudes."""
    re = body_radius_m if body_radius_m is not None else R_EARTH_EQUATORIAL.value
    rp = re + perigee_altitude_m
    ra = re + apogee_altitude_m
    if ra < rp:
        raise ValueError(
            f"apogee altitude ({apogee_altitude_m}) is below perigee ({perigee_altitude_m})"
        )
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    mu = MU_EARTH.value

    p = a * (1.0 - e * e)
    nu = true_anomaly_rad
    r = p / (1.0 + e * math.cos(nu))

    r_pqw = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    v_pqw = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    ci, si = math.cos(inclination_rad), math.sin(inclination_rad)
    co, so = math.cos(raan_rad), math.sin(raan_rad)
    cw, sw = math.cos(argument_of_perigee_rad), math.sin(argument_of_perigee_rad)

    rot = np.array(
        [
            [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
            [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
            [sw * si, cw * si, ci],
        ]
    )
    return rot @ r_pqw, rot @ v_pqw


def j2_acceleration(position_m: np.ndarray) -> np.ndarray:
    """J2 oblateness perturbation.

    Included because J2 dominates every other non-drag perturbation in LEO by
    orders of magnitude. It does not itself cause secular decay, but it
    controls perigee precession, which decides where in the atmosphere the
    vehicle spends its low passes — and therefore how fast it decays.
    """
    mu = MU_EARTH.value
    j2 = J2_EARTH.value
    re = R_EARTH_EQUATORIAL.value

    x, y, z = position_m
    r = float(np.linalg.norm(position_m))
    r2 = r * r
    z2_r2 = (z * z) / r2
    factor = 1.5 * j2 * mu * re * re / (r2 * r2 * r)

    return np.array(
        [
            factor * x * (5.0 * z2_r2 - 1.0),
            factor * y * (5.0 * z2_r2 - 1.0),
            factor * z * (5.0 * z2_r2 - 3.0),
        ]
    )


DensityFunction = Callable[[_dt.datetime, float, float, float], float]


class DragPropagator:
    """Cowell propagator with J2 and free-molecular drag.

    ``density_function`` is injected, which is what makes the counterfactual
    possible: the same trajectory can be run against a quiet atmosphere and a
    storm atmosphere with nothing else changed, so the difference is
    attributable to the atmosphere and to nothing else.
    """

    def __init__(
        self,
        density_function: DensityFunction,
        spacecraft: SpacecraftProperties,
        *,
        include_j2: bool = True,
        include_atmosphere_rotation: bool = True,
    ) -> None:
        self.density_function = density_function
        self.spacecraft = spacecraft
        self.include_j2 = include_j2
        self.include_atmosphere_rotation = include_atmosphere_rotation
        self._density_calls = 0

    def _derivative(
        self, t_s: float, state: np.ndarray, epoch: _dt.datetime
    ) -> np.ndarray:
        position = state[:3]
        velocity = state[3:]
        r = float(np.linalg.norm(position))

        acceleration = -MU_EARTH.value * position / (r**3)
        if self.include_j2:
            acceleration = acceleration + j2_acceleration(position)

        when = epoch + _dt.timedelta(seconds=float(t_s))
        altitude = geodetic_altitude(position)

        if altitude < REENTRY_ALTITUDE_M:
            # Below the reentry floor the density model is out of its domain;
            # the terminating event fires here, so return gravity only.
            return np.concatenate([velocity, acceleration])

        gmst = greenwich_sidereal_time(when)
        latitude, longitude = geodetic_latlon(position, gmst)
        density = self.density_function(when, altitude, latitude, longitude)
        self._density_calls += 1

        if self.include_atmosphere_rotation:
            omega = np.array([0.0, 0.0, OMEGA_EARTH.value])
            v_rel = velocity - np.cross(omega, position)
        else:
            v_rel = velocity

        v_rel_mag = float(np.linalg.norm(v_rel))
        if v_rel_mag > 0.0 and density > 0.0:
            b = self.spacecraft.inverse_ballistic_coefficient
            drag = -0.5 * b * density * v_rel_mag * v_rel
            acceleration = acceleration + drag

        return np.concatenate([velocity, acceleration])

    def propagate(
        self,
        initial: OrbitState,
        duration_days: float,
        *,
        sample_interval_s: float = 60.0,
        rtol: float = 1e-9,
        atol: float = 1e-6,
        max_step_s: float = 30.0,
    ) -> PropagationResult:
        """Integrate until reentry or until the duration elapses."""
        if duration_days <= 0:
            raise ValueError(f"duration must be positive, got {duration_days}")

        epoch = initial.time_utc
        y0 = np.array([*initial.position_m, *initial.velocity_m_s], dtype=float)
        t_end = duration_days * 86400.0

        def reentry_event(t_s: float, state: np.ndarray) -> float:
            return geodetic_altitude(state[:3]) - REENTRY_ALTITUDE_M

        reentry_event.terminal = True  # type: ignore[attr-defined]
        reentry_event.direction = -1.0  # type: ignore[attr-defined]

        def escape_event(t_s: float, state: np.ndarray) -> float:
            return MAX_ALTITUDE_M - geodetic_altitude(state[:3])

        escape_event.terminal = True  # type: ignore[attr-defined]
        escape_event.direction = -1.0  # type: ignore[attr-defined]

        # np.arange overshoots t_end whenever the interval does not divide the
        # duration evenly, and solve_ivp rejects any t_eval outside t_span.
        t_eval = np.arange(0.0, t_end, sample_interval_s)
        if t_eval.size == 0 or t_eval[-1] < t_end:
            t_eval = np.append(t_eval, t_end)

        solution = solve_ivp(
            fun=lambda t, y: self._derivative(t, y, epoch),
            t_span=(0.0, t_end),
            y0=y0,
            method="DOP853",
            t_eval=t_eval,
            events=[reentry_event, escape_event],
            rtol=rtol,
            atol=atol,
            max_step=max_step_s,
            dense_output=False,
        )

        if not solution.success and solution.status != 1:
            raise RuntimeError(f"integration failed: {solution.message}")

        reentered = len(solution.t_events[0]) > 0
        reentry_time = (
            epoch + _dt.timedelta(seconds=float(solution.t_events[0][0]))
            if reentered
            else None
        )

        times: list[_dt.datetime] = []
        altitudes: list[float] = []
        sma: list[float] = []
        ecc: list[float] = []
        perigee: list[float] = []
        apogee: list[float] = []
        densities: list[float] = []
        drag_accels: list[float] = []
        positions: list[tuple[float, float, float]] = []
        velocities: list[tuple[float, float, float]] = []

        re = R_EARTH_EQUATORIAL.value
        for i, t_s in enumerate(solution.t):
            position = solution.y[:3, i]
            velocity = solution.y[3:, i]
            when = epoch + _dt.timedelta(seconds=float(t_s))
            altitude = geodetic_altitude(position)

            times.append(when)
            altitudes.append(altitude)
            positions.append((float(position[0]), float(position[1]), float(position[2])))
            velocities.append((float(velocity[0]), float(velocity[1]), float(velocity[2])))
            try:
                a, e, _ = classical_elements(position, velocity)
            except ValueError:
                a, e = float("nan"), float("nan")
            sma.append(a)
            ecc.append(e)
            perigee.append(a * (1.0 - e) - re)
            apogee.append(a * (1.0 + e) - re)

            if altitude >= REENTRY_ALTITUDE_M:
                gmst = greenwich_sidereal_time(when)
                lat, lon = geodetic_latlon(position, gmst)
                rho = self.density_function(when, altitude, lat, lon)
            else:
                rho = float("nan")
            densities.append(rho)

            if math.isfinite(rho):
                omega = np.array([0.0, 0.0, OMEGA_EARTH.value])
                v_rel = velocity - np.cross(omega, position)
                v_mag = float(np.linalg.norm(v_rel))
                drag_accels.append(
                    0.5 * self.spacecraft.inverse_ballistic_coefficient * rho * v_mag * v_mag
                )
            else:
                drag_accels.append(float("nan"))

        notes = [
            "Integration frame treated as inertial; precession, nutation and "
            "polar motion omitted. Negligible over decay timescales relative "
            "to density and ballistic-coefficient uncertainty.",
            "Atmosphere co-rotation "
            + ("included" if self.include_atmosphere_rotation else "OMITTED")
            + ".",
            f"J2 {'included' if self.include_j2 else 'OMITTED'}.",
            f"Reentry threshold {REENTRY_ALTITUDE_M / 1000:.0f} km geodetic.",
            f"Density evaluations during integration: {self._density_calls}.",
        ]

        return PropagationResult(
            times_utc=times,
            altitudes_m=altitudes,
            semi_major_axis_m=sma,
            eccentricity=ecc,
            perigee_altitude_m=perigee,
            apogee_altitude_m=apogee,
            density_kg_m3=densities,
            drag_acceleration_m_s2=drag_accels,
            reentered=reentered,
            reentry_time_utc=reentry_time,
            spacecraft=self.spacecraft,
            positions_m=positions,
            velocities_m_s=velocities,
            notes=notes,
        )
