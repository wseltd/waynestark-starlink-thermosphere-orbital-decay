"""Two different energy quantities that must never be reported as one.

Drag removes energy from an orbit, and there are two defensible scalars for
"how much", which are NOT equal:

    inertial orbital-energy change      d(epsilon)/dt = dot(a_drag, v)
    atmosphere-relative dissipation     P/m           = -dot(a_drag, v_rel)

They differ whenever the spacecraft velocity and the atmosphere-relative
velocity differ — that is, whenever the atmosphere co-rotates, or a neutral
wind blows. At 210 km the co-rotation term is roughly 400 m/s against an
orbital 7 800 m/s, so the two integrals differ by a few percent; storm-time
winds of several hundred m/s move them further apart. Neither is wrong. Which
one a number IS, is what must not be left ambiguous.

Physically: ``dot(a_drag, v)`` is what the ORBIT loses, and it is the quantity
that governs decay. ``-dot(a_drag, v_rel)`` is what the GAS receives, and it is
non-negative by construction because ``a_drag`` is antiparallel to ``v_rel``.
The difference between them is work done against the rotating atmosphere's own
momentum, which changes the orbit without dissipating into the gas.

WHY THIS MODULE EXISTS. ``scripts/run_wam_ipe_decay.py`` reported three
quantities named ``drag_work_*_J_per_kg``. They were neither of the above: the
script integrated the MAGNITUDE of drag acceleration over time, which has units
of m/s and is a delta-v, and it formed that magnitude from the INERTIAL speed
rather than the relative speed, so the co-rotation the propagator carefully
includes was dropped. The number was a delta-v computed on the wrong velocity
and published as an energy. The ratios it was used for survived, because they
are ratios of the same wrong quantity; the labels did not.

ON J2. When J2 is active the two-body specific energy ``v^2/2 - mu/r`` is not
conserved even without drag: it oscillates at the short period with an
amplitude of order ten kilometres' worth of energy at this altitude, which
swamps the secular drag loss over a few revolutions. Differencing that
quantity across a window therefore does NOT measure drag. This module takes the
other route named in the instruction: it integrates drag power directly, so the
ledger never has to interpret a short-period oscillation. :func:`two_body_
specific_energy` is provided for diagnostics and its docstring says plainly
that it must not be differenced to obtain drag loss.

Canonical tickets: P03_05_T031, P06_08_T024.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..domain.constants import MU_EARTH, OMEGA_EARTH

#: Names the ledger emits. Deliberately long: a reader must not have to guess
#: which velocity a number was formed against, and "drag work" does not say.
INERTIAL_KEY = "inertial_orbital_energy_change_J_kg"
DISSIPATION_KEY = "atmosphere_relative_dissipation_J_kg"
DELTA_V_KEY = "drag_delta_v_m_s"


class EnergyLedgerError(ValueError):
    """Raised when a ledger cannot be formed from the states supplied."""


def corotating_velocity_m_s(position_m: np.ndarray) -> np.ndarray:
    """Velocity of a co-rotating atmosphere at ``position_m``, in the inertial frame."""
    omega = np.array([0.0, 0.0, OMEGA_EARTH.value], dtype=np.float64)
    return np.cross(omega, position_m)


def relative_velocity_m_s(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    *,
    wind_m_s: np.ndarray | None = None,
    include_corotation: bool = True,
) -> np.ndarray:
    """Spacecraft velocity relative to the local air.

    ``wind_m_s`` is an inertial-frame neutral wind added to the co-rotation
    term. It is optional because this project's default is an explicit,
    labelled zero wind; when a wind provider is in use its vector belongs here,
    or the dissipation figure silently describes a still atmosphere.
    """
    air = np.zeros(3, dtype=np.float64)
    if include_corotation:
        air = air + corotating_velocity_m_s(position_m)
    if wind_m_s is not None:
        air = air + np.asarray(wind_m_s, dtype=np.float64)
    return np.asarray(velocity_m_s, dtype=np.float64) - air


def drag_acceleration_m_s2(
    inverse_ballistic_coefficient_m2_kg: float,
    density_kg_m3: float,
    relative_velocity: np.ndarray,
) -> np.ndarray:
    """``-0.5 * (C_d A / m) * rho * |v_rel| * v_rel``, the drag acceleration vector.

    Returned as a VECTOR. The magnitude alone cannot form either energy
    quantity, because both are dot products with a velocity, and it was the
    magnitude that was previously integrated and mislabelled.
    """
    if inverse_ballistic_coefficient_m2_kg < 0:
        raise EnergyLedgerError(
            "inverse ballistic coefficient must be non-negative, got "
            f"{inverse_ballistic_coefficient_m2_kg} m^2/kg"
        )
    if density_kg_m3 < 0:
        raise EnergyLedgerError(f"density must be non-negative, got {density_kg_m3}")
    speed = float(np.linalg.norm(relative_velocity))
    if speed == 0.0:
        return np.zeros(3, dtype=np.float64)
    return (
        -0.5
        * inverse_ballistic_coefficient_m2_kg
        * density_kg_m3
        * speed
        * np.asarray(relative_velocity, dtype=np.float64)
    )


def inertial_orbital_energy_rate_w_kg(
    drag_acceleration: np.ndarray, velocity_m_s: np.ndarray
) -> float:
    """``dot(a_drag, v)`` — the rate the ORBIT loses specific energy.

    Negative for a decaying orbit. This is the quantity that governs decay, and
    it is formed against the INERTIAL velocity.
    """
    return float(np.dot(drag_acceleration, np.asarray(velocity_m_s, dtype=np.float64)))


def atmosphere_relative_dissipation_rate_w_kg(
    drag_acceleration: np.ndarray, relative_velocity: np.ndarray
) -> float:
    """``-dot(a_drag, v_rel)`` — the rate energy is dissipated into the GAS.

    Non-negative by construction: drag acceleration is antiparallel to the
    relative velocity, so the dot product is never positive and its negation
    never negative. A negative value here would mean the model had produced
    thrust, and is treated as an error rather than reported.
    """
    value = -float(
        np.dot(drag_acceleration, np.asarray(relative_velocity, dtype=np.float64))
    )
    if value < 0.0:
        raise EnergyLedgerError(
            f"atmosphere-relative dissipation is negative ({value} W/kg). Drag "
            "cannot add energy to the gas; the acceleration and relative "
            "velocity are not antiparallel, so one of them is wrong"
        )
    return value


def two_body_specific_energy_j_kg(
    position_m: np.ndarray, velocity_m_s: np.ndarray
) -> float:
    """``v^2/2 - mu/r``. DIAGNOSTIC ONLY — do not difference this to obtain drag loss.

    Under J2 this oscillates at the short period with an amplitude far larger
    than the secular drag loss over a few revolutions, so a difference across a
    window measures where in its oscillation each endpoint happened to fall.
    Use :func:`integrate_ledger`, which integrates drag power directly and never
    has to interpret that oscillation.
    """
    r = float(np.linalg.norm(position_m))
    v = float(np.linalg.norm(velocity_m_s))
    if r <= 0.0:
        raise EnergyLedgerError("radius must be positive")
    return 0.5 * v * v - MU_EARTH.value / r


@dataclass(frozen=True, slots=True)
class EnergyLedger:
    """Both integrals over one window, with the counts behind them."""

    inertial_orbital_energy_change_j_kg: float
    atmosphere_relative_dissipation_j_kg: float
    drag_delta_v_m_s: float
    n_samples: int
    window_seconds: float
    corotation_included: bool
    wind_included: bool

    @property
    def difference_j_kg(self) -> float:
        """Dissipation MINUS the magnitude of the orbital-energy change.

        Non-zero exactly when the two velocities differ. Reported so a reader
        can see the size of the effect that makes the two names necessary,
        rather than being asked to take it on trust.
        """
        return self.atmosphere_relative_dissipation_j_kg - abs(
            self.inertial_orbital_energy_change_j_kg
        )

    def as_dict(self) -> dict[str, object]:
        """Both quantities under explicit names, never one under the other's."""
        return {
            INERTIAL_KEY: self.inertial_orbital_energy_change_j_kg,
            DISSIPATION_KEY: self.atmosphere_relative_dissipation_j_kg,
            DELTA_V_KEY: self.drag_delta_v_m_s,
            "dissipation_minus_orbital_change_J_kg": self.difference_j_kg,
            "n_samples": self.n_samples,
            "window_seconds": self.window_seconds,
            "corotation_included": self.corotation_included,
            "wind_included": self.wind_included,
            "note": (
                "inertial_orbital_energy_change_J_kg is dot(a_drag, v) and is what "
                "the ORBIT loses; atmosphere_relative_dissipation_J_kg is "
                "-dot(a_drag, v_rel) and is what the GAS receives. They are not "
                "equal when the atmosphere co-rotates or a wind blows. "
                "drag_delta_v_m_s is the time integral of |a_drag| and is a "
                "delta-v in m/s, NOT an energy."
            ),
        }


def integrate_ledger(
    times_utc: Sequence[_dt.datetime],
    positions_m: Sequence[npt.ArrayLike],
    velocities_m_s: Sequence[npt.ArrayLike],
    densities_kg_m3: Sequence[float],
    inverse_ballistic_coefficient_m2_kg: float,
    *,
    winds_m_s: Sequence[npt.ArrayLike] | None = None,
    include_corotation: bool = True,
) -> EnergyLedger:
    """Integrate both energy rates, and the delta-v, over the sampled window.

    Trapezoidal in time. All three integrals use the SAME samples and the same
    relative velocity, so the difference between the two energies is physical
    rather than an artefact of integrating them differently.
    """
    n = len(times_utc)
    if n < 2:
        raise EnergyLedgerError(
            f"{n} sample(s); at least 2 are needed to integrate over a window"
        )
    lengths = {n, len(positions_m), len(velocities_m_s), len(densities_kg_m3)}
    if winds_m_s is not None:
        lengths.add(len(winds_m_s))
    if lengths != {n}:
        raise EnergyLedgerError(
            f"sample sequences have lengths {sorted(lengths)}; they must all match"
        )

    seconds = np.array([t.timestamp() for t in times_utc], dtype=np.float64)
    if not np.all(np.diff(seconds) > 0):
        raise EnergyLedgerError("sample times are not strictly increasing")

    orbital_rate = np.zeros(n, dtype=np.float64)
    dissipation_rate = np.zeros(n, dtype=np.float64)
    acceleration_magnitude = np.zeros(n, dtype=np.float64)

    for i in range(n):
        position = np.asarray(positions_m[i], dtype=np.float64)
        velocity = np.asarray(velocities_m_s[i], dtype=np.float64)
        wind = None if winds_m_s is None else np.asarray(winds_m_s[i], dtype=np.float64)
        v_rel = relative_velocity_m_s(
            position, velocity, wind_m_s=wind, include_corotation=include_corotation
        )
        a_drag = drag_acceleration_m_s2(
            inverse_ballistic_coefficient_m2_kg, float(densities_kg_m3[i]), v_rel
        )
        orbital_rate[i] = inertial_orbital_energy_rate_w_kg(a_drag, velocity)
        dissipation_rate[i] = atmosphere_relative_dissipation_rate_w_kg(a_drag, v_rel)
        acceleration_magnitude[i] = float(np.linalg.norm(a_drag))

    return EnergyLedger(
        inertial_orbital_energy_change_j_kg=float(
            np.trapezoid(orbital_rate, seconds)
        ),
        atmosphere_relative_dissipation_j_kg=float(
            np.trapezoid(dissipation_rate, seconds)
        ),
        drag_delta_v_m_s=float(np.trapezoid(acceleration_magnitude, seconds)),
        n_samples=n,
        window_seconds=float(seconds[-1] - seconds[0]),
        corotation_included=include_corotation,
        wind_included=winds_m_s is not None,
    )
