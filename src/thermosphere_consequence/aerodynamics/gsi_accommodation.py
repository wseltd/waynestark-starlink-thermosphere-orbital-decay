"""AO-conditioned accommodation: the Pilinski altitude prior and the Langmuir branch.

Two ways of stating the same physical claim — that adsorbed atomic oxygen
drives conventional exposed LEO surfaces toward high, nearly diffuse energy
accommodation below ~300 km — with very different evidential standing, kept
as SEPARATE providers so a result always says which one produced it.

BRANCH A, the nominal public-data prior. Pilinski, Argrow & Palo (2010,
J. Spacecraft & Rockets 47(6):951, doi:10.2514/1.49330) published
orbit-derived effective energy-accommodation coefficients by altitude and
solar activity. The primary paper is paywalled; the table below is carried
as reproduced in this project's research brief and controlling instruction,
and is consistent with the ADBSat verification paper's independent use of
the same model (alpha = 1 at 200 km, 0.97 at 300 km) — the access limitation
is recorded in planning/species_gsi_worklog.md rather than hidden.

BRANCH B, the sensitivity transfer. The same authors' Langmuir-isotherm form

    alpha_E = K' * n_O * T_i / (1 + K' * n_O * T_i)

with historical fitted K' ~ 7.5e-17 m^3/K. That K' was calibrated against a
2010-era atmosphere model, and Pilinski et al. caution that it must be
recalibrated when the atmosphere model changes materially. Combining the
historical constant with NRLMSIS 2.1 is therefore a COEFFICIENT TRANSFER,
not an independently validated model — the provider's name says so, K' is a
swept parameter, and the exact T_i convention of the original fit (which
this project could not verify from an accessible primary source) is absorbed
into that sweep, since only the product K'*T_i enters.

Neither branch supplies per-species values: the orbit-effective coefficient
describes the AO-conditioned SURFACE, and every incident species meets the
same surface. Per-species overrides remain the job of
:class:`~.accommodation.LiteratureAccommodation`'s sourced-override seam.
"""

from __future__ import annotations

import math
from typing import Literal

from ..domain.species import get_species
from .kernels import reemission_temperature_k

__all__ = [
    "HISTORICAL_LANGMUIR_K_PRIME_M3_PER_K",
    "LangmuirAOAccommodation",
    "NOMINAL_AO_ENVELOPE",
    "PILINSKI_TABLE",
    "PilinskiAltitudeAccommodation",
    "STRESS_AO_ENVELOPE",
    "incident_kinetic_temperature_k",
]

#: The published orbit-derived effective alpha_E by altitude and activity.
#: Keys are altitude in metres.
#:
#: PROVENANCE, upgraded 11 September 2026. These values were carried here from a
#: project research brief rather than from the primary, which is paywalled. An
#: independent retrieval of Pilinski, Argrow & Palo (2010), Table 2, p. 953
#: ("Accommodation coefficients computed by Bowman and Moe") reports exactly the
#: six numbers below. The transcription is therefore corroborated by a reader
#: who saw the primary table; the primary itself is still not hash-pinned in
#: this repository, because no stable direct file URL was obtained. Corroborated
#: is not the same as retrieved, and the distinction is kept.
#:
#: The 0.998 this project applies at 210 km is the linear interpolation between
#: the 200 km and 250 km solar-maximum knots, which is what the class below
#: computes; it is not a value printed in the source.
PILINSKI_TABLE: dict[str, dict[float, float]] = {
    "solar_maximum": {200_000.0: 1.00, 250_000.0: 0.99, 300_000.0: 0.98},
    "solar_minimum": {200_000.0: 0.99, 250_000.0: 0.97, 300_000.0: 0.93},
}

#: The nominal 180-300 km AO-conditioned envelope the brief supports.
NOMINAL_AO_ENVELOPE: tuple[float, float] = (0.93, 1.00)

#: The deliberately conservative stress envelope. 0.85 is NOT a measured
#: Starlink lower bound; it is a stress case near the edge of the Pilinski
#: model's advised validity (the authors advise against relying on results
#: once accommodation falls much below ~0.85).
STRESS_AO_ENVELOPE: tuple[float, float] = (0.85, 1.00)

#: Pilinski et al.'s historical fitted Langmuir constant [m^3/K]. Tied to the
#: atmosphere model of the original fit; a transfer, not a universal constant.
HISTORICAL_LANGMUIR_K_PRIME_M3_PER_K = 7.5e-17

_ACTIVITIES = ("solar_maximum", "solar_minimum")

#: The altitude interval the published table actually covers.
_TABLE_LOW_M = 200_000.0
_TABLE_HIGH_M = 300_000.0

#: The study interval's lower edge. Between here and the table's lower edge
#: the 200 km value is HELD, with the reason stated on the provider: AO
#: coverage increases downward, so the 200 km row is a lower bound on
#: accommodation there, and the model's own validity improves as altitude
#: falls. This is a labelled edge-hold, not a silent extrapolation.
_STUDY_LOW_M = 180_000.0


def incident_kinetic_temperature_k(speed_m_s: float, molecular_mass_kg: float) -> float:
    """T_i = m V^2 / (3 k_B) — the kinetic temperature of a monoenergetic beam.

    A DIAGNOSTIC, and explicitly NOT the temperature of the Langmuir isotherm.
    It is retained so the size of a corrected error stays computable.

    WHAT IT WAS WRONGLY USED FOR. Branch B once passed this to
    :class:`LangmuirAOAccommodation`. The published isotherm is in the
    atomic-oxygen partial pressure at the AMBIENT temperature,
    P_O = n_AO T_inf (R / M_AO), so the substitution inflated the isotherm's
    loading by the ratio T_i / T_inf, about 52 at 210 km.

    WHY THE OLD DEFENCE FAILED, since it reads plausibly. This docstring
    previously argued that only the product K' * T enters the isotherm, so any
    convention difference is a rescaling of K' that the sensitivity sweep
    already covers. That is true only if the two temperatures differ by a
    CONSTANT factor. They do not: T_i carries the mixture's mean particle mass
    and the relative speed, while T_inf is an independent state variable, so
    their ratio varies with composition, altitude and epoch. The argument also
    quietly coupled the factorial's A and C axes, because a T_i built from
    m_bar changes when composition changes.
    """
    if speed_m_s <= 0.0:
        raise ValueError(f"speed must be positive, got {speed_m_s}")
    if molecular_mass_kg <= 0.0:
        raise ValueError(f"molecular mass must be positive, got {molecular_mass_kg}")
    k_b = 1.380649e-23
    return molecular_mass_kg * speed_m_s * speed_m_s / (3.0 * k_b)


class PilinskiAltitudeAccommodation:
    """Branch A: the orbit-effective AO-conditioned prior, by altitude and activity.

    Linear interpolation INSIDE the published 200-300 km interval; a labelled
    edge-hold from 180 km up to the table's lower edge; a refusal above
    300 km, where this project has no business and the prior thins.
    """

    def __init__(
        self,
        *,
        altitude_m: float,
        activity: Literal["solar_maximum", "solar_minimum"],
        wall_temperature_k: float,
    ) -> None:
        if activity not in _ACTIVITIES:
            raise ValueError(
                f"activity must be one of {_ACTIVITIES}, got {activity!r}"
            )
        if wall_temperature_k <= 0.0:
            raise ValueError(
                f"wall temperature must be positive, got {wall_temperature_k}"
            )
        if not math.isfinite(altitude_m):
            raise ValueError(f"altitude must be finite, got {altitude_m}")
        if altitude_m > _TABLE_HIGH_M:
            raise ValueError(
                f"altitude {altitude_m/1000:.0f} km is above the published "
                f"table's 300 km upper edge; the prior is not extrapolated "
                f"upward, where accommodation falls and the model's own "
                f"authors advise caution"
            )
        if altitude_m < _STUDY_LOW_M:
            raise ValueError(
                f"altitude {altitude_m/1000:.0f} km is below the 180 km study "
                f"interval; nothing below it was sourced"
            )
        self.name = "pilinski_altitude_prior"
        self.activity = activity
        self.altitude_m = float(altitude_m)
        self.wall_temperature_k = float(wall_temperature_k)
        self.edge_held = altitude_m < _TABLE_LOW_M
        self._alpha = self._interpolate(max(altitude_m, _TABLE_LOW_M), activity)

    @staticmethod
    def _interpolate(altitude_m: float, activity: str) -> float:
        table = PILINSKI_TABLE[activity]
        knots = sorted(table)
        for low, high in zip(knots[:-1], knots[1:], strict=False):
            if low <= altitude_m <= high:
                fraction = (altitude_m - low) / (high - low)
                return table[low] + fraction * (table[high] - table[low])
        raise AssertionError(f"altitude {altitude_m} escaped the guarded interval")

    def energy_accommodation(self, species: str) -> float:
        """The orbit-effective coefficient; the same AO-conditioned surface
        meets every species, so the value is species-independent by design."""
        get_species(species)  # an invented species raises rather than passing
        return self._alpha

    def reemission_temperature_k(
        self, species: str, speed_ratio: float, freestream_temperature_k: float
    ) -> float:
        """Re-emission temperature for one species at this speed ratio [K]."""
        return reemission_temperature_k(
            self.energy_accommodation(species),
            speed_ratio,
            self.wall_temperature_k,
            freestream_temperature_k,
        )

    def provenance(self) -> dict[str, str]:
        """Where the value came from, including the access limitation."""
        return {
            "provider": self.name,
            "activity": self.activity,
            "altitude_m": f"{self.altitude_m}",
            "alpha": f"{self._alpha:.4f}",
            "edge_held": (
                "yes - altitude below the table's 200 km lower edge uses the "
                "200 km value (AO coverage increases downward); a labelled "
                "hold, not an extrapolation"
                if self.edge_held
                else "no"
            ),
            "table_source": (
                "Pilinski, Argrow & Palo (2010) doi:10.2514/1.49330, as "
                "reproduced in the project research brief; primary paywalled "
                "- access limitation recorded in "
                "planning/species_gsi_worklog.md"
            ),
            "nominal_envelope": f"{NOMINAL_AO_ENVELOPE}",
            "stress_envelope": (
                f"{STRESS_AO_ENVELOPE} - 0.85 is a conservative stress bound, "
                f"not a measured Starlink value"
            ),
        }


class LangmuirAOAccommodation:
    """Branch B: historical_coefficient_transfer_sensitivity.

    The Langmuir-isotherm form with the HISTORICAL K'. Never the nominal:
    the fitted constant belongs to the atmosphere model it was calibrated
    against, and pairing it with NRLMSIS 2.1 is a labelled transfer whose
    K' the sensitivity sweep varies.

    WHICH TEMPERATURE ENTERS THE ISOTHERM, and a correction. The published
    form is an adsorption isotherm in the atomic-oxygen PARTIAL PRESSURE,

        theta = K P_O / (1 + K P_O),    P_O = n_AO T_inf (R / M_AO),

    i.e. the ideal-gas partial pressure at the AMBIENT freestream neutral
    temperature. An earlier version of this class was given the DIRECTED
    kinetic temperature of the oncoming beam, m_bar V_rel^2 / (3 k_B), which
    at 210 km is about 47,800 K against an ambient 924 K - a factor of 52 in
    the isotherm's loading. Both have units of kelvin and both drive alpha
    towards unity, which is exactly why the substitution survived review: it
    could not be caught by a dimension check or by the sign of the answer.

    The parameter is therefore named for the quantity the source defines.
    Passing a beam temperature here is a modelling choice that departs from
    the published calibration, and the constant K' carries that calibration.
    """

    def __init__(
        self,
        *,
        atomic_oxygen_number_density_m3: float,
        ambient_temperature_k: float,
        wall_temperature_k: float,
        k_prime_m3_per_k: float = HISTORICAL_LANGMUIR_K_PRIME_M3_PER_K,
    ) -> None:
        if atomic_oxygen_number_density_m3 < 0.0:
            raise ValueError(
                f"n_O must be non-negative, got {atomic_oxygen_number_density_m3}"
            )
        if ambient_temperature_k <= 0.0:
            raise ValueError(
                f"ambient temperature must be positive, got "
                f"{ambient_temperature_k}"
            )
        if wall_temperature_k <= 0.0:
            raise ValueError(
                f"wall temperature must be positive, got {wall_temperature_k}"
            )
        if k_prime_m3_per_k <= 0.0:
            raise ValueError(f"K' must be positive, got {k_prime_m3_per_k}")
        self.name = "langmuir_ao_historical_coefficient_transfer_sensitivity"
        self.wall_temperature_k = float(wall_temperature_k)
        self.k_prime_m3_per_k = float(k_prime_m3_per_k)
        self.atomic_oxygen_number_density_m3 = float(atomic_oxygen_number_density_m3)
        self.ambient_temperature_k = float(ambient_temperature_k)
        loading = (
            self.k_prime_m3_per_k
            * self.atomic_oxygen_number_density_m3
            * self.ambient_temperature_k
        )
        self._alpha = loading / (1.0 + loading)

    def energy_accommodation(self, species: str) -> float:
        """AO surface coverage sets one effective coefficient for all species."""
        get_species(species)
        return self._alpha

    def reemission_temperature_k(
        self, species: str, speed_ratio: float, freestream_temperature_k: float
    ) -> float:
        """Re-emission temperature for one species at this speed ratio [K]."""
        return reemission_temperature_k(
            self.energy_accommodation(species),
            speed_ratio,
            self.wall_temperature_k,
            freestream_temperature_k,
        )

    def provenance(self) -> dict[str, str]:
        """Everything a report needs to label this branch correctly."""
        return {
            "provider": self.name,
            "alpha": f"{self._alpha:.4f}",
            "k_prime_m3_per_k": f"{self.k_prime_m3_per_k:.3e}",
            "historical_k_prime_m3_per_k": (
                f"{HISTORICAL_LANGMUIR_K_PRIME_M3_PER_K:.3e}"
            ),
            "n_o_m3": f"{self.atomic_oxygen_number_density_m3:.4e}",
            "ambient_temperature_k": f"{self.ambient_temperature_k:.1f}",
            "isotherm_temperature_convention": (
                "AMBIENT freestream neutral temperature, per the published "
                "partial-pressure form P_O = n_AO T_inf (R / M_AO). NOT the "
                "directed beam temperature m_bar V^2 / (3 k_B), which an "
                "earlier version of this project used in error"
            ),
            "status": (
                "historical_coefficient_transfer_sensitivity - the fitted K' "
                "belongs to the 2010-era atmosphere model; pairing it with "
                "NRLMSIS 2.x is a labelled transfer, never independently "
                "validated, never the nominal"
            ),
        }
