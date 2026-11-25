"""Per-facet, per-species gas-surface interaction kernels.

Each kernel answers the same question for one flat facet and one gas species:
given the speed ratio, the incidence cosine and the surface's thermal state,
what are the drag and lift coefficients REFERENCED TO THAT FACET'S OWN AREA?

FORCE DIRECTION, and the reason lift is signed the way it is. Write u_hat for
the direction the free stream travels, n_hat for the facet's outward normal,
gamma = -n_hat . u_hat = cos(theta) and ell = sin(theta). Decompose

    n_hat = -gamma * u_hat + ell * w_hat        (w_hat perpendicular to u_hat)

Then the force on one facet of area A under dynamic pressure q is

    F = q * A * (C_D * u_hat  -  C_L * w_hat)

Drag is along the flow; lift is perpendicular to it, in the plane containing
the flow and the normal, positive when it pushes the facet the way a plate at
incidence is pushed. Getting the lift sign wrong leaves drag untouched and
quietly rotates the total force, which is why the decomposition is written out
here and checked against a hand-integrated limit in the tests.

DERIVATION, so the coefficients are not three unexplained formulae. For a
Maxwellian at T_inf drifting at V, the incident normal and tangential momentum
fluxes on a surface element are the standard Schaaf & Chambre results

    p_i / q = gamma*P/sqrt(pi) + G*Z + gamma^2*Z
    tau_i / q = ell*P/sqrt(pi) + ell*gamma*Z

with P = exp(-(gamma*s)^2)/s, G = 1/(2 s^2), Q = 1 + G, Z = 1 + erf(gamma*s).
Diffuse re-emission adds a purely normal term p_re/q = (V_re/V_inf)/2 *
(P + sqrt(pi)*gamma*Z) and carries away no tangential momentum; specular
reflection doubles the normal term and cancels the tangential one exactly.
Resolving p and tau onto u_hat and w_hat gives, for the diffuse case,

    C_D = P/sqrt(pi) + gamma*Q*Z + (gamma/2)*(V_re/V_inf)*(gamma*sqrt(pi)*Z + P)
    C_L = ell*G*Z   + (ell/2)  *(V_re/V_inf)*(gamma*sqrt(pi)*Z + P)

The C_D line is exactly ``sentman.sentman_plate_drag_coefficient``, which is
already validated against its analytic anchors, so it is CALLED rather than
transcribed a second time. Only the C_L line is new algebra and it is
implemented once, here.

VALIDITY, common to every kernel in this module:
  * free-molecular flow only, Knudsen number >> 1. At 210 km the mean free
    path is of order a kilometre against a metre-scale vehicle, so Kn ~ 10^3.
    These forms are meaningless in the transitional and continuum regimes.
  * flat facets. Curvature within a facet is not modelled; a curved surface
    must be discretised until the flat assumption holds.
  * no facet-to-facet re-impingement. A molecule that leaves one facet is
    assumed to escape rather than strike another. That is exact for a convex
    body and an approximation for a concave one.
  * a single accommodation coefficient per species per surface, applied to
    energy. Separate normal and tangential momentum accommodation is the CLL
    model, which this repository cannot source coefficients for; see
    :func:`cll`.

Primary sources:
  Sentman, L.H. (1961) LMSC-448514.
  Schaaf, S.A. & Chambre, P.L. (1961) "Flow of Rarefied Gases", Princeton.
  Mehta, P.M. et al. (2014) Adv. Space Res. 54:1590 (Sentman, Schaaf-Chambre,
    DRIA, CLL and Maxwell formulations side by side).
  Moe, K. & Moe, M.M. (2005) Planet. Space Sci. 53:793.

Canonical tickets: P11_12_T004, P11_12_T007, P11_12_T008.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from .cll_constants import cll_constants
from .geometry import EDGE_ON_COSINE_TOLERANCE
from .sentman import REEMISSION_FACTORS, sentman_plate_drag_coefficient

__all__ = [
    "FacetCoefficients",
    "cll",
    "dria",
    "maxwell_mixture",
    "pressure_shear_from_drag_lift",
    "reemission_temperature_k",
    "schaaf_chambre",
    "sentman_diffuse",
    "specular",
]


class FacetCoefficients(NamedTuple):
    """(drag, lift) for one facet and one species, referenced to the facet area.

    A NamedTuple so it unpacks as ``cd, cl = kernel(...)`` while still naming
    its fields at the call sites that only want one of them.
    """

    drag: float
    lift: float


_ZERO = FacetCoefficients(0.0, 0.0)


def _validate(speed_ratio: float, cos_incidence: float) -> None:
    if not math.isfinite(speed_ratio) or speed_ratio <= 0.0:
        raise ValueError(f"speed ratio must be positive and finite, got {speed_ratio}")
    if not math.isfinite(cos_incidence):
        raise ValueError(f"incidence cosine must be finite, got {cos_incidence}")
    if cos_incidence > 1.0 + 1e-9:
        raise ValueError(f"incidence cosine cannot exceed 1, got {cos_incidence}")


def _theta_rad(cos_incidence: float) -> float:
    """The incidence angle, clamping the cosine into acos's domain."""
    return math.acos(min(1.0, max(-1.0, cos_incidence)))


def reemission_temperature_k(
    energy_accommodation: float,
    speed_ratio: float,
    wall_temperature_k: float,
    freestream_temperature_k: float,
) -> float:
    """Temperature of the re-emitted flux under partial accommodation [K].

    Under the standard energy-accommodation definition the re-emitted flux
    leaves with most-probable speed V_re where

        (V_re/V_inf)^2 = (1/2) * (1 + alpha*(2*T_w/(T_inf*s^2) - 1))

    Writing that as a temperature via V_re^2 = 2*k*T_re/m gives

        T_re = alpha*T_w + (1 - alpha) * T_inf * s^2 / 2

    which is a linear blend between the wall and HALF the incident kinetic
    temperature m*V_inf^2/(2k). alpha = 1 lands exactly on the wall, alpha = 0
    leaves the molecule as energetic as it arrived. Exposed as a temperature
    rather than kept as a speed ratio because a temperature is the quantity a
    thermal model and an accommodation provider can both speak about.
    """
    if not 0.0 <= energy_accommodation <= 1.0:
        raise ValueError(
            f"energy accommodation must lie in [0, 1], got {energy_accommodation}"
        )
    if wall_temperature_k <= 0.0 or freestream_temperature_k <= 0.0:
        raise ValueError("temperatures must be positive")
    if speed_ratio <= 0.0:
        raise ValueError(f"speed ratio must be positive, got {speed_ratio}")
    incident_kinetic_k = 0.5 * freestream_temperature_k * speed_ratio * speed_ratio
    return energy_accommodation * wall_temperature_k + (
        1.0 - energy_accommodation
    ) * incident_kinetic_k


def _reemission_speed_ratio(
    reemission_temperature: float, speed_ratio: float, freestream_temperature_k: float
) -> float:
    """V_re / V_inf from the re-emission temperature."""
    return math.sqrt(
        reemission_temperature / (freestream_temperature_k * speed_ratio * speed_ratio)
    )


def _lift_coefficient(
    speed_ratio: float, cos_incidence: float, reemission_speed_ratio: float
) -> float:
    """C_L = ell*G*Z + (ell/2)*(V_re/V_inf)*(gamma*sqrt(pi)*Z + P).

    The one piece of algebra in this module that ``sentman.py`` does not
    already provide. Zero at normal incidence by construction (ell = 0), which
    is the symmetry a flat plate square to the flow must obey.
    """
    gamma = cos_incidence
    ell = math.sqrt(max(0.0, 1.0 - gamma * gamma))
    if ell <= EDGE_ON_COSINE_TOLERANCE:
        return 0.0
    s = speed_ratio
    p = math.exp(-((gamma * s) ** 2)) / s
    g = 1.0 / (2.0 * s * s)
    z = 1.0 + math.erf(gamma * s)
    return ell * g * z + (ell / 2.0) * reemission_speed_ratio * (
        gamma * math.sqrt(math.pi) * z + p
    )


def sentman_diffuse(
    speed_ratio: float,
    cos_incidence: float,
    accommodation: float,
    wall_temperature_k: float,
    freestream_temperature_k: float,
    *,
    reemission_factor: str = "koppenwallner_half",
) -> FacetCoefficients:
    """Sentman diffuse reflection with energy accommodation ``accommodation``.

    Molecules are absorbed and re-emitted with a cosine angular distribution
    about the surface normal, at a temperature set by the accommodation
    coefficient. This is the default model for spacecraft surfaces in LEO:
    Moe & Moe (2005) find near-complete accommodation on the adsorbed atomic
    oxygen layer that covers everything below roughly 500 km.

    VALIDITY, in addition to the module-wide limits: assumes a fully
    thermalised cosine re-emission. It cannot represent the lobed reflection
    that a clean, smooth surface produces at grazing incidence — for that the
    CLL model is needed, and :func:`cll` explains why it is unavailable here.

    The drag coefficient is delegated to the validated
    ``sentman.sentman_plate_drag_coefficient``; only the lift is computed here.
    """
    _validate(speed_ratio, cos_incidence)
    # NO INCIDENCE GUARD. The closed form is continuous through grazing and
    # remains valid for cos < 0: checked against the quadrature oracle from 90
    # to 180 degrees for N2, O, He and H, agreeing to a ratio of 1.0000
    # everywhere the oracle can resolve the answer. A rear-facing facet still
    # receives the thermal tail of the shifted Maxwellian. Which facets the gas
    # can REACH is geometry and is decided in faceted.py; it is not decided by
    # the sign of a cosine.
    drag = sentman_plate_drag_coefficient(
        speed_ratio,
        _theta_rad(cos_incidence),
        accommodation,
        wall_temperature_k,
        freestream_temperature_k,
        reemission_factor=reemission_factor,
    )
    constant = REEMISSION_FACTORS[reemission_factor]
    ratio = wall_temperature_k / (
        freestream_temperature_k * speed_ratio * speed_ratio
    )
    v_re = math.sqrt(
        max(constant * (1.0 - accommodation) + accommodation * ratio, 0.0)
    )
    lift = _lift_coefficient(speed_ratio, cos_incidence, v_re)
    return FacetCoefficients(drag, lift)


def specular(speed_ratio: float, cos_incidence: float) -> FacetCoefficients:
    """Perfectly specular reflection: mirror the normal velocity, keep the tangential.

    Closed form. Specular reflection doubles the incident normal momentum flux
    and cancels the tangential one exactly, so the force is purely normal:

        C_p = 2 * (gamma*P/sqrt(pi) + G*Z + gamma^2*Z)
        C_D = gamma * C_p ,   C_L = ell * C_p

    Takes no accommodation coefficient and no wall temperature BY DEFINITION:
    a specular reflection exchanges no energy with the surface, so the wall's
    temperature cannot enter. The hyperthermal normal-incidence limit is
    C_D -> 4, twice the diffuse cold-wall limit of 2, because the molecule's
    normal momentum is reversed rather than merely absorbed.

    VALIDITY: an idealisation, and a COLD-BEAM one. Real spacecraft surfaces in
    LEO are neither atomically smooth nor clean, and Moe & Moe (2005) show the
    adsorbed-oxygen layer drives them close to fully diffuse. Use this as the
    upper bound of a sensitivity envelope, never as a nominal.

    WHAT THIS KERNEL IS NOT VALIDATED FOR, stated because the diffuse kernels
    beside it now are. The diffuse family was checked against direct
    quadrature of the shifted Maxwellian across the whole incidence range,
    including past grazing. This kernel was NOT: it is used here as a
    hyperthermal cold-beam limiting case, in which the incident distribution is
    treated as a directed beam with no thermal spread, and its guard below
    reflects that idealisation rather than a shifted-Maxwellian result.

    Two consequences a reader should not have to infer. Zero TANGENTIAL shear at
    exactly grazing is a real statement about specular reflection - the
    molecule keeps its tangential velocity - and it holds for any incident
    distribution. Zero NORMAL momentum transfer on a leeward face is NOT: at
    finite temperature the thermal tail reaches a downstream-facing surface and
    a specular reflection reverses its normal component, so a shifted-Maxwellian
    specular kernel would return a small non-zero pressure there. This one
    returns zero. That is a declared limitation of the cold-beam idealisation,
    not a claim about the physics, and no result in this project rests on
    specular leeward behaviour.
    """
    _validate(speed_ratio, cos_incidence)
    # SPECULAR KEEPS A SYMMETRIC GUARD, for two different reasons that should
    # not be conflated. At exactly grazing: a specularly reflected molecule
    # keeps its tangential velocity, so a plate whose mean flow is parallel
    # transfers no net tangential momentum and feels no grazing drag. That is
    # physics and holds for any incident distribution. On the leeward side: this
    # is a COLD-BEAM idealisation with no thermal spread, so nothing reaches a
    # downstream-facing surface by construction. A shifted-Maxwellian specular
    # kernel would return a small non-zero NORMAL pressure there. Returning zero
    # is a limit of the idealisation and is labelled as one in the docstring; it
    # is not the same statement as the grazing one above.
    if abs(cos_incidence) <= EDGE_ON_COSINE_TOLERANCE or cos_incidence < 0.0:
        return _ZERO
    gamma = cos_incidence
    ell = math.sqrt(max(0.0, 1.0 - gamma * gamma))
    s = speed_ratio
    p = math.exp(-((gamma * s) ** 2)) / s
    g = 1.0 / (2.0 * s * s)
    z = 1.0 + math.erf(gamma * s)
    pressure = 2.0 * (gamma * p / math.sqrt(math.pi) + g * z + gamma * gamma * z)
    return FacetCoefficients(gamma * pressure, ell * pressure)


def maxwell_mixture(
    speed_ratio: float,
    cos_incidence: float,
    accommodation: float,
    wall_temperature_k: float,
    freestream_temperature_k: float,
    *,
    specular_fraction: float,
) -> FacetCoefficients:
    """Maxwell's two-population model: a specular fraction, the rest diffuse.

    A fraction ``specular_fraction`` of incident molecules reflects specularly
    and the remainder reflects diffusely with the given energy accommodation.
    Because both populations act on the same facet under the same free stream,
    and free-molecular fluxes superpose linearly with no collisions between
    them, the mixture coefficient is the plain linear blend

        C = (1 - f) * C_diffuse + f * C_specular

    VALIDITY: one parameter is asked to stand for the whole gas-surface
    interaction, so the model can match either the normal or the tangential
    momentum transfer but not both independently. That limitation is exactly
    what CLL exists to remove. Physically the two populations are a
    convenience, not a claim that some molecules truly reflect specularly.
    """
    if not 0.0 <= specular_fraction <= 1.0:
        raise ValueError(
            f"specular fraction must lie in [0, 1], got {specular_fraction}"
        )
    diffuse = sentman_diffuse(
        speed_ratio,
        cos_incidence,
        accommodation,
        wall_temperature_k,
        freestream_temperature_k,
    )
    mirror = specular(speed_ratio, cos_incidence)
    blend = specular_fraction
    return FacetCoefficients(
        (1.0 - blend) * diffuse.drag + blend * mirror.drag,
        (1.0 - blend) * diffuse.lift + blend * mirror.lift,
    )


def dria(
    speed_ratio: float,
    cos_incidence: float,
    energy_accommodation: float,
    wall_temperature_k: float,
    freestream_temperature_k: float,
    *,
    reemission_temperature: float | None = None,
) -> FacetCoefficients:
    """DRIA: Diffuse Reflection with Incomplete Accommodation (Mehta et al. 2014).

    DRIA is the Sentman closed form read as a physical model rather than as
    algebra: molecules reflect diffusely, but they leave at a temperature
    between the wall and the incident stream, set by the energy accommodation
    coefficient. This entry point exists because the literature names the
    model this way, and because it lets the re-emission temperature be
    supplied DIRECTLY by a thermal or accommodation model instead of being
    derived from alpha and the wall temperature alone.

    The identity that makes the delegation exact: diffuse re-emission at
    temperature T_re is indistinguishable from FULL accommodation to a wall
    held at T_re, because T_re is the only way the surface enters the force.
    So this calls the validated Sentman coefficient with alpha = 1 and
    T_w = T_re, and when ``reemission_temperature`` is left to default the two
    routes agree to the last bit — a property the tests pin.

    VALIDITY: in addition to the module-wide limits, DRIA attributes the whole
    gas-surface interaction to a single ENERGY accommodation coefficient. It
    cannot represent a surface whose normal and tangential momentum
    accommodate differently.
    """
    _validate(speed_ratio, cos_incidence)
    # NO INCIDENCE GUARD. The closed form is continuous through grazing and
    # remains valid for cos < 0: checked against the quadrature oracle from 90
    # to 180 degrees for N2, O, He and H, agreeing to a ratio of 1.0000
    # everywhere the oracle can resolve the answer. A rear-facing facet still
    # receives the thermal tail of the shifted Maxwellian. Which facets the gas
    # can REACH is geometry and is decided in faceted.py; it is not decided by
    # the sign of a cosine.
    if reemission_temperature is None:
        reemission_temperature = reemission_temperature_k(
            energy_accommodation,
            speed_ratio,
            wall_temperature_k,
            freestream_temperature_k,
        )
    elif reemission_temperature <= 0.0:
        raise ValueError(
            f"re-emission temperature must be positive, got {reemission_temperature}"
        )
    drag = sentman_plate_drag_coefficient(
        speed_ratio,
        _theta_rad(cos_incidence),
        1.0,
        reemission_temperature,
        freestream_temperature_k,
    )
    lift = _lift_coefficient(
        speed_ratio,
        cos_incidence,
        _reemission_speed_ratio(
            reemission_temperature, speed_ratio, freestream_temperature_k
        ),
    )
    return FacetCoefficients(drag, lift)


def pressure_shear_from_drag_lift(
    drag: float, lift: float, cos_incidence: float
) -> tuple[float, float]:
    """(c_p, c_tau) from (c_d, c_l) — the exact rotation, not an approximation.

    The pair (drag, lift) lives in flow coordinates and (pressure, shear) in
    facet coordinates; they are the same force decomposed two ways, related by
    the involutory rotation of the ADBSat methodology paper (eqs. 4-5):

        c_p   = c_d * cos(delta) + c_l * sin(delta)
        c_tau = c_d * sin(delta) - c_l * cos(delta)

    Exposed so a per-facet result can report a pressure/shear split without
    every kernel growing two more return values.
    """
    gamma = cos_incidence
    ell = math.sqrt(max(0.0, 1.0 - gamma * gamma))
    return (drag * gamma + lift * ell, drag * ell - lift * gamma)


def schaaf_chambre(
    speed_ratio: float,
    cos_incidence: float,
    normal_momentum_accommodation: float,
    tangential_momentum_accommodation: float,
    wall_temperature_k: float,
    freestream_temperature_k: float,
) -> FacetCoefficients:
    """Schaaf & Chambre flat-plate coefficients with separate sigma_N, sigma_T.

    The two-coefficient description of the ADBSat methodology paper
    (arXiv:2104.05543v2, eqs. 9-10):

        C_p = (1/s^2) [ ((2-sigma_N)/sqrt(pi) * s*g + (sigma_N/2)*sqrt(Tw/Tinf)) * E
              + ( (2-sigma_N)(s^2 g^2 + 1/2)
                  + (sigma_N/2)*sqrt(pi*Tw/Tinf)*s*g ) * Z ]
        C_tau = (sigma_T * l/(s*sqrt(pi))) * [ E + s*sqrt(pi)*g*Z ]

    with g = cos(delta), l = sin(delta), E = exp(-s^2 g^2), Z = 1+erf(s*g).
    Limits the tests pin: sigma_N = sigma_T = 0 reproduces the specular
    kernel exactly, and the alpha_N = 1 branch of :func:`cll` is this
    pressure form by construction.
    """
    _validate(speed_ratio, cos_incidence)
    for name, sigma in (
        ("sigma_N", normal_momentum_accommodation),
        ("sigma_T", tangential_momentum_accommodation),
    ):
        if not 0.0 <= sigma <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {sigma}")
    if wall_temperature_k <= 0.0 or freestream_temperature_k <= 0.0:
        raise ValueError("temperatures must be positive")
    # NO INCIDENCE GUARD. The closed form is continuous through grazing and
    # remains valid for cos < 0: checked against the quadrature oracle from 90
    # to 180 degrees for N2, O, He and H, agreeing to a ratio of 1.0000
    # everywhere the oracle can resolve the answer. A rear-facing facet still
    # receives the thermal tail of the shifted Maxwellian. Which facets the gas
    # can REACH is geometry and is decided in faceted.py; it is not decided by
    # the sign of a cosine.
    s = speed_ratio
    gamma = cos_incidence
    ell = math.sqrt(max(0.0, 1.0 - gamma * gamma))
    sigma_n = normal_momentum_accommodation
    sigma_t = tangential_momentum_accommodation
    t_ratio = wall_temperature_k / freestream_temperature_k
    exp_term = math.exp(-(s * gamma) ** 2)
    z = 1.0 + math.erf(s * gamma)
    pressure = (
        ((2.0 - sigma_n) / math.sqrt(math.pi) * s * gamma
         + (sigma_n / 2.0) * math.sqrt(t_ratio)) * exp_term
        + ((2.0 - sigma_n) * (s * s * gamma * gamma + 0.5)
           + (sigma_n / 2.0) * math.sqrt(math.pi * t_ratio) * s * gamma) * z
    ) / (s * s)
    shear = (sigma_t * ell / (s * math.sqrt(math.pi))) * (
        exp_term + s * math.sqrt(math.pi) * gamma * z
    )
    return FacetCoefficients(
        pressure * gamma + shear * ell,
        pressure * ell - shear * gamma,
    )


#: Recorded because the earlier release refused to implement CLL at all, and
#: the refusal was pinned by a test. What has CHANGED, and what has not: the
#: closed-form equations and the species-fitted constants are now sourced and
#: verified against two independent published reproductions (see
#: cll_constants.py and planning/species_gsi_worklog.md). What is STILL not
#: sourced is any per-surface Starlink value for alpha_N or sigma_T — those
#: remain swept sensitivity parameters with stated envelopes, and any claim
#: that a particular pair is "the Starlink surface" remains prohibited.
CLL_COEFFICIENT_STATUS = (
    "CLL closed form: Walker, Mehta & Koller (2014) via the ADBSat "
    "methodology paper (arXiv:2104.05543v2, eqs. 11-14) and the Manchester "
    "reproduction (species table verified value-by-value). Surface inputs "
    "alpha_N and sigma_T are swept parameters, never measured Starlink "
    "values. Species without published constants (Ar, anomalous O, NO) are "
    "refused by cll_constants.cll_constants and their mass fraction is "
    "reported by the faceted model."
)


def cll(
    speed_ratio: float,
    cos_incidence: float,
    normal_energy_accommodation: float,
    tangential_momentum_accommodation: float,
    wall_temperature_k: float,
    freestream_temperature_k: float,
    *,
    species: str,
) -> FacetCoefficients:
    """Modified Walker-Mehta-Koller CLL coefficients for one species and facet.

    The closed-form approximation to CLL DSMC behaviour, exactly as published
    (ADBSat methodology paper eqs. 11-14; exponent delta_j + 1/2 confirmed by
    the Manchester reproduction):

        G1 = (1/sqrt(pi)) [ s*g*E + (sqrt(pi)/2)(1 + 2 s^2 g^2) Z ]
        G2 = (1/sqrt(pi)) [ E + s*sqrt(pi)*g*Z ]
        alpha_N < 1:
          C_p = (1/s^2) [ (1 + sqrt(1-alpha_N)) G1
                + (sqrt(pi) zeta_j/(2 s)) e^(-beta_j (1-alpha_N)^gamma_j)
                  (Tw/Tinf)^(delta_j + 1/2) G2 ]
        alpha_N = 1:
          C_p = Schaaf-Chambre pressure with sigma_N -> alpha_N
        C_tau = sigma_T * sin(delta) * G2 / s          (both branches)

    with g = cos(delta), E = exp(-s^2 g^2), Z = 1 + erf(s*g), and the species
    constants of :mod:`.cll_constants`. Species without published constants
    raise :class:`~.cll_constants.CllUnsupportedSpeciesError` — the Sentman
    baseline covers those, and the model comparison reports the excluded mass
    fraction rather than inventing numbers.

    The published model is a fit to DSMC and the ADBSat authors advise against
    using it independently of comparable DSMC cases; in this project it is an
    UNCERTAINTY BRANCH beside the Sentman nominal, never the nominal itself.
    """
    _validate(speed_ratio, cos_incidence)
    if not 0.0 <= normal_energy_accommodation <= 1.0:
        raise ValueError(
            f"alpha_N must lie in [0, 1], got {normal_energy_accommodation}"
        )
    if not 0.0 <= tangential_momentum_accommodation <= 1.0:
        raise ValueError(
            f"sigma_T must lie in [0, 1], got {tangential_momentum_accommodation}"
        )
    if wall_temperature_k <= 0.0 or freestream_temperature_k <= 0.0:
        raise ValueError("temperatures must be positive")
    # NO INCIDENCE GUARD. The closed form is continuous through grazing and
    # remains valid for cos < 0: checked against the quadrature oracle from 90
    # to 180 degrees for N2, O, He and H, agreeing to a ratio of 1.0000
    # everywhere the oracle can resolve the answer. A rear-facing facet still
    # receives the thermal tail of the shifted Maxwellian. Which facets the gas
    # can REACH is geometry and is decided in faceted.py; it is not decided by
    # the sign of a cosine.

    s = speed_ratio
    gamma = cos_incidence
    ell = math.sqrt(max(0.0, 1.0 - gamma * gamma))
    exp_term = math.exp(-(s * gamma) ** 2)
    z = 1.0 + math.erf(s * gamma)
    g2 = (exp_term + s * math.sqrt(math.pi) * gamma * z) / math.sqrt(math.pi)

    alpha_n = normal_energy_accommodation
    if alpha_n < 1.0:
        constants = cll_constants(species, alpha_n)
        g1 = (
            s * gamma * exp_term
            + (math.sqrt(math.pi) / 2.0) * (1.0 + 2.0 * s * s * gamma * gamma) * z
        ) / math.sqrt(math.pi)
        t_ratio = wall_temperature_k / freestream_temperature_k
        pressure = (
            (1.0 + math.sqrt(1.0 - alpha_n)) * g1
            + (math.sqrt(math.pi) * constants.zeta / (2.0 * s))
            * math.exp(-constants.beta * (1.0 - alpha_n) ** constants.gamma)
            * t_ratio ** (constants.delta + 0.5)
            * g2
        ) / (s * s)
    else:
        cll_constants(species, 0.0)  # refuse unsupported species on this branch too
        sc = schaaf_chambre(
            speed_ratio,
            cos_incidence,
            alpha_n,
            0.0,
            wall_temperature_k,
            freestream_temperature_k,
        )
        pressure, _shear_zero = pressure_shear_from_drag_lift(
            sc.drag, sc.lift, cos_incidence
        )
    shear = tangential_momentum_accommodation * ell * g2 / s
    return FacetCoefficients(
        pressure * gamma + shear * ell,
        pressure * ell - shear * gamma,
    )
