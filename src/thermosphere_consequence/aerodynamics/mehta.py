"""Mehta et al. (2014) drag-coefficient response surfaces. INTERFACE ONLY.

WHAT THE MODEL IS. Mehta, Walker, Lawrence, Linares, Higdon and Koller (2014),
"Modeling satellite drag coefficients with response surfaces", Adv. Space Res.
54:1590-1607 (doi:10.1016/j.asr.2014.06.033), replace a closed-form drag
coefficient with a SURROGATE for a Test Particle Monte Carlo simulation of a
specific vehicle. A Latin hypercube of 1000 ensemble members is run through
TPMC, a Gaussian process is fitted to the results, and MCMC characterises the
posterior. The product is a set of fitted coefficients that reproduce the TPMC
drag coefficient of THAT GEOMETRY, over that parameter box, at a fraction of
the cost. It is a different kind of object from ``kernels.sentman_diffuse`` or
``kernels.dria``: those are analytic free-molecular results for a flat facet,
this is a regression onto simulations of a whole satellite.

WHY THIS RAISES, which is NOT the reason the repository previously recorded.
The blocker was carried as "no open coefficient dataset exists". That is wrong
and is corrected here. The coefficients were located, in the open, and the bar
turns out to be three other things:

  1. LICENCE. The fitted coefficients ship in the LANL Response Surface
     Modeling Tool Suite, https://github.com/AndrewCWalker/rsm_tool_suite
     (RSM_Cd/data/GRACE_RSM.dat and RSM_Cd/data/CHAMP_RSM.dat), authored by two
     of the paper's co-authors and released as DOE software under OSTI ID
     1310545, https://www.osti.gov/biblio/1310545 . Its LICENSE file is a LANL
     US-Government-rights notice plus "the GNU General Public License ...
     either version 2 of the License, or (at your option) any later version" --
     GPL-2.0-or-later, which appears by name in ``excluded_licences`` in
     config/licence_policy.yaml. This is the same position this project already
     takes on ADBSat (GPL-3.0) and Nyx (AGPL-3.0): the licence PERMITS
     commercial use, and this project declines the copyleft obligation. It is a
     reversible business decision, not a legal bar, and calling it either one
     by the other's name would be a false statement.

  2. GEOMETRY. The released coefficients are for GRACE and CHAMP (and, added
     later, CYGNSS). They are surrogates for those hulls at those attitudes.
     This project's vehicle is a Starlink v1.5, and no coefficient file for it
     exists anywhere, because none has been fitted. The sphere case the paper
     validates against has no released coefficient file either. Using GRACE's
     response surface for a Starlink would not be an approximation, it would be
     a category error -- and that objection survives any licence decision.

  3. VALIDATION. The paper is paywalled at Elsevier, has no arXiv preprint and
     no indexed open-access copy, so the per-condition drag coefficients an
     implementation would be checked against are not openly readable. Only
     aggregate error statistics circulate in secondary summaries. An
     implementation this repository could not check against its own source's
     numbers is not one this repository should ship.

WHAT WOULD BE NEEDED, named precisely, because a blocker that does not say what
is missing is not actionable. Each released ``*_RSM.dat`` carries, per gas
species (H, He, N, N2, O, O2) and in the file's own labels:

    beta[6][7]      Gaussian-process correlation parameters, one row per
                    species, one column per model input
    xmin[6][7]      input box lower corner, used to normalise the inputs
    xrange[6][7]    input box width
    mean[6], sd[6]  output standardisation
    lamz[6]         marginal precision of the Gaussian process
    lamws[6]        nugget / noise precision
    x[6][1000][7]   the 1000-point Latin hypercube design, per species
    wt[1000][6]     the fitted weights

and the model inputs are, per the suite's own RSM_Cd/README: relative velocity
between vehicle and gas (5500-7500 m/s), vehicle surface temperature
(100-500 K), atmospheric translational temperature (200-2000 K), the H/He/N/
N2/O/O2 mole fractions, total number density (which enters only through the
adsorbed atomic-oxygen fraction, under a Langmuir or Freundlich isotherm),
pitch and yaw, and the surface material's particle mass.

TERMINOLOGY, because the suite and this repository disagree and the disagreement
is the dangerous kind. RSM_Cd prints "Ballistic Coefficient, BC = 3.689233e-03
m^2/kg" for C_d = 3.464208, A = 0.1064957 m^2, m = 100 kg. That is C_d*A/m,
which this repository calls the INVERSE ballistic coefficient
(``inverse_ballistic_coefficient_m2_kg``) and never calls a ballistic
coefficient. Anyone wiring these files in must convert the name, not only the
number.

WHAT THIS PROJECT DOES INSTEAD. ``kernels.sentman_diffuse`` and ``kernels.dria``
with a sourced energy accommodation coefficient, composed over a faceted
geometry in ``aerodynamics.faceted``. That is analytic, its assumptions are
stated, and it needs no third-party coefficient file.

Canonical lineage: the Mehta et al. (2014) formulations already cited by
``aerodynamics.kernels`` and ``aerodynamics.sentman``; blocker registry entry
for the response-surface coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

__all__ = [
    "MEHTA_RELEASED_GEOMETRIES",
    "MEHTA_RESPONSE_SURFACE_BLOCKER",
    "MEHTA_RSM_COEFFICIENT_ARRAYS",
    "MehtaResponseSurface",
]

#: The coefficient arrays a response-surface evaluation needs, in the labels the
#: released files use. Named so the blocker message cannot say "the
#: coefficients" and leave a reader to guess which.
MEHTA_RSM_COEFFICIENT_ARRAYS: Final[tuple[str, ...]] = (
    "beta[6][7]",
    "xmin[6][7]",
    "xrange[6][7]",
    "mean[6]",
    "sd[6]",
    "lamz[6]",
    "lamws[6]",
    "x[6][1000][7]",
    "wt[1000][6]",
)

#: The vehicles for which fitted coefficients were actually released. Not this
#: project's vehicle, which is the objection that outlives the licence one.
MEHTA_RELEASED_GEOMETRIES: Final[tuple[str, ...]] = ("GRACE", "CHAMP", "CYGNSS")

#: Stated once so the raised message and the test that pins it cannot drift.
MEHTA_RESPONSE_SURFACE_BLOCKER = (
    "MehtaResponseSurface is not implemented. The blocker is NOT the one previously "
    "recorded ('no open coefficient dataset exists') -- the coefficients were located "
    "in the open and the recorded reason is corrected here. "
    "WHERE THEY WERE SOUGHT AND FOUND: the LANL Response Surface Modeling Tool Suite, "
    "https://github.com/AndrewCWalker/rsm_tool_suite , files RSM_Cd/data/GRACE_RSM.dat "
    "and RSM_Cd/data/CHAMP_RSM.dat, released as DOE software OSTI ID 1310545 "
    "(https://www.osti.gov/biblio/1310545) by two of the paper's co-authors. "
    "WHERE THEY WERE SOUGHT AND NOT FOUND: the Elsevier article "
    "doi:10.1016/j.asr.2014.06.033 is paywalled with no arXiv preprint and no indexed "
    "open-access copy, so the paper's own per-condition reference drag coefficients are "
    "not openly readable; no Zenodo, Figshare, Dryad or data.gov record exists; and the "
    "successor suite https://github.com/ASSISTLabOrg/WVU_RSM_Suite ships the pipeline "
    "without pre-fitted coefficient files. "
    "WHY IT STILL RAISES, three reasons, each sufficient on its own: "
    "(1) LICENCE -- the suite is GPL-2.0-or-later, which config/licence_policy.yaml "
    "lists by name in excluded_licences. The licence PERMITS commercial use; this "
    "project declines the copyleft obligation, exactly as it does for ADBSat and Nyx. "
    "(2) GEOMETRY -- the released coefficients fit GRACE, CHAMP and CYGNSS. They are "
    "surrogates for those hulls, not for this project's Starlink v1.5, and no fitted "
    "file for it exists because none has been produced. The sphere case the paper "
    "validates against has no released coefficient file either. "
    "(3) VALIDATION -- with the paper paywalled there are no openly readable "
    "per-condition values to check an implementation against. "
    "WHAT WOULD BE NEEDED: per-species (H, He, N, N2, O, O2) arrays "
    + ", ".join(MEHTA_RSM_COEFFICIENT_ARRAYS)
    + ", fitted for THIS project's geometry, under terms compatible with a "
    "strict-permissive policy, together with reference values to validate against. "
    "TO UNBLOCK: either take a recorded decision to accept GPL-2.0-or-later and run "
    "the suite as a separate executable in research mode (never linked into this "
    "library), or run the Automated_RSM TPMC pipeline on the Starlink geometry to "
    "produce coefficients this project owns. Until then use kernels.dria or "
    "kernels.sentman_diffuse over aerodynamics.faceted, which are analytic and need no "
    "third-party coefficient file."
)


class MehtaResponseSurface:
    """A Mehta et al. (2014) response surface. INTERFACE ONLY -- always raises.

    The signatures are the ones a real implementation would need, so that the
    day defensible coefficients exist for this project's geometry the call
    sites do not move. Every entry point raises :class:`NotImplementedError`
    carrying :data:`MEHTA_RESPONSE_SURFACE_BLOCKER`, which names the coefficient
    arrays, where they were sought, where they were found, and the three
    independent reasons they are not used.

    Constructing this is refused as well as calling it: a surrogate that could
    be built and only failed later would invite a caller to hold one and assume
    the physics behind it had been settled.
    """

    #: Fitted for a named hull, over a named parameter box. Not a general model.
    released_geometries = MEHTA_RELEASED_GEOMETRIES

    def __init__(
        self,
        *,
        geometry: str,
        coefficient_path: str,
        adsorption_model: str = "Freundlich",
    ) -> None:
        """Load a fitted response surface for one geometry.

        Raises:
            NotImplementedError: always, by design.
        """
        del geometry, coefficient_path, adsorption_model
        raise NotImplementedError(MEHTA_RESPONSE_SURFACE_BLOCKER)

    def drag_coefficient(
        self,
        *,
        relative_speed_m_s: float,
        surface_temperature_k: float,
        translational_temperature_k: float,
        species_mole_fractions: Mapping[str, float],
        total_number_density_m3: float,
        pitch_angle_deg: float = 0.0,
        yaw_angle_deg: float = 0.0,
    ) -> float:
        """Drag coefficient referenced to the projected area, dimensionless.

        Raises:
            NotImplementedError: always, by design.
        """
        del relative_speed_m_s, surface_temperature_k, translational_temperature_k
        del species_mole_fractions, total_number_density_m3
        del pitch_angle_deg, yaw_angle_deg
        raise NotImplementedError(MEHTA_RESPONSE_SURFACE_BLOCKER)

    def inverse_ballistic_coefficient_m2_kg(
        self,
        *,
        relative_speed_m_s: float,
        surface_temperature_k: float,
        translational_temperature_k: float,
        species_mole_fractions: Mapping[str, float],
        total_number_density_m3: float,
        mass_kg: float,
        pitch_angle_deg: float = 0.0,
        yaw_angle_deg: float = 0.0,
    ) -> float:
        """C_d * A / m in m^2/kg, using the suite's projected-area look-up.

        Named for what it is. The reference implementation prints this quantity
        under the label "Ballistic Coefficient", which is its reciprocal's name;
        the conversion of the NAME is part of wiring these files in.

        Raises:
            NotImplementedError: always, by design.
        """
        del relative_speed_m_s, surface_temperature_k, translational_temperature_k
        del species_mole_fractions, total_number_density_m3, mass_kg
        del pitch_angle_deg, yaw_angle_deg
        raise NotImplementedError(MEHTA_RESPONSE_SURFACE_BLOCKER)
