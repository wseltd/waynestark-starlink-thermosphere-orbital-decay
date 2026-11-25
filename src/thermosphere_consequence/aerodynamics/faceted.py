"""Faceted free-molecular force and torque model.

Assembles the per-facet, per-species coefficients of
:mod:`~thermosphere_consequence.aerodynamics.kernels` into a total force
VECTOR, an aerodynamic torque about a supplied centre of mass, and an
effective drag coefficient referenced to a stated reference area.

WHY A VECTOR MODEL AND NOT A SCALAR ONE. The existing effective-C_d model
answers "how much drag" given a projected area. It cannot answer three
questions this one can:

  * the force is not generally antiparallel to the flow. A flat panel at
    incidence produces a lift component perpendicular to the velocity, which a
    scalar model absorbs into C_d and thereby misattributes.
  * the drag depends on ATTITUDE, not only on area. The same body edge-on and
    face-on differs by more than an order of magnitude in C_d*A/m, and a
    scalar model can only represent one of those at a time.
  * an off-centre force produces a TORQUE, which is what determines whether a
    tumbling body settles into a high-drag or a low-drag attitude. A scalar
    model has no moment arm at all.

REFERENCE AREA. C_d alone is meaningless without the area it is referenced to.
The model takes one explicitly and reports it beside every coefficient. The
combination C_d * A_ref is reference-area INVARIANT — it equals the drag force
divided by dynamic pressure — which is why
:meth:`FacetedAerodynamicResult.drag_area_m2` and the inverse ballistic
coefficient derived from it are the quantities safe to compare across models.
The tests pin that invariance.

SPECIES SUPERPOSITION. In free-molecular flow the constituents do not collide
with each other, so each species contributes its own partial dynamic pressure
q_i = rho_i * V^2 / 2 at its own speed ratio s_i, and the total force is the
plain sum. This is not an approximation and it is not a mass-flux-weighted
mean of coefficients; the weighting emerges from the sum. It matters at 210 km
because atomic oxygen and molecular nitrogen sit at speed ratios of 7.7 and
10.2 respectively and are therefore in measurably different flow regimes.

WHERE THIS MODEL SITS, and it is not in the propagation loop. Nothing in
:mod:`~thermosphere_consequence.propagation` calls this class. The trajectory is
integrated under the effective scalar model — one mass-flux-weighted C_d against
the total density — and this module is evaluated separately, as the comparison
that says how far the scalar answer could be wrong (see
``scripts/run_faceted_aero_comparison.py`` and
``evidence/FACETED_AERODYNAMICS.md``). Two further things stop the sum above
from being a species-resolved force CLOSURE, and both are honest limits rather
than oversights: the accommodation provider returns the SAME sourced nominal for
every species, because no open dataset supplies per-species values (see
:mod:`.accommodation`), so the species dependence here is entirely in the speed
ratio and the partial pressure; and the surface model that separates normal
and tangential momentum accommodation, CLL, is available as an UNCERTAINTY
BRANCH: its closed form and species constants are sourced and verified (see
:data:`.kernels.CLL_COEFFICIENT_STATUS`), while its two per-surface inputs
alpha_N and sigma_T remain swept parameters supplied explicitly through
:class:`CllParameters`, and species without published constants are excluded
with their mass fraction reported rather than invented.

VALIDITY: everything in the kernels module's validity list, plus the
single-reflection assumption — a molecule leaving one facet is assumed to
escape rather than strike another. Exact for a convex body; an approximation
for a concave one, and one that UNDERSTATES drag in a cavity.

Canonical tickets: P11_12_T009 through P11_12_T014.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np

from ..domain.drag_parameters import inverse_ballistic_coefficient_m2_kg
from ..domain.species import species_mass
from .accommodation import LiteratureAccommodation, SpeciesAccommodationProvider
from .cll_constants import is_cll_supported
from .geometry import (
    EDGE_ON_COSINE_TOLERANCE,
    Facet,
    Vector3,
    incidence_angles,
    projected_area,
    rotate_to_flow_frame,
    self_shadowing_visibility,
)
from .kernels import (
    FacetCoefficients,
    cll,
    dria,
    maxwell_mixture,
    schaaf_chambre,
    sentman_diffuse,
    specular,
)
from .sentman import molecular_speed_ratio

__all__ = [
    "CllParameters",
    "FacetContribution",
    "FacetedAerodynamicResult",
    "FacetedFreeMolecularModel",
    "GasSurfaceModel",
]

#: Which gas-surface kernel the model applies to every facet.
GasSurfaceModel = Literal[
    "sentman_diffuse", "dria", "specular", "maxwell", "cll", "schaaf_chambre"
]


@dataclass(frozen=True, slots=True)
class CllParameters:
    """The two per-surface inputs of the CLL and Schaaf-Chambre kernels.

    SWEPT PARAMETERS, never asserted Starlink values: no public source gives
    a flight-conditioned alpha_N or sigma_T for this vehicle, so a result is
    always conditional on the pair carried here, and the pair is reported
    beside every output that used it.
    """

    normal_accommodation: float
    tangential_momentum_accommodation: float

    def __post_init__(self) -> None:
        for name, value in (
            ("normal_accommodation", self.normal_accommodation),
            (
                "tangential_momentum_accommodation",
                self.tangential_momentum_accommodation,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")

#: Derived from the Literal rather than restated, so the runtime check and the
#: type annotation cannot end up listing different models.
_MODELS: tuple[str, ...] = get_args(GasSurfaceModel)


@dataclass(frozen=True, slots=True)
class FacetContribution:
    """What one facet did, kept so a result can be audited facet by facet.

    ``normal_body`` is carried so the pressure/shear split is recoverable:
    the normal (pressure) component of ``force_n`` is its projection on
    -normal, the shear component is the remainder.
    """

    index: int
    area_m2: float
    cos_incidence: float
    visible: bool
    force_n: Vector3
    surface_class: str = "default"
    normal_body: Vector3 = (0.0, 0.0, 1.0)

    def pressure_force_n(self) -> Vector3:
        """The component of the facet force along the inward normal."""
        normal = np.asarray(self.normal_body, dtype=np.float64)
        force = np.asarray(self.force_n, dtype=np.float64)
        normal_component = float(force @ normal)
        pressure = normal_component * normal
        return (float(pressure[0]), float(pressure[1]), float(pressure[2]))

    def shear_force_n(self) -> Vector3:
        """The in-plane remainder of the facet force."""
        pressure = np.asarray(self.pressure_force_n(), dtype=np.float64)
        force = np.asarray(self.force_n, dtype=np.float64)
        shear = force - pressure
        return (float(shear[0]), float(shear[1]), float(shear[2]))


@dataclass(frozen=True, slots=True)
class FacetedAerodynamicResult:
    """One evaluation of the faceted model, with everything it depended on."""

    force_n: Vector3
    torque_n_m: Vector3
    drag_coefficient: float
    lift_coefficient: float
    reference_area_m2: float
    projected_area_m2: float
    visible_area_m2: float
    mass_density_kg_m3: float
    dynamic_pressure_pa: float
    speed_m_s: float
    flow_unit_body: Vector3
    centre_of_mass_body: Vector3
    gas_surface_model: str
    per_species_drag_coefficient: Mapping[str, float]
    facets: tuple[FacetContribution, ...]
    #: Species the selected kernel has no published constants for (CLL), with
    #: the mass fraction of the atmosphere they carried. Empty for kernels
    #: that cover every species. The excluded species contribute NO force;
    #: their absence is reported, never silently absorbed.
    kernel_excluded_species: tuple[str, ...] = ()
    kernel_excluded_mass_fraction: float = 0.0
    #: Drag area (drag force / dynamic pressure) summed by surface class [m^2].
    per_surface_class_drag_area_m2: Mapping[str, float] = field(default_factory=dict)

    def drag_area_m2(self) -> float:
        """C_d * A_ref [m^2] — the drag force divided by dynamic pressure.

        Reference-area invariant: changing A_ref changes C_d by the reciprocal
        factor and leaves this unchanged. It is the honest quantity to compare
        between a faceted model and a scalar one, because the two need not
        agree on what area they referenced.
        """
        return self.drag_coefficient * self.reference_area_m2

    def effective_inverse_ballistic_coefficient_m2_kg(self, mass_kg: float) -> float:
        """C_d * A / m in m^2/kg, directly comparable with the scalar model.

        This is the INVERSE ballistic coefficient. It is NOT the ballistic
        coefficient, which is its reciprocal in kg/m^2 and moves the other
        way. The arithmetic is delegated to
        ``domain.drag_parameters.inverse_ballistic_coefficient_m2_kg`` so the
        package has exactly one definition of it.
        """
        return inverse_ballistic_coefficient_m2_kg(
            self.drag_coefficient, self.reference_area_m2, mass_kg
        )


class FacetedFreeMolecularModel:
    """Free-molecular force, torque and effective C_d for a faceted body.

    ``facets`` are in the BODY frame. ``attitude_quaternion`` at evaluation
    time rotates them; leaving it out means the body frame already is the
    frame the velocity is expressed in.

    ``reference_area_m2`` is the area every reported coefficient is referenced
    to. Pass the projected area if you want a C_d comparable with the
    conventional 2.2; pass a fixed geometric area if you want the attitude
    dependence to show up in C_d rather than in the area.

    ``apply_self_shadowing`` may be turned off, and then the model states so in
    its result; it is on by default because silently double-counting a
    shadowed facet inflates drag.
    """

    def __init__(
        self,
        facets: Sequence[Facet],
        *,
        reference_area_m2: float,
        accommodation: SpeciesAccommodationProvider | None = None,
        gas_surface_model: GasSurfaceModel = "sentman_diffuse",
        specular_fraction: float = 0.0,
        cll_parameters: CllParameters | None = None,
        surface_accommodation: Mapping[str, SpeciesAccommodationProvider] | None = None,
        centre_of_mass_body: Vector3 = (0.0, 0.0, 0.0),
        apply_self_shadowing: bool = True,
    ) -> None:
        if not facets:
            raise ValueError("a faceted body needs at least one facet")
        if not math.isfinite(reference_area_m2) or reference_area_m2 <= 0.0:
            raise ValueError(
                f"reference area must be positive and finite, got {reference_area_m2}"
            )
        if gas_surface_model not in _MODELS:
            raise ValueError(
                f"unknown gas-surface model {gas_surface_model!r}; expected one of {_MODELS}"
            )
        if not 0.0 <= specular_fraction <= 1.0:
            raise ValueError(
                f"specular fraction must lie in [0, 1], got {specular_fraction}"
            )
        if gas_surface_model in ("cll", "schaaf_chambre"):
            if cll_parameters is None:
                raise ValueError(
                    f"the {gas_surface_model!r} kernel needs explicit "
                    f"CllParameters(alpha_N, sigma_T); they are swept "
                    f"sensitivity inputs and there is no defensible default"
                )
        elif cll_parameters is not None:
            raise ValueError(
                f"cll_parameters were supplied but the kernel is "
                f"{gas_surface_model!r}; refusing to carry parameters the "
                f"kernel would silently ignore"
            )
        if len(centre_of_mass_body) != 3:
            raise ValueError("centre of mass must be a 3-vector")

        self.facets = tuple(facets)
        self.reference_area_m2 = float(reference_area_m2)
        self.accommodation: SpeciesAccommodationProvider = (
            accommodation if accommodation is not None else LiteratureAccommodation()
        )
        self.gas_surface_model = gas_surface_model
        self.specular_fraction = float(specular_fraction)
        self.cll_parameters = cll_parameters
        self.surface_accommodation: dict[str, SpeciesAccommodationProvider] = dict(
            surface_accommodation or {}
        )
        self.centre_of_mass_body: Vector3 = (
            float(centre_of_mass_body[0]),
            float(centre_of_mass_body[1]),
            float(centre_of_mass_body[2]),
        )
        self.apply_self_shadowing = bool(apply_self_shadowing)

    # ------------------------------------------------------------------
    # kernel dispatch
    # ------------------------------------------------------------------

    def _provider_for(self, surface_class: str) -> SpeciesAccommodationProvider:
        """The accommodation provider a facet's surface class routes to."""
        return self.surface_accommodation.get(surface_class, self.accommodation)

    def _coefficients(
        self,
        species: str,
        speed_ratio: float,
        cos_incidence: float,
        temperature_k: float,
        surface_class: str,
    ) -> FacetCoefficients | None:
        """Kernel coefficients, or None when the kernel cannot represent the
        species (CLL without published constants) — the caller records the
        exclusion instead of this method inventing a number."""
        provider = self._provider_for(surface_class)
        wall = provider.wall_temperature_k
        if self.gas_surface_model == "specular":
            return specular(speed_ratio, cos_incidence)
        if self.gas_surface_model == "cll":
            if not is_cll_supported(species):
                return None
            parameters = self.cll_parameters
            assert parameters is not None  # enforced at construction
            return cll(
                speed_ratio,
                cos_incidence,
                parameters.normal_accommodation,
                parameters.tangential_momentum_accommodation,
                wall,
                temperature_k,
                species=species,
            )
        if self.gas_surface_model == "schaaf_chambre":
            parameters = self.cll_parameters
            assert parameters is not None  # enforced at construction
            return schaaf_chambre(
                speed_ratio,
                cos_incidence,
                parameters.normal_accommodation,
                parameters.tangential_momentum_accommodation,
                wall,
                temperature_k,
            )
        alpha = provider.energy_accommodation(species)
        if self.gas_surface_model == "maxwell":
            return maxwell_mixture(
                speed_ratio,
                cos_incidence,
                alpha,
                wall,
                temperature_k,
                specular_fraction=self.specular_fraction,
            )
        if self.gas_surface_model == "dria":
            return dria(
                speed_ratio,
                cos_incidence,
                alpha,
                wall,
                temperature_k,
                reemission_temperature=provider.reemission_temperature_k(
                    species, speed_ratio, temperature_k
                ),
            )
        return sentman_diffuse(speed_ratio, cos_incidence, alpha, wall, temperature_k)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        number_densities: Mapping[str, float],
        temperature_k: float,
        relative_velocity_m_s: Sequence[float],
        *,
        attitude_quaternion: Sequence[float] | None = None,
        centre_of_mass_body: Sequence[float] | None = None,
    ) -> FacetedAerodynamicResult:
        """Force, torque and coefficients for one atmospheric state and attitude.

        ``relative_velocity_m_s`` is the SPACECRAFT's velocity relative to the
        atmosphere, in the same frame as the facets. The free stream therefore
        travels along its negative, and that negative is the ``flow_unit_body``
        every incidence cosine is measured against. Reporting it on the result
        is deliberate: reversing this vector reverses the drag and the mistake
        is otherwise invisible.
        """
        if not number_densities:
            raise ValueError("no species supplied; cannot form a free-molecular force")
        if temperature_k <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature_k}")
        if len(relative_velocity_m_s) != 3:
            raise ValueError("relative velocity must be a 3-vector")

        velocity = np.asarray(relative_velocity_m_s, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError(f"relative speed must be positive and finite, got {speed}")
        flow = -velocity / speed
        flow_unit: Vector3 = (float(flow[0]), float(flow[1]), float(flow[2]))

        facets = (
            rotate_to_flow_frame(self.facets, attitude_quaternion)
            if attitude_quaternion is not None
            else self.facets
        )
        centre = np.asarray(
            self.centre_of_mass_body
            if centre_of_mass_body is None
            else centre_of_mass_body,
            dtype=np.float64,
        )
        if centre.shape != (3,):
            raise ValueError("centre of mass must be a 3-vector")

        cosines = incidence_angles(facets, flow_unit)
        # `visible` answers ONE question: does the MEAN stream reach this facet,
        # or does another part of the body stand in the way? It is False for
        # every rear-facing facet, because the mean stream does not reach one.
        # `>= 0.0`, matching `self_shadowing_visibility`. It was `> 0.0`, and
        # the two branches therefore disagreed about an EXACTLY grazing facet:
        # the shadowing path kept it, the no-shadowing path dropped it. For a
        # thin panel edge-on that silently discarded both broad faces - 98.5 %
        # of the grazing drag - so the grazing correction was only ever in
        # force on one of the two code paths. No published result came through
        # here, because every research script leaves self-shadowing on, but the
        # sphere validation battery runs with it off and had been measuring a
        # body whose grazing band was missing.
        mean_stream_visible = (
            self_shadowing_visibility(facets, flow_unit)
            if self.apply_self_shadowing
            else tuple(c >= 0.0 for c in cosines)
        )
        # A REAR-FACING FACET IS NOT AN UNREACHED FACET. The incident
        # distribution is a shifted Maxwellian, not a beam: at finite
        # temperature the thermal tail still delivers molecules to a facet whose
        # MEAN normal velocity points away from the flow. The Sentman closed
        # form is continuous through grazing and remains valid there - verified
        # against the quadrature oracle to a ratio of 1.0000 from 90 degrees to
        # 180 degrees for N2, O, He and H, wherever the oracle itself is
        # numerically resolvable. Excluding those facets put a cliff in a
        # continuous function, exactly as the grazing cutoff did: at 90.1
        # degrees the discarded C_D is 5.5e-2 for N2 and 3.0e-1 for H, which is
        # most of the grazing value, not a negligible tail.
        #
        # SIGN, which is not what one might guess. Just past grazing the
        # discarded term is POSITIVE drag, so excluding it biased the force low.
        # Deep in the lee it is NEGATIVE - thermal molecules striking a
        # downstream-facing surface push it forward - so excluding it there
        # biased the force high. The two do not cancel in general and the
        # direction of the old approximation was therefore not even uniform.
        #
        # WHAT IS STILL APPROXIMATE. `self_shadowing_visibility` ray-casts along
        # the mean flow only, so obstruction is resolved for mean-stream
        # directions and NOT for thermal-tail directions. A leeward facet that
        # another part of the body hides from the downstream hemisphere is still
        # counted here. For a convex body that case cannot arise and the
        # treatment is exact; for the bus-and-array proxy it can, so the leeward
        # contribution is an upper bound on an already small quantity. No
        # particle tracing is implemented and none is implied.
        reached = tuple(
            mean_stream_visible[i] if cosines[i] >= 0.0 else True
            for i in range(len(facets))
        )
        visible = mean_stream_visible

        # Per-facet flow geometry, computed once and reused for every species.
        normals = np.asarray([f.normal_body for f in facets], dtype=np.float64)
        lift_directions = np.zeros_like(normals)
        for index, cosine in enumerate(cosines):
            sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
            if sine > EDGE_ON_COSINE_TOLERANCE:
                lift_directions[index] = (normals[index] + cosine * flow) / sine

        facet_forces = np.zeros_like(normals)
        per_species_drag_area: dict[str, float] = {}
        per_class_drag_area: dict[str, float] = {}
        excluded: dict[str, float] = {}
        total_mass_density = 0.0

        for symbol, number_density in number_densities.items():
            mass = species_mass(symbol)  # unknown species raises, by design
            if number_density <= 0.0:
                # A supplied species with no density contributes a real zero,
                # which is a different statement from a species that was never
                # supplied at all; the diagnostic records it as such.
                per_species_drag_area[symbol] = 0.0
                continue
            mass_density = number_density * mass
            total_mass_density += mass_density
            partial_q = 0.5 * mass_density * speed * speed
            speed_ratio = molecular_speed_ratio(speed, temperature_k, symbol)

            species_drag_area = 0.0
            species_excluded = False
            for index, facet in enumerate(facets):
                # Geometric obstruction only. A rear-facing facet is included
                # and evaluated by the same closed form; see the note on
                # `reached` above for what that does and does not resolve.
                if not reached[index]:
                    continue
                coefficients = self._coefficients(
                    symbol,
                    speed_ratio,
                    cosines[index],
                    temperature_k,
                    facet.surface_class,
                )
                if coefficients is None:
                    species_excluded = True
                    break
                drag, lift = coefficients
                scale = partial_q * facet.area_m2
                facet_forces[index] += scale * (drag * flow - lift * lift_directions[index])
                species_drag_area += facet.area_m2 * drag
                per_class_drag_area[facet.surface_class] = (
                    per_class_drag_area.get(facet.surface_class, 0.0)
                    + facet.area_m2 * drag * mass_density
                )
            if species_excluded:
                excluded[symbol] = mass_density
                per_species_drag_area[symbol] = 0.0
            else:
                per_species_drag_area[symbol] = species_drag_area

        if total_mass_density <= 0.0:
            raise ValueError("total mass density is zero; cannot form a force")

        dynamic_pressure = 0.5 * total_mass_density * speed * speed
        force = facet_forces.sum(axis=0)
        arms = np.asarray([f.centroid_body for f in facets], dtype=np.float64) - centre
        torque = np.cross(arms, facet_forces).sum(axis=0)

        drag_force = float(force @ flow)
        lift_vector = force - drag_force * flow
        denominator = dynamic_pressure * self.reference_area_m2

        visible_area = sum(
            f.area_m2 * cosines[i]
            for i, f in enumerate(facets)
            if visible[i] and cosines[i] >= 0.0
        )

        return FacetedAerodynamicResult(
            force_n=_vector3(force),
            torque_n_m=_vector3(torque),
            drag_coefficient=drag_force / denominator,
            lift_coefficient=float(np.linalg.norm(lift_vector)) / denominator,
            reference_area_m2=self.reference_area_m2,
            projected_area_m2=projected_area(facets, flow_unit),
            visible_area_m2=float(visible_area),
            mass_density_kg_m3=total_mass_density,
            dynamic_pressure_pa=dynamic_pressure,
            speed_m_s=speed,
            flow_unit_body=flow_unit,
            centre_of_mass_body=_vector3(centre),
            gas_surface_model=self.gas_surface_model,
            per_species_drag_coefficient={
                symbol: area / self.reference_area_m2
                for symbol, area in per_species_drag_area.items()
            },
            facets=tuple(
                FacetContribution(
                    index=index,
                    area_m2=facet.area_m2,
                    cos_incidence=cosines[index],
                    visible=visible[index],
                    force_n=_vector3(facet_forces[index]),
                    surface_class=facet.surface_class,
                    normal_body=facet.normal_body,
                )
                for index, facet in enumerate(facets)
            ),
            kernel_excluded_species=tuple(sorted(excluded)),
            kernel_excluded_mass_fraction=(
                sum(excluded.values()) / total_mass_density if excluded else 0.0
            ),
            per_surface_class_drag_area_m2={
                # The mass-density weighting used during accumulation divides
                # back out so the value is the class's drag-force contribution
                # over dynamic pressure, in square metres.
                surface_class: area / total_mass_density
                for surface_class, area in per_class_drag_area.items()
            },
        )

    def effective_inverse_ballistic_coefficient_m2_kg(
        self,
        mass_kg: float,
        number_densities: Mapping[str, float],
        temperature_k: float,
        relative_velocity_m_s: Sequence[float],
        *,
        attitude_quaternion: Sequence[float] | None = None,
    ) -> float:
        """C_d * A / m [m^2/kg] for this body, attitude and atmosphere.

        The single number that makes the faceted model directly comparable
        with the scalar effective-C_d model, since orbital decay constrains
        this combination and not its factors. NOT a ballistic coefficient.
        """
        result = self.evaluate(
            number_densities,
            temperature_k,
            relative_velocity_m_s,
            attitude_quaternion=attitude_quaternion,
        )
        return result.effective_inverse_ballistic_coefficient_m2_kg(mass_kg)


def _vector3(values: np.ndarray) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))
