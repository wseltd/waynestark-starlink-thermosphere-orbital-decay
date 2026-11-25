"""Public-data proxy geometries: sphere, bus, bus-plus-array, Starlink-LIKE.

NO PUBLIC EXACT STARLINK CAD MODEL EXISTS, and this module does not pretend
otherwise. Every builder returns a :class:`ProxyGeometry` whose metadata
states its dimensions, reference area, centre of mass, facet count, surface
classes and — most importantly — its ``assumption_status``: which numbers are
assumptions, which are derived, and what none of them are (SpaceX data).

SURFACE CLASSES. Facets carry the public-data proxy classes of the research
brief. At 180-300 km every conventional exposed class shares the
AO-conditioned orbital accommodation prior; the classes exist so wall
temperature and sensitivity WIDTH can differ per class, and so the class-level
force breakdown is reportable. ``unknown_proprietary_coating`` never receives
an invented coefficient — it routes to the widest sensitivity treatment.

THE SPHERE. A deterministic latitude-longitude tessellation used for mesh
convergence against the analytical sphere solution. Facet areas are the EXACT
spherical patch areas (they sum to 4*pi*r^2 to rounding), so refinement
probes only the angular resolution of the incidence distribution, never a
surface-area deficit. A sphere is convex, so the shadowing short-circuit
applies and the forward-facing cosine test is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Facet, Vector3, box_geometry

__all__ = [
    "PROXY_SURFACE_CLASSES",
    "ProxyGeometry",
    "bus_with_array_proxy",
    "cuboid_bus_proxy",
    "sphere_geometry",
    "starlink_like_proxy",
    "with_surface_class",
]

#: The public-data proxy classes of the research brief, in one place so a
#: typo cannot mint a sixth class silently.
PROXY_SURFACE_CLASSES: tuple[str, ...] = (
    "solar_array_glass_or_oxide",
    "bus_ao_conditioned_oxide_or_coating",
    "metallic_structure_ao_conditioned_al_oxide",
    "polymer_or_mli_ao_aged_proxy",
    "unknown_proprietary_coating",
)


@dataclass(frozen=True, slots=True)
class ProxyGeometry:
    """A proxy body plus everything a report must state about it."""

    name: str
    facets: tuple[Facet, ...]
    reference_area_m2: float
    centre_of_mass_body: Vector3
    dimensions_m: dict[str, float] = field(default_factory=dict)
    assumption_status: str = ""

    def __post_init__(self) -> None:
        if not self.facets:
            raise ValueError("a proxy geometry needs at least one facet")
        if self.reference_area_m2 <= 0.0:
            raise ValueError(
                f"reference area must be positive, got {self.reference_area_m2}"
            )
        if not self.assumption_status.strip():
            raise ValueError(
                "a proxy geometry must state its assumption status; a body "
                "whose provenance is unstated reads as a measured geometry"
            )

    @property
    def surface_classes(self) -> tuple[str, ...]:
        """The distinct surface classes present, sorted."""
        return tuple(sorted({facet.surface_class for facet in self.facets}))

    @property
    def total_area_m2(self) -> float:
        return sum(facet.area_m2 for facet in self.facets)


def with_surface_class(facets: tuple[Facet, ...], surface_class: str) -> tuple[Facet, ...]:
    """The same facets, stamped with one surface class."""
    if surface_class != "default" and surface_class not in PROXY_SURFACE_CLASSES:
        raise ValueError(
            f"unknown surface class {surface_class!r}; expected 'default' or "
            f"one of {PROXY_SURFACE_CLASSES}"
        )
    return tuple(
        Facet(
            area_m2=facet.area_m2,
            normal_body=facet.normal_body,
            centroid_body=facet.centroid_body,
            vertices_body=facet.vertices_body,
            surface_class=surface_class,
        )
        for facet in facets
    )


def sphere_geometry(
    radius_m: float, *, polar_bands: int, azimuthal_steps: int
) -> ProxyGeometry:
    """Deterministic latitude-longitude sphere tessellation.

    Each band between polar angles carries ``azimuthal_steps`` facets whose
    areas are the exact spherical patch areas r^2 * dlambda * (cos t1 - cos t2)
    and whose normals are radial at the patch's angular midpoint. Total area
    is exactly 4*pi*r^2 up to rounding, by construction.
    """
    if radius_m <= 0.0:
        raise ValueError(f"radius must be positive, got {radius_m}")
    if polar_bands < 2 or azimuthal_steps < 3:
        raise ValueError(
            f"need at least 2 polar bands and 3 azimuthal steps, got "
            f"{polar_bands} and {azimuthal_steps}"
        )
    facets: list[Facet] = []
    d_lambda = 2.0 * math.pi / azimuthal_steps
    for band in range(polar_bands):
        theta_1 = math.pi * band / polar_bands
        theta_2 = math.pi * (band + 1) / polar_bands
        theta_mid = 0.5 * (theta_1 + theta_2)
        band_area = radius_m * radius_m * d_lambda * (
            math.cos(theta_1) - math.cos(theta_2)
        )
        for step in range(azimuthal_steps):
            lambda_mid = d_lambda * (step + 0.5)
            normal = (
                math.sin(theta_mid) * math.cos(lambda_mid),
                math.sin(theta_mid) * math.sin(lambda_mid),
                math.cos(theta_mid),
            )
            centroid = tuple(radius_m * component for component in normal)
            facets.append(
                Facet(
                    area_m2=band_area,
                    normal_body=normal,
                    centroid_body=(centroid[0], centroid[1], centroid[2]),
                )
            )
    return ProxyGeometry(
        name=f"sphere r={radius_m} m ({polar_bands}x{azimuthal_steps})",
        facets=tuple(facets),
        reference_area_m2=math.pi * radius_m * radius_m,
        centre_of_mass_body=(0.0, 0.0, 0.0),
        dimensions_m={"radius": radius_m},
        assumption_status=(
            "analytical test body, not a spacecraft; reference area is the "
            "projected (cross-sectional) area pi*r^2, matching the published "
            "sphere benchmark convention"
        ),
    )


def cuboid_bus_proxy(
    length_m: float, width_m: float, height_m: float
) -> ProxyGeometry:
    """A rectangular bus, every face on the AO-conditioned bus class."""
    facets = with_surface_class(
        box_geometry(length_m, width_m, height_m),
        "bus_ao_conditioned_oxide_or_coating",
    )
    return ProxyGeometry(
        name=f"cuboid bus {length_m}x{width_m}x{height_m} m",
        facets=facets,
        reference_area_m2=length_m * width_m,
        centre_of_mass_body=(0.0, 0.0, 0.0),
        dimensions_m={"length": length_m, "width": width_m, "height": height_m},
        assumption_status=(
            "generic analytical bus; dimensions are the caller's assumption, "
            "reference area is the length x width face"
        ),
    )


def _panel_pair(
    span_m: float,
    chord_m: float,
    centre: Vector3,
    front_class: str,
    back_class: str,
) -> tuple[Facet, ...]:
    """A zero-thickness planar panel as an explicit front/back facet pair.

    Normals along +z (front) and -z (back), so exactly one side sees any
    given flow. Vertices are supplied for the shadowing ray-caster.
    """
    area = span_m * chord_m
    half_span, half_chord = 0.5 * span_m, 0.5 * chord_m
    x, y, z = centre
    corners = (
        (x - half_chord, y - half_span, z),
        (x + half_chord, y - half_span, z),
        (x + half_chord, y + half_span, z),
        (x - half_chord, y + half_span, z),
    )
    front = Facet(
        area_m2=area,
        normal_body=(0.0, 0.0, 1.0),
        centroid_body=centre,
        vertices_body=corners,
        surface_class=front_class,
    )
    back = Facet(
        area_m2=area,
        normal_body=(0.0, 0.0, -1.0),
        centroid_body=centre,
        vertices_body=tuple(reversed(corners)),
        surface_class=back_class,
    )
    return (front, back)


def bus_with_array_proxy(
    *,
    bus_length_m: float,
    bus_width_m: float,
    bus_height_m: float,
    array_span_m: float,
    array_chord_m: float,
    array_offset_m: float,
) -> ProxyGeometry:
    """A cuboid bus plus one planar solar-array facet pair above it."""
    bus = with_surface_class(
        box_geometry(bus_length_m, bus_width_m, bus_height_m),
        "bus_ao_conditioned_oxide_or_coating",
    )
    panel = _panel_pair(
        array_span_m,
        array_chord_m,
        (0.0, 0.0, array_offset_m),
        "solar_array_glass_or_oxide",
        "polymer_or_mli_ao_aged_proxy",
    )
    return ProxyGeometry(
        name="cuboid bus + planar array",
        facets=bus + panel,
        reference_area_m2=bus_length_m * bus_width_m + array_span_m * array_chord_m,
        centre_of_mass_body=(0.0, 0.0, 0.0),
        dimensions_m={
            "bus_length": bus_length_m,
            "bus_width": bus_width_m,
            "bus_height": bus_height_m,
            "array_span": array_span_m,
            "array_chord": array_chord_m,
            "array_offset": array_offset_m,
        },
        assumption_status=(
            "generic bus-plus-array proxy; every dimension is the caller's "
            "assumption; array modelled as a zero-thickness facet pair with "
            "a glass/oxide front and a polymer-proxy back"
        ),
    )


def starlink_like_proxy() -> ProxyGeometry:
    """The Starlink-LIKE public-data proxy geometry. Not the Starlink geometry.

    Round-number assumed dimensions for a flat-panel bus with a single large
    solar array, chosen to be of the publicly reported CLASS of the vehicle
    (flat bus, one big array) without claiming any proprietary value:

      bus 2.8 x 1.4 x 0.2 m; array 8.0 x 2.8 m, offset 1.0 m above the bus
      plane; array front is PV glass, array back a polymer proxy, bus faces
      the AO-conditioned coating class, bus edges the metallic class.

    Reference area is the bus face + array area (the maximum-drag broadside
    projection). Mass is deliberately NOT part of the geometry: the caller
    supplies it explicitly wherever a per-mass quantity is needed.

    NOT CALIBRATED TO ANY PUBLISHED COEFFICIENT, and that is checkable rather
    than merely asserted. This project carries an external reference value of
    0.085 m^2/kg (Shambaugh 2024, a CAPELLA satellite, not these spacecraft).
    Reproducing it with the geometry above at the nominal 30 degree attitude
    would require a mass of 327.5 kg; the mass actually used is 297 kg, from a
    third-party estimate whose own stated range is 275-303 kg and therefore
    excludes 327.5. Matching it by attitude instead would require about 27
    degrees rather than 30. Every dimension above is a round number. None of
    the three was tuned toward that coefficient, and any agreement with it is
    a coincidence rather than a check.
    """
    bus_length, bus_width, bus_height = 2.8, 1.4, 0.2
    array_span, array_chord, array_offset = 8.0, 2.8, 1.0
    bus_faces = box_geometry(bus_length, bus_width, bus_height)
    # The four thin edge faces (normals in the x-y plane) are metallic
    # structure; the two broad faces keep the coating class.
    reclassed = tuple(
        Facet(
            area_m2=facet.area_m2,
            normal_body=facet.normal_body,
            centroid_body=facet.centroid_body,
            vertices_body=facet.vertices_body,
            surface_class=(
                "bus_ao_conditioned_oxide_or_coating"
                if abs(facet.normal_body[2]) > 0.5
                else "metallic_structure_ao_conditioned_al_oxide"
            ),
        )
        for facet in bus_faces
    )
    panel = _panel_pair(
        array_span,
        array_chord,
        (0.0, 0.0, array_offset),
        "solar_array_glass_or_oxide",
        "polymer_or_mli_ao_aged_proxy",
    )
    facets = reclassed + panel
    return ProxyGeometry(
        name="Starlink-like public-data proxy geometry",
        facets=facets,
        reference_area_m2=bus_length * bus_width + array_span * array_chord,
        centre_of_mass_body=(0.0, 0.0, 0.0),
        dimensions_m={
            "bus_length": bus_length,
            "bus_width": bus_width,
            "bus_height": bus_height,
            "array_span": array_span,
            "array_chord": array_chord,
            "array_offset": array_offset,
        },
        assumption_status=(
            "Starlink-LIKE public-data proxy: every dimension is a "
            "round-number assumption representing the publicly reported "
            "vehicle class (flat-panel bus, single large array), NOT SpaceX "
            "data, NOT the proprietary geometry; per-mass quantities compare "
            "against the external Capella SAR reference (0.085 m^2/kg, "
            "Shambaugh 2024), not proprietary hardware"
        ),
    )
