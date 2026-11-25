"""Faceted body geometry: facets, attitude, incidence and self-shadowing.

This module supplies the geometric half of a faceted free-molecular force
model. It knows nothing about gas-surface physics; it answers only "which
surfaces does the free stream see, at what incidence, and where are they".

SIGN CONVENTION, stated once and used everywhere in this package.

  ``flow_unit_vector`` is the direction in which the FREE-STREAM GAS TRAVELS,
  expressed in the same frame as the facet normals. For a spacecraft moving
  through a static atmosphere it is the negative of the spacecraft's
  atmosphere-relative velocity direction.

  A facet's incidence cosine is

      cos(theta_i) = - n_hat_i . u_hat

  so a facet whose outward normal points straight into the oncoming gas has
  cos(theta) = +1, and a facet on the lee side has cos(theta) < 0. This matches
  the ``theta_rad`` convention of :mod:`~thermosphere_consequence.aerodynamics.sentman`,
  where theta = 0 is a plate normal to the flow.

  Getting this backwards silently reverses the drag, which is why the
  convention is asserted in the tests rather than only written down.

EDGE-ON FACETS. ``math.cos(math.pi/2)`` is 6.1e-17, not 0. A bare ``> 0`` test
therefore does NOT reject an exactly edge-on facet, and the repository has
already been bitten by that: the facet picks up a spurious tangential residual
and then drops discontinuously to zero one ULP of angle later. Every cosine
test in this package is therefore made against
:data:`EDGE_ON_COSINE_TOLERANCE`, which is the same value and the same
reasoning as the guard inside ``sentman.sentman_plate_drag_coefficient``.

CONVEXITY. A convex body cannot shadow itself: every ray arriving from
infinity meets at most one forward-facing surface. ``self_shadowing_visibility``
therefore proves convexity from the facet planes and short-circuits, and only
falls back to trimesh ray casting for genuinely non-convex bodies. That is not
an optimisation for its own sake — the pure-Python ray engine is the slowest
thing in this package, and most spacecraft busses are convex.

Third-party components: trimesh (MIT) for mesh loading and ray casting,
scipy.spatial.transform.Rotation (BSD-3-Clause) for attitude. Both are recorded
as licence-approved in the project resource registry.

Canonical tickets: P11_12_T001, P11_12_T002, P11_12_T003.
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

__all__ = [
    "EDGE_ON_COSINE_TOLERANCE",
    "Facet",
    "Vector3",
    "box_geometry",
    "flat_panel_geometry",
    "incidence_angles",
    "is_convex",
    "load_mesh",
    "projected_area",
    "rotate_to_flow_frame",
    "self_shadowing_visibility",
    "translate",
    "unit_vector",
]

#: A 3-vector in a stated frame, in metres or dimensionless as documented.
Vector3 = tuple[float, float, float]

#: Cosines at or below this magnitude are SNAPPED to exactly 0.0, so that an
#: exactly grazing facet arrives at the kernels as a clean zero rather than as
#: 6.1e-17. It is a numerical snapping tolerance and NOT a statement that such
#: a facet contributes nothing.
#:
#: IT USED TO MEAN THAT, AND THAT WAS WRONG. A shifted Maxwellian retains finite
#: normal thermal velocities even when its mean stream is exactly parallel to a
#: plate, so the incident flux onto an unobscured grazing face is
#: Gamma_j = n_j sqrt(k_B T / (2 pi m_j)) > 0. Diffuse re-emission from a
#: stationary wall returns zero mean tangential velocity, so that flux exerts
#: drag: C_D = 1/(sqrt(pi) s_j) per exposed face, referenced to its own
#: planform. The Sentman expression already reproduces exactly that value in the
#: limit; the old guard discarded it as a "spurious tangential residual" and
#: introduced a discontinuity at 90 degrees. On the Starlink-like proxy at
#: edge-on it removed 53.76 m^2 of the body's 54.32 m^2 from the calculation.
EDGE_ON_COSINE_TOLERANCE = 1e-12

#: A facet normal must be a unit vector to this absolute tolerance. Loose
#: enough for a normal reconstructed from float32 mesh vertices, tight enough
#: that an un-normalised vector is rejected rather than silently scaling a force.
_UNIT_NORMAL_TOLERANCE = 1e-9

#: Relative tolerance for the convexity plane test, scaled by the body size.
_CONVEXITY_RELATIVE_TOLERANCE = 1e-9

#: Ray origins are lifted off the surface by this fraction of the body size so
#: a facet does not detect itself as its own occluder.
_RAY_OFFSET_FRACTION = 1e-6


@dataclass(frozen=True, slots=True)
class Facet:
    """One flat surface element of a body, in the body frame.

    ``normal_body`` is the OUTWARD unit normal — outward meaning away from the
    material. ``centroid_body`` is the area centroid, and is what the torque
    arm is measured to. ``vertices_body`` is the polygon outline; it is
    optional because an analytic single-plate case has no outline, but it is
    REQUIRED before a multi-facet body can be ray-cast for self-shadowing,
    and :func:`self_shadowing_visibility` says so rather than guessing.
    """

    area_m2: float
    normal_body: Vector3
    centroid_body: Vector3
    vertices_body: tuple[Vector3, ...] = ()
    #: Which public-data surface-class proxy this facet belongs to. "default"
    #: means the body-wide accommodation provider applies; any other value is
    #: looked up in the faceted model's per-class assignment map. A CLASS
    #: LABEL, never a material claim.
    surface_class: str = "default"

    def __post_init__(self) -> None:
        if not math.isfinite(self.area_m2) or self.area_m2 <= 0.0:
            raise ValueError(f"facet area must be positive and finite, got {self.area_m2}")
        if len(self.normal_body) != 3:
            raise ValueError(f"facet normal must be a 3-vector, got {len(self.normal_body)}")
        if len(self.centroid_body) != 3:
            raise ValueError(f"facet centroid must be a 3-vector, got {len(self.centroid_body)}")
        norm = math.sqrt(sum(c * c for c in self.normal_body))
        if abs(norm - 1.0) > _UNIT_NORMAL_TOLERANCE:
            raise ValueError(
                f"facet normal must be a unit vector, got length {norm} for {self.normal_body}"
            )
        for index, vertex in enumerate(self.vertices_body):
            if len(vertex) != 3:
                raise ValueError(
                    f"facet vertex {index} must be a 3-vector, got {len(vertex)} components"
                )
        if self.vertices_body and len(self.vertices_body) < 3:
            raise ValueError(
                f"a facet outline needs at least 3 vertices, got {len(self.vertices_body)}"
            )


def unit_vector(vector: Sequence[float]) -> Vector3:
    """Normalise a 3-vector, rejecting the zero vector rather than dividing by it."""
    if len(vector) != 3:
        raise ValueError(f"expected a 3-vector, got {len(vector)} components")
    x, y, z = (float(c) for c in vector)
    norm = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"cannot normalise the vector {(x, y, z)}: its length is {norm}")
    return (x / norm, y / norm, z / norm)


def _as_array(facets: Sequence[Facet], attribute: str) -> NDArray[np.float64]:
    return np.asarray([getattr(f, attribute) for f in facets], dtype=np.float64).reshape(
        len(facets), 3
    )


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def box_geometry(
    length_m: float,
    width_m: float,
    height_m: float,
    *,
    centre_body: Vector3 = (0.0, 0.0, 0.0),
) -> tuple[Facet, ...]:
    """Six quadrilateral facets of a rectangular box, outward normals.

    ``length_m`` runs along body x, ``width_m`` along y, ``height_m`` along z,
    and the box is centred on ``centre_body``. Faces are whole rectangles
    rather than triangle pairs so that a hand-computed projected area is a
    one-line calculation in a test instead of a sum over a triangulation.

    Exists so the tests and the comparison script do not need an external mesh
    file: a box is the shape whose projected area, self-shadowing behaviour
    and zero-torque symmetry can all be worked out on paper.
    """
    extents = (length_m, width_m, height_m)
    for name, value in zip(("length", "width", "height"), extents, strict=True):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"box {name} must be positive and finite, got {value}")

    half = tuple(0.5 * e for e in extents)
    basis = np.eye(3, dtype=np.float64)
    centre = np.asarray(centre_body, dtype=np.float64)

    facets: list[Facet] = []
    for axis in range(3):
        # (axis, first, second) is a right-handed cycle, so first x second =
        # +axis and the counter-clockwise vertex order below yields the
        # outward normal by the right-hand rule.
        first, second = (axis + 1) % 3, (axis + 2) % 3
        area = extents[first] * extents[second]
        for sign in (+1.0, -1.0):
            normal = sign * basis[axis]
            face_centre = centre + normal * half[axis]
            corners = [(-1.0, -1.0), (+1.0, -1.0), (+1.0, +1.0), (-1.0, +1.0)]
            if sign < 0.0:
                corners.reverse()
            vertices = tuple(
                _to_vector3(
                    face_centre
                    + a * half[first] * basis[first]
                    + b * half[second] * basis[second]
                )
                for a, b in corners
            )
            facets.append(
                Facet(
                    area_m2=area,
                    normal_body=_to_vector3(normal),
                    centroid_body=_to_vector3(face_centre),
                    vertices_body=vertices,
                )
            )
    return tuple(facets)


def flat_panel_geometry(
    area_m2: float,
    thickness_m: float,
    *,
    centre_body: Vector3 = (0.0, 0.0, 0.0),
) -> tuple[Facet, ...]:
    """A square flat panel of the given face AREA, with a finite thickness.

    Parameterised by area rather than by side length because area is the
    quantity the aerodynamics and the inverse ballistic coefficient are
    written in; the side is sqrt(area) and is an implementation detail.

    The thickness is finite and non-zero on purpose: a zero-thickness panel
    has no edge facets, so its edge-on drag would be exactly zero by
    construction rather than by physics, which would make the edge-on case
    untestable. The two large faces are normal to +/- body z.
    """
    if not math.isfinite(area_m2) or area_m2 <= 0.0:
        raise ValueError(f"panel area must be positive and finite, got {area_m2}")
    if not math.isfinite(thickness_m) or thickness_m <= 0.0:
        raise ValueError(f"panel thickness must be positive and finite, got {thickness_m}")
    side = math.sqrt(area_m2)
    return box_geometry(side, side, thickness_m, centre_body=centre_body)


def translate(facets: Sequence[Facet], offset_body: Sequence[float]) -> tuple[Facet, ...]:
    """Rigidly shift every facet, leaving areas and normals untouched.

    Needed to assemble a compound body (a panel plus an offset chassis) from
    the primitive builders, and to move a body off its own centre of mass so
    that a torque test has a non-zero arm to measure.
    """
    if len(offset_body) != 3:
        raise ValueError(f"offset must be a 3-vector, got {len(offset_body)} components")
    shift = np.asarray(offset_body, dtype=np.float64)
    return tuple(
        Facet(
            area_m2=f.area_m2,
            normal_body=f.normal_body,
            centroid_body=_to_vector3(np.asarray(f.centroid_body) + shift),
            vertices_body=tuple(
                _to_vector3(np.asarray(v) + shift) for v in f.vertices_body
            ),
            surface_class=f.surface_class,
        )
        for f in facets
    )


def load_mesh(path: str | pathlib.Path) -> tuple[Facet, ...]:
    """Load a triangulated surface mesh and return its facets.

    Uses trimesh (MIT), which is the only file reader in this package's
    otherwise pure-kernel aerodynamics layer; it is here because a faceted
    model without a way to ingest a real CAD surface is not a faceted model.

    OUTWARD NORMALS. trimesh's face normals follow the file's winding, which
    is not guaranteed to be consistent. When the mesh is watertight the
    winding can be repaired unambiguously and it is; when it is not, the
    file's own winding is kept and the caller is responsible for it, because
    guessing "outward" for an open surface is not well posed and a silently
    flipped normal reverses the force on that facet.
    """
    import trimesh  # imported lazily: heavy, and only this one function needs it

    resolved = pathlib.Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"mesh file not found: {resolved}")

    loaded = trimesh.load_mesh(str(resolved), process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            f"{resolved} did not load as a triangulated surface mesh "
            f"(trimesh returned {type(loaded).__name__})"
        )
    if len(loaded.faces) == 0:
        raise ValueError(f"{resolved} contains no faces")
    if loaded.is_watertight:
        loaded.fix_normals()

    areas = np.asarray(loaded.area_faces, dtype=np.float64)
    normals = np.asarray(loaded.face_normals, dtype=np.float64)
    centroids = np.asarray(loaded.triangles_center, dtype=np.float64)
    triangles = np.asarray(loaded.triangles, dtype=np.float64)

    facets: list[Facet] = []
    for index in range(len(areas)):
        if areas[index] <= 0.0:
            # A degenerate face carries no momentum and has no defined normal.
            continue
        facets.append(
            Facet(
                area_m2=float(areas[index]),
                normal_body=unit_vector(normals[index]),
                centroid_body=_to_vector3(centroids[index]),
                vertices_body=tuple(_to_vector3(v) for v in triangles[index]),
            )
        )
    if not facets:
        raise ValueError(f"{resolved} contains only degenerate (zero-area) faces")
    return tuple(facets)


# --------------------------------------------------------------------------
# attitude and incidence
# --------------------------------------------------------------------------


def rotate_to_flow_frame(
    facets: Sequence[Facet], quaternion: Sequence[float]
) -> tuple[Facet, ...]:
    """Apply an attitude rotation to every facet normal, centroid and vertex.

    ``quaternion`` is SCALAR-LAST, (x, y, z, w), which is the convention
    ``scipy.spatial.transform.Rotation.from_quat`` takes. Passing a
    scalar-first quaternion silently produces a different, valid-looking
    attitude, so the ordering is stated here and asserted in the tests.

    Areas are invariant under rotation and are carried through unchanged
    rather than recomputed, so no floating-point area drift accumulates when
    a body is rotated through a sequence of attitudes.
    """
    if len(quaternion) != 4:
        raise ValueError(f"quaternion must have 4 components (x, y, z, w), got {len(quaternion)}")
    norm = math.sqrt(sum(float(c) ** 2 for c in quaternion))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"quaternion must be non-zero and finite, got norm {norm}")
    rotation = Rotation.from_quat(np.asarray(quaternion, dtype=np.float64))

    rotated: list[Facet] = []
    for facet in facets:
        vertices = (
            tuple(
                _to_vector3(v)
                for v in rotation.apply(np.asarray(facet.vertices_body, dtype=np.float64))
            )
            if facet.vertices_body
            else ()
        )
        rotated.append(
            Facet(
                area_m2=facet.area_m2,
                normal_body=unit_vector(rotation.apply(np.asarray(facet.normal_body))),
                centroid_body=_to_vector3(rotation.apply(np.asarray(facet.centroid_body))),
                vertices_body=vertices,
                surface_class=facet.surface_class,
            )
        )
    return tuple(rotated)


def incidence_angles(
    facets: Sequence[Facet], flow_unit_vector: Sequence[float]
) -> tuple[float, ...]:
    """cos(theta_i) for every facet, where theta_i is the incidence angle.

    Returns the COSINE rather than the angle. Every free-molecular kernel
    consumes cos(theta) directly, and round-tripping a cosine through
    ``acos`` and back is what turns an exact edge-on zero into 6.1e-17.
    Cosines below :data:`EDGE_ON_COSINE_TOLERANCE` in magnitude are snapped to
    exactly 0.0 for the same reason.

    Positive means the facet faces into the oncoming gas; negative means it
    lies on the lee side. See the module docstring for the flow convention.
    """
    if not facets:
        return ()
    flow = np.asarray(unit_vector(flow_unit_vector), dtype=np.float64)
    cosines = -(_as_array(facets, "normal_body") @ flow)
    return tuple(
        0.0 if abs(c) <= EDGE_ON_COSINE_TOLERANCE else float(c) for c in cosines
    )


def projected_area(facets: Sequence[Facet], flow_unit_vector: Sequence[float]) -> float:
    """sum(A_i * max(0, cos theta_i)) — the area the free stream sees [m^2].

    This is the geometric projection and it does NOT account for
    self-shadowing: for a convex body the two are identical, and for a
    non-convex body the projection is an upper bound. Combine with
    :func:`self_shadowing_visibility` when that distinction matters.
    """
    cosines = incidence_angles(facets, flow_unit_vector)
    return float(
        sum(f.area_m2 * c for f, c in zip(facets, cosines, strict=True) if c > 0.0)
    )


# --------------------------------------------------------------------------
# visibility
# --------------------------------------------------------------------------


def is_convex(facets: Sequence[Facet]) -> bool:
    """True when every vertex lies on or behind every facet plane.

    That is the definition of a convex polyhedron and it is decided from the
    facets themselves, so no mesh object has to be carried around beside them.
    A body whose facets carry no outline cannot be tested and is reported as
    NOT convex, because "no evidence of convexity" must not read as "convex".
    """
    if not facets:
        return True
    if any(not f.vertices_body for f in facets):
        return False

    vertices = np.asarray(
        [v for f in facets for v in f.vertices_body], dtype=np.float64
    )
    scale = float(np.max(np.abs(vertices))) if vertices.size else 1.0
    tolerance = _CONVEXITY_RELATIVE_TOLERANCE * max(scale, 1.0)

    normals = _as_array(facets, "normal_body")
    centroids = _as_array(facets, "centroid_body")
    # heights[i, j] is the signed height of vertex j above the plane of facet i.
    offsets = vertices[None, :, :] - centroids[:, None, :]
    heights = np.einsum("ijk,ik->ij", offsets, normals)
    return bool(np.all(heights <= tolerance))


def self_shadowing_visibility(
    facets: Sequence[Facet], flow_unit_vector: Sequence[float]
) -> tuple[bool, ...]:
    """Which facets the free stream actually reaches, as a per-facet mask.

    A facet is visible when it faces into the flow AND no other part of the
    body lies between it and the oncoming gas.

    CONVEX SHORT-CIRCUIT. A convex body cannot occlude itself: a ray from
    infinity crosses its boundary exactly twice, once on a forward-facing
    facet and once on a rear-facing one, so every forward-facing facet is
    visible and no ray casting is needed. :func:`is_convex` proves this from
    the facet planes and the routine returns the forward-facing mask directly.
    A single facet is the degenerate case of the same argument and is handled
    without requiring an outline.

    NON-CONVEX BODIES fall back to ray casting on trimesh's ray-triangle
    primitives: one ray per forward-facing facet, launched from just off the
    facet centroid straight back up the flow, and the facet is shadowed if
    that ray meets the body again. This is a CENTROID test, so a facet that is
    only partly occluded is scored all-or-nothing; refining it needs sub-facet
    sampling and a triangulation fine enough to justify one.

    Why trimesh's ``mesh.ray`` accelerator is not called: its broad phase
    requires ``rtree``, an OPTIONAL trimesh dependency that is not part of this
    project's licence-approved dependency set and is not installed. The
    intersection mathematics used below — ``planes_lines`` and
    ``points_to_barycentric`` — is trimesh's own; only the spatial index is
    replaced, by testing every triangle against every ray. That is
    O(rays * triangles) and is the right trade for the tens-of-facets bodies
    this model is built for; it would need revisiting for a mesh of thousands.
    """
    cosines = incidence_angles(facets, flow_unit_vector)
    # >= 0.0, not > 0.0. A facet whose plane contains the flow direction is
    # REACHED by the gas - nothing occludes it - and it carries the grazing
    # thermal flux described at EDGE_ON_COSINE_TOLERANCE. Excluding it was a
    # defect, not a convention.
    forward = tuple(c >= 0.0 for c in cosines)
    if len(facets) <= 1 or is_convex(facets):
        return forward
    if not any(forward):
        return forward

    # Imported lazily: only non-convex bodies need the ray-triangle machinery.
    from trimesh import intersections as _intersections
    from trimesh import triangles as _triangles

    corners: list[NDArray[np.float64]] = []
    for facet in facets:
        if not facet.vertices_body:
            raise ValueError(
                "self-shadowing on a body that is not provably convex needs facet "
                "outlines, and a facet was supplied without vertices_body. Build "
                "the body with box_geometry, flat_panel_geometry or load_mesh so "
                "the outlines exist, or assert that it cannot shadow itself by "
                "constructing the model with apply_self_shadowing=False"
            )
        outline = np.asarray(facet.vertices_body, dtype=np.float64)
        # Fan triangulation: valid for the convex polygons these builders emit.
        for k in range(1, len(outline) - 1):
            corners.append(np.stack((outline[0], outline[k], outline[k + 1])))
    triangles = np.stack(corners)

    scale = float(np.max(np.abs(triangles)))
    offset = _RAY_OFFSET_FRACTION * max(scale, 1.0)
    flow = np.asarray(unit_vector(flow_unit_vector), dtype=np.float64)
    upstream = -flow

    lit = [i for i, is_forward in enumerate(forward) if is_forward]
    origins = (
        np.asarray([facets[i].centroid_body for i in lit], dtype=np.float64)
        + upstream * offset
    )

    # Every triangle is a candidate for every ray: see the note above.
    ray_index = np.repeat(np.arange(len(origins)), len(triangles))
    triangle_index = np.tile(np.arange(len(triangles)), len(origins))
    candidates = triangles[triangle_index]
    normals, usable = _triangles.normals(candidates)
    if not bool(np.all(usable)):
        raise ValueError(
            "the body contains a degenerate triangle with no defined normal; "
            "self-shadowing cannot be decided"
        )

    location, valid, distance = _intersections.planes_lines(
        plane_origins=candidates[:, 0, :],
        plane_normals=normals,
        line_origins=origins[ray_index],
        line_directions=np.broadcast_to(upstream, (len(ray_index), 3)),
        return_distance=True,
    )
    visible = list(forward)
    if not bool(np.any(valid)):
        return tuple(visible)

    # A hit only counts when it lies AHEAD of the ray origin (positive
    # distance) and inside the triangle (all barycentric coordinates >= 0).
    barycentric = _triangles.points_to_barycentric(candidates[valid], location)
    inside = np.all(barycentric >= -1e-9, axis=1)
    ahead = distance > offset
    blocked_rays = np.unique(ray_index[valid][inside & ahead])
    for ray in blocked_rays:
        visible[lit[int(ray)]] = False
    return tuple(visible)


def _to_vector3(values: NDArray[np.float64] | Sequence[float]) -> Vector3:
    x, y, z = (float(c) for c in values)
    return (x, y, z)
