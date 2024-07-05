# HyTeG Operator Generator
# Copyright (C) 2024  HyTeG Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import List
import sympy as sp

from hog.exception import HOGException

from hog.element_geometry import ElementGeometry, TriangleElement
from hog.symbolizer import Symbolizer
from hog.blending import GeometryMap, IdentityMap
from hog.fem_helpers import jac_affine_to_physical
from hog.external_functions import BlendingFEmbeddedTriangle


def embedded_normal(
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """Returns an unoriented unit normal vector for an embedded triangle."""

    if not (isinstance(geometry, TriangleElement) and geometry.space_dimension == 3):
        raise HOGException(
            "Embedded normal vectors are only defined for triangles embedded in 3D space."
        )

    vert_points = symbolizer.affine_vertices_as_vectors(
        geometry.space_dimension, geometry.num_vertices
    )
    span0 = vert_points[1] - vert_points[0]
    span1 = vert_points[2] - vert_points[0]

    if not isinstance(blending, IdentityMap):
        blending_jac = jac_affine_to_physical(geometry, symbolizer)
        span0 = blending_jac.T * span0
        span1 = blending_jac.T * span1

    normal = sp.Matrix(
        [
            span0[1] * span1[2] - span0[2] * span1[1],
            span0[2] * span1[0] - span0[0] * span1[2],
            span0[0] * span1[1] - span0[1] * span1[0],
        ]
    )
    normal = normal / (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
    return normal


def face_projection(
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """Returns a projection matrix for an embedded triangle."""

    if not (isinstance(geometry, TriangleElement) and geometry.space_dimension == 3):
        raise HOGException(
            "Projection matrices are only defined for triangles embedded in 3D space."
        )

    normal = embedded_normal(geometry, symbolizer, blending=blending)
    projection = sp.Matrix(
        [
            [1.0 - normal[0] ** 2, -normal[0] * normal[1], -normal[0] * normal[2]],
            [-normal[0] * normal[1], 1.0 - normal[1] ** 2, -normal[1] * normal[2]],
            [-normal[0] * normal[2], -normal[1] * normal[2], 1.0 - normal[2] ** 2],
        ]
    )
    return projection
