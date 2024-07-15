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


from hog.element_geometry import ElementGeometry
from hog.function_space import FunctionSpace
from hog.symbolizer import Symbolizer
from hog.blending import GeometryMap, IdentityMap
from hog.integrand import process_integrand, Form


def mass_boundary(
    trial: FunctionSpace,
    test: FunctionSpace,
    volume_geometry: ElementGeometry,
    boundary_geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Mass operator.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})

    ∫ uv ds
"""

    from hog.recipes.integrands.boundary.mass import integrand as integrand

    return process_integrand(
        integrand,
        trial,
        test,
        volume_geometry,
        symbolizer,
        blending=blending,
        boundary_geometry=boundary_geometry,
        is_symmetric=trial == test,
        docstring=docstring,
    )
