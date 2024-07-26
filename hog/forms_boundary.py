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


from typing import Optional
from hog.element_geometry import ElementGeometry
from hog.function_space import FunctionSpace, TrialSpace, TestSpace
from hog.symbolizer import Symbolizer
from hog.blending import GeometryMap, IdentityMap
from hog.integrand import process_integrand, Form


def mass_boundary(
    trial: TrialSpace,
    test: TestSpace,
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
        is_symmetric=trial == test,  # type: ignore[comparison-overlap]
        docstring=docstring,
    )


def freeslip_momentum_weak_boundary(
    trial: TrialSpace,
    test: TestSpace,
    volume_geometry: ElementGeometry,
    boundary_geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    function_space_mu: Optional[FunctionSpace] = None,
) -> Form:
    docstring = f"""
Weak (Nitsche) free-slip boundary operator for an analytical outward normal.

The normal is specified via an affine mapping of the coordinates:

    n(x) = A x + b

where x = (x_1, x_2, x_3) are the physical coordinates of the position on the boundary.

n is normalized automatically.

This enables for instance to specify some simple cases like

    normals away from the origin:
        A = I, b = 0

    normals towards the origin:
        A = -I, b = 0

    normals in one coordinate direction (e.g. in x-direction)
        A = 0, b = (1, 0, 0)ᵀ

Weak formulation

    From 
        Davies et al.
        Towards automatic finite-element methods for geodynamics via Firedrake
        in Geosci. Model Dev. (2022)
        DOI: 10.5194/gmd-15-5127-2022


    u: trial function (space: {trial})
    v: test function  (space: {test})
    n: outward normal
    
    − ∫_Γ 𝑣 ⋅ 𝑛 𝑛 ⋅ (𝜇 [∇𝑢 + (∇𝑢)ᵀ]) ⋅ 𝑛 𝑑𝑠
    − ∫_Γ 𝑛 ⋅ (𝜇 [∇𝑣 + (∇𝑣)ᵀ]) ⋅ 𝑛 𝑢 ⋅ 𝑛 𝑑𝑠
    + ∫_Γ C_n 𝜇 𝑣 ⋅ 𝑛 𝑢 ⋅ 𝑛 𝑑𝑠


Geometry map: {blending}
    
"""

    from hog.recipes.integrands.boundary.freeslip_nitsche_momentum import (
        integrand as integrand,
    )

    return process_integrand(
        integrand,
        trial,
        test,
        volume_geometry,
        symbolizer,
        blending=blending,
        boundary_geometry=boundary_geometry,
        is_symmetric=trial == test,  # type: ignore[comparison-overlap]
        fe_coefficients={"mu": function_space_mu},
        docstring=docstring,
    )


def freeslip_divergence_weak_boundary(
    trial: TrialSpace,
    test: TestSpace,
    volume_geometry: ElementGeometry,
    boundary_geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Weak (Nitsche) free-slip boundary operator for an analytical outward normal.

From
    Davies et al.
    Towards automatic finite-element methods for geodynamics via Firedrake
    in Geosci. Model Dev. (2022)
    DOI: 10.5194/gmd-15-5127-2022

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    n: outward normal

    − ∫_Γ 𝑣 𝑛 ⋅ 𝑢 𝑑𝑠

Geometry map: {blending}

"""

    from hog.recipes.integrands.boundary.freeslip_nitsche_divergence import (
        integrand as integrand,
    )

    return process_integrand(
        integrand,
        trial,
        test,
        volume_geometry,
        symbolizer,
        blending=blending,
        boundary_geometry=boundary_geometry,
        docstring=docstring,
    )


def freeslip_gradient_weak_boundary(
    trial: TrialSpace,
    test: TestSpace,
    volume_geometry: ElementGeometry,
    boundary_geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Weak (Nitsche) free-slip boundary operator for an analytical outward normal.

From
    Davies et al.
    Towards automatic finite-element methods for geodynamics via Firedrake
    in Geosci. Model Dev. (2022)
    DOI: 10.5194/gmd-15-5127-2022

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    n: outward normal

    − ∫_Γ 𝑛 ⋅ 𝑣 𝑢 𝑑𝑠

Geometry map: {blending}

"""

    from hog.recipes.integrands.boundary.freeslip_nitsche_gradient import (
        integrand as integrand,
    )

    return process_integrand(
        integrand,
        trial,
        test,
        volume_geometry,
        symbolizer,
        blending=blending,
        boundary_geometry=boundary_geometry,
        docstring=docstring,
    )
