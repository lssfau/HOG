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

from sympy.core.cache import clear_cache

from hog.operator_generation.loop_strategies import CUBES
from hog.operator_generation.optimizer import Opts
from hog.element_geometry import (
    ElementGeometry,
    LineElement,
    TriangleElement,
    TetrahedronElement,
)
from hog.function_space import LagrangianFunctionSpace, TrialSpace, TestSpace
from hog.operator_generation.operators import HyTeGElementwiseOperator
from hog.symbolizer import Symbolizer
from hog.quadrature import Quadrature, select_quadrule
from hog.forms import div_k_grad
from hog.forms_boundary import mass_boundary
from hog.operator_generation.kernel_types import ApplyWrapper, AssembleWrapper
from hog.operator_generation.types import hyteg_type
from hog.blending import AnnulusMap, GeometryMap, IcosahedralShellMap


def test_opgen_smoke():
    """
    Just a simple smoke test to check that an operator can be generated.

    If something is really broken, this will make the CI fail early.

    We are generating a matvec method here for

        ∫ k ∇u · ∇v dx + ∫ uv ds

    with either integral being evaluated in their own kernel.

    That may not be reasonable but tests some features.
    """

    clear_cache()

    symbolizer = Symbolizer()

    name = f"P2DivKGradBlendingPlusBoundaryMass"

    dims = [2]

    trial = TrialSpace(LagrangianFunctionSpace(2, symbolizer))
    test = TestSpace(LagrangianFunctionSpace(2, symbolizer))
    coeff = LagrangianFunctionSpace(2, symbolizer)

    type_descriptor = hyteg_type()

    kernel_types = [
        ApplyWrapper(
            trial,
            test,
            type_descriptor=type_descriptor,
            dims=dims,
        ),
        AssembleWrapper(
            trial,
            test,
            type_descriptor=type_descriptor,
            dims=dims,
        ),
    ]

    operator = HyTeGElementwiseOperator(
        name,
        symbolizer=symbolizer,
        kernel_wrapper_types=kernel_types,
        type_descriptor=type_descriptor,
    )

    opts_volume = {Opts.MOVECONSTANTS, Opts.VECTORIZE, Opts.TABULATE, Opts.QUADLOOPS}
    opts_boundary = {Opts.MOVECONSTANTS}

    for d in dims:
        if d == 2:
            volume_geometry: ElementGeometry = TriangleElement()
            boundary_geometry: ElementGeometry = LineElement(space_dimension=2)
            blending_map: GeometryMap = AnnulusMap()
        else:
            volume_geometry = TetrahedronElement()
            boundary_geometry = TriangleElement(space_dimension=3)
            blending_map = IcosahedralShellMap()

        quad_volume = Quadrature(select_quadrule(2, volume_geometry), volume_geometry)
        quad_boundary = Quadrature(
            select_quadrule(5, boundary_geometry), boundary_geometry
        )

        divkgrad = div_k_grad(
            trial, test, volume_geometry, symbolizer, blending_map, coeff
        )
        mass_b = mass_boundary(
            trial,
            test,
            volume_geometry,
            boundary_geometry,
            symbolizer,
            blending_map,
        )

        operator.add_volume_integral(
            name="div_k_grad",
            volume_geometry=volume_geometry,
            quad=quad_volume,
            blending=blending_map,
            form=divkgrad,
            loop_strategy=CUBES(),
            optimizations=opts_volume,
        )

        operator.add_boundary_integral(
            name="mass_boundary",
            volume_geometry=volume_geometry,
            quad=quad_boundary,
            blending=blending_map,
            form=mass_b,
            optimizations=opts_boundary,
        )

    operator.generate_class_code(
        ".",
    )


if __name__ == "__main__":
    test_opgen_smoke()
