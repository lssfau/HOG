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

from hog.operator_generation.loop_strategies import CUBES, BOUNDARY
from hog.operator_generation.optimizer import Opts
from hog.element_geometry import LineElement, TriangleElement
from hog.function_space import LagrangianFunctionSpace
from hog.operator_generation.operators import (
    HyTeGElementwiseOperator,
    MacroIntegrationDomain,
)
from hog.symbolizer import Symbolizer
from hog.quadrature import Quadrature, select_quadrule
from hog.forms import div_k_grad
from hog.forms_boundary import mass_boundary
from hog.operator_generation.kernel_types import ApplyWrapper, AssembleWrapper
from hog.operator_generation.types import hyteg_type
from hog.blending import AnnulusMap


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
    volume_geometry = TriangleElement()
    boundary_geometry = LineElement(space_dimension=2)

    name = f"P2DivKGradBlendingPlusBoundaryMass"

    trial = LagrangianFunctionSpace(2, symbolizer)
    test = LagrangianFunctionSpace(2, symbolizer)
    coeff = LagrangianFunctionSpace(2, symbolizer)
    quad_volume = Quadrature(select_quadrule(2, volume_geometry), volume_geometry)
    quad_boundary = Quadrature(select_quadrule(5, boundary_geometry), boundary_geometry)

    divkgrad = div_k_grad(trial, test, volume_geometry, symbolizer, AnnulusMap(), coeff)
    mass_b = mass_boundary(
        trial, test, volume_geometry, boundary_geometry, symbolizer, AnnulusMap()
    )

    type_descriptor = hyteg_type()

    kernel_types = [
        ApplyWrapper(
            test,
            trial,
            type_descriptor=type_descriptor,
            dims=[2],
        ),
        AssembleWrapper(
            test,
            trial,
            type_descriptor=type_descriptor,
            dims=[2],
        ),
    ]

    opts_volume = {Opts.MOVECONSTANTS, Opts.VECTORIZE, Opts.TABULATE, Opts.QUADLOOPS}
    opts_boundary = {Opts.MOVECONSTANTS}

    operator = HyTeGElementwiseOperator(
        name,
        symbolizer=symbolizer,
        kernel_wrapper_types=kernel_types,
        type_descriptor=type_descriptor,
    )

    operator.add_volume_integral(
        name="div_k_grad",
        volume_geometry=volume_geometry,
        quad=quad_volume,
        blending=AnnulusMap(),
        form=divkgrad,
        loop_strategy=CUBES(),
        optimizations=opts_volume,
    )

    operator.add_boundary_integral(
        name="mass_boundary",
        volume_geometry=volume_geometry,
        quad=quad_boundary,
        blending=AnnulusMap(),
        form=mass_b,
        optimizations=opts_boundary,
    )

    operator.generate_class_code(
        ".",
    )


if __name__ == "__main__":
    test_opgen_smoke()
