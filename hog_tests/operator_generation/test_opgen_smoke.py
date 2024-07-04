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

from hog.blending import IdentityMap
from hog.operator_generation.loop_strategies import CUBES
from hog.operator_generation.optimizer import Opts
from hog.element_geometry import TriangleElement
from hog.function_space import LagrangianFunctionSpace
from hog.operator_generation.operators import (
    HyTeGElementwiseOperator,
    MacroIntegrationDomain,
)
from hog.symbolizer import Symbolizer
from hog.quadrature import Quadrature, select_quadrule
from hog.forms import div_k_grad, mass
from hog.operator_generation.kernel_types import ApplyWrapper
from hog.operator_generation.types import hyteg_type
from hog.blending import AnnulusMap


def test_opgen_smoke():
    """
    Just a simple smoke test to check that an operator can be generated.

    If something is really broken, this will make the CI fail early.

    We are generating a matvec method here for

        ∫ k ∇u · ∇v dx + ∫ uv dx

    with either integral being evaluated in their own kernel.

    That may not be reasonable but tests some features.
    """
    clear_cache()

    symbolizer = Symbolizer()
    volume_geometry = TriangleElement()

    name = f"P2DivKGradBlendingPlusMass"

    trial = LagrangianFunctionSpace(2, symbolizer)
    test = LagrangianFunctionSpace(2, symbolizer)
    coeff = LagrangianFunctionSpace(2, symbolizer)
    quad = Quadrature(select_quadrule(2, volume_geometry), volume_geometry)

    divkgrad = div_k_grad(trial, test, volume_geometry, symbolizer, AnnulusMap(), coeff)
    m = mass(trial, test, volume_geometry, symbolizer, AnnulusMap())

    type_descriptor = hyteg_type()

    kernel_types = [
        ApplyWrapper(
            test,
            trial,
            type_descriptor=type_descriptor,
            dims=[2],
        )
    ]

    opts = {Opts.MOVECONSTANTS, Opts.VECTORIZE, Opts.TABULATE, Opts.QUADLOOPS}

    operator = HyTeGElementwiseOperator(
        name,
        symbolizer=symbolizer,
        kernel_wrapper_types=kernel_types,
        opts=opts,
        type_descriptor=type_descriptor,
    )

    operator.add_integral(
        name="div_k_grad",
        dim=volume_geometry.dimensions,
        geometry=volume_geometry,
        integration_domain=MacroIntegrationDomain.VOLUME,
        quad=quad,
        blending=AnnulusMap(),
        form=divkgrad,
        loop_strategy=CUBES(),
    )

    operator.add_integral(
        name="mass",
        dim=volume_geometry.dimensions,
        geometry=volume_geometry,
        integration_domain=MacroIntegrationDomain.VOLUME,
        quad=quad,
        blending=AnnulusMap(),
        form=m,
        loop_strategy=CUBES(),
    )

    operator.generate_class_code(
        ".",
    )


if __name__ == "__main__":
    test_opgen_smoke()
