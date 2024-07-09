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
import logging

from sympy.core.cache import clear_cache

from hog.blending import AnnulusMap, IcosahedralShellMap
from hog.element_geometry import LineElement, TriangleElement, TetrahedronElement
from hog.function_space import LagrangianFunctionSpace
from hog.operator_generation.operators import (
    HyTeGElementwiseOperator,
)
from hog.symbolizer import Symbolizer
from hog.quadrature import Quadrature, select_quadrule
from hog.operator_generation.kernel_types import ApplyWrapper
from hog.operator_generation.types import hyteg_type
from hog.forms_boundary import mass_boundary
from hog.logger import TimedLogger


def test_boundary_loop():

    # TimedLogger.set_log_level(logging.DEBUG)

    dims = [2, 3]

    clear_cache()

    symbolizer = Symbolizer()

    name = f"P2MassBoundary"

    trial = LagrangianFunctionSpace(2, symbolizer)
    test = LagrangianFunctionSpace(2, symbolizer)

    type_descriptor = hyteg_type()

    kernel_types = [
        ApplyWrapper(
            test,
            trial,
            type_descriptor=type_descriptor,
            dims=dims,
        )
    ]

    operator = HyTeGElementwiseOperator(
        name,
        symbolizer=symbolizer,
        kernel_wrapper_types=kernel_types,
        type_descriptor=type_descriptor,
    )

    for dim in dims:

        if dim == 2:

            volume_geometry = TriangleElement()
            boundary_geometry = LineElement(space_dimension=2)
            blending = AnnulusMap()

        else:

            volume_geometry = TetrahedronElement()
            boundary_geometry = TriangleElement(space_dimension=3)
            blending = IcosahedralShellMap()

        quad = Quadrature(select_quadrule(5, boundary_geometry), boundary_geometry)

        form = mass_boundary(
            trial,
            test,
            volume_geometry,
            boundary_geometry,
            symbolizer,
            blending=blending,
        )

        operator.add_boundary_integral(
            name=f"boundary_mass",
            volume_geometry=volume_geometry,
            quad=quad,
            blending=blending,
            form=form,
        )

    operator.generate_class_code(
        ".",
        clang_format_binary="clang-format",
    )


if __name__ == "__main__":
    test_boundary_loop()
