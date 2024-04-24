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

import quadpy

from typing import Union, Tuple

from sympy.core.cache import clear_cache

from hog.blending import IdentityMap
from hog.operator_generation.loop_strategies import BOUNDARY
from hog.operator_generation.optimizer import Opts
from hog.element_geometry import TetrahedronElement, TriangleElement, ElementGeometry
from hog.function_space import LagrangianFunctionSpace
from hog.logger import get_logger
from hog.operator_generation import indexing
from hog.operator_generation.indexing import (
    CellType,
    DoFIndex,
    DoFType,
    EdgeType,
    FaceType,
    IndexingInfo,
)
from hog.operator_generation.operators import (
    HyTeGElementwiseOperator,
    MacroIntegrationDomain,
)
import sympy as sp
from hog.symbolizer import Symbolizer
from hog.quadrature import Quadrature, select_quadrule
from hog.forms import diffusion
import hog.operator_generation.indexing
from hog.operator_generation.kernel_types import Apply
from hog.operator_generation.indexing import (
    VolumeDoFMemoryLayout,
    num_microfaces_per_face,
    num_microcells_per_cell,
)
from hog.operator_generation.types import hyteg_type


def test_boundary_loop():
    clear_cache()

    symbolizer = Symbolizer()
    geometry = TriangleElement()
    name = f"P1DiffusionBoundary"

    trial = LagrangianFunctionSpace(1, symbolizer)
    test = LagrangianFunctionSpace(1, symbolizer)
    quad = Quadrature(select_quadrule(1, geometry), geometry)

    form = diffusion(trial, test, geometry, symbolizer)

    type_descriptor = hyteg_type()

    kernel_types = [
        Apply(
            test,
            trial,
            type_descriptor=type_descriptor,
            dims=[2],
        )
    ]

    opts = set()

    operator = HyTeGElementwiseOperator(
        name,
        symbolizer=symbolizer,
        kernel_types=kernel_types,
        opts=opts,
        type_descriptor=type_descriptor,
    )

    operator.set_element_matrix(
        dim=geometry.dimensions,
        geometry=geometry,
        integration_domain=MacroIntegrationDomain.VOLUME,
        quad=quad,
        blending=IdentityMap(),
        form=form,
    )

    operator.generate_class_code(
        ".",
        loop_strategy=BOUNDARY(facet_id=2),
        clang_format_binary="clang-format",
    )


if __name__ == "__main__":
    test_boundary_loop()
