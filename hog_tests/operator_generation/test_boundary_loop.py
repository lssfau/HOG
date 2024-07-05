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

from hog.blending import IdentityMap, GeometryMap, AnnulusMap
from hog.exception import HOGException
from hog.fem_helpers import create_empty_element_matrix, element_matrix_iterator
from hog.operator_generation.loop_strategies import BOUNDARY, SAWTOOTH
from hog.operator_generation.optimizer import Opts
from hog.element_geometry import (
    TetrahedronElement,
    TriangleElement,
    LineElement,
    EmbeddedLine,
    ElementGeometry,
)
from hog.function_space import LagrangianFunctionSpace, FunctionSpace
from hog.logger import get_logger, TimedLogger
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
from hog.quadrature import Quadrature, select_quadrule, Tabulation
from hog.forms import diffusion, Form
import hog.operator_generation.indexing
from hog.operator_generation.kernel_types import ApplyWrapper
from hog.operator_generation.indexing import (
    VolumeDoFMemoryLayout,
    num_microfaces_per_face,
    num_microcells_per_cell,
)
from hog.operator_generation.types import hyteg_type
from hog.forms_boundary import mass_boundary


def test_boundary_loop():
    clear_cache()

    symbolizer = Symbolizer()
    volume_geometry = TriangleElement()
    boundary_geometry = EmbeddedLine()
    blending = AnnulusMap()

    name = f"P2MassBoundary"

    trial = LagrangianFunctionSpace(2, symbolizer)
    test = LagrangianFunctionSpace(2, symbolizer)
    quad = Quadrature(select_quadrule(5, boundary_geometry), boundary_geometry)

    form = mass_boundary(
        trial,
        test,
        volume_geometry,
        boundary_geometry,
        symbolizer,
        blending=blending,
    )

    type_descriptor = hyteg_type()

    kernel_types = [
        ApplyWrapper(
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
        kernel_wrapper_types=kernel_types,
        opts=opts,
        type_descriptor=type_descriptor,
    )

    for facet_id in range(3):

        operator.add_integral(
            name=f"boundarymass_{facet_id}",
            dim=volume_geometry.dimensions,
            geometry=volume_geometry,
            integration_domain=MacroIntegrationDomain.DOMAIN_BOUNDARY_FREESLIP,
            quad=quad,
            blending=blending,
            form=form,
            loop_strategy=BOUNDARY(facet_id=facet_id),
        )

    operator.generate_class_code(
        ".",
        clang_format_binary="clang-format",
    )


if __name__ == "__main__":
    test_boundary_loop()
