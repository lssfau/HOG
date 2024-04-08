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

from typing import Union, Tuple

from sympy.core.cache import clear_cache

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
import sympy as sp
from hog.symbolizer import Symbolizer
import hog.operator_generation.indexing
from hog.operator_generation.indexing import (
    VolumeDoFMemoryLayout,
    num_microfaces_per_face,
    num_microcells_per_cell,
)


def test_micro_element_to_vertex_indices():
    clear_cache()

    assert indexing.micro_element_to_vertex_indices(
        TriangleElement(), FaceType.GRAY, (0, 0, 0)
    ) == [
        DoFIndex((0, 0, 0), DoFType.VERTEX),
        DoFIndex((1, 0, 0), DoFType.VERTEX),
        DoFIndex((0, 1, 0), DoFType.VERTEX),
    ]

    assert indexing.micro_element_to_vertex_indices(
        TriangleElement(), FaceType.BLUE, (2, 4, 0)
    ) == [
        DoFIndex((3, 4, 0), DoFType.VERTEX),
        DoFIndex((2, 5, 0), DoFType.VERTEX),
        DoFIndex((3, 5, 0), DoFType.VERTEX),
    ]

    assert indexing.micro_element_to_vertex_indices(
        TetrahedronElement(), CellType.WHITE_UP, (2, 4, 6)
    ) == [
        DoFIndex((2, 4, 6), DoFType.VERTEX),
        DoFIndex((3, 4, 6), DoFType.VERTEX),
        DoFIndex((2, 5, 6), DoFType.VERTEX),
        DoFIndex((2, 4, 7), DoFType.VERTEX),
    ]

    assert indexing.micro_element_to_vertex_indices(
        TetrahedronElement(), CellType.WHITE_DOWN, (2, 4, 6)
    ) == [
        DoFIndex((3, 5, 6), DoFType.VERTEX),
        DoFIndex((3, 4, 7), DoFType.VERTEX),
        DoFIndex((2, 5, 7), DoFType.VERTEX),
        DoFIndex((3, 5, 7), DoFType.VERTEX),
    ]

    assert indexing.micro_element_to_vertex_indices(
        TetrahedronElement(), CellType.BLUE_UP, (2, 4, 6)
    ) == [
        DoFIndex((3, 4, 6), DoFType.VERTEX),
        DoFIndex((2, 5, 6), DoFType.VERTEX),
        DoFIndex((3, 5, 6), DoFType.VERTEX),
        DoFIndex((3, 4, 7), DoFType.VERTEX),
    ]

    assert indexing.micro_element_to_vertex_indices(
        TetrahedronElement(), CellType.BLUE_DOWN, (2, 4, 6)
    ) == [
        DoFIndex((2, 5, 6), DoFType.VERTEX),
        DoFIndex((2, 4, 7), DoFType.VERTEX),
        DoFIndex((3, 4, 7), DoFType.VERTEX),
        DoFIndex((2, 5, 7), DoFType.VERTEX),
    ]

    assert indexing.micro_element_to_vertex_indices(
        TetrahedronElement(), CellType.GREEN_UP, (2, 4, 6)
    ) == [
        DoFIndex((3, 4, 6), DoFType.VERTEX),
        DoFIndex((2, 5, 6), DoFType.VERTEX),
        DoFIndex((2, 4, 7), DoFType.VERTEX),
        DoFIndex((3, 4, 7), DoFType.VERTEX),
    ]

    assert indexing.micro_element_to_vertex_indices(
        TetrahedronElement(), CellType.GREEN_DOWN, (2, 4, 6)
    ) == [
        DoFIndex((2, 5, 6), DoFType.VERTEX),
        DoFIndex((3, 5, 6), DoFType.VERTEX),
        DoFIndex((3, 4, 7), DoFType.VERTEX),
        DoFIndex((2, 5, 7), DoFType.VERTEX),
    ]


def test_micro_vertex_to_edge_indices():
    clear_cache()

    # GRAY
    assert indexing.micro_vertex_to_edge_indices(
        TriangleElement(),
        [
            DoFIndex((5, 4, 0), DoFType.VERTEX),
            DoFIndex((6, 4, 0), DoFType.VERTEX),
            DoFIndex((5, 5, 0), DoFType.VERTEX),
        ],
    ) == [
        DoFIndex((5, 4, 0), DoFType.EDGE, EdgeType.XY),
        DoFIndex((5, 4, 0), DoFType.EDGE, EdgeType.Y),
        DoFIndex((5, 4, 0), DoFType.EDGE, EdgeType.X),
    ]

    # BLUE
    assert indexing.micro_vertex_to_edge_indices(
        TriangleElement(),
        [
            DoFIndex((6, 4, 0), DoFType.VERTEX),
            DoFIndex((5, 5, 0), DoFType.VERTEX),
            DoFIndex((6, 5, 0), DoFType.VERTEX),
        ],
    ) == [
        DoFIndex((5, 5, 0), DoFType.EDGE, EdgeType.X),
        DoFIndex((6, 4, 0), DoFType.EDGE, EdgeType.Y),
        DoFIndex((5, 4, 0), DoFType.EDGE, EdgeType.XY),
    ]

    # WHITE UP
    assert indexing.micro_vertex_to_edge_indices(
        TetrahedronElement(),
        [
            DoFIndex((2, 4, 6), DoFType.VERTEX),
            DoFIndex((3, 4, 6), DoFType.VERTEX),
            DoFIndex((2, 5, 6), DoFType.VERTEX),
            DoFIndex((2, 4, 7), DoFType.VERTEX),
        ],
    ) == [
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.YZ),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.XZ),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.XY),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.Z),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.Y),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.X),
    ]

    # WHITE DOWN
    assert indexing.micro_vertex_to_edge_indices(
        TetrahedronElement(),
        [
            DoFIndex((3, 5, 6), DoFType.VERTEX),
            DoFIndex((3, 4, 7), DoFType.VERTEX),
            DoFIndex((2, 5, 7), DoFType.VERTEX),
            DoFIndex((3, 5, 7), DoFType.VERTEX),
        ],
    ) == [
        DoFIndex((2, 5, 7), DoFType.EDGE, EdgeType.X),
        DoFIndex((3, 4, 7), DoFType.EDGE, EdgeType.Y),
        DoFIndex((2, 4, 7), DoFType.EDGE, EdgeType.XY),
        DoFIndex((3, 5, 6), DoFType.EDGE, EdgeType.Z),
        DoFIndex((2, 5, 6), DoFType.EDGE, EdgeType.XZ),
        DoFIndex((3, 4, 6), DoFType.EDGE, EdgeType.YZ),
    ]

    # BLUE UP
    assert indexing.micro_vertex_to_edge_indices(
        TetrahedronElement(),
        [
            DoFIndex((3, 4, 6), DoFType.VERTEX),
            DoFIndex((2, 5, 6), DoFType.VERTEX),
            DoFIndex((3, 5, 6), DoFType.VERTEX),
            DoFIndex((3, 4, 7), DoFType.VERTEX),
        ],
    ) == [
        DoFIndex((3, 4, 6), DoFType.EDGE, EdgeType.YZ),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.XYZ),
        DoFIndex((2, 5, 6), DoFType.EDGE, EdgeType.X),
        DoFIndex((3, 4, 6), DoFType.EDGE, EdgeType.Z),
        DoFIndex((3, 4, 6), DoFType.EDGE, EdgeType.Y),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.XY),
    ]

    # BLUE DOWN
    assert indexing.micro_vertex_to_edge_indices(
        TetrahedronElement(),
        [
            DoFIndex((2, 5, 6), DoFType.VERTEX),
            DoFIndex((2, 4, 7), DoFType.VERTEX),
            DoFIndex((3, 4, 7), DoFType.VERTEX),
            DoFIndex((2, 5, 7), DoFType.VERTEX),
        ],
    ) == [
        DoFIndex((2, 4, 7), DoFType.EDGE, EdgeType.XY),
        DoFIndex((2, 4, 7), DoFType.EDGE, EdgeType.Y),
        DoFIndex((2, 4, 7), DoFType.EDGE, EdgeType.X),
        DoFIndex((2, 5, 6), DoFType.EDGE, EdgeType.Z),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.XYZ),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.YZ),
    ]

    # GREEN UP
    assert indexing.micro_vertex_to_edge_indices(
        TetrahedronElement(),
        [
            DoFIndex((3, 4, 6), DoFType.VERTEX),
            DoFIndex((2, 5, 6), DoFType.VERTEX),
            DoFIndex((2, 4, 7), DoFType.VERTEX),
            DoFIndex((3, 4, 7), DoFType.VERTEX),
        ],
    ) == [
        DoFIndex((2, 4, 7), DoFType.EDGE, EdgeType.X),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.XYZ),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.YZ),
        DoFIndex((3, 4, 6), DoFType.EDGE, EdgeType.Z),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.XZ),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.XY),
    ]

    # GREEN DOWN
    assert indexing.micro_vertex_to_edge_indices(
        TetrahedronElement(),
        [
            DoFIndex((2, 5, 6), DoFType.VERTEX),
            DoFIndex((3, 5, 6), DoFType.VERTEX),
            DoFIndex((3, 4, 7), DoFType.VERTEX),
            DoFIndex((2, 5, 7), DoFType.VERTEX),
        ],
    ) == [
        DoFIndex((2, 4, 7), DoFType.EDGE, EdgeType.XY),
        DoFIndex((2, 5, 6), DoFType.EDGE, EdgeType.XZ),
        DoFIndex((3, 4, 6), DoFType.EDGE, EdgeType.YZ),
        DoFIndex((2, 5, 6), DoFType.EDGE, EdgeType.Z),
        DoFIndex((2, 4, 6), DoFType.EDGE, EdgeType.XYZ),
        DoFIndex((2, 5, 6), DoFType.EDGE, EdgeType.X),
    ]


def test_micro_volume_to_volume_indices():
    clear_cache()
    indexingInfo = IndexingInfo()
    hog.operator_generation.indexing.USE_SYMPY_INT_DIV = True

    def test_element_type_on_level(
        geometry: ElementGeometry,
        level: int,
        indexing_info: IndexingInfo,
        n_dofs_per_primitive,
        primitive_type: Union[FaceType, CellType],
        primitive_index: Tuple[int, int, int],
        target_array_index: int,
        intra_primitive_index: int = 0,
        memory_layout: VolumeDoFMemoryLayout = VolumeDoFMemoryLayout.AoS,
    ):
        indexing_info.level = level
        dof_indices = indexing.micro_element_to_volume_indices(
            primitive_type, primitive_index, n_dofs_per_primitive, memory_layout
        )
        array_index = sp.simplify(
            dof_indices[intra_primitive_index].array_index(geometry, indexing_info)
        )
        print(array_index)
        assert array_index == target_array_index

    # 2D, P0:
    test_element_type_on_level(
        TriangleElement(), 2, indexingInfo, 1, FaceType.GRAY, (2, 0, 0), 2
    )
    test_element_type_on_level(
        TriangleElement(), 2, indexingInfo, 1, FaceType.GRAY, (1, 2, 0), 8
    )
    test_element_type_on_level(
        TriangleElement(), 2, indexingInfo, 1, FaceType.BLUE, (1, 1, 0), 14
    )
    test_element_type_on_level(
        TriangleElement(), 2, indexingInfo, 1, FaceType.BLUE, (0, 2, 0), 15
    )

    # 2D, P1:
    level = 2
    indexingInfo.num_microfaces_per_face = (
        TriangleElement(),
        hog.operator_generation.indexing.num_microfaces_per_face(level),
    )
    test_element_type_on_level(
        TriangleElement(),
        level,
        indexingInfo,
        3,
        FaceType.GRAY,
        (2, 0, 0),
        7,
        intra_primitive_index=1,
    )
    test_element_type_on_level(
        TriangleElement(),
        level,
        indexingInfo,
        3,
        FaceType.GRAY,
        (1, 2, 0),
        26,
        intra_primitive_index=2,
    )
    test_element_type_on_level(
        TriangleElement(),
        level,
        indexingInfo,
        3,
        FaceType.BLUE,
        (1, 1, 0),
        30 + 13,
        intra_primitive_index=1,
    )
    test_element_type_on_level(
        TriangleElement(),
        level,
        indexingInfo,
        3,
        FaceType.BLUE,
        (0, 2, 0),
        30 + 17,
        intra_primitive_index=2,
    )

    # 2D, P1, SoA:
    level = 2
    indexingInfo.num_microfaces_per_face = num_microfaces_per_face(level)

    test_element_type_on_level(
        TriangleElement(),
        level,
        indexingInfo,
        3,
        FaceType.GRAY,
        (2, 0, 0),
        18,
        intra_primitive_index=1,
        memory_layout=VolumeDoFMemoryLayout.SoA,
    )
    test_element_type_on_level(
        TriangleElement(),
        level,
        indexingInfo,
        3,
        FaceType.GRAY,
        (1, 1, 0),
        32 + 5,
        intra_primitive_index=2,
        memory_layout=VolumeDoFMemoryLayout.SoA,
    )
    test_element_type_on_level(
        TriangleElement(),
        level,
        indexingInfo,
        3,
        FaceType.BLUE,
        (1, 1, 0),
        26 + 4,
        intra_primitive_index=1,
        memory_layout=VolumeDoFMemoryLayout.SoA,
    )
    test_element_type_on_level(
        TriangleElement(),
        level,
        indexingInfo,
        3,
        FaceType.BLUE,
        (0, 2, 0),
        42 + 5,
        intra_primitive_index=2,
        memory_layout=VolumeDoFMemoryLayout.SoA,
    )

    # 3D, P0:
    test_element_type_on_level(
        TetrahedronElement(), 2, indexingInfo, 1, CellType.WHITE_UP, (0, 1, 2), 18
    )
    test_element_type_on_level(
        TetrahedronElement(), 2, indexingInfo, 1, CellType.WHITE_DOWN, (0, 0, 1), 43
    )
    test_element_type_on_level(
        TetrahedronElement(), 2, indexingInfo, 1, CellType.BLUE_UP, (0, 0, 2), 29
    )
    test_element_type_on_level(
        TetrahedronElement(), 2, indexingInfo, 1, CellType.GREEN_DOWN, (0, 1, 1), 62
    )

    # 3D, P1:
    test_element_type_on_level(
        TetrahedronElement(),
        2,
        indexingInfo,
        4,
        CellType.WHITE_UP,
        (0, 1, 2),
        75,
        intra_primitive_index=3,
    )
    test_element_type_on_level(
        TetrahedronElement(),
        2,
        indexingInfo,
        4,
        CellType.WHITE_DOWN,
        (0, 0, 1),
        174,
        intra_primitive_index=2,
    )
    test_element_type_on_level(
        TetrahedronElement(),
        2,
        indexingInfo,
        4,
        CellType.BLUE_UP,
        (0, 0, 2),
        117,
        intra_primitive_index=1,
    )
    test_element_type_on_level(
        TetrahedronElement(),
        2,
        indexingInfo,
        4,
        CellType.GREEN_DOWN,
        (0, 1, 1),
        248,
        intra_primitive_index=0,
    )

    # 3D, P1, SoA:
    indexingInfo.num_microcells_per_cell = num_microcells_per_cell(level)

    test_element_type_on_level(
        TetrahedronElement(),
        2,
        indexingInfo,
        4,
        CellType.WHITE_UP,
        (2, 0, 0),
        64 + 2,
        intra_primitive_index=1,
        memory_layout=VolumeDoFMemoryLayout.SoA,
    )
    test_element_type_on_level(
        TetrahedronElement(),
        2,
        indexingInfo,
        4,
        CellType.WHITE_DOWN,
        (0, 0, 1),
        2 * 64 + 40 + 3,
        intra_primitive_index=2,
        memory_layout=VolumeDoFMemoryLayout.SoA,
    )
    test_element_type_on_level(
        TetrahedronElement(),
        2,
        indexingInfo,
        4,
        CellType.GREEN_UP,
        (0, 1, 1),
        3 * 64 + 30 + 8,
        intra_primitive_index=3,
        memory_layout=VolumeDoFMemoryLayout.SoA,
    )
    hog.operator_generation.indexing.USE_SYMPY_INT_DIV = False
