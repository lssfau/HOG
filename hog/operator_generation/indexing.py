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

from enum import Enum
from typing import Any, List, Tuple, Union, Optional

import operator
import sympy as sp

from hog.element_geometry import ElementGeometry, TriangleElement, TetrahedronElement
from hog.exception import HOGException
from hog.symbolizer import Symbolizer

from pystencils.integer_functions import int_div
from pystencils import TypedSymbol
from enum import Enum
from typing import Tuple, List, Union

import operator
import sympy as sp

from hog.element_geometry import ElementGeometry, TriangleElement, TetrahedronElement
from hog.exception import HOGException
from hog.symbolizer import Symbolizer

from pystencils.integer_functions import int_div

# only for testing, switch off casts etc to make a pure sympy AST that can be evaluated
USE_SYMPY_INT_DIV = False


class DoFType(Enum):
    """
    Used to differentiate the different DoFs defined in HyTeG.
    """

    VERTEX = "VERTEX"
    EDGE = "EDGE"
    VOLUME = "VOLUME"


class EdgeType(Enum):
    """
    The six different edge types that occur during regular refinement of tetrahedrons as in Bey's paper.

    In 2D, we use X, Y, and XY.
    """

    X = "X"
    Y = "Y"
    Z = "Z"
    XY = "XY"
    XZ = "XZ"
    YZ = "YZ"
    XYZ = "XYZ"


class FaceType(Enum):
    """
    Face types. Currently only meaningful in 2D.
    """

    GRAY = "GRAY"
    BLUE = "BLUE"


class CellType(Enum):
    """
    The six different cell types that occur during regular refinement of tetrahedrons as in Bey's paper.
    """

    WHITE_UP = "WHITE_UP"
    WHITE_DOWN = "WHITE_DOWN"
    BLUE_UP = "BLUE_UP"
    BLUE_DOWN = "BLUE_DOWN"
    GREEN_UP = "GREEN_UP"
    GREEN_DOWN = "GREEN_DOWN"


class Point:
    """
    Resembles a point in 2D or 3D to make index computations more visual.
    """

    def _init_(self, x=0, y=0, z=None):
        self.x = x
        self.y = y
        self.z = z


class VolumeDoFMemoryLayout(Enum):
    """
    Array of structs or stuct of array; memory layouts for DG
    """

    SoA = "SoA"
    AoS = "AoS"


def sympy_int_div(x: float, y: float) -> int:
    """Int division that does not introduce pystencils nodes into the AST. That keeps it evaluatable."""
    return sp.floor(x / y)


def generalized_macro_face_index(
    logical_index: Tuple[int, int, int], width: int
) -> int:
    """Indexes macro faces. width depends on the level and quantifies the amount of primitives in one direction of the refined triangle."""
    x, y, _ = logical_index
    if USE_SYMPY_INT_DIV:
        row_offset = y * (width + 1) - sympy_int_div(((y + 1) * y), 2)
    else:
        row_offset = y * (width + 1) - int_div(((y + 1) * y), 2)
    return row_offset + x


def generalized_macro_cell_index(
    logical_index: Tuple[int, int, int], width: int
) -> int:
    """Indexes macro tetrahedra. width depends on the level and quantifies the amount of primitives in one direction of the refined tetrahedron."""

    x, y, z = logical_index
    width_minus_slice = width - z
    if USE_SYMPY_INT_DIV:
        slice_offset = int_div(((width + 2) * (width + 1) * width), 6) - sympy_int_div(
            ((width_minus_slice + 2) * (width_minus_slice + 1) * width_minus_slice), 6
        )
        row_offset = y * (width_minus_slice + 1) - sympy_int_div(((y + 1) * y), 2)
    else:
        slice_offset = int_div(((width + 2) * (width + 1) * width), 6) - int_div(
            ((width_minus_slice + 2) * (width_minus_slice + 1) * width_minus_slice), 6
        )
        row_offset = y * (width_minus_slice + 1) - int_div(((y + 1) * y), 2)
    return slice_offset + row_offset + x


def num_microvertices_per_face_from_width(width: int) -> int:
    """Computes the number of microvertices in a refined macro triangle. width depends on the level and quantifies the amount of primitives in one direction of the refined triangle."""

    return (width * (width + 1)) >> 1


def num_microvertices_per_cell_from_width(width: int) -> int:
    """Computes the number of microvertices in a refined macro tetrahedron. width depends on the level and quantifies the amount of primitives in one direction of the refined triangle."""

    if USE_SYMPY_INT_DIV:
        return sympy_int_div(((width + 2) * (width + 1) * width), 6)
    else:
        return int_div(((width + 2) * (width + 1) * width), 6)


def num_microfaces_per_face(level: int) -> int:
    return 4**level


def num_microcells_per_cell(level: int) -> int:
    return 8 ** (level)


def linear_macro_cell_size(width: int) -> int:
    if USE_SYMPY_INT_DIV:
        return sympy_int_div((width + 2) * (width + 1) * width, 6)
    else:
        return int_div((width + 2) * (width + 1) * width, 6)


def num_microvertices_per_edge(level: int) -> int:
    return 2**level + 1


def num_microedges_per_edge(level: int) -> int:
    return num_microvertices_per_edge(level) - 1


def num_faces_per_row_by_type(
    level: int, faceType: Union[None, EdgeType, FaceType, CellType]
) -> int:
    if faceType == FaceType.GRAY:
        return num_microedges_per_edge(level)
    elif faceType == FaceType.BLUE:
        return num_microedges_per_edge(level) - 1
    else:
        raise HOGException(f"Unexpected face type: {faceType}")


def num_cells_per_row_by_type(
    level: int, cellType: Union[None, EdgeType, FaceType, CellType]
) -> int:
    if cellType == CellType.WHITE_UP:
        return num_microedges_per_edge(level)
    elif cellType == CellType.GREEN_UP:
        return num_microedges_per_edge(level) - 1
    elif cellType == CellType.BLUE_UP:
        return num_microedges_per_edge(level) - 1
    elif cellType == CellType.WHITE_DOWN:
        return num_microedges_per_edge(level) - 2
    elif cellType == CellType.BLUE_DOWN:
        return num_microedges_per_edge(level) - 1
    elif cellType == CellType.GREEN_DOWN:
        return num_microedges_per_edge(level) - 1
    else:
        raise HOGException(f"Unexpected cell type: {cellType}")


def num_micro_faces_per_macro_face(level: int, faceType: FaceType) -> int:
    return num_microvertices_per_face_from_width(
        num_faces_per_row_by_type(level, faceType)
    )


def num_micro_cells_per_macro_cell(level: int, cellType: CellType) -> int:
    return num_microvertices_per_cell_from_width(
        num_cells_per_row_by_type(level, cellType)
    )


def linear_macro_face_index(width: int, x: int, y: int) -> int:
    if USE_SYMPY_INT_DIV:
        rowOffset = y * ((width) + 1) - sympy_int_div(((y + 1) * (y)), 2)
    else:
        rowOffset = y * ((width) + 1) - int_div(((y + 1) * (y)), 2)
    return rowOffset + x


def linear_macro_cell_index(width: int, x: int, y: int, z: int) -> int:
    widthMinusSlice = width - z
    if USE_SYMPY_INT_DIV:
        sliceOffset = linear_macro_cell_size(width) - sympy_int_div(
            (widthMinusSlice + 2) * (widthMinusSlice + 1) * widthMinusSlice, 6
        )
        rowOffset = y * (widthMinusSlice + 1) - sympy_int_div((y + 1) * y, 2)
    else:
        sliceOffset = linear_macro_cell_size(width) - int_div(
            (widthMinusSlice + 2) * (widthMinusSlice + 1) * widthMinusSlice, 6
        )
        rowOffset = y * (widthMinusSlice + 1) - int_div((y + 1) * y, 2)
    return sliceOffset + rowOffset + x


def facedof_index(
    level: int,
    index: Tuple[int, int, int],
    faceType: Union[None, EdgeType, FaceType, CellType],
) -> int:
    """Indexes triangles/faces. Used to compute offsets in volume dof indexing in 2D and AoS layout."""
    x, y, _ = index
    width = num_faces_per_row_by_type(level, faceType)
    if faceType == FaceType.GRAY:
        return linear_macro_face_index(width, x, y)
    elif faceType == FaceType.BLUE:
        return num_micro_faces_per_macro_face(
            level, FaceType.GRAY
        ) + linear_macro_face_index(width, x, y)
    else:
        raise HOGException(f"Unexpected face type: {faceType}")


def celldof_index(
    level: int,
    index: Tuple[int, int, int],
    cellType: Union[None, EdgeType, FaceType, CellType],
) -> int:
    """Indexes cells/tetrahedra. Used to compute offsets in volume dof indexing in 3D and AoS layout."""
    x, y, z = index
    width = num_cells_per_row_by_type(level, cellType)  # gives expr(level)
    if cellType == CellType.WHITE_UP:
        return linear_macro_cell_index(width, x, y, z)
    elif cellType == CellType.BLUE_UP:
        return num_micro_cells_per_macro_cell(  # gives expr(level)
            level, CellType.WHITE_UP
        ) + linear_macro_cell_index(width, x, y, z)
    elif cellType == CellType.GREEN_UP:
        return (
            num_micro_cells_per_macro_cell(level, CellType.WHITE_UP)
            + num_micro_cells_per_macro_cell(level, CellType.BLUE_UP)
            + linear_macro_cell_index(width, x, y, z)
        )
    elif cellType == CellType.WHITE_DOWN:
        return (
            num_micro_cells_per_macro_cell(level, CellType.WHITE_UP)
            + num_micro_cells_per_macro_cell(level, CellType.BLUE_UP)
            + num_micro_cells_per_macro_cell(level, CellType.GREEN_UP)
            + linear_macro_cell_index(width, x, y, z)
        )
    elif cellType == CellType.BLUE_DOWN:
        return (
            num_micro_cells_per_macro_cell(level, CellType.WHITE_UP)
            + num_micro_cells_per_macro_cell(level, CellType.BLUE_UP)
            + num_micro_cells_per_macro_cell(level, CellType.GREEN_UP)
            + num_micro_cells_per_macro_cell(level, CellType.WHITE_DOWN)
            + linear_macro_cell_index(width, x, y, z)
        )
    elif cellType == CellType.GREEN_DOWN:
        return (
            num_micro_cells_per_macro_cell(level, CellType.WHITE_UP)
            + num_micro_cells_per_macro_cell(level, CellType.BLUE_UP)
            + num_micro_cells_per_macro_cell(level, CellType.GREEN_UP)
            + num_micro_cells_per_macro_cell(level, CellType.WHITE_DOWN)
            + num_micro_cells_per_macro_cell(level, CellType.BLUE_DOWN)
            + linear_macro_cell_index(width, x, y, z)
        )
    else:
        raise HOGException()


class IndexingInfo:
    """Encapsulates essential properties of the refined mesh: micro_edges_per_macro_edge in conforming function spaces
    and num_microfaces_per_face, num_microcells_per_cell, for discontinuous galerkin. These are required for all indexing
    maps."""

    def __init__(self):
        self.level = TypedSymbol("level", int)
        self.micro_edges_per_macro_edge = TypedSymbol("micro_edges_per_macro_edge", int)
        self.num_microfaces_per_face = TypedSymbol("num_microfaces_per_face", int)
        self.num_microcells_per_cell = TypedSymbol("num_microcells_per_cell", int)
        self.micro_edges_per_macro_edge_float = sp.Symbol(
            "micro_edges_per_macro_edge_float"
        )


def all_element_types(dimensions: int) -> List[Union[FaceType, CellType]]:
    if dimensions == 2:
        return [FaceType.GRAY, FaceType.BLUE]
    if dimensions == 3:
        return list(CellType)
    else:
        raise HOGException("Invalid dim.")


class DoFIndex:
    """
    Class that summarizes all relevant information of a dof index.

    :param primitive_index: index of the geometric primitive the dof is located on.
    :param dof_type: type of the geometric primitive the dof is located on, e.g. a vertex or volume.
    :param dof_sub_type: second primitive typing only relevant for edge dofs.
    :param n_dofs_per_primitive: number of dofs on each primitive. For nodal, conforming Galerkin,
           we only support n_dofs_per_primitive == 1 on vertices and edges. Then, primitive_index directly
           indexes the dof.
    :param intra_primitive_index: index within the dofs associated with a single primitive; currently only relevant
           for volume dofs. For all other primitives, we only have a single dof per primitive, such that
           intra_primitive_index == 0 always.
    :param memory_layout: memory layout for the dofs associated with a single primitive; currently only relevant
           for volume dofs.
    """

    def __init__(
        self,
        primitive_index: Tuple[int, int, int],
        dof_type: DoFType = DoFType.VERTEX,
        dof_sub_type: Union[None, EdgeType, FaceType, CellType] = None,
        n_dofs_per_primitive: int = 1,
        intra_primitive_index: int = 0,
        memory_layout: VolumeDoFMemoryLayout = VolumeDoFMemoryLayout.AoS,
    ):
        if intra_primitive_index > 0 and (
            dof_type == DoFType.VERTEX or dof_type == DoFType.EDGE
        ):
            raise HOGException(
                "intra_primitive_index > 0 not supported yet for edge and vertex DoFs."
            )

        self.primitive_index = primitive_index
        self.dof_type = dof_type
        self.dof_sub_type = dof_sub_type
        self.mem_layout = memory_layout
        self.intra_primitive_index = intra_primitive_index
        self.n_dofs_per_primitive = n_dofs_per_primitive

    def __getitem__(self, i):
        return self.primitive_index[i]

    def array_index(
        self, geometry: ElementGeometry, indexing_info: IndexingInfo
    ) -> int:
        """
        Computes the array index of the passed DoF.
        """
        if self.dof_type == DoFType.VERTEX:
            width = indexing_info.micro_edges_per_macro_edge + 1
            if isinstance(geometry, TriangleElement):
                return generalized_macro_face_index(self.primitive_index, width)
            elif isinstance(geometry, TetrahedronElement):
                return generalized_macro_cell_index(self.primitive_index, width)
            else:
                raise HOGException(
                    "Indexing function not implemented for this geometry."
                )
        elif self.dof_type == DoFType.EDGE:
            width = indexing_info.micro_edges_per_macro_edge
            if isinstance(geometry, TriangleElement):
                micro_edges_one_type_per_macro_face = int_div(
                    (indexing_info.micro_edges_per_macro_edge + 1)
                    * indexing_info.micro_edges_per_macro_edge,
                    2,
                )

                order: List[Union[None, EdgeType, FaceType, CellType]] = [
                    EdgeType.X,
                    EdgeType.XY,
                    EdgeType.Y,
                ]
                return order.index(
                    self.dof_sub_type
                ) * micro_edges_one_type_per_macro_face + generalized_macro_face_index(
                    self.primitive_index, width
                )
            elif isinstance(geometry, TetrahedronElement):
                order = [
                    EdgeType.X,
                    EdgeType.Y,
                    EdgeType.Z,
                    EdgeType.XY,
                    EdgeType.XZ,
                    EdgeType.YZ,
                    EdgeType.XYZ,
                ]
                return order.index(
                    self.dof_sub_type
                ) * num_microvertices_per_cell_from_width(
                    width
                ) + generalized_macro_cell_index(
                    self.primitive_index,
                    width - (1 if self.dof_sub_type == EdgeType.XYZ else 0),
                )
            else:
                raise HOGException(
                    "Indexing function not implemented for this geometry."
                )
        elif self.dof_type == DoFType.VOLUME:
            if isinstance(geometry, TriangleElement):
                numMicroVolumes = indexing_info.num_microfaces_per_face

                microVolume = facedof_index(
                    indexing_info.level, self.primitive_index, self.dof_sub_type
                )

                if self.mem_layout == VolumeDoFMemoryLayout.SoA:
                    idx = numMicroVolumes * self.intra_primitive_index + microVolume
                    return idx
                else:
                    idx = (
                        microVolume * self.n_dofs_per_primitive
                        + self.intra_primitive_index
                    )
                    return idx
            elif isinstance(geometry, TetrahedronElement):
                numMicroVolumes = indexing_info.num_microcells_per_cell

                microVolume = celldof_index(
                    indexing_info.level, self.primitive_index, self.dof_sub_type
                )

                if self.mem_layout == VolumeDoFMemoryLayout.SoA:
                    idx = numMicroVolumes * self.intra_primitive_index + microVolume
                    return idx
                else:
                    idx = (
                        microVolume * self.n_dofs_per_primitive
                        + self.intra_primitive_index
                    )
                    return idx

            else:
                raise HOGException(
                    "Indexing function not implemented for this geometry."
                )
        else:
            raise HOGException(
                "Indexing function not implemented for this function space."
            )

    def __eq__(self, o: Any) -> bool:
        return (
            type(self) == type(o)
            and self.primitive_index == o.primitive_index
            and self.dof_type == o.dof_type
            and self.dof_sub_type == o.dof_sub_type
        )

    def __repr__(self) -> str:
        return f"DoFIndex({self.primitive_index}, {self.dof_type}, {self.dof_sub_type})"


def micro_element_to_vertex_indices(
    geometry: ElementGeometry,
    element_type: Union[FaceType, CellType],
    element_index: Tuple[int, int, int],
) -> List[DoFIndex]:
    """
    Returns the micro vertex indices of the given micro element, ordered such
    that orientations of mesh entities are locally consistent.
    """
    if isinstance(geometry, TriangleElement):
        # fmt: off
        if element_type == FaceType.GRAY:
            return [
                DoFIndex((element_index[0]    , element_index[1]    , 0)),
                DoFIndex((element_index[0] + 1, element_index[1]    , 0)),
                DoFIndex((element_index[0]    , element_index[1] + 1, 0)),
            ]
        elif element_type == FaceType.BLUE:
            return [
                DoFIndex((element_index[0] + 1, element_index[1]    , 0)),
                DoFIndex((element_index[0]    , element_index[1] + 1, 0)),
                DoFIndex((element_index[0] + 1, element_index[1] + 1, 0)),
            ]
        else:
            raise HOGException(f"Invalid face type {element_type} for 2D.")
        # fmt: on
    elif isinstance(geometry, TetrahedronElement):
        # fmt: off
        if element_type == CellType.WHITE_UP:
            return [
                DoFIndex((element_index[0]    , element_index[1]    , element_index[2])),
                DoFIndex((element_index[0] + 1, element_index[1]    , element_index[2])),
                DoFIndex((element_index[0]    , element_index[1] + 1, element_index[2])),
                DoFIndex((element_index[0]    , element_index[1]    , element_index[2] + 1)),
            ]
        elif element_type == CellType.WHITE_DOWN:
            return [
                DoFIndex((element_index[0] + 1, element_index[1] + 1, element_index[2])),
                DoFIndex((element_index[0] + 1, element_index[1]    , element_index[2] + 1)),
                DoFIndex((element_index[0]    , element_index[1] + 1, element_index[2] + 1)),
                DoFIndex((element_index[0] + 1, element_index[1] + 1, element_index[2] + 1)),
            ]
        elif element_type == CellType.BLUE_UP:
            return [
                DoFIndex((element_index[0] + 1, element_index[1]    , element_index[2])),
                DoFIndex((element_index[0]    , element_index[1] + 1, element_index[2])),
                DoFIndex((element_index[0] + 1, element_index[1] + 1, element_index[2])),
                DoFIndex((element_index[0] + 1, element_index[1]    , element_index[2] + 1)),
            ]
        elif element_type == CellType.BLUE_DOWN:
            return [
                DoFIndex((element_index[0]    , element_index[1] + 1, element_index[2])),
                DoFIndex((element_index[0]    , element_index[1]    , element_index[2] + 1)),
                DoFIndex((element_index[0] + 1, element_index[1]    , element_index[2] + 1)),
                DoFIndex((element_index[0]    , element_index[1] + 1, element_index[2] + 1)),
            ]
        elif element_type == CellType.GREEN_UP:
            return [
                DoFIndex((element_index[0] + 1, element_index[1]    , element_index[2])),
                DoFIndex((element_index[0]    , element_index[1] + 1, element_index[2])),
                DoFIndex((element_index[0]    , element_index[1]    , element_index[2] + 1)),
                DoFIndex((element_index[0] + 1, element_index[1]    , element_index[2] + 1)),
            ]
        elif element_type == CellType.GREEN_DOWN:
            return [
                DoFIndex((element_index[0]    , element_index[1] + 1, element_index[2])),
                DoFIndex((element_index[0] + 1, element_index[1] + 1, element_index[2])),
                DoFIndex((element_index[0] + 1, element_index[1]    , element_index[2] + 1)),
                DoFIndex((element_index[0]    , element_index[1] + 1, element_index[2] + 1)),
            ]
        else:
            raise HOGException(f"Invalid cell type {element_type} for 3D.")
        # fmt: on
    else:
        raise HOGException("Indexing function not implemented for this geometry.")


def micro_element_to_volume_indices(
    primitive_type: Union[FaceType, CellType],
    primitive_index: Tuple[int, int, int],
    n_dofs_per_primitive: int,
    memory_layout: VolumeDoFMemoryLayout,
) -> List[DoFIndex]:
    """
    Returns the DoFIndices for all intra-primitive indices of the provided volume.
    """
    return [
        DoFIndex(
            primitive_index,
            DoFType.VOLUME,
            primitive_type,
            n_dofs_per_primitive,
            pidx,
            memory_layout,
        )
        for pidx in range(0, n_dofs_per_primitive)
    ]


def micro_vertex_to_edge_indices(
    geometry: ElementGeometry, vertex_indices: List[DoFIndex]
) -> List[DoFIndex]:
    """
    Returns a collection of correctly oriented edges in form of DoFIndices from a given set of vertex DoFIndices.
    """

    edge_indices: List[DoFIndex] = []

    edges: List[Tuple[int, int]]
    if isinstance(geometry, TriangleElement):
        edges = [(1, 2), (0, 2), (0, 1)]
    elif isinstance(geometry, TetrahedronElement):
        edges = [(2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)]
    else:
        raise HOGException("Indexing function not implemented for this geometry.")

    for vertex_start, vertex_end in edges:
        vertex_idx_start: Tuple[int, int, int]
        vertex_idx_end: Tuple[int, int, int]
        if isinstance(geometry, TriangleElement):
            vertex_idx_start = (
                vertex_indices[vertex_start][0],
                vertex_indices[vertex_start][1],
                0,
            )
            vertex_idx_end = (
                vertex_indices[vertex_end][0],
                vertex_indices[vertex_end][1],
                0,
            )
        elif isinstance(geometry, TetrahedronElement):
            vertex_idx_start = vertex_indices[vertex_start].primitive_index
            vertex_idx_end = vertex_indices[vertex_end].primitive_index

        # determine orientation of the edge between the two vertices
        edge_orientation = calc_edge_orientation(vertex_idx_start, vertex_idx_end)

        # get the edge dof index of the edge between the two vertices, append with the corresponding edge orientation
        edge_dof_index = DoFIndex(
            get_edge_dof_index(vertex_idx_start, vertex_idx_end, edge_orientation),
            DoFType.EDGE,
            edge_orientation,
        )
        edge_indices.append(edge_dof_index)

    return edge_indices


def array_access_from_dof_index():
    pass


def element_vertex_coordinates(
    geometry: ElementGeometry,
    element_index: Tuple[int, int, int],
    element_type: Union[FaceType, CellType],
    micro_edges_per_macro_edge: sp.Symbol,
    macro_vertex_coordinates: List[sp.Matrix],
    symbolizer: Symbolizer,
) -> List[sp.Matrix]:
    """
    Computes the coordinates of the vertices of the current micro element.

    :param geometry: the geometry of the micro/macro element
    :param element_index: the logical index of the current element as a tuple (e.g. (idx_x, idx_y) in 2D)
    :param element_type: the type of the current element
    :param micro_edges_per_macro_edge: number of micro-edges along a macro-edge
    :param macro_vertex_coordinates: list of the macro-vertex coordinates
    :param symbolizer: a symbolizer instance
    """

    # First we compute the vertex dof indices of the current element.
    # This is equivalent to what is done in HyTeG
    vertex_indices = micro_element_to_vertex_indices(
        geometry, element_type, element_index
    )

    # Now we compute the coordinates of the vertex indices.
    # The next lines compute the increments
    h = 1.0 / micro_edges_per_macro_edge
    steps = [
        h * (macro_vertex_coordinates[i] - macro_vertex_coordinates[0])
        for i in range(1, geometry.num_vertices)
    ]

    # These lines add the increments up so that we arrive at the respective micro-vertices.
    # We have to do this for each micro-vertex of the current element.
    vertex_coordinates = []
    for vertex_index in vertex_indices:
        vertex_coordinate = macro_vertex_coordinates[0]
        for i in range(geometry.dimensions):
            vertex_coordinate += steps[i] * vertex_index.primitive_index[i]
        vertex_coordinates.append(vertex_coordinate)

    return vertex_coordinates


def calc_edge_orientation(
    vertex_idx_start: Tuple[int, int, int], vertex_idx_end: Tuple[int, int, int]
) -> EdgeType:
    """
    This function computes the edge type/edge orientation between two given vertex dof indices.
    """
    x = vertex_idx_end[0] - vertex_idx_start[0]
    y = vertex_idx_end[1] - vertex_idx_start[1]
    z = vertex_idx_end[2] - vertex_idx_start[2]

    if x == 1 and y == 0 and z == 0:
        return EdgeType.X
    if x == -1 and y == 0 and z == 0:
        return EdgeType.X

    if x == 0 and y == 1 and z == 0:
        return EdgeType.Y
    if x == 0 and y == -1 and z == 0:
        return EdgeType.Y

    if x == 0 and y == 0 and z == 1:
        return EdgeType.Z
    if x == 0 and y == 0 and z == -1:
        return EdgeType.Z

    if x == -1 and y == 1 and z == 0:
        return EdgeType.XY
    if x == 1 and y == -1 and z == 0:
        return EdgeType.XY

    if x == 1 and y == 0 and z == -1:
        return EdgeType.XZ
    if x == -1 and y == 0 and z == 1:
        return EdgeType.XZ

    if x == 0 and y == 1 and z == -1:
        return EdgeType.YZ
    if x == 0 and y == -1 and z == 1:
        return EdgeType.YZ

    if x == 1 and y == -1 and z == 1:
        return EdgeType.XYZ
    if x == -1 and y == 1 and z == -1:
        return EdgeType.XYZ

    raise HOGException("Invalid index offset.")


def get_edge_dof_index(
    vertex_idx_start: Tuple[int, int, int],
    vertex_idx_end: Tuple[int, int, int],
    orientation: EdgeType,
) -> Tuple[int, int, int]:
    """
    This function maps the edge between the vertex dof indices vertex_idx_start and
    vertex_idx_end to the corresponding edge dof index.
    """
    if orientation == EdgeType.X:
        return (
            vertex_idx_start
            if vertex_idx_start[0] < vertex_idx_end[0]
            else vertex_idx_end
        )
    elif orientation == EdgeType.Y:
        return (
            vertex_idx_start
            if vertex_idx_start[1] < vertex_idx_end[1]
            else vertex_idx_end
        )
    elif orientation == EdgeType.Z:
        return (
            vertex_idx_start
            if vertex_idx_start[2] < vertex_idx_end[2]
            else vertex_idx_end
        )
    elif orientation == EdgeType.XY:
        return tuple(
            map(
                operator.add,
                (
                    vertex_idx_start
                    if vertex_idx_start[0] < vertex_idx_end[0]
                    else vertex_idx_end
                ),
                (0, -1, 0),
            )
        )  # type: ignore
    elif orientation == EdgeType.XZ:
        return tuple(
            map(
                operator.add,
                (
                    vertex_idx_start
                    if vertex_idx_start[0] < vertex_idx_end[0]
                    else vertex_idx_end
                ),
                (0, 0, -1),
            )
        )  # type: ignore
    elif orientation == EdgeType.YZ:
        return tuple(
            map(
                operator.add,
                (
                    vertex_idx_start
                    if vertex_idx_start[1] < vertex_idx_end[1]
                    else vertex_idx_end
                ),
                (0, 0, -1),
            )
        )  # type: ignore
    elif orientation == EdgeType.XYZ:
        return tuple(
            map(
                operator.add,
                (
                    vertex_idx_start
                    if vertex_idx_start[0] < vertex_idx_end[0]
                    else vertex_idx_end
                ),
                (0, -1, 0),
            )
        )  # type: ignore
    else:
        raise HOGException("Unexpected orientation.")
