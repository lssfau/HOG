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

import itertools
import numpy as np
import sympy as sp
from typing import List, Tuple, Set, Union, Dict

import pystencils as ps
from pystencils import Field, FieldType
from pystencils.backends.cbackend import CustomCodeNode

from hog.element_geometry import ElementGeometry
from hog.exception import HOGException
from hog.function_space import (
    FunctionSpace,
    LagrangianFunctionSpace,
    TensorialVectorFunctionSpace,
    N1E1Space,
)
from hog.operator_generation.indexing import (
    CellType,
    FaceType,
    VolumeDoFMemoryLayout,
    micro_element_to_vertex_indices,
    micro_vertex_to_edge_indices,
    micro_element_to_volume_indices,
    IndexingInfo,
)
from hog.operator_generation.types import HOGType
from hog.operator_generation.function_space_implementations.function_space_impl_base import (
    FunctionSpaceImpl,
)


class P2FunctionSpaceImpl(FunctionSpaceImpl):
    def __init__(
        self,
        fe_space: FunctionSpace,
        name: str,
        type_descriptor: HOGType,
        is_pointer: bool = False,
    ):
        super().__init__(fe_space, name, type_descriptor, is_pointer)
        self.f_vertex = self._create_field(name + "Vertex")
        self.f_edge = self._create_field(name + "Edge")

    def pre_communication(self, dim: int) -> str:
        if dim == 2:
            return f"communication::syncFunctionBetweenPrimitives( {self.name}, level, communication::syncDirection_t::LOW2HIGH );"
        else:
            return (
                f"{self._deref()}.communicate< Face, Cell >( level );\n"
                f"{self._deref()}.communicate< Edge, Cell >( level );\n"
                f"{self._deref()}.communicate< Vertex, Cell >( level );"
            )

    def zero_halos(self, dim: int) -> str:
        if dim == 2:
            return (
                f"for ( const auto& idx : vertexdof::macroface::Iterator( level ) )\n"
                f"{{\n"
                f"    if ( vertexdof::macroface::isVertexOnBoundary( level, idx ) )\n"
                f"    {{\n"
                f"        auto arrayIdx = vertexdof::macroface::index( level, idx.x(), idx.y() );\n"
                f"        _data_{self.name }Vertex[arrayIdx] = walberla::numeric_cast< {self.type_descriptor.pystencils_type} >( 0 );\n"
                f"    }}\n"
                f"}}\n"
                f"for ( const auto& idx : edgedof::macroface::Iterator( level ) )\n"
                f"{{\n"
                f"    for ( const auto& orientation : edgedof::faceLocalEdgeDoFOrientations )\n"
                f"    {{\n"
                f"        if ( !edgedof::macroface::isInnerEdgeDoF( level, idx, orientation ) )\n"
                f"        {{\n"
                f"            auto arrayIdx = edgedof::macroface::index( level, idx.x(), idx.y(), orientation );\n"
                f"            _data_{self.name }Edge[arrayIdx] = walberla::numeric_cast< {self.type_descriptor.pystencils_type} >( 0 );\n"
                f"        }}\n"
                f"    }}\n"
                f"}}"
            )
        else:
            return (
                f"for ( const auto& idx : vertexdof::macrocell::Iterator( level ) )\n"
                f"{{\n"
                f"if ( !vertexdof::macrocell::isOnCellFace( idx, level ).empty() )\n"
                f"    {{\n"
                f"        auto arrayIdx = vertexdof::macrocell::index( level, idx.x(), idx.y(), idx.z() );\n"
                f"        _data_{self.name}Vertex[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"    }}\n"
                f"}}\n"
                f"edgedof::macrocell::setBoundaryToZero( level, cell, {self._deref()}.getEdgeDoFFunction().getCellDataID() );"
            )

    def post_communication(
        self, dim: int, params: str, transform_basis: bool = True
    ) -> str:
        if dim == 2:
            return (
                f"{self._deref()}.getVertexDoFFunction().communicateAdditively< Face, Edge >( {params} );\n"
                f"{self._deref()}.getVertexDoFFunction().communicateAdditively< Face, Vertex >( {params} );\n"
                f"{self._deref()}.getEdgeDoFFunction().communicateAdditively< Face, Edge >( {params} );"
            )
        else:
            return (
                f"{self._deref()}.getVertexDoFFunction().communicateAdditively< Cell, Face >( {params} );\n"
                f"{self._deref()}.getVertexDoFFunction().communicateAdditively< Cell, Edge >( {params} );\n"
                f"{self._deref()}.getVertexDoFFunction().communicateAdditively< Cell, Vertex >( {params} );\n"
                f"{self._deref()}.getEdgeDoFFunction().communicateAdditively< Cell, Face >( {params} );\n"
                f"{self._deref()}.getEdgeDoFFunction().communicateAdditively< Cell, Edge >( {params} );"
            )

    def pointer_retrieval(self, dim: int) -> str:
        """C++ code for retrieving pointers to the numerical data stored in the macro primitives `face` (2d) or `cell` (3d)."""
        Macro = {2: "Face", 3: "Cell"}[dim]
        macro = {2: "face", 3: "cell"}[dim]

        return (
            f"{self.type_descriptor.pystencils_type}* _data_{self.name}Vertex = {macro}.getData( {self._deref()}.getVertexDoFFunction().get{Macro}DataID() )->getPointer( level );\n"
            f"{self.type_descriptor.pystencils_type}* _data_{self.name}Edge   = {macro}.getData( {self._deref()}.getEdgeDoFFunction().get{Macro}DataID() )->getPointer( level );"
        )

    def invert_elementwise(self, dim: int) -> str:
        return f"{self._deref()}.invertElementwise( level );"

    def local_dofs(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        indexing_info: IndexingInfo,
        element_vertex_ordering: List[int],
    ) -> List[Field.Access]:
        vertex_dof_indices = micro_element_to_vertex_indices(
            geometry, element_type, element_index
        )

        vertex_dof_indices = [vertex_dof_indices[i] for i in element_vertex_ordering]

        edge_dof_indices = micro_vertex_to_edge_indices(geometry, vertex_dof_indices)

        vrtx_array_idcs = [
            dof_idx.array_index(geometry, indexing_info)
            for dof_idx in vertex_dof_indices
        ]
        edge_array_idcs = [
            dof_idx.array_index(geometry, indexing_info) for dof_idx in edge_dof_indices
        ]

        # NOTE: The order of DoFs must match the order of shape functions
        #       defined in the FunctionSpace. Holds within the individual lists
        #       and overall.
        return [
            self.f_vertex.absolute_access((idx,), (0,)) for idx in vrtx_array_idcs
        ] + [self.f_edge.absolute_access((idx,), (0,)) for idx in edge_array_idcs]

    def func_type_string(self) -> str:
        return f"P2Function< {self.type_descriptor.pystencils_type} >"

    def includes(self) -> Set[str]:
        return {f"hyteg/p2functionspace/P2Function.hpp"}

    def dof_transformation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        element_vertex_ordering: List[int],
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        return (
            CustomCodeNode("", [], []),
            sp.Identity(self.fe_space.num_dofs(geometry)),
        )


class P2VectorFunctionSpaceImpl(FunctionSpaceImpl):
    def __init__(
        self,
        fe_space: FunctionSpace,
        name: str,
        type_descriptor: HOGType,
        is_pointer: bool = False,
    ):
        super().__init__(fe_space, name, type_descriptor, is_pointer)
        self.fields: Dict[Tuple[str, int], Field] = {}

    def _field_name(self, component: int, dof_type: str) -> str:
        """dof_type should either be 'vertex' or 'edge'"""
        return self.name + f"_{dof_type}_{component}"

    def _raw_pointer_name(self, component: int, dof_type: str) -> str:
        """dof_type should either be 'vertex' or 'edge'"""
        return f"_data_" + self._field_name(component, dof_type)

    def pre_communication(self, dim: int) -> str:
        if dim == 2:
            return f"communication::syncVectorFunctionBetweenPrimitives( {self.name}, level, communication::syncDirection_t::LOW2HIGH );"
        else:
            ret_str = ""
            for i in range(dim):
                ret_str += (
                    f"{self._deref()}[{i}].communicate< Face, Cell >( level );\n"
                    f"{self._deref()}[{i}].communicate< Edge, Cell >( level );\n"
                    f"{self._deref()}[{i}].communicate< Vertex, Cell >( level );"
                )
            return ret_str

    def zero_halos(self, dim: int) -> str:
        if dim == 2:
            return (
                f"for ( const auto& idx : vertexdof::macroface::Iterator( level ) )\n"
                f"{{\n"
                f"    if ( vertexdof::macroface::isVertexOnBoundary( level, idx ) )\n"
                f"    {{\n"
                f"        auto arrayIdx = vertexdof::macroface::index( level, idx.x(), idx.y() );\n"
                f"        {self._raw_pointer_name(0, 'vertex')}[arrayIdx] = walberla::numeric_cast< {self.type_descriptor.pystencils_type} >( 0 );\n"
                f"        {self._raw_pointer_name(1, 'vertex')}[arrayIdx] = walberla::numeric_cast< {self.type_descriptor.pystencils_type} >( 0 );\n"
                f"    }}\n"
                f"}}\n"
                f"for ( const auto& idx : edgedof::macroface::Iterator( level ) )\n"
                f"{{\n"
                f"    for ( const auto& orientation : edgedof::faceLocalEdgeDoFOrientations )\n"
                f"    {{\n"
                f"        if ( !edgedof::macroface::isInnerEdgeDoF( level, idx, orientation ) )\n"
                f"        {{\n"
                f"            auto arrayIdx = edgedof::macroface::index( level, idx.x(), idx.y(), orientation );\n"
                f"            {self._raw_pointer_name(0, 'edge')}[arrayIdx] = walberla::numeric_cast< {self.type_descriptor.pystencils_type} >( 0 );\n"
                f"            {self._raw_pointer_name(1, 'edge')}[arrayIdx] = walberla::numeric_cast< {self.type_descriptor.pystencils_type} >( 0 );\n"
                f"        }}\n"
                f"    }}\n"
                f"}}"
            )
        else:
            return (
                f"for ( const auto& idx : vertexdof::macrocell::Iterator( level ) )\n"
                f"{{\n"
                f"if ( !vertexdof::macrocell::isOnCellFace( idx, level ).empty() )\n"
                f"    {{\n"
                f"        auto arrayIdx = vertexdof::macrocell::index( level, idx.x(), idx.y(), idx.z() );\n"
                f"        {self._raw_pointer_name(0, 'vertex')}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"        {self._raw_pointer_name(1, 'vertex')}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"        {self._raw_pointer_name(2, 'vertex')}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"    }}\n"
                f"}}\n"
                f"edgedof::macrocell::setBoundaryToZero( level, cell, {self._deref()}[0].getEdgeDoFFunction().getCellDataID() );\n"
                f"edgedof::macrocell::setBoundaryToZero( level, cell, {self._deref()}[1].getEdgeDoFFunction().getCellDataID() );\n"
                f"edgedof::macrocell::setBoundaryToZero( level, cell, {self._deref()}[2].getEdgeDoFFunction().getCellDataID() );\n"
            )

    def post_communication(
        self, dim: int, params: str, transform_basis: bool = True
    ) -> str:
        if dim == 2:
            ret_str = ""
            for i in range(dim):
                ret_str += (
                    f"{self._deref()}[{i}].getVertexDoFFunction().communicateAdditively< Face, Edge >( {params} );\n"
                    f"{self._deref()}[{i}].getVertexDoFFunction().communicateAdditively< Face, Vertex >( {params} );\n"
                    f"{self._deref()}[{i}].getEdgeDoFFunction().communicateAdditively< Face, Edge >( {params} );"
                )
            return ret_str
        else:
            ret_str = ""
            for i in range(dim):
                ret_str += (
                    f"{self._deref()}[{i}].getVertexDoFFunction().communicateAdditively< Cell, Face >( {params} );\n"
                    f"{self._deref()}[{i}].getVertexDoFFunction().communicateAdditively< Cell, Edge >( {params} );\n"
                    f"{self._deref()}[{i}].getVertexDoFFunction().communicateAdditively< Cell, Vertex >( {params} );\n"
                    f"{self._deref()}[{i}].getEdgeDoFFunction().communicateAdditively< Cell, Face >( {params} );\n"
                    f"{self._deref()}[{i}].getEdgeDoFFunction().communicateAdditively< Cell, Edge >( {params} );"
                )
            return ret_str

    def pointer_retrieval(self, dim: int) -> str:
        """C++ code for retrieving pointers to the numerical data stored in the macro primitives `face` (2d) or `cell` (3d)."""
        Macro = {2: "Face", 3: "Cell"}[dim]
        macro = {2: "face", 3: "cell"}[dim]

        ret_str = ""
        for i in range(dim):
            ret_str += f"{self.type_descriptor.pystencils_type}* {self._raw_pointer_name(i, 'vertex')} = {macro}.getData( {self._deref()}[{i}].getVertexDoFFunction().get{Macro}DataID() )->getPointer( level );\n"
            ret_str += f"{self.type_descriptor.pystencils_type}* {self._raw_pointer_name(i, 'edge')} = {macro}.getData( {self._deref()}[{i}].getEdgeDoFFunction().get{Macro}DataID() )->getPointer( level );\n"
        return ret_str

    def invert_elementwise(self, dim: int) -> str:
        ret_str = ""
        for i in range(dim):
            ret_str += f"{self._deref()}[{i}].invertElementwise( level );\n"
        return ret_str

    def local_dofs(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        indexing_info: IndexingInfo,
        element_vertex_ordering: List[int],
    ) -> List[Field.Access]:
        """
        Returns the element-local DoFs (field accesses) in a list (i.e., linearized).

        See P1VectorFunctionSpaceImpl::local_dofs() for details.
        """

        for dt in ["vertex", "edge"]:
            for c in range(geometry.dimensions):
                if (dt, c) not in self.fields:
                    self.fields[(dt, c)] = self._create_field(self._field_name(c, dt))

        vertex_dof_indices = micro_element_to_vertex_indices(
            geometry, element_type, element_index
        )

        vertex_dof_indices = [vertex_dof_indices[i] for i in element_vertex_ordering]

        edge_dof_indices = micro_vertex_to_edge_indices(geometry, vertex_dof_indices)

        vertex_array_indices = [
            dof_idx.array_index(geometry, indexing_info)
            for dof_idx in vertex_dof_indices
        ]

        edge_array_indices = [
            dof_idx.array_index(geometry, indexing_info) for dof_idx in edge_dof_indices
        ]

        loc_dofs = []

        for c in range(geometry.dimensions):
            loc_dofs += [
                self.fields[("vertex", c)].absolute_access((idx,), (0,))
                for idx in vertex_array_indices
            ]
            loc_dofs += [
                self.fields[("edge", c)].absolute_access((idx,), (0,))
                for idx in edge_array_indices
            ]

        return loc_dofs

    def func_type_string(self) -> str:
        return f"P2VectorFunction< {self.type_descriptor.pystencils_type} >"

    def includes(self) -> Set[str]:
        return {
            f"hyteg/p2functionspace/P2VectorFunction.hpp",
            f"hyteg/p2functionspace/P2Function.hpp",
        }

    def dof_transformation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        element_vertex_ordering: List[int],
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        return (
            CustomCodeNode("", [], []),
            sp.Identity(self.fe_space.num_dofs(geometry)),
        )
