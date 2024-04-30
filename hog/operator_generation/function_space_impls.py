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

from abc import ABC, abstractmethod

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
    micro_element_to_vertex_indices,
    micro_vertex_to_edge_indices,
    IndexingInfo,
)
from hog.operator_generation.types import HOGType


class FunctionSpaceImpl(ABC):
    """A FunctionSpaceImpl is the counterpart of a Function in HyTeG.

    An instance of this class represents an instance of one of HyTeG's Function
    classes. It is associated with a mathematical function space, an identifier
    (variable name) and can optionally be a pointer. This class is intended
    to abstract printing code for e.g. communication in an FE space and C++
    implementation agnostic way.

    It is impossible to create an instance of this abstract base class directly.
    Preferrably, use the static method `create_impl` which selects the correct
    derived class for the `FunctionSpace`.
    """

    def __init__(
        self,
        fe_space: FunctionSpace,
        name: str,
        type_descriptor: HOGType,
        is_pointer: bool,
    ):
        """Records the passed parameters in member variables.

        It is impossible to create an instance of this abstract base class
        directly. This __init__ method is to be called by the derived classes.
        Preferrably, use the static method `create_impl` which selects the
        correct derived class for the `FunctionSpace`.
        """
        self.fe_space = fe_space
        self.name = name
        self.type_descriptor = type_descriptor
        self.is_pointer = is_pointer

    @staticmethod
    def create_impl(  # type: ignore[no-untyped-def] # returns Self but Self is kind of new
        func_space: FunctionSpace,
        name: str,
        type_descriptor: HOGType,
        is_pointer: bool = False,
    ):
        """Takes a mathematical function space and produces the corresponding function space implementation.

        :param func_space:      The mathematical function space.
        :param name:            The C++ variable name/identifier of this function.
        :param type_descriptor: The value type of this function.
        :param is_pointer:      Whether the C++ variable is of a pointer type. Used
                                to print member accesses.
        """
        impl_class: type

        if isinstance(func_space, LagrangianFunctionSpace):
            if func_space.degree == 2:
                impl_class = P2FunctionSpaceImpl
            elif func_space.degree == 1:
                impl_class = P1FunctionSpaceImpl
            else:
                raise HOGException("Lagrangian function space must be of order 1 or 2.")
        elif isinstance(func_space, TensorialVectorFunctionSpace):
            if isinstance(func_space.component_function_space, LagrangianFunctionSpace):
                if func_space.component_function_space.degree == 1:
                    impl_class = P1VectorFunctionSpaceImpl
                else:
                    raise HOGException(
                        "TensorialVectorFunctionSpaces are only supported for P1 components."
                    )
            else:
                raise HOGException(
                    "TensorialVectorFunctionSpaces are only supported with Lagrangian component spaces."
                )
        elif isinstance(func_space, N1E1Space):
            impl_class = N1E1FunctionSpaceImpl
        else:
            raise HOGException("Unexpected function space")

        return impl_class(func_space, name, type_descriptor, is_pointer)

    def _create_field(self, name: str) -> Field:
        """Creates a pystencils field with a given name and stride 1."""
        f = Field.create_generic(
            name,
            1,
            dtype=self.type_descriptor.pystencils_type,
            field_type=FieldType.CUSTOM,
        )
        f.strides = tuple([1 for _ in f.strides])
        return f

    @abstractmethod
    def pre_communication(self, dim: int) -> str:
        """C++ code for the function space communication prior to the macro-primitive loop."""
        ...

    @abstractmethod
    def zero_halos(self, dim: int) -> str:
        """C++ code to zero the halos on a macro in HyTeG."""
        ...

    @abstractmethod
    def post_communication(
        self, dim: int, params: str, transform_basis: bool = True
    ) -> str:
        """C++ code for the function space communication after the macro-primitive loop."""
        ...

    @abstractmethod
    def pointer_retrieval(self, dim: int) -> str:
        """C++ code for retrieving pointers to the numerical data stored in the macro primitives `face` (2d) or `cell` (3d)."""
        ...

    @abstractmethod
    def invert_elementwise(self, dim: int) -> str:
        """C++ code for inverting each DoF of the linalg vector."""
        ...

    @abstractmethod
    def local_dofs(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        indexing_info: IndexingInfo,
    ) -> List[Field.Access]:
        """Returns a list of local dof values on the current element."""
        ...

    @abstractmethod
    def func_type_string(self) -> str:
        """Returns the C++ function class type as a string."""
        ...

    @abstractmethod
    def includes(self) -> Set[str]:
        """Returns the import location for the function space in HyTeG."""
        ...

    @abstractmethod
    def dof_transformation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        """Returns HyTeG code that computes the basis/DoF transformation and a symbolic expression of the result.

        In general FEM spaces, neighboring elements must agree on the
        orientation of shared mesh entities. For example, in N1E1 the
        orientations of edges define the sign of the DoFs, while in P3
        reorienting an edge swaps the two DoFs located on this edge. Basis
        transformations between these spaces allow us to view DoFs from
        neighboring elements in our local orientation.
        Recommended reading:
          Scroggs et al., "Construction of Arbitrary Order Finite Element Degree-
          of-Freedom Maps on Polygonal and Polyhedral Cell Meshes," 2022, doi:
          https://doi.org/10.1145/3524456.

        This function returns a matrix that, when applied to a local DoF vector,
        transforms the DoFs from the owning primitives to the basis of the
        current macro primitive.
        For example:
          - P1/P2: The identity.
          - P3:    A permutation matrix.
          - N1E1:  A diagonal matrix with |aᵢᵢ| = 1.

        If the micro element is in the interior of the macro-cell, the
        transformation is the identity. In matrix-free computations the
        communication is responsible for the basis transformations. Only when
        assembling operators into matrices, these transformations must be
        "baked into" the matrix because vectors are assembled locally and our
        communication routine is not performed during the operator application.
        """
        ...

    def _deref(self) -> str:
        if self.is_pointer:
            return f"(*{self.name})"
        else:
            return self.name


class P1FunctionSpaceImpl(FunctionSpaceImpl):
    def __init__(
        self,
        fe_space: FunctionSpace,
        name: str,
        type_descriptor: HOGType,
        is_pointer: bool = False,
    ):
        super().__init__(fe_space, name, type_descriptor, is_pointer)
        self.field = self._create_field(name)

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
                f"for ( const auto& idx : vertexdof::macroface::Iterator( level ) ) {{\n"
                f"    if ( vertexdof::macroface::isVertexOnBoundary( level, idx ) ) {{\n"
                f"        auto arrayIdx = vertexdof::macroface::index( level, idx.x(), idx.y() );\n"
                f"        _data_{self.name}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"    }}\n"
                f"}}"
            )
        else:
            return (
                f"for ( const auto& idx : vertexdof::macrocell::Iterator( level ) ) {{\n"
                f"    if ( !vertexdof::macrocell::isOnCellFace( idx, level ).empty() ) {{\n"
                f"        auto arrayIdx = vertexdof::macrocell::index( level, idx.x(), idx.y(), idx.z() );\n"
                f"        _data_{self.name}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"    }}\n"
                f"}}"
            )

    def post_communication(
        self, dim: int, params: str, transform_basis: bool = True
    ) -> str:
        if dim == 2:
            return (
                f"{self._deref()}.communicateAdditively < Face, Edge > ( {params} );\n"
                f"{self._deref()}.communicateAdditively < Face, Vertex > ( {params} );"
            )
        else:
            return (
                f"{self._deref()}.communicateAdditively< Cell, Face >( {params} );\n"
                f"{self._deref()}.communicateAdditively< Cell, Edge >( {params} );\n"
                f"{self._deref()}.communicateAdditively< Cell, Vertex >( {params} );"
            )

    def pointer_retrieval(self, dim: int) -> str:
        """C++ code for retrieving pointers to the numerical data stored in the macro primitives `face` (2d) or `cell` (3d)."""
        Macro = {2: "Face", 3: "Cell"}[dim]
        macro = {2: "face", 3: "cell"}[dim]

        return f"{self.type_descriptor.pystencils_type}* _data_{self.name} = {macro}.getData( {self._deref()}.get{Macro}DataID() )->getPointer( level );"

    def invert_elementwise(self, dim: int) -> str:
        return f"{self._deref()}.invertElementwise( level );"

    def local_dofs(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        indexing_info: IndexingInfo,
    ) -> List[Field.Access]:
        vertex_dof_indices = micro_element_to_vertex_indices(
            geometry, element_type, element_index
        )
        vertex_array_indices = [
            dof_idx.array_index(geometry, indexing_info)
            for dof_idx in vertex_dof_indices
        ]

        return [
            self.field.absolute_access((idx,), (0,)) for idx in vertex_array_indices
        ]

    def func_type_string(self) -> str:
        return f"P1Function< {self.type_descriptor.pystencils_type} >"

    def includes(self) -> Set[str]:
        return {f"hyteg/p1functionspace/P1Function.hpp"}

    def dof_transformation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        return (
            CustomCodeNode("", [], []),
            sp.Identity(self.fe_space.num_dofs(geometry)),
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
    ) -> List[Field.Access]:
        vertex_dof_indices = micro_element_to_vertex_indices(
            geometry, element_type, element_index
        )
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
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        return (
            CustomCodeNode("", [], []),
            sp.Identity(self.fe_space.num_dofs(geometry)),
        )


class P1VectorFunctionSpaceImpl(FunctionSpaceImpl):
    def __init__(
        self,
        fe_space: FunctionSpace,
        name: str,
        type_descriptor: HOGType,
        is_pointer: bool = False,
    ):
        super().__init__(fe_space, name, type_descriptor, is_pointer)
        self.fields: Dict = {}

    def _field_name(self, component: int) -> str:
        return self.name + f"_{component}"

    def _raw_pointer_name(self, component: int) -> str:
        return f"_data_" + self._field_name(component)

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
                f"for ( const auto& idx : vertexdof::macroface::Iterator( level ) ) {{\n"
                f"    if ( vertexdof::macroface::isVertexOnBoundary( level, idx ) ) {{\n"
                f"        auto arrayIdx = vertexdof::macroface::index( level, idx.x(), idx.y() );\n"
                f"        {self._raw_pointer_name(0)}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"        {self._raw_pointer_name(1)}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"    }}\n"
                f"}}"
            )
        else:
            return (
                f"for ( const auto& idx : vertexdof::macrocell::Iterator( level ) ) {{\n"
                f"    if ( !vertexdof::macrocell::isOnCellFace( idx, level ).empty() ) {{\n"
                f"        auto arrayIdx = vertexdof::macrocell::index( level, idx.x(), idx.y(), idx.z() );\n"
                f"        {self._raw_pointer_name(0)}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"        {self._raw_pointer_name(1)}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"        {self._raw_pointer_name(2)}[arrayIdx] = {self.type_descriptor.pystencils_type}( 0 );\n"
                f"    }}\n"
                f"}}"
            )

    def post_communication(
        self, dim: int, params: str, transform_basis: bool = True
    ) -> str:
        if dim == 2:
            ret_str = ""
            for i in range(dim):
                ret_str += (
                    f"{self._deref()}[{i}].communicateAdditively < Face, Edge > ( {params} );\n"
                    f"{self._deref()}[{i}].communicateAdditively < Face, Vertex > ( {params} );\n"
                )
            return ret_str
        else:
            ret_str = ""
            for i in range(dim):
                ret_str += (
                    f"{self._deref()}[{i}].communicateAdditively< Cell, Face >( {params} );\n"
                    f"{self._deref()}[{i}].communicateAdditively< Cell, Edge >( {params} );\n"
                    f"{self._deref()}[{i}].communicateAdditively< Cell, Vertex >( {params} );"
                )
            return ret_str

    def pointer_retrieval(self, dim: int) -> str:
        """C++ code for retrieving pointers to the numerical data stored in the macro primitives `face` (2d) or `cell` (3d)."""
        Macro = {2: "Face", 3: "Cell"}[dim]
        macro = {2: "face", 3: "cell"}[dim]

        ret_str = ""
        for i in range(dim):
            ret_str += f"{self.type_descriptor.pystencils_type}* {self._raw_pointer_name(i)} = {macro}.getData( {self._deref()}[{i}].get{Macro}DataID() )->getPointer( level );\n"
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
    ) -> List[Field.Access]:
        """
        Returns the element-local DoFs (field accesses) in a list (i.e., linearized).

        The order here has to match that of the function space implementation.
        TODO: This is a little concerning since that order has to match in two very different parts of this
              implementation. Here, and in the TensorialVectorFunctionSpace. Maybe we can define this order in a single
              location.
        We return them in "SoA" order: first all DoFs corresponding to the function with the first component != 0,
        then all DoFs corresponding to the function with the second component != 0 etc.

        For instance, in 2D we get the DoFs corresponding to the 6 shape functions in the following order:

        First component != 0

        list[0]: ⎡-x_ref_0 - x_ref_1 + 1⎤
                 ⎢                      ⎥
                 ⎣          0           ⎦

        list[1]: ⎡x_ref_0⎤
                 ⎢       ⎥
                 ⎣   0   ⎦

        list[2]: ⎡x_ref_1⎤
                 ⎢       ⎥
                 ⎣   0   ⎦

        Second component != 0

        list[3]: ⎡          0           ⎤
                 ⎢                      ⎥
                 ⎣-x_ref_0 - x_ref_1 + 1⎦

        list[4]: ⎡   0   ⎤
                 ⎢       ⎥
                 ⎣x_ref_0⎦

        list[5]: ⎡   0   ⎤
                 ⎢       ⎥
                 ⎣x_ref_1⎦
        """

        for c in range(geometry.dimensions):
            if c not in self.fields:
                self.fields[c] = self._create_field(self._field_name(c))

        vertex_dof_indices = micro_element_to_vertex_indices(
            geometry, element_type, element_index
        )
        vertex_array_indices = [
            dof_idx.array_index(geometry, indexing_info)
            for dof_idx in vertex_dof_indices
        ]

        return [
            self.fields[c].absolute_access((idx,), (0,))
            for c in range(geometry.dimensions)
            for idx in vertex_array_indices
        ]

    def func_type_string(self) -> str:
        return f"P1VectorFunction< {self.type_descriptor.pystencils_type} >"

    def includes(self) -> Set[str]:
        return {f"hyteg/p1functionspace/P1VectorFunction.hpp"}

    def dof_transformation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        return (
            CustomCodeNode("", [], []),
            sp.Identity(self.fe_space.num_dofs(geometry)),
        )


class N1E1FunctionSpaceImpl(FunctionSpaceImpl):
    def __init__(
        self,
        fe_space: FunctionSpace,
        name: str,
        type_descriptor: HOGType,
        is_pointer: bool = False,
    ):
        super().__init__(fe_space, name, type_descriptor, is_pointer)
        self.field = self._create_field(name)

    def pre_communication(self, dim: int) -> str:
        if dim == 2:
            return ""
        else:
            return (
                f"{self._deref()}.communicate< Face, Cell >( level );\n"
                f"{self._deref()}.communicate< Edge, Cell >( level );"
            )

    def zero_halos(self, dim: int) -> str:
        if dim == 2:
            return ""
        else:
            return f"edgedof::macrocell::setBoundaryToZero( level, cell, {self._deref()}.getDoFs()->getCellDataID() );"

    def post_communication(
        self, dim: int, params: str, transform_basis: bool = True
    ) -> str:
        if dim == 2:
            return ""
        else:
            dofs = ""
            if not transform_basis:
                dofs = "getDoFs()->"
            return (
                f"{self._deref()}.{dofs}communicateAdditively< Cell, Face >( {params} );\n"
                f"{self._deref()}.{dofs}communicateAdditively< Cell, Edge >( {params} );"
            )

    def pointer_retrieval(self, dim: int) -> str:
        """C++ code for retrieving pointers to the numerical data stored in the macro primitives `face` (2d) or `cell` (3d)."""
        Macro = {2: "Face", 3: "Cell"}[dim]
        macro = {2: "face", 3: "cell"}[dim]

        return f"{self.type_descriptor.pystencils_type}* _data_{self.name} = {macro}.getData( {self._deref()}.getDoFs()->get{Macro}DataID() )->getPointer( level );"

    def invert_elementwise(self, dim: int) -> str:
        return f"{self._deref()}.getDoFs()->invertElementwise( level );"

    def local_dofs(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        indexing_info: IndexingInfo,
    ) -> List[Field.Access]:
        vertex_dof_indices = micro_element_to_vertex_indices(
            geometry, element_type, element_index
        )
        edge_dof_indices = micro_vertex_to_edge_indices(geometry, vertex_dof_indices)
        edge_array_indices = [
            dof_idx.array_index(geometry, indexing_info) for dof_idx in edge_dof_indices
        ]

        return [self.field.absolute_access((idx,), (0,)) for idx in edge_array_indices]

    def func_type_string(self) -> str:
        return f"n1e1::N1E1VectorFunction< {self.type_descriptor.pystencils_type} >"

    def includes(self) -> Set[str]:
        return {
            "hyteg/n1e1functionspace/N1E1VectorFunction.hpp",
            "hyteg/n1e1functionspace/N1E1MacroCell.hpp",
        }

    def dof_transformation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        Macro = {2: "Face", 3: "Cell"}[geometry.dimensions]
        macro = {2: "face", 3: "cell"}[geometry.dimensions]
        name = "basisTransformation"
        n_dofs = self.fe_space.num_dofs(geometry)

        # NOTE: Types are added manually because pystencils won't add types to accessed/
        #       defined symbols of custom code nodes. As a result the symbols
        #       in the matrix will be typed but not their definition, leading to
        #       undefined symbols.
        # NOTE: The type is `real_t` (not `self._type_descriptor.pystencils_type`)
        #       because this function is implemented manually in HyTeG with
        #       this signature. Passing `np.float64` is not ideal (if `real_t !=
        #       double`) but it makes sure that casts are inserted if necessary
        #       (though some might be superfluous).
        symbols = [
            ps.TypedSymbol(f"{name}.diagonal()({i})", np.float64) for i in range(n_dofs)
        ]
        return (
            CustomCodeNode(
                f"const Eigen::DiagonalMatrix< real_t, {n_dofs} > {name} = "
                f"n1e1::macro{macro}::basisTransformation( level, {macro}, {{{element_index[0]}, {element_index[1]}, {element_index[2]}}}, {macro}dof::{Macro}Type::{element_type.name} );",
                [
                    ps.TypedSymbol("level", "const uint_t"),
                    ps.TypedSymbol(f"{macro}", f"const {Macro}&"),
                ],
                symbols,
            ),
            sp.diag(*symbols),
        )
