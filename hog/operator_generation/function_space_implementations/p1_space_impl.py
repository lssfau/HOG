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

import hog.operator_generation.function_space_impls


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
        element_vertex_ordering: List[int],
    ) -> List[Field.Access]:
        vertex_dof_indices = micro_element_to_vertex_indices(
            geometry, element_type, element_index
        )

        vertex_dof_indices = [vertex_dof_indices[i] for i in element_vertex_ordering]

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
        element_vertex_ordering: List[int],
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
        self.fields: Dict[int, Field] = {}

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
        element_vertex_ordering: List[int],
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

        vertex_dof_indices = [vertex_dof_indices[i] for i in element_vertex_ordering]

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
        element_vertex_ordering: List[int],
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        return (
            CustomCodeNode("", [], []),
            sp.Identity(self.fe_space.num_dofs(geometry)),
        )
