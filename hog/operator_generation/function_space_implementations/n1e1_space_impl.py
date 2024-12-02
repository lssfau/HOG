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
        element_vertex_ordering: List[int],
    ) -> List[Field.Access]:
        vertex_dof_indices = micro_element_to_vertex_indices(
            geometry, element_type, element_index
        )

        vertex_dof_indices = [vertex_dof_indices[i] for i in element_vertex_ordering]

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
        element_vertex_ordering: List[int],
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        if element_vertex_ordering != [0, 1, 2, 3]:
            raise HOGException(
                "Element vertex re-ordering not supported for Nédélec elements (yet)."
            )

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
