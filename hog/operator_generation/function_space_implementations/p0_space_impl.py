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


class P0FunctionSpaceImpl(FunctionSpaceImpl):
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
        return ""

    def zero_halos(self, dim: int) -> str:
        return ""

    def post_communication(
        self, dim: int, params: str, transform_basis: bool = True
    ) -> str:
        return ""

    def pointer_retrieval(self, dim: int) -> str:
        """C++ code for retrieving pointers to the numerical data stored in the macro primitives `face` (2d) or `cell` (3d)."""
        Macro = {2: "Face", 3: "Cell"}[dim]
        macro = {2: "face", 3: "cell"}[dim]

        return f"{self.type_descriptor.pystencils_type}* _data_{self.name} = {self._deref()}.getDGFunction()->volumeDoFFunction()->dofMemory( it.first, level );"

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
        volume_dof_indices = micro_element_to_volume_indices(
            element_type, element_index, 1, VolumeDoFMemoryLayout.SoA
        )

        vertex_array_indices = [
            dof_idx.array_index(geometry, indexing_info)
            for dof_idx in volume_dof_indices
        ]

        return [
            self.field.absolute_access((idx,), (0,)) for idx in vertex_array_indices
        ]

    def func_type_string(self) -> str:
        return f"P0Function< {self.type_descriptor.pystencils_type} >"

    def includes(self) -> Set[str]:
        return {f"hyteg/p0functionspace/P0Function.hpp"}

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
