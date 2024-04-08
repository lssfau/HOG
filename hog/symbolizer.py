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

from typing import List
import sympy as sp
import string
from hog.exception import HOGException


class Symbolizer:
    """Helper class to manage the names of sympy variables."""

    def __init__(
        self,
        symbol_ref_coords=lambda coord: f"x_ref_{coord}",
        symbol_affine_vertices=lambda point, coord, suffix: f"p_affine{suffix}_{point}_{coord}",
        input_affine_vertices_name="coords",
        input_affine_vertices_access=lambda name, point, coord: f"{name}[{point}][{coord}]",
        symbol_element_matrix=lambda row, col: f"a_{row}_{col}",
        output_element_matrix_name="elMat",
        output_element_matrix_access=lambda name, row, col: f"{name}({row},{col})",
        symbol_dof=lambda idx: f"dof_{idx}",
        dof_name="dofs",
        dof_access=lambda name, idx: f"{name}[{idx}]",
        symbol_affine_eval_coord=lambda coord: f"x_affine_eval_{coord}",
        affine_eval_coord_name=lambda coord: f"x_affine_eval",
        affine_eval_coord_access=lambda name, coord: f"{name}[{coord}]",
        tmp_prefix="tmp_",
        symbol_jac_affine="jac_affine",
        symbol_jac_affine_inv="jac_affine_inv",
        symbol_abs_det_jac_affine="abs_det_jac_affine",
        symbol_blending_parameter_prefix="blending_param_",
        float_loop_ctr_array_prefix="float_loop_ctr_array_dim_",
    ):
        """Creates a Symbolizer instance.

        Allows specifying lambdas that define the variable names.

        :param symbol_ref_coords: lambda for the names of the coordinates on the reference element
        :param symbol_affine_vertices: lambda for the names of the entries of the affine element vertices
        :param symbol_element_matrix: lambda for the names of the element matrix entries
        :param tmp_prefix: string that is used as a prefix for temporary variables
        """
        self._symbol_ref_coords = symbol_ref_coords
        self._symbol_affine_vertices = symbol_affine_vertices
        self._input_affine_vertices_name = input_affine_vertices_name
        self._input_affine_vertices_access = input_affine_vertices_access
        self._symbol_element_matrix = symbol_element_matrix
        self._output_element_matrix_name = output_element_matrix_name
        self._output_element_matrix_access = output_element_matrix_access
        self._symbol_dof = symbol_dof
        self._dof_name = dof_name
        self._dof_access = dof_access
        self._symbol_affine_eval_coord = symbol_affine_eval_coord
        self._affine_eval_coord_name = affine_eval_coord_name
        self._affine_eval_coord_access = affine_eval_coord_access
        self._tmp_prefix = tmp_prefix
        self._symbol_jac_affine = symbol_jac_affine
        self._symbol_jac_affine_inv = symbol_jac_affine_inv
        self._symbol_abs_det_jac_affine = symbol_abs_det_jac_affine
        self._symbol_blending_parameter_prefix = symbol_blending_parameter_prefix
        self._float_loop_ctr_array_prefix = float_loop_ctr_array_prefix

    def ref_coords_as_list(self, dimensions: int) -> List[sp.Symbol]:
        """Returns a list of symbols that correspond to the coordinates on the reference element."""
        return sp.symbols([self._symbol_ref_coords(i) for i in range(dimensions)])

    def ref_coords_as_vector(self, dimensions: int) -> sp.Matrix:
        """Returns a vector of symbols that correspond to the coordinates on the reference element."""
        symbols = self.ref_coords_as_list(dimensions)
        return sp.Matrix([[s] for s in symbols])

    def affine_vertices_as_vectors(
        self,
        dimensions: int,
        num_vertices: int,
        first_vertex: int = 0,
        symbol_suffix: str = "",
    ) -> List[sp.Matrix]:
        """Returns a list of vectors that correspond to the coordinates of the vertices of the affinely mapped (computational) element."""
        symbols = sp.symbols(
            [
                self._symbol_affine_vertices(p, d, symbol_suffix)
                for p in range(first_vertex, num_vertices + first_vertex)
                for d in range(dimensions)
            ]
        )
        vertices = []
        for p in range(num_vertices):
            vertices.append(
                sp.Matrix([[symbols[p * dimensions + d]] for d in range(dimensions)])
            )
        return vertices

    def element_matrix_entry(self, row: int, col: int) -> sp.Symbol:
        """Returns a symbol for the respective element matrix entry."""
        return sp.symbols(self._symbol_element_matrix(row, col))

    def input_affine_vertex_access(self, point: int, coord: int) -> sp.Symbol:
        return sp.symbols(
            self._input_affine_vertices_access(
                self._input_affine_vertices_name, point, coord
            )
        )

    def output_element_matrix_access(self, row: int, col: int) -> sp.Symbol:
        return sp.Symbol(
            self._output_element_matrix_access(
                self._output_element_matrix_name, row, col
            )
        )

    def dof_symbols_names_as_list(self, num_dofs: int, function_id: str) -> List[str]:
        """Returns a list of strings that correspond to the local element's DoFs.

        :param num_dofs: number of DoFs
        :param function_id: a string identifier for the corresponding finite element function
        """
        allowed = set(string.ascii_letters + string.digits + "_")
        if not set(function_id).issubset(allowed):
            raise HOGException("Bad function identifier.")

        return [f"{function_id}_" + self._symbol_dof(i) for i in range(num_dofs)]

    def dof_symbols_as_vector(self, num_dofs: int, function_id: str) -> sp.Matrix:
        """Returns a sympy vector of symbols that correspond to the local element's DoFs.

        :param num_dofs: number of DoFs
        :param function_id: a string identifier for the corresponding finite element function
        """
        return sp.Matrix(
            list(map(sp.Symbol, self.dof_symbols_names_as_list(num_dofs, function_id)))
        )

    def affine_eval_coords_as_list(self, dimensions: int) -> List[sp.Symbol]:
        """Returns a list of symbols that correspond to the coordinates on the reference element."""
        return sp.symbols(
            [self._symbol_affine_eval_coord(i) for i in range(dimensions)]
        )

    def affine_eval_coords_as_vector(self, dimensions: int) -> sp.Matrix:
        """Returns a vector of symbols that correspond to the coordinates on the evaluation point in affine space."""
        symbols = self.affine_eval_coords_as_list(dimensions)
        return sp.Matrix([[s] for s in symbols])

    def tmp_prefix(self) -> str:
        return self._tmp_prefix

    def jac_ref_to_affine(self, dimensions: int) -> sp.Matrix:
        return sp.Matrix(
            [
                [
                    sp.Symbol(f"{self._symbol_jac_affine}_{i}_{j}")
                    for j in range(dimensions)
                ]
                for i in range(dimensions)
            ]
        )

    def jac_ref_to_affine_inv(self, dimensions: int) -> sp.Matrix:
        return sp.Matrix(
            [
                [
                    sp.Symbol(f"{self._symbol_jac_affine_inv}_{i}_{j}")
                    for j in range(dimensions)
                ]
                for i in range(dimensions)
            ]
        )

    def abs_det_jac_ref_to_affine(self) -> sp.Symbol:
        return sp.Symbol(self._symbol_abs_det_jac_affine)

    def blending_parameter_symbols(self, num_symbols: int) -> List[sp.Symbol]:
        return sp.symbols(
            [f"{self._symbol_blending_parameter_prefix}{i}" for i in range(num_symbols)]
        )

    def float_loop_ctr_array(self, dimensions: int) -> List[sp.Symbol]:
        return sp.symbols(
            [f"{self._float_loop_ctr_array_prefix}{d}" for d in range(dimensions)]
        )
