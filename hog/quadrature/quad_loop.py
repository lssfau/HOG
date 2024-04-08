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

import logging
import sympy as sp
from typing import Iterable, List, Optional

import pystencils as ps
import pystencils.astnodes as ast
from pystencils.typing import BasicType

from .quadrature import Quadrature
import hog
from hog.cse import CseImplementation
from hog.logger import TimedLogger
from hog.operator_generation.pystencils_extensions import create_field_access
from hog.operator_generation.types import HOGType
from hog.symbolizer import Symbolizer
from hog.sympy_extensions import fast_subs


class QuadLoop:
    """Implements a quadrature scheme by an explicit loop over quadrature points."""

    q_ctr = ps.TypedSymbol("q", BasicType(int))

    def __init__(
        self,
        symbolizer: Symbolizer,
        quadrature: Quadrature,
        integrand: sp.MatrixBase,
        type_descriptor: HOGType,
        symmetric: bool,
    ):
        self.symbolizer = symbolizer
        self.quadrature = quadrature
        self.mat_integrand = integrand
        self.type_descriptor = type_descriptor
        self.symmetric = symmetric

        self.w_array_name = "q_w"
        self.p_array_names = [
            "q_p_" + str(idx) for idx in range(quadrature.geometry.dimensions)
        ]

        self.mat = sp.Matrix.zeros(self.mat_integrand.rows, self.mat_integrand.cols)
        for row in range(self.mat_integrand.rows):
            for col in range(self.mat_integrand.cols):
                if self.symmetric and row > col:
                    continue

                q_acc = ps.TypedSymbol(
                    f"q_acc_{row}_{col}",
                    BasicType(self.type_descriptor.pystencils_type),
                    const=False,
                )
                self.mat[row, col] = q_acc

                if self.symmetric:
                    self.mat[col, row] = q_acc

    def construct_quad_loop(
        self,
        accessed_mat_entries: Iterable[ps.TypedSymbol],
        cse: Optional[CseImplementation] = None,
    ) -> List[ast.Node]:
        ref_symbols = self.symbolizer.ref_coords_as_list(
            self.quadrature.geometry.dimensions
        )

        accumulator_declarations = []
        quadrature_assignments = []
        accumulator_updates = []

        for row in range(self.mat_integrand.rows):
            for col in range(self.mat_integrand.cols):
                tmp_symbol = sp.Symbol(f"q_tmp_{row}_{col}")
                q_acc = self.mat[row, col]

                if (self.symmetric and row > col) or not q_acc in accessed_mat_entries:
                    continue

                coord_subs_dict = {
                    symbol: create_field_access(
                        self.p_array_names[dim],
                        self.type_descriptor.pystencils_type,
                        self.q_ctr,
                    )
                    for dim, symbol in enumerate(ref_symbols)
                }

                weight = create_field_access(
                    self.w_array_name, self.type_descriptor.pystencils_type, self.q_ctr
                )
                integrated = weight * fast_subs(
                    self.mat_integrand[row, col], coord_subs_dict
                )

                accumulator_declarations.append(
                    ast.SympyAssignment(q_acc, 0.0, is_const=False)
                )
                quadrature_assignments.append(
                    ast.SympyAssignment(tmp_symbol, integrated)
                )
                accumulator_updates.append(
                    ast.SympyAssignment(q_acc, q_acc + tmp_symbol, is_const=False)
                )

        # common subexpression elimination
        if cse:
            with TimedLogger("cse on quad loop body", logging.DEBUG):
                quadrature_assignments = hog.cse.cse(
                    quadrature_assignments,
                    cse,
                    "tmp_qloop",
                    return_type=ast.SympyAssignment,
                )

        return accumulator_declarations + [
            ast.ForLoop(
                ast.Block(quadrature_assignments + accumulator_updates),
                self.q_ctr,
                0,
                len(self.quadrature.weights()),
            )
        ]

    def point_weight_decls(self) -> List[ast.ArrayDeclaration]:
        """Returns statements that declare the quadrature rules' points and weights as c arrays."""
        quad_decls = []
        quad_decls.append(
            ast.ArrayDeclaration(
                ast.FieldPointerSymbol(
                    self.w_array_name,
                    BasicType(self.type_descriptor.pystencils_type),
                    False,
                ),
                *(sp.Float(w) for _, w in self.quadrature.weights()),
            )
        )
        for dim in range(0, self.quadrature.geometry.dimensions):
            quad_decls.append(
                ast.ArrayDeclaration(
                    ast.FieldPointerSymbol(
                        self.p_array_names[dim],
                        BasicType(self.type_descriptor.pystencils_type),
                        False,
                    ),
                    *(sp.Float(point[dim]) for point in self.quadrature._points),
                )
            )
        return quad_decls