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
from hog.fem_helpers import (
    jac_blending_evaluate,
    abs_det_jac_blending_eval_symbols,
    jac_blending_inv_eval_symbols,
)
from hog.element_geometry import ElementGeometry
from hog.blending import GeometryMap, IdentityMap


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
        blending: GeometryMap = IdentityMap(),
    ):
        self.symbolizer = symbolizer
        self.quadrature = quadrature
        self.mat_integrand = integrand
        self.type_descriptor = type_descriptor
        self.symmetric = symmetric
        self.blending = blending

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

        coord_subs_dict = {
            symbol: create_field_access(
                self.p_array_names[dim],
                self.type_descriptor.pystencils_type,
                self.q_ctr,
            )
            for dim, symbol in enumerate(ref_symbols)
        }

        if not self.blending.is_affine():
            jac = jac_blending_evaluate(
                self.symbolizer, self.quadrature.geometry, self.blending
            )
            jac_evaluated = fast_subs(jac, coord_subs_dict)

            quadrature_assignments += self.blending_quad_loop_assignments(
                self.quadrature.geometry, self.symbolizer, jac_evaluated
            )

        for row in range(self.mat_integrand.rows):
            for col in range(self.mat_integrand.cols):
                tmp_symbol = sp.Symbol(f"q_tmp_{row}_{col}")
                q_acc = self.mat[row, col]

                if (self.symmetric and row > col) or not q_acc in accessed_mat_entries:
                    continue

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

    def blending_quad_loop_assignments(
        self,
        geometry: ElementGeometry,
        symbolizer: Symbolizer,
        jac_evaluated: sp.Matrix,
    ) -> List[ast.SympyAssignment]:
        quadrature_assignments = []

        jac_symbols = symbolizer.jac_affine_to_blending(geometry.dimensions)

        abs_det_jac_blending = abs_det_jac_blending_eval_symbols(geometry, symbolizer)

        jac_blending_inv = symbolizer.jac_affine_to_blending_inv(geometry.dimensions)
        jac_blending_inv_eval = jac_blending_inv_eval_symbols(geometry, symbolizer)

        dim = geometry.dimensions
        quadrature_assignments += [
            ast.SympyAssignment(jac_symbols[i, j], jac_evaluated[i, j], is_const=False)
            for i in range(dim)
            for j in range(dim)
        ]
        quadrature_assignments.append(
            ast.SympyAssignment(
                symbolizer.abs_det_jac_affine_to_blending(),
                abs_det_jac_blending,
                is_const=False,
            )
        )
        quadrature_assignments += [
            ast.SympyAssignment(
                jac_blending_inv[i, j], jac_blending_inv_eval[i, j], is_const=False
            )
            for i in range(dim)
            for j in range(dim)
        ]

        return quadrature_assignments
