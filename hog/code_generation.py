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

import sympy as sp
from typing import Dict, List, Optional, Tuple, Union
import logging
from uuid import UUID

from pystencils.astnodes import SympyAssignment, Node

from hog.ast import Assignment, CodeBlock, FunctionCall, FunctionDefinition
import hog.cse
from hog.element_geometry import ElementGeometry
from hog.exception import HOGException
from hog.fem_helpers import jac_ref_to_affine, trafo_ref_to_affine
from hog.logger import TimedLogger
from hog.multi_assignment import Member, MultiAssignment
from hog.quadrature import Quadrature
from hog.symbolizer import Symbolizer
from hog.blending import GeometryMap, ExternalMap


def has_nested_multi_assignment(expr: sp.Expr) -> bool:
    """Returns True if the expression has a MultiAssignment instance somewehere in its expression tree.

    If the expression itself is a MultiAssignment but there is no nested MultiAssignment, this function returns False.
    """

    def recursive_search(expr: sp.Expr) -> bool:
        if isinstance(expr, MultiAssignment):
            return True
        elif hasattr(expr, "args"):
            if not expr.args:
                return False
            for arg in expr.args:
                if recursive_search(arg):
                    return True
        return False

    if hasattr(expr, "args"):
        if not expr.args:
            return False
        for arg in expr.args:
            if recursive_search(arg):
                return True
    return False


def replace_multi_assignments(
    element_matrix: sp.Matrix,
    symbolizer: Symbolizer,
) -> Tuple[
    sp.Matrix,
    List[FunctionCall],
    List[FunctionDefinition],
    List[sp.Symbol],
    List[Member],
]:
    """
    Traverses the element matrix an replaces MultiAssignment objects with sympy symbols.

    The required function calls, definitions and members are collected and returned.
    """

    def gather_identical_func_calls(
        expr: sp.Expr,
        multi_assignments_to_replace: Dict[int, List[MultiAssignment]],
    ) -> None:
        """Let's first identify MultiAssignments to replace and gather all those that refer to the same function call in the same lists."""
        if isinstance(expr, MultiAssignment) and not has_nested_multi_assignment(expr):
            new_call_id = len(multi_assignments_to_replace)
            for call_id, multi_assignments in multi_assignments_to_replace.items():
                if expr == multi_assignments[0]:
                    new_call_id = call_id
                    break
            if new_call_id in multi_assignments_to_replace:
                multi_assignments_to_replace[new_call_id].append(expr)
            else:
                multi_assignments_to_replace[new_call_id] = [expr]

        elif hasattr(expr, "args") and expr.args:
            for arg in expr.args:
                if isinstance(arg, MultiAssignment) or has_nested_multi_assignment(arg):
                    gather_identical_func_calls(arg, multi_assignments_to_replace)

    def replace_func_calls(
        expr: sp.Expr,
        multi_assignments_replacement_symbols: Dict[UUID, sp.Symbol],
    ) -> sp.Expr:
        """Since each multi assignment object has a unique identifier, we can easily replace all those marked during the gathering."""
        if (
            isinstance(expr, MultiAssignment)
            and expr.unique_identifier in multi_assignments_replacement_symbols
        ):
            return multi_assignments_replacement_symbols[expr.unique_identifier]

        elif not hasattr(expr, "args"):
            return expr

        else:
            args = [
                replace_func_calls(arg, multi_assignments_replacement_symbols)
                for arg in expr.args
            ]
            return expr if not args else expr.func(*args)

    # This will be a dict filled with:
    # - key:   int                   -> each key corresponds to a single function call
    # - value: List[MultiAssignment] -> all MultiAssignments with same function type, name, and input arguments (but may hva any output arguments)
    multi_assignments_to_replace: Dict[int, List[MultiAssignment]] = {}

    rows, cols = element_matrix.shape
    contributions = [element_matrix[i, j] for i in range(rows) for j in range(cols)]

    with TimedLogger(f"looking for function calls", level=logging.DEBUG):
        for i, expr in enumerate(contributions):
            with TimedLogger(
                f"contribution {i + 1}/{len(contributions)}", level=logging.DEBUG
            ):
                gather_identical_func_calls(expr, multi_assignments_to_replace)

    # This will be a dict filled with:
    # - key:   UUID -> UUIDs of all MultiAssignments to be replaced in this pass
    # - value: sp.Symbol -> symbol that replaces the corresponding MultiAssignment
    multi_assignments_replacement_symbols = {}

    # A list of all output variables, since we need to initialize them with zero.
    undefined_output_variables = []

    for call_id, multi_assignments in multi_assignments_to_replace.items():
        replacement_symbols = {}
        # Preparing symbols to replace the MA.
        # Let's take any MA since they all have the same input arguments.
        for oa in range(multi_assignments[0].num_output_args()):
            replacement_symbol = sp.Symbol(
                multi_assignments[0].symbol_name(call_id, oa)
            )
            replacement_symbols[oa] = replacement_symbol
            undefined_output_variables.append(replacement_symbol)
        # Actually filling the dict.
        for ma in multi_assignments:
            replacement_symbol = replacement_symbols[ma.output_arg()]
            multi_assignments_replacement_symbols[ma.unique_identifier] = (
                replacement_symbol
            )

    if multi_assignments_replacement_symbols:
        with TimedLogger(
            f"substitution of function calls with symbols",
            level=logging.DEBUG,
        ):
            for row in range(rows):
                for col in range(cols):
                    element_matrix[row, col] = replace_func_calls(
                        element_matrix[row, col],
                        multi_assignments_replacement_symbols,
                    )

    func_defs = []
    func_calls = []
    members = []

    for call_id, mas in multi_assignments_to_replace.items():
        ma = mas[0]
        func_def = FunctionDefinition(
            f"{ma.function_name()}_{ma.variable_name()}",
            ma.num_input_args(),
            ma.num_output_args(),
            ma.implementation(),
        )
        func_call = FunctionCall(
            f"{ma.function_name()}_{ma.variable_name()}",
            ma.input_args(),
            [ma.symbol_name(call_id, oa) for oa in range(ma.num_output_args())],
        )
        members += ma.members()

        if func_def not in func_defs:
            func_defs.append(func_def)
        func_calls.append(func_call)

    return element_matrix, func_calls, func_defs, undefined_output_variables, members


def jacobi_matrix_assignments(
    element_matrix: sp.Matrix,
    quad_stmts: List[sp.codegen.ast.CodegenAST],
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    affine_points: Optional[List[sp.Matrix]] = None,
) -> List[Assignment]:
    """
    Generates a list of assignments computing the Jacobian (and if needed its inverse and absolute determinant).

    The element_matrix and quad_stmts are searched for symbols for the Jacobian, its inverse and its absolute
    determinant. Returns assignments which take the affine coordinates and compute the found symbols. Only those
    assignments that are necessary to compute the expressions in the element matrix and quad statements are returned to
    avoid having unused variables.
    """

    # Our given expressions may contain any of the following symbols:
    #
    # - entries of the Jacobian
    # - entries of the inverse of the Jacobian
    # - the determinant of the Jacobian
    #
    # The inverse and the determinant shall be replaced by expressions that consist of the Jacobian entries.
    # So we have to proceed in steps to find the entries of the Jacobian that we really need.
    #
    # 1. We check which entries of the inverse of the Jacobian are required and whether the determinant is also
    #    required (i.e., check if they are part of the passed expressions).
    # 2. We then express those through the entries of the Jacobian.
    # 3. We parse
    #    - the expressions of the element matrix,
    #    - the expressions of the quad statements,
    #    - the expressions that compose the inverse Jacobian entries and the determinant,
    #    and keep track of all the entries of the Jacobian that are in there.

    assignments = []

    jac_aff_symbol = symbolizer.jac_ref_to_affine(geometry.dimensions)
    jac_aff_inv_symbol = symbolizer.jac_ref_to_affine_inv(geometry.dimensions)
    jac_aff_det_symbol = symbolizer.abs_det_jac_ref_to_affine()

    free_symbols = element_matrix.free_symbols | {
        free_symbol
        for stmt in quad_stmts
        for free_symbol in stmt.undefined_symbols
        if isinstance(stmt, Node)
    }

    # Steps 1 and 2.
    jac_affine_inv_in_expr = set(jac_aff_inv_symbol).intersection(free_symbols)
    abs_det_jac_affine_in_expr = jac_aff_det_symbol in free_symbols

    # Just an early exit. Not strictly required, but might accelerate this process in some cases.
    if jac_affine_inv_in_expr:
        jac_aff_inv_expr = jac_aff_symbol.inv()
        for s_ij, e_ij in zip(jac_aff_inv_symbol, jac_aff_inv_expr):
            if s_ij in jac_affine_inv_in_expr:
                assignments.append(SympyAssignment(s_ij, e_ij))

    if abs_det_jac_affine_in_expr:
        assignments.append(
            SympyAssignment(jac_aff_det_symbol, sp.Abs(jac_aff_symbol.det()))
        )

    # Collecting all expressions to parse for step 3.
    free_symbols |= {free_symbol for a in assignments for free_symbol in a.rhs.atoms()}

    jac_affine_in_expr = set(jac_aff_symbol).intersection(free_symbols)

    # Just an early exit. Not strictly required, but might accelerate this process in some cases.
    if jac_affine_in_expr:
        jac_aff_expr = jac_ref_to_affine(geometry, symbolizer, affine_points)
        for s_ij, e_ij in zip(jac_aff_symbol, jac_aff_expr):
            if s_ij in jac_affine_in_expr:
                assignments.append(SympyAssignment(s_ij, e_ij))

    return assignments


def blending_jacobi_matrix_assignments(
    element_matrix: sp.Matrix,
    quad_stmts: List[sp.codegen.ast.CodegenAST],
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    affine_points: Optional[List[sp.Matrix]],
    blending: GeometryMap,
    quad_info: Quadrature,
) -> List[Assignment]:

    assignments = []

    free_symbols = element_matrix.free_symbols | {
        free_symbol
        for stmt in quad_stmts
        for free_symbol in stmt.undefined_symbols
        if isinstance(stmt, Node)
    }

    for i_q_pt, point in enumerate(quad_info._point_symbols):
        jac_blend_symbol = symbolizer.jac_affine_to_blending(
            geometry.dimensions, q_pt=f"_q_{i_q_pt}"
        )
        jac_blend_inv_symbol = symbolizer.jac_affine_to_blending_inv(
            geometry.dimensions, q_pt=f"_q_{i_q_pt}"
        )
        jac_blend_det_symbol = symbolizer.abs_det_jac_affine_to_blending(
            q_pt=f"_q_{i_q_pt}"
        )

        jac_blending_inv_in_expr = set(jac_blend_inv_symbol).intersection(free_symbols)
        abs_det_jac_blending_in_expr = jac_blend_det_symbol in free_symbols

        if jac_blending_inv_in_expr:
            jac_blend_inv_expr = jac_blend_symbol.inv()
            for s_ij, e_ij in zip(jac_blend_inv_symbol, jac_blend_inv_expr):
                if s_ij in jac_blending_inv_in_expr:
                    assignments.append(SympyAssignment(s_ij, e_ij))

        if abs_det_jac_blending_in_expr:
            assignments.append(
                SympyAssignment(jac_blend_det_symbol, sp.Abs(jac_blend_symbol.det()))
            )

        free_symbols |= {
            free_symbol for a in assignments for free_symbol in a.rhs.atoms()
        }

        jac_blending_in_expr = set(jac_blend_symbol).intersection(free_symbols)

        # Just an early exit. Not strictly required, but might accelerate this process in some cases.
        if jac_blending_in_expr:
            if isinstance(blending, ExternalMap):
                HOGException("Not implemented or cannot be?")

            # Collecting all expressions to parse for step 3.
            jac_blend_expr = blending.jacobian(
                trafo_ref_to_affine(geometry, symbolizer, affine_points)
            )
            spat_coord_subs = {}
            for idx, symbol in enumerate(
                symbolizer.ref_coords_as_list(geometry.dimensions)
            ):
                spat_coord_subs[symbol] = point[idx]
            jac_blend_expr_sub = jac_blend_expr.subs(spat_coord_subs)
            for s_ij, e_ij in zip(jac_blend_symbol, jac_blend_expr_sub):
                if s_ij in jac_blending_in_expr:
                    assignments.append(SympyAssignment(s_ij, e_ij))

    assignments.reverse()

    return assignments


def code_block_from_element_matrix(
    element_matrix: sp.Matrix,
    quad: Quadrature,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    coefficientAveraging: Union[bool, str] = False,
    cse: hog.cse.CseImplementation = hog.cse.CseImplementation.SYMPY,
) -> Tuple[CodeBlock, List[FunctionDefinition], List[Member]]:
    """Helper function that creates a CodeBlock object, a list of required function definition objects, and a list of
    required member variables from the local element matrix."""

    code_block = CodeBlock()
    func_calls = []
    func_defs = []
    undefined_variables = []
    members = []

    if isinstance(element_matrix, sp.Expr) and not isinstance(
        element_matrix, sp.Matrix
    ):
        element_matrix = sp.Matrix([element_matrix])

    rows, cols = element_matrix.shape

    # Replacing MultiAssignments (MAs) by function calls.
    #
    # We need to perform multiple passes here.
    # The reason is that MAs might be composed, i.e. input arguments might be MAs themselves.
    #
    # So we search and replace in each pass all MAs that do not have any MAs in their input arguments.
    #
    # As soon as no MAs are in the contribution/matrix entry, we are finished.
    with TimedLogger("replacing function call symbols", level=logging.DEBUG):
        parser_pass = 0
        while True:
            mat_has_multiassignment = False
            contributions = [
                element_matrix[i, j] for i in range(rows) for j in range(cols)
            ]
            for c in contributions:
                if has_nested_multi_assignment(c):
                    mat_has_multiassignment = True
                    parser_pass += 1
                    break

            if mat_has_multiassignment:
                with TimedLogger(f"pass {parser_pass}", level=logging.DEBUG):
                    element_matrix, fc, fd, undef, m = replace_multi_assignments(
                        element_matrix, symbolizer
                    )
                    func_calls += fc
                    func_defs += fd
                    undefined_variables += undef
                    members += m
            else:
                break

    undef_assignments = []
    for v in undefined_variables:
        assignment = Assignment(v, 0)
        undef_assignments.append(assignment)
    code_block.statements = undef_assignments + code_block.statements

    for func_call in func_calls:
        code_block.statements.append(func_call)

    with TimedLogger("building assignments", logging.DEBUG):
        mat_assignments = [
            Assignment(symbolizer.element_matrix_entry(i, j), element_matrix[i, j])
            for i in range(rows)
            for j in range(cols)
        ]

        jacobi_assignments = jacobi_matrix_assignments(
            element_matrix, [], geometry, symbolizer
        )
        assignments = jacobi_assignments + mat_assignments

    # common subexpression elimination for all matrix entries
    with TimedLogger("cse", logging.DEBUG):
        assignments = hog.cse.cse(assignments, cse, "tmp")
        code_block.statements.extend(assignments)

    # replace quadrature weights and points with values

    if quad:
        with TimedLogger(
            "substitution of quadrature point and weight symbols", level=logging.DEBUG
        ):
            for stmt in code_block.statements:
                if isinstance(stmt, FunctionCall):
                    for p_symbol, p_value in quad.points():
                        stmt.input_args = [
                            ia.subs(p_symbol, p_value) for ia in stmt.input_args
                        ]

                    for w_symbol, w_value in quad.weights():
                        stmt.input_args = [
                            ia.subs(w_symbol, w_value) for ia in stmt.input_args
                        ]

                elif isinstance(stmt, Assignment):
                    if hasattr(stmt.rhs, "subs"):
                        for p_symbol, p_value in quad.points():
                            stmt.rhs = stmt.rhs.subs(p_symbol, p_value)

                        for w_symbol, w_value in quad.weights():
                            stmt.rhs = stmt.rhs.subs(w_symbol, w_value)
                else:
                    raise HOGException(
                        "Cannot replace quadrature points and weights in unknown statement type."
                    )

    if coefficientAveraging == "mean" or coefficientAveraging == "harmonic":
        coefficient_names = []
        n_coeffs = 0
        sumExpr = 0.0
        for s in code_block.statements:
            if not isinstance(s, FunctionCall) and (
                s.lhs.name.startswith("Scalar_Variable_Coefficient_2D_mu_out0_id")
                or s.lhs.name.startswith("Scalar_Variable_Coefficient_3D_mu_out0_id")
            ):
                coefficient_names += [s.lhs.name]
                n_coeffs = n_coeffs + 1
                if coefficientAveraging == "mean":
                    sumExpr = sumExpr + sp.Symbol(s.lhs.name)
                elif coefficientAveraging == "harmonic":
                    sumExpr = sumExpr + 1.0 / sp.Symbol(s.lhs.name)

        if n_coeffs > 0:
            if coefficientAveraging == "mean":
                code_block.statements.insert(
                    2 * n_coeffs, Assignment(sp.Symbol("coeffSum"), sumExpr / n_coeffs)
                )
            elif coefficientAveraging == "harmonic":
                code_block.statements.insert(
                    2 * n_coeffs,
                    Assignment(sp.Symbol("coeffSum"), 1.0 / (sumExpr / n_coeffs)),
                )

            newStmtsCounter = 0
            for ce in coefficient_names:
                code_block.statements.insert(
                    2 * n_coeffs + 1 + newStmtsCounter,
                    Assignment(sp.Symbol(ce), sp.Symbol("coeffSum"), False),
                )
                newStmtsCounter = newStmtsCounter + 1

    return code_block, func_defs, members
