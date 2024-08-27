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
from typing import Dict, List, Tuple, Union

import pystencils.astnodes as ast
from pystencils.typing import BasicType
from pystencils.typing.cast_functions import CastFunc

from . import QuadLoop, Quadrature
from hog.operator_generation.pystencils_extensions import create_field_access
from hog.operator_generation.types import HOGType
from hog.symbolizer import Symbolizer
from hog.sympy_extensions import fast_subs


class Table:
    def __init__(self, name: str):
        self.name = name
        # Using the epression as the dictionary key automatically removes
        # duplicate table entries. This catches tables only depending on row
        # *exclusive or* column counters. Furthemore, it resolves symmetry and
        # (almost) halves the size of the generated code.
        self.entries: Dict[sp.Expr, Tuple[int, sp.Symbol]] = {}

    def insert(self, entry: sp.Expr) -> sp.Symbol:
        if entry not in self.entries:
            idx = len(self.entries)  # code below relies on 0-based, consecutive indices
            symbol = sp.Symbol(f"{self.name}_{idx}")
            self.entries[entry] = (idx, symbol)
            return symbol
        else:
            _, symbol = self.entries[entry]
            return symbol


class Tabulation:
    """Hoist/pull element-independent, quadrature-point-dependent assignments out of the loop nest."""

    def __init__(self, symbolizer: Symbolizer):
        self.symbolizer = symbolizer
        self.tables: Dict[str, Table] = {}

    def register_factor(
        self, factor_name: str, factor: sp.Matrix | int | float
    ) -> sp.Matrix:
        """Register a factor of the weak form that can be tabulated. Returns
        symbols replacing the expression for the factor. The symbols are returned
        in the same form as the factor was given. E.g. in case of a blended full
        Stokes operator we might encounter J_F^-1 grad phi being a matrix."""

        if not isinstance(factor, sp.MatrixBase):
            factor = sp.Matrix([factor])

        replacement_symbols = sp.zeros(factor.rows, factor.cols)
        for r in range(factor.rows):
            for c in range(factor.cols):
                table_name = f"{factor_name}_{r}_{c}"
                # get or insert empty table
                table = self.tables.setdefault(table_name, Table(table_name))
                replacement_symbols[r, c] = table.insert(factor[r, c])

        return replacement_symbols

    def construct_tables(
        self, quadrature: Quadrature, type_descriptor: HOGType
    ) -> List[ast.ArrayDeclaration]:
        arrays = []
        for table in self.tables.values():
            # make list of lists from dict
            # sorting is technically not necessary because Python dicts are ordered
            lists_of_qp_evals = [
                quadrature.evaluate_on_quadpoints(local_mat_entry, self.symbolizer)
                for local_mat_entry, _ in sorted(
                    table.entries.items(), key=lambda entry: entry[1][0]
                )
            ]
            # transpose for better memory layout: evaluation of the factor for all basis functions consecutively instead of
            # evaluation of the factor for a single basis function on all quad points consecutively
            transposed_lists_of_qp_evals = [
                [row[i] for row in lists_of_qp_evals]
                for i in range(len(lists_of_qp_evals[0]))
            ]
            # linearize
            linearized_qp_evals = [
                CastFunc(item, type_descriptor.pystencils_type)
                for sublist in transposed_lists_of_qp_evals
                for item in sublist
            ]
            arrays.append(
                ast.ArrayDeclaration(
                    ast.FieldPointerSymbol(
                        table.name, BasicType(type_descriptor.pystencils_type), False
                    ),
                    *linearized_qp_evals,
                )
            )

        return arrays

    def resolve_table_accesses(
        self, mat: sp.MatrixBase, type_descriptor: HOGType
    ) -> sp.MatrixBase:
        """Replaces all dependencies on table entries by array accesses."""

        return fast_subs(
            mat,
            {
                access_symbol: create_field_access(
                    table.name,
                    type_descriptor.pystencils_type,
                    i + len(table.entries) * QuadLoop.q_ctr,
                )
                for table in self.tables.values()
                for i, access_symbol in table.entries.values()
            },
        )

    def inline_tables(self, mat: sp.MatrixBase) -> sp.MatrixBase:
        """Resolves all dependencies on table entries by inlining.

        This method assumes that tables do not reference other tables. If this
        is not the case and you need "deep substitutions", call this method
        several times until all references are resolved.
        """

        subs_dict: Dict[sp.Symbol, sp.Expr] = {}
        for table in self.tables.values():
            subs_dict |= {symbol: expr for expr, (_, symbol) in table.entries.items()}
        return fast_subs(mat, subs_dict)

    def register_phi_evals(
        self, phis: Union[List[sp.Expr], List[sp.MatrixBase]]
    ) -> List[sp.Expr]:
        """Convenience function to register factors for the evaluation of basis functions."""
        phi_eval_symbols = []
        for idx, phi in enumerate(phis):
            phi_eval_symbols.append(self.register_factor("phi", sp.Matrix([phi])))
        return phi_eval_symbols
