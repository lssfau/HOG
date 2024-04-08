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

import enum
from typing import Dict, Iterable, List, Tuple, Union

import poly_cse_py.sympy_interface as poly_cse
import pystencils as ps
import sympy as sp

import hog.ast
from hog.exception import HOGException

Assignment = Union[hog.ast.Assignment, ps.astnodes.SympyAssignment]


class CseImplementation(enum.Enum):
    SYMPY = 1
    POLYCSE = 2

    def __str__(self):
        return self.name


def cse(
    assignments: Iterable[Assignment],
    impl: CseImplementation,
    tmp_symbol_prefix: str,
    return_type: type = hog.ast.Assignment,
) -> List[Assignment]:
    if impl == CseImplementation.POLYCSE:
        ass_tuples: List[Tuple[str, sp.Expr]] = []
        lhs_expressions: Dict[str, sp.Expr] = {}
        for s in assignments:
            ass_tuples.append((s.lhs.name, s.rhs))
            lhs_expressions[s.lhs.name] = s.lhs

        after_cse = poly_cse.extract_multi_terms(ass_tuples, tmp_symbol_prefix)

        return [
            return_type(lhs_expressions.get(s[0], sp.Symbol(s[0])), s[1])
            for s in after_cse
        ]
    else:
        tmp_symbols = sp.numbered_symbols(tmp_symbol_prefix + "_")
        ass_sp = map(lambda a: sp.codegen.Assignment(a.lhs, a.rhs), assignments)
        code_block = sp.codegen.CodeBlock(*ass_sp)

        after_cse = code_block.cse(symbols=tmp_symbols).args

        return [return_type(a.lhs, a.rhs) for a in after_cse]
