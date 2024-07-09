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
from typing import List, Dict, Tuple, Optional, Callable, TypeVar
from functools import partial

T = TypeVar("T")


def fast_subs(
    expression: T,
    substitutions: Dict[sp.Expr, sp.Expr],
    skip: Optional[Callable[[sp.Expr], bool]] = None,
) -> T:
    """Similar to sympy subs function.

    Args:
        expression: expression where parts should be substituted
        substitutions: dict defining substitutions by mapping from old to new terms
        skip: function that marks expressions to be skipped (if True is returned) - that means that in these skipped
              expressions no substitutions are done

    This version is much faster for big substitution dictionaries than sympy version
    """
    if type(expression) is sp.Matrix:
        return expression.copy().applyfunc(
            partial(fast_subs, substitutions=substitutions)
        )

    def visit(expr):
        if skip and skip(expr):
            return expr
        if hasattr(expr, "fast_subs"):
            return expr.fast_subs(substitutions, skip)
        if expr in substitutions:
            return substitutions[expr]
        if not hasattr(expr, "args"):
            return expr
        param_list = [visit(a) for a in expr.args]
        return expr if not param_list else expr.func(*param_list)

    if len(substitutions) == 0:
        return expression
    else:
        return visit(expression)
