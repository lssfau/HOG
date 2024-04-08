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
from sympy.printing.c import C99CodePrinter

from hog.exception import HOGException


class CustomCCodePrinter(C99CodePrinter):
    """Custom C code printer to compile sympy expressions to C code."""

    def _print_Pow(self, expr: sp.Expr) -> str:
        if expr.exp.is_integer and expr.exp.is_number and 0 < expr.exp < 1000:
            return f"({self._print(sp.Mul(*[expr.base] * expr.exp, evaluate=False))})"
        elif expr.exp.is_integer and expr.exp.is_number and -1000 < expr.exp < 0:
            return f"1.0 / ({self._print(sp.Mul(*([expr.base] * -expr.exp), evaluate=False))})"
        elif expr.exp.is_integer and expr.exp.is_number and expr.exp == 0:
            return f"1.0"
        else:
            return f"std::pow({super(CustomCCodePrinter, self)._print(expr.base)}, {self._print(expr.exp)})"

    def _print_Abs(self, expr):
        if len(expr.args) != 1:
            raise HOGException("Abs may only have one argument.")
        return f"std::abs({super(CustomCCodePrinter, self)._print(expr.args[0])})"

    def _print_Function(self, expr):
        if hasattr(expr, "to_c"):
            return expr.to_c(self._print)
        else:
            return super(CustomCCodePrinter, self)._print_Function(expr)


def print_c_code(expr: sp.Expr) -> str:
    """Calls custom c code printer on the passed expression and returns a string formatted as c code."""
    return CustomCCodePrinter().doprint(expr)
