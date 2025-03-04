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

import tabulate
from typing import List, Set

import sympy as sp

from hog.exception import HOGException
import hog.printer


class Operations:
    def __init__(self):
        self.adds = 0
        self.muls = 0
        self.divs = 0
        self.pows = 0
        self.abs = 0
        self.assignments = 0
        self.function_calls = 0
        self.unknown_ops = 0

    def to_table(self) -> str:
        d = vars(self)
        return tabulate.tabulate([d.values()], headers=list(d.keys()))


def count_operations(
    expr: sp.Expr, operations: Operations, loop_factor: int = 1
) -> None:
    def pre(expr, operations):
        if not hasattr(expr, "func"):
            return
        if expr.func == sp.core.add.Add:
            operations.adds += loop_factor * (len(expr.args) - 1)
        elif expr.func == sp.core.mul.Mul:
            operations.muls += loop_factor * (len(expr.args) - 1)
            for mul_arg in expr.args:
                if mul_arg in [-1, 1]:
                    operations.muls -= loop_factor * (1)
        elif expr.func is sp.Pow:
            if expr.exp.is_integer and expr.exp.is_number:
                if expr.exp >= 0:
                    operations.muls += loop_factor * (int(expr.exp) - 1)
                else:
                    operations.divs += loop_factor * (1)
                    operations.muls += loop_factor * ((-int(expr.exp)) - 1)
            elif expr.exp.is_number:
                operations.pows += loop_factor * (1)
            else:
                operations.pows += loop_factor * (1)
                pre(expr.exp, operations)
        elif expr.func == sp.Abs:
            operations.abs += loop_factor * (1)
        for arg in expr.args:
            pre(arg, operations)

    pre(expr, operations)


class Statement:
    pass


class Assignment(Statement):
    """Simple container for an assignment of a sympy expression to a sympy symbol."""

    def __init__(self, lhs: sp.Symbol, rhs: sp.Expr, is_declaration: bool = True):
        self.lhs = lhs
        self.rhs = rhs
        self.is_declaration = is_declaration

    def count_operations(self, operations: Operations) -> None:
        operations.assignments += 1
        count_operations(self.rhs, operations)

    def to_code(self) -> str:
        a = f"{hog.printer.print_c_code(self.lhs)} = {hog.printer.print_c_code(self.rhs)};"
        if self.is_declaration:
            a = "real_t " + a
        return a

    def atoms(self) -> Set[sp.Atom]:
        return self.lhs.atoms() | self.rhs.atoms()


class FunctionCall(Statement):
    """Simple container for an external function call."""

    def __init__(
        self, name: str, input_args: List[sp.Expr], output_args: List[sp.Symbol]
    ):
        self.name = name
        self.input_args = input_args
        self.output_args = output_args

    def count_operations(self, operations: Operations) -> None:
        operations.function_calls += 1
        for ia in self.input_args:
            count_operations(ia, operations)

    def to_code(self) -> str:
        return (
            f'{self.name}( {", ".join([hog.printer.print_c_code(arg) for arg in self.input_args])}, '
            f'{", ".join([f"&{hog.printer.print_c_code(arg)}" for arg in self.output_args])} );'
        )


class CodeBlock:
    """Container for the entire block."""

    def __init__(self):
        self.statements = []

    def count_operations(self) -> Operations:
        operations = Operations()
        for statement in self.statements:
            statement.count_operations(operations)
        return operations

    def to_code(self) -> str:
        code = []
        for statement in self.statements:
            code.append(statement.to_code())
        return "\n".join(code)


class FunctionDefinition:
    """Container for a function definition."""

    def __init__(
        self,
        name: str,
        num_input_args: int,
        num_output_args: int,
        implementation_string: str,
    ):
        self.name = name
        self.num_input_args = num_input_args
        self.num_output_args = num_output_args
        self.implementation_string = implementation_string

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.num_input_args == other.num_input_args
            and self.num_output_args == other.num_output_args
        )

    def _decl(self, name_prefix: str = "") -> str:
        input_args = ", ".join([f"real_t in_{a}" for a in range(self.num_input_args)])
        output_args = ", ".join(
            [f"real_t * out_{a}" for a in range(self.num_output_args)]
        )
        return f"void {name_prefix}{self.name}( {input_args}, {output_args} ) const"

    def declaration(self) -> str:
        return self._decl() + ";"

    def implementation(self, name_prefix: str = "") -> str:
        impl_str = "   " + "\n   ".join(self.implementation_string.splitlines())
        return f"""{self._decl(name_prefix)}
{{
{impl_str}
}}"""
