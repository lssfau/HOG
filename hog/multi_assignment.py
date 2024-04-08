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

from typing import Any, List
import sympy as sp
import uuid

from hog.exception import HOGException


class Member:
    """Representing a class member variable that needs to be set through the constructor."""

    def __init__(self, name_constructor: str, name_member: str, dtype: str):
        self.name_constructor = name_constructor
        self.name_member = name_member
        self.dtype = dtype

    def __eq__(self, other):
        return (
            self.name_constructor == other.name_constructor
            and self.name_member == other.name_member
            and self.dtype == other.dtype
        )

    def __hash__(self):
        return hash((self.name_constructor, self.name_member, self.dtype))

    def __str__(self):
        return "Member variable: " + str(vars(self))


class MultiAssignment(sp.Function):
    """
    Base class to realize external function calls that assign multiple variables.

    The evaluation of a vector or matrix-valued function with multiple parameters
    is not easily possible using only standard sympy assignments and code printing.

    Using the MultiAssignment class allows embedding call to functions of the form

        void func ( dtype in_0, dtype in_1, ..., dtype * out_0, dtype * out_1, ... );

    For specialization, you should inherit from MultiAssignment. Usage example:

    Let's say we have a C function with two input and two output arguments

        void my_func_alpha( double in_0, double in_1, double * out_0, double * out_1 )
        {
            ...
        }

    and we want to embed the result out_1 in our sympy calculations.

    First subclass MultiAssignment and override all members that are relevant (indicated in docstring).

    To use the result out_1 in sympy computations, simply create a sympy.Function
    (treat it as standard sympy.Expr), by

        # index of the output parameter
        out_idx = 1

        # input parameters as list of sympy expressions
        input_parameters = [in_0, in_1]

        out_1_expr = MyFunc("alpha", out_idx, input_parameters)
    """
    unique_identifier: uuid.UUID

    @classmethod
    def num_input_args(cls) -> int:
        """
        Should be overridden by subclass.

        Returns number of input arguments of the corresponding C-function.
        """
        raise HOGException("Number of input arguments not known.")

    @classmethod
    def num_output_args(cls) -> int:
        """
        Should be overridden by subclass.

        Returns number of output arguments of the corresponding C-function.
        """
        raise HOGException("Number of output arguments not known.")

    def implementation(self) -> str:
        """
        Should be overridden by subclass.
        
        Returns the implementation (only code block) of the C-function.
        """
        raise HOGException("No implementation has been defined.")

    def members(self) -> List[Member]:
        """
        Should be overridden by subclass if there are required members.

        Implement to specify required member variables that need to be added to the generated c++ class.
        """
        return []

    nargs = list(range(2, 100))

    def function_name(self) -> str:
        """
        Should be overridden by subclass.

        Returns a general function name to identify the type of function. E.g. 'scalar_coefficient'.
        """
        raise HOGException("function_name() not overwritten.")

    def variable_name(self) -> str:
        """
        Returns the name of a specific instance of the function. 
        If there are e.g. multiple scalar coefficients, both may have the same function_name() 
        but different variable_name() (e.g. 'alpha' and 'beta').
        """
        return self.args[0]

    def symbol_name(self, call_id: int, output_arg: int) -> str:
        """Returns a string that serves as a sympy symbol name."""
        return f"{self.function_name()}_{self.variable_name()}_out{output_arg}_id{call_id}"

    def output_arg(self) -> int:
        return self.args[1]

    def input_args(self) -> List[int]:
        return self.args[2:]

    def __eq__(self, other: Any) -> bool:
        """Two MultiAssignments are equal if their type, their name, and input args are equal."""
        return (
            type(self) == type(other)
            and self.function_name() == other.function_name()
            and self.variable_name() == other.variable_name()
            and all(
                [
                    (self_ia - other_ia).expand() == 0
                    for self_ia, other_ia in zip(self.input_args(), other.input_args())
                ]
            )
        )

    def __hash__(self):
        return self.unique_identifier.int

    def __new__(cls, *args):

        arg = args[0]
        if not isinstance(arg, sp.Symbol):
            raise HOGException("First argument of MultiAssignment must be a symbol.")

        arg = args[1]
        if type(arg) != int or arg < 0:
            if not hasattr(arg, "is_integer") or not arg.is_integer or arg < 0:
                raise HOGException(
                    "Second argument of MultiAssignment must be an integer "
                    "(either python int or sympy.Integer) and >= 0. That argument specifies "
                    "the output index."
                )

        obj = super().__new__(cls, *args)
        obj.unique_identifier = uuid.uuid4()
        return obj
