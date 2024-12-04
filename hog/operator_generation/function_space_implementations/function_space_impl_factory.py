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

from hog.exception import HOGException

from hog.function_space import (
    FunctionSpace,
    LagrangianFunctionSpace,
    TensorialVectorFunctionSpace,
    N1E1Space,
)

from hog.operator_generation.function_space_implementations.function_space_impl_base import (
    FunctionSpaceImpl,
)

from hog.operator_generation.function_space_implementations.p0_space_impl import (
    P0FunctionSpaceImpl,
)
from hog.operator_generation.function_space_implementations.p1_space_impl import (
    P1FunctionSpaceImpl,
    P1VectorFunctionSpaceImpl,
)
from hog.operator_generation.function_space_implementations.p2_space_impl import (
    P2FunctionSpaceImpl,
    P2VectorFunctionSpaceImpl,
)
from hog.operator_generation.function_space_implementations.n1e1_space_impl import (
    N1E1FunctionSpaceImpl,
)

from hog.operator_generation.types import HOGType


def create_impl(
    func_space: FunctionSpace,
    name: str,
    type_descriptor: HOGType,
    is_pointer: bool = False,
) -> FunctionSpaceImpl:
    """Takes a mathematical function space and produces the corresponding function space implementation.

    :param func_space:      The mathematical function space.
    :param name:            The C++ variable name/identifier of this function.
    :param type_descriptor: The value type of this function.
    :param is_pointer:      Whether the C++ variable is of a pointer type. Used
                            to print member accesses.
    """
    impl_class: type

    if isinstance(func_space, LagrangianFunctionSpace):
        if func_space.degree == 2:
            impl_class = P2FunctionSpaceImpl
        elif func_space.degree == 1:
            impl_class = P1FunctionSpaceImpl
        elif func_space.degree == 0:
            impl_class = P0FunctionSpaceImpl
        else:
            raise HOGException("Lagrangian function space must be of order 1 or 2.")
    elif isinstance(func_space, TensorialVectorFunctionSpace):
        if isinstance(func_space.component_function_space, LagrangianFunctionSpace):
            if func_space.component_function_space.degree == 1:
                if func_space.single_component is None:
                    impl_class = P1VectorFunctionSpaceImpl
                else:
                    impl_class = P1FunctionSpaceImpl
            elif func_space.component_function_space.degree == 2:
                if func_space.single_component is None:
                    impl_class = P2VectorFunctionSpaceImpl
                else:
                    impl_class = P2FunctionSpaceImpl
            else:
                raise HOGException(
                    "TensorialVectorFunctionSpaces not supported for the chosen components."
                )
        else:
            raise HOGException(
                "TensorialVectorFunctionSpaces are only supported with Lagrangian component spaces."
            )
    elif isinstance(func_space, N1E1Space):
        impl_class = N1E1FunctionSpaceImpl
    else:
        raise HOGException("Unexpected function space")

    return impl_class(func_space, name, type_descriptor, is_pointer)
