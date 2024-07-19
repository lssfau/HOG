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

from copy import deepcopy
import sympy as sp
from hog.function_space import FunctionSpace
from hog.element_geometry import ElementGeometry


class DoFSymbol(sp.Symbol):
    """
    This is a simple wrapper around sympy symbols for DoF processing in the code generator.

    In particular this class just adds some attributes to sp.Symbol so that it can later be replaced by the
    corresponding array accesses.
    """

    def __new__(  # type: ignore[no-untyped-def] # returns Self but Self is kind of new
        cls,
        name: str,
        function_space: FunctionSpace,
        dof_id: int,
        function_id: str,
    ):
        """
        Creates a DoFSymbol

        :param function_space: the FE function space it is associated with
        :param dof_id: the index of the dof in the reference element
        :param function_id: some string identifies of the corresponding FE function
        """
        obj = sp.Symbol.__new__(cls, name)
        obj.function_space = function_space
        obj.dof_id = dof_id
        obj.function_id = function_id
        return obj

    def __deepcopy__(self, memo):
        return DoFSymbol(
            deepcopy(self.name),
            deepcopy(self.function_space),
            deepcopy(self.dof_id),
            deepcopy(self.function_id),
        )
