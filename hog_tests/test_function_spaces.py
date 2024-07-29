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
from hog.element_geometry import TriangleElement
from hog.function_space import (
    FunctionSpace,
    LagrangianFunctionSpace,
    TensorialVectorFunctionSpace,
)
from hog.symbolizer import Symbolizer
from hog.exception import HOGException


def test_function_spaces():
    symbolizer = Symbolizer()
    geometry = TriangleElement()

    print()

    f: FunctionSpace = LagrangianFunctionSpace(1, symbolizer)
    f_shape = f.shape(geometry)
    f_grad_shape = f.grad_shape(geometry)
    print(f)
    print()
    print("- shape")
    for s in f_shape:
        if s.shape != (1, 1):
            raise HOGException("Shape function has wrong shape (sic).")
        sp.pprint(s)
    print()
    print("- grad shape")
    for g in f_grad_shape:
        if g.shape != (geometry.dimensions, 1):
            raise HOGException("Gradient has wrong shape.")
        sp.pprint(g)

    print()
    print()

    f_scalar = LagrangianFunctionSpace(1, symbolizer)
    f = TensorialVectorFunctionSpace(f_scalar)
    f_shape = f.shape(geometry)
    f_grad_shape = f.grad_shape(geometry)
    print(f)
    print()
    print("- shape")
    for s in f_shape:
        if s.shape != (geometry.dimensions, 1):
            raise HOGException("Shape function has wrong shape (sic).")
        sp.pprint(s)
    print()
    print("- grad shape")
    for g in f_grad_shape:
        if g.shape != (geometry.dimensions, geometry.dimensions):
            raise HOGException("Gradient has wrong shape.")
        sp.pprint(g)
