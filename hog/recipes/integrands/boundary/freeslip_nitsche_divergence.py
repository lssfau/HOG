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

from hog.recipes.common import *


def integrand(*, u, v, x, jac_a_boundary, jac_b, matrix, **_):

    space_dim = len(x)

    A = matrix("A", space_dim, space_dim)
    b = matrix("b", space_dim, 1)

    n = A * x + b
    n = n / n.norm()

    ds = abs(det(jac_a_boundary.T * jac_b.T * jac_b * jac_a_boundary)) ** 0.5
    return -v * dot(n, u) * ds
