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

def integrand(
    *,
    u,
    v,
    jac_a_abs_det,
    jac_b_abs_det,
    k,
    tabulate,
    component_index,
    x,
    test_is_vectorial,
    **_,
):
    g = -x/x.norm()

    if test_is_vectorial:
        return (
            -k["rho"] * dot(g, v) * tabulate(u * jac_a_abs_det) * jac_b_abs_det
        )
    else:
        return (
            -k["rho"] * g[component_index] * tabulate(u * v * jac_a_abs_det) * jac_b_abs_det
        ) 