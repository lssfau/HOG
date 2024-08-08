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
    v,
    jac_a_inv,
    jac_a_abs_det,
    jac_b_inv,
    jac_b_abs_det,
    grad_u,
    grad_v,
    k,
    volume_geometry,
    tabulate,
    **_,
):
    if volume_geometry.dimensions > 2:
        u = sp.Matrix([[k["ux"]], [k["uy"]], [k["uz"]]])
    else:
        u = sp.Matrix([[k["ux"]], [k["uy"]]])

    return (
        k["cp_times_delta"]
        * dot(jac_b_inv.T * tabulate(jac_a_inv.T * grad_u), u)
        * dot(jac_b_inv.T * tabulate(jac_a_inv.T * grad_v), u)
        * tabulate(jac_a_abs_det)
        * jac_b_abs_det
    )
