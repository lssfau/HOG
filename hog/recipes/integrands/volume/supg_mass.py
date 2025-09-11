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
    jac_a_inv,
    jac_a_abs_det,
    jac_b_inv,
    jac_b_abs_det,
    u,
    grad_v,
    k,
    volume_geometry,
    tabulate,
    scalars,
    x_ref,
    affine_diameter,
    **_,
):
    if volume_geometry.dimensions > 2:
        u_vec = sp.Matrix([[k["ux"]], [k["uy"]], [k["uz"]]])
    else:
        u_vec = sp.Matrix([[k["ux"]], [k["uy"]]])

    # delta function
    if "delta" in k.keys():
        delta = k["delta"]
    else:
        delta = delta_supg(x_ref, u_vec, affine_diameter, scalars("thermal_conductivity"), True)

    return (
        delta
        * dot(jac_b_inv.T * tabulate(jac_a_inv.T * grad_v), u_vec)
        * tabulate(u * jac_a_abs_det)
        * jac_b_abs_det
    )