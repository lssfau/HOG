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
    grad_v,
    jac_a_abs_det,
    jac_b_abs_det,
    k,
    volume_geometry,
    tabulate,
    x,
    xRef,
    scalars,
    affine_diameter,
    jac_a_inv,
    jac_b_inv,
    **_,
):
    if volume_geometry.dimensions > 2:
        uVec = sp.Matrix([[k["ux"]], [k["uy"]], [k["uz"]]])
    else:
        uVec = sp.Matrix([[k["ux"]], [k["uy"]]])

    g = -x/x.norm()

    # delta function
    # delta function
    if "delta" in k.keys():
        delta = k["delta"]
    else:
        delta = deltaSUPG(xRef, uVec, affine_diameter, scalars("thermalConductivity"), True)

    return (- delta * dot(uVec, g) * dot(uVec, jac_b_inv.T * tabulate(jac_a_inv.T * grad_v)) *  tabulate(u * jac_a_abs_det) * jac_b_abs_det)