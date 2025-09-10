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
from hog.blending import IdentityMap
from hog.fem_helpers import hessian_shape_affine_ref_pullback, hessian_shape_blending_ref_pullback

def integrand(
    *,
    jac_a_inv,
    jac_b_inv,
    hessian_b,
    jac_a_abs_det,
    jac_b_abs_det,
    grad_u,
    grad_v,
    hessian_u,
    k,
    volume_geometry,
    tabulate,
    blending,
    xRef,
    affine_diameter,
    scalars,
    **_,
):
    if volume_geometry.dimensions > 2:
        uVec = sp.Matrix([[k["ux"]], [k["uy"]], [k["uz"]]])
    else:
        uVec = sp.Matrix([[k["ux"]], [k["uy"]]])    

    # delta function
    if "delta" in k.keys():
        delta = k["delta"]
    else:
        delta = deltaSUPG(xRef, uVec, affine_diameter, scalars("thermalConductivity"), True)        

    # scaling with 1/rho
    if "rho" in k.keys():
        scaling = sp.S(1) / k["rho"]
    else:
        scaling = sp.S(1)

    if isinstance(blending, IdentityMap):
        hessian_affine = hessian_shape_affine_ref_pullback(hessian_u, jac_a_inv)

        laplacian = sum([hessian_affine[i, i] for i in range(volume_geometry.dimensions)])

        supg = dot(uVec, tabulate(jac_a_inv.T * grad_v))

        form = (
            -delta * scaling * tabulate(laplacian) * supg * tabulate(jac_a_abs_det)
        )
    else:
        hessian_blending = hessian_shape_blending_ref_pullback(
            volume_geometry,
            grad_u,
            hessian_u,
            jac_a_inv,
            hessian_b,
            jac_b_inv,
        )

        laplacian = sum(
            [hessian_blending[i, i] for i in range(volume_geometry.dimensions)]
        )

        supg = dot(uVec, jac_b_inv.T * tabulate(jac_a_inv.T * grad_v))

        form = (
            -delta * scaling * laplacian * supg * tabulate(jac_a_abs_det) * jac_b_abs_det
        )
    
    return form