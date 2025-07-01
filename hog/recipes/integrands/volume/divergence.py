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
    grad_u,
    v,
    tabulate,
    component_index,
    **_,
):
    # working with vector-valued functions
    if grad_u.is_square:
        return (
            -(jac_b_inv.T * tabulate(jac_a_inv.T * grad_u)).trace()
            * tabulate(v * jac_a_abs_det)
            * jac_b_abs_det
        )

    # working with scalar-valued functions (backward compatibility)
    else:
        return (
            -(jac_b_inv.T * tabulate(jac_a_inv.T * grad_u))[component_index]
            * tabulate(v * jac_a_abs_det)
            * jac_b_abs_det
        )


def integrand_compressible(
    *,
    jac_a_inv,
    jac_a_abs_det,
    jac_b_inv,
    jac_b_abs_det,
    u,
    grad_u,
    v,
    k,
    grad_k,
    tabulate,
    **_,
):
    rho = k["rho"]
    grad_rho = jac_b_inv.T * tabulate(jac_a_inv.T * grad_k["rho"])

    div_u = (jac_b_inv.T * tabulate(jac_a_inv.T * grad_u)).trace()
    div_rho_u = rho * div_u + dot(grad_rho, u)[0]

    return (-div_rho_u) * tabulate(v * jac_a_abs_det) * jac_b_abs_det
