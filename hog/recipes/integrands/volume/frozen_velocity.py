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
    v,
    k,
    grad_k,
    tabulate,
    trial_is_vectorial,
    component_index,
    **_,
):
    inv_rho_scaling = (sp.S(1) / k["rho"] if not ("inv_rho" in k.keys()) else k["inv_rho"])

    if trial_is_vectorial:
        return (
            inv_rho_scaling
            * dot((jac_b_inv.T * jac_a_inv.T * grad_k["rho"]), u)
            * tabulate(v * jac_a_abs_det)
            * jac_b_abs_det
        )
    else:
        return (
            inv_rho_scaling
            * (jac_b_inv.T * jac_a_inv.T * grad_k["rho"])[component_index]
            * tabulate(u * v * jac_a_abs_det)
            * jac_b_abs_det
        )
