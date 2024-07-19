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
    grad_v,
    k,
    tabulate,
    **_,
):

    grad_u_chain = jac_b_inv.T * tabulate(jac_a_inv.T * grad_u)
    grad_v_chain = jac_b_inv.T * tabulate(jac_a_inv.T * grad_v)

    def symm_grad(w):
        return 0.5 * (w + w.T)

    symm_grad_u = symm_grad(grad_u_chain)
    symm_grad_v = symm_grad(grad_v_chain)

    return (
        double_contraction(2 * k[0] * symm_grad_u, symm_grad_v)
        * tabulate(jac_a_abs_det)
        * jac_b_abs_det
    )
