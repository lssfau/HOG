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

def integrand_recipe(
    use_dim,
    *,
    jac_a_inv,
    jac_a_abs_det,
    jac_b_inv,
    jac_b_abs_det,
    grad_u,
    grad_v,
    k,
    tabulate,
    volume_geometry,
    **_,
):
    grad_u_chain = jac_b_inv.T * tabulate(jac_a_inv.T * grad_u)
    grad_v_chain = jac_b_inv.T * tabulate(jac_a_inv.T * grad_v)

    def symm_grad(w):
        return 0.5 * (w + w.T)

    symm_grad_u = symm_grad(grad_u_chain)
    symm_grad_v = symm_grad(grad_v_chain)

    div_u = (jac_b_inv.T * tabulate(jac_a_inv.T * grad_u)).trace()
    div_v = (jac_b_inv.T * tabulate(jac_a_inv.T * grad_v)).trace()

    divdiv_scaling = sp.Rational(2, volume_geometry.dimensions) if use_dim else sp.Rational(2, 3)

    return k["mu"] * (
        (
            double_contraction(2 * symm_grad_u, symm_grad_v)
            * tabulate(jac_a_abs_det)
            * jac_b_abs_det
        )
        - divdiv_scaling * div_u * div_v * tabulate(jac_a_abs_det) * jac_b_abs_det
    )

def integrand(**kwargs):
    return integrand_recipe(True, **kwargs)

def integrand_pseudo_3D(**kwargs):
    return integrand_recipe(False, **kwargs)
