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
    grad_u,
    grad_v,
    jac_a_inv,
    jac_a_boundary,
    jac_b,
    jac_b_inv,
    x,
    k,
    scalars,
    matrix,
    **_,
):
    space_dim = len(x)

    c_penalty = scalars("c_penalty")
    A = matrix("A", space_dim, space_dim)
    b = matrix("b", space_dim, 1)

    n = A * x + b
    n = n / n.norm()

    mu = k["mu"]

    grad_u_chain = jac_b_inv.T * jac_a_inv.T * grad_u
    grad_v_chain = jac_b_inv.T * jac_a_inv.T * grad_v

    sym_grad_u = mu * (grad_u_chain + grad_u_chain.T)
    sym_grad_v = mu * (grad_v_chain + grad_v_chain.T)

    term_consistency = -dot(v, n) * dot(dot(n, sym_grad_u).T, n)
    term_symmetry = -dot(dot(n, sym_grad_v).T, n) * dot(u, n)
    term_penalty = c_penalty * mu * dot(v, n) * dot(u, n)

    ds = abs(det(jac_a_boundary.T * jac_b.T * jac_b * jac_a_boundary)) ** 0.5

    return (term_consistency + term_symmetry + term_penalty) * ds
