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
    volume_geometry,
    tabulate,
    **_,
):
    if volume_geometry.dimensions == 2:
        volume = jac_b_abs_det * 0.5 * jac_a_abs_det
        tau = -volume * 0.2
    else:
        volume = jac_b_abs_det * jac_a_abs_det / 6.0
        tau = -pow(volume, 2.0 / 3.0) / 12.0

    return tau * (
        double_contraction(
            jac_b_inv.T * tabulate(jac_a_inv.T * grad_u),
            jac_b_inv.T * tabulate(jac_a_inv.T * grad_v),
        )
        * tabulate(jac_a_abs_det)
        * jac_b_abs_det
    )
