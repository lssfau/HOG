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
    grad_u,
    grad_v,
    tabulate,
    **_,
):
    return tabulate(
        double_contraction(
            jac_a_inv.T * grad_u,
            jac_a_inv.T * grad_v,
        )
        * jac_a_abs_det
    )
