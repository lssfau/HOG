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
from hog.math_helpers import simpleViscosityProfile, expApprox


def integrand(
    use_dim,
    viscosity,
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
    scalars,
    x,
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

    if viscosity == "frank_kamenetskii_type1_simple_viscosity":
        etaRef = scalars("etaRef")
        temperatureSurface = scalars("temperatureSurface")
        rockChemicalCompositionParameter = scalars("rockChemicalCompositionParameter")
        depthDependency = scalars("depthDependency")
        radiusSurface = scalars("radiusSurface")
        radiusCMB = scalars("radiusCMB")
        additiveOffSet = scalars("additiveOffSet")

        # pos = ( radiusSurface - norm(x) );
        # x01 = ( norm - radiusCMB );
        # eta = eta0(x01) * exp( -rockChemicalCompositionParameter * temperature + depthDependency * pos + additiveOffSet );
        norm = x.norm()
        x01 = norm - radiusCMB

        etaSimple = simpleViscosityProfile(x01) / etaRef
        
        T_mod = k["T_extra"]-temperatureSurface
        pos = radiusSurface - norm
        
        exp_input = -rockChemicalCompositionParameter * T_mod + depthDependency * pos  + additiveOffSet
        
        exp_approx = expApprox(exp_input)
        
        eta = etaSimple * exp_approx
    else: # viscosity == "general"
        eta = k["mu"]

    return eta * (
        (
            double_contraction(2 * symm_grad_u, symm_grad_v)
            * tabulate(jac_a_abs_det)
            * jac_b_abs_det
        )
        - divdiv_scaling * div_u * div_v * tabulate(jac_a_abs_det) * jac_b_abs_det
    )
