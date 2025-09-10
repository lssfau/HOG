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
    include_inv_rho_scaling,
    use_dim,
    viscosity,
    surface_cutoff,
    *,
    jac_a_inv,
    jac_b_inv,
    jac_a_abs_det,
    jac_b_abs_det,
    u,
    v,
    k,
    grad_k,
    volume_geometry,
    tabulate,
    x,
    scalars,
    **_,
):
    dim = volume_geometry.dimensions

    inv_rho_scaling = (sp.S(1) / k["rho"] if include_inv_rho_scaling else sp.S(1))
    divdiv_scaling = sp.Rational(1, dim) if use_dim else sp.Rational(1, 3)

    if surface_cutoff:
        norm = x.norm()
        pos = scalars("radiusSurface") - norm

        pos_scaling = sp.Piecewise(
            (0.0, pos < scalars("cutoff") ),
            (1.0, sp.sympify(True) )
        )
    else:
        pos_scaling = sp.S(1)

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
        eta = k["eta"]        

    grad_ux = jac_b_inv.T * jac_a_inv.T * grad_k["ux"]
    grad_uy = jac_b_inv.T * jac_a_inv.T * grad_k["uy"]
    
    grad_u = grad_ux.row_join(grad_uy)
    if dim == 3:
        grad_uz = jac_b_inv.T * jac_a_inv.T * grad_k["uz"]
        grad_u = grad_u.row_join(grad_uz)

    sym_grad_w = sp.Rational(1,2) * (grad_u + grad_u.T)

    divdiv = grad_u.trace() * sp.eye(dim)

    tau = 2 * (sym_grad_w - divdiv_scaling * divdiv)

    return (
        pos_scaling
        * inv_rho_scaling
        * eta
        * (double_contraction(tau, grad_u)[0])
        * jac_b_abs_det
        * tabulate(jac_a_abs_det * u * v)
    )