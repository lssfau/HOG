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
from functools import partial

def integrand_recipe(
    use_dim,
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

    inv_rho_scaling = (sp.S(1) / k["rho"] if ("rho" in k.keys()) else sp.S(1))
    divdiv_scaling = sp.Rational(1, dim) if use_dim else sp.Rational(1, 3)

    if surface_cutoff:
        norm = x.norm()
        pos = scalars("radius_surface") - norm

        pos_scaling = sp.Piecewise(
            (0.0, pos < scalars("cutoff") ),
            (1.0, sp.sympify(True) )
        )
    else:
        pos_scaling = sp.S(1)

    # define specific viscosity
    eta_ref = scalars("eta_ref")
    temperature_surface = scalars("temperature_surface")
    rock_chemical_composition_parameter = scalars("rock_chemical_composition_parameter")
    depth_dependency = scalars("depth_dependency")
    radius_surface = scalars("radius_surface")
    radius_CMB = scalars("radius_CMB")
    additive_offset = scalars("additive_offset")

    # pos = ( radius_surface - norm(x) );
    # x_01 = ( norm - radius_CMB );
    # eta = eta0(x_01) * exp( -rock_chemical_composition_parameter * temperature + depth_dependency * pos + additive_offset );
    norm = x.norm()
    x_01 = norm - radius_CMB

    eta_simple = simple_viscosity_profile(x_01) / eta_ref
    
    T_mod = k["T_extra"] - temperature_surface
    pos = radius_surface - norm
    
    exp_input = -rock_chemical_composition_parameter * T_mod + depth_dependency * pos  + additive_offset
    
    exp_approx = exp_approx(exp_input)
    
    eta = eta_simple * exp_approx      

    # build form
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

def integrand():
    return partial(integrand_recipe(True, False))

def integrand_pseudo_3D():
    return partial(integrand_recipe(False, False))

def integrand_with_cutoff():
    return partial(integrand_recipe(True, True))

def integrand_with_cutoff_pseudo_3D():
    return partial(integrand_recipe(False, True))