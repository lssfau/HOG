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
    scalars,
    x,
    **_,
):
    # setup
    grad_u_chain = jac_b_inv.T * tabulate(jac_a_inv.T * grad_u)
    grad_v_chain = jac_b_inv.T * tabulate(jac_a_inv.T * grad_v)

    def symm_grad(w):
        return 0.5 * (w + w.T)

    symm_grad_u = symm_grad(grad_u_chain)
    symm_grad_v = symm_grad(grad_v_chain)

    div_u = (jac_b_inv.T * tabulate(jac_a_inv.T * grad_u)).trace()
    div_v = (jac_b_inv.T * tabulate(jac_a_inv.T * grad_v)).trace()

    divdiv_scaling = sp.Rational(2, volume_geometry.dimensions) if use_dim else sp.Rational(2, 3)

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
    
    T_mod = k["T"] - temperature_surface
    pos = radius_surface - norm
    
    exp_input = -rock_chemical_composition_parameter * T_mod + depth_dependency * pos  + additive_offset
    
    exp_val = exp_approx(exp_input)
    
    eta = eta_simple * exp_val

    return eta * (
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
