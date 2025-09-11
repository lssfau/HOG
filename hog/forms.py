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

import logging
import sympy as sp
from typing import Optional, Callable, Any

from hog.element_geometry import ElementGeometry
from hog.exception import HOGException
from hog.fem_helpers import (
    trafo_ref_to_affine,
    trafo_ref_to_physical,    
    jac_ref_to_affine,
    jac_affine_to_physical,
    hessian_shape_affine_ref_pullback,
    hessian_shape_blending_ref_pullback,
    create_empty_element_matrix,
    element_matrix_iterator,
    scalar_space_dependent_coefficient,
    vector_space_dependent_coefficient,
    fem_function_on_element,
    fem_function_gradient_on_element,
)
from hog.function_space import (
    FunctionSpace,
    LagrangianFunctionSpace,
    N1E1Space,
    P2PlusBubbleSpace,
    TestSpace,
    TrialSpace,
)
from hog.math_helpers import dot, grad, inv, abs, det, double_contraction
from hog.quadrature import Quadrature, Tabulation
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger
from hog.blending import GeometryMap, ExternalMap, IdentityMap
from hog.integrand import process_integrand, Form


def diffusion(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Diffusion operator without coefficients.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    
    ∫ ∇u : ∇v
    
    Note that the double contraction (:) reduces to the dot product for scalar function spaces, i.e. the form becomes
    
    ∫ ∇u · ∇v 
"""

    from hog.recipes.integrands.volume.diffusion import integrand
    from hog.recipes.integrands.volume.diffusion_affine import (
        integrand as integrand_affine,
    )

    # mypy type checking supposedly cannot figure out ternaries.
    # https://stackoverflow.com/a/70832391
    integr: Callable[..., Any] = integrand
    if blending == IdentityMap():
        integr = integrand_affine

    return process_integrand(
        integr,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
    )


def mass(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Mass operator.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})

    ∫ uv
"""

    from hog.recipes.integrands.volume.mass import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
    )


def div_k_grad(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    coefficient_function_space: Optional[FunctionSpace] = None,
) -> Form:
    docstring = f"""
Diffusion operator with a scalar coefficient.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    k: coefficient    (space: {coefficient_function_space})

    ∫ k ∇u · ∇v
"""

    from hog.recipes.integrands.volume.div_k_grad import integrand
    from hog.recipes.integrands.volume.div_k_grad_affine import (
        integrand as integrand_affine,
    )

    # mypy type checking supposedly cannot figure out ternaries.
    # https://stackoverflow.com/a/70832391
    integr: Callable[..., Any] = integrand
    if blending == IdentityMap():
        integr = integrand_affine

    return process_integrand(
        integr,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        fe_coefficients={"k": coefficient_function_space},
        is_symmetric=trial == test,
        docstring=docstring,
    )


def nonlinear_diffusion(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    coefficient_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Diffusion operator with coefficient function.

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    c: FE coefficient function (space: {coefficient_function_space})
    
    ∫ a(c) ∇u · ∇v

Note: :math:`a(c) = 1/8 + u^2` is currently hard-coded and the form is intended for :math:`c = u`,
      hence, the naming, but will work for other coefficients, too.
"""

    if blending != IdentityMap():
        raise HOGException(
            "The nonlinear-diffusion form does currently not support blending."
        )

    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble non-linear diffusion matrix."
        )

    def integrand(
        *,
        jac_a_inv,
        jac_a_abs_det,
        grad_u,
        grad_v,
        k,
        tabulate,
        **_,
    ):
        a = sp.Rational(1, 8) + k["u"] * k["u"]
        return a * tabulate(
            double_contraction(
                jac_a_inv.T * grad_u,
                jac_a_inv.T * grad_v,
            )
            * jac_a_abs_det
        )

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
        fe_coefficients={"u": coefficient_function_space},
    )


def nonlinear_diffusion_newton_galerkin(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    coefficient_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
    only_newton_galerkin_part_of_form: Optional[bool] = True,
) -> Form:
    docstring = f"""

Bi-linear form for the solution of the non-linear diffusion equation by a Newton-Galerkin approach

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    k: FE coefficient function (space: {coefficient_function_space})
    
    ∫ a(k) ∇u · ∇v + ∫ a'(k) u ∇k · ∇v

Note: :math:`a(k) = 1/8 + k^2` is currently hard-coded and the form is intended for :math:`k = u`.
"""
    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble diffusion matrix."
        )

    if blending != IdentityMap():
        raise HOGException(
            "The nonlinear_diffusion_newton_galerkin form does currently not support blending."
        )

    def integrand(
        *,
        jac_a_inv,
        jac_a_abs_det,
        u,
        grad_u,
        grad_v,
        k,
        grad_k,
        tabulate,
        **_,
    ):
        a = sp.Rational(1, 8) + k["k"] * k["k"]
        a_prime = 2 * k["k"]

        diffusion_term = a * tabulate(
            dot(jac_a_inv.T * grad_u, jac_a_inv.T * grad_v) * jac_a_abs_det
        )

        newton_galerkin_term = (
            a_prime
            * u
            * dot(jac_a_inv.T * grad_k["k"], tabulate(jac_a_inv.T * grad_v))
            * tabulate(jac_a_abs_det)
        )

        if only_newton_galerkin_part_of_form:
            return newton_galerkin_term
        else:
            return diffusion_term + newton_galerkin_term

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=False,
        docstring=docstring,
        fe_coefficients={"k": coefficient_function_space},
    )


def epsilon(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_trial: int = 0,
    component_test: int = 0,
    variable_viscosity: bool = True,
    coefficient_function_space: Optional[FunctionSpace] = None,
    rotation_wrapper: bool = False,
) -> Form:
    docstring = f"""
"Epsilon" operator.

Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})
    μ: coefficient    (scalar space:    {coefficient_function_space})

    ∫ 2 μ ε(u) : ε(v)
    
where
    
    ε(w) := (1/2) (∇w + (∇w)ᵀ)
"""

    if not variable_viscosity:
        raise HOGException("Constant viscosity currently not supported.")

    from hog.recipes.integrands.volume.rotation import RotationType
    from hog.recipes.integrands.volume.epsilon import integrand
    from hog.recipes.integrands.volume.epsilon_affine import (
        integrand as integrand_affine,
    )

    # mypy type checking supposedly cannot figure out ternaries.
    # https://stackoverflow.com/a/70832391
    integr: Callable[..., Any] = integrand
    if blending == IdentityMap():
        integr = integrand_affine

    return process_integrand(
        integr,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
        fe_coefficients={"mu": coefficient_function_space},
        rot_type=RotationType.PRE_AND_POST_MULTIPLY
        if rotation_wrapper
        else RotationType.NO_ROTATION,
    )


def k_mass(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    coefficient_function_space: Optional[FunctionSpace] = None,
) -> Form:
    docstring = f"""
Mass operator scaled with a coefficient.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    k: coefficient    (space: {coefficient_function_space})

    ∫ k uv
"""

    from hog.recipes.integrands.volume.k_mass import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        fe_coefficients={"k": coefficient_function_space},
        docstring=docstring,
    )


def pspg(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
PSPG stabilisation.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})

    ∫ τ ∇u · ∇v

    where tau is an element-wise factor given by

    2D: 
        τ = -CellVolume(triangle) / 5               
    3D:
        τ = -(CellVolume(tetrahedron))**(2/3) / 12

See e.g. 

    Brezzi, F., & Douglas, J. (1988). 
    Stabilized mixed methods for the Stokes problem. 
    Numerische Mathematik, 53, 225-235.

or 

    Hughes, T. J., Franca, L. P., & Balestra, M. (1986). 
    A new finite element formulation for computational fluid dynamics: V. 
    Circumventing the Babuška-Brezzi condition: A stable Petrov-Galerkin formulation of the Stokes problem accommodating
    equal-order interpolations. 
    Computer Methods in Applied Mechanics and Engineering, 59(1), 85-99.

for details.
"""

    from hog.recipes.integrands.volume.pspg import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
    )


def linear_form(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """
    Implements a linear form of type:

        (k(x), psi)

    where psi a test function and k = k(x) a scalar, external function.
    """

    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble linear form (jep this is weird, but linear forms are implemented as diagonal matrices)."
        )

    if quad.is_exact() and isinstance(blending, ExternalMap):
        raise HOGException(
            "Exact integration is not supported for externally defined blending functions."
        )

    with TimedLogger("assembling linear form", level=logging.DEBUG):
        jac_affine = jac_ref_to_affine(geometry, symbolizer)
        jac_affine_det = abs(det(jac_affine))

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))

        mat = create_empty_element_matrix(trial, test, geometry)

        it = element_matrix_iterator(trial, test, geometry)

        if isinstance(trial, N1E1Space):
            jac_affine_inv_T = inv(jac_affine).T
            jac_blending_inv_T = inv(jac_blending).T
            space_dependent_coefficient = vector_space_dependent_coefficient
        else:
            space_dependent_coefficient = scalar_space_dependent_coefficient
        k = space_dependent_coefficient("k", geometry, symbolizer, blending=blending)

        with TimedLogger(
            f"integrating {mat.shape[0]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                if data.row == data.col:
                    psi = data.test_shape

                    if isinstance(trial, N1E1Space):
                        form = (
                            dot(k, jac_blending_inv_T * jac_affine_inv_T * psi)
                            * jac_affine_det
                            * jac_blending_det
                        )
                    else:
                        form = k * psi * jac_affine_det * jac_blending_det

                    mat[data.row, data.col] = quad.integrate(form, symbolizer)

    return mat


def divergence(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_index: int = 0,
    rotation_wrapper: bool = False,
    compressible: bool = False,
    density_function_space: Optional[FunctionSpace] = None,
) -> Form:
    if compressible:
        docstring = f"""
Divergence Compressible.

Component:    {component_index}
Geometry map: {blending}

Weak formulation

    u: trial function   (vectorial space: {trial})\n
    v: test function    (scalar space:    {test})\n
    𝜌: density function (scalar space:    {density_function_space})
    
    ∫ - ( ∇ · ( 𝜌u ) ) v

    which can be then be written as,
    
    ∫ -( 𝜌 ∇·u + u · ∇𝜌 ) v
"""
    else:
        docstring = f"""
Divergence.

Component:    {component_index}
Geometry map: {blending}

Weak formulation

    u: trial function (vectorial space: {trial})\n
    v: test function  (scalar space:    {test})\n

    ∫ - ( ∇ · u ) v
"""

    from hog.recipes.integrands.volume.divergence import integrand
    from hog.recipes.integrands.volume.divergence import integrand_compressible
    from hog.recipes.integrands.volume.rotation import RotationType

    if compressible:
        if not trial.is_vectorial:
            raise HOGException(
                "Compressible version of divergence form only supports vectorial functions"
            )

        return process_integrand(
            integrand_compressible,
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            is_symmetric=False,
            docstring=docstring,
            fe_coefficients={"rho": density_function_space},
            rot_type=RotationType.POST_MULTIPLY
            if rotation_wrapper
            else RotationType.NO_ROTATION,
        )
    else:
        return process_integrand(
            integrand,
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_index=component_index,
            is_symmetric=False,
            docstring=docstring,
            rot_type=RotationType.POST_MULTIPLY
            if rotation_wrapper
            else RotationType.NO_ROTATION,
        )


def gradient(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_index: int = 0,
    rotation_wrapper: bool = False,
) -> Form:
    docstring = f"""
    Gradient.

    Component:    {component_index}
    Geometry map: {blending}

    Weak formulation

        u: trial function (scalar space:    {trial})
        v: test function  (vectorial space: {test})

        ∫ - ( ∇ · v ) u
    """

    from hog.recipes.integrands.volume.gradient import integrand
    from hog.recipes.integrands.volume.rotation import RotationType

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        component_index=component_index,
        is_symmetric=False,
        docstring=docstring,
        rot_type=RotationType.PRE_MULTIPLY
        if rotation_wrapper
        else RotationType.NO_ROTATION,
    )


def full_stokes(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_trial: int = 0,
    component_test: int = 0,
    coefficient_function_space: Optional[FunctionSpace] = None,
    rotation_wrapper: bool = False,
    use_dim: bool = False,
    viscosity: str = "general"
) -> Form:
    newline = "\n"
    dimstring = "(2/dim)" if use_dim else "(2/3)"
    pseudo_note = f"{newline}Note that the factor 2/3 means that for 2D this is the pseudo-3D form of the operator.{newline}" if not use_dim else ""
    docstring = f"""
Implements the fully coupled viscous operator of the Stokes problem.
The latter is the extension of the Epsilon operator to the case where
the velocity field need not be divergence-free. This is e.g. the case
in the (truncated) anelastic liquid approximation of mantle convection.

The strong representation of the operator is given by:

   - div[ μ (grad(u)+grad(u)ᵀ) ] + {dimstring} grad[ μ div(u) ]
{pseudo_note}
Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})
    μ: coefficient    (scalar space:    {coefficient_function_space})

    ∫ μ {{ ( 2 ε(u) : ε(v) ) - {dimstring} [ ( ∇ · u ) · ( ∇ · v ) ] }}
    
where
    
    ε(w) := (1/2) (∇w + (∇w)ᵀ)
"""

    from hog.recipes.integrands.volume.rotation import RotationType
    
    # supported viscosities are
    #   "general" (as a FEM function coefficient)
    #   "frank_kamenetskii_type1_simple_viscosity" (see integrand for the formula)

    if viscosity == "frank_kamenetskii_type1_simple_viscosity":
        FEM_functions = {"T_extra": coefficient_function_space}    
        if use_dim:
            from hog.recipes.integrands.volume.full_stokes_frank_kamenetskii_type1 import integrand
        else:
            from hog.recipes.integrands.volume.full_stokes_frank_kamenetskii_type1 import integrand_pseudo_3D as integrand
    else: # "general"
        FEM_functions = {"mu": coefficient_function_space}
        if use_dim:
            from hog.recipes.integrands.volume.full_stokes import integrand
        else:
            from hog.recipes.integrands.volume.full_stokes import integrand_pseudo_3D as integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
        fe_coefficients=FEM_functions,
        rot_type=RotationType.PRE_AND_POST_MULTIPLY
        if rotation_wrapper
        else RotationType.NO_ROTATION,
    )


def shear_heating(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    viscosity_function_space: Optional[FunctionSpace] = None,
    velocity_function_space: Optional[FunctionSpace] = None,
    density_function_space: Optional[FunctionSpace] = None,
    coefficient_function_space: Optional[FunctionSpace] = None,
    use_dim: bool = False,
    viscosity: str = "general",
    include_inv_rho_scaling: bool = False,
    surface_cutoff: bool = False,
) -> Form:
    newline = "\n"
    dimstring = "1/dim" if use_dim else "1/3"
    rhostring_scaling_string = "1/rho " if include_inv_rho_scaling else ""
    rhostring_coeff_string = f"rho: coefficient (scalar space: {density_function_space}){newline}" if include_inv_rho_scaling else ""
    pseudo_note = f"{newline}Note that the factor 1/3 means that for 2D this is the pseudo-3D form of the operator.{newline}" if not use_dim else "" 
    cufoff_note = f"""
This operator was designed for annulus and spherical shell use and includes a scaling factor of 0 close to the surface.
Define the (usually nondimensional) surface radius and cutoff distance to the surface via the constructor.     
""" if surface_cutoff else ""
    docstring = f"""
Implements the fully coupled viscous operator for the shear heating term.
The latter is the extension of the Epsilon operator to the case where
the velocity field need not be divergence-free. This is e.g. the case
in the (truncated) anelastic liquid approximation of mantle convection.

https://doi.org/10.1111/j.1365-246X.2009.04413.x
(3) and (5)

https://doi.org/10.5194/gmd-15-5127-2022
Listing 2

Intended for RHS use.

Geometry map: {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    w: test function (scalar space: {test})
    u: coefficient (vector space: {velocity_function_space})
    eta: coefficient (scalar space: {viscosity_function_space})
    {rhostring_coeff_string}
    ∫ {rhostring_scaling_string}( tau(u,eta) : ∇u ) T * w

    or equivalently
    
    ∫ {rhostring_scaling_string}( tau(u,eta) : eps(u) ) T * w

    with

    tau(u,eta) = 2 eta eps(u)
    eps(u) := 1/2 ∇u + 1/2 (∇u)^T - {dimstring} (∇ · u) I
    I := Identity Matrix      
{pseudo_note}{cufoff_note}    
Typical usage sets T = 1, i.e. applying the operator to a function containing only ones.
"""
    FEM_functions = {
        "ux": velocity_function_space,
        "uy": velocity_function_space,
        "uz": velocity_function_space,
    }

    # supported viscosities are
    #   "general" (as a FEM function coefficient)
    #   "frank_kamenetskii_type1_simple_viscosity" (see integrand for the formula)

    if viscosity == "frank_kamenetskii_type1_simple_viscosity":
        FEM_functions.update({"T_extra": coefficient_function_space})  
        if use_dim:
            if surface_cutoff:
                from hog.recipes.integrands.volume.shear_heating_frank_kamenetskii_type1 import integrand_with_cutoff as integrand
            else:
                from hog.recipes.integrands.volume.shear_heating_frank_kamenetskii_type1 import integrand
        else:
            if surface_cutoff:
                from hog.recipes.integrands.volume.shear_heating_frank_kamenetskii_type1 import integrand_with_cutoff_pseudo_3D as integrand
            else:
                from hog.recipes.integrands.volume.shear_heating_frank_kamenetskii_type1 import integrand_pseudo_3D as integrand
    else: # "general"
        FEM_functions.update({"eta": viscosity_function_space})  
        if use_dim:
            if surface_cutoff:
                from hog.recipes.integrands.volume.shear_heating import integrand_with_cutoff as integrand
            else:
                from hog.recipes.integrands.volume.shear_heating import integrand
        else:
            if surface_cutoff:
                from hog.recipes.integrands.volume.shear_heating import integrand_with_cutoff_pseudo_3D as integrand
            else:
                from hog.recipes.integrands.volume.shear_heating import integrand_pseudo_3D as integrand

    if include_inv_rho_scaling:
        FEM_functions.update({"rho": density_function_space})  

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        fe_coefficients=FEM_functions,
        is_symmetric=trial == test,
        docstring=docstring,
    )

def supg_shear_heating(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    viscosity_function_space: Optional[FunctionSpace] = None,
    velocity_function_space: Optional[FunctionSpace] = None,
    density_function_space: Optional[FunctionSpace] = None,
    coefficient_function_space: Optional[FunctionSpace] = None,
    delta_function_space: Optional[FunctionSpace] = None,
    use_dim: bool = False,
    viscosity: str = "general",
    include_inv_rho_scaling: bool = False,
    surface_cutoff: bool = False,
    coefficient_delta: bool = False,
) -> Form:
    newline = "\n"
    dimstring = "1/dim" if use_dim else "1/3"
    rhostring_scaling_string = "1/rho " if include_inv_rho_scaling else ""
    rhostring_coeff_string = f"rho: coefficient (scalar space: {density_function_space}){newline}" if include_inv_rho_scaling else ""
    pseudo_note = f"{newline}Note that the factor 1/3 means that for 2D this is the pseudo-3D form of the operator.{newline}" if not use_dim else "" 
    cufoff_note = f"""
This operator was designed for annulus and spherical shell use and includes a scaling factor of 0 close to the surface.
Define the (usually nondimensional) surface radius and cutoff distance to the surface via the constructor.     
""" if surface_cutoff else ""
    deltastring = f"""
The scaling function for the supg stabilisation is hard coded into the form.

delta( u_abs ) := h / ( 2 * u_abs ) * xi( Pe )

with:
    xi( Pe ) :=  ( 1 + 2 / ( exp( 2* Pe ) - 1 ) - 1 / Pe ) = coth( Pe ) - 1 / Pe
    Pe := u_abs * h / ( 2 * k ) as the local Peclet number
    h as the element diameter
    k as the thermal conductivity coefficient
    u_abs as the norm of the velocity vector at the element centroid

Note: h is calculated from the affine element geometry. If your blending map changes the element diameter too drastically, this will no longer work.
""" if not coefficient_delta else ""   
    coefficientstring = f"    delta: coefficient (scalar space: {delta_function_space}){newline}" if coefficient_delta else "" 
    docstring = f"""
Shear heating SUPG operator for the TALA

Intended for RHS use.
{deltastring}
Geometry map: {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    w: test function (scalar space: {test})
    u: coefficient (vector space: {velocity_function_space})
    eta: coefficient (scalar space: {viscosity_function_space})
    {rhostring_coeff_string}{coefficientstring}
    ∫ delta {rhostring_scaling_string}( tau(u,eta) : ∇u ) T * ( u · ∇w )

    or equivalently
    
    ∫ delta {rhostring_scaling_string}( tau(u,eta) : eps(u) ) T * ( u · ∇w )

    with

    tau(u,eta) = 2 eta eps(u)
    eps(u) := 1/2 ∇u + 1/2 (∇u)^T - {dimstring} (∇ · u) I
    I := Identity Matrix      
{pseudo_note}{cufoff_note}    
Typical usage sets T = 1, i.e. applying the operator to a function containing only ones.
"""
    FEM_functions = {
        "ux": velocity_function_space,
        "uy": velocity_function_space,
        "uz": velocity_function_space,
    }

    if coefficient_delta:
        FEM_functions.update({"delta": delta_function_space})

    # supported viscosities are
    #   "general" (as a FEM function coefficient)
    #   "frank_kamenetskii_type1_simple_viscosity" (see integrand for the formula)

    if viscosity == "frank_kamenetskii_type1_simple_viscosity":
        FEM_functions.update({"T_extra": coefficient_function_space})  
        if use_dim:
            if surface_cutoff:
                from hog.recipes.integrands.volume.supg_shear_heating_frank_kamenetskii_type1 import integrand_with_cutoff as integrand
            else:
                from hog.recipes.integrands.volume.supg_shear_heating_frank_kamenetskii_type1 import integrand
        else:
            if surface_cutoff:
                from hog.recipes.integrands.volume.supg_shear_heating_frank_kamenetskii_type1 import integrand_with_cutoff_pseudo_3D as integrand
            else:
                from hog.recipes.integrands.volume.supg_shear_heating_frank_kamenetskii_type1 import integrand_pseudo_3D as integrand
    else: # "general"
        FEM_functions.update({"eta": viscosity_function_space})  
        if use_dim:
            if surface_cutoff:
                from hog.recipes.integrands.volume.supg_shear_heating import integrand_with_cutoff as integrand
            else:
                from hog.recipes.integrands.volume.supg_shear_heating import integrand
        else:
            if surface_cutoff:
                from hog.recipes.integrands.volume.supg_shear_heating import integrand_with_cutoff_pseudo_3D as integrand
            else:
                from hog.recipes.integrands.volume.supg_shear_heating import integrand_pseudo_3D as integrand

    if include_inv_rho_scaling:
        FEM_functions.update({"rho": density_function_space})  

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        fe_coefficients=FEM_functions,
        is_symmetric=False,
        docstring=docstring,
    )

def divdiv(
    trial: TrialSpace,
    test: TestSpace,
    component_trial: int,
    component_test: int,
    geometry: ElementGeometry,
    quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
divdiv operator which is used as a stabilization term that is taken from

    Blank, L. (2014).
    On Divergence-Free Finite Element Methods for the Stokes Equations (Freie Universität Berlin).
    p. 84, eq. (6.2)

for the P1-P0 stabilized Stokes discretization.

Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})

    ∫ ( ∇ · u ) · ( ∇ · v )
"""

    from hog.recipes.integrands.volume.divdiv import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
    )


def k_divdiv(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    coefficient_function_space: Optional[FunctionSpace] = None,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
divdiv operator which is the compressible part of full Stokes operator

Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})

    ∫ μ ( ∇ · u ) · ( ∇ · v )
"""

    from hog.recipes.integrands.volume.divdiv import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
        fe_coefficients={"k": coefficient_function_space},
    )

def adiabatic_heating(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    velocity_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Adiabatic heating operator for the TALA

Intended for LHS use.

The gravity vector is hard coded for annulus and icosahedral shell,
i.e. g := -x / norm(x).

Geometry map: {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    w: test function (scalar space: {test})
    u: velocity function (vectorial space: {velocity_function_space})

    - ∫ T(u · g) w
    """

    from hog.recipes.integrands.volume.adiabatic_heating import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=(trial == test),
        docstring=docstring,
        fe_coefficients={
            "ux": velocity_function_space,
            "uy": velocity_function_space,
            "uz": velocity_function_space,
        },
    )

def supg_adiabatic_heating(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    velocity_function_space: FunctionSpace,
    coefficient_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
    coefficient_delta: bool = False,
) -> Form:
    newline = "\n"
    deltastring = f"""
The scaling function for the supg stabilisation is hard coded into the form.

delta( u_abs ) := h / ( 2 * u_abs ) * xi( Pe )

with:
    xi( Pe ) :=  ( 1 + 2 / ( exp( 2* Pe ) - 1 ) - 1 / Pe ) = coth( Pe ) - 1 / Pe
    Pe := u_abs * h / ( 2 * k ) as the local Peclet number
    h as the element diameter
    k as the thermal conductivity coefficient
    u_abs as the norm of the velocity vector at the element centroid

Note: h is calculated from the affine element geometry. If your blending map changes the element diameter too drastically, this will no longer work.
""" if not coefficient_delta else ""
    coefficientstring = f"delta: coefficient (scalar space: {coefficient_function_space}){newline}" if coefficient_delta else ""
    docstring = f"""
Adiabatic heating SUPG operator for the TALA

Intended for LHS use.

The gravity vector is hard coded for annulus and icosahedral shell,
i.e. g := -x / norm(x).
{deltastring}
Geometry map: {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    w: test function (scalar space: {test})
    u: coefficient (vector space: {velocity_function_space})
    {coefficientstring}
    - ∫ delta T (u · g) ( u · ∇w )
    """

    FEM_functions = {
        "ux": velocity_function_space,
        "uy": velocity_function_space,
        "uz": velocity_function_space,
    }
    if coefficient_delta:
        FEM_functions.update({"delta": coefficient_function_space})

    from hog.recipes.integrands.volume.supg_adiabatic_heating import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=False,
        docstring=docstring,
        fe_coefficients=FEM_functions,
    )     


def advection(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    velocity_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Advection operator for the TALA.
Can be used in combination with SUPG.

Intended for LHS use.

Geometry map:    {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    s: test function  (scalar space: {test})
    u: velocity function (vectorial space: {velocity_function_space})

    ∫ ( u · ∇T ) s
"""

    from hog.recipes.integrands.volume.advection import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=False,
        docstring=docstring,
        fe_coefficients={
            "ux": velocity_function_space,
            "uy": velocity_function_space,
            "uz": velocity_function_space,
        },
    )

def supg_advection(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    velocity_function_space: FunctionSpace,
    coefficient_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
    coefficient_delta: bool = False,
) -> Form:
    newline = "\n"
    deltastring = f"""
The scaling function for the supg stabilisation is hard coded into the form.

delta( u_abs ) := h / ( 2 * u_abs ) * xi( Pe )

with:
    xi( Pe ) :=  ( 1 + 2 / ( exp( 2* Pe ) - 1 ) - 1 / Pe ) = coth( Pe ) - 1 / Pe
    Pe := u_abs * h / ( 2 * k ) as the local Peclet number
    h as the element diameter
    k as the thermal conductivity coefficient
    u_abs as the norm of the velocity vector at the element centroid

Note: h is calculated from the affine element geometry. If your blending map changes the element diameter too drastically, this will no longer work.
""" if not coefficient_delta else ""
    coefficientstring = f"delta: coefficient (scalar space: {coefficient_function_space}){newline}" if coefficient_delta else ""

    docstring = f"""
Advection SUPG operator for the TALA

Intended for LHS use.
{deltastring}
Geometry map: {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    w: test function (scalar space: {test})
    u: coefficient (vector space: {velocity_function_space})
    {coefficientstring}
    ∫ delta (u · ∇T) (u · ∇w)
"""
    
    FEM_functions = {
        "ux": velocity_function_space,
        "uy": velocity_function_space,
        "uz": velocity_function_space,
    }
    if coefficient_delta:
        FEM_functions.update({"delta": coefficient_function_space})    

    from hog.recipes.integrands.volume.supg_advection import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
        fe_coefficients=FEM_functions,
    )

def diffusion_inv_rho(
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    coefficient_function_space: Optional[FunctionSpace] = None
) -> Form:
    docstring = f"""
    Diffusion inv rho operator for the TALA
    
    Intended for LHS use.

    Geometry map: {blending}

    Weak formulation

        T: trial function (scalar space: {trial})
        w: test function (scalar space: {test})
        rho: coefficient (scalar space: {coefficient_function_space})

        - ∫ w/(rho*rho) ∇T · ∇rho         
    """

    from hog.recipes.integrands.volume.diffusion_inv_rho import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=False,
        docstring=docstring,
        fe_coefficients={
            "rho": coefficient_function_space
        },
    )

def supg_diffusion(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    velocity_function_space: FunctionSpace,
    coefficient_function_space: FunctionSpace,
    density_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
    coefficient_delta: bool = False,
    include_invrho: bool = True,
) -> Form:
    newline = "\n"
    deltastring = f"""
We make some assumptions here:
    - k is constant
    - T is in H^2 on each element ( trivially fulfilled for polynomial elements )

The scaling function for the supg stabilisation is hard coded into the form.

delta( u_abs ) := h / ( 2 * u_abs ) * xi( Pe )

with:
    xi( Pe ) :=  ( 1 + 2 / ( exp( 2* Pe ) - 1 ) - 1 / Pe ) = coth( Pe ) - 1 / Pe
    Pe := u_abs * h / ( 2 * k ) as the local Peclet number
    h as the element diameter
    k as the thermal conductivity coefficient
    u_abs as the norm of the velocity vector at the element centroid

Note: h is calculated from the affine element geometry. If your blending map changes the element diameter too drastically, this will no longer work.
    """ if not coefficient_delta else ""    
    coefficientstring_delta = f"rho: coefficient (scalar space: {density_function_space}){newline}" if include_invrho else ""
    coefficientstring_rho = f"delta: coefficient (scalar space: {coefficient_function_space}){newline}" if coefficient_delta else ""
    weakstring_rho = f" 1/rho" if include_invrho else ""
    docstring = f"""
SUPG diffusion operator for the TALA.

This corresponds to the SUPG bilinear form for both div_k_grad and diffusion_inv_rho.

Intended for LHS use.
{deltastring}
Geometry map: {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    w: test function (scalar space: {test})
    u: velocity function (vector space: {velocity_function_space})
    {coefficientstring_delta}{coefficientstring_rho}
    -∫ delta{weakstring_rho} (∇ · ∇T) (u · ∇w)
"""

    FEM_functions = {
        "ux": velocity_function_space,
        "uy": velocity_function_space,
        "uz": velocity_function_space,
    }
    if coefficient_delta:
        FEM_functions.update({"delta": coefficient_function_space})
    if include_invrho:
        FEM_functions.update({"rho": density_function_space})        

    from hog.recipes.integrands.volume.supg_diffusion import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=False,
        docstring=docstring,
        fe_coefficients=FEM_functions,
    )


def grad_rho_by_rho_dot_u(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    density_function_space: Optional[FunctionSpace] = None,
    include_inv_rho: bool = False,
    component_index: int = 0,
) -> Form:
    newline = "\n"
    coefficientstring = f"inv_rho: coefficient (scalar space: {density_function_space}){newline}" if include_inv_rho else ""
    docstring = f"""
Operator for the frozen velocity approach.

Intended for RHS use.

Geometry map: {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (space: {test})
    rho: coefficient    (space: {density_function_space})
    {coefficientstring}
    ∫ ((∇rho / rho) · u) v
"""

    from hog.recipes.integrands.volume.frozen_velocity import integrand

    FEM_functions = {
        "rho": density_function_space,
    }
    if include_inv_rho:
        FEM_functions.update({"inv_rho": density_function_space})

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        component_index=component_index,
        is_symmetric=False,
        docstring=docstring,
        fe_coefficients=FEM_functions,
    )   

def grad_rho_by_rho_dot_u_divergence(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    density_function_space: Optional[FunctionSpace] = None,
    include_inv_rho: bool = False,
    component_index: int = 0,
) -> Form:
    newline = "\n"
    coefficientstring = f"invRho: coefficient (scalar space: {density_function_space}){newline}" if include_inv_rho else ""
    docstring = f"""
Divergence + Rho stokes operator for the compressible case.

Can be used as a B Block if we want the grad_rho_rho term to be implicit.

Intended for LHS use.

Geometry map: {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (space: {test})
    rho: coefficient    (space: {density_function_space})
    {coefficientstring}
    - ∫ ( ∇ · u ) v - ∫ ( ∇rho/rho · u ) v
"""

    from hog.recipes.integrands.volume.grad_rho_rho_divergence import integrand

    FEM_functions = {
        "rho": density_function_space,
    }
    if include_inv_rho:
        FEM_functions.update({"inv_rho": density_function_space})

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        component_index=component_index,
        is_symmetric=False,
        docstring=docstring,
        fe_coefficients=FEM_functions,
    )   

def supg_mass(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    velocity_function_space: FunctionSpace,
    coefficient_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
    coefficient_delta: bool = False,
) -> Form:
    newline = "\n"
    deltastring = f"""
The scaling function for the supg stabilisation is hard coded into the form.

delta( u_abs ) := h / ( 2 * u_abs ) * xi( Pe )

with:
    xi( Pe ) :=  ( 1 + 2 / ( exp( 2* Pe ) - 1 ) - 1 / Pe ) = coth( Pe ) - 1 / Pe
    Pe := u_abs * h / ( 2 * k ) as the local Peclet number
    h as the element diameter
    k as the thermal conductivity coefficient
    u_abs as the norm of the velocity vector at the element centroid

Note: h is calculated from the affine element geometry. If your blending map changes the element diameter too drastically, this will no longer work.
""" if not coefficient_delta else ""
    coefficientstring = f"delta: coefficient (scalar space: {coefficient_function_space}){newline}" if coefficient_delta else ""

    docstring = f"""
SUPG mass operator for the TALA

Intended for LHS use.
{deltastring}
Geometry map: {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    w: test function (scalar space: {test})
    u: coefficient (vector space: {velocity_function_space})
    {coefficientstring}
    ∫ delta T ( u · ∇w )
"""
    
    FEM_functions = {
        "ux": velocity_function_space,
        "uy": velocity_function_space,
        "uz": velocity_function_space,
    }
    if coefficient_delta:
        FEM_functions.update({"delta": coefficient_function_space})    

    from hog.recipes.integrands.volume.supg_mass import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=False,
        docstring=docstring,
        fe_coefficients=FEM_functions,
    )

def rho_g_mass(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    density_function_space: Optional[FunctionSpace] = None,
    component_index: int = 0
) -> Form:
    docstring = f"""
Density scaled mass operator for the TALA
    
Intended for RHS use in case of the TALA RHS.
Intended for LHS use in case of the additional ALA pressure term.

In the second case the trial function T would act as the pressure and
we would generate a P1ToP2Vector operator.

The gravity vector is hard coded for annulus and icosahedral shell,
i.e. g := -x / norm(x).

Geometry map: {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    v: test function (vector space: {test})
    rho: coefficient (scalar space: {density_function_space})

    - ∫ rho T (g · v)
"""

    from hog.recipes.integrands.volume.rho_g_mass import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=False,
        fe_coefficients={"rho": density_function_space},
        component_index=component_index,
        docstring=docstring,
    )

def zero_form(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    from hog.recipes.integrands.volume.zero import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring="",
    )
