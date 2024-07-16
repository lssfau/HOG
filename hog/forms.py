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

from hog.element_geometry import ElementGeometry, TetrahedronElement
from hog.exception import HOGException
from hog.fem_helpers import (
    trafo_ref_to_affine,
    jac_ref_to_affine,
    jac_affine_to_physical,
    hessian_ref_to_affine,
    hessian_affine_to_blending,
    create_empty_element_matrix,
    element_matrix_iterator,
    scalar_space_dependent_coefficient,
    vector_space_dependent_coefficient,
    fem_function_on_element,
    fem_function_gradient_on_element,
)
from hog.function_space import FunctionSpace, N1E1Space
from hog.math_helpers import dot, inv, abs, det, double_contraction
from hog.quadrature import Quadrature, Tabulation
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger
from hog.blending import GeometryMap, ExternalMap, IdentityMap
from hog.integrand import process_integrand, Form


def diffusion(
    trial: FunctionSpace,
    test: FunctionSpace,
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
    if blending == IdentityMap():
        integr: Callable[..., Any] = integrand_affine
    else:
        integr: Callable[..., Any] = integrand

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
    trial: FunctionSpace,
    test: FunctionSpace,
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
    trial: FunctionSpace,
    test: FunctionSpace,
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
    if blending == IdentityMap():
        integr: Callable[..., Any] = integrand_affine
    else:
        integr: Callable[..., Any] = integrand

    return process_integrand(
        integr,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        fe_coefficients=[("k", coefficient_function_space)],
        is_symmetric=trial == test,
        docstring=docstring,
    )


def nonlinear_diffusion(
    trial: FunctionSpace,
    test: FunctionSpace,
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
        a = sp.Matrix([sp.Rational(1, 8)]) + k[0] * k[0]
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
        fe_coefficients=[("u", coefficient_function_space)],
    )


def nonlinear_diffusion_newton_galerkin(
    trial: FunctionSpace,
    test: FunctionSpace,
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
        a = sp.Matrix([sp.Rational(1, 8)]) + k[0] * k[0]
        a_prime = 2 * k[0]

        diffusion_term = a * tabulate(
            dot(jac_a_inv.T * grad_u, jac_a_inv.T * grad_v) * jac_a_abs_det
        )

        newton_galerkin_term = (
            a_prime
            * u
            * dot(jac_a_inv.T * grad_k, tabulate(jac_a_inv.T * grad_v))
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
        fe_coefficients=[("k", coefficient_function_space)],
    )


def epsilon(
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_trial: int = 0,
    component_test: int = 0,
    variable_viscosity: bool = True,
    coefficient_function_space: Optional[FunctionSpace] = None,
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

    from hog.recipes.integrands.volume.epsilon import integrand
    from hog.recipes.integrands.volume.epsilon_affine import (
        integrand as integrand_affine,
    )

    # mypy type checking supposedly cannot figure out ternaries.
    # https://stackoverflow.com/a/70832391
    if blending == IdentityMap():
        integr: Callable[..., Any] = integrand_affine
    else:
        integr: Callable[..., Any] = integrand

    return process_integrand(
        integr,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
        fe_coefficients=[("mu", coefficient_function_space)],
    )


def k_mass(
    trial: FunctionSpace,
    test: FunctionSpace,
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
        docstring=docstring,
    )


def pspg(
    trial: FunctionSpace,
    test: FunctionSpace,
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
    trial: FunctionSpace,
    test: FunctionSpace,
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
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_index: int = 0,
) -> Form:

    docstring = f"""
Divergence.

Component:    {component_index}
Geometry map: {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (scalar space:    {test})

    ∫ - ( ∇ · u ) v
"""

    from hog.recipes.integrands.volume.divergence import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=False,
        docstring=docstring,
    )


def gradient(
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_index: int = 0,
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

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=False,
        docstring=docstring,
    )


def full_stokes(
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_trial: int = 0,
    component_test: int = 0,
    variable_viscosity: bool = True,
    coefficient_function_space: Optional[FunctionSpace] = None,
) -> Form:
    docstring = f"""
Implements the fully coupled viscous operator of the Stokes problem.
The latter is the extension of the Epsilon operator to the case where
the velocity field need not be divergence-free. This is e.g. the case
in the (truncated) anelastic liquid approximation of mantle convection.

The strong representation of the operator is given by:

   - div[ μ (grad(u)+grad(u)ᵀ) ] + 2/3 grad[ μ div(u) ]

Note that the factor 2/3 means that for 2D this is the pseudo-3D form
of the operator.

Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})
    μ: coefficient    (scalar space:    {coefficient_function_space})

    ∫ μ {{ ( 2 ε(u) : ε(v) ) - (2/3) [ ( ∇ · u ) · ( ∇ · v ) ] }}
    
where
    
    ε(w) := (1/2) (∇w + (∇w)ᵀ)
"""

    from hog.recipes.integrands.volume.full_stokes import integrand

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        is_symmetric=trial == test,
        docstring=docstring,
        fe_coefficients=[("mu", coefficient_function_space)],
    )


def shear_heating(
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_trial: int = 0,
    component_test: int = 0,
    variable_viscosity: bool = True,
    viscosity_function_space: Optional[FunctionSpace] = None,
    velocity_function_space: Optional[FunctionSpace] = None,
) -> Form:
    docstring = f"""
Implements the fully coupled viscous operator for the shear heating term.
The latter is the extension of the Epsilon operator to the case where
the velocity field need not be divergence-free. This is e.g. the case
in the (truncated) anelastic liquid approximation of mantle convection.

https://doi.org/10.1111/j.1365-246X.2009.04413.x
(3) and (5)

https://doi.org/10.5194/gmd-15-5127-2022
Listing 2

The strong representation of the operator is given by:

    𝜏(u) : grad(u)
    2 {{[ μ (grad(u)+grad(u)ᵀ) / 2 ] - 1/dim [ μ div(u) ]I}} : grad(u)

Note that the factor 1/dim means that for 2D this is the pseudo-3D form
of the operator.

Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    T: trial function (scalar space:    {trial})
    s: test function  (scalar space:    {test})
    μ: coefficient    (scalar space:    {viscosity_function_space})
    u: velocity       (vectorial space: {velocity_function_space})

    ∫ {{ 2 {{[ μ (grad(u)+grad(u)ᵀ) / 2 ] - 1/3 [ μ div(u) ]I}} : grad(u) }} T_h s_h
    
The resulting matrix must be multiplied with a vector of ones to be used as the shear heating term in the RHS
"""

    if variable_viscosity == False:
        raise HOGException("Constant viscosity currently not supported.")
        # TODO fix issue with undeclared p_affines

    if geometry.dimensions < 3 and (component_trial > 1 or component_test > 1):
        return create_empty_element_matrix(trial, test, geometry)
    with TimedLogger("assembling shear heating matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = symbolizer.jac_ref_to_affine(geometry)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            jac_blending = symbolizer.jac_affine_to_blending(geometry.dimensions)
            jac_blending_inv = symbolizer.jac_affine_to_blending_inv(
                geometry.dimensions
            )
            jac_blending_det = symbolizer.abs_det_jac_affine_to_blending()
            # affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            # jac_blending = blending.jacobian(affine_coords)

        # jac_blending_inv = inv(jac_blending)
        # jac_blending_det = abs(det(jac_blending))

        ref_symbols_list = symbolizer.ref_coords_as_list(geometry.dimensions)

        mu: sp.Expr = 1
        if viscosity_function_space:
            phi_eval_symbols = tabulation.register_phi_evals(
                viscosity_function_space.shape(geometry)
            )

            mu, _ = fem_function_on_element(
                viscosity_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="mu",
                basis_eval=phi_eval_symbols,
            )
        else:
            raise HOGException(
                "scalar_space_dependent_coefficient currently not supported in opgen."
            )
            # mu = scalar_space_dependent_coefficient(
            #     "mu", geometry, symbolizer, blending=blending
            # )

        if velocity_function_space:
            phi_eval_symbols_u = tabulation.register_phi_evals(
                velocity_function_space.shape(geometry)
            )
            ux, dof_symbols_ux = fem_function_on_element(
                velocity_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="ux",
                basis_eval=phi_eval_symbols_u,
            )

            grad_ux, _ = fem_function_gradient_on_element(
                velocity_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="grad_ux",
                dof_symbols=dof_symbols_ux,
            )

            uy, dof_symbols_uy = fem_function_on_element(
                velocity_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="uy",
                basis_eval=phi_eval_symbols_u,
            )

            grad_uy, _ = fem_function_gradient_on_element(
                velocity_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="grad_uy",
                dof_symbols=dof_symbols_uy,
            )

            # if geometry.dimensions > 2:
            uz, dof_symbols_uz = fem_function_on_element(
                velocity_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="uz",
                basis_eval=phi_eval_symbols_u,
            )

            grad_uz, _ = fem_function_gradient_on_element(
                velocity_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="grad_uz",
                dof_symbols=dof_symbols_uz,
            )

        else:
            raise HOGException("velocity function needed as an external function")

        if blending != IdentityMap():
            grad_ux = jac_blending_inv.T * jac_affine_inv.T * grad_ux
            grad_uy = jac_blending_inv.T * jac_affine_inv.T * grad_uy
            grad_uz = jac_blending_inv.T * jac_affine_inv.T * grad_uz
        else:
            grad_ux = jac_affine_inv.T * grad_ux
            grad_uy = jac_affine_inv.T * grad_uy
            grad_uz = jac_affine_inv.T * grad_uz

        grad_u = grad_ux.row_join(grad_uy)

        dim = geometry.dimensions
        if dim == 2:
            u = sp.Matrix([[ux], [uy]])
        elif dim == 3:
            u = sp.Matrix([[ux], [uy], [uz]])
            grad_u = grad_u.row_join(grad_uz)

        _sym_grad_u = (grad_u + grad_u.T) / 2

        # Compute div(u)

        divdiv = grad_u.trace() * sp.eye(dim)

        tau = 2 * (_sym_grad_u - sp.Rational(1, dim) * divdiv)

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            phi = data.trial_shape
            psi = data.test_shape

            if blending != IdentityMap():
                affine_factor = (
                    tabulation.register_factor(
                        "affine_factor_symbol",
                        sp.Matrix([phi * psi * jac_affine_det]),
                    )
                )[0]
                form = (
                    mu[0]
                    * (double_contraction(tau, grad_u)[0])
                    * jac_blending_det
                    * affine_factor
                )
            else:
                shear_heating_det_symbol = (
                    tabulation.register_factor(
                        "shear_heating_det_symbol",
                        (double_contraction(tau, grad_u)) * phi * psi * jac_affine_det,
                    )
                )[0]
                form = mu[0] * shear_heating_det_symbol

            mat[data.row, data.col] = form

    return Form(
        mat,
        tabulation,
        symmetric=component_trial == component_test,
        docstring=docstring,
    )


def divdiv(
    trial: FunctionSpace,
    test: FunctionSpace,
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


def supg_diffusion(
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    velocity_function_space: FunctionSpace,
    diffusivityXdelta_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Form for SUPGDiffusion operator used for SUPG stabilisation

Geometry map: {blending}

Weak formulation

    T: trial function (space: {trial})
    s: test function  (space: {test})
    u: velocity function (space: {velocity_function_space})
   k𝛿: FE function representing k·𝛿 (space: {diffusivityXdelta_function_space})
    
    For OpGen,

    ∫ k(ΔT) · 𝛿(u · ∇s)

    -------------------

    For ExternalMap (only for testing, currently not supported),

    ∫ (ΔT) s
"""

    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble SUPG diffusion matrix."
        )

    with TimedLogger("assembling second derivative matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = symbolizer.jac_ref_to_affine(geometry)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            HOGException("ExternalMap is not supported")
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = symbolizer.jac_affine_to_blending(geometry.dimensions)
            jac_blending_inv = symbolizer.jac_affine_to_blending_inv(
                geometry.dimensions
            )
            jac_blending_det = symbolizer.abs_det_jac_affine_to_blending()
            if not isinstance(blending, IdentityMap):
                # hessian_blending_map = blending.hessian(affine_coords)
                hessian_blending_map = symbolizer.hessian_blending_map(
                    geometry.dimensions
                )

        # jac_blending_det = abs(det(jac_blending))
        # with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
        #     jac_blending_inv = inv(jac_blending)

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        if velocity_function_space != None and diffusivityXdelta_function_space != None:
            u_eval_symbols = tabulation.register_phi_evals(
                velocity_function_space.shape(geometry)
            )

            ux, _ = fem_function_on_element(
                velocity_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="ux",
                basis_eval=u_eval_symbols,
            )

            uy, _ = fem_function_on_element(
                velocity_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="uy",
                basis_eval=u_eval_symbols,
            )

            if isinstance(geometry, TetrahedronElement):
                uz, _ = fem_function_on_element(
                    velocity_function_space,
                    geometry,
                    symbolizer,
                    domain="reference",
                    function_id="uz",
                    basis_eval=u_eval_symbols,
                )
                u = sp.Matrix([[ux], [uy], [uz]])
            else:
                u = sp.Matrix([[ux], [uy]])

            kdelta_eval_symbols = tabulation.register_phi_evals(
                diffusivityXdelta_function_space.shape(geometry)
            )

            kdelta, _ = fem_function_on_element(
                diffusivityXdelta_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="kdelta",
                basis_eval=kdelta_eval_symbols,
            )

        for data in it:
            psi = data.test_shape
            grad_phi = data.trial_shape_grad
            grad_psi = data.test_shape_grad
            hessian_phi = data.trial_shape_hessian
            hessian_affine = hessian_ref_to_affine(
                geometry, hessian_phi, jac_affine_inv
            )

            hessian_affine_symbols = tabulation.register_factor(
                "hessian_affine",
                hessian_affine,
            )

            jac_affine_inv_T_grad_phi_symbols = tabulation.register_factor(
                "jac_affine_inv_T_grad_phi",
                jac_affine_inv.T * grad_phi,
            )

            jac_affine_inv_T_grad_psi_symbols = tabulation.register_factor(
                "jac_affine_inv_T_grad_psi",
                jac_affine_inv.T * grad_psi,
            )

            # jac_blending_inv_T_jac_affine_inv_T_grad_psi_symbols = tabulation.register_factor(
            #     "jac_affine_inv_T_grad_psi",
            #     jac_blending_inv.T * jac_affine_inv_T_grad_psi_symbols,
            # )

            if isinstance(blending, IdentityMap):
                laplacian = sum(
                    [hessian_affine_symbols[i, i] for i in range(geometry.dimensions)]
                )
                form = (
                    laplacian
                    * dot(u, jac_affine_inv_T_grad_psi_symbols)
                    * kdelta
                    * jac_affine_det
                )
            else:
                hessian_blending = hessian_affine_to_blending(
                    geometry,
                    hessian_affine,
                    hessian_blending_map,
                    jac_blending_inv.T,
                    jac_affine_inv_T_grad_phi_symbols,
                )

                laplacian = sum(
                    [hessian_blending[i, i] for i in range(geometry.dimensions)]
                )

                form = (
                    laplacian
                    * dot(u, jac_blending_inv.T * jac_affine_inv.T * grad_psi)
                    * kdelta
                    * jac_affine_det
                    * jac_blending_det
                )
                # HOGException("Only for testing with Blending map")

            mat[data.row, data.col] = form

    return Form(mat, tabulation, symmetric=False, docstring=docstring)


def zero_form(
    trial: FunctionSpace,
    test: FunctionSpace,
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
