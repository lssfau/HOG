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
    TensorialVectorFunctionSpace,
)
from hog.math_helpers import dot, inv, abs, det, double_contraction
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
    trial: TensorialVectorFunctionSpace,
    test: TensorialVectorFunctionSpace,
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
    trial: TensorialVectorFunctionSpace,
    test: TensorialVectorFunctionSpace,
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
        fe_coefficients={"mu": coefficient_function_space},
    )


def shear_heating(
    trial: TrialSpace,
    test: TestSpace,
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

    𝜏(w) : grad(w)
    2 {{[ μ (grad(w)+grad(w)ᵀ) / 2 ] - 1/dim [ μ div(w) ]I}} : grad(w)

Note that the factor 1/dim means that for 2D this is the pseudo-3D form
of the operator.

Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    T: trial function (scalar space:    {trial})
    s: test function  (scalar space:    {test})
    μ: coefficient    (scalar space:    {viscosity_function_space})
    w: velocity       (vectorial space: {velocity_function_space})

    ∫ {{ 2 {{[ μ (grad(w)+grad(w)ᵀ) / 2 ] - 1/dim [ μ div(w) ]I}} : grad(w) }} T_h s_h
    
The resulting matrix must be multiplied with a vector of ones to be used as the shear heating term in the RHS
"""

    def integrand(
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
        **_,
    ):
        """First function: mu, other functions: ux, uy, uz."""

        mu = k["mu"]

        # grad_k[0] is grad_mu_ref
        grad_wx = jac_b_inv.T * jac_a_inv.T * grad_k["wx"]
        grad_wy = jac_b_inv.T * jac_a_inv.T * grad_k["wy"]
        grad_wz = jac_b_inv.T * jac_a_inv.T * grad_k["wz"]

        grad_w = grad_wx.row_join(grad_wy)
        dim = volume_geometry.dimensions
        if dim == 3:
            grad_w = grad_w.row_join(grad_wz)

        sym_grad_w = 0.5 * (grad_w + grad_w.T)

        divdiv = grad_w.trace() * sp.eye(dim)

        tau = 2 * (sym_grad_w - sp.Rational(1, dim) * divdiv)

        return (
            mu
            * (double_contraction(tau, grad_w)[0])
            * jac_b_abs_det
            * tabulate(jac_a_abs_det * u * v)
        )

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        fe_coefficients={
            "mu": viscosity_function_space,
            "wx": velocity_function_space,
            "wy": velocity_function_space,
            "wz": velocity_function_space,
        },
        is_symmetric=trial == test,
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


def advection(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    velocity_function_space: FunctionSpace,
    coefficient_function_space: FunctionSpace,
    constant_cp: bool = False,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
advection operator which needs to be used in combination with SUPG

Geometry map:    {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    s: test function  (scalar space: {test})
    u: velocity function (vectorial space: {velocity_function_space})

    ∫ cp ( u · ∇T ) s
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
        }
        if constant_cp
        else {
            "ux": velocity_function_space,
            "uy": velocity_function_space,
            "uz": velocity_function_space,
            "cp": coefficient_function_space,
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
) -> Form:
    docstring = f"""
advection operator which needs to be used in combination with SUPG

Geometry map:    {blending}

Weak formulation

    T: trial function (scalar space: {trial})
    s: test function  (scalar space: {test})
    u: velocity function (vectorial space: {velocity_function_space})

    ∫ cp ( u · ∇T ) 𝛿(u · ∇s)
"""

    from hog.recipes.integrands.volume.supg_advection import integrand

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
            "cp_times_delta": coefficient_function_space,
        },
    )


def supg_diffusion(
    trial: TrialSpace,
    test: TestSpace,
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
    w: velocity function (space: {velocity_function_space})
   k𝛿: FE function representing k·𝛿 (space: {diffusivityXdelta_function_space})
    
    For OpGen,

    ∫ k(ΔT) · 𝛿(w · ∇s)

    -------------------

    For ExternalMap (only for testing, currently not supported),

    ∫ (ΔT) s
"""

    def integrand(
        *,
        jac_a_inv,
        jac_b_inv,
        hessian_b,
        jac_a_abs_det,
        jac_b_abs_det,
        grad_u,
        grad_v,
        hessian_u,
        k,
        volume_geometry,
        tabulate,
        **_,
    ):
        """First function: k𝛿, other functions: ux, uy, uz."""

        k_times_delta = k["diffusivity_times_delta"]
        wx = k["wx"]
        wy = k["wy"]
        wz = k["wz"]

        dim = volume_geometry.dimensions
        if dim == 2:
            w = sp.Matrix([[wx], [wy]])
        elif dim == 3:
            w = sp.Matrix([[wx], [wy], [wz]])

        if isinstance(blending, IdentityMap):
            hessian_affine = hessian_shape_affine_ref_pullback(hessian_u, jac_a_inv)

            laplacian = sum([hessian_affine[i, i] for i in range(geometry.dimensions)])

            form = (
                k_times_delta
                * tabulate(laplacian)
                * dot(w, tabulate(jac_a_inv.T * grad_v))
                * jac_a_abs_det
            )

        else:
            hessian_blending = hessian_shape_blending_ref_pullback(
                geometry,
                grad_u,
                hessian_u,
                jac_a_inv,
                hessian_b,
                jac_b_inv,
            )

            laplacian = sum(
                [hessian_blending[i, i] for i in range(geometry.dimensions)]
            )

            form = (
                laplacian
                * dot(w, jac_b_inv.T * tabulate(jac_a_inv.T * grad_v))
                * k_times_delta
                * jac_a_abs_det
                * jac_b_abs_det
            )

        return form

    return process_integrand(
        integrand,
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        fe_coefficients={
            "diffusivity_times_delta": diffusivityXdelta_function_space,
            "wx": velocity_function_space,
            "wy": velocity_function_space,
            "wz": velocity_function_space,
        },
    )


def grad_rho_by_rho_dot_u(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    density_function_space: Optional[FunctionSpace] = None,
) -> Form:
    docstring = f"""
RHS operator for the frozen velocity approach.

Geometry map: {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (space: {test})
    rho: coefficient    (space: {density_function_space})

    ∫ ((∇ρ / ρ) · u) v
"""

    from hog.recipes.integrands.volume.frozen_velocity import integrand

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
            "rho": density_function_space,
        },
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
