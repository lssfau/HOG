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

from dataclasses import dataclass
import logging
import sympy as sp
from typing import List, Optional, Tuple

from hog.ast import Operations, count_operations
from hog.element_geometry import ElementGeometry
from hog.exception import HOGException
from hog.fem_helpers import (
    trafo_ref_to_affine,
    jac_ref_to_affine,
    jac_affine_to_physical,
    create_empty_element_matrix,
    element_matrix_iterator,
    scalar_space_dependent_coefficient,
    vector_space_dependent_coefficient,
    fem_function_on_element,
    fem_function_gradient_on_element,
)
from hog.function_space import FunctionSpace, EnrichedGalerkinFunctionSpace, N1E1Space
from hog.math_helpers import dot, grad, inv, abs, det, double_contraction, e_vec
from hog.quadrature import Quadrature, Tabulation
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger, get_logger
from hog.blending import GeometryMap, ExternalMap, IdentityMap


@dataclass
class Form:
    integrand: sp.MatrixBase
    tabulation: Tabulation
    symmetric: bool
    docstring: str = ""

    def integrate(self, quad: Quadrature, symbolizer: Symbolizer) -> sp.Matrix:
        """Integrates the form using the passed quadrature directly, i.e. without tabulations or loops."""
        mat = self.tabulation.inline_tables(self.integrand)

        for row in range(mat.rows):
            for col in range(mat.cols):
                if self.symmetric and row > col:
                    mat[row, col] = mat[col, row]
                else:
                    mat[row, col] = quad.integrate(mat[row, col], symbolizer)

        return mat


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
"""

    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble diffusion matrix."
        )

    with TimedLogger("assembling diffusion matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = symbolizer.jac_ref_to_affine(geometry.dimensions)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry.dimensions)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
            jac_blending_det = abs(det(jac_blending))
            with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
                jac_blending_inv = inv(jac_blending)
        else:
            # affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            # jac_blending = blending.jacobian(affine_coords)
            jac_blending = symbolizer.jac_affine_to_blending(geometry.dimensions)
            jac_blending_inv = symbolizer.jac_affine_to_blending_inv(
                geometry.dimensions
            )
            jac_blending_det = symbolizer.abs_det_jac_affine_to_blending()

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            grad_phi = data.trial_shape_grad
            grad_psi = data.test_shape_grad
            if blending != IdentityMap():
                jac_affine_inv_T_grad_phi_symbols = tabulation.register_factor(
                    "jac_affine_inv_T_grad_phi",
                    jac_affine_inv.T * grad_phi,
                )
                jac_affine_inv_T_grad_psi_symbols = tabulation.register_factor(
                    "jac_affine_inv_T_grad_psi",
                    jac_affine_inv.T * grad_psi,
                )
                form = (
                    double_contraction(
                        jac_blending_inv.T
                        * sp.Matrix(jac_affine_inv_T_grad_phi_symbols),
                        jac_blending_inv.T
                        * sp.Matrix(jac_affine_inv_T_grad_psi_symbols),
                    )
                    * jac_affine_det
                    * jac_blending_det
                )
            else:
                jac_affine_inv_grad_phi_jac_affine_inv_grad_psi_det_symbol = (
                    tabulation.register_factor(
                        "jac_affine_inv_grad_phi_jac_affine_inv_grad_psi_det",
                        double_contraction(
                            jac_affine_inv.T * grad_phi,
                            jac_affine_inv.T * grad_psi,
                        )
                        * jac_affine_det,
                    )
                )[0]
                form = jac_affine_inv_grad_phi_jac_affine_inv_grad_psi_det_symbol

            mat[data.row, data.col] = form

    return Form(mat, tabulation, symmetric=True, docstring=docstring)


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
    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble mass matrix."
        )

    with TimedLogger("assembling mass matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        # TODO tabulate when blending is implemented in the FOG,
        # without blending move_constants is enough

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                phi = data.trial_shape
                psi = data.test_shape
                phi_psi_det_jac_aff = tabulation.register_factor(
                    "phi_psi_det_jac_aff", sp.Matrix([phi * psi * jac_affine_det])
                )[0]
                if blending != IdentityMap():
                    form = phi_psi_det_jac_aff * jac_blending_det
                    mat[data.row, data.col] = form
                else:
                    form = phi_psi_det_jac_aff
                    mat[data.row, data.col] = form

    return Form(mat, tabulation, symmetric=True, docstring=docstring)


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

    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble diffusion matrix."
        )

    with TimedLogger("assembling div-k-grad matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = symbolizer.jac_ref_to_affine(geometry.dimensions)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry.dimensions)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))
        with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
            jac_blending_inv = inv(jac_blending)

        if coefficient_function_space:
            phi_eval_symbols = tabulation.register_phi_evals(
                coefficient_function_space.shape(geometry)
            )

            k, _ = fem_function_on_element(
                coefficient_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="k",
                basis_eval=phi_eval_symbols,
            )
        else:
            k = scalar_space_dependent_coefficient(
                "k", geometry, symbolizer, blending=blending
            )

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            if blending != IdentityMap():
                # the following factors of the weak form can always be tabulated
                jac_affine_inv_T_grad_phi_symbols = tabulation.register_factor(
                    "jac_affine_inv_T_grad_phi",
                    jac_affine_inv.T * data.trial_shape_grad,
                )
                jac_affine_inv_T_grad_psi_symbols = tabulation.register_factor(
                    "jac_affine_inv_T_grad_psi",
                    jac_affine_inv.T * data.test_shape_grad,
                )
                form = (
                    k
                    * dot(
                        jac_blending_inv.T
                        * sp.Matrix(jac_affine_inv_T_grad_phi_symbols),
                        jac_blending_inv.T
                        * sp.Matrix(jac_affine_inv_T_grad_psi_symbols),
                    )
                    * jac_affine_det
                    * jac_blending_det
                )
            else:
                jac_affine_inv_grad_phi_jac_affine_inv_grad_psi_det_symbol = (
                    tabulation.register_factor(
                        "jac_affine_inv_grad_phi_jac_affine_inv_grad_psi_det",
                        dot(
                            jac_affine_inv.T * data.test_shape_grad,
                            jac_affine_inv.T * data.trial_shape_grad,
                        )
                        * jac_affine_det,
                    )
                )[0]
                form = k * jac_affine_inv_grad_phi_jac_affine_inv_grad_psi_det_symbol

            mat[data.row, data.col] = form

    return Form(mat, tabulation, symmetric=True, docstring=docstring)


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

    with TimedLogger("assembling non-linear diffusion matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = symbolizer.jac_ref_to_affine(geometry.dimensions)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry.dimensions)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))
        with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
            jac_blending_inv = inv(jac_blending)

        phi_eval_symbols = tabulation.register_phi_evals(trial.shape(geometry))

        if coefficient_function_space:
            u, _ = fem_function_on_element(
                coefficient_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="u",
                basis_eval=phi_eval_symbols,
            )
        else:
            raise HOGException("Not implemented.")

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            grad_phi = jac_affine_inv.T * data.trial_shape_grad
            grad_psi = jac_affine_inv.T * data.test_shape_grad

            a = sp.Matrix([sp.Rational(1, 8)]) + u * u

            dot_grad_phi_grad_psi_symbol = tabulation.register_factor(
                "dot_grad_phi_grad_psi", dot(grad_phi, grad_psi) * jac_affine_det
            )[0]

            mat[data.row, data.col] = a * dot_grad_phi_grad_psi_symbol

    return Form(mat, tabulation, symmetric=True, docstring=docstring)


def nonlinear_diffusion_newton_galerkin(
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    coefficient_function_space: FunctionSpace,
    blending: GeometryMap = IdentityMap(),
    onlyNewtonGalerkinPartOfForm: Optional[bool] = True,
) -> Form:
    docstring = f"""

Bi-linear form for the solution of the non-linear diffusion equation by a Newton-Galerkin approach

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    c: FE coefficient function (space: {coefficient_function_space})
    
    ∫ a(c) ∇u · ∇v + ∫ a'(c) u ∇c · ∇v

Note: :math:`a(c) = 1/8 + u^2` is currently hard-coded and the form is intended for :math:`c = u`.
"""
    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble diffusion matrix."
        )

    if blending != IdentityMap():
        raise HOGException(
            "The nonlinear_diffusion_newton_galerkin form does currently not support blending."
        )

    with TimedLogger(
        "assembling nonlinear_diffusion_newton_galerkin matrix", level=logging.DEBUG
    ):
        tabulation = Tabulation(symbolizer)

        jac_affine = symbolizer.jac_ref_to_affine(geometry.dimensions)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry.dimensions)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))
        with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
            jac_blending_inv = inv(jac_blending)

        phi_eval_symbols = tabulation.register_phi_evals(trial.shape(geometry))

        u, dof_symbols = fem_function_on_element(
            coefficient_function_space,
            geometry,
            symbolizer,
            domain="reference",
            function_id="u",
            basis_eval=phi_eval_symbols,
        )

        grad_u, _ = fem_function_gradient_on_element(
            coefficient_function_space,
            geometry,
            symbolizer,
            domain="reference",
            function_id="grad_u",
            dof_symbols=dof_symbols,
        )

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            phi = data.trial_shape
            grad_psi = jac_affine_inv.T * data.test_shape_grad

            a = sp.Matrix([sp.Rational(1, 8)]) + u * u

            aPrime = 2 * u

            grad_psi_symbol = tabulation.register_factor("grad_psi", grad_psi)

            if onlyNewtonGalerkinPartOfForm:
                mat[data.row, data.col] = (
                    aPrime
                    * phi
                    * dot(jac_affine_inv.T * grad_u, grad_psi_symbol)
                    * jac_affine_det
                )
            else:
                grad_phi = jac_affine_inv.T * data.trial_shape_grad

                dot_grad_phi_grad_psi_symbol = tabulation.register_factor(
                    "dot_grad_phi_grad_psi", dot(grad_phi, grad_psi) * jac_affine_det
                )[0]

                mat[data.row, data.col] = (
                    a * dot_grad_phi_grad_psi_symbol
                    + aPrime
                    * phi
                    * dot(jac_affine_inv.T * grad_u, grad_psi_symbol)
                    * jac_affine_det
                )

    return Form(mat, tabulation, symmetric=False, docstring=docstring)


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

Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})
    μ: coefficient    (scalar space:    {coefficient_function_space})

    ∫ 2 μ ε(u) : ε(v)
    
where
    
    ε(w) := (1/2) (∇w + (∇w)ᵀ)
"""
    if variable_viscosity == False:
        raise HOGException("Constant viscosity currently not supported.")
        # TODO fix issue with undeclared p_affines

    if geometry.dimensions < 3 and (component_trial > 1 or component_test > 1):
        return create_empty_element_matrix(trial, test, geometry)
    with TimedLogger("assembling epsilon matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = symbolizer.jac_ref_to_affine(geometry.dimensions)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry.dimensions)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
            jac_blending_inv = inv(jac_blending)
        jac_blending_det = abs(det(jac_blending))

        ref_symbols_list = symbolizer.ref_coords_as_list(geometry.dimensions)

        mu: sp.Expr = 1
        if variable_viscosity:
            if coefficient_function_space:
                phi_eval_symbols = tabulation.register_phi_evals(
                    coefficient_function_space.shape(geometry)
                )

                mu, _ = fem_function_on_element(
                    coefficient_function_space,
                    geometry,
                    symbolizer,
                    domain="reference",
                    function_id="mu",
                    basis_eval=phi_eval_symbols,
                )
            else:
                mu = scalar_space_dependent_coefficient(
                    "mu", geometry, symbolizer, blending=blending
                )

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            phi = data.trial_shape
            psi = data.test_shape
            grad_phi_vec = data.trial_shape_grad
            grad_psi_vec = data.test_shape_grad

            # EG gradient transformation TODO move to function space
            if isinstance(trial, EnrichedGalerkinFunctionSpace):
                # for EDG, the shape function is already vectorial and does not have to be multiplied by e_vec
                grad_phi_vec = jac_affine * grad_phi_vec
            else:
                grad_phi_vec = (
                    (e_vec(geometry.dimensions, component_trial) * phi)
                    .jacobian(ref_symbols_list)
                    .T
                )
            # same for test space
            if isinstance(test, EnrichedGalerkinFunctionSpace):
                # for EDG, the shape function is already vectorial and does not have to be multiplied by e_vec
                grad_psi_vec = jac_affine * grad_psi_vec
            else:
                grad_psi_vec = (
                    (e_vec(geometry.dimensions, component_test) * psi)
                    .jacobian(ref_symbols_list)
                    .T
                )

            # setup of the form expression with tabulation
            if blending != IdentityMap():
                # chain rule, premultiply with transposed inverse jacobians of the affine trafo
                # the results are tensors of order 2
                # + tabulate affine transformed gradients (can only do this due to incoming, micro-element dependent blending jacobian)
                jac_affine_inv_T_grad_phi_symbols = sp.Matrix(
                    tabulation.register_factor(
                        "jac_affine_inv_T_grad_phi",
                        jac_affine_inv.T * grad_phi_vec,
                    )
                )
                jac_affine_inv_T_grad_psi_symbols = sp.Matrix(
                    tabulation.register_factor(
                        "jac_affine_inv_T_grad_psi",
                        jac_affine_inv.T * grad_psi_vec,
                    )
                )

                # transform gradients according to blending map
                jac_blending_T_jac_affine_inv_T_grad_phi = (
                    jac_blending_inv.T * jac_affine_inv_T_grad_phi_symbols
                )
                jac_blending_T_jac_affine_inv_T_grad_psi = (
                    jac_blending_inv.T * jac_affine_inv_T_grad_psi_symbols
                )

                # extract the symmetric part
                sym_grad_phi = 0.5 * (
                    jac_blending_T_jac_affine_inv_T_grad_phi
                    + jac_blending_T_jac_affine_inv_T_grad_phi.T
                )
                sym_grad_psi = 0.5 * (
                    jac_blending_T_jac_affine_inv_T_grad_psi
                    + jac_blending_T_jac_affine_inv_T_grad_psi.T
                )

                # double contract everything + determinants
                form = (
                    double_contraction(2 * mu[0, 0] * sym_grad_phi, sym_grad_psi)
                    * jac_affine_det
                    * jac_blending_det
                )

            else:
                # chain rule, premultiply with transposed inverse jacobians of affine trafo
                # the results are tensors of order 2
                jac_affine_inv_T_grad_phi = jac_affine_inv.T * grad_phi_vec
                jac_affine_inv_T_grad_psi = jac_affine_inv.T * grad_psi_vec

                # now let's extract the symmetric part
                sym_grad_phi = 0.5 * (
                    jac_affine_inv_T_grad_phi + jac_affine_inv_T_grad_phi.T
                )
                sym_grad_psi = 0.5 * (
                    jac_affine_inv_T_grad_psi + jac_affine_inv_T_grad_psi.T
                )

                # double contract everything + determinants, tabulate the whole contraction
                # TODO maybe shorten naming, although its nice to have everything in the name
                contract_2_jac_affine_inv_sym_grad_phi_jac_affine_inv_sym_grad_psi_det_symbol = (
                    tabulation.register_factor(
                        "contract_2_jac_affine_inv_sym_grad_phi_jac_affine_inv_sym_grad_psi_det_symbol",
                        double_contraction(2 * sym_grad_phi, sym_grad_psi)
                        * jac_affine_det,
                    )
                )[
                    0
                ]
                form = (
                    mu
                    * contract_2_jac_affine_inv_sym_grad_phi_jac_affine_inv_sym_grad_psi_det_symbol
                )

            mat[data.row, data.col] = form

    return Form(
        mat,
        tabulation,
        symmetric=component_trial == component_test,
        docstring=docstring,
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

    if trial != test:
        TimedLogger(
            "Trial and test space can be different, but please make sure this is intensional!",
            level=logging.INFO
        ).log()

    with TimedLogger("assembling k-mass matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))

        mat = create_empty_element_matrix(trial, test, geometry)

        it = element_matrix_iterator(trial, test, geometry)

        if coefficient_function_space:
            k, _ = fem_function_on_element(
                coefficient_function_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="k",
            )
        else:
            k = scalar_space_dependent_coefficient(
                "k", geometry, symbolizer, blending=blending
            )

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                phi = data.trial_shape
                psi = data.test_shape

                phi_psi_jac_affine_det = tabulation.register_factor(
                    "phi_psi_jac_affine_det",
                    sp.Matrix([phi * psi * jac_affine_det]),
                )[0]
                if blending == IdentityMap():
                    form = k * phi_psi_jac_affine_det
                else:
                    form = k * phi_psi_jac_affine_det * jac_blending_det
                mat[data.row, data.col] = form

    return Form(mat, tabulation, symmetric=trial == test, docstring=docstring)


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

    if quad.is_exact() and isinstance(blending, ExternalMap):
        raise HOGException(
            "Exact integration is not supported for externally defined blending functions."
        )

    with TimedLogger("assembling diffusion matrix", level=logging.DEBUG):
        jac_affine = jac_ref_to_affine(geometry, symbolizer)
        with TimedLogger("inverting affine Jacobian", level=logging.DEBUG):
            jac_affine_inv = inv(jac_affine)
        jac_affine_det = abs(det(jac_affine))

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))
        with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
            jac_blending_inv = inv(jac_blending)

        if geometry.dimensions == 2:
            volume = jac_blending_det * 0.5 * jac_affine_det
            tau = -volume * 0.2
        else:
            volume = jac_blending_det * jac_affine_det / 6.0
            tau = -pow(volume, 2.0 / 3.0) / 12.0

        mat = create_empty_element_matrix(trial, test, geometry)

        it = element_matrix_iterator(trial, test, geometry)
        # TODO tabulate

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                grad_phi = data.trial_shape_grad
                grad_psi = data.test_shape_grad
                form = (
                    dot(
                        jac_blending_inv.T * jac_affine_inv.T * grad_phi,
                        jac_blending_inv.T * jac_affine_inv.T * grad_psi,
                    )
                    * jac_affine_det
                    * tau
                    * jac_blending_det
                )
                mat[data.row, data.col] = quad.integrate(form, symbolizer)

    return Form(mat, Tabulation(symbolizer), symmetric=True, docstring=docstring)


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
    transpose: bool = False,
) -> Form:
    if transpose:
        docstring = f"""
Gradient.

Component:    {component_index}
Geometry map: {blending}

Weak formulation

    u: trial function (scalar space:    {trial})
    v: test function  (vectorial space: {test})

    ∫ - ( ∇ · v ) u
"""
    else:
        docstring = f"""
Divergence.

Component:    {component_index}
Geometry map: {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (scalar space:    {test})

    ∫ - ( ∇ · u ) v
"""

    with TimedLogger(
        f"assembling divergence {'transpose' if transpose else ''} matrix",
        level=logging.DEBUG,
    ):
        jac_affine = symbolizer.jac_ref_to_affine(geometry.dimensions)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry.dimensions)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))
        with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
            jac_blending_inv = inv(jac_blending)

        mat = create_empty_element_matrix(trial, test, geometry)

        it = element_matrix_iterator(trial, test, geometry)
        # TODO tabulate

        # guard in 2D against the z derivative, which is not defined here:
        component_index = min(component_index, geometry.dimensions - 1)

        for data in it:
            if transpose:
                phi = data.trial_shape
                grad_phi = data.test_shape_grad
            else:
                phi = data.test_shape
                grad_phi = data.trial_shape_grad
            form = (
                -(jac_blending_inv.T * jac_affine_inv.T * grad_phi)[component_index]
                * phi
                * jac_affine_det
                * jac_blending_det
            )

            mat[data.row, data.col] = form

    return Form(mat, Tabulation(symbolizer), symmetric=False, docstring=docstring)


def gradient(
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_index: int = 0,
) -> Form:
    """See divergence form. Just calls that with the transpose argument set to True."""
    return divergence(
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
        component_index=component_index,
        transpose=True,
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

    if variable_viscosity == False:
        raise HOGException("Constant viscosity currently not supported.")
        # TODO fix issue with undeclared p_affines

    if geometry.dimensions < 3 and (component_trial > 1 or component_test > 1):
        return create_empty_element_matrix(trial, test, geometry)
    with TimedLogger("assembling full stokes matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = symbolizer.jac_ref_to_affine(geometry.dimensions)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry.dimensions)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_inv = inv(jac_blending)
        jac_blending_det = abs(det(jac_blending))

        ref_symbols_list = symbolizer.ref_coords_as_list(geometry.dimensions)

        mu: sp.Expr = 1
        if variable_viscosity:
            if coefficient_function_space:
                phi_eval_symbols = tabulation.register_phi_evals(
                    coefficient_function_space.shape(geometry)
                )

                mu, _ = fem_function_on_element(
                    coefficient_function_space,
                    geometry,
                    symbolizer,
                    domain="reference",
                    function_id="mu",
                    basis_eval=phi_eval_symbols,
                )
            else:
                mu = scalar_space_dependent_coefficient(
                    "mu", geometry, symbolizer, blending=blending
                )

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            phi = data.trial_shape
            psi = data.test_shape
            grad_phi_vec = data.trial_shape_grad
            grad_psi_vec = data.test_shape_grad

            # gradient of e_i * phi, where i is the trial space component
            # results in a order 2 tensor,
            # equal to the transposed Jacobian of e_i * phi
            grad_phi_vec = (
                (e_vec(geometry.dimensions, component_trial) * phi)
                .jacobian(ref_symbols_list)
                .T
            )

            # same for test space
            grad_psi_vec = (
                (e_vec(geometry.dimensions, component_test) * psi)
                .jacobian(ref_symbols_list)
                .T
            )

            # setup of the form expression with tabulation
            if blending != IdentityMap():
                # chain rule, premultiply with transposed inverse jacobians of the affine trafo
                # the results are tensors of order 2
                # + tabulate affine transformed gradients (can only do this due to incoming, micro-element dependent blending jacobian)
                jac_affine_inv_T_grad_phi_symbols = sp.Matrix(
                    tabulation.register_factor(
                        "jac_affine_inv_T_grad_phi",
                        jac_affine_inv.T * grad_phi_vec,
                    )
                )
                jac_affine_inv_T_grad_psi_symbols = sp.Matrix(
                    tabulation.register_factor(
                        "jac_affine_inv_T_grad_psi",
                        jac_affine_inv.T * grad_psi_vec,
                    )
                )

                # transform gradients according to blending map
                jac_blending_T_jac_affine_inv_T_grad_phi = (
                    jac_blending_inv.T * jac_affine_inv_T_grad_phi_symbols
                )
                jac_blending_T_jac_affine_inv_T_grad_psi = (
                    jac_blending_inv.T * jac_affine_inv_T_grad_psi_symbols
                )

                # extract the symmetric part
                sym_grad_phi = 0.5 * (
                    jac_blending_T_jac_affine_inv_T_grad_phi
                    + jac_blending_T_jac_affine_inv_T_grad_phi.T
                )
                sym_grad_psi = 0.5 * (
                    jac_blending_T_jac_affine_inv_T_grad_psi
                    + jac_blending_T_jac_affine_inv_T_grad_psi.T
                )

                # form divdiv part
                # ( div(e_i*phi), div(e_j*psi) )_Omega results in
                #
                # ( \partial phi / \partial x_i ) * ( \partial psi / \partial x_j )
                #
                # for which we have to take the distortions by the two mappings into account
                divdiv = sp.Matrix(
                    [
                        jac_blending_T_jac_affine_inv_T_grad_phi[
                            component_trial, component_trial
                        ]
                        * jac_blending_T_jac_affine_inv_T_grad_psi[
                            component_test, component_test
                        ]
                    ]
                )

                # double contract sym grads + divdiv + determinants
                form = (
                    mu
                    * (
                        double_contraction(2 * sym_grad_phi, sym_grad_psi)
                        - sp.Rational(2, 3) * divdiv
                    )
                    * jac_affine_det
                    * jac_blending_det
                )

            else:
                # chain rule, premultiply with transposed inverse jacobians of affine trafo
                # the results are tensors of order 2
                jac_affine_inv_T_grad_phi = jac_affine_inv.T * grad_phi_vec
                jac_affine_inv_T_grad_psi = jac_affine_inv.T * grad_psi_vec

                # now let's extract the symmetric part
                sym_grad_phi = 0.5 * (
                    jac_affine_inv_T_grad_phi + jac_affine_inv_T_grad_phi.T
                )
                sym_grad_psi = 0.5 * (
                    jac_affine_inv_T_grad_psi + jac_affine_inv_T_grad_psi.T
                )

                divdiv = sp.Matrix(
                    [
                        jac_affine_inv_T_grad_phi[component_trial, component_trial]
                        * jac_affine_inv_T_grad_psi[component_test, component_test]
                    ]
                )

                # double contract sym grads + divdiv + determinants + tabulate the whole expression
                # TODO maybe shorten naming, although its nice to have everything in the name
                contract_2_jac_affine_inv_sym_grad_phi_jac_affine_inv_sym_grad_psi__min_2third_divdiv_det_symbol = (
                    tabulation.register_factor(
                        "contract_2_jac_affine_inv_sym_grad_phi_jac_affine_inv_sym_grad_psi_det_plus_min2third_divdiv",
                        (
                            double_contraction(2 * sym_grad_phi, sym_grad_psi)
                            - sp.Rational(2, 3) * divdiv
                        )
                        * jac_affine_det,
                    )
                )[
                    0
                ]
                form = (
                    mu
                    * contract_2_jac_affine_inv_sym_grad_phi_jac_affine_inv_sym_grad_psi__min_2third_divdiv_det_symbol
                )

            mat[data.row, data.col] = form

    return Form(
        mat,
        tabulation,
        symmetric=component_trial == component_test,
        docstring=docstring,
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

        jac_affine = symbolizer.jac_ref_to_affine(geometry.dimensions)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry.dimensions)
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
            raise HOGException("scalar_space_dependent_coefficient currently not supported in opgen.")
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
                )[
                    0
                ]
                form = (
                    mu[0]
                    * (
                        double_contraction(tau, grad_u)[0]
                    )
                    * jac_blending_det
                    * affine_factor
                )
            else:
                shear_heating_det_symbol = (
                    tabulation.register_factor(
                        "shear_heating_det_symbol",
                        (
                            double_contraction(tau, grad_u)
                        )
                        * phi
                        * psi 
                        * jac_affine_det,
                    )
                )[
                    0
                ]
                form = (
                    mu[0] * shear_heating_det_symbol
                )

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

    if geometry.dimensions < 3 and (component_trial > 1 or component_test > 1):
        return create_empty_element_matrix(trial, test, geometry)

    jac_affine = jac_ref_to_affine(geometry, symbolizer)
    jac_affine_inv = inv(jac_affine)
    jac_affine_det = abs(det(jac_affine))

    if isinstance(blending, ExternalMap):
        jac_blending = jac_affine_to_physical(geometry, symbolizer)
    else:
        affine_coords = trafo_ref_to_affine(geometry, symbolizer)
        jac_blending = blending.jacobian(affine_coords)

    jac_blending_inv = inv(jac_blending)
    jac_blending_det = abs(det(jac_blending))

    mat = create_empty_element_matrix(trial, test, geometry)

    it = element_matrix_iterator(trial, test, geometry)

    ref_symbols_list = symbolizer.ref_coords_as_list(geometry.dimensions)
    # TODO tabulate

    with TimedLogger(
        f"integrating {mat.shape[0] * mat.shape[1]} expressions",
        level=logging.DEBUG,
    ):
        for data in it:
            phi = data.trial_shape
            psi = data.test_shape

            # gradient of e_i * phi, where i is the trial space component
            # results in a order 2 tensor,
            # equal to the transposed Jacobian of e_i * phi
            grad_phi_vec = (
                (e_vec(geometry.dimensions, component_trial) * phi)
                .jacobian(ref_symbols_list)
                .T
            )

            # same for test space
            grad_psi_vec = (
                (e_vec(geometry.dimensions, component_test) * psi)
                .jacobian(ref_symbols_list)
                .T
            )

            # chain rule, premultiply with transposed inverse jacobians of blending and affine trafos
            # the results are tensors of order 2
            grad_phi_vec_chain = jac_blending_inv.T * jac_affine_inv.T * grad_phi_vec
            grad_psi_vec_chain = jac_blending_inv.T * jac_affine_inv.T * grad_psi_vec

            # ( div(e_i*phi), div(e_j*psi) )_Omega results in
            #
            # ( \partial phi / \partial x_i ) * ( \partial psi / \partial x_j )
            #
            # for which we have to take the distortians by the two mappings into account
            divdiv = (
                grad_phi_vec_chain[component_trial, component_trial]
                * grad_psi_vec_chain[component_test, component_test]
            )

            form = divdiv * jac_affine_det * jac_blending_det

            mat[data.row, data.col] = quad.integrate(form, symbolizer)

    return Form(
        mat,
        Tabulation(symbolizer),
        symmetric=component_trial == component_test,
        docstring=docstring,
    )


def zero_form(
    trial: FunctionSpace, test: FunctionSpace, geometry: ElementGeometry
) -> sp.Matrix:
    rows = test.num_dofs(geometry)
    cols = trial.num_dofs(geometry) if trial is not None else 1
    return sp.zeros(rows, cols)
