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

import sympy as sp
from typing import List, Optional, Tuple
import logging

from hog.blending import GeometryMap, IdentityMap, ExternalMap
from hog.element_geometry import ElementGeometry, TetrahedronElement
from hog.exception import HOGException
from hog.fem_helpers import (
    trafo_ref_to_affine,
    fem_function_on_element,
    jac_ref_to_affine,
    jac_affine_to_physical,
    create_empty_element_matrix,
    element_matrix_iterator,
    scalar_space_dependent_coefficient,
)
from hog.integrand import Form
from hog.function_space import (
    FunctionSpace,
    EnrichedGalerkinFunctionSpace,
    N1E1Space,
    TrialSpace,
    TestSpace,
)
from hog.math_helpers import inv, abs, det, double_contraction, dot, curl
from hog.quadrature import Quadrature, Tabulation
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger
from hog.sympy_extensions import fast_subs


def diffusion_vectorial(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    if quad.is_exact() and isinstance(blending, ExternalMap):
        raise HOGException(
            "Exact integration is not supported for externally defined blending functions."
        )

    with TimedLogger("assembling diffusion matrix", level=logging.DEBUG):
        jac_affine = jac_ref_to_affine(geometry, symbolizer)
        with TimedLogger("inverting affine Jacobian", level=logging.DEBUG):
            jac_affine_inv = inv(jac_affine)
        jac_affine_det = sp.Abs(sp.det(jac_affine))

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

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                grad_phi = data.trial_shape_grad
                grad_psi = data.test_shape_grad

                if isinstance(trial, EnrichedGalerkinFunctionSpace):
                    grad_phi = jac_blending * jac_affine * grad_phi
                if isinstance(test, EnrichedGalerkinFunctionSpace):
                    grad_psi = jac_blending * jac_affine * grad_psi

                form = (
                    double_contraction(
                        grad_phi * jac_affine_inv * jac_blending_inv,
                        grad_psi * jac_affine_inv * jac_blending_inv,
                    )
                    * jac_affine_det
                    * jac_blending_det
                )
                mat[data.row, data.col] = quad.integrate(form, symbolizer)

    return mat


def mass_vectorial(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    if quad.is_exact() and isinstance(blending, ExternalMap):
        raise HOGException(
            "Exact integration is not supported for externally defined blending functions."
        )

    with TimedLogger("assembling mass matrix", level=logging.DEBUG):
        jac_affine = jac_ref_to_affine(geometry, symbolizer)
        jac_affine_det = sp.Abs(sp.det(jac_affine))

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))

        mat = create_empty_element_matrix(trial, test, geometry)

        it = element_matrix_iterator(trial, test, geometry)

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                phi = data.trial_shape
                psi = data.test_shape

                if isinstance(trial, EnrichedGalerkinFunctionSpace):
                    # TODO Blending?
                    phi = jac_affine * phi
                if isinstance(test, EnrichedGalerkinFunctionSpace):
                    # TODO Blending?
                    psi = jac_affine * psi

                form = dot(phi, psi) * jac_affine_det * jac_blending_det
                mat[data.row, data.col] = quad.integrate(form, symbolizer)

    return mat


def mass_n1e1(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    if not isinstance(trial, N1E1Space) or not isinstance(test, N1E1Space):
        raise HOGException("mass_n1e1 form is only implemented for N1E1.")

    with TimedLogger("assembling mass matrix", level=logging.DEBUG):
        jac_affine = symbolizer.jac_ref_to_affine(geometry)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry)
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

        for data in it:
            phi = data.trial_shape
            psi = data.test_shape

            form = (
                jac_blending_det
                * jac_affine_det
                * dot(
                    jac_blending_inv.T * jac_affine_inv.T * phi,
                    jac_blending_inv.T * jac_affine_inv.T * psi,
                )
            )

            mat[data.row, data.col] = form

    return Form(mat, Tabulation(symbolizer), symmetric=True)


def divergence_vectorial(
    trial: TrialSpace,
    test: TestSpace,
    transpose: bool,
    geometry: ElementGeometry,
    quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    if quad.is_exact() and isinstance(blending, ExternalMap):
        raise HOGException(
            "Exact integration is not supported for externally defined blending functions."
        )

    with TimedLogger("assembling diffusion matrix", level=logging.DEBUG):
        jac_affine = jac_ref_to_affine(geometry, symbolizer)
        with TimedLogger("inverting affine Jacobian", level=logging.DEBUG):
            jac_affine_inv = inv(jac_affine)
        jac_affine_det = sp.Abs(sp.det(jac_affine))

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
            jac_blending_inv = inv(jac_blending)
        jac_blending_det = abs(det(jac_blending))

        mat = create_empty_element_matrix(trial, test, geometry)

        it = element_matrix_iterator(trial, test, geometry)

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                if transpose:
                    phi = data.trial_shape
                    grad_phi = data.test_shape_grad
                    if isinstance(trial, EnrichedGalerkinFunctionSpace):
                        # TODO Blending?
                        phi = jac_affine * phi
                    if isinstance(test, EnrichedGalerkinFunctionSpace):
                        # TODO Blending?
                        grad_phi = jac_affine * grad_phi
                else:
                    phi = data.test_shape
                    grad_phi = data.trial_shape_grad

                    if isinstance(test, EnrichedGalerkinFunctionSpace):
                        # TODO Blending?
                        phi = jac_affine * phi
                    if isinstance(trial, EnrichedGalerkinFunctionSpace):
                        # TODO Blending?
                        grad_phi = jac_affine * grad_phi

                form = (
                    sp.Trace(-(grad_phi * jac_affine_inv * jac_blending_inv)).doit()
                    * phi
                    * jac_affine_det
                    * jac_blending_det
                )
                mat[data.row, data.col] = quad.integrate(form, symbolizer)
    return mat


def curl_curl(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    """
    Implements a bilinear form of type:

        (curl phi, curl psi),

    where phi is a trial function and psi a test function.
    phi and psi are assumed to be three-dimensional.

    The strong form of this operator is:

        curl curl u.
    """

    if not isinstance(trial, N1E1Space) or not isinstance(test, N1E1Space):
        raise HOGException("curl-curl form is only implemented for N1E1.")

    with TimedLogger("assembling curl-curl matrix", level=logging.DEBUG):
        jac_affine = symbolizer.jac_ref_to_affine(geometry)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))

        symbols = symbolizer.ref_coords_as_list(geometry.dimensions)
        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            curl_phi = curl(data.trial_shape, symbols)
            curl_psi = curl(data.test_shape, symbols)

            form = (
                1
                / (jac_blending_det * jac_affine_det)
                * dot(
                    jac_blending * jac_affine * curl_phi,
                    jac_blending * jac_affine * curl_psi,
                )
            )
            mat[data.row, data.col] = form

    return Form(mat, Tabulation(symbolizer), symmetric=True)


def curl_curl_plus_mass(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    alpha_fem_space: Optional[FunctionSpace] = None,
    beta_fem_space: Optional[FunctionSpace] = None,
) -> Form:
    docstring = f"""
Linear combination of double curl and mass.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    α: coefficient    (space: {alpha_fem_space})
    β: coefficient    (space: {beta_fem_space})

    ∫ α curl u · curl v + ∫ β u · v

Strong formulation

    α curl curl u + β u
"""

    if not isinstance(trial, N1E1Space) or not isinstance(test, N1E1Space):
        raise HOGException("curl_curl_plus_mass form is only implemented for N1E1.")

    with TimedLogger("assembling curl_curl_plus_mass matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = symbolizer.jac_ref_to_affine(geometry)
        jac_affine_inv = symbolizer.jac_ref_to_affine_inv(geometry)
        jac_affine_det = symbolizer.abs_det_jac_ref_to_affine()

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            affine_coords = trafo_ref_to_affine(geometry, symbolizer)
            jac_blending = blending.jacobian(affine_coords)

        jac_blending_det = abs(det(jac_blending))
        with TimedLogger("inverting blending Jacobian", level=logging.DEBUG):
            jac_blending_inv = inv(jac_blending)

        if alpha_fem_space:
            phi_eval_symbols = tabulation.register_phi_evals(
                alpha_fem_space.shape(geometry)
            )

            alpha, _ = fem_function_on_element(
                alpha_fem_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="alpha",
                basis_eval=phi_eval_symbols,
            )
        else:
            alpha = scalar_space_dependent_coefficient(
                "alpha", geometry, symbolizer, blending=blending
            )

        if beta_fem_space:
            if beta_fem_space != alpha_fem_space:
                phi_eval_symbols = tabulation.register_phi_evals(
                    beta_fem_space.shape(geometry)
                )

            beta, _ = fem_function_on_element(
                beta_fem_space,
                geometry,
                symbolizer,
                domain="reference",
                function_id="beta",
                basis_eval=phi_eval_symbols,
            )
        else:
            beta = scalar_space_dependent_coefficient(
                "beta", geometry, symbolizer, blending=blending
            )

        symbols = symbolizer.ref_coords_as_list(geometry.dimensions)
        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            phi = data.trial_shape
            psi = data.test_shape
            curl_phi = curl(data.trial_shape, symbols)
            curl_psi = curl(data.test_shape, symbols)

            if blending != IdentityMap():
                affine_curl_phi_symbols = tabulation.register_factor(
                    "affine_curl_phi", jac_affine * curl_phi / jac_affine_det
                )
                affine_curl_psi_symbols = tabulation.register_factor(
                    "affine_curl_psi", jac_affine * curl_psi / jac_affine_det
                )
                affine_phi_symbols = tabulation.register_factor(
                    "affine_phi", jac_affine_inv.T * phi * jac_affine_det
                )
                affine_psi_symbols = tabulation.register_factor(
                    "affine_psi", jac_affine_inv.T * psi * jac_affine_det
                )

                curl_curl = (
                    dot(
                        jac_blending * affine_curl_phi_symbols,
                        jac_blending * affine_curl_psi_symbols,
                    )
                    / jac_blending_det
                )
                mass = (
                    dot(
                        jac_blending_inv.T * affine_phi_symbols,
                        jac_blending_inv.T * affine_psi_symbols,
                    )
                    * jac_blending_det
                )

                form = alpha * curl_curl + beta * mass
            else:
                curl_curl = dot(
                    jac_blending * jac_affine * curl_phi,
                    jac_blending * jac_affine * curl_psi,
                ) / (jac_blending_det * jac_affine_det)
                mass = (
                    dot(
                        jac_blending_inv.T * jac_affine_inv.T * phi,
                        jac_blending_inv.T * jac_affine_inv.T * psi,
                    )
                    * jac_blending_det
                    * jac_affine_det
                )

                curl_curl_symbol = tabulation.register_factor(
                    "curl_curl_det", curl_curl
                )
                mass_symbol = tabulation.register_factor("mass_det", mass)

                form = alpha * curl_curl_symbol + beta * mass_symbol

            mat[data.row, data.col] = form

    return Form(mat, tabulation, symmetric=True, docstring=docstring)


def linear_form_vectorial(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """
    Implements a linear form of type:

        ([f1(x), f2(x), f3(x)], [psi_vec1(x), psi_vec2(x),psi_vec3(x)])

    where psi_vec a vectorial test function and f = [f1,f2,f3] a vectorial, external function.
    """

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

        f1 = scalar_space_dependent_coefficient(
            "f1", geometry, symbolizer, blending=blending
        )
        f2 = scalar_space_dependent_coefficient(
            "f2", geometry, symbolizer, blending=blending
        )
        if isinstance(geometry, TetrahedronElement):
            f3 = scalar_space_dependent_coefficient(
                "f3", geometry, symbolizer, blending=blending
            )

        with TimedLogger(
            f"integrating {mat.shape[0]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                if data.row == data.col:
                    psi_vec = data.test_shape
                    if isinstance(geometry, TetrahedronElement):
                        form = (
                            dot(sp.Matrix([f1, f2, f3]), psi_vec)
                            * jac_affine_det
                            * jac_blending_det
                        )
                    else:
                        form = (
                            dot(sp.Matrix([f1, f2]), psi_vec)
                            * jac_affine_det
                            * jac_blending_det
                        )
                    mat[data.row, data.col] = quad.integrate(form, symbolizer)

    return mat
