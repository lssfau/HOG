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

from hog.element_geometry import ElementGeometry, TriangleElement
from hog.exception import HOGException
from hog.fem_helpers import (
    trafo_ref_to_affine,
    trafo_ref_to_physical,
    jac_ref_to_affine,
    jac_affine_to_physical,
    create_empty_element_matrix,
    element_matrix_iterator,
)
from hog.function_space import TrialSpace, TestSpace
from hog.math_helpers import dot, inv, abs, det, double_contraction, e_vec
from hog.quadrature import Tabulation
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger
from hog.blending import GeometryMap, ExternalMap, IdentityMap
from hog.integrand import Form

from hog.manifold_helpers import face_projection, embedded_normal


def laplace_beltrami(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Laplace beltrami operator.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    G: first fundamental form of manifold

    ∫ ∇u · G^(-1) · ∇v · (det(G))^0.5
"""

    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble laplace beltrami matrix."
        )

    if not (isinstance(geometry, TriangleElement) and geometry.space_dimension == 3):
        raise HOGException(
            "Laplace Beltrami only works for triangles embedded in 3D space."
        )

    with TimedLogger("assembling laplace beltrami matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = jac_ref_to_affine(geometry, symbolizer)

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            jac_blending = blending.jacobian(trafo_ref_to_affine(geometry, symbolizer))

        fundamental_form = jac_affine.T * jac_blending.T * jac_blending * jac_affine
        with TimedLogger("inverting first fundamental form", level=logging.DEBUG):
            fundamental_form_inv = inv(fundamental_form)
        fundamental_form_det = abs(det(fundamental_form))

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                with TimedLogger(
                    f"Integrating row = {data.row} , col = {data.col}",
                    level=logging.DEBUG,
                ):
                    grad_phi = data.trial_shape_grad
                    grad_psi = data.test_shape_grad
                    laplace_beltrami_symbol = sp.Matrix(
                        tabulation.register_factor(
                            "laplace_beltrami_symbol",
                            dot(
                                grad_phi,
                                fundamental_form_inv * grad_psi,
                            )
                            * (fundamental_form_det**0.5),
                        )
                    )
                    form = laplace_beltrami_symbol[0]
                    mat[data.row, data.col] = form

    return Form(mat, tabulation, symmetric=True, docstring=docstring)


def manifold_mass(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Manifold mass operator.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})
    G: first fundamental form of manifold

    ∫ uv · (det(G))^0.5
"""

    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble laplace beltrami matrix."
        )

    if not (isinstance(geometry, TriangleElement) and geometry.space_dimension == 3):
        raise HOGException(
            "Laplace Beltrami only works for triangles embedded in 3D space."
        )

    with TimedLogger("assembling laplace beltrami matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = jac_ref_to_affine(geometry, symbolizer)

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            jac_blending = blending.jacobian(trafo_ref_to_affine(geometry, symbolizer))

        fundamental_form_det = abs(
            det(jac_affine.T * jac_blending.T * jac_blending * jac_affine)
        )

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                with TimedLogger(
                    f"Integrating row = {data.row} , col = {data.col}",
                    level=logging.DEBUG,
                ):
                    phi = data.trial_shape
                    psi = data.test_shape
                    manifold_mass_symbol = sp.Matrix(
                        tabulation.register_factor(
                            "manifold_mass_symbol",
                            sp.Matrix([phi * psi * fundamental_form_det**0.5]),
                        )
                    )
                    form = manifold_mass_symbol[0]
                    mat[data.row, data.col] = form

    return Form(mat, tabulation, symmetric=True, docstring=docstring)


def manifold_vector_mass(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_trial: int = 0,
    component_test: int = 0,
) -> Form:
    docstring = f"""
Manifold vector mass operator operator.

Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})
    P: projection matrix onto face

    ∫ Pu · Pv · (det(G))^0.5
    
"""
    with TimedLogger("assembling manifold vector mass matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        if not (
            isinstance(geometry, TriangleElement) and geometry.space_dimension == 3
        ):
            raise HOGException(
                "Laplace Beltrami only works for triangles embedded in 3D space."
            )

        projection = face_projection(geometry, symbolizer, blending=blending)

        jac_affine = jac_ref_to_affine(geometry, symbolizer)

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            jac_blending = sp.eye(3)

        fundamental_form_det = abs(
            det(jac_affine.T * jac_blending.T * jac_blending * jac_affine)
        )

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            phi = data.trial_shape
            psi = data.test_shape
            phi_vec = e_vec(geometry.space_dimension, component_trial) * phi
            psi_vec = e_vec(geometry.space_dimension, component_test) * psi
            projected_phi = projection * phi_vec
            projected_psi = projection * psi_vec

            form = sp.Matrix(
                tabulation.register_factor(
                    "manifold_vector_mass_symbol",
                    dot(projected_phi, projected_psi) * fundamental_form_det**0.5,
                )
            )[0]
            mat[data.row, data.col] = form

    return Form(
        mat,
        tabulation,
        symmetric=component_trial == component_test,
        docstring=docstring,
    )


def manifold_normal_penalty(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_trial: int = 0,
    component_test: int = 0,
) -> Form:
    docstring = f"""
Manifold normal penaly operator.

Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})
    n: normal vector of the face

    ∫ (n · u)(n · v) · (det(G))^0.5
    
"""
    with TimedLogger("assembling manifold normal penalty matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        if not (
            isinstance(geometry, TriangleElement) and geometry.space_dimension == 3
        ):
            raise HOGException(
                "Laplace Beltrami only works for triangles embedded in 3D space."
            )

        normal = embedded_normal(geometry, symbolizer, blending)

        jac_affine = jac_ref_to_affine(geometry, symbolizer)

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            jac_blending = sp.eye(3)

        fundamental_form_det = abs(
            det(jac_affine.T * jac_blending.T * jac_blending * jac_affine)
        )

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            phi = data.trial_shape
            psi = data.test_shape
            phi_vec = e_vec(geometry.space_dimension, component_trial) * phi
            psi_vec = e_vec(geometry.space_dimension, component_test) * psi
            phi_normal = dot(phi_vec, normal)
            psi_normal = dot(psi_vec, normal)

            form = sp.Matrix(
                tabulation.register_factor(
                    "manifold_normal_penalty_symbol",
                    phi_normal * psi_normal * fundamental_form_det**0.5,
                )
            )[0]
            mat[data.row, data.col] = form

    return Form(
        mat,
        tabulation,
        symmetric=component_trial == component_test,
        docstring=docstring,
    )


def manifold_divergence(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_index: int = 0,
    transpose: bool = False,
) -> Form:
    docstring = f"""
Manifold divergence operator.

Component index: {component_index}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (space: {test})

    ∫ u · grad_Gamma(v) · (det(G))^0.5
    
"""

    with TimedLogger("assembling manifold div matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        if not (
            isinstance(geometry, TriangleElement) and geometry.space_dimension == 3
        ):
            raise HOGException(
                "Laplace Beltrami only works for triangles embedded in 3D space."
            )

        jac_affine = jac_ref_to_affine(geometry, symbolizer)

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            jac_blending = sp.eye(3)

        jac_total = jac_blending * jac_affine
        fundamental_form = jac_total.T * jac_total
        fundamental_form_inv = inv(fundamental_form)
        fundamental_form_det = abs(det(fundamental_form))

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            if not transpose:
                phi = data.trial_shape
                phi_grad = data.test_shape_grad
            else:
                phi = data.test_shape
                phi_grad = data.trial_shape_grad

            form = (
                (jac_total * fundamental_form_inv * phi_grad)[component_index]
                * phi
                * fundamental_form_det**0.5
            )

            mat[data.row, data.col] = form

    return Form(
        mat,
        tabulation,
        symmetric=False,
        docstring=docstring,
    )


def manifold_vector_divergence(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_index: int = 0,
    transpose: bool = False,
) -> Form:
    docstring = f"""
Manifold vector divergence operator.

Component index: {component_index}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (space: {test})
    P: projection matrix
    
    ∫ div_Gamma(Pu) · v · (det(G))^0.5
    
"""

    with TimedLogger("assembling manifold div matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        logging.info(
            f"WARNING: Manifold vector divergence does NOT compute derivative of matrix P yet. Generated form might not work as intended."
        )

        if not (
            isinstance(geometry, TriangleElement) and geometry.space_dimension == 3
        ):
            raise HOGException(
                "Laplace Beltrami only works for triangles embedded in 3D space."
            )

        projection_mat = face_projection(geometry, symbolizer, blending=blending)

        jac_affine = jac_ref_to_affine(geometry, symbolizer)

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            jac_blending = sp.eye(3)

        jac_total = jac_blending * jac_affine
        fundamental_form = jac_total.T * jac_total
        fundamental_form_inv = inv(fundamental_form)
        fundamental_form_det = abs(det(fundamental_form))

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            ref_symbols_list = symbolizer.ref_coords_as_list(geometry.dimensions)
            if not transpose:
                phi_vec = (
                    projection_mat
                    * (
                        e_vec(geometry.space_dimension, component_index) * data.trial_shape
                    ).jacobian(ref_symbols_list)
                ).T
                phi = data.test_shape
            else:
                phi = data.trial_shape
                phi_vec = (
                    projection_mat
                    * (
                        e_vec(geometry.space_dimension, component_index) * data.test_shape
                    ).jacobian(ref_symbols_list)
                ).T

            form = (
                (jac_total * fundamental_form_inv * phi_vec).trace()
                * phi
                * fundamental_form_det**0.5
            )

            mat[data.row, data.col] = form

    return Form(
        mat,
        tabulation,
        symmetric=False,
        docstring=docstring,
    )


def manifold_epsilon(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_trial: int = 0,
    component_test: int = 0,
) -> Form:
    docstring = f"""
Manifold epsilon operator.

Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})
    G: 2D manifold embedded in 3D space

    ∫ epsilon_Gamma(Pu) : epsilon_Gamma(Pv) · (det(G))^0.5
    
"""

    with TimedLogger("assembling manifold epsilon matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        if not (
            isinstance(geometry, TriangleElement) and geometry.space_dimension == 3
        ):
            raise HOGException(
                "Laplace Beltrami only works for triangles embedded in 3D space."
            )

        projection_mat = face_projection(geometry, symbolizer, blending=blending)

        jac_affine = jac_ref_to_affine(geometry, symbolizer)

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            jac_blending = sp.eye(3)

        jac_total = jac_blending * jac_affine
        fundamental_form = jac_total.T * jac_total
        fundamental_form_inv = inv(fundamental_form)
        fundamental_form_det = abs(det(fundamental_form))

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            ref_symbols_list = symbolizer.ref_coords_as_list(geometry.dimensions)
            phi = data.trial_shape
            psi = data.test_shape
            unscaled_phi_projected_grad = (
                projection_mat
                * (e_vec(geometry.space_dimension, component_trial) * phi).jacobian(
                    ref_symbols_list
                )
            ).T
            unscaled_psi_projected_grad = (
                projection_mat
                * (e_vec(geometry.space_dimension, component_test) * psi).jacobian(
                    ref_symbols_list
                )
            ).T

            phi_projected_grad = (
                jac_total * fundamental_form_inv * unscaled_phi_projected_grad
            )
            psi_projected_grad = (
                jac_total * fundamental_form_inv * unscaled_psi_projected_grad
            )

            phi_epsilon = 0.5 * (phi_projected_grad + phi_projected_grad.T)
            psi_epsilon = 0.5 * (psi_projected_grad + psi_projected_grad.T)

            form = tabulation.register_factor(
                "epsilon_epsilon_prod",
                double_contraction(phi_epsilon, psi_epsilon)
                * fundamental_form_det**0.5,
            )

            mat[data.row, data.col] = form

    return Form(
        mat,
        tabulation,
        symmetric=component_trial == component_test,
        docstring=docstring,
    )


def vector_laplace_beltrami(
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    component_trial: int = 0,
    component_test: int = 0,
) -> Form:
    docstring = f"""
Manifold vector laplace beltrami operator.

Component trial: {component_trial}
Component test:  {component_test}
Geometry map:    {blending}

Weak formulation

    u: trial function (vectorial space: {trial})
    v: test function  (vectorial space: {test})
    G: 2D manifold embedded in 3D space

    ∫ grad_Gamma(u) : grad_Gamma(v) · (det(G))^0.5
    
"""

    with TimedLogger("assembling vector laplace beltrami matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        if not (
            isinstance(geometry, TriangleElement) and geometry.space_dimension == 3
        ):
            raise HOGException(
                "Laplace Beltrami only works for triangles embedded in 3D space."
            )

        projection_mat = face_projection(geometry, symbolizer, blending=blending)

        jac_affine = jac_ref_to_affine(geometry, symbolizer)

        if isinstance(blending, ExternalMap):
            jac_blending = jac_affine_to_physical(geometry, symbolizer)
        else:
            jac_blending = sp.eye(3)

        jac_total = jac_blending * jac_affine
        fundamental_form = jac_total.T * jac_total
        fundamental_form_inv = inv(fundamental_form)
        fundamental_form_det = abs(det(fundamental_form))

        mat = create_empty_element_matrix(trial, test, geometry)
        it = element_matrix_iterator(trial, test, geometry)

        for data in it:
            ref_symbols_list = symbolizer.ref_coords_as_list(geometry.dimensions)
            phi = data.trial_shape
            psi = data.test_shape
            unscaled_phi_projected_grad = (
                projection_mat
                * (e_vec(geometry.space_dimension, component_trial) * phi).jacobian(
                    ref_symbols_list
                )
            ).T
            unscaled_psi_projected_grad = (
                projection_mat
                * (e_vec(geometry.space_dimension, component_test) * psi).jacobian(
                    ref_symbols_list
                )
            ).T

            phi_projected_grad = (
                jac_total * fundamental_form_inv * unscaled_phi_projected_grad
            )
            psi_projected_grad = (
                jac_total * fundamental_form_inv * unscaled_psi_projected_grad
            )

            form = tabulation.register_factor(
                "phi_psi_prod",
                double_contraction(phi_projected_grad, psi_projected_grad)
                * fundamental_form_det**0.5,
            )

            mat[data.row, data.col] = form

    return Form(
        mat,
        tabulation,
        symmetric=component_trial == component_test,
        docstring=docstring,
    )
