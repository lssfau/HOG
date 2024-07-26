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
from typing import List, Tuple, Union
import logging

from hog.blending import GeometryMap, IdentityMap
from hog.element_geometry import ElementGeometry, TriangleElement, TetrahedronElement
from hog.exception import HOGException
from hog.fem_helpers import (
    trafo_ref_to_affine,
    trafo_affine_point_to_ref,
    jac_ref_to_affine,
    create_empty_element_matrix,
    element_matrix_iterator,
    scalar_space_dependent_coefficient,
)
from hog.external_functions import (
    ScalarVariableCoefficient2D,
    ScalarVariableCoefficient3D,
)
from hog.function_space import FunctionSpace, TrialSpace, TestSpace
from hog.math_helpers import (
    dot,
    inv,
    abs,
    vol,
)
from hog.quadrature import Quadrature
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger, get_logger


beta_0 = 1
sigma_0 = 3


def _affine_element_vertices(
    volume_element_geometry: ElementGeometry, symbolizer: Symbolizer
) -> Tuple[
    List[sp.Matrix], List[sp.Matrix], List[sp.Matrix], sp.Matrix, sp.Matrix, sp.Matrix
]:
    """Helper function that returns the symbols of the affine points of two neighboring elements.

    Returns a tuple of

    1) a list of the vertices of the affine element E1 (inner element)
    2) a list of the vertices of the affine element E2 (outer element)
    3) a list of the vertices of the interface I
    4) the vertex that is part of E1 but not on I
    5) the vertex that is part of E2 but not on I
    6) the normal from E1 to E2
    """
    affine_points_E1 = symbolizer.affine_vertices_as_vectors(
        volume_element_geometry.dimensions, volume_element_geometry.num_vertices, 0
    )

    affine_points_E2 = symbolizer.affine_vertices_as_vectors(
        volume_element_geometry.dimensions,
        volume_element_geometry.num_vertices,
        volume_element_geometry.num_vertices,
    )

    affine_points_I = symbolizer.affine_vertices_as_vectors(
        volume_element_geometry.dimensions,
        volume_element_geometry.num_vertices - 1,
        2 * volume_element_geometry.num_vertices,
    )

    affine_point_E1_opposite = symbolizer.affine_vertices_as_vectors(
        volume_element_geometry.dimensions,
        1,
        3 * volume_element_geometry.num_vertices - 1,
    )[0]

    affine_point_E2_opposite = symbolizer.affine_vertices_as_vectors(
        volume_element_geometry.dimensions,
        1,
        3 * volume_element_geometry.num_vertices,
    )[0]

    outward_normal = symbolizer.affine_vertices_as_vectors(
        volume_element_geometry.dimensions,
        1,
        3 * volume_element_geometry.num_vertices + 1,
    )[0]

    return (
        affine_points_E1,
        affine_points_E2,
        affine_points_I,
        affine_point_E1_opposite,
        affine_point_E2_opposite,
        outward_normal,
    )


def trafo_ref_interface_to_ref_element(
    volume_element_geometry: ElementGeometry,
    symbolizer: Symbolizer,
    element: str = "inner",
) -> sp.Matrix:
    """Maps a point on the reference space of the interface to the reference space of the volume element.

    Given coordinates in interface-reference space (e.g. (0, 1) line in 2D, where the quadrature points are
    defined) we need the corresponding points in the element-reference space. This is how we evaluate the shape
    functions and the gradients so that we do not need to sort the DoFs if the element vertices are passed in
    the correct order.

    For now this transformation is done in a "brute-force" manner.
    First, the interface-reference point is mapped onto the affine interface space.
    From there, individually for each element, we map is back to the element-reference space.

    We need an additional point to the vertices of the interface - so we simply choose any point that is in
    one of the elements but not on the interface.

    :param element: either "inner" or "outer"
    """

    (
        affine_points_E1,
        affine_points_E2,
        affine_points_I,
        affine_point_E1_opposite,
        affine_point_E2_opposite,
        outward_normal,
    ) = _affine_element_vertices(volume_element_geometry, symbolizer)

    if element == "inner":
        affine_points_E = affine_points_E1
        affine_point_E_opposite = affine_point_E1_opposite
    elif element == "outer":
        affine_points_E = affine_points_E2
        affine_point_E_opposite = affine_point_E2_opposite
    else:
        raise HOGException("Invalid element type (should be 'inner' or 'outer')")

    # First we compute the transformation from the interface reference space to the affine interface.
    trafo_ref_interface_to_affine_interface_E = trafo_ref_to_affine(
        volume_element_geometry,
        symbolizer,
        affine_points=affine_points_I + [affine_point_E_opposite],
    )

    # Then transform the result from affine space to the reference volume.
    trafo_ref_interface_to_ref_element_E = trafo_affine_point_to_ref(
        volume_element_geometry,
        symbolizer,
        affine_eval_point=trafo_ref_interface_to_affine_interface_E,
        affine_element_points=affine_points_E,
    )

    return trafo_ref_interface_to_ref_element_E


def stokes_p0_stabilization(
    interface_type: str,
    test_element: TestSpace,
    trial_element: TrialSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    r"""
    Interface integrals for the stabilization term

        \gamma |\meas{e}| ( [[p_h]]_e, [[q_h]]_e )_e

    for the P1-P0 Stokes discretization as found in

        Blank, L. (2014).
        On Divergence-Free Finite Element Methods for the Stokes Equations (Freie Universität Berlin).
        p. 84, eq. (6.2)

    Interface integrals are performed from both sides. interface_type has to be

        'inner' coupling of element unknowns with themselves

            \int_e \gamma \meas{e} p q

        'outer' coupling of element unknowns with those opposite of the interface

            - \int_e \gamma \meas{e} p q
    """

    if not isinstance(blending, IdentityMap):
        raise HOGException(
            "Blending is not implemented for the the P0-Stokes stabilization."
        )

    with TimedLogger("assembling interface stabilization matrix", level=logging.DEBUG):
        # Grabbing the symbols for the vertices of both elements, the interface, and the outward normal.
        (
            affine_points_E1,
            affine_points_E2,
            affine_points_I,
            _,
            _,
            outward_normal,
        ) = _affine_element_vertices(volume_element_geometry, symbolizer)

        # These trafos are required to evaluate correct points in the reference space of the elements.
        # The reference coordinates of the interface (where the quadrature happens) are mapped to element ref space.

        trafo_ref_interface_to_ref_element_E1 = trafo_ref_interface_to_ref_element(
            volume_element_geometry, symbolizer, "inner"
        )
        trafo_ref_interface_to_ref_element_E2 = trafo_ref_interface_to_ref_element(
            volume_element_geometry, symbolizer, "outer"
        )

        # Finally, we need the determinant of the integration space trafo.

        volume_interface = abs(vol(affine_points_I))
        # if volume_element_geometry.dimensions != 2:
        #    raise HOGException("Volume must be scaled in 3D.")
        if isinstance(volume_element_geometry, TetrahedronElement):
            volume_interface *= 2

        mat = create_empty_element_matrix(
            trial_element, test_element, volume_element_geometry
        )
        it = element_matrix_iterator(
            trial_element, test_element, volume_element_geometry
        )

        reference_symbols = symbolizer.ref_coords_as_list(
            volume_element_geometry.dimensions
        )

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                # TODO: fix this by introducing extra symbols for the shape functions
                phi = data.trial_shape
                psi = data.test_shape

                shape_symbols = ["xi_shape_0", "xi_shape_1"]
                phi = phi.subs(zip(reference_symbols, shape_symbols))
                psi = psi.subs(zip(reference_symbols, shape_symbols))

                # The evaluation is performed on the reference space of the _elements_ and _not_ in the reference space
                # of the interface. This way we do not need to sort DoFs. So apply the trafo to the trial functions.
                if interface_type == "outer":
                    phi = phi.subs(
                        zip(shape_symbols, trafo_ref_interface_to_ref_element_E2)
                    )
                else:
                    phi = phi.subs(
                        zip(shape_symbols, trafo_ref_interface_to_ref_element_E1)
                    )

                psi = psi.subs(
                    zip(shape_symbols, trafo_ref_interface_to_ref_element_E1)
                )

                gamma = 0.1

                if interface_type == "inner":
                    form = ((gamma * volume_interface) * phi * psi) * volume_interface

                elif interface_type == "outer":
                    form = (-(gamma * volume_interface) * phi * psi) * volume_interface

                mat[data.row, data.col] = facet_quad.integrate(form, symbolizer)[
                    0
                ].subs(reference_symbols[volume_element_geometry.dimensions - 1], 0)

    return mat


def diffusion_sip_facet(
    interface_type: str,
    test_element_1: TestSpace,
    trial_element_2: TrialSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    r"""
    Interface integrals for the symmetric interior penalty formulation for the (constant-coeff.) Laplacian.

    Performs only "one-sided" evaluation - the integration has to be repeated from the other side.

    Let v and w be the test and trial functions respectively, n is the normal pointing from the "inner" to the "outer"
    element (E1 and E2).

    All "interface types" are handled by this function. The type has to be specified by the interface_type argument:
    Possible types:

        'inner': coupling of element unknowns with themselves (both, v and w restricted to E1)

            - 0.5 * \int_e J_v^{-\top} \nabla v \cdot n w
            - 0.5 * \int_e J_w^{-\top} \nabla w \cdot n v
            + \frac{\sigma}{\meas(e)^\beta} \int_e v w

        'outer': coupling of element unknowns with those opposite of the interface (v restricted to E1, w restricted to
                 E2)

            + 0.5 * \int_e J_v^{-\top} \nabla v \cdot n w
            - 0.5 * \int_e J_w^{-\top} \nabla w \cdot n v
            - \frac{\sigma}{\meas(e)^\beta} \int_e v w

        'dirichlet': interface is at Dirichlet boundary (both, v and w restricted to E1)

            - \int_e J_v^{-\top} \nabla v \cdot n w
            - \int_e J_w^{-\top} \nabla w \cdot n v
            + 2 \frac{\sigma}{\meas(e)^\beta} \int_e v w

    Implementation:

    Three "reference" spaces are considered:

    - the reference space of the interface  (S_e)
    - the reference space of the element(s) (S_E1, S_E2)
    - the affine space                      (S)

    The integrals are evaluated over e in reference space of e (S_e).

    v, w, \nabla v and \nabla w are evaluated on the _element_ reference space. This involves trafos
    T_e^{E1}: S_e \rightarrow S_E1, and T_e^{E2}: S_e \rightarrow S_E2. So v above is actually v( T_e^{E1}( x ) ).

    The Jacobians are formed from the transformation from S_E1 to S and S_E2 to S.

    Parameters: the first set of affine vertices belongs to the interior element, the second set to the neighboring
    element. An additional set of affine vertices corresponds to the interface vertices. Overall, e.g. in 2D we have

        p_0, p_1, p_2 for the inner triangular element
        p_3, p_4, p_5 for the outer triangular element
        p_6, p_7      for the interface
        p_8           inner triangle vertex that is not on the interface
        p_9           outer triangle vertex that is not on the interface
        p_10          for the normal from E1 to E2

    In 3D we get

        p_0, p_1, p_2, p_3 for the inner tet element
        p_4, p_5, p_6, p_7 for the outer tet element
        p_8, p_9, p_10     for the interface
        p_11               inner triangle vertex that is not on the interface
        p_12               outer triangle vertex that is not on the interface
        p_13               for the normal from E1 to E2

    Note that many of these points are equal. This avoids handling reordering of the matrices later on and also avoids
    any conditionals in the generated forms.
    """

    if not isinstance(blending, IdentityMap):
        raise HOGException("Blending is not implemented for the SIP.")

    with TimedLogger(
        "assembling interface diffusion matrix with SIP", level=logging.DEBUG
    ):
        # Grabbing the symbols for the vertices of both elements, the interface, and the outward normal.
        (
            affine_points_E1,
            affine_points_E2,
            affine_points_I,
            _,
            _,
            outward_normal,
        ) = _affine_element_vertices(volume_element_geometry, symbolizer)

        # These Jacobians are for the trafo from the element-local reference space to the affine element space.
        # Both are required for application of the chain rule.

        jac_affine_inv_E1 = inv(
            jac_ref_to_affine(
                volume_element_geometry, symbolizer, affine_points=affine_points_E1
            )
        )

        jac_affine_inv_E2 = inv(
            jac_ref_to_affine(
                volume_element_geometry, symbolizer, affine_points=affine_points_E2
            )
        )

        # These trafos are required to evaluate correct points in the reference space of the elements.
        # The reference coordinates of the interface (where the quadrature happens) are mapped to element ref space.

        trafo_ref_interface_to_ref_element_E1 = trafo_ref_interface_to_ref_element(
            volume_element_geometry, symbolizer, "inner"
        )
        trafo_ref_interface_to_ref_element_E2 = trafo_ref_interface_to_ref_element(
            volume_element_geometry, symbolizer, "outer"
        )

        # Finally, we need the determinant of the integration space trafo.
        volume_interface = abs(vol(affine_points_I))
        if isinstance(volume_element_geometry, TetrahedronElement):
            volume_interface *= 2

        mat = create_empty_element_matrix(
            trial_element_2, test_element_1, volume_element_geometry
        )
        it = element_matrix_iterator(
            trial_element_2, test_element_1, volume_element_geometry
        )

        reference_symbols = symbolizer.ref_coords_as_list(
            volume_element_geometry.dimensions
        )

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                # TODO: fix this by introducing extra symbols for the shape functions
                phi = data.trial_shape
                psi = data.test_shape
                grad_phi = data.trial_shape_grad
                grad_psi = data.test_shape_grad

                shape_symbols = ["xi_shape_0", "xi_shape_1", "xi_shape_2"][
                    : volume_element_geometry.dimensions
                ]
                phi = phi.subs(zip(reference_symbols, shape_symbols))
                psi = psi.subs(zip(reference_symbols, shape_symbols))
                grad_phi = grad_phi.subs(zip(reference_symbols, shape_symbols))
                grad_psi = grad_psi.subs(zip(reference_symbols, shape_symbols))

                # The evaluation is performed on the reference space of the _elements_ and _not_ in the reference space
                # of the interface. This way we do not need to sort DoFs. So apply the trafo to the trial functions.
                if interface_type == "outer":
                    phi = phi.subs(
                        zip(shape_symbols, trafo_ref_interface_to_ref_element_E2)
                    )
                    grad_phi = grad_phi.subs(
                        zip(shape_symbols, trafo_ref_interface_to_ref_element_E2)
                    )
                else:
                    phi = phi.subs(
                        zip(shape_symbols, trafo_ref_interface_to_ref_element_E1)
                    )
                    grad_phi = grad_phi.subs(
                        zip(shape_symbols, trafo_ref_interface_to_ref_element_E1)
                    )

                psi = psi.subs(
                    zip(shape_symbols, trafo_ref_interface_to_ref_element_E1)
                )
                grad_psi = grad_psi.subs(
                    zip(shape_symbols, trafo_ref_interface_to_ref_element_E1)
                )

                if interface_type == "inner":
                    form = (
                        -0.5
                        * dot(grad_psi * jac_affine_inv_E1, outward_normal)[0, 0]
                        * phi
                        - 0.5
                        * dot(grad_phi * jac_affine_inv_E1, outward_normal)[0, 0]
                        * psi
                        + (sigma_0 / volume_interface**beta_0) * phi * psi
                    ) * volume_interface

                elif interface_type == "outer":
                    form = (
                        0.5
                        * dot(grad_psi * jac_affine_inv_E1, outward_normal)[0, 0]
                        * phi
                        - 0.5
                        * dot(grad_phi * jac_affine_inv_E2, outward_normal)[0, 0]
                        * psi
                        - (sigma_0 / volume_interface**beta_0) * phi * psi
                    ) * volume_interface

                elif interface_type == "dirichlet":
                    form = (
                        -dot(grad_psi * jac_affine_inv_E1, outward_normal)[0, 0] * phi
                        - dot(grad_phi * jac_affine_inv_E1, outward_normal)[0, 0] * psi
                        + (4 * sigma_0 / volume_interface**beta_0) * phi * psi
                    ) * volume_interface

                mat[data.row, data.col] = facet_quad.integrate(form, symbolizer)[
                    0
                ].subs(reference_symbols[volume_element_geometry.dimensions - 1], 0)

    return mat


def diffusion_sip_rhs_dirichlet(
    function_space: FunctionSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    r"""
    Right-hand side Dirichlet boundary integrals for the symmetric interior penalty formulation for the
    (constant-coeff.) Laplacian.

    This evaluates

        - \int_e ( \nabla v \cdot n + \frac{\sigma}{\meas(e)^\beta} v ) g_D

    where g_D is an external function that evaluates to the Dirichlet boundary condition.

    Implementation:

    Three "reference" spaces are considered:

    - the reference space of the interface  (S_e)
    - the reference space of the element(s) (S_E1, S_E2)
    - the affine space                      (S)

    The integrals are evaluated over e in reference space of e (S_e).

    v, w, \nabla v and \nabla w are evaluated on the _element_ reference space. This involves trafos
    T_e^{E1}: S_e \rightarrow S_E1, and T_e^{E2}: S_e \rightarrow S_E2. So v above is actually v( T_e^{E1}( x ) ).

    The Jacobians are formed from the transformation from S_E1 to S and S_E2 to S.

    Parameters: the first set of affine vertices belongs to the interior element, the second set to the neighboring
    element. An additional set of affine vertices corresponds to the interface vertices. Overall, e.g. in 2D we have

        p_0, p_1, p_2 for the inner triangular element
        p_6, p_7      for the interface
        p_8           inner triangle vertex that is not on the interface
        p_10          for the normal from E1 to E2

    In 3D we get

        p_0, p_1, p_2, p_3 for the inner tet element
        p_8, p_9, p_10     for the interface
        p_11               inner triangle vertex that is not on the interface
        p_13               for the normal from E1 to E2

    Note that many of these points are equal. This avoids handling reordering of the matrices later on and also avoids
    any conditionals in the generated forms.
    """

    if not isinstance(blending, IdentityMap):
        raise HOGException("Blending is not implemented for the SIP.")

    with TimedLogger(
        "assembling interface diffusion matrix with SIP", level=logging.DEBUG
    ):
        # Grabbing the symbols for the vertices of the element, the interface, and the outward normal.
        (
            affine_points_E1,
            _,
            affine_points_I,
            affine_point_E_opposite,
            _,
            outward_normal,
        ) = _affine_element_vertices(volume_element_geometry, symbolizer)

        # This Jacobian is for the trafo from the element-local reference space to the affine element space.
        # Both are required for application of the chain rule.

        jac_affine_inv_E1 = inv(
            jac_ref_to_affine(
                volume_element_geometry, symbolizer, affine_points=affine_points_E1
            )
        )

        # This trafo is required to evaluate correct points in the reference space of the elements.
        # The reference coordinates of the interface (where the quadrature happens) are mapped to element ref space.

        trafo_ref_interface_to_ref_element_E1 = trafo_ref_interface_to_ref_element(
            volume_element_geometry, symbolizer, "inner"
        )

        # Finally, we need the determinant of the integration space trafo.
        volume_interface = abs(vol(affine_points_I))
        if isinstance(volume_element_geometry, TetrahedronElement):
            volume_interface *= 2

        mat = sp.zeros(function_space.num_dofs(volume_element_geometry), 1)

        reference_symbols = symbolizer.ref_coords_as_list(
            volume_element_geometry.dimensions
        )

        # TODO: outsource the scalar coeff for boundary integrals to some function
        # We cannot simply use the standard function to retrieve a scalar coefficient since we need to transform back
        # to affine space first to evaluate it. So let's do that manually for now.
        trafo_ref_interface_to_affine_interface = trafo_ref_to_affine(
            volume_element_geometry,
            symbolizer,
            affine_points=affine_points_I + [affine_point_E_opposite],
        )
        coeff_class: Union[
            type[ScalarVariableCoefficient2D], type[ScalarVariableCoefficient3D]
        ]
        if isinstance(volume_element_geometry, TriangleElement):
            coeff_class = ScalarVariableCoefficient2D
        elif isinstance(volume_element_geometry, TetrahedronElement):
            coeff_class = ScalarVariableCoefficient3D
        g = coeff_class(sp.Symbol("g"), 0, *trafo_ref_interface_to_affine_interface)

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for i in range(function_space.num_dofs(volume_element_geometry)):
                # TODO: fix this by introducing extra symbols for the shape functions
                phi = function_space.shape(volume_element_geometry)[i]
                grad_phi = function_space.grad_shape(volume_element_geometry)[i]

                shape_symbols = ["xi_shape_0", "xi_shape_1"]
                phi = phi.subs(zip(reference_symbols, shape_symbols))
                grad_phi = grad_phi.subs(zip(reference_symbols, shape_symbols))

                # The evaluation is performed on the reference space of the _elements_ and _not_ in the reference space
                # of the interface. This way we do not need to sort DoFs. So apply the trafo to the trial functions.

                phi = phi.subs(
                    zip(shape_symbols, trafo_ref_interface_to_ref_element_E1)
                )
                grad_phi = grad_phi.subs(
                    zip(shape_symbols, trafo_ref_interface_to_ref_element_E1)
                )

                form = (
                    1
                    * (
                        -dot(jac_affine_inv_E1.T * grad_phi, outward_normal)[0, 0]
                        + (4 * sigma_0 / volume_interface**beta_0) * phi
                    )
                    * g
                    * volume_interface
                )

                # form = phi * g * volume_interface

                mat[i, 0] = facet_quad.integrate(form, symbolizer)[0].subs(
                    reference_symbols[volume_element_geometry.dimensions - 1], 0
                )

    return mat
