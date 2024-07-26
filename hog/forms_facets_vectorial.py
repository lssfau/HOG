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
import logging

from hog.blending import GeometryMap, IdentityMap
from hog.element_geometry import ElementGeometry, TriangleElement, TetrahedronElement
from hog.exception import HOGException
from hog.fem_helpers import (
    trafo_ref_to_affine,
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
from hog.logger import TimedLogger
from hog.forms_facets import (
    _affine_element_vertices,
    trafo_ref_interface_to_ref_element,
    sigma_0,
    beta_0,
)
from hog.function_space import EnrichedGalerkinFunctionSpace


def diffusion_sip_facet_vectorial(
    interface_type: str,
    test_element_1: TestSpace,
    trial_element_2: TrialSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    mixed_bc: bool,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """
    See; diffusion_sip_facet_vectorial
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

        jac_affine_E1 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E1
        )
        jac_affine_E2 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E2
        )

        jac_affine_inv_E1 = inv(jac_affine_E1)
        jac_affine_inv_E2 = inv(jac_affine_E2)

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
            beta_0 = 0.5
            volume_interface *= 2
        else:
            beta_0 = 1

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

                if isinstance(trial_element_2, EnrichedGalerkinFunctionSpace):
                    if interface_type == "outer":
                        phi = jac_affine_E2 * phi
                        grad_phi = jac_affine_E2 * grad_phi
                    else:
                        phi = jac_affine_E1 * phi
                        grad_phi = jac_affine_E1 * grad_phi

                if isinstance(test_element_1, EnrichedGalerkinFunctionSpace):
                    psi = jac_affine_E1 * psi
                    grad_psi = jac_affine_E1 * grad_psi

                if interface_type == "inner":
                    form = (
                        -0.5
                        * dot(grad_psi * jac_affine_inv_E1 * outward_normal, phi)[0, 0]
                        - 0.5
                        * dot(grad_phi * jac_affine_inv_E1 * outward_normal, psi)[0, 0]
                        + (sigma_0 / volume_interface**beta_0) * dot(phi, psi)[0, 0]
                    ) * volume_interface

                elif interface_type == "outer":
                    form = (
                        +0.5
                        * dot(grad_psi * jac_affine_inv_E1 * outward_normal, phi)[0, 0]
                        - 0.5
                        * dot(grad_phi * jac_affine_inv_E2 * outward_normal, psi)[0, 0]
                        - (sigma_0 / volume_interface**beta_0) * dot(phi, psi)[0, 0]
                    ) * volume_interface

                elif interface_type == "dirichlet":
                    # form = (
                    #    -dot(grad_phi * jac_affine_inv_E1 *
                    #         outward_normal, psi)[0, 0]
                    #    - sp.S(0) if trial_element_2.is_continuous else dot(grad_psi *
                    #                                                        jac_affine_inv_E1 * outward_normal, phi)[0, 0]
                    #    + sp.S(0) if trial_element_2.is_continuous else (1 *
                    #                                                     sigma_0 / volume_interface ** beta_0) * dot(phi, psi)[0, 0]
                    # ) * volume_interface

                    form = sp.S(0)
                    form += -dot(grad_phi * jac_affine_inv_E1 * outward_normal, psi)[
                        0, 0
                    ]
                    if not mixed_bc or not trial_element_2.is_continuous:
                        form += -dot(
                            grad_psi * jac_affine_inv_E1 * outward_normal, phi
                        )[0, 0]
                    if not mixed_bc or not (trial_element_2.is_continuous):
                        form += (sigma_0 / volume_interface**beta_0) * dot(phi, psi)[
                            0, 0
                        ]
                    form *= volume_interface

                mat[data.row, data.col] = facet_quad.integrate(form, symbolizer)[
                    0
                ].subs(reference_symbols[volume_element_geometry.dimensions - 1], 0)
    return mat


def diffusion_iip_facet_vectorial(
    interface_type: str,
    test_element_1: TestSpace,
    trial_element_2: TrialSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """
    See; diffusion_sip_facet_vectorial
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

        jac_affine_E1 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E1
        )
        jac_affine_E2 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E2
        )

        jac_affine_inv_E1 = inv(jac_affine_E1)
        jac_affine_inv_E2 = inv(jac_affine_E2)

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
            beta_0 = 0.5
            volume_interface *= 2
        else:
            beta_0 = 1

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

                if isinstance(trial_element_2, EnrichedGalerkinFunctionSpace):
                    if interface_type == "outer":
                        phi = jac_affine_E2 * phi
                        grad_phi = jac_affine_E2 * grad_phi
                    else:
                        phi = jac_affine_E1 * phi
                        grad_phi = jac_affine_E1 * grad_phi

                if isinstance(test_element_1, EnrichedGalerkinFunctionSpace):
                    psi = jac_affine_E1 * psi
                    grad_psi = jac_affine_E1 * grad_psi

                if interface_type == "inner":
                    form = (
                        -0.5
                        * dot(grad_phi * jac_affine_inv_E1 * outward_normal, psi)[0, 0]
                        + (sigma_0 / volume_interface**beta_0) * dot(phi, psi)[0, 0]
                    ) * volume_interface

                elif interface_type == "outer":
                    form = (
                        -0.5
                        * dot(grad_phi * jac_affine_inv_E2 * outward_normal, psi)[0, 0]
                        - (sigma_0 / volume_interface**beta_0) * dot(phi, psi)[0, 0]
                    ) * volume_interface

                elif interface_type == "dirichlet":
                    form = (
                        -dot(grad_phi * jac_affine_inv_E1 * outward_normal, psi)[0, 0]
                        + sp.S(0)
                        if trial_element_2.is_continuous
                        else (1 * sigma_0 / volume_interface**beta_0)
                        * dot(phi, psi)[0, 0]
                    ) * volume_interface

                mat[data.row, data.col] = facet_quad.integrate(form, symbolizer)[
                    0
                ].subs(reference_symbols[volume_element_geometry.dimensions - 1], 0)
    return mat


def divergence_facet_vectorial(
    interface_type: str,
    test_element_1: TestSpace,
    trial_element_2: TrialSpace,
    transpose: bool,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    mixed_bc: bool,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """
    See; diffusion_sip_facet_vectorial
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

        jac_affine_E1 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E1
        )
        jac_affine_E2 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E2
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

                if isinstance(trial_element_2, EnrichedGalerkinFunctionSpace):
                    if interface_type == "outer":
                        phi = jac_affine_E2 * phi
                    else:
                        phi = jac_affine_E1 * phi

                if isinstance(test_element_1, EnrichedGalerkinFunctionSpace):
                    psi = jac_affine_E1 * psi

                if interface_type == "inner":
                    form = (
                        0.5 * dot(outward_normal, phi * psi)[0, 0]
                    ) * volume_interface

                elif interface_type == "outer":
                    if not transpose:
                        form = (
                            -(0.5 * dot(outward_normal, phi * psi)[0, 0])
                            * volume_interface
                        )
                    else:
                        form = (
                            +(0.5 * dot(outward_normal, phi * psi)[0, 0])
                            * volume_interface
                        )

                elif interface_type == "dirichlet":
                    # form = sp.S(0) if not transpose and trial_element_2.is_continuous else (
                    #    dot(outward_normal, phi * psi)[0, 0]) * volume_interface
                    form = sp.S(0)
                    if not mixed_bc or (
                        not transpose and trial_element_2.is_continuous and mixed_bc
                    ):
                        form += (
                            dot(outward_normal, phi * psi)[0, 0]
                        ) * volume_interface

                mat[data.row, data.col] = facet_quad.integrate(form, symbolizer)[
                    0
                ].subs(reference_symbols[volume_element_geometry.dimensions - 1], 0)

    return mat


def symm_grad(grad, jac):
    return 0.5 * ((grad * jac) + sp.transpose((grad * jac)))


def epsilon_sip_facet_vectorial(
    interface_type: str,
    test_element_1: TestSpace,
    trial_element_2: TrialSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    mixed_bc: bool,
    blending: GeometryMap = IdentityMap(),
    variable_viscosity: bool = False,
) -> sp.Matrix:
    """
    See; diffusion_sip_facet_vectorial
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
            affine_point_E_opposite,
            _,
            outward_normal,
        ) = _affine_element_vertices(volume_element_geometry, symbolizer)

        # These Jacobians are for the trafo from the element-local reference space to the affine element space.
        # Both are required for application of the chain rule.

        jac_affine_E1 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E1
        )
        jac_affine_E2 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E2
        )

        jac_affine_inv_E1 = inv(jac_affine_E1)
        jac_affine_inv_E2 = inv(jac_affine_E2)

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
            beta_0 = 0.5
            volume_interface *= 2
        else:
            beta_0 = 1

        mat = create_empty_element_matrix(
            trial_element_2, test_element_1, volume_element_geometry
        )
        it = element_matrix_iterator(
            trial_element_2, test_element_1, volume_element_geometry
        )

        reference_symbols = symbolizer.ref_coords_as_list(
            volume_element_geometry.dimensions
        )

        mu = 1
        if variable_viscosity:
            # mu = scalar_space_dependent_coefficient(
            #    "mu", facet_element_geometry, symbolizer, blending=blending
            # )
            trafo_ref_interface_to_affine_interface = trafo_ref_to_affine(
                volume_element_geometry,
                symbolizer,
                affine_points=affine_points_I + [affine_point_E_opposite],
            )
            if isinstance(volume_element_geometry, TriangleElement):
                coeff_class = ScalarVariableCoefficient2D
            elif isinstance(volume_element_geometry, TetrahedronElement):
                coeff_class = ScalarVariableCoefficient3D

            mu = coeff_class(
                sp.Symbol("mu"), 0, *trafo_ref_interface_to_affine_interface
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

                if isinstance(trial_element_2, EnrichedGalerkinFunctionSpace):
                    if interface_type == "outer":
                        phi = jac_affine_E2 * phi
                        grad_phi = jac_affine_E2 * grad_phi
                    else:
                        phi = jac_affine_E1 * phi
                        grad_phi = jac_affine_E1 * grad_phi

                if isinstance(test_element_1, EnrichedGalerkinFunctionSpace):
                    psi = jac_affine_E1 * psi
                    grad_psi = jac_affine_E1 * grad_psi

                if interface_type == "inner":
                    form = (
                        2
                        * mu
                        * (
                            -0.5
                            * dot(
                                symm_grad(grad_psi, jac_affine_inv_E1) * outward_normal,
                                phi,
                            )[0, 0]
                            - 0.5
                            * dot(
                                symm_grad(grad_phi, jac_affine_inv_E1) * outward_normal,
                                psi,
                            )[0, 0]
                            + (sigma_0 / volume_interface**beta_0)
                            * dot(phi, psi)[0, 0]
                        )
                        * volume_interface
                    )

                elif interface_type == "outer":
                    form = (
                        2
                        * mu
                        * (
                            0.5
                            * dot(
                                symm_grad(grad_psi, jac_affine_inv_E1) * outward_normal,
                                phi,
                            )[0, 0]
                            - 0.5
                            * dot(
                                symm_grad(grad_phi, jac_affine_inv_E2) * outward_normal,
                                psi,
                            )[0, 0]
                            - (sigma_0 / volume_interface**beta_0)
                            * dot(phi, psi)[0, 0]
                        )
                        * volume_interface
                    )

                elif interface_type == "dirichlet":
                    # form = 2 * mu * (
                    #    - dot(symm_grad(grad_phi, jac_affine_inv_E1) *
                    #          outward_normal, psi)[0, 0]
                    #    - sp.S(0) if trial_element_2.is_continuous else dot(symm_grad(grad_psi, jac_affine_inv_E1) *
                    #                                                        outward_normal, phi)[0, 0]
                    #    + sp.S(0) if trial_element_2.is_continuous else (1*sigma_0 / volume_interface **
                    #                                                     beta_0) * dot(phi, psi)[0, 0]
                    # ) * volume_interface

                    form = sp.S(0)
                    form += -dot(
                        symm_grad(grad_phi, jac_affine_inv_E1) * outward_normal, psi
                    )[0, 0]
                    if not mixed_bc or not trial_element_2.is_continuous:
                        form += -dot(
                            symm_grad(grad_psi, jac_affine_inv_E1) * outward_normal, phi
                        )[0, 0]
                    if not mixed_bc or not (trial_element_2.is_continuous):
                        form += (sigma_0 / volume_interface**beta_0) * dot(phi, psi)[
                            0, 0
                        ]
                        form *= 2 * mu * volume_interface
                mat[data.row, data.col] = facet_quad.integrate(form, symbolizer)[
                    0
                ].subs(reference_symbols[volume_element_geometry.dimensions - 1], 0)

    return mat


def epsilon_nip_facet_vectorial(
    interface_type: str,
    test_element_1: TestSpace,
    trial_element_2: TrialSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    variable_viscosity: bool = False,
) -> sp.Matrix:
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
            affine_point_E_opposite,
            _,
            outward_normal,
        ) = _affine_element_vertices(volume_element_geometry, symbolizer)

        # These Jacobians are for the trafo from the element-local reference space to the affine element space.
        # Both are required for application of the chain rule.

        jac_affine_E1 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E1
        )
        jac_affine_E2 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E2
        )

        jac_affine_inv_E1 = inv(jac_affine_E1)
        jac_affine_inv_E2 = inv(jac_affine_E2)

        # These trafos are required to evaluate correct points in the reference space of the elements.
        # The reference coordinates of the interface (where the quadrature happens) are mapped to element ref space.

        trafo_ref_interface_to_ref_element_E1 = trafo_ref_interface_to_ref_element(
            volume_element_geometry, symbolizer, "inner"
        )

        print(
            "trafo_ref_interface_to_ref_element_E1: "
            + str(trafo_ref_interface_to_ref_element_E1)
        )
        trafo_ref_interface_to_ref_element_E2 = trafo_ref_interface_to_ref_element(
            volume_element_geometry, symbolizer, "outer"
        )

        # Finally, we need the determinant of the integration space trafo.
        volume_interface = abs(vol(affine_points_I))
        if isinstance(volume_element_geometry, TetrahedronElement):
            beta_0 = 0.5
            volume_interface *= 2
        else:
            beta_0 = 1
        mat = create_empty_element_matrix(
            trial_element_2, test_element_1, volume_element_geometry
        )
        it = element_matrix_iterator(
            trial_element_2, test_element_1, volume_element_geometry
        )

        reference_symbols = symbolizer.ref_coords_as_list(
            volume_element_geometry.dimensions
        )

        mu = 1
        if variable_viscosity:
            # mu = scalar_space_dependent_coefficient(
            #    "mu", facet_element_geometry, symbolizer, blending=blending
            # )
            trafo_ref_interface_to_affine_interface = trafo_ref_to_affine(
                volume_element_geometry,
                symbolizer,
                affine_points=affine_points_I + [affine_point_E_opposite],
            )
            if isinstance(volume_element_geometry, TriangleElement):
                coeff_class = ScalarVariableCoefficient2D
            elif isinstance(volume_element_geometry, TetrahedronElement):
                coeff_class = ScalarVariableCoefficient3D

            mu = coeff_class(
                sp.Symbol("mu"), 0, *trafo_ref_interface_to_affine_interface
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

                if isinstance(trial_element_2, EnrichedGalerkinFunctionSpace):
                    if interface_type == "outer":
                        phi = jac_affine_E2 * phi
                        grad_phi = jac_affine_E2 * grad_phi
                    else:
                        phi = jac_affine_E1 * phi
                        grad_phi = jac_affine_E1 * grad_phi

                if isinstance(test_element_1, EnrichedGalerkinFunctionSpace):
                    psi = jac_affine_E1 * psi
                    grad_psi = jac_affine_E1 * grad_psi

                if interface_type == "inner":
                    form = (
                        2
                        * mu
                        * (
                            +(sigma_0 / volume_interface**beta_0)
                            * dot(phi, psi)[0, 0]
                        )
                        * volume_interface
                    )

                elif interface_type == "outer":
                    form = (
                        2
                        * mu
                        * (
                            -(sigma_0 / volume_interface**beta_0)
                            * dot(phi, psi)[0, 0]
                        )
                        * volume_interface
                    )

                elif interface_type == "dirichlet":
                    form = (
                        2
                        * mu
                        * (
                            +sp.S(0)
                            if trial_element_2.is_continuous
                            else (1 * sigma_0 / volume_interface**beta_0)
                            * dot(phi, psi)[0, 0]
                        )
                        * volume_interface
                    )

                mat[data.row, data.col] = facet_quad.integrate(form, symbolizer)[
                    0
                ].subs(reference_symbols[volume_element_geometry.dimensions - 1], 0)

    return mat


def epsilon_sip_rhs_dirichlet_vectorial(
    function_space: FunctionSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    variable_viscosity: bool = False,
) -> sp.Matrix:
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

        jac_affine_E1 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E1
        )
        jac_affine_inv_E1 = inv(jac_affine_E1)

        # This trafo is required to evaluate correct points in the reference space of the elements.
        # The reference coordinates of the interface (where the quadrature happens) are mapped to element ref space.

        trafo_ref_interface_to_ref_element_E1 = trafo_ref_interface_to_ref_element(
            volume_element_geometry, symbolizer, "inner"
        )

        # Finally, we need the determinant of the integration space trafo.
        volume_interface = abs(vol(affine_points_I))
        if isinstance(volume_element_geometry, TetrahedronElement):
            beta_0 = 0.5
            volume_interface *= 2
        else:
            beta_0 = 1

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
        if isinstance(volume_element_geometry, TriangleElement):
            coeff_class = ScalarVariableCoefficient2D
        elif isinstance(volume_element_geometry, TetrahedronElement):
            coeff_class = ScalarVariableCoefficient3D
        g = sp.Matrix(
            [
                coeff_class(
                    sp.Symbol(f"g{i}"), 0, *trafo_ref_interface_to_affine_interface
                )
                for i in range(volume_element_geometry.dimensions)
            ]
        )

        mu = 1
        if variable_viscosity:
            # mu = scalar_space_dependent_coefficient(
            #    "mu", facet_element_geometry, symbolizer, blending=blending
            # )
            trafo_ref_interface_to_affine_interface = trafo_ref_to_affine(
                volume_element_geometry,
                symbolizer,
                affine_points=affine_points_I + [affine_point_E_opposite],
            )
            if isinstance(volume_element_geometry, TriangleElement):
                coeff_class = ScalarVariableCoefficient2D
            elif isinstance(volume_element_geometry, TetrahedronElement):
                coeff_class = ScalarVariableCoefficient3D

            mu = coeff_class(
                sp.Symbol("mu"), 0, *trafo_ref_interface_to_affine_interface
            )

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for i in range(function_space.num_dofs(volume_element_geometry)):
                # TODO: fix this by introducing extra symbols for the shape functions
                phi = function_space.shape(volume_element_geometry)[i]
                grad_phi = function_space.grad_shape(volume_element_geometry)[i]

                if isinstance(function_space, EnrichedGalerkinFunctionSpace):
                    phi = jac_affine_E1 * phi
                    grad_phi = jac_affine_E1 * grad_phi

                shape_symbols = ["xi_shape_0", "xi_shape_1", "xi_shape_2"][
                    : volume_element_geometry.dimensions
                ]
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
                    2
                    * mu
                    * (
                        dot(
                            -(symm_grad(grad_phi, jac_affine_inv_E1) * outward_normal)
                            + (1 * sigma_0 / volume_interface**beta_0) * phi,
                            g,
                        )
                    )
                    * volume_interface
                )

                # form = phi * g * volume_interface

                mat[i, 0] = facet_quad.integrate(form, symbolizer)[0].subs(
                    reference_symbols[volume_element_geometry.dimensions - 1], 0
                )

    return mat


def diffusion_nip_facet_vectorial(
    interface_type: str,
    test_element_1: TestSpace,
    trial_element_2: TrialSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """
    See; diffusion_nip_facet_vectorial
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

        jac_affine_E1 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E1
        )
        jac_affine_E2 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E2
        )

        jac_affine_inv_E1 = inv(jac_affine_E1)
        jac_affine_inv_E2 = inv(jac_affine_E2)

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
            beta_0 = 0.5
            volume_interface *= 2
        else:
            beta_0 = 1

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

                if isinstance(trial_element_2, EnrichedGalerkinFunctionSpace):
                    if interface_type == "outer":
                        phi = jac_affine_E2 * phi
                        grad_phi = jac_affine_E2 * grad_phi
                    else:
                        phi = jac_affine_E1 * phi
                        grad_phi = jac_affine_E1 * grad_phi

                if isinstance(test_element_1, EnrichedGalerkinFunctionSpace):
                    psi = jac_affine_E1 * psi
                    grad_psi = jac_affine_E1 * grad_psi

                if interface_type == "inner":
                    form = (
                        (sigma_0 / volume_interface**beta_0) * dot(phi, psi)[0, 0]
                    ) * volume_interface

                elif interface_type == "outer":
                    form = (
                        -(sigma_0 / volume_interface**beta_0) * dot(phi, psi)[0, 0]
                    ) * volume_interface

                elif interface_type == "dirichlet":
                    form = (
                        +sp.S(0)
                        if trial_element_2.is_continuous
                        else (1 * sigma_0 / volume_interface**beta_0)
                        * dot(phi, psi)[0, 0]
                    ) * volume_interface

                mat[data.row, data.col] = facet_quad.integrate(form, symbolizer)[
                    0
                ].subs(reference_symbols[volume_element_geometry.dimensions - 1], 0)

    return mat


def divergence_facet_rhs_dirichlet_vectorial(
    function_space: FunctionSpace,
    transpose: bool,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """
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

        jac_affine_E1 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E1
        )

        jac_affine_inv_E1 = inv(jac_affine_E1)

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
        if isinstance(volume_element_geometry, TriangleElement):
            coeff_class = ScalarVariableCoefficient2D
        elif isinstance(volume_element_geometry, TetrahedronElement):
            coeff_class = ScalarVariableCoefficient3D
        g = sp.Matrix(
            [
                coeff_class(
                    sp.Symbol(f"g{i}"), 0, *trafo_ref_interface_to_affine_interface
                )
                for i in range(volume_element_geometry.dimensions)
            ]
        )

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for i in range(function_space.num_dofs(volume_element_geometry)):
                # TODO: fix this by introducing extra symbols for the shape functions
                phi = function_space.shape(volume_element_geometry)[i]
                grad_phi = function_space.grad_shape(volume_element_geometry)[i]

                if isinstance(function_space, EnrichedGalerkinFunctionSpace):
                    phi = jac_affine_E1 * phi
                    grad_phi = jac_affine_E1 * grad_phi

                shape_symbols = ["xi_shape_0", "xi_shape_1", "xi_shape_2"][
                    : volume_element_geometry.dimensions
                ]
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

                if transpose:
                    form = sp.S(0)
                else:
                    form = (dot(outward_normal, phi * g)[0, 0]) * volume_interface

                mat[i, 0] = facet_quad.integrate(form, symbolizer)[0].subs(
                    reference_symbols[volume_element_geometry.dimensions - 1], 0
                )

    return mat


def diffusion_sip_rhs_dirichlet_vectorial(
    function_space: FunctionSpace,
    volume_element_geometry: ElementGeometry,
    facet_quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """
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

        jac_affine_E1 = jac_ref_to_affine(
            volume_element_geometry, symbolizer, affine_points=affine_points_E1
        )

        jac_affine_inv_E1 = inv(jac_affine_E1)

        # This trafo is required to evaluate correct points in the reference space of the elements.
        # The reference coordinates of the interface (where the quadrature happens) are mapped to element ref space.

        trafo_ref_interface_to_ref_element_E1 = trafo_ref_interface_to_ref_element(
            volume_element_geometry, symbolizer, "inner"
        )

        # Finally, we need the determinant of the integration space trafo.
        volume_interface = abs(vol(affine_points_I))
        if isinstance(volume_element_geometry, TetrahedronElement):
            beta_0 = 0.5
            volume_interface *= 2
        else:
            beta_0 = 1

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
        if isinstance(volume_element_geometry, TriangleElement):
            coeff_class = ScalarVariableCoefficient2D
        elif isinstance(volume_element_geometry, TetrahedronElement):
            coeff_class = ScalarVariableCoefficient3D
        g = sp.Matrix(
            [
                coeff_class(
                    sp.Symbol(f"g{i}"), 0, *trafo_ref_interface_to_affine_interface
                )
                for i in range(volume_element_geometry.dimensions)
            ]
        )

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for i in range(function_space.num_dofs(volume_element_geometry)):
                # TODO: fix this by introducing extra symbols for the shape functions
                phi = function_space.shape(volume_element_geometry)[i]
                grad_phi = function_space.grad_shape(volume_element_geometry)[i]

                if isinstance(function_space, EnrichedGalerkinFunctionSpace):
                    phi = jac_affine_E1 * phi
                    grad_phi = jac_affine_E1 * grad_phi

                shape_symbols = ["xi_shape_0", "xi_shape_1", "xi_shape_2"][
                    : volume_element_geometry.dimensions
                ]
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
                    dot(
                        -(grad_phi * jac_affine_inv_E1 * outward_normal)
                        + (1 * sigma_0 / volume_interface**beta_0) * phi,
                        g,
                    )[0, 0]
                ) * volume_interface

                mat[i, 0] = facet_quad.integrate(form, symbolizer)[0].subs(
                    reference_symbols[volume_element_geometry.dimensions - 1], 0
                )

    return mat
