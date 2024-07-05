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
from hog.element_geometry import ElementGeometry, TriangleElement, TetrahedronElement
from hog.exception import HOGException
from hog.fem_helpers import (
    trafo_ref_to_affine,
    trafo_ref_to_physical,
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
from hog.function_space import FunctionSpace, EnrichedGalerkinFunctionSpace, N1E1Space
from hog.math_helpers import dot, grad, inv, abs, det, double_contraction, e_vec
from hog.quadrature import Quadrature, Tabulation
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger, get_logger
from hog.blending import GeometryMap, ExternalMap, IdentityMap
from hog.forms import Form


def mass_boundary(
    trial: FunctionSpace,
    test: FunctionSpace,
    volume_geometry: ElementGeometry,
    boundary_geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> Form:
    docstring = f"""
Mass operator.

Geometry map: {blending}

Weak formulation

    u: trial function (space: {trial})
    v: test function  (space: {test})

    ∫ uv ds
"""
    if trial != test:
        raise HOGException(
            "Trial space must be equal to test space to assemble mass matrix."
        )

    with TimedLogger("assembling mass matrix", level=logging.DEBUG):
        tabulation = Tabulation(symbolizer)

        jac_affine = jac_ref_to_affine(boundary_geometry, symbolizer)
        jac_blending = blending.jacobian(
            trafo_ref_to_affine(boundary_geometry, symbolizer)
        )

        fundamental_form_det = abs(
            det(jac_affine.T * jac_blending.T * jac_blending * jac_affine)
        )

        mat = create_empty_element_matrix(trial, test, volume_geometry)
        it = element_matrix_iterator(trial, test, volume_geometry)

        with TimedLogger(
            f"integrating {mat.shape[0] * mat.shape[1]} expressions",
            level=logging.DEBUG,
        ):
            for data in it:
                phi = data.trial_shape
                psi = data.test_shape

                form = phi * psi * fundamental_form_det**0.5

                mat[data.row, data.col] = form

    return Form(mat, tabulation, symmetric=True, docstring=docstring)
