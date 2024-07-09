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

import dataclasses
from typing import Callable
from dataclasses import dataclass

import sympy as sp

from hog.function_space import (
    FunctionSpace,
    LagrangianFunctionSpace,
    TensorialVectorFunctionSpace,
)
from hog.element_geometry import ElementGeometry
from hog.quadrature import Tabulation
from hog.symbolizer import Symbolizer
from hog.blending import GeometryMap, IdentityMap
from hog.fem_helpers import (
    create_empty_element_matrix,
    element_matrix_iterator,
)
from hog.math_helpers import dot
from hog.exception import HOGException
from hog.forms import Form


@dataclass
class IntegrandSymbols:
    J_a: sp.Matrix = None
    J_a_inv: sp.Matrix = None
    J_a_det: sp.Symbol = None

    J_b: sp.Matrix = None
    J_b_inv: sp.Matrix = None
    J_b_det: sp.Symbol = None

    u: sp.Expr = None
    grad_u: sp.Matrix = None

    v: sp.Expr = None
    grad_v: sp.Matrix = None

    dx: sp.Expr = None

    dot_grad_u_grad_v_dx: sp.Matrix = None


def diffusion(*, grad_u, grad_v, dx, **_):
    return dot(grad_u, grad_v) * dx


def mass(*, u, v, dx, **_):
    return u * v * dx


def process_integrand(
    integrand: Callable,
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    is_symmetric: bool = False,
    docstring: str = "",
):

    tabulation = Tabulation(symbolizer)

    s = IntegrandSymbols()

    s.J_a = symbolizer.jac_ref_to_affine(geometry)
    s.J_a_inv = symbolizer.jac_ref_to_affine_inv(geometry)
    s.J_a_det = symbolizer.abs_det_jac_ref_to_affine()

    if isinstance(blending, IdentityMap):
        s.J_b = sp.eye(geometry.space_dimension)
        s.J_b_inv = sp.eye(geometry.space_dimension)
        s.J_b_det = 1
    else:
        s.J_b = symbolizer.jac_affine_to_blending(geometry.space_dimension)
        s.J_b_inv = symbolizer.jac_affine_to_blending_inv(geometry.space_dimension)
        s.J_b_det = symbolizer.abs_det_jac_affine_to_blending()

    s.dx = s.J_a_det * s.J_b_det

    mat = create_empty_element_matrix(trial, test, geometry)
    it = element_matrix_iterator(trial, test, geometry)

    for data in it:

        if isinstance(trial, LagrangianFunctionSpace) or (
            isinstance(trial, TensorialVectorFunctionSpace)
            and isinstance(trial.component_function_space, LagrangianFunctionSpace)
        ):
            s.u = data.trial_shape
            s.grad_u = s.J_b_inv.T * tabulation.register_factor(
                "jac_affine_inv_T_grad_phi", s.J_a_inv.T * data.trial_shape_grad
            )
        else:
            raise HOGException(
                f"New form construction not implemented for the function space {trial}."
            )

        if isinstance(test, LagrangianFunctionSpace) or (
            isinstance(test, TensorialVectorFunctionSpace)
            and isinstance(test.component_function_space, LagrangianFunctionSpace)
        ):
            s.v = data.test_shape
            s.grad_v = s.J_b_inv.T * tabulation.register_factor(
                "jac_affine_inv_T_grad_psi", s.J_a_inv.T * data.test_shape_grad
            )
        else:
            raise HOGException(
                f"New form construction not implemented for the function space {test}."
            )

        # TODO:
        # - other function spaces
        # - manifold/boundary handling (fundamental forms)
        # - tabulation of more involved terms
        # - coefficients

        mat[data.row, data.col] = integrand(**dataclasses.asdict(s))

    return Form(mat, tabulation, symmetric=is_symmetric, docstring=docstring)
