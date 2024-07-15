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

"""
As of writing this documentation, the HOG supports the generation of operators from bilinear forms with the structure

    ∫ F dx + ∫ G ds

where F and G are integrands that typically contain trial and test functions, their derivatives, and/or other terms.

In the HOG, the integrands are expressed with respect to the reference element!
That means, the integral transformations from the physical space to the reference space have to be done manually.
Specifically, the HOG takes care of approximating integrals of the form

    ∫_T K d(x_T)

where T is a reference element and x_T are the reference coordinates. That means that various pull-back mappings,
the transformation theorem (https://en.wikipedia.org/wiki/Integration_by_substitution#Substitution_for_multiple_variables),
etc. have to be written out manually. There are several reasons for that, e.g.:

    1. It is not clear if/how we can decide what the additional terms look like (for instance, they look different for
       volume, boundary, and manifold integrals - even when the same function spaces are used).
       Although it is certainly possible to automate that, it requires a careful design. FEniCS, Firedrake and co have
       been doing that successfully, however it would require a lot of effort to implement here.

    2. Tabulation (i.e., the precomputation of certain inter-element-invariant terms) is very hard to automate
       efficiently. However, one central goal of the HOG is aggressive optimization, and we do not want to remove that
       granularity.

    3. Developer time-constraints :)

This module contains various functions and data structures to formulate the integrands, i.e. what is K above.
The actual integration and code generation is handled somewhere else.
"""

from typing import Callable, List, Union, Tuple
from dataclasses import dataclass, asdict, fields

import sympy as sp

from hog.exception import HOGException
from hog.function_space import FunctionSpace
from hog.element_geometry import ElementGeometry
from hog.quadrature import Quadrature, Tabulation
from hog.symbolizer import Symbolizer
from hog.blending import GeometryMap, IdentityMap, ExternalMap
from hog.fem_helpers import (
    create_empty_element_matrix,
    element_matrix_iterator,
    fem_function_on_element,
    fem_function_gradient_on_element,
    scalar_space_dependent_coefficient,
    jac_affine_to_physical,
)
from hog.math_helpers import inv, det


@dataclass
class Form:
    """
    Wrapper class around the local system matrix that carries some additional information such as whether the bilinear
    form is symmetric and a docstring.
    """

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


@dataclass
class IntegrandSymbols:
    """
    The members of this class are the terms that are supported by the HOG and therefore available to the user for
    the formulation of integrands of integrals over reference elements.

    Always make sure to check whether a term that is required in an integrand is available here first before
    constructing it otherwise.

    See :func:`~process_integrand` for more details.
    """

    @staticmethod
    def fields():
        """
        Convenience method to return a list of all symbols/functions that can be used for the integrand construction.
        """
        return [f.name for f in fields(IntegrandSymbols) if not f.name.startswith("_")]

    # Jacobian from reference to affine space.
    jac_a: sp.Matrix = None
    # Its inverse.
    jac_a_inv: sp.Matrix = None
    # The absolute of its determinant.
    jac_a_abs_det: sp.Symbol = None

    # Jacobian from affine to physical space.
    jac_b: sp.Matrix = None
    # Its inverse.
    jac_b_inv: sp.Matrix = None
    # The absolute of its determinant.
    jac_b_abs_det: sp.Symbol = None

    # The trial shape function (reference space!).
    u: sp.Expr = None
    # The gradient of the trial shape function (reference space!).
    grad_u: sp.Matrix = None

    # The test shape function (reference space!).
    v: sp.Expr = None
    # The gradient of the test shape function (reference space!).
    grad_v: sp.Matrix = None

    # A list of scalar constants.
    c: List[sp.Symbol] = None

    # A list of finite element functions that can be used as function parameters.
    k: List[sp.Symbol] = None

    # The geometry of the volume element.
    volume_geometry: ElementGeometry = None

    # The geometry of the boundary element.
    boundary_geometry: ElementGeometry = None

    # If a boundary geometry is available, this is populated with the Jacobian of the affine mapping from the reference
    # space of the boundary element to the computational (affine) space.
    # The reference space has the dimensions of the boundary element.
    # The affine space has the space dimension (aka the dimension of the space it is embedded in) of the boundary
    # element.
    jac_a_boundary: sp.Matrix = None

    # A callback to tabulate (aka precompute) terms that are identical on all elements of the same type.
    # Use at your own risk, you may get wrong code if used on terms that are not element-invariant!
    tabulate: Callable = None
    # You can also give the tabulated variable a name. That has no effect other than the generated code to be more
    # readable. So not encouraged. But nice for debugging.
    _tabulate_named: Callable = None


def process_integrand(
    integrand: Callable,
    trial: FunctionSpace,
    test: FunctionSpace,
    volume_geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    boundary_geometry: ElementGeometry = None,
    scalar_coefficients: List[str] = None,
    fe_coefficients: List[Tuple[str, Union[FunctionSpace, None]]] = None,
    is_symmetric: bool = False,
    docstring: str = "",
) -> Form:
    """
    Constructs an element matrix (:class:`~Form` object) from an integrand.

    Note that this function does not specify the loop structure of the kernel.
    Make sure to pass the result into the correct methods later on (specifically, take care that boundary integrals are
    actually executed on the boundary, volume integrals over all elements).

    Integrands are passed in as a callable (aka function). For instance:

    ```python
    # The arguments of the function must begin with an asterisk (*), followed by keyword arguments, followed by the
    # unused keyword arguments (**_). All keyword arguments must be members of the IntegrandSymbols class.
    # The function must return the integrand. You can use functions from the module hog.math_helpers module.
    # Many integrands are already implemented under hog/recipes/integrands/.

    def my_diffusion_integrand(
        *,
        jac_a_inv,
        jac_a_abs_det,
        jac_b_inv,
        jac_b_abs_det,
        grad_u,
        grad_v,
        tabulate,
        **_,
    ):
        return (
            double_contraction(
                jac_b_inv.T * tabulate(jac_a_inv.T * grad_u),
                jac_b_inv.T * tabulate(jac_a_inv.T * grad_v),
            )
            * tabulate(jac_a_abs_det)
            * jac_b_abs_det
        )
    ```

    The callable (here `my_diffusion_integrand`, not `my_diffusion_integrand()`) is then passed to this function.

    :param integrand: an integrand callable
    :param trial: the finite-element trial function space
    :param test: the finite-element test function space
    :param volume_geometry: the geometry of the volume element
    :param symbolizer: a Symbolizer instance
    :param blending: an optional blending map e.g., for curved geometries
    :param boundary_geometry: the geometry to integrate over for boundary integrals - passed through to the callable via
                              the IntegrandSymbols object
    :param scalar_coefficients: a list of strings that are names for scalar coefficients, they will be available to the
                                callable as `c`
    :param fe_coefficients: a list of tuples of type (str, FunctionSpace) that are names and spaces for scalar
                            finite-element function coefficients, they will be available to the callable as `k`
                            supply None as the FunctionSpace for a std::function-type coeff (only works for old forms)
    :param is_symmetric: whether the bilinear form is symmetric - this is exploited by the generator
    :param docstring: documentation of the integrand/bilinear form - will end up in the docstring of the generated code
    """

    if scalar_coefficients is None:
        scalar_coefficients = []

    if fe_coefficients is None:
        fe_coefficients = []

    s = IntegrandSymbols()

    tabulation = Tabulation(symbolizer)

    def _tabulate(factor: Union[sp.Expr, sp.Matrix]):
        if isinstance(factor, sp.Expr):
            factor = sp.Matrix([factor])

        return tabulation.register_factor(
            f"tabulated_factor_{symbolizer.get_next_running_integer()}", factor
        )

    def _tabulate_named(factor_name: str, factor: sp.Matrix):
        return tabulation.register_factor(factor_name, factor)

    s.tabulate = _tabulate
    s._tabulate_named = _tabulate_named

    s.volume_geometry = volume_geometry

    s.jac_a = symbolizer.jac_ref_to_affine(volume_geometry)
    s.jac_a_inv = symbolizer.jac_ref_to_affine_inv(volume_geometry)
    s.jac_a_abs_det = symbolizer.abs_det_jac_ref_to_affine()

    if isinstance(blending, IdentityMap):
        s.jac_b = sp.eye(volume_geometry.space_dimension)
        s.jac_b_inv = sp.eye(volume_geometry.space_dimension)
        s.jac_b_abs_det = 1
    elif isinstance(blending, ExternalMap):
        s.jac_b = jac_affine_to_physical(volume_geometry, symbolizer)
        s.jac_b_inv = inv(s.jac_b)
        s.jac_b_abs_det = abs(det(s.jac_b))
    else:
        s.jac_b = symbolizer.jac_affine_to_blending(volume_geometry.space_dimension)
        s.jac_b_inv = symbolizer.jac_affine_to_blending_inv(
            volume_geometry.space_dimension
        )
        s.jac_b_abs_det = symbolizer.abs_det_jac_affine_to_blending()

    if boundary_geometry is not None:

        if boundary_geometry.dimensions != boundary_geometry.space_dimension - 1:
            raise HOGException(
                "Since you are integrating over a boundary, the boundary element's space dimension should be larger "
                "than its dimension."
            )

        if boundary_geometry.space_dimension != volume_geometry.space_dimension:
            raise HOGException("All geometries must be embedded in the same space.")

        s.boundary_geometry = boundary_geometry
        s.jac_a_boundary = symbolizer.jac_ref_to_affine(boundary_geometry)

    s.c = [sp.Symbol(s) for s in scalar_coefficients]

    s.k = []
    s.grad_k = []
    for name, coefficient_function_space in fe_coefficients:
        if coefficient_function_space is None:
            k = scalar_space_dependent_coefficient(
                name, volume_geometry, symbolizer, blending=blending
            )
            grad_k = None
        else:
            k, dof_symbols = fem_function_on_element(
                coefficient_function_space,
                volume_geometry,
                symbolizer,
                domain="reference",
                function_id=name,
                basis_eval=tabulation.register_phi_evals(
                    coefficient_function_space.shape(volume_geometry)
                ),
            )

            if isinstance(k, sp.Matrix) and k.shape == (1, 1):
                k = k[0, 0]

            grad_k, _ = fem_function_gradient_on_element(
                coefficient_function_space,
                volume_geometry,
                symbolizer,
                domain="reference",
                function_id=f"grad_{name}",
                dof_symbols=dof_symbols,
            )
        s.k.append(k)
        s.grad_k.append(grad_k)

    mat = create_empty_element_matrix(trial, test, volume_geometry)
    it = element_matrix_iterator(trial, test, volume_geometry)

    for data in it:
        s.u = data.trial_shape
        s.grad_u = data.trial_shape_grad

        s.v = data.test_shape
        s.grad_v = data.test_shape_grad

        mat[data.row, data.col] = integrand(**asdict(s))

    return Form(mat, tabulation, symmetric=is_symmetric, docstring=docstring)
