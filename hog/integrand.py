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

from typing import Callable, List, Union, Tuple, Any, Dict
from dataclasses import dataclass, asdict, fields, field

import sympy as sp

from hog.exception import HOGException
from hog.function_space import (
    FunctionSpace,
    TrialSpace,
    TestSpace,
    TensorialVectorFunctionSpace,
    LagrangianFunctionSpace,
)
from hog.element_geometry import ElementGeometry
from hog.quadrature import Quadrature, Tabulation
from hog.symbolizer import Symbolizer
from hog.blending import GeometryMap, IdentityMap, ExternalMap, ParametricMap
from hog.fem_helpers import (
    create_empty_element_matrix,
    element_matrix_iterator,
    fem_function_on_element,
    fem_function_gradient_on_element,
    scalar_space_dependent_coefficient,
    jac_affine_to_physical,
    trafo_ref_to_physical,
)
from hog.math_helpers import inv, det
from hog.recipes.integrands.volume.rotation import RotationType


@dataclass
class Form:
    """
    Wrapper class around the local system matrix that carries some additional information such as whether the bilinear
    form is symmetric and a docstring.
    """

    integrand: sp.MatrixBase
    tabulation: Tabulation
    symmetric: bool
    free_symbols: List[sp.Symbol] = field(default_factory=lambda: list())
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
    jac_a: sp.Matrix | None = None
    # Its inverse.
    jac_a_inv: sp.Matrix | None = None
    # The absolute of its determinant.
    jac_a_abs_det: sp.Symbol | None = None

    # Jacobian from affine to physical space.
    jac_b: sp.Matrix | None = None
    # Its inverse.
    jac_b_inv: sp.Matrix | None = None
    # The absolute of its determinant.
    jac_b_abs_det: sp.Symbol | None = None

    # Hessian of the mapping from affine to physical space.
    hessian_b: sp.Matrix | None = None

    # The trial shape function (reference space!).
    u: sp.Expr | None = None
    # The gradient of the trial shape function (reference space!).
    grad_u: sp.Matrix | None = None
    # The Hessian of the trial shape function (reference space!).
    hessian_u: sp.Matrix | None = None

    # The test shape function (reference space!).
    v: sp.Expr | None = None
    # The gradient of the test shape function (reference space!).
    grad_v: sp.Matrix | None = None
    # The Hessian of the test shape function (reference space!).
    hessian_v: sp.Matrix | None = None

    # The physical coordinates.
    x: sp.Matrix | None = None

    # A dict of finite element functions that can be used as function parameters.
    # The keys are specified by the strings that are passed to process_integrand.
    k: Dict[str, sp.Symbol] | None = None
    # A list of the gradients of the parameter finite element functions.
    # The keys are specified by the strings that are passed to process_integrand.
    grad_k: Dict[str, sp.Matrix] | None = None

    # The geometry of the volume element.
    volume_geometry: ElementGeometry | None = None

    # The geometry of the boundary element.
    boundary_geometry: ElementGeometry | None = None

    # If a boundary geometry is available, this is populated with the Jacobian of the affine mapping from the reference
    # space of the boundary element to the computational (affine) space.
    # The reference space has the dimensions of the boundary element.
    # The affine space has the space dimension (aka the dimension of the space it is embedded in) of the boundary
    # element.
    jac_a_boundary: sp.Matrix | None = None

    # A callback to generate free symbols that can be chosen by the user later.
    #
    # To get (and register new symbols) simply pass a list of symbol names to this function.
    # It returns sympy symbols that can be safely used in the integrand:
    #
    #     n_x, n_y, n_z = scalars(["n_x", "n_y", "n_z"])
    #
    # or for a single symbol
    #
    #     a = scalars("a")
    #
    # or use the sympy-like space syntax
    #
    #     a, b = scalars("a b")
    #
    # Simply using sp.Symbol will not work since the symbols must be registered in the generator.
    scalars: Callable[[str | List[str]], sp.Symbol | List[sp.Symbol]] | None = None

    # Same as scalars, but returns the symbols arranged as matrices.
    #
    # For a matrix with three rows and two columns:
    #
    #     A = matrix("A", 3, 2)
    #
    matrix: Callable[[str, int, int], sp.Matrix] | None = None

    # A callback to tabulate (aka precompute) terms that are identical on all elements of the same type.
    #
    # Simply enclose such a factor with this function, e.g., replace
    #
    #     some_term = jac_a_inv.T * grad_u
    #
    # with
    #
    #     some_term = tabulate(jac_a_inv.T * grad_u)
    #
    # Use at your own risk, you may get wrong code if used on terms that are not element-invariant!
    #
    # For debugging, you can also give the table an optional name:
    #
    #     some_term = tabulate(jac_a_inv.T * grad_u, factor_name="jac_grad_u")
    #
    tabulate: Callable[[Union[sp.Expr, sp.Matrix], str], sp.Matrix] | None = None

    # For backward compatibility with (sub-)form generation this integer allows to select a component
    component_index: int | None = None


def process_integrand(
    integrand: Callable[..., Any],
    trial: Union[TrialSpace, TensorialVectorFunctionSpace],
    test: Union[TestSpace, TensorialVectorFunctionSpace],
    volume_geometry: ElementGeometry,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
    boundary_geometry: ElementGeometry | None = None,
    fe_coefficients: Dict[str, Union[FunctionSpace, None]] | None = None,
    component_index: int | None = None,
    is_symmetric: bool = False,
    rot_type: RotationType = RotationType.NO_ROTATION,
    docstring: str = "",
) -> Form:
    """
    Constructs an element matrix (:class:`~Form` object) from an integrand.

    Note that this function does not specify the loop structure of the kernel.
    Make sure to pass the result into the correct methods later on (specifically, take care that boundary integrals are
    actually executed on the boundary, volume integrals over all elements).

    Integrands are passed in as a callable (aka function). For instance:

    .. code-block:: python

        # The arguments of the function must begin with an asterisk (*), followed by
        # keyword arguments, followed by the unused keyword arguments (**_). All
        # keyword arguments must be members of the IntegrandSymbols class.
        #
        # The function must return the integrand. You can use functions from the
        # module hog.math_helpers module.
        #
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

    The callable (here `my_diffusion_integrand`, not `my_diffusion_integrand()`) is then passed to this function:

    .. code-block:: python

        form = process_integrand( my_diffusion_integrand, trial, test, ... )


    :param integrand: an integrand callable
    :param trial: the finite-element trial function space
    :param test: the finite-element test function space
    :param volume_geometry: the geometry of the volume element
    :param symbolizer: a Symbolizer instance
    :param blending: an optional blending map e.g., for curved geometries
    :param boundary_geometry: the geometry to integrate over for boundary integrals - passed through to the callable via
                              the IntegrandSymbols object
    :param fe_coefficients: a dictionary of type (str, FunctionSpace) that are names and spaces for scalar
                            finite-element function coefficients, they will be available to the callable as `k`
                            supply None as the FunctionSpace for a std::function-type coeff (only works for old forms)
    :param is_symmetric: whether the bilinear form is symmetric - this is exploited by the generator
    :param rot_type: whether the  operator has to be wrapped with rotation matrix and the type of rotation that needs
                     to be applied, only applicable for Vectorial spaces
    :param docstring: documentation of the integrand/bilinear form - will end up in the docstring of the generated code
    """

    if fe_coefficients is None:
        fe_coefficients = {}

    s = IntegrandSymbols()

    ####################
    # Element geometry #
    ####################

    s.volume_geometry = volume_geometry

    if boundary_geometry is not None:
        if boundary_geometry.dimensions != boundary_geometry.space_dimension - 1:
            raise HOGException(
                "Since you are integrating over a boundary, the boundary element's space dimension should be larger "
                "than its dimension."
            )

        if boundary_geometry.space_dimension != volume_geometry.space_dimension:
            raise HOGException("All geometries must be embedded in the same space.")

        s.boundary_geometry = boundary_geometry

    ##############
    # Tabulation #
    ##############

    tabulation = Tabulation(symbolizer)

    def _tabulate(
        factor: Union[sp.Expr, sp.Matrix], factor_name: str = "tabulated_and_untitled"
    ) -> sp.Matrix:
        if isinstance(factor, sp.Expr):
            factor = sp.Matrix([factor])

        return tabulation.register_factor(factor_name, factor)

    s.tabulate = _tabulate

    ################################
    # Scalar and matrix parameters #
    ################################

    free_symbols = set()

    def _scalars(symbol_names: str | List[str]) -> sp.Symbol | List[sp.Symbol]:
        nonlocal free_symbols
        symbs = sp.symbols(symbol_names)
        if isinstance(symbs, list):
            free_symbols |= set(symbs)
        elif isinstance(symbs, sp.Symbol):
            free_symbols.add(symbs)
        else:
            raise HOGException(
                f"I did not expect sp.symbols() to return whatever this is: {type(symbs)}"
            )
        return symbs

    def _matrix(base_name: str, rows: int, cols: int) -> sp.Matrix:
        symbs = _scalars(
            [f"{base_name}_{row}_{col}" for row in range(rows) for col in range(cols)]
        )
        return sp.Matrix(symbs).reshape(rows, cols)

    s.scalars = _scalars
    s.matrix = _matrix

    ###################
    # FE coefficients #
    ###################

    fe_coefficients_modified = {k: v for k, v in fe_coefficients.items()}

    special_name_of_micromesh_coeff = "micromesh"
    if isinstance(blending, ParametricMap):
        # We add a vector coefficient for the parametric mapping here.
        if special_name_of_micromesh_coeff in fe_coefficients:
            raise HOGException(
                f"You cannot use the name {special_name_of_micromesh_coeff} for your FE coefficient."
                f"It is reserved."
            )
        fe_coefficients_modified[
            special_name_of_micromesh_coeff
        ] = TensorialVectorFunctionSpace(
            LagrangianFunctionSpace(blending.degree, symbolizer)
        )

    s.k = dict()
    s.grad_k = dict()
    for name, coefficient_function_space in fe_coefficients_modified.items():
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
        s.k[name] = k
        s.grad_k[name] = grad_k

    ##############################
    # Jacobians and determinants #
    ##############################

    s.jac_a = symbolizer.jac_ref_to_affine(volume_geometry)
    s.jac_a_inv = symbolizer.jac_ref_to_affine_inv(volume_geometry)
    s.jac_a_abs_det = symbolizer.abs_det_jac_ref_to_affine()

    if boundary_geometry is not None:
        s.jac_a_boundary = symbolizer.jac_ref_to_affine(boundary_geometry)

    if isinstance(blending, IdentityMap):
        s.jac_b = sp.eye(volume_geometry.space_dimension)
        s.jac_b_inv = sp.eye(volume_geometry.space_dimension)
        s.jac_b_abs_det = 1
    elif isinstance(blending, ExternalMap):
        s.jac_b = jac_affine_to_physical(volume_geometry, symbolizer)
        s.jac_b_inv = inv(s.jac_b)
        s.jac_b_abs_det = abs(det(s.jac_b))
    elif isinstance(blending, ParametricMap):
        s.jac_a = sp.eye(volume_geometry.space_dimension)
        s.jac_a_inv = sp.eye(volume_geometry.space_dimension)
        s.jac_a_abs_det = 1

        if boundary_geometry is not None:
            raise HOGException(
                "Boundary integrals not tested with parametric mappings yet. "
                "We have to handle/set the affine Jacobian at the boundary appropriately.\n"
                "Dev note to future me: since we assume the boundary element to have the last ref coord zero, "
                "I suppose we can set this thing to:\n"
                " ⎡ 1 ⎤ \n"
                " ⎣ 0 ⎦ \n"
                "in 2D and to\n"
                " ⎡ 1 0 ⎤ \n"
                " | 0 1 | \n"
                " ⎣ 0 0 ⎦ \n"
                "in 3D."
            )

        s.jac_b = s.grad_k[special_name_of_micromesh_coeff].T
        s.jac_b_inv = inv(s.jac_b)
        s.jac_b_abs_det = abs(det(s.jac_b))

        s.x = s.k[special_name_of_micromesh_coeff]

    else:
        s.jac_b = symbolizer.jac_affine_to_blending(volume_geometry.space_dimension)
        s.jac_b_inv = symbolizer.jac_affine_to_blending_inv(
            volume_geometry.space_dimension
        )
        s.jac_b_abs_det = symbolizer.abs_det_jac_affine_to_blending()
        s.hessian_b = symbolizer.hessian_blending_map(volume_geometry.dimensions)

    if not isinstance(blending, ParametricMap):
        s.x = trafo_ref_to_physical(volume_geometry, symbolizer, blending)

    #######################################
    # Assembling the local element matrix #
    #######################################

    mat = create_empty_element_matrix(trial, test, volume_geometry)
    it = element_matrix_iterator(trial, test, volume_geometry)

    if component_index is not None:
        s.component_index = component_index

    for data in it:
        s.u = data.trial_shape
        s.grad_u = data.trial_shape_grad
        s.hessian_u = data.trial_shape_hessian

        s.v = data.test_shape
        s.grad_v = data.test_shape_grad
        s.hessian_v = data.test_shape_hessian

        mat[data.row, data.col] = integrand(**asdict(s))

    free_symbols_sorted = sorted(list(free_symbols), key=lambda x: str(x))

    if not rot_type == RotationType.NO_ROTATION:
        if rot_type == RotationType.PRE_AND_POST_MULTIPLY:
            if not trial == test:
                HOGException(
                    "Trial and Test spaces must be the same for RotationType.PRE_AND_POST_MULTIPLY"
                )

            if not trial.is_vectorial:
                raise HOGException(
                    "Rotation wrapper can only work with vectorial spaces"
                )

            rot_space = TensorialVectorFunctionSpace(
                LagrangianFunctionSpace(trial.degree, symbolizer)
            )

        elif rot_type == RotationType.PRE_MULTIPLY:
            if not test.is_vectorial:
                raise HOGException(
                    "Rotation wrapper can only work with vectorial spaces"
                )

            rot_space = TensorialVectorFunctionSpace(
                LagrangianFunctionSpace(test.degree, symbolizer)
            )

        elif rot_type == RotationType.POST_MULTIPLY:
            if not trial.is_vectorial:
                raise HOGException(
                    "Rotation wrapper can only work with vectorial spaces"
                )

            rot_space = TensorialVectorFunctionSpace(
                LagrangianFunctionSpace(trial.degree, symbolizer)
            )

        from hog.recipes.integrands.volume.rotation import rotation_matrix

        normal_fspace = rot_space.component_function_space

        normals = (
            ["nx_rotation", "ny_rotation"]
            if volume_geometry.dimensions == 2
            else ["nx_rotation", "ny_rotation", "nz_rotation"]
        )

        n_dof_symbols = []

        for normal in normals:
            if normal in fe_coefficients:
                raise HOGException(
                    f"You cannot use the name {normal} for your FE coefficient."
                    f"It is reserved."
                )
            if normal_fspace is None:
                raise HOGException("Invalid normal function space")
            else:
                _, nc_dof_symbols = fem_function_on_element(
                    normal_fspace,
                    volume_geometry,
                    symbolizer,
                    domain="reference",
                    function_id=normal,
                )

                n_dof_symbols.append(nc_dof_symbols)

        rotmat = rotation_matrix(
            rot_space.num_dofs(volume_geometry),
            int(rot_space.num_dofs(volume_geometry) / volume_geometry.dimensions),
            n_dof_symbols,
            volume_geometry,
        )

        rot_doc_string = ""

        if rot_type == RotationType.PRE_AND_POST_MULTIPLY:
            mat = rotmat * mat * rotmat.T
            rot_doc_string = "RKRᵀ uᵣ"
        elif rot_type == RotationType.PRE_MULTIPLY:
            mat = rotmat * mat
            rot_doc_string = "RK uᵣ"
        elif rot_type == RotationType.POST_MULTIPLY:
            mat = mat * rotmat.T
            rot_doc_string = "KRᵀ uᵣ"

        docstring += f"""

And the assembled FE matrix (K) is wrapped with a Rotation matrix (R) locally as below,

    {rot_doc_string}

where
    R : Rotation matrix calculated with the normal vector (n̂) at the DoF
    uᵣ: FE function but the components rotated at the boundaries according to the normal FE function passed
    
    n̂ : normals (vectorial space: {normal_fspace})
        * The passed normal vector must be normalized
        * The radial component of the rotated vector will be pointing in the given normal direction
        * If the normals are zero at a DoF, the rotation matrix is identity matrix
    """

    return Form(
        mat,
        tabulation,
        symmetric=is_symmetric,
        free_symbols=free_symbols_sorted,
        docstring=docstring,
    )
