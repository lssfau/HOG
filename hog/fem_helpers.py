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
from typing import Iterator, List, Optional, Tuple, Union, Dict
from dataclasses import dataclass

from hog.blending import (
    GeometryMap,
    ExternalMap,
    IdentityMap,
    AnnulusMap,
    IcosahedralShellMap,
)
from hog.element_geometry import (
    ElementGeometry,
    TriangleElement,
    TetrahedronElement,
    LineElement,
)
from hog.exception import HOGException
from hog.function_space import FunctionSpace
from hog.math_helpers import inv, det
from hog.multi_assignment import MultiAssignment
from hog.symbolizer import Symbolizer
from hog.external_functions import (
    BlendingFTriangle,
    BlendingFTetrahedron,
    BlendingDFTetrahedron,
    BlendingDFTriangle,
    ScalarVariableCoefficient2D,
    ScalarVariableCoefficient3D,
    VectorVariableCoefficient3D,
)
from hog.dof_symbol import DoFSymbol
import pystencils.astnodes as ast


def create_empty_element_matrix(
        trial: FunctionSpace, test: FunctionSpace, geometry: ElementGeometry
) -> sp.Matrix:
    """
    Returns a sympy matrix of the required size corresponding to the trial and test spaces, initialized with zeros.
    """
    return sp.zeros(test.num_dofs(geometry), trial.num_dofs(geometry))


@dataclass
class ElementMatrixData:
    """Class for holding the relevant shape functions and matrix indices to conveniently fill an element matrix."""

    trial_shape: sp.Expr
    test_shape: sp.Expr
    trial_shape_grad: sp.MatrixBase
    test_shape_grad: sp.MatrixBase
    row: int
    col: int


def element_matrix_iterator(
        trial: FunctionSpace, test: FunctionSpace, geometry: ElementGeometry
) -> Iterator[ElementMatrixData]:
    """Call this to create a generator to conveniently fill the element matrix."""
    for row, (psi, grad_psi) in enumerate(
            zip(test.shape(geometry), test.grad_shape(geometry))
    ):
        for col, (phi, grad_phi) in enumerate(
                zip(trial.shape(geometry), trial.grad_shape(geometry))
        ):
            yield ElementMatrixData(
                trial_shape=phi,
                trial_shape_grad=grad_phi,
                test_shape=psi,
                test_shape_grad=grad_psi,
                row=row,
                col=col,
            )


def trafo_ref_to_affine(
        geometry: ElementGeometry,
        symbolizer: Symbolizer,
        affine_points: Optional[List[sp.Matrix]] = None,
) -> sp.Matrix:
    """Returns the transformation of a point from the reference element to the affine (computational) element.

    :param geometry: the element geometry
    :param symbolizer: a symbolizer object
    :param affine_points: the symbols of the vertices of the affine element - if omitted, they default to the first d+1
                          vector symbols, useful for example if the trafo of two or more different elements is required
    """
    ref_symbols_vector = symbolizer.ref_coords_as_vector(geometry.dimensions)
    if affine_points is None:
        affine_points = symbolizer.affine_vertices_as_vectors(
            geometry.dimensions, geometry.num_vertices
        )
    else:
        if len(affine_points) != geometry.num_vertices:
            raise HOGException("The number of affine points must match the geometry.")

    trafo = sp.Matrix(
        [
            [
                affine_points[p][d] - affine_points[0][d]
                for p in range(1, geometry.num_vertices)
            ]
            for d in range(geometry.dimensions)
        ]
    )
    trafo = trafo * ref_symbols_vector
    trafo = trafo + sp.Matrix(
        [[affine_points[0][d]] for d in range(geometry.dimensions)]
    )
    return trafo


def trafo_affine_point_to_ref(
        geometry: ElementGeometry,
        symbolizer: Symbolizer,
        affine_eval_point: Optional[sp.Matrix] = None,
        affine_element_points: Optional[List[sp.Matrix]] = None,
) -> sp.Matrix:
    """Returns the transformation of an affine evaluation point on the affine element to the reference element.

    :param geometry: the element geometry
    :param symbolizer: a symbolizer object
    :param affine_eval_point: the point to transform to reference space
    :param affine_element_points: the symbols of the vertices of the affine element - if omitted, they default to the
                                  first d+1 vector symbols, useful for example if the trafo of two or more different
                                  elements is required
    """
    if affine_eval_point is None:
        affine_eval_point = symbolizer.affine_eval_coords_as_vector(
            dimensions=geometry.dimensions
        )
    else:
        if affine_eval_point.shape != (geometry.dimensions, 1):
            HOGException(
                f"Affine evaluation point has incorrect shape {affine_eval_point.shape}"
            )

    if affine_element_points is None:
        affine_element_points = symbolizer.affine_vertices_as_vectors(
            geometry.dimensions, geometry.num_vertices
        )
    else:
        if len(affine_element_points) != geometry.num_vertices:
            raise HOGException("The number of affine points must match the geometry.")

    trafo = affine_eval_point - affine_element_points[0]
    inv_jac = inv(
        jac_ref_to_affine(geometry, symbolizer, affine_points=affine_element_points)
    )
    trafo = inv_jac * trafo
    return trafo


def jac_ref_to_affine(
        geometry: ElementGeometry,
        symbolizer: Symbolizer,
        affine_points: Optional[List[sp.Matrix]] = None,
) -> sp.Matrix:
    """Returns the Jacobian of the transformation from the reference to the affine (computational) element.

    :param geometry: the element geometry
    :param symbolizer: a symbolizer object
    :param affine_points: the symbols of the vertices of the affine element - if omitted, they default to the first d+1
                          vector symbols, useful for example if the trafo of two or more different elements is required
    """
    ref_symbols_list = symbolizer.ref_coords_as_list(geometry.dimensions)
    trafo = trafo_ref_to_affine(geometry, symbolizer, affine_points=affine_points)
    return trafo.jacobian(ref_symbols_list)


def trafo_ref_to_physical(
        geometry: ElementGeometry, symbolizer: Symbolizer
) -> sp.Matrix:
    """Returns the transformation of a point in the reference space to the physical element."""
    blending_class: type[MultiAssignment]
    if isinstance(geometry, TriangleElement):
        blending_class = BlendingFTriangle
    elif isinstance(geometry, TetrahedronElement):
        blending_class = BlendingFTetrahedron
    else:
        raise HOGException("Blending not implemented for the passed element geometry.")

    t = trafo_ref_to_affine(geometry, symbolizer)
    phy = sp.zeros(geometry.dimensions, 1)
    for coord in range(geometry.dimensions):
        phy[coord] = blending_class(sp.Symbol("blend"), coord, *t)
    return phy


def jac_affine_to_physical(
        geometry: ElementGeometry, symbolizer: Symbolizer
) -> sp.Matrix:
    """Returns the Jacobian of the transformation from the affine (computational) to the physical element."""
    blending_class: type[MultiAssignment]
    if isinstance(geometry, TriangleElement):
        blending_class = BlendingDFTriangle
    elif isinstance(geometry, TetrahedronElement):
        blending_class = BlendingDFTetrahedron
    else:
        raise HOGException("Blending not implemented for the passed element geometry.")

    t = trafo_ref_to_affine(geometry, symbolizer)
    jac = sp.zeros(geometry.dimensions, geometry.dimensions)
    rows, cols = jac.shape
    for row in range(rows):
        for col in range(cols):
            output_arg = row * cols + col
            if len(t) != blending_class.num_input_args():
                raise HOGException(
                    f"Wrong number of input arguments to {blending_class.name()}."
                )
            if output_arg >= blending_class.num_output_args():
                raise HOGException(
                    f"{blending_class.name()} output argument index out of range."
                )
            jac[row, col] = blending_class(sp.Symbol("blend"), row * cols + col, *t)
    return jac


def blending_supported_geometries(blending: GeometryMap) -> List[ElementGeometry]:
    if isinstance(blending, IdentityMap):
        return [LineElement(), TriangleElement(), TetrahedronElement()]
    elif isinstance(blending, AnnulusMap):
        return [TriangleElement()]
    elif isinstance(blending, IcosahedralShellMap):
        return [TetrahedronElement()]
    elif isinstance(blending, GeometryMap):
        return []
    else:
        HOGException("blending must be an instance of GeometryMap")


def jac_blending_evaluate(
    symbolizer: Symbolizer, geometry: ElementGeometry
) -> sp.Matrix:
    affine_points = symbolizer.affine_vertices_as_vectors(
        geometry.dimensions, geometry.num_vertices
    )
    jac = geometry.blending.jacobian(
        trafo_ref_to_affine(geometry, symbolizer, affine_points)
    )
    return jac


def abs_det_jac_blending_eval_symbols(
    geometry: ElementGeometry, symbolizer: Symbolizer, q_pt: str = ""
) -> sp.Expr:
    jac_blending = symbolizer.jac_affine_to_blending(geometry.dimensions, q_pt)
    return det(jac_blending)


def jac_blending_inv_eval_symbols(
    geometry: ElementGeometry, symbolizer: Symbolizer, q_pt: str = ""
) -> sp.Matrix:
    jac_blending = symbolizer.jac_affine_to_blending(geometry.dimensions, q_pt)
    return inv(jac_blending)


def blending_quad_loop_assignments(
    geometry: ElementGeometry, symbolizer: Symbolizer, jac_evaluated: sp.Matrix
) -> List[ast.SympyAssignment]:
    quadrature_assignments = []

    jac_symbols = symbolizer.jac_affine_to_blending(geometry.dimensions)

    abs_det_jac_blending = abs_det_jac_blending_eval_symbols(geometry, symbolizer)

    jac_blending_inv = symbolizer.jac_affine_to_blending_inv(geometry.dimensions)
    jac_blending_inv_eval = jac_blending_inv_eval_symbols(geometry, symbolizer)

    dim = geometry.dimensions
    quadrature_assignments += [
        ast.SympyAssignment(jac_symbols[i, j], jac_evaluated[i, j], is_const=False)
        for i in range(dim)
        for j in range(dim)
    ]
    quadrature_assignments.append(
        ast.SympyAssignment(
            symbolizer.abs_det_jac_affine_to_blending(),
            abs_det_jac_blending,
            is_const=False,
        )
    )
    quadrature_assignments += [
        ast.SympyAssignment(
            jac_blending_inv[i, j], jac_blending_inv_eval[i, j], is_const=False
        )
        for i in range(dim)
        for j in range(dim)
    ]

    return quadrature_assignments


def scalar_space_dependent_coefficient(
        name: str,
        geometry: ElementGeometry,
        symbolizer: Symbolizer,
        blending: GeometryMap = IdentityMap(),
) -> Union[ScalarVariableCoefficient2D, ScalarVariableCoefficient3D]:
    """Returns a symbol for an externally defined, space dependent scalar coefficient."""
    if isinstance(blending, IdentityMap):
        t = trafo_ref_to_affine(geometry, symbolizer)
    elif isinstance(blending, ExternalMap):
        t = trafo_ref_to_physical(geometry, symbolizer)
    else:
        raise HOGException(
            "Reminder: revisit this function for compiled blending functions != id."
        )

    coeff_class: Union[
        type[ScalarVariableCoefficient2D], type[ScalarVariableCoefficient3D]
    ]
    if isinstance(geometry, TriangleElement):
        coeff_class = ScalarVariableCoefficient2D
    elif isinstance(geometry, TetrahedronElement):
        coeff_class = ScalarVariableCoefficient3D
    elif isinstance(geometry, LineElement):
        coeff_class = ScalarVariableCoefficient3D

    return coeff_class(sp.Symbol(name), 0, *t)


def vector_space_dependent_coefficient(
        name: str,
        geometry: ElementGeometry,
        symbolizer: Symbolizer,
        blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    """Returns a symbol for an externally defined, space dependent vector coefficient."""
    if isinstance(blending, IdentityMap):
        t = trafo_ref_to_affine(geometry, symbolizer)
    elif isinstance(blending, ExternalMap):
        t = trafo_ref_to_physical(geometry, symbolizer)
    else:
        raise HOGException(
            "Reminder: revisit this function for compiled blending functions != id."
        )

    if isinstance(geometry, TriangleElement):
        raise HOGException("2D is not supported")
    elif isinstance(geometry, TetrahedronElement):
        coeff_class = VectorVariableCoefficient3D

    eval = sp.Matrix([coeff_class(sp.Symbol(name), i, *t) for i in [0, 1, 2]])
    return eval


def create_dof_symbols(function_space: FunctionSpace,
                       geometry: ElementGeometry,
                       symbolizer: Symbolizer, function_id: str) -> List[DoFSymbol]:
    """ Creates a list of DoF symbols for a given function space."""
    return [
        DoFSymbol(
            d,
            function_space=function_space,
            dof_id=dof_id,
            function_id=function_id,
        )
        for dof_id, d in enumerate(
            symbolizer.dof_symbols_names_as_list(
                function_space.num_dofs(geometry), function_id
            )
        )
    ]


def fem_function_on_element(
        function_space: FunctionSpace,
        geometry: ElementGeometry,
        symbolizer: Symbolizer,
        domain: str = "affine",
        function_id: str = "f",
        dof_map: Optional[List[int]] = None,
        basis_eval: Union[str, List[sp.Expr]] = "default",
        dof_symbols : Optional[List[DoFSymbol]] = None
) -> Tuple[sp.Expr, List[DoFSymbol]]:
    """Returns an expression that is the element-local polynomial, either in affine or reference coordinates.

    The expression is build using DoFSymbol instances so that the DoFs can be resolved later.

    :param function_space: the underlying function space
    :param geometry: the element geometry
    :param symbolizer: a Symbolizer instance
    :param domain: 'reference' if evaluation shall be performed on the reference element or 'affine' if evaluation
                   shall be performed with affine coordinates
    :param function_id: a string identifier that identifies the corresponding (HyTeG) finite element function
    :param dof_map: this list can be used to specify (remap) the DoF ordering of the element
    :param dof_symbols: list of dof symbols that can be passed if they are generated outside of this function
    """

    dofs = create_dof_symbols(function_space, geometry, symbolizer,  function_id) if dof_symbols is None else dof_symbols

    if basis_eval != "default":
        assert (
                domain == "reference"
        ), "Tabulating the basis evaluation not implemented for affine domain."

    if domain == "reference":
        # On the reference domain, the reference coordinates symbols can be used directly, so no substitution
        # has to be performed for the shape functions.
        s = sp.zeros(1,1)
        for dof, phi in zip(
                dofs,
                function_space.shape(geometry=geometry, domain="reference", dof_map=dof_map)
                if basis_eval == "default"
                else basis_eval,
        ):
            s += dof * sp.Matrix([phi])
    elif domain == "affine":
        # On the affine / computational domain, the evaluation point is first mapped to reference space and then
        # the reference space coordinate symbols are substituted with the transformed point.
        eval_point_on_ref = trafo_affine_point_to_ref(geometry, symbolizer=symbolizer)
        s = sp.zeros(1, 1)
        for dof, phi in zip(
                dofs,
                function_space.shape(
                    geometry=geometry, domain="reference", dof_map=dof_map
                ),
        ):
            s += dof * sp.Matrix(phi.subs(
                {
                    ref_c: trafo_c
                    for ref_c, trafo_c in zip(
                        symbolizer.ref_coords_as_list(dimensions=geometry.dimensions),
                        eval_point_on_ref,
                    )
                })
            )
    else:
        raise HOGException(
            f"Invalid domain '{domain}': cannot evaluate local polynomial here."
        )
    return s, dofs



def fem_function_gradient_on_element(
        function_space: FunctionSpace,
        geometry: ElementGeometry,
        symbolizer: Symbolizer,
        domain: str = "affine",
        function_id: str = "f",
        dof_map: Optional[List[int]] = None,
        basis_eval: Union[str, List[sp.Expr]] = "default",
        dof_symbols: Optional[List[DoFSymbol]] = None
) -> sp.Matrix:
    """Returns an expression that is the gradient of the element-local polynomial, either in affine or reference coordinates.

    The expression is build using DoFSymbol instances so that the DoFs can be resolved later.

    :param function_space: the underlying function space
    :param geometry: the element geometry
    :param symbolizer: a Symbolizer instance
    :param domain: 'reference' if evaluation shall be performed on the reference element or 'affine' if evaluation
                   shall be performed with affine coordinates
    :param function_id: a string identifier that identifies the corresponding (HyTeG) finite element function
    :param dof_map: this list can be used to specify (remap) the DoF ordering of the element
    :param dof_symbols: list of dof symbols that can be passed if they are generated outside of this function

    """

    dofs = create_dof_symbols(function_space, geometry, symbolizer,  function_id) if dof_symbols is None else dof_symbols


    if basis_eval != "default":
        assert (
                domain == "reference"
        ), "Tabulating the basis evaluation not implemented for affine domain."

    if domain == "reference":
        # On the reference domain, the reference coordinates symbols can be used directly, so no substitution
        # has to be performed for the shape functions.
        s = sp.zeros(geometry.dimensions, 1)
        for dof, grad_phi in zip(
                dofs,
                function_space.grad_shape(geometry=geometry, domain="reference", dof_map=dof_map)
                if basis_eval == "default"
                else basis_eval,
        ):
            s += dof * grad_phi
    elif domain == "affine":
        raise HOGException(
            "Not implemented."
        )
    else:
        raise HOGException(
            f"Invalid domain '{domain}': cannot evaluate local polynomial here."
        )
    return s, dofs
