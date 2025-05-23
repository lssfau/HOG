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

from abc import ABC, abstractmethod
from pyclbr import Function
from typing import Any, List, NewType, Optional, Union
import sympy as sp

from hog.element_geometry import (
    ElementGeometry,
    TriangleElement,
    TetrahedronElement,
)
from hog.exception import HOGException
from hog.math_helpers import grad, hessian
from hog.symbolizer import Symbolizer


class FunctionSpace(ABC):
    """Representation of a finite element function space."""

    @property
    @abstractmethod
    def family(self) -> str:
        """The common name of this FEM space."""
        ...

    @property
    @abstractmethod
    def is_vectorial(self) -> bool:
        """Whether shape functions are scalar or vector valued."""
        ...

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Whether functions in this space are continuous across elements."""
        ...

    @property
    @abstractmethod
    def degree(self) -> int:
        """The polynomial degree of the shape functions."""
        ...

    @property
    @abstractmethod
    def symbolizer(self) -> Symbolizer:
        """The symbolizer used to construct this object."""
        ...

    @abstractmethod
    def shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.MatrixBase]:
        """The basis functions of this FEM space."""
        ...

    def grad_shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.MatrixBase]:
        """Returns a list containing the gradients of the shape functions on the element.

        Particularly, for each (vector- or scalar-valued) shape function N = (N_1, ..., N_n) returns the *transposed*
        Jacobian

                          ⎡∂N₁/∂x₁  ···  ∂Nₙ/∂x₁⎤
                          ⎢   ·             ·   ⎥
        grad(N) = J(N)ᵀ = ⎢   ·             ·   ⎥
                          ⎢   ·             ·   ⎥
                          ⎣∂N₁/∂xₖ  ···  ∂Nₙ/∂xₖ⎦

        i.e., the returned gradient is a column vector for scalar shape functions.

        :param dof_map: this list can be used to specify (remap) the DoF ordering of the element
        """
        if domain in ["ref", "reference"]:
            symbols = self.symbolizer.ref_coords_as_list(geometry.dimensions)
            basis_functions_gradients = [
                grad(f, symbols)
                for f in self.shape(geometry, domain=domain, dof_map=dof_map)
            ]
            return basis_functions_gradients

        raise HOGException(
            f"Gradient of shape function not available for domain type {domain}"
        )

    def hessian_shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.MatrixBase]:
        """Returns a list containing the hessians of the shape functions on the element.

        :param dof_map: this list can be used to specify (remap) the DoF ordering of the element
        """
        if domain in ["ref", "reference"]:
            symbols = self.symbolizer.ref_coords_as_list(geometry.dimensions)
            basis_functions_hessians = [
                hessian(f, symbols)
                for f in self.shape(geometry, domain=domain, dof_map=dof_map)
            ]
            return basis_functions_hessians

        raise HOGException(
            f"Hessian of shape function not available for domain type {domain}"
        )

    def num_dofs(self, geometry: ElementGeometry) -> int:
        """The number of DoFs per element."""
        return len(self.shape(geometry))

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        ...


TrialSpace = NewType("TrialSpace", FunctionSpace)
TestSpace = NewType("TestSpace", FunctionSpace)


class LagrangianFunctionSpace(FunctionSpace):
    """Representation of a finite element function spaces.

    Instances of this class provide the shape functions on the reference element as sympy expression.
    """

    def __init__(self, degree: int, symbolizer: Symbolizer):
        """Creates a FunctionSpace of Lagrangian family.

        :param degree: the order of the shape functions
        :param symbolizer: a Symbolizer instance
        """
        if degree not in [0, 1, 2]:
            raise HOGException("Only degree 0, 1 and 2 are supported.")

        self._degree = degree
        self._symbolizer = symbolizer

    @property
    def is_vectorial(self) -> bool:
        return False

    @property
    def is_continuous(self) -> bool:
        if self._degree == 0:
            return False
        else:
            return True

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def symbolizer(self) -> Symbolizer:
        return self._symbolizer

    @property
    def family(self) -> str:
        return "Lagrange"

    def shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.MatrixBase]:
        """Returns a list containing the shape functions on the element.

        :param dof_map: this list can be used to specify (remap) the DoF ordering of the element
        """
        if domain in ["ref", "reference"]:
            symbols = self.symbolizer.ref_coords_as_list(geometry.dimensions)

            basis_functions = []

            if (
                isinstance(geometry, TriangleElement)
                and self.family in ["Lagrange"]
                and self._degree == 0
            ):
                basis_functions = [sp.sympify(1)]

            elif (
                isinstance(geometry, TriangleElement)
                and self.family in ["Lagrange"]
                and self._degree == 1
            ):
                basis_functions = [
                    1 - symbols[0] - symbols[1],
                    symbols[0],
                    symbols[1],
                ]

            elif (
                isinstance(geometry, TriangleElement)
                and self.family in ["Lagrange"]
                and self._degree == 2
            ):
                x = symbols[0]
                y = symbols[1]
                basis_functions = [
                    2 * x**2 + 4 * x * y - 3 * x + 2 * y**2 - 3 * y + 1,
                    2 * x**2 - x,
                    2 * y**2 - y,
                    4 * x * y,
                    -4 * x * y - 4 * y**2 + 4 * y,
                    -4 * x**2 - 4 * x * y + 4 * x,
                ]

            elif (
                isinstance(geometry, TetrahedronElement)
                and self.family in ["Lagrange"]
                and self._degree == 0
            ):
                basis_functions = [sp.sympify(1)]

            elif (
                isinstance(geometry, TetrahedronElement)
                and self.family in ["Lagrange"]
                and self._degree == 1
            ):
                basis_functions = [
                    1 - symbols[0] - symbols[1] - symbols[2],
                    symbols[0],
                    symbols[1],
                    symbols[2],
                ]

            elif (
                isinstance(geometry, TetrahedronElement)
                and self.family in ["Lagrange"]
                and self._degree == 2
            ):
                xi_1 = symbols[0]
                xi_2 = symbols[1]
                xi_3 = symbols[2]
                basis_functions = [
                    (
                        2.0 * xi_1 * xi_1
                        + 4.0 * xi_1 * xi_2
                        + 4.0 * xi_1 * xi_3
                        - 3.0 * xi_1
                        + 2.0 * xi_2 * xi_2
                        + 4.0 * xi_2 * xi_3
                        - 3.0 * xi_2
                        + 2.0 * xi_3 * xi_3
                        - 3.0 * xi_3
                        + 1.0
                    ),
                    (2.0 * xi_1 * xi_1 - 1.0 * xi_1),
                    (2.0 * xi_2 * xi_2 - 1.0 * xi_2),
                    (2.0 * xi_3 * xi_3 - 1.0 * xi_3),
                    (4.0 * xi_2 * xi_3),
                    (4.0 * xi_1 * xi_3),
                    (4.0 * xi_1 * xi_2),
                    (
                        -4.0 * xi_1 * xi_3
                        - 4.0 * xi_2 * xi_3
                        - 4.0 * xi_3 * xi_3
                        + 4.0 * xi_3
                    ),
                    (
                        -4.0 * xi_1 * xi_2
                        - 4.0 * xi_2 * xi_2
                        - 4.0 * xi_2 * xi_3
                        + 4.0 * xi_2
                    ),
                    (
                        -4.0 * xi_1 * xi_1
                        - 4.0 * xi_1 * xi_2
                        - 4.0 * xi_1 * xi_3
                        + 4.0 * xi_1
                    ),
                ]

            else:
                raise HOGException(
                    "Basis functions not implemented for the specified element type and geometry."
                )

            if dof_map:
                raise HOGException("DoF reordering not implemented.")

            basis_functions = [sp.Matrix([b]) for b in basis_functions]
            return basis_functions

        raise HOGException(f"Shape function not available for domain type {domain}")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LagrangianFunctionSpace):
            return False
        return self.family == other.family and self._degree == other._degree

    def __str__(self):
        return f"{self.family}, degree: {self._degree}"

    def __repr__(self):
        return str(self)


class TensorialVectorFunctionSpace(FunctionSpace):
    """
    Given a scalar function space, this class can be used to construct a tensorial vector-function space.

    Each shape function of the resulting tensorial space has only one non-zero component that is set to one of the
    shape functions of the given scalar function space.

    For instance, a tensorial function space of dim d = 3 that is based on a P1 (Lagrangian) scalar function space with
    n shape functions N_1, ..., N_n has the shape functions::

      (N_1, 0, 0),
      ...
      (N_n, 0, 0),
      (0, N_1, 0),
      ...
      (0, N_n, 0),
      (0, 0, N_1),
      ...
      (0, 0, N_n).


    This class also enables specifying only a single component to get the shape functions, e.g. only for component 1
    (starting to count components at 0)::

      (0, N_1, 0),
      ...
      (0, N_n, 0),

    """

    def __init__(
        self, function_space: FunctionSpace, single_component: Union[None, int] = None
    ):
        """
        Initializes a tensorial vector function space from a scalar function space.

        :param function_space: the (scalar) component function space
        :param single_component: set to the component that shall be non-zero - None if all components shall be present
        """
        self._component_function_space = function_space
        self._single_component = single_component

    @property
    def is_vectorial(self) -> bool:
        return True

    @property
    def degree(self) -> int:
        return self._component_function_space.degree

    @property
    def symbolizer(self) -> Symbolizer:
        return self._component_function_space.symbolizer

    @property
    def family(self) -> str:
        return self._component_function_space.family

    @property
    def is_continuous(self) -> bool:
        return self._component_function_space.is_continuous

    @property
    def component_function_space(self) -> FunctionSpace:
        return self._component_function_space

    @property
    def single_component(self) -> Union[int, None]:
        return self._single_component

    def _to_vector(
        self, phi: sp.MatrixBase, component: int, dimensions: int
    ) -> sp.MatrixBase:
        if phi.shape != (1, 1):
            raise HOGException("Component of tensorial space must be scalar.")
        return sp.Matrix(
            [[phi if c == component else sp.sympify(0)] for c in range(dimensions)]
        )

    def shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.MatrixBase]:
        dim = geometry.dimensions

        shape_functions = self._component_function_space.shape(
            geometry, domain, dof_map
        )

        if self._single_component is None:
            return [
                self._to_vector(phi, c, dim)
                for c in range(dim)
                for phi in shape_functions
            ]
        else:
            return [
                self._to_vector(phi, self._single_component, dim)
                for phi in shape_functions
            ]

    def __eq__(self, other: Any) -> bool:
        if type(self) != type(other):
            return False
        if not hasattr(other, "_component_function_space"):
            return False
        if not hasattr(other, "single_component"):
            return False
        return (
            self.component_function_space == other.component_function_space
            and self.single_component == other.single_component
        )

    def __str__(self):
        component = (
            ""
            if self.single_component is None
            else f", component {self.single_component}"
        )
        return f"TensorialVectorSpace({self._component_function_space}{component})"

    def __repr__(self):
        return str(self)


class EnrichedGalerkinFunctionSpace(FunctionSpace):
    def __init__(
        self,
        symbolizer: Symbolizer,
    ):
        self._symbolizer = symbolizer
        self._affine = None

    @property
    def family(self) -> str:
        return "EnrichedGalerkin"

    @property
    def is_vectorial(self) -> bool:
        return True

    def init_affine(self, affine):
        self._affine = affine

    @property
    def is_continuous(self) -> bool:
        return False

    @property
    def degree(self) -> int:
        return 1

    @property
    def symbolizer(self) -> Symbolizer:
        return self._symbolizer

    def shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.Expr]:
        if domain not in ["reference", "ref"] or dof_map is not None:
            raise HOGException(
                "Unsupported parameters for EnrichedGalerkinFunctionSpace"
            )
        symbols = self._symbolizer.ref_coords_as_list(geometry.dimensions)
        midpoint = sp.Matrix(
            [
                [sp.Rational(1, geometry.dimensions + 1)]
                for i in range(geometry.dimensions)
            ]
        )
        x = sp.Matrix([[symbols[i]] for i in range(geometry.dimensions)])
        # return [self._affine * (x - midpoint)]
        return [(x - midpoint)]

    def grad_shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.MatrixBase]:
        # return [self._affine]
        return sp.Matrix([[1, 0], [0, 1]])

    def num_dofs(self, geometry: ElementGeometry) -> int:
        """Returns the number of DoFs per element."""
        return len(self.shape(geometry))

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other)

    def __str__(self):
        return f"EnrichedDG"

    def __repr__(self):
        return str(self)


class N1E1Space(FunctionSpace):
    """Nedelec edge elements of type I and order 1 for problems in H(curl)."""

    def __init__(self, symbolizer: Symbolizer):
        self._symbolizer = symbolizer

    @property
    def is_vectorial(self) -> bool:
        return True

    @property
    def is_continuous(self) -> bool:
        return False

    @property
    def degree(self) -> int:
        return 1

    @property
    def symbolizer(self) -> Symbolizer:
        return self._symbolizer

    @property
    def family(self) -> str:
        return "N1E1"

    def shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.Expr]:
        if not isinstance(geometry, TetrahedronElement):
            raise HOGException(
                f"N1E1Space is only implemented for tetrahedral elements"
            )

        if domain in ["ref", "reference"]:
            symbols = self.symbolizer.ref_coords_as_list(geometry.dimensions)
            x, y, z = symbols

            basis = [
                sp.Matrix([0, -z, y]),
                sp.Matrix([-z, 0, x]),
                sp.Matrix([-y, x, 0]),
                sp.Matrix([z, z, -x - y + 1]),
                sp.Matrix([y, -x - z + 1, y]),
                sp.Matrix([-y - z + 1, x, x]),
            ]

            if dof_map:
                raise HOGException("DoF reordering not implemented.")

            return basis

        raise HOGException(f"Shape function not available for domain type {domain}")

    def grad_shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.MatrixBase]:
        return 6 * [None]

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other)

    def __str__(self):
        return f"N1E1"

    def __repr__(self):
        return str(self)


class P2PlusBubbleSpace(FunctionSpace):
    """
    Space of continuous piecewise quadratic functions enhanced with local bubbles vanishing on element boundaries.

    In the 2D case the triplet :math:`(K,P,\Sigma)` describing the Finite Element is given by the following:

    #. K represents the standard 2D-simplex, i.e. the unit triangle.
    #. The local function space :math:`P` on :math:`K` is given by
       :math:`P = \mathbb{P}_2 \oplus \ell_1 \ell_2 \ell_3 \mathbb{R}`.
       Here the :math:`\ell_i` represent barycentric coordinates. Thus, any function :math:`v\in P` can uniquely be
       composed into :math:`v = p + b`, where p is a quadratic polynomial and b a bubble function.
    #. The set :math:`\Sigma` of linear functionals :math:`L_k : P \rightarrow \mathbb{R}, k = 1,\ldots,7` is given
       by

    .. math::
            L_m( v ) = v(x_m), m = 1,\ldots 6 \\
            L_7( v ) = b(x_7) 

    Here :math:`x_1, \ldots, x_3` represents the vertices of :math:`K`, :math:`x_4,\ldots,x_6` the midpoints
    of its edges and :math:`x_7` its barycenter.
    """

    def __init__(self, symbolizer: Symbolizer):
        self._symbolizer = symbolizer

    @property
    def is_vectorial(self) -> bool:
        return False

    @property
    def is_continuous(self) -> bool:
        return True

    @property
    def degree(self) -> int:
        return 2

    @property
    def symbolizer(self) -> Symbolizer:
        return self._symbolizer

    @property
    def family(self) -> str:
        return "P2 enhanced with Bubble"

    def shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.MatrixBase]:
        """Returns a list containing the shape functions on the element.

        :param dof_map: this list can be used to specify (remap) the DoF ordering of the element
        """
        if domain in ["ref", "reference"]:
            symbols = self.symbolizer.ref_coords_as_list(geometry.dimensions)

            basis_functions = []

            if isinstance(geometry, TriangleElement):
                L_1 = 1 - symbols[0] - symbols[1]
                L_2 = symbols[0]
                L_3 = symbols[1]
                basis_functions = [
                    L_1 * (2 * L_1 - 1),
                    L_2 * (2 * L_2 - 1),
                    L_3 * (2 * L_3 - 1),
                    4 * L_2 * L_3,
                    4 * L_1 * L_3,
                    4 * L_1 * L_2,
                    27 * L_1 * L_2 * L_3,
                ]

            elif isinstance(geometry, TetrahedronElement):
                raise HOGException(
                    f"P2PlusBubbleSpace is currently only implemented for triangle elements"
                )
            else:
                raise HOGException(
                    "Basis functions not implemented for the specified element type and geometry."
                )

            if dof_map:
                raise HOGException("DoF reordering not implemented.")

            basis_functions = [sp.Matrix([b]) for b in basis_functions]
            return basis_functions

        raise HOGException(
            "Basis functions not implemented for the specified element type and geometry."
        )

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other)

    def __str__(self):
        return f"P2PlusBubble"

    def __repr__(self):
        return str(self)


class DGFunctionSpace(FunctionSpace):
    """
    Space of discontinuous piecewise polynomial functions (Discontinuous-Galerkin)
    """

    def __init__(self, degree: int, symbolizer: Symbolizer):
        """Creates a FunctionSpace of the Discontinuous Galerkin family.

        :param degree: the order of the shape functions
        :param symbolizer: a Symbolizer instance
        """
        if degree not in [1, 2]:
            raise HOGException("Only degrees 1 and 2 are supported.")

        self._degree = degree
        self._symbolizer = symbolizer

    @property
    def is_vectorial(self) -> bool:
        return False

    @property
    def is_continuous(self) -> bool:
        return False

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def symbolizer(self) -> Symbolizer:
        return self._symbolizer

    @property
    def family(self) -> str:
        return "DG"

    def shape(
        self,
        geometry: ElementGeometry,
        domain: str = "reference",
        dof_map: Optional[List[int]] = None,
    ) -> List[sp.MatrixBase]:
        """Returns a list containing the shape functions on the element.

        :param dof_map: this list can be used to specify (remap) the DoF ordering of the element
        """
        if domain in ["ref", "reference"]:
            symbols = self.symbolizer.ref_coords_as_list(geometry.dimensions)

            basis_functions = []

            if (
                isinstance(geometry, TriangleElement)
                and self.family in ["DG"]
                and self._degree == 1
            ):
                basis_functions = [
                    1 - symbols[0] - symbols[1],
                    symbols[0],
                    symbols[1],
                ]

            elif (
                isinstance(geometry, TriangleElement)
                and self.family in ["DG"]
                and self._degree == 2
            ):
                x = symbols[0]
                y = symbols[1]
                basis_functions = [
                    2 * x**2 + 4 * x * y - 3 * x + 2 * y**2 - 3 * y + 1,
                    2 * x**2 - x,
                    2 * y**2 - y,
                    4 * x * y,
                    -4 * x * y - 4 * y**2 + 4 * y,
                    -4 * x**2 - 4 * x * y + 4 * x,
                ]

            elif (
                isinstance(geometry, TetrahedronElement)
                and self.family in ["DG"]
                and self._degree == 1
            ):
                basis_functions = [
                    1 - symbols[0] - symbols[1] - symbols[2],
                    symbols[0],
                    symbols[1],
                    symbols[2],
                ]

            elif (
                isinstance(geometry, TetrahedronElement)
                and self.family in ["DG"]
                and self._degree == 2
            ):
                xi_1 = symbols[0]
                xi_2 = symbols[1]
                xi_3 = symbols[2]
                basis_functions = [
                    (
                        2.0 * xi_1 * xi_1
                        + 4.0 * xi_1 * xi_2
                        + 4.0 * xi_1 * xi_3
                        - 3.0 * xi_1
                        + 2.0 * xi_2 * xi_2
                        + 4.0 * xi_2 * xi_3
                        - 3.0 * xi_2
                        + 2.0 * xi_3 * xi_3
                        - 3.0 * xi_3
                        + 1.0
                    ),
                    (2.0 * xi_1 * xi_1 - 1.0 * xi_1),
                    (2.0 * xi_2 * xi_2 - 1.0 * xi_2),
                    (2.0 * xi_3 * xi_3 - 1.0 * xi_3),
                    (4.0 * xi_2 * xi_3),
                    (4.0 * xi_1 * xi_3),
                    (4.0 * xi_1 * xi_2),
                    (
                        -4.0 * xi_1 * xi_3
                        - 4.0 * xi_2 * xi_3
                        - 4.0 * xi_3 * xi_3
                        + 4.0 * xi_3
                    ),
                    (
                        -4.0 * xi_1 * xi_2
                        - 4.0 * xi_2 * xi_2
                        - 4.0 * xi_2 * xi_3
                        + 4.0 * xi_2
                    ),
                    (
                        -4.0 * xi_1 * xi_1
                        - 4.0 * xi_1 * xi_2
                        - 4.0 * xi_1 * xi_3
                        + 4.0 * xi_1
                    ),
                ]

            else:
                raise HOGException(
                    "Basis functions not implemented for the specified element type and geometry."
                )

            if dof_map:
                raise HOGException("DoF reordering not implemented.")

            basis_functions = [sp.Matrix([b]) for b in basis_functions]
            return basis_functions

        raise HOGException(f"Shape function not available for domain type {domain}")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LagrangianFunctionSpace):
            return False
        return self.family == other.family and self._degree == other._degree

    def __str__(self):
        return f"{self.family}, degree: {self._degree}"

    def __repr__(self):
        return str(self)
