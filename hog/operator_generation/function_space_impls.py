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

import itertools
import numpy as np
import sympy as sp
from typing import List, Tuple, Set, Union, Dict

import pystencils as ps
from pystencils import Field, FieldType
from pystencils.backends.cbackend import CustomCodeNode

from hog.element_geometry import ElementGeometry
from hog.exception import HOGException
from hog.function_space import (
    FunctionSpace,
    LagrangianFunctionSpace,
    TensorialVectorFunctionSpace,
    N1E1Space,
)
from hog.operator_generation.indexing import (
    CellType,
    FaceType,
    VolumeDoFMemoryLayout,
    micro_element_to_vertex_indices,
    micro_vertex_to_edge_indices,
    micro_element_to_volume_indices,
    IndexingInfo,
)
from hog.operator_generation.types import HOGType


class FunctionSpaceImpl(ABC):
    """A FunctionSpaceImpl is the counterpart of a Function in HyTeG.

    An instance of this class represents an instance of one of HyTeG's Function
    classes. It is associated with a mathematical function space, an identifier
    (variable name) and can optionally be a pointer. This class is intended
    to abstract printing code for e.g. communication in an FE space and C++
    implementation agnostic way.

    It is impossible to create an instance of this abstract base class directly.
    Preferrably, use the static method `create_impl` which selects the correct
    derived class for the `FunctionSpace`.
    """

    def __init__(
        self,
        fe_space: FunctionSpace,
        name: str,
        type_descriptor: HOGType,
        is_pointer: bool,
    ):
        """Records the passed parameters in member variables.

        It is impossible to create an instance of this abstract base class
        directly. This __init__ method is to be called by the derived classes.
        Preferrably, use the static method `create_impl` which selects the
        correct derived class for the `FunctionSpace`.
        """
        self.fe_space = fe_space
        self.name = name
        self.type_descriptor = type_descriptor
        self.is_pointer = is_pointer

    @staticmethod
    def create_impl(  # type: ignore[no-untyped-def] # returns Self but Self is kind of new
        func_space: FunctionSpace,
        name: str,
        type_descriptor: HOGType,
        is_pointer: bool = False,
    ):
        """Takes a mathematical function space and produces the corresponding function space implementation.

        :param func_space:      The mathematical function space.
        :param name:            The C++ variable name/identifier of this function.
        :param type_descriptor: The value type of this function.
        :param is_pointer:      Whether the C++ variable is of a pointer type. Used
                                to print member accesses.
        """
        impl_class: type

        import hog.operator_generation.function_space_implementations.p0_space_impl
        import hog.operator_generation.function_space_implementations.p1_space_impl
        import hog.operator_generation.function_space_implementations.p2_space_impl
        import hog.operator_generation.function_space_implementations.n1e1_space_impl

        if isinstance(func_space, LagrangianFunctionSpace):
            if func_space.degree == 2:
                impl_class = P2FunctionSpaceImpl
            elif func_space.degree == 1:
                impl_class = P1FunctionSpaceImpl
            elif func_space.degree == 0:
                impl_class = P0FunctionSpaceImpl
            else:
                raise HOGException("Lagrangian function space must be of order 1 or 2.")
        elif isinstance(func_space, TensorialVectorFunctionSpace):
            if isinstance(func_space.component_function_space, LagrangianFunctionSpace):
                if func_space.component_function_space.degree == 1:
                    if func_space.single_component is None:
                        impl_class = P1VectorFunctionSpaceImpl
                    else:
                        impl_class = P1FunctionSpaceImpl
                elif func_space.component_function_space.degree == 2:
                    if func_space.single_component is None:
                        impl_class = P2VectorFunctionSpaceImpl
                    else:
                        impl_class = P2FunctionSpaceImpl
                else:
                    raise HOGException(
                        "TensorialVectorFunctionSpaces not supported for the chosen components."
                    )
            else:
                raise HOGException(
                    "TensorialVectorFunctionSpaces are only supported with Lagrangian component spaces."
                )
        elif isinstance(func_space, N1E1Space):
            impl_class = N1E1FunctionSpaceImpl
        else:
            raise HOGException("Unexpected function space")

        return impl_class(func_space, name, type_descriptor, is_pointer)

    def _create_field(self, name: str) -> Field:
        """Creates a pystencils field with a given name and stride 1."""
        f = Field.create_generic(
            name,
            1,
            dtype=self.type_descriptor.pystencils_type,
            field_type=FieldType.CUSTOM,
        )
        f.strides = tuple([1 for _ in f.strides])
        return f

    @abstractmethod
    def pre_communication(self, dim: int) -> str:
        """C++ code for the function space communication prior to the macro-primitive loop."""
        ...

    @abstractmethod
    def zero_halos(self, dim: int) -> str:
        """C++ code to zero the halos on a macro in HyTeG."""
        ...

    @abstractmethod
    def post_communication(
        self, dim: int, params: str, transform_basis: bool = True
    ) -> str:
        """C++ code for the function space communication after the macro-primitive loop."""
        ...

    @abstractmethod
    def pointer_retrieval(self, dim: int) -> str:
        """C++ code for retrieving pointers to the numerical data stored in the macro primitives `face` (2d) or `cell` (3d)."""
        ...

    @abstractmethod
    def invert_elementwise(self, dim: int) -> str:
        """C++ code for inverting each DoF of the linalg vector."""
        ...

    @abstractmethod
    def local_dofs(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        indexing_info: IndexingInfo,
        element_vertex_ordering: List[int],
    ) -> List[Field.Access]:
        """
        Returns a list of local dof values on the current element.

        The element_vertex_ordering is a list that specifies the ordering of the reference vertices.
        The ordering in which the DoFs are returned depends on this list. The "default" ordering is
        [0, 1, ..., num_vertices - 1].
        """
        ...

    @abstractmethod
    def func_type_string(self) -> str:
        """Returns the C++ function class type as a string."""
        ...

    @abstractmethod
    def includes(self) -> Set[str]:
        """Returns the import location for the function space in HyTeG."""
        ...

    @abstractmethod
    def dof_transformation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        element_vertex_ordering: List[int],
    ) -> Tuple[CustomCodeNode, sp.MatrixBase]:
        """Returns HyTeG code that computes the basis/DoF transformation and a symbolic expression of the result.

        In general FEM spaces, neighboring elements must agree on the
        orientation of shared mesh entities. For example, in N1E1 the
        orientations of edges define the sign of the DoFs, while in P3
        reorienting an edge swaps the two DoFs located on this edge. Basis
        transformations between these spaces allow us to view DoFs from
        neighboring elements in our local orientation.
        Recommended reading:
          Scroggs et al., "Construction of Arbitrary Order Finite Element Degree-
          of-Freedom Maps on Polygonal and Polyhedral Cell Meshes," 2022, doi:
          https://doi.org/10.1145/3524456.

        This function returns a matrix that, when applied to a local DoF vector,
        transforms the DoFs from the owning primitives to the basis of the
        current macro primitive.
        For example:
          - P1/P2: The identity.
          - P3:    A permutation matrix.
          - N1E1:  A diagonal matrix with |aᵢᵢ| = 1.

        If the micro element is in the interior of the macro-cell, the
        transformation is the identity. In matrix-free computations the
        communication is responsible for the basis transformations. Only when
        assembling operators into matrices, these transformations must be
        "baked into" the matrix because vectors are assembled locally and our
        communication routine is not performed during the operator application.

        The element_vertex_ordering is a list that specifies the ordering of the reference vertices.
        The ordering in which the DoFs are returned depends on this list. The "default" ordering is
        [0, 1, ..., num_vertices - 1].
        """
        ...

    def _deref(self) -> str:
        if self.is_pointer:
            return f"(*{self.name})"
        else:
            return self.name
