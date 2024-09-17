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

from dataclasses import dataclass, field
from enum import auto, Enum
import logging
from typing import Dict, List, Optional, Set, Tuple, Union
import os
from textwrap import indent

import pystencils.astnodes
import numpy as np
import sympy as sp

from hog.cpp_printing import (
    CppClass,
    CppFilePair,
    CppMethod,
    CppMethodWithVariants,
    CppMemberVariable,
    CppConstructor,
    CppVariable,
    CppComment,
    CppInclude,
    CppFileRepresentation,
    apply_clang_format,
)
from hog.hyteg_code_generation import (
    COPYRIGHT_COMMENT,
    GENERATED_COMMENT,
    PRAGMA_ONCE,
    GCC_WARNING_WORKAROUND,
)
from hog.logger import TimedLogger
from hog.operator_generation.function_space_impls import FunctionSpaceImpl
from hog.operator_generation.pystencils_extensions import create_generic_fields

import pystencils as ps
from pystencils import (
    Backend,
    CreateKernelConfig,
    Target,
    TypedSymbol,
    Field,
)
from pystencils.astnodes import (
    ArrayDeclaration,
    Block,
    KernelFunction,
    LoopOverCoordinate,
    SympyAssignment,
    ForLoop,
)
from pystencils.cpu.cpujit import make_python_function
from pystencils.transformations import resolve_field_accesses
from pystencils.typing.transformations import add_types
from pystencils.backends.cbackend import CustomCodeNode
from pystencils.typing.typed_sympy import FieldPointerSymbol

from hog.integrand import Form
from hog.ast import Operations, count_operations
from hog.blending import GeometryMap
import hog.code_generation
import hog.cse
from hog.dof_symbol import DoFSymbol
from hog.element_geometry import (
    ElementGeometry,
    TriangleElement,
    TetrahedronElement,
)
from hog.exception import HOGException
from hog.operator_generation.indexing import (
    all_element_types,
    element_vertex_coordinates,
    IndexingInfo,
    FaceType,
    CellType,
)
from hog.operator_generation.kernel_types import KernelWrapperType
from hog.operator_generation.kernel_types import Assemble, KernelType
from hog.operator_generation.loop_strategies import (
    LoopStrategy,
    SAWTOOTH,
    BOUNDARY,
    CUBES,
)
from hog.operator_generation.optimizer import Optimizer, Opts
from hog.quadrature import QuadLoop, Quadrature
from hog.symbolizer import Symbolizer
from hog.operator_generation.types import HOGType
from hog.integrand import RotationType


class MacroIntegrationDomain(Enum):
    """Enum type to specify where to integrate."""

    # Integration over the volume element.
    VOLUME = "Volume"

    # Integration over the boundary of the domain (for forms like ∫ ... d(∂Ω)).
    #
    # Note: Having one flag for all boundaries of one "type" is not very flexible.
    #       At some point there should be additional logic that uses HyTeG's BoundaryUIDs.
    #       To avoid ambiguity, the operator should (at least if there are boundary integrals) have a
    #       hyteg::BoundaryCondition constructor parameter, and for each boundary integral one hyteg::BoundaryUID
    #       parameter. Then the BC is tested by comparing those for each facet.
    #
    #       Like so (application pseudocode in HyTeG):
    #
    #         // generating an operator with one volume and one free-slip boundary integral
    #         //   ∫ F dx + ∫ G ds
    #         // where ds corresponds to integrating over parts of the boundary that are marked with a specific
    #         // BoundaryUID
    #
    #         // see HyTeG's documentation and/or BC tutorial
    #         BoundaryCondition someBC( ... );
    #
    #         // setting up the BCs
    #         BoundaryUID freeslipBC = someBC.createFreeslipBC( ... );
    #
    #         // the generated operator
    #         MyFancyFreeSlipOperator op( storage,
    #                                     ...,
    #                                     someBC,    // this is what will be tested against - no ambiguity because src
    #                                                // and dst func might have different BCs!
    #                                     freeslipBC // this is the BCUID that will be used for the boundary integral
    #                                                // if there are more boundary integrals there are more UIDs in the
    #                                                // constructor (this BCUID is linked to 'ds' above)
    #                                     );
    #

    DOMAIN_BOUNDARY = "Domain boundary"


@dataclass
class IntegrationInfo:
    """Data associated with one integral term and the corresponding loop pattern."""

    geometry: ElementGeometry  # geometry of the element, e.g. tetrahedron
    integration_domain: (
        MacroIntegrationDomain  # entity of geometry to integrate over, e.g. facet
    )

    quad: Quadrature  # quadrature over integration domain of geometry, e.g. triangle
    blending: GeometryMap

    tables: List[ArrayDeclaration]
    quad_loop: Optional[QuadLoop]
    mat: sp.MatrixBase

    loop_strategy: LoopStrategy

    optimizations: Set[Opts]

    free_symbols: List[sp.Symbol]

    name: str = "unknown_integral"
    docstring: str = ""
    boundary_uid_name: str = ""
    integrand_name: str = "unknown_integrand"

    def _str_(self):
        return f"Integration Info: {self.name}, {self.geometry}, {self.integration_domain}, mat shape {self.mat.shape}, quad degree {self.quad.degree}, blending {self.blending}"

    def _repr_(self):
        return str(self)


@dataclass
class OperatorMethod:
    """Collection of kernels and metadata required for each method of an operator."""

    kernel_wrapper_type: KernelWrapperType
    kernel_functions: List[KernelFunction]
    platform_dependent_funcs: List[Dict[str, KernelFunction]]
    operation_counts: List[str]
    integration_infos: List[IntegrationInfo]


class CppClassFiles(Enum):
    """How to split a Cpp class to a set of files."""

    HEADER_ONLY = auto()
    HEADER_AND_IMPL = auto()
    HEADER_IMPL_AND_VARIANTS = auto()


def micro_vertex_permutation_for_facet(
    volume_geometry: ElementGeometry,
    element_type: Union[FaceType, CellType],
    facet_id: int,
) -> List[int]:
    """
    Provides a re-ordering of the micro-vertices such that the facet of the micro-element that coincides with the
    macro-facet (which is given by the facet_id parameter) is spanned by the first three returned vertex positions.

    The reordering can then for instance be executed by

    ```python
        element_vertex_order = shuffle_order_for_element_micro_vertices( ... )

        el_vertex_coordinates = [
            el_vertex_coordinates[i] for i in element_vertex_order
        ]
    ```

    """

    if volume_geometry == TriangleElement():

        if element_type == FaceType.BLUE:
            return [0, 1, 2]

        shuffle_order_gray = {
            0: [0, 1, 2],
            1: [0, 2, 1],
            2: [1, 2, 0],
        }

        return shuffle_order_gray[facet_id]

    elif volume_geometry == TetrahedronElement():

        if element_type == CellType.WHITE_DOWN:
            return [0, 1, 2, 3]

        # All element types but WHITE_UP only overlap with a single macro-facet.
        # It's a different element type for each facet. WHITE_DOWN is never at the boundary.
        shuffle_order: Dict[Union[FaceType, CellType], Dict[int, List[int]]] = {
            CellType.WHITE_UP: {
                0: [0, 1, 2, 3],
                1: [0, 1, 3, 2],
                2: [0, 2, 3, 1],
                3: [1, 2, 3, 0],
            },
            CellType.BLUE_UP: {0: [0, 1, 2, 3]},
            CellType.GREEN_UP: {1: [0, 2, 3, 1]},
            CellType.BLUE_DOWN: {2: [0, 1, 3, 2]},
            CellType.GREEN_DOWN: {3: [1, 2, 3, 0]},
        }

        return shuffle_order[element_type][facet_id]

    else:
        raise HOGException("Not implemented.")


class HyTeGElementwiseOperator:
    """
    This class handles the code generation of HyTeG-type 'elementwise' operators.

    TODO: extend documentation!
    """

    INCLUDES = {
        "core/DataTypes.h",
        "hyteg/communication/Syncing.hpp",
        "hyteg/edgedofspace/EdgeDoFMacroCell.hpp",
        "hyteg/primitivestorage/PrimitiveStorage.hpp",
        "hyteg/LikwidWrapper.hpp",
        "hyteg/boundary/BoundaryConditions.hpp",
        "hyteg/types/types.hpp",
    }

    VAR_NAME_MICRO_EDGES_PER_MACRO_EDGE = "micro_edges_per_macro_edge"
    VAR_NAME_MICRO_EDGES_PER_MACRO_EDGE_FLOAT = "micro_edges_per_macro_edge_float"

    @staticmethod
    def _var_names_macro_vertex_coords(i: int, comp: int) -> str:
        return f"macro_vertex_coord_id_{i}comp{comp}"

    def __init__(
        self,
        name: str,
        symbolizer: Symbolizer,
        kernel_wrapper_types: List[KernelWrapperType],
        type_descriptor: HOGType,
    ):
        self.name = name
        self.symbolizer = symbolizer
        self.kernel_wrapper_types = kernel_wrapper_types  # type of kernel: e.g. GEMV

        # Each IntegrationInfo object represents one integral of the weak formulation.
        self.integration_infos: Dict[int, List[IntegrationInfo]] = {}

        # set the precision in which the operations are to be performed
        self._type_descriptor = type_descriptor

        # coefficients
        self.coeffs: Dict[str, FunctionSpaceImpl] = {}
        # implementations for each kernel, generated at a later stage
        self.operator_methods: List[OperatorMethod] = []

    def _add_integral(
        self,
        name: str,
        volume_geometry: ElementGeometry,
        integration_domain: MacroIntegrationDomain,
        quad: Quadrature,
        blending: GeometryMap,
        form: Form,
        loop_strategy: LoopStrategy,
        boundary_uid_name: str,
        optimizations: Set[Opts],
        integrand_name: str | None = None,
    ) -> None:
        """
        Use this method to add integrals to the operator if you know what you are doing. There are helper methods for
        adding integrals that are a little simpler to use.

        :param name: some name for this integral (no spaces please)
        :param volume_geometry: the volume element (even for boundary integrals pass the element with dim == space_dim)
        :param integration_domain: where to integrate - see MacroIntegrationDomain
        :param quad: the employed quadrature scheme
        :param blending: the same geometry map that has been passed to the form
        :param form: the integrand
        :param loop_strategy: loop pattern over the refined macro-volume - must somehow be compatible with the
                              integration domain
        :param boundary_uid_name: string that defines the name of the boundary UID if this is a boundary integral
                                  the parameter is ignored for volume integrals
        :param optimizations: optimizations that shall be applied to this integral
        :param integrand_name: while each integral is a separate kernel with a name, for some types of integrals (e.g.,
                               boundary integrals) more than one kernel with the same integrand is added - for some
                               features (e.g., symbol naming) it is convenient to be able to identify all those
                               integrals by a string - this (optional) integrand_name does not have to be unique (other
                               than the name parameter which has to be unique) but can be the same for all integrals
                               with different domains but the same integrand
        """

        if "".join(name.split()) != name:
            raise HOGException(
                "Please give the integral an identifier without white space."
            )

        if integrand_name is None:
            integrand_name = name

        if volume_geometry.space_dimension in self.integration_infos:
            if name in [
                ii.name
                for ii in self.integration_infos[volume_geometry.space_dimension]
            ]:
                raise HOGException(f"Integral with name {name} already added!")

        if volume_geometry.space_dimension not in [2, 3]:
            raise HOGException("Only supporting 2D and 3D. Dim should be in [2, 3]")

        if integration_domain == MacroIntegrationDomain.VOLUME and not (
            isinstance(loop_strategy, SAWTOOTH) or isinstance(loop_strategy, CUBES)
        ):
            raise HOGException("Invalid loop strategy for volume integrals.")

        tables = []
        quad_loop = None
        mat = form.integrand

        if Opts.TABULATE in optimizations:
            mat = form.tabulation.resolve_table_accesses(mat, self._type_descriptor)
            with TimedLogger(f"constructing tables", level=logging.DEBUG):
                tables = form.tabulation.construct_tables(quad, self._type_descriptor)
        else:
            mat = form.tabulation.inline_tables(mat)

        if Opts.QUADLOOPS in optimizations:
            quad_loop = QuadLoop(
                self.symbolizer,
                quad,
                mat,
                self._type_descriptor,
                form.symmetric,
                blending,
            )
            mat = quad_loop.mat
        else:
            with TimedLogger(
                f"integrating {mat.shape[0] * mat.shape[1]} expressions",
                level=logging.DEBUG,
            ):
                for row in range(mat.rows):
                    for col in range(mat.cols):
                        if form.symmetric and row > col:
                            mat[row, col] = mat[col, row]
                        else:
                            mat[row, col] = quad.integrate(
                                mat[row, col], self.symbolizer, blending
                            )

        if not form.rotmat.is_zero_matrix:
            if form.rot_type == RotationType.PRE_AND_POST_MULTIPLY:
                mat = form.rotmat * mat * form.rotmat.T
            elif form.rot_type == RotationType.PRE_MULTIPLY:
                mat = form.rotmat * mat
            elif form.rot_type == RotationType.POST_MULTIPLY:
                mat = mat * form.rotmat.T
            else:
                raise HOGException("Not implemented")


        if volume_geometry.space_dimension not in self.integration_infos:
            self.integration_infos[volume_geometry.space_dimension] = []

        self.integration_infos[volume_geometry.space_dimension].append(
            IntegrationInfo(
                name=name,
                geometry=volume_geometry,
                integration_domain=integration_domain,
                quad=quad,
                blending=blending,
                tables=tables,
                quad_loop=quad_loop,
                mat=mat,
                docstring=form.docstring,
                loop_strategy=loop_strategy,
                boundary_uid_name=boundary_uid_name,
                optimizations=optimizations,
                free_symbols=form.free_symbols,
                integrand_name=integrand_name,
            )
        )

    def add_volume_integral(
        self,
        name: str,
        volume_geometry: ElementGeometry,
        quad: Quadrature,
        blending: GeometryMap,
        form: Form,
        loop_strategy: LoopStrategy,
        optimizations: Union[None, Set[Opts]] = None,
    ) -> None:
        """
        Adds a volume integral to the operator. Wrapper around _add_integral() for volume integrals.

        :param name: some name for this integral (no spaces please)
        :param volume_geometry: the volume element
        :param quad: the employed quadrature scheme
        :param blending: the same geometry map that has been passed to the form
        :param form: the integrand
        :param loop_strategy: loop pattern over the refined macro-volume - must somehow be compatible with the
                              integration domain
        :param optimizations: optimization applied to this integral
        """
        if optimizations is None:
            optimizations = set()

        if volume_geometry.dimensions != quad.geometry.dimensions:
            raise HOGException(
                "The quadrature geometry does not match the volume geometry."
            )

        self._add_integral(
            name,
            volume_geometry,
            MacroIntegrationDomain.VOLUME,
            quad,
            blending,
            form,
            loop_strategy,
            "",
            optimizations,
        )

    def add_boundary_integral(
        self,
        name: str,
        volume_geometry: ElementGeometry,
        quad: Quadrature,
        blending: GeometryMap,
        form: Form,
        optimizations: Union[None, Set[Opts]] = None,
    ) -> None:
        """
        Adds a boundary integral to the operator. Wrapper around _add_integral() for boundary integrals.

        :param name: some name for this integral (no spaces please)
        :param volume_geometry: the volume element (not the geometry of the boundary, also no embedded elements, just
                                the volume element geometry)
        :param quad: the employed quadrature scheme - this must use the embedded geometry (e.g., for boundary integrals
                     in 2D, this should use a LineElement(space_dimension=2))
        :param blending: the same map that has been passed to the form
        :param form: the integrand
        :param optimizations: optimization applied to this integral
        """

        if optimizations is None:
            optimizations = set()

        allowed_boundary_optimizations = {Opts.MOVECONSTANTS}
        if optimizations - allowed_boundary_optimizations != set():
            raise HOGException(
                f"Only allowed (aka tested and working) optimizations for boundary integrals are "
                f"{allowed_boundary_optimizations}."
            )

        if volume_geometry not in [TriangleElement(), TetrahedronElement()]:
            raise HOGException(
                "Boundary integrals only implemented for triangle and tetrahedral elements."
            )

        if volume_geometry.dimensions - 1 != quad.geometry.dimensions:
            raise HOGException(
                "The quadrature geometry does not match the boundary geometry."
            )

        # Since we will only integrate over the reference facet that lies on the x-line (2D) or xy-plane (3D) we need to
        # set the last reference coordinate to zero since it will otherwise appear as a free, uninitialized variable.
        #
        # This has to be repeated later before the quadrature is applied in case we are working with symbols.
        #
        form.integrand = form.integrand.subs(
            self.symbolizer.ref_coords_as_list(volume_geometry.dimensions)[-1], 0
        )

        for facet_id in range(volume_geometry.num_vertices):
            self._add_integral(
                name + f"_facet_id_{facet_id}",
                volume_geometry,
                MacroIntegrationDomain.DOMAIN_BOUNDARY,
                quad,
                blending,
                form,
                BOUNDARY(facet_id=facet_id),
                name + "_boundary_uid",
                optimizations,
                integrand_name=name,
            )

    def coefficients(self) -> List[FunctionSpaceImpl]:
        """Returns all coefficients sorted by name.

        During generation coefficents are detected in the element matrix and
        stored in the `coeffs` field. Being a Python dictionary, iterating over
        it yields the coefficients in an arbitrary order. Whenever generating
        code for all coefficents it is a good idea to access them in a well-defined
        order. Most importantly, the order of constructor arguments must
        be deterministic.
        """
        return sorted(self.coeffs.values(), key=lambda c: c.name)

    def generate_class_code(
        self,
        dir_path: str,
        class_files: CppClassFiles = CppClassFiles.HEADER_AND_IMPL,
        clang_format_binary: Optional[str] = None,
    ) -> None:
        """
        Invokes the code generation process, writing the full operator C++ code to file.

        :param dir_path:            directory where to write the files - the file names are built automatically
        :param class_files:         determines whether header and or impl files are generated
        :param clang_format_binary: path and/or name of binary for clang-format, defaults to None, which turns
                                    off formatting
        """

        with TimedLogger(
            f"Generating kernels for operator {self.name}", level=logging.INFO
        ):

            # Generate each kernel type (apply, gemv, ...).
            self.generate_kernels()

        # Setting up the final C++ class.
        operator_cpp_class = CppClass(
            name=self.name,
            base_classes=sorted(
                {base for kt in self.kernel_wrapper_types for base in kt.base_classes()}
            ),
        )

        # Adding form docstring to C++ class
        form_docstrings = set()
        for d, ios in self.integration_infos.items():
            for io in ios:
                form_docstrings.add(io.docstring)
        for form_docstring in form_docstrings:
            form_docstring_with_slashes = "/// ".join(form_docstring.splitlines(True))
            operator_cpp_class.add(CppComment(form_docstring_with_slashes, where="all"))

        for kernel_wrapper_type in self.kernel_wrapper_types:
            # Setting up communication.
            kernel_wrapper_type.substitute(
                {
                    "comm_fe_functions_2D": "\n".join(
                        coeff.pre_communication(2) for coeff in self.coefficients()
                    ),
                    "comm_fe_functions_3D": "\n".join(
                        coeff.pre_communication(3) for coeff in self.coefficients()
                    ),
                }
            )

            # Wrapper methods ("hand-crafted")
            for kernel_wrapper_cpp_method in kernel_wrapper_type.wrapper_methods():
                operator_cpp_class.add(kernel_wrapper_cpp_method)

            # Member variables
            for member in kernel_wrapper_type.member_variables():
                operator_cpp_class.add(member)

        with TimedLogger("Generating C code from kernel AST(s)"):
            # Add all kernels to the class.
            for operator_method in self.operator_methods:

                num_integrals = len(operator_method.integration_infos)

                if num_integrals != len(
                    operator_method.kernel_functions
                ) or num_integrals != len(operator_method.operation_counts):
                    raise HOGException(
                        "There should be as many IntegrationInfo (aka integrals) as KernelFunctions (aka kernels)."
                    )

                for (
                    integration_info,
                    sub_kernel,
                    operation_count,
                    platform_dependent_funcs,
                ) in zip(
                    operator_method.integration_infos,
                    operator_method.kernel_functions,
                    operator_method.operation_counts,
                    operator_method.platform_dependent_funcs,
                ):
                    kernel_docstring = "\n".join(
                        [
                            f"\nIntegral: {integration_info.name}",
                            f"- volume element:  {integration_info.geometry}",
                            f"- kernel type:     {operator_method.kernel_wrapper_type.name}",
                            f"- loop strategy:   {integration_info.loop_strategy}",
                            f"- quadrature rule: {integration_info.quad}",
                            f"- blending map:    {integration_info.blending}",
                            f"- operations per element:",
                            operation_count,
                        ]
                    )

                    if class_files == CppClassFiles.HEADER_IMPL_AND_VARIANTS:
                        operator_cpp_class.add(
                            CppMethodWithVariants(
                                {
                                    platform: CppMethod.from_kernel_function(
                                        plat_dep_kernel,
                                        is_const=True,
                                        visibility="private",
                                        docstring=indent(
                                            kernel_docstring,
                                            "/// ",
                                        ),
                                    )
                                    for platform, plat_dep_kernel in platform_dependent_funcs.items()
                                }
                            )
                        )
                    else:
                        operator_cpp_class.add(
                            CppMethod.from_kernel_function(
                                sub_kernel,
                                is_const=True,
                                visibility="private",
                                docstring=indent(
                                    kernel_docstring,
                                    "/// ",
                                ),
                            )
                        )

        # Free symbols that shall be settable through the ctor.
        # Those are only free symbols that have been explicitly defined by the user. Other undefined sympy symbols are
        # not (and are not supposed to be) handled here.
        #
        # We append the name of the integrand (!= name of the kernel) to the free symbols we found in the
        # integrand to make sure that two different integrands (e.g., a boundary and a volume integrand)
        # that use the same symbol name do not clash.
        #
        # However, if more than one kernel is added for the same integrand by the HOG (e.g. for boundary
        # integrals, a separate kernel per side of the simplex is added) this name will (and should) clash
        # to make sure all kernels use the same symbols.
        #
        free_symbol_vars_set = set()
        for integration_infos in self.integration_infos.values():
            for integration_info in integration_infos:
                for fs in integration_info.free_symbols:
                    free_symbol_vars_set.add(
                        f"{str(fs)}_{integration_info.integrand_name}"
                    )

        free_symbol_vars = [
            CppVariable(
                name=fs,
                type=str(self._type_descriptor.pystencils_type),
            )
            for fs in free_symbol_vars_set
        ]

        free_symbol_vars = sorted(free_symbol_vars, key=lambda x: x.name)

        free_symbol_vars_members = [
            CppVariable(name=fsv.name + "_", type=fsv.type) for fsv in free_symbol_vars
        ]

        # Let's now check whether we need ctor arguments and member variables for boundary integrals.
        boundary_condition_vars = []
        for integration_infos in self.integration_infos.values():
            if not all(
                ii.integration_domain == MacroIntegrationDomain.VOLUME
                for ii in integration_infos
            ):
                bc_var = CppVariable(name="boundaryCondition", type="BoundaryCondition")
                if bc_var not in boundary_condition_vars:
                    boundary_condition_vars.append(bc_var)

            for ii in integration_infos:
                if ii.integration_domain == MacroIntegrationDomain.DOMAIN_BOUNDARY:
                    bcuid_var = CppVariable(
                        name=ii.boundary_uid_name, type="BoundaryUID"
                    )
                    if bcuid_var not in boundary_condition_vars:
                        boundary_condition_vars.append(bcuid_var)

        boundary_condition_vars_members = [
            CppVariable(name=bcv.name + "_", type=bcv.type)
            for bcv in boundary_condition_vars
        ]

        # Finally we know what fields we need and can build the constructors, member variables, and includes.

        # Constructors ...
        operator_cpp_class.add(
            CppConstructor(
                arguments=[
                    CppVariable(
                        name="storage",
                        type="std::shared_ptr< PrimitiveStorage >",
                        is_const=True,
                        is_reference=True,
                    ),
                    CppVariable(name="minLevel", type="size_t"),
                    CppVariable(name="maxLevel", type="size_t"),
                ]
                + [
                    CppVariable(
                        name=f"_{coeff.name}",
                        type=coeff.func_type_string(),
                        is_const=True,
                        is_reference=True,
                    )
                    for coeff in self.coefficients()
                ]
                + free_symbol_vars
                + boundary_condition_vars,
                initializer_list=["Operator( storage, minLevel, maxLevel )"]
                + [f"{coeff.name}( _{coeff.name} )" for coeff in self.coefficients()]
                + [
                    f"{fsv[0].name}( {fsv[1].name} )"
                    for fsv in zip(free_symbol_vars_members, free_symbol_vars)
                ]
                + [
                    f"{bcv[0].name}( {bcv[1].name} )"
                    for bcv in zip(
                        boundary_condition_vars_members, boundary_condition_vars
                    )
                ],
            )
        )

        # Member variables ...
        for coeff in self.coefficients():
            operator_cpp_class.add(
                CppMemberVariable(
                    CppVariable(
                        name=coeff.name,
                        type=coeff.func_type_string(),
                    ),
                    visibility="private",
                )
            )

        for fsv in free_symbol_vars_members:
            operator_cpp_class.add(CppMemberVariable(fsv, visibility="private"))

        for bcv in boundary_condition_vars_members:
            operator_cpp_class.add(CppMemberVariable(bcv, visibility="private"))

        # Create path if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        output_path_header = os.path.join(dir_path, f"{self.name}.hpp")
        output_path_impl = os.path.join(dir_path, f"{self.name}.cpp")

        # Checking which files have to be included.
        func_space_includes = set().union(
            *[coeff.includes() for coeff in self.coefficients()]
        )
        kernel_includes = set().union(
            *[kt.includes() for kt in self.kernel_wrapper_types]
        )
        blending_includes = set()
        for dim, integration_infos in self.integration_infos.items():

            if not all(
                [
                    integration_infos[0].blending.coupling_includes()
                    == io.blending.coupling_includes()
                    for io in integration_infos
                ]
            ):
                raise HOGException(
                    "Seems that there are different blending functions in one bilinear form (likely in two different "
                    "integrals). This is not supported yet. :("
                )

            for inc in integration_infos[0].blending.coupling_includes():
                blending_includes.add(inc)
        all_includes = (
            self.INCLUDES | func_space_includes | kernel_includes | blending_includes
        )

        operator_cpp_file = self._build_cpp_file_pair(
            all_includes, class_files == CppClassFiles.HEADER_ONLY, operator_cpp_class
        )

        if class_files == CppClassFiles.HEADER_ONLY:
            with TimedLogger(
                f"Writing operator {self.name} to {output_path_header}", logging.INFO
            ):
                with open(output_path_header, "w") as f:
                    f.write(
                        operator_cpp_file.to_code(
                            representation=CppFileRepresentation.HEADER_ONLY
                        )
                    )
        elif class_files == CppClassFiles.HEADER_AND_IMPL:
            with TimedLogger(
                f"Writing operator {self.name} to {output_path_header}", logging.INFO
            ):
                with open(output_path_header, "w") as f:
                    f.write(
                        operator_cpp_file.to_code(
                            representation=CppFileRepresentation.HEADER_SPLIT
                        )
                    )
            with TimedLogger(
                f"Writing operator {self.name} to {output_path_impl}", logging.INFO
            ):
                with open(output_path_impl, "w") as f:
                    f.write(
                        operator_cpp_file.to_code(
                            representation=CppFileRepresentation.IMPL_SPLIT
                        )
                    )
        else:
            with TimedLogger(
                f"Writing operator {self.name} to {dir_path}", logging.INFO
            ):
                operator_cpp_file.write(dir_path, self.name)

        if clang_format_binary is not None:
            with TimedLogger(f"Applying clang-format (header)", logging.INFO):
                apply_clang_format(output_path_header, clang_format_binary)
            if class_files != CppClassFiles.HEADER_ONLY:
                with TimedLogger(
                    f"Applying clang-format (implementation)", logging.INFO
                ):
                    apply_clang_format(output_path_impl, clang_format_binary)

    def _build_cpp_file_pair(
        self,
        includes: Set[str],
        inline_functions: bool,
        operator_cpp_class: CppClass,
    ) -> CppFilePair:
        operator_cpp_file = CppFilePair()

        operator_cpp_file.add(CppComment(comment=COPYRIGHT_COMMENT))
        operator_cpp_file.add(CppComment(comment=GENERATED_COMMENT))
        operator_cpp_file.add(CppComment(comment=PRAGMA_ONCE, where="header"))
        operator_cpp_file.add(CppComment(comment=GCC_WARNING_WORKAROUND, where="impl"))

        for include in sorted(includes):
            operator_cpp_file.add(CppInclude(file_to_include=include, where="header"))

        operator_cpp_file.add(
            CppInclude(file_to_include=f"{self.name}.hpp", where="impl")
        )
        operator_cpp_file.add(
            CppInclude(file_to_include=f"../{self.name}.hpp", where="variant")
        )

        operator_cpp_file.add(
            CppComment(
                comment=f"#define FUNC_PREFIX {'inline' if inline_functions else ''} "
            )
        )

        operator_cpp_file.add(CppComment(comment=f"namespace hyteg {{"))
        operator_cpp_file.add(CppComment(comment=f"namespace operatorgeneration {{"))

        operator_cpp_file.add(operator_cpp_class)

        operator_cpp_file.add(CppComment(comment=f"}} // namespace operatorgeneration"))
        operator_cpp_file.add(CppComment(comment=f"}} // namespace hyteg"))

        return operator_cpp_file

    def _compute_micro_element_coordinates(
        self,
        integration_info: IntegrationInfo,
        element_index: List[sp.Symbol],
        geometry: ElementGeometry,
    ) -> Tuple[List[int | sp.Symbol | Field.Access], List[CustomCodeNode]]:
        """
        Computes coordinates of the micro-element. This is _not_ to be confused with the coordinates of the element's
        vertices!

        We need to distinguish two purposes of the loop counters / element indices - they are used
        (a) inside array accesses (i.e., for the computation of array indices - as integers)
        (b) for the computation of coordinates (i.e., inside expressions - after being cast to a float type)

        Although we already have the (integer) symbols of the element as an input parameter, this method returns symbols
        of the element indices that can be used for _computations_ - as opposed to array accesses. That is, we sort out
        case (b).

        In short:
        - use element_index for array accesses
        - use what is returned from this method for computations that involve the element index (or loop counters
          respectively, it is the same thing more or less)

        Why? Optimizations are not straightforward otherwise (includes purely affine transformations and especially
        vectorization whenever there are loop-counter dependent expressions).

        :param integration_info: container that specifies various properties of the kernel
        :param element_index: index of the micro-element, this is practically a list of symbols that should be equal to
                              the loop counter symbols
        :param geometry: the element's geometry
        :returns: a tuple of the element's vertex coordinates to be used for computations, and a list of CustomCodeNodes
                  that needs to be included at the beginning of the innermost loop's body
        """

        # This list is only filled if we want to vectorize.
        loop_counter_custom_code_nodes = []

        if (
            Opts.VECTORIZE not in integration_info.optimizations
            and Opts.VECTORIZE512 not in integration_info.optimizations
        ):
            # The Jacobians are loop-counter dependent, and we do not care about vectorization.
            # So we just use the indices. pystencils will handle casting them to float.
            el_matrix_element_index = element_index.copy()

        else:
            # Vectorization and loop-counter dependencies
            #
            # Easier said than done.
            #
            # At the time of writing this comment, pystencils` vectorizer has a problem with integers
            # (or more general: with vectorizing loop counter-dependent expressions that are not array accesses).
            #
            # It follows a hack to circumvent this problem that can hopefully be removed as soon as pytencils`
            # vectorizer is supporting such endeavors.
            # Instead of directly vectorizing integer expressions, we use a CustomCodeNode to first cast the loop
            # counters to the desired floating point format, and then write the cast loop counters to an array.
            # The size of that array must be at least as large as the width of the respective vector data types
            # (i.e. for AVX256 and 64bit counters it must be of length 4).
            # Since the counters are now in an array (or more specifically a pystencils.Field data structure)
            # the vectorizer just treats the (float) counters as any other data and can apply vectorization.
            #
            # For this to work, the array access (to the array that contains the cast loop counters) must contain
            # the loop counter (otherwise it may be interpreted as a constant and moved in front of the loop).
            # However, we only want the array to be of size <width-of-vector-datatype>. If we use the loop
            # counter to access the array, at first glance one would think it needs to be as large as the
            # iteration space. To avoid that, this hack introduces a "phantom" counter that is just set to the
            # same value as the loop counter, and we subtract it from the access. We define that thing inside a
            # CustomCodeNode, so it cannot be evaluated by sympy.
            #
            # Later we add the pystencils.Fields that we introduce here to the "global variables" of the
            # KernelFunction object such that they do not end up in the kernel parameters.
            #
            # Whenever the loop counters are not part of expressions at all (e.g., when the Jacobians are
            # constant on the entire macro) or when we do not vectorize, none of this is necessary.
            #
            # Phew.

            # Those are the fields that hold the loop counters.
            # We need one for each loop (i.e., one for each dimension).
            float_loop_ctr_arrays = create_generic_fields(
                [
                    s.name
                    for s in self.symbolizer.float_loop_ctr_array(geometry.dimensions)
                ],
                self._type_descriptor.pystencils_type,
            )

            # This is just some symbol that we set to the innermost loop ctr.
            # However, we make sure sympy never sees this.
            phantom_ctr = TypedSymbol("phantom_ctr_0", int)

            # Those are the actual counters (i.e. the coordinates of the micro-element) for use in expressions that
            # are _not_ array accesses. We assign them to the first entry of the float loop counter array. The
            # vectorizer can automatically assign multiple variables here.
            # Note that the first, i.e., innermost loop counter is chosen for the array access for all dimensions!
            el_matrix_element_index = [
                float_loop_ctr_arrays[d].absolute_access(
                    (element_index[0] - phantom_ctr,), (0,)
                )
                for d in range(geometry.dimensions)
            ]

            # Let's fill the array.
            float_ctr_array_size = (
                8 if Opts.VECTORIZE512 in integration_info.optimizations else 4
            )

            custom_code = ""
            custom_code += f"const int64_t phantom_ctr_0 = ctr_0;\n"
            for d in range(geometry.dimensions):
                array_name = (
                    "_data_"
                    + self.symbolizer.float_loop_ctr_array(geometry.dimensions)[d].name
                )
                custom_code += f"{str(self._type_descriptor.pystencils_type)} {array_name}[{float_ctr_array_size}];\n"
                for i in range(float_ctr_array_size):
                    custom_code += f"{array_name}[{i}] = ({str(self._type_descriptor.pystencils_type)}) ctr_{d}"
                    if d == 0:
                        # We only vectorize the innermost loop.
                        # Only that counter is increased. The others are constant.
                        custom_code += f"+ {i}"
                    custom_code += ";\n"

            # We need the fields' symbols to signal that they have been defined inside the CustomCodeNode and do not
            # end up in the kernel parameters.
            float_loop_ctr_array_symbols = [
                FieldPointerSymbol(f.name, f.dtype, False)
                for f in float_loop_ctr_arrays
            ]

            # Creating our custom node.
            loop_counter_custom_code_nodes = [
                CustomCodeNode(
                    custom_code,
                    # The (integer) loop counters. This makes sure the custom code node has to stay inside the loop.
                    # Without this, it would not depend on the counter.
                    element_index,
                    # We use the symbols_defined parameter to signal pystencils that the phantom_ctr and the loop
                    # counter arrays are being defined inside the custom code node and do not appear in the kernel
                    # parameters.
                    [phantom_ctr] + float_loop_ctr_array_symbols,
                )
            ]

        return el_matrix_element_index, loop_counter_custom_code_nodes

    def _generate_kernel(
        self,
        dim: int,
        integration_info: IntegrationInfo,
        kernel_type: KernelType,
        src_fields: List[FunctionSpaceImpl],
        dst_fields: List[FunctionSpaceImpl],
    ) -> Tuple[ps.astnodes.Block, str]:
        """
        This method generates an AST that represents the passed kernel type.

        It does not return the corresponding C++ code. That is generated in a second step.

        :param integration_info: IntegrationInfo object holding the symbolic element matrix, quadrature rule,
                                 element geometry, etc.
        :param kernel_type:      specifies the kernel to execute - this could be e.g., a matrix-vector
                                 multiplication
        :returns: tuple (pre_loop_stmts, loop, operations_table)
        """

        geometry = integration_info.geometry
        mat = integration_info.mat
        quadrature = integration_info.quad

        rows, cols = mat.shape

        optimizer = Optimizer(integration_info.optimizations)
        optimizer.check_opts_validity()

        kernel_config = CreateKernelConfig(
            default_number_float=self._type_descriptor.pystencils_type,
            data_type=self._type_descriptor.pystencils_type,
        )

        # Fully symbolic kernel operation (matrix vector multiplication or
        # gemv):
        #
        #  - matvec: z <- mat * x,
        #  - gemv:   z <- alpha * mat * x + beta * y,
        #
        # where mat is the local element matrix that we get from the HOG form,
        # and x, y, z are the local element vectors. The elements of the vectors
        # are symbols of DoFs on a micro element. Adequate array accesses will
        # be inserted below.

        # Create symbols for DoFs.
        src_vecs_symbols = [
            self.symbolizer.dof_symbols_as_vector(
                src_field.fe_space.num_dofs(geometry), src_field.name
            )
            for src_field in src_fields
        ]

        dst_vecs_symbols = [
            self.symbolizer.dof_symbols_as_vector(
                dst_field.fe_space.num_dofs(geometry), dst_field.name
            )
            for dst_field in dst_fields
        ]

        # Do the kernel operation.
        kernel_op_assignments = kernel_type.kernel_operation(
            src_vecs_symbols, dst_vecs_symbols, mat, rows
        )

        # Common subexpression elimination.
        with TimedLogger("cse on kernel operation", logging.DEBUG):
            cse_impl = optimizer.cse_impl()
            kernel_op_assignments = hog.cse.cse(
                kernel_op_assignments,
                cse_impl,
                "tmp_kernel_op",
                return_type=SympyAssignment,
            )

        if integration_info.quad_loop:
            accessed_mat_entries = mat.atoms(TypedSymbol)
            accessed_mat_entries &= set().union(
                *[ass.undefined_symbols for ass in kernel_op_assignments]
            )

            with TimedLogger("constructing quadrature loops"):
                quad_loop = integration_info.quad_loop.construct_quad_loop(
                    accessed_mat_entries, optimizer.cse_impl()
                )
        else:
            quad_loop = []

        # Some required input parameters
        indexing_info = IndexingInfo()

        # Vertex coordinates of the micro element.
        # Only used for non-affine blending maps.
        element_vertex_coordinates_symbols = self.symbolizer.affine_vertices_as_vectors(
            geometry.dimensions, geometry.num_vertices
        )
        # Vertex coordinates of the micro element with index 0.
        # These are used to determine the affine Jacobi matrix.
        coord_symbols_for_jac_affine = self.symbolizer.affine_vertices_as_vectors(
            geometry.dimensions,
            geometry.num_vertices,
            first_vertex=0,
            symbol_suffix="_const",
        )

        jacobi_assignments = hog.code_generation.jacobi_matrix_assignments(
            mat,
            integration_info.tables + quad_loop,
            geometry,
            self.symbolizer,
            affine_points=coord_symbols_for_jac_affine,
        )

        macro_vertex_coordinates = [
            sp.Matrix(
                [
                    [sp.Symbol(self._var_names_macro_vertex_coords(i, c))]
                    for c in range(geometry.dimensions)
                ]
            )
            for i in range(geometry.num_vertices)
        ]

        # create loop according to loop strategy
        element_index = [
            LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(dim)
        ]
        loop = integration_info.loop_strategy.create_loop(
            dim, element_index, indexing_info.micro_edges_per_macro_edge
        )

        # element coordinates, jacobi matrix, tabulations, etc.
        preloop_stmts = {}

        # Deciding on which element types we want to iterate over.
        # We skip certain element types for macro-volume boundary integrals.
        element_types: List[Union[FaceType, CellType]] = all_element_types(
            geometry.dimensions
        )
        if isinstance(integration_info.loop_strategy, BOUNDARY):
            element_types = list(integration_info.loop_strategy.element_loops.keys())

        for element_type in element_types:

            # Re-ordering micro-element vertices for the handling of domain boundary integrals.
            #
            # Boundary integrals are handled by looping over all (volume-)elements that have a facet at one of the
            # macro-element boundaries. There are three such boundaries in 2D and four in 3D. For each case, a separate
            # kernel has to be generated. The logic that decides which of these kernels are called on which
            # macro-volumes is handled by the HyTeG operator that is generated around these kernels.
            #
            # Instead of integrating over the micro-volume, we need to integrate over one of the micro-element facets.
            # For that, the micro-element vertices are re-ordered such that the transformation from the affine element
            # to the reference element results in the integration (boundary-)domain being mapped to the reference
            # domain.
            #
            # E.g., in 2D, if the integration over the xy-boundary of the macro-volume (== macro-face) shall be
            # generated this is signalled here by loop_strategy.facet_id == 2.
            # In 2D all boundaries are only touched by the GRAY micro-elements (this is a little more complicated in 3D
            # where at each macro-volume boundary, two types of elements overlap).
            # Thus, we
            #   a) Shuffle the affine element vertices such that the xy-boundary of the GRAY elements is mapped to the
            #      (0, 1) line, which is the reference line for integration.
            #   b) Only iterate over the GRAY elements (handled later below).

            # Default order.
            element_vertex_order = list(range(geometry.num_vertices))

            # Shuffling vertices if a boundary integral is requested.
            if (
                integration_info.integration_domain
                == MacroIntegrationDomain.DOMAIN_BOUNDARY
                and isinstance(integration_info.loop_strategy, BOUNDARY)
            ):
                element_vertex_order = micro_vertex_permutation_for_facet(
                    volume_geometry=geometry,
                    element_type=element_type,
                    facet_id=integration_info.loop_strategy.facet_id,
                )

            # Create array accesses to the source and destination vector(s) for
            # the kernel.
            src_vecs_accesses = [
                src_field.local_dofs(
                    geometry,
                    element_index,  # type: ignore[arg-type] # list of sympy expressions also works
                    element_type,
                    indexing_info,
                    element_vertex_order,
                )
                for src_field in src_fields
            ]
            dst_vecs_accesses = [
                dst_field.local_dofs(
                    geometry,
                    element_index,  # type: ignore[arg-type] # list of sympy expressions also works
                    element_type,
                    indexing_info,
                    element_vertex_order,
                )
                for dst_field in dst_fields
            ]

            # Load source DoFs.
            load_vecs = [
                SympyAssignment(s, a)
                for symbols, accesses in zip(src_vecs_symbols, src_vecs_accesses)
                for s, a in zip(symbols, accesses)
            ]

            kernel_op_post_assignments = kernel_type.kernel_post_operation(
                geometry,
                element_index,  # type: ignore[arg-type] # list of sympy expressions also works
                element_type,
                src_vecs_accesses,
                dst_vecs_accesses,
                element_vertex_order,
            )

            # Load DoFs of coefficients. Those appear whenever a form is
            # generated that uses DoFs from some external function (think a
            # coefficient that "lives" on a FEM function).
            dof_symbols_set: Set[DoFSymbol] = {
                a
                for ass in kernel_op_assignments + quad_loop
                for a in ass.atoms(DoFSymbol)
            }

            dof_symbols = sorted(dof_symbols_set, key=lambda ds: ds.name)
            coeffs = dict(
                (
                    dof_symbol.function_id,
                    FunctionSpaceImpl.create_impl(
                        dof_symbol.function_space,
                        dof_symbol.function_id,
                        self._type_descriptor,
                    ),
                )
                for dof_symbol in dof_symbols
            )
            self.coeffs |= coeffs
            for dof_symbol in dof_symbols:
                # Assign to the DoF symbol the corresponding array access.
                load_vecs.append(
                    SympyAssignment(
                        sp.Symbol(dof_symbol.name),
                        coeffs[dof_symbol.function_id].local_dofs(
                            geometry,
                            element_index,
                            element_type,
                            indexing_info,
                            element_vertex_order,
                        )[dof_symbol.dof_id],
                    )
                )

            # Compute coordinates of the micro-element that can safely be used for all optimizations.
            #
            # Those should not be used for the computation of the Jacobians from the reference to the affine space since
            # we can also exploit translation invariance there. Here, we cannot exploit that since the following symbols
            # are meant for other computations that have to be executed at quadrature points.
            #
            # See docstring of the called method for details.
            (
                el_matrix_element_index,
                loop_counter_custom_code_nodes,
            ) = self._compute_micro_element_coordinates(
                integration_info, element_index, geometry
            )

            el_vertex_coordinates = element_vertex_coordinates(
                geometry,
                el_matrix_element_index,  # type: ignore[arg-type] # list of sympy expressions also works
                element_type,
                indexing_info.micro_edges_per_macro_edge_float,
                macro_vertex_coordinates,
                self.symbolizer,
            )

            # Re-ordering the element vertex coordinates
            # (for all computations but affine transformation Jacobians - those are re-ordered later).
            # See comment on boundary integrals above.
            el_vertex_coordinates = [
                el_vertex_coordinates[i] for i in element_vertex_order
            ]

            coords_assignments = [
                SympyAssignment(
                    element_vertex_coordinates_symbols[vertex][component],
                    el_vertex_coordinates[vertex][component],
                )
                for vertex in range(geometry.num_vertices)
                for component in range(geometry.dimensions)
            ]

            if integration_info.blending.is_affine() or optimizer[Opts.QUADLOOPS]:
                blending_assignments = []
            else:
                blending_assignments = (
                    hog.code_generation.blending_jacobi_matrix_assignments(
                        mat,
                        integration_info.tables + quad_loop,
                        geometry,
                        self.symbolizer,
                        affine_points=element_vertex_coordinates_symbols,
                        blending=integration_info.blending,
                        quad_info=integration_info.quad,
                    )
                )

                blending_assignments += hog.code_generation.hessian_matrix_assignments(
                    mat,
                    integration_info.tables + quad_loop,
                    geometry,
                    self.symbolizer,
                    affine_points=element_vertex_coordinates_symbols,
                    blending=integration_info.blending,
                    quad_info=integration_info.quad,
                )

                with TimedLogger("cse on blending operation", logging.DEBUG):
                    cse_impl = optimizer.cse_impl()
                    blending_assignments = hog.cse.cse(
                        blending_assignments,
                        cse_impl,
                        "tmp_blending_op",
                        return_type=SympyAssignment,
                    )

            body = (
                loop_counter_custom_code_nodes
                + coords_assignments
                + blending_assignments
                + load_vecs
                + quad_loop
                + kernel_op_assignments
                + kernel_op_post_assignments
            )

            if (
                integration_info.integration_domain
                == MacroIntegrationDomain.DOMAIN_BOUNDARY
                and isinstance(integration_info.loop_strategy, BOUNDARY)
            ):
                with TimedLogger(
                    "boundary integrals: setting unused reference coordinate to 0",
                    logging.DEBUG,
                ):
                    for node in body:
                        node.subs(
                            {
                                self.symbolizer.ref_coords_as_list(geometry.dimensions)[
                                    -1
                                ]: 0
                            }
                        )

            if not optimizer[Opts.QUADLOOPS]:
                # Only now we replace the quadrature points and weights - if there are any.
                # We also setup sympy assignments in body
                with TimedLogger(
                    "replacing quadrature points and weigths", logging.DEBUG
                ):
                    if not quadrature.is_exact() and not quadrature.inline_values:
                        subs_dict = dict(quadrature.points() + quadrature.weights())
                        for node in body:
                            node.subs(subs_dict)

            # count operations
            ops = Operations()
            for stmt in body:
                if isinstance(stmt, ForLoop):
                    for stmt2 in stmt.body.args:
                        count_operations(
                            stmt2.rhs, ops, loop_factor=stmt.stop - stmt.start
                        )
                elif isinstance(stmt, SympyAssignment) or isinstance(
                    stmt, hog.ast.Assignment
                ):
                    count_operations(stmt.rhs, ops)
                elif isinstance(
                    stmt, (ps.astnodes.EmptyLine, ps.astnodes.SourceCodeComment)
                ):
                    pass
                else:
                    ops.unknown_ops += 1

            body = add_types(body, kernel_config)

            # add the created loop body of statements to the loop according to the loop strategy
            integration_info.loop_strategy.add_body_to_loop(loop, body, element_type)

            # This actually replaces the abstract field access instances with
            # array accesses that contain the index computations.
            resolve_field_accesses(loop)

            with TimedLogger(
                f"cse on jacobi matrix computation ({element_type})", logging.DEBUG
            ):
                # The affine Jacobi matrix is always computed from the micro
                # element at index 0. This makes sure that tables can reference
                # the affine Jacobian even in the blending case.
                el_vertex_coordinates = element_vertex_coordinates(
                    geometry,
                    tuple(0 for _ in element_index),  # type: ignore[arg-type] # length of tuple is correct
                    element_type,
                    indexing_info.micro_edges_per_macro_edge_float,
                    macro_vertex_coordinates,
                    self.symbolizer,
                )

                # Re-ordering the element vertex coordinates for the Jacobians.
                # See comment on boundary integrals above.
                el_vertex_coordinates = [
                    el_vertex_coordinates[i] for i in element_vertex_order
                ]

                coords_assignments = [
                    SympyAssignment(
                        coord_symbols_for_jac_affine[vertex][component],
                        el_vertex_coordinates[vertex][component],
                    )
                    for vertex in range(geometry.num_vertices)
                    for component in range(geometry.dimensions)
                ]

                elem_dependent_stmts = (
                    hog.cse.cse(
                        coords_assignments + jacobi_assignments,
                        optimizer.cse_impl(),
                        "tmp_coords_jac",
                        return_type=SympyAssignment,
                    )
                    + integration_info.tables  # array declarations
                )

                preloop_stmts[element_type] = add_types(
                    elem_dependent_stmts, kernel_config
                )

        with TimedLogger(
            "renaming loop bodies and preloop stmts for element types", logging.DEBUG
        ):
            for element_type, preloop in preloop_stmts.items():
                loop = integration_info.loop_strategy.add_preloop_for_loop(
                    loop, preloop, element_type
                )

        # Add quadrature points and weights array declarations, but only those
        # which are actually needed.
        if integration_info.quad_loop:
            q_decls = integration_info.quad_loop.point_weight_decls()
            undefined = Block(loop).undefined_symbols
            body = [q_decl for q_decl in q_decls if q_decl.lhs in undefined] + loop
        else:
            body = loop

        return (Block(body), ops.to_table())

    def generate_kernels(self) -> None:
        """
        TODO: Split this up in a meaningful way.
              Currently does a lot of stuff and modifies the kernel_types.
        TODO: I think this is the only place where the kernel type template
              should be filled. Currently, the template contains coefficient
              communication placeholders *for all dimensions*. This means that
              all dimensions must be substituted (outside of this function).
              IMO the template should be fixed to only include the requested
              dimensions as is the case for all other placeholders. Once this
              is sorted out, the substitution can be moved to its own function,
              making this one more clear.

        For each kernel, there is at least one kernel wrapper.
        To generate the wrapper, we need some information about the underlying kernel.
        This information is gathered here, and written to the kernel type object,
        """

        for kernel_wrapper_type in self.kernel_wrapper_types:
            for dim, integration_infos in self.integration_infos.items():

                kernel_functions = []
                kernel_op_counts = []
                platform_dep_kernels = []

                macro_type = {2: "face", 3: "cell"}

                geometry = integration_infos[0].geometry
                if not all([geometry == io.geometry for io in integration_infos]):
                    raise HOGException(
                        "All element geometries should be the same. Regardless of whether you integrate over their "
                        "boundary or volume. Dev note: this information seems to be redundant and we should only have "
                        "it in one place."
                    )

                for integration_info in integration_infos:

                    # generate AST of kernel loop
                    with TimedLogger(
                        f"Generating kernel {integration_info.name} ({kernel_wrapper_type.name}, {dim}D)",
                        logging.INFO,
                    ):

                        (
                            function_body,
                            kernel_op_count,
                        ) = self._generate_kernel(
                            dim,
                            integration_info,
                            kernel_wrapper_type.kernel_type,
                            kernel_wrapper_type.src_fields,
                            kernel_wrapper_type.dst_fields,
                        )

                        kernel_function = KernelFunction(
                            function_body,
                            Target.CPU,
                            Backend.C,
                            make_python_function,
                            ghost_layers=None,
                            function_name=f"{kernel_wrapper_type.name}_{integration_info.name}_macro_{geometry.dimensions}D",
                            assignments=None,
                        )

                        kernel_functions.append(kernel_function)
                        kernel_op_counts.append(kernel_op_count)

                    # optimizer applies optimizations
                    with TimedLogger(
                        f"Optimizing kernel: {kernel_function.function_name} in {dim}D",
                        logging.INFO,
                    ):
                        optimizer = Optimizer(integration_info.optimizations)
                        optimizer.check_opts_validity()

                        if isinstance(kernel_wrapper_type.kernel_type, Assemble):
                            optimizer = optimizer.copy_without_vectorization()

                        platform_dep_kernels.append(
                            optimizer.apply_to_kernel(
                                kernel_function, dim, integration_info.loop_strategy
                            )
                        )

                # Setup kernel wrapper string and op count table
                #
                # In the following, we insert the sub strings of the final kernel string:
                # coefficients (retrieving pointers), setup of scalar parameters, kernel function call
                # This is done as follows:
                # - the kernel type has the skeleton string in which the sub string must be substituted
                # - the function space impl knows from which function type a src/dst/coefficient field is and can return
                #   the corresponding sub string

                # Retrieve coefficient pointers
                kernel_wrapper_type.substitute(
                    {
                        f"pointer_retrieval_{dim}D": "\n".join(
                            coeff.pointer_retrieval(dim)
                            for coeff in self.coefficients()
                        )
                    }
                )

                # Setting up required scalar parameters.
                scalar_parameter_setup = (
                    f"const auto {self.VAR_NAME_MICRO_EDGES_PER_MACRO_EDGE} = (int64_t) levelinfo::num_microedges_per_edge( level );\n"
                    f"const auto {self.VAR_NAME_MICRO_EDGES_PER_MACRO_EDGE_FLOAT} = ({self._type_descriptor.pystencils_type}) levelinfo::num_microedges_per_edge( level );\n"
                )
                scalar_parameter_setup += "\n".join(
                    [
                        f"const {self._type_descriptor.pystencils_type} {self._var_names_macro_vertex_coords(i, c)}"
                        f" ={' (' + str(self._type_descriptor.pystencils_type) + ') ' if not isinstance(self._type_descriptor.pystencils_type.numpy_dtype, np.float64) else ''}"
                        f" {macro_type[dim]}.getCoordinates()[{i}][{c}];"
                        for i in range(geometry.num_vertices)
                        for c in range(geometry.dimensions)
                    ]
                )

                blending_parameter_coupling_code = integration_infos[
                    0
                ].blending.parameter_coupling_code()
                if not all(
                    [
                        blending_parameter_coupling_code
                        == io.blending.parameter_coupling_code()
                        for io in integration_infos
                    ]
                ):
                    raise HOGException(
                        "It seems you specified different blending maps for one bilinear form. "
                        "This may be desired, but it is certainly not supported :)."
                    )

                scalar_parameter_setup += blending_parameter_coupling_code

                kernel_wrapper_type.substitute(
                    {f"scalar_parameter_setup_{dim}D": scalar_parameter_setup}
                )

                # Kernel function call(s).
                kernel_function_call_strings = []
                for kernel_function, integration_info in zip(
                    kernel_functions, integration_infos
                ):

                    pre_call_code = ""
                    post_call_code = ""

                    if (
                        integration_info.integration_domain
                        == MacroIntegrationDomain.DOMAIN_BOUNDARY
                    ):

                        if not isinstance(integration_info.loop_strategy, BOUNDARY):
                            raise HOGException(
                                "The loop strategy should be BOUNDARY for boundary integrals."
                            )

                        facet_type = "Edge" if dim == 2 else "Face"

                        neighbor_facet = (
                            f"getStorage()->get{facet_type}( "
                            f"{macro_type[dim]}.getLowerDimNeighbors()"
                            f"[{integration_info.loop_strategy.facet_id}] )"
                        )

                        pre_call_code = (
                            f"if ( boundaryCondition_.getBoundaryUIDFromMeshFlag( "
                            f"{neighbor_facet}->getMeshBoundaryFlag() ) == {integration_info.boundary_uid_name}_ ) {{"
                        )
                        post_call_code = "}"

                    kernel_parameters = kernel_function.get_parameters()

                    # We append the name of the integrand (!= name of the kernel) to the free symbols we found in the
                    # integrand to make sure that two different integrands (e.g., a boundary and a volume integrand)
                    # that use the same symbol name do not clash.
                    #
                    # However, if more than one kernel is added for the same integrand by the HOG (e.g. for boundary
                    # integrals, a separate kernel per side of the simplex is added) this name will (and should) clash
                    # to make sure all kernels use the same symbols.

                    kernel_parameters_updated = []
                    for prm in kernel_parameters:
                        if str(prm) in [
                            str(fs) for fs in integration_info.free_symbols
                        ]:
                            kernel_parameters_updated.append(
                                f"{str(prm)}_{integration_info.integrand_name}_"
                            )
                        else:
                            kernel_parameters_updated.append(str(prm))

                    kernel_function_call_parameter_string = ",\n".join(
                        kernel_parameters_updated
                    )

                    kernel_function_call_strings.append(
                        f"""
                                    {pre_call_code}\n
                                    {kernel_function.function_name}(\n
                                    {kernel_function_call_parameter_string});\n
                                    {post_call_code}\n"""
                    )

                kernel_wrapper_type.substitute(
                    {
                        f"kernel_function_call_{dim}D": "\n".join(
                            kernel_function_call_strings
                        )
                    }
                )

                # Collect all information
                kernel = OperatorMethod(
                    kernel_wrapper_type,
                    kernel_functions,
                    platform_dep_kernels,
                    kernel_op_counts,
                    integration_infos,
                )
                self.operator_methods.append(kernel)
