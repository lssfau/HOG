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
from enum import auto, Enum
import logging
from typing import Dict, List, Optional, Set, Tuple
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

from hog.forms import Form
from hog.ast import Operations, count_operations
from hog.blending import GeometryMap
import hog.code_generation
import hog.cse
from hog.dof_symbol import DoFSymbol
from hog.element_geometry import ElementGeometry
from hog.exception import HOGException
from hog.operator_generation.indexing import (
    all_element_types,
    element_vertex_coordinates,
    IndexingInfo,
)
from hog.operator_generation.kernel_types import Assemble, KernelType
from hog.operator_generation.loop_strategies import LoopStrategy, SAWTOOTH
from hog.operator_generation.optimizer import Optimizer, Opts
from hog.quadrature import QuadLoop, Quadrature
from hog.symbolizer import Symbolizer
from hog.operator_generation.types import HOGType


class MacroIntegrationDomain(Enum):
    """Enum type to specify where to integrate."""

    # Integration over the volume element
    VOLUME = "Volume"


@dataclass
class IntegrationInfo:
    geometry: ElementGeometry  # geometry of the element, e.g. tetrahedron
    integration_domain: (
        MacroIntegrationDomain  # entity of geometry to integrate over, e.g. facet
    )
    quad: Quadrature  # quadrature over integration domain of geometry, e.g. triangle
    blending: GeometryMap

    tables: List[ArrayDeclaration]
    quad_loop: Optional[QuadLoop]
    mat: sp.MatrixBase

    docstring: str = ""

    def _str_(self):
        return f"Integration Info: {self.geometry}, {self.integration_domain}, mat shape {self.mat.shape}, quad degree {self.quad.degree}, blending {self.blending}"

    def _repr_(self):
        return str(self)


@dataclass
class Kernel:
    kernel_type: KernelType
    kernel_function: KernelFunction
    platform_dependent_funcs: Dict[str, KernelFunction]
    operation_count: str
    integration_info: IntegrationInfo


class CppClassFiles(Enum):
    """How to split a Cpp class to a set of files."""

    HEADER_ONLY = auto()
    HEADER_AND_IMPL = auto()
    HEADER_IMPL_AND_VARIANTS = auto()


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
        kernel_types: List[KernelType],
        opts: Set[Opts],
        type_descriptor: HOGType,
    ):
        self.name = name
        self.symbolizer = symbolizer
        self.kernel_types = kernel_types  # type of kernel: e.g. GEMV

        # Holds one element matrix and quad scheme for each ElementGeometry.
        self.element_matrices: Dict[int, IntegrationInfo] = {}

        self._optimizer = Optimizer(opts)
        self._optimizer_no_vec = Optimizer(opts - {Opts.VECTORIZE, Opts.VECTORIZE512})

        # set the precision in which the operations are to be performed
        self._type_descriptor = type_descriptor

        # coefficients
        self.coeffs: Dict[str, FunctionSpaceImpl] = {}
        # implementations for each kernel, generated at a later stage
        self.kernels: List[Kernel] = []

    def set_element_matrix(
        self,
        dim: int,
        geometry: ElementGeometry,
        integration_domain: MacroIntegrationDomain,
        quad: Quadrature,
        blending: GeometryMap,
        form: Form,
    ) -> None:
        """
        Use this method to add element matrices to the operator.

        :param dim: the dimensionality of the domain, i.e. the volume - it may be that this does not match the geometry
                    of the element, e.g., when facet integrals are required, note that only one routine per dim is
                    created
        :param geometry: geometry that shall be integrated over
        :param integration_domain: where to integrate - see MacroIntegrationDomain
        :param mat: the local element matrix
        :param quad: the employed quadrature scheme - should match what has been used to integrate the weak form
        :param blending: the same geometry map that has been passed to the form
        """

        if dim not in [2, 3]:
            raise HOGException("Only supporting 2D and 3D. Dim should be in [2, 3]")

        if integration_domain != MacroIntegrationDomain.VOLUME:
            raise HOGException("Only volume integrals supported as of now.")
        if dim != geometry.dimensions:
            raise HOGException("Only volume integrals supported as of now.")

        if dim in self.element_matrices:
            raise HOGException(
                "You are trying to overwrite an already specified integration routine by calling this method twice with "
                "the same dim argument."
            )

        tables = []
        quad_loop = None
        mat = form.integrand

        if self._optimizer[Opts.TABULATE]:
            mat = form.tabulation.resolve_table_accesses(mat, self._type_descriptor)
            with TimedLogger(f"constructing tables", level=logging.DEBUG):
                tables = form.tabulation.construct_tables(quad, self._type_descriptor)
        else:
            mat = form.tabulation.inline_tables(mat)

        if self._optimizer[Opts.QUADLOOPS]:
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

        self.element_matrices[dim] = IntegrationInfo(
            geometry=geometry,
            integration_domain=integration_domain,
            quad=quad,
            blending=blending,
            tables=tables,
            quad_loop=quad_loop,
            mat=mat,
            docstring=form.docstring,
        )

    def coefficients(self) -> List[FunctionSpaceImpl]:
        """Returns all coefficients sorted by name.

        During generation coefficents are detected in the element matrix and
        stored in the `coeffs` field. Being a Python dictionary, iterating over
        it yields the coefficients in an arbitrary order. Whenever generating
        code for all coefficents it is a good idea to access them in a well
        defined order. Most importantly, the order of constructor arguments must
        be deterministic.
        """
        return sorted(self.coeffs.values(), key=lambda c: c.name)

    def generate_class_code(
        self,
        dir_path: str,
        loop_strategy: LoopStrategy = SAWTOOTH(),
        class_files: CppClassFiles = CppClassFiles.HEADER_AND_IMPL,
        clang_format_binary: Optional[str] = None,
    ) -> None:
        """
        Invokes the code generation process, writing the full operator C++ code to file.

        :param dir_path:      directory where to write the files - the file names are built automatically
        :param loop_strategy: iteration pattern
        :param header_only:   if True, the entire class (incl. implementation) is written into a single file
        :clang_format_binary: path and/or name of binary for clang-format, defaults to None, which turns
                              off formatting
        """

        # Asking the optimizer if optimizations are valid.
        self._optimizer.check_opts_validity(loop_strategy)
        # Generate each kernel type (apply, gemv, ...).
        self.generate_kernels(loop_strategy)

        # Setting up the final C++ class.
        operator_cpp_class = CppClass(
            name=self.name,
            base_classes=sorted(
                {base for kt in self.kernel_types for base in kt.base_classes()}
            ),
        )

        # Adding form docstring to C++ class
        form_docstrings = set()
        for d, io in self.element_matrices.items():
            form_docstrings.add(io.docstring)
        for form_docstring in form_docstrings:
            form_docstring_with_slashes = "/// ".join(form_docstring.splitlines(True))
            operator_cpp_class.add(CppComment(form_docstring_with_slashes, where="all"))

        for kernel_type in self.kernel_types:
            # Setting up communication.
            kernel_type.substitute(
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
            for kernel_wrapper_cpp_method in kernel_type.wrapper_methods():
                operator_cpp_class.add(kernel_wrapper_cpp_method)

            # Member variables
            for member in kernel_type.member_variables():
                operator_cpp_class.add(member)

        # Add all kernels to the class.
        for kernel in self.kernels:
            kernel_op_count = "\n".join(
                [
                    f"Kernel type: {kernel.kernel_type.name}",
                    f"- quadrature rule: {kernel.integration_info.quad}",
                    f"- operations per element:",
                    kernel.operation_count,
                ]
            )

            if class_files == CppClassFiles.HEADER_IMPL_AND_VARIANTS:
                operator_cpp_class.add(
                    CppMethodWithVariants(
                        {
                            platform: CppMethod.from_kernel_function(
                                kernel,
                                is_const=True,
                                visibility="private",
                                docstring=indent(kernel_op_count, "/// "),
                            )
                            for platform, kernel in kernel.platform_dependent_funcs.items()
                        }
                    )
                )
            else:
                operator_cpp_class.add(
                    CppMethod.from_kernel_function(
                        kernel.kernel_function,
                        is_const=True,
                        visibility="private",
                        docstring=indent(kernel_op_count, "/// "),
                    )
                )

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
                ],
                initializer_list=["Operator( storage, minLevel, maxLevel )"]
                + [f"{coeff.name}( _{coeff.name} )" for coeff in self.coefficients()],
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

        os.makedirs(dir_path, exist_ok=True)  # Create path if it doesn't exist

        output_path_header = os.path.join(dir_path, f"{self.name}.hpp")
        output_path_impl = os.path.join(dir_path, f"{self.name}.cpp")

        # Checking which files have to be included.
        func_space_includes = set().union(
            *[coeff.includes() for coeff in self.coefficients()]
        )
        kernel_includes = set().union(*[kt.includes() for kt in self.kernel_types])
        blending_includes = set()
        for dim, integration_info in self.element_matrices.items():
            for inc in integration_info.blending.coupling_includes():
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

        if not (self._optimizer[Opts.VECTORIZE] or self._optimizer[Opts.VECTORIZE512]):
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
            float_ctr_array_size = 8 if self._optimizer[Opts.VECTORIZE512] else 4

            custom_code = ""
            custom_code += f"const int64_t phantom_ctr_0 = ctr_0;\n"
            for d in range(geometry.dimensions):
                array_name = (
                    "_data_"
                    + self.symbolizer.float_loop_ctr_array(geometry.dimensions)[d].name
                )
                custom_code += f"real_t {array_name}[{float_ctr_array_size}];\n"
                for i in range(float_ctr_array_size):
                    custom_code += f"{array_name}[{i}] = (real_t) ctr_{d}"
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
        loop_strategy: LoopStrategy,
        kernel_type: KernelType,
    ) -> Tuple[ps.astnodes.Block, str]:
        """
        This method generates an AST that represents the passed kernel type.

        It does not return the corresponding C++ code. That is generated in a second step.

        :param integration_info: IntegrationInfo object holding the symbolic element matrix, quadrature rule,
                                 element geometry, etc.
        :param loop_strategy:    defines the iteration pattern
        :param kernel_type:      specifies the kernel to execute - this could be e.g., a matrix-vector
                                 multiplication
        :returns: tuple (pre_loop_stmts, loop, operations_table)
        """

        geometry = integration_info.geometry
        mat = integration_info.mat
        quadrature = integration_info.quad

        rows, cols = mat.shape

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
            for src_field in kernel_type.src_fields
        ]

        dst_vecs_symbols = [
            self.symbolizer.dof_symbols_as_vector(
                dst_field.fe_space.num_dofs(geometry), dst_field.name
            )
            for dst_field in kernel_type.dst_fields
        ]

        # Do the kernel operation.
        kernel_op_assignments = kernel_type.kernel_operation(
            src_vecs_symbols, dst_vecs_symbols, mat, rows
        )

        # Common subexpression elimination.
        with TimedLogger("cse on kernel operation", logging.DEBUG):
            cse_impl = self._optimizer.cse_impl()
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
                    accessed_mat_entries, self._optimizer.cse_impl()
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
        loop = loop_strategy.create_loop(
            dim, element_index, indexing_info.micro_edges_per_macro_edge
        )

        # element coordinates, jacobi matrix, tabulations, etc.
        preloop_stmts = {}

        for element_type in all_element_types(geometry.dimensions):
            # Create array accesses to the source and destination vector(s) for
            # the kernel.
            src_vecs_accesses = [
                src_field.local_dofs(
                    geometry,
                    element_index,  # type: ignore[arg-type] # list of sympy expressions also works
                    element_type,
                    indexing_info,
                )
                for src_field in kernel_type.src_fields
            ]
            dst_vecs_accesses = [
                dst_field.local_dofs(
                    geometry,
                    element_index,  # type: ignore[arg-type] # list of sympy expressions also works
                    element_type,
                    indexing_info,
                )
                for dst_field in kernel_type.dst_fields
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
                            geometry, element_index, element_type, indexing_info
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

            coords_assignments = [
                SympyAssignment(
                    element_vertex_coordinates_symbols[vertex][component],
                    el_vertex_coordinates[vertex][component],
                )
                for vertex in range(geometry.num_vertices)
                for component in range(geometry.dimensions)
            ]

            if integration_info.blending.is_affine() or self._optimizer[Opts.QUADLOOPS]:
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

                if integration_info.blending.hessian_used:
                    blending_assignments += (
                        hog.code_generation.hessian_matrix_assignments(
                            mat,
                            integration_info.tables + quad_loop,
                            geometry,
                            self.symbolizer,
                            affine_points=element_vertex_coordinates_symbols,
                            blending=integration_info.blending,
                            quad_info=integration_info.quad,
                        )
                    )

                with TimedLogger("cse on blending operation", logging.DEBUG):
                    cse_impl = self._optimizer.cse_impl()
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

            if not self._optimizer[Opts.QUADLOOPS]:
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
            loop_strategy.add_body_to_loop(loop, body, element_type)

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
                        self._optimizer.cse_impl(),
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
                loop = loop_strategy.add_preloop_for_loop(loop, preloop, element_type)

        # Add quadrature points and weights array declarations, but only those
        # which are actually needed.
        if integration_info.quad_loop:
            q_decls = integration_info.quad_loop.point_weight_decls()
            undefined = Block(loop).undefined_symbols
            body = [q_decl for q_decl in q_decls if q_decl.lhs in undefined] + loop
        else:
            body = loop

        return (Block(body), ops.to_table())

    def generate_kernels(self, loop_strategy: LoopStrategy) -> None:
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

        for kernel_type in self.kernel_types:
            for dim, integration_info in self.element_matrices.items():
                geometry = integration_info.geometry
                macro_type = {2: "face", 3: "cell"}

                # generate AST of kernel loop
                with TimedLogger(
                    f"Generating kernel: {kernel_type.name} in {dim}D", logging.INFO
                ):
                    (
                        function_body,
                        kernel_op_count,
                    ) = self._generate_kernel(
                        dim, integration_info, loop_strategy, kernel_type
                    )

                kernel_function = KernelFunction(
                    function_body,
                    Target.CPU,
                    Backend.C,
                    make_python_function,
                    ghost_layers=None,
                    function_name=f"{kernel_type.name}_macro_{geometry.dimensions}D",
                    assignments=None,
                )

                # optimizer applies optimizations
                with TimedLogger(
                    f"Optimizing kernel: {kernel_type.name} in {dim}D", logging.INFO
                ):
                    optimizer = (
                        self._optimizer
                        if not isinstance(kernel_type, Assemble)
                        else self._optimizer_no_vec
                    )
                    platform_dep_kernels = optimizer.apply_to_kernel(
                        kernel_function, dim, loop_strategy
                    )

                kernel_parameters = kernel_function.get_parameters()

                # setup kernel string and op count table
                #
                # in the following, we insert the sub strings of the final kernel string:
                # coefficients (retrieving pointers), setup of scalar parameters, kernel function call
                # This is done as follows:
                # - the kernel type has the skeleton string in which the sub string must be substituted
                # - the function space impl knows from which function type a src/dst/coefficient field is and can return the
                #   corresponding sub string

                # Retrieve coefficient pointers
                kernel_type.substitute(
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

                scalar_parameter_setup += (
                    integration_info.blending.parameter_coupling_code()
                )

                kernel_type.substitute(
                    {f"scalar_parameter_setup_{dim}D": scalar_parameter_setup}
                )

                # Kernel function call.
                kernel_function_call_parameter_string = ",\n".join(
                    [str(prm) for prm in kernel_parameters]
                )
                kernel_type.substitute(
                    {
                        f"kernel_function_call_{dim}D": f"""
                                {kernel_function.function_name}(\n
                                {kernel_function_call_parameter_string});"""
                    }
                )

                # Collect all information
                kernel = Kernel(
                    kernel_type,
                    kernel_function,
                    platform_dep_kernels,
                    kernel_op_count,
                    integration_info,
                )
                self.kernels.append(kernel)
