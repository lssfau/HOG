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
from string import Template
from textwrap import indent
from typing import List, Mapping, Set, Tuple, Union

import numpy as np
import pystencils as ps
import sympy as sp
from pystencils.astnodes import SympyAssignment
from pystencils.backends.cbackend import CustomCodeNode
from pystencils.field import Field
from pystencils.typing.cast_functions import CastFunc
from pystencils.typing.typed_sympy import FieldPointerSymbol, TypedSymbol

from hog.cpp_printing import (
    CppDefaultArgument,
    CppMemberVariable,
    CppMethod,
    CppVariable,
    INDENT,
)
from hog.element_geometry import ElementGeometry
from hog.exception import HOGException
from hog.function_space import FunctionSpace, TrialSpace, TestSpace
from hog.operator_generation.function_space_impls import FunctionSpaceImpl
from hog.operator_generation.indexing import FaceType, CellType
from hog.operator_generation.pystencils_extensions import create_generic_fields
from hog.operator_generation.types import HOGPrecision, HOGType, hyteg_type
from hog.operator_generation.loop_strategies import LoopStrategy, SAWTOOTH


class KernelType(ABC):
    """
    A HyTeG operator may implement multiple types of methods that perform some kind of operation on a function that
    is inferred from the bilinear form. For instance, a (matrix-free) matrix-vector multiplication or the assembly
    of its diagonal.

    Certain operations such as setting boundary data to zero or communication have to be executed before the actual
    kernel can be executed.

    E.g.:

    ```
    void apply() {
        communication()       // "pre-processing"
        kernel()              // actual kernel
        post_communication()  // "post-processing"
    }
    ```

    For some applications, it might be necessary or at least comfortable to execute multiple kernels inside such a
    method. For instance, if boundary conditions are applied:

    ```
    // form is sum of volume and boundary integral:
    //     ∫ ... dΩ + ∫ ... dS
    void assemble() {
        communication()       // "pre-processing"
        kernel()              // volume kernel (∫ ... dΩ)
        kernel_boundary()     // boundary kernel (∫ ... dS | additive execution)
        post_communication()  // "post-processing"
    }
    ```

    This class (KernelType) describes the "action" of the kernel (matvec, assembly, ...).
    Another class (KernelWrapperType) then describes what kind of pre- and post-processing is required.

    ```
    void assemble() {         // from KernelTypeWrapper
        communication()       // from KernelTypeWrapper
        kernel()              // from KernelType + IntegrationInfo 1
        kernel_boundary()     // from KernelType + IntegrationInfo 2
        post_communication()  // from KernelTypeWrapper
    }
    ```
    """

    @abstractmethod
    def kernel_operation(
        self,
        src_vecs: List[sp.MatrixBase],
        dst_vecs: List[sp.MatrixBase],
        mat: sp.MatrixBase,
        rows: int,
    ) -> List[SympyAssignment]:
        """Applies the actual kernel operation to a set of src and destination vectors.

        All inputs are intended to be "purely" symbolic. No resolution of array accesses has been performed.
        This method shall return a list of SympyAssignments to temporary variables.
        """
        ...

    @abstractmethod
    def kernel_post_operation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        src_vecs_accesses: List[List[Field.Access]],
        dst_vecs_accesses: List[List[Field.Access]],
        element_vertex_ordering: List[int],
    ) -> List[ps.astnodes.Node]:
        """
        Operations to be executed after the access resolution and common subexpression elimination on the symbols
        that are used for the kernel operation.

        At this point you can e.g. write results to vectors using the passed accesses. They correspond to the symbols
        passed to kernel_operation().

        For example, the Apply kernel can additively update the global
        destination vector, while the Assembly can insert values into the sparse
        matrix.
        """
        ...


class Apply(KernelType):
    def __init__(self):
        self.result_prefix = "elMatVec_"

    def kernel_operation(
        self,
        src_vecs: List[sp.MatrixBase],
        dst_vecs: List[sp.MatrixBase],
        mat: sp.MatrixBase,
        rows: int,
    ) -> List[SympyAssignment]:
        kernel_ops = mat * src_vecs[0]

        tmp_symbols = sp.numbered_symbols(self.result_prefix)

        kernel_op_assignments = [
            SympyAssignment(tmp, kernel_op)
            for tmp, kernel_op in zip(tmp_symbols, kernel_ops)
        ]

        return kernel_op_assignments

    def kernel_post_operation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        src_vecs_accesses: List[List[Field.Access]],
        dst_vecs_accesses: List[List[Field.Access]],
        element_vertex_ordering: List[int],
    ) -> List[ps.astnodes.Node]:
        tmp_symbols = sp.numbered_symbols(self.result_prefix)

        # Add and store result to destination.
        store_dst_vecs = [
            SympyAssignment(a, a + s) for a, s in zip(dst_vecs_accesses[0], tmp_symbols)
        ]
        return store_dst_vecs


class AssembleDiagonal(KernelType):
    def __init__(self):
        self.result_prefix = "elMatDiag_"

    def kernel_operation(
        self,
        src_vecs: List[sp.MatrixBase],
        dst_vecs: List[sp.MatrixBase],
        mat: sp.MatrixBase,
        rows: int,
    ) -> List[SympyAssignment]:
        kernel_ops = mat.diagonal()

        tmp_symbols = sp.numbered_symbols(self.result_prefix)

        kernel_op_assignments = [
            SympyAssignment(tmp, kernel_op)
            for tmp, kernel_op in zip(tmp_symbols, kernel_ops)
        ]

        return kernel_op_assignments

    def kernel_post_operation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        src_vecs_accesses: List[List[Field.Access]],
        dst_vecs_accesses: List[List[Field.Access]],
        element_vertex_ordering: List[int],
    ) -> List[ps.astnodes.Node]:
        tmp_symbols = sp.numbered_symbols(self.result_prefix)

        # Add and store result to destination.
        store_dst_vecs = [
            SympyAssignment(a, a + s) for a, s in zip(dst_vecs_accesses[0], tmp_symbols)
        ]
        return store_dst_vecs


class Assemble(KernelType):
    def __init__(
        self,
        src: FunctionSpaceImpl,
        dst: FunctionSpaceImpl,
    ):
        self.result_prefix = "elMat_"

        self.src = src
        self.dst = dst

    def kernel_operation(
        self,
        src_vecs: List[sp.MatrixBase],
        dst_vecs: List[sp.MatrixBase],
        mat: sp.MatrixBase,
        rows: int,
    ) -> List[SympyAssignment]:
        return [
            SympyAssignment(sp.Symbol(f"{self.result_prefix}{r}_{c}"), mat[r, c])
            for r in range(mat.shape[0])
            for c in range(mat.shape[1])
        ]

    def kernel_post_operation(
        self,
        geometry: ElementGeometry,
        element_index: Tuple[int, int, int],
        element_type: Union[FaceType, CellType],
        src_vecs_accesses: List[List[Field.Access]],
        dst_vecs_accesses: List[List[Field.Access]],
        element_vertex_ordering: List[int],
    ) -> List[ps.astnodes.Node]:
        src, dst = dst_vecs_accesses
        el_mat = sp.Matrix(
            [
                [sp.Symbol(f"{self.result_prefix}{r}_{c}") for c in range(len(src))]
                for r in range(len(dst))
            ]
        )

        # apply basis/dof transformations
        transform_src_code, transform_src_mat = self.src.dof_transformation(
            geometry, element_index, element_type, element_vertex_ordering
        )
        transform_dst_code, transform_dst_mat = self.dst.dof_transformation(
            geometry, element_index, element_type, element_vertex_ordering
        )
        transformed_el_mat = transform_dst_mat.T * el_mat * transform_src_mat

        nr = len(dst)
        nc = len(src)
        mat_size = nr * nc

        rowIdx, colIdx = create_generic_fields(["rowIdx", "colIdx"], dtype=np.uint64)
        # NOTE: The type is 'hyteg_type().pystencils_type', i.e. `real_t`
        #       (not `self._type_descriptor.pystencils_type`)
        #       because this function is implemented manually in HyTeG with
        #       this signature. Passing `np.float64` is not ideal (if `real_t !=
        #       double`) but it makes sure that casts are inserted if necessary
        #       (though some might be superfluous).
        mat = create_generic_fields(
            ["mat"], dtype=hyteg_type(HOGPrecision.REAL_T).pystencils_type
        )[0]
        rowIdxSymb = FieldPointerSymbol(rowIdx.name, rowIdx.dtype, False)
        colIdxSymb = FieldPointerSymbol(colIdx.name, colIdx.dtype, False)
        matSymb = FieldPointerSymbol(mat.name, mat.dtype, False)

        body: List[ps.astnodes.Node] = [
            CustomCodeNode(
                f"std::vector< uint_t > {rowIdxSymb.name}( {nr} );\n"
                f"std::vector< uint_t > {colIdxSymb.name}( {nc} );\n"
                f"std::vector< {mat.dtype} > {matSymb.name}( {mat_size} );",
                [],
                [rowIdxSymb, colIdxSymb, matSymb],
            ),
            ps.astnodes.EmptyLine(),
        ]
        body += [
            SympyAssignment(
                rowIdx.absolute_access((k,), (0,)), CastFunc(dst_access, np.uint64)
            )
            for k, dst_access in enumerate(dst)
        ]
        body += [
            SympyAssignment(
                colIdx.absolute_access((k,), (0,)), CastFunc(src_access, np.uint64)
            )
            for k, src_access in enumerate(src)
        ]

        body += [
            ps.astnodes.EmptyLine(),
            ps.astnodes.SourceCodeComment("Apply basis transformation"),
            *sorted({transform_src_code, transform_dst_code}, key=lambda n: n._code),
            ps.astnodes.EmptyLine(),
        ]

        body += [
            SympyAssignment(
                mat.absolute_access((r * nc + c,), (0,)),
                CastFunc(transformed_el_mat[r, c], mat.dtype),
            )
            for r in range(nr)
            for c in range(nc)
        ]

        body += [
            ps.astnodes.EmptyLine(),
            CustomCodeNode(
                f"mat->addValues( {rowIdxSymb.name}, {colIdxSymb.name}, {matSymb.name} );",
                [
                    TypedSymbol("mat", "std::shared_ptr< SparseMatrixProxy >"),
                    rowIdxSymb,
                    colIdxSymb,
                    matSymb,
                ],
                [],
            ),
        ]

        return body


class KernelWrapperType(ABC):
    name: str
    src_fields: List[FunctionSpaceImpl]
    dst_fields: List[FunctionSpaceImpl]
    _template: Template
    """
    See documentation of class KernelType. 
    """

    @property
    @abstractmethod
    def kernel_type(self) -> KernelType:
        ...

    @abstractmethod
    def includes(self) -> Set[str]:
        ...

    @abstractmethod
    def base_classes(self) -> List[str]:
        ...

    @abstractmethod
    def wrapper_methods(self) -> List[CppMethod]:
        ...

    @abstractmethod
    def member_variables(self) -> List[CppMemberVariable]:
        ...

    def substitute(self, subs: Mapping[str, object]) -> None:
        self._template = Template(self._template.safe_substitute(subs))


class ApplyWrapper(KernelWrapperType):
    def __init__(
        self,
        src_space: TrialSpace,
        dst_space: TestSpace,
        type_descriptor: HOGType,
        dims: List[int] = [2, 3],
    ):
        self.name = "apply"
        self.src: FunctionSpaceImpl = FunctionSpaceImpl.create_impl(
            src_space, "src", type_descriptor
        )
        self.dst: FunctionSpaceImpl = FunctionSpaceImpl.create_impl(
            dst_space, "dst", type_descriptor
        )
        self.src_fields = [self.src]
        self.dst_fields = [self.dst]
        self.dims = dims
        self.result_prefix = "elMatVec_"

        def macro_loop(dim: int) -> str:
            Macro = {2: "Face", 3: "Cell"}[dim]
            macro = {2: "face", 3: "cell"}[dim]

            if dim in dims:
                return (
                    f"for ( auto& it : storage_->get{Macro}s() )\n"
                    f"{{\n"
                    f"    {Macro}& {macro} = *it.second;\n"
                    f"\n"
                    f"    // get hold of the actual numerical data in the functions\n"
                    f"{indent(self.src.pointer_retrieval(dim), INDENT)}\n"
                    f"{indent(self.dst.pointer_retrieval(dim), INDENT)}\n"
                    f"    $pointer_retrieval_{dim}D\n"
                    f"\n"
                    f"    // Zero out dst halos only\n"
                    f"    //\n"
                    f"    // This is also necessary when using update type == Add.\n"
                    f"    // During additive comm we then skip zeroing the data on the lower-dim primitives.\n"
                    f"{indent(self.dst.zero_halos(dim), INDENT)}\n"
                    f"\n"
                    f"    $scalar_parameter_setup_{dim}D\n"
                    f"\n"
                    f'    this->timingTree_->start( "kernel" );\n'
                    f"    $kernel_function_call_{dim}D\n"
                    f'    this->timingTree_->stop( "kernel" );\n'
                    f"}}\n"
                    f"\n"
                    f"// Push result to lower-dimensional primitives\n"
                    f"//\n"
                    f'this->timingTree_->start( "post-communication" );\n'
                    f"// Note: We could avoid communication here by implementing the apply() also for the respective\n"
                    f"//       lower dimensional primitives!\n"
                    f"{self.dst.post_communication(dim, 'level, DoFType::All ^ flag, *storage_, updateType == Replace')}\n"
                    f'this->timingTree_->stop( "post-communication" );'
                )
            else:
                return 'WALBERLA_ABORT( "Not implemented." );'

        def halo_update(dim: int) -> str:
            if dim in dims:
                if dim == 2:
                    return (
                        f"{indent(self.src.pre_communication(2), INDENT)}\n"
                        f"    $comm_fe_functions_2D\n"
                    )
                elif dim == 3:
                    return (
                        f"    // Note that the order of communication is important, since the face -> cell communication may overwrite\n"
                        f"    // parts of the halos that carry the macro-vertex and macro-edge unknowns.\n"
                        f"{indent(self.src.pre_communication(3), INDENT)}\n"
                        f"    $comm_fe_functions_3D\n"
                    )
                else:
                    raise HOGException("Dim not supported.")
            else:
                return 'WALBERLA_ABORT( "Not implemented." );'

        self._template = Template(
            f'this->startTiming( "{self.name}" );\n'
            f"\n"
            f"// Make sure that halos are up-to-date\n"
            f'this->timingTree_->start( "pre-communication" );\n'
            f"if ( this->storage_->hasGlobalCells() )\n"
            f"{{\n"
            f"{halo_update(3)}"
            f"}}\n"
            f"else\n"
            f"{{\n"
            f"{halo_update(2)}"
            f"}}\n"
            f'this->timingTree_->stop( "pre-communication" );\n'
            f"\n"
            f"if ( updateType == Replace )\n"
            f"{{\n"
            f"    // We need to zero the destination array (including halos).\n"
            f"    // However, we must not zero out anything that is not flagged with the specified BCs.\n"
            f"    // Therefore, we first zero out everything that flagged, and then, later,\n"
            f"    // the halos of the highest dim primitives.\n"
            f"    dst.interpolate( walberla::numeric_cast< {self.dst.type_descriptor.pystencils_type} >( 0 ), level, flag );\n"
            f"}}\n"
            f"\n"
            f"if ( storage_->hasGlobalCells() )\n"
            f"{{\n"
            f"{indent(macro_loop(3), INDENT)}\n"
            f"}}\n"
            f"else\n"
            f"{{\n"
            f"{indent(macro_loop(2), INDENT)}\n"
            f"}}\n"
            f"\n"
            f'this->stopTiming( "{self.name}" );'
        )

    @property
    def kernel_type(self) -> KernelType:
        return Apply()

    def includes(self) -> Set[str]:
        return (
            {"hyteg/operators/Operator.hpp"} | self.src.includes() | self.dst.includes()
        )

    def base_classes(self) -> List[str]:
        return [
            f"public Operator< {self.src.func_type_string()}, {self.dst.func_type_string()} >",
        ]

    def wrapper_methods(self) -> List[CppMethod]:
        return [
            CppMethod(
                name=self.name,
                arguments=[
                    CppVariable(
                        name=self.src.name,
                        type=self.src.func_type_string(),
                        is_const=True,
                        is_reference=True,
                    ),
                    CppVariable(
                        name=self.dst.name,
                        type=self.dst.func_type_string(),
                        is_const=True,
                        is_reference=True,
                    ),
                    CppVariable(name="level", type="uint_t"),
                    CppVariable(name="flag", type="DoFType"),
                    CppDefaultArgument(
                        variable=CppVariable(name="updateType", type="UpdateType"),
                        default_value="Replace",
                    ),
                ],
                return_type="void",
                is_const=True,
                content=self._template.template,
            )
        ]

    def member_variables(self) -> List[CppMemberVariable]:
        return []


class GEMVWrapper(KernelWrapperType):
    def __init__(
        self,
        type_descriptor: HOGType,
        name: str = "gemv",
        src_fields: List[str] = ["src1", "src2"],
        dst_fields: List[str] = ["dst"],
        scalar_params: List[str] = ["alpha", "beta"],
        dims: List[int] = [2, 3],
    ):
        pass

    def kernel_operation(
        self,
        src_vecs: List[sp.MatrixBase],
        dst_vecs: List[sp.MatrixBase],
        mat: sp.MatrixBase,
        rows: int,
    ) -> List[SympyAssignment]:
        # TODO invent a solution for the macro-boundary-infinite-neighbor problem for function-update + operator apply kernels
        raise HOGException("Not implemented")


class AssembleDiagonalWrapper(KernelWrapperType):
    def __init__(
        self,
        fe_space: FunctionSpace,
        type_descriptor: HOGType,
        dst_field: str = "invDiag_",
        dims: List[int] = [2, 3],
    ):
        self.name = "computeInverseDiagonalOperatorValues"
        self.dst: FunctionSpaceImpl = FunctionSpaceImpl.create_impl(
            fe_space, dst_field, type_descriptor, is_pointer=True
        )
        self.src_fields = []
        self.dst_fields = [self.dst]
        self.dims = dims

        def macro_loop(dim: int) -> str:
            Macro = {2: "Face", 3: "Cell"}[dim]
            macro = {2: "face", 3: "cell"}[dim]

            if dim in dims:
                return (
                    f"for ( auto& it : storage_->get{Macro}s() )\n"
                    f"{{\n"
                    f"    {Macro}& {macro} = *it.second;\n"
                    f"\n"
                    f"    // get hold of the actual numerical data\n"
                    f"{indent(self.dst.pointer_retrieval(dim), INDENT)}\n"
                    f"    $pointer_retrieval_{dim}D\n"
                    f"\n"
                    f"    $scalar_parameter_setup_{dim}D\n"
                    f"\n"
                    f'    this->timingTree_->start( "kernel" );\n'
                    f"    $kernel_function_call_{dim}D\n"
                    f'    this->timingTree_->stop( "kernel" );\n'
                    f"}}\n"
                    f"\n"
                    f"// Push result to lower-dimensional primitives\n"
                    f"//\n"
                    f'this->timingTree_->start( "post-communication" );\n'
                    f"// Note: We could avoid communication here by implementing the apply() also for the respective\n"
                    f"//       lower dimensional primitives!\n"
                    f"{self.dst.post_communication(dim, 'level', transform_basis=False)}\n"
                    f'this->timingTree_->stop( "post-communication" );'
                )
            else:
                return 'WALBERLA_ABORT( "Not implemented." );'

        fnc_name = "inverse diagonal entries"

        self._template = Template(
            f'this->startTiming( "{self.name}" );\n'
            f"\n"
            f"if ( {self.dst.name} == nullptr )\n"
            f"{{\n"
            f"    {self.dst.name} =\n"
            f'        std::make_shared< {self.dst.func_type_string()} >( "{fnc_name}", storage_, minLevel_, maxLevel_ );\n'
            f"}}\n"
            f"\n"
            f"for ( uint_t level = minLevel_; level <= maxLevel_; level++ )\n"
            f"{{\n"
            f"    {self.dst.name}->setToZero( level );\n"
            f"\n"
            f"    if ( storage_->hasGlobalCells() )\n"
            f"    {{\n"
            f'        this->timingTree_->start( "pre-communication" );\n'
            f"        $comm_fe_functions_3D\n"
            f'        this->timingTree_->stop( "pre-communication" );\n'
            f"\n"
            f"{indent(macro_loop(3), 2 * INDENT)}\n"
            f"{indent(self.dst.invert_elementwise(3), 2 * INDENT)}\n"
            f"    }}\n"
            f"    else\n"
            f"    {{\n"
            f'        this->timingTree_->start( "pre-communication" );\n'
            f"        $comm_fe_functions_2D\n"
            f'        this->timingTree_->stop( "pre-communication" );\n'
            f"\n"
            f"{indent(macro_loop(2), 2 * INDENT)}\n"
            f"{indent(self.dst.invert_elementwise(2), 2 * INDENT)}\n"
            f"    }}\n"
            f"\n"
            f"}}\n"
            f"\n"
            f'this->stopTiming( "{self.name}" );'
        )

    @property
    def kernel_type(self) -> KernelType:
        return AssembleDiagonal()

    def includes(self) -> Set[str]:
        return {"hyteg/solvers/Smoothables.hpp"} | self.dst.includes()

    def base_classes(self) -> List[str]:
        return [
            f"public OperatorWithInverseDiagonal< {self.dst.func_type_string()} >",
        ]

    def wrapper_methods(self) -> List[CppMethod]:
        return [
            CppMethod(
                name=self.name,
                arguments=[],
                return_type="void",
                content=self._template.template,
            ),
            CppMethod(
                name="getInverseDiagonalValues",
                arguments=[],
                return_type=f"std::shared_ptr< {self.dst.func_type_string()} >",
                is_const=True,
                content=f"return {self.dst.name};",
            ),
        ]

    def member_variables(self) -> List[CppMemberVariable]:
        return [
            CppMemberVariable(
                variable=CppVariable(
                    name=self.dst.name,
                    type=f"std::shared_ptr< {self.dst.func_type_string()} >",
                )
            ),
        ]


class AssembleWrapper(KernelWrapperType):
    def __init__(
        self,
        src_space: TrialSpace,
        dst_space: TestSpace,
        type_descriptor: HOGType,
        dims: List[int] = [2, 3],
    ):
        idx_t = HOGType("idx_t", np.int64)
        self.name = "toMatrix"
        self.src: FunctionSpaceImpl = FunctionSpaceImpl.create_impl(
            src_space, "src", idx_t
        )
        self.dst: FunctionSpaceImpl = FunctionSpaceImpl.create_impl(
            dst_space, "dst", idx_t
        )

        # Treating both src and dst as dst_fields because src_fields are loaded
        # explicitly from memory prior to the kernel operation but we do not
        # need that here.
        self.src_fields = []
        self.dst_fields = [self.src, self.dst]

        self.type_descriptor = type_descriptor
        self.dims = dims

        def macro_loop(dim: int) -> str:
            Macro = {2: "Face", 3: "Cell"}[dim]
            macro = {2: "face", 3: "cell"}[dim]

            if dim in dims:
                return (
                    f"for ( auto& it : storage_->get{Macro}s() )\n"
                    f"{{\n"
                    f"    {Macro}& {macro} = *it.second;\n"
                    f"\n"
                    f"    // get hold of the actual numerical data\n"
                    f"{indent(self.src.pointer_retrieval(dim), INDENT)}\n"
                    f"{indent(self.dst.pointer_retrieval(dim), INDENT)}\n"
                    f"    $pointer_retrieval_{dim}D\n"
                    f"\n"
                    f"    $scalar_parameter_setup_{dim}D\n"
                    f"\n"
                    f'    this->timingTree_->start( "kernel" );\n'
                    f"    $kernel_function_call_{dim}D\n"
                    f'    this->timingTree_->stop( "kernel" );\n'
                    f"}}"
                )
            else:
                return 'WALBERLA_ABORT( "Not implemented." );'

        self._template = Template(
            f'this->startTiming( "{self.name}" );\n'
            f"\n"
            f"// We currently ignore the flag provided!\n"
            f"if ( flag != All )\n"
            f"{{\n"
            f'    WALBERLA_LOG_WARNING_ON_ROOT( "Input flag ignored in {self.name}; using flag = All" );\n'
            f"}}\n"
            f"\n"
            f"if ( storage_->hasGlobalCells() )\n"
            f"{{\n"
            f'    this->timingTree_->start( "pre-communication" );\n'
            f"    $comm_fe_functions_3D\n"
            f'    this->timingTree_->stop( "pre-communication" );\n'
            f"\n"
            f"{indent(macro_loop(3), INDENT)}\n"
            f"}}\n"
            f"else\n"
            f"{{\n"
            f'    this->timingTree_->start( "pre-communication" );\n'
            f"    $comm_fe_functions_2D\n"
            f'    this->timingTree_->stop( "pre-communication" );\n'
            f"\n"
            f"{indent(macro_loop(2), INDENT)}\n"
            f"}}\n"
            f'this->stopTiming( "{self.name}" );'
        )

    @property
    def kernel_type(self) -> KernelType:
        return Assemble(self.src, self.dst)

    def includes(self) -> Set[str]:
        return (
            {
                "hyteg/operators/Operator.hpp",
                "hyteg/sparseassembly/SparseMatrixProxy.hpp",
            }
            | self.src.includes()
            | self.dst.includes()
        )

    def base_classes(self) -> List[str]:
        # TODO The toMatrix method is part of the Operator interface but we do
        #      not know the value types of the src and dst functions.
        return []

    def wrapper_methods(self) -> List[CppMethod]:
        return [
            CppMethod(
                name=self.name,
                arguments=[
                    CppVariable(
                        name="mat",
                        type="std::shared_ptr< SparseMatrixProxy >",
                        is_const=True,
                        is_reference=True,
                    ),
                    CppVariable(
                        name=self.src.name,
                        type=self.src.func_type_string(),
                        is_const=True,
                        is_reference=True,
                    ),
                    CppVariable(
                        name=self.dst.name,
                        type=self.dst.func_type_string(),
                        is_const=True,
                        is_reference=True,
                    ),
                    CppVariable(name="level", type="uint_t"),
                    CppVariable(name="flag", type="DoFType"),
                ],
                return_type="void",
                is_const=True,
                content=self._template.template,
            )
        ]

    def member_variables(self) -> List[CppMemberVariable]:
        return []
