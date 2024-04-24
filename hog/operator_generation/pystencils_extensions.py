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

from typing import Dict, List, Tuple, Union

import sympy as sp
from pystencils import FieldType, Field
from pystencils.astnodes import Block, LoopOverCoordinate, Node, get_next_parent_of_type
import pystencils as ps
from pystencils.astnodes import (
    Block,
    LoopOverCoordinate,
    Node,
    Conditional,
    KernelFunction,
)
from pystencils.backends.cbackend import CBackend, CFunction
from pystencils.enums import Backend
from pystencils.typing.types import BasicType

from hog.exception import HOGException
from hog.operator_generation.indexing import FaceType, CellType
from hog.operator_generation.types import HOGType


def loop_over_simplex(
    dim: int, width: int, cut_innermost: int = 0
) -> LoopOverCoordinate:
    """
    Arranges loop (ast-)nodes implementing an iteration over a structured simplex of arbitrary dimension.

    The width is the size in each direction. For dim = 1, this is just a one dimensional loop.
    Dim = 2 gives a loop over a structured triangle. Illustrated for width == 5 below:

                        +
                        + +
                        + + +
                      ^ + + + +
    second coordinate | + + + + +
                        -> first coordinate

    This is implemented by a dependency of all loop counters on all other _outside_ loop counters.
    In C, a two dim loop would look like this:

    // y-direction
    for ( int idx_1 = 0; idx_1 < width; idx_1++ )
    {
        // x-direction
        for ( int idx_0 = 0; idx_0 < width - idx_1; idx_0++ )
        {
            // body
        }
    }

    Returns the outermost loop ast node. The cut_innermost parameter restricts the innermost loop to a
    range of width - sum(outer_loop_counter_symbols) - cut_innermost
    """

    loops = {}
    for d in range(dim):
        outer_loop_counter_symbols = [
            LoopOverCoordinate.get_loop_counter_symbol(coord)
            for coord in range(d + 1, dim)
        ]
        loops[d] = LoopOverCoordinate(
            Block([]),
            d,
            0,
            width - sum(outer_loop_counter_symbols) - (cut_innermost if d == 0 else 0),
        )

    for d in range(1, dim):
        loops[d] = loops[d].new_loop_with_different_body(loops[d - 1])

    return loops[dim - 1]


def loop_over_simplex_facet(dim: int, width: int, facet_id: int) -> LoopOverCoordinate:
    """
    Loops over one boundary facet of a simplex (e.g., one edge in 2D, one face in 3D).

    The facet is specified by an integer. It is required that

        0 ≤ facet_id ≤ dim

    Let [x_0, x_1, ..., x_(dim-1)] be the coordinate of one element that is looped over.

    For facet_id < dim, we have that

        x_(dim - 1 - facet_id) = 0

    and the remaining boundary is selected with facet_id == dim.

    So in 2D for example, we get

        facet_id = 0: (x_0, 0  )
        facet_id = 1: (0,   x_1)
        facet_id = 2: the xy- (or "diagonal") boundary
    """
    loop = loop_over_simplex(dim, width)
    innermost_loops = get_innermost_loop(loop)
    if len(innermost_loops) != 1:
        raise HOGException("There should be only one innermost loop.")
    innermost_loop: LoopOverCoordinate = innermost_loops[0]

    if facet_id not in range(0, dim + 1):
        raise HOGException(f"Bad facet_id ({facet_id}) for dim {dim}.")

    if facet_id == dim:
        # For the "diagonal" loop we can just iterate as usual but skip all but the last element of the innermost loop.
        # I hope.
        innermost_loop.start = innermost_loop.stop - 1
        return loop

    # For the other facet_ids we need to find the corresponding loop and set that counter to 0.
    # I am doing this here by just traversing the loops from the inside out, assuming that no loop cutting etc.
    # occurred.
    loop_with_counter_to_be_set_to_zero = innermost_loop
    for d in range(dim - 1 - facet_id):
        loop_with_counter_to_be_set_to_zero = get_next_parent_of_type(
            loop_with_counter_to_be_set_to_zero, LoopOverCoordinate
        )
    if loop_with_counter_to_be_set_to_zero is None:
        raise HOGException("There was no parent loop. This should not happen. I think.")

    loop_with_counter_to_be_set_to_zero.start = 0
    loop_with_counter_to_be_set_to_zero.stop = 1

    return loop


def create_micro_element_loops(
    dim: int, micro_edges_per_macro_edge: int
) -> Dict[Union[FaceType, CellType], LoopOverCoordinate]:
    element_loops: Dict[Union[FaceType, CellType], LoopOverCoordinate] = {}
    if dim == 2:
        element_loops[FaceType.GRAY] = loop_over_simplex(
            dim, micro_edges_per_macro_edge
        )
        element_loops[FaceType.BLUE] = loop_over_simplex(
            dim, micro_edges_per_macro_edge, cut_innermost=1
        )
    elif dim == 3:
        cutoff_for_celltype = {
            CellType.WHITE_UP: 0,
            CellType.WHITE_DOWN: 2,
            CellType.BLUE_UP: 1,
            CellType.BLUE_DOWN: 1,
            CellType.GREEN_UP: 1,
            CellType.GREEN_DOWN: 1,
        }
        for cellType in CellType:
            element_loops[cellType] = loop_over_simplex(
                dim,
                micro_edges_per_macro_edge,
                cut_innermost=cutoff_for_celltype[cellType],
            )

    else:
        raise HOGException("Wrong dim")
    return element_loops


def fuse_loops_over_simplex(
    loops: List[LoopOverCoordinate], dim_to_fuse: int, max_dim: int
) -> Tuple[LoopOverCoordinate, List[Node]]:
    """Takes a list of simplex loops over max_dim dimensions and fuses them at dim_to_fuse.
    E.g. for dim_to_fuse == 0:  L_z(L_y(L_x_1(...))) + L_z(L_y(L_x_2(...))) = L_z(L_y([L_x_1(...), L_x_2(...)]))
    """

    # fused loop will be constructed here
    current_loop = loops[0]
    fused_loops = {}
    for d in range(max_dim, dim_to_fuse, -1):
        if not isinstance(current_loop, LoopOverCoordinate):
            raise HOGException(f"Non-loop encountered: {current_loop}")

        # reconstruct current loop
        fused_loops[d] = current_loop.new_loop_with_different_body(Block([]))

        # assert fusability
        # ranges = [(loop.start, loop.step, loop.stop) for loop in loops]
        # is_same = reduce(lambda p, q: p[0] - q[0] + p[1] - q[1] + p[2] - q[2] == 0, ranges, 0)
        # if not is_same:
        #    raise HOGException(f"Loop ranges are not the same for dimension {d}!")

        # iterate loop
        current_loop = current_loop.body

    # collect bodies, add to constructed loop
    dim_to_fuse_loops = []
    for loop in loops:
        current_loop = loop
        for d in range(max_dim - 1, dim_to_fuse, -1):
            current_loop = current_loop.body
        dim_to_fuse_loops.append(current_loop.body)

    offset = 0 if max_dim == 2 else 1
    fused_loops[max_dim - offset].body = Block(dim_to_fuse_loops)
    for d in range(max_dim, dim_to_fuse + 1, -1):
        fused_loops[d] = fused_loops[d].new_loop_with_different_body(fused_loops[d - 1])

    return (fused_loops[max_dim], [loop.body for loop in dim_to_fuse_loops])


def get_innermost_loop(
    ast_node: Node, shift_to_outer: int = 0, return_all_inner: bool = False
) -> List[LoopOverCoordinate]:
    """Returns the (innermost + shift_to_outer) loop of the given ast node. If there are more than one, throws an exception.
    For example, get_innermost_loop(ast_node, 1) returns the loop one further out from the innermost loop.
    An exception is raised if shift_to_outer is larger than the overall loop depth."""
    all_loops: List[LoopOverCoordinate] = ast_node.atoms(LoopOverCoordinate)
    inner_loops = [loop for loop in all_loops if loop.is_innermost_loop]
    if len(inner_loops) != 1:
        if not return_all_inner:
            raise Exception("There are less or more than 1 innermost loops.")
    if shift_to_outer > 0:
        # TODO This assumes that all innermost loops share the same parent
        inner_loop = inner_loops[0]
        for s in range(0, shift_to_outer):
            inner_loop = inner_loop.parent
            while isinstance(inner_loop, Block):
                inner_loop = inner_loop.parent

            if inner_loop == ast_node:
                raise Exception("Specified shifted-to-outer loop is the outest loop!")

        return [inner_loop]
    else:
        return inner_loops


def remove_blocks(ast_node: Node) -> None:
    """Traverses the node and removes all obsolete blocks"""
    for arg in ast_node.args:
        remove_blocks(arg)
        if isinstance(arg, Block) and isinstance(arg.parent, Block):
            for argarg in arg.args:
                argarg.parent = arg.parent
                arg.parent.args.append(argarg)
            arg.parent.args.remove(arg)


def create_generic_fields(
    names: List[str], dtype: Union[BasicType, type]
) -> List[Field]:
    field_list = []
    for name in names:
        f = Field.create_generic(name, 1, dtype=dtype, field_type=FieldType.CUSTOM)
        f.strides = tuple([1 for _ in f.strides])
        field_list.append(f)
    return field_list


def create_field_access(
    name: str, dtype: Union[BasicType, type], idx: sp.Expr
) -> Field.Access:
    f = Field.create_generic(name, 1, dtype=dtype, field_type=FieldType.CUSTOM)
    f.strides = tuple([1 for _ in f.strides])
    return f.absolute_access((idx,), (0,))
