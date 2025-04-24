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
import re
from typing import Type, Union, Dict

from pystencils import TypedSymbol
from pystencils.astnodes import (
    Block,
    Conditional,
    ResolvedFieldAccess,
    SourceCodeComment,
    LoopOverCoordinate,
)
from pystencils.sympyextensions import fast_subs
from pystencils.typing import FieldPointerSymbol

import sympy as sp

from hog.exception import HOGException
from hog.operator_generation.pystencils_extensions import (
    loop_over_simplex,
    loop_over_simplex_facet,
    get_innermost_loop,
    create_field_access,
    create_micro_element_loops,
    fuse_loops_over_simplex,
)
from hog.operator_generation.indexing import (
    all_element_types,
    FaceType,
    CellType,
)


def get_element_replacement(node, element_type):
    if isinstance(node, ResolvedFieldAccess):
        return create_field_access(
            re.sub("_data_", "", node.base.name) + "_" + str(element_type.name),
            node.base.dtype,
            node.args[1],
        )
    elif isinstance(node, FieldPointerSymbol):
        return FieldPointerSymbol(
            re.sub("_data_", "", node.name) + "_" + str(element_type.name),
            node.dtype,
            False,
        )
    elif isinstance(node, TypedSymbol):
        return TypedSymbol(node.name + "_" + str(element_type.name), node.dtype)
    else:
        raise HOGException(f"Unexpected node {node}")


class LoopStrategy(ABC):
    """Each loop strategy creates a different combination and number of inner/outer loops and adds the eleemnt_type specific kernels
    in a specific way to the loop construct."""

    def __init__(self):
        pass

    @abstractmethod
    def create_loop(self, dim, element_index, micro_edges_per_macro_edge):
        """Creates a loop structure corresponding to the strategy."""
        pass

    @abstractmethod
    def add_body_to_loop(self, loop, body, element_type):
        """Inserts given loop bodies to the loop structure."""
        pass

    @abstractmethod
    def get_inner_bodies(self, loop_body):
        """Returns the inner loop bodies of a loop structure."""
        pass

    @abstractmethod
    def add_preloop_for_loop(self, loops, preloop_stmts, element_type):
        """Add given list of statements directly in front of the loop corresponding to element type."""

    @abstractmethod
    def __str__(self):
        pass


class CUBES(LoopStrategy):
    """
    For the "cubes" loop strategy, we want to update all micro-elements in the current "cube", i.e. loop over all
    element types for the current micro-element index before incrementing.
    This way we only have one loop, but need to insert conditionals that take care of those element types that
    are not existing at the loop boundary.
    The hope is that this strategy induces better cache locality, and that the conditionals can be automatically
    transformed by the loop cutting features of pystencils.
    """

    def __init__(self):
        super(CUBES, self).__init__()

    def create_loop(self, dim, element_index, micro_edges_per_macro_edge):
        """We now build all the conditional blocks for all element types. They are filled later."""
        self.conditional_blocks = {}
        for element_type in all_element_types(dim):
            if (dim, element_type) in [
                (2, FaceType.GRAY),
                (3, CellType.WHITE_UP),
            ]:
                cb = Conditional(sp.Eq(0, 0), Block([]))
            elif (dim, element_type) in [
                (3, CellType.WHITE_DOWN),
            ]:
                cb = Conditional(
                    sp.Lt(
                        element_index[0],
                        micro_edges_per_macro_edge - 2 - sum(element_index[1:]),
                    ),
                    Block([]),
                )
            else:
                cb = Conditional(
                    sp.Lt(
                        element_index[0],
                        micro_edges_per_macro_edge - 1 - sum(element_index[1:]),
                    ),
                    Block([]),
                )
            self.conditional_blocks[element_type] = cb

        # For the "cubes" loop strategy we only need one loop.
        # The different element types are handled later via conditionals.
        return loop_over_simplex(dim, micro_edges_per_macro_edge)

    def add_body_to_loop(self, loop, body, element_type):
        """Adds all conditionals to the innermost loop."""
        conditional_body = Block(body)
        self.conditional_blocks[element_type].true_block.append(conditional_body)
        conditional_body.parent = self.conditional_blocks[element_type].true_block

        body = Block([cb for cb in self.conditional_blocks.values()])
        innermost_loop = get_innermost_loop(loop)
        innermost_loop[0].body = body
        body.parent = innermost_loop[0]

    def get_inner_bodies(self, loop_body):
        bodies = [
            inner_loop.body
            for inner_loop in get_innermost_loop(loop_body, return_all_inner=True)
        ]
        assert len(bodies) == 1, "Expecting only one block as body here."
        return bodies[0].args

    def add_preloop_for_loop(self, loops, preloop_stmts, element_type):
        """add given list of statements directly in front of the loop corresponding to element_type."""
        if not isinstance(loops, list):
            loops = [loops]
        preloop_stmts_lhs_subs = {
            stmt.lhs: get_element_replacement(stmt.lhs, element_type)
            for stmt in preloop_stmts
        }

        self.conditional_blocks[element_type] = fast_subs(
            self.conditional_blocks[element_type], preloop_stmts_lhs_subs
        )

        new_preloop_stmts = [
            stmt.fast_subs(preloop_stmts_lhs_subs) for stmt in preloop_stmts
        ]

        return new_preloop_stmts + loops

    def __str__(self):
        return "CUBES"


class SAWTOOTH(LoopStrategy):
    """The naive way to loop over the mesh: construct a loop for each element type (e.g. BLUE and GRAY in 2D) which are executed
    in succession. The pattern of looping over upward facing triangles and then over downward facing triangles gives the loop
     strategy its name."""

    def __init__(self):
        super(SAWTOOTH, self).__init__()

    def create_loop(self, dim, element_index, micro_edges_per_macro_edge):
        """create loops for each element type and pack them into a block that is returned
        loops over certain elements have to be cut in the innermost dimension"""
        self.element_loops = create_micro_element_loops(dim, micro_edges_per_macro_edge)
        return Block(
            [
                Block([SourceCodeComment(str(element_type)), loop])
                for element_type, loop in self.element_loops.items()
            ]
        )

    def add_body_to_loop(self, loop, body, element_type):
        """Register a given loop body to innermost loop of the outer loop corresponding to element_type"""
        innermost_loop = get_innermost_loop(self.element_loops[element_type])
        body = Block(body)
        innermost_loop[0].body = body
        body.parent = innermost_loop[0]

    def get_inner_bodies(self, loop_body):
        return [
            inner_loop.body
            for inner_loop in get_innermost_loop(loop_body, return_all_inner=True)
        ]

    def add_preloop_for_loop(self, loops, preloop_stmts, element_type):
        """add given list of statements directly in front of the loop corresponding to element_type."""
        preloop_stmts_lhs_subs = {
            stmt.lhs: get_element_replacement(stmt.lhs, element_type)
            for stmt in preloop_stmts
        }

        if not isinstance(loops, Block):
            loops = Block(loops)

        blocks = loops.take_child_nodes()
        new_blocks = []
        for block in blocks:
            idx_to_slice_at = -1
            for idx, stmt in enumerate(block.args):
                if stmt == self.element_loops[element_type]:
                    idx_to_slice_at = idx
                    break

            if idx_to_slice_at == -1:
                new_blocks.append(block)
            else:
                new_stmts = (
                    block.args[0:idx_to_slice_at]
                    + preloop_stmts
                    + block.args[idx_to_slice_at:]
                )
                new_block = Block(new_stmts)
                new_block.fast_subs(preloop_stmts_lhs_subs)
                new_blocks.append(new_block)

        return new_blocks

    def __str__(self):
        return "SAWTOOTH"


class FUSEDROWS(LoopStrategy):
    """Modification of SAWTOOTH: fuse the z and y loops of the standard SAWTOOTH element loops together, such that for
    each (z,y) row of the     macro tetrahedron the x-dimension of each element loop (blue, gray etc) is executed in
    order. Up to certain refinement level, a whole row of micro tetrahedra can be held in cache, such that the DoFs
    overlapping with other micro tetrahedra of a different typ are still cached when the next row is computed. E.g. in
    3D at position (z,y), the WHITE_DOWN row/x-dim loop will access DoFs overlapping with elements of the WHITE-UP row,
    that was computed just prior to it and loaded the DoFs from main memory. Thereby, a similar positive effect as for
    CUBES is achieved with less complexity: only fusing the z and y loops is required. Additionally, penalty from
    frequent switches between vectorized and remainder loop is avoided."""

    def __init__(self):
        super(FUSEDROWS, self).__init__()

    def create_loop(self, dim, element_index, micro_edges_per_macro_edge):
        """create a single loop with and x-dim loop for each micro element type in the innermost dimension."""
        element_loops = create_micro_element_loops(dim, micro_edges_per_macro_edge)
        (fused_loop, bodies) = fuse_loops_over_simplex(
            [elem_loop for elem_loop in element_loops.values()], 1, dim
        )
        assert len(bodies) == 2 * 3 ** (dim - 2)
        if dim == 2:
            element_type: Union[Type[FaceType], Type[CellType]] = FaceType
        else:
            element_type = CellType
        self.bodies = {
            eType: body
            for eType, body in zip([eType for eType in element_type], bodies)
        }

        return fused_loop

    def add_body_to_loop(self, loop, body, element_type):
        body = Block(body)
        self.bodies[element_type].parent.body = body
        body.parent = self.bodies[element_type].parent
        self.bodies[element_type] = body

    def get_inner_bodies(self, loop_body):
        return self.bodies.values()

    def add_preloop_for_loop(self, loops, preloop_stmts, element_type):
        if not isinstance(loops, list):
            loops = [loops]
        preloop_stmts_lhs_subs = {
            stmt.lhs: get_element_replacement(stmt.lhs, element_type)
            for stmt in preloop_stmts
        }

        body = self.bodies[element_type]
        assert isinstance(body, Block), f"Encountered body that is not a Block: {body}"
        body.fast_subs(preloop_stmts_lhs_subs)

        new_preloop_stmts = [
            stmt.fast_subs(preloop_stmts_lhs_subs) for stmt in preloop_stmts
        ]

        return new_preloop_stmts + loops

    def __str__(self):
        return "FUSEDROWS"


class BOUNDARY(LoopStrategy):
    """
    Special loop strategy that only loops over elements of a specified boundary.

    Concretely, this means that it iterates over all elements with facets (edges in 2D, faces in 3D) that fully overlap
    with the specified boundary (one of 3 macro-edges in 2D, one of 4 macro-faces in 3D).

    To loop over multiple boundaries, just construct multiple loops.

    The loop is constructed using conditionals, see the CUBES loop strategy.
    """

    def __init__(self, facet_id: int):
        """
        Constructs and initializes the BOUNDARY loop strategy.

        :param facet_id: in [0, 2] for 2D, in [0, 3] for 3D
        """
        super(BOUNDARY, self).__init__()
        self.facet_id = facet_id
        self.element_loops: Dict[Union[FaceType, CellType], LoopOverCoordinate] = dict()

    def create_loop(self, dim, element_index, micro_edges_per_macro_edge):
        if dim == 2:
            if self.facet_id not in [0, 1, 2]:
                raise HOGException("Invalid facet ID for BOUNDARY loop strategy in 2D.")

            self.element_loops = {
                FaceType.GRAY: loop_over_simplex_facet(
                    dim, micro_edges_per_macro_edge, self.facet_id
                ),
            }

        elif dim == 3:
            if self.facet_id not in [0, 1, 2, 3]:
                raise HOGException("Invalid facet ID for BOUNDARY loop strategy in 3D.")

            second_cell_type = {
                0: CellType.BLUE_UP,
                1: CellType.GREEN_UP,
                2: CellType.BLUE_DOWN,
                3: CellType.GREEN_DOWN,
            }

            self.element_loops = {
                CellType.WHITE_UP: loop_over_simplex_facet(
                    dim, micro_edges_per_macro_edge, self.facet_id
                ),
                second_cell_type[self.facet_id]: loop_over_simplex_facet(
                    dim, micro_edges_per_macro_edge - 1, self.facet_id
                ),
            }

        return Block(
            [
                Block([SourceCodeComment(str(element_type)), loop])
                for element_type, loop in self.element_loops.items()
            ]
        )

    def add_body_to_loop(self, loop, body, element_type):
        """Register a given loop body to innermost loop of the outer loop corresponding to element_type"""
        if element_type not in self.element_loops:
            return
        innermost_loop = get_innermost_loop(self.element_loops[element_type])
        body = Block(body)
        innermost_loop[0].body = body
        body.parent = innermost_loop[0]

    def get_inner_bodies(self, loop_body):
        return [
            inner_loop.body
            for inner_loop in get_innermost_loop(loop_body, return_all_inner=True)
        ]

    def add_preloop_for_loop(self, loops, preloop_stmts, element_type):
        """add given list of statements directly in front of the loop corresponding to element_type."""

        if element_type not in self.element_loops:
            return loops

        preloop_stmts_lhs_subs = {
            stmt.lhs: get_element_replacement(stmt.lhs, element_type)
            for stmt in preloop_stmts
        }

        if not isinstance(loops, Block):
            loops = Block(loops)

        blocks = loops.take_child_nodes()
        new_blocks = []
        for block in blocks:
            idx_to_slice_at = -1
            for idx, stmt in enumerate(block.args):
                if stmt == self.element_loops[element_type]:
                    idx_to_slice_at = idx
                    break

            if idx_to_slice_at == -1:
                new_blocks.append(block)
            else:
                new_stmts = (
                    block.args[0:idx_to_slice_at]
                    + preloop_stmts
                    + block.args[idx_to_slice_at:]
                )
                new_block = Block(new_stmts)
                new_block.fast_subs(preloop_stmts_lhs_subs)
                new_blocks.append(new_block)

        return new_blocks

    def __str__(self):
        return "BOUNDARY"
