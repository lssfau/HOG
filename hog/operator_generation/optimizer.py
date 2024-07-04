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

from copy import deepcopy
import enum
import logging
import sympy as sp
from typing import Dict, Iterable, List, Set

from pystencils import TypedSymbol
from pystencils.astnodes import (
    Block,
    KernelFunction,
    LoopOverCoordinate,
    SympyAssignment,
    ResolvedFieldAccess,
)
from pystencils.cpu.vectorization import vectorize
from pystencils.transformations import (
    cut_loop,
    move_constants_before_loop,
    simplify_conditionals,
)

from hog.ast import Operations, count_operations
import hog.cse
from hog.exception import HOGException
from hog.logger import TimedLogger
from hog.operator_generation.loop_strategies import (
    LoopStrategy,
    CUBES,
    SAWTOOTH,
    FUSEDROWS,
)
from hog.operator_generation.pystencils_extensions import (
    get_innermost_loop,
)


class Opts(enum.Enum):
    POLYCSE = "_polycse"  # use polynomial CSE instead of tree based
    VECTORIZE = "_vect"  # vectorize the element loop: handle 4 elements simultaneously by avx registers
    VECTORIZE512 = "_vect512"  # same with 8 elements
    MOVECONSTANTS = (
        "_const"  # move loop-counter-independent expressions further outside loops
    )
    QUADLOOPS = "_quadloops"  # use loops to sum up the evaluations of the quad rule instead of unrolling it
    ELIMTMPS = "_elimtmps"  # eliminate all temporaries that substitute less than x flops and dont contain a complicated flop like div
    REORDER = "_reorder"  # reorder the kernel stmts such that an access to a tmp is closer to its definition (reduce reuse distance)
    TABULATE = "_tab"  # hoist quad-dependent assignments


opts_arg_mapping = {
    "POLYCSE": Opts.POLYCSE,
    "VECTORIZE": Opts.VECTORIZE,
    "VECTORIZE512": Opts.VECTORIZE512,
    "MOVECONSTANTS": Opts.MOVECONSTANTS,
    "QUADLOOPS": Opts.QUADLOOPS,
    "ELIMTMPS": Opts.ELIMTMPS,
    "REORDER": Opts.REORDER,
    "TABULATE": Opts.TABULATE,
}


def ordered_opts_suffix(loop_strategy: LoopStrategy, opts: Iterable[Opts]) -> str:
    # introduces a specific ordering of optimizations in the name
    opts_suffix = ""
    if isinstance(loop_strategy, CUBES):
        opts_suffix += "_cubes"
    if isinstance(loop_strategy, FUSEDROWS):
        opts_suffix += "_fused_rows"
    if Opts.MOVECONSTANTS in opts:
        opts_suffix += Opts.MOVECONSTANTS.value
    if Opts.VECTORIZE in opts:
        opts_suffix += Opts.VECTORIZE.value
    if Opts.VECTORIZE512 in opts:
        opts_suffix += Opts.VECTORIZE512.value
    if Opts.QUADLOOPS in opts:
        opts_suffix += "_fused_quadloops"
    if Opts.REORDER in opts:
        opts_suffix += Opts.REORDER.value
    if Opts.ELIMTMPS in opts:
        opts_suffix += Opts.ELIMTMPS.value
    if Opts.TABULATE in opts:
        opts_suffix += "_tab"
    if Opts.POLYCSE in opts:
        opts_suffix += Opts.POLYCSE.value
    return opts_suffix


class Optimizer:
    def __init__(self, opts: Set[Opts]):
        self._opts = set(opts)

    def __getitem__(self, opt):
        return opt in self._opts

    def check_opts_validity(self) -> None:
        """Checks if the desired optimizations are valid."""

        if Opts.VECTORIZE512 in self._opts and not Opts.VECTORIZE in self._opts:
            raise HOGException("Optimization VECTORIZE512 requires VECTORIZE.")

        if Opts.TABULATE in self._opts and not Opts.QUADLOOPS in self._opts:
            raise HOGException("Optimization TABULATE requires QUADLOOPS.")

        if Opts.REORDER in self._opts and Opts.TABULATE in self._opts:
            raise HOGException(
                "Optimization REORDER currently not supported in combination with TABULATE."
            )

        if Opts.ELIMTMPS in self._opts or Opts.REORDER in self._opts:
            raise HOGException(
                "Optimizations ELIMTMPS and REORDER are currently mostly disfunctional."
            )

    def cse_impl(self) -> hog.cse.CseImplementation:
        if self[Opts.POLYCSE]:
            return hog.cse.CseImplementation.POLYCSE
        else:
            return hog.cse.CseImplementation.SYMPY

    def apply_to_kernel(
        self, kernel_function: KernelFunction, dim: int, loop_strategy: LoopStrategy
    ) -> Dict[str, KernelFunction]:
        """Applies optimizations to the given kernel function.

        The optimizations are applied to the passed kernel function which is
        modified in place. Additionally, optimizations which require hardware
        support (i.e. vectorization) are applied in isolation. This method
        returns a dictionary mapping platform features to kernel functions. The
        kernel functions in this dictionary only require the hardware feature
        specified by the dictionary key.
        For example, when vectorization (avx) is requested the returned dict looks like
            { "noarch": (optimized kernel function *without vectorization*)
            , "avx"   : (vectorized kernel function)
            }.
        The "avx" kernel function is the same Python object as the argument to
        this function.
        """

        # We cut the innermost loop so that the conditionals can be simplified/removed.
        # In particular, we split the innermost loop at the last (2D) and next to last (3D) iterations since we know
        # for the elementwise iteration that those are the points where the conditionals can be safely evaluated to
        # True or False.
        if not isinstance(kernel_function.body, Block):
            raise HOGException("Expecting a block around the kernel loops. ")

        if isinstance(loop_strategy, CUBES):
            with TimedLogger("cutting loops", logging.DEBUG):
                loops = [
                    loop
                    for loop in kernel_function.body.args
                    if isinstance(loop, LoopOverCoordinate)
                ]
                assert len(loops) == 1, f"Expecting a single loop here, not {loops}"
                loop = loops[0]

                innermost_loop = get_innermost_loop(loop, return_all_inner=True)[0]
                if dim == 2:
                    new_loops = cut_loop(
                        innermost_loop,
                        [innermost_loop.stop - 1],
                        replace_loops_with_length_one=False,
                    )
                    loop.body = new_loops
                    new_loops.parent = loop

                elif dim == 3:
                    new_loops = cut_loop(
                        innermost_loop,
                        [innermost_loop.stop - 2, innermost_loop.stop - 1],
                        with_conditional=True,
                        replace_loops_with_length_one=False,
                    )
                    innermost_loop.parent.body = new_loops
                    new_loops.parent = innermost_loop.parent

            with TimedLogger("simplifying conditionals", logging.DEBUG):
                simplify_conditionals(loop, loop_counter_simplification=True)

        if self[Opts.MOVECONSTANTS]:
            with TimedLogger("moving constants out of loop", logging.DEBUG):
                # This has to be done twice because sometimes constants are not moved completely to the surrounding block but
                # only to an outer loop. Then an additional move_constants moves them entirely out of the loop.
                ignore_pred = lambda x: (
                    isinstance(x, TypedSymbol) and x.name.startswith("q_acc")
                ) or (
                    isinstance(x, ResolvedFieldAccess)
                    and (
                        x.field.name.startswith("q_w") or x.field.name.startswith("q_p")
                    )
                )

                symbols_generator = sp.numbered_symbols("tmp_moved_constant_")

                move_constants_before_loop(
                    kernel_function.body,
                    lhs_symbol_ignore_predicate=ignore_pred,
                    tmp_symbols_generator=symbols_generator,
                )
                move_constants_before_loop(
                    kernel_function.body,
                    lhs_symbol_ignore_predicate=ignore_pred,
                    tmp_symbols_generator=symbols_generator,
                )

        self.elimtmps_and_reorder(kernel_function)

        platform_dependent_kernels = {"noarch": deepcopy(kernel_function)}

        if self[Opts.VECTORIZE]:
            instruction_set = "avx512" if self[Opts.VECTORIZE512] else "avx"

            with TimedLogger("vectorizing loops", logging.DEBUG):
                vectorize(
                    kernel_function,
                    instruction_set=instruction_set,
                    replace_loops_with_length_one=False,
                )
            platform_dependent_kernels[instruction_set] = kernel_function

        return platform_dependent_kernels

    def reduce_reuse_dist(self, stmts: List[SympyAssignment]) -> List[SympyAssignment]:
        # collect the accessed symbols of each statement (accessed on their rhs) in a dict
        accessed_symbols = {}
        for stmt in stmts:
            accessed_symbols[stmt.lhs.name] = [
                s.name for s in stmt.rhs.atoms(TypedSymbol, sp.Symbol)
            ]

        ctr = 0
        # we want to find a position close to the point where all its accessed symbols/dependencies are defined
        while ctr < len(stmts) - 1:
            # take the current statement
            stmt = stmts.pop(ctr)
            ctr2 = 0
            # move upwards in the statement list
            while ctr - ctr2 >= 0:
                #  print("ctr, ctr2=(" + str(ctr) + "," + str(ctr2) + ")")
                cur_pos = ctr - ctr2
                stmt2 = stmts[cur_pos]
                # if the current stmt2 we compare with defines a symbol that is accessed by stmt we stop and insert at that position
                if stmt2.lhs.name in accessed_symbols[stmt.lhs.name]:
                    stmts.insert(cur_pos + 1, stmt)
                    break
                # if we are at the beginning of the list stmt is not depending on any definition in the kernel (except for the loop counter)
                # and can be put at its beginning
                if ctr - ctr2 == 0:
                    stmts.insert(0, stmt)
                ctr2 += 1

            ctr += 1
        return stmts

    def elimtmps_and_reorder(self, kernel_function: KernelFunction) -> None:
        main_body = kernel_function.body

        # eliminate temporaries with only a small number of flops
        if self[Opts.ELIMTMPS]:
            with TimedLogger(
                "inlining tmps with small rhs (< 4 mul/add)", logging.DEBUG
            ):

                def eliminate(stmt, stmts, ctr):
                    tmp_symbol = deepcopy(stmt.lhs)
                    for stmt_inner in stmts[ctr + 1 :]:
                        if (
                            # TODO do not simply ignore other AST nodes
                            isinstance(stmt_inner, SympyAssignment)
                            and stmt_inner.lhs.name != tmp_symbol.name
                        ):
                            stmt_inner.subs({tmp_symbol: stmt.rhs})

                ctr = 0
                stmts_outer = main_body.take_child_nodes()
                for stmt_outer in stmts_outer:
                    if isinstance(stmt_outer, SympyAssignment):
                        ops = Operations()
                        count_operations(stmt_outer.rhs, ops)
                        flops = vars(ops)
                        if (
                            flops["adds"] + flops["muls"] < 4
                            and flops["divs"] == 0
                            and flops["pows"] == 0
                            and flops["abs"] == 0
                        ):
                            eliminate(stmt_outer, stmts_outer, ctr)
                        else:
                            main_body.append(stmt_outer)
                    else:
                        # TODO recurse into blocks, loop bodies, etc.
                        main_body.append(stmt_outer)
                    ctr += 1

        if self[Opts.REORDER]:
            with TimedLogger(
                "reordering stmts to reduce locality of access for tmps", logging.DEBUG
            ):
                # TODO recurse into blocks, loop bodies, etc.
                main_body = Block(self.reduce_reuse_dist(main_body.args))

        kernel_function.body = main_body
