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

import argparse
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from functools import partial
import logging
import os
import re
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import sympy as sp
from sympy.core.cache import clear_cache
from tabulate import tabulate

from hog.blending import GeometryMap, IdentityMap, AnnulusMap, IcosahedralShellMap
from hog.cse import CseImplementation
from hog.element_geometry import ElementGeometry, TriangleElement, TetrahedronElement
from hog.exception import HOGException
from hog.forms import (
    diffusion,
    divergence,
    div_k_grad,
    shear_heating,
    epsilon,
    full_stokes,
    nonlinear_diffusion,
    nonlinear_diffusion_newton_galerkin,
    supg_diffusion,
)
from hog.forms_vectorial import curl_curl, curl_curl_plus_mass, mass_n1e1
from hog.function_space import (
    FunctionSpace,
    LagrangianFunctionSpace,
    N1E1Space,
    TensorialVectorFunctionSpace,
)
from hog.logger import get_logger, TimedLogger
from hog.operator_generation.kernel_types import (
    ApplyWrapper,
    AssembleWrapper,
    AssembleDiagonalWrapper,
    KernelWrapperType,
)
from hog.operator_generation.loop_strategies import (
    LoopStrategy,
    SAWTOOTH,
    CUBES,
    FUSEDROWS,
)
from hog.operator_generation.operators import (
    HyTeGElementwiseOperator,
    MacroIntegrationDomain,
)

from operator import itemgetter
from hog.operator_generation.optimizer import (
    Opts,
    opts_arg_mapping,
    ordered_opts_suffix,
)
from hog.quadrature import Quadrature
from hog.symbolizer import Symbolizer
from generate_all_hyteg_forms import valid_base_dir
from hog.operator_generation.types import parse_argument_type, HOGType, HOGPrecision
import quadpy
from hog.quadrature.quadrature import select_quadrule

ALL_GEOMETRY_MAPS = [
    IdentityMap(),
    AnnulusMap(),
    IcosahedralShellMap(),
]


def parse_arguments():
    parser = ArgumentParser(
        description="Generates a collection of elementwise operators."
    )

    list_or_gen = parser.add_mutually_exclusive_group(required=True)
    list_or_gen.add_argument(
        "-l",
        "--list-operators",
        action="store_true",
        help="List all available operators.",
    )
    list_or_gen.add_argument(
        "--list-quadratures",
        nargs=1,
        help="List all available quadrature rules for a given geometry (t1: Line, t2: Triangle, t3: Tetrahedron)",
    )

    list_or_gen.add_argument(
        "--hyteg",
        help=(
            "Path to the HyTeG repository. The generated code is written to the correct directory in the source tree."
        ),
    )
    list_or_gen.add_argument(
        "--output",
        help=("Path to output directory."),
    )

    parser.add_argument(
        "-s",
        "--space-mapping",
        default=".*?",
        help="Filter by function space, e.g. 'P1' or 'P1toP2' (regex).",
        # TODO add choices list and type=string.lower
    )

    parser.add_argument(
        "-f",
        "--form",
        default=".*?",
        help="Filter by form, e.g. 'CurlCurl' or 'Diffusion' (regex).",
        # TODO add choices list and type=string.lower
    )

    parser.add_argument(
        "-o",
        "--opts",
        default="",
        nargs="+",
        help=f"""
    Enter a space separated list of optimizations. 
    Not all combinations are currently supported. Available optimizations are: {Opts.__members__.keys()}""",
        # TODO add choices list and type=string.lower
    )
    # TODO explanations for optimizations e.g. CUTLOOPS is not straight-forward

    parser.add_argument(
        "--loop-strategy",
        default="SAWTOOTH",
        help="Enter a loop strategy for the operator. Available strategies are SAWTOOTH, CUBES.",
        # TODO add choices list and type=string.lower
    )

    quad_rule_or_degree = parser.add_mutually_exclusive_group(required=False)
    quad_rule_or_degree.add_argument(
        "--quad-rule",
        type=str,
        nargs=2,
        default=None,
        help="""Supply two rules by their identifier (leftmost column when calling with --list-quadratures),
                the first will be used for triangles and the second for tetrahedra.""",
    )
    quad_rule_or_degree.add_argument(
        "--quad-degree",
        type=int,
        nargs=2,
        default=[-1, -1],
        help="""Supply two polynomial degrees (as integers) up to which the quadrature should be exact. 
                The generator will chose the rule with minimal points that integrates the specified 
                degree exactly. The first integer will be used for triangles and the second for tetrahedra.""",
    )

    prec_or_mpfr = parser.add_mutually_exclusive_group(required=False)
    prec_or_mpfr.add_argument(
        "-p",
        "--precision",
        type=str.lower,
        help=(
            "Defines the precision in which the kernels are calculated; "
            "If not specified we choose fp64(double) precision"
        ),
        choices=list(choice.value for choice in HOGPrecision),
        default="fp64",
    )

    parser.add_argument(
        "-b",
        "--blending",
        default=str(IdentityMap()),
        help=f"Enter a blending type for the operator. Note some geometry maps only work in 2D or 3D respectively. "
        f"Available geometry maps: {[str(m) for m in ALL_GEOMETRY_MAPS]}",
        # TODO add choices list and type=string.lower
    )

    parser.add_argument(
        "--name",
        help=f"Specify the name of the generated C++ class.",
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[2, 3],
        help=f"Domain dimensions for which operators shall be generated.",
    )

    logging_group = parser.add_mutually_exclusive_group(required=False)
    logging_group.add_argument(
        "-v",
        "--verbose",
        type=int,
        help="specify verbosity level (0 = silent, 1 = info, 2 = debug)",
        choices=[0, 1, 2],
    )
    logging_group.add_argument(
        "--loglevel",
        type=str.lower,
        help=(
            "Defines which messages are shown and which not; "
            "If not specified 'warning' is chosen."
        ),
        choices=["debug", "info", "warning", "error", "critical"],
    )

    clang_format_group = parser.add_mutually_exclusive_group(required=False)
    clang_format_group.add_argument(
        "--no-clang-format",
        dest="clang_format",
        action="store_false",
        help="Do not apply clang-format on generated files.",
    )

    clang_format_group.add_argument(
        "--clang-format-binary",
        default="clang-format",
        help=f"Allows to specify the name of the clang-format binary and/or optionally its full path."
        f" By default 'clang-format' will be used.",
    )

    args = parser.parse_args()
    return parser, args


strategy_args_mapping = {
    "CUBES": CUBES(),
    "SAWTOOTH": SAWTOOTH(),
    "FUSEDROWS": FUSEDROWS(),
}


def main():
    clear_cache()

    parser, args = parse_arguments()

    # Adapt clang_format_binary according to the value of clang_format
    if "clang_format" in vars(args):
        if not args.clang_format:
            args.clang_format_binary = None

    if args.verbose:
        if args.verbose == 0:
            log_level = logging.CRITICAL
        elif args.verbose == 1:
            log_level = logging.INFO
        elif args.verbose == 2:
            log_level = logging.DEBUG
        else:
            raise HOGException(f"Invalid verbosity level {args.verbose}.")
    elif args.loglevel:
        if args.loglevel == "debug":
            log_level = logging.DEBUG
        elif args.loglevel == "info":
            log_level = logging.INFO
        elif args.loglevel == "warning":
            log_level = logging.WARNING
        elif args.loglevel == "error":
            log_level = logging.ERROR
        elif args.loglevel == "critical":
            log_level = logging.CRITICAL
        else:
            raise HOGException(f"Invalid log level {args.loglevel}.")
    else:
        # The default log level is chosen here
        log_level = logging.DEBUG

    logger = get_logger(log_level)
    TimedLogger.set_log_level(log_level)

    logger.info(f"Running {__file__} with arguments: {' '.join(sys.argv[1:])}")

    symbolizer = Symbolizer()

    if args.list_quadratures:
        geometry = args.list_quadratures[0]
        logger.debug(f"Listing quadrature rules for geometry: {geometry}.")
        if geometry == "t3":
            schemes = quadpy.t3.schemes
        elif geometry == "t2":
            schemes = quadpy.t2.schemes
        else:
            raise HOGException(f"Unexpected geometry: {geometry}")

        all_schemes = []
        for key, s in schemes.items():
            try:
                scheme = s()
                entry = {
                    "key": key,
                    "name": scheme.name,
                    "points": scheme.points.shape[1],
                    "degree": scheme.degree,
                    "test_tolerance": scheme.test_tolerance,
                }
                all_schemes.append(entry)
            except TypeError as e:
                # Some schemes seem to require parameters, we simply exclude these with this litte try-except hack
                pass

        all_schemes = sorted(all_schemes, key=itemgetter("degree", "points"))
        table = tabulate(all_schemes, headers="keys", tablefmt="grid")
        print(table)
        exit()

    blending = None
    for m in ALL_GEOMETRY_MAPS:
        if str(m) == args.blending:
            blending = m
    if blending is None:
        raise HOGException("Invalid geometry map.")
    logger.debug(f"Blending: {blending}")

    type_descriptor = parse_argument_type(args)

    logger.debug(f"Data type name is {type_descriptor.pystencils_type}.")

    opts = {opts_arg_mapping[opt] for opt in args.opts}
    loop_strategy = strategy_args_mapping[args.loop_strategy]
    re_form = re.compile(args.form)

    if (
        not args.list_operators
        and args.quad_degree == [-1, -1]
        and args.quad_rule is None
    ):
        parser.error("Either --quad-degree or --quad-rule are required.")
    quad = {
        geometry: info
        for geometry, info in zip(
            [TriangleElement(), TetrahedronElement()],
            args.quad_degree if args.quad_rule is None else args.quad_rule,
        )
    }

    enabled_geometries: Set[TriangleElement | TetrahedronElement] = set()
    if 2 in args.dimensions:
        enabled_geometries.add(TriangleElement())
    if 3 in args.dimensions:
        enabled_geometries.add(TetrahedronElement())

    operators = all_operators(
        symbolizer,
        [(opts, loop_strategy, ordered_opts_suffix(loop_strategy, opts))],
        type_descriptor,
        blending,
        geometries=enabled_geometries,
    )

    filtered_operators = list(
        filter(
            lambda o: re.fullmatch(args.space_mapping, o.mapping, re.IGNORECASE)
            and re.fullmatch(args.form, o.name, re.IGNORECASE),
            operators,
        )
    )

    filtered_operators = [f for f in filtered_operators if f.geometries]

    if len(filtered_operators) == 0:
        raise HOGException(
            f"Form {args.form} does not match any available forms. Run --list for a list."
        )

    if args.list_operators:
        print(
            tabulate(
                sorted(
                    [
                        (
                            o.mapping,
                            o.name,
                            ", ".join(
                                str(geo.dimensions) + "D" for geo in o.geometries
                            ),
                            len(o.opts),
                        )
                        for o in filtered_operators
                    ]
                ),
                headers=(
                    "Space mapping",
                    "Form",
                    "Dimensions",
                    "Optimizations",
                ),
            )
        )
        exit()

    if args.hyteg:
        args.output = os.path.join(args.hyteg, "src/hyteg/operatorgeneration/generated")
        if not valid_base_dir(args.hyteg):
            raise HOGException(
                "The specified directory does not seem to be the HyTeG directory."
            )

    for operator in filtered_operators:
        for opt, loop, opt_name in operator.opts:
            blending_str = (
                "_" + str(blending) if not isinstance(blending, IdentityMap) else ""
            )
            if args.name:
                name = args.name
            else:
                name = (
                    f"{operator.mapping}Elementwise{operator.name}{blending_str}{opt_name}"
                    f"{'_' + str(type_descriptor) if type_descriptor.add_file_suffix else ''}"
                )
            with TimedLogger(f"Generating {name}", level=logging.INFO):
                generate_elementwise_op(
                    args,
                    symbolizer,
                    operator,
                    name,
                    opt,
                    loop,
                    blending,
                    quad,
                    type_descriptor=type_descriptor,
                )


def all_opts_sympy_cse() -> List[Tuple[Set[Opts], LoopStrategy, str]]:
    return [
        (
            set(
                {
                    Opts.MOVECONSTANTS,
                    Opts.REORDER,
                    Opts.ELIMTMPS,
                    Opts.QUADLOOPS,
                    Opts.VECTORIZE,
                }
            ),
            SAWTOOTH(),
            "_const_vect_fused_quadloops_reordered_elimtmps",
        ),
    ]


def all_opts_poly_cse() -> List[Tuple[Set[Opts], LoopStrategy, str]]:
    return [
        (o | {Opts.POLYCSE}, l, n + "_polycse") for (o, l, n) in all_opts_sympy_cse()
    ]


def all_opts_both_cses() -> List[Tuple[Set[Opts], LoopStrategy, str]]:
    return all_opts_sympy_cse() + all_opts_poly_cse()


@dataclass
class OperatorInfo:
    mapping: str
    name: str
    trial_space: FunctionSpace
    test_space: FunctionSpace
    form: Callable[
        [
            FunctionSpace,
            FunctionSpace,
            ElementGeometry,
            Symbolizer,
            GeometryMap,
        ],
        sp.Matrix,
    ]
    type_descriptor: HOGType
    kernel_types: List[KernelWrapperType] = None  # type: ignore[assignment] # will definitely be initialized in __post_init__
    geometries: Sequence[ElementGeometry] = field(
        default_factory=lambda: [TriangleElement(), TetrahedronElement()]
    )
    opts: List[Tuple[Set[Opts], LoopStrategy, str]] = field(
        default_factory=all_opts_sympy_cse
    )
    blending: GeometryMap = field(default_factory=lambda: IdentityMap())

    def __post_init__(self):
        # Removing geometries that are not supported by blending.
        # I don't like this, but it looks like the collection of operators has to be refactored anyway.
        self.geometries = list(
            set(self.geometries) & set(self.blending.supported_geometries())
        )

        if self.kernel_types is None:
            dims = [g.dimensions for g in self.geometries]
            self.kernel_types = [
                ApplyWrapper(
                    self.test_space,
                    self.trial_space,
                    type_descriptor=self.type_descriptor,
                    dims=dims,
                )
            ]

            all_opts = set().union(*[o for (o, _, _) in self.opts])
            if not ({Opts.VECTORIZE, Opts.VECTORIZE512}.intersection(all_opts)):
                self.kernel_types.append(
                    AssembleWrapper(
                        self.test_space,
                        self.trial_space,
                        type_descriptor=self.type_descriptor,
                        dims=dims,
                    )
                )

            if self.test_space == self.trial_space:
                self.kernel_types.append(
                    AssembleDiagonalWrapper(
                        self.test_space,
                        type_descriptor=self.type_descriptor,
                        dims=dims,
                    )
                )


def all_operators(
    symbolizer: Symbolizer,
    opts: List[Tuple[Set[Opts], LoopStrategy, str]],
    type_descriptor: HOGType,
    blending: GeometryMap,
    geometries: Set[Union[TriangleElement, TetrahedronElement]],
) -> List[OperatorInfo]:
    P1 = LagrangianFunctionSpace(1, symbolizer)
    P1Vector = TensorialVectorFunctionSpace(P1)
    P2 = LagrangianFunctionSpace(2, symbolizer)
    P2Vector = TensorialVectorFunctionSpace(P2)
    N1E1 = N1E1Space(symbolizer)

    two_d = list(geometries & {TriangleElement()})
    three_d = list(geometries & {TetrahedronElement()})

    ops: List[OperatorInfo] = []

    # fmt: off
    # TODO switch to manual specification of opts for now/developement, later use default factory
    ops.append(OperatorInfo(mapping="N1E1", name="CurlCurl", trial_space=N1E1, test_space=N1E1, form=curl_curl,
                            type_descriptor=type_descriptor, geometries=three_d, opts=opts, blending=blending))
    ops.append(OperatorInfo(mapping="N1E1", name="Mass", trial_space=N1E1, test_space=N1E1, form=mass_n1e1,
                            type_descriptor=type_descriptor, geometries=three_d, opts=opts, blending=blending))
    ops.append(OperatorInfo(mapping="N1E1", name="CurlCurlPlusMass", trial_space=N1E1, test_space=N1E1,
                            form=partial(curl_curl_plus_mass, alpha_fem_space=P1, beta_fem_space=P1),
                            type_descriptor=type_descriptor, geometries=three_d, opts=opts, blending=blending))
    ops.append(OperatorInfo(mapping="P1", name="Diffusion", trial_space=P1, test_space=P1, form=diffusion,
                            type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))
    ops.append(OperatorInfo(mapping="P1", name="DivKGrad", trial_space=P1, test_space=P1,
                            form=partial(div_k_grad, coefficient_function_space=P1),
                            type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))

    ops.append(OperatorInfo(mapping="P2", name="Diffusion", trial_space=P2, test_space=P2, form=diffusion,
                            type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))
    ops.append(OperatorInfo(mapping="P2", name="DivKGrad", trial_space=P2, test_space=P2,
                            form=partial(div_k_grad, coefficient_function_space=P2),
                            type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))

    ops.append(OperatorInfo(mapping="P2", name="ShearHeating", trial_space=P2, test_space=P2,
                            form=partial(shear_heating, viscosity_function_space=P2, velocity_function_space=P2),
                            type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))

    ops.append(OperatorInfo(mapping="P1", name="NonlinearDiffusion", trial_space=P1, test_space=P1,
                            form=partial(nonlinear_diffusion, coefficient_function_space=P1),
                            type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))
    ops.append(OperatorInfo(mapping="P1", name="NonlinearDiffusionNewtonGalerkin", trial_space=P1,
                            test_space=P1, form=partial(nonlinear_diffusion_newton_galerkin,
                            coefficient_function_space=P1, onlyNewtonGalerkinPartOfForm=False),
                            type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))

    ops.append(OperatorInfo(mapping="P1Vector", name="Diffusion", trial_space=P1Vector, test_space=P1Vector,
                            form=diffusion, type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))

    ops.append(OperatorInfo(mapping="P2", name="SUPGDiffusion", trial_space=P2, test_space=P2, 
                            form=partial(supg_diffusion, velocity_function_space=P2, diffusivityXdelta_function_space=P2), 
                            type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))

    # fmt: on

    p2vec_epsilon = partial(
        epsilon,
        variable_viscosity=True,
        coefficient_function_space=P2,
    )
    ops.append(
        OperatorInfo(
            mapping=f"P2Vector",
            name=f"Epsilon",
            trial_space=P2Vector,
            test_space=P2Vector,
            form=p2vec_epsilon,
            type_descriptor=type_descriptor,
            geometries=list(geometries),
            opts=opts,
            blending=blending,
        )
    )

    for c in [0, 1, 2]:
        # fmt: off
        if c == 2:
            div_geometries = three_d
        else:
            div_geometries = list(geometries)
        ops.append(OperatorInfo(mapping=f"P2ToP1", name=f"Div_{c}", trial_space=P1, test_space=P2,
                                form=partial(divergence, transpose=False, component_index=c),
                                type_descriptor=type_descriptor, opts=opts, geometries=div_geometries,
                                blending=blending))
        ops.append(OperatorInfo(mapping=f"P1ToP2", name=f"DivT_{c}", trial_space=P2, test_space=P1,
                                form=partial(divergence, transpose=True, component_index=c),
                                type_descriptor=type_descriptor, opts=opts, geometries=div_geometries,
                                blending=blending))
        # fmt: on

    for c in [0, 1]:
        for r in [0, 1]:
            p2_epsilon = partial(
                epsilon,
                variable_viscosity=True,
                coefficient_function_space=P2,
                component_trial=c,
                component_test=r,
            )

            p2_full_stokes = partial(
                full_stokes,
                variable_viscosity=True,
                coefficient_function_space=P2,
                component_trial=c,
                component_test=r,
            )
            # fmt: off
            ops.append(
                OperatorInfo(mapping=f"P2", name=f"Epsilon_{r}_{c}", trial_space=P2, test_space=P2, form=p2_epsilon,
                              type_descriptor=type_descriptor, geometries=list(geometries), opts=opts, blending=blending))
            ops.append(OperatorInfo(mapping=f"P2", name=f"FullStokes_{r}_{c}", trial_space=P2, test_space=P2,
                                    form=p2_full_stokes, type_descriptor=type_descriptor, geometries=list(geometries), opts=opts,
                                    blending=blending))
            # fmt: on
    for c, r in [(0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]:
        p2_epsilon = partial(
            epsilon,
            variable_viscosity=True,
            coefficient_function_space=P2,
            component_trial=c,
            component_test=r,
        )

        p2_full_stokes = partial(
            full_stokes,
            variable_viscosity=True,
            coefficient_function_space=P2,
            component_trial=c,
            component_test=r,
        )
        # fmt: off
        ops.append(
            OperatorInfo(mapping=f"P2", name=f"Epsilon_{r}_{c}", trial_space=P2, test_space=P2, form=p2_epsilon,
                         type_descriptor=type_descriptor, geometries=three_d, opts=opts, blending=blending))
        ops.append(
            OperatorInfo(mapping=f"P2", name=f"FullStokes_{r}_{c}", trial_space=P2, test_space=P2, form=p2_full_stokes,
                          type_descriptor=type_descriptor, geometries=three_d, opts=opts, blending=blending))
        # fmt: on

    # Removing all operators without viable element types (e.g.: some ops only support 2D, but a blending map maybe only
    # supports 3D).
    ops = [o for o in ops if o.geometries]

    return ops


def generate_elementwise_op(
    args: argparse.Namespace,
    symbolizer: Symbolizer,
    op_info: OperatorInfo,
    name: str,
    optimizations: Set[Opts],
    loop_strategy: LoopStrategy,
    blending: GeometryMap,
    quad_info: dict[ElementGeometry, int | str],
    type_descriptor: HOGType,
) -> None:
    """Generates a single operator and writes it to the HyTeG directory."""

    operator = HyTeGElementwiseOperator(
        name,
        symbolizer,
        kernel_wrapper_types=op_info.kernel_types,
        type_descriptor=type_descriptor,
    )

    for geometry in op_info.geometries:
        # Is this necessary? Currently, the decision of the geometry is all over the place.
        if geometry not in blending.supported_geometries():
            continue

        quad = Quadrature(
            select_quadrule(quad_info[geometry], geometry),
            geometry,
        )

        form = op_info.form(
            op_info.test_space,
            op_info.trial_space,
            geometry,
            symbolizer,
            blending=blending,  # type: ignore[call-arg] # kw-args are not supported by Callable
        )

        operator.add_volume_integral(
            name="".join(name.split()),
            volume_geometry=geometry,
            quad=quad,
            blending=blending,
            form=form,
            loop_strategy=loop_strategy,
            optimizations=optimizations,
        )

    dir_path = os.path.join(args.output, op_info.name.split("_")[0])
    operator.generate_class_code(
        dir_path,
        clang_format_binary=args.clang_format_binary,
    )


if __name__ == "__main__":
    main()
