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

import pytest
from sympy.core.cache import clear_cache
import logging

from hog.element_geometry import LineElement, TriangleElement, TetrahedronElement
from hog.function_space import LagrangianFunctionSpace
from hog.fem_helpers import fem_function_on_element, trafo_ref_to_affine
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger
from hog.code_generation import code_block_from_element_matrix
from hog.forms import diffusion, linear_form, mass
from hog.forms_facets import (
    diffusion_sip_facet,
    diffusion_sip_rhs_dirichlet,
    _affine_element_vertices,
    trafo_affine_point_to_ref,
    stokes_p0_stabilization,
)
from hog.quadrature import Quadrature, integrate_exact_over_reference


@pytest.mark.skip(reason="test is not really meaningful, but keeping code for now")
def test_trafos():

    volume_element_geometry = TriangleElement()
    symbolizer = Symbolizer()

    (
        affine_points_E1,
        affine_points_E2,
        affine_points_I,
        _,
        _,
        outward_normal,
    ) = _affine_element_vertices(volume_element_geometry, symbolizer)

    function_space_trial = LagrangianFunctionSpace(1, symbolizer)

    phi = function_space_trial.shape(TriangleElement())[0]

    print("phi")
    print(phi)

    import sympy as sp

    affine_p = trafo_ref_to_affine(
        TriangleElement(),
        symbolizer,
        affine_points=affine_points_I
        + [[sp.Symbol("p_affine_8_0"), sp.Symbol("p_affine_8_1")]],
    )

    print()
    print("affine space")
    print(affine_p)
    # print(affine_p.subs("x_ref_0", 0.5).subs("x_ref_1", 0))

    p = trafo_affine_point_to_ref(
        TriangleElement(),
        symbolizer,
        affine_eval_point=affine_p,
        affine_element_points=affine_points_E1,
    )

    print()
    print("elem space")
    print(p)

    p = p.subs("p_affine_8_0", "p_affine_0_0")
    p = p.subs("p_affine_8_1", "p_affine_0_1")

    p = p.subs("p_affine_6_0", "p_affine_1_0")
    p = p.subs("p_affine_6_1", "p_affine_1_1")

    p = p.subs("p_affine_7_0", "p_affine_2_0")
    p = p.subs("p_affine_7_1", "p_affine_2_1")

    p = p.subs("p_affine_0_0", 0)
    p = p.subs("p_affine_0_1", 0)

    p = p.subs("p_affine_1_0", 1)
    p = p.subs("p_affine_1_1", 0)

    p = p.subs("p_affine_2_0", 0)
    p = p.subs("p_affine_2_1", 1)

    phi = phi.subs("x_ref_0", "x_ref_phi_0")
    phi = phi.subs("x_ref_1", "x_ref_phi_1")

    print()
    print("new phi")
    print(phi)

    phi = phi.subs(zip(["x_ref_phi_0", "x_ref_phi_1"], p))

    phi = phi.subs("x_ref_1", 0)

    p.simplify()

    print()
    print("p")
    print(p)
    print()
    print("phi(p)")
    print(phi)


@pytest.mark.skip(reason="test is not really meaningful, but keeping code for now")
def test_line_integration():

    symbolizer = Symbolizer()
    x, y = symbolizer.ref_coords_as_list(2)

    # print(integrate_exact_over_reference(x ** 2, TriangleElement(), symbolizer))
    # print(integrate_exact_over_reference(y ** 2, TriangleElement(), symbolizer))
    # print(
    #     integrate_exact_over_reference((1 - x - y) ** 2, TriangleElement(), symbolizer)
    # )
    # print(
    #     integrate_exact_over_reference(x * (1 - x - y), TriangleElement(), symbolizer)
    # )
    # print(integrate_exact_over_reference(x * y, TriangleElement(), symbolizer))
    #
    # print("### laplace form line ###")
    facet_geometry = TriangleElement()
    volume_geometry = TetrahedronElement()
    degree = 1

    quad = Quadrature(degree, facet_geometry, tolerance=1e-13)

    function_space_trial = LagrangianFunctionSpace(degree, symbolizer)
    function_space_test = LagrangianFunctionSpace(degree, symbolizer)

    interface_type = "inner"

    # print(interface_type)

    if True:
        dform = diffusion_sip_facet(
            interface_type,
            function_space_test,
            function_space_trial,
            volume_geometry,
            quad,
            symbolizer,
        )

        # dform = stokes_p0_stabilization(
        #     interface_type,
        #     function_space_test,
        #     function_space_trial,
        #     volume_geometry,
        #     quad,
        #     symbolizer,
        # )

    else:
        dform = diffusion_sip_rhs_dirichlet(
            function_space_test,
            volume_geometry,
            quad,
            symbolizer,
        )

    # print(dform)

    code_block, function_defs, members = code_block_from_element_matrix(
        dform, quad, volume_geometry, symbolizer
    )

    members = list(set(members))

    print(code_block.to_code())
    print()
    for fd in function_defs:
        print(fd.declaration())
        print(fd.implementation())

    print()
    for m in members:
        print(m)


@pytest.mark.skip(reason="test is not really meaningful, but keeping code for now")
def test_polynomial_evaluation():
    clear_cache()

    TimedLogger.set_log_level(logging.DEBUG)

    symbolizer = Symbolizer()

    geometry = TetrahedronElement()

    degree = 1

    function_space = LagrangianFunctionSpace(degree, symbolizer)

    evaluation = evaluate_fem_function_on_element(
        function_space, geometry, symbolizer, "reference"
    )
    print("### evaluation ###")
    print(evaluation)
    print()

    code_block, function_defs, members = code_block_from_element_matrix(
        evaluation, None, geometry, symbolizer
    )

    print("// body")
    print(code_block.to_code())

    print()
    print("### linear form ###")
    quad = Quadrature(4, geometry)
    lform = linear_form(function_space, function_space, geometry, quad, symbolizer)

    code_block, function_defs, members = code_block_from_element_matrix(
        lform, quad, geometry, symbolizer
    )

    members = list(set(members))

    print(code_block.to_code())
    print()
    for fd in function_defs:
        print(fd.declaration())
        print(fd.implementation())

    print()
    for m in members:
        print(m)

    print()
    print("### mass form ###")

    quad = Quadrature("exact", geometry)
    mform = mass(function_space, function_space, geometry, quad, symbolizer)

    code_block, function_defs, members = code_block_from_element_matrix(
        mform, quad, geometry, symbolizer
    )

    members = list(set(members))

    print(code_block.to_code())
    print()
    for fd in function_defs:
        print(fd.declaration())
        print(fd.implementation())

    print()
    for m in members:
        print(m)

    print("### laplace form volume ###")

    quad = Quadrature(2, geometry)
    dform = diffusion(function_space, function_space, geometry, quad, symbolizer)

    code_block, function_defs, members = code_block_from_element_matrix(
        dform, quad, geometry, symbolizer
    )

    members = list(set(members))

    print(code_block.to_code())
    print()
    for fd in function_defs:
        print(fd.declaration())
        print(fd.implementation())

    print()
    for m in members:
        print(m)


if __name__ == "__main__":
    test_polynomial_evaluation()
    # test_line_integration()
    # test_trafos()
