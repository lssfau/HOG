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

from sympy.core.cache import clear_cache
import logging

from hog.blending import IdentityMap, ExternalMap
from hog.element_geometry import TriangleElement, TetrahedronElement
from hog.forms import diffusion
from hog.function_space import LagrangianFunctionSpace
from hog.hyteg_form_template import HyTeGForm, HyTeGFormClass, HyTeGIntegrator
from hog.quadrature import Quadrature, select_quadrule
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger


def test_diffusion_p1_affine():
    """Simple integration test (pun intended) to check if the pipeline has any major issues.

    Generates (and prints) a HyTeG form for the 2D and 3D affine diffusion for P1 finite elements.
    """

    clear_cache()

    TimedLogger.set_log_level(logging.DEBUG)

    symbolizer = Symbolizer()

    geometries = [TriangleElement(), TetrahedronElement()]
    schemes = {TriangleElement() : 2, TetrahedronElement() : 2 }
    blending = IdentityMap()

    class_name = f"P1DiffusionAffine"

    form_codes = []

    for geometry in geometries:
        trial = LagrangianFunctionSpace(1, symbolizer)
        test = LagrangianFunctionSpace(1, symbolizer)
        quad = Quadrature(select_quadrule(schemes[geometry], geometry), geometry)

        mat = diffusion(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
        ).integrate(quad, symbolizer)
        form_codes.append(
            HyTeGIntegrator(
                class_name, mat, geometry, quad, symbolizer, integrate_rows=[0, 1]
            )
        )

    form_class = HyTeGFormClass(class_name, trial, test, form_codes)
    form_hyteg = HyTeGForm(class_name, trial, test, [form_class])
    print(form_hyteg.to_code())
    print(form_hyteg.to_code(header=False))


def test_diffusion_p2_blending_2D():
    """Simple integration test (pun intended) to check if the pipeline has any major issues.

    Generates (and prints) a HyTeG form for the 2D diffusion with blending for P2 finite elements.
    """

    clear_cache()

    TimedLogger.set_log_level(logging.DEBUG)

    symbolizer = Symbolizer()

    geometry = TriangleElement()
    blending = ExternalMap()

    class_name = f"P2DiffusionBlending2D"

    form_codes = []

    trial = LagrangianFunctionSpace(2, symbolizer)
    test = LagrangianFunctionSpace(2, symbolizer)
    schemes = {TriangleElement(): 4, TetrahedronElement(): 4}

    quad = Quadrature(select_quadrule(schemes[geometry], geometry), geometry)

    mat = diffusion(
        trial,
        test,
        geometry,
        symbolizer,
        blending=blending,
    ).integrate(quad, symbolizer)
    form_codes.append(
        HyTeGIntegrator(
            class_name, mat, geometry, quad, symbolizer, integrate_rows=[0, 1]
        )
    )

    form_class = HyTeGFormClass(class_name, trial, test, form_codes)
    form_hyteg = HyTeGForm(class_name, trial, test, [form_class])
    print(form_hyteg.to_code())
    print(form_hyteg.to_code(header=False))
