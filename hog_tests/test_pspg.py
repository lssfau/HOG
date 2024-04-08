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

from hog.blending import IdentityMap
from hog.element_geometry import TriangleElement, TetrahedronElement
from hog.forms import pspg
from hog.function_space import LagrangianFunctionSpace
from hog.hyteg_form_template import HyTeGForm, HyTeGFormClass, HyTeGIntegrator
from hog.quadrature import Quadrature
from hog.symbolizer import Symbolizer
from hog.logger import TimedLogger


def test_pspg_p1_affine():
    """Tests PSPG form as it requires std::pow with rational exponent."""

    clear_cache()

    TimedLogger.set_log_level(logging.DEBUG)

    symbolizer = Symbolizer()

    geometries = [TriangleElement()]
    blending = IdentityMap()

    class_name = f"P1PSPGAffine"

    form_codes = []

    for geometry in geometries:
        trial = LagrangianFunctionSpace(1, symbolizer)
        test = LagrangianFunctionSpace(1, symbolizer)
        quad = Quadrature("exact", geometry)

        form = pspg(
            trial,
            test,
            geometry,
            quad,
            symbolizer,
            blending=blending,
        )
        form_codes.append(
            HyTeGIntegrator(
                class_name, form.integrand, geometry, quad, symbolizer, integrate_rows=[0]
            )
        )

    form_class = HyTeGFormClass(class_name, trial, test, form_codes)
    form_hyteg = HyTeGForm(class_name, trial, test, [form_class])
    print(form_hyteg.to_code())
    print(form_hyteg.to_code(header=False))
