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

from hog.element_geometry import LineElement, TriangleElement, TetrahedronElement
from hog.quadrature import Quadrature, select_quadrule
from hog.exception import HOGException


def test_smoke():
    """Just a brief test to see if the quadrature class does _something_."""

    geometries = [TriangleElement(), TetrahedronElement()] # TODO fix quad for lines

    for geometry in geometries:
        schemes = {TriangleElement(): "exact", TetrahedronElement(): "exact"}

        quad = Quadrature(select_quadrule(schemes[geometry], geometry), geometry)
        print("points", quad.points())
        print("weights", quad.weights())

        for deg in range(1, 10):
            schemes = {TriangleElement(): deg, TetrahedronElement(): deg}

            quad = Quadrature(select_quadrule(schemes[geometry], geometry), geometry)
            points = quad.points()
            weights = quad.weights()
            if not points or not weights:
                raise HOGException("There should be points and weights...")
            print("points", quad.points())
            print("weights", quad.weights())

test_smoke()