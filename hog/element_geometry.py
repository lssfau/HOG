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

from hog.exception import HOGException


class ElementGeometry:
    def __init__(self, dimensions: int, num_vertices: int, space_dimension: int):
        if space_dimension < dimensions:
            raise HOGException(
                "The space dimension should be larger or equal to the dimension of the geometry."
            )

        self.dimensions = dimensions
        self.num_vertices = num_vertices
        self.space_dimension = space_dimension

    def __str__(self):
        return f"ElementGeometry(dim: {self.dimensions}, vertices: {self.num_vertices}), space dim: {self.space_dimension}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.dimensions, self.num_vertices, self.space_dimension))

    def __eq__(self, other):
        if isinstance(other, ElementGeometry):
            return (
                self.dimensions == other.dimensions
                and self.num_vertices == other.num_vertices
                and self.space_dimension == other.space_dimension
            )
        return False


class LineElement(ElementGeometry):
    def __init__(self, space_dimension: int = 1):
        super().__init__(1, 2, space_dimension=space_dimension)

    def __str__(self):
        return f"line, dim: 1, vertices: 2, spacedim: {self.space_dimension}"

    def __repr__(self):
        return str(self)


class TriangleElement(ElementGeometry):
    def __init__(self, space_dimension: int = 2):
        super().__init__(2, 3, space_dimension=space_dimension)

    def __str__(self):
        return f"triangle, dim: 2, vertices: 3, spacedim: {self.space_dimension}"

    def __repr__(self):
        return str(self)


class TetrahedronElement(ElementGeometry):
    def __init__(self, space_dimension: int = 3):
        super().__init__(3, 4, space_dimension=space_dimension)

    def __str__(self):
        return f"tetrahedron, dim: 3, vertices: 4, spacedim: {self.space_dimension}"

    def __repr__(self):
        return str(self)
