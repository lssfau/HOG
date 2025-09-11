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


# Collects common imports for forming integrand and operator recipes.
import sympy as sp
from hog.math_helpers import dot, abs, det, inv, double_contraction, diameter
from hog.approximation_functions import delta_supg, exp_approx, sin_approx, cos_approx, simple_viscosity_profile

__all__ = ["sp", "dot", "abs", "det", "inv", "double_contraction", "diameter", "delta_supg", "exp_approx", "sin_approx", "cos_approx", "simple_viscosity_profile"]
