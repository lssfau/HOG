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

from hog.forms_new import process_integrand
from hog.symbolizer import Symbolizer
from hog.element_geometry import TriangleElement
from hog.blending import AnnulusMap
from hog.function_space import LagrangianFunctionSpace
from hog.math_helpers import dot

import hog.forms


def test_my_form() -> None:

    symbolizer = Symbolizer()

    trial = LagrangianFunctionSpace(2, symbolizer)
    test = LagrangianFunctionSpace(2, symbolizer)

    geometry = TriangleElement()
    blending = AnnulusMap()

    # This is what we have right now.
    old_form_diffusion = hog.forms.diffusion(
        trial, test, geometry, symbolizer, blending=blending
    )
    old_form_mass = hog.forms.mass(trial, test, geometry, symbolizer, blending=blending)

    # Proposal:
    #
    # Have short (one-liner) functions that only describe the integrand.
    # The arguments are automatically handed in from a populated IntegrandSymbols object.
    # They may already be tabulated.
    # Also, we can apply the chain rule already depending on the function spaces, such that writing forms becomes much
    # easier.
    def my_new_diffusion(*, grad_u, grad_v, dx, **kwargs):
        return dot(grad_u, grad_v) * dx

    # We can add all arguments here that are given by the IntegrandSymbols object!
    def my_new_mass(*, u, v, dx, **kwargs):
        return u * v * dx

    integrands = [my_new_diffusion, my_new_mass]

    # Where the magic happens.
    # The only variable argument is the form!
    new_forms = [
        process_integrand(
            integrand,
            trial,
            test,
            geometry,
            symbolizer,
            blending,
            is_symmetric=True,
        )
        for integrand in integrands
    ]

    # Prints "True" on my machine :)
    print(new_forms[0].integrand == old_form_diffusion.integrand)

    # Mass is not tabulated correctly, but you get the point.


if __name__ == "__main__":
    test_my_form()
