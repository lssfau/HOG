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

from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Union
import os
import logging
import itertools
import sympy as sp
from sympy.core.cache import clear_cache
import re
import quadpy

from hog.blending import GeometryMap, IdentityMap, ExternalMap, AnnulusMap
from hog.element_geometry import (
    TriangleElement,
    TetrahedronElement,
    ElementGeometry,
)
from hog.function_space import (
    LagrangianFunctionSpace,
    N1E1Space,
    P2PlusBubbleSpace,
    TrialSpace,
    TestSpace,
)
from hog.forms import (
    mass,
    diffusion,
    div_k_grad,
    epsilon,
    k_mass,
    pspg,
    linear_form,
    divergence,
    gradient,
    full_stokes,
    divdiv,
    supg_diffusion,
)
from hog.manifold_forms import (
    laplace_beltrami,
    manifold_mass,
    manifold_vector_mass,
    manifold_normal_penalty,
    manifold_divergence,
    manifold_vector_divergence,
    manifold_epsilon,
    vector_laplace_beltrami,
)
from hog.forms_vectorial import mass_n1e1, curl_curl
from hog.quadrature import Quadrature, select_quadrule
from hog.exception import HOGException
from hog.logger import TimedLogger
from hog.symbolizer import Symbolizer
from hog.hyteg_form_template import HyTeGForm, HyTeGFormClass, HyTeGIntegrator
from hog.logger import TimedLogger, get_logger


# For a default scalar form we assume that it will be well-defined for 2D and 3D
def is_implemented_for_scalar(row: int, col: int, dim: int) -> bool:
    return True


# For a default vector-to-vector form we assume that its sub-forms will be
# well-defined in 2D only for dimension values 0,1
# Note: dim argument should be 2 or 3, while row and col are 0-based
def is_implemented_for_vector_to_vector(row: int, col: int, dim: int) -> bool:
    if (row == 2 or col == 2) and dim == 2:
        return False
    else:
        return True


# For a default vector-to-scalar form we assume that its sub-forms will be
# well-defined in 2D only for dimension values 0,1
# Note: dim argument should be 2 or 3, while row and col are 0-based
def is_implemented_for_vector_to_scalar(row: int, col: int, dim: int) -> bool:
    if col == 2 and dim == 2:
        return False
    else:
        return True


# For a default scalar-to-vector form we assume that its sub-forms will be
# well-defined in 2D only for dimension values 0,1
# Note: dim argument should by 2 or 3, while row and col are 0-based
def is_implemented_for_scalar_to_vector(row: int, col: int, dim: int) -> bool:
    if row == 2 and dim == 2:
        return False
    else:
        return True


@dataclass
class FormInfo:
    form_name: str
    trial_degree: int
    test_degree: int
    trial_family: str = "Lagrange"
    test_family: str = "Lagrange"
    quad_schemes: Dict[int, Union[int, str]] = field(
        default_factory=lambda: {2: 2, 3: 2}
    )
    blending: GeometryMap = field(default_factory=lambda: IdentityMap())
    description: str = ""
    integrate_rows: List[int] = field(default_factory=lambda: [0])
    row_dim: int = 1
    col_dim: int = 1
    is_implemented: Callable[[int, int, int], bool] = is_implemented_for_scalar
    inline_quad: bool = False

    def min_quad_scheme(self) -> Union[int, str]:
        """The minimal quadrature degree for all dimensions or 'exact'."""
        return min(
            self.quad_schemes.values(), key=lambda x: 2e31 - 1 if x == "exact" else x
        )

    def space_desc(self) -> str:
        """A compact representation of the function spaces."""
        if self.trial_family == "N1E1":
            return "n1e1"
        elif self.trial_family == "P2 enhanced with Bubble":
            return "p2_plus_bubble"
        elif self.trial_degree == self.test_degree:
            return f"p{self.trial_degree}"
        else:
            return f"p{self.trial_degree}_to_p{self.test_degree}"

    def blending_desc(self) -> str:
        """The type of transformation from the reference element. Either 'blending' or 'affine'."""
        if isinstance(self.blending, ExternalMap):
            return "blending"
        else:
            return "affine"

    def quadrature_desc(self) -> str:
        """A compact representation of the quadrature scheme."""
        min_scheme = self.min_quad_scheme()
        if "exact" == min_scheme:
            return "qe"
        elif isinstance(min_scheme, int):
            return f"q{min_scheme}"
        else:
            raise HOGException("Invalid quadrature scheme.")

    def file_path(self, hyteg_base_path: str, header: bool = True) -> str:
        ending = ".cpp"
        if header:
            ending = ".hpp"

        sub_dir = "p1"
        if self.trial_family == "N1E1":
            sub_dir = "n1e1"
        elif self.trial_family == "P2 enhanced with Bubble":
            return "p2_plus_bubble"
        elif self.trial_degree == self.test_degree:
            sub_dir = f"p{self.trial_degree}"
        else:
            sub_dir = f"p{self.trial_degree}_to_p{self.test_degree}"
        file_name = self.full_form_name() + ending
        return os.path.join(hyteg_base_path, HYTEG_GENERATED_PATH, sub_dir, file_name)

    def class_name(self, row=0, col=0):
        """HyTeG class name assembled from the input parameters."""
        space = self.space_desc()
        b = self.blending_desc()
        q = self.quadrature_desc()

        if self.row_dim == 1 and self.col_dim == 1:
            return f"{space}_{self.form_name}_{b}_{q}"
        elif self.row_dim == 1:
            return f"{space}_{self.form_name}_{col}_{b}_{q}"
        elif self.col_dim == 1:
            return f"{space}_{self.form_name}_{row}_{b}_{q}"
        else:
            return f"{space}_{self.form_name}_{row}_{col}_{b}_{q}"

    def full_form_name(self):
        """HyTeG class name assembled from the input parameters."""
        return f"{self.space_desc()}_{self.form_name}_{self.blending_desc()}_{self.quadrature_desc()}"

    def __str__(self):
        return (
            f"{self.full_form_name()}, trial: ({self.trial_family}, degree {self.trial_degree}), test: ({self.test_family}, "
            f"degree {self.test_degree}), dimension: {self.row_dim} x {self.col_dim}, "
            f"quadrature schemes/degree (dimension): {self.quad_schemes}, blending: {self.blending}"
        )

    def __repr__(self):
        return str(self)


HYTEG_GENERATED_PATH = "src/hyteg/forms/form_hyteg_generated/"

form_infos = [
    FormInfo(
        "diffusion",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: "exact", 3: 2},
    ),
    FormInfo(
        "diffusion",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 1, 3: 1},
    ),
    FormInfo(
        "diffusion",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 3, 3: 3},
        blending=ExternalMap(),
    ),
    FormInfo(
        "diffusion",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 1, 3: 1},
        blending=ExternalMap(),
    ),
    FormInfo(
        "diffusion",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: "exact", 3: 2},
    ),
    FormInfo(
        "diffusion",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: 3, 3: 3},
        blending=ExternalMap(),
    ),
    FormInfo(
        "diffusion",
        trial_degree=2,
        test_degree=2,
        trial_family="P2 enhanced with Bubble",
        test_family="P2 enhanced with Bubble",
        quad_schemes={2: "exact"},
        integrate_rows=[],
    ),
    FormInfo(
        "mass",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: "exact", 3: "exact"},
    ),
    FormInfo(
        "mass",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 4, 3: 4},
        blending=ExternalMap(),
    ),
    FormInfo(
        "mass",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: "exact", 3: "exact"},
    ),
    FormInfo(
        "mass",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: 4, 3: 4},
        blending=ExternalMap(),
    ),
    FormInfo(
        "mass",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        quad_schemes={3: 2},
        integrate_rows=[],
    ),
    FormInfo(
        "mass",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        quad_schemes={3: "exact"},
        integrate_rows=[],
    ),
    FormInfo(
        "mass",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        quad_schemes={3: 2},
        blending=ExternalMap(),
        integrate_rows=[],
    ),
    FormInfo(
        "mass",
        trial_degree=2,
        test_degree=2,
        trial_family="P2 enhanced with Bubble",
        test_family="P2 enhanced with Bubble",
        quad_schemes={2: "exact"},
        integrate_rows=[],
    ),
    FormInfo(
        "curl_curl",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        quad_schemes={3: 0},
        integrate_rows=[],
    ),
    FormInfo(
        "curl_curl",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        quad_schemes={3: 2},
        blending=ExternalMap(),
        integrate_rows=[],
    ),
    FormInfo(
        "div_k_grad",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 3, 3: 3},
    ),
    FormInfo(
        "div_k_grad",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: 4, 3: 4},
    ),
    FormInfo(
        "div_k_grad",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 3, 3: 3},
        blending=ExternalMap(),
    ),
    FormInfo(
        "div_k_grad",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: 4, 3: 4},
        blending=ExternalMap(),
    ),
    FormInfo(
        "k_mass",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 4, 3: 4},
        description="Implements a linear form of type (k(x) * phi, psi) where phi is a trial function, psi a test function and k = k(x) a scalar, spatially varying coefficient.",
    ),
    FormInfo(
        "k_mass",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: 4, 3: 4},
        description="Implements a bilinear form of type (k(x) * phi, psi) where phi is a trial function, psi a test function and k = k(x) a scalar, spatially varying coefficient.",
    ),
    FormInfo(
        "pspg",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: "exact", 3: 2},
        description="Implements bilinear form for PSPG stabilisation.",
    ),
    FormInfo(
        "pspg",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 2, 3: 2},
        blending=ExternalMap(),
        description="Implements bilinear form for PSPG stabilisation.",
    ),
    FormInfo(
        "linear_form",
        trial_degree=0,
        test_degree=0,
        quad_schemes={2: 5, 3: 5},
        blending=ExternalMap(),
        description="Implements a linear form of type: (k(x), psi) where psi a test function and k = k(x) a scalar, external function.",
        inline_quad=True,
    ),
    FormInfo(
        "linear_form",
        trial_degree=0,
        test_degree=0,
        quad_schemes={2: 7, 3: 7},
        blending=ExternalMap(),
        description="Implements a linear form of type: (k(x), psi) where psi a test function and k = k(x) a scalar, external function.",
        inline_quad=True,
    ),
    FormInfo(
        "linear_form",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 6, 3: 6},
        description="Implements a linear form of type: (k(x), psi) where psi a test function and k = k(x) a scalar, external function.",
    ),
    FormInfo(
        "linear_form",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 5, 3: 5},
        blending=ExternalMap(),
        description="Implements a linear form of type: (k(x), psi) where psi a test function and k = k(x) a scalar, external function.",
        inline_quad=True,
    ),
    FormInfo(
        "linear_form",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: 6, 3: 6},
        description="Implements a linear form of type: (k(x), psi) where psi a test function and k = k(x) a scalar, external function.",
    ),
    FormInfo(
        "linear_form",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: 7, 3: 7},
        blending=ExternalMap(),
        description="Implements a linear form of type: (k(x), psi) where psi a test function and k = k(x) a scalar, external function.",
        inline_quad=True,
    ),
    FormInfo(
        "linear_form",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        quad_schemes={3: 6},
        description="Implements a linear form of type: (k(x), psi) where psi a test function and k = k(x) a vectorial, external function.",
        integrate_rows=[],
    ),
    FormInfo(
        "linear_form",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        quad_schemes={3: 6},
        blending=ExternalMap(),
        description="Implements a linear form of type: (k(x), psi) where psi a test function and k = k(x) a vectorial, external function.",
        integrate_rows=[],
    ),
    FormInfo(
        "supg_diffusion",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: 4, 3: 4},
        blending=ExternalMap(),
    ),
    FormInfo(
        "supg_diffusion",
        trial_degree=2,
        test_degree=2,
        quad_schemes={2: 4, 3: 4},
        blending=AnnulusMap(),
    ),
    FormInfo(
        "laplace_beltrami",
        trial_degree=1,
        test_degree=1,
        quad_schemes={3: 1},
    ),
    FormInfo(
        "laplace_beltrami",
        trial_degree=1,
        test_degree=1,
        quad_schemes={3: 3},
        blending=ExternalMap(),
    ),
    FormInfo("manifold_mass", trial_degree=1, test_degree=1, quad_schemes={2: 1}),
    FormInfo(
        "manifold_mass",
        trial_degree=1,
        test_degree=1,
        quad_schemes={2: 3},
        blending=ExternalMap(),
    ),
]

for d in [1, 2]:
    for epstype in ["cc", "var"]:
        form_infos.append(
            FormInfo(
                f"epsilon{epstype}",
                trial_degree=d,
                test_degree=d,
                quad_schemes={2: 2, 3: 2},
                row_dim=3,
                col_dim=3,
                is_implemented=is_implemented_for_vector_to_vector,
            )
        )

for d in [1, 2]:
    for epstype in ["cc", "var"]:
        form_infos.append(
            FormInfo(
                f"epsilon{epstype}",
                trial_degree=d,
                test_degree=d,
                # Theoretically correct integration of polynomials of degree 2*(d-1) should suffice
                # (also for blending and variable coeff) for correct O(h^k), but might give bad
                # constant? Need to check on that.
                quad_schemes={2: 2 * (d - 1) + 1, 3: 2 * (d - 1) + 1},
                blending=ExternalMap(),
                row_dim=3,
                col_dim=3,
                is_implemented=is_implemented_for_vector_to_vector,
            )
        )

        form_infos.append(
            FormInfo(
                f"epsilon{epstype}",
                trial_degree=d,
                test_degree=d,
                quad_schemes={2: 4, 3: 4},
                row_dim=3,
                col_dim=3,
                is_implemented=is_implemented_for_vector_to_vector,
            )
        )

for d in [1, 2]:
    for type in ["cc", "var"]:
        form_infos.append(
            FormInfo(
                f"full_stokes{type}",
                trial_degree=d,
                test_degree=d,
                quad_schemes={2: 2 * (d - 1) + 1, 3: 2 * (d - 1) + 1},
                row_dim=3,
                col_dim=3,
                is_implemented=is_implemented_for_vector_to_vector,
            )
        )

for d in [1, 2]:
    for type in ["cc", "var"]:
        form_infos.append(
            FormInfo(
                f"full_stokes{type}",
                trial_degree=d,
                test_degree=d,
                quad_schemes={2: 2 * (d - 1) + 1, 3: 2 * (d - 1) + 1},
                blending=ExternalMap(),
                row_dim=3,
                col_dim=3,
                is_implemented=is_implemented_for_vector_to_vector,
            )
        )

for d in [1]:
    form_infos.append(
        FormInfo(
            f"divdiv",
            trial_degree=d,
            test_degree=d,
            quad_schemes={2: 2, 3: 2},
            row_dim=3,
            col_dim=3,
            is_implemented=is_implemented_for_vector_to_vector,
        )
    )

# theoretically degree 2*(d-1) should suffice (also for blending and variable coeff)
# for correct O(h^k), but might give bad constant? Need to check on that
for trial_deg, test_deg, transpose in [
    (0, 1, True),
    (1, 0, False),
    (1, 2, True),
    (2, 1, False),
    (1, 1, True),
    (1, 1, False),
]:
    for blending in [IdentityMap(), ExternalMap()]:
        if not transpose:
            form_infos.append(
                FormInfo(
                    f"div",
                    trial_degree=trial_deg,
                    test_degree=test_deg,
                    quad_schemes={
                        2: trial_deg + test_deg - 1,
                        3: trial_deg + test_deg - 1,
                    },
                    row_dim=1,
                    col_dim=3,
                    is_implemented=is_implemented_for_vector_to_scalar,
                    blending=blending,
                )
            )
        else:
            form_infos.append(
                FormInfo(
                    f"divt",
                    trial_degree=trial_deg,
                    test_degree=test_deg,
                    quad_schemes={
                        2: trial_deg + test_deg - 1,
                        3: trial_deg + test_deg - 1,
                    },
                    row_dim=3,
                    col_dim=1,
                    is_implemented=is_implemented_for_scalar_to_vector,
                    blending=blending,
                )
            )

for blending in [IdentityMap(), ExternalMap()]:
    form_infos.append(
        FormInfo(
            "manifold_vector_mass",
            trial_degree=2,
            test_degree=2,
            quad_schemes={2: 3},
            row_dim=3,
            col_dim=3,
            is_implemented=is_implemented_for_vector_to_vector,
            blending=blending,
        )
    )

    form_infos.append(
        FormInfo(
            "manifold_normal_penalty",
            trial_degree=2,
            test_degree=2,
            quad_schemes={2: 3},
            row_dim=3,
            col_dim=3,
            is_implemented=is_implemented_for_vector_to_vector,
            blending=blending,
        )
    )

    form_infos.append(
        FormInfo(
            "manifold_epsilon",
            trial_degree=2,
            test_degree=2,
            quad_schemes={2: 3},
            row_dim=3,
            col_dim=3,
            is_implemented=is_implemented_for_vector_to_vector,
            blending=blending,
        )
    )

    form_infos.append(
        FormInfo(
            "manifold_epsilon",
            trial_degree=2,
            test_degree=2,
            quad_schemes={2: 6},
            row_dim=3,
            col_dim=3,
            is_implemented=is_implemented_for_vector_to_vector,
            blending=blending,
        )
    )

    form_infos.append(
        FormInfo(
            "vector_laplace_beltrami",
            trial_degree=2,
            test_degree=2,
            quad_schemes={2: 3},
            row_dim=3,
            col_dim=3,
            is_implemented=is_implemented_for_vector_to_vector,
            blending=blending,
        )
    )

for trial_deg, test_deg, transpose in [(1, 2, True), (2, 1, False)]:
    for blending in [IdentityMap(), ExternalMap()]:
        if not transpose:
            form_infos.append(
                FormInfo(
                    "manifold_div",
                    trial_degree=trial_deg,
                    test_degree=test_deg,
                    quad_schemes={2: 3},
                    row_dim=1,
                    col_dim=3,
                    is_implemented=is_implemented_for_vector_to_scalar,
                    blending=blending,
                )
            )
        else:
            form_infos.append(
                FormInfo(
                    "manifold_divt",
                    trial_degree=trial_deg,
                    test_degree=test_deg,
                    quad_schemes={2: 3},
                    row_dim=3,
                    col_dim=1,
                    is_implemented=is_implemented_for_scalar_to_vector,
                    blending=blending,
                )
            )

for trial_deg, test_deg, transpose in [
    (1, 2, True),
    (2, 1, False),
    (0, 2, True),
    (2, 0, False),
]:
    for blending in [IdentityMap(), ExternalMap()]:
        if not transpose:
            form_infos.append(
                FormInfo(
                    "manifold_vector_div",
                    trial_degree=trial_deg,
                    test_degree=test_deg,
                    quad_schemes={2: 3},
                    row_dim=1,
                    col_dim=3,
                    is_implemented=is_implemented_for_vector_to_scalar,
                    blending=blending,
                )
            )
        else:
            form_infos.append(
                FormInfo(
                    "manifold_vector_divt",
                    trial_degree=trial_deg,
                    test_degree=test_deg,
                    quad_schemes={2: 3},
                    row_dim=3,
                    col_dim=1,
                    is_implemented=is_implemented_for_scalar_to_vector,
                    blending=blending,
                )
            )


def form_func(
    name: str,
    row: int,
    col: int,
    trial: TrialSpace,
    test: TestSpace,
    geometry: ElementGeometry,
    quad: Quadrature,
    symbolizer: Symbolizer,
    blending: GeometryMap = IdentityMap(),
) -> sp.Matrix:
    if name == "diffusion":
        return diffusion(
            trial, test, geometry, symbolizer, blending=blending
        ).integrate(quad, symbolizer)
    elif name == "mass" and isinstance(trial, N1E1Space):
        return mass_n1e1(
            trial, test, geometry, symbolizer, blending=blending
        ).integrate(quad, symbolizer)
    elif name == "mass":
        return mass(trial, test, geometry, symbolizer, blending=blending).integrate(
            quad, symbolizer
        )
    elif name == "div_k_grad":
        return div_k_grad(
            trial, test, geometry, symbolizer, blending=blending
        ).integrate(quad, symbolizer)
    elif name == "curl_curl":
        return curl_curl(
            trial, test, geometry, symbolizer, blending=blending
        ).integrate(quad, symbolizer)
    elif name.startswith("epsilon"):
        if name.startswith("epsiloncc"):
            var_visc = False
        elif name.startswith("epsilonvar"):
            var_visc = True
        else:
            raise HOGException("Epsilon operator not supported.")
        if row not in [0, 1, 2] or col not in [0, 1, 2]:
            raise HOGException("Invalid call to epsilon form.")
        # the input parameters for the epsilon operators are intended to be switched below (col ~ trial component, row ~ test component)
        return epsilon(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_trial=col,
            component_test=row,
            variable_viscosity=var_visc,
        ).integrate(quad, symbolizer)
    elif name.startswith("full_stokes"):
        if name.startswith("full_stokescc"):
            var_visc = False
        elif name.startswith("full_stokesvar"):
            var_visc = True
        else:
            raise HOGException("Full_Stokes operator not supported.")
        if row not in [0, 1, 2] or col not in [0, 1, 2]:
            raise HOGException("Invalid call to epsilon form.")
        return full_stokes(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_trial=col,
            component_test=row,
            variable_viscosity=var_visc,
        ).integrate(quad, symbolizer)
    elif name.startswith("k_mass"):
        return k_mass(trial, test, geometry, symbolizer, blending=blending).integrate(
            quad, symbolizer
        )
    elif name.startswith("pspg"):
        return pspg(trial, test, geometry, quad, symbolizer, blending=blending)
    elif name.startswith("linear_form"):
        return linear_form(trial, test, geometry, quad, symbolizer, blending=blending)
    elif name.startswith("divt"):
        if row not in [0, 1, 2] or col != 0:
            raise HOGException("Invalid call to divt form.")
        return gradient(
            trial,
            test,
            geometry,
            symbolizer,
            component_index=row,
            blending=blending,
        ).integrate(quad, symbolizer)
    elif name.startswith("divdiv"):
        return divdiv(
            trial, test, col, row, geometry, quad, symbolizer, blending=blending
        )
    elif name.startswith("div"):
        if col not in [0, 1, 2] or row != 0:
            raise HOGException("Invalid call to div form.")
        return divergence(
            trial,
            test,
            geometry,
            symbolizer,
            component_index=col,
            blending=blending,
        ).integrate(quad, symbolizer)
    elif name.startswith("supg_d"):
        raise HOGException(f"SUPG Diffusion is not supported for form generation")
    elif name.startswith("laplace_beltrami"):
        return laplace_beltrami(
            trial, test, geometry, symbolizer, blending=blending
        ).integrate(quad, symbolizer)
    elif name.startswith("manifold_mass"):
        return manifold_mass(
            trial, test, geometry, symbolizer, blending=blending
        ).integrate(quad, symbolizer)
    elif name.startswith("manifold_vector_mass"):
        return manifold_vector_mass(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_trial=col,
            component_test=row,
        ).integrate(quad, symbolizer)
    elif name.startswith("manifold_normal_penalty"):
        return manifold_normal_penalty(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_trial=col,
            component_test=row,
        ).integrate(quad, symbolizer)
    elif name.startswith("manifold_divt"):
        return manifold_divergence(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_index=row,
            transpose=True,
        ).integrate(quad, symbolizer)
    elif name.startswith("manifold_div"):
        return manifold_divergence(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_index=col,
            transpose=False,
        ).integrate(quad, symbolizer)
    elif name.startswith("manifold_vector_divt"):
        return manifold_vector_divergence(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_index=row,
            transpose=True,
        ).integrate(quad, symbolizer)
    elif name.startswith("manifold_vector_div"):
        return manifold_vector_divergence(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_index=col,
            transpose=False,
        ).integrate(quad, symbolizer)
    elif name.startswith("manifold_epsilon"):
        return manifold_epsilon(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_trial=col,
            component_test=row,
        ).integrate(quad, symbolizer)
    elif name.startswith("vector_laplace_beltrami"):
        return vector_laplace_beltrami(
            trial,
            test,
            geometry,
            symbolizer,
            blending=blending,
            component_trial=col,
            component_test=row,
        ).integrate(quad, symbolizer)
    else:
        raise HOGException(f"Cannot call form function with name {name}.")


def parse_arguments():
    parser = ArgumentParser(description="Generates all HyTeG forms.")
    parser.add_argument(
        "hyteg_base_path",
        type=str,
        help=(
            "path to the HyTeG repository, if specified, the generated code is written to the corresponding files"
        ),
        nargs="?",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        help="only generate forms that include the passed string (which can be a Python regular expression) (works in combination with --list to show all filtered forms and abort)",
        default="",
    )
    parser.add_argument(
        "-g",
        "--geometry",
        type=str,
        help="build form(s) only for triangle (2D) or tetrahedron (3D) elements; if not speficied we do both",
        nargs="?",
        choices=["triangle", "tetrahedron", "embedded_triangle", "both"],
        default="both",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list all available forms by name and abort (works in combination with --filter to show all filtered forms and abort)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="file where all generated code shall be written to",
    )
    parser.add_argument(
        "-s",
        "--stdout",
        action="store_true",
        help="all generated code is written to stdout",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        help="specify verbosity level (0 = silent, 1 = info, 2 = debug)",
    )
    args = parser.parse_args()
    return parser, args


def valid_base_dir(hyteg_base_path):
    directory_structure = os.listdir(hyteg_base_path)

    for d in ["src", "tests", "apps", "doc"]:
        if d not in directory_structure:
            return False

    return True


def main():
    clear_cache()

    parser, args = parse_arguments()
    if args.verbose == 0:
        loglevel = logging.CRITICAL
    elif args.verbose == 1:
        loglevel = logging.INFO
    elif args.verbose == 2:
        loglevel = logging.DEBUG
    else:
        raise HOGException(f"Invalid verbosity level {args.verbose}.")

    logger = get_logger(loglevel)
    TimedLogger.set_log_level(loglevel)

    if not any([args.hyteg_base_path, args.stdout, args.output_file, args.list]):
        parser.print_help()
        return

    if args.output_file:
        # need to write in append mode later
        with open(args.output_file, "w") as f:
            f.write("")

    logger.info("################################")
    logger.info("### HyTeG Operator Generator ###")
    logger.info("################################")

    filtered_form_infos = [
        fi for fi in form_infos if re.search(args.filter, fi.full_form_name())
    ]

    if args.list:
        logger.info("Available forms:")
        for form_info in filtered_form_infos:
            logger.info(f"- {form_info.full_form_name()}")
        logger.info("Bye.")
        return

    if args.hyteg_base_path:
        if not valid_base_dir(args.hyteg_base_path):
            raise HOGException(
                "The specified directory does not seem to be the HyTeG directory."
            )

    symbolizer = Symbolizer()

    # determine geometries to use
    geometries: List[ElementGeometry]
    if args.geometry == "triangle":
        logger.info(f"- selected geometry: triangle")
        geometries = [TriangleElement()]
    elif args.geometry == "tetrahedron":
        logger.info(f"- selected geometry: tetrahedron")
        geometries = [TetrahedronElement()]
    elif args.geometry == "embedded_triangle":
        logger.info(f"- selected geometry: embedded triangle")
        geometries = [TriangleElement(space_dimension=3)]
    else:
        logger.info(f"- selected geometries: triangle, tetrahedron")
        geometries = [TriangleElement(), TetrahedronElement()]

    logger.info(
        f"Generating {len(filtered_form_infos)} form{'s' if len(filtered_form_infos) > 1 else ''}:"
    )
    for form_info in filtered_form_infos:
        logger.info(f"- {form_info.full_form_name()}")

    logger.info(f"Patience you must have young padawan")
    for form_info in filtered_form_infos:
        logger.info(f"{form_info}")

        trial: TrialSpace
        if form_info.trial_family == "N1E1":
            trial = TrialSpace(N1E1Space(symbolizer))
        elif form_info.trial_family == "P2 enhanced with Bubble":
            trial = TrialSpace(P2PlusBubbleSpace(symbolizer))
        else:
            trial = TrialSpace(
                LagrangianFunctionSpace(form_info.trial_degree, symbolizer)
            )

        test: TestSpace
        if form_info.test_family == "N1E1":
            test = TestSpace(N1E1Space(symbolizer))
        elif form_info.test_family == "P2 enhanced with Bubble":
            test = TestSpace(P2PlusBubbleSpace(symbolizer))
        else:
            test = TestSpace(LagrangianFunctionSpace(form_info.test_degree, symbolizer))

        form_classes = []

        for row in range(0, form_info.row_dim):
            for col in range(0, form_info.col_dim):
                form_codes = []

                for geometry in geometries:
                    if geometry.dimensions in form_info.quad_schemes.keys():
                        with TimedLogger(
                            f"- Generating code for class {form_info.class_name(row,col)}, {geometry.dimensions}D"
                        ):
                            quad = Quadrature(
                                select_quadrule(
                                    form_info.quad_schemes[geometry.dimensions],
                                    geometry,
                                ),
                                geometry,
                                inline_values=form_info.inline_quad,
                            )

                            if form_info.is_implemented( row, col, geometry.dimensions ):
                                mat = form_func(
                                    form_info.form_name,
                                    row,
                                    col,
                                    trial,
                                    test,
                                    geometry,
                                    quad,
                                    symbolizer,
                                    blending=form_info.blending,
                                )
                                form_codes.append(
                                    HyTeGIntegrator(
                                        form_info.class_name(row, col),
                                        mat,
                                        geometry,
                                        quad,
                                        symbolizer,
                                        integrate_rows=form_info.integrate_rows,
                                    )
                                )
                form_classes.append(
                    HyTeGFormClass(
                        form_info.class_name(row, col),
                        trial,
                        test,
                        form_codes,
                        description=form_info.description,
                    )
                )

        form_hyteg = HyTeGForm(
            form_info.full_form_name(),
            trial,
            test,
            form_classes,
            description=form_info.description,
        )

        for header in [True, False]:
            output = form_hyteg.to_code(header=header)

            if args.hyteg_base_path:
                file_path = form_info.file_path(args.hyteg_base_path, header=header)
                dir_path = os.path.dirname(file_path)
                os.makedirs(dir_path, exist_ok=True)

                with open(file_path, "w") as f:
                    logger.info(
                        f"- Writing form class {form_info.full_form_name()} to {file_path}"
                    )
                    f.write(output)

            if args.output_file:
                with open(args.output_file, "a") as f:
                    logger.info(
                        f"- Writing form class {form_info.full_form_name()} to {args.output_file}: header={header}"
                    )
                    f.write(output)

            if args.stdout:
                print(output)

    logger.info(f"Done. Bye.")


if __name__ == "__main__":
    main()
