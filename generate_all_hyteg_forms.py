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

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Union
import os
import logging
import itertools
import sympy as sp
from sympy.core.cache import clear_cache
import re
import quadpy
from functools import total_ordering

from hog.blending import GeometryMap, IdentityMap, ExternalMap, AnnulusMap
from hog.element_geometry import (
    TriangleElement,
    TetrahedronElement,
    ElementGeometry,
)
from hog.function_space import (
    LagrangianFunctionSpace,
    DGFunctionSpace,
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
@total_ordering
class FormInfo:
    form_name: str
    trial_degree: int
    test_degree: int
    trial_family: str = "Lagrange"
    test_family: str = "Lagrange"
    supported_geometry_options: List[str] = field(
        default_factory=lambda: ["triangle", "tetrahedron"]
    )
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
        descr_string = ""
        if self.trial_family == "N1E1":
            descr_string = "n1e1"
        elif self.trial_family == "P2 enhanced with Bubble":
            if self.test_family == "P2 enhanced with Bubble":
                descr_string = "p2_plus_bubble"
            elif self.test_family == "DG":
                descr_string = f"p2_plus_bubble_to_dg{self.test_degree}"
        elif self.trial_family == "Lagrange" and self.test_family == "Lagrange":
            if self.trial_degree == self.test_degree:
                descr_string = f"p{self.trial_degree}"
            else:
                descr_string = f"p{self.trial_degree}_to_p{self.test_degree}"
        elif self.trial_family == "DG":
            if self.test_family == "DG":
                if self.trial_degree == self.test_degree:
                    descr_string = f"dg{self.trial_degree}"
                else:
                    descr_string = f"dg{self.trial_degree}_to_dg{self.test_degree}"
            elif self.test_family == "Lagrange":
                descr_string = f"dg{self.trial_degree}_to_p{self.test_degree}"
            elif self.test_family == "P2 enhanced with Bubble":
                descr_string = f"dg{self.trial_degree}_to_p2_plus_bubble"
            else:
                raise HOGException(
                    f"Do not know how to name combination of DGFunctionSpace with {self.test_family}."
                )
        return descr_string

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
        elif (
            self.trial_family == "P2 enhanced with Bubble"
            and self.test_family == "P2 enhanced with Bubble"
        ):
            sub_dir = "p2"
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

    def supports_geometry(self, geometry: str) -> bool:
        """Check if form supports a certain type of geometric element."""
        if geometry == "triangle+tetrahedron":
            return ("triangle" in self.supported_geometry_options) and (
                "tetrahedron" in self.supported_geometry_options
            )
        else:
            return geometry in self.supported_geometry_options

    def __eq__(self, other):
        return self.full_form_name() == other.full_form_name()

    def __lt__(self, other):
        return self.full_form_name() < other.full_form_name()

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
        supported_geometry_options=["triangle"],
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
        supported_geometry_options=["tetrahedron"],
        quad_schemes={3: 2},
        integrate_rows=[],
    ),
    FormInfo(
        "mass",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        supported_geometry_options=["tetrahedron"],
        quad_schemes={3: "exact"},
        integrate_rows=[],
    ),
    FormInfo(
        "mass",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        supported_geometry_options=["tetrahedron"],
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
        supported_geometry_options=["triangle"],
        quad_schemes={2: "exact"},
        integrate_rows=[],
    ),
    FormInfo(
        "curl_curl",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        supported_geometry_options=["tetrahedron"],
        quad_schemes={3: 0},
        integrate_rows=[],
    ),
    FormInfo(
        "curl_curl",
        trial_degree=1,
        test_degree=1,
        trial_family="N1E1",
        test_family="N1E1",
        supported_geometry_options=["tetrahedron"],
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
        supported_geometry_options=["tetrahedron"],
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
        supported_geometry_options=["tetrahedron"],
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
        supported_geometry_options=["embedded_triangle"],
        quad_schemes={3: 1},
    ),
    FormInfo(
        "laplace_beltrami",
        trial_degree=1,
        test_degree=1,
        supported_geometry_options=["embedded_triangle"],
        quad_schemes={3: 3},
        blending=ExternalMap(),
    ),
    FormInfo(
        "manifold_mass",
        trial_degree=1,
        test_degree=1,
        supported_geometry_options=["embedded_triangle"],
        quad_schemes={2: 1},
    ),
    FormInfo(
        "manifold_mass",
        trial_degree=1,
        test_degree=1,
        supported_geometry_options=["embedded_triangle"],
        quad_schemes={2: 3},
        blending=ExternalMap(),
    ),
]

for d in [1, 2]:
    for epstype in ["var"]:
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
    for epstype in ["var"]:
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

for trial_deg, test_deg, transpose in [
    (1, 2, True),
    (2, 1, False),
]:
    for blending in [IdentityMap(), ExternalMap()]:
        if not transpose:
            form_infos.append(
                FormInfo(
                    f"div",
                    trial_degree=trial_deg,
                    test_degree=test_deg,
                    trial_family="P2 enhanced with Bubble",
                    test_family="DG",
                    supported_geometry_options=["triangle"],
                    quad_schemes={
                        2: 3 + test_deg - 1,
                        3: 4 + test_deg - 1,
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
                    trial_family="DG",
                    test_family="P2 enhanced with Bubble",
                    supported_geometry_options=["triangle"],
                    quad_schemes={
                        2: trial_deg + 3 - 1,
                        3: trial_deg + 4 - 1,
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
            supported_geometry_options=["embedded_triangle"],
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
            supported_geometry_options=["embedded_triangle"],
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
            supported_geometry_options=["embedded_triangle"],
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
            supported_geometry_options=["embedded_triangle"],
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
            supported_geometry_options=["embedded_triangle"],
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
                    supported_geometry_options=["embedded_triangle"],
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
                    supported_geometry_options=["embedded_triangle"],
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
                    supported_geometry_options=["embedded_triangle"],
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
                    supported_geometry_options=["embedded_triangle"],
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
        return pspg(
            trial, test, geometry, quad, symbolizer, blending=blending
        ).integrate(quad, symbolizer)
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
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description="""Generates (selected) HyTeG forms

Selection of forms to generate is based on the --filter and --geometry arguments:

* if neither --geometry, nor --filter is given, we generate all defined forms
* if only --filter is given, all forms matching the filter expression are generated
* if only --geometry is given, all forms that support the given geometric element type are generated
* if both --filter and --geometry are given, the intersection of both sets is generated""",
    )

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
        help="only generate forms that include the passed string (which can be a Python regular expression)",
        default="",
    )
    parser.add_argument(
        "-g",
        "--geometry",
        type=str,
        help="build form(s) only for triangle (2D), tetrahedron (3D) elements or embedded triangles (manifolds)",
        nargs="?",
        choices=[
            "triangle",
            "tetrahedron",
            "triangle+tetrahedron",
            "embedded_triangle",
        ],
        default="",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list all forms that would be generated for given values of --filter and --geometry; but do not generate anything",
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


def assemble_list_of_forms_to_generate(
    form_infos: List[FormInfo], logger: logging.Logger
) -> List[FormInfo]:
    """From the list of all defined forms extract those to generate.

    The extraction works as follows, depending on the command-line
    options given:

    - if neither --geometry, nor --filter was given, we return all defined forms
    - if only --filter was given, it is applied to the list of all defined forms
    - if only --geometry was given, we extract all forms that support the given geometry element type
    - if both --filter and --geometry was given, determine the intersection of both sets
    """

    form_list: List[FormInfo] = []

    parser, args = parse_arguments()

    if args.geometry == "" and args.filter == "":
        form_list = form_infos
        logger.info(
            "No '--filter' and no '--geometry' given: selecting all forms available"
        )

    elif args.geometry != "" and args.filter == "":
        logger.info(
            f"No '--filter' given, extracting all forms supporting '--geometry {args.geometry}'"
        )
        for fi in form_infos:
            if fi.supports_geometry(args.geometry):
                form_list.append(fi)

    elif args.geometry == "" and args.filter != "":
        logger.info(f"Extracting forms based on '--filter {args.filter}'")
        form_list = [
            fi for fi in form_infos if re.search(args.filter, fi.full_form_name())
        ]

    else:
        logger.info(f"Extracting forms based on '--filter {args.filter}'")
        form_list = [
            fi for fi in form_infos if re.search(args.filter, fi.full_form_name())
        ]
        logger.info(f"Found {len(form_list)} matching forms")
        for fi in form_list:
            logger.info(f"* {fi.full_form_name()}")

        logger.info(f"Checking forms against '--geometry {args.geometry}'")

        aux_list = form_list.copy()
        for fi in form_list:
            if not fi.supports_geometry(args.geometry):
                logger.info(f"* deselecting '{fi.full_form_name()}' again")
                aux_list.remove(fi)
        form_list = aux_list

        # sort alphabetically by full form name
        form_list.sort()

    return form_list


def geometry_string_to_list(geometry_string: str) -> List[str]:
    glist: List[str] = []
    if geometry_string == "triangle+tetrahedron":
        glist.append("triangle")
        glist.append("tetrahedron")
    else:
        glist.append(geometry_string)
    return glist


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

    filtered_form_infos = assemble_list_of_forms_to_generate(form_infos, logger)

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

    # no forms -> nothing to do
    if len(filtered_form_infos) == 0:
        logger.info(f"Found no matching forms to generate.")
        logger.info(f"Bye.")
        quit()

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

        # determine geometries to use for this form
        geometries: List[ElementGeometry] = []

        target_geometries = []
        if args.geometry == "":
            target_geometries = form_info.supported_geometry_options
        else:
            target_geometries = geometry_string_to_list(args.geometry)

        if (
            "triangle" in target_geometries
            or "triangle+tetrahedron" in target_geometries
        ):
            geometries.append(TriangleElement())

        if (
            "tetrahedron" in target_geometries
            or "triangle+tetrahedron" in target_geometries
        ):
            geometries.append(TetrahedronElement())

        if "embedded_triangle" in target_geometries:
            geometries.append(TriangleElement(space_dimension=3))

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

                            if form_info.is_implemented(row, col, geometry.dimensions):
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
