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

import sympy as sp
from typing import List, Tuple, Union

from hog.ast import Assignment, CodeBlock
from hog.exception import HOGException
from hog.quadrature import Quadrature
from hog.symbolizer import Symbolizer
from hog.function_space import N1E1Space, TrialSpace, TestSpace
from hog.element_geometry import ElementGeometry
from hog.code_generation import code_block_from_element_matrix
from hog.multi_assignment import Member
from hog.hyteg_code_generation import GENERATED_COMMENT, COPYRIGHT_COMMENT, PRAGMA_ONCE


class HyTeGIntegrator:
    integrate_matrix_elements: List[Tuple[str, int]]

    def __init__(
        self,
        class_name: str,
        element_matrix: sp.Matrix,
        geometry: ElementGeometry,
        quad: Quadrature,
        symbolizer: Symbolizer,
        not_implemented: bool = False,
        integrate_rows: List[int] = [],
    ):
        """Represents the required components for the HyTeG method 'integrateAll()'."""
        self.class_name = class_name
        self.element_matrix = element_matrix
        self.geometry = geometry
        self.quad = quad
        self.symbolizer = symbolizer
        self.not_implemented = not_implemented
        self.integrate_matrix_elements = []
        for row in integrate_rows:
            self.integrate_matrix_elements.append(("row", row))
        self.integrate_matrix_elements.append(("all", 0))

        (
            self.integrate_all_decl,
            self.integrate_all_impl,
            self.integrate_other_decl,
            self.integrate_other_impl,
            self.helper_methods_decl,
            self.helper_methods_impl,
            self.members,
        ) = self._setup_methods()

    def _docstring(
        self,
        code_block: CodeBlock,
    ) -> str:
        """Returns a string formatted as c-style comment containing infos about the generated form."""

        if self.not_implemented:
            return "/// \\brief Not implemented - does nothing."

        operations_table = code_block.count_operations().to_table()
        operations_table = "\n".join(
            [
                f"///                                           {r}"
                for r in operations_table.splitlines()
            ]
        )
        info = (
            f"/// \\brief Integrates the weak form over the passed element (vertices in computational space).\n"
            f"///\n"
            f"/// - element geometry:                       {self.geometry}\n"
            f"/// - element matrix dimensions (rows, cols): {self.element_matrix.shape}\n"
            f"/// - quadrature rule:                        {self.quad}\n"
            f"/// - floating point operations:\n"
            f"{operations_table}\n"
            f"///"
        )
        return info

    def _setup_methods(
        self,
    ) -> Tuple[str, str, List[str], List[str], List[str], List[str], List[Member]]:
        rows, cols = self.element_matrix.shape

        # read from input array of computational vertices
        input_assignments = []
        internal_vertex_symbols = self.symbolizer.affine_vertices_as_vectors(
            self.geometry.dimensions, self.geometry.num_vertices
        )
        for point in range(self.geometry.num_vertices):
            for coord in range(self.geometry.dimensions):
                lhs = internal_vertex_symbols[point][coord]
                rhs = self.symbolizer.input_affine_vertex_access(point, coord)
                input_assignments.append(Assignment(lhs, rhs))

        integrate_decl = {}
        integrate_impl = {}

        for integrate_matrix_element in self.integrate_matrix_elements:
            if integrate_matrix_element[0] == "all":
                method_name = "integrateAll"
                element_matrix_sliced = self.element_matrix
                cpp_override = True

                # output assignments to output matrix
                output_assignments = []
                for row in range(rows):
                    for col in range(cols):
                        lhs = self.symbolizer.output_element_matrix_access(row, col)
                        rhs = self.symbolizer.element_matrix_entry(row, col)
                        output_assignments.append(
                            Assignment(lhs, rhs, is_declaration=False)
                        )

            elif integrate_matrix_element[0] == "row":
                integrate_row = integrate_matrix_element[1]

                method_name = f"integrateRow{integrate_row}"
                element_matrix_sliced = sp.zeros(1, cols)
                for col in range(cols):
                    element_matrix_sliced[0, col] = self.element_matrix[
                        integrate_row, col
                    ]
                cpp_override = True

                # output assignments to output matrix
                output_assignments = []
                for col in range(cols):
                    lhs = self.symbolizer.output_element_matrix_access(0, col)
                    rhs = self.symbolizer.element_matrix_entry(integrate_row, col)
                    output_assignments.append(
                        Assignment(lhs, rhs, is_declaration=False)
                    )

            output_rows, output_cols = element_matrix_sliced.shape

            # generate code from sympy matrix
            code_block, func_defs, members = code_block_from_element_matrix(
                element_matrix_sliced, self.quad, self.geometry, self.symbolizer
            )

            code_block.statements = (
                input_assignments + code_block.statements + output_assignments
            )

            if self.not_implemented:
                code_block_code = ""
            else:
                code_block_code = "\n      ".join(code_block.to_code().splitlines())

            hyteg_matrix_type = f"Matrix< real_t, {output_rows}, {output_cols} >"

            def integrate_f(
                method_name: str,
                prefix: str = "",
                header: bool = True,
                without_argnames: bool = False,
                override: bool = True,
            ) -> str:
                override_str = ""
                if header and override:
                    override_str = " override"

                return f"   void {prefix}{method_name}( const std::array< Point3D, {self.geometry.num_vertices} >& {'' if without_argnames else 'coords'}, {hyteg_matrix_type}& {'' if without_argnames else 'elMat'} ) const{override_str}"

            integrate_decl[integrate_matrix_element] = (
                "   " + "\n   ".join(self._docstring(code_block).splitlines()) + "\n"
            )
            integrate_decl[
                integrate_matrix_element
            ] += f"{integrate_f(method_name, override=cpp_override)};"

            integrate_impl[
                integrate_matrix_element
            ] = f"""{integrate_f(method_name, prefix=self.class_name + "::", header=False, without_argnames=self.not_implemented, override=cpp_override)}
   {{
      {code_block_code}
   }}"""

        integrate_all_decl = integrate_decl[("all", 0)]
        integrate_all_impl = integrate_impl[("all", 0)]
        integrate_other_decl = [
            decl for k, decl in integrate_decl.items() if k != ("all", 0)
        ]
        integrate_other_impl = [
            impl for k, impl in integrate_impl.items() if k != ("all", 0)
        ]

        helper_methods_decl = []
        helper_methods_impl = []
        if func_defs:
            for fd in func_defs:
                fd_code = "   " + "\n   ".join(fd.declaration().splitlines())
                helper_methods_decl.append(fd_code)
                fd_code = "   " + "\n   ".join(
                    fd.implementation(name_prefix=self.class_name + "::").splitlines()
                )
                helper_methods_impl.append(fd_code)

        return (
            integrate_all_decl,
            integrate_all_impl,
            integrate_other_decl,
            integrate_other_impl,
            helper_methods_decl,
            helper_methods_impl,
            members,
        )


class HyTeGFormClass:
    def __init__(
        self,
        name: str,
        trial: TrialSpace,
        test: TestSpace,
        integrators: List[HyTeGIntegrator],
        description: str = "",
    ):
        self.name = name
        self.trial = trial
        self.test = test
        self.integrators = integrators
        self.description = description

    def _docstring(
        self,
    ) -> str:
        """Returns a string formatted as c-style comment containing infos about the generated class."""
        info = (
            f"/// Implementation of the integration of a weak form over an element.\n"
            f"///\n"
            f"/// - name:        {self.name}\n"
            f"/// - description: {self.description}\n"
            f"/// - trial space: {self.trial}\n"
            f"/// - test space:  {self.test}\n"
            f"///"
        )
        return info

    def _constructor(self) -> str:
        members = []

        for f in self.integrators:
            members += f.members

        members = sorted(set(members), key=lambda m: m.name_constructor)

        if not members:
            return ""

        default_constructor = f'{self.name}() {{ WALBERLA_ABORT("Not implemented."); }}'

        ctr_prms = ", ".join([f"{m.dtype} {m.name_constructor}" for m in members])

        init_list = "\n   , ".join(
            [f"{m.name_member}({m.name_constructor})" for m in members]
        )

        member_decl = "\n   ".join([f"{m.dtype} {m.name_member};" for m in members])

        constructor_string = f""" public:

   {default_constructor}

   {self.name}( {ctr_prms} )
   : {init_list}
   {{}}

 private:

   {member_decl}
"""
        return constructor_string

    def to_code(self, header: bool = True) -> str:
        file_string = []

        if isinstance(self.trial, N1E1Space):
            super_class = f"n1e1::N1E1Form"
        elif self.trial.degree == self.test.degree:
            super_class = f"P{self.trial.degree}FormHyTeG"
        else:
            super_class = f"P{self.trial.degree}ToP{self.test.degree}FormHyTeG"

        info = self._docstring()

        class_open = f"{info}\nclass {self.name} : public {super_class}\n{{"
        class_close = "};"

        public = " public:"
        private = " private:"

        helper_methods = [f.helper_methods_decl for f in self.integrators]
        has_helpers = any([hmm for hm in helper_methods for hmm in hm])

        if header:
            file_string.append(class_open)
            file_string.append(self._constructor())
            file_string.append(public)
            for integrator in self.integrators:
                file_string.append(integrator.integrate_all_decl)
                for other_integrator_decl in integrator.integrate_other_decl:
                    file_string.append(other_integrator_decl)
            if has_helpers:
                file_string.append(private)
                for integrator in self.integrators:
                    for hm in integrator.helper_methods_decl:
                        file_string.append(hm)
            file_string.append(class_close)
        else:
            for integrator in self.integrators:
                file_string.append(integrator.integrate_all_impl)
                for other_integrator_impl in integrator.integrate_other_impl:
                    file_string.append(other_integrator_impl)
            for helper_method in self.integrators:
                for hm in helper_method.helper_methods_impl:
                    file_string.append(hm)

        return "\n\n".join(file_string)


class HyTeGForm:
    NAMESPACE_OPEN = "namespace hyteg {\nnamespace forms {"

    NAMESPACE_CLOSE = "} // namespace forms\n} // namespace hyteg"

    def __init__(
        self,
        name: str,
        trial: TrialSpace,
        test: TestSpace,
        formClasses: List[HyTeGFormClass],
        description: str = "",
    ):
        self.name = name
        self.trial = trial
        self.test = test
        self.formClasses = formClasses
        self.description = description

    def to_code(self, header: bool = True) -> str:
        file_string = []

        if isinstance(self.trial, N1E1Space):
            super_class = f"N1E1Form"
        elif self.trial.degree == self.test.degree:
            super_class = f"form_hyteg_base/P{self.trial.degree}FormHyTeG"
        else:
            super_class = (
                f"form_hyteg_base/P{self.trial.degree}ToP{self.test.degree}FormHyTeG"
            )

        if header:
            includes = "\n".join(
                [
                    '#include "hyteg/geometry/GeometryMap.hpp"',
                    f'#include "hyteg/forms/{super_class}.hpp"',
                ]
            )
        else:
            includes = f'#include "{self.name}.hpp"'

        file_string.append(COPYRIGHT_COMMENT)
        file_string.append(GENERATED_COMMENT)

        if header:
            file_string.append(PRAGMA_ONCE)

        file_string.append(includes)
        file_string.append(HyTeGForm.NAMESPACE_OPEN)

        for fc in self.formClasses:
            classCode = fc.to_code(header)
            file_string.append(classCode)

        file_string.append(HyTeGForm.NAMESPACE_CLOSE)

        return "\n\n".join(file_string) + "\n"
