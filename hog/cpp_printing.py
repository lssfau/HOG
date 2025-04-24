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

from copy import deepcopy
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
import shutil
import subprocess
import sys
import os
from textwrap import indent

from pystencils.backends.cbackend import generate_c
import pystencils.astnodes

from hog.exception import HOGException


"""

This file contains a collection of helper classes to facilitate printing C++ code.

The test 
    
    hog_tests/test_cpp_printing.py
    
provides some insight on how to work with it. Supporting all C++ features is not the primary goal of the present
implementation. It shall just provide a _simple_ interface to avoid boilerplate and spaghetti code when writing C++ code
to file.

"""

# Eventually, a formatting program shall be run to format all output.
# If that is not supported for some reason, simple indentation at least provides some eye candy.
INDENT = "    "


class CppFileRepresentation(Enum):
    """Defines ways to represent/split the C++ file contents."""

    HEADER_ONLY = auto()
    HEADER_SPLIT = auto()
    IMPL_SPLIT = auto()
    VARIANT_IMPL_SPLIT = auto()


class CppVariable:
    """
    Represents a C++ variable. Can be used for several purposes (arguments, member variables, ...).
    """

    def __init__(
        self,
        name: str,
        type: str,
        is_const: bool = False,
        is_reference: bool = False,
        is_pointer: bool = False,
        is_pointer_const: bool = False,
    ):
        self.name = name
        self.type = type
        self.is_const = is_const
        self.is_reference = is_reference
        self.is_pointer = is_pointer
        self.is_pointer_const = is_pointer_const

        if self.is_reference and self.is_pointer:
            raise HOGException("References to pointers are not supported.")

    def to_code(
        self, representation: CppFileRepresentation = CppFileRepresentation.HEADER_ONLY
    ) -> str:
        """Returns a C++ string representation."""
        s = []
        if self.is_const:
            s += ["const"]
        s += [self.type]
        if self.is_pointer:
            s += ["*"]
            if self.is_pointer_const:
                s += ["const"]
        if self.is_reference:
            s += ["&"]
        s += [self.name]

        return " ".join(s)

    def __eq__(self, o: Any) -> bool:
        return type(self) == type(o) and (
            self.name,
            self.type,
            self.is_const,
            self.is_reference,
            self.is_pointer,
            self.is_pointer_const,
        ) == (
            o.name,
            o.type,
            o.is_const,
            o.is_reference,
            o.is_pointer,
            o.is_pointer_const,
        )


class CppDefaultArgument:
    """
    Represents a C++ default argument.
    """

    def __init__(self, variable: CppVariable, default_value: str):
        self.variable = variable
        self.default_value = default_value

    def to_code(
        self, representation: CppFileRepresentation = CppFileRepresentation.HEADER_ONLY
    ) -> str:
        """Returns a C++ string representation."""
        if representation in [
            CppFileRepresentation.HEADER_ONLY,
            CppFileRepresentation.HEADER_SPLIT,
        ]:
            return self.variable.to_code() + " = " + self.default_value
        else:
            return self.variable.to_code()


class CppFunction:
    """
    Represents a (free) C++ function (or function template).

    To create methods, please use the corresponding class CppMethod.
    """

    def __init__(self):
        pass


class CppMethod:
    """Represents a C++ method and is used for printing purposes."""

    def __init__(
        self,
        name: str,
        arguments: Optional[List[Union[CppVariable, CppDefaultArgument]]] = None,
        return_type: str = "void",
        is_const: bool = False,
        content: str = "",
        visibility: str = "public",
        docstring: str = "",
    ):
        self.name = name
        self.arguments: List[Union[CppVariable, CppDef]] = deepcopy(arguments)  # type: ignore
        if not self.arguments:
            self.arguments = []
        self.return_type = return_type
        self.is_const = is_const
        self.content = content
        self.visibility = visibility
        self.docstring = docstring

        self.kernel_function = None

    @classmethod
    def from_kernel_function(  # type: ignore[no-untyped-def] # returns Self but Self is kind of new
        cls,
        kernel: pystencils.astnodes.KernelFunction,
        return_type: str = "void",
        is_const: bool = False,
        visibility: str = "public",
        docstring: str = "",
    ):
        """Creates a CppMethod object from a pystencils KernelFunction."""
        # Without this, we cannot generate C++ code from the body :(
        kernel.body.instruction_set = kernel.instruction_set

        return CppMethod(
            name=kernel.function_name,
            arguments=[
                CppVariable(name=p.symbol.name, type=str(p.symbol.dtype))
                for p in kernel.get_parameters()
            ],
            content=generate_c(kernel.body),
            return_type=return_type,
            is_const=is_const,
            visibility=visibility,
            docstring=docstring,
        )


class CppMethodWithVariants:
    """Represents different implementation variants of the same method for different target architectures."""

    def __init__(self, variants: Dict[str, CppMethod]):
        if len(variants) == 0:
            raise HOGException("No variants.")

        v = next(iter(variants.values()))

        self.name = v.name
        self.arguments = v.arguments
        self.return_type = v.return_type
        self.is_const = v.is_const
        self.visibility = v.visibility
        self.docstring = v.docstring

        for _, variant in variants.items():
            if (
                variant.name != self.name
                or variant.arguments != self.arguments
                or variant.return_type != self.return_type
                or variant.is_const != self.is_const
                or variant.visibility != self.visibility
                or variant.docstring != self.docstring
            ):
                print(f"{self.name        = }", file=sys.stderr)
                print(f"{self.arguments   = }", file=sys.stderr)
                print(f"{self.return_type = }", file=sys.stderr)
                print(f"{self.is_const    = }", file=sys.stderr)
                print(f"{self.visibility  = }", file=sys.stderr)
                print(f"{self.docstring   = }", file=sys.stderr)
                print(f"{variant.name        = }", file=sys.stderr)
                print(f"{variant.arguments   = }", file=sys.stderr)
                print(f"{variant.return_type = }", file=sys.stderr)
                print(f"{variant.is_const    = }", file=sys.stderr)
                print(f"{variant.visibility  = }", file=sys.stderr)
                print(f"{variant.docstring   = }", file=sys.stderr)
                raise HOGException(
                    "Variants may only differ in their implementation (contents)."
                )

        self.variants = variants


class CppConstructor:
    """Represents a C++ constructor and is used for printing purposes."""

    def __init__(
        self,
        arguments: Optional[List[CppVariable]] = None,
        initializer_list: Optional[List[str]] = None,
        content: str = "",
        visibility: str = "public",
    ):
        self.arguments: List[CppVariable] = deepcopy(arguments)  # type: ignore
        if not self.arguments:
            self.arguments = []
        self.initializer_list = deepcopy(initializer_list)
        if not self.initializer_list:
            self.initializer_list = []
        self.content = content
        self.visibility = visibility


class CppMemberVariable:
    """Represents a C++ class member variable and is used for printing purposes."""

    def __init__(self, variable: CppVariable, visibility: str = "private"):
        self.variable = variable
        if visibility not in ["public", "protected", "private"]:
            raise HOGException("Visibility not supported.")
        self.visibility = visibility

    def to_code(self):
        return self.variable.to_code() + ";"


class CppComment:
    """Just a C++ comment."""

    def __init__(self, comment: str, where: str = "all"):
        """
        Creates a C++ comment.

        :param comment: any string (does not strictly have to be a comment and can be misused ;))
        :param where:   either "header", "impl", "variant" or "all" - stating where to emit this comment
        """
        self.comment = comment
        self.where = where

        if where not in ["header", "impl", "variant", "all"]:
            raise HOGException("Invalid where parameter in CppComment.")

    def to_code(
        self, representation: CppFileRepresentation = CppFileRepresentation.HEADER_ONLY
    ) -> str:
        if (
            (
                representation
                in [
                    CppFileRepresentation.HEADER_ONLY,
                    CppFileRepresentation.HEADER_SPLIT,
                ]
                and self.where in ["all", "header"]
            )
            or (
                representation == CppFileRepresentation.IMPL_SPLIT
                and self.where in ["all", "impl"]
            )
            or (
                representation == CppFileRepresentation.VARIANT_IMPL_SPLIT
                and self.where in ["all", "variant"]
            )
        ):
            return self.comment
        else:
            return ""


class CppClass:
    """Represents a C++ class (or class template) and is used for printing purposes."""

    def __init__(self, name: str, base_classes: Optional[List[str]] = None):
        self.name = name
        self.base_classes = deepcopy(base_classes)
        if not self.base_classes:
            self.base_classes = []
        self.constructors: List[CppConstructor] = []
        self.methods: List[CppMethod] = []
        self.methods_with_variants: List[CppMethodWithVariants] = []
        self.variables: List[CppMemberVariable] = []
        self.comments: List[CppComment] = []

    def add(
        self,
        class_content: Union[
            CppConstructor,
            CppMethod,
            CppMethodWithVariants,
            CppMemberVariable,
            CppComment,
        ],
    ) -> None:
        if isinstance(class_content, CppConstructor):
            self.constructors.append(class_content)
        elif isinstance(class_content, CppMethod):
            self.methods.append(class_content)
        elif isinstance(class_content, CppMethodWithVariants):
            self.methods_with_variants.append(class_content)
        elif isinstance(class_content, CppMemberVariable):
            self.variables.append(class_content)
        elif isinstance(class_content, CppComment):
            self.comments.append(class_content)
        else:
            raise HOGException("Cannot add the passed content to a class.")

    def to_code(
        self, representation: CppFileRepresentation = CppFileRepresentation.HEADER_ONLY
    ) -> str:
        if representation == CppFileRepresentation.HEADER_ONLY:
            s = []
            for c in self.comments:
                s += [c.to_code(representation=representation)]
            s += [f"class {self.name}"]
            if self.base_classes:
                s += [indent(": " + ", ".join(self.base_classes), INDENT)]
            s += ["{"]

            for visibility in ["public", "protected", "private"]:
                s += [f"{visibility}:"]

                # Constructors
                for constructor in self.constructors:
                    if constructor.visibility == visibility:
                        s += [
                            indent(
                                self.name
                                + "( "
                                + ", ".join(
                                    [
                                        a.to_code(representation=representation)
                                        for a in constructor.arguments
                                    ]
                                )
                                + " )",
                                INDENT,
                            )
                        ]
                        if constructor.initializer_list:
                            s += [
                                indent(
                                    ": " + ", ".join(constructor.initializer_list),
                                    2 * INDENT,
                                )
                            ]
                        s += [indent("{", INDENT)]
                        s += [indent(constructor.content, 2 * INDENT)]
                        s += [indent("}", INDENT)]

                s += [""]

                # Methods
                for method in self.methods:
                    if method.visibility == visibility:
                        s += [method.docstring]
                        s += [
                            INDENT
                            + method.return_type
                            + " "
                            + method.name
                            + "( "
                            + ", ".join(
                                [
                                    a.to_code(representation=representation)
                                    for a in method.arguments
                                ]
                            )
                            + " )"
                            + (" const" if method.is_const else "")
                        ]
                        s += [indent("{", INDENT)]
                        s += [indent(method.content, 2 * INDENT)]
                        s += [indent("}", INDENT)]

                s += [""]

                # Variables
                for variable in self.variables:
                    if variable.visibility == visibility:
                        s += [indent(variable.to_code(), 2 * INDENT)]

                s += [""]

            s += ["};"]
            return "\n".join(s)

        elif representation == CppFileRepresentation.HEADER_SPLIT:
            s = []
            for c in self.comments:
                s += [c.to_code(representation=representation)]
            s += [f"class {self.name}"]
            if self.base_classes:
                s += [indent(": " + ", ".join(self.base_classes), INDENT)]
            s += ["{"]

            for visibility in ["public", "protected", "private"]:
                s += [f"{visibility}:"]

                # Constructors
                for constructor in self.constructors:
                    if constructor.visibility == visibility:
                        s += [
                            indent(
                                self.name
                                + "( "
                                + ", ".join(
                                    [
                                        a.to_code(representation=representation)
                                        for a in constructor.arguments
                                    ]
                                )
                                + " );",
                                INDENT,
                            )
                        ]

                s += [""]

                # Methods
                for method_decl in self.methods + self.methods_with_variants:
                    if method_decl.visibility == visibility:
                        s += [indent(method_decl.docstring, INDENT)]
                        s += [
                            indent(
                                method_decl.return_type
                                + " "
                                + method_decl.name
                                + "( "
                                + ", ".join(
                                    [
                                        a.to_code(representation=representation)
                                        for a in method_decl.arguments
                                    ]
                                )
                                + " )"
                                + (" const" if method_decl.is_const else "")
                                + ";",
                                INDENT,
                            )
                        ]

                s += [""]

                # Variables
                for variable in self.variables:
                    if variable.visibility == visibility:
                        s += [indent(variable.to_code(), INDENT)]

                s += [""]

            s += ["};"]
            return "\n".join(s)

        elif representation == CppFileRepresentation.IMPL_SPLIT:
            s = []

            # Constructors
            for constructor in self.constructors:
                s += [
                    self.name
                    + "::"
                    + self.name
                    + "( "
                    + ", ".join(
                        [
                            a.to_code(representation=representation)
                            for a in constructor.arguments
                        ]
                    )
                    + " )"
                ]
                if constructor.initializer_list:
                    s += [
                        indent(": " + ", ".join(constructor.initializer_list), INDENT)
                    ]
                s += ["{"]
                s += [indent(constructor.content, INDENT)]
                s += ["}"]

            s += [""]

            # Methods
            for method in self.methods:
                s += [
                    method.return_type
                    + " "
                    + self.name
                    + "::"
                    + method.name
                    + "( "
                    + ", ".join(
                        [
                            a.to_code(representation=representation)
                            for a in method.arguments
                        ]
                    )
                    + " )"
                    + (" const" if method.is_const else "")
                ]
                s += ["{"]
                s += [indent(method.content, INDENT)]
                s += ["}"]

            s += [""]

            return "\n".join(s)

        elif representation == CppFileRepresentation.VARIANT_IMPL_SPLIT:
            return ""

        else:
            raise HOGException("Invalid file representation.")


class CppInclude:
    """Just a C++ include statement."""

    def __init__(self, file_to_include: str, quotes: bool = True, where: str = "all"):
        """
        Creates a C++ include statement.

        :param file_to_include: path to file to be included (without quotes or anything)
        :param quotes:          using quotes vs angle brackets
        :param where:           either "header", "impl", "variant" or "all" - stating where to include this file
        """
        self.file_to_include = file_to_include
        self.quotes = quotes
        self.where = where

        if where not in ["header", "impl", "variant", "all"]:
            raise HOGException("Invalid where parameter in CppInclude.")

    def to_code(
        self, representation: CppFileRepresentation = CppFileRepresentation.HEADER_ONLY
    ) -> str:
        ql = '"' if self.quotes else "<"
        qr = '"' if self.quotes else ">"
        include = f"#include {ql}{self.file_to_include}{qr}"

        if (
            (
                representation
                in [
                    CppFileRepresentation.HEADER_ONLY,
                    CppFileRepresentation.HEADER_SPLIT,
                ]
                and self.where in ["all", "header"]
            )
            or (
                representation == CppFileRepresentation.IMPL_SPLIT
                and self.where in ["all", "impl"]
            )
            or (
                representation == CppFileRepresentation.VARIANT_IMPL_SPLIT
                and self.where in ["all", "variant"]
            )
        ):
            return include

        else:
            return ""


class CppFilePair:
    """Represents a C++ file pair (header and implementation)."""

    def __init__(self):
        self.content = []

    def add(
        self, file_content: Union[CppClass, CppFunction, CppComment, CppInclude]
    ) -> None:
        supported_content_types = (CppClass, CppFunction, CppComment, CppInclude)
        if not isinstance(file_content, supported_content_types):
            raise HOGException("Content type not supported.")
        self.content.append(file_content)

    def to_code(
        self, representation: CppFileRepresentation = CppFileRepresentation.HEADER_ONLY
    ) -> str:
        """
        Returns the entire code of this file as a string.

        The actual content depends on the passed representation parameter.

        :param representation: can either be set to
            "HEADER_ONLY"  (returns the entire class 'inline' to be written to a header file)
            "HEADER_SPLIT" (returns the header part of a .hpp/.cpp split version including all methods with and without variants)
            "IMPL_SPLIT"   (returns the implementation part of a .hpp/.cpp split version including only methods without variants)
        """

        s = ""
        for c in self.content:
            s += c.to_code(representation=representation) + "\n\n"
        return s

    def write(self, dir_name: str, file_basename: str) -> None:
        """Writes a header file and all implementations files to disk.

        Creates a header file `dir_name/file_basename.hpp` an implementation
        `dir_name/file_basename.cpp` and an implementation for each variant in
        `dir_name/key/file_basename_variantname.cpp`.
        """
        os.makedirs(dir_name, exist_ok=True)

        # Header file including all methods with/without variants
        header_file_name = os.path.join(dir_name, f"{file_basename}.hpp")
        with open(header_file_name, "w") as f:
            f.write(self.to_code(representation=CppFileRepresentation.HEADER_SPLIT))

        # Main implementation of all methods without variants
        impl_file_name = os.path.join(dir_name, f"{file_basename}.cpp")
        with open(impl_file_name, "w") as f:
            f.write(self.to_code(representation=CppFileRepresentation.IMPL_SPLIT))

        # One additional implementation for each variant
        for cpp_class in filter(lambda c: isinstance(c, CppClass), self.content):
            for method_with_variants in cpp_class.methods_with_variants:
                for key, method in method_with_variants.variants.items():
                    dir = os.path.join(dir_name, key)
                    file_name = os.path.join(dir, f"{file_basename}_{method.name}.cpp")
                    os.makedirs(dir, exist_ok=True)

                    with open(file_name, "w") as f:
                        s = []
                        for c in self.content:
                            if isinstance(c, CppClass):
                                s += [
                                    method.return_type
                                    + " "
                                    + c.name
                                    + "::"
                                    + method.name
                                    + "( "
                                    + ", ".join(
                                        [
                                            a.to_code(
                                                representation=CppFileRepresentation.VARIANT_IMPL_SPLIT
                                            )
                                            for a in method.arguments
                                        ]
                                    )
                                    + " )"
                                    + (" const" if method.is_const else "")
                                ]
                                s += ["{"]
                                s += [indent(method.content, INDENT)]
                                s += ["}"]
                            else:
                                s += [
                                    c.to_code(
                                        representation=CppFileRepresentation.VARIANT_IMPL_SPLIT
                                    )
                                    + "\n"
                                ]

                        f.write("\n".join(s))


def apply_clang_format(
    cpp_file_path: str,
    binary: str,
    style: Optional[str] = None,
    fail_if_no_binary: bool = False,
) -> None:
    """
    Applies clang-format to the passed C++ source file. Modifies the contents in-place.

    :param cpp_file_path:     path to the cpp file that shall be formatted
    :param style:             None or one of [LLVM, GNU, Google, Chromium, Microsoft, Mozilla, WebKit] - if None,
        clang-format looks for a .clang-format config file in parent directories and applies that if found, otherwise
        it defaults to some default style
    :param binary:            the name and/or path of the clang-format binary
    :param fail_if_no_binary: if True, this function raises an exception if the clang-format binary does not exist,
        otherwise just exists and does nothing if no binary is available
    """

    if not os.path.exists(cpp_file_path):
        raise HOGException("Cannot apply clang-format to non-existing file.")

    default_styles = [
        "LLVM",
        "GNU",
        "Google",
        "Chromium",
        "Microsoft",
        "Mozilla",
        "WebKit",
    ]

    if not shutil.which(binary):
        if not fail_if_no_binary:
            return
        else:
            raise HOGException(f"Could not find clang-format binary '{binary}'.")

    cmd = [binary, "-i", cpp_file_path]

    if style in default_styles:
        cmd += [f"--style={style}"]
    elif style is not None:
        raise HOGException(
            f"Invalid clang-format style. Should either be one of {default_styles} or None."
        )

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        raise HOGException(
            f"clang-format exited with errors:\nstdout:\n{result.stdout.decode()}\nstderr:\n{result.stderr.decode()}"
        )
