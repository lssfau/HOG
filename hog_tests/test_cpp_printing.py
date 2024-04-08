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

from hog.cpp_printing import (
    CppClass,
    CppFunction,
    CppConstructor,
    CppMethod,
    CppMemberVariable,
    CppFilePair,
    CppFileRepresentation,
    CppVariable,
)
from hog.exception import HOGException


def test_cpp_printer():
    c = CppClass(name="MyOperatorClass")

    # Adding some member variables

    var_1 = CppVariable("someInt_", "int")
    mvar_1 = CppMemberVariable(var_1)

    var_2 = CppVariable("storage_", "std::shared_ptr< PrimitiveStorage >")
    mvar_2 = CppMemberVariable(var_2, visibility="public")

    c.add(mvar_1)
    c.add(mvar_2)

    # Adding a constructor

    ctor_1 = CppConstructor(
        arguments=[
            CppVariable("storage", "PrimitiveStorage", is_const=True, is_reference=True)
        ],
        initializer_list=[f"{var_2.name}( storage )", f"{var_1.name}( 0 )"],
        content="",
    )

    c.add(ctor_1)

    # Adding a method

    method_1 = CppMethod("apply", content=" // some computation ")

    c.add(method_1)

    file = CppFilePair()
    file.add(c)

    print()
    print()
    s = file.to_code()
    print(s)
