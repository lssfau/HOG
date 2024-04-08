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

COPYRIGHT_COMMENT = f"""/*
* Copyright (c) 2017-2024 Nils Kohl, Daniel Bauer, Fabian Böhm.
*
* This file is part of HyTeG
* (see https://i10git.cs.fau.de/hyteg/hyteg).
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/"""

GENERATED_COMMENT = f"""/*
* The entire file was generated with the HyTeG Operator Generator.
*
* Avoid modifying this file. If buggy, consider fixing the generator itself.
*/"""

PRAGMA_ONCE = "#pragma once"

GCC_WARNING_WORKAROUND = (
    f"// Unfortunately, the inverse diagonal kernel wrapper triggers a GCC bug (maybe\n"
    f"// (related to) https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107087) causing a\n"
    f"// warning in an internal standard library header (bits/stl_algobase.h). As a\n"
    f"// workaround, we disable the warning and include this header indirectly through\n"
    f"// a public header.\n"
    f"#include <waLBerlaDefinitions.h>\n"
    f"#ifdef WALBERLA_CXX_COMPILER_IS_GNU\n"
    f"#pragma GCC diagnostic push\n"
    f'#pragma GCC diagnostic ignored "-Wnonnull"\n'
    f"#endif\n"
    f"#include <cmath>\n"
    f"#ifdef WALBERLA_CXX_COMPILER_IS_GNU\n"
    f"#pragma GCC diagnostic pop\n"
    f"#endif"
)
