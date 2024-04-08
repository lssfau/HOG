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

from argparse import Namespace
from enum import Enum
from typing import Union

import numpy as np
from numpy.typing import DTypeLike

from pystencils.typing.types import BasicType


class HOGPrecision(Enum):
    """
    This enum represents all floating-point types that are currently supported and used by HyTeG.
    It is used as an argument in the function 'hyteg_type' which constructs HOGType objects.
    """

    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"
    REAL_T = "real_t"


# TODO Since right now this class only stores a Basic type. It's actually not needed anymore. Think about replacing it.
class HOGType:
    """
    This class handles the types supported by HyTeG and the variables and objects needed by the generator.

     Attributes:
        pystencils_type  Is the Pystencils BasicType that holds the information 'numpy_dtype', 'const' and 'identifier'.
        add_file_suffix  Stores whether the type should be added as a suffix to the filename.

    Params:
        dtype_identifier This is the data type identifier. It is a 'str' value that should later be used by the printer.
                             (Right now, this is also the value passed to pystencils
                             to determine the default floating point type.
                             Within pystencils it converted to a np.type.
                             So right now, only identifiers that can be understood by numpy can be used.)
        ptype            Stands for 'python type', which is used in pystencils to better work with types.
                         For some optimizations, python has to know to identify the precision.
        const            Stores whether it's a const type or not.
        add_file_suffix  If set to yes, the created operator will have the type_name in its name.

    Default: If nothing is specified, double is used as a default.
    """

    def __init__(
        self,
        dtype_identifier: Union[str, None],
        ptype: DTypeLike,
        const: bool = False,
        add_file_suffix: bool = False,
    ):
        self.pystencils_type = BasicType(
            dtype=ptype, const=const, identifier=dtype_identifier
        )
        self.add_file_suffix = add_file_suffix

    def __str__(self) -> str:
        return f'{self.pystencils_type.identifier.split("::")[-1]}'


def hyteg_type(type_name: HOGPrecision = HOGPrecision.FP64) -> HOGType:
    """
    This method can be used to construct an HOGType object, i.e. a 'type_descriptor' of a HyTeG type.
    """
    prefix = "walberla::"
    if type_name == HOGPrecision.FP16:
        return HOGType(
            dtype_identifier=f"{prefix}float16",
            ptype=np.float16,
            add_file_suffix=True,
        )
    elif type_name == HOGPrecision.FP32:
        return HOGType(
            dtype_identifier=f"{prefix}float32",
            ptype=np.float32,
            add_file_suffix=True,
        )
    elif type_name == HOGPrecision.FP64:
        return HOGType(
            dtype_identifier=f"{prefix}float64",
            ptype=np.float64,
            add_file_suffix=True,
        )
    elif type_name == HOGPrecision.REAL_T:
        return HOGType(
            dtype_identifier=f"real_t",
            ptype=np.float64,
            add_file_suffix=True,
        )
    else:
        raise ValueError(
            f"Something went wrong, "
            f"the given precision '{type_name.value}' can't be chosen since it is not implemented yet.\n"
            f"Supported types are {{{list(choice.value for choice in HOGPrecision)}}}"
        )


def parse_argument_type(args: Namespace) -> HOGType:
    """Since for the argument parser, it is easier to use strings instead of enums,
    this function is used to parse the input arguments and map the given strings to the enum HOGPrecision.
    """
    if args.precision:
        type_name = args.precision
        if type_name == "fp16":
            return hyteg_type(HOGPrecision.FP16)
        if type_name == "fp32":
            return hyteg_type(HOGPrecision.FP32)
        if type_name == "fp64":
            return hyteg_type(HOGPrecision.FP64)
        if type_name == "real_t":
            return hyteg_type(HOGPrecision.REAL_T)
        else:
            raise ValueError(
                f"Something went wrong, "
                f"the given precision string '{type_name}' can't be interpreted as a member of the HOGPrecision enum.\n"
                f"Supported precisions are {{{list(choice.value for choice in HOGPrecision)}}}"
            )
    else:
        # When no precision was defined, the HyTeG default is chosen.
        return hyteg_type()
