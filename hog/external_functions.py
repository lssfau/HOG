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

from hog.multi_assignment import MultiAssignment, Member


class BlendingFTriangle(MultiAssignment):
    def function_name(self):
        return "Blending_F_Triangle"

    def num_input_args(self):
        return 2

    def num_output_args(self):
        return 2

    def implementation(self):
        return """Point3D  in( {in_0, in_1, 0} );
Point3D out;
geometryMap_->evalF( in, out );
*out_0 = out[0];
*out_1 = out[1];"""


class BlendingFEmbeddedTriangle(MultiAssignment):
    def function_name(self):
        return "Blending_F_EmbeddedTriangle"

    def num_input_args(self):
        return 3

    def num_output_args(self):
        return 3

    def implementation(self):
        return """Point3D  in( {in_0, in_1, in_2} );
Point3D out;
geometryMap_->evalF( in, out );
*out_0 = out[0];
*out_1 = out[1];
*out_2 = out[2];"""


class BlendingFTetrahedron(MultiAssignment):
    def function_name(self):
        return "Blending_F_Tetrahedron"

    def num_input_args(self):
        return 3

    def num_output_args(self):
        return 3

    def implementation(self):
        return """Point3D  in( {in_0, in_1, in_2} );
Point3D out;
geometryMap_->evalF( in, out );
*out_0 = out[0];
*out_1 = out[1];
*out_2 = out[2];"""


class BlendingDFTriangle(MultiAssignment):
    def function_name(self):
        return "Blending_DF_Triangle"

    @classmethod
    def num_input_args(cls):
        return 2

    @classmethod
    def num_output_args(cls):
        return 2 * 2

    def implementation(self):
        return """Point3D  mappedPt( {in_0, in_1, 0} );
Matrix2r DPsi;
geometryMap_->evalDF( mappedPt, DPsi );
*out_0 = DPsi( 0, 0 );
*out_1 = DPsi( 0, 1 );
*out_2 = DPsi( 1, 0 );
*out_3 = DPsi( 1, 1 );"""


class BlendingDFEmbeddedTriangle(MultiAssignment):
    def function_name(self):
        return "Blending_DF_EmbeddedTriangle"

    @classmethod
    def num_input_args(cls):
        return 3

    @classmethod
    def num_output_args(cls):
        return 3 * 3

    def implementation(self):
        return """Point3D  mappedPt( {in_0, in_1, in_2} );
Matrix3r DPsi;
geometryMap_->evalDF( mappedPt, DPsi );
*out_0 = DPsi( 0, 0 );
*out_1 = DPsi( 0, 1 );
*out_2 = DPsi( 0, 2 );
*out_3 = DPsi( 1, 0 );
*out_4 = DPsi( 1, 1 );
*out_5 = DPsi( 1, 2 );
*out_6 = DPsi( 2, 0 );
*out_7 = DPsi( 2, 1 );
*out_8 = DPsi( 2, 2 );"""


class BlendingDFInvDFTriangle(MultiAssignment):
    def function_name(self):
        return "Blending_DFInvDF_Triangle"

    @classmethod
    def num_input_args(cls):
        return 2

    @classmethod
    def num_output_args(cls):
        return 2 * 2 * 2

    def implementation(self):
        return """Point3D  mappedPt( {in_0, in_1, 0} );
Matrixr< 2, 4 > DInvDPsi;
geometryMap_->evalDFinvDF( mappedPt, DInvDPsi );
*out_0 = DInvDPsi( 0, 0 );
*out_1 = DInvDPsi( 0, 1 );
*out_2 = DInvDPsi( 0, 2 );
*out_3 = DInvDPsi( 0, 3 );
*out_4 = DInvDPsi( 1, 0 );
*out_5 = DInvDPsi( 1, 1 );
*out_6 = DInvDPsi( 1, 2 );
*out_7 = DInvDPsi( 1, 3 );"""


class BlendingDFTetrahedron(MultiAssignment):
    def function_name(self):
        return "Blending_DF_Tetrahedron"

    @classmethod
    def num_input_args(cls):
        return 3

    @classmethod
    def num_output_args(cls):
        return 3 * 3

    def implementation(self):
        return """Point3D  mappedPt( {in_0, in_1, in_2} );
Matrix3r DPsi;
geometryMap_->evalDF( mappedPt, DPsi );
*out_0 = DPsi( 0, 0 );
*out_1 = DPsi( 0, 1 );
*out_2 = DPsi( 0, 2 );
*out_3 = DPsi( 1, 0 );
*out_4 = DPsi( 1, 1 );
*out_5 = DPsi( 1, 2 );
*out_6 = DPsi( 2, 0 );
*out_7 = DPsi( 2, 1 );
*out_8 = DPsi( 2, 2 );"""


class ScalarVariableCoefficient2D(MultiAssignment):
    def function_name(self):
        return "Scalar_Variable_Coefficient_2D"

    @classmethod
    def num_input_args(cls):
        return 2

    @classmethod
    def num_output_args(cls):
        return 1

    def implementation(self):
        return f"*out_0 = {self._callback_name()}( Point3D( {{in_0, in_1, 0}} ) );"

    def _callback_name(self):
        return f"callback_{self.function_name()}_{self.variable_name()}"

    def members(self):
        return [
            Member(
                f"_{self._callback_name()}",
                f"{self._callback_name()}",
                "std::function< real_t ( const Point3D & ) >",
            )
        ]


class ScalarVariableCoefficient3D(MultiAssignment):
    def function_name(self):
        return "Scalar_Variable_Coefficient_3D"

    @classmethod
    def num_input_args(cls):
        return 3

    @classmethod
    def num_output_args(cls):
        return 1

    def implementation(self):
        return f"*out_0 = {self._callback_name()}( Point3D( {{in_0, in_1, in_2}} ) );"

    def _callback_name(self):
        return f"callback_{self.function_name()}_{self.variable_name()}"

    def members(self):
        return [
            Member(
                f"_{self._callback_name()}",
                f"{self._callback_name()}",
                "std::function< real_t ( const Point3D & ) >",
            )
        ]


class VectorVariableCoefficient3D(MultiAssignment):
    def function_name(self):
        return "Vector_Variable_Coefficient_3D"

    @classmethod
    def num_input_args(cls):
        return 3

    @classmethod
    def num_output_args(cls):
        return 3

    def implementation(self):
        return f"""Point3D {self._result_name()} = {self._callback_name()}( Point3D{{ in_0, in_1, in_2 }} );
*out_0 = {self._result_name()}[0];
*out_1 = {self._result_name()}[1];
*out_2 = {self._result_name()}[2];"""

    def _callback_name(self):
        return f"callback_{self.function_name()}_{self.variable_name()}"

    def _result_name(self):
        return f"result_{self.function_name()}_{self.variable_name()}"

    def members(self):
        return [
            Member(
                f"_{self._callback_name()}",
                f"{self._callback_name()}",
                "std::function< Point3D ( const Point3D & ) >",
            )
        ]
