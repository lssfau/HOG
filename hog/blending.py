# HyTeG Operator Generator
# Copyright (C) 2024-2025  HyTeG Team
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
from typing import Any, List
from hog.exception import HOGException
from hog.element_geometry import (
    ElementGeometry,
    LineElement,
    TriangleElement,
    TetrahedronElement,
)


class GeometryMap:
    """
    Superclass for geometry maps to apply "blending".

    Instances of subclasses can be used to compile a blending function directly "into" the operator.
    """

    def supported_geometries(self) -> List[ElementGeometry]:
        """Returns the element types that the map can be executed on."""
        return []

    def is_affine(self) -> bool:
        """Shall return true if the map is an affine mapping. Certain critical optimizations depend on this.
        Returns false by default - this does not alter accuracy. Better overwrite when true!
        """
        return False

    def evaluate(self, x: sp.Matrix) -> sp.Matrix:
        """Evaluates the geometry map at the passed point."""
        raise HOGException("evaluate() not implemented for this map.")

    def jacobian(self, x: sp.Matrix) -> sp.Matrix:
        """Evaluates the Jacobian of the geometry map at the passed point."""
        raise HOGException("jacobian() not implemented for this map.")

    def hessian(self, x: sp.Matrix) -> List[sp.Matrix]:
        """Evaluates the hessian of the geometry map at the passed point."""
        raise HOGException("hessian() not implemented for this map.")

    def coupling_includes(self) -> List[str]:
        """Returns a list of files that better be included into the C++ files when this map is used."""
        raise HOGException("coupling_includes() not implemented for this map.")

    def parameter_coupling_code(self) -> str:
        """Returns the code that is required to retrieve all parameters from the C++ GeometryMap."""
        raise HOGException("parameter_coupling_code() not implemented for this map.")

    def __str__(self):
        return self.__class__.__name__


class IdentityMap(GeometryMap):
    """
    Just the identity. Nothing happens.
    """

    def supported_geometries(self) -> List[ElementGeometry]:
        return [LineElement(), TriangleElement(), TetrahedronElement()]

    def is_affine(self) -> bool:
        return True

    def evaluate(self, x: sp.Matrix) -> sp.Matrix:
        return x

    def jacobian(self, x: sp.Matrix) -> sp.Matrix:
        return sp.eye(len(x))

    def coupling_includes(self) -> List[str]:
        return []

    def parameter_coupling_code(self) -> str:
        return ""

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other)


class ExternalMap(GeometryMap):
    """This is a special map that indicates that the actual map is defined externally (through a C++ class in HyTeG)."""

    def supported_geometries(self) -> List[ElementGeometry]:
        return [LineElement(), TriangleElement(), TetrahedronElement()]


class ParametricMap(GeometryMap):
    """
    This is a special map that indicates that you want a parametric mapping.

    It uses HyTeG's MicroMesh implementation and introduces a vector finite element coefficient that represents the
    coordinates at each node. Locally, the Jacobians are computed through the gradients of the shape functions.

    Depending on your choice of polynomial degree, you can use this blending map to construct sub-, super-, and
    isoparametric mappings.

    The affine Jacobians in all integrands will automatically be set to the identity if you use this map.
    The blending Jacobians will be set to the transpose of the gradient of the shape functions.
    """

    def __init__(self, degree: int):
        if degree not in [1, 2]:
            raise HOGException(
                "Only first and second order parametric maps are supported."
            )

        self.degree = degree

    def supported_geometries(self) -> List[ElementGeometry]:
        return [TriangleElement(), TetrahedronElement()]

    def parameter_coupling_code(self) -> str:
        return ""

    def coupling_includes(self) -> List[str]:
        return [
            "hyteg/p1functionspace/P1VectorFunction.hpp",
            "hyteg/p2functionspace/P2VectorFunction.hpp",
        ]

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.degree == other.degree

    def __str__(self):
        return self.__class__.__name__ + f"P{self.degree}"


class AnnulusMap(GeometryMap):
    """Projects HyTeG's approximate annulus to the actual curved geometry.

    This will ONLY work with the full operator generation. For forms, use the ExternalMap class and HyTeG's geometry maps.
    """

    def __init__(self):
        self.radRefVertex = sp.Symbol("radRefVertex")
        self.radRayVertex = sp.Symbol("radRayVertex")
        self.refVertex = sp.Matrix([["refVertex_0"], ["refVertex_1"]])
        self.rayVertex = sp.Matrix([["rayVertex_0"], ["rayVertex_1"]])
        self.thrVertex = sp.Matrix([["thrVertex_0"], ["thrVertex_1"]])

    def supported_geometries(self) -> List[ElementGeometry]:
        return [TriangleElement()]

    def evaluate(self, x: sp.Matrix) -> sp.Matrix:
        """Evaluates the geometry map at the passed point."""

        radRefVertex = self.radRefVertex
        radRayVertex = self.radRayVertex
        refVertex = self.refVertex
        rayVertex = self.rayVertex
        thrVertex = self.thrVertex

        areaT = (refVertex[0] - rayVertex[0]) * (thrVertex[1] - rayVertex[1]) - (
            refVertex[1] - rayVertex[1]
        ) * (thrVertex[0] - rayVertex[0])

        areaX = (x[0] - rayVertex[0]) * (thrVertex[1] - rayVertex[1]) - (
            x[1] - rayVertex[1]
        ) * (thrVertex[0] - rayVertex[0])

        factor = areaX / areaT

        oldRad = sp.sqrt(x[0] * x[0] + x[1] * x[1])

        newRad = radRayVertex + factor * (radRefVertex - radRayVertex)

        xnew = sp.zeros(2, 1)
        xnew[0] = x[0] * newRad / oldRad
        xnew[1] = x[1] * newRad / oldRad
        return xnew

    def jacobian(self, x: sp.Matrix) -> sp.Matrix:
        """Evaluates the Jacobian of the geometry map at the passed point."""

        if sp.shape(x) != (2, 1):
            raise HOGException(f"Invalid input shape {sp.shape(x)} for AnnulusMap.")

        radRefVertex = self.radRefVertex
        radRayVertex = self.radRayVertex
        refVertex = self.refVertex
        rayVertex = self.rayVertex
        thrVertex = self.thrVertex

        dist = radRefVertex - radRayVertex
        areaT = (refVertex[0] - rayVertex[0]) * (thrVertex[1] - rayVertex[1]) - (
            refVertex[1] - rayVertex[1]
        ) * (thrVertex[0] - rayVertex[0])
        areaX = (x[0] - rayVertex[0]) * (thrVertex[1] - rayVertex[1]) - (
            x[1] - rayVertex[1]
        ) * (thrVertex[0] - rayVertex[0])
        bary = areaX / areaT
        oldRad = sp.sqrt(x[0] * x[0] + x[1] * x[1])
        newRad = radRayVertex + bary * dist

        invNorm = 1.0 / oldRad
        invNorm3 = invNorm * invNorm * invNorm
        tmp0 = invNorm * dist / areaT
        tmp1 = x[0] * tmp0
        tmp2 = x[1] * tmp0
        tmp3 = thrVertex[1] - rayVertex[1]
        tmp4 = thrVertex[0] - rayVertex[0]
        tmp5 = x[0] * invNorm3 * newRad
        tmp6 = x[1] * invNorm3 * newRad

        jac = sp.Matrix(
            [
                [x[1] * tmp6 + tmp1 * tmp3, -x[0] * tmp6 - tmp1 * tmp4],
                [-x[1] * tmp5 + tmp2 * tmp3, x[0] * tmp5 - tmp2 * tmp4],
            ]
        )

        return jac

    def hessian(self, x: sp.Matrix) -> List[sp.Matrix]:
        """Evaluates the derivatives of the inverse Jacobian matrix of the geometry map at the passed point."""

        if sp.shape(x) != (2, 1):
            raise HOGException("Invalid input shape for AnnulusMap.")

        radRefVertex = self.radRefVertex
        radRayVertex = self.radRayVertex
        refVertex = self.refVertex
        rayVertex = self.rayVertex
        thrVertex = self.thrVertex

        xAnnulus, yAnnulus = sp.symbols("x y")

        dist = radRefVertex - radRayVertex
        areaT = (refVertex[0] - rayVertex[0]) * (thrVertex[1] - rayVertex[1]) - (
            refVertex[1] - rayVertex[1]
        ) * (thrVertex[0] - rayVertex[0])
        areaX = (xAnnulus - rayVertex[0]) * (thrVertex[1] - rayVertex[1]) - (
            yAnnulus - rayVertex[1]
        ) * (thrVertex[0] - rayVertex[0])
        bary = areaX / areaT
        oldRad = sp.sqrt(xAnnulus * xAnnulus + yAnnulus * yAnnulus)
        newRad = radRayVertex + bary * dist

        invNorm = 1.0 / oldRad
        invNorm3 = invNorm * invNorm * invNorm
        tmp0 = invNorm * dist / areaT
        tmp1 = xAnnulus * tmp0
        tmp2 = yAnnulus * tmp0
        tmp3 = thrVertex[1] - rayVertex[1]
        tmp4 = thrVertex[0] - rayVertex[0]
        tmp5 = xAnnulus * invNorm3 * newRad
        tmp6 = yAnnulus * invNorm3 * newRad

        jac = sp.Matrix(
            [
                [yAnnulus * tmp6 + tmp1 * tmp3, -xAnnulus * tmp6 - tmp1 * tmp4],
                [-yAnnulus * tmp5 + tmp2 * tmp3, xAnnulus * tmp5 - tmp2 * tmp4],
            ]
        ).T

        hess = [
            sp.diff(jac, xAnnulus).subs([(xAnnulus, x[0]), (yAnnulus, x[1])]),
            sp.diff(jac, yAnnulus).subs([(xAnnulus, x[0]), (yAnnulus, x[1])]),
        ]

        return hess

    def coupling_includes(self) -> List[str]:
        return ["hyteg/geometry/AnnulusMap.hpp"]

    def parameter_coupling_code(self) -> str:
        code = []

        code += [
            f"WALBERLA_CHECK_NOT_NULLPTR( std::dynamic_pointer_cast< AnnulusMap >( face.getGeometryMap() ),"
            f'"This operator requires the AnnulusMap to be registered as GeometryMap on every macro-cell." )'
        ]

        code += [
            f"real_t {self.radRefVertex} = std::dynamic_pointer_cast< AnnulusMap >( face.getGeometryMap() )->radRefVertex();",
            f"real_t {self.radRayVertex} = std::dynamic_pointer_cast< AnnulusMap >( face.getGeometryMap() )->radRayVertex();",
        ]

        for i in range(2):
            code += [
                f"real_t {self.refVertex[i]} = std::dynamic_pointer_cast< AnnulusMap >( face.getGeometryMap() )->refVertex()[{i}];",
                f"real_t {self.rayVertex[i]} = std::dynamic_pointer_cast< AnnulusMap >( face.getGeometryMap() )->rayVertex()[{i}];",
                f"real_t {self.thrVertex[i]} = std::dynamic_pointer_cast< AnnulusMap >( face.getGeometryMap() )->thrVertex()[{i}];",
            ]

        return "\n".join(code)


class IcosahedralShellMap(GeometryMap):
    """Projects HyTeG's approximate thick spherical shell to the actual curved geometry.

    This will ONLY work with the full operator generation. For forms, use the ExternalMap class and HyTeG's geometry maps.
    """

    def __init__(self):
        self.radRefVertex = sp.Symbol("radRefVertex")
        self.radRayVertex = sp.Symbol("radRayVertex")
        self.refVertex = sp.Matrix([["refVertex_0"], ["refVertex_1"], ["refVertex_2"]])
        self.rayVertex = sp.Matrix([["rayVertex_0"], ["rayVertex_1"], ["rayVertex_2"]])
        self.thrVertex = sp.Matrix([["thrVertex_0"], ["thrVertex_1"], ["thrVertex_2"]])
        self.forVertex = sp.Matrix([["forVertex_0"], ["forVertex_1"], ["forVertex_2"]])

    def supported_geometries(self) -> List[ElementGeometry]:
        return [TetrahedronElement()]

    def evaluate(self, x: sp.Matrix) -> sp.Matrix:
        """Evaluates the geometry map at the passed point."""

        radRefVertex = self.radRefVertex
        radRayVertex = self.radRayVertex
        refVertex = self.refVertex
        rayVertex = self.rayVertex
        thrVertex = self.thrVertex
        forVertex = self.forVertex

        tmp0 = -rayVertex[2]
        tmp1 = refVertex[2] + tmp0
        tmp2 = -rayVertex[0]
        tmp3 = thrVertex[0] + tmp2
        tmp4 = -rayVertex[1]
        tmp5 = forVertex[1] + tmp4
        tmp6 = tmp3 * tmp5
        tmp7 = refVertex[1] + tmp4
        tmp8 = thrVertex[2] + tmp0
        tmp9 = forVertex[0] + tmp2
        tmp10 = tmp8 * tmp9
        tmp11 = refVertex[0] + tmp2
        tmp12 = thrVertex[1] + tmp4
        tmp13 = forVertex[2] + tmp0
        tmp14 = tmp12 * tmp13
        tmp15 = tmp13 * tmp3
        tmp16 = tmp12 * tmp9
        tmp17 = tmp5 * tmp8
        tmp18 = tmp0 + x[2]
        tmp19 = tmp4 + x[1]
        tmp20 = tmp2 + x[0]

        volT = (
            -tmp1 * tmp16
            + tmp1 * tmp6
            + tmp10 * tmp7
            + tmp11 * tmp14
            - tmp11 * tmp17
            - tmp15 * tmp7
        )

        volX = (
            tmp10 * tmp19
            + tmp14 * tmp20
            - tmp15 * tmp19
            - tmp16 * tmp18
            - tmp17 * tmp20
            + tmp18 * tmp6
        )

        bary = sp.Abs(volX / volT)

        oldRad = sp.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])

        newRad = radRayVertex + bary * (radRefVertex - radRayVertex)

        xnew = sp.zeros(3, 1)
        xnew[0] = x[0] * newRad / oldRad
        xnew[1] = x[1] * newRad / oldRad
        xnew[2] = x[2] * newRad / oldRad

        return xnew

    def jacobian(self, x: sp.Matrix) -> sp.Matrix:
        """Evaluates the Jacobian of the geometry map at the passed point."""

        if sp.shape(x) != (3, 1):
            raise HOGException("Invalid input shape for IcosahedralShellMap.")

        radRefVertex = self.radRefVertex
        radRayVertex = self.radRayVertex
        refVertex = self.refVertex
        rayVertex = self.rayVertex
        thrVertex = self.thrVertex
        forVertex = self.forVertex

        tmp0 = x[0] * x[0]
        tmp1 = rayVertex[2] - refVertex[2]
        tmp2 = rayVertex[0] - thrVertex[0]
        tmp3 = rayVertex[1] - forVertex[1]
        tmp4 = tmp2 * tmp3
        tmp5 = rayVertex[1] - refVertex[1]
        tmp6 = rayVertex[0] - forVertex[0]
        tmp7 = rayVertex[2] - thrVertex[2]
        tmp8 = tmp6 * tmp7
        tmp9 = rayVertex[0] - refVertex[0]
        tmp10 = rayVertex[1] - thrVertex[1]
        tmp11 = rayVertex[2] - forVertex[2]
        tmp12 = tmp10 * tmp11
        tmp13 = tmp11 * tmp2
        tmp14 = tmp10 * tmp6
        tmp15 = tmp3 * tmp7
        tmp16 = (
            -tmp1 * tmp14
            + tmp1 * tmp4
            + tmp12 * tmp9
            - tmp13 * tmp5
            - tmp15 * tmp9
            + tmp5 * tmp8
        )
        tmp17 = radRayVertex - radRefVertex
        tmp18 = rayVertex[2] - x[2]
        tmp19 = rayVertex[1] - x[1]
        tmp20 = rayVertex[0] - x[0]
        tmp21 = radRayVertex * tmp16 - tmp17 * (
            tmp12 * tmp20
            - tmp13 * tmp19
            - tmp14 * tmp18
            - tmp15 * tmp20
            + tmp18 * tmp4
            + tmp19 * tmp8
        )
        tmp22 = x[1] * x[1]
        tmp23 = x[2] * x[2]
        tmp24 = tmp0 + tmp22 + tmp23
        tmp25 = tmp17 * (tmp12 - tmp15)
        tmp26 = 1.0 / (tmp16 * tmp24 ** (sp.Rational(3, 2)))
        tmp27 = tmp13 - tmp8
        tmp28 = tmp17 * tmp24
        tmp29 = tmp21 * x[1] + tmp27 * tmp28
        tmp30 = tmp26 * x[0]
        tmp31 = -tmp14 + tmp4
        tmp32 = -tmp21 * x[2] + tmp28 * tmp31
        tmp33 = -tmp21 * x[0] + tmp24 * tmp25
        tmp34 = tmp26 * x[1]
        tmp35 = tmp26 * x[2]

        jac = sp.Matrix(
            [
                [
                    tmp26 * (-tmp0 * tmp21 + tmp24 * (tmp21 + tmp25 * x[0])),
                    -tmp29 * tmp30,
                    tmp30 * tmp32,
                ],
                [
                    tmp33 * tmp34,
                    tmp26 * (-tmp21 * tmp22 + tmp24 * (-tmp17 * tmp27 * x[1] + tmp21)),
                    tmp32 * tmp34,
                ],
                [
                    tmp33 * tmp35,
                    -tmp29 * tmp35,
                    tmp26 * (-tmp21 * tmp23 + tmp24 * (tmp17 * tmp31 * x[2] + tmp21)),
                ],
            ]
        )

        return jac

    def hessian(self, x_: sp.Matrix) -> List[sp.Matrix]:
        """Evaluates the derivatives of the inverse Jacobian matrix of the geometry map at the passed point."""

        if sp.shape(x_) != (3, 1):
            raise HOGException("Invalid input shape for IcosahedralShellMap.")

        xAnnulus, yAnnulus, zAnnulus = sp.symbols("x y z")

        x = [xAnnulus, yAnnulus, zAnnulus]

        radRefVertex = self.radRefVertex
        radRayVertex = self.radRayVertex
        refVertex = self.refVertex
        rayVertex = self.rayVertex
        thrVertex = self.thrVertex
        forVertex = self.forVertex

        tmp0 = x[0] * x[0]
        tmp1 = rayVertex[2] - refVertex[2]
        tmp2 = rayVertex[0] - thrVertex[0]
        tmp3 = rayVertex[1] - forVertex[1]
        tmp4 = tmp2 * tmp3
        tmp5 = rayVertex[1] - refVertex[1]
        tmp6 = rayVertex[0] - forVertex[0]
        tmp7 = rayVertex[2] - thrVertex[2]
        tmp8 = tmp6 * tmp7
        tmp9 = rayVertex[0] - refVertex[0]
        tmp10 = rayVertex[1] - thrVertex[1]
        tmp11 = rayVertex[2] - forVertex[2]
        tmp12 = tmp10 * tmp11
        tmp13 = tmp11 * tmp2
        tmp14 = tmp10 * tmp6
        tmp15 = tmp3 * tmp7
        tmp16 = (
            -tmp1 * tmp14
            + tmp1 * tmp4
            + tmp12 * tmp9
            - tmp13 * tmp5
            - tmp15 * tmp9
            + tmp5 * tmp8
        )
        tmp17 = radRayVertex - radRefVertex
        tmp18 = rayVertex[2] - x[2]
        tmp19 = rayVertex[1] - x[1]
        tmp20 = rayVertex[0] - x[0]
        tmp21 = radRayVertex * tmp16 - tmp17 * (
            tmp12 * tmp20
            - tmp13 * tmp19
            - tmp14 * tmp18
            - tmp15 * tmp20
            + tmp18 * tmp4
            + tmp19 * tmp8
        )
        tmp22 = x[1] * x[1]
        tmp23 = x[2] * x[2]
        tmp24 = tmp0 + tmp22 + tmp23
        tmp25 = tmp17 * (tmp12 - tmp15)
        tmp26 = 1.0 / (tmp16 * tmp24 ** (sp.Rational(3, 2)))
        tmp27 = tmp13 - tmp8
        tmp28 = tmp17 * tmp24
        tmp29 = tmp21 * x[1] + tmp27 * tmp28
        tmp30 = tmp26 * x[0]
        tmp31 = -tmp14 + tmp4
        tmp32 = -tmp21 * x[2] + tmp28 * tmp31
        tmp33 = -tmp21 * x[0] + tmp24 * tmp25
        tmp34 = tmp26 * x[1]
        tmp35 = tmp26 * x[2]

        jac = sp.Matrix(
            [
                [
                    tmp26 * (-tmp0 * tmp21 + tmp24 * (tmp21 + tmp25 * x[0])),
                    -tmp29 * tmp30,
                    tmp30 * tmp32,
                ],
                [
                    tmp33 * tmp34,
                    tmp26 * (-tmp21 * tmp22 + tmp24 * (-tmp17 * tmp27 * x[1] + tmp21)),
                    tmp32 * tmp34,
                ],
                [
                    tmp33 * tmp35,
                    -tmp29 * tmp35,
                    tmp26 * (-tmp21 * tmp23 + tmp24 * (tmp17 * tmp31 * x[2] + tmp21)),
                ],
            ]
        ).T

        hess = [
            sp.diff(jac, xAnnulus).subs(
                [(xAnnulus, x_[0]), (yAnnulus, x_[1]), (zAnnulus, x_[2])]
            ),
            sp.diff(jac, yAnnulus).subs(
                [(xAnnulus, x_[0]), (yAnnulus, x_[1]), (zAnnulus, x_[2])]
            ),
            sp.diff(jac, zAnnulus).subs(
                [(xAnnulus, x_[0]), (yAnnulus, x_[1]), (zAnnulus, x_[2])]
            ),
        ]

        return hess

    def coupling_includes(self) -> List[str]:
        return ["hyteg/geometry/IcosahedralShellMap.hpp"]

    def parameter_coupling_code(self) -> str:
        code = []

        code += [
            f"WALBERLA_CHECK_NOT_NULLPTR( std::dynamic_pointer_cast< IcosahedralShellMap >( cell.getGeometryMap() ),"
            f'"This operator requires the IcosahedralShellMap to be registered as GeometryMap on every macro-cell." )'
        ]

        code += [
            f"real_t {self.radRefVertex} = std::dynamic_pointer_cast< IcosahedralShellMap >( cell.getGeometryMap() )->radRefVertex();",
            f"real_t {self.radRayVertex} = std::dynamic_pointer_cast< IcosahedralShellMap >( cell.getGeometryMap() )->radRayVertex();",
        ]

        for i in range(3):
            code += [
                f"real_t {self.refVertex[i]} = std::dynamic_pointer_cast< IcosahedralShellMap >( cell.getGeometryMap() )->refVertex()[{i}];",
                f"real_t {self.rayVertex[i]} = std::dynamic_pointer_cast< IcosahedralShellMap >( cell.getGeometryMap() )->rayVertex()[{i}];",
                f"real_t {self.thrVertex[i]} = std::dynamic_pointer_cast< IcosahedralShellMap >( cell.getGeometryMap() )->thrVertex()[{i}];",
                f"real_t {self.forVertex[i]} = std::dynamic_pointer_cast< IcosahedralShellMap >( cell.getGeometryMap() )->forVertex()[{i}];",
            ]

        return "\n".join(code)


class AffineMap2D(GeometryMap):
    """
    This blending map uses a 2x2 matrix M and a shift vector v to
    construct an affine mapping x -> M * x + v.

    This is mostly intended for testing purposes, as we normally
    would not want to have the overhead of blending in such a situation.
    """

    def __init__(self):
        self.mat = sp.Matrix([["bMat_00", "bMat_01"], ["bMat_10", "bMat_11"]])
        self.vec = sp.Matrix([["bVec_0"], ["bVec_1"]])

    def supported_geometries(self) -> List[ElementGeometry]:
        return [TriangleElement()]

    def is_affine(self) -> bool:
        # is_affine is supposed to indicate that the map does not have to be re-computed
        # for each quadrature point - however, there seems to currently be a problem with
        # the implementation of that optimization. Thus, we return False for the moment.
        return False

    def evaluate(self, x: sp.Matrix) -> sp.Matrix:
        return self.mat * x + self.vec

    def jacobian(self, x: sp.Matrix) -> sp.Matrix:
        jac = self.mat.copy()
        return jac

    def hessian(self, x: sp.Matrix) -> List[sp.Matrix]:
        hess = [sp.zeros(2, 2), sp.zeros(2, 2)]
        return hess

    def coupling_includes(self) -> List[str]:
        return ["hyteg/geometry/AffineMap2D.hpp"]

    def parameter_coupling_code(self) -> str:
        code = []

        code += [
            f"WALBERLA_CHECK_NOT_NULLPTR( std::dynamic_pointer_cast< AffineMap2D >( face.getGeometryMap() ),"
            f'"This operator requires the AffineMap2D to be registered as GeometryMap on every macro-face." )'
        ]

        code += [
            f"real_t {self.mat[0,0]} = std::dynamic_pointer_cast< AffineMap2D >( face.getGeometryMap() )->getMatrix()(0,0);",
            f"real_t {self.mat[0,1]} = std::dynamic_pointer_cast< AffineMap2D >( face.getGeometryMap() )->getMatrix()(0,1);",
            f"real_t {self.mat[1,0]} = std::dynamic_pointer_cast< AffineMap2D >( face.getGeometryMap() )->getMatrix()(1,0);",
            f"real_t {self.mat[1,1]} = std::dynamic_pointer_cast< AffineMap2D >( face.getGeometryMap() )->getMatrix()(1,1);",
        ]

        for i in range(2):
            code += [
                f"real_t {self.vec[i]} = std::dynamic_pointer_cast< AffineMap2D >( face.getGeometryMap() )->getVector()[{i}];",
            ]

        return "\n".join(code)
