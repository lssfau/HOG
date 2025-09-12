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

from typing import List, Union
import sympy as sp

from hog.exception import HOGException


def det(mat: sp.Matrix) -> sp.Expr:
    if mat.rows != mat.cols:
        raise HOGException("det() of non-square matrix?")

    if mat.rows == 0:
        return mat.one
    elif mat.rows == 1:
        return mat[0, 0]
    elif mat.rows == 2:
        return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    elif mat.rows == 3:
        return (
            mat[0, 0] * mat[1, 1] * mat[2, 2]
            + mat[0, 1] * mat[1, 2] * mat[2, 0]
            + mat[0, 2] * mat[1, 0] * mat[2, 1]
            - mat[0, 2] * mat[1, 1] * mat[2, 0]
            - mat[0, 0] * mat[1, 2] * mat[2, 1]
            - mat[0, 1] * mat[1, 0] * mat[2, 2]
        )
    return mat.det()


def abs(expr: sp.Expr) -> sp.Expr:
    return sp.Abs(expr)

def pad_rows(vec: sp.Matrix, expected_dim: int, default_value: float) -> sp.Matrix:
    """Makes sure that a given matrix has an expected amount of rows. If not, new rows initialized with default_value will be added."""
    if vec.shape[0] >= expected_dim:
        return vec
    else:
        return vec.col_join(sp.ones(expected_dim-vec.shape[0], vec.shape[1])*default_value)
    
def pad_cols(vec: sp.Matrix, expected_dim: int, default_value: float) -> sp.Matrix:
    """Makes sure that a given matrix has an expected amount of columns. If not, new columns initialized with default_value will be added."""
    if vec.shape[1] >= expected_dim:
        return vec
    else:
        return vec.row_join(sp.ones(vec.shape[0], expected_dim-vec.shape[1])*default_value)    

def grad(f: Union[sp.Expr, sp.MatrixBase], symbols: List[sp.Symbol]) -> sp.MatrixBase:
    """Returns the gradient of the passed sympy expression with respect to the passed symbols."""
    if isinstance(f, sp.MatrixBase):
        return sp.simplify(f.jacobian(symbols).T)
    elif isinstance(f, sp.Expr):
        return sp.simplify(sp.Matrix([[sp.diff(f, s)] for s in symbols]))
    raise HOGException("Invalid data type in grad().")


def hessian(f: sp.Expr, symbols: List[sp.Symbol]) -> sp.MatrixBase:
    """Returns the hessian of the passed sympy expression with respect to the passed symbols."""
    df = grad(f, symbols)
    return sp.Matrix([[sp.simplify(sp.diff(g, s)) for s in symbols] for g in df])


def curl(u: sp.Matrix, symbols: List[sp.Symbol]) -> sp.Expr:
    """Returns the curl of the passed sympy matrix with respect to the passed symbols."""
    if u.shape != (3, 1) or len(symbols) != 3:
        raise HOGException("Curl is only defined for 3D vectors.")

    curl = sp.Matrix(
        [
            sp.diff(u[2], symbols[1]) - sp.diff(u[1], symbols[2]),
            sp.diff(u[0], symbols[2]) - sp.diff(u[2], symbols[0]),
            sp.diff(u[1], symbols[0]) - sp.diff(u[0], symbols[1]),
        ]
    )
    return sp.simplify(curl)


def dot(f: sp.MatrixBase, g: sp.MatrixBase) -> sp.MatrixBase:
    """Simple wrapper for the dot product of two sympy expressions."""
    return f.T * g


def cross(a: sp.Matrix, b: sp.Matrix) -> sp.MatrixBase:
    """Cross product of two 3D vectors."""
    if a.shape != b.shape:
        raise HOGException("Shape mismatch in cross().")
    if a.shape != (3, 1):
        raise HOGException("Invalid shape for cross().")
    r0 = a[1] * b[2] - a[2] * b[1]
    r1 = a[2] * b[0] - a[0] * b[2]
    r2 = a[0] * b[1] - a[1] * b[0]
    return sp.Matrix([r0, r1, r2])


def norm(a: sp.MatrixBase) -> sp.Expr:
    return a.norm()


def double_contraction(f: sp.Matrix, g: sp.Matrix) -> sp.MatrixBase:
    """Double contraction of two order 2 tensors."""
    if f.shape != g.shape:
        raise HOGException(
            f"Cannot perform double contraction of tensors of sizes {f.shape} and {g.shape}."
        )
    rows, cols = f.shape
    result = 0
    for i in range(rows):
        for j in range(cols):
            result += f[i, j] * g[i, j]
    return sp.Matrix([result])


def e_vec(dim: int, idx: int) -> sp.Matrix:
    """Returns a basis vector e of length dim with e_i = kronecker(i, idx)."""
    e = sp.zeros(dim, 1)
    e[idx] = 1
    return e


def inv(mat: sp.MatrixBase) -> sp.Matrix:
    """Optimized implementation of matrix inverse for 2x2, and 3x3 matrices. Use this instead of sympy's mat**-1."""
    if isinstance(mat, sp.MatrixBase):
        rows, cols = mat.shape
        if rows != cols:
            raise HOGException("Matrix is not square - cannot be inverted.")
        if rows == 2:
            a = mat[0, 0]
            b = mat[0, 1]
            c = mat[1, 0]
            d = mat[1, 1]
            det = a * d - b * c
            invmat = (1 / det) * sp.Matrix([[d, -b], [-c, a]])
            return invmat
        elif rows == 3:
            a = mat[0, 0]
            b = mat[0, 1]
            c = mat[0, 2]
            d = mat[1, 0]
            e = mat[1, 1]
            f = mat[1, 2]
            g = mat[2, 0]
            h = mat[2, 1]
            i = mat[2, 2]
            det = a * e * i + b * f * g + c * d * h - g * e * c - h * f * a - i * d * b
            invmat = (1 / det) * sp.Matrix(
                [
                    [e * i - f * h, c * h - b * i, b * f - c * e],
                    [f * g - d * i, a * i - c * g, c * d - a * f],
                    [d * h - e * g, b * g - a * h, a * e - b * d],
                ]
            )
            return invmat
        else:
            return mat**-1
    elif isinstance(mat, sp.Expr):
        return 1 / mat


def normal(plane: List[sp.Matrix], d: sp.Matrix) -> sp.Matrix:
    """Returns a unit normal to the passed plane. d is a point on one side of the plane in which the normal shall point."""
    if len(plane) == 2:
        # Find projection P of d onto the plane
        plane_vec = plane[1] - plane[0]
        plane_anchor = plane[0]
        s = dot((d - plane_anchor), plane_vec)[0, 0] / dot(plane_vec, plane_vec)[0, 0]
        P = plane_anchor + s * plane_vec
        n = d - P
        nd = dot(n, n)
        if nd.shape != (1, 1):
            raise HOGException("Dot product should be scalar.")
        res = n / sp.sqrt(nd[0, 0])
        return res
    elif len(plane) == 3:
        raise HOGException("Normal vector not implemented.")
    else:
        raise HOGException("Normal vector not implemented.")


def vol(vertices: List[sp.Matrix]) -> sp.Expr:
    """Returns the volume of the passed simplex."""
    if len(vertices) == 2:
        # a line
        v0 = vertices[0]
        v1 = vertices[1]
        d = v1 - v0
        dd = dot(d, d)
        if dd.shape != (1, 1):
            raise HOGException("Dot product should be scalar.")
        return sp.sqrt(dd[0, 0])
    elif len(vertices) == 3:
        # a triangle
        ab = pad_rows(vertices[1] - vertices[0], 3, 0)
        ac = pad_rows(vertices[2] - vertices[0], 3, 0)
        return sp.Rational(1,2) * norm(cross(ab, ac))
    elif len(vertices) == 4:
        # a tetrahedron
        ad = vertices[0]-vertices[3]
        bd = vertices[1]-vertices[3]
        cd = vertices[2]-vertices[3]

        return sp.Abs(ad.dot(bd.cross(cd))) * sp.Rational(1,6)  
    else:
        raise HOGException(f"Not implemented for {len(vertices)} vertices")
    
def diameter(vertices: List[sp.Matrix]) -> sp.Expr:
    """Returns the diameter of the passed simplex, calculated as double the circumcircle / circumsphere radius."""
    if len(vertices) == 2:
        # a line
        return norm(vertices[1]-vertices[0])
    elif len(vertices) == 3:
        # a triangle
        a = norm(vertices[1]-vertices[2])
        b = norm(vertices[2]-vertices[0])
        c = norm(vertices[0]-vertices[1])

        return 2 * ( a * b * c / (4 * vol(vertices)) )
    elif len(vertices) == 4:
        # a tetrahedron
        a       = norm(vertices[0]-vertices[3])
        b       = norm(vertices[1]-vertices[3])
        c       = norm(vertices[2]-vertices[3])
        a_tilde = norm(vertices[1]-vertices[2])
        b_tilde = norm(vertices[2]-vertices[0])
        c_tilde = norm(vertices[0]-vertices[1])

        # heron type formula (from Cayley Menger determinant)
        s = (a*a_tilde + b*b_tilde + c*c_tilde) * sp.Rational(1,2)

        return 2 * ( sp.sqrt(s * (s - a*a_tilde) * (s- b*b_tilde) * (s - c*c_tilde)) / (6 * vol(vertices)) ) 
    else:
        raise HOGException(f"Not implemented for {len(vertices)} vertices")    
