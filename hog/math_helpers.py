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

def padRows(vec: sp.Matrix, expectedDim: int, defaultValue: float) -> sp.Matrix:
    if vec.shape[0] >= expectedDim:
        return vec
    else:
        return vec.col_join(sp.ones(expectedDim-vec.shape[0], vec.shape[1])*defaultValue)
    
def padCols(vec: sp.Matrix, expectedDim: int, defaultValue: float) -> sp.Matrix:
    if vec.shape[1] >= expectedDim:
        return vec
    else:
        return vec.row_join(sp.ones(vec.shape[0], expectedDim-vec.shape[1])*defaultValue)    

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
        ab = padRows(vertices[1] - vertices[0], 3, 0)
        ac = padRows(vertices[2] - vertices[0], 3, 0)
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

def deltaSUPG(xRef: sp.Matrix, uVec: sp.Matrix, h: sp.Expr, k: sp. Expr, approximateXi: bool = True) -> sp.Expr:
    """
    Returns an expression for the SUPG scaling delta.
    The exp function cannot be vectorized in pystencils yet.
    If approximateXi is True this returns an approximation that can be vectorized.
    """
    dim = uVec.shape[0]

    centroid = {xRef[i]: sp.Rational(1,dim+1) for i in range(dim)}
    uCentroid = [sp.simplify(uVec[i].subs(centroid)) for i in range(dim)]
    abs_u_centroid = sp.sqrt(sum([uCentroid[i]*uCentroid[i] for i in range(dim)]))

    Pe = abs_u_centroid * h / ( 2 * k )
    
    if approximateXi:
        # Taylor near 0, Cubic spline in a second region, 1 - 1/Pe for the rest
        xi = sp.Piecewise(
            (Pe / sp.S(3) - Pe**3 / sp.S(45)                                                                   , Pe <= 0.58999997807912),
            (-0.00415539574874170*Pe**3 - 0.0357573472773785*Pe**2 + 0.359398130738144*Pe - 0.00649542596882084, Pe <= 1.50796949894352),
            ( 0.00608120216343659*Pe**3 - 0.0820667795509200*Pe**2 + 0.429231342120035*Pe - 0.04159754356120940, Pe <= 3.3717174059858 ),
            ( 0.00311255780252234*Pe**3 - 0.0520384899592918*Pe**2 + 0.327984435431960*Pe + 0.07219444229959040, Pe <= 4.85000000293136),
            ((sp.S(1) - sp.S(1)/Pe)                                                                            , sp.sympify(True)      )
        )
    else:
        xi = sp.S(1) + sp.S(2) / ( sp.exp( 2 * Pe ) - sp.S(1) ) - sp.S(1) / Pe

    return h / ( 2 * abs_u_centroid ) * xi 

def simpleViscosityProfile(x: sp.Expr) -> sp.Expr:
    """
    Returns a simple viscosity profile with jumps. 
    Viscosity profile similar to the one used in
     
    Lin, Yi-An and Colli, Lorenzo and Wu, Jonny
    NW Pacific-Panthalassa Intra-Oceanic Subduction During Mesozoic Times From Mantle Convection and Geoid Models
    Geochemistry, Geophysics, Geosystems, Volume 23, Number 11, Pages e2022GC010514, 2022
    https://doi.org/10.1029/2022GC010514
       
    but with constant viscosity in the lower mantle.

    Can be used to compare to TERRA (https://doi.org/10.1126/science.280.5360.91) models.
    """

    etaSimple = sp.Piecewise(
        (1.25e+23, x <= 0.75),
        (3.54619628745821e+26*x**2 - 5.43063135866058e+26*x + 2.07948810730019e+26, x <= 0.758405172413793),
        (1.86568634321297e+26*x**2 - 2.88161649064377e+26*x + 1.11289507706839e+26, x <= 0.764008620689655),
        (1.06929133385375e+26*x**2 - 1.66471118539444e+26*x + 6.48032005181657e+25, x <= 0.772413793103448),
        (1.5e+22 , x <= 0.85172414),
        (3.85797403183421e+25*x**2 - 6.69652358427785e+25*x + 2.90638521562008e+25, x <= 0.85948275862069),
        (2.2385551110116e+25*x**2 - 3.91279830141555e+25*x + 1.71010327294176e+25 , x <= 0.864655172413793),
        (1.38463307937214e+25*x**2 - 2.43610209842523e+25*x + 1.07168676794207e+25, x <= 0.872413793103448),
        (2.5e+21, x <= 0.93793103),
        (1.95514620595374e+25*x**2 - 3.65353225743697e+25*x + 1.70704057747143e+25, x <= 0.953448275862069),
        (5.10752322498446e+25*x**2 - 9.66478912476108e+25*x + 4.57275182542852e+25, x <= 0.963793103448276),
        (1.28735188889197e+26*x**2 - 2.46344152493811e+26*x + 1.17865630354825e+26, x <= 0.974137931034483),
        (2.75420301084874e+26*x**2 - 5.32127215909528e+26*x + 2.57061691415066e+26, x <= 0.98448275862069),
        (8.78545671271681e+26*x**2 - 1.71966027238079e+27*x + 8.41614601109111e+26, True)
    )

    return etaSimple

def expApprox(x: sp.Expr, betterApprox: bool = False) -> sp.Expr:
    """
    Returns a piecewise polynomial approximation of order 5 or lower for the exp function on the interval [-5,5].
    Do not use this approximation on any other value range!
    
    This can to be used for a Frank-Kamenetskii flow, 
    see e.g.
    
    D.A. May and J. Brown and L. Le Pourhiet,
    A scalable, matrix-free multigrid preconditioner for finite element discretizations of heterogeneous Stokes flow,
    Computer Methods in Applied Mechanics and Engineering, Volume 290, p.496-523, 2015,
    https://doi.org/10.1016/j.cma.2015.03.014
    
    or
    
    Lin, Yi-An and Colli, Lorenzo and Wu, Jonny
    NW Pacific-Panthalassa Intra-Oceanic Subduction During Mesozoic Times From Mantle Convection and Geoid Models
    Geochemistry, Geophysics, Geosystems, Volume 23, Number 11, Pages e2022GC010514,
    https://doi.org/10.1029/2022GC010514
    """

    if betterApprox:
        # The absolute approximation error is < 1e-6 on [-10, 10].
        expApprox = sp.Piecewise(
            (7.25210133510516e-6*x**4 + 0.000282942334426625*x**3 + 0.0041745133816066*x**2 + 0.0276563994532016*x + 0.0695796376930537, (x <= -7.5)),
            (1.75106244595648e-5*x**5 + 0.000634293514527288*x**4 + 0.00933736479006806*x**3 + 0.0701247329226285*x**2 + 0.270099538417808*x + 0.429574895170266, (x >= -7.5) & (x <= -5.0)),
            (3.44438559712425e-5*x**6 + 0.000983924751952837*x**5 + 0.0121609784800846*x**4 + 0.0842879213906417*x**3 + 0.350611974210844*x**2 + 0.842800345781279*x + 0.927398687928547, (x >= -5.0) & (x <= -2.5)),
            (5.98991880319994e-5*x**7 + 0.000944797334515148*x**6 + 0.0075043607939413*x**5 + 0.0407686155561299*x**4 + 0.166127564540268*x**3 + 0.499840761384455*x**2 + 0.99998268968172*x + 0.999999721711981, (x >= -2.5) & (x <= 0.0)),
            (0.0026500400085104*x**6 + 0.00589484232125402*x**5 + 0.0438758284577314*x**4 + 0.165665610174163*x**3 + 0.500210697231479*x**2 + 0.999984340972463*x + 1.00000010932421, (x >= 0.0) & (x <= 1.25)),
            (0.00132406706306152*x**7 - 0.00819082089596664*x**6 + 0.0487176322377378*x**5 - 0.0586015490288259*x**4 + 0.32065944421455*x**3 + 0.355622838212876*x**2 + 1.0759463571123*x + 0.9827889619419, (x >= 1.25) & (x <= 2.5)),
            (0.00457383493682491*x**7 - 0.0679898561858632*x**6 + 0.526463877350017*x**5 - 2.20371451832871*x**4 + 6.15815530999186*x**3 - 9.2578966190885*x**2 + 9.93494996343822*x - 2.53665485746765, (x >= 2.5) & (x <= 3.75)),
            (0.0159988772939167*x**7 - 0.378007670093177*x**6 + 4.15380984197933*x**5 - 25.9167207358983*x**4 + 99.6572759882102*x**3 - 231.511741716953*x**2 + 304.706068019523*x - 170.730876497915, (x >= 3.75) & (x <= 5.0)),
            (0.28304672901609*x**6 - 7.32423576117582*x**5 + 83.1772935362035*x**4 - 515.374669099028*x**3 + 1829.96482993567*x**2 - 3505.67116251976*x + 2829.30501341813, (x >= 5.0) & (x <= 5.625)),
            (0.0756394354170271*x**7 - 2.61545104309386*x**6 + 40.3352976275334*x**5 - 352.740006988615*x**4 + 1880.034848318*x**3 - 6078.56320604615*x**2 + 11020.3104345257*x - 8621.62805346233, (x >= 5.625) & (x <= 6.25)),
            (0.984390925509542*x**6 - 32.8357987212767*x**5 + 471.01362058624*x**4 - 3669.13970668305*x**3 + 16296.1647904057*x**2 - 38991.0139914872*x + 39191.5336961241, (x >= 6.25) & (x <= 6.875)),
            (0.264786524967526*x**7 - 11.4783860803262*x**6 + 218.771142782824*x**5 - 2353.44533689661*x**4 + 15371.774396392*x**3 - 60792.5713795176*x**2 + 134561.549298232*x - 128423.620324519, (x >= 6.875) & (x <= 7.5)),
            (0.491899673679648*x**7 - 23.4563456258799*x**6 + 489.623601053712*x**5 - 5757.48746236456*x**4 + 41051.5626841241*x**3 - 177076.740661143*x**2 + 427217.110968214*x - 444210.251905855, (x >= 7.5) & (x <= 8.125)),
            (0.915875896543658*x**7 - 47.6590510587373*x**6 + 1081.96248546365*x**5 - 13814.3070074952*x**4 + 106827.435257423*x**3 - 499389.764359219*x**2 + 1304969.78174018*x - 1469021.13720182, (x >= 8.125) & (x <= 8.75)),
            (1.70077487181909*x**7 - 95.8705823356372*x**6 + 2351.50779693372*x**5 - 32392.7367538162*x**4 + 270003.374783309*x**3 - 1359568.15741888*x**2 + 3824858.54827785*x - 4633696.71688371, (x >= 8.75) & (x <= 9.375)),
            (3.2435880276896*x**7 - 197.491525035967*x**6 + 5221.01244864561*x**5 - 77421.2779642087*x**4 + 694084.420142809*x**3 - 3756683.18331828*x**2 + 11354652.3415925*x - 14773419.2245149, True)
        )      
    else:
        # The absolute approximation error is < 1e-5 on [-5,5].
        expApprox = sp.Piecewise(
            (0.000210799310036196*x**5 + 0.00501190078696588*x**4 + 0.0494402419981363*x**3 + 0.256198878294768*x**2 + 0.708019277890397*x + 0.848198640564163, (x <= -2.5)),
            (0.00042229066167372*x**6 + 0.00571373198524667*x**5 + 0.0377262341425878*x**4 + 0.163477078891512*x**3 + 0.49874653174446*x**2 + 0.999816711696631*x + 0.999996312029598, (x >= -2.5) & (x <= 0.0)),
            (0.0156585460585765*x**5 + 0.030456484353984*x**4 + 0.174117960503455*x**3 + 0.497815769030432*x**2 + 1.00023114817759*x + 0.999996256406322, (x >= 0.0) & (x <= 1.25)),
            (0.00919017664768228*x**6 - 0.0481169944065158*x**5 + 0.238134656591533*x**4 - 0.219386056681326*x**3 + 0.9392688549289*x**2 + 0.729152335884409*x + 1.0701841156343, (x >= 1.25) & (x <= 2.5)),
            (0.0319292326734341*x**6 - 0.405841148550303*x**5 + 2.61245893835647*x**4 - 8.71819053684828*x**3 + 18.2169577645378*x**2 - 18.15777498739*x + 9.73119878687425, (x >= 2.5) & (x <= 3.75)),
            (0.484334107122001*x**5 - 7.40404535751216*x**4 + 50.0684690859892*x**3 - 172.708281678359*x**2 + 308.465883701428*x - 220.837411495519, (x >= 3.75) & (x <= 4.375)),
            (0.151551246308766*x**6 - 3.35366863270683*x**5 + 33.1760064840383*x**4 - 179.24929535351*x**3 + 557.837525873006*x**2 - 935.639594566522*x + 664.057105334284, True)
        )

    return expApprox

def sinApprox(y: sp.Expr) -> sp.Expr:
    """
    Returns a piecewise polynomial approximation of order 5 or lower for the sin function on the interval [0,2*pi].
    """

    x = sp.Mod(y,2*sp.pi)

    # The absolute approximation error is < 1e-6 on [0,2*pi].
    sinApprox = sp.Piecewise(
        (0.00773818879926878*x**5 + 0.000663917492157654*x**4 - 0.166960828483897*x**3 + 5.7434780036691e-5*x**2 + 0.999995926533202*x + 1.7994603566393e-8, (x >= 0.0) & (x <= 0.698131700797732)),
        (0.00420170714734027*x**5 + 0.0138578225244834*x**4 - 0.187465420963805*x**3 + 0.0165054340051712*x**2 + 0.993247160594985*x + 0.00112423649240849, (x >= 0.698131700797732) & (x <= 1.39626340159546)),
        (-0.00143824992707594*x**5 + 0.0533165312740673*x**4 - 0.29946605531422*x**3 + 0.177607398472308*x**2 + 0.875945336819296*x + 0.0356650994670281, (x >= 1.39626340159546) & (x <= 2.0943951023932)),
        (-0.00633799362383502*x**5 + 0.104045320588459*x**4 - 0.510842932361512*x**3 + 0.620702661340308*x**2 + 0.408705828295634*x + 0.233911060187348, (x >= 2.0943951023932) & (x <= 2.79252680319093)),
        (-0.00828950249413584*x**5 + 0.130211435475182*x**4 - 0.651480159639251*x**3 + 0.99950976677439*x**2 - 0.102673348874465*x + 0.510739296173431, (x >= 2.79252680319093) & (x <= 3.49065850398866)),
        (-0.00634152261425901*x**5 + 0.0951275765135822*x**4 - 0.398422670612051*x**3 + 0.0858090634662075*x**2 + 1.54866111221362*x - 0.684281519676061, (x >= 3.49065850398866) & (x <= 4.18879020478639)),
        (-0.00143231347720746*x**5 - 0.00829114212991957*x**4 + 0.474379799339795*x**3 - 3.60280116003682*x**2 + 9.35465413943443*x - 7.30170174500153, (x >= 4.18879020478639) & (x <= 4.88692190558412)),
        (0.0041878724649904*x**5 - 0.145503556088125*x**4 + 1.81595032939344*x**3 - 10.1690919186447*x**2 + 25.4429812545492*x - 23.0877026710384, (x >= 4.88692190558412) & (x <= 5.58505360638185)),
        (0.00774243943485163*x**5 - 0.243882227305696*x**4 + 2.90589581738653*x**3 - 16.2115250644173*x**2 + 42.2049589169532*x - 41.7016057269896, True),
    )

    return sinApprox

def cosApprox(y: sp.Expr) -> sp.Expr:
    """
    Returns a piecewise polynomial approximation of order 5 or lower for the cos function on the interval [0,2*pi].
    """

    return sinApprox(y+sp.pi/2)