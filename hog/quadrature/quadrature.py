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

import logging
import warnings
from typing import Collection, List, Optional, Tuple, Union
import quadpy
from quadpy.tn._helpers import TnScheme
import numpy as np
import numpy.typing as npt
import sympy as sp

from hog.element_geometry import (
    ElementGeometry,
    TriangleElement,
    EmbeddedTriangle,
    TetrahedronElement,
    LineElement,
)
from hog.logger import TimedLogger, get_logger
from hog.exception import HOGException
from hog.symbolizer import Symbolizer
from hog.sympy_extensions import fast_subs
from hog.blending import GeometryMap, IdentityMap


class HOGIntegrationException(HOGException):
    pass


class HOGExactIntegrationException(HOGIntegrationException):
    pass


def select_quadrule(
    scheme_info: Union[str | int], geometry: ElementGeometry
) -> Union[TnScheme | str]:
    """Checks for availability of a specified quadrature rule and chooses a rule with minimal points
    if only a degree is given."""

    # TODO for now, leave out line elements as we have no use for them without DG
    logger = get_logger()

    # quadrule given by name, just check if it exists and return it
    if scheme_info == "exact":
        return scheme_info
    if isinstance(scheme_info, str):
        # if isinstance(geometry, LineElement) and info[geometry] not in [scheme for scheme in quadpy.t1.schemes]:
        #    raise HOGException(f"Not quadrature rule with name {info[geometry]} found for lines.")
        if isinstance(geometry, TriangleElement) and scheme_info not in [
            scheme for scheme in quadpy.t2.schemes
        ]:
            raise HOGException(
                f"No quadrature rule with name {scheme_info} found for triangles. Use --list-quadratures t2 to query rules for triangles."
            )
        if isinstance(geometry, TetrahedronElement) and scheme_info not in [
            scheme for scheme in quadpy.t3.schemes
        ]:
            raise HOGException(
                f"No quadrature rule with name {scheme_info} found for tetrahedra. Use --list-quadratures t3 to query rules for tetrahedra."
            )

        if isinstance(geometry, LineElement):
            scheme = quadpy.t1.schemes[scheme_info]()
        elif isinstance(geometry, TriangleElement):
            scheme = quadpy.t2.schemes[scheme_info]()
        elif isinstance(geometry, TetrahedronElement):
            scheme = quadpy.t3.schemes[scheme_info]()
        else:
            raise HOGException(f"Geometry not implemented: {geometry}")
    # degree given: choose quadrule with min points that integrates that degree exactly
    elif isinstance(scheme_info, int):
        if int(scheme_info) == 0:
            # Quadpy's degree 0 rules integrate linear polynomials exactly as well and therefore are all registered
            # as degree-1-exact-integration rules. Increasing degree to 1 such that rules can be found in the database.
            scheme_info = 1

        def choose_rule_with_min_points(
            degree: int, geometry: ElementGeometry
        ) -> TnScheme:
            with warnings.catch_warnings():
                # suppressing quadpy precision warnings - the precision is checked by ourselves
                warnings.simplefilter("ignore")

                all_schemes = []
                if isinstance(geometry, TriangleElement) or isinstance(
                    geometry, EmbeddedTriangle
                ):
                    schemes = quadpy.t2.schemes
                elif isinstance(geometry, TetrahedronElement):
                    schemes = quadpy.t3.schemes
                for key, s in schemes.items():
                    try:
                        scheme = s()
                    except TypeError as e:
                        pass
                    all_schemes.append(scheme)

            def select_degree(x: TnScheme) -> int:
                if x.degree >= degree:
                    return x.points.shape[1]
                else:
                    return 10**10000  # just a large number

            return min(all_schemes, key=lambda x: select_degree(x))

        scheme = choose_rule_with_min_points(int(scheme_info), geometry)
    else:
        raise HOGException(f"Unexpected {scheme_info}")

    logger.info(
        f"Integrating over {geometry} with rule: {scheme.name} (degree: {scheme.degree}, #points: {scheme.points.shape[1]})."
    )
    return scheme


def integrate_exact_over_reference(
    expr: sp.Expr, geometry: ElementGeometry, symbolizer: Symbolizer
) -> sp.Expr:
    """Integrates the passed expression analytically over the reference triangle."""
    if isinstance(geometry, TriangleElement):
        x, y = symbolizer.ref_coords_as_list(2)
        result = sp.integrate(sp.integrate(expr, (x, 0, 1 - y)), (y, 0, 1))
        fs = result.free_symbols
        if x in fs or y in fs:
            raise HOGExactIntegrationException(
                f"Could not successfully perform exact integration of {expr}. Consider using a quadrature rule."
            )
        return result
    elif isinstance(geometry, TetrahedronElement):
        x, y, z = symbolizer.ref_coords_as_list(3)
        result = sp.integrate(
            sp.integrate(sp.integrate(expr, (x, 0, 1 - y - z)), (y, 0, 1 - z)),
            (z, 0, 1),
        )
        fs = result.free_symbols
        if x in fs or y in fs or z in fs:
            raise HOGExactIntegrationException(
                f"Could not successfully perform exact integration of {expr}. Consider using a quadrature rule."
            )
        return result
    else:
        raise HOGException("Cannot integrate over passed geometry.")


class Quadrature:
    """
    Implements numerical (and exact, symbolic) integration of sympy expressions over the reference element.
    Exact integration is performed via sympy.
    """

    _points: Collection[npt.NDArray[np.float_]]
    _weights: Collection[float]

    def __init__(
        self,
        scheme: Union[TnScheme | str],
        geometry: ElementGeometry,
        inline_values: bool = False,
    ):
        """Instantiates a quadrature/integration rule.

        There are two ways to do so:

        a) Symbolic, exact integration can be requested.
           In practice this may not be possible, in which case an exception is thrown. Also, might be really slow.

        b) A certain quadrature rule is requested.

        :param scheme: Defines the quadrature rule to be applied. Possible values:
                       - 'exact' for analytical integration (might fail of course - an exception is thrown in this
                         case),
                       - otherwise, the specified quadrature rule is applied
        :param geometry: the element geometry
        :param inline_values: if True, points and weights are not treated as symbols but inlined right away
                              this may or may not make a performance difference, but seems to produce different
                              op counts in generated code for no obvious reason
        """
        self._geometry = geometry
        self._inline_values = inline_values

        if scheme == "exact":
            self._scheme_name = "exact"
            self._points = []
            self._weights = []
            self._point_symbols: List[List[sp.Symbol]] = []
            self._weight_symbols: List[sp.Symbol] = []
        else:
            (
                self._points,
                self._weights,
                self._degree,
                self._scheme_name,
            ) = Quadrature._prepare_rule(scheme, geometry)
            self._point_symbols = []
            self._weight_symbols = []

            for i, point in enumerate(self._points):
                self._point_symbols.append(
                    sp.symbols([f"q_p_{i}_{d}" for d in range(len(point))])
                )
                self._weight_symbols.append(sp.symbols(f"w_p_{i}"))

    def integrate(
        self, f: sp.Expr, symbolizer: Symbolizer, blending: GeometryMap = IdentityMap()
    ) -> sp.Expr:
        """Integrates the passed sympy expression over the reference element."""
        if hasattr(f, "shape"):
            if f.shape == (1, 1):
                f = f[0]
            else:
                raise HOGException(
                    f"Cannot apply quadrature rule to matrix of shape {f.shape}: {f}."
                )
        ref_symbols = symbolizer.ref_coords_as_list(self._geometry.dimensions)
        if isinstance(self._geometry, EmbeddedTriangle):
            ref_symbols = symbolizer.ref_coords_as_list(self._geometry.dimensions - 1)

        if self._scheme_name == "exact":
            mat_entry = integrate_exact_over_reference(f, self._geometry, symbolizer)
        else:
            mat_entry = 0
            inline_points: Collection[
                Union[npt.NDArray[np.float_], List[sp.Symbol]]
            ] = self._point_symbols
            inline_weights: Collection[Union[float, sp.Symbol]] = self._weight_symbols
            if self._inline_values:
                inline_points = self._points
                inline_weights = self._weights

            # unroll the quadrature loop, stick body for all iterations over quad points together
            for i, (point, weight) in enumerate(zip(inline_points, inline_weights)):
                spat_coord_subs = {}
                for idx, symbol in enumerate(ref_symbols):
                    spat_coord_subs[symbol] = point[idx]
                if not blending.is_affine():
                    for symbol in symbolizer.quadpoint_dependent_free_symbols(
                        self._geometry.dimensions
                    ):
                        spat_coord_subs[symbol] = sp.Symbol(symbol.name + f"_q_{i}")
                f_sub = fast_subs(f, spat_coord_subs)
                mat_entry += f_sub * weight

        return mat_entry

    def evaluate_on_quadpoints(
        self, expr: sp.Expr, symbolizer: Symbolizer
    ) -> List[sp.Expr]:
        ref_symbols = symbolizer.ref_coords_as_list(self._geometry.dimensions)
        evaluated_exprs = []
        for point in self._points:
            spat_coord_subs = {}
            for idx, symbol in enumerate(ref_symbols):
                spat_coord_subs[symbol] = sp.sympify(point[idx])
            evaluated_exprs.append(fast_subs(expr, spat_coord_subs))
        return evaluated_exprs

    @staticmethod
    def _prepare_rule(
        scheme: TnScheme, geometry: ElementGeometry
    ) -> Tuple[
        Collection[npt.NDArray[np.float_]], Collection[float], Optional[int], str
    ]:
        """Returns list of points and list of weights for quadrature.

        :param scheme: a quadpy integration scheme
        :param geometry: the element geometry
        """

        if isinstance(geometry, TetrahedronElement):
            vertices = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            degree = scheme.degree
        elif isinstance(geometry, TriangleElement):
            vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            degree = scheme.degree
        elif isinstance(geometry, LineElement):
            vertices = np.asarray([[0.0], [1.0]])
            degree = scheme.degree
        elif isinstance(geometry, EmbeddedTriangle):
            vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            degree = scheme.degree
        else:
            raise HOGException("Invalid geometry for quadrature.")

        if isinstance(geometry, LineElement):
            points = [np.array([0.5 * (p + 1)]) for p in scheme.points]
            weights = 0.5 * scheme.weights
        else:
            vol = quadpy.tn.get_vol(vertices)
            points = quadpy.tn.transform(scheme.points, vertices.T).T
            weights = vol * scheme.weights

        return points, weights, degree, scheme.name

    def points(self) -> List[Tuple[sp.Symbol, float]]:
        result = []
        for point, vals in zip(self._point_symbols, self._points):
            for coord, val in zip(point, vals):
                result.append((coord, val))
        return result

    def weights(self) -> List[Tuple[sp.Symbol, float]]:
        return list(zip(self._weight_symbols, self._weights))

    def is_exact(self) -> bool:
        """Returns True if exact integration was chosen instead of a quadrature scheme."""
        return self._scheme_name == "exact"

    @property
    def geometry(self) -> ElementGeometry:
        """Returns the integration domain."""
        return self._geometry

    @property
    def inline_values(self) -> bool:
        """Returns True if the quadrature points and weights are inlined."""
        return self._inline_values

    @property
    def degree(self) -> Optional[int]:
        return self._degree

    @property
    def name(self) -> str:
        return self._scheme_name

    def __str__(self):
        points_weights = ""
        if not self.is_exact():
            points_weights = f" | points: {len(self._points)}, degree: {self.degree}"
        return f"{self._scheme_name}{points_weights}"

    def __repr__(self):
        return str(self)
