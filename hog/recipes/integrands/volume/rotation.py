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

from hog.recipes.common import *
from hog.exception import HOGException


def rotation_matrix(
    mat_size,
    mat_comp_size,
    n_dof_symbols,
    geometry
):

    rotmat = sp.zeros(mat_size, mat_size)

    for row in range(mat_comp_size):
        for col in range(mat_comp_size):
            if row != col:
                continue

            nx_ = n_dof_symbols[0][row]
            ny_ = n_dof_symbols[1][row]

            if geometry.dimensions == 2:
                n_ = sp.Matrix([[nx_], [ny_]])
                nnorm = sp.sqrt(nx_ * nx_ + ny_ * ny_)
                rotmat_ = sp.Matrix([[-ny_, nx_], [nx_, ny_]])
            elif geometry.dimensions == 3:
                nz_ = n_dof_symbols[2][row]
                n_ = sp.Matrix([[nx_], [ny_], [nz_]])
                nnorm = sp.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_)

                ex = sp.Matrix([[1.0], [0.0], [0.0]])
                ey = sp.Matrix([[0.0], [1.0], [0.0]])
                ez = sp.Matrix([[0.0], [0.0], [1.0]])

                ncross = [n_.cross(ex), n_.cross(ey), n_.cross(ez)]
                ncrossnorm = [nc_.norm() for nc_ in ncross]

                t1all = [
                    n_.cross(ex).normalized(),
                    n_.cross(ey).normalized(),
                    n_.cross(ez).normalized(),
                ]

                getrotmat = lambda t1: (t1.row_join(n_.cross(t1)).row_join(n_)).T

                machine_eps = 1e-10
                rotmat_ = sp.eye(geometry.dimensions)
                for iRot in range(geometry.dimensions):
                    for jRot in range(geometry.dimensions):
                        rotmat_[iRot, jRot] = sp.Piecewise(
                            (
                                getrotmat(t1all[0])[iRot, jRot],
                                sp.And(
                                    ncrossnorm[0] + machine_eps > ncrossnorm[1],
                                    ncrossnorm[0] + machine_eps > ncrossnorm[2],
                                ),
                            ),
                            (
                                getrotmat(t1all[1])[iRot, jRot],
                                sp.And(
                                    ncrossnorm[1] + machine_eps > ncrossnorm[0],
                                    ncrossnorm[1] + machine_eps > ncrossnorm[2],
                                ),
                            ),
                            (getrotmat(t1all[2])[iRot, jRot], True),
                        )

                # print("rotmat_ = ", rotmat_.subs([(nx_, 0.0), (ny_, 0.0), (nz_, 1.0)]))
                # print("rotmat_ = ", rotmat_[0, 0])
                # exit()
            else:
                raise HOGException(
                    "We have not reached your dimensions, supreme being or little one"
                )


            rotmatid_ = sp.eye(geometry.dimensions)

            ndofs = mat_comp_size

            for iRot in range(geometry.dimensions):
                for jRot in range(geometry.dimensions):
                    rotmat[row + iRot * ndofs, col + jRot * ndofs] = sp.Piecewise(
                        (rotmat_[iRot, jRot], nnorm > 0.5),
                        (rotmatid_[iRot, jRot], True),
                    )

    return rotmat
