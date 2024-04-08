/*
 * HyTeG Operator Generator
 * Copyright (C) 2017-2024  Nils Kohl, Fabian Böhm, Daniel Bauer
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
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "core/DataTypes.h"

#include "hyteg/elementwiseoperators/P2ToP1ElementwiseOperator.hpp"

#include "Div/TestOpDiv.hpp"
#include "OperatorGenerationTest.hpp"

using namespace hyteg;
using walberla::real_t;

int main( int argc, char* argv[] )
{
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   walberla::MPIManager::instance()->useWorldComm();

   const uint_t level = 3;
   StorageSetup storageSetup(
       "cube_6el", MeshInfo::fromGmshFile( "../hyteg/data/meshes/3D/cube_6el.msh" ), GeometryMap::Type::IDENTITY );

   real_t thresholdOverMachineEpsApply = 225;

   compareApply< REF_OP, operatorgeneration::TestOpDiv >(
       level, storageSetup, storageSetup.description() + " Apply", thresholdOverMachineEpsApply );

   return EXIT_SUCCESS;
}
