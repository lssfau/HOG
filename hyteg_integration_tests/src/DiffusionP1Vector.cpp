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

#include "hyteg/elementwiseoperators/P1ElementwiseOperator.hpp"
#include "mixed_operator/VectorLaplaceOperator.hpp"

#include "Diffusion/TestOpDiffusion.hpp"
#include "OperatorGenerationTest.hpp"

using namespace hyteg;
using walberla::real_t;

int main( int argc, char* argv[] )
{
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   walberla::MPIManager::instance()->useWorldComm();

   const uint_t level = 3;

   for ( uint_t d = 2; d <= 3; ++d )
   {
      StorageSetup storageSetup;
      if ( d == 2 )
      {
         storageSetup = StorageSetup(
             "quad_4el", MeshInfo::fromGmshFile( "../hyteg/data/meshes/quad_4el.msh" ), GeometryMap::Type::IDENTITY );
      }
      else
      {
         storageSetup = StorageSetup(
             "cube_6el", MeshInfo::fromGmshFile( "../hyteg/data/meshes/3D/cube_6el.msh" ), GeometryMap::Type::IDENTITY );
      }

      testOperators< P1VectorFunction< real_t >, P1ElementwiseVectorLaplaceOperator, operatorgeneration::TestOpDiffusion >(
          level, storageSetup, 225, 9.0e6 );

#ifdef TEST_ASSEMBLE
      compareAssembledMatrix< P1ElementwiseVectorLaplaceOperator, operatorgeneration::TestOpDiffusion >(
          level, storageSetup, storageSetup.description() + " Assembly", 360 );
#endif
   }

   return EXIT_SUCCESS;
}
