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
#include "hyteg/p1functionspace/P1Function.hpp"

#include "Diffusion/TestOpDiffusion.hpp"
#include "OperatorGenerationTest.hpp"

using namespace hyteg;
using walberla::real_t;

int main( int argc, char* argv[] )
{
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   walberla::MPIManager::instance()->useWorldComm();

   const uint_t level = 7;

   StorageSetup storageSetupAnnulus(
       "Annulus", MeshInfo::meshAnnulus( 1.0, 2.0, MeshInfo::CRISS, 12, 2 ), GeometryMap::Type::ANNULUS );

   testOperators< P1Function< real_t >, P1ElementwiseBlendingLaplaceOperator, operatorgeneration::TestOpDiffusion >(
       level, storageSetupAnnulus, 2e4, 2e4 );

#ifdef TEST_ASSEMBLE
   compareAssembledMatrix< P1ElementwiseBlendingLaplaceOperator, operatorgeneration::TestOpDiffusion >(
       level, storageSetupAnnulus, storageSetupAnnulus.description() + " Assembly", 3e5 );
#endif

   return EXIT_SUCCESS;
}
