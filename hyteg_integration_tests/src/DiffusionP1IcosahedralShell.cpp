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

   const uint_t level = 4;

   StorageSetup storageSetupIcosahedralShell(
       "IcosahedralShell", MeshInfo::meshSphericalShell( 3, 2, 0.5, 1.0 ), GeometryMap::Type::ICOSAHEDRAL_SHELL );

   testOperators< P1Function< real_t >, P1ElementwiseBlendingLaplaceOperator, operatorgeneration::TestOpDiffusion >(
       level, storageSetupIcosahedralShell, 4e5, 4e9 );

#ifdef TEST_ASSEMBLE
   compareAssembledMatrix< P1ElementwiseBlendingLaplaceOperator, operatorgeneration::TestOpDiffusion >(
       level, storageSetupIcosahedralShell, storageSetupIcosahedralShell.description() + " Assembly", 2e8 );
#endif

   return EXIT_SUCCESS;
}
