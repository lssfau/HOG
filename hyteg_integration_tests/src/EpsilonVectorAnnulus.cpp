/*
 * HyTeG Operator Generator
 * Copyright (C) 2017-2026  Nils Kohl, Fabian Böhm, Daniel Bauer, Marcus Mohr
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

#include <type_traits>

#include "core/DataTypes.h"

#include "hyteg/elementwiseoperators/P2ElementwiseEpsilonOperator.hpp"
#include "hyteg/elementwiseoperators/P2ElementwiseOperator.hpp"
#include "hyteg/forms/form_hyteg_generated/p2/p2_epsilonvar_affine_q4.hpp"
#include "hyteg/p2functionspace/P2Function.hpp"

#include "Epsilon/TestOpEpsilon.hpp"
#include "OperatorGenerationTest.hpp"

using namespace hyteg;
using walberla::real_t;

real_t k( const hyteg::Point3D& x )
{
   // The operator also works with a non-constant viscosity.
   // However, the reference operator seems to have such a low quadrature order that the difference between the
   // reference and generated operator is relatively large.
   // Under refinement that error vanishes, but not quickly enough that we can afford refinement in this (quick) test.
   // For simplicity, the viscosity is therefore set to 1 here, such that the comparison succeeds (just blending, no var
   // viscosity). Note that there is a dedicated test with a variable viscosity, but without blending.
   return 1.0;
}

P2ElementwiseBlendingEpsilonOperator makeRefOp( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel )
{
   return P2ElementwiseBlendingEpsilonOperator( storage, minLevel, maxLevel, k );
};

template < class Op >
Op makeTestOp( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel )
{
   P2Function< real_t > k( "k", storage, minLevel, maxLevel );
   for ( size_t lvl = minLevel; lvl <= maxLevel; ++lvl )
   {
      k.interpolate( ::k, lvl );
   }
   return Op( storage, minLevel, maxLevel, k );
};

int main( int argc, char* argv[] )
{
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   walberla::MPIManager::instance()->useWorldComm();

   const uint_t level = 3;

   real_t thresholdOverMachineEpsApply    = 2e3;
   real_t thresholdOverMachineEpsInvDiag  = 9.0e6;
   real_t thresholdOverMachineEpsAssembly = 360;

   // Testing only 2D since generation of 3D operator just takes too long.

   StorageSetup storageSetup(
       "Annulus", MeshInfo::meshAnnulus( 1.0, 2.0, MeshInfo::CRISS, 12, 2 ), GeometryMap::Type::ANNULUS );

   compareApply< P2ElementwiseBlendingEpsilonOperator, operatorgeneration::TestOpEpsilon >(
       makeRefOp,
       makeTestOp< operatorgeneration::TestOpEpsilon >,
       level,
       storageSetup,
       storageSetup.description() + " Apply",
       thresholdOverMachineEpsApply );

#ifdef TEST_DIAG
   compareInvDiag< P2Function< real_t >, P2ElementwiseBlendingEpsilonOperator, operatorgeneration::TestOpEpsilon >(
       makeRefOp,
       makeTestOp< operatorgeneration::TestOpEpsilon >,
       level,
       storageSetup,
       storageSetup.description() + " Inverse Diagonal",
       thresholdOverMachineEpsInvDiag );
#endif

#ifdef TEST_ASSEMBLE
   compareAssembledMatrix< P2ElementwiseBlendingEpsilonOperator, operatorgeneration::TestOpEpsilon >(
       makeRefOp,
       makeTestOp< operatorgeneration::TestOpEpsilon >,
       level,
       storageSetup,
       storageSetup.description() + " Assembly",
       thresholdOverMachineEpsAssembly );
#endif

   return EXIT_SUCCESS;
}
