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

#include <type_traits>

#include "core/DataTypes.h"

#include "hyteg/elementwiseoperators/P2ElementwiseOperator.hpp"
#include "hyteg/forms/form_hyteg_generated/p2/p2_epsilonvar_affine_q4.hpp"
#include "hyteg/p2functionspace/P2Function.hpp"

#include "Epsilon/TestOpEpsilon.hpp"
#include "OperatorGenerationTest.hpp"

using namespace hyteg;
using walberla::real_t;

constexpr bool only3d = std::is_constructible< FORM, std::function< real_t( const Point3D& ) > >::value;

real_t k( const hyteg::Point3D& x )
{
   return x[0] * x[0] - x[2] * x[2] - x[1] * x[1] + 3 * x[0] + x[1] - 5 * x[2] + 2;
}

template < class Form >
P2ElementwiseOperator< Form > makeRefOp( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel )
{
   if constexpr ( only3d )
   {
      return P2ElementwiseOperator( storage, minLevel, maxLevel, Form{ k } );
   }
   else
   {
      return P2ElementwiseOperator( storage, minLevel, maxLevel, Form{ k, k } );
   }
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

   real_t thresholdOverMachineEpsApply    = 225;
   real_t thresholdOverMachineEpsInvDiag  = 9.0e6;
   real_t thresholdOverMachineEpsAssembly = 360;

   for ( uint_t d = only3d ? 3 : 2; d <= 3; ++d )
   {
      StorageSetup storageSetup;
      if ( d == 2 )
      {
         storageSetup = StorageSetup(
             "quad_4el", MeshInfo::fromGmshFile( prependHyTeGMeshDir( "2D/quad_4el.msh" ) ), GeometryMap::Type::IDENTITY );
      }
      else
      {
         storageSetup = StorageSetup(
             "cube_6el", MeshInfo::fromGmshFile( prependHyTeGMeshDir( "3D/cube_6el.msh" ) ), GeometryMap::Type::IDENTITY );
      }

      compareApply< P2ElementwiseOperator< FORM >, operatorgeneration::TestOpEpsilon >(
          makeRefOp< FORM >,
          makeTestOp< operatorgeneration::TestOpEpsilon >,
          level,
          storageSetup,
          storageSetup.description() + " Apply",
          thresholdOverMachineEpsApply );

#ifdef TEST_DIAG
      compareInvDiag< P2Function< real_t >, P2ElementwiseOperator< FORM >, operatorgeneration::TestOpEpsilon >(
          makeRefOp< FORM >,
          makeTestOp< operatorgeneration::TestOpEpsilon >,
          level,
          storageSetup,
          storageSetup.description() + " Inverse Diagonal",
          thresholdOverMachineEpsInvDiag );
#endif

#ifdef TEST_ASSEMBLE
      compareAssembledMatrix< P2ElementwiseOperator< FORM >, operatorgeneration::TestOpEpsilon >(
          makeRefOp< FORM >,
          makeTestOp< operatorgeneration::TestOpEpsilon >,
          level,
          storageSetup,
          storageSetup.description() + " Assembly",
          thresholdOverMachineEpsAssembly );
#endif
   }

   return EXIT_SUCCESS;
}
