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

#include "hyteg/elementwiseoperators/N1E1ElementwiseOperator.hpp"
#include "hyteg/n1e1functionspace/N1E1VectorFunction.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"

#include "CurlCurlPlusMass/TestOpCurlCurlPlusMass.hpp"
#include "OperatorGenerationTest.hpp"

using namespace hyteg;
using walberla::real_t;
using Fnc = n1e1::N1E1VectorFunction< real_t >;

constexpr real_t alpha = 0.6;
constexpr real_t beta  = 1.4;

template < class Op >
Op makeTestOp( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel )
{
   P1Function< real_t > alpha( "alpha", storage, minLevel, maxLevel );
   P1Function< real_t > beta( "beta", storage, minLevel, maxLevel );
   for ( size_t lvl = minLevel; lvl <= maxLevel; ++lvl )
   {
      alpha.interpolate( ::alpha, lvl );
      beta.interpolate( ::beta, lvl );
   }
   return Op( storage, minLevel, maxLevel, alpha, beta );
};

n1e1::N1E1ElementwiseLinearCombinationOperator
    makeRefOp( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel )
{
   static forms::n1e1_curl_curl_affine_q0 curlCurlForm;
   static forms::n1e1_mass_affine_qe      massForm;
   return n1e1::N1E1ElementwiseLinearCombinationOperator(
       storage, minLevel, maxLevel, { { alpha, beta }, { &curlCurlForm, &massForm } } );
};

int main( int argc, char* argv[] )
{
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   walberla::MPIManager::instance()->useWorldComm();

   const uint_t       level = 2;
   const StorageSetup storageSetup(
       "cube_6el", MeshInfo::fromGmshFile( prependHyTeGMeshDir( "3D/cube_6el.msh" ) ), GeometryMap::Type::IDENTITY );

   real_t thresholdOverMachineEpsApply    = 225;
   real_t thresholdOverMachineEpsInvDiag  = 9.0e6;
   real_t thresholdOverMachineEpsAssembly = 360;

   compareApply< n1e1::N1E1ElementwiseLinearCombinationOperator, operatorgeneration::TestOpCurlCurlPlusMass >(
       makeRefOp,
       makeTestOp< operatorgeneration::TestOpCurlCurlPlusMass >,
       level,
       storageSetup,
       storageSetup.description() + " Apply",
       thresholdOverMachineEpsApply );

   compareInvDiag< n1e1::N1E1VectorFunction< real_t >,
                   n1e1::N1E1ElementwiseLinearCombinationOperator,
                   operatorgeneration::TestOpCurlCurlPlusMass >( makeRefOp,
                                                                 makeTestOp< operatorgeneration::TestOpCurlCurlPlusMass >,
                                                                 level,
                                                                 storageSetup,
                                                                 storageSetup.description() + " Inverse Diagonal",
                                                                 thresholdOverMachineEpsInvDiag );

#ifdef TEST_ASSEMBLE
   compareAssembledMatrix< n1e1::N1E1ElementwiseLinearCombinationOperator, operatorgeneration::TestOpCurlCurlPlusMass >(
       makeRefOp,
       makeTestOp< operatorgeneration::TestOpCurlCurlPlusMass >,
       level,
       storageSetup,
       storageSetup.description() + " Assembly",
       thresholdOverMachineEpsAssembly );
#endif

   return EXIT_SUCCESS;
}
