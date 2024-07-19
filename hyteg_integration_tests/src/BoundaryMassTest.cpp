/*
 * Copyright (c) 2024 Marcus Mohr, Nils Kohl.
 *
 * This file is part of HyTeG
 * (see https://i10git.cs.fau.de/hyteg/hyteg).
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
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <core/Environment.h>
#include <core/math/Constants.h>
#include <core/timing/Timer.h>

#include "hyteg/boundary/BoundaryConditions.hpp"
#include "hyteg/dataexport/VTKOutput/VTKOutput.hpp"
#include "hyteg/geometry/AnnulusMap.hpp"
#include "hyteg/geometry/IcosahedralShellMap.hpp"
#include "hyteg/mesh/MeshInfo.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/primitivestorage/Visualization.hpp"

#include "BoundaryMass/TestOpBoundaryMass.hpp"

using walberla::real_t;
using walberla::uint_c;
using walberla::uint_t;
using walberla::math::pi;

namespace hyteg {

real_t sphericalSurface( uint_t dim, real_t radius )
{
   return dim == 2 ? 2 * radius * pi : 4 * radius * radius * pi;
}

real_t runTestSphericalAnyDim( uint_t dim, uint_t level, bool inside, real_t innerRad, real_t outerRad )
{
   bool beVerbose = true;

   std::shared_ptr< SetupPrimitiveStorage > setupStorage;

   uint_t markerInnerBoundary = 11;
   uint_t markerOuterBoundary = 22;

   if ( dim == 2 )
   {
      uint_t nLayers = 2;

      MeshInfo meshInfo = MeshInfo::meshAnnulus( innerRad, outerRad, MeshInfo::CROSS, 8, nLayers );

      setupStorage =
          std::make_shared< SetupPrimitiveStorage >( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      AnnulusMap::setMap( *setupStorage );
   }
   else
   {
      uint_t nRad = 2;
      uint_t nTan = 2;

      MeshInfo meshInfo = MeshInfo::meshSphericalShell( nTan, nRad, innerRad, outerRad );

      setupStorage =
          std::make_shared< SetupPrimitiveStorage >( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      IcosahedralShellMap::setMap( *setupStorage );
   }

   real_t tol         = 1e-6;
   real_t boundaryRad = 0.0;
   // flag the inner and outer boundary by assigning different values
   auto onBoundary = [&boundaryRad, tol]( const Point3D& x ) {
      real_t radius = std::sqrt( x[0] * x[0] + x[1] * x[1] + x[2] * x[2] );
      return std::abs( boundaryRad - radius ) < tol;
   };

   boundaryRad = outerRad;
   setupStorage->setMeshBoundaryFlagsByVertexLocation( markerOuterBoundary, onBoundary, true );

   boundaryRad = innerRad;
   setupStorage->setMeshBoundaryFlagsByVertexLocation( markerInnerBoundary, onBoundary, true );

   auto storage = std::make_shared< PrimitiveStorage >( *setupStorage );

   writeDomainPartitioningVTK( storage, ".", "domain" );

   // -----------------------
   //  Function Manipulation
   // -----------------------

   P2Function< real_t > one( "one", storage, level, level );
   P2Function< real_t > result( "result", storage, level, level );

   // generate bc object and set different conditions on inside and outside
   BoundaryCondition bcs;
   BoundaryUID       massBoundaryUID = inside ? bcs.createDirichletBC( "massBoundary", markerInnerBoundary ) :
                                                bcs.createDirichletBC( "massBoundary", markerOuterBoundary );

   one.setBoundaryCondition( bcs );
   result.setBoundaryCondition( bcs );

   one.interpolate( 1.0, level, All );

   operatorgeneration::TestOpBoundaryMass boundaryMassOp( storage, level, level, bcs, massBoundaryUID );

   boundaryMassOp.apply( one, result, level, All );

   communication::syncFunctionBetweenPrimitives( result, level );

   auto sumFreeslipBoundary = result.sumGlobal( level, All );

   if ( beVerbose )
   {
      std::string fPath = ".";
      std::string fName = "BoundaryIntegralTest";
      VTKOutput   vtkOutput( fPath, fName, storage );
      vtkOutput.add( one );
      vtkOutput.add( result );
      vtkOutput.write( level );
   }

   return sumFreeslipBoundary;
}

} // namespace hyteg

int main( int argc, char* argv[] )
{
   walberla::Environment walberlaEnv( argc, argv );
   walberla::MPIManager::instance()->useWorldComm();

   real_t innerRad = 1.1;
   real_t outerRad = 2.1;

   for ( uint_t dim : { 2u } )
   {
      for ( bool inside : { true, false } )
      {
         WALBERLA_LOG_INFO_ON_ROOT( "" << ( dim == 2 ? "Annulus " : "IcoShell " ) << ( inside ? "inside " : "outside " )
                                       << "test" )
         real_t lastError = 0;
         real_t rate      = 1;

         uint_t maxLevel = dim == 2 ? 7 : 5;

         for ( uint_t level = 2; level < maxLevel; level++ )
         {
            auto surfaceComputed = hyteg::runTestSphericalAnyDim( dim, level, inside, innerRad, outerRad );

            auto error = std::abs( surfaceComputed - hyteg::sphericalSurface( dim, inside ? innerRad : outerRad ) );

            if ( level > 2 )
            {
               rate = error / lastError;
               if ( error > 1e-13 )
               {
                    WALBERLA_CHECK_LESS( rate, 0.016 );
               }
               else
               {
                    WALBERLA_CHECK_LESS( error, 1e-13 );
               }

            }

            WALBERLA_LOG_INFO_ON_ROOT( "level: " << level << " | surface computed: " << surfaceComputed << " | error: " << error
                                                 << " | rate: " << rate );

            lastError = error;
         }
      }
   }

   return 0;
}
