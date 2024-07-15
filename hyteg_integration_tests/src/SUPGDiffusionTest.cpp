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

#include "hyteg/p2functionspace/P2Function.hpp"

#include "SUPGDiffusion/TestOpSUPGDiffusion.hpp"
#include "OperatorGenerationTest.hpp"

using namespace hyteg;
using walberla::real_t;

int main( int argc, char* argv[] )
{
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   walberla::MPIManager::instance()->useWorldComm();

   Point2D lowerLeft(0.0, 0.0);
   Point2D upperRight(1.0, 1.0);

   uint_t nx = 10U, ny = 10U, nz = 10U;
   auto meshInfo = MeshInfo::meshRectangle(lowerLeft, upperRight, MeshInfo::CRISS, nx, ny);

    StorageSetup storageSetupRectangle(
       "Rectangle", MeshInfo::meshRectangle(lowerLeft, upperRight, MeshInfo::CRISS, nx, ny), GeometryMap::Type::IDENTITY );

    auto storage = storageSetupRectangle.createStorage();

    const uint_t level = 2U;

   P2Function< real_t >       T( "T", storage, level, level );
   P2Function< real_t >       f( "f", storage, level, level );
   P2Function< real_t >       sh( "sh", storage, level, level );
   P2Function< real_t >       kd( "kd", storage, level, level );

   P2VectorFunction< real_t > u("u", storage, level, level);

   std::function<real_t(const Point3D&)> uX = [](const Point3D& x)
   {
    return x[0];
   };

   std::function<real_t(const Point3D&)> uY = [](const Point3D& x)
   {
    return x[1];
   };
   
   std::function<real_t(const Point3D&)> uZ = [](const Point3D& x)
   {
    return x[2];
   };

   std::function<real_t(const Point3D&)> kdFunc = [](const Point3D& x)
   {
      real_t r = x.norm();
    return r * r;
   };

   std::function<real_t(const Point3D&)> shFunc = [](const Point3D& x)
   {
      real_t r = x.norm();
    return r * r;
   };

   std::function<real_t(const Point3D&)> TFunc = [](const Point3D& x)
   {
      real_t r = x.norm();
    return r * r;
   };

   T.interpolate(TFunc, level, All);
   u.interpolate({uX, uY}, level, All);
   kd.interpolate(kdFunc, level, All);
   sh.interpolate(shFunc, level, All);

   operatorgeneration::TestOpSUPGDiffusion testOperator(storage, level, level, kd, u.component(0U), u.component(1U));

   testOperator.apply(T, f, level, All);

   real_t integral = sh.dotGlobal(f, level, All);
   real_t expected = 224.0/45.0;

   real_t error = std::abs(integral - expected);

   real_t threshold = 1e-7;

   WALBERLA_CHECK_LESS(error, threshold);

   return EXIT_SUCCESS;
}