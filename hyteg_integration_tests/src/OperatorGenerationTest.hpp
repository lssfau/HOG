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

#include <memory>
#include <utility>

#include "core/DataTypes.h"
#include "core/logging/Logging.h"
#include "core/math/Random.h"

#include "hyteg/dataexport/VTKOutput/VTKOutput.hpp"
#include "hyteg/eigen/EigenSparseMatrix.hpp"
#include "hyteg/geometry/AnnulusMap.hpp"
#include "hyteg/geometry/IcosahedralShellMap.hpp"
#include "hyteg/mesh/MeshInfo.hpp"
#include "hyteg/n1e1functionspace/N1E1VectorFunction.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/primitivestorage/loadbalancing/SimpleBalancer.hpp"
#include "hyteg/solvers/Smoothables.hpp"
#include "hyteg/sparseassembly/SparseMatrixProxy.hpp"

namespace hyteg {
/// Testing for mixed precision is implemented only for 'compareApply', since this is the only function implemented in the current dev branch of HOG (21.12.2023).
/// TODO: Extend the other tests for mixed precision as well and choose a feasible error acceptance limit for those cases.
///  At least 'compareInvDiag' must be implemented as well, otherwise the test will fail for the mixed precision operators.
/// FIXME: I would suggest to put the entire OpGen testing suite in a own namespace? Things like compareInvDiag could be a conflicting name.

using walberla::real_t;
using walberla::uint_t;

template < typename Op >
using OperatorFactory = std::function< Op( std::shared_ptr< PrimitiveStorage >, uint_t, uint_t ) >;

namespace precisionDict {
/// The machine precision (epsilon) is calculated according to Higham's "Accuracy and Stability of Numerical Algorithms"
template < typename ValueType >
real_t epsilon()
{
   WALBERLA_ASSERT( false,
                    "" << typeid( ValueType ).name() << " is no known ValueType, therefore no accuracy can be specified." );
   return 0.;
}
#ifdef WALBERLA_BUILD_WITH_HALF_PRECISION_SUPPORT
template <>
constexpr real_t epsilon< walberla::float16 >()
{
   return 9.77e-04;
};
#endif
template <>
constexpr real_t epsilon< walberla::float32 >()
{
   return 1.19e-07;
};
template <>
constexpr real_t epsilon< walberla::float64 >()
{
   return 2.22e-16;
};
} // namespace precisionDict

class StorageSetup
{
 public:
   StorageSetup( std::string description, MeshInfo meshInfo, GeometryMap::Type geometryMapType )
   : description_( std::move( description ) )
   , meshInfo_( std::move( meshInfo ) )
   , geometryMapType_( geometryMapType )
   {}

   StorageSetup()
   : StorageSetup( "NOT DEFINED", MeshInfo::emptyMeshInfo(), GeometryMap::Type::IDENTITY )
   {}

   const std::string& description() const { return description_; }

   [[nodiscard]] std::shared_ptr< PrimitiveStorage > createStorage() const
   {
      WALBERLA_CHECK_GREATER( meshInfo_.getVertices().size(), 0, "MeshInfo is empty." )

      SetupPrimitiveStorage setupStorage( meshInfo_, walberla::uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );

      switch ( geometryMapType_ )
      {
      case GeometryMap::Type::IDENTITY:
         break;
      case GeometryMap::Type::ANNULUS:
         AnnulusMap::setMap( setupStorage );
         break;
      case GeometryMap::Type::ICOSAHEDRAL_SHELL:
         IcosahedralShellMap::setMap( setupStorage );
         break;
      default:
         WALBERLA_ABORT( "Geometry map not supported." )
      }

      setupStorage.setMeshBoundaryFlagsOnBoundary( 1, 0, true );
      loadbalancing::roundRobin( setupStorage );
      std::shared_ptr< hyteg::PrimitiveStorage > storage = std::make_shared< hyteg::PrimitiveStorage >( setupStorage );
      return storage;
   }

 private:
   std::string       description_;
   MeshInfo          meshInfo_;
   GeometryMap::Type geometryMapType_;
};

template < typename RefOpType, typename OpType >
void compareApply( const uint_t        level,
                   const StorageSetup& storageSetup,
                   const std::string&  message,
                   const real_t        thresholdOverMachineEps,
                   const bool          writeVTK = false )
{
   OperatorFactory< RefOpType > makeRefOp = []( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel ) {
      return RefOpType( storage, minLevel, maxLevel );
   };
   OperatorFactory< OpType > makeTestOp = []( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel ) {
      return OpType( storage, minLevel, maxLevel );
   };
   compareApply( makeRefOp, makeTestOp, level, storageSetup, message, thresholdOverMachineEps, writeVTK );
}

template < typename RefOpType, typename OpType >
void compareApply( OperatorFactory< RefOpType > refOpFactory,
                   OperatorFactory< OpType >    testOpFactory,
                   const uint_t                 level,
                   const StorageSetup&          storageSetup,
                   const std::string&           message,
                   const real_t                 thresholdOverMachineEps,
                   const bool                   writeVTK = false )
{
   static_assert( std::is_same_v< typename RefOpType::srcType::valueType, typename RefOpType::dstType::valueType >,
                  "Reference operator has a different ValueType for source and destination." );
   static_assert( std::is_same_v< typename OpType::srcType::valueType, typename OpType::dstType::valueType >,
                  "Tested operator has a different ValueType for source and destination." );

   using RefType  = typename RefOpType::srcType::valueType;
   using TestType = typename OpType::srcType::valueType;

   static_assert( std::is_same_v< typename RefOpType::dstType, typename OpType::dstType::template FunctionType< RefType > >,
                  "Reference and test operator have different destination types." );
   static_assert( std::is_same_v< typename RefOpType::srcType, typename OpType::srcType::template FunctionType< RefType > >,
                  "Reference and test operator have different source types." );

   using RefSrcFncType  = typename RefOpType::srcType::template FunctionType< RefType >;
   using RefDstFncType  = typename RefOpType::dstType::template FunctionType< RefType >;
   using TestSrcFncType = typename OpType::srcType::template FunctionType< TestType >;
   using TestDstFncType = typename OpType::dstType::template FunctionType< TestType >;

   WALBERLA_LOG_INFO_ON_ROOT( message )

   std::shared_ptr< hyteg::PrimitiveStorage > storage = storageSetup.createStorage();

   TestSrcFncType src( "src", storage, level, level );
   TestDstFncType dst( "dst", storage, level, level );
   RefSrcFncType  refSrc( "refSrc", storage, level, level );
   RefDstFncType  refDst( "refDst", storage, level, level );

   auto srcFunction = []( const hyteg::Point3D& x ) -> RefType {
      return walberla::numeric_cast< RefType >( x[0] * x[0] * x[0] * x[0] * std::sinh( x[1] ) * std::cos( x[2] ) );
   };
   auto srcFunctionTest = []( const hyteg::Point3D& x ) -> TestType {
      return walberla::numeric_cast< TestType >( x[0] * x[0] * x[0] * x[0] * std::sinh( x[1] ) * std::cos( x[2] ) );
   };

   auto rand = []( const hyteg::Point3D& ) -> RefType {
      return numeric_cast< RefType >( walberla::math::realRandom< RefType >() );
   };
   auto randTest = []( const hyteg::Point3D& ) -> TestType {
      return numeric_cast< TestType >( walberla::math::realRandom< RefType >() );
   };

   // init operand and fill destinations with random noise
   src.interpolate( srcFunctionTest, level, All );
   dst.interpolate( randTest, level, All );
   refSrc.interpolate( srcFunction, level, All );
   refDst.interpolate( rand, level, All );

   // apply reference and test operators
   RefOpType opRef = refOpFactory( storage, level, level );
   opRef.apply( refSrc, refDst, level, All, Replace );

   OpType op = testOpFactory( storage, level, level );
   op.apply( src, dst, level, All, Replace );

   // compare
   RefDstFncType err( "error", storage, level, level );
   if constexpr ( !std::is_same_v< TestType, RefType > && std::is_same_v< RefDstFncType, P1Function< RefType > > )
   {
      RefDstFncType cmpDst( "cmpDst", storage, level, level );
      cmpDst.copyFrom( dst, level );
      err.assign( { 1.0, -1.0 }, { cmpDst, refDst }, level, All );
   }
   else
   {
      static_assert( (std::is_same_v< TestType, RefType >),
                     "Comparing Operators of different ValueType is only implemented for P1 functions, yet." );
      err.assign( { 1.0, -1.0 }, { dst, refDst }, level, All );
   }

   real_t maxAbs = 1.0;
   if constexpr ( std::is_same< RefDstFncType, n1e1::N1E1VectorFunction< RefType > >::value )
   {
      maxAbs = err.getDoFs()->getMaxMagnitude( level );
   }
   else if constexpr ( std::is_same_v< RefDstFncType, P1VectorFunction< RefType > > || std::is_same_v< RefDstFncType, P2VectorFunction< RefType > > )
   {
      maxAbs = err.getMaxComponentMagnitude( level, All );
   }
   else
   {
      maxAbs = err.getMaxMagnitude( level );
   }

   const real_t threshold = thresholdOverMachineEps * precisionDict::epsilon< TestType >();
   WALBERLA_LOG_INFO_ON_ROOT( "Maximum magnitude of error = " << maxAbs << " (threshold: " << threshold << ")" );

   if ( writeVTK )
   {
      VTKOutput vtkOutput( "../../output", "OperatorGenerationTest", storage );
      vtkOutput.add( src );
      vtkOutput.add( dst );
      vtkOutput.add( refSrc );
      vtkOutput.add( refDst );
      vtkOutput.add( err );
      vtkOutput.write( level );
   }

   WALBERLA_CHECK_LESS( maxAbs, threshold );
}

template < typename SrcFncType, typename DstFncType, typename RefOpType, typename OpType >
void compareGEMV( const uint_t        level,
                  const StorageSetup& storageSetup,
                  const std::string&  message,
                  const real_t        thresholdOverMachineEps,
                  const bool          writeVTK = false )
{
   WALBERLA_LOG_INFO_ON_ROOT( message )

   using TestType = typename OpType::srcType::valueType;

   std::shared_ptr< hyteg::PrimitiveStorage > storage = storageSetup.createStorage();

   SrcFncType src( "src", storage, level, level );
   SrcFncType src2( "src2", storage, level, level );
   DstFncType dst( "dst", storage, level, level );
   DstFncType tmp( "tmp", storage, level, level );
   DstFncType refDst( "refDst", storage, level, level );
   DstFncType err( "error", storage, level, level );

   std::function< real_t( const hyteg::Point3D& ) > srcFunction = []( const hyteg::Point3D& x ) {
      return x[0] * x[0] * x[0] * x[0] * std::sinh( x[1] ) * std::cos( x[2] );
   };
   std::function< real_t( const hyteg::Point3D& ) > srcFunction2 = []( const hyteg::Point3D& x ) {
      return x[0] * x[0] * std::cos( x[2] ) + 3 * x[1] * x[1] + 5 * x[1] - 3 * x[0];
   };
   std::function< real_t( const hyteg::Point3D& ) > rand = []( const hyteg::Point3D& ) {
      return walberla::math::realRandom< real_t >();
   };

   // init operand and fill destinations with random noise
   src.interpolate( srcFunction, level, All );
   src2.interpolate( srcFunction2, level, All );
   dst.interpolate( rand, level, All );
   refDst.interpolate( rand, level, All );

   // apply reference and test operators
   // Constant operator
   VTKOutput vtkOutput( "../../output", "OperatorGenerationTest", storage );
   vtkOutput.add( src );
   vtkOutput.add( src2 );
   vtkOutput.add( dst );
   vtkOutput.add( refDst );
   vtkOutput.add( err );
   vtkOutput.write( level, 0 );
   RefOpType opRef( storage, level, level );
   opRef.apply( src, tmp, level, All, Replace );
   refDst.assign( { 0.0, 1.0 }, { tmp, src2 }, level, All );

   vtkOutput.write( level, 1 );
   OpType op( storage, level, level );
   // op.gemv( src, src2, dst, 5.0, 0.0, level, All, Replace );

   // compare
   err.assign( { 1.0, -1.0 }, { dst, refDst }, level, All );
   real_t maxAbs = 1.0;
   if constexpr ( std::is_same< DstFncType, n1e1::N1E1VectorFunction< real_t > >::value )
   {
      maxAbs = err.getDoFs()->getMaxMagnitude( level );
   }
   else if constexpr ( std::is_same_v< DstFncType, P1VectorFunction< real_t > > || std::is_same_v< DstFncType, P2VectorFunction< real_t > > )
   {
      maxAbs = err.getMaxComponentMagnitude( level, All );
   }
   else
   {
      maxAbs = err.getMaxMagnitude( level );
   }

   const real_t threshold = thresholdOverMachineEps * precisionDict::epsilon< TestType >();
   WALBERLA_LOG_INFO_ON_ROOT( "Maximum magnitude of error = " << maxAbs << " (threshold: " << threshold << ")" );

   vtkOutput.write( level, 2 );

   WALBERLA_CHECK_LESS( maxAbs, threshold );
}

template < typename DiagType, typename RefOpType, typename OpType >
void compareInvDiag( const uint_t        level,
                     const StorageSetup& storageSetup,
                     const std::string&  message,
                     const real_t        thresholdOverMachineEps,
                     const bool          writeVTK = false )
{
   OperatorFactory< RefOpType > makeRefOp = []( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel ) {
      return RefOpType( storage, minLevel, maxLevel );
   };
   OperatorFactory< OpType > makeTestOp = []( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel ) {
      return OpType( storage, minLevel, maxLevel );
   };
   compareInvDiag< DiagType, RefOpType, OpType >(
       makeRefOp, makeTestOp, level, storageSetup, message, thresholdOverMachineEps, writeVTK );
}

template < typename DiagType, typename RefOpType, typename TestOpType >
void compareInvDiag( OperatorFactory< RefOpType >  refOpFactory,
                     OperatorFactory< TestOpType > testOpFactory,
                     const uint_t                  level,
                     const StorageSetup&           storageSetup,
                     const std::string&            message,
                     const real_t                  thresholdOverMachineEps,
                     const bool                    writeVTK = false )
{
   // Label Value Types
   using RefType  = typename RefOpType::srcType::valueType;
   using TestType = typename TestOpType::srcType::valueType;
   static_assert( std::is_same_v< RefType, typename DiagType::valueType >,
                  "The RefOpType is of different ValueType than DiagType." );

   using RefDiagType  = typename DiagType::template FunctionType< RefType >;
   using TestDiagType = typename DiagType::template FunctionType< TestType >;

   WALBERLA_LOG_INFO_ON_ROOT( message )

   std::shared_ptr< hyteg::PrimitiveStorage > storage = storageSetup.createStorage();

   // apply reference and test operators
   RefOpType opRef = refOpFactory( storage, level, level );
   opRef.computeInverseDiagonalOperatorValues();
   std::shared_ptr< RefDiagType > diagRef = opRef.getInverseDiagonalValues();

   TestOpType opTest = testOpFactory( storage, level, level );
   opTest.computeInverseDiagonalOperatorValues();
   std::shared_ptr< TestDiagType > diagTest = opTest.getInverseDiagonalValues();

   // compare
   RefDiagType err( "error", storage, level, level );
   if constexpr ( !std::is_same_v< TestType, RefType > && std::is_same_v< DiagType, P1Function< RefType > > )
   {
      RefDiagType diagCmp( "diagCmp", storage, level, level );
      diagCmp.copyFrom( *diagTest, level );
      err.assign( { 1.0, -1.0 }, { diagCmp, *diagRef }, level, All );
   }
   else
   {
      static_assert( (std::is_same_v< TestType, RefType >),
                     "Comparing Operators of different ValueType is only implemented for P1 functions, yet." );
      err.assign( { 1.0, -1.0 }, { *diagTest, *diagRef }, level, All );
   }

   real_t maxAbs = 1.0;
   if constexpr ( std::is_same< RefDiagType, n1e1::N1E1VectorFunction< RefType > >::value )
   {
      maxAbs = err.getDoFs()->getMaxMagnitude( level );
   }
   else if constexpr ( std::is_same_v< RefDiagType, P1VectorFunction< RefType > > || std::is_same_v< RefDiagType, P2VectorFunction< RefType > > )
   {
      maxAbs = err.getMaxComponentMagnitude( level, All );
   }
   else
   {
      maxAbs = err.getMaxMagnitude( level );
   }

   const real_t threshold = thresholdOverMachineEps * precisionDict::epsilon< TestType >();

   WALBERLA_LOG_INFO_ON_ROOT( "Maximum magnitude of error = " << maxAbs << " (threshold: " << threshold << ")" );

   if ( writeVTK )
   {
      VTKOutput vtkOutput( "../../output", "OperatorGenerationTest", storage );
      vtkOutput.add( *diagRef );
      vtkOutput.add( *diagTest );
      vtkOutput.add( err );
      vtkOutput.write( level );
   }

   WALBERLA_CHECK_LESS( maxAbs, threshold );
}

template < typename RefOpType, typename OpType >
void compareAssembledMatrix( const uint_t        level,
                             const StorageSetup& storageSetup,
                             const std::string&  message,
                             const real_t        thresholdOverMachineEps )

{
   OperatorFactory< RefOpType > makeRefOp = []( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel ) {
      return RefOpType( storage, minLevel, maxLevel );
   };
   OperatorFactory< OpType > makeTestOp = []( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel ) {
      return OpType( storage, minLevel, maxLevel );
   };

   compareAssembledMatrix< RefOpType, OpType >( makeRefOp, makeTestOp, level, storageSetup, message, thresholdOverMachineEps );
}

template < typename RefOpType, typename OpType >
void compareAssembledMatrix( OperatorFactory< RefOpType > refOpFactory,
                             OperatorFactory< OpType >    testOpFactory,
                             const uint_t                 level,
                             const StorageSetup&          storageSetup,
                             const std::string&           message,
                             const real_t                 thresholdOverMachineEps )
{
   static_assert( std::is_same_v< typename RefOpType::srcType, typename OpType::srcType >,
                  "Reference and test operator have different source types." );
   static_assert( std::is_same_v< typename RefOpType::dstType, typename OpType::dstType >,
                  "Reference and test operator have different destination types." );
   using SrcFncType = typename RefOpType::srcType::template FunctionType< idx_t >;
   using DstFncType = typename RefOpType::dstType::template FunctionType< idx_t >;

   WALBERLA_LOG_INFO_ON_ROOT( message )

   std::shared_ptr< hyteg::PrimitiveStorage > storage = storageSetup.createStorage();

   SrcFncType src( "src", storage, level, level );
   DstFncType dst( "dst", storage, level, level );
   src.enumerate( level );
   dst.enumerate( level );

   // assemble reference and test operators
   RefOpType                                      opRef  = refOpFactory( storage, level, level );
   Eigen::SparseMatrix< real_t, Eigen::RowMajor > matRef = createEigenSparseMatrixFromOperator( opRef, level, src, dst );

   OpType                                         op  = testOpFactory( storage, level, level );
   Eigen::SparseMatrix< real_t, Eigen::RowMajor > mat = createEigenSparseMatrixFromOperator( op, level, src, dst );

   // compare
   const real_t maxAbs    = ( mat - matRef ).eval().coeffs().cwiseAbs().maxCoeff();
   const real_t threshold = thresholdOverMachineEps * precisionDict::epsilon< typename OpType::srcType::valueType >();
   WALBERLA_LOG_INFO_ON_ROOT( "Maximum magnitude of error = " << maxAbs << " (threshold: " << threshold << ")" );
   WALBERLA_CHECK_LESS( maxAbs, threshold );
}

template < class Diagonal, class Reference, class... Operators >
void testOperators( const uint_t        level,
                    const StorageSetup& storageSetup,
                    const real_t        thresholdOverMachineEpsApply,
                    const real_t        thresholdOverMachineEpsInvDiag,
                    const bool          writeVTK = false )
{
   WALBERLA_LOG_INFO_ON_ROOT( "\n" << storageSetup.description() << " Apply\n" )
   ( compareApply< Reference, Operators >(
         level, storageSetup, typeid( Operators ).name(), thresholdOverMachineEpsApply, writeVTK ),
     ... );

   WALBERLA_LOG_INFO_ON_ROOT( "\n" << storageSetup.description() << " Inverse Diagonal\n" )
   ( compareInvDiag< Diagonal, Reference, Operators >(
         level, storageSetup, typeid( Operators ).name(), thresholdOverMachineEpsInvDiag, writeVTK ),
     ... );
}

} // namespace hyteg
