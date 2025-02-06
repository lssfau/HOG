cmake_minimum_required(VERSION 3.20)
project(hyteg_integration_tests)

add_compile_options( "-Wno-unused-variable" )

list(APPEND WALBERLA_MODULE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/hyteg/src" "${CMAKE_CURRENT_SOURCE_DIR}/hyteg/tests")
add_subdirectory(hyteg SYSTEM)
waLBerla_import()

# add a convenience interface target to link against
add_library(hyteg::hyteg INTERFACE IMPORTED)
set_target_properties(hyteg::hyteg PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/hyteg/src"
  INTERFACE_LINK_DIRECTORIES "${hyteg_LIBRARY_DIR}"
  INTERFACE_LINK_LIBRARIES hyteg
)

add_subdirectory(src)
