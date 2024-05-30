enable_testing()

# Generate an operator, compile a test executable and register the test.
#
# Params:
#  - FILE     Source file which implements the test.
#  - DEF      Define preprocessor variables.
#  - FORM     Name of the form.
#  - ABBR     Short string describing the configuration used to generate the
#             operator. Must be unique for each FORM, non-empty and a valid
#             file name.
#  - GEN_ARGS Arguments passed to the code generator.
#  - LIBS     Additional libraries to link against.
#
function(add_operator_test)
  set(options)
  set(oneValueArgs FILE FORM ABBR)
  set(multiValueArgs DEF GEN_ARGS LIBS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  string(REGEX MATCH "^[^_]*" FORM_NAME ${ARG_FORM})
  set(OPERATOR "TestOp${FORM_NAME}")
  set(TARGET "${OPERATOR}_${ARG_ABBR}")

  add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/${ARG_ABBR}/${FORM_NAME}/${OPERATOR}.cpp ${CMAKE_CURRENT_BINARY_DIR}/${ARG_ABBR}/${FORM_NAME}/${OPERATOR}.hpp
    COMMAND python3 generate_all_operators.py --output ${CMAKE_CURRENT_BINARY_DIR}/${ARG_ABBR} --name ${OPERATOR} --no-clang-format -f '${ARG_FORM}' ${ARG_GEN_ARGS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
  )

  add_executable(${TARGET} ${ARG_FILE} OperatorGenerationTest.hpp ${CMAKE_CURRENT_BINARY_DIR}/${ARG_ABBR}/${FORM_NAME}/${OPERATOR}.cpp)
  target_compile_definitions(${TARGET} PRIVATE ${ARG_DEF})
  target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${ARG_ABBR})
  target_link_libraries(${TARGET} hyteg::hyteg core ${ARG_LIBS})
  add_test(NAME ${TARGET} COMMAND ${TARGET})
endfunction()

# tests without blending


add_operator_test(FILE EpsilonVector.cpp        DEF           FORM Epsilon                                             ABBR VecVIQT  GEN_ARGS -s P2Vector               -o MOVECONSTANTS QUADLOOPS TABULATE  --quad-degree 2 2 --dimensions 2 LIBS constant_stencil_operator)
add_operator_test(FILE EpsilonVectorAnnulus.cpp DEF           FORM Epsilon                                             ABBR VecbVIQT GEN_ARGS -s P2Vector -b AnnulusMap -o MOVECONSTANTS QUADLOOPS TABULATE  --quad-degree 2 2 --dimensions 2 LIBS constant_stencil_operator)
