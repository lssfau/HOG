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

add_operator_test(FILE CurlCurl.cpp DEF TEST_ASSEMBLE FORM CurlCurl ABBR noopts GEN_ARGS --quad-degree 0 0)
add_operator_test(FILE CurlCurl.cpp                   FORM CurlCurl ABBR VIQ    GEN_ARGS --quad-degree 0 0 -o MOVECONSTANTS VECTORIZE QUADLOOPS)
add_operator_test(FILE CurlCurl.cpp                   FORM CurlCurl ABBR CVIP   GEN_ARGS --quad-degree 0 0 --loop-strategy CUBES -o MOVECONSTANTS VECTORIZE POLYCSE)

add_operator_test(FILE CurlCurlPlusMass.cpp DEF TEST_ASSEMBLE FORM CurlCurlPlusMass ABBR noopts GEN_ARGS --quad-degree 2 2)
add_operator_test(FILE CurlCurlPlusMass.cpp                   FORM CurlCurlPlusMass ABBR VI     GEN_ARGS --quad-degree 2 2 -o MOVECONSTANTS VECTORIZE)

add_operator_test(FILE DiffusionP1.cpp       DEF TEST_ASSEMBLE FORM Diffusion ABBR 1       GEN_ARGS -s P1       --quad-degree 0 0)
add_operator_test(FILE DiffusionP1.cpp                         FORM Diffusion ABBR 1VIP    GEN_ARGS -s P1       --quad-degree 0 0 -o MOVECONSTANTS VECTORIZE POLYCSE)
add_operator_test(FILE DiffusionP1.cpp                         FORM Diffusion ABBR 1CVI    GEN_ARGS -s P1       --quad-degree 0 0 --loop-strategy CUBES -o MOVECONSTANTS VECTORIZE)
add_operator_test(FILE DiffusionP2.cpp       DEF TEST_ASSEMBLE FORM Diffusion ABBR 2       GEN_ARGS -s P2       --quad-degree 2 2)
add_operator_test(FILE DiffusionP1Vector.cpp DEF TEST_ASSEMBLE FORM Diffusion ABBR 1Vec    GEN_ARGS -s P1Vector --quad-degree 0 0)

add_operator_test(FILE DivKGrad.cpp DEF TEST_ASSEMBLE FORM DivKGrad ABBR noopts GEN_ARGS -s P2 --quad-degree 4 4)
add_operator_test(FILE DivKGrad.cpp                   FORM DivKGrad ABBR VIQ    GEN_ARGS -s P2 --quad-degree 4 4 -o MOVECONSTANTS VECTORIZE QUADLOOPS )
add_operator_test(FILE DivKGrad.cpp DEF TEST_ASSEMBLE FORM DivKGrad ABBR IQT    GEN_ARGS -s P2 --quad-degree 4 4 -o MOVECONSTANTS QUADLOOPS TABULATE )

add_operator_test(FILE Div.cpp  DEF REF_OP=P2ToP1ElementwiseDivxOperator  FORM Div_0  ABBR VIQT GEN_ARGS -s P2ToP1 -o MOVECONSTANTS QUADLOOPS TABULATE VECTORIZE --quad-degree 2 2)
add_operator_test(FILE DivT.cpp DEF REF_OP=P1ToP2ElementwiseDivTzOperator FORM DivT_2 ABBR VIQT GEN_ARGS -s P1ToP2 -o MOVECONSTANTS QUADLOOPS TABULATE VECTORIZE --quad-degree 2 2)

add_operator_test(FILE Epsilon.cpp       DEF TEST_DIAG FORM=forms::p2_epsilonvar_0_0_affine_q4 FORM Epsilon_0_0 ABBR 00VIQT  GEN_ARGS -s P2       -o MOVECONSTANTS QUADLOOPS TABULATE VECTORIZE --quad-degree 4 4)
add_operator_test(FILE Epsilon.cpp       DEF           FORM=forms::p2_epsilonvar_2_1_affine_q4 FORM Epsilon_2_1 ABBR 21VIQT  GEN_ARGS -s P2       -o MOVECONSTANTS QUADLOOPS TABULATE VECTORIZE --quad-degree 4 4)
add_operator_test(FILE EpsilonVector.cpp DEF           FORM Epsilon                                             ABBR VecVIQT GEN_ARGS -s P2Vector -o MOVECONSTANTS QUADLOOPS TABULATE VECTORIZE --quad-degree 4 4)

# tests with blending

add_operator_test(FILE DiffusionP1Annulus.cpp DEF TEST_ASSEMBLE FORM Diffusion ABBR b2     GEN_ARGS -s P1 -b AnnulusMap --quad-rule hillion_07 yu_3)
add_operator_test(FILE DiffusionP1Annulus.cpp                   FORM Diffusion ABBR b2VIQT GEN_ARGS -s P1 -b AnnulusMap -o MOVECONSTANTS QUADLOOPS TABULATE VECTORIZE --quad-rule hillion_07 yu_3)
add_operator_test(FILE DiffusionP1IcosahedralShell.cpp          FORM Diffusion ABBR b3VIQT GEN_ARGS -s P1 -b IcosahedralShellMap -o MOVECONSTANTS QUADLOOPS TABULATE VECTORIZE --quad-rule hillion_07 yu_3)

add_operator_test(FILE FullStokes.cpp DEF TEST_DIAG FORM=forms::p2_full_stokesvar_0_0_blending_q3 FORM FullStokes_0_0 ABBR 00b3VIQT GEN_ARGS -s P2 -b IcosahedralShellMap -o MOVECONSTANTS QUADLOOPS TABULATE VECTORIZE --quad-rule yu_3 yu_3)
add_operator_test(FILE FullStokes.cpp DEF           FORM=forms::p2_full_stokesvar_2_1_blending_q3 FORM FullStokes_2_1 ABBR 21b3VIQT GEN_ARGS -s P2 -b IcosahedralShellMap -o MOVECONSTANTS QUADLOOPS TABULATE VECTORIZE --quad-rule yu_3 yu_3)
