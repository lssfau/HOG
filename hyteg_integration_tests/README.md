# Integration tests for HyTeG

This directory contains integration tests which compare the generated operators against a manual reference implementation in HyTeG.

# How to build

First clone HyTeG or create a symlink to an existing working tree in the current directory (`hyteg_integration_tests`).

```sh
hyteg_integration_tests$ ln -s path/to/hyteg
```

Next, use CMake and make/ninja to configure the build and compile the tests, e.g.:

```sh
hyteg_integration_tests$ source path-to-your-env/bin/activate
hyteg_integration_tests$ mkdir build
hyteg_integration_tests$ cd build
hyteg_integration_tests/build$ CXXFLAGS='-fdiagnostics-color=always' cmake -G Ninja ..
hyteg_integration_tests/build$ ccmake .
hyteg_integration_tests/build$ ninja -j 6 src/all
```

As part of the build, all tested operators will be automatically generated.
Therefore, do not forget to activate your Python environment.
Note that changing the generator itself (i.e. Python code) will not trigger regeneration of the operators.
As of now you have to take care of that manually.
Use of ninja is advised because make does not capture the output of the code generator leading to mangled output in parallel builds.

Finally, execute the tests.

```sh
hyteg_integration_tests/build$ ctest --test-dir src -j 6 --output-on-failure
```

# Writing new tests

Writing a new test for a new form requires two steps:

1. Write the C++ code (in the `src` directory).
   Take a look at what is already there and copy/pasta.
   The test can include the test operator analogous to `#include "Diffusion/TestOpDiffusion.hpp"`.
   Change `Diffusion` to the name of the form (twice).
2. Add the test to CMake (at `src/CMakeLists.txt`).
   Again, replicate what is already there.
   All you need to do is add function calls to `add_operator_test`.
   You can reuse the same C++ test to test different sets of optimizations.
   In some cases the preprocessor is useful for increasing the reusability.
