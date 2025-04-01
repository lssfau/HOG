# HyTeG Operator Generator

The HyTeG Operator Generator (HOG) is a small library with the purpose to automatically
generate kernels that integrate weak forms over elements and set up the corresponding
element matrices.

While the entries of the element matrices are assembled as [sympy](https://www.sympy.org/ "sympy web page") symbols, and may be
accessed as such, the generator creates C++ code that can be directly used in [HyTeG](https://i10git.cs.fau.de/hyteg/hyteg "HyTeG GitLab").

For more information visit the [full documentation](https://hyteg.pages.i10git.cs.fau.de/hog/) and have a look at our [arXiv preprint](https://arxiv.org/abs/2404.08371).

To cite us, please use the following reference:

* Böhm, F., Bauer, D., Kohl, N., Alappat, C., Thönnes, D., Mohr, M., Köstler, H. & Rüde, U. (2024). 
  _Code Generation and Performance Engineering for Matrix-Free Finite Element Methods on Hybrid Tetrahedral Grids_. 
  Submitted. arXiv preprint [arXiv:2404.08371](https://arxiv.org/abs/2404.08371).

# How to generate operators

Python version `3.10` or higher is required.

## 1. Alternative: HyTeG submodule
The dedicated submodule `hyteg-operators` within the HyTeG repository generates and integrates them into the build system such that they can be used in HyTeG apps.

### Step 1: Form definition in the HyTeG Operator Generator (HOG)
Declare the desired weak form in the HOG repository locally on your machine in
hog/forms.py:
```python
def new_form(
    trial: FunctionSpace,
    test: FunctionSpace,
    geometry: ElementGeometry,
    symbolizer: Symbolizer,
    ...
) -> Form:
    ...
        form = (double_contraction(2 * mu * sym_grad_phi, sym_grad_psi)
           - sp.Rational(2, 3)  * mu * divdiv)
           * jac_affine_det
           * jac_blending_det
        )
```

### Step 2: Registration in the submodule (`hyteg-operators`)
Register the form with desired discretization spaces, quadrature degree, optimizations etc. in [operators.toml](https://i10git.cs.fau.de/hyteg/hyteg-operators/-/blob/main/operators.toml) at `hyteg/src/hyteg/hyteg-operators`:
```toml
[[new_form]]
trial-space   = "P2"
test-space    = "P2"
form-space-args.coefficient_function_space = "P2"
dimensions    = [2, 3]
quadrature    = 2
loop-strategy = "sawtooth"
optimizations = ["moveconstants", "vectorize", "quadloops", "tabulate"]
```
The reason for this module is to administer different versions of each operator. Depending on the machine,
it might be possible to use AVX512, AVX2 or no vectorization at all. The module will choose the correct type of vectorization for you, generate and build it.
(More precisely, it generates a non-vectorized and a vectorized version alongside logic which chooses the desired version at build time (automatically).)

Depending on the operator you are adding you might also need to extend `hyteg-operators/generate/generate.py`, e.g. if you want to use a new FE space or
blending map.

### Step 3: Operator generation
Generate the operator by creating a virtual environment including the HOG and running
`hyteg-operators/generate/generate.py`:
```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python generate.py -o ../operators ../operators.toml
```
The generated operator will be saved at `hyteg-operators/operators`.
Note: the pointer to HOG in the `requirements.txt` should point to the commit you published the form on or, if you have not published it, to you local HOG repository.

For a full list of options run
```sh
python generate.py -h
```

### Step 4: Linking and including in Apps
Link your application against the operator in the corresponding `CMakeLists.txt`:
```cmake
waLBerla_add_executable(NAME your_app
      FILES your_app.cpp
      DEPENDS hyteg opgen-new_form)
```
E.g. in `hyteg/apps/your_app_dir/CMakeLists.txt` for an app residing in `your_app_dir`.
Then, include the operator in your_app.cpp:
```cpp
#include "hyteg-operators/operators/new_form/P2ElementwiseNewForm.hpp"
```

## 2. Alternative: Command Line Script
You can also generate operators via the command line using the `generate_all_operators.py` script provided
with HOG. The script takes various inputs about the form, optimizations etc. (run with `-h` for more information).
A desired form can be defined as in HyTeG submodule: Step 1 but then also has to be registered in the
command line script `generate_all_operators.py`:
```python
    ops.append(OperatorInfo(mapping="P2", name="new_form", trial_space=P2, test_space=P2 ...))
 ```
You have to take care of the integration of the generated operator into the HyTeG build system yourself
when generating via command line.

# Formatting

Please use [black](https://black.readthedocs.io "black formatter") to format the source python files if you add to the generator.

# Building the documentation

To build the documentation make sure to install all packages in `requirements.txt` and `dev-requirements.txt`.
Then simply run the script `build_documentation.sh`.
It cleanly (re-)generates the documentation and writes html files to `doc/html`.
Simply open `doc/html/index.html` in your browser.

So for instance
```
$ pip install -r requirements.txt      # if not already done
$ pip install -r dev-requirements.txt  # if not already done
$ bash build_documentation.sh          # deletes previous generated data to ensure a clean documtation from scratch 
$ firefox doc/html/index.html          # open it up
```
