# Getting Things To Run

## Initialize the Submodule

Run `git submodule update --init --recursive` in the parent directory

## Compile the C shared library

Inside of `build`, run `cmake ..` and `cmake --build .`
This builds us 3 executables
  - `pool` main program, although unsure if we will use this
  - `pool_tests` unit testing
  - `libpool_shared` (different name on windows) the shared library which will be called by pooltool python library.

## Set up Pooltool

In the `pooltool` folder, follow the installation guide from [pooltool](https://pooltool.readthedocs.io/en/latest/getting_started/install.html).
I read the from source section, dont know if pip is enough.
You should now be able to run `run-pooltool` inside the python environment.

On startup it should print something like `Resolver settings file is located at /home/nils/.config/pooltool/physics/resolver.yaml` in the console.
Open this file and change line 7 from `model: frictional_inelastic` to `model: frictional_asl`, this will change the resolver from ball to ball collision to our custom resolver where we hopefully will pass the data to our C code.

Since im on ubuntu it could be that you need to change the following variable in the resolver to match your system.
`LIBRARY_PATH = "../build/libpool_shared.so"

### Important Files
  - Our resolver is located at `pooltools/physics/resolve/ball_ball/frictional_asl/__init__.py`
  - The original resolver at `pooltools/physics/resolve/ball_ball/frictional_mathavan/__init__.py`
